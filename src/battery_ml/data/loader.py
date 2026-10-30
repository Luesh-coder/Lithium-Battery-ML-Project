"""Read the Severson et al. (2019) batch files into a clean 124-cell dataset.

The three MATLAB v7.3 structs from data.matr.io are HDF5 files underneath, so they are
read with h5py. Cleaning follows the authors' starter code
(github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation):

* batch 1: drop 5 cells that never reached 80% of nominal capacity
* batch 2: 5 cells are continuations of batch-1 cells; append them to those cells
* batch 3: drop 6 noisy channels

Each cell is stored as a dict:
    cycle_life  float, cycles until capacity falls to 80% of nominal
    policy      fast-charging protocol string, e.g. "3.6C(80%)-3.6C"
    batch       "b1" | "b2" | "b3"
    n_cycles    total cycles recorded
    summary     per-cycle arrays: cycle, QD, QC, IR, Tavg, Tmin, Tmax, chargetime
    Qdlin       (keep_cycles, 1000) discharge capacity on VOLTAGE_GRID, row k = cycle k + 1
    Tdlin       (keep_cycles, 1000) temperature on VOLTAGE_GRID, row k = cycle k + 1

Cycles are numbered from 1, as in summary["cycle"], which has one entry per curve row.
In batch 1, row 0 (cycle 1) is all NaN.
"""

import pickle

import h5py
import numpy as np
from tqdm import tqdm

N_VOLTAGE_POINTS = 1000
VOLTAGE_GRID = np.linspace(3.6, 2.0, N_VOLTAGE_POINTS)

SUMMARY_FIELDS = {  # name in the .mat struct -> name used here
    "cycle": "cycle",
    "QDischarge": "QD",
    "QCharge": "QC",
    "IR": "IR",
    "Tavg": "Tavg",
    "Tmin": "Tmin",
    "Tmax": "Tmax",
    "chargetime": "chargetime",
}
CURVE_FIELDS = ("Qdlin", "Tdlin")

BATCH1_NO_EOL = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]
BATCH3_NOISY = ["b3c2", "b3c23", "b3c32", "b3c37", "b3c42", "b3c43"]
# batch-2 cell -> (batch-1 cell it continues, cycles to add to that cell's cycle life)
CONTINUED_CELLS = {
    "b2c7": ("b1c0", 662),
    "b2c8": ("b1c1", 981),
    "b2c9": ("b1c2", 1060),
    "b2c15": ("b1c3", 208),
    "b2c16": ("b1c4", 482),
}


def read_batch(path, batch, keep_cycles):
    """Read one batch file into {cell_id: cell dict}."""
    cells = {}
    with h5py.File(path, "r") as f:
        structs = f["batch"]
        n_cells = structs["summary"].shape[0]
        for i in tqdm(range(n_cells), desc=f"reading {batch}"):
            summary_group = f[structs["summary"][i, 0]]
            summary = {
                short: np.asarray(summary_group[name][0, :], dtype=float)
                for name, short in SUMMARY_FIELDS.items()
            }

            cycles = f[structs["cycles"][i, 0]]
            n_cycles = cycles["Qdlin"].shape[0]
            curves = {}
            for field in CURVE_FIELDS:
                arr = np.full((min(n_cycles, keep_cycles), N_VOLTAGE_POINTS), np.nan, dtype=np.float32)
                for j in range(len(arr)):
                    values = np.asarray(f[cycles[field][j, 0]][()]).ravel()
                    if values.size == N_VOLTAGE_POINTS:
                        arr[j] = values
                curves[field] = arr

            policy = f[structs["policy_readable"][i, 0]][()].tobytes()[::2].decode()
            cycle_life = float(np.asarray(f[structs["cycle_life"][i, 0]][()]).ravel()[0])
            cells[f"{batch}c{i}"] = {
                "cycle_life": cycle_life,
                "policy": policy,
                "batch": batch,
                "n_cycles": n_cycles,
                "summary": summary,
                **curves,
            }
    return cells


def merge_continued_cells(b1, b2, keep_cycles):
    """Append the batch-2 continuation data to its batch-1 cell, then drop it from batch 2."""
    for b2_id, (b1_id, extra_cycles) in CONTINUED_CELLS.items():
        first, cont = b1[b1_id], b2.pop(b2_id)
        first["cycle_life"] += extra_cycles

        n_first = len(first["summary"]["cycle"])
        for name, values in first["summary"].items():
            tail = cont["summary"][name]
            if name == "cycle":
                tail = tail + n_first
            first["summary"][name] = np.concatenate([values, tail])

        for field in CURVE_FIELDS:
            missing = keep_cycles - len(first[field])
            if missing > 0:
                first[field] = np.concatenate([first[field], cont[field][:missing]])
        first["n_cycles"] += cont["n_cycles"]


def build_dataset(raw_dir, batch_files, keep_cycles):
    """Read all three batches, apply the cleaning steps, and return {cell_id: cell}."""
    b1 = read_batch(raw_dir / batch_files["b1"], "b1", keep_cycles)
    b2 = read_batch(raw_dir / batch_files["b2"], "b2", keep_cycles)
    b3 = read_batch(raw_dir / batch_files["b3"], "b3", keep_cycles)

    for cell_id in BATCH1_NO_EOL:
        del b1[cell_id]
    merge_continued_cells(b1, b2, keep_cycles)
    for cell_id in BATCH3_NOISY:
        del b3[cell_id]

    return {**b1, **b2, **b3}


def save_cells(cells, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cells, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cells(path):
    with open(path, "rb") as f:
        return pickle.load(f)
