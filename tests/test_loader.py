import itertools

import h5py
import numpy as np

from battery_ml.data import build_dataset
from battery_ml.data.loader import N_VOLTAGE_POINTS, SUMMARY_FIELDS

N_CYCLES = 6
KEEP_CYCLES = 10
POLICY = "5C(67%)-4C"
BATCH_FILES = {"b1": "b1.mat", "b2": "b2.mat", "b3": "b3.mat"}


def write_fake_batch(path, n_cells):
    """Write an HDF5 file laid out like the MATLAB v7.3 batch structs on data.matr.io."""
    with h5py.File(path, "w") as f:
        refs, batch = f.create_group("#refs#"), f.create_group("batch")
        fields = {
            name: batch.create_dataset(name, (n_cells, 1), dtype=h5py.ref_dtype)
            for name in ("summary", "cycles", "cycle_life", "policy_readable")
        }
        counter = itertools.count()

        def store(data):
            return refs.create_dataset(str(next(counter)), data=data).ref

        for i in range(n_cells):
            summary = refs.create_group(f"summary{i}")
            for name in SUMMARY_FIELDS:
                values = np.arange(1, N_CYCLES + 1, dtype=float) if name == "cycle" else np.full(N_CYCLES, float(i))
                summary.create_dataset(name, data=values[None, :])
            fields["summary"][i, 0] = summary.ref

            cycles = refs.create_group(f"cycles{i}")
            for field in ("Qdlin", "Tdlin"):
                ds = cycles.create_dataset(field, (N_CYCLES, 1), dtype=h5py.ref_dtype)
                for j in range(N_CYCLES):
                    ds[j, 0] = store(np.full((1, N_VOLTAGE_POINTS), i + j / 100))
            fields["cycles"][i, 0] = cycles.ref

            fields["cycle_life"][i, 0] = store(np.array([[500.0 + i]]))
            chars = np.frombuffer(POLICY.encode(), dtype=np.uint8).astype(np.uint16)
            fields["policy_readable"][i, 0] = store(chars[:, None])


def test_build_dataset_matches_published_cleaning(tmp_path):
    # Real batch sizes before cleaning: 46, 48, 46 cells.
    for name, n_cells in zip(BATCH_FILES.values(), (46, 48, 46)):
        write_fake_batch(tmp_path / name, n_cells)

    cells = build_dataset(tmp_path, BATCH_FILES, KEEP_CYCLES)

    assert len(cells) == 124
    assert "b1c8" not in cells and "b3c37" not in cells and "b2c7" not in cells

    # b1c0 is continued by b2c7: cycle life, summary, and curves are extended.
    b1c0 = cells["b1c0"]
    assert b1c0["cycle_life"] == 500.0 + 662
    assert b1c0["policy"] == POLICY
    assert b1c0["n_cycles"] == 2 * N_CYCLES
    np.testing.assert_array_equal(b1c0["summary"]["cycle"], np.arange(1, 2 * N_CYCLES + 1))
    assert b1c0["Qdlin"].shape == (KEEP_CYCLES, N_VOLTAGE_POINTS)
    np.testing.assert_allclose(b1c0["Qdlin"][N_CYCLES], 7.0)  # first continuation cycle of b2c7

    assert cells["b3c0"]["Qdlin"].shape == (N_CYCLES, N_VOLTAGE_POINTS)
