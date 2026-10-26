"""Step 1: read the three raw batch files and write data/preprocessed/cells.pkl.

    python scripts/preprocess.py
"""

import argparse

import pandas as pd

from battery_ml.config import load_config
from battery_ml.data import build_dataset, save_cells, severson_splits

DATA_URL = "https://data.matr.io/1/projects/5c48dd2bc625d700019f3204"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir, out_dir = cfg["paths"]["raw_dir"], cfg["paths"]["preprocessed_dir"]
    batch_files = cfg["data"]["batch_files"]

    missing = [name for name in batch_files.values() if not (raw_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"Missing in {raw_dir}:\n  " + "\n  ".join(missing) + f"\nDownload them from {DATA_URL}"
        )

    cells = build_dataset(raw_dir, batch_files, cfg["data"]["keep_cycles"])
    out_path = out_dir / cfg["data"]["output_file"]
    save_cells(cells, out_path)

    split_of = {cid: split for split, ids in severson_splits(cells).items() for cid in ids}
    table = pd.DataFrame(
        {
            "batch": [c["batch"] for c in cells.values()],
            "policy": [c["policy"] for c in cells.values()],
            "cycle_life": [c["cycle_life"] for c in cells.values()],
            "n_cycles": [c["n_cycles"] for c in cells.values()],
            "split": [split_of[cid] for cid in cells],
        },
        index=pd.Index(list(cells), name="cell_id"),
    )
    table.to_csv(out_dir / "cells_summary.csv")

    print(f"Saved {len(cells)} cells to {out_path}")
    print(table["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
