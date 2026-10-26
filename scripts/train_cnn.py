"""Steps 2-3 (Weeks 3-4): train the 1D CNN on early-cycle Q(V) curves for each window.

    python scripts/train_cnn.py --windows 100 50 20 10

Implement battery_ml.models.cnn.CycleLifeCNN first. Save checkpoints to results/models/
and scores to results/metrics/cnn.csv in the same format as train_baselines.py.
"""

import argparse

from battery_ml.config import load_config
from battery_ml.data import load_cells, severson_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--windows", type=int, nargs="+", default=None, help="e.g. 100 50 20 10")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cells = load_cells(cfg["paths"]["preprocessed_dir"] / cfg["data"]["output_file"])
    splits = severson_splits(cells)

    # TODO (Weeks 3-4):
    #   1. inputs: battery_ml.features.delta_q_curves(cells, splits[...], ref, last)
    #   2. standardize curves and log10(cycle life) using training-set statistics only
    #   3. train CycleLifeCNN on cfg["cnn"] settings (GPU if torch.cuda.is_available())
    #   4. score each split with battery_ml.evaluation.score_point
    raise NotImplementedError("CNN training is not implemented yet")


if __name__ == "__main__":
    main()
