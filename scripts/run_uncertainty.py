"""Step 4 (Weeks 5-6): compare Gaussian process regression with deep ensembles.

    python scripts/run_uncertainty.py --windows 100 50 20 10

Implement battery_ml.models.gaussian_process and battery_ml.models.deep_ensemble first.
Score each method with battery_ml.evaluation.score_intervals (RMSE, MAPE, coverage,
width) at cfg["uncertainty"]["interval_level"] and save to results/metrics/uncertainty.csv.
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

    # TODO (Weeks 5-6): for each window, fit the GP and the deep ensemble on the training
    # cells, predict intervals for both test sets, and score them with score_intervals.
    raise NotImplementedError("Uncertainty comparison is not implemented yet")


if __name__ == "__main__":
    main()
