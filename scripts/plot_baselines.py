"""Figures for the baseline results: RMSE and MAPE bars, and predicted vs observed cycle life.

    python scripts/plot_baselines.py                 # 100-cycle window
    python scripts/plot_baselines.py --window 50

Reads results/metrics/baselines_<feature set>.csv and _predictions.csv (from
train_baselines.py) and writes PNG + PDF figures to results/figures/. At 100 cycles the
elastic net bars carry the published values from Severson et al. (2019), Table 1.
"""

import argparse

import pandas as pd

from battery_ml.config import load_config
from battery_ml.evaluation.plots import (
    SEVERSON_TABLE1,
    plot_metric_bars,
    plot_predicted_vs_observed,
    save_figure,
    use_style,
)
from battery_ml.features import FEATURE_SETS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--window", type=int, default=100, help="early-prediction window to plot")
    parser.add_argument("--feature-sets", choices=FEATURE_SETS, nargs="+", default=list(FEATURE_SETS))
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics_dir = cfg["paths"]["results_dir"] / "metrics"
    figures_dir = cfg["paths"]["results_dir"] / "figures"

    missing = [fs for fs in args.feature_sets if not (metrics_dir / f"baselines_{fs}.csv").exists()]
    if missing:
        raise SystemExit(f"No baseline results for {missing}; run scripts/train_baselines.py first")

    scores = pd.concat(
        pd.read_csv(metrics_dir / f"baselines_{fs}.csv").assign(feature_set=fs) for fs in args.feature_sets
    )
    predictions = pd.concat(
        pd.read_csv(metrics_dir / f"baselines_{fs}_predictions.csv", index_col="cell_id").assign(feature_set=fs)
        for fs in args.feature_sets
    )
    scores = scores[scores["window"] == args.window]
    predictions = predictions[predictions["window"] == args.window]
    if scores.empty:
        raise SystemExit(f"No baseline results for the {args.window}-cycle window")

    use_style()
    reference = SEVERSON_TABLE1 if args.window == 100 else None
    stem = f"baselines_w{args.window}"
    for metric, name in (("rmse", "RMSE"), ("mape", "MAPE")):
        fig = plot_metric_bars(scores, metric, f"Baseline {name}, first {args.window} cycles", reference)
        save_figure(fig, figures_dir / f"{stem}_{metric}")
    fig = plot_predicted_vs_observed(predictions, scores, f"Predicted vs observed cycle life, first {args.window} cycles")
    save_figure(fig, figures_dir / f"{stem}_pred_vs_obs")

    print(f"Saved {stem}_rmse, {stem}_mape, {stem}_pred_vs_obs (.png, .pdf) to {figures_dir}")


if __name__ == "__main__":
    main()
