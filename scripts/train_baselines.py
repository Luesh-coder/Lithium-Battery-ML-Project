"""Steps 2-3: classical baselines on Severson features, over every early-prediction window.

    python scripts/train_baselines.py                          # all windows, all models
    python scripts/train_baselines.py --feature-set full --windows 100 50
"""

import argparse

import numpy as np
import pandas as pd

from battery_ml.config import load_config
from battery_ml.data import SPLITS, load_cells, severson_splits
from battery_ml.evaluation import score_point
from battery_ml.features import FEATURE_SETS, build_feature_table
from battery_ml.models import BASELINES, make_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default=None)
    parser.add_argument("--windows", type=int, nargs="+", default=None, help="e.g. 100 50 20 10")
    parser.add_argument("--models", choices=BASELINES, nargs="+", default=list(BASELINES))
    args = parser.parse_args()

    cfg = load_config(args.config)
    feature_set = args.feature_set or cfg["features"]["feature_set"]
    log_target = cfg["features"]["log_target"]
    windows = args.windows or list(cfg["windows"])

    cells = load_cells(cfg["paths"]["preprocessed_dir"] / cfg["data"]["output_file"])
    splits = severson_splits(cells)

    scores, predictions = [], []
    for window in windows:
        ref, last = cfg["windows"][window]["ref"], cfg["windows"][window]["last"]
        data = {split: build_feature_table(cells, splits[split], feature_set, ref, last) for split in SPLITS}

        X_train, y_train = data["train"]
        target = np.log10(y_train) if log_target else y_train

        for name in args.models:
            model = make_baseline(name, cfg["baselines"][name], cfg["seed"])
            model.fit(X_train, target)
            for split, (X, y) in data.items():
                pred = model.predict(X)
                pred = 10**pred if log_target else pred
                scores.append({"window": window, "model": name, "split": split, **score_point(y, pred)})
                predictions.append(
                    pd.DataFrame({"window": window, "model": name, "split": split, "cycle_life": y, "predicted": pred})
                )

    out_dir = cfg["paths"]["results_dir"] / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(scores)
    scores.to_csv(out_dir / f"baselines_{feature_set}.csv", index=False)
    pd.concat(predictions).rename_axis("cell_id").to_csv(out_dir / f"baselines_{feature_set}_predictions.csv")

    table = scores.pivot_table(index=["window", "model"], columns="split", values=["rmse", "mape"])
    print(f"Feature set: {feature_set}  (RMSE in cycles, MAPE in %)")
    print(table.round(1).sort_index(ascending=[False, True]).to_string())


if __name__ == "__main__":
    main()
