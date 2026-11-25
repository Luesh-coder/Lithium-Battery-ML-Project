"""Steps 2-3 (Weeks 3-4): train the 1D CNN on early-cycle delta Q(V) curves for each window.

    python scripts/train_cnn.py                                  # every window, curves only
    python scripts/train_cnn.py --windows 100 --scalar-features full

The network sees delta Q(V) = Q_last(V) - Q_ref(V) for each window's (ref, last) cycles.
With --scalar-features (default: cnn.scalar_features in the config), a Severson feature
set [1] is also fed to its dense head. It is fit on the train split only, with early
stopping on held-out training cells, and scored on all three splits.

Outputs, with <name> = cnn or cnn_<feature set>:
    results/metrics/<name>.csv              window, model, split, rmse, mape
    results/metrics/<name>_predictions.csv  per-cell predictions
    results/metrics/<name>_history.csv      train / validation loss per epoch
    results/models/<name>_w<window>.pt      checkpoint for CNNRegressor.from_checkpoint
"""

import argparse

import numpy as np
import pandas as pd
import torch

from battery_ml.config import load_config
from battery_ml.data import SPLITS, load_cells, severson_splits
from battery_ml.evaluation import score_point
from battery_ml.features import FEATURE_SETS, build_feature_table, delta_q_curves
from battery_ml.models.cnn import CNNRegressor


def split_inputs(cells, cell_ids, ref, last, scalar_features):
    """(curves, scalar features or None, cycle life) for one split."""
    curves = delta_q_curves(cells, cell_ids, ref, last)
    y = np.array([cells[cid]["cycle_life"] for cid in cell_ids], dtype=float)
    if scalar_features is None:
        return curves, None, y
    X, _ = build_feature_table(cells, cell_ids, scalar_features, ref, last)
    return curves, X.to_numpy(dtype=float), y


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--windows", type=int, nargs="+", default=None, help="e.g. 100 50 20 10")
    parser.add_argument(
        "--scalar-features",
        choices=[*FEATURE_SETS, "none"],
        default=None,
        help="Severson feature set for the dense head (default: cnn.scalar_features in the config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cnn_cfg = cfg["cnn"]
    scalar_features = args.scalar_features or cnn_cfg["scalar_features"]
    scalar_features = None if scalar_features in (None, "none") else scalar_features
    name = "cnn" if scalar_features is None else f"cnn_{scalar_features}"
    log_target = cfg["features"]["log_target"]
    windows = args.windows or list(cfg["windows"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training {name} on {device}")

    cells = load_cells(cfg["paths"]["preprocessed_dir"] / cfg["data"]["output_file"])
    splits = severson_splits(cells)
    model_dir = cfg["paths"]["results_dir"] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    scores, predictions, histories = [], [], []
    for window in windows:
        ref, last = cfg["windows"][window]["ref"], cfg["windows"][window]["last"]
        data = {split: split_inputs(cells, splits[split], ref, last, scalar_features) for split in SPLITS}

        train_curves, train_scalars, y_train = data["train"]
        target = np.log10(y_train) if log_target else y_train
        reg = CNNRegressor(cnn_cfg["architecture"], cnn_cfg["training"], cfg["seed"], device)
        reg.fit(train_curves, target, train_scalars)
        torch.save(reg.checkpoint(), model_dir / f"{name}_w{window}.pt")

        history = pd.DataFrame(reg.history)
        histories.append(history.assign(window=window)[["window", *history.columns]])
        best = history.iloc[reg.best_epoch - 1]
        print(
            f"window {window}: kept epoch {reg.best_epoch}/{len(history)} "
            f"(train loss {best.train_loss:.3f}, val loss {best.val_loss:.3f} on {len(reg.val_index)} cells)"
        )

        for split, (curves, scalars, y) in data.items():
            pred = reg.predict(curves, scalars)
            pred = 10**pred if log_target else pred
            scores.append({"window": window, "model": name, "split": split, **score_point(y, pred)})
            predictions.append(
                pd.DataFrame(
                    {"window": window, "model": name, "split": split, "cycle_life": y, "predicted": pred},
                    index=pd.Index(splits[split], name="cell_id"),
                )
            )

    out_dir = cfg["paths"]["results_dir"] / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(scores)
    scores.to_csv(out_dir / f"{name}.csv", index=False)
    pd.concat(predictions).to_csv(out_dir / f"{name}_predictions.csv")
    pd.concat(histories).to_csv(out_dir / f"{name}_history.csv", index=False)

    table = scores.pivot_table(index=["window", "model"], columns="split", values=["rmse", "mape"])
    print(f"\n{name}  (RMSE in cycles, MAPE in %)")
    print(table.round(1).sort_index(ascending=[False, True]).to_string())


if __name__ == "__main__":
    main()
