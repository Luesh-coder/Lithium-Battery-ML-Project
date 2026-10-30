import matplotlib as mpl
import pandas as pd
import pytest

from battery_ml.data import SPLITS
from battery_ml.evaluation.plots import (
    SEVERSON_TABLE1,
    STYLE,
    plot_metric_bars,
    plot_predicted_vs_observed,
    save_figure,
)


@pytest.fixture
def results():
    """Scores and predictions in the format written by train_baselines.py."""
    scores, predictions = [], []
    for fs in ("variance", "full"):
        for model, rmse in (("elastic_net", 120.0), ("gradient_boosting", 0.5)):
            for split in SPLITS:
                scores.append({"window": 100, "model": model, "split": split, "rmse": rmse, "mape": 12.0, "feature_set": fs})
                ids = pd.Index([f"{split}_a", f"{split}_b"], name="cell_id")
                predictions.append(
                    pd.DataFrame(
                        {"window": 100, "model": model, "split": split, "cycle_life": [500.0, 1000.0],
                         "predicted": [550.0, 0.0], "feature_set": fs},
                        index=ids,
                    )
                )
    return pd.DataFrame(scores), pd.concat(predictions)


def test_figures_render_and_save(results, tmp_path):
    scores, predictions = results
    with mpl.rc_context(STYLE):
        figures = {
            "rmse": plot_metric_bars(scores, "rmse", "RMSE", SEVERSON_TABLE1),
            "mape": plot_metric_bars(scores, "mape", "MAPE"),
            "pred_vs_obs": plot_predicted_vs_observed(predictions, scores, "Predicted vs observed"),
        }
        for name, fig in figures.items():
            save_figure(fig, tmp_path / name)
    for name in figures:
        assert (tmp_path / f"{name}.png").stat().st_size > 0
        assert (tmp_path / f"{name}.pdf").stat().st_size > 0
