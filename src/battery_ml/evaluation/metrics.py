"""Evaluation metrics from the proposal. All inputs are in cycles, not log cycles."""

import numpy as np


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    """Mean absolute percentage error, in percent."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(100 * np.mean(np.abs((y_true - y_pred) / y_true)))


def interval_coverage(y_true, lower, upper):
    """Prediction-interval coverage probability: fraction of targets inside [lower, upper]."""
    y_true = np.asarray(y_true, float)
    return float(np.mean((y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))))


def interval_width(lower, upper):
    """Mean prediction-interval width, in cycles."""
    return float(np.mean(np.asarray(upper, float) - np.asarray(lower, float)))


def score_point(y_true, y_pred):
    return {"rmse": rmse(y_true, y_pred), "mape": mape(y_true, y_pred)}


def score_intervals(y_true, y_pred, lower, upper):
    return {
        **score_point(y_true, y_pred),
        "coverage": interval_coverage(y_true, lower, upper),
        "width": interval_width(lower, upper),
    }
