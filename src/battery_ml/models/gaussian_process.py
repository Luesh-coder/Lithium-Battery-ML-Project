"""Gaussian process regression with prediction intervals (Weeks 5-6).

Fit on the hand-crafted features in log10(cycle life) space, then map the interval
bounds back to cycles with 10 ** bound so they can be scored by
battery_ml.evaluation.score_intervals.
"""


def fit_gp(X, y_log, seed):
    """Fit a GP (e.g. sklearn GaussianProcessRegressor with RBF + WhiteKernel) on standardized X."""
    raise NotImplementedError


def predict_interval(model, X, level):
    """Return (mean, lower, upper) in cycles for a central interval at the given level."""
    raise NotImplementedError
