"""Models. The PyTorch modules (cnn, deep_ensemble) are imported directly so that the
classical baselines work without torch installed."""

from .baselines import BASELINES, make_baseline

__all__ = ["BASELINES", "make_baseline"]
