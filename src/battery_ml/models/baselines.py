"""Classical baselines (Weeks 1-2): elastic net from [1], random forest, gradient boosting."""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASELINES = ("elastic_net", "random_forest", "gradient_boosting")


def make_baseline(name, params, seed):
    """Build an unfitted scikit-learn regressor from its config section."""
    if name == "elastic_net":
        return make_pipeline(
            StandardScaler(),
            ElasticNetCV(l1_ratio=params["l1_ratio"], cv=params["cv"], max_iter=100_000, random_state=seed),
        )
    if name == "random_forest":
        return RandomForestRegressor(random_state=seed, **params)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=seed, **params)
    raise ValueError(f"unknown baseline {name!r}; expected one of {BASELINES}")
