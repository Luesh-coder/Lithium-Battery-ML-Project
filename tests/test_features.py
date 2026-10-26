import numpy as np
import pytest

from battery_ml.features import FEATURE_SETS, build_feature_table, delta_q, delta_q_curves


@pytest.mark.parametrize("feature_set,n_features", [("variance", 1), ("discharge", 6), ("full", 9)])
@pytest.mark.parametrize("ref,last", [(10, 100), (2, 10)])
def test_feature_table_shapes(cells, feature_set, n_features, ref, last):
    X, y = build_feature_table(cells, list(cells), feature_set, ref, last)
    assert X.shape == (len(cells), n_features)
    assert np.isfinite(X.to_numpy()).all()
    assert (y.index == X.index).all()


def test_delta_q_is_curve_difference(cells):
    cell = cells["b1c0"]
    np.testing.assert_allclose(delta_q(cell, 10, 100), cell["Qdlin"][100] - cell["Qdlin"][10], atol=1e-6)


def test_delta_q_rejects_unkept_cycles(cells):
    with pytest.raises(ValueError):
        delta_q(cells["b1c0"], 10, 500)


def test_delta_q_curves_shape(cells):
    assert delta_q_curves(cells, list(cells), 10, 100).shape == (len(cells), 1, 1000)


def test_unknown_feature_set(cells):
    assert "bogus" not in FEATURE_SETS
    with pytest.raises(ValueError):
        build_feature_table(cells, list(cells), "bogus", 10, 100)
