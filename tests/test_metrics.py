import pytest

from battery_ml.evaluation import interval_coverage, interval_width, mape, rmse


def test_point_metrics():
    assert rmse([100, 200], [110, 190]) == pytest.approx(10.0)
    assert mape([100, 200], [110, 190]) == pytest.approx(7.5)


def test_interval_metrics():
    y = [100, 200, 300, 400]
    lower, upper = [90, 210, 250, 350], [110, 230, 350, 450]
    assert interval_coverage(y, lower, upper) == pytest.approx(0.75)
    assert interval_width(lower, upper) == pytest.approx(60.0)
