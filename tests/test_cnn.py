import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import make_cell  # noqa: E402

from battery_ml.features import delta_q_curves  # noqa: E402
from battery_ml.models.cnn import CNNRegressor, CycleLifeCNN, validation_split  # noqa: E402

ARCH = {"conv_channels": [4, 8], "kernel_size": 5, "pool_size": 4, "pooled_length": 4, "hidden_units": 8, "dropout": 0.0}
TRAINING = {
    "curve_transform": "asinh",
    "val_fraction": 0.25,
    "epochs": 40,
    "batch_size": 8,
    "learning_rate": 1e-2,
    "weight_decay": 0.0,
}


@pytest.fixture
def signal():
    """30 synthetic cells whose cycle life is set by their capacity fade, so delta Q(V) predicts it.

    Returns delta Q curves (ref 10, last 100), log10 cycle life, and per-cell scalar features.
    """
    cells = {f"b1c{i}": make_cell(i) for i in range(30)}
    for cell in cells.values():
        cell["cycle_life"] = 0.1 / (cell["summary"]["QD"][0] - cell["summary"]["QD"][1])
    ids = list(cells)
    curves = delta_q_curves(cells, ids, 10, 100)
    y_log = np.log10([cells[c]["cycle_life"] for c in ids])
    scalars = np.stack([cells[c]["summary"]["Tavg"][:3] for c in ids])
    return curves, y_log, scalars


@pytest.mark.parametrize("n_scalars,n_outputs", [(0, 1), (3, 1), (3, 2)])
def test_forward_shape(n_scalars, n_outputs):
    model = CycleLifeCNN(2, n_scalars, n_outputs=n_outputs, **ARCH)
    out = model(torch.zeros(5, 2, 1000), torch.zeros(5, n_scalars) if n_scalars else None)
    assert out.shape == (5, n_outputs)


def test_forward_requires_scalars():
    with pytest.raises(ValueError):
        CycleLifeCNN(1, 3, **ARCH)(torch.zeros(2, 1, 1000))


def test_validation_split_spans_targets():
    y = np.arange(41.0)[::-1]
    fit, val = validation_split(y, 0.2, seed=42)
    assert len(val) == 8
    assert set(fit).isdisjoint(val) and len(fit) + len(val) == 41
    assert y[val].min() < 5 and y[val].max() > 35  # spread over the target range
    np.testing.assert_array_equal(val, validation_split(y, 0.2, seed=42)[1])
    assert len(validation_split(y, 0, seed=42)[1]) == 0


def test_learns_signal_from_training_cells_only(signal):
    curves, y_log, _ = signal
    reg = CNNRegressor(ARCH, TRAINING, seed=0, device="cpu").fit(curves[:24], y_log[:24])

    # scaling statistics come from the cells passed to fit, not the held-out ones
    np.testing.assert_allclose(reg.scalers["target"][0].ravel(), y_log[:24].mean())
    pred = reg.predict(curves[24:])
    assert pred.shape == (6,) and np.isfinite(pred).all()
    rmse = np.sqrt(np.mean((pred - y_log[24:]) ** 2))
    rmse_mean_only = np.sqrt(np.mean((y_log[:24].mean() - y_log[24:]) ** 2))
    assert rmse < 0.25 * rmse_mean_only
    assert 1 <= reg.best_epoch <= TRAINING["epochs"] and len(reg.history) == TRAINING["epochs"]


def test_checkpoint_round_trip(signal, tmp_path):
    curves, y_log, scalars = signal
    reg = CNNRegressor(ARCH, {**TRAINING, "epochs": 5}, seed=0, device="cpu").fit(curves, y_log, scalars)
    torch.save(reg.checkpoint(), tmp_path / "cnn.pt")
    loaded = CNNRegressor.from_checkpoint(torch.load(tmp_path / "cnn.pt", weights_only=True), device="cpu")
    np.testing.assert_allclose(loaded.predict(curves, scalars), reg.predict(curves, scalars), rtol=1e-6)


def test_scalars_must_match_fit(signal):
    curves, y_log, scalars = signal
    reg = CNNRegressor(ARCH, {**TRAINING, "epochs": 2}, seed=0, device="cpu").fit(curves, y_log, scalars)
    with pytest.raises(ValueError):
        reg.predict(curves)


def test_no_validation_keeps_last_epoch(signal):
    curves, y_log, _ = signal
    reg = CNNRegressor(ARCH, {**TRAINING, "epochs": 3, "val_fraction": 0}, seed=0, device="cpu").fit(curves, y_log)
    assert reg.best_epoch == 3 and len(reg.val_index) == 0


def test_rejects_nan_curves(signal):
    curves, y_log, _ = signal
    curves = curves.copy()
    curves[0, 0, 0] = np.nan  # e.g. ref = 1 on a batch-1 cell
    with pytest.raises(ValueError, match="ref >= 2"):
        CNNRegressor(ARCH, TRAINING, seed=0, device="cpu").fit(curves, y_log)
