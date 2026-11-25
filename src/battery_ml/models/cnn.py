"""1D CNN on early-cycle delta Q(V) curves (Weeks 3-4).

Input:  curves of shape (batch, channels, n_voltage_points), e.g. delta Q(V) from
        battery_ml.features.delta_q_curves.
Extra:  optional scalar features (e.g. a Severson `full` feature row [1]) concatenated
        after the convolutional trunk, before the dense head.
Output: standardized log10(cycle life), shape (batch, n_outputs). The deep ensemble [3]
        can use n_outputs=2 for a mean and a variance.

CNNRegressor wraps the network in a fit / predict interface like the sklearn baselines.
The training split has only 41 cells, so the network is small and training uses dropout,
weight decay, and early stopping on training cells held out for validation.
"""

import copy

import numpy as np
import torch
from torch import nn

CURVE_TRANSFORMS = ("none", "asinh")


class CycleLifeCNN(nn.Module):
    """Conv1d -> ReLU -> MaxPool1d blocks over the voltage axis, then a dense head."""

    def __init__(
        self,
        in_channels=1,
        n_scalar_features=0,
        conv_channels=(8, 16, 32),
        kernel_size=7,
        pool_size=4,
        pooled_length=8,
        hidden_units=32,
        dropout=0.2,
        n_outputs=1,
    ):
        super().__init__()
        layers, width = [], in_channels
        for channels in conv_channels:
            layers += [
                nn.Conv1d(width, channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
            ]
            width = channels
        layers += [nn.AdaptiveAvgPool1d(pooled_length), nn.Flatten()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(width * pooled_length + n_scalar_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, n_outputs),
        )
        self.in_channels = in_channels
        self.n_scalar_features = n_scalar_features

    def forward(self, curves: torch.Tensor, scalars: torch.Tensor | None = None) -> torch.Tensor:
        z = self.trunk(curves)
        if self.n_scalar_features:
            if scalars is None:
                raise ValueError(f"this model takes {self.n_scalar_features} scalar features")
            z = torch.cat([z, scalars], dim=1)
        return self.head(z)


def validation_split(y, val_fraction, seed):
    """Split training cells into (fit, validation) indices that both span the cycle-life range.

    Cells are sorted by target and every k-th one (k = round(1 / val_fraction)) is held
    out, starting from a seed-dependent offset.
    """
    n = len(y)
    if val_fraction <= 0:
        return np.arange(n), np.array([], dtype=int)
    if val_fraction >= 1:
        raise ValueError(f"val_fraction must be below 1, got {val_fraction}")
    k = max(2, round(1 / val_fraction))
    val = np.sort(np.argsort(y, kind="stable")[seed % k :: k])
    return np.setdiff1d(np.arange(n), val), val


def _finite(x, name, hint=""):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contain NaN or inf{hint}")
    return x


def _fit_scaler(x, axis):
    """(mean, std) over `axis`; a zero std becomes 1."""
    mean, std = x.mean(axis=axis, keepdims=True), x.std(axis=axis, keepdims=True)
    return mean, np.where(std > 0, std, 1.0)


class CNNRegressor:
    """CycleLifeCNN with a fit / predict interface, like the sklearn baselines.

    fit() learns every scaling statistic from the cells it is given, so pass the training
    split only. It holds out `val_fraction` of those cells and keeps the weights from the
    epoch with the lowest validation loss. predict() returns the target on the scale that
    fit() was given (log10 cycles in scripts/train_cnn.py).
    """

    def __init__(self, architecture, training, seed, device=None):
        if training["curve_transform"] not in CURVE_TRANSFORMS:
            raise ValueError(f"curve_transform must be one of {CURVE_TRANSFORMS}")
        self.architecture = dict(architecture)
        self.training = dict(training)
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def fit(self, curves, y, scalars=None):
        curves = _finite(curves, "curves", " (batch-1 curves are all-NaN at cycle 1; use ref >= 2)")
        y = _finite(y, "y")
        median = float(np.median(np.abs(curves)))
        self.curve_scale = median if median > 0 else 1.0
        self.scalers = {
            "curve": _fit_scaler(self._transform(curves), axis=(0, 2)),  # one (mean, std) per channel
            "scalar": None if scalars is None else _fit_scaler(_finite(scalars, "scalars"), axis=0),
            "target": _fit_scaler(y, axis=0),
        }

        torch.manual_seed(self.seed)
        n_scalars = 0 if scalars is None else np.shape(scalars)[1]
        self.model = CycleLifeCNN(curves.shape[1], n_scalars, **self.architecture).to(self.device)
        xc, xs = self._inputs(curves, scalars)
        mean, std = self.scalers["target"]
        target = self._tensor(((y - mean) / std)[:, None])
        fit_idx, self.val_index = validation_split(y, self.training["val_fraction"], self.seed)
        self.history, self.best_epoch = self._train(xc, xs, target, fit_idx, self.val_index)
        return self

    def predict(self, curves, scalars=None):
        xc, xs = self._inputs(curves, scalars)
        self.model.eval()
        with torch.no_grad():
            out = self.model(xc, xs)[:, 0].cpu().numpy().astype(float)
        mean, std = self.scalers["target"]
        return out * std + mean

    def checkpoint(self):
        """A dict for torch.save. Rebuild with from_checkpoint(torch.load(path))."""
        return {
            "architecture": self.architecture,
            "training": self.training,
            "seed": self.seed,
            "in_channels": self.model.in_channels,
            "n_scalar_features": self.model.n_scalar_features,
            "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "curve_scale": self.curve_scale,
            "scalers": {
                k: None if v is None else [torch.from_numpy(np.asarray(a)) for a in v] for k, v in self.scalers.items()
            },
            "val_index": self.val_index.tolist(),
            "best_epoch": self.best_epoch,
            "history": self.history,
        }

    @classmethod
    def from_checkpoint(cls, ckpt, device=None):
        reg = cls(ckpt["architecture"], ckpt["training"], ckpt["seed"], device)
        reg.model = CycleLifeCNN(ckpt["in_channels"], ckpt["n_scalar_features"], **reg.architecture).to(reg.device)
        reg.model.load_state_dict(ckpt["state_dict"])
        reg.curve_scale = ckpt["curve_scale"]
        reg.scalers = {k: None if v is None else tuple(a.numpy() for a in v) for k, v in ckpt["scalers"].items()}
        reg.val_index = np.array(ckpt["val_index"], dtype=int)
        reg.best_epoch, reg.history = ckpt["best_epoch"], ckpt["history"]
        return reg

    def _transform(self, curves):
        """asinh(x / s) is linear for |x| << s and logarithmic for |x| >> s. delta Q spans
        decades across cells, and its log variance is the main feature in [1]."""
        if self.training["curve_transform"] == "asinh":
            return np.arcsinh(curves / self.curve_scale)
        return curves

    def _tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _inputs(self, curves, scalars):
        """Scaled float32 tensors on the device."""
        if (scalars is None) != (self.scalers["scalar"] is None):
            raise ValueError("pass scalars to predict() if and only if fit() got them")
        mean, std = self.scalers["curve"]
        xc = self._tensor((self._transform(_finite(curves, "curves")) - mean) / std)
        if scalars is None:
            return xc, None
        mean, std = self.scalers["scalar"]
        return xc, self._tensor((_finite(scalars, "scalars") - mean) / std)

    def _forward(self, xc, xs, idx):
        return self.model(xc[idx], None if xs is None else xs[idx])

    def _loss(self, xc, xs, target, idx):
        self.model.eval()
        with torch.no_grad():
            return nn.functional.mse_loss(self._forward(xc, xs, idx), target[idx]).item()

    def _train(self, xc, xs, target, fit_idx, val_idx):
        """Minibatch AdamW on MSE. Returns (history, epoch whose weights are kept)."""
        t = self.training
        opt = torch.optim.AdamW(self.model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
        shuffle = torch.Generator().manual_seed(self.seed)
        fit_idx = torch.as_tensor(fit_idx, device=self.device)
        val_idx = torch.as_tensor(val_idx, dtype=torch.long, device=self.device)
        history, best_loss, best_epoch, best_state = [], float("inf"), t["epochs"], None

        for epoch in range(1, t["epochs"] + 1):
            self.model.train()
            order = fit_idx[torch.randperm(len(fit_idx), generator=shuffle).to(self.device)]
            for batch in order.split(t["batch_size"]):
                opt.zero_grad()
                loss = nn.functional.mse_loss(self._forward(xc, xs, batch), target[batch])
                loss.backward()
                opt.step()
            val_loss = self._loss(xc, xs, target, val_idx) if len(val_idx) else float("nan")
            history.append({"epoch": epoch, "train_loss": self._loss(xc, xs, target, fit_idx), "val_loss": val_loss})
            if val_loss < best_loss:  # never true without validation cells: keep the last epoch
                best_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history, best_epoch
