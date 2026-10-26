"""1D CNN on early-cycle Q(V) curves (Weeks 3-4).

Input:  curves of shape (batch, channels, 1000), e.g. delta Q(V) from
        battery_ml.features.delta_q_curves; optionally extra channels such as
        Q_last(V), T(V), or dQ/dV.
Extra:  per-cycle scalar measurements (e.g. a `full` feature row) concatenated
        after the convolutional trunk, before the dense head.
Output: standardized log10(cycle life).
"""

import torch
from torch import nn


class CycleLifeCNN(nn.Module):
    def __init__(self, in_channels=1, n_scalar_features=0):
        super().__init__()
        # TODO: convolutional trunk (Conv1d -> ReLU -> pooling blocks) over the voltage axis,
        # then a dense head that also takes the n_scalar_features inputs.
        raise NotImplementedError

    def forward(self, curves: torch.Tensor, scalars: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
