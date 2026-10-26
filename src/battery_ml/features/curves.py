"""Curve-shaped inputs for the deep models (1D CNN, deep ensembles)."""

import numpy as np

from .severson import delta_q


def delta_q_curves(cells, cell_ids, ref, last):
    """Stack delta Q(V) curves into an array of shape (n_cells, 1, n_voltage_points)."""
    return np.stack([delta_q(cells[cid], ref, last) for cid in cell_ids])[:, None, :]
