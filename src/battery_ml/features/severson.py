"""Hand-crafted features from Severson et al. (2019), generalized to any cycle window.

All three feature sets are built on delta Q(V) = Q_last(V) - Q_ref(V), the difference
between two discharge capacity curves on the shared voltage grid. With ref=10, last=100
this is the paper's delta Q_{100-10}(V). Summary-based features use cycles 2..last.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

FEATURE_SETS = ("variance", "discharge", "full")


def delta_q(cell, ref, last):
    """delta Q(V) between cycles `last` and `ref` (cycles start at 1; curve row k holds cycle k + 1)."""
    q = cell["Qdlin"]
    if ref < 1 or last > len(q):
        raise ValueError(f"cycles must be in 1..{len(q)} (the curves that were kept); got {ref} and {last}")
    return q[last - 1].astype(float) - q[ref - 1].astype(float)


def _summary_window(cell, key, first, last):
    """Summary values for cycles first..last (inclusive), looked up by cycle number."""
    s = cell["summary"]
    mask = (s["cycle"] >= first) & (s["cycle"] <= last)
    return s["cycle"][mask], s[key][mask]


def _summary_at(cell, key, cycle):
    _, values = _summary_window(cell, key, cycle, cycle)
    if values.size == 0:
        raise ValueError(f"no summary value for {key} at cycle {cycle}")
    return values[0]


def _log_abs(x):
    return np.log10(np.abs(x))


def cell_features(cell, feature_set, ref, last):
    """Feature dict for one cell."""
    dq = delta_q(cell, ref, last)
    feats = {"log_var_dq": _log_abs(np.var(dq))}
    if feature_set == "variance":
        return feats

    qd2 = _summary_at(cell, "QD", 2)
    feats["log_min_dq"] = _log_abs(np.min(dq))
    feats["qd_cycle2"] = qd2

    if feature_set == "discharge":
        _, qd = _summary_window(cell, "QD", 2, last)
        feats["log_skew_dq"] = _log_abs(skew(dq))
        feats["log_kurt_dq"] = _log_abs(kurtosis(dq))
        feats["qd_max_minus_cycle2"] = np.max(qd) - qd2
        return feats

    if feature_set == "full":
        cycles, qd = _summary_window(cell, "QD", 2, last)
        slope, intercept = np.polyfit(cycles, qd, 1)
        _, charge_time = _summary_window(cell, "chargetime", 2, 6)
        t_cycles, t_avg = _summary_window(cell, "Tavg", 2, last)
        _, ir = _summary_window(cell, "IR", 2, last)
        feats["qd_fade_slope"] = slope
        feats["qd_fade_intercept"] = intercept
        feats["mean_charge_time_first5"] = np.mean(charge_time)
        feats["temperature_integral"] = np.trapezoid(t_avg, t_cycles)
        feats["min_ir"] = np.min(ir[ir > 0])
        feats["ir_last_minus_cycle2"] = _summary_at(cell, "IR", last) - _summary_at(cell, "IR", 2)
        return feats

    raise ValueError(f"unknown feature set {feature_set!r}; expected one of {FEATURE_SETS}")


def build_feature_table(cells, cell_ids, feature_set, ref, last):
    """Return (X, y): feature DataFrame and cycle-life Series, both indexed by cell id."""
    rows = {cid: cell_features(cells[cid], feature_set, ref, last) for cid in cell_ids}
    X = pd.DataFrame.from_dict(rows, orient="index")
    y = pd.Series({cid: cells[cid]["cycle_life"] for cid in cell_ids}, name="cycle_life")
    return X, y
