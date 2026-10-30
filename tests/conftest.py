import numpy as np
import pytest

from battery_ml.data.loader import N_VOLTAGE_POINTS


def make_cell(seed, n_cycles=120, cycle_life=800.0):
    """A synthetic cell shaped like the output of battery_ml.data.loader."""
    rng = np.random.default_rng(seed)
    fade = 1e-4 * (1 + rng.random())
    cycles = np.arange(1, n_cycles + 1, dtype=float)  # numbered from 1, like the real summary
    v = np.linspace(0, 1, N_VOLTAGE_POINTS)
    qdlin = 1.1 * v * (1 - fade * cycles[:, None]) + 1e-4 * rng.standard_normal((n_cycles, N_VOLTAGE_POINTS))
    return {
        "cycle_life": cycle_life,
        "policy": "5C(67%)-4C",
        "batch": "b1",
        "n_cycles": n_cycles,
        "summary": {
            "cycle": cycles,
            "QD": 1.1 - fade * cycles,
            "QC": 1.1 - fade * cycles,
            "IR": 0.016 + 1e-6 * cycles,
            "Tavg": 31 + rng.random(n_cycles),
            "Tmin": 30 + rng.random(n_cycles),
            "Tmax": 33 + rng.random(n_cycles),
            "chargetime": 10 + rng.random(n_cycles),
        },
        "Qdlin": qdlin.astype(np.float32),
        "Tdlin": np.full((n_cycles, N_VOLTAGE_POINTS), 31, dtype=np.float32),
    }


@pytest.fixture
def cells():
    return {f"b1c{i}": make_cell(i, cycle_life=500.0 + 50 * i) for i in range(12)}
