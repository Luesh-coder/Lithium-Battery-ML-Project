from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path=None):
    """Load a YAML config and resolve its `paths` entries against the repo root."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {name: REPO_ROOT / p for name, p in cfg["paths"].items()}
    return cfg
