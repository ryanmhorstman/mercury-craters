"""Load the YAML config and resolve paths against the repo root."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


def load_config(path=None):
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {key: resolve(value) for key, value in cfg["paths"].items()}
    return cfg


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
