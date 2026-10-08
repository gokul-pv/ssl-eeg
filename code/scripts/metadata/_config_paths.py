"""Resolve raw-data locations for the metadata builders from the preprocessing
configs, so that no machine-specific path is hard-coded in Python.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def config_value(config_path: str | Path, key: str) -> str:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    if not cfg.get(key):
        raise SystemExit(
            f"'{key}' is not set in {config_path}; pass it on the command line instead."
        )
    return str(cfg[key])
