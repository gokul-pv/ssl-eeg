"""Config loading and CLI override system."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values win."""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file."""
    config_path = Path(config_path)
    cfg = load_yaml(config_path)

    for sub_key in ("dataset_config", "model_config"):
        if sub_key in cfg:
            sub_path = Path(cfg.pop(sub_key))
            # Resolve relative to cwd (caller should run from code/ root)
            sub_cfg = load_yaml(sub_path)
            # merge: sub_cfg is base, main cfg overrides
            cfg = deep_merge(sub_cfg, cfg)

    return cfg


def apply_cli_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply key=value overrides from CLI to the config dict."""
    if not overrides:
        return cfg

    cfg = deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid CLI override '{item}'. Expected format: key=value"
            )
        key, raw_value = item.split("=", 1)
        value = _infer_type(raw_value)

        # Navigate nested keys (e.g. "model_kwargs.drop_prob")
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    return cfg


def _infer_type(value: str):
    """Convert string to int, float, bool, None, or keep as str."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def cfg_to_flat_dict(cfg: dict, prefix: str = "") -> dict:
    """Flatten a nested config dict for logging (e.g. to WandB)."""
    flat = {}
    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(cfg_to_flat_dict(v, prefix=full_key))
        else:
            flat[full_key] = v
    return flat
