"""Cache key for extracted feature matrices."""

from __future__ import annotations

import hashlib
import json


def feature_cache_key(cfg: dict) -> str:
    """Hash of everything that determines a feature matrix: the feature settings,
    the preprocessed data folder, the channel selection, the task, the sampling
    rate and the normalisation. (The original key only hashed the feature
    settings, so e.g. a 61- and a 51-channel FEP run would have shared one
    cache entry.)
    """
    payload = {
        "ml_features": cfg.get("ml_features", {}),
        "data_dir": str(cfg.get("data_dir")),
        "dataset_id": cfg.get("dataset_id"),
        "save_folder": cfg.get("save_folder"),
        "common_ch_names": cfg.get("common_ch_names"),
        "classify_choice": cfg.get("classify_choice"),
        "sfreq": cfg.get("sfreq", 200),
        "normalize": cfg.get("normalize"),
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
