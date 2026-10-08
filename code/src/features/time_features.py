"""Time-domain EEG features per channel."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


# Feature names per channel (ordered)
_TIME_FEATURE_NAMES_PER_CHANNEL = [
    "mean", "variance", "skewness", "kurtosis", "rms",
    "zcr", "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
    "line_length", "peak_to_peak",
]
N_TIME_FEATURES_PER_CHANNEL = len(_TIME_FEATURE_NAMES_PER_CHANNEL)


def compute_time_features(epoch: np.ndarray) -> np.ndarray:
    """Compute time-domain features for one EEG window."""
    n_channels = epoch.shape[0]
    features = np.empty((n_channels, N_TIME_FEATURES_PER_CHANNEL), dtype=np.float64)

    for c in range(n_channels):
        x = epoch[c].astype(np.float64)
        features[c] = _channel_time_features(x)

    return features.ravel()


def _channel_time_features(x: np.ndarray) -> np.ndarray:
    """Compute time-domain stats for a 1D signal."""
    n = len(x)
    if n == 0:
        return np.zeros(N_TIME_FEATURES_PER_CHANNEL)

    mean_ = float(np.mean(x))
    var_ = float(np.var(x, ddof=1)) if n > 1 else 0.0
    skew_ = float(skew(x)) if n > 2 else 0.0
    kurt_ = float(kurtosis(x)) if n > 3 else 0.0
    rms_ = float(np.sqrt(np.mean(x ** 2)))

    # Zero-crossing rate (per sample pair)
    zero_crossings = float(np.sum(np.diff(np.sign(x)) != 0)) / max(n - 1, 1)

    # Hjorth parameters
    dx = np.diff(x)
    ddx = np.diff(dx)
    activity = float(np.var(x, ddof=0))
    mob_num = float(np.var(dx, ddof=0))
    mob = float(np.sqrt(mob_num / activity)) if activity > 0 else 0.0

    mob_dx_num = float(np.var(ddx, ddof=0))
    mob_dx = float(np.sqrt(mob_dx_num / mob_num)) if mob_num > 0 else 0.0
    complexity = float(mob_dx / mob) if mob > 0 else 0.0

    # Line length (sum of absolute differences)
    line_len = float(np.sum(np.abs(dx))) / max(n - 1, 1)

    # Peak-to-peak
    p2p = float(np.max(x) - np.min(x))

    return np.array([
        mean_, var_, skew_, kurt_, rms_,
        zero_crossings, activity, mob, complexity,
        line_len, p2p,
    ], dtype=np.float64)


def time_feature_names(
    n_channels: int,
    ch_names: list[str] | None = None,
) -> list[str]:
    """Return human-readable feature names for time-domain features."""
    names = []
    for c in range(n_channels):
        ch = ch_names[c] if ch_names is not None else f"ch{c:03d}"
        for feat in _TIME_FEATURE_NAMES_PER_CHANNEL:
            names.append(f"{ch}_time_{feat}")
    return names
