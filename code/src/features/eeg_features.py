"""Top-level EEG feature extraction for traditional ML baselines."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np

from .band_power import compute_band_powers, band_power_feature_names
from .time_features import compute_time_features, time_feature_names
from .wavelet_features import (
    compute_wavelet_features,
    wavelet_feature_names,
    get_dwt_level,
    _PYWT_AVAILABLE,
)
from .cwt_features import compute_cwt_features, cwt_feature_names

logger = logging.getLogger(__name__)

# Standard 5-band clinical EEG frequency decomposition
DEFAULT_BAND_LIMITS = [
    [0.5,  4.0],   # delta
    [4.0,  8.0],   # theta
    [8.0,  13.0],  # alpha
    [13.0, 30.0],  # beta
    [30.0, 45.0],  # gamma
]

DEFAULT_DOMAINS = ["dft", "cwt", "dwt", "time"]


# ---------------------------------------------------------------------------
# Per-window extraction (picklable for multiprocessing)
# ---------------------------------------------------------------------------


def _extract_one(
    epoch: np.ndarray,
    sfreq: float,
    domains: list[str],
    band_limits: list[list[float]],
    wavelet: str,
    cwt_wavelet: str,
    dwt_level: int | None,
) -> np.ndarray:
    """Extract feature vector from a single (C, T) epoch."""
    parts = []
    if "dft" in domains:
        parts.append(compute_band_powers(epoch, sfreq, band_limits))
    if "cwt" in domains:
        parts.append(compute_cwt_features(epoch, sfreq, band_limits, cwt_wavelet))
    if "dwt" in domains:
        parts.append(compute_wavelet_features(epoch, wavelet=wavelet, level=dwt_level))
    if "time" in domains:
        parts.append(compute_time_features(epoch))
    if not parts:
        raise ValueError(f"No valid domains specified. Got: {domains}")
    return np.concatenate(parts, dtype=np.float64)


def _extract_one_wrapper(args):
    """Top-level wrapper so it's picklable for multiprocessing."""
    epoch, sfreq, domains, band_limits, wavelet, cwt_wavelet, dwt_level = args
    return _extract_one(epoch, sfreq, domains, band_limits, wavelet, cwt_wavelet, dwt_level)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_features_from_windows(
    X_np: np.ndarray,
    sfreq: float,
    cfg: dict,
    ch_names: list[str] | None = None,
) -> np.ndarray:
    """Extract a 2D feature matrix from a batch of EEG windows."""
    ml_cfg = cfg.get("ml_features", {})
    domains = list(ml_cfg.get("domains", DEFAULT_DOMAINS))
    band_limits = list(ml_cfg.get("band_limits", DEFAULT_BAND_LIMITS))
    wavelet = str(ml_cfg.get("wavelet", "db4"))
    cwt_wavelet = str(ml_cfg.get("cwt_wavelet", "morl"))
    n_jobs = int(ml_cfg.get("n_jobs", 1))

    # Validate frequency bands against Nyquist
    nyquist = sfreq / 2.0
    valid_bands = [[lo, min(hi, nyquist - 0.1)] for lo, hi in band_limits if lo < nyquist]
    if len(valid_bands) < len(band_limits):
        logger.warning(
            f"Removed {len(band_limits) - len(valid_bands)} bands exceeding Nyquist ({nyquist} Hz)."
        )
    band_limits = valid_bands

    # DWT availability check
    if "dwt" in domains and not _PYWT_AVAILABLE:
        logger.warning("PyWavelets not available; removing 'dwt' from feature domains.")
        domains = [d for d in domains if d != "dwt"]

    n_windows, n_channels, n_times = X_np.shape

    # Determine DWT level from first window
    dwt_level: int | None = None
    if "dwt" in domains:
        try:
            dwt_level = get_dwt_level(n_times, wavelet)
        except Exception:
            dwt_level = None

    logger.info(
        f"Extracting features from {n_windows} windows "
        f"({n_channels}ch × {n_times}t @ {sfreq}Hz) | "
        f"domains={domains} | bands={len(band_limits)} | dwt_level={dwt_level} | n_jobs={n_jobs}"
    )

    args_list = [
        (X_np[i], sfreq, domains, band_limits, wavelet, cwt_wavelet, dwt_level)
        for i in range(n_windows)
    ]

    if n_jobs == 1:
        feats = [_extract_one_wrapper(a) for a in args_list]
    else:
        # Use process pool; limit workers to avoid memory blowup
        actual_jobs = min(n_jobs, n_windows)
        feats = [None] * n_windows
        with ProcessPoolExecutor(max_workers=actual_jobs) as executor:
            futures = {executor.submit(_extract_one_wrapper, a): i for i, a in enumerate(args_list)}
            for fut in as_completed(futures):
                idx = futures[fut]
                feats[idx] = fut.result()

    X_feat = np.stack(feats, axis=0)

    # Replace any NaN / Inf from degenerate signals
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Feature matrix shape: {X_feat.shape}")
    return X_feat.astype(np.float32)


def get_feature_names(
    n_channels: int,
    sfreq: float,
    cfg: dict,
    ch_names: list[str] | None = None,
) -> list[str]:
    """Return ordered list of feature names matching extract_features_from_windows output."""
    ml_cfg = cfg.get("ml_features", {})
    domains = list(ml_cfg.get("domains", DEFAULT_DOMAINS))
    band_limits = list(ml_cfg.get("band_limits", DEFAULT_BAND_LIMITS))
    wavelet = str(ml_cfg.get("wavelet", "db4"))
    cwt_wavelet = str(ml_cfg.get("cwt_wavelet", "morl"))

    # dummy n_times to get dwt_level
    dummy_n_times = 800  # 4s @ 200Hz
    dwt_level = get_dwt_level(dummy_n_times, wavelet) if "dwt" in domains and _PYWT_AVAILABLE else 0

    names = []
    if "dft" in domains:
        names.extend(band_power_feature_names(n_channels, band_limits, ch_names))
    if "cwt" in domains:
        names.extend(cwt_feature_names(n_channels, band_limits, ch_names))
    if "dwt" in domains and _PYWT_AVAILABLE:
        names.extend(wavelet_feature_names(n_channels, dwt_level, ch_names))
    if "time" in domains:
        names.extend(time_feature_names(n_channels, ch_names))
    return names
