"""Discrete Wavelet Transform (DWT) energy features."""

from __future__ import annotations

import numpy as np

try:
    import pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False


def _check_pywt() -> None:
    if not _PYWT_AVAILABLE:
        raise ImportError(
            "PyWavelets (`pywt`) is required for DWT features. "
            "Install it with: pip install PyWavelets"
        )


def compute_wavelet_features(
    epoch: np.ndarray,
    wavelet: str = "db4",
    level: int | None = None,
) -> np.ndarray:
    """Compute DWT energy features for one EEG window."""
    _check_pywt()

    n_channels, n_times = epoch.shape
    if level is None:
        level = pywt.dwt_max_level(n_times, wavelet)

    n_levels = level + 1  # detail levels (1..level) + approximation (0)
    features = np.empty((n_channels, n_levels), dtype=np.float64)

    for c in range(n_channels):
        x = epoch[c].astype(np.float64)
        coeffs = pywt.wavedec(x, wavelet, level=level)
        # coeffs[0] = approximation, coeffs[1..] = details (finest → coarsest reversed)
        energies = np.array([np.sum(c_ ** 2) for c_ in coeffs], dtype=np.float64)
        total = energies.sum()
        # Log-normalised energy; if total is 0 all features are 0
        if total > 0:
            features[c] = np.log1p(energies / total * total)  # = log1p(energies)
        else:
            features[c] = 0.0

    return features.ravel()


def wavelet_feature_names(
    n_channels: int,
    level: int,
    ch_names: list[str] | None = None,
) -> list[str]:
    """Return human-readable feature names for DWT features."""
    names = []
    for c in range(n_channels):
        ch = ch_names[c] if ch_names is not None else f"ch{c:03d}"
        names.append(f"{ch}_dwt_approx")
        for lv in range(1, level + 1):
            names.append(f"{ch}_dwt_detail_{lv}")
    return names


def get_dwt_level(n_times: int, wavelet: str = "db4") -> int:
    """Return max DWT decomposition level for a signal length."""
    _check_pywt()
    return pywt.dwt_max_level(n_times, wavelet)
