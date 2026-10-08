"""Continuous Wavelet Transform (CWT) features — mirrors ETH EEG-Bench brainfeatures."""

from __future__ import annotations

import logging

import numpy as np

try:
    import pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Number of log-spaced frequencies to sample per band when building scales
_SCALES_PER_BAND = 8


def _build_scales(
    sfreq: float,
    band_limits: list[list[float]],
    wavelet: str = "morl",
) -> np.ndarray:
    """Build a sorted scale array covering all requested frequency bands."""
    nyquist = sfreq / 2.0
    centre_freq = pywt.central_frequency(wavelet)  # cycles per sample

    freqs: set[float] = set()
    for low, high in band_limits:
        low_safe = max(low, 0.5)
        high_safe = min(high, nyquist - 0.5)
        if low_safe >= high_safe:
            continue
        freqs.update(
            np.logspace(np.log10(low_safe), np.log10(high_safe), _SCALES_PER_BAND).tolist()
        )

    if not freqs:
        return np.array([1.0])

    # scale = centre_freq * sfreq / freq  (standard CWT scale-frequency relation)
    scales = np.array([centre_freq * sfreq / f for f in freqs])
    return np.sort(scales)  # ascending scales = descending frequencies


def compute_cwt_features(
    epoch: np.ndarray,
    sfreq: float,
    band_limits: list[list[float]],
    wavelet: str = "morl",
    wavelet_w: float = 5.0,      # kept for API compatibility, unused with pywt
    n_scales_per_band: int = 8,  # kept for API compatibility, unused
) -> np.ndarray:
    """Compute CWT-based band energy features for one EEG window."""
    if not _PYWT_AVAILABLE:
        logger.warning("PyWavelets not available; returning zeros for CWT features.")
        return np.zeros(epoch.shape[0] * len(band_limits), dtype=np.float64)

    n_channels, n_times = epoch.shape
    n_bands = len(band_limits)
    nyquist = sfreq / 2.0

    # Build scales and get their corresponding centre frequencies in Hz
    scales = _build_scales(sfreq, band_limits, wavelet)
    centre_freqs_hz = pywt.scale2frequency(wavelet, scales) * sfreq

    # Pre-compute boolean mask: which scales fall in each band
    band_masks = []
    for low, high in band_limits:
        low_safe = max(low, 0.5)
        high_safe = min(high, nyquist - 0.5)
        mask = (centre_freqs_hz >= low_safe) & (centre_freqs_hz < high_safe)
        band_masks.append(mask)

    features = np.empty((n_channels, n_bands), dtype=np.float64)
    for c in range(n_channels):
        x = epoch[c].astype(np.float64)
        # pywt.cwt returns (coeffs, freqs_hz); coeffs shape: (n_scales, n_times)
        coeffs, _ = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / sfreq)
        power = np.abs(coeffs) ** 2  # (n_scales, n_times)

        for b, mask in enumerate(band_masks):
            if mask.any():
                features[c, b] = np.log1p(float(power[mask].mean()))
            else:
                features[c, b] = 0.0

    return features.ravel()


def cwt_feature_names(
    n_channels: int,
    band_limits: list[list[float]],
    ch_names: list[str] | None = None,
) -> list[str]:
    """Return human-readable feature names for CWT features."""
    names = []
    for c in range(n_channels):
        ch = ch_names[c] if ch_names is not None else f"ch{c:03d}"
        for low, high in band_limits:
            names.append(f"{ch}_cwt_{low:.1f}_{high:.1f}Hz")
    return names
