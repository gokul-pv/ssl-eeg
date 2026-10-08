"""Band-power features computed via Welch's PSD method."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def compute_band_powers(
    epoch: np.ndarray,
    sfreq: float,
    band_limits: list[list[float]],
    nperseg_factor: float = 4.0,
) -> np.ndarray:
    """Compute mean band power in each frequency band for each channel via Welch PSD."""
    n_channels = epoch.shape[0]
    n_bands = len(band_limits)
    nperseg = min(int(sfreq * nperseg_factor), epoch.shape[1])

    features = np.empty((n_channels, n_bands), dtype=np.float64)
    for c in range(n_channels):
        freqs, psd = welch(epoch[c], fs=sfreq, nperseg=nperseg)
        df = freqs[1] - freqs[0]  # frequency resolution
        for b, (low, high) in enumerate(band_limits):
            idx = np.where((freqs >= low) & (freqs < high))[0]
            if len(idx) == 0:
                features[c, b] = 0.0
            else:
                # Mean power (absolute) then log-transform for stability
                mean_pwr = float(np.trapezoid(psd[idx], freqs[idx]))
                features[c, b] = np.log1p(max(mean_pwr, 0.0))

    return features.ravel()


def band_power_feature_names(
    n_channels: int,
    band_limits: list[list[float]],
    ch_names: list[str] | None = None,
) -> list[str]:
    """Return human-readable feature names for band power features."""
    names = []
    for c in range(n_channels):
        ch = ch_names[c] if ch_names is not None else f"ch{c:03d}"
        for low, high in band_limits:
            names.append(f"{ch}_bp_{low:.1f}_{high:.1f}Hz")
    return names
