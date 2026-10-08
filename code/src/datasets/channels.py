"""Channel names and shape information handed to braindecode models."""

from __future__ import annotations


def infer_ch_names(windows_ds) -> list[str] | None:
    """Channel names of the first recording (fallback when no channel selection is set)."""
    first = windows_ds.datasets[0] if windows_ds.datasets else None
    raw = getattr(first, "raw", None)
    return list(raw.ch_names) if raw is not None else None


def resolve_ch_names(ds, fallback: list[str] | None) -> list[str] | None:
    """Channel names in the exact row order ``ds`` yields."""
    selected = getattr(ds, "selected_ch_names", None)
    if selected:
        return list(selected)
    if fallback is None:
        return None
    if len(fallback) != ds.n_channels:
        raise ValueError(
            f"Inferred {len(fallback)} channel names but the dataset yields "
            f"{ds.n_channels} channels; set common_ch_names in the dataset config."
        )
    return list(fallback)


def make_data_info(ds, cfg: dict, ch_names: list[str] | None) -> dict:
    """``data_info`` dict expected by ``src.models.build_model``."""
    info = {
        "n_chans": ds.n_channels,
        "n_classes": ds.n_classes,
        "n_times": ds.n_timesteps,
        "sfreq": float(cfg.get("sfreq", 200)),
    }
    if ch_names is not None:
        info["ch_names"] = list(ch_names)
    return info
