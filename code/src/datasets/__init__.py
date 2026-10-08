"""Dataset registry and build functions."""

from __future__ import annotations

import logging

from torch.utils.data import DataLoader, Dataset

from .adftd import ADFTD_KEEP_LABELS, ADFTDDataset
from .base import (
    SubjectSplitDataset,
    _build_subject_to_windows,
    filter_sids_by_label,
    get_loso_folds,
    get_or_create_pretrain_reserve,
    get_pretrain_reserve_sids,
    get_rolling_cv_folds,
    get_subject_split,
    load_windows_dataset,
    pretrain_reserve_for,
)
from .collect import collect_numpy_split
from .partitions import downstream_subject_ids
from .fep import FEPDataset
from .mdd import MDDDataset
from .scz import SCZDataset
from .pretrain import (
    MultiDatasetPretrainDataset,
    PretrainDataset,
    build_pretrain_dataloader,
    build_pretrain_dataset,
)
from .split_io import (
    get_windows_for_subjects,
    load_pretrain_reserve_sids,
    save_pretrain_reserve,
    save_rolling_cv_split,
    save_subject_split,
    split_file_exists,
)

logger = logging.getLogger(__name__)

# ── Registry ──────────────────────────────────────────────────────────────
DATASET_REGISTRY: dict[str, type] = {
    "adftd": ADFTDDataset,
    "fep": FEPDataset,
    "mdd": MDDDataset,
    # scz = Schizophrenia vs HC resting-state EEG (Zenodo 14808296, Racz et al. 2025)
    "scz": SCZDataset,
}

# ── Public API ────────────────────────────────────────────────────────────


def build_dataset(
    windows_ds,
    window_indices: list[int],
    cfg: dict,
) -> SubjectSplitDataset:
    """Instantiate the appropriate Dataset class for the given cfg["dataset_name"]."""
    dataset_name = cfg.get("dataset_name")
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset_name='{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    cls = DATASET_REGISTRY[dataset_name]
    kwargs = dict(
        windows_ds=windows_ds,
        window_indices=window_indices,
        normalize=bool(cfg.get("normalize", True)),
    )

    # Pass dataset-specific kwargs
    if cls is ADFTDDataset:
        kwargs["classify_choice"] = cfg.get("classify_choice", "multi_class")

    # If common_ch_names is present in cfg, pass it to kwargs
    if "common_ch_names" in cfg:
        kwargs["common_ch_names"] = cfg["common_ch_names"]

    return cls(**kwargs)


def build_dataloader(dataset: Dataset, cfg: dict, split: str) -> DataLoader:
    """Build a PyTorch DataLoader for the given split."""
    is_train = split == "train"
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=is_train,
        drop_last=is_train,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=True,
        persistent_workers=bool(cfg.get("persistent_workers", False))
        and int(cfg.get("num_workers", 0)) > 0,
    )


__all__ = [
    # Supervised datasets
    "DATASET_REGISTRY",
    "ADFTDDataset",
    "ADFTD_KEEP_LABELS",
    "FEPDataset",
    "MDDDataset",
    "SCZDataset",
    "SubjectSplitDataset",
    # Self-supervised / pretraining
    "PretrainDataset",
    "MultiDatasetPretrainDataset",
    "build_pretrain_dataset",
    "build_pretrain_dataloader",
    # Split I/O
    "save_rolling_cv_split",
    "save_subject_split",
    "save_pretrain_reserve",
    "load_pretrain_reserve_sids",
    "split_file_exists",
    "get_windows_for_subjects",
    # Supervised build helpers
    "build_dataset",
    "build_dataloader",
    "collect_numpy_split",
    "get_rolling_cv_folds",
    "get_loso_folds",
    "get_subject_split",
    "get_pretrain_reserve_sids",
    "get_or_create_pretrain_reserve",
    "pretrain_reserve_for",
    "downstream_subject_ids",
    "filter_sids_by_label",
    "load_windows_dataset",
]
