"""Multi-dataset pre-training dataloader for self-supervised EEG learning."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from .base import (
    get_or_create_pretrain_reserve,
    get_subject_split,
    load_windows_dataset,
    normalize_ch_name as _normalize_ch_name,
)
from .split_io import (
    get_windows_for_subjects,
    load_split,
    save_subject_split,
    split_file_exists,
)

logger = logging.getLogger(__name__)

_DEFAULT_SPLIT_DIR = "metadata"
_EXPECTED_N_TIMES = 3000  # 15 s × 200 Hz


# ---------------------------------------------------------------------------
# Channel name helpers
# ---------------------------------------------------------------------------


# The ACNS 10-20 synonym normaliser (_normalize_ch_name) is imported from .base —
# the same helper SubjectSplitDataset applies when selecting common_ch_names, so the
# pretraining intersection and downstream channel selection can never disagree.


# Canonical 10-20 spatial order: anterior → posterior, left → right within each row.
# Used to sort common_ch_names so that channel index i always corresponds to the same
# scalp region, preserving spatial topology for interpretability and future spatial models.
_STANDARD_1020_ORDER: list[str] = [
    "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1", "OZ", "O2",
]
_STANDARD_1020_RANK: dict[str, int] = {ch: i for i, ch in enumerate(_STANDARD_1020_ORDER)}


def _sort_channels_spatially(channels: set[str]) -> list[str]:
    """Sort channel names by standard 10-20 spatial order (anterior→posterior, L→R)."""
    known = sorted(
        [ch for ch in channels if ch in _STANDARD_1020_RANK],
        key=lambda ch: _STANDARD_1020_RANK[ch],
    )
    unknown = sorted(ch for ch in channels if ch not in _STANDARD_1020_RANK)
    return known + unknown


def _get_recording_ch_names(base_dataset) -> list[str]:
    """Return the uppercase and standardised EEG channel names for a single braindecode BaseDataset.
    """
    raw = getattr(base_dataset, "raw", None)
    if raw is not None:
        ch_names = getattr(raw, "ch_names", None)
        if ch_names is not None:
            return [_normalize_ch_name(ch) for ch in ch_names]
    raise RuntimeError(
        "Cannot recover channel names from braindecode BaseDataset. "
        "Ensure datasets are loaded with preload=True so that raw.ch_names is available. "
        "Channel-name intersection (Option B) requires raw.ch_names on every recording."
    )


def _compute_global_ch_intersection(all_ch_names: list[list[str]]) -> list[str]:
    """Compute the strict global intersection of channel names across all recordings."""
    if not all_ch_names:
        return []
    common = set(all_ch_names[0])
    for names in all_ch_names[1:]:
        common &= set(names)
    return _sort_channels_spatially(common)



# ---------------------------------------------------------------------------
# PretrainDataset
# ---------------------------------------------------------------------------


class PretrainDataset(Dataset):
    """PyTorch Dataset for self-supervised pretraining on a single EEG source."""

    def __init__(
        self,
        windows_ds,
        window_indices: list[int],
        normalize: bool = True,
        dataset_id: str = "",
        expected_n_times: int = _EXPECTED_N_TIMES,
        return_unnormalized: bool = False,
    ) -> None:
        self.windows_ds = windows_ds
        self.return_unnormalized = return_unnormalized
        self.window_indices = np.array(window_indices, dtype=np.int64)
        self.normalize = normalize
        self.dataset_id = dataset_id
        self._expected_n_times = expected_n_times

        # common_ch_names set externally by MultiDatasetPretrainDataset
        self._common_ch_names: list[str] | None = None
        # Per-recording: list of indices into that recording's ch list
        # Populated once common_ch_names is set.
        self._rec_ch_indices: list[list[int]] = []

        # ── Collect channel names per recording ──────────────────────────
        self._rec_ch_names: list[list[str]] = []
        for bd in windows_ds.datasets:
            self._rec_ch_names.append(_get_recording_ch_names(bd))

        # ── Build global-window → recording index map (O(n_recordings)) ──
        desc = windows_ds.description.reset_index(drop=True)
        cum = windows_ds.cumulative_sizes
        self._win_to_rec: dict[int, int] = {}
        for i, _ in desc.iterrows():
            start = cum[i - 1] if i > 0 else 0
            end = cum[i]
            for wi in range(start, end):
                self._win_to_rec[wi] = i

        # ── Validate time dimension using first window ────────────────────
        first_idx = int(self.window_indices[0])
        X_check, _, _ = windows_ds[first_idx]
        n_times = np.asarray(X_check).shape[1]
        if n_times != expected_n_times:
            raise ValueError(
                f"[{dataset_id}] Time dimension mismatch: "
                f"expected {expected_n_times} samples, "
                f"got {n_times} for window index {first_idx}. "
                f"Check that all datasets were preprocessed with seg_len={expected_n_times}."
            )
        self._n_times = n_times

    # ── Common channel name interface ─────────────────────────────────────

    def set_common_ch_names(self, common_ch_names: list[str]) -> None:
        """Set the intersection channel list and pre-compute per-recording
        channel selection indices.
        """
        self._common_ch_names = common_ch_names
        self._rec_ch_indices = []

        for rec_idx, rec_names in enumerate(self._rec_ch_names):
            rec_name_to_pos = {ch: pos for pos, ch in enumerate(rec_names)}
            indices = []
            for ch in common_ch_names:
                if ch not in rec_name_to_pos:
                    raise ValueError(
                        f"[{self.dataset_id}] Recording #{rec_idx} is missing "
                        f"channel '{ch}' from the common intersection. "
                        "This should not happen if intersection was computed "
                        "from the same recordings. Check for data inconsistency."
                    )
                indices.append(rec_name_to_pos[ch])
            self._rec_ch_indices.append(indices)

        logger.info(
            f"PretrainDataset [{self.dataset_id}]: "
            f"{len(self.window_indices):,} windows | "
            f"T={self._n_times} | "
            f"C_common={len(common_ch_names)} (from {self.native_c_range})"
        )

    @property
    def common_ch_names(self) -> list[str]:
        """Intersection channel names (must be set via set_common_ch_names)."""
        if self._common_ch_names is None:
            raise RuntimeError(
                f"[{self.dataset_id}] common_ch_names not set. "
                "Wrap in MultiDatasetPretrainDataset or call set_common_ch_names() manually."
            )
        return self._common_ch_names

    @property
    def c_common(self) -> int:
        """Number of channels in the intersection."""
        return len(self.common_ch_names)

    @property
    def native_c_range(self) -> str:
        """String summary of native channel count range across recordings."""
        counts = [len(names) for names in self._rec_ch_names]
        if not counts:
            return "N/A"
        return f"min={min(counts)}, max={max(counts)}"

    @property
    def n_times(self) -> int:
        """Window length in samples."""
        return self._n_times

    # ── Standard Dataset interface ─────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Fetch, normalise, and select intersection channels from one EEG window."""
        real_idx = int(self.window_indices[idx])
        rec_idx = self._win_to_rec[real_idx]
        sel_indices = self._rec_ch_indices[rec_idx]

        # braindecode returns (X_np [C_i, T], y_scalar, crop_inds)
        X_np, _, _ = self.windows_ds[real_idx]
        X = np.asarray(X_np, dtype=np.float32)  # (C_i, T)

        # ── Sanity-check time dimension ──────────────────────────────────
        T = X.shape[1]
        if T != self._n_times:
            raise ValueError(
                f"[{self.dataset_id}] Runtime time-dimension mismatch "
                f"at window {real_idx}: expected {self._n_times}, got {T}."
            )

        # ── Select intersection channels ─────────────────────────────────
        X_sel = X[sel_indices, :]  # (C_common, T)
        X_unnormalized = X_sel

        # ── Per-channel z-score normalisation ────────────────────────────
        if self.normalize:
            mean = X_sel.mean(axis=-1, keepdims=True)   # (C_common, 1)
            std = X_sel.std(axis=-1, keepdims=True)      # (C_common, 1)
            std[std == 0.0] = 1.0                        # avoid /0
            X_sel = (X_sel - mean) / std

        if self.return_unnormalized:
            return torch.from_numpy(X_sel), torch.from_numpy(np.ascontiguousarray(X_unnormalized))
        return (torch.from_numpy(X_sel),)               # (C_common, T)


# ---------------------------------------------------------------------------
# MultiDatasetPretrainDataset
# ---------------------------------------------------------------------------


class MultiDatasetPretrainDataset(ConcatDataset):
    """Combines multiple ``PretrainDataset`` instances into one joint dataset."""

    def __init__(self, pretrain_datasets: list[PretrainDataset],
                 common_ch_names: list[str] | None = None) -> None:
        if not pretrain_datasets:
            raise ValueError("pretrain_datasets must be a non-empty list.")

        # ── Validate time dimension consistency ────────────────────────────
        n_times_vals = {ds.n_times for ds in pretrain_datasets}
        if len(n_times_vals) != 1:
            mismatch = {ds.dataset_id: ds.n_times for ds in pretrain_datasets}
            raise ValueError(
                f"All PretrainDatasets must have the same n_times. "
                f"Found mismatches: {mismatch}"
            )
        self.n_times: int = n_times_vals.pop()

        # ── Compute strict global channel intersection ────────────────────
        all_ch_names: list[list[str]] = []
        for ds in pretrain_datasets:
            all_ch_names.extend(ds._rec_ch_names)

        if common_ch_names is not None:
            self.common_ch_names: list[str] = list(common_ch_names)
        else:
            self.common_ch_names = _compute_global_ch_intersection(all_ch_names)
        self.c_common: int = len(self.common_ch_names)

        if self.c_common == 0:
            raise ValueError(
                "Channel intersection is empty! No channel appears in every "
                "recording across all datasets. Check that all datasets were "
                "preprocessed with compatible channel sets."
            )

        # ── Channel intersection report (Case-Insensitive Global Intersection) ──
        logger.info("=" * 70)
        logger.info("Channel Global Intersection Report (Standardised Casing)")
        logger.info("=" * 70)
        for ds in pretrain_datasets:
            ds_union = set()
            for rec_names in ds._rec_ch_names:
                ds_union.update(rec_names)
            dropped = sorted(list(ds_union - set(self.common_ch_names)))
            counts = [len(n) for n in ds._rec_ch_names]
            
            logger.info(
                f"  {ds.dataset_id:>12} | {len(ds._rec_ch_names):>4} recordings | "
                f"ch_per_recording: min={min(counts)} max={max(counts)} | "
                f"union_size={len(ds_union)}"
            )
            if dropped:
                logger.info(
                    f"    Dropped {len(dropped)} channel(s) from union: {dropped[:10]}"
                    + (f" ... +{len(dropped) - 10} more" if len(dropped) > 10 else "")
                )
            else:
                logger.info("    Dropped: 0 channels (all in intersection)")
        logger.info("-" * 70)
        logger.info(f"  Global intersection: {self.c_common} channels")
        logger.info(f"  Common channels: {self.common_ch_names}")
        logger.info("=" * 70)

        # ── Set common channels on each sub-dataset ────────────────────────
        for ds in pretrain_datasets:
            ds.set_common_ch_names(self.common_ch_names)

        # ConcatDataset init (provides __len__ / __getitem__ / index routing)
        super().__init__(pretrain_datasets)


    # ── Balanced sampler ──────────────────────────────────────────────────

    def build_weighted_sampler(self) -> WeightedRandomSampler:
        """Build a ``WeightedRandomSampler`` that gives each *dataset* equal
        expected weight per epoch, regardless of dataset size.
        """
        weights: list[float] = []
        for ds in self.datasets:
            n = len(ds)
            if n == 0:
                continue
            per_sample_weight = 1.0 / n
            weights.extend([per_sample_weight] * n)

        total = sum(weights)
        weights_norm = [w / total for w in weights]

        return WeightedRandomSampler(
            weights=weights_norm,
            num_samples=len(self),
            replacement=True,
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _load_pretrain_split(
    windows_ds,
    entry: dict,
    cfg: dict,
) -> list[int]:
    """Return global window indices for one dataset entry in the pretrain config."""
    split = entry.get("split", "all")
    dataset_id = entry["dataset_id"]

    if split == "all":
        all_idx = list(range(len(windows_ds)))
        logger.info(
            f"[{dataset_id}] split=all → using all {len(all_idx):,} windows."
        )
        return all_idx

    elif split in ("train", "val"):
        pretrain_seed = int(cfg.get("pretrain_seed", 42))
        split_dir = cfg.get("split_save_dir", _DEFAULT_SPLIT_DIR)
        field = "train_subject_ids" if split == "train" else "val_subject_ids"

        from src.config import load_config
        ds_cfg = load_config(entry["dataset_config"])
        task = ds_cfg.get("classify_choice")

        if split_file_exists(dataset_id, pretrain_seed, split_dir, task=task):
            payload = load_split(dataset_id, pretrain_seed, split_dir, task=task)
            sids = set(int(s) for s in payload[field])
            idx = get_windows_for_subjects(windows_ds, sids)
            logger.info(
                f"[{dataset_id}] split={split} → loaded from disk: "
                f"{len(sids)} subjects, {len(idx):,} windows."
            )
        else:
            suffix = f"_{task}" if task else ""
            logger.warning(
                f"[{dataset_id}] No saved split found at "
                f"'{split_dir}/{dataset_id}/split_seed{pretrain_seed}{suffix}.json'. "
                "Computing it now with pretrain_seed and saving for future runs."
            )
            ds_cfg["seed"] = pretrain_seed
            train_idx, val_idx, test_idx, train_sids, val_sids, test_sids = (
                get_subject_split(windows_ds, ds_cfg, return_sids=True)
            )
            save_subject_split(
                dataset_id=dataset_id,
                seed=pretrain_seed,
                train_sids=train_sids,
                val_sids=val_sids,
                test_sids=test_sids,
                cross_val=str(ds_cfg.get("cross_val", "mccv")),
                task=task,
                out_dir=split_dir,
            )
            idx = train_idx if split == "train" else val_idx

        return idx

    elif split == "pretrain_reserve":
        pretrain_seed = int(cfg.get("pretrain_seed", 42))
        split_dir = cfg.get("split_save_dir", _DEFAULT_SPLIT_DIR)

        reserve_sids = get_or_create_pretrain_reserve(
            windows_ds, dataset_id, seed=pretrain_seed, out_dir=split_dir
        )
        reserve_idx = get_windows_for_subjects(windows_ds, reserve_sids)
        logger.info(
            f"[{dataset_id}] split=pretrain_reserve → "
            f"{len(reserve_sids)} subjects, {len(reserve_idx):,} windows."
        )
        return reserve_idx

    else:
        raise ValueError(
            f"[{dataset_id}] Unknown split='{split}'. "
            "Use 'all', 'train', 'val', or 'pretrain_reserve'."
        )


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def build_pretrain_dataset(cfg: dict, common_ch_names: list[str] | None = None) -> MultiDatasetPretrainDataset:
    """Build a ``MultiDatasetPretrainDataset`` from a pretrain config dict."""
    from src.config import load_config  # avoid circular import
    import os

    entries = cfg.get("pretrain_datasets", [])
    if not entries:
        raise ValueError(
            "cfg['pretrain_datasets'] is empty. "
            "Specify at least one dataset entry in the pretrain config."
        )

    # ── Dynamically override n_jobs for parallel loading ─────────────────
    # If SLURM allocates multiple CPUs, or num_workers is set, we leverage
    # Joblib's multi-process loading (each subject shard is loaded in parallel).
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    default_n_jobs = int(slurm_cpus) if slurm_cpus else int(cfg.get("num_workers", 4))
    n_jobs = int(cfg.get("n_jobs", default_n_jobs))
    if n_jobs < 1:
        n_jobs = 1

    logger.info("=" * 70)
    logger.info(f"Loading pretraining datasets (parallel loading n_jobs={n_jobs})")
    logger.info("=" * 70)

    normalize = bool(cfg.get("normalize", True))
    return_unnormalized = bool(cfg.get("return_unnormalized", False))

    pretrain_datasets: list[PretrainDataset] = []

    for entry in entries:
        dataset_id = entry["dataset_id"]
        ds_config_path = entry.get("dataset_config")

        logger.info(f"Loading dataset: {dataset_id} (config: {ds_config_path})")

        if ds_config_path:
            ds_cfg = load_config(ds_config_path)
        else:
            ds_cfg = {}
        ds_cfg.update({k: v for k, v in entry.items() if k != "dataset_config"})
        
        # Override dataset-specific n_jobs with our high-throughput value
        ds_cfg["n_jobs"] = n_jobs

        windows_ds = load_windows_dataset(ds_cfg)
        window_indices = _load_pretrain_split(windows_ds, entry, cfg)

        pretrain_ds = PretrainDataset(
            windows_ds=windows_ds,
            window_indices=window_indices,
            normalize=normalize,
            dataset_id=dataset_id,
            expected_n_times=int(cfg.get("expected_n_times", _EXPECTED_N_TIMES)),
            return_unnormalized=return_unnormalized,
        )
        pretrain_datasets.append(pretrain_ds)

    return MultiDatasetPretrainDataset(pretrain_datasets, common_ch_names=common_ch_names)



def build_pretrain_dataloader(
    multi_ds: MultiDatasetPretrainDataset,
    cfg: dict,
) -> DataLoader:
    """Build a ``DataLoader`` for the combined pretraining dataset."""
    use_balanced = bool(cfg.get("pretrain_balanced_sampling", False))
    num_workers = int(cfg.get("num_workers", 0))
    batch_size = int(cfg.get("batch_size", 64))
    persistent = bool(cfg.get("persistent_workers", False)) and num_workers > 0

    if use_balanced:
        sampler = multi_ds.build_weighted_sampler()
        shuffle = False
        logger.info(
            f"PretrainDataLoader: WeightedRandomSampler enabled "
            f"({len(multi_ds.datasets)} datasets, equal weight per dataset)."
        )
    else:
        sampler = None
        shuffle = True
        logger.info(
            "PretrainDataLoader: standard random sampler (shuffle=True)."
        )

    def _collate_fn(batch):
        """Collate ``(X,)`` / ``(X, X_unnormalized)`` items → stacked tuples."""
        return tuple(torch.stack([item[i] for item in batch]) for i in range(len(batch[0])))

    return DataLoader(
        multi_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=True,     # avoids incomplete last batch during MAE training
        collate_fn=_collate_fn,
    )
