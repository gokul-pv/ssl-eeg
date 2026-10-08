"""Base dataset that wraps a saved braindecode WindowsDataset."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from braindecode.datautil import load_concat_dataset
from braindecode.datasets import BaseConcatDataset

logger = logging.getLogger(__name__)


# Legacy -> modern 10-20 channel labels.
CH_SYNONYMS: dict[str, str] = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


def normalize_ch_name(ch: str) -> str:
    """Uppercase *ch* and map legacy 10-20 labels to their modern equivalents."""
    ch_upper = ch.strip().upper()
    return CH_SYNONYMS.get(ch_upper, ch_upper)


def _is_int(name: str) -> bool:
    """Return True if *name* can be cast to int (i.e. a braindecode data subdir)."""
    try:
        int(name)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Subject-level split helper
# ---------------------------------------------------------------------------


def _build_subject_to_windows(windows_ds) -> dict[int, list[int]]:
    """Maps subject_id → list of global window indices."""
    desc = windows_ds.description.reset_index(drop=True)
    cum = windows_ds.cumulative_sizes
    result: dict[int, list[int]] = {}
    for i, row in desc.iterrows():
        sid = int(row["subject"])
        start = cum[i - 1] if i > 0 else 0
        result.setdefault(sid, []).extend(range(start, cum[i]))
    return result


def _get_subjects_by_class(windows_ds) -> dict[int, list[int]]:
    """Maps class label → sorted list of subject IDs."""
    desc = windows_ds.description.reset_index(drop=True)
    by_class: dict[int, list[int]] = {}
    for _, row in desc.iterrows():
        label = int(row["target"])
        sid = int(row["subject"])
        by_class.setdefault(label, []).append(sid)
    return {label: sorted(set(sids)) for label, sids in by_class.items()}


def filter_sids_by_label(windows_ds, sids: set[int], keep_labels: set[int]) -> set[int]:
    """Drop any subject whose diagnosis (`target`) isn't in `keep_labels`."""
    desc = windows_ds.description.reset_index(drop=True)
    sid_label = {int(row["subject"]): int(row["target"]) for _, row in desc.iterrows()}
    return {sid for sid in sids if sid_label.get(sid) in keep_labels}


def get_pretrain_reserve_sids(
    windows_ds,
    reserve_frac: float = 0.2,
    seed: int = 42,
) -> set[int]:
    """Carve out a small, diagnosis-stratified pool of subjects reserved for
    pretraining only (never used in any downstream train/val/test split or
    rolling-CV fold).
    """
    subjects_by_class = _get_subjects_by_class(windows_ds)
    rng = random.Random(seed)
    reserve_sids: set[int] = set()
    for sids in subjects_by_class.values():
        sids = sorted(sids)
        rng.shuffle(sids)
        n_reserve = int(round(reserve_frac * len(sids)))
        reserve_sids.update(sids[:n_reserve])
    return reserve_sids


def get_or_create_pretrain_reserve(
    windows_ds,
    dataset_id: str,
    seed: int = 42,
    reserve_frac: float = 0.2,
    out_dir: str = "metadata",
) -> set[int]:
    """Load the saved pretrain-reserve pool for `dataset_id`, computing and
    saving it on first use.
    """
    from .split_io import load_pretrain_reserve_sids, save_pretrain_reserve

    reserve_sids = load_pretrain_reserve_sids(dataset_id, seed, out_dir)
    if reserve_sids:
        return reserve_sids

    reserve_sids = get_pretrain_reserve_sids(windows_ds, reserve_frac=reserve_frac, seed=seed)
    save_pretrain_reserve(dataset_id, seed, reserve_sids, out_dir=out_dir)
    return reserve_sids


def pretrain_reserve_for(windows_ds, cfg: dict) -> set[int]:
    """Pretraining-reserve participants of ``cfg["dataset_id"]`` for downstream use."""
    return get_or_create_pretrain_reserve(
        windows_ds,
        cfg["dataset_id"],
        seed=int(cfg.get("pretrain_seed", 42)),
        out_dir=cfg.get("split_save_dir", "metadata"),
    )


def get_subject_split(
    windows_ds,
    cfg: dict,
    return_sids: bool = False,
    exclude_sids: set[int] | None = None,
):
    """Compute window-level indices for train / val / test splits using a
    subject-level, class-stratified partition.
    """
    desc = windows_ds.description.reset_index(drop=True)
    cum_sizes = windows_ds.cumulative_sizes  # [n_win_0, n_win_0+n_win_1, ...]
    exclude_sids = exclude_sids or set()

    def _windows_for_sub_ds(i: int) -> list[int]:
        start = cum_sizes[i - 1] if i > 0 else 0
        end = cum_sizes[i]
        return list(range(start, end))

    # Build per-class subject lists. Each subject may have multiple rows in
    # `desc` (multiple sessions/runs) — dedupe to one entry per subject so a
    # subject can never be shuffled into two different cut-point buckets.
    subjects_by_class: dict[int, list[int]] = {}
    seen_subjects: set[int] = set()
    for i, row in desc.iterrows():
        sid = int(row["subject"])
        if sid in seen_subjects or sid in exclude_sids:
            continue
        seen_subjects.add(sid)
        label = int(row["target"])
        subjects_by_class.setdefault(label, []).append(sid)

    cross_val = cfg.get("cross_val", "fixed")
    ratio_a = float(cfg.get("ratio_a", 0.6))
    ratio_b = float(cfg.get("ratio_b", 0.8))
    seed = int(cfg.get("seed", 42))

    train_sids: set[int] = set()
    val_sids: set[int] = set()
    test_sids: set[int] = set()

    if cross_val in ("fixed", "mccv"):
        rng = random.Random(42 if cross_val == "fixed" else seed)
        for label, sids in subjects_by_class.items():
            sids = sorted(sids)
            rng.shuffle(sids)
            n = len(sids)
            n_train = int(ratio_a * n)
            n_val = int(ratio_b * n)
            train_sids.update(sids[:n_train])
            val_sids.update(sids[n_train:n_val])
            test_sids.update(sids[n_val:])

    else:
        raise ValueError(
            f"Unknown cross_val='{cross_val}'. Use 'fixed' or 'mccv' "
            "(LOSO and five-fold CV: get_loso_folds / get_rolling_cv_folds)."
        )

    overlap = (train_sids & val_sids) | (train_sids & test_sids) | (val_sids & test_sids)
    assert not overlap, f"Subject leakage detected across splits: {sorted(overlap)}"

    # Map sub-dataset index → set of window indices
    train_idx, val_idx, test_idx = [], [], []
    for i, row in desc.iterrows():
        sid = int(row["subject"])
        wins = _windows_for_sub_ds(i)
        if sid in train_sids:
            train_idx.extend(wins)
        if sid in val_sids:
            val_idx.extend(wins)
        if sid in test_sids:
            test_idx.extend(wins)

    logger.info(
        f"Split ({cross_val}) | "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} windows"
    )
    logger.info(f"  train subjects : {sorted(train_sids)}")
    logger.info(f"  val subjects   : {sorted(val_sids)}")
    logger.info(f"  test subjects  : {sorted(test_sids)}")

    if return_sids:
        return train_idx, val_idx, test_idx, train_sids, val_sids, test_sids
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class SubjectSplitDataset(Dataset):
    """PyTorch Dataset wrapping a braindecode BaseConcatDataset (WindowsDataset)."""

    def __init__(
        self,
        windows_ds,
        window_indices: list[int],
        normalize: bool = True,
        common_ch_names: list[str] | None = None,
    ) -> None:
        self.windows_ds = windows_ds
        self.window_indices = np.array(window_indices, dtype=np.int64)
        self.normalize = normalize
        self.common_ch_names = common_ch_names

        # Pre-build a fast lookup: window_global_idx → (subject_id, label)
        # using description + cumulative_sizes (avoids per-item searches)
        desc = windows_ds.description.reset_index(drop=True)
        cum = windows_ds.cumulative_sizes
        self._win_subject: dict[int, int] = {}
        self._win_label: dict[int, int] = {}
        self._win_to_rec: dict[int, int] = {}
        for i, row in desc.iterrows():
            start = cum[i - 1] if i > 0 else 0
            end = cum[i]
            for wi in range(start, end):
                self._win_subject[wi] = int(row["subject"])
                self._win_label[wi] = int(row["target"])
                self._win_to_rec[wi] = i

        # ── Pre-compute channel selection indices if common_ch_names is set ────
        # Channel names in the row order __getitem__ yields (None: recording order).
        self.selected_ch_names: list[str] | None = None
        self._rec_ch_indices: list[list[int]] = []
        if common_ch_names is not None:
            # Normalise to uppercase AND map legacy synonyms so legacy names in
            # common_ch_names (e.g. 'T3') correctly match recordings that use
            # the modern equivalent ('T7').
            upper_common = [normalize_ch_name(ch) for ch in common_ch_names]
            self.selected_ch_names = list(upper_common)

            for rec_idx, bd in enumerate(windows_ds.datasets):
                raw = getattr(bd, "raw", None)
                if raw is not None and getattr(raw, "ch_names", None) is not None:
                    # Normalise and map synonyms to match upper_common perfectly
                    rec_names = [normalize_ch_name(ch) for ch in raw.ch_names]
                else:
                    raise RuntimeError(
                        f"Cannot recover channel names for recording #{rec_idx} in "
                        f"SubjectSplitDataset. Ensure datasets are preloaded."
                    )
                
                rec_name_to_pos = {ch: pos for pos, ch in enumerate(rec_names)}
                indices = []
                for ch in upper_common:
                    if ch not in rec_name_to_pos:
                        raise ValueError(
                            f"Recording #{rec_idx} is missing common channel '{ch}' "
                            f"from the specified common list: {upper_common}."
                        )
                    indices.append(rec_name_to_pos[ch])
                self._rec_ch_indices.append(indices)

            logger.info(
                f"SubjectSplitDataset: selecting {len(upper_common)} common channels "
                f"across {len(windows_ds.datasets)} recordings. "
                f"Channels: {list(common_ch_names)}"
            )

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.window_indices[idx])
        rec_idx = self._win_to_rec[real_idx]

        # braindecode returns (X_np [C,T], y_scalar, crop_inds)
        X_np, y_raw, _ = self.windows_ds[real_idx]
        X = np.asarray(X_np, dtype=np.float32)  # (C, T)

        # ── Select common channels if provided ───────────────────────────
        if self.common_ch_names is not None:
            sel_indices = self._rec_ch_indices[rec_idx]
            X = X[sel_indices, :]

        if self.normalize:
            mean = X.mean(axis=-1, keepdims=True)
            std = X.std(axis=-1, keepdims=True)
            std[std == 0.0] = 1.0
            X = (X - mean) / std

        label = self._win_label.get(real_idx, int(y_raw))
        subject_id = self._win_subject.get(real_idx, -1)

        return (
            torch.from_numpy(X),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(subject_id, dtype=torch.long),
        )

    @property
    def n_channels(self) -> int:
        if self.common_ch_names is not None:
            return len(self.common_ch_names)
        X, _, _ = self.windows_ds[int(self.window_indices[0])]
        return X.shape[0]

    @property
    def n_timesteps(self) -> int:
        X, _, _ = self.windows_ds[int(self.window_indices[0])]
        return X.shape[1]

    @property
    def n_classes(self) -> int:
        labels = set(self._win_label[wi] for wi in self.window_indices)
        return len(labels)


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------


def load_windows_dataset(cfg: dict):
    """Load a saved braindecode WindowsDataset from disk."""
    dataset_path = Path(cfg["data_dir"]) / cfg["dataset_id"] / cfg["save_folder"]
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {dataset_path}. "
            f"Run preprocess.py first."
        )

    preload = bool(cfg.get("preload", True))
    n_jobs = int(cfg.get("n_jobs", 1))

    # ── Detect layout ─────────────────────────────────────────────────────
    # A flat braindecode dataset contains integer-named subdirectories (0, 1...)
    # A sharded layout contains subdirectories like "sub-1448".
    sub_dirs = sorted(p for p in dataset_path.iterdir() if p.is_dir())
    
    is_sharded = False
    if sub_dirs:
        # Sharded layout is indicated by at least one sub-directory whose name
        # is NOT a plain integer (braindecode's flat layout uses 0/, 1/, ...).
        is_sharded = any(not _is_int(d.name) for d in sub_dirs)

    if is_sharded:
        # ── Sharded layout: load each subject shard and merge ──────────────
        logger.info(
            f"Detected per-subject sharded layout in {dataset_path} "
            f"({len(sub_dirs)} shards). Merging into one dataset..."
        )
        all_datasets = []
        for shard_dir in sub_dirs:
            # Skip shards of excluded subjects (metadata only, no braindecode dirs).
            bd_subdirs = [p for p in shard_dir.iterdir() if p.is_dir() and _is_int(p.name)]
            if not bd_subdirs:
                logger.debug(
                    f"  Skipping shard {shard_dir.name}: no braindecode data dirs found "
                    f"(likely excluded/failed subject with only metadata)"
                )
                continue
            try:
                shard_ds = load_concat_dataset(
                    str(shard_dir),
                    preload=preload,
                    n_jobs=n_jobs,
                )
                all_datasets.extend(shard_ds.datasets)
                logger.debug(f"  Loaded shard: {shard_dir.name} ({len(shard_ds)} windows)")
            except Exception as exc:
                logger.warning(f"  Skipping shard {shard_dir.name}: {exc}")

        if not all_datasets:
            raise RuntimeError(
                f"No valid subject shards found under {dataset_path}. "
                "Check that preprocessing completed successfully for at least one subject."
            )

        windows_ds = BaseConcatDataset(all_datasets)
        logger.info(
            f"Merged {len(sub_dirs)} shards → {len(windows_ds)} windows "
            f"from {len(windows_ds.datasets)} recording(s)."
        )
    else:
        # ── Flat layout ────────────────────────────────────────────────────
        logger.info(f"Loading WindowsDataset from {dataset_path} ...")
        windows_ds = load_concat_dataset(
            str(dataset_path),
            preload=preload,
            n_jobs=n_jobs,
        )
        logger.info(
            f"Loaded {len(windows_ds)} windows from {len(windows_ds.datasets)} recording(s)."
        )

    return windows_ds


# ---------------------------------------------------------------------------
# Five-fold rolling cross-validation and leave-one-subject-out
# ---------------------------------------------------------------------------


def get_rolling_cv_folds(
    windows_ds,
    all_sids: list[int],
    cfg: dict,
    return_sids: bool = False,
):
    """Generate rolling 5-fold CV splits over all subjects (no outer test set)."""
    n_folds = int(cfg.get("n_folds", 5))
    if n_folds != 5:
        raise ValueError(
            f"get_rolling_cv_folds requires n_folds=5, got n_folds={n_folds}. "
            "The rolling pattern is defined for exactly 5 folds."
        )
    seed = int(cfg.get("seed", 42))

    all_sid_set = set(all_sids)
    subjects_by_class = _get_subjects_by_class(windows_ds)
    subjects_by_class = {
        label: [s for s in sids if s in all_sid_set]
        for label, sids in subjects_by_class.items()
    }
    sid_to_windows = _build_subject_to_windows(windows_ds)

    # Stratified subject buckets, one per fold (seeded shuffle within class).
    rng = random.Random(seed)
    buckets: list[set[int]] = [set() for _ in range(n_folds)]
    for label, sids in subjects_by_class.items():
        sids = sorted(sids)     # deterministic starting order
        rng.shuffle(sids)       # shuffle within class using seeded rng
        n = len(sids)
        if n == 0:
            continue
        bucket_size = n / n_folds
        for k in range(n_folds):
            start = int(round(k * bucket_size))
            end = int(round((k + 1) * bucket_size))
            buckets[k].update(sids[start:end])

    logger.info("Rolling CV fold assignment (subject-level, stratified):")
    for k, bkt in enumerate(buckets):
        logger.info(f"  Fold {k + 1}: {sorted(bkt)}")

    # Generate 5 runs with the rolling pattern.
    runs: list[tuple[list[int], list[int], list[int]]] = []
    runs_sids: list[tuple[set[int], set[int], set[int]]] = []
    for run_i in range(n_folds):
        test_k = (run_i + 4) % n_folds   # 0-indexed: folds 4,0,1,2,3
        val_k = (run_i + 3) % n_folds    # 0-indexed: folds 3,4,0,1,2
        train_ks = {k for k in range(n_folds) if k != test_k and k != val_k}

        train_sids: set[int] = set().union(*(buckets[k] for k in train_ks))
        val_sids: set[int] = buckets[val_k]
        test_sids: set[int] = buckets[test_k]

        train_win: list[int] = []
        val_win: list[int] = []
        test_win: list[int] = []
        for sid in all_sids:
            wins = sid_to_windows.get(sid, [])
            if sid in train_sids:
                train_win.extend(wins)
            elif sid in val_sids:
                val_win.extend(wins)
            elif sid in test_sids:
                test_win.extend(wins)

        logger.info(
            f"Rolling run {run_i + 1}/{n_folds} | "
            f"train folds={sorted(k + 1 for k in train_ks)} "
            f"val fold={val_k + 1} test fold={test_k + 1}"
        )
        logger.info(
            f"  train={len(train_win)} val={len(val_win)} test={len(test_win)} windows"
        )
        logger.info(f"  train subjects : {sorted(train_sids)}")
        logger.info(f"  val subjects   : {sorted(val_sids)}")
        logger.info(f"  test subjects  : {sorted(test_sids)}")

        runs.append((train_win, val_win, test_win))
        runs_sids.append((train_sids, val_sids, test_sids))

    if return_sids:
        return runs, runs_sids
    return runs


def get_loso_folds(
    windows_ds,
    all_sids: list[int],
    cfg: dict,
    return_sids: bool = False,
):
    """Generate leave-one-subject-out (LOSO) CV splits over all subjects."""
    n_subj = len(all_sids)
    if n_subj < 3:
        raise ValueError(
            f"get_loso_folds requires at least 3 subjects (need >=1 train, "
            f"1 val, 1 test), got {n_subj}."
        )

    sid_to_windows = _build_subject_to_windows(windows_ds)

    logger.info(f"LOSO CV fold assignment over {n_subj} subjects: {all_sids}")

    runs: list[tuple[list[int], list[int], list[int]]] = []
    runs_sids: list[tuple[set[int], set[int], set[int]]] = []
    for run_i in range(n_subj):
        test_sid = all_sids[run_i]
        val_sid = all_sids[(run_i + 1) % n_subj]
        train_sids = {s for s in all_sids if s not in (test_sid, val_sid)}
        val_sids = {val_sid}
        test_sids = {test_sid}

        train_win: list[int] = []
        val_win: list[int] = []
        test_win: list[int] = []
        for sid in all_sids:
            wins = sid_to_windows.get(sid, [])
            if sid in train_sids:
                train_win.extend(wins)
            elif sid in val_sids:
                val_win.extend(wins)
            elif sid in test_sids:
                test_win.extend(wins)

        logger.info(
            f"LOSO run {run_i + 1}/{n_subj} | "
            f"train={len(train_win)} val={len(val_win)} test={len(test_win)} windows | "
            f"val subject={val_sid} test subject={test_sid}"
        )

        runs.append((train_win, val_win, test_win))
        runs_sids.append((train_sids, val_sids, test_sids))

    if return_sids:
        return runs, runs_sids
    return runs
