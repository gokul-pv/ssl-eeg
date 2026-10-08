"""Persistence of participant-level partitions (JSON files under ``metadata/``)."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_OUT_DIR = Path("metadata")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _split_path(
    dataset_id: str,
    seed: int,
    out_dir: str | Path = _DEFAULT_OUT_DIR,
    task: str | None = None,
) -> Path:
    """Return the canonical path for a split JSON file."""
    suffix = f"_{task}" if task else ""
    return Path(out_dir) / dataset_id / f"split_seed{seed}{suffix}.json"


def _cv_split_path(
    dataset_id: str,
    seed: int,
    out_dir: str | Path = _DEFAULT_OUT_DIR,
    task: str | None = None,
    strategy: str | None = None,
) -> Path:
    """Return the canonical path for a CV split JSON file."""
    suffix = f"_{task}" if task else ""
    strategy_suffix = f"_{strategy}" if strategy else ""
    return Path(out_dir) / dataset_id / f"cv_seed{seed}{suffix}{strategy_suffix}.json"


def _pretrain_reserve_path(
    dataset_id: str,
    seed: int,
    out_dir: str | Path = _DEFAULT_OUT_DIR,
) -> Path:
    """Return the canonical path for a pretrain-reserve pool JSON file."""
    return Path(out_dir) / dataset_id / f"pretrain_reserve_seed{seed}.json"


# ---------------------------------------------------------------------------
# Raw subject-ID cross-referencing
# ---------------------------------------------------------------------------


def _master_table_path(dataset_id: str, metadata_dir: str | Path) -> Path:
    """``metadata/<id>/master_subject_table_<id>.csv``, matched case-insensitively
    (LEMON's table is ``master_subject_table_lemon.csv``; Linux file systems are
    case-sensitive).
    """
    folder = Path(metadata_dir) / dataset_id
    expected = f"master_subject_table_{dataset_id}.csv"
    if folder.is_dir():
        for candidate in folder.iterdir():
            if candidate.name.lower() == expected.lower():
                return candidate
    return folder / expected


def _load_subject_id_map(
    dataset_id: str,
    metadata_dir: str | Path = "metadata",
) -> dict[int, list[str]]:
    """Map each integer subject pid → the raw BIDS subject-ID string(s) it came from."""
    csv_path = _master_table_path(dataset_id, metadata_dir)
    if not csv_path.exists():
        logger.warning(
            f"No master_subject_table found at {csv_path} — split JSON will "
            "only contain integer subject IDs, not raw BIDS subject strings."
        )
        return {}

    mapping: dict[int, list[str]] = {}
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if "subject_id" not in (reader.fieldnames or []):
                logger.warning(
                    f"{csv_path} has no 'subject_id' column — skipping ID enrichment."
                )
                return {}
            for row in reader:
                raw_id = row["subject_id"]
                if not raw_id:
                    continue
                digits = "".join(filter(str.isdigit, raw_id))
                if not digits:
                    continue
                pid = int(digits)
                raw_list = mapping.setdefault(pid, [])
                if raw_id not in raw_list:
                    raw_list.append(raw_id)
    except OSError as exc:
        logger.warning(f"Could not read {csv_path}: {exc} — skipping ID enrichment.")
        return {}

    return mapping


def _enrich_sids(sids: set[int] | list[int], id_map: dict[int, list[str]]) -> list[dict]:
    """Pair each sorted pid with its raw subject-ID string(s), if known."""
    return [
        {"pid": pid, "subject_ids": id_map.get(pid, [])}
        for pid in sorted(int(s) for s in sids)
    ]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_subject_split(
    dataset_id: str,
    seed: int,
    train_sids: set[int] | list[int],
    val_sids: set[int] | list[int],
    test_sids: set[int] | list[int],
    cross_val: str = "mccv",
    task: str | None = None,
    out_dir: str | Path = _DEFAULT_OUT_DIR,
    overwrite: bool = False,
) -> Path:
    """Persist a subject-level train/val/test split to JSON on disk."""
    path = _split_path(dataset_id, seed, out_dir, task=task)

    if path.exists() and not overwrite:
        logger.info(
            f"Split file already exists at {path} — skipping save. "
            "Set overwrite=True to replace it."
        )
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    id_map = _load_subject_id_map(dataset_id, metadata_dir=out_dir)

    payload = {
        "dataset_id": dataset_id,
        "seed": seed,
        "cross_val": cross_val,
        "task": task,
        "train_subject_ids": sorted(int(s) for s in train_sids),
        "val_subject_ids": sorted(int(s) for s in val_sids),
        "test_subject_ids": sorted(int(s) for s in test_sids),
        "train_subjects": _enrich_sids(train_sids, id_map),
        "val_subjects": _enrich_sids(val_sids, id_map),
        "test_subjects": _enrich_sids(test_sids, id_map),
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        f"Saved subject split → {path}  "
        f"(train={len(payload['train_subject_ids'])} "
        f"val={len(payload['val_subject_ids'])} "
        f"test={len(payload['test_subject_ids'])} subjects)"
    )
    return path


def save_rolling_cv_split(
    dataset_id: str,
    seed: int,
    runs_sids: list[tuple[set[int] | list[int], set[int] | list[int], set[int] | list[int]]],
    task: str | None = None,
    out_dir: str | Path = _DEFAULT_OUT_DIR,
    overwrite: bool = False,
    strategy: str | None = None,
) -> Path:
    """Persist rolling 5-fold CV subject-level splits to JSON on disk."""
    path = _cv_split_path(dataset_id, seed, out_dir, task=task, strategy=strategy)

    if path.exists() and not overwrite:
        logger.info(
            f"CV split file already exists at {path} — skipping save. "
            "Set overwrite=True to replace it."
        )
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    id_map = _load_subject_id_map(dataset_id, metadata_dir=out_dir)

    payload = {
        "dataset_id": dataset_id,
        "seed": seed,
        "task": task,
        "strategy": strategy or "rolling",
        "n_folds": len(runs_sids),
        "folds": [
            {
                "run": run_i + 1,
                "train_subject_ids": sorted(int(s) for s in train_sids),
                "val_subject_ids": sorted(int(s) for s in val_sids),
                "test_subject_ids": sorted(int(s) for s in test_sids),
                "train_subjects": _enrich_sids(train_sids, id_map),
                "val_subjects": _enrich_sids(val_sids, id_map),
                "test_subjects": _enrich_sids(test_sids, id_map),
            }
            for run_i, (train_sids, val_sids, test_sids) in enumerate(runs_sids)
        ],
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Saved rolling CV split → {path}  ({payload['n_folds']} folds)")
    return path


def save_pretrain_reserve(
    dataset_id: str,
    seed: int,
    reserve_sids: set[int] | list[int],
    out_dir: str | Path = _DEFAULT_OUT_DIR,
    overwrite: bool = False,
) -> Path:
    """Persist a pretrain-reserve subject pool to JSON on disk."""
    path = _pretrain_reserve_path(dataset_id, seed, out_dir)

    if path.exists() and not overwrite:
        logger.info(
            f"Pretrain-reserve file already exists at {path} — skipping save. "
            "Set overwrite=True to replace it."
        )
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    id_map = _load_subject_id_map(dataset_id, metadata_dir=out_dir)

    payload = {
        "dataset_id": dataset_id,
        "seed": seed,
        "reserve_subject_ids": sorted(int(s) for s in reserve_sids),
        "reserve_subjects": _enrich_sids(reserve_sids, id_map),
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        f"Saved pretrain-reserve pool → {path}  "
        f"({len(payload['reserve_subject_ids'])} subjects)"
    )
    return path


def load_pretrain_reserve_sids(
    dataset_id: str,
    seed: int,
    out_dir: str | Path = _DEFAULT_OUT_DIR,
) -> set[int]:
    """Load the pretrain-reserve subject pool for a dataset, if it exists."""
    path = _pretrain_reserve_path(dataset_id, seed, out_dir)
    if not path.exists():
        return set()
    with open(path) as f:
        payload = json.load(f)
    return set(int(s) for s in payload["reserve_subject_ids"])


# ---------------------------------------------------------------------------
# Query / Load
# ---------------------------------------------------------------------------


def split_file_exists(
    dataset_id: str,
    seed: int,
    out_dir: str | Path = _DEFAULT_OUT_DIR,
    task: str | None = None,
) -> bool:
    """Return True if the split JSON file exists on disk."""
    return _split_path(dataset_id, seed, out_dir, task=task).exists()


def load_split(
    dataset_id: str,
    seed: int,
    out_dir: str | Path = _DEFAULT_OUT_DIR,
    task: str | None = None,
) -> dict:
    """Load the full split JSON and return the raw dict."""
    path = _split_path(dataset_id, seed, out_dir, task=task)
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. "
            "It is created on first use by the pretraining / training entrypoints."
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Window index lookup
# ---------------------------------------------------------------------------


def get_windows_for_subjects(
    windows_ds,
    subject_ids: set[int] | list[int],
) -> list[int]:
    """Return global window indices for all windows belonging to the given subjects."""
    sid_set = set(int(s) for s in subject_ids)
    desc = windows_ds.description.reset_index(drop=True)
    cum = windows_ds.cumulative_sizes  # list[int], length = n_recordings

    window_indices: list[int] = []
    for i, row in desc.iterrows():
        if int(row["subject"]) in sid_set:
            start = cum[i - 1] if i > 0 else 0
            end = cum[i]
            window_indices.extend(range(start, end))

    logger.debug(
        f"get_windows_for_subjects: {len(sid_set)} subjects → "
        f"{len(window_indices)} windows"
    )
    return window_indices
