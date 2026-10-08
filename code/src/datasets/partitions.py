"""The downstream participant pool — one definition for every model family."""

from __future__ import annotations

import logging

from .adftd import ADFTD_KEEP_LABELS
from .base import filter_sids_by_label, pretrain_reserve_for

logger = logging.getLogger(__name__)


def downstream_subject_ids(windows_ds, cfg: dict) -> list[int]:
    """Sorted participant IDs of the downstream pool for ``cfg``'s dataset/task."""
    reserve = pretrain_reserve_for(windows_ds, cfg)
    sids = {
        int(s) for s in windows_ds.description["subject"].unique() if int(s) not in reserve
    }
    classify_choice = cfg.get("classify_choice")
    if cfg.get("dataset_name") == "adftd" and classify_choice:
        sids = filter_sids_by_label(windows_ds, sids, ADFTD_KEEP_LABELS[classify_choice])
    pool = sorted(sids)
    logger.info(
        f"Downstream pool: {len(pool)} participants "
        f"({len(reserve)} pretraining-reserve participants excluded) — {pool}"
    )
    return pool
