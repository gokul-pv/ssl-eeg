"""MDD Dataset — ds003478 (Major Depressive Disorder vs Healthy Controls)."""

from __future__ import annotations

import logging

from .base import SubjectSplitDataset

logger = logging.getLogger(__name__)


class MDDDataset(SubjectSplitDataset):
    """MDD dataset for ds003478."""

    def __init__(self, windows_ds, window_indices: list[int], normalize: bool = True, **kwargs) -> None:
        super().__init__(windows_ds, window_indices, normalize=normalize, **kwargs)
        logger.info(
            f"MDDDataset: {len(self.window_indices)} windows, n_classes={self.n_classes}"
        )

    @property
    def n_classes(self) -> int:
        return 2
