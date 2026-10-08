"""SCZ Dataset — Schizophrenia Resting-State EEG (Zenodo 14808296)."""

from __future__ import annotations

import logging

from .base import SubjectSplitDataset

logger = logging.getLogger(__name__)


class SCZDataset(SubjectSplitDataset):
    """Schizophrenia vs Healthy Controls resting-state EEG dataset."""

    def __init__(
        self,
        windows_ds,
        window_indices: list[int],
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(windows_ds, window_indices, normalize=normalize, **kwargs)
        logger.info(
            f"SCZDataset: {len(self.window_indices)} windows, "
            f"n_classes={self.n_classes}"
        )

    @property
    def n_classes(self) -> int:
        return 2  # always binary for SCZ
