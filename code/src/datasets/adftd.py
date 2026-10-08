"""ADFTD Dataset — ds004504 (Alzheimer / FTD / Healthy Controls)."""

from __future__ import annotations

import logging

import numpy as np
import torch

from .base import SubjectSplitDataset

logger = logging.getLogger(__name__)

# Original ADFTD labels from preprocessing
_LABEL_HC = 0
_LABEL_AD = 1
_LABEL_FTD = 2

_CLASSIFY_CHOICES = frozenset(
    ["multi_class", "ad_vs_hc", "ftd_vs_hc", "hc_vs_abnormal", "ad_vs_nonad"]
)

# Diagnosis labels kept by each classify_choice (used to filter subject splits).
ADFTD_KEEP_LABELS: dict[str, set[int]] = {
    "multi_class": {_LABEL_HC, _LABEL_AD, _LABEL_FTD},
    "ad_vs_hc": {_LABEL_HC, _LABEL_AD},
    "ftd_vs_hc": {_LABEL_HC, _LABEL_FTD},
    "hc_vs_abnormal": {_LABEL_HC, _LABEL_AD, _LABEL_FTD},
    "ad_vs_nonad": {_LABEL_HC, _LABEL_AD, _LABEL_FTD},
}


class ADFTDDataset(SubjectSplitDataset):
    """ADFTD dataset with classify_choice support."""

    def __init__(
        self,
        windows_ds,
        window_indices: list[int],
        classify_choice: str = "multi_class",
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(windows_ds, window_indices, normalize=normalize, **kwargs)

        if classify_choice not in _CLASSIFY_CHOICES:
            raise ValueError(
                f"Invalid classify_choice='{classify_choice}'. "
                f"Choose from: {sorted(_CLASSIFY_CHOICES)}"
            )

        self.classify_choice = classify_choice
        self.window_indices, self._label_remap = self._apply_classify_choice(
            self.window_indices, classify_choice
        )
        logger.info(
            f"classify_choice='{classify_choice}' → "
            f"{len(self.window_indices)} windows, "
            f"n_classes={len(set(self._label_remap.values()))}"
        )

    def _apply_classify_choice(
        self, window_indices: np.ndarray, classify_choice: str
    ) -> tuple[np.ndarray, dict[int, int]]:
        """Return filtered indices and label remap dict."""
        all_labels = {wi: self._win_label[wi] for wi in window_indices}

        if classify_choice == "multi_class":
            keep_labels = {_LABEL_HC, _LABEL_AD, _LABEL_FTD}
            remap = {_LABEL_HC: 0, _LABEL_AD: 1, _LABEL_FTD: 2}

        elif classify_choice == "ad_vs_hc":
            keep_labels = {_LABEL_HC, _LABEL_AD}
            remap = {_LABEL_HC: 0, _LABEL_AD: 1}

        elif classify_choice == "ftd_vs_hc":
            keep_labels = {_LABEL_HC, _LABEL_FTD}
            remap = {_LABEL_HC: 0, _LABEL_FTD: 1}

        elif classify_choice == "hc_vs_abnormal":
            keep_labels = {_LABEL_HC, _LABEL_AD, _LABEL_FTD}
            remap = {_LABEL_HC: 0, _LABEL_AD: 1, _LABEL_FTD: 1}

        elif classify_choice == "ad_vs_nonad":
            keep_labels = {_LABEL_HC, _LABEL_AD, _LABEL_FTD}
            remap = {_LABEL_HC: 0, _LABEL_FTD: 0, _LABEL_AD: 1}

        else:
            raise ValueError(f"Unknown classify_choice: {classify_choice}")

        filtered = np.array(
            [wi for wi, lbl in all_labels.items() if lbl in keep_labels],
            dtype=np.int64,
        )
        return filtered, remap

    def __getitem__(self, idx: int):
        X, label_tensor, sid_tensor = super().__getitem__(idx)
        remapped = self._label_remap.get(label_tensor.item(), label_tensor.item())
        return X, torch.tensor(remapped, dtype=torch.long), sid_tensor

    @property
    def n_classes(self) -> int:
        return len(set(self._label_remap.values()))
