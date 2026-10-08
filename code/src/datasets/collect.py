"""Utility to collect a SubjectSplitDataset into numpy arrays."""

from __future__ import annotations

import logging

import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def collect_numpy_split(
    dataset: Dataset,
    desc: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterate a SubjectSplitDataset and return stacked numpy arrays."""
    n = len(dataset)
    label_str = f" ({desc})" if desc else ""
    logger.info(f"Collecting {n} windows{label_str} into numpy arrays...")

    X_list, y_list, sid_list = [], [], []
    for i in range(n):
        x_t, y_t, sid_t = dataset[i]
        X_list.append(x_t.numpy())      # (C, T)
        y_list.append(int(y_t.item()))
        sid_list.append(int(sid_t.item()))

    X = np.stack(X_list, axis=0).astype(np.float32)   # (N, C, T)
    y = np.array(y_list, dtype=np.int64)               # (N,)
    subject_ids = np.array(sid_list, dtype=np.int64)   # (N,)

    logger.info(
        f"  {desc}: X={X.shape}  y={y.shape}  "
        f"classes={sorted(set(y.tolist()))}  "
        f"subjects={len(set(sid_list))}"
    )
    return X, y, subject_ids
