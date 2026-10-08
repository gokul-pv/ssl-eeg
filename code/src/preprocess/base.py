"""Base preprocessor abstract class."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


def numeric_subject_id(sub_id: str) -> int:
    """Integer participant ID from the digits of a BIDS subject label."""
    digits = "".join(filter(str.isdigit, str(sub_id)))
    if not digits:
        raise ValueError(
            f"Cannot derive a numeric participant ID from subject label '{sub_id}'."
        )
    return int(digits)


class BasePreprocessor(ABC):
    """Abstract base for dataset preprocessors."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    @property
    def output_dir(self) -> Path:
        """Canonical output directory: <data_dir>/<dataset_id>/<save_folder>/"""
        return (
            Path(self.cfg["data_dir"])
            / self.cfg["dataset_id"]
            / self.cfg["save_folder"]
        )

    @abstractmethod
    def run(self) -> None:
        """Execute the full preprocessing pipeline and save to output_dir."""
        ...

    def save_meta(self, meta: dict) -> None:
        """Write the preprocessing metadata next to (not inside) the braindecode
        save directory.
        """
        if self.cfg.get("subject_id") is not None:
            tag = str(self.cfg["subject_id"])
        else:
            tag = Path(str(self.cfg["save_folder"])).name
        path = self.output_dir.parent / f"preprocess_meta_{tag.replace('/', '_')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved preprocess metadata → {path}")

    def log_cfg(self) -> None:
        logger.info("Preprocess config:")
        for k, v in self.cfg.items():
            logger.info(f"  {k:30s} = {v}")
