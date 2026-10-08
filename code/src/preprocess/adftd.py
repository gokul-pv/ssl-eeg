"""Preprocessor for ds004504 — ADFTD Resting State EEG."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

import mne
import pandas as pd
from mne.preprocessing import ICA

from braindecode.datasets import BaseConcatDataset
from braindecode.datasets.base import BaseDataset
from braindecode.preprocessing import (
    EEGPrep,
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)

try:
    from mne_icalabel import label_components as icalabel_label_components
    _ICALABEL_AVAILABLE = True
except ImportError:
    _ICALABEL_AVAILABLE = False
    warnings.warn(
        "mne-icalabel is not installed; ICA step will be skipped. "
        "Install with: pip install mne-icalabel",
        ImportWarning,
        stacklevel=2,
    )

from .base import BasePreprocessor, numeric_subject_id

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

LABEL_MAP = {
    "alzheimer": 1,
    "frontotemporaldementia": 2,
    "control": 0,
}

# Default ICLabel artifact types to reject
_DEFAULT_EXCLUDE_LABELS: list[str] = [
    "muscle artifact",
    "eye blink",
    "heart beat",
    "channel noise",
    "line noise",
]


class ADFTD_Preprocessor(BasePreprocessor):
    """Preprocessing pipeline for ds004504 (ADFTD-RS) with EEGPrep + ICA."""

    # ------------------------------------------------------------------
    # Helper: read subject metadata
    # ------------------------------------------------------------------

    def _build_subject_info(self) -> dict[str, dict]:
        master_file = Path(self.cfg["metadata_file"])
        df = pd.read_csv(master_file, sep=",")

        missing = {"subject_id", "diagnosis"} - set(df.columns)
        if missing:
            raise ValueError(f"{master_file} is missing column(s) {sorted(missing)}")

        sub_info: dict[str, dict] = {}
        for row in df.itertuples(index=False):
            sub_name = row.subject_id   # e.g. "sub-001"
            diag_code = row.diagnosis   # e.g. "alzheimer" / "frontotemporaldementia" / "control"
            pid = numeric_subject_id(sub_name)
            label = LABEL_MAP.get(diag_code)
            if label is None:
                logger.warning(
                    f"Unknown diagnosis '{diag_code}' for {sub_name} — skipping."
                )
                continue
            sub_info[sub_name] = {"label": label, "pid": pid}

        logger.info(f"Subjects in metadata: {len(sub_info)}")

        # ── Single-subject mode (for SLURM array jobs) ──────────────
        subject_id = self.cfg.get("subject_id")
        if subject_id is not None:
            if subject_id not in sub_info:
                raise ValueError(
                    f"subject_id '{subject_id}' not found in metadata. "
                    f"Available: {sorted(sub_info.keys())}"
                )
            sub_info = {subject_id: sub_info[subject_id]}
            logger.info(f"Single-subject mode: processing only {subject_id}")

        logger.info(
            f"Subjects to process: {len(sub_info)} "
            f"(HC={sum(1 for v in sub_info.values() if v['label'] == 0)}, "
            f"AD={sum(1 for v in sub_info.values() if v['label'] == 1)}, "
            f"FTD={sum(1 for v in sub_info.values() if v['label'] == 2)})"
        )
        return sub_info

    # ------------------------------------------------------------------
    # Helper: load raw recordings
    # ------------------------------------------------------------------

    def _fix_montage(self, raw: mne.io.Raw) -> None:
        """Apply standard 10-20 montage to EEGLAB .set recordings."""
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)
        logger.info("    Montage set: standard_1020")

    def _load_base_datasets(self, sub_info: dict) -> list[BaseDataset]:
        """Load all raw MNE recordings and wrap in braindecode BaseDataset."""
        raw_data_path = Path(self.cfg["raw_data_path"])
        derivatives_root = raw_data_path / "derivatives"

        datasets: list[BaseDataset] = []
        for sub in sorted(sub_info.keys()):
            # sub_path = derivatives_root / sub / "eeg"
            sub_path = raw_data_path / sub / "eeg"
            if not sub_path.exists():
                logger.warning(f"EEG folder not found for {sub}: {sub_path} — skipping.")
                continue

            info = sub_info[sub]
            logger.info(f"  Subject {sub} (label={info['label']})")

            set_files = sorted(f for f in os.listdir(sub_path) if f.endswith(".set"))
            if not set_files:
                logger.warning(f"  No .set files found for {sub}.")
                continue

            for fname in set_files:
                file_path = sub_path / fname
                logger.info(f"    Loading {fname}")
                try:
                    raw = mne.io.read_raw_eeglab(
                        str(file_path), preload=True, verbose=False
                    )
                    self._fix_montage(raw)
                except Exception as exc:
                    logger.error(f"    Failed to load {file_path}: {exc}")
                    continue

                description = {
                    "subject":      info["pid"],
                    "target":       info["label"],
                    "subject_name": sub,
                    "filename":     fname,
                }
                ds = BaseDataset(raw, description=description, target_name="target")
                datasets.append(ds)

        logger.info(f"Total recordings loaded: {len(datasets)}")
        return datasets

    # ------------------------------------------------------------------
    # Helper: ICA per recording
    # ------------------------------------------------------------------

    def _apply_ica(
        self,
        raw: mne.io.Raw,
        n_components: int,
        prob_threshold: float,
        exclude_labels: list[str],
    ) -> int:
        """Fit ICA on a 1 Hz high-pass copy of *raw* and apply it to *raw*."""
        if not _ICALABEL_AVAILABLE:
            logger.warning("mne-icalabel not available — skipping ICA.")
            return 0

        # ── Re-apply montage (EEGPrep's internal conversion may drop it) ──
        n_eeg = len(mne.pick_types(raw.info, eeg=True))
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)

        # ── Cap n_components to available EEG channels ─────────────────────
        effective_n_components = min(n_components, n_eeg - 1)
        if effective_n_components < n_components:
            logger.warning(
                f"  ICA: reduced n_components {n_components} → {effective_n_components} "
                f"(only {n_eeg} EEG channels remain after EEGPrep)"
            )

        logger.debug(
            f"  ICA: fitting on 1 Hz HP copy (n_components={effective_n_components})"
        )
        raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        raw_hp.set_montage(montage, match_case=False, on_missing="warn", verbose=False)

        ica = ICA(
            n_components=effective_n_components,
            method="fastica",
            random_state=42,
            max_iter="auto",
        )
        ica.fit(raw_hp, verbose=False)

        # Automatic component labeling via ICLabel
        labels_info = icalabel_label_components(raw_hp, ica, method="iclabel")
        component_labels: list[str] = labels_info["labels"]
        proba_matrix = labels_info["y_pred_proba"]  # shape (n_components, n_classes)

        exclude_idx = [
            i for i, lbl in enumerate(component_labels)
            if lbl in exclude_labels
            and proba_matrix[i].max() >= prob_threshold
        ]
        ica.exclude = exclude_idx

        n_excluded = len(exclude_idx)
        logger.info(
            f"  ICA: excluded {n_excluded}/{effective_n_components} components "
            f"{[component_labels[i] for i in exclude_idx]}"
        )

        # Apply to the ORIGINAL broadband signal
        ica.apply(raw, verbose=False)
        return n_excluded

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.log_cfg()
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {out_dir}")

        # ── 1. Load subject metadata ──────────────────────────────────
        sub_info = self._build_subject_info()

        # ── 2. Load raw MNE recordings ────────────────────────────────
        logger.info("Loading raw EEG recordings...")
        datasets = self._load_base_datasets(sub_info)
        if not datasets:
            raise RuntimeError(
                "No datasets loaded. Check 'raw_data_path' in config."
            )
        concat_ds = BaseConcatDataset(datasets)

        # ── 3. Braindecode preprocessing chain ────────────────────────
        preprocessors: list[Preprocessor | EEGPrep] = []

        # 3a. Keep EEG channels only
        preprocessors.append(
            Preprocessor("pick_types", eeg=True, meg=False, stim=False, verbose=False)
        )

        # 3b. Scale V → µV
        scale = self.cfg.get("scale")
        if scale is not None:
            _scale = float(scale)
            preprocessors.append(Preprocessor(lambda x, s=_scale: x * s))

        # 3c. EEGPrep  (DC offset · resample · flatline · drift HP ·
        #               bad channels · ASR · bad windows ·
        #               reinterpolation · avg reference)
        if self.cfg.get("eegprep", True):
            eegprep_kwargs: dict[str, Any] = {
                "resample_to":               float(self.cfg.get("eegprep_resample_to", 200)),
                "burst_removal_cutoff":      float(self.cfg.get("eegprep_burst_cutoff", 20)),
                "bad_channel_corr_threshold": float(
                    self.cfg.get("eegprep_corr_threshold", 0.75)
                ),
                "common_avg_ref":            bool(self.cfg.get("eegprep_common_avg_ref", True)),
                "bad_channel_reinterpolate": True,
            }
            logger.info(f"EEGPrep config: {eegprep_kwargs}")
            preprocessors.append(EEGPrep(**eegprep_kwargs))

        # 3d. Bandpass filter (applied AFTER EEGPrep / at resampled rate)
        l_freq = self.cfg.get("l_freq")
        h_freq = self.cfg.get("h_freq")
        if l_freq is not None or h_freq is not None:
            preprocessors.append(
                Preprocessor("filter", l_freq=l_freq, h_freq=h_freq, verbose=False)
            )

        # 3e. Notch filter (50 Hz European line noise — optional)
        notch_freq = self.cfg.get("notch_freq")
        if notch_freq is not None:
            preprocessors.append(
                Preprocessor("notch_filter", freqs=float(notch_freq), verbose=False)
            )

        logger.info(f"Applying {len(preprocessors)} preprocessing step(s)...")
        preprocess(concat_ds, preprocessors, n_jobs=int(self.cfg.get("n_jobs", 1)))

        # ── 4. ICA per recording ──────────────────────────────────────
        ica_excluded_per_subject: dict[str, int] = {}
        apply_ica = self.cfg.get("apply_ica", True)

        if apply_ica:
            if not _ICALABEL_AVAILABLE:
                logger.warning(
                    "apply_ica=true but mne-icalabel is not installed. "
                    "Skipping ICA step."
                )
            else:
                n_components = int(self.cfg.get("ica_n_components", 20))
                prob_threshold = float(self.cfg.get("ica_prob_threshold", 0.8))
                exclude_labels: list[str] = list(
                    self.cfg.get("ica_exclude_labels", _DEFAULT_EXCLUDE_LABELS)
                )

                logger.info(
                    "Applying ICA per recording "
                    f"(n_components={n_components}, prob_threshold={prob_threshold})"
                )
                for ds in concat_ds.datasets:
                    sub_name = ds.description.get("subject_name", "unknown")
                    fname = ds.description.get("filename", "")
                    key = f"{sub_name}/{fname}"
                    logger.info(f"  ICA: {key}")
                    n_excl = self._apply_ica(
                        ds.raw, n_components, prob_threshold, exclude_labels
                    )
                    ica_excluded_per_subject[key] = n_excl

                logger.info(
                    "ICA done. "
                    "Mean excluded: "
                    f"{sum(ica_excluded_per_subject.values()) / max(len(ica_excluded_per_subject), 1):.1f} "
                    "components per recording."
                )

        # ── 5. Fixed-length windowing ─────────────────────────────────
        seg_len = int(self.cfg["seg_len"])
        overlap = int(self.cfg.get("overlap", 0))
        stride = seg_len - overlap
        if stride <= 0:
            raise ValueError(f"overlap ({overlap}) must be < seg_len ({seg_len})")

        logger.info(
            f"Creating windows: seg_len={seg_len} overlap={overlap} stride={stride}"
        )
        windows_ds = create_fixed_length_windows(
            concat_ds,
            start_offset_samples=0,
            stop_offset_samples=None,
            window_size_samples=seg_len,
            window_stride_samples=stride,
            drop_last_window=bool(self.cfg.get("drop_last_window", True)),
            preload=bool(self.cfg.get("preload", True)),
            n_jobs=int(self.cfg.get("n_jobs", 1)),
        )

        n_windows = len(windows_ds)
        n_chans = windows_ds[0][0].shape[0]
        actual_sfreq = concat_ds.datasets[0].raw.info["sfreq"]
        logger.info(
            f"Total windows: {n_windows} | Channels: {n_chans} | sfreq: {actual_sfreq} Hz"
        )
        logger.info(
            "Subject summary:\n"
            + windows_ds.description[["subject", "target"]].to_string()
        )

        # ── 6. Save braindecode dataset ───────────────────────────────
        logger.info(f"Saving WindowsDataset → {out_dir} ...")
        windows_ds.save(str(out_dir), overwrite=True)

        # ── 7. Save preprocess metadata ───────────────────────────────
        self.save_meta(
            {
                "dataset_id":                self.cfg["dataset_id"],
                "seg_len":                   seg_len,
                "overlap":                   overlap,
                "stride":                    stride,
                "n_chans":                   n_chans,
                "sfreq":                     actual_sfreq,
                "l_freq":                    l_freq,
                "h_freq":                    h_freq,
                "notch_freq":                notch_freq,
                "scale":                     scale,
                "eegprep":                   bool(self.cfg.get("eegprep", True)),
                "eegprep_resample_to":       self.cfg.get("eegprep_resample_to", 200),
                "eegprep_burst_cutoff":      self.cfg.get("eegprep_burst_cutoff", 20),
                "eegprep_corr_threshold":    self.cfg.get("eegprep_corr_threshold", 0.75),
                "apply_ica":                 apply_ica,
                "ica_n_components":          self.cfg.get("ica_n_components", 20),
                "ica_prob_threshold":        self.cfg.get("ica_prob_threshold", 0.8),
                "ica_excluded_per_subject":  ica_excluded_per_subject,
                "n_windows":                 n_windows,
                "n_subjects":                len(datasets),
                "label_map":                 LABEL_MAP,
                "save_folder":               self.cfg["save_folder"],
            }
        )
        logger.info("✓ Preprocessing complete.")
