"""Preprocessor for ds003775 — SRM Resting-state EEG (Healthy Controls)."""

from __future__ import annotations

import json
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

# All subjects are healthy controls
LABEL_MAP: dict[str, int] = {"control": 0}

# BIDS type → MNE channel type mapping (same as FEP preprocessor)
_BIDS_TO_MNE_TYPE: dict[str, str] = {
    "EEG":   "eeg",
    "EOG":   "eog",
    "ECG":   "ecg",
    "EMG":   "emg",
    "MISC":  "misc",
    "STIM":  "stim",
    "REF":   "misc",
    "OTHER": "misc",
}

# Default ICLabel artifact types to reject
_DEFAULT_EXCLUDE_LABELS: list[str] = [
    "muscle artifact",
    "eye blink",
    "heart beat",
    "channel noise",
    "line noise",
]

# BioSemi session sub-directory names to search (in preference order)
_SESSION_DIRS = ("ses-t1", "ses-t2")


class SRM_Preprocessor(BasePreprocessor):
    """Preprocessing pipeline for ds003775 (SRM Resting-state EEG)."""

    # ------------------------------------------------------------------
    # Helper: read subject metadata
    # ------------------------------------------------------------------

    def _build_subject_info(self) -> dict[str, dict]:
        """Read participants.tsv and return {sub_folder: {'label': int, 'pid': int}}."""
        raw_data_path = Path(self.cfg["raw_data_path"])
        tsv_path = raw_data_path / "participants.tsv"

        if not tsv_path.exists():
            raise FileNotFoundError(f"participants.tsv not found at {tsv_path}")

        df = pd.read_csv(tsv_path, sep="\t")

        sub_info: dict[str, dict] = {}
        for row in df.itertuples(index=False):
            sub_id = str(getattr(row, "participant_id", row[0])).strip()
            if not sub_id.startswith("sub-"):
                continue
            pid = numeric_subject_id(sub_id)
            sub_info[sub_id] = {"label": 0, "pid": pid}

        logger.info(
            f"Subjects in participants.tsv: {len(sub_info)} "
            f"(all healthy controls, label=0)"
        )

        # ── Single-subject mode (for SLURM array jobs) ──────────────
        subject_id = self.cfg.get("subject_id")
        if subject_id is not None:
            if subject_id not in sub_info:
                raise ValueError(
                    f"subject_id '{subject_id}' not found in participants.tsv. "
                    f"Available: {sorted(sub_info.keys())}"
                )
            sub_info = {subject_id: sub_info[subject_id]}
            logger.info(f"Single-subject mode: processing only {subject_id}")

        return sub_info

    # ------------------------------------------------------------------
    # Helper: fix BioSemi channel names via BIDS channels.tsv
    # ------------------------------------------------------------------

    def _fix_channels(self, raw: mne.io.Raw, edf_path: Path) -> None:
        """Rename BioSemi internal channel labels (A1–A32, B1–B32 …) to
        10-10 standard names and apply the standard_1020 montage.
        """
        # Derive *_channels.tsv path:
        #   sub-01_ses-t1_task-Rest_eeg.edf → sub-01_ses-t1_task-Rest_channels.tsv
        stem = edf_path.stem  # e.g. "sub-01_ses-t1_task-Rest_eeg"
        if stem.endswith("_eeg"):
            stem = stem[: -len("_eeg")]
        channels_tsv = edf_path.parent / f"{stem}_channels.tsv"

        # ── Log channel names BEFORE renaming ──────────────────────────
        logger.info(
            f"    Channels BEFORE rename ({len(raw.ch_names)}): "
            f"{raw.ch_names}"
        )

        if channels_tsv.exists():
            ch_df = pd.read_csv(channels_tsv, sep="\t")
            all_names = ch_df["name"].tolist()
            all_types = ch_df["type"].str.upper().tolist()

            n_raw = len(raw.ch_names)
            n_tsv = len(all_names)

            if n_tsv == n_raw:
                # 1. Rename all channels
                rename_map = {old: new for old, new in zip(raw.ch_names, all_names)}
                raw.rename_channels(rename_map)
                logger.info(
                    f"    Channels AFTER rename ({n_raw}): {raw.ch_names}"
                )

                # 2. Set correct channel types from tsv
                type_map: dict[str, str] = {}
                for ch_name, bids_type in zip(all_names, all_types):
                    mne_type = _BIDS_TO_MNE_TYPE.get(bids_type)
                    if mne_type:
                        type_map[ch_name] = mne_type
                if type_map:
                    raw.set_channel_types(type_map, verbose=False)
                    n_eeg = sum(1 for t in type_map.values() if t == "eeg")
                    n_eog = sum(1 for t in type_map.values() if t == "eog")
                    n_misc = sum(1 for t in type_map.values() if t == "misc")
                    logger.info(
                        f"    Channel types set: {n_eeg} EEG, "
                        f"{n_eog} EOG, {n_misc} misc"
                    )
            else:
                logger.warning(
                    f"    Channel count mismatch: raw={n_raw}, tsv={n_tsv} "
                    "— skipping rename. ICLabel may fail."
                )
        else:
            logger.warning(
                f"    No _channels.tsv sidecar found at {channels_tsv}. "
                "BioSemi channel names will NOT be renamed — ICLabel may fail."
            )

        # ── Apply standard_1020 montage (required for ICLabel) ─────────
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)
        logger.info("    Montage set: standard_1020")

    # ------------------------------------------------------------------
    # Helper: load raw EDF recordings for a single subject
    # ------------------------------------------------------------------

    def _load_recordings(
        self, sub_id: str, info: dict
    ) -> list[tuple[mne.io.Raw, str, str]]:
        """Locate and load all .edf files for *sub_id*, across sessions."""
        raw_data_path = Path(self.cfg["raw_data_path"])
        results: list[tuple[mne.io.Raw, str, str]] = []

        # Build candidate EEG directories (session sub-dirs first, then flat)
        candidate_dirs: list[tuple[Path, str]] = []
        for ses in _SESSION_DIRS:
            p = raw_data_path / sub_id / ses / "eeg"
            if p.exists():
                candidate_dirs.append((p, ses))
        # Fallback: no session subdirectory
        flat_eeg = raw_data_path / sub_id / "eeg"
        if flat_eeg.exists() and not candidate_dirs:
            candidate_dirs.append((flat_eeg, "ses-t1"))

        if not candidate_dirs:
            logger.warning(
                f"  {sub_id}: no EEG directory found "
                f"(searched {_SESSION_DIRS} + flat layout) — skipping."
            )
            return results

        for eeg_dir, ses_label in candidate_dirs:
            edf_files = sorted(f for f in os.listdir(eeg_dir) if f.endswith(".edf"))
            if not edf_files:
                logger.warning(
                    f"  {sub_id}/{ses_label}: no .edf files found in {eeg_dir}"
                )
                continue

            for fname in edf_files:
                file_path = eeg_dir / fname
                logger.info(f"    Loading {ses_label}/{fname}")
                try:
                    raw = mne.io.read_raw_edf(
                        str(file_path), preload=True, verbose=False
                    )
                    self._fix_channels(raw, file_path)
                except Exception as exc:
                    logger.error(f"    Failed to load {file_path}: {exc}")
                    continue
                results.append((raw, fname, ses_label))

        return results

    # ------------------------------------------------------------------
    # Helper: build braindecode BaseDatasets
    # ------------------------------------------------------------------

    def _load_base_datasets(self, sub_info: dict) -> list[BaseDataset]:
        """Load all MNE recordings and wrap in braindecode BaseDataset."""
        datasets: list[BaseDataset] = []
        failed: dict[str, str] = {}

        for sub_id, info in sorted(sub_info.items()):
            logger.info(f"  Subject {sub_id} (label={info['label']})")
            recordings = self._load_recordings(sub_id, info)

            if not recordings:
                reason = "no usable .edf recordings found"
                logger.warning(f"  {sub_id}: {reason} — skipping.")
                failed[sub_id] = reason
                continue

            for raw, fname, ses_label in recordings:
                description = {
                    "subject":      info["pid"],
                    "target":       info["label"],
                    "subject_name": sub_id,
                    "session":      ses_label,
                    "filename":     fname,
                }
                ds = BaseDataset(raw, description=description, target_name="target")
                datasets.append(ds)

        logger.info(
            f"Total recordings loaded: {len(datasets)} | "
            f"Subjects with load failures: {len(failed)}"
        )
        if failed:
            logger.warning(
                "Subjects not preprocessed (data issue):\n  "
                + "\n  ".join(f"{k}: {v}" for k, v in failed.items())
            )
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

        # Re-apply montage (EEGPrep's internal conversion may drop it)
        n_eeg = len(mne.pick_types(raw.info, eeg=True))
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)

        # Cap n_components to available EEG channels
        eff_n = min(n_components, n_eeg - 1)
        if eff_n < n_components:
            logger.warning(
                f"  ICA: n_components capped {n_components}→{eff_n} "
                f"(only {n_eeg} EEG channels remain after EEGPrep)"
            )

        raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        raw_hp.set_montage(montage, match_case=False, on_missing="warn", verbose=False)

        ica = ICA(
            n_components=eff_n,
            method="fastica",
            random_state=42,
            max_iter="auto",
        )
        ica.fit(raw_hp, verbose=False)

        labels_info = icalabel_label_components(raw_hp, ica, method="iclabel")
        component_labels: list[str] = labels_info["labels"]
        proba = labels_info["y_pred_proba"]

        exclude_idx = [
            i for i, lbl in enumerate(component_labels)
            if lbl in exclude_labels and proba[i].max() >= prob_threshold
        ]
        ica.exclude = exclude_idx
        logger.info(
            f"  ICA: excluded {len(exclude_idx)}/{eff_n} components "
            f"{[component_labels[i] for i in exclude_idx]}"
        )
        ica.apply(raw, verbose=False)
        return len(exclude_idx)

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
        preprocessors: list[Any] = []

        # 3a. Keep EEG channels only
        preprocessors.append(
            Preprocessor("pick_types", eeg=True, meg=False, stim=False, verbose=False)
        )

        # 3b. Scale V → µV
        scale = self.cfg.get("scale")
        if scale is not None:
            _s = float(scale)
            preprocessors.append(Preprocessor(lambda x, s=_s: x * s))

        # 3c. EEGPrep (DC offset · resample · flatline · drift HP ·
        #               RANSAC · ASR · bad-window rejection ·
        #               reinterpolation · avg re-reference)
        if self.cfg.get("eegprep", True):
            ep_kw: dict[str, Any] = {
                "resample_to":               float(self.cfg.get("eegprep_resample_to", 200)),
                "burst_removal_cutoff":      float(self.cfg.get("eegprep_burst_cutoff", 20)),
                "bad_channel_corr_threshold": float(
                    self.cfg.get("eegprep_corr_threshold", 0.75)
                ),
                "common_avg_ref":            bool(self.cfg.get("eegprep_common_avg_ref", True)),
                "bad_channel_reinterpolate": True,
            }
            logger.info(f"EEGPrep config: {ep_kw}")
            preprocessors.append(EEGPrep(**ep_kw))

        # 3d. Bandpass filter (applied AFTER EEGPrep / at resampled rate)
        l_freq = self.cfg.get("l_freq")
        h_freq = self.cfg.get("h_freq")
        if l_freq is not None or h_freq is not None:
            preprocessors.append(
                Preprocessor("filter", l_freq=l_freq, h_freq=h_freq, verbose=False)
            )

        # 3e. Notch filter (50 Hz European power line)
        notch_freq = self.cfg.get("notch_freq")
        if notch_freq is not None:
            preprocessors.append(
                Preprocessor("notch_filter", freqs=float(notch_freq), verbose=False)
            )

        logger.info(f"Applying {len(preprocessors)} preprocessing step(s)...")
        preprocess(concat_ds, preprocessors, n_jobs=int(self.cfg.get("n_jobs", 1)))

        # ── 4. ICA per recording ──────────────────────────────────────
        ica_excluded: dict[str, int] = {}
        apply_ica = self.cfg.get("apply_ica", True)

        if apply_ica:
            if not _ICALABEL_AVAILABLE:
                logger.warning(
                    "apply_ica=true but mne-icalabel is not installed — skipping."
                )
            else:
                n_comp = int(self.cfg.get("ica_n_components", 20))
                prob_thr = float(self.cfg.get("ica_prob_threshold", 0.8))
                excl_lbls: list[str] = list(
                    self.cfg.get("ica_exclude_labels", _DEFAULT_EXCLUDE_LABELS)
                )

                logger.info(
                    f"Applying ICA per recording "
                    f"(n_components={n_comp}, threshold={prob_thr})"
                )
                for ds in concat_ds.datasets:
                    sub_name = ds.description.get("subject_name", "unknown")
                    ses = ds.description.get("session", "")
                    fname = ds.description.get("filename", "")
                    key = f"{sub_name}/{ses}/{fname}"
                    logger.info(f"  ICA: {key}")
                    ica_excluded[key] = self._apply_ica(
                        ds.raw, n_comp, prob_thr, excl_lbls
                    )

                mean_excl = sum(ica_excluded.values()) / max(len(ica_excluded), 1)
                logger.info(
                    f"ICA done. Mean excluded: {mean_excl:.1f} components/recording."
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
            + windows_ds.description[["subject", "target", "session"]].to_string()
        )

        # ── 6. Save braindecode dataset ───────────────────────────────
        logger.info(f"Saving WindowsDataset → {out_dir} ...")
        windows_ds.save(str(out_dir), overwrite=True)

        # ── 7. Save preprocess metadata ───────────────────────────────
        self.save_meta(
            {
                "dataset_id":               self.cfg["dataset_id"],
                "label_map":                LABEL_MAP,
                "n_subjects_processed":     len(sub_info),
                "seg_len":                  seg_len,
                "overlap":                  overlap,
                "stride":                   stride,
                "n_chans":                  n_chans,
                "sfreq":                    actual_sfreq,
                "l_freq":                   l_freq,
                "h_freq":                   h_freq,
                "notch_freq":               notch_freq,
                "scale":                    scale,
                "eegprep":                  bool(self.cfg.get("eegprep", True)),
                "eegprep_resample_to":      self.cfg.get("eegprep_resample_to", 200),
                "eegprep_burst_cutoff":     self.cfg.get("eegprep_burst_cutoff", 20),
                "eegprep_corr_threshold":   self.cfg.get("eegprep_corr_threshold", 0.75),
                "apply_ica":                apply_ica,
                "ica_n_components":         self.cfg.get("ica_n_components", 20),
                "ica_prob_threshold":       self.cfg.get("ica_prob_threshold", 0.8),
                "ica_excluded_per_subject": ica_excluded,
                "n_windows":                n_windows,
                "n_recordings":             len(datasets),
                "save_folder":              self.cfg["save_folder"],
                "status":                   "success",
            }
        )
        logger.info("✓ Preprocessing complete.")
