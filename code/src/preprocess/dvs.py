"""Preprocessor for ds005385 — Dortmund Vital Study Resting-state EEG."""

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
        "mne-icalabel is not installed; ICA step will be skipped.",
        ImportWarning,
        stacklevel=2,
    )

from .base import BasePreprocessor, numeric_subject_id

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

# All subjects are healthy controls
LABEL_MAP: dict[str, int] = {"control": 0}

_BIDS_TO_MNE_TYPE: dict[str, str] = {
    "EEG": "eeg", "EOG": "eog", "ECG": "ecg",
    "EMG": "emg", "MISC": "misc", "STIM": "stim",
    "REF": "misc", "OTHER": "misc",
}

_DEFAULT_EXCLUDE_LABELS: list[str] = [
    "muscle artifact", "eye blink", "heart beat", "channel noise", "line noise",
]

# All four conditions available in the dataset
_ALL_CONDITIONS = [
    "task-EyesClosed_acq-pre",
    "task-EyesClosed_acq-post",
    "task-EyesOpen_acq-pre",
    "task-EyesOpen_acq-post",
]

# Session labels and their corresponding late-trigger column in participants.tsv
_SESSION_LATE_COL = {
    "ses-1": "late_ses1",
    "ses-2": "late_ses2",
}


class DVS_Preprocessor(BasePreprocessor):
    """Preprocessing pipeline for ds005385 (Dortmund Vital Study)."""

    # ------------------------------------------------------------------
    # Subject metadata
    # ------------------------------------------------------------------

    def _build_subject_info(
        self,
    ) -> tuple[dict[str, dict], dict[str, str]]:
        """Read participants.tsv → (retained_per_session, skipped)."""
        tsv = Path(self.cfg["raw_data_path"]) / "participants.tsv"
        if not tsv.exists():
            raise FileNotFoundError(f"participants.tsv not found: {tsv}")

        df = pd.read_csv(tsv, sep="\t")

        sessions_cfg: list[str] = list(
            self.cfg.get("sessions", ["ses-1", "ses-2"])
        )
        late_thresh = int(self.cfg.get("late_trigger_threshold", 0))

        sub_info: dict[str, dict] = {}
        skipped: dict[str, str] = {}

        n_tsv_subjects = 0
        n_participated_ses1 = 0
        n_participated_ses2 = 0
        n_skipped_ses1 = 0
        n_skipped_ses2 = 0

        for row in df.itertuples(index=False):
            sub_id = str(getattr(row, "participant_id", row[0])).strip()
            if not sub_id.startswith("sub-"):
                continue

            n_tsv_subjects += 1

            pid = numeric_subject_id(sub_id)

            valid_sessions: list[str] = []
            late_counts: dict[str, int] = {}

            for ses in sessions_cfg:
                late_col = _SESSION_LATE_COL.get(ses)
                # Check if this subject has this session at all
                ses_col = f"session{ses.split('-')[1]}"
                ses_present = str(getattr(row, ses_col, "no")).strip().lower()
                if ses_present != "yes":
                    continue  # subject has no data for this session

                if ses == "ses-1":
                    n_participated_ses1 += 1
                elif ses == "ses-2":
                    n_participated_ses2 += 1

                late = 0
                if late_col and hasattr(row, late_col):
                    raw_late = getattr(row, late_col)
                    try:
                        late = int(float(str(raw_late)))
                    except (ValueError, TypeError):
                        late = 0

                late_counts[ses] = late

                if late > late_thresh:
                    key = f"{sub_id}/{ses}"
                    reason = (
                        f"late triggers={late} > threshold={late_thresh} "
                        "(recording probably not continuous)"
                    )
                    logger.info(f"  {key}: skipped — {reason}")
                    skipped[key] = reason
                    if ses == "ses-1":
                        n_skipped_ses1 += 1
                    elif ses == "ses-2":
                        n_skipped_ses2 += 1
                else:
                    valid_sessions.append(ses)

            if valid_sessions:
                sub_info[sub_id] = {
                    "label": 0,
                    "pid": pid,
                    "sessions": valid_sessions,
                    "late": late_counts,
                }

        n_retained_subjects = len(sub_info)
        n_fully_skipped = n_tsv_subjects - n_retained_subjects
        n_retained_ses1 = n_participated_ses1 - n_skipped_ses1
        n_retained_ses2 = n_participated_ses2 - n_skipped_ses2

        logger.info(
            f"\n"
            f"================================================================\n"
            f"   DORTMUND VITAL STUDY (ds005385) COHORT SUMMARY\n"
            f"================================================================\n"
            f"  • Total unique subjects in participants.tsv: {n_tsv_subjects}\n"
            f"  • Subjects fully skipped (all recorded sessions bad): {n_fully_skipped}\n"
            f"  • Subjects retained (at least 1 clean session): {n_retained_subjects}\n"
            f"\n"
            f"  Session-level breakdown:\n"
            f"    - Session 1 (Baseline):\n"
            f"      * Total recorded: {n_participated_ses1}\n"
            f"      * Skipped (late triggers > {late_thresh}): {n_skipped_ses1}\n"
            f"      * Clean & retain: {n_retained_ses1}\n"
            f"    - Session 2 (5-year follow-up):\n"
            f"      * Total recorded: {n_participated_ses2}\n"
            f"      * Skipped (late triggers > {late_thresh}): {n_skipped_ses2}\n"
            f"      * Clean & retain: {n_retained_ses2}\n"
            f"================================================================"
        )

        # Single-subject SLURM mode
        subject_id = self.cfg.get("subject_id")
        if subject_id is not None:
            if subject_id not in sub_info:
                # Could be entirely skipped
                matching = {k: v for k, v in skipped.items()
                            if k.startswith(f"{subject_id}/")}
                if matching:
                    logger.info(
                        f"Subject '{subject_id}' has all sessions skipped "
                        f"(late triggers). Reasons: {matching}"
                    )
                    return {}, matching
                raise ValueError(
                    f"subject_id '{subject_id}' not found in participants.tsv."
                )
            sub_info = {subject_id: sub_info[subject_id]}
            logger.info(f"Single-subject mode: {subject_id}")

        return sub_info, skipped

    # ------------------------------------------------------------------
    # Channel renaming
    # ------------------------------------------------------------------

    def _fix_channels(self, raw: mne.io.Raw, edf_path: Path) -> None:
        """Rename BrainVision channel labels to 10-20 names via channels.tsv."""
        # sub-001_ses-1_task-EyesClosed_acq-pre_eeg.edf
        #   → sub-001_ses-1_task-EyesClosed_acq-pre_channels.tsv
        stem = edf_path.stem
        if stem.endswith("_eeg"):
            stem = stem[: -len("_eeg")]
        channels_tsv = edf_path.parent / f"{stem}_channels.tsv"

        logger.info(
            f"    Channels BEFORE rename ({len(raw.ch_names)}): {raw.ch_names}"
        )

        if channels_tsv.exists():
            ch_df = pd.read_csv(channels_tsv, sep="\t")
            all_names = ch_df["name"].tolist()
            all_types = ch_df["type"].str.upper().tolist()
            n_raw, n_tsv = len(raw.ch_names), len(all_names)

            # Often the raw EDF has an extra 'Status' channel not in the TSV
            # We rename the first N channels that match the TSV length
            if n_tsv <= n_raw:
                rename_map = {
                    old: new
                    for old, new in zip(raw.ch_names[:n_tsv], all_names)
                    if old != new
                }
                if rename_map:
                    raw.rename_channels(rename_map)
                    logger.info(f"    Renamed {len(rename_map)} channels via TSV")
                
                type_map = {
                    name: _BIDS_TO_MNE_TYPE.get(btype)
                    for name, btype in zip(raw.ch_names[:n_tsv], all_types)
                    if btype in _BIDS_TO_MNE_TYPE
                }
                
                # Automatically type 'Status' as stim if present
                if "Status" in raw.ch_names and "Status" not in type_map:
                    type_map["Status"] = "stim"

                if type_map:
                    raw.set_channel_types(type_map, verbose=False)
                    n_eeg = sum(1 for t in type_map.values() if t == "eeg")
                    logger.info(
                        f"    Types: {n_eeg} EEG, "
                        f"{sum(1 for t in type_map.values() if t=='eog')} EOG, "
                        f"{sum(1 for t in type_map.values() if t=='misc')} misc, "
                        f"{sum(1 for t in type_map.values() if t=='stim')} stim"
                    )
                
                logger.info(
                    f"    Channels AFTER rename ({len(raw.ch_names)}): {raw.ch_names}"
                )
            else:
                logger.warning(
                    f"    Channel count mismatch raw={n_raw} tsv={n_tsv} "
                    "— TSV has MORE channels than raw. Skipping rename/types."
                )
        else:
            logger.warning(
                f"    No _channels.tsv at {channels_tsv} — skipping rename."
            )

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)
        logger.info("    Montage set: standard_1020")

    # ------------------------------------------------------------------
    # Load recordings
    # ------------------------------------------------------------------

    def _load_base_datasets(
        self, sub_info: dict
    ) -> tuple[list[BaseDataset], dict[str, str]]:
        """Load all EDF recordings for each subject × session × condition."""
        raw_data_path = Path(self.cfg["raw_data_path"])
        conditions: list[str] = list(
            self.cfg.get("conditions", _ALL_CONDITIONS)
        )

        datasets: list[BaseDataset] = []
        failed: dict[str, str] = {}

        for sub_id, info in sorted(sub_info.items()):
            logger.info(f"  Subject {sub_id}")
            for ses in info["sessions"]:
                eeg_dir = raw_data_path / sub_id / ses / "eeg"
                if not eeg_dir.exists():
                    reason = f"EEG directory not found: {eeg_dir}"
                    logger.warning(f"  {sub_id}/{ses}: {reason}")
                    failed[f"{sub_id}/{ses}"] = reason
                    continue

                for condition in conditions:
                    # Expected filename pattern:
                    # sub-001_ses-1_task-EyesClosed_acq-pre_eeg.edf
                    pattern = f"{sub_id}_{ses}_{condition}_eeg.edf"
                    file_path = eeg_dir / pattern
                    if not file_path.exists():
                        # Try glob in case of minor naming variation
                        matches = sorted(eeg_dir.glob(f"*{condition}*_eeg.edf"))
                        if matches:
                            file_path = matches[0]
                        else:
                            logger.warning(
                                f"    {sub_id}/{ses}/{condition}: "
                                f"file not found — skipping."
                            )
                            failed[f"{sub_id}/{ses}/{condition}"] = "file not found"
                            continue

                    logger.info(f"    Loading {ses}/{condition}: {file_path.name}")
                    try:
                        raw = mne.io.read_raw_edf(
                            str(file_path),
                            preload=True,
                            stim_channel="auto",
                            verbose=False,
                        )
                        self._fix_channels(raw, file_path)
                    except Exception as exc:
                        reason = f"load error: {exc}"
                        logger.error(f"    {sub_id}/{ses}/{condition}: {reason}")
                        failed[f"{sub_id}/{ses}/{condition}"] = reason
                        continue

                    description = {
                        "subject":    info["pid"],
                        "target":     info["label"],
                        "subject_name": sub_id,
                        "session":    ses,
                        "condition":  condition,
                        "filename":   file_path.name,
                    }
                    ds = BaseDataset(raw, description=description, target_name="target")
                    datasets.append(ds)

        logger.info(
            f"Total recordings loaded: {len(datasets)} | "
            f"Failed: {len(failed)}"
        )
        if failed:
            logger.warning(
                "Failed recordings:\n  "
                + "\n  ".join(f"{k}: {v}" for k, v in failed.items())
            )
        return datasets, failed

    # ------------------------------------------------------------------
    # ICA
    # ------------------------------------------------------------------

    def _apply_ica(
        self,
        raw: mne.io.Raw,
        n_components: int,
        prob_threshold: float,
        exclude_labels: list[str],
    ) -> int:
        if not _ICALABEL_AVAILABLE:
            logger.warning("mne-icalabel not available — skipping ICA.")
            return 0

        n_eeg = len(mne.pick_types(raw.info, eeg=True))
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)

        eff_n = min(n_components, n_eeg - 1)
        if eff_n < n_components:
            logger.warning(f"  ICA: n_components capped {n_components}→{eff_n}")

        raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        raw_hp.set_montage(montage, match_case=False, on_missing="warn", verbose=False)

        ica = ICA(n_components=eff_n, method="fastica", random_state=42, max_iter="auto")
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
            f"  ICA: excluded {len(exclude_idx)}/{eff_n} "
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

        # 1. Subject metadata
        sub_info, skipped = self._build_subject_info()

        if not sub_info:
            logger.info("No subjects to process — saving metadata and exiting.")
            self.save_meta({
                "dataset_id":    self.cfg["dataset_id"],
                "skipped":       skipped,
                "n_skipped":     len(skipped),
                "status":        "skipped",
            })
            return

        # 2. Load recordings
        logger.info("Loading raw EEG recordings...")
        datasets, failed = self._load_base_datasets(sub_info)
        if not datasets:
            logger.warning("No recordings loaded — saving failure metadata.")
            self.save_meta({
                "dataset_id": self.cfg["dataset_id"],
                "skipped":    skipped,
                "failed":     failed,
                "status":     "failed",
            })
            return

        concat_ds = BaseConcatDataset(datasets)

        # 3. Preprocessing chain
        preprocessors: list[Any] = []
        preprocessors.append(
            Preprocessor("pick_types", eeg=True, meg=False, stim=False, verbose=False)
        )

        scale = self.cfg.get("scale")
        if scale is not None:
            _s = float(scale)
            preprocessors.append(Preprocessor(lambda x, s=_s: x * s))

        if self.cfg.get("eegprep", True):
            ep_kw: dict[str, Any] = {
                "resample_to":               float(self.cfg.get("eegprep_resample_to", 200)),
                "burst_removal_cutoff":      float(self.cfg.get("eegprep_burst_cutoff", 20)),
                "bad_channel_corr_threshold": float(self.cfg.get("eegprep_corr_threshold", 0.75)),
                "common_avg_ref":            bool(self.cfg.get("eegprep_common_avg_ref", True)),
                "bad_channel_reinterpolate": True,
            }
            logger.info(f"EEGPrep config: {ep_kw}")
            preprocessors.append(EEGPrep(**ep_kw))

        l_freq = self.cfg.get("l_freq")
        h_freq = self.cfg.get("h_freq")
        if l_freq is not None or h_freq is not None:
            preprocessors.append(
                Preprocessor("filter", l_freq=l_freq, h_freq=h_freq, verbose=False)
            )

        notch_freq = self.cfg.get("notch_freq")
        if notch_freq is not None:
            preprocessors.append(
                Preprocessor("notch_filter", freqs=float(notch_freq), verbose=False)
            )

        logger.info(f"Applying {len(preprocessors)} preprocessing step(s)...")
        preprocess(concat_ds, preprocessors, n_jobs=int(self.cfg.get("n_jobs", 1)))

        # 4. ICA
        ica_excluded: dict[str, int] = {}
        apply_ica = self.cfg.get("apply_ica", True)
        if apply_ica:
            if not _ICALABEL_AVAILABLE:
                logger.warning("apply_ica=true but mne-icalabel not installed — skipping.")
            else:
                n_comp = int(self.cfg.get("ica_n_components", 20))
                prob_thr = float(self.cfg.get("ica_prob_threshold", 0.8))
                excl_lbls: list[str] = list(
                    self.cfg.get("ica_exclude_labels", _DEFAULT_EXCLUDE_LABELS)
                )
                logger.info(f"Applying ICA (n_components={n_comp}, threshold={prob_thr})")
                for ds in concat_ds.datasets:
                    key = (
                        f"{ds.description.get('subject_name')}/"
                        f"{ds.description.get('session')}/"
                        f"{ds.description.get('condition')}"
                    )
                    logger.info(f"  ICA: {key}")
                    ica_excluded[key] = self._apply_ica(ds.raw, n_comp, prob_thr, excl_lbls)

                mean_excl = sum(ica_excluded.values()) / max(len(ica_excluded), 1)
                logger.info(f"ICA done. Mean excluded: {mean_excl:.1f} components/recording.")

        # 5. Windowing
        seg_len = int(self.cfg["seg_len"])
        overlap = int(self.cfg.get("overlap", 0))
        stride = seg_len - overlap
        if stride <= 0:
            raise ValueError(f"overlap ({overlap}) must be < seg_len ({seg_len})")

        logger.info(f"Windowing: seg_len={seg_len}, overlap={overlap}, stride={stride}")
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
            f"Windows: {n_windows} | Channels: {n_chans} | sfreq: {actual_sfreq} Hz"
        )
        logger.info(
            "Subject summary:\n"
            + windows_ds.description[
                ["subject", "target", "session", "condition"]
            ].to_string()
        )

        # 6. Save
        logger.info(f"Saving WindowsDataset → {out_dir} ...")
        windows_ds.save(str(out_dir), overwrite=True)

        # 7. Metadata
        self.save_meta({
            "dataset_id":               self.cfg["dataset_id"],
            "label_map":                LABEL_MAP,
            "conditions":               list(self.cfg.get("conditions", _ALL_CONDITIONS)),
            "sessions":                 list(self.cfg.get("sessions", ["ses-1", "ses-2"])),
            "late_trigger_threshold":   int(self.cfg.get("late_trigger_threshold", 0)),
            "skipped_sessions":         skipped,
            "n_skipped_sessions":       len(skipped),
            "failed_recordings":        failed,
            "n_failed":                 len(failed),
            "n_recordings":             len(datasets),
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
            "ica_excluded_per_recording": ica_excluded,
            "n_windows":                n_windows,
            "save_folder":              self.cfg["save_folder"],
            "status":                   "success",
        })
        logger.info("✓ Preprocessing complete.")
