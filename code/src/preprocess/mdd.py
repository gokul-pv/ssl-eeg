"""Preprocessor for ds003478 — MDD vs Healthy Controls Resting State EEG."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

import json
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
        "mne-icalabel not installed; ICA will be skipped.",
        ImportWarning, stacklevel=2,
    )

from .base import BasePreprocessor, numeric_subject_id

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

# SCID → label mapping
SCID_LABEL_MAP: dict[str, int] = {
    "No Interview":  0,   # HC
    "Current MDD":   1,   # MDD (current episode)
    "Past MDD":      1,   # MDD (remission, included)
}
# Any value containing "do not meet" (case-insensitive) → also MDD (label 1)

_DEFAULT_EXCLUDE_LABELS: list[str] = [
    "muscle artifact", "eye blink", "heart beat", "channel noise", "line noise",
]

# trial_type prefixes used for block detection
_EC_ONSET_MARKER = "Eyes Closed: Every 2000 ms"
_EO_ONSET_MARKER = "Eyes Open: Every 2000 ms"
_EC_PREFIX = "Eyes Closed"
_EO_PREFIX = "Eyes Open"


class MDD_Preprocessor(BasePreprocessor):
    """Preprocessing pipeline for ds003478 (MDD vs HC)."""

    # ------------------------------------------------------------------
    # Subject metadata
    # ------------------------------------------------------------------

    def _build_subject_info(self) -> tuple[dict[str, dict], dict[str, str]]:
        """Read participants.tsv → (retained, excluded)."""
        tsv = Path(self.cfg["raw_data_path"]) / "participants.tsv"
        if not tsv.exists():
            raise FileNotFoundError(f"participants.tsv not found: {tsv}")

        df = pd.read_csv(tsv, sep="\t")

        # Locate scid column
        scid_col = next((c for c in df.columns if c.lower() == "scid"), None)
        if scid_col is None:
            raise ValueError(f"No 'scid' column in participants.tsv. Found: {list(df.columns)}")

        hamd_col = next((c for c in df.columns if c.lower() in ("hamd", "ham_d", "ham-d")), None)
        hamd_hc_warn = float(self.cfg.get("hamd_hc_warn_above", 7))
        hamd_mdd_warn = float(self.cfg.get("hamd_mdd_warn_below", 14))

        sub_info: dict[str, dict] = {}
        excluded: dict[str, str] = {}

        for row in df.itertuples(index=False):
            sub_id = str(getattr(row, "participant_id", row[0]))
            raw_scid = str(getattr(row, scid_col, "")).strip()

            # Determine label
            if "do not meet" in raw_scid.lower():
                label = 1  # included as MDD
            elif raw_scid in SCID_LABEL_MAP:
                label = SCID_LABEL_MAP[raw_scid]
            else:
                reason = f"unknown SCID='{raw_scid}'"
                logger.info(f"  {sub_id}: excluded ({reason})")
                excluded[sub_id] = reason
                continue

            # HAM-D sanity check (warning only)
            hamd_val = None
            if hamd_col is not None:
                raw_hamd = getattr(row, hamd_col, None)
                try:
                    hamd_val = float(raw_hamd)
                    if label == 0 and hamd_val >= hamd_hc_warn:
                        logger.warning(
                            f"  {sub_id}: HC but hamd={hamd_val:.1f} >= {hamd_hc_warn} (QC warning)"
                        )
                    elif label == 1 and hamd_val < hamd_mdd_warn:
                        logger.warning(
                            f"  {sub_id}: MDD but hamd={hamd_val:.1f} < {hamd_mdd_warn} (QC warning)"
                        )
                except (ValueError, TypeError):
                    pass

            pid = numeric_subject_id(sub_id)

            sub_info[sub_id] = {"label": label, "pid": pid, "hamd": hamd_val}

        logger.info(
            f"Subjects retained: {len(sub_info)} "
            f"(HC={sum(1 for v in sub_info.values() if v['label']==0)}, "
            f"MDD={sum(1 for v in sub_info.values() if v['label']==1)}, "
            f"excluded={len(excluded)})"
        )
        if excluded:
            logger.info("Excluded subjects: " + ", ".join(
                f"{k} ({v})" for k, v in excluded.items()
            ))

        # Single-subject mode (SLURM)
        subject_id = self.cfg.get("subject_id")
        if subject_id is not None:
            if subject_id not in sub_info:
                # Subject exists but is excluded — log and exit gracefully
                if subject_id in excluded:
                    logger.info(
                        f"Subject '{subject_id}' is excluded "
                        f"({excluded[subject_id]}) — nothing to preprocess."
                    )
                    return {}, {subject_id: excluded[subject_id]}
                raise ValueError(f"subject_id '{subject_id}' not found in participants.tsv.")
            sub_info = {subject_id: sub_info[subject_id]}
            logger.info(f"Single-subject mode: {subject_id}")

        return sub_info, excluded

    # ------------------------------------------------------------------
    # Event-based segment cropping
    # ------------------------------------------------------------------

    def _find_events_tsv(self, set_path: Path) -> Path | None:
        """Derive BIDS events.tsv path from .set file path."""
        stem = set_path.stem  # e.g. "sub-001_task-Rest_eeg"
        if stem.endswith("_eeg"):
            stem = stem[:-4]
        p = set_path.parent / f"{stem}_events.tsv"
        return p if p.exists() else None

    def _get_all_blocks(
        self, events_tsv: Path, resting_state: str
    ) -> dict[str, list[tuple[float, float]]]:
        """Parse events.tsv and return ALL blocks for each requested state."""
        df = pd.read_csv(events_tsv, sep="\t")
        if "trial_type" not in df.columns or "onset" not in df.columns:
            raise ValueError(f"events.tsv missing required columns: {events_tsv}")

        df = df.sort_values("onset").reset_index(drop=True)
        tt = df["trial_type"].astype(str)
        onsets = df["onset"].astype(float).values

        result: dict[str, list[tuple[float, float]]] = {}

        def _collect_blocks(onset_marker: str, opposite_prefix: str, label: str):
            """Find all blocks starting with onset_marker; each ends at next opposite_prefix."""
            blocks: list[tuple[float, float]] = []
            start_indices = df.index[tt == onset_marker].tolist()
            if not start_indices:
                logger.warning(f"    No '{onset_marker}' in {events_tsv.name}")
                return
            for si in start_indices:
                tmin = float(onsets[si])
                
                # A block ends at the next opposite-state marker or the next run of the same state.
                mask_opp = tt.str.startswith(opposite_prefix) & (onsets > tmin)
                mask_same = (tt == onset_marker) & (onsets > tmin)
                
                next_opp = float(onsets[mask_opp.values][0]) if mask_opp.any() else float("inf")
                next_same = float(onsets[mask_same.values][0]) if mask_same.any() else float("inf")
                
                tmax = min(next_opp, next_same)
                blocks.append((tmin, tmax))
            # Merge adjacent blocks: if two EC blocks are separated by only STATUS
            # (no EO events in between), they're actually the same continuous EC period.
            # Detect: if an EC block's tmax == inf (no EO follows) but another EC block
            # starts shortly after → they are separated by a STATUS gap, NOT by EO.
            # In ds003478 the only thing that separates EC runs is STATUS + another
            # EC onset_marker, so we DON'T merge: each onset_marker IS a new run.
            result[label] = blocks
            logger.info(
                f"    Found {len(blocks)} '{label}' block(s): "
                + ", ".join(f"{a:.1f}–{'end' if b==float('inf') else f'{b:.1f}'}s"
                            for a, b in blocks)
            )

        if resting_state in ("eyes_closed", "both", "first"):
            _collect_blocks(_EC_ONSET_MARKER, _EO_PREFIX, "eyes_closed")

        if resting_state in ("eyes_open", "both", "first"):
            _collect_blocks(_EO_ONSET_MARKER, _EC_PREFIX, "eyes_open")

        return result

    def _crop_first_block(
        self,
        raw: mne.io.Raw,
        blocks: list[tuple[float, float]],
        condition_label: str,
        min_samples: int = 0,
    ) -> mne.io.Raw | None:
        """Crop raw to the first block of sufficient length."""
        if not blocks:
            logger.warning(f"    No '{condition_label}' blocks found — skipping.")
            return None
        sfreq = raw.info["sfreq"]
        for idx, (tmin, tmax) in enumerate(blocks):
            t_end = min(tmax, raw.times[-1])
            if t_end <= tmin:
                logger.warning(
                    f"    '{condition_label}' block {idx}: tmin={tmin:.2f} >= t_end={t_end:.2f} — skipping."
                )
                continue
            n_samples = int((t_end - tmin) * sfreq)
            if n_samples < min_samples:
                logger.warning(
                    f"    '{condition_label}' block {idx}: {n_samples} samples "
                    f"< min_samples {min_samples} ({(t_end - tmin):.1f}s) — trying next block."
                )
                continue
            logger.info(
                f"    Cropping '{condition_label}' block {idx}: {tmin:.2f}–{t_end:.2f}s "
                f"({n_samples} samples)"
            )
            return raw.copy().crop(tmin=tmin, tmax=t_end)
        logger.warning(
            f"    No '{condition_label}' block with >= {min_samples} samples found — skipping."
        )
        return None

    # ------------------------------------------------------------------
    # Montage
    # ------------------------------------------------------------------

    def _fix_montage(self, raw: mne.io.Raw) -> None:
        # Re-type non-standard / non-EEG channels so they don't break ICLabel.
        # Any channel not in the 10-20 system must be correctly typed so that
        # pick_types(eeg=True) drops them before ICA.
        mapping = {
            "HEOG": "eog",
            "VEOG": "eog",
            "EKG":  "ecg",   # cardiac channel present in some subjects
            "ECG":  "ecg",
            "EMG":  "emg",
            "CB1":  "misc",  # cerebellar — not in 10-20
            "CB2":  "misc",
        }
        valid_mapping = {ch: t for ch, t in mapping.items() if ch in raw.ch_names}
        if valid_mapping:
            logger.info(f"    Re-typing channels: {valid_mapping}")
            raw.set_channel_types(valid_mapping, verbose=False)

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)

    # ------------------------------------------------------------------
    # Load recordings
    # ------------------------------------------------------------------

    def _load_base_datasets(
        self, sub_info: dict
    ) -> tuple[list[BaseDataset], dict[str, str]]:
        raw_data_path = Path(self.cfg["raw_data_path"])
        resting_state = self.cfg.get("resting_state", "full")
        seg_len = int(self.cfg["seg_len"])
        datasets: list[BaseDataset] = []
        failed: dict[str, str] = {}  # sub_id → reason

        for sub_id, info in sorted(sub_info.items()):
            sub_eeg = raw_data_path / sub_id / "eeg"
            if not sub_eeg.exists():
                sub_eeg = raw_data_path / sub_id
            if not sub_eeg.exists():
                reason = "EEG folder not found"
                logger.warning(f"  {sub_id}: {reason} — skipping.")
                failed[sub_id] = reason
                continue

            set_files = sorted(f for f in os.listdir(sub_eeg) if f.endswith(".set"))
            if not set_files:
                reason = "no .set files found"
                logger.warning(f"  {sub_id}: {reason} — skipping.")
                failed[sub_id] = reason
                continue

            logger.info(f"  Subject {sub_id} (label={info['label']})")
            sub_loaded = False

            for fname in set_files:
                file_path = sub_eeg / fname
                logger.info(f"    Loading {fname}")
                try:
                    raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)
                    self._fix_montage(raw)
                except Exception as exc:
                    logger.error(f"    Failed to load {file_path}: {exc}")
                    continue

                # Full-recording mode: skip event parsing entirely
                if resting_state == "full":
                    logger.info(f"    resting_state='full': using entire recording ({raw.times[-1]:.1f}s)")
                    items = [("full", [(0.0, float("inf"))])]
                else:
                    # Find events.tsv sidecar
                    events_tsv = self._find_events_tsv(file_path)
                    if events_tsv is None:
                        logger.warning(f"    No events.tsv for {fname} — using full recording.")
                        all_blocks: dict[str, list[tuple[float, float]]] = {"full": [(0.0, float("inf"))]}
                    else:
                        try:
                            all_blocks = self._get_all_blocks(events_tsv, resting_state)
                        except Exception as exc:
                            logger.error(f"    Event parsing failed: {exc} — using full recording.")
                            all_blocks = {"full": [(0.0, float("inf"))]}

                    if not all_blocks:
                        reason = (
                            f"no '{resting_state}' event markers found in events.tsv "
                            f"(only STATUS/metadata events present)"
                        )
                        logger.warning(f"    {sub_id}/{fname}: {reason} — skipping.")
                        if sub_id not in failed:
                            failed[sub_id] = reason
                        continue

                # Determine which (state, blocks) pairs to process
                if resting_state == "full":
                    pass  # items already set above
                elif resting_state == "first":
                    ec_blocks = all_blocks.get("eyes_closed", [])
                    eo_blocks = all_blocks.get("eyes_open", [])
                    ec_onset = ec_blocks[0][0] if ec_blocks else float("inf")
                    eo_onset = eo_blocks[0][0] if eo_blocks else float("inf")
                    if ec_onset == float("inf") and eo_onset == float("inf"):
                        reason = "no EC or EO blocks found — skipping."
                        logger.warning(f"    {sub_id}/{fname}: {reason}")
                        if sub_id not in failed:
                            failed[sub_id] = reason
                        continue
                    chosen = "eyes_closed" if ec_onset <= eo_onset else "eyes_open"
                    logger.info(
                        f"    resting_state='first': selected '{chosen}' "
                        f"(EC onset={ec_onset:.1f}s, EO onset={eo_onset:.1f}s)"
                    )
                    items = [(chosen, all_blocks[chosen])]
                elif resting_state == "both":
                    items = list(all_blocks.items())
                else:
                    state = resting_state  # "eyes_closed" or "eyes_open"
                    items = [(state, all_blocks.get(state, []))]

                for state, blocks in items:
                    cropped = self._crop_first_block(raw, blocks, state, min_samples=seg_len)
                    if cropped is None:
                        if sub_id not in failed:
                            failed[sub_id] = f"no '{state}' block >= {seg_len} samples"
                        continue

                    logger.info(f"    State '{state}': cropped {cropped.times[-1]:.1f}s")
                    description = {
                        "subject":        info["pid"],
                        "target":         info["label"],
                        "subject_name":   sub_id,
                        "filename":       fname,
                        "resting_state":  state,
                    }
                    ds = BaseDataset(cropped, description=description, target_name="target")
                    datasets.append(ds)
                    sub_loaded = True

            # If the subject had .set files but none produced usable data
            if not sub_loaded and sub_id not in failed:
                failed[sub_id] = "no usable segments produced (all files failed)"

        logger.info(
            f"Total recordings loaded: {len(datasets)} | "
            f"Subjects with load failures: {len(failed)}"
        )
        if failed:
            logger.warning(
                "Subjects not preprocessed (data issue):\n  "
                + "\n  ".join(f"{k}: {v}" for k, v in failed.items())
            )
        return datasets, failed

    # ------------------------------------------------------------------
    # ICA (identical logic to FEP/ADFTD)
    # ------------------------------------------------------------------

    def _apply_ica(
        self, raw: mne.io.Raw, n_components: int,
        prob_threshold: float, exclude_labels: list[str],
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

        sub_info, excluded = self._build_subject_info()

        # In single-subject SLURM mode, _build_subject_info returns ({}, {sub: reason})
        # when the subject is excluded. Exit cleanly but save metadata first.
        if not sub_info:
            logger.info("Subject excluded — saving reason to metadata and exiting.")
            self.save_meta({
                "dataset_id":        self.cfg["dataset_id"],
                "excluded_subjects": excluded,
                "n_excluded":        len(excluded),
                "failed_subjects":   {},
                "n_failed":          0,
                "status":            "excluded"
            })
            return

        logger.info("Loading raw EEG recordings...")
        datasets, failed_subjects = self._load_base_datasets(sub_info)
        if not datasets:
            logger.warning(
                "No datasets loaded — all recordings failed or contained only STATUS events. "
                "Saving failure reason to metadata and exiting."
            )
            self.save_meta({
                "dataset_id":        self.cfg["dataset_id"],
                "excluded_subjects": excluded,
                "n_excluded":        len(excluded),
                "failed_subjects":   failed_subjects,
                "n_failed":          len(failed_subjects),
                "status":            "failed"
            })
            return

        concat_ds = BaseConcatDataset(datasets)

        # Preprocessing chain
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
                "resample_to":              float(self.cfg.get("eegprep_resample_to", 200)),
                "burst_removal_cutoff":     float(self.cfg.get("eegprep_burst_cutoff", 20)),
                "bad_channel_corr_threshold": float(self.cfg.get("eegprep_corr_threshold", 0.75)),
                "common_avg_ref":           bool(self.cfg.get("eegprep_common_avg_ref", True)),
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

        # ICA
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
                    key = f"{ds.description.get('subject_name')}/{ds.description.get('filename')}"
                    logger.info(f"  ICA: {key}")
                    ica_excluded[key] = self._apply_ica(ds.raw, n_comp, prob_thr, excl_lbls)

                mean_excl = sum(ica_excluded.values()) / max(len(ica_excluded), 1)
                logger.info(f"ICA done. Mean excluded: {mean_excl:.1f} components/recording.")

        # Windowing
        seg_len = int(self.cfg["seg_len"])
        overlap = int(self.cfg.get("overlap", 0))
        stride = seg_len - overlap
        if stride <= 0:
            raise ValueError(f"overlap ({overlap}) must be < seg_len ({seg_len})")

        # Drop any recordings shortened below seg_len by EEGPrep/ASR after cropping
        valid = [ds for ds in concat_ds.datasets if ds.raw.n_times >= seg_len]
        dropped = len(concat_ds.datasets) - len(valid)
        if dropped:
            short = [
                f"{ds.description.get('subject_name')} ({ds.raw.n_times} samples)"
                for ds in concat_ds.datasets if ds.raw.n_times < seg_len
            ]
            logger.warning(
                f"Dropping {dropped} recording(s) shorter than seg_len={seg_len} "
                f"after preprocessing: {short}"
            )
            concat_ds = BaseConcatDataset(valid)

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
        logger.info(f"Windows: {n_windows} | Channels: {n_chans} | sfreq: {actual_sfreq} Hz")
        logger.info(
            "Subject summary:\n"
            + windows_ds.description[["subject", "target", "resting_state"]].to_string()
        )

        logger.info(f"Saving WindowsDataset → {out_dir} ...")
        windows_ds.save(str(out_dir), overwrite=True)

        self.save_meta({
            "dataset_id":               self.cfg["dataset_id"],
            "resting_state":            self.cfg.get("resting_state", "full"),
            "scid_label_map":           {k: v for k, v in SCID_LABEL_MAP.items() if v is not None},
            "excluded_subjects":        excluded,
            "n_excluded":               len(excluded),
            "failed_subjects":          failed_subjects,
            "n_failed":                 len(failed_subjects),
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
            "n_subjects":               len(datasets),
            "save_folder":              self.cfg["save_folder"],
            "status":                   "success"
        })
        logger.info("✓ Preprocessing complete.")
