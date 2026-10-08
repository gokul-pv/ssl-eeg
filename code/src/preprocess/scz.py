"""Preprocessor for scz — Schizophrenia Resting-State EEG (Zenodo 14808296)."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import mne
import numpy as np
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

LABEL_MAP: dict[str, int] = {
    "HC":  0,
    "SZ":  1,
}

_DEFAULT_EXCLUDE_LABELS: list[str] = [
    "muscle artifact",
    "eye blink",
    "heart beat",
    "channel noise",
    "line noise",
]

# Neuroscan aux channel → MNE channel type
_AUX_TYPE_MAP: dict[str, str] = {
    "HEO":     "eog",
    "VEO":     "eog",
    "EKG":     "ecg",
    "EMG":     "emg",
    "Trigger": "stim",
}


# ---------------------------------------------------------------------------
# Neuroscan DAT reader
# ---------------------------------------------------------------------------

def _parse_rs3_labels(rs3_path: Path) -> tuple[list[str], list[str]]:
    """Parse a Neuroscan .rs3 file and return (eeg_labels, aux_labels)."""
    eeg_labels: list[str] = []
    aux_labels: list[str] = []

    in_eeg_block  = False
    in_aux_block  = False

    with open(rs3_path, encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("LABELS START_LIST"):
                in_eeg_block = True
                continue
            if stripped.startswith("LABELS END_LIST"):
                in_eeg_block = False
                continue
            if stripped.startswith("LABELS_OTHERS START_LIST"):
                in_aux_block = True
                continue
            if stripped.startswith("LABELS_OTHERS END_LIST"):
                in_aux_block = False
                continue

            if in_eeg_block and stripped:
                eeg_labels.append(stripped)
            elif in_aux_block and stripped:
                aux_labels.append(stripped)

    return eeg_labels, aux_labels


def _parse_dap_meta(dap_path: Path) -> dict[str, str]:
    """Parse a Neuroscan .dap file (key = value text) into a flat dict."""
    meta: dict[str, str] = {}
    with open(dap_path, encoding="utf-8-sig") as f:
        for line in f:
            if "=" in line:
                k, _, v = line.partition("=")
                meta[k.strip()] = v.strip()
    return meta


def _read_neuroscan_dat(
    dap_path: Path,
    dat_path: Path,
    rs3_path: Path,
) -> mne.io.RawArray:
    """Load a Neuroscan SynAmps recording into an MNE RawArray."""
    meta = _parse_dap_meta(dap_path)

    n_chans  = int(meta["NumChannels"])    # 69
    n_samples = int(meta["NumSamples"])    # 121700
    sfreq    = float(meta["SampleFreqHz"]) # 1000.0

    eeg_labels, aux_labels = _parse_rs3_labels(rs3_path)
    all_labels = eeg_labels + aux_labels

    if len(all_labels) != n_chans:
        logger.warning(
            f"Channel count mismatch in {dap_path.name}: "
            f"expected {n_chans}, got {len(all_labels)} from .rs3. "
            "Proceeding anyway — check electrode assignments."
        )

    ch_types = [
        _AUX_TYPE_MAP.get(name, "eeg") for name in all_labels
    ]

    # Read binary float32 (little-endian), shape (NumSamples, NumChannels) → (C, T)
    data = np.fromfile(str(dat_path), dtype="<f4")
    data = data.reshape(n_samples, n_chans).T  # (n_chans, n_samples)

    # Data is in µV; MNE expects V internally
    data_V = data * 1e-6

    info = mne.create_info(
        ch_names=all_labels[:n_chans],
        sfreq=sfreq,
        ch_types=ch_types[:n_chans],
    )
    raw = mne.io.RawArray(data_V, info, verbose=False)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)

    # Channels without a standard_1020 position (CB1, CB2) become 'misc' so that
    # EEGPrep, ICA and windowing skip them.
    montage_ch_set = {ch.upper() for ch in montage.ch_names}
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    no_pos = {
        raw.ch_names[p]: "misc"
        for p in eeg_picks
        if raw.ch_names[p].upper() not in montage_ch_set
    }
    if no_pos:
        logger.warning(
            f"  Retyping {list(no_pos.keys())} → 'misc': not in standard_1020 montage. "
            "These are non-scalp channels and will be excluded from EEGPrep, ICA, "
            "and windowing via pick_types(eeg=True)."
        )
        raw.set_channel_types(no_pos)

    # Retype mastoid reference electrodes as 'misc' — M1/M2 appear in the Neuroscan
    # LABELS block as recording channels but are not scalp EEG sites and must not
    # enter the EEGPrep average reference computation.
    _MASTOID_NAMES = {"M1", "M2", "A1", "A2"}
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    mastoid_to_misc = {
        raw.ch_names[p]: "misc"
        for p in eeg_picks
        if raw.ch_names[p].upper() in _MASTOID_NAMES
    }
    if mastoid_to_misc:
        logger.warning(
            f"  Retyping {list(mastoid_to_misc.keys())} → 'misc': mastoid reference electrodes."
        )
        raw.set_channel_types(mastoid_to_misc)

    eeg_final = mne.pick_types(raw.info, eeg=True, exclude=[])
    misc_final = mne.pick_types(raw.info, misc=True, exclude=[])
    logger.info(
        f"  Final channel layout: {len(eeg_final)} EEG, {len(misc_final)} misc. "
        f"EEG channels: {[raw.ch_names[p] for p in eeg_final]}"
    )

    return raw


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class SCZ_Preprocessor(BasePreprocessor):
    """Preprocessing pipeline for scz (Schizophrenia Resting-State EEG)."""

    # ------------------------------------------------------------------
    # Helper: read subject metadata from Excel
    # ------------------------------------------------------------------

    def _build_subject_info(self) -> dict[str, dict]:
        """Read subject IDs and labels from the Excel Demographic sheet."""
        excel_path = Path(self.cfg["excel_path"])
        if not excel_path.exists():
            raise FileNotFoundError(f"Excel metadata not found at {excel_path}")

        df = pd.read_excel(excel_path, sheet_name="Demographic", engine="openpyxl")

        if "code" not in df.columns or "BNO" not in df.columns:
            raise ValueError(
                f"Expected 'code' and 'BNO' columns in Demographic sheet. "
                f"Found: {list(df.columns)}"
            )

        logger.warning(
            "SCZ dataset: 77 raw .dat files are deposited on Zenodo, but the paper's "
            "analyzed sample is only 61 subjects — 16 subjects (8 SZ + 8 HC) were "
            "excluded for low data quality. Their subject IDs are NOT documented. "
            "All subjects with .dat files are processed here; EEGPrep RANSAC/ASR "
            "provides quality gating. Poor-quality subjects will yield fewer or zero "
            "clean windows after artifact subspace reconstruction."
        )

        sub_info: dict[str, dict] = {}
        for row in df.itertuples(index=False):
            sub_id = str(row.code).strip()
            if not sub_id or sub_id.lower() == "nan":
                continue

            bno = str(row.BNO).strip() if pd.notna(row.BNO) else ""
            # Split on comma to handle dual-diagnosis BNO values (e.g. "F2520, F2010").
            # A subject is SZ if ANY individual code starts with "F20".
            bno_codes = [c.strip() for c in bno.split(",") if c.strip()]
            label = 1 if any(c.startswith("F20") for c in bno_codes) else 0

            pid = numeric_subject_id(sub_id)

            sub_info[sub_id] = {"label": label, "pid": pid}

        logger.info(
            f"Excel metadata loaded: {len(sub_info)} subjects "
            f"(HC={sum(1 for v in sub_info.values() if v['label'] == 0)}, "
            f"SZ={sum(1 for v in sub_info.values() if v['label'] == 1)})"
        )

        # ── Single-subject mode (for SLURM array jobs) ──────────────
        subject_id = self.cfg.get("subject_id")
        if subject_id is not None:
            if subject_id not in sub_info:
                raise ValueError(
                    f"subject_id '{subject_id}' not found in Excel metadata. "
                    f"Available: {sorted(sub_info.keys())}"
                )
            sub_info = {subject_id: sub_info[subject_id]}
            logger.info(f"Single-subject mode: processing only {subject_id}")

        return sub_info

    # ------------------------------------------------------------------
    # Helper: load raw recordings
    # ------------------------------------------------------------------

    def _load_base_datasets(self, sub_info: dict) -> list[BaseDataset]:
        """Load Neuroscan .dat recordings from the flat raw directory."""
        raw_data_path = Path(self.cfg["raw_data_path"])
        if not raw_data_path.exists():
            raise FileNotFoundError(f"raw_data_path not found: {raw_data_path}")

        # Discover all .dap files → subject IDs
        dap_files = sorted(raw_data_path.glob("*.dap"))
        if not dap_files:
            raise RuntimeError(f"No .dap files found in {raw_data_path}")

        # Map sub_id → dap_path for discovered files
        discovered: dict[str, Path] = {}
        for dap in dap_files:
            # e.g. "sch_002_ec.dap" → "sch_002"
            stem = dap.stem  # "sch_002_ec"
            sub_id = stem.replace("_ec", "")
            discovered[sub_id] = dap

        # Warn: Excel subjects with no .dat file
        for sub_id in sorted(sub_info.keys()):
            if sub_id not in discovered:
                logger.warning(
                    f"Subject '{sub_id}' is in Excel metadata but has no .dat file "
                    f"in {raw_data_path} — skipping."
                )

        # Warn: .dat files with no Excel entry.
        # Suppressed in single-subject mode — sub_info is already filtered to one
        # subject so every other .dat file would trigger the warning spuriously.
        if self.cfg.get("subject_id") is None:
            for sub_id in sorted(discovered.keys()):
                if sub_id not in sub_info:
                    logger.warning(
                        f"Subject '{sub_id}' has a .dat file but is not in Excel metadata "
                        "— skipping (no label available)."
                    )

        # Process the intersection
        to_process = {k: v for k, v in sub_info.items() if k in discovered}
        logger.info(
            f"Subjects to process: {len(to_process)} "
            f"(HC={sum(1 for v in to_process.values() if v['label'] == 0)}, "
            f"SZ={sum(1 for v in to_process.values() if v['label'] == 1)})"
        )

        datasets: list[BaseDataset] = []
        for sub_id, info in sorted(to_process.items()):
            dap_path = discovered[sub_id]
            stem     = dap_path.stem  # "sch_002_ec"
            dat_path = dap_path.with_suffix(".dat")
            rs3_path = dap_path.with_suffix(".rs3")

            if not dat_path.exists():
                logger.warning(f"  .dat file missing for {sub_id}: {dat_path} — skipping.")
                continue
            if not rs3_path.exists():
                logger.warning(f"  .rs3 file missing for {sub_id}: {rs3_path} — skipping.")
                continue

            logger.info(f"  Subject {sub_id} (label={info['label']})")
            try:
                raw = _read_neuroscan_dat(dap_path, dat_path, rs3_path)
            except Exception as exc:
                logger.error(f"  Failed to load {sub_id}: {exc}")
                continue

            description = {
                "subject":      info["pid"],
                "target":       info["label"],
                "subject_name": sub_id,
                "filename":     f"{stem}.dat",
            }
            ds = BaseDataset(raw, description=description, target_name="target")
            datasets.append(ds)

        logger.info(f"Total recordings loaded: {len(datasets)}")
        return datasets

    # ------------------------------------------------------------------
    # Helper: ICA per recording (same as FEP_Preprocessor)
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

        n_eeg = len(mne.pick_types(raw.info, eeg=True))
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)

        effective_n_components = min(n_components, n_eeg - 1)
        if effective_n_components < n_components:
            logger.warning(
                f"  ICA: reduced n_components {n_components} → {effective_n_components} "
                f"(only {n_eeg} EEG channels remain after EEGPrep)"
            )

        logger.debug(f"  ICA: fitting on 1 Hz HP copy (n_components={effective_n_components})")
        raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        raw_hp.set_montage(montage, match_case=False, on_missing="warn", verbose=False)

        ica = ICA(
            n_components=effective_n_components,
            method="fastica",
            random_state=42,
            max_iter="auto",
        )
        ica.fit(raw_hp, verbose=False)

        labels_info = icalabel_label_components(raw_hp, ica, method="iclabel")
        component_labels: list[str] = labels_info["labels"]
        proba_matrix = labels_info["y_pred_proba"]

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

        # ── 2. Load raw recordings ────────────────────────────────────
        logger.info("Loading raw EEG recordings...")
        datasets = self._load_base_datasets(sub_info)
        if not datasets:
            raise RuntimeError(
                "No datasets loaded. Check 'raw_data_path' and 'excel_path'."
            )
        concat_ds = BaseConcatDataset(datasets)

        # ── 3. Braindecode preprocessing chain ────────────────────────
        preprocessors: list[Preprocessor | EEGPrep] = []

        # 3a. Keep EEG channels only (drops HEO, VEO, EKG, EMG, Trigger)
        preprocessors.append(
            Preprocessor("pick_types", eeg=True, meg=False, stim=False, verbose=False)
        )

        # 3b. Scale (null → skip; data already in µV after reader's V→µV conversion
        #     which is reversed by the pipeline back to µV at window output)
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

        # 3e. Notch filter (50 Hz — European power line, Budapest Hungary)
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
                n_components   = int(self.cfg.get("ica_n_components", 20))
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
                    fname    = ds.description.get("filename", "")
                    key      = f"{sub_name}/{fname}"
                    logger.info(f"  ICA: {key}")
                    n_excl = self._apply_ica(
                        ds.raw, n_components, prob_threshold, exclude_labels
                    )
                    ica_excluded_per_subject[key] = n_excl

                logger.info(
                    "ICA done. Mean excluded: "
                    f"{sum(ica_excluded_per_subject.values()) / max(len(ica_excluded_per_subject), 1):.1f} "
                    "components per recording."
                )

        # ── 5. Fixed-length windowing ─────────────────────────────────
        seg_len = int(self.cfg["seg_len"])
        overlap = int(self.cfg.get("overlap", 0))
        stride  = seg_len - overlap
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

        n_windows  = len(windows_ds)
        n_chans    = windows_ds[0][0].shape[0]
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
                "label_map":                 {str(k): v for k, v in LABEL_MAP.items()},
                "save_folder":               self.cfg["save_folder"],
            }
        )
        logger.info("✓ Preprocessing complete.")
