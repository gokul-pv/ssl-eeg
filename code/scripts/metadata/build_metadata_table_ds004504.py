#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from _config_paths import config_value

import mne
import pandas as pd

mne.set_log_level("ERROR")   # suppress MNE console chatter


# ---------------------------------------------------------------------------
# Sex normalisation
# ---------------------------------------------------------------------------

_SEX_NORM: dict[str, str] = {
    "m": "M", "male": "M", "1": "M", "man": "M",
    "f": "F", "female": "F", "2": "F", "woman": "F",
}


def _normalise_sex(raw: str) -> str:
    return _SEX_NORM.get(str(raw).strip().lower(), str(raw).strip())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_bids_filename(stem: str) -> dict:
    """Extract BIDS key-value entities from a filename stem."""
    _SUFFIX_TOKENS = {"eeg", "channels", "events", "electrodes", "coordsystem"}
    entities: dict = {}
    for match in re.finditer(r"([a-zA-Z]+)-([^_]+)", stem):
        key, val = match.group(1), match.group(2)
        if key not in _SUFFIX_TOKENS:
            entities[key] = val
    return entities


def load_channels_tsv(channels_tsv: Path) -> dict:
    """Read a BIDS *_channels.tsv (columns: name, type, units)."""
    empty = dict(
        n_total_channels="",
        n_eeg_channels="",
        n_other_channels="",
        channel_types_tsv="",
        channel_unit="",
        channel_names="",
    )
    if not channels_tsv.exists():
        print(f"  [WARN] channels.tsv not found: {channels_tsv.name}")
        return empty
    try:
        df = pd.read_csv(channels_tsv, sep="\t", dtype=str)
    except Exception as exc:
        print(f"  [WARN] channels.tsv parse error {channels_tsv.name}: {exc}")
        return empty

    df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)

    n_total = len(df)
    result: dict = dict(
        n_total_channels=n_total,
        n_eeg_channels="",
        n_other_channels="",
        channel_types_tsv="",
        channel_unit="",
        channel_names="",
    )

    if "type" not in df.columns:
        print(f"  [WARN] 'type' column missing in {channels_tsv.name}")
        return result

    type_upper = df["type"].str.upper().fillna("")
    type_counts = Counter(t for t in type_upper if t)
    result["channel_types_tsv"] = ";".join(
        f"{t}:{c}" for t, c in sorted(type_counts.items())
    )

    n_eeg = type_counts.get("EEG", 0)
    result["n_eeg_channels"] = n_eeg
    result["n_other_channels"] = n_total - n_eeg

    eeg_mask = type_upper == "EEG"

    if "name" in df.columns:
        eeg_names = df.loc[eeg_mask, "name"].tolist()
        result["channel_names"] = ";".join(str(n) for n in eeg_names if str(n).strip())

    if "units" in df.columns:
        eeg_units = df.loc[eeg_mask, "units"]
        non_empty = eeg_units[eeg_units.str.strip().astype(bool)]
        if not non_empty.empty:
            unit_counts = Counter(non_empty.str.strip())
            result["channel_unit"] = ";".join(
                f"{u}:{c}" for u, c in sorted(unit_counts.items())
            )

    return result


def load_eeg_json(eeg_json: Path) -> dict:
    """Read a BIDS *_eeg.json sidecar. Returns {} on failure."""
    if not eeg_json.exists():
        print(f"  [WARN] _eeg.json not found: {eeg_json.name}")
        return {}
    try:
        with open(eeg_json, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"  [WARN] _eeg.json parse error {eeg_json.name}: {exc}")
        return {}


def load_dataset_description(bids_root: Path) -> dict:
    """Read dataset_description.json. Returns {} on failure."""
    path = bids_root / "dataset_description.json"
    if not path.exists():
        print("[WARN] dataset_description.json not found")
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"[WARN] dataset_description.json parse error: {exc}")
        return {}


def read_mne_filter_info(set_path: Path) -> tuple[float | str, float | str]:
    """Load the EEGLAB .set header with MNE (preload=False) and return
    (lowpass_hz, highpass_hz) as stored in raw.info.
    """
    try:
        raw = mne.io.read_raw_eeglab(str(set_path), preload=False, verbose=False)
        return raw.info["lowpass"], raw.info["highpass"]
    except Exception as exc:
        print(f"  [WARN] MNE read error {set_path.name}: {exc}")
        return "", ""


def _infer_eyes_condition(eeg_json: dict) -> str:
    """Scan _eeg.json text fields for open/closed eye keywords.
    Checks: TaskName, Instructions, TaskDescription (in that order).
    Returns "open", "closed", or "".
    """
    for key in ("TaskName", "Instructions", "TaskDescription"):
        val = str(eeg_json.get(key, "")).lower()
        if not val:
            continue
        if "closed" in val:
            return "closed"
        if "open" in val:
            return "open"
    return ""


# ---------------------------------------------------------------------------
# CSV column definition — must match existing master_subject_table_ds004504.csv
# ---------------------------------------------------------------------------

COLUMNS = [
    # --- Identifiers ---
    "new_id",           # 4-digit counter, 0001-based
    "old_id",           # subject_id (BIDS participant_id)
    "dataset_id",       # "ds004504"
    "subject_id",       # BIDS participant_id, e.g. "sub-001"
    "session_id",       # BIDS ses entity (empty if absent)
    "session_num",      # numeric session number (empty if absent)
    "run",              # BIDS run entity value (empty if absent)
    # --- Demographics (participants.tsv) ---
    "age",              # Age column
    "sex",              # Gender column (M/F)
    "mmse",             # MMSE column
    # --- Diagnosis (participants.tsv Group column) ---
    "diagnosis",        # alzheimer | frontotemporaldementia | control
    # --- Task (BIDS filename + _eeg.json) ---
    "task_label",       # BIDS task entity, e.g. "eyesclosed"
    "task_name",        # _eeg.json → TaskName
    "task_description", # _eeg.json → TaskDescription (empty if absent)
    "eyes_condition",   # inferred from TaskName; empty if unclear
    # --- EEG acquisition (_eeg.json) ---
    "sampling_rate_hz",       # SamplingFrequency
    "powerline_freq_hz",      # PowerLineFrequency
    "recording_duration_sec", # RecordingDuration
    "n_timepoints",           # RecordingDuration × SamplingFrequency
    "recording_type",         # RecordingType
    # --- Channel counts (_channels.tsv primary, JSON fallback) ---
    "n_total_channels",
    "n_eeg_channels",
    "n_other_channels",
    "channel_types_tsv",   # e.g. "EEG:19"
    "channel_unit",        # e.g. "microV:19"
    "channel_names",       # semicolon-separated EEG channel names
    # --- EEG setup (_eeg.json) ---
    "eeg_reference",       # EEGReference
    "eeg_ground",          # EEGGround (empty if absent from JSON)
    "eeg_placement_scheme",# EEGPlacementScheme
    "software_filters",    # SoftwareFilters (serialised)
    "hardware_filters",    # HardwareFilters (empty if absent from JSON)
    # --- Filter bounds from MNE .set header ---
    "mne_lowpass_hz",      # raw.info['lowpass'] from MNE
    "mne_highpass_hz",     # raw.info['highpass'] from MNE
    # --- Hardware / institution (_eeg.json) ---
    "cap_manufacturer",    # CapManufacturer
    "cap_model",           # CapManufacturersModelName
    "institution",         # InstitutionName
    # --- File paths ---
    "file_name",           # basename of the .set file
    "file_extension",      # ".set"
    "file_path_raw",       # absolute path to .set
    "subject_dir",         # absolute path to sub-XXX/ directory
    "eeg_json_path",       # absolute path to *_eeg.json
    "channels_tsv_path",   # absolute path to *_channels.tsv
    # --- BIDS dataset metadata (dataset_description.json) ---
    "bids_dataset_name",   # Name
    "bids_version",        # BIDSVersion
    "bids_license",        # License
    "bids_dataset_doi",    # DatasetDOI
    # --- Data availability ---
    "file_missing",        # True if no EEG .set file found for this subject
    "table_generated_at",
]

# participants.tsv Group → normalised diagnosis label
_GROUP_MAP: dict[str, str] = {
    "a": "alzheimer",
    "f": "frontotemporaldementia",
    "c": "control",
}


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def build_row(
    new_id: int,
    sub_id: str,
    sub_row: "pd.Series",
    set_path: Path,
    dataset_desc: dict,
    now_str: str,
) -> dict:
    """Return one CSV row dict for a single .set recording file."""

    # ── Parse BIDS filename entities ──────────────────────────────────────
    # e.g. "sub-001_task-eyesclosed_eeg"
    stem = set_path.stem
    entities = parse_bids_filename(stem)

    session_id = f"ses-{entities['ses']}" if "ses" in entities else ""
    run_id     = entities.get("run", "")
    task_label = entities.get("task", "")

    session_num: int | str = ""
    if session_id:
        digits = "".join(c for c in session_id if c.isdigit())
        session_num = int(digits) if digits else ""

    # ── Locate sidecar files ──────────────────────────────────────────────
    # Strip "_eeg" suffix: "sub-001_task-eyesclosed_eeg" → "sub-001_task-eyesclosed"
    base = stem[:-4] if stem.endswith("_eeg") else stem

    eeg_json_path     = set_path.parent / f"{base}_eeg.json"
    channels_tsv_path = set_path.parent / f"{base}_channels.tsv"

    # ── Load sidecar data ─────────────────────────────────────────────────
    eeg_json = load_eeg_json(eeg_json_path)
    ch_info  = load_channels_tsv(channels_tsv_path)

    # ── EEG acquisition values from JSON ─────────────────────────────────
    sfreq: float | str = eeg_json.get("SamplingFrequency", "")
    if sfreq != "":
        sfreq = float(sfreq)

    rec_duration: float | str = eeg_json.get("RecordingDuration", "")
    if rec_duration != "":
        rec_duration = float(rec_duration)

    n_timepoints: int | str = ""
    if rec_duration != "" and sfreq != "":
        n_timepoints = int(float(rec_duration) * float(sfreq))

    # ── MNE filter bounds from .set header ────────────────────────────────
    mne_lowpass, mne_highpass = read_mne_filter_info(set_path)

    # ── Channel counts: TSV primary, JSON fallback ────────────────────────
    n_eeg = ch_info["n_eeg_channels"]
    if n_eeg == "" and "EEGChannelCount" in eeg_json:
        n_eeg = int(eeg_json["EEGChannelCount"])

    n_total = ch_info["n_total_channels"]
    if n_total == "" and eeg_json:
        total_from_json = sum(
            int(v) for k, v in eeg_json.items()
            if k.endswith("ChannelCount") and isinstance(v, (int, float))
        )
        if total_from_json > 0:
            n_total = total_from_json

    n_other = ch_info["n_other_channels"]
    if n_other == "" and n_total != "" and n_eeg != "":
        n_other = int(n_total) - int(n_eeg)

    # ── Demographics (participants.tsv) ───────────────────────────────────
    age  = sub_row.get("Age", "")
    sex  = _normalise_sex(sub_row.get("Gender", ""))
    mmse = sub_row.get("MMSE", "")

    raw_group = str(sub_row.get("Group", "")).strip()
    diagnosis = _GROUP_MAP.get(raw_group.lower(), raw_group.lower())

    # ── BIDS dataset metadata ─────────────────────────────────────────────
    bids_name    = dataset_desc.get("Name", "")
    bids_version = dataset_desc.get("BIDSVersion", "")
    bids_license = dataset_desc.get("License", "")
    bids_doi     = dataset_desc.get("DatasetDOI", dataset_desc.get("DOI", ""))

    # ── Assemble row ──────────────────────────────────────────────────────
    return {
        "new_id":                 f"{new_id:04d}",
        "old_id":                 sub_id,
        "dataset_id":             "ds004504",
        "subject_id":             sub_id,
        "session_id":             session_id,
        "session_num":            session_num,
        "run":                    run_id,
        "age":                    age,
        "sex":                    sex,
        "mmse":                   mmse,
        "diagnosis":              diagnosis,
        "task_label":             task_label,
        "task_name":              eeg_json.get("TaskName", ""),
        "task_description":       eeg_json.get("TaskDescription", ""),
        "eyes_condition":         _infer_eyes_condition(eeg_json),
        "sampling_rate_hz":       sfreq,
        "powerline_freq_hz":      eeg_json.get("PowerLineFrequency", ""),
        "recording_duration_sec": rec_duration,
        "n_timepoints":           n_timepoints,
        "recording_type":         eeg_json.get("RecordingType", ""),
        "n_total_channels":       n_total,
        "n_eeg_channels":         n_eeg,
        "n_other_channels":       n_other,
        "channel_types_tsv":      ch_info["channel_types_tsv"],
        "channel_unit":           ch_info["channel_unit"],
        "channel_names":          ch_info["channel_names"],
        "eeg_reference":          eeg_json.get("EEGReference", ""),
        "eeg_ground":             eeg_json.get("EEGGround", ""),
        "eeg_placement_scheme":   eeg_json.get("EEGPlacementScheme", ""),
        # SoftwareFilters may be a dict; serialise to string (matches existing CSV)
        "software_filters":       str(eeg_json.get("SoftwareFilters", "")),
        "hardware_filters":       str(eeg_json.get("HardwareFilters", "")) if "HardwareFilters" in eeg_json else "",
        "mne_lowpass_hz":         mne_lowpass,
        "mne_highpass_hz":        mne_highpass,
        "cap_manufacturer":       eeg_json.get("CapManufacturer", ""),
        "cap_model":              eeg_json.get("CapManufacturersModelName", ""),
        "institution":            eeg_json.get("InstitutionName", ""),
        "file_name":              set_path.name,
        "file_extension":         set_path.suffix,
        "file_path_raw":          str(set_path),
        "subject_dir":            str(set_path.parent.parent),   # sub-XXX/
        "eeg_json_path":          str(eeg_json_path)      if eeg_json_path.exists()      else "",
        "channels_tsv_path":      str(channels_tsv_path)  if channels_tsv_path.exists()  else "",
        "bids_dataset_name":      bids_name,
        "bids_version":           bids_version,
        "bids_license":           bids_license,
        "bids_dataset_doi":       bids_doi,
        "file_missing":           False,
        "table_generated_at":     now_str,
    }


def build_missing_row(
    new_id: int,
    sub_id: str,
    sub_row: "pd.Series",
    sub_dir: Path,
    dataset_desc: dict,
    now_str: str,
) -> dict:
    """Return a stub CSV row for a subject whose EEG .set file is absent."""
    age  = sub_row.get("Age", "")
    sex  = _normalise_sex(sub_row.get("Gender", ""))
    mmse = sub_row.get("MMSE", "")
    raw_group = str(sub_row.get("Group", "")).strip()
    diagnosis = _GROUP_MAP.get(raw_group.lower(), raw_group.lower())

    return {
        "new_id":                 f"{new_id:04d}",
        "old_id":                 sub_id,
        "dataset_id":             "ds004504",
        "subject_id":             sub_id,
        "session_id":             "",
        "session_num":            "",
        "run":                    "",
        "age":                    age,
        "sex":                    sex,
        "mmse":                   mmse,
        "diagnosis":              diagnosis,
        "task_label":             "",
        "task_name":              "",
        "task_description":       "",
        "eyes_condition":         "",
        "sampling_rate_hz":       "",
        "powerline_freq_hz":      "",
        "recording_duration_sec": "",
        "n_timepoints":           "",
        "recording_type":         "",
        "n_total_channels":       "",
        "n_eeg_channels":         "",
        "n_other_channels":       "",
        "channel_types_tsv":      "",
        "channel_unit":           "",
        "channel_names":          "",
        "eeg_reference":          "",
        "eeg_ground":             "",
        "eeg_placement_scheme":   "",
        "software_filters":       "",
        "hardware_filters":       "",
        "mne_lowpass_hz":         "",
        "mne_highpass_hz":        "",
        "cap_manufacturer":       "",
        "cap_model":              "",
        "institution":            "",
        "file_name":              "",
        "file_extension":         "",
        "file_path_raw":          "",
        "subject_dir":            str(sub_dir),
        "eeg_json_path":          "",
        "channels_tsv_path":      "",
        "bids_dataset_name":      dataset_desc.get("Name", ""),
        "bids_version":           dataset_desc.get("BIDSVersion", ""),
        "bids_license":           dataset_desc.get("License", ""),
        "bids_dataset_doi":       dataset_desc.get("DatasetDOI", dataset_desc.get("DOI", "")),
        "file_missing":           True,
        "table_generated_at":     now_str,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build master_subject_table_ds004504.csv from BIDS sidecar files.\n"
            "Run on the server where the raw dataset is accessible."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/preprocess/adftd.yaml",
        help="Preprocessing config that provides the default raw-data paths.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Path to the BIDS root directory of ds004504. Default: raw_data_path from --config.",
    )
    parser.add_argument(
        "--out_csv",
        default="metadata/ds004504/master_subject_table_ds004504.csv",
        help="Output CSV path (relative to project root / code/, or absolute).",
    )
    args = parser.parse_args()
    if args.data_root is None:
        args.data_root = config_value(args.config, "raw_data_path")

    bids_root = Path(args.data_root)
    out_csv   = Path(args.out_csv)

    if not bids_root.exists():
        raise FileNotFoundError(
            f"BIDS root not found: {bids_root}\n"
            "Run this script on the server where the dataset is mounted."
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. participants.tsv ───────────────────────────────────────────────
    participants_tsv = bids_root / "participants.tsv"
    if not participants_tsv.exists():
        raise FileNotFoundError(f"participants.tsv not found: {participants_tsv}")

    participants_df = pd.read_csv(participants_tsv, sep="\t", dtype=str)
    participants_df = participants_df.set_index("participant_id")
    print(f"Loaded {len(participants_df)} participants from participants.tsv")
    print(f"  Columns : {list(participants_df.columns)}")
    print(f"  Groups  :\n{participants_df['Group'].value_counts().to_string()}")

    # ── 2. dataset_description.json ──────────────────────────────────────
    dataset_desc = load_dataset_description(bids_root)
    print(f"\nDataset : {dataset_desc.get('Name', '(unnamed)')}")
    print(f"Version : {dataset_desc.get('BIDSVersion', '?')}")

    # ── 3. Walk subject directories ───────────────────────────────────────
    now_str  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    rows: list[dict] = []
    new_id   = 1
    skipped  = 0

    sub_dirs = sorted(
        d for d in bids_root.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )
    print(f"\nFound {len(sub_dirs)} sub-* directories\n")

    for sub_dir in sub_dirs:
        sub_id = sub_dir.name   # e.g. "sub-001"

        if sub_id not in participants_df.index:
            print(f"  [WARN] {sub_id} not in participants.tsv — skipping")
            skipped += 1
            continue

        sub_row = participants_df.loc[sub_id]

        # .set files sit in sub-XXX/eeg/ (also handles ses-XX/eeg/ via rglob)
        set_files = sorted(sub_dir.rglob("*.set"))
        if not set_files:
            print(f"  [{new_id:04d}] {sub_id} — no .set file found, marking file_missing=True")
            row = build_missing_row(
                new_id=new_id,
                sub_id=sub_id,
                sub_row=sub_row,
                sub_dir=sub_dir,
                dataset_desc=dataset_desc,
                now_str=now_str,
            )
            rows.append(row)
            new_id += 1
            continue

        for set_path in set_files:
            print(f"  [{new_id:04d}] {set_path.relative_to(bids_root)}")
            row = build_row(
                new_id=new_id,
                sub_id=sub_id,
                sub_row=sub_row,
                set_path=set_path,
                dataset_desc=dataset_desc,
                now_str=now_str,
            )
            rows.append(row)
            new_id += 1

    # ── 4. Write CSV ──────────────────────────────────────────────────────
    if not rows:
        print(
            "\n[ERROR] No rows generated.\n"
            "Check --data_root and that sub-* directories contain .set files."
        )
        return

    out_df = pd.DataFrame(rows, columns=COLUMNS)
    out_df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    n_missing = out_df["file_missing"].sum()
    print("\n" + "=" * 60)
    print(f"✓  Written {len(out_df)} rows  →  {out_csv}")
    print(f"   Subjects processed : {out_df['subject_id'].nunique()}")
    print(f"   Subjects skipped   : {skipped}")
    print(f"   File missing       : {n_missing}")
    if n_missing:
        print(f"   Missing subjects   : {out_df.loc[out_df['file_missing'], 'subject_id'].tolist()}")
    print(f"   Diagnosis counts   :\n{out_df['diagnosis'].value_counts().to_string()}")
    sfreqs = out_df["sampling_rate_hz"].dropna().unique()
    if len(sfreqs):
        print(f"   Sampling rates Hz  : {sorted(sfreqs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
