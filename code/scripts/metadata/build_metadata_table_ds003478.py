#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from _config_paths import config_value

import mne
import pandas as pd

mne.set_log_level("ERROR")


# ---------------------------------------------------------------------------
# Diagnosis derivation — mirrors MDD_Preprocessor.SCID_LABEL_MAP
# ---------------------------------------------------------------------------

_SCID_MAP: dict[str, str] = {
    "no interview":  "control",
    "current mdd":   "mdd",
    "past mdd":      "mdd",
}


def _derive_diagnosis(raw_scid: str) -> str:
    """Map a raw SCID cell value to a normalised diagnosis label.
    Matches the logic in MDD_Preprocessor._build_subject_info().
    """
    val = str(raw_scid).strip().lower()
    if "do not meet" in val:
        return "mdd"
    return _SCID_MAP.get(val, "mdd")


# ---------------------------------------------------------------------------
# Sex normalisation
# ---------------------------------------------------------------------------

_SEX_MAP: dict[str, str] = {
    "one": "F",   # female
    "two": "M",   # male
}


# ---------------------------------------------------------------------------
# Helpers — file parsers
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
    """Read a BIDS *_channels.tsv file (columns: name, type, units)."""
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

    # All channel names (regardless of type)
    if "name" in df.columns:
        all_names = df["name"].tolist()
        result["channel_names"] = ";".join(str(n) for n in all_names if str(n).strip())

    if "type" not in df.columns:
        return result

    type_upper = df["type"].str.upper().fillna("")
    # Only meaningful types (not n/a)
    valid_types = Counter(t for t in type_upper if t and t != "N/A")

    if valid_types:
        result["channel_types_tsv"] = ";".join(
            f"{t}:{c}" for t, c in sorted(valid_types.items())
        )
        n_eeg = valid_types.get("EEG", 0)
        result["n_eeg_channels"] = n_eeg
        result["n_other_channels"] = n_total - n_eeg

        eeg_mask = type_upper == "EEG"
        # Override channel_names with EEG-only names when types are available
        if eeg_mask.any() and "name" in df.columns:
            eeg_names = df.loc[eeg_mask, "name"].tolist()
            result["channel_names"] = ";".join(
                str(n) for n in eeg_names if str(n).strip()
            )

        if "units" in df.columns:
            non_na_units = df.loc[eeg_mask, "units"]
            non_na_units = non_na_units[
                non_na_units.str.strip().astype(bool)
                & (non_na_units.str.upper() != "N/A")
            ]
            if not non_na_units.empty:
                unit_counts = Counter(non_na_units.str.strip())
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


def read_mne_filter_info(set_path: Path) -> tuple[float | str, float | str]:
    """Load the EEGLAB .set header with MNE (preload=False) and return
    (lowpass_hz, highpass_hz) as stored in raw.info.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*boundary.*", category=RuntimeWarning)
            raw = mne.io.read_raw_eeglab(str(set_path), preload=False, verbose=False)
        return raw.info["lowpass"], raw.info["highpass"]
    except Exception as exc:
        print(f"  [WARN] MNE read error {set_path.name}: {exc}")
        return "", ""


def infer_eyes_condition_from_events(events_tsv: Path) -> str:
    """Check the *_events.tsv for Eyes Closed / Eyes Open markers."""
    if not events_tsv.exists():
        return ""
    try:
        df = pd.read_csv(events_tsv, sep="\t", dtype=str)
    except Exception:
        return ""

    if "trial_type" not in df.columns:
        return ""

    tt = df["trial_type"].str.lower().fillna("")
    has_ec = tt.str.contains("eyes closed").any()
    has_eo = tt.str.contains("eyes open").any()

    if has_ec and has_eo:
        return "both"
    if has_ec:
        return "closed"
    if has_eo:
        return "open"
    return ""


# ---------------------------------------------------------------------------
# CSV column definition
# ---------------------------------------------------------------------------

COLUMNS = [
    # --- Identifiers ---
    "new_id",           # 4-digit counter, 0001-based, per-dataset
    "old_id",           # same as subject_id
    "original_id",      # participants.tsv → Original_ID
    "dataset_id",       # "ds003478"
    "subject_id",       # BIDS participant_id, e.g. "sub-001"
    "session_id",       # BIDS ses entity (empty if absent)
    "session_num",      # numeric session number (empty if absent)
    "run",              # BIDS run entity value (empty if absent)
    # --- Demographics (participants.tsv) ---
    "age",              # age in years
    "sex",              # F / M  (normalised from "one"/"two")
    # --- Clinical scores (participants.tsv) ---
    "bdi",              # Beck Depression Inventory score
    "stai",             # Speilberger Trait Anxiety Inventory score
    "scid",             # SCID raw value
    "scid_notes",       # SCID outcome notes
    "hamd",             # Hamilton Depression Rating Scale
    # --- Diagnosis (derived from SCID column) ---
    "diagnosis",        # control | mdd | excluded
    # --- Task (BIDS filename + _eeg.json) ---
    "task_label",       # BIDS task entity, e.g. "Rest"
    "task_name",        # _eeg.json → TaskName
    "task_description", # _eeg.json → TaskDescription
    "eyes_condition",   # derived from *_events.tsv: "closed" | "open" | "both" | ""
    # --- EEG acquisition (_eeg.json) ---
    "sampling_rate_hz",       # SamplingFrequency
    "powerline_freq_hz",      # PowerLineFrequency
    "recording_duration_sec", # RecordingDuration
    "n_timepoints",           # RecordingDuration × SamplingFrequency
    "recording_type",         # RecordingType
    # --- Channel counts (_channels.tsv primary, _eeg.json fallback) ---
    "n_total_channels",
    "n_eeg_channels",
    "n_other_channels",
    "channel_types_tsv",   # e.g. "EEG:64;MISC:66"
    "channel_unit",        # from tsv units (empty if all n/a)
    "channel_names",       # semicolon-separated channel names
    # --- EEG setup (_eeg.json) ---
    "eeg_reference",        # EEGReference
    "eeg_ground",           # EEGGround
    "eeg_placement_scheme", # EEGPlacementScheme
    "software_filters",     # SoftwareFilters (serialised)
    "hardware_filters",     # HardwareFilters (empty if absent)
    "cap_manufacturer",     # CapManufacturer
    "cap_model",            # ManufacturersModelName
    "institution",          # InstitutionName
    "subject_artefact",     # SubjectArtefactDescription
    # --- Filter bounds from MNE .set header ---
    "mne_lowpass_hz",      # raw.info['lowpass']
    "mne_highpass_hz",     # raw.info['highpass']
    # --- File paths ---
    "file_name",           # basename of the .set file
    "file_extension",      # ".set"
    "file_path_raw",       # absolute path to .set
    "subject_dir",         # absolute path to sub-XXX/ directory
    "eeg_json_path",       # absolute path to *_eeg.json
    "channels_tsv_path",   # absolute path to *_channels.tsv
    "events_tsv_path",     # absolute path to *_events.tsv (empty if absent)
    "table_generated_at",
]


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def build_row(
    new_id: int,
    sub_id: str,
    sub_row: "pd.Series",
    set_path: Path,
    now_str: str,
) -> dict:
    """Return one CSV row dict for a single .set recording file."""

    # ── Parse BIDS filename entities ──────────────────────────────────────
    stem = set_path.stem   # e.g. "sub-001_task-Rest_eeg"
    entities = parse_bids_filename(stem)

    session_id = f"ses-{entities['ses']}" if "ses" in entities else ""
    run_id     = entities.get("run", "")
    task_label = entities.get("task", "")

    session_num: int | str = ""
    if session_id:
        digits = "".join(c for c in session_id if c.isdigit())
        session_num = int(digits) if digits else ""

    # ── Locate sidecar files ──────────────────────────────────────────────
    # Strip "_eeg" suffix: "sub-001_task-Rest_eeg" → "sub-001_task-Rest"
    base = stem[:-4] if stem.endswith("_eeg") else stem

    eeg_json_path     = set_path.parent / f"{base}_eeg.json"
    channels_tsv_path = set_path.parent / f"{base}_channels.tsv"
    events_tsv_path   = set_path.parent / f"{base}_events.tsv"

    # ── Load sidecar data ─────────────────────────────────────────────────
    eeg_json  = load_eeg_json(eeg_json_path)
    ch_info   = load_channels_tsv(channels_tsv_path)

    # ── MNE filter bounds from .set header ────────────────────────────────
    mne_lowpass, mne_highpass = read_mne_filter_info(set_path)

    # ── EEG acquisition from JSON ─────────────────────────────────────────
    sfreq: float | str = eeg_json.get("SamplingFrequency", "")
    if sfreq != "":
        sfreq = float(sfreq)

    rec_duration: float | str = eeg_json.get("RecordingDuration", "")
    if rec_duration != "":
        rec_duration = float(rec_duration)

    n_timepoints: int | str = ""
    if rec_duration != "" and sfreq != "":
        n_timepoints = int(float(rec_duration) * float(sfreq))

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

    # ── Eyes condition from events.tsv ────────────────────────────────────
    eyes_condition = infer_eyes_condition_from_events(events_tsv_path)

    # ── Demographics and clinical (participants.tsv) ──────────────────────
    age       = sub_row.get("age", "")
    raw_sex   = str(sub_row.get("sex", "")).strip().lower()
    sex       = _SEX_MAP.get(raw_sex, raw_sex)   # "one"→"F", "two"→"M"

    bdi       = sub_row.get("BDI", "")
    stai      = sub_row.get("STAI", "")
    raw_scid  = str(sub_row.get("SCID", "")).strip()
    scid_notes = sub_row.get("SCID_notes", "")
    hamd      = sub_row.get("HamD", "")

    original_id = sub_row.get("Original_ID", "")
    diagnosis  = _derive_diagnosis(raw_scid)

    # ── Assemble row ──────────────────────────────────────────────────────
    return {
        "new_id":                 f"{new_id:04d}",
        "old_id":                 sub_id,
        "original_id":            original_id,
        "dataset_id":             "ds003478",
        "subject_id":             sub_id,
        "session_id":             session_id,
        "session_num":            session_num,
        "run":                    run_id,
        "age":                    age,
        "sex":                    sex,
        "bdi":                    bdi,
        "stai":                   stai,
        "scid":                   raw_scid,
        "scid_notes":             scid_notes,
        "hamd":                   hamd,
        "diagnosis":              diagnosis,
        "task_label":             task_label,
        "task_name":              eeg_json.get("TaskName", ""),
        "task_description":       eeg_json.get("TaskDescription", ""),
        "eyes_condition":         eyes_condition,
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
        "software_filters":       str(eeg_json.get("SoftwareFilters", "")),
        "hardware_filters":       str(eeg_json.get("HardwareFilters", "")) if "HardwareFilters" in eeg_json else "",
        "cap_manufacturer":       eeg_json.get("CapManufacturer", ""),
        "cap_model":              eeg_json.get("ManufacturersModelName", ""),
        "institution":            eeg_json.get("InstitutionName", ""),
        "subject_artefact":       eeg_json.get("SubjectArtefactDescription", ""),
        "mne_lowpass_hz":         mne_lowpass,
        "mne_highpass_hz":        mne_highpass,
        "file_name":              set_path.name,
        "file_extension":         set_path.suffix,
        "file_path_raw":          str(set_path),
        "subject_dir":            str(set_path.parent.parent),   # sub-XXX/
        "eeg_json_path":          str(eeg_json_path)      if eeg_json_path.exists()      else "",
        "channels_tsv_path":      str(channels_tsv_path)  if channels_tsv_path.exists()  else "",
        "events_tsv_path":        str(events_tsv_path)    if events_tsv_path.exists()    else "",
        "table_generated_at":     now_str,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build master_subject_table_ds003478.csv from BIDS sidecar files.\n"
            "Run on the server where the raw dataset is accessible."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/preprocess/mdd.yaml",
        help="Preprocessing config that provides the default raw-data paths.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Path to the BIDS root directory of ds003478. Default: raw_data_path from --config.",
    )
    parser.add_argument(
        "--out_csv",
        default="metadata/ds003478/master_subject_table_ds003478.csv",
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

    # Derive diagnosis for all subjects and show summary
    if "SCID" in participants_df.columns:
        participants_df["_diagnosis"] = participants_df["SCID"].apply(_derive_diagnosis)
        print(f"  Diagnosis (from SCID):\n{participants_df['_diagnosis'].value_counts().to_string()}")

    # ── 2. Walk subject directories ───────────────────────────────────────
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

        # .set files sit in sub-XXX/eeg/ (rglob handles ses-XX/eeg/ too)
        set_files = sorted(sub_dir.rglob("*.set"))
        if not set_files:
            print(f"  [WARN] No .set files found for {sub_id}")
            skipped += 1
            continue

        for set_path in set_files:
            print(f"  [{new_id:04d}] {set_path.relative_to(bids_root)}")
            row = build_row(
                new_id=new_id,
                sub_id=sub_id,
                sub_row=sub_row,
                set_path=set_path,
                now_str=now_str,
            )
            rows.append(row)
            new_id += 1

    # ── 3. Write CSV ──────────────────────────────────────────────────────
    if not rows:
        print(
            "\n[ERROR] No rows generated.\n"
            "Check --data_root and that sub-* directories contain .set files."
        )
        return

    out_df = pd.DataFrame(rows, columns=COLUMNS)
    out_df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print("\n" + "=" * 60)
    print(f"✓  Written {len(out_df)} rows  →  {out_csv}")
    print(f"   Subjects processed : {out_df['subject_id'].nunique()}")
    print(f"   Subjects skipped   : {skipped}")
    print(f"   Diagnosis counts   :\n{out_df['diagnosis'].value_counts().to_string()}")
    eyes = out_df["eyes_condition"].value_counts()
    if not eyes.empty:
        print(f"   Eyes condition     :\n{eyes.to_string()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
