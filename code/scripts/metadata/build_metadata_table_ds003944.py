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

import pandas as pd


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
# Helpers — file parsers
# ---------------------------------------------------------------------------


def parse_vhdr(vhdr_path: Path) -> dict:
    """Parse a BrainVision .vhdr header file (INI-like text, no MNE needed)."""
    result: dict = {}
    try:
        with open(vhdr_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("SamplingInterval="):
                    try:
                        result["sampling_interval_us"] = float(
                            line.split("=", 1)[1].strip()
                        )
                    except ValueError:
                        pass
                elif line.startswith("DataPoints="):
                    try:
                        result["n_timepoints"] = int(line.split("=", 1)[1].strip())
                    except ValueError:
                        pass
    except Exception as exc:
        print(f"  [WARN] vhdr parse error {vhdr_path.name}: {exc}")
    return result


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
    """Read a BIDS *_channels.tsv file."""
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

    # Strip whitespace from all string cells
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

    # EEG channel names (from 'name' column)
    if "name" in df.columns:
        eeg_names = df.loc[eeg_mask, "name"].tolist()
        result["channel_names"] = ";".join(str(n) for n in eeg_names if str(n).strip())

    # Unit summary for EEG channels (from 'units' column; skip empty cells)
    if "units" in df.columns:
        eeg_units = df.loc[eeg_mask, "units"]
        non_empty_units = eeg_units[eeg_units.str.strip().astype(bool)]
        if not non_empty_units.empty:
            unit_counts = Counter(non_empty_units.str.strip())
            result["channel_unit"] = ";".join(
                f"{u}:{c}" for u, c in sorted(unit_counts.items())
            )

    return result


def load_eeg_json(eeg_json: Path) -> dict:
    """Read a BIDS *_eeg.json sidecar.  Returns {} if the file does not exist or
    cannot be parsed.
    """
    if not eeg_json.exists():
        print(f"  [WARN] _eeg.json not found: {eeg_json.name}")
        return {}
    try:
        with open(eeg_json, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"  [WARN] _eeg.json parse error {eeg_json.name}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# CSV column definition
# ---------------------------------------------------------------------------

# All columns that will appear in the output CSV, in order.
# Only fields that are readable from the actual source files are included.
COLUMNS = [
    # --- Identifiers ---
    "new_id",           # 4-digit counter, 0001-based, per-dataset
    "old_id",           # same as subject_id (BIDS participant_id)
    "dataset_id",       # "ds003944"
    "subject_id",       # BIDS participant_id, e.g. "sub-1448"
    "session_id",       # BIDS ses entity, e.g. "ses-01" (empty if absent)
    "session_num",      # numeric session number (empty if absent)
    "run",              # BIDS run entity value, e.g. "01" (empty if absent)
    # --- Demographics (participants.tsv) ---
    "age",              # age in years
    "sex",              # M / F  (from 'gender' column in participants.tsv)
    "race",             # participants.tsv race column
    "ethnicity",        # participants.tsv ethnicity column
    # --- Diagnosis (participants.tsv 'type' column) ---
    "diagnosis",        # normalised: "control" or "psychosis"
    # --- Task (BIDS filename + _eeg.json) ---
    "task_label",       # BIDS task entity value, e.g. "Rest"
    "task_name",        # _eeg.json → TaskName
    "task_description", # _eeg.json → TaskDescription
    "eyes_condition",   # inferred from Instructions / TaskDescription; empty if unclear
    # --- EEG acquisition (_eeg.json, with vhdr fallback) ---
    "sampling_rate_hz",       # SamplingFrequency (JSON) or 1e6/SamplingInterval (vhdr)
    "powerline_freq_hz",      # PowerLineFrequency
    "recording_duration_sec", # RecordingDuration (JSON) or n_timepoints/sfreq
    "n_timepoints",           # DataPoints (vhdr) or duration×sfreq
    "recording_type",         # RecordingType
    # --- Channel counts (_channels.tsv primary, _eeg.json fallback) ---
    "n_total_channels",
    "n_eeg_channels",
    "n_other_channels",
    "channel_types_tsv",  # e.g. "EEG:61;EOG:1;ECG:1;MISC:1"
    "channel_unit",       # e.g. "microV:61"  (EEG channels only, non-empty units)
    "channel_names",      # semicolon-separated EEG channel names
    # --- EEG setup (_eeg.json) ---
    "eeg_reference",        # EEGReference
    "eeg_ground",           # EEGGround
    "eeg_placement_scheme", # EEGPlacementScheme
    "software_filters",     # SoftwareFilters (serialised string)
    "hardware_filters",     # HardwareFilters (empty if absent)
    "cap_manufacturer",     # CapManufacturer
    "cap_model",            # ManufacturersModelName
    "institution",          # InstitutionName
    # --- File paths ---
    "file_name",          # basename of the .vhdr file
    "file_extension",     # ".vhdr"
    "file_path_raw",      # absolute path to .vhdr
    "subject_dir",        # absolute path to sub-XXXX/ directory
    "eeg_json_path",      # absolute path to *_eeg.json  (empty if missing)
    "channels_tsv_path",  # absolute path to *_channels.tsv (empty if missing)
    "table_generated_at",
]

# Map participants.tsv 'type' values → normalised diagnosis label
_DIAGNOSIS_MAP: dict[str, str] = {
    "control":   "control",
    "psychosis": "psychosis",
    "fep":       "psychosis",
}


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def _infer_eyes_condition(eeg_json: dict) -> str:
    """Scan _eeg.json text fields for open/closed eye keywords.
    Checks: Instructions, TaskDescription, TaskName (in that order).
    Returns "open", "closed", or "" if not determinable.
    """
    for key in ("Instructions", "TaskDescription", "TaskName"):
        val = str(eeg_json.get(key, "")).lower()
        if not val:
            continue
        if "closed" in val:
            return "closed"
        if "open" in val:
            return "open"
    return ""


def build_row(
    new_id: int,
    sub_id: str,
    sub_row: "pd.Series",
    vhdr_path: Path,
    now_str: str,
) -> dict:
    """Return one CSV row dict for a single .vhdr recording file."""

    # ── Parse BIDS filename entities ──────────────────────────────────────
    stem = vhdr_path.stem   # e.g. "sub-1448_task-Rest_eeg"
    entities = parse_bids_filename(stem)

    session_id = f"ses-{entities['ses']}" if "ses" in entities else ""
    run_id     = entities.get("run", "")
    task_label = entities.get("task", "")

    session_num: int | str = ""
    if session_id:
        digits = "".join(c for c in session_id if c.isdigit())
        session_num = int(digits) if digits else ""

    # ── Locate sidecar files ──────────────────────────────────────────────
    # Strip trailing "_eeg" suffix to get the shared BIDS prefix
    # e.g. "sub-1448_task-Rest_eeg" → "sub-1448_task-Rest"
    base = stem[:-4] if stem.endswith("_eeg") else stem

    eeg_json_path     = vhdr_path.parent / f"{base}_eeg.json"
    channels_tsv_path = vhdr_path.parent / f"{base}_channels.tsv"

    # ── Load sidecar data ─────────────────────────────────────────────────
    eeg_json  = load_eeg_json(eeg_json_path)
    ch_info   = load_channels_tsv(channels_tsv_path)
    vhdr_info = parse_vhdr(vhdr_path)

    # ── Sampling rate ─────────────────────────────────────────────────────
    # Priority: _eeg.json SamplingFrequency → vhdr SamplingInterval
    sfreq: float | str
    if "SamplingFrequency" in eeg_json:
        sfreq = float(eeg_json["SamplingFrequency"])
    elif "sampling_interval_us" in vhdr_info:
        sfreq = round(1e6 / vhdr_info["sampling_interval_us"], 6)
    else:
        sfreq = ""

    # ── n_timepoints ──────────────────────────────────────────────────────
    # Priority: vhdr DataPoints → computed from RecordingDuration × sfreq
    n_timepoints: int | str
    if "n_timepoints" in vhdr_info:
        n_timepoints = vhdr_info["n_timepoints"]
    elif "RecordingDuration" in eeg_json and sfreq != "":
        n_timepoints = int(float(eeg_json["RecordingDuration"]) * float(sfreq))
    else:
        n_timepoints = ""

    # ── Recording duration ────────────────────────────────────────────────
    rec_duration: float | str
    if "RecordingDuration" in eeg_json:
        rec_duration = float(eeg_json["RecordingDuration"])
    elif n_timepoints != "" and sfreq != "":
        rec_duration = round(int(n_timepoints) / float(sfreq), 3)
    else:
        rec_duration = ""

    # ── Channel counts: TSV primary, JSON fallback ────────────────────────
    # If _channels.tsv was missing, try to fill from JSON channel count keys
    n_eeg = ch_info["n_eeg_channels"]
    if n_eeg == "" and "EEGChannelCount" in eeg_json:
        n_eeg = int(eeg_json["EEGChannelCount"])

    n_total = ch_info["n_total_channels"]
    if n_total == "" and eeg_json:
        # Sum all *ChannelCount keys present in JSON
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
    age       = sub_row.get("age", "")
    sex       = _normalise_sex(sub_row.get("gender", ""))  # column named 'gender' in this dataset
    race      = sub_row.get("race", "")
    ethnicity = sub_row.get("ethnicity", "")
    raw_type  = str(sub_row.get("type", "")).strip()
    diagnosis = _DIAGNOSIS_MAP.get(raw_type.lower(), raw_type.lower())

    # ── Assemble row ──────────────────────────────────────────────────────
    return {
        "new_id":                 f"{new_id:04d}",
        "old_id":                 sub_id,
        "dataset_id":             "ds003944",
        "subject_id":             sub_id,
        "session_id":             session_id,
        "session_num":            session_num,
        "run":                    run_id,
        "age":                    age,
        "sex":                    sex,
        "race":                   race,
        "ethnicity":              ethnicity,
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
        # SoftwareFilters may be the string "n/a" or a dict — serialise both
        "software_filters":       str(eeg_json.get("SoftwareFilters", "")),
        "hardware_filters":       str(eeg_json.get("HardwareFilters", "")) if "HardwareFilters" in eeg_json else "",
        "cap_manufacturer":       eeg_json.get("CapManufacturer", ""),
        "cap_model":              eeg_json.get("ManufacturersModelName", ""),
        "institution":            eeg_json.get("InstitutionName", ""),
        "file_name":              vhdr_path.name,
        "file_extension":         vhdr_path.suffix,
        "file_path_raw":          str(vhdr_path),
        "subject_dir":            str(vhdr_path.parent.parent),   # sub-XXXX/
        "eeg_json_path":          str(eeg_json_path)      if eeg_json_path.exists()      else "",
        "channels_tsv_path":      str(channels_tsv_path)  if channels_tsv_path.exists()  else "",
        "table_generated_at":     now_str,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build master_subject_table_ds003944.csv from BIDS sidecar files.\n"
            "Run on the server where the raw dataset is accessible."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/preprocess/fep.yaml",
        help="Preprocessing config that provides the default raw-data paths.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Path to the BIDS root directory of ds003944. Default: raw_data_path from --config.",
    )
    parser.add_argument(
        "--out_csv",
        default="metadata/ds003944/master_subject_table_ds003944.csv",
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
    print(f"  Diagnoses:\n{participants_df['type'].value_counts().to_string()}")

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
        sub_id = sub_dir.name   # e.g. "sub-1448"

        if sub_id not in participants_df.index:
            print(f"  [WARN] {sub_id} not in participants.tsv — skipping")
            skipped += 1
            continue

        sub_row = participants_df.loc[sub_id]

        # Recursively find all .vhdr files (covers ses-XX/eeg/ sub-directories)
        vhdr_files = sorted(sub_dir.rglob("*.vhdr"))
        if not vhdr_files:
            print(f"  [WARN] No .vhdr files found for {sub_id}")
            skipped += 1
            continue

        for vhdr_path in vhdr_files:
            print(f"  [{new_id:04d}] {vhdr_path.relative_to(bids_root)}")
            row = build_row(
                new_id=new_id,
                sub_id=sub_id,
                sub_row=sub_row,
                vhdr_path=vhdr_path,
                now_str=now_str,
            )
            rows.append(row)
            new_id += 1

    # ── 3. Write CSV ──────────────────────────────────────────────────────
    if not rows:
        print(
            "\n[ERROR] No rows generated.\n"
            "Check --data_root and that sub-* directories contain .vhdr files."
        )
        return

    out_df = pd.DataFrame(rows, columns=COLUMNS)
    out_df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print("\n" + "=" * 60)
    print(f"✓  Written {len(out_df)} rows  →  {out_csv}")
    print(f"   Subjects processed : {out_df['subject_id'].nunique()}")
    print(f"   Subjects skipped   : {skipped}")
    print(f"   Diagnosis counts   :\n{out_df['diagnosis'].value_counts().to_string()}")
    sfreqs = out_df["sampling_rate_hz"].dropna().unique()
    if len(sfreqs):
        print(f"   Sampling rates Hz  : {sorted(sfreqs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
