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


def parse_bids_filename(stem: str) -> dict:
    """Extract BIDS key-value entities from a filename stem."""
    _SUFFIX_TOKENS = {"eeg", "channels", "events", "electrodes", "coordsystem"}
    entities: dict = {}
    for match in re.finditer(r"([a-zA-Z]+)-([^_]+)", stem):
        key, val = match.group(1), match.group(2)
        if key not in _SUFFIX_TOKENS:
            entities[key] = val
    return entities


def load_eeg_json(eeg_json: Path) -> dict:
    """Read a BIDS *_eeg.json sidecar. Returns {} on missing or parse error."""
    if not eeg_json.exists():
        print(f"  [WARN] _eeg.json not found: {eeg_json.name}")
        return {}
    try:
        with open(eeg_json, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"  [WARN] _eeg.json parse error {eeg_json.name}: {exc}")
        return {}


def _parse_cutoff(value: str) -> float | str:
    """Convert a channels.tsv cutoff string to float.
    "DC" → 0.0,  numeric strings → float,  anything else → "".
    """
    v = value.strip().upper()
    if v in ("DC", "0", "0.0"):
        return 0.0
    try:
        return float(v)
    except ValueError:
        return ""


def load_channels_tsv(channels_tsv: Path) -> dict:
    """Read a BIDS *_channels.tsv (columns: name, type, units, low_cutoff, high_cutoff)."""
    empty = dict(
        n_total_channels="",
        n_eeg_channels="",
        n_other_channels="",
        channel_types_tsv="",
        channel_unit="",
        channel_names="",
        hw_lowpass_hz="",
        hw_highpass_hz="",
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
        hw_lowpass_hz="",
        hw_highpass_hz="",
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

    # Hardware filter cutoffs — take first EEG-row value (all rows share the same values)
    if "low_cutoff" in df.columns:
        eeg_low = df.loc[eeg_mask, "low_cutoff"].dropna()
        if not eeg_low.empty:
            result["hw_lowpass_hz"] = _parse_cutoff(eeg_low.iloc[0])

    if "high_cutoff" in df.columns:
        eeg_high = df.loc[eeg_mask, "high_cutoff"].dropna()
        if not eeg_high.empty:
            result["hw_highpass_hz"] = _parse_cutoff(eeg_high.iloc[0])

    return result


def load_sessions_tsv(sub_dir: Path, sub_id: str) -> dict[str, str]:
    """Read sub-{id}/sub-{id}_sessions.tsv.
    Returns a dict mapping session_id → recording_year (as string).
    """
    sessions_tsv = sub_dir / f"{sub_id}_sessions.tsv"
    if not sessions_tsv.exists():
        return {}
    try:
        df = pd.read_csv(sessions_tsv, sep="\t", dtype=str)
        df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)
        if "session_id" not in df.columns or "recording_year" not in df.columns:
            return {}
        return dict(zip(df["session_id"], df["recording_year"]))
    except Exception as exc:
        print(f"  [WARN] sessions.tsv parse error for {sub_id}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# CSV column definition
# ---------------------------------------------------------------------------

COLUMNS = [
    # --- Identifiers ---
    "new_id",           # 4-digit counter, 0001-based, per-dataset
    "old_id",           # same as subject_id (BIDS participant_id)
    "dataset_id",       # "ds005385"
    "subject_id",       # BIDS participant_id, e.g. "sub-001"
    "session_id",       # BIDS ses entity, e.g. "ses-1"
    "session_num",      # numeric session number
    "run",              # "" (no run entity; conditions separated by task/acq entities)
    # --- Demographics (participants.tsv) ---
    "age",
    "sex",
    "handedness",
    # --- Session participation flags (participants.tsv) ---
    "session1_participated",   # "yes" / "no"
    "late_ses1",               # count of late triggers in session 1
    "session2_participated",   # "yes" / "no"
    "late_ses2",               # count of late triggers in session 2
    # --- Late-trigger exclusion for this specific recording ---
    "late_triggers_session",   # late count for the session of this row (0 = clean)
    "late_excluded",           # True if dvs.py would skip this recording (late > 0)
    # --- Diagnosis (all healthy) ---
    "diagnosis",
    # --- Task (BIDS filename + _eeg.json) ---
    "task_label",        # BIDS task entity value, e.g. "EyesClosed"
    "task_name",         # _eeg.json → TaskName
    "task_description",  # _eeg.json → TaskDescription
    "acq",               # BIDS acq entity value: "pre" or "post"
    "eyes_condition",    # derived from task entity: "closed" or "open"
    # --- Recording context ---
    "recording_year",    # from sub-{id}_sessions.tsv
    # --- EEG acquisition (_eeg.json) ---
    "sampling_rate_hz",
    "powerline_freq_hz",
    "recording_duration_sec",
    "n_timepoints",           # RecordingDuration × SamplingFrequency
    "recording_type",
    # --- Channel info (_channels.tsv primary, _eeg.json fallback) ---
    "n_total_channels",
    "n_eeg_channels",
    "n_other_channels",
    "channel_types_tsv",
    "channel_unit",
    "channel_names",
    # --- Hardware filter cutoffs (_channels.tsv) ---
    "hw_lowpass_hz",    # low_cutoff column; "DC" → 0.0
    "hw_highpass_hz",   # high_cutoff column
    # --- EEG setup (_eeg.json) ---
    "eeg_reference",
    "eeg_ground",
    "eeg_placement_scheme",
    "software_filters",
    "hardware_filters",
    "cap_manufacturer",
    "cap_model",
    "institution",
    # --- File paths ---
    "file_name",
    "file_extension",
    "file_path_raw",
    "subject_dir",
    "eeg_json_path",
    "channels_tsv_path",
    "table_generated_at",
]


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def _infer_eyes_condition(task_label: str) -> str:
    """Derive eyes condition directly from the BIDS task entity value."""
    t = task_label.lower()
    if "closed" in t:
        return "closed"
    if "open" in t:
        return "open"
    return ""


def build_row(
    new_id: int,
    sub_id: str,
    sub_row: "pd.Series",
    edf_path: Path,
    sessions_map: dict[str, str],
    now_str: str,
) -> dict:
    """Return one CSV row dict for a single .edf recording file."""

    # ── Parse BIDS filename entities ──────────────────────────────────────
    stem = edf_path.stem   # e.g. "sub-001_ses-1_task-EyesClosed_acq-pre_eeg"
    entities = parse_bids_filename(stem)

    session_id  = f"ses-{entities['ses']}" if "ses" in entities else ""
    task_label  = entities.get("task", "")
    acq         = entities.get("acq", "")

    session_num: int | str = ""
    if session_id:
        digits = "".join(c for c in session_id if c.isdigit())
        session_num = int(digits) if digits else ""

    # ── Locate sidecar files ──────────────────────────────────────────────
    # Strip trailing "_eeg" suffix to get the shared BIDS prefix
    base = stem[:-4] if stem.endswith("_eeg") else stem
    eeg_json_path     = edf_path.parent / f"{base}_eeg.json"
    channels_tsv_path = edf_path.parent / f"{base}_channels.tsv"

    # ── Load sidecar data ─────────────────────────────────────────────────
    eeg_json = load_eeg_json(eeg_json_path)
    ch_info  = load_channels_tsv(channels_tsv_path)

    # ── Sampling rate ─────────────────────────────────────────────────────
    sfreq: float | str = ""
    if "SamplingFrequency" in eeg_json:
        sfreq = float(eeg_json["SamplingFrequency"])

    # ── n_timepoints — computed from duration × sfreq ────────────────────
    n_timepoints: int | str = ""
    rec_duration: float | str = ""
    if "RecordingDuration" in eeg_json:
        rec_duration = float(eeg_json["RecordingDuration"])
        if sfreq != "":
            n_timepoints = int(rec_duration * sfreq)

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
    age         = sub_row.get("age", "")
    sex         = _normalise_sex(sub_row.get("sex", ""))
    handedness  = sub_row.get("handedness", "")
    ses1_part   = sub_row.get("session1", "")
    late_ses1   = sub_row.get("late_ses1", "")
    ses2_part   = sub_row.get("session2", "")
    late_ses2   = sub_row.get("late_ses2", "")

    # ── Late-trigger exclusion flag (mirrors dvs.py late_trigger_threshold=0) ──
    # Resolve the late count for this specific session so each row is self-contained.
    try:
        late_triggers_session = int(float(str(late_ses1 if session_num == 1 else late_ses2)))
    except (ValueError, TypeError):
        late_triggers_session = ""
    late_excluded = bool(late_triggers_session) if late_triggers_session != "" else ""

    # ── Recording year (sessions.tsv) ─────────────────────────────────────
    recording_year = sessions_map.get(session_id, "")

    # ── Assemble row ──────────────────────────────────────────────────────
    return {
        "new_id":                   f"{new_id:04d}",
        "old_id":                   sub_id,
        "dataset_id":               "ds005385",
        "subject_id":               sub_id,
        "session_id":               session_id,
        "session_num":              session_num,
        "run":                      "",
        "age":                      age,
        "sex":                      sex,
        "handedness":               handedness,
        "session1_participated":    ses1_part,
        "late_ses1":                late_ses1,
        "session2_participated":    ses2_part,
        "late_ses2":                late_ses2,
        "late_triggers_session":    late_triggers_session,
        "late_excluded":            late_excluded,
        "diagnosis":                "control",
        "task_label":               task_label,
        "task_name":                eeg_json.get("TaskName", ""),
        "task_description":         eeg_json.get("TaskDescription", ""),
        "acq":                      acq,
        "eyes_condition":           _infer_eyes_condition(task_label),
        "recording_year":           recording_year,
        "sampling_rate_hz":         sfreq,
        "powerline_freq_hz":        eeg_json.get("PowerLineFrequency", ""),
        "recording_duration_sec":   rec_duration,
        "n_timepoints":             n_timepoints,
        "recording_type":           eeg_json.get("RecordingType", ""),
        "n_total_channels":         n_total,
        "n_eeg_channels":           n_eeg,
        "n_other_channels":         n_other,
        "channel_types_tsv":        ch_info["channel_types_tsv"],
        "channel_unit":             ch_info["channel_unit"],
        "channel_names":            ch_info["channel_names"],
        "hw_lowpass_hz":            ch_info["hw_lowpass_hz"],
        "hw_highpass_hz":           ch_info["hw_highpass_hz"],
        "eeg_reference":            eeg_json.get("EEGReference", ""),
        "eeg_ground":               eeg_json.get("EEGGround", ""),
        "eeg_placement_scheme":     eeg_json.get("EEGPlacementScheme", ""),
        "software_filters":         str(eeg_json.get("SoftwareFilters", "")),
        "hardware_filters":         str(eeg_json.get("HardwareFilters", "")) if "HardwareFilters" in eeg_json else "",
        "cap_manufacturer":         eeg_json.get("CapManufacturer", ""),
        "cap_model":                eeg_json.get("CapManufacturersModelName", ""),
        "institution":              eeg_json.get("InstitutionName", ""),
        "file_name":                edf_path.name,
        "file_extension":           edf_path.suffix,
        "file_path_raw":            str(edf_path),
        "subject_dir":              str(edf_path.parent.parent),
        "eeg_json_path":            str(eeg_json_path)      if eeg_json_path.exists()      else "",
        "channels_tsv_path":        str(channels_tsv_path)  if channels_tsv_path.exists()  else "",
        "table_generated_at":       now_str,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build master_subject_table_ds005385.csv from BIDS sidecar files.\n"
            "Run on the server where the raw dataset is accessible."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/preprocess/dvs.yaml",
        help="Preprocessing config that provides the default raw-data paths.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Path to the BIDS root directory of ds005385. Default: raw_data_path from --config.",
    )
    parser.add_argument(
        "--out_csv",
        default="metadata/ds005385/master_subject_table_ds005385.csv",
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
    print(f"  Columns: {list(participants_df.columns)}")

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

        sub_row      = participants_df.loc[sub_id]
        sessions_map = load_sessions_tsv(sub_dir, sub_id)

        edf_files = sorted(sub_dir.rglob("*.edf"))
        if not edf_files:
            print(f"  [WARN] No .edf files found for {sub_id}")
            skipped += 1
            continue

        for edf_path in edf_files:
            print(f"  [{new_id:04d}] {edf_path.relative_to(bids_root)}")
            row = build_row(
                new_id=new_id,
                sub_id=sub_id,
                sub_row=sub_row,
                edf_path=edf_path,
                sessions_map=sessions_map,
                now_str=now_str,
            )
            rows.append(row)
            new_id += 1

    # ── 3. Write CSV ──────────────────────────────────────────────────────
    if not rows:
        print(
            "\n[ERROR] No rows generated.\n"
            "Check --data_root and that sub-* directories contain .edf files."
        )
        return

    out_df = pd.DataFrame(rows, columns=COLUMNS)
    out_df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print("\n" + "=" * 60)
    print(f"✓  Written {len(out_df)} rows  →  {out_csv}")
    print(f"   Subjects processed   : {out_df['subject_id'].nunique()}")
    print(f"   Subjects skipped     : {skipped}")
    print(f"   Session distribution :\n{out_df['session_id'].value_counts().to_string()}")
    print(f"   Eyes condition       :\n{out_df['eyes_condition'].value_counts().to_string()}")
    print(f"   Acq distribution     :\n{out_df['acq'].value_counts().to_string()}")
    sfreqs = out_df["sampling_rate_hz"].dropna().unique()
    if len(sfreqs):
        print(f"   Sampling rates Hz    : {sorted(sfreqs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
