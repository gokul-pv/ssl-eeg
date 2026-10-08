#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from _config_paths import config_value

import pandas as pd


# ---------------------------------------------------------------------------
# .vhdr parser
# ---------------------------------------------------------------------------

# EOG channel names embedded in LEMON .vhdr files (BrainVision marks all
# channels as EEG by default; we re-classify by name, matching the
# preprocessor's _fix_channels logic in code/src/preprocess/lemon.py)
_EOG_NAMES: frozenset[str] = frozenset({"VEOG", "HEOG", "LEOG", "REOG"})

# Sex normalisation — LEMON encodes sex as 1 (male) / 2 (female) in participants.csv
_SEX_NORM: dict[str, str] = {
    "m": "M", "male": "M", "1": "M", "man": "M",
    "f": "F", "female": "F", "2": "F", "woman": "F",
}


def _normalise_sex(raw: str) -> str:
    return _SEX_NORM.get(str(raw).strip().lower(), str(raw).strip())


def parse_vhdr_lemon(vhdr_path: Path) -> dict:
    """Parse a LEMON BrainVision .vhdr header file."""
    result: dict = dict(
        sfreq="",
        n_timepoints="",
        n_total_channels="",
        n_eeg_channels="",
        n_other_channels="",
        channel_types_tsv="",
        channel_unit="",
        channel_names="",
    )

    try:
        text = vhdr_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"  [WARN] Cannot read {vhdr_path.name}: {exc}")
        return result

    # ── Common Infos ──────────────────────────────────────────────────────
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("SamplingInterval="):
            try:
                interval_us = float(line.split("=", 1)[1].strip())
                result["sfreq"] = round(1e6 / interval_us, 6)
            except ValueError:
                pass
        elif line.startswith("DataPoints="):
            try:
                result["n_timepoints"] = int(line.split("=", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("NumberOfChannels="):
            try:
                result["n_total_channels"] = int(line.split("=", 1)[1].strip())
            except ValueError:
                pass

    # ── Channel Infos ─────────────────────────────────────────────────────
    # Match lines like: Ch1=Fp1,,0.1,µV  or  Ch12=VEOG,,0.1,µV
    ch_pattern = re.compile(r"^Ch\d+=(.+)$", re.IGNORECASE)
    eeg_names: list[str] = []
    eeg_units: list[str] = []
    type_counter: Counter = Counter()

    for line in text.splitlines():
        m = ch_pattern.match(line.strip())
        if not m:
            continue
        parts = m.group(1).split(",")
        name = parts[0].strip() if parts else ""
        unit = parts[3].strip() if len(parts) > 3 else ""

        if name.upper() in _EOG_NAMES:
            type_counter["EOG"] += 1
        else:
            type_counter["EEG"] += 1
            eeg_names.append(name)
            if unit:
                eeg_units.append(unit)

    if type_counter:
        result["channel_types_tsv"] = ";".join(
            f"{t}:{c}" for t, c in sorted(type_counter.items())
        )
        n_eeg = type_counter.get("EEG", 0)
        n_other = type_counter.get("EOG", 0)
        result["n_eeg_channels"]   = n_eeg
        result["n_other_channels"] = n_other
        # Prefer NumberOfChannels from header; fall back to sum of parsed channels
        if result["n_total_channels"] == "":
            result["n_total_channels"] = n_eeg + n_other

    if eeg_names:
        result["channel_names"] = ";".join(eeg_names)

    if eeg_units:
        unit_counts = Counter(eeg_units)
        result["channel_unit"] = ";".join(
            f"{u}:{c}" for u, c in sorted(unit_counts.items())
        )

    return result


# ---------------------------------------------------------------------------
# participants.csv loader
# ---------------------------------------------------------------------------

# Maps raw participants.csv column names → clean output names.
# Unnamed / empty-header columns are dropped before renaming.
_COL_RENAME: dict[str, str] = {
    "sex":                                                      "sex",
    "age":                                                      "age",
    "Handedness":                                               "handedness",
    "Education":                                                "education",
    "DRUG":                                                     "drug",
    "DRUG_0=negative_1=Positive":                               "drug_positive",
    "Smoking":                                                  "smoking",
    "Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)": "smoking_num",
    "SKID_Diagnoses":                                           "skid_diagnosis",
    "Hamilton_Scale":                                           "hamilton_scale",
    "BSL23_sumscore":                                           "bsl23_sumscore",
    "BSL23_behavior":                                           "bsl23_behavior",
    "AUDIT":                                                    "audit",
    "Standard_Alcoholunits_Last_28days":                        "alcohol_units_28d",
    "Alcohol_Dependence_In_1st-3rd_Degree_relative":            "alcohol_dep_relative",
    "Relationship_Status":                                      "relationship_status",
}

_CLINICAL_COLS: list[str] = list(_COL_RENAME.values())


def load_participants_csv(data_root: Path) -> pd.DataFrame:
    """Read participants.csv (full clinical metadata)."""
    csv_path = data_root / "participants.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"participants.csv not found: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0, dtype=str)

    # Drop columns whose header is empty or NaN
    df = df.loc[:, df.columns.str.strip().astype(bool)]

    # Rename known columns; keep extras as-is (harmless)
    df = df.rename(columns=_COL_RENAME)

    # Ensure index is string and strip whitespace
    df.index = df.index.astype(str).str.strip()

    return df


# ---------------------------------------------------------------------------
# CSV column definition
# ---------------------------------------------------------------------------

COLUMNS: list[str] = [
    # --- Identifiers ---
    "new_id",       # 4-digit counter, 0001-based, per-dataset
    "old_id",       # same as subject_id
    "dataset_id",   # "LEMON"
    "subject_id",   # e.g. "sub-032301"
    "session_id",   # "" (single-session dataset; no ses-* dirs)
    "session_num",  # "" (no session entity in filename)
    "run",          # "" (no run entity in filename)
    # --- Demographics (participants.csv) ---
    "age",          # age range string, e.g. "20-25"
    "sex",          # normalised to M/F (source: 1=male, 2=female)
    "handedness",
    "education",
    # --- Lifestyle / clinical (participants.csv) ---
    "drug",
    "drug_positive",
    "smoking",
    "smoking_num",
    "skid_diagnosis",
    "hamilton_scale",
    "bsl23_sumscore",
    "bsl23_behavior",
    "audit",
    "alcohol_units_28d",
    "alcohol_dep_relative",
    "relationship_status",
    # --- Diagnosis (all healthy) ---
    "diagnosis",
    # --- Task (fixed for all LEMON recordings) ---
    "task_label",      # "" (no task entity in filename)
    "task_name",       # "RestingState"
    "task_description", # "" (no JSON sidecar)
    "eyes_condition",  # "open" (fixation cross, eyes open — documented fact)
    # --- EEG acquisition (from .vhdr) ---
    "sampling_rate_hz",
    "powerline_freq_hz",    # 50 Hz (European; hardcoded — no JSON sidecar)
    "recording_duration_sec",
    "n_timepoints",
    "recording_type",       # "" (no JSON sidecar)
    # --- Channel info (from .vhdr [Channel Infos]) ---
    "n_total_channels",
    "n_eeg_channels",
    "n_other_channels",
    "channel_types_tsv",  # e.g. "EEG:58;EOG:4"
    "channel_unit",     # e.g. "µV:58"
    "channel_names",    # semicolon-separated EEG channel names
    # --- EEG setup (no JSON sidecar — all empty) ---
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
    "eeg_json_path",    # "" (no JSON sidecar exists)
    "channels_tsv_path", # "" (no channels TSV exists)
    "table_generated_at",
]


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def build_row(
    new_id: int,
    sub_id: str,
    sub_row: "pd.Series",
    vhdr_path: Path,
    now_str: str,
) -> dict:
    """Return one CSV row dict for a single .vhdr recording file."""

    vhdr_info = parse_vhdr_lemon(vhdr_path)

    # ── Recording duration ────────────────────────────────────────────────
    rec_duration: float | str = ""
    sfreq = vhdr_info["sfreq"]
    n_tp  = vhdr_info["n_timepoints"]
    if n_tp != "" and sfreq != "":
        rec_duration = round(int(n_tp) / float(sfreq), 3)

    # ── Demographics / clinical ───────────────────────────────────────────
    clinical = {col: sub_row.get(col, "") for col in _CLINICAL_COLS}
    clinical["sex"] = _normalise_sex(str(clinical.get("sex", "")))

    return {
        "new_id":                f"{new_id:04d}",
        "old_id":                sub_id,
        "dataset_id":            "LEMON",
        "subject_id":            sub_id,
        "session_id":            "",
        "session_num":           "",
        "run":                   "",
        **clinical,
        "diagnosis":             "control",
        "task_label":            "",
        "task_name":             "RestingState",
        "task_description":      "",
        "eyes_condition":        "open",
        "sampling_rate_hz":      sfreq,
        "powerline_freq_hz":     50,
        "recording_duration_sec": rec_duration,
        "n_timepoints":          n_tp,
        "recording_type":        "",
        "n_total_channels":      vhdr_info["n_total_channels"],
        "n_eeg_channels":        vhdr_info["n_eeg_channels"],
        "n_other_channels":      vhdr_info["n_other_channels"],
        "channel_types_tsv":     vhdr_info["channel_types_tsv"],
        "channel_unit":          vhdr_info["channel_unit"],
        "channel_names":         vhdr_info["channel_names"],
        "eeg_reference":         "",
        "eeg_ground":            "",
        "eeg_placement_scheme":  "",
        "software_filters":      "",
        "hardware_filters":      "",
        "cap_manufacturer":      "",
        "cap_model":             "",
        "institution":           "",
        "file_name":             vhdr_path.name,
        "file_extension":        vhdr_path.suffix,
        "file_path_raw":         str(vhdr_path),
        "subject_dir":           str(vhdr_path.parent.parent),
        "eeg_json_path":         "",
        "channels_tsv_path":     "",
        "table_generated_at":    now_str,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build master_subject_table_lemon.csv from LEMON BrainVision headers.\n"
            "Run on the server where the raw dataset is accessible."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/preprocess/lemon.yaml",
        help="Preprocessing config that provides the default raw-data paths.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Path to the LEMON dataset root (contains participants.csv + sub-* dirs). Default: raw_data_path from --config.",
    )
    parser.add_argument(
        "--out_csv",
        default="metadata/LEMON/master_subject_table_lemon.csv",
        help="Output CSV path (relative to project root / code/, or absolute).",
    )
    args = parser.parse_args()
    if args.data_root is None:
        args.data_root = config_value(args.config, "raw_data_path")

    data_root = Path(args.data_root)
    out_csv   = Path(args.out_csv)

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root not found: {data_root}\n"
            "Run this script on the server where the dataset is mounted."
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. participants.csv ───────────────────────────────────────────────
    participants_df = load_participants_csv(data_root)
    print(f"Loaded {len(participants_df)} subjects from participants.csv")
    print(f"  Columns: {list(participants_df.columns)}")

    # ── 2. Walk subject directories ───────────────────────────────────────
    now_str  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    rows: list[dict] = []
    new_id   = 1
    skipped  = 0

    sub_dirs = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )
    print(f"\nFound {len(sub_dirs)} sub-* directories\n")

    for sub_dir in sub_dirs:
        sub_id = sub_dir.name   # e.g. "sub-032301"

        if sub_id not in participants_df.index:
            print(f"  [WARN] {sub_id} not in participants.csv — skipping")
            skipped += 1
            continue

        sub_row   = participants_df.loc[sub_id]
        rseeg_dir = sub_dir / "RSEEG"

        if not rseeg_dir.exists():
            print(f"  [WARN] No RSEEG/ directory for {sub_id} — skipping")
            skipped += 1
            continue

        vhdr_files = sorted(rseeg_dir.glob("*.vhdr"))
        if not vhdr_files:
            print(f"  [WARN] No .vhdr files found for {sub_id}")
            skipped += 1
            continue

        for vhdr_path in vhdr_files:
            print(f"  [{new_id:04d}] {vhdr_path.relative_to(data_root)}")
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
            "Check --data_root and that sub-* directories contain RSEEG/*.vhdr files."
        )
        return

    out_df = pd.DataFrame(rows, columns=COLUMNS)
    out_df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print("\n" + "=" * 60)
    print(f"✓  Written {len(out_df)} rows  →  {out_csv}")
    print(f"   Subjects processed : {out_df['subject_id'].nunique()}")
    print(f"   Subjects skipped   : {skipped}")
    sfreqs = out_df["sampling_rate_hz"].dropna().unique()
    if len(sfreqs):
        print(f"   Sampling rates Hz  : {sorted(sfreqs)}")
    n_eeg_vals = out_df["n_eeg_channels"].dropna().unique()
    if len(n_eeg_vals):
        print(f"   EEG channel counts : {sorted(n_eeg_vals)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
