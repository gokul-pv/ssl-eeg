#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from _config_paths import config_value

import pandas as pd


# ---------------------------------------------------------------------------
# Neuroscan .dap parser  (copied from src/preprocess/scz.py)
# ---------------------------------------------------------------------------


def _parse_dap_meta(dap_path: Path) -> dict[str, str]:
    """Parse a Neuroscan .dap file (key = value text) into a flat dict."""
    meta: dict[str, str] = {}
    with open(dap_path, encoding="utf-8-sig") as f:
        for line in f:
            if "=" in line:
                k, _, v = line.partition("=")
                meta[k.strip()] = v.strip()
    return meta


# ---------------------------------------------------------------------------
# Neuroscan .rs3 parser  (copied from src/preprocess/scz.py)
# ---------------------------------------------------------------------------


def _parse_rs3_labels(rs3_path: Path) -> tuple[list[str], list[str]]:
    """Parse a Neuroscan .rs3 file and return (eeg_labels, aux_labels)."""
    eeg_labels: list[str] = []
    aux_labels: list[str] = []
    in_eeg_block = False
    in_aux_block = False

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


# ---------------------------------------------------------------------------
# .rs3 → channel info dict
# ---------------------------------------------------------------------------

# Aux channel → channel type string (for channel_types_tsv summary)
_AUX_TYPE_MAP: dict[str, str] = {
    "HEO":     "EOG",
    "VEO":     "EOG",
    "EKG":     "ECG",
    "EMG":     "EMG",
    "Trigger": "STIM",
}


def parse_rs3(rs3_path: Path) -> dict:
    """Parse .rs3 and return a channel-info dict:
      n_total_channels, n_eeg_channels, n_other_channels,
      channel_types_tsv, channel_unit, channel_names
    """
    empty: dict = dict(
        n_total_channels="",
        n_eeg_channels="",
        n_other_channels="",
        channel_types_tsv="",
        channel_unit="",
        channel_names="",
    )
    try:
        eeg_labels, aux_labels = _parse_rs3_labels(rs3_path)
    except Exception as exc:
        print(f"  [WARN] .rs3 parse error {rs3_path.name}: {exc}")
        return empty

    if not eeg_labels and not aux_labels:
        print(f"  [WARN] No channel labels found in {rs3_path.name}")
        return empty

    n_eeg   = len(eeg_labels)   # 64
    n_other = len(aux_labels)   # 5
    n_total = n_eeg + n_other   # 69

    # Build type counter (EEG + aux mapped types)
    type_counter: Counter = Counter({"EEG": n_eeg})
    for aux in aux_labels:
        type_counter[_AUX_TYPE_MAP.get(aux, "MISC")] += 1

    channel_types_tsv = ";".join(
        f"{t}:{c}" for t, c in sorted(type_counter.items())
    )

    # channel_unit is intentionally left out here — it is set in build_row()
    # using the DataUnit from .dap so it reflects the actual recorded unit.
    return dict(
        n_total_channels=n_total,
        n_eeg_channels=n_eeg,
        n_other_channels=n_other,
        channel_types_tsv=channel_types_tsv,
        channel_names=";".join(eeg_labels),
    )


# ---------------------------------------------------------------------------
# .dap → acquisition info dict
# ---------------------------------------------------------------------------


def parse_dap(dap_path: Path) -> dict:
    """Parse .dap and return an acquisition-info dict:
      sampling_rate_hz, n_timepoints, recording_duration_sec,
      n_channels_dap (NumChannels from .dap, for cross-checking with .rs3),
      channel_unit   (formatted from DataUnit field, e.g. "µV")
    """
    empty = dict(
        sampling_rate_hz="",
        n_timepoints="",
        recording_duration_sec="",
        n_channels_dap="",
        channel_unit_prefix="",
    )
    try:
        meta = _parse_dap_meta(dap_path)
    except Exception as exc:
        print(f"  [WARN] .dap parse error {dap_path.name}: {exc}")
        return empty

    sfreq   = meta.get("SampleFreqHz", "")
    n_samp  = meta.get("NumSamples", "")
    n_chans = meta.get("NumChannels", "")

    # Normalise DataUnit: "uV" / "microvolts" / "µV" → "µV"
    raw_unit = meta.get("DataUnit", "").lower().strip()
    if raw_unit in ("uv", "µv", "microvolt", "microvolts", "microV"):
        unit_prefix = "µV"
    elif raw_unit:
        unit_prefix = raw_unit
    else:
        unit_prefix = "µV"  # fallback — known from paper

    rec_dur: float | str = ""
    if sfreq and n_samp:
        try:
            rec_dur = round(int(n_samp) / float(sfreq), 3)
        except (ValueError, ZeroDivisionError):
            pass

    return dict(
        sampling_rate_hz=float(sfreq) if sfreq else "",
        n_timepoints=int(n_samp) if n_samp else "",
        recording_duration_sec=rec_dur,
        n_channels_dap=int(n_chans) if n_chans else "",
        channel_unit_prefix=unit_prefix,
    )


# ---------------------------------------------------------------------------
# Excel demographics loader
# ---------------------------------------------------------------------------

_SEX_NORM: dict[str, str] = {
    "m": "M", "male": "M", "1": "M", "man": "M",
    "f": "F", "female": "F", "2": "F", "woman": "F",
}


def _normalise_sex(raw: str) -> str:
    return _SEX_NORM.get(str(raw).strip().lower(), str(raw).strip())


def load_excel(excel_path: Path) -> pd.DataFrame:
    """Read the Demographic sheet from the SCZ Excel file."""
    df = pd.read_excel(excel_path, sheet_name="Demographic", dtype=str)

    # Strip whitespace from all cells and column names
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)

    if "code" not in df.columns:
        raise ValueError(
            f"'code' column not found in Demographic sheet of {excel_path.name}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.set_index("code")
    df.index = df.index.str.strip()
    return df


# ---------------------------------------------------------------------------
# CSV column definition
# ---------------------------------------------------------------------------

# Core columns shared across all dataset metadata tables.
# SCZ-specific addition: bno_code (will be NaN in other datasets on pd.concat).
COLUMNS: list[str] = [
    # --- Identifiers ---
    "new_id",        # 4-digit counter, 0001-based, per-dataset
    "old_id",        # same as subject_id
    "dataset_id",    # "scz"
    "subject_id",    # e.g. "sch_002"
    "session_id",    # "" (single-session dataset)
    "session_num",   # "" (no session entity)
    "run",           # "" (no run entity)
    # --- Demographics (from Excel Demographic sheet) ---
    "age",
    "sex",
    "edu_years",     # years of formal education — standard demographic covariate
    # --- Diagnosis ---
    "diagnosis",     # "control" or "scz"  (matches inter-dataset convention)
    "bno_code",      # SCZ-specific: raw ICD-10 value(s), e.g. "F20.0"; blank for HC
    "years_scz",     # years since first diagnosis (illness duration); blank for HC
    # --- Task (hardcoded — no JSON sidecar) ---
    "task_label",
    "task_name",
    "task_description",
    "eyes_condition",
    # --- EEG acquisition (from .dap) ---
    "sampling_rate_hz",
    "powerline_freq_hz",
    "recording_duration_sec",
    "n_timepoints",
    "recording_type",
    # --- Channel info (from .rs3) ---
    "n_total_channels",
    "n_eeg_channels",
    "n_other_channels",
    "channel_types_tsv",
    "channel_unit",
    "channel_names",
    # --- EEG setup (hardcoded — no JSON sidecar) ---
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
    "file_path_raw",    # absolute path to .dat file
    "subject_dir",      # flat raw directory (all subjects share one dir)
    "eeg_json_path",    # "" (no BIDS sidecar)
    "channels_tsv_path", # "" (no BIDS sidecar)
    "table_generated_at",
]


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------


def build_row(
    new_id: int,
    sub_id: str,
    sub_row: "pd.Series",
    dat_path: Path,
    dap_path: Path,
    rs3_path: Path,
    now_str: str,
) -> dict:
    """Return one CSV row dict for a single subject recording."""

    acq  = parse_dap(dap_path)
    ch   = parse_rs3(rs3_path)

    # Cross-check NumChannels from .dap against .rs3 label count
    n_dap = acq["n_channels_dap"]
    n_rs3 = ch["n_total_channels"]
    if n_dap and n_rs3 and int(n_dap) != int(n_rs3):
        print(
            f"  [WARN] Channel count mismatch for {sub_id}: "
            f".dap NumChannels={n_dap}, .rs3 label count={n_rs3}"
        )

    # channel_unit built from .dap DataUnit + total channel count
    unit_prefix = acq.get("channel_unit_prefix", "µV")
    n_total_for_unit = n_rs3 if n_rs3 else (n_dap if n_dap else 69)
    channel_unit = f"{unit_prefix}:{n_total_for_unit}"

    # ── Diagnosis ────────────────────────────────────────────────────────
    # BNO may contain comma-separated dual diagnoses (e.g. "F2520, F2010").
    # Classify as "scz" if ANY individual code starts with "F20".
    bno_val = sub_row.get("BNO", "")
    bno_raw = "" if (bno_val is None or str(bno_val).strip().lower() in ("", "nan")) else str(bno_val).strip()
    bno_codes = [c.strip() for c in bno_raw.split(",") if c.strip()]
    is_scz    = any(c.startswith("F20") for c in bno_codes)
    diagnosis = "scz" if is_scz else "control"
    bno_code  = bno_raw  # empty string for HC, raw ICD-10 value(s) for SZ

    # ── Demographics (from Demographic sheet) ────────────────────────────
    age: str = ""
    for col in ("age", "Age", "AGE"):
        if col in sub_row.index:
            age = str(sub_row[col]).strip()
            break

    sex: str = ""
    for col in ("sex", "Sex", "SEX", "gender", "Gender", "GENDER"):
        if col in sub_row.index:
            sex = _normalise_sex(str(sub_row[col]))
            break

    edu_years_val = sub_row.get("edu_years", "")
    edu_years = "" if (edu_years_val is None or str(edu_years_val).strip().lower() in ("", "nan")) else str(edu_years_val).strip()

    years_scz_val = sub_row.get("years_scz", "")
    years_scz = "" if (years_scz_val is None or str(years_scz_val).strip().lower() in ("", "nan")) else str(years_scz_val).strip()

    return {
        "new_id":                 f"{new_id:04d}",
        "old_id":                 sub_id,
        "dataset_id":             "scz",
        "subject_id":             sub_id,
        "session_id":             "",
        "session_num":            "",
        "run":                    "",
        "age":                    age,
        "sex":                    sex,
        "edu_years":              edu_years,
        "diagnosis":              diagnosis,
        "bno_code":               bno_code,
        "years_scz":              years_scz,
        "task_label":             "RestEC",
        "task_name":              "Resting State Eyes Closed",
        "task_description":       "2-minute eyes-closed resting-state EEG (Racz et al. 2025)",
        "eyes_condition":         "closed",
        "sampling_rate_hz":       acq["sampling_rate_hz"],
        "powerline_freq_hz":      50,   # Budapest, Hungary — European power grid
        "recording_duration_sec": acq["recording_duration_sec"],
        "n_timepoints":           acq["n_timepoints"],
        "recording_type":         "continuous",
        "n_total_channels":       ch["n_total_channels"],
        "n_eeg_channels":         ch["n_eeg_channels"],
        "n_other_channels":       ch["n_other_channels"],
        "channel_types_tsv":      ch["channel_types_tsv"],
        "channel_unit":           channel_unit,
        "channel_names":          ch["channel_names"],
        "eeg_reference":          "Linked mastoids (M1/M2)",
        "eeg_ground":             "",
        "eeg_placement_scheme":   "standard_1020",
        "software_filters":       "n/a",
        "hardware_filters":       "",
        "cap_manufacturer":       "Neuroscan",
        "cap_model":              "SynAmps",
        "institution":            "University of Pécs, Hungary",
        "file_name":              dat_path.name,
        "file_extension":         dat_path.suffix,
        "file_path_raw":          str(dat_path),
        "subject_dir":            str(dat_path.parent),
        "eeg_json_path":          "",
        "channels_tsv_path":      "",
        "table_generated_at":     now_str,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build master_subject_table_scz.csv from Neuroscan .dap/.rs3 files.\n"
            "Run on the server where the raw dataset is accessible."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/preprocess/scz.yaml",
        help="Preprocessing config that provides the default raw-data paths.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Path to the flat folder containing sch_XXX_ec.{dat,dap,rs3} files. Default: raw_data_path from --config.",
    )
    parser.add_argument(
        "--excel_path",
        default=None,
        help="Path to schizophrenia_EI_dynamics_table.xlsx. Default: excel_path from --config.",
    )
    parser.add_argument(
        "--out_csv",
        default="metadata/scz/master_subject_table_scz.csv",
        help="Output CSV path (relative to code/ directory, or absolute).",
    )
    args = parser.parse_args()
    if args.data_root is None:
        args.data_root = config_value(args.config, "raw_data_path")
    if args.excel_path is None:
        args.excel_path = config_value(args.config, "excel_path")

    data_root  = Path(args.data_root)
    excel_path = Path(args.excel_path)
    out_csv    = Path(args.out_csv)

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root not found: {data_root}\n"
            "Run this script on the server where the dataset is mounted."
        )
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel file not found: {excel_path}"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Excel demographics ────────────────────────────────────────
    participants_df = load_excel(excel_path)
    print(f"Loaded {len(participants_df)} rows from Excel Demographic sheet")
    print(f"  Columns : {list(participants_df.columns)}")
    bno_col = participants_df.get("BNO", pd.Series(dtype=str))
    scz_count = bno_col.apply(
        lambda v: any(c.strip().startswith("F20") for c in str(v).split(","))
        if pd.notna(v) and str(v).strip().lower() not in ("", "nan") else False
    ).sum()
    print(f"  Diagnosis: control={len(participants_df) - scz_count}, scz={scz_count}")

    excel_ids: set[str] = set(participants_df.index)

    # ── 2. Discover .dap files in flat directory ──────────────────────────
    dap_files = sorted(data_root.glob("*.dap"))
    print(f"\nFound {len(dap_files)} .dap files in {data_root}\n")

    # Build map: subject_id → dap_path (strip "_ec" suffix from stem)
    file_map: dict[str, Path] = {}
    for dap in dap_files:
        sub_id = dap.stem.replace("_ec", "")
        file_map[sub_id] = dap

    file_ids: set[str] = set(file_map)

    # Warn about mismatches
    for sub_id in sorted(excel_ids - file_ids):
        print(f"  [WARN] Excel subject {sub_id!r} has no .dat file — skipping")
    for sub_id in sorted(file_ids - excel_ids):
        print(f"  [WARN] File {sub_id!r} has no Excel entry — skipping (no label)")

    intersection = sorted(excel_ids & file_ids)
    print(f"\nProcessing {len(intersection)} subjects (Excel ∩ .dat files)\n")

    # ── 3. Build rows ─────────────────────────────────────────────────────
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    rows: list[dict] = []
    new_id  = 1
    skipped = 0

    for sub_id in intersection:
        dap_path = file_map[sub_id]
        dat_path = dap_path.with_suffix(".dat")
        rs3_path = dap_path.with_suffix(".rs3")

        if not dat_path.exists():
            print(f"  [WARN] .dat missing for {sub_id} — skipping")
            skipped += 1
            continue
        if not rs3_path.exists():
            print(f"  [WARN] .rs3 missing for {sub_id} — skipping")
            skipped += 1
            continue

        sub_row = participants_df.loc[sub_id]
        print(f"  [{new_id:04d}] {dat_path.name}")

        row = build_row(
            new_id=new_id,
            sub_id=sub_id,
            sub_row=sub_row,
            dat_path=dat_path,
            dap_path=dap_path,
            rs3_path=rs3_path,
            now_str=now_str,
        )
        rows.append(row)
        new_id += 1

    # ── 4. Write CSV ──────────────────────────────────────────────────────
    if not rows:
        print(
            "\n[ERROR] No rows generated.\n"
            "Check --data_root and --excel_path arguments."
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

    n_eeg_vals = out_df["n_eeg_channels"].dropna().unique()
    if len(n_eeg_vals):
        print(f"   EEG channel counts : {sorted(n_eeg_vals)}")

    rec_durs = out_df["recording_duration_sec"].dropna()
    if not rec_durs.empty:
        print(f"   Duration (sec)     : min={rec_durs.min():.1f}  max={rec_durs.max():.1f}")

    print("=" * 60)
    print()
    print(
        "WARNING: The paper (Racz et al. 2025) analyzed 61 subjects but deposited\n"
        "all 77 raw .dat files including 16 quality-excluded subjects (8 SZ + 8 HC).\n"
        "Their IDs are NOT documented. The preprocessor uses EEGPrep RANSAC/ASR to\n"
        "gate data quality — poor-quality subjects yield fewer or zero clean windows.\n"
        "This metadata table covers all subjects with both an Excel row and a .dat file."
    )


if __name__ == "__main__":
    main()
