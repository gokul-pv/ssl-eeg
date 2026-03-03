import os
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import mne
from bids import BIDSLayout
from tqdm import tqdm


def _safe_get(d: dict, *keys, default=None):
    """Nested dict lookup that never raises."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def load_json(path) -> dict:
    """Load a JSON sidecar; return empty dict on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_tsv(path) -> pd.DataFrame:
    """Load a TSV file; return empty DataFrame on any error."""
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()


def extract_raw_info(eeg_path: str) -> dict:
    """
    Open the raw EEG file with MNE (no preload) and pull low-level metadata.
    Works for .set (EEGLAB), .edf, .bdf, .fif, .vhdr (BrainVision).
    """
    info = {
        "sampling_rate_hz":         None,
        "n_total_channels":         None,
        "n_eeg_channels":           None,
        "n_other_channels":         None,
        "recording_duration_sec":   None,
        "n_timepoints":             None,
        "mne_lowpass_hz":           None,
        "mne_highpass_hz":          None,
        "channel_names":            None,
    }

    ext = Path(eeg_path).suffix.lower()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if ext == ".set":
                raw = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
            elif ext == ".edf":
                raw = mne.io.read_raw_edf(eeg_path, preload=False, verbose=False)
            elif ext == ".bdf":
                raw = mne.io.read_raw_bdf(eeg_path, preload=False, verbose=False)
            elif ext in (".fif", ".fif.gz"):
                raw = mne.io.read_raw_fif(eeg_path, preload=False, verbose=False)
            elif ext == ".vhdr":
                raw = mne.io.read_raw_brainvision(eeg_path, preload=False, verbose=False)
            else:
                return info  # unsupported format — leave all None

        sr = raw.info["sfreq"]
        ch_types = raw.get_channel_types()
        n_eeg = sum(1 for t in ch_types if t == "eeg")
        n_other = len(ch_types) - n_eeg

        info.update({
            "sampling_rate_hz":         sr,
            "n_total_channels":         len(raw.ch_names),
            "n_eeg_channels":           n_eeg,
            "n_other_channels":         n_other,
            "recording_duration_sec":   round(raw.n_times / sr, 3),
            "n_timepoints":             raw.n_times,
            "mne_lowpass_hz":           raw.info.get("lowpass"),
            "mne_highpass_hz":          raw.info.get("highpass"),
            "channel_names":            ";".join(raw.ch_names),
        })
    except Exception as e:
        print(f"  [WARN] MNE could not read {eeg_path}: {e}")

    return info


def extract_eeg_json(json_path: str) -> dict:
    """
    Parse the BIDS *_eeg.json sidecar.
    Spec: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html
    """
    d = load_json(json_path)
    return {
        "json_task_name":               d.get("TaskName"),
        "json_task_description":        d.get("TaskDescription"),
        "json_eeg_channel_count":       d.get("EEGChannelCount"),
        "json_eog_channel_count":       d.get("EOGChannelCount"),
        "json_ecg_channel_count":       d.get("ECGChannelCount"),
        "json_emg_channel_count":       d.get("EMGChannelCount"),
        # "json_instructions":            d.get("Instructions"),
        "json_sampling_frequency":      d.get("SamplingFrequency"),
        "json_powerline_frequency":     d.get("PowerLineFrequency"),
        "json_software_filters":        str(d.get("SoftwareFilters", "")),
        "json_hardware_filters":        str(d.get("HardwareFilters", "")),
        "json_eeg_reference":           d.get("EEGReference"),
        "json_eeg_ground":              d.get("EEGGround"),
        "json_eeg_placement_scheme":    d.get("EEGPlacementScheme"),
        "json_cap_manufacturer":        d.get("CapManufacturer"),
        "json_cap_model":               d.get("CapManufacturersModelName"),
        # "json_amplifier_manufacturer":  d.get("Manufacturer"),
        # "json_amplifier_model":         d.get("ManufacturersModelName"),
        # "json_amplifier_serial":        d.get("DeviceSerialNumber"),
        # "json_software":                d.get("SoftwareVersions"),
        "json_institution":             d.get("InstitutionName"),
        "json_recording_type":          d.get("RecordingType"),        # continuous / epoched
        # "json_epoch_length":            d.get("EpochLength"),
        # "json_cog_atlas_id":            d.get("CogAtlasID"),
        "json_recording_duration":      d.get("RecordingDuration"),    # sidecar value (sec)
    }


def extract_channels_tsv(tsv_path: str) -> dict:
    """
    Summarise the *_channels.tsv sidecar.
    Returns counts per channel type and the reference column if present.
    """
    out = {
        "ch_tsv_n_rows":        None,
        "ch_tsv_names":         None,   # e.g. "Fp1,Fp2,F3"
        "ch_tsv_units":         None,   # e.g. "microV:66"
        "ch_tsv_types":         None,   # e.g. "EEG:64;EOG:2;ECG:1"
    }

    df = load_tsv(tsv_path)
    if df.empty:
        return out

    out["ch_tsv_n_rows"] = len(df)

    if "type" in df.columns:
        counts = df["type"].value_counts().to_dict()
        out["ch_tsv_types"] = ";".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    
    if "units" in df.columns:
        unit_counts = df["units"].value_counts().to_dict()
        out["ch_tsv_units"] = ";".join(f"{k}:{v}" for k, v in sorted(unit_counts.items()))
    
    if "name" in df.columns:
        out["ch_tsv_names"] = ",".join(df["name"].astype(str).tolist())

    return out


def extract_dataset_description(dataset_path: str) -> dict:
    """Parse the top-level dataset_description.json."""
    desc_path = Path(dataset_path) / "dataset_description.json"
    d = load_json(desc_path)
    return {
        "bids_dataset_name":    d.get("Name"),
        "bids_version":         d.get("BIDSVersion"),
        "bids_license":         d.get("License"),
        "bids_authors":         ";".join(d.get("Authors", [])),
        "bids_funding":         ";".join(d.get("Funding", [])) if isinstance(d.get("Funding"), list) else d.get("Funding"),
        "bids_dataset_doi":     _safe_get(d, "DatasetDOI"),
    }



def build_master_table(dataset_path: str, dataset_id: str, output_path: str) -> pd.DataFrame:
    """
    Walk every subject x session x task x run in the BIDS layout and produce
    one row per EEG recording file.
    """
    dataset_path = str(Path(dataset_path).resolve())
    output_path  = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Dataset : {dataset_id}")
    print(f"  Path    : {dataset_path}")
    print(f"  Output  : {output_path}")
    print(f"{'='*60}\n")

    # ── BIDS layout ───────────────────────────────────────────────────────────
    layout = BIDSLayout(dataset_path, validate=False)

    # ── Dataset-level metadata ────────────────────────────────────────────────
    dataset_meta = extract_dataset_description(dataset_path)

    # ── Participants table ────────────────────────────────────────────────────
    participants_path = Path(dataset_path) / "participants.tsv"
    participants_df   = load_tsv(participants_path)

    # Normalise participant_id column (keep original for merge)
    if not participants_df.empty and "participant_id" in participants_df.columns:
        participants_df["_subj_key"] = (
            participants_df["participant_id"]
            .str.replace("sub-", "", regex=False)
            .str.strip()
        )

    # ── EEG file extensions to look for (in preference order) ─────────────────
    EEG_EXTENSIONS = [".set", ".edf", ".bdf", ".fif", ".vhdr"]

    subjects = layout.get_subjects()
    print(f"Found {len(subjects)} subjects.\n")

    records = []
    row_idx = 1  # global NEW_ID counter (1-based, zero-padded later)

    for subj in tqdm(subjects, desc="Subjects"):
        # Collect all EEG files for this subject (across sessions / runs / tasks)
        eeg_files = []
        for ext in EEG_EXTENSIONS:
            eeg_files.extend(layout.get(subject=subj, suffix="eeg", extension=ext))

        if not eeg_files:
            tqdm.write(f"  [SKIP] sub-{subj}: no EEG files found")
            continue

        # ── Participant-level info ────────────────────────────────────────────
        if not participants_df.empty:
            p_row = participants_df[participants_df["_subj_key"] == subj]
            p_info = p_row.iloc[0].to_dict() if not p_row.empty else {}
        else:
            p_info = {}

        p_info = {k.lower(): v for k, v in p_info.items()}

        age       = p_info.get("age")
        sex       = p_info.get("sex") or p_info.get("gender")
        group     = p_info.get("group") or p_info.get("diagnosis") or p_info.get("pathology")
        mmse      = p_info.get("mmse")
        old_id    = p_info.get("old_id") or p_info.get("participant_id", f"sub-{subj}")

        for eeg_bids_file in eeg_files:
            path_obj = Path(eeg_bids_file.path)
            entities = eeg_bids_file.get_entities()

            task    = entities.get("task")
            session = entities.get("session")
            run     = entities.get("run")
            # acq     = entities.get("acquisition")
            ext     = path_obj.suffix.lower()

            # ── EEG sidecar JSON ──────────────────────────────────────────────
            json_kwargs = dict(subject=subj, suffix="eeg", extension=".json")
            if task:    json_kwargs["task"]        = task
            if session: json_kwargs["session"]     = session
            if run:     json_kwargs["run"]         = run
            # if acq:     json_kwargs["acquisition"] = acq

            json_files = layout.get(**json_kwargs)
            eeg_json_meta = extract_eeg_json(json_files[0].path) if json_files else extract_eeg_json("")

            # ── Channels TSV ──────────────────────────────────────────────────
            ch_kwargs = {**json_kwargs, "suffix": "channels", "extension": ".tsv"}
            ch_files  = layout.get(**ch_kwargs)
            ch_meta   = extract_channels_tsv(ch_files[0].path) if ch_files else extract_channels_tsv("")

            # ── Raw signal info (MNE) ─────────────────────────────────────────
            raw_meta = extract_raw_info(str(path_obj))

            # ── Infer eyes condition from task name ───────────────────────────
            task_lower = (task or "").lower()
            if "eyesclosed" in task_lower or "eyes_closed" in task_lower or "ec" == task_lower:
                eyes = "closed"
            elif "eyesopen" in task_lower or "eyes_open" in task_lower or "eo" == task_lower:
                eyes = "open"
            else:
                eyes = None

            # ── Assemble row ──────────────────────────────────────────────────
            record = {
                # ── Identifiers ──────────────────────────────────────────────
                "new_id":               str(row_idx).zfill(4),
                "old_id":               old_id,
                "dataset_id":           dataset_id,
                "subject_id":           f"sub-{subj}",
                "session_id":           f"ses-{session}" if session else None,
                "session_num":          session,
                "run":                  run,
                # "acquisition":          acq,

                # ── Demographics ─────────────────────────────────────────────
                "age":                  age,
                "sex":                  sex,
                "mmse":                 mmse,
                "diagnosis":            group,

                # ── Task / paradigm ───────────────────────────────────────────
                "task_label":           task,
                "task_name":            eeg_json_meta.get("json_task_name"),
                "task_description":     eeg_json_meta.get("json_task_description"),
                # "task_instructions":    eeg_json_meta.get("json_instructions"),
                "eyes_condition":       eyes,
                # "cog_atlas_id":         eeg_json_meta.get("json_cog_atlas_id"),

                # ── Recording parameters ──────────────────────────────────────
                "sampling_rate_hz":     raw_meta["sampling_rate_hz"] or eeg_json_meta.get("json_sampling_frequency"),
                "powerline_freq_hz":    eeg_json_meta.get("json_powerline_frequency"),
                "recording_duration_sec": raw_meta["recording_duration_sec"] or eeg_json_meta.get("json_recording_duration"),
                "n_timepoints":         raw_meta["n_timepoints"],
                "recording_type":       eeg_json_meta.get("json_recording_type"),
                # "epoch_length_sec":     eeg_json_meta.get("json_epoch_length"),

                # ── Channels ──────────────────────────────────────────────────
                "n_total_channels":     raw_meta["n_total_channels"] or ch_meta["ch_tsv_n_rows"],
                "n_eeg_channels":       raw_meta["n_eeg_channels"] or eeg_json_meta["json_eeg_channel_count"],
                "n_other_channels":     raw_meta["n_other_channels"],
                "channel_types_tsv":    ch_meta["ch_tsv_types"],
                "channel_unit":         ch_meta["ch_tsv_units"],
                # "n_bad_channels":       ch_meta["ch_tsv_status_bad"],
                "channel_names":        raw_meta["channel_names"] or ch_meta["ch_tsv_names"],

                # ── Filters / reference ───────────────────────────────────────
                "eeg_reference":        eeg_json_meta.get("json_eeg_reference"),
                "eeg_ground":           eeg_json_meta.get("json_eeg_ground"),
                # "reference_channels_tsv": ch_meta["ch_tsv_reference_col"],
                "eeg_placement_scheme": eeg_json_meta.get("json_eeg_placement_scheme"),
                "software_filters":     eeg_json_meta.get("json_software_filters"),
                "hardware_filters":     eeg_json_meta.get("json_hardware_filters"),
                "mne_lowpass_hz":       raw_meta["mne_lowpass_hz"],
                "mne_highpass_hz":      raw_meta["mne_highpass_hz"],

                # ── Hardware / acquisition ────────────────────────────────────
                # "amplifier_manufacturer": eeg_json_meta.get("json_amplifier_manufacturer"),
                # "amplifier_model":      eeg_json_meta.get("json_amplifier_model"),
                # "amplifier_serial":     eeg_json_meta.get("json_amplifier_serial"),
                "cap_manufacturer":     eeg_json_meta.get("json_cap_manufacturer"),
                "cap_model":            eeg_json_meta.get("json_cap_model"),
                # "acquisition_software": eeg_json_meta.get("json_software"),
                "institution":          eeg_json_meta.get("json_institution"),
                # "acquisition_date":     acq_date,

                # ── File paths ────────────────────────────────────────────────
                "file_name":            path_obj.name,
                "file_extension":       ext,
                "file_path_raw":        str(path_obj),
                "subject_dir":          str(path_obj.parent.parent),
                "eeg_json_path":        json_files[0].path if json_files else None,
                "channels_tsv_path":    ch_files[0].path  if ch_files  else None,

                # ── Dataset-level ─────────────────────────────────────────────
                "bids_dataset_name":    dataset_meta["bids_dataset_name"],
                "bids_version":         dataset_meta["bids_version"],
                "bids_license":         dataset_meta["bids_license"],
                "bids_dataset_doi":     dataset_meta["bids_dataset_doi"],

                # # ── QC / provenance placeholders (fill in manually later) ─────
                # "downloaded":           True,
                # "preprocessed":         False,
                # "reason_missing_preproc": None,
                # "preprocessed_file_dir": None,
                # "notes":                None,

                # ── Script provenance ─────────────────────────────────────────
                "table_generated_at":   datetime.now().isoformat(timespec="seconds"),
            }

            records.append(record)
            row_idx += 1

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Master table saved → {output_path}  ({len(df)} rows × {len(df.columns)} columns)")
    return df


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a comprehensive BIDS EEG master subject/recording table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset_path", required=True,
                   help="Path to the root of the BIDS dataset.")
    p.add_argument("--dataset_id",   required=True,
                   help="Short identifier, e.g. ds004504.")
    p.add_argument("--output_path",  default="metadata/master_subject_table.csv",
                   help="Where to write the output CSV.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_master_table(
        dataset_path=args.dataset_path,
        dataset_id=args.dataset_id,
        output_path=args.output_path,
    )