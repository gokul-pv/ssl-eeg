import pyrootutils

root = pyrootutils.setup_root(
        search_from=__file__,
        indicator="pyproject.toml",
        pythonpath=True,
        cwd=True,
    )

import os
import json
import warnings
from collections import defaultdict, Counter
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import resample


warnings.filterwarnings("ignore")
mne.set_log_level('WARNING')

# Multi-scale sampling rates (Hz)
# SAMPLE_RATE_LIST = [200, 100, 50]
SAMPLE_RATE_LIST = [200]
DATASET = "ds004504"
DATASET_PATH = f"/mnt/datasets_v7/gokul/EEGData/{DATASET}"

# Fixed segment length and overlap in samples
SEG_LEN = 400
OVERLAP = 200
assert 0 <= OVERLAP < SEG_LEN, "OVERLAP must be in [0, SEG_LEN)."


def build_subject_info(master_file):
    participants_df = pd.read_csv(master_file, sep=",")
    label_map = {"alzheimer": 1, "frontotemporaldementia": 2, "control": 0}

    # "sub-XXX" -> (label, pid)
    sub_info = {}
    for row in participants_df.itertuples(index=False):
        sub_name = row[3]               # e.g., sub-001
        diag_code = row[10]             # e.g., A/F/C
        pid = int(sub_name[-3:])        # 001 -> 1
        label = label_map[diag_code]
        sub_info[sub_name] = (label, pid)

    return sub_info


def scan_dataset_stats(derivatives_root):
    bad_channel_list = []
    sampling_freq_list = []
    data_shape_list = []

    for sub in sorted(os.listdir(derivatives_root)):
        if "sub-" not in sub:
            continue
        sub_path = derivatives_root / sub / "eeg"
        if not sub_path.exists():
            continue

        for fname in os.listdir(sub_path):
            if not fname.endswith(".set"):
                continue
            file_path = sub_path / fname
            raw = mne.io.read_raw_eeglab(file_path, preload=True)

            bad_channel_list.append(raw.info["bads"])
            sampling_freq_list.append(raw.info["sfreq"])
            data_shape_list.append(raw.get_data().shape)  # (C, T)

    print("Bad channels:", bad_channel_list[:5], "...")
    if data_shape_list:
        print("First raw data shape:", data_shape_list[0])
        print("Raw channel number counter:", Counter(i[0] for i in data_shape_list))
    print("Raw sampling rate counter:", Counter(sampling_freq_list))


def find_common_channels(derivatives_root):
    common_channels = None

    for sub in sorted(os.listdir(derivatives_root)):
        if "sub-" not in sub:
            continue
        sub_path = derivatives_root / sub / "eeg"
        if not sub_path.exists():
            continue

        for fname in os.listdir(sub_path):
            if not fname.endswith(".set"):
                continue
            file_path = sub_path / fname
            raw = mne.io.read_raw_eeglab(file_path, preload=False)
            ch_set = set(raw.info["ch_names"])

            if common_channels is None:
                common_channels = ch_set
            else:
                common_channels &= ch_set

    if common_channels is None:
        raise RuntimeError("No EEG .set files found.")

    common_channels = sorted(list(common_channels))
    print("Common channels:", common_channels)
    print("Common channel count:", len(common_channels))
    print("-" * 40)

    return common_channels


def compute_step(seg_len, overlap):
    step = seg_len - overlap
    if step <= 0:
        raise ValueError(f"Invalid overlap={overlap}: step <= 0.")
    return step


def compute_num_segments(num_samples, seg_len, step):
    if num_samples < seg_len:
        return 0
    return 1 + (num_samples - seg_len) // step


def resample_time_series(data, original_fs, target_fs):
    """
    Resample each channel independently.
    data shape: (T_raw, C)
    return shape: (T_new, C)
    """
    t_raw, n_ch = data.shape
    new_length = int(t_raw * target_fs / original_fs)
    out = np.zeros((new_length, n_ch), dtype=np.float32)
    for i in range(n_ch):
        out[:, i] = resample(data[:, i], new_length)
    return out


def count_segments(derivatives_root, sub_info, sample_rates, seg_len, overlap):
    subject_segment_counts = defaultdict(lambda: defaultdict(int))
    n_total = 0
    step = compute_step(seg_len, overlap)
    print("SEG_LEN =", seg_len, "OVERLAP =", overlap, "STEP =", step)

    for sub in sorted(os.listdir(derivatives_root)):
        if "sub-" not in sub:
            continue
        if sub not in sub_info:
            continue

        sub_path = derivatives_root / sub / "eeg"
        if not sub_path.exists():
            continue

        print(f"[PASS 1] Subject: {sub}")
        for fname in os.listdir(sub_path):
            if not fname.endswith(".set"):
                continue

            file_path = sub_path / fname
            print("  reading header:", file_path)
            raw = mne.io.read_raw_eeglab(file_path, preload=False)
            original_fs = raw.info["sfreq"]
            t_raw = raw.n_times

            for fs in sample_rates:
                t_res = int(t_raw * fs / original_fs)
                n_seg = compute_num_segments(t_res, seg_len, step)
                subject_segment_counts[sub][fs] += n_seg
                n_total += n_seg
                print(f"    fs={fs}Hz: T_res={t_res}, STEP={step}, n_seg={n_seg}")

        print("-" * 40)

    print("Total segments N_total =", n_total)
    if n_total == 0:
        raise RuntimeError("No segments found. Check SEG_LEN/OVERLAP.")

    return subject_segment_counts, n_total, step


def write_memmaps(
    derivatives_root,
    sub_info,
    common_channels,
    subject_segment_counts,
    n_total,
    sample_rates,
    seg_len,
    step,
    x_path,
    y_path,
):
    n_channels = len(common_channels)

    x_mm = np.memmap(x_path, dtype="float32", mode="w+", shape=(n_total, seg_len, n_channels))
    y_mm = np.memmap(y_path, dtype="float32", mode="w+", shape=(n_total, 2))

    cur = 0
    total_seconds_all = 0.0

    for sub in sorted(os.listdir(derivatives_root)):
        if "sub-" not in sub:
            continue
        if sub not in sub_info:
            continue

        label, pid = sub_info[sub]
        sub_path = derivatives_root / sub / "eeg"
        if not sub_path.exists():
            continue

        total_seg_sub = sum(subject_segment_counts[sub][fs] for fs in sample_rates)
        if total_seg_sub == 0:
            continue

        print(f"[PASS 2] Subject: {sub}, expected total segments={total_seg_sub}")

        for fname in os.listdir(sub_path):
            if not fname.endswith(".set"):
                continue

            file_path = sub_path / fname
            print("  load:", file_path)
            raw = mne.io.read_raw_eeglab(file_path, preload=True)

            # Keep only common channels and fixed channel order
            raw.pick(common_channels)

            original_fs = raw.info["sfreq"]
            data_raw = raw.get_data().T.astype("float32")  # (C, T) -> (T, C)
            total_seconds_all += data_raw.shape[0] / original_fs
            print("raw shape:", data_raw.shape)

            for fs in sample_rates:
                data = resample_time_series(data_raw, original_fs, fs)
                t_res, _ = data.shape

                starts = np.arange(0, t_res - seg_len + 1, step, dtype=int)
                print(f"    fs={fs}Hz: resampled shape={data.shape}, segments={len(starts)}")

                for s in starts:
                    if cur >= n_total:
                        raise RuntimeError("Exceeded predicted N_total.")

                    x_mm[cur] = data[s : s + seg_len]
                    y_mm[cur, 0] = float(label)  # label
                    y_mm[cur, 1] = float(pid)    # subject id
                    # y_mm[cur, 2] = float(fs)     # sampling rate scale
                    cur += 1

        print("-" * 40)

    total_hours_all = total_seconds_all / 3600.0
    print("DONE: cur =", cur, "expected N_total =", n_total)
    print(f"Total hours across all subjects: {total_hours_all:.2f} hours")

    del x_mm
    del y_mm

    return cur, total_hours_all


def save_meta(meta_path, n_total, seg_len, n_channels, sample_rates, overlap, step, x_path, y_path):
    meta = {
        "N": int(n_total),
        "T": int(seg_len),
        "C": int(n_channels),
        "SAMPLE_RATE_LIST": list(sample_rates),
        "OVERLAP": int(overlap),
        "STEP": int(step),
        "X_path": str(x_path),
        "y_path": str(y_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved meta:", meta_path)


def verify_output(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)

    n = meta["N"]
    t = meta["T"]
    c = meta["C"]
    x_path = meta["X_path"]
    y_path = meta["y_path"]

    print("Meta:")
    print("  N =", n)
    print("  T =", t)
    print("  C =", c)
    print("  X_path =", x_path)
    print("  y_path =", y_path)

    x_mm = np.memmap(x_path, dtype="float32", mode="r", shape=(n, t, c))
    y_mm = np.memmap(y_path, dtype="float32", mode="r", shape=(n, 2))

    subject_ids = np.unique(y_mm[:, 1]).astype(int)
    print(f"{subject_ids=}")
    for sid in sorted(subject_ids):
        idx = np.where(y_mm[:, 1] == sid)[0]
        n_seg = len(idx)
        x_shape = (n_seg, t, c)
        y_shape = (n_seg, 2)
        print(f"Subject ID {sid:03d}: X shape={x_shape}, y shape={y_shape}")

    del x_mm
    del y_mm


def main():
    master_file = root / "metadata" / "ds004504" / "master_subject_table_ds004504.csv"
    derivatives_root = Path(DATASET_PATH) / "derivatives"
    print(f"{derivatives_root=}")

    sub_folder_path = f"L{SEG_LEN}"
    output_root = Path("data") / DATASET / sub_folder_path
    output_root.mkdir(parents=True, exist_ok=True)

    x_path = output_root / "X.dat"
    y_path = output_root / "y.dat"
    meta_path = output_root / "meta.json"

    print("X path:", x_path)
    print("y path:", y_path)

    sub_info = build_subject_info(master_file)
    print("Subjects loaded:", len(sub_info))

    # scan_dataset_stats(derivatives_root)
    common_channels = find_common_channels(derivatives_root)

    subject_segment_counts, n_total, step = count_segments(
        derivatives_root=derivatives_root,
        sub_info=sub_info,
        sample_rates=SAMPLE_RATE_LIST,
        seg_len=SEG_LEN,
        overlap=OVERLAP,
    )

    cur, total_hours_all = write_memmaps(
        derivatives_root=derivatives_root,
        sub_info=sub_info,
        common_channels=common_channels,
        subject_segment_counts=subject_segment_counts,
        n_total=n_total,
        sample_rates=SAMPLE_RATE_LIST,
        seg_len=SEG_LEN,
        step=step,
        x_path=x_path,
        y_path=y_path,
    )

    if cur != n_total:
        print(f"Warning: cur ({cur}) != N_total ({n_total})")

    save_meta(
        meta_path=meta_path,
        n_total=n_total,
        seg_len=SEG_LEN,
        n_channels=len(common_channels),
        sample_rates=SAMPLE_RATE_LIST,
        overlap=OVERLAP,
        step=step,
        x_path=x_path,
        y_path=y_path,
    )

    verify_output(meta_path)


if __name__ == "__main__":
    main()
