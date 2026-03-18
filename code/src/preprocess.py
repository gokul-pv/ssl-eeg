import os, sys
import mne
import pandas as pd
from mne_icalabel import label_components
from asrpy import ASR

# ======================
# CONFIG
# ======================
DATASET = "ds004504"
MASTER_TABLE_PATH = f"/work/gokul/ssl-eeg/code/metadata/{DATASET}/master_subject_table_{DATASET}.csv"
INPUT_ROOT = "/mnt/datasets_v7/gokul/EEGData/ds004504"
OUTPUT_ROOT = "/mnt/datasets_v7/gokul/EEGData/ds004504/cleaned"

# ======================
# PATH UTILS
# ======================
def get_output_path(input_path):
    rel = os.path.relpath(input_path, INPUT_ROOT)
    return os.path.join(OUTPUT_ROOT, rel.replace(".set", ".fif"))

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ======================
# PREPROCESS
# ======================
def preprocess_mne_raw(file_path):
    raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose="ERROR")
    raw.pick(["eeg"])
    raw.notch_filter(freqs=[50, 60], verbose=False)
    raw.filter(l_freq=1.0, h_freq=100, verbose=False)
    raw.set_eeg_reference(ref_channels="average", verbose=False)

    asr = ASR(sfreq=raw.info["sfreq"], cutoff=20)
    asr.fit(raw)
    raw = asr.transform(raw)

    ica = mne.preprocessing.ICA(
        n_components=None, random_state=42,
        method="infomax", fit_params=dict(extended=True),
    )
    ica.fit(raw, verbose=False)
    ic_labels = label_components(raw, ica, method="iclabel")
    exclude_idx = [
        i for i, label in enumerate(ic_labels["labels"])
        if label not in ["brain"]
    ]
    ica.apply(raw, exclude=exclude_idx, verbose=False)
    return raw

# ======================
# MAIN
# ======================
def main():
    recording_idx = int(sys.argv[1])

    df = pd.read_csv(MASTER_TABLE_PATH)
    file_path = df.iloc[recording_idx]["file_path_raw"]
    out_path = get_output_path(file_path)

    if os.path.exists(out_path):
        print(f"[Task {recording_idx}] Already exists, skipping: {out_path}")
        return

    print(f"[Task {recording_idx}] Processing: {file_path}")
    raw_clean = preprocess_mne_raw(file_path)

    ensure_dir(out_path)
    print(f"[Task {recording_idx}] Saving -> {out_path}")
    raw_clean.save(out_path, overwrite=True, verbose=False)
    print(f"[Task {recording_idx}] Done.")

if __name__ == "__main__":
    main()