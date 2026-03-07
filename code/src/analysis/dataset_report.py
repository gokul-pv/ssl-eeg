import os
import json
import numpy as np
import pandas as pd
import mne
import wandb
from tqdm import tqdm
from scipy.stats import skew, kurtosis

DATASET = "ds004504"
MASTER_TABLE_PATH = f"metadata/master_subject_table_{DATASET}.csv"
OUTPUT_DIR = "outputs/reports/data_exploration"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_signal_statistics(raw):
    data = raw.get_data()

    stats = {
        "global_mean": float(np.mean(data)),
        "global_std": float(np.std(data)),
        "global_min": float(np.min(data)),
        "global_max": float(np.max(data)),
        "global_energy": float(np.sum(data ** 2))
    }

    channel_means = np.mean(data, axis=1)
    channel_stds = np.std(data, axis=1)

    stats["mean_channel_std"] = float(np.mean(channel_stds))
    stats["max_channel_std"] = float(np.max(channel_stds))
    stats["min_channel_std"] = float(np.min(channel_stds))

    stats["mean_skewness"] = float(np.mean(skew(data, axis=1)))
    stats["mean_kurtosis"] = float(np.mean(kurtosis(data, axis=1)))

    # Flat channel detection
    stats["flat_channels"] = int(np.sum(channel_stds < 1e-6))

    return stats


def compute_psd_features(raw):
    psd, freqs = mne.time_frequency.psd_array_welch(
        raw.get_data(),
        sfreq=raw.info["sfreq"],
        fmin=0.5,
        fmax=45,
        verbose=False
    )

    total_power = np.sum(psd)
    peak_freq = freqs[np.argmax(np.mean(psd, axis=0))]

    return float(total_power), float(peak_freq)


def main():

    wandb.init(
        entity="team-ssl-eeg",
        project="data_exploration",
        name=f"{DATASET}_v1",
        config={"stage": "data_exploration"}
    )

    df = pd.read_csv(MASTER_TABLE_PATH)

    exploration_records = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        raw = mne.io.read_raw_eeglab(row["file_path_raw"], preload=True, verbose=False)

        signal_stats = compute_signal_statistics(raw)
        total_power, peak_freq = compute_psd_features(raw)

        record = {
            "subject_id": row["subject_id"],
            "class_label": row["diagnosis"],
            "sampling_rate": row["sampling_rate_hz"],
            "n_channels": row["n_total_channels"],
            "n_eeg_channels": row["n_eeg_channels"],
            "duration_sec": row["recording_duration_sec"],
            "age": row["age"],
            "gender": row["sex"],
            "powerline_freq": row["powerline_freq_hz"],
            "unit": row["channel_unit"],
            "reference": row["eeg_reference"],
            "eeg_placement_scheme": row["eeg_placement_scheme"],
            "total_power": total_power,
            "peak_frequency": peak_freq
        }

        record.update(signal_stats)

        exploration_records.append(record)

    exploration_df = pd.DataFrame(exploration_records)

    # Save detailed subject-level stats
    exploration_df.to_csv(
        os.path.join(OUTPUT_DIR, f"subject_level_statistics_{DATASET}.csv"),
        index=False
    )

    # Create global summary
    summary = {
        "total_subjects": len(exploration_df),
        "class_distribution": exploration_df["class_label"].value_counts().to_dict(),
        "sampling_rates_unique": exploration_df["sampling_rate"].unique().tolist(),
        "channel_counts_unique": exploration_df["n_eeg_channels"].unique().tolist(),
        "mean_age": float(exploration_df["age"].mean()),
        "gender_distribution": exploration_df["gender"].value_counts().to_dict(),
        "mean_duration_sec": float(exploration_df["duration_sec"].mean()),
        "mean_global_std": float(exploration_df["global_std"].mean()),
        "mean_total_power": float(exploration_df["total_power"].mean()),
        "mean_peak_frequency": float(exploration_df["peak_frequency"].mean()),
        "subjects_with_flat_channels": int(np.sum(exploration_df["flat_channels"] > 0))
    }

    with open(os.path.join(OUTPUT_DIR, f"exploration_summary_{DATASET}.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Save numerical description
    exploration_df.describe().to_csv(
        os.path.join(OUTPUT_DIR, f"exploration_numerical_summary{DATASET}.csv")
    )

    wandb.log(summary)

    print(json.dumps(summary, indent=4))

    wandb.finish()


if __name__ == "__main__":
    main()