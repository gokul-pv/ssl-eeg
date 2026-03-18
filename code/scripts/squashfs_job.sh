#!/bin/bash
#SBATCH --job-name=eeg_preprocess
#SBATCH --output=/work/gokul/ssl-eeg/code/outputs/logs/ds004504/preprocess/eeg_%A_%a.out
#SBATCH --error=/work/gokul/ssl-eeg/code/outputs/logs/ds004504/preprocess/eeg_%A_%a.err
#SBATCH --array=0,1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=all

# ======================
# MOUNT SQUASHFS ENV
# ======================
VENV_MOUNT=/mnt/scratch/gokul/ssl-eeg/.venv
SQUASHFS_IMG=/work/gokul/uv-env.squashfs

sudo mkdir -p "$VENV_MOUNT"

if ! mountpoint -q "$VENV_MOUNT"; then
    sudo mount -t squashfs -o loop "$SQUASHFS_IMG" "$VENV_MOUNT"
fi

# ======================
# TRAP — unmount on exit
# ======================
cleanup() {
    LOCKFILE="/tmp/squashfs_umount_${SLURM_JOB_ID}_$(hostname -s).lock"

    (
        flock -x 200

        if mountpoint -q "$VENV_MOUNT"; then
            sudo umount "$VENV_MOUNT" && echo "Unmounted." || echo "Umount failed."
        else
            echo "Already unmounted, skipping."
        fi

    ) 200>"$LOCKFILE"

    rm -f "$LOCKFILE"
}

trap cleanup EXIT   # fires on: normal exit, error, SIGTERM (job cancel)

source "$VENV_MOUNT/bin/activate"

# ======================
# SKIP IF ALREADY DONE
# ======================
MASTER_CSV="/work/gokul/ssl-eeg/code/metadata/ds004504/master_subject_table_ds004504.csv"

OUTPUT_FILE=$(python -c "
import os, pandas as pd
df = pd.read_csv('$MASTER_CSV')
f = df.iloc[$SLURM_ARRAY_TASK_ID]['file_path_raw']
inp = '/mnt/datasets_v7/gokul/EEGData/ds004504'
out = '/mnt/datasets_v7/gokul/EEGData/ds004504/cleaned'
rel = os.path.relpath(f, inp).replace('.set', '.fif')
print(os.path.join(out, rel))
")

if [ -f "$OUTPUT_FILE" ]; then
    echo "[Task $SLURM_ARRAY_TASK_ID] Already exists, skipping: $OUTPUT_FILE"
    exit 0
fi

# ======================
# RUN
# ======================
python /work/gokul/ssl-eeg/code/src/preprocess.py $SLURM_ARRAY_TASK_ID
