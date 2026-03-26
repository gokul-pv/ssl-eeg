#!/bin/bash
#SBATCH --job-name=eeg_preprocess
#SBATCH --output=/work/gokul/ssl-eeg/code/outputs/logs/ds004504/preprocess/eeg_%A_%a.out
#SBATCH --error=/work/gokul/ssl-eeg/code/outputs/logs/ds004504/preprocess/eeg_%A_%a.err
#SBATCH --array=0-87%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=all

module load singularity/4.1.2

srun singularity exec --userns \
    --bind /mnt/datasets_v7:/mnt/datasets_v7 \
    /work/gokul/env_cpu.sif \
    python /work/gokul/ssl-eeg/code/src/preprocess.py $SLURM_ARRAY_TASK_ID
