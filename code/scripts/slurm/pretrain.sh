#!/bin/bash
# =============================================================================
# Self-supervised pretraining of a BrainLM-EEG variant (pretrain.py).
#
#   bash scripts/slurm/submit.sh scripts/slurm/pretrain.sh configs/pretrain/brainlm_eeg.yaml
#   bash scripts/slurm/submit.sh scripts/slurm/pretrain.sh configs/pretrain/brainlm_eeg_rope.yaml
#   bash scripts/slurm/submit.sh scripts/slurm/pretrain.sh configs/pretrain/brainlm_eeg_microstate.yaml
#
# Trailing KEY=VALUE arguments are passed to --set. About 1.5 min/epoch on one
# H200 (500 epochs, early stopping).
# =============================================================================
#SBATCH --job-name=pretrain
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00

source "${SSL_EEG_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/slurm/common.sh"

run_config_job pretrain.py "$@"
