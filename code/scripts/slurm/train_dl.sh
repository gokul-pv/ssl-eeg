#!/bin/bash
# =============================================================================
# Supervised deep-learning baselines (EEGNeX, EEGConformer, ATCNet; train_dl.py).
#
#   bash scripts/slurm/submit.sh scripts/slurm/train_dl.sh configs/train_dl/train_atcnet_adftd.yaml
#   for c in configs/train_dl/*.yaml; do bash scripts/slurm/submit.sh scripts/slurm/train_dl.sh "$c"; done
# =============================================================================
#SBATCH --job-name=train_dl
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

source "${SSL_EEG_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/slurm/common.sh"

run_config_job train_dl.py "$@"
