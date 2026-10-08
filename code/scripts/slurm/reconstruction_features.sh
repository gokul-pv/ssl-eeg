#!/bin/bash
# =============================================================================
# Classifying BrainLM-EEG reconstructions vs. the original input
# (experiments/reconstruction_features.py).
#
#   for c in configs/experiments/reconstruction_features_*.yaml; do
#       bash scripts/slurm/submit.sh scripts/slurm/reconstruction_features.sh "$c"; done
# =============================================================================
#SBATCH --job-name=reconstruction_features
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00

source "${SSL_EEG_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/slurm/common.sh"

run_config_job experiments/reconstruction_features.py "$@"
