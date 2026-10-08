#!/bin/bash
# =============================================================================
# CLS-token perturbation test (experiments/cls_token_ablation.py).
#
#   bash scripts/slurm/submit.sh scripts/slurm/cls_token_ablation.sh configs/experiments/cls_token_ablation.yaml
# =============================================================================
#SBATCH --job-name=cls_token_ablation
#SBATCH --partition=gpu_v100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00

source "${SSL_EEG_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/slurm/common.sh"

run_config_job experiments/cls_token_ablation.py "$@"
