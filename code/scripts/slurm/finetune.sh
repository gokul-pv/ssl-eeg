#!/bin/bash
# =============================================================================
# BrainLM-EEG linear probe or finetuning (finetune.py). The protocol (five-fold
# rolling CV, LOSO, or all FEP participants) is set by the config.
#
#   bash scripts/slurm/submit.sh scripts/slurm/finetune.sh configs/linear_probe/brainlm_eeg_lp_adftd.yaml
#   bash scripts/slurm/submit.sh scripts/slurm/finetune.sh configs/finetune/brainlm_eeg_ft_mdd_loso.yaml
#
#   # all linear-probe and finetuning configs:
#   for c in configs/linear_probe/*.yaml configs/finetune/*.yaml; do
#       bash scripts/slurm/submit.sh scripts/slurm/finetune.sh "$c"; done
# =============================================================================
#SBATCH --job-name=finetune
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00

source "${SSL_EEG_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/slurm/common.sh"

run_config_job finetune.py "$@"
