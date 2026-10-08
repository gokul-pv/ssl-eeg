#!/bin/bash
# =============================================================================
# Layer-wise probing of a frozen BrainLM-EEG encoder (experiments/layer_analysis.py).
#
#   for c in configs/experiments/layer_analysis_*.yaml; do
#       bash scripts/slurm/submit.sh scripts/slurm/layer_analysis.sh "$c"; done
# =============================================================================
#SBATCH --job-name=layer_analysis
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00

source "${SSL_EEG_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/slurm/common.sh"

run_config_job experiments/layer_analysis.py "$@"
