#!/bin/bash
# =============================================================================
# Classical baselines (LDA, SVM, XGBoost on handcrafted features; train_ml.py). CPU only.
#
#   bash scripts/slurm/submit.sh scripts/slurm/train_ml.sh configs/train_ml/baseline_lda_adftd.yaml
#   for c in configs/train_ml/*.yaml; do bash scripts/slurm/submit.sh scripts/slurm/train_ml.sh "$c"; done
#
# Feature extraction uses all allocated CPUs (ml_features.n_jobs).
# =============================================================================
#SBATCH --job-name=train_ml
#SBATCH --partition=high_prio
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source "${SSL_EEG_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/slurm/common.sh"

[ $# -ge 1 ] || { echo "Usage: $(basename "$0") <config.yaml> [KEY=VALUE ...]" >&2; exit 1; }
# n_jobs first, so a KEY=VALUE given on the command line still wins.
run_config_job --cpu train_ml.py "$1" "ml_features.n_jobs=${SLURM_CPUS_PER_TASK:-8}" "${@:2}"
