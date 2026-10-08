#!/bin/bash
# =============================================================================
# Cross-dataset evaluation FEP → SCZ (evaluate.py).
#
#   bash scripts/slurm/submit.sh scripts/slurm/evaluate.sh <eval config> [--rolling] [KEY=VALUE ...]
#
#   --rolling   evaluate the five five-fold-CV models (rolling_checkpoint_dir in
#               the config) and average them; without it, the model trained on
#               all FEP participants (checkpoint_path).
#
#   # every model, both settings:
#   for c in configs/eval/*.yaml; do
#       bash scripts/slurm/submit.sh scripts/slurm/evaluate.sh "$c" --rolling
#       bash scripts/slurm/submit.sh scripts/slurm/evaluate.sh "$c"
#   done
#
# Classical-ML configs (those with an ml_model key) run on the CPU.
# =============================================================================
#SBATCH --job-name=evaluate
#SBATCH --partition=gpu_v100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00

source "${SSL_EEG_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/slurm/common.sh"

if [ $# -lt 1 ]; then
    echo "Usage: $(basename "$0") <eval config> [--rolling] [KEY=VALUE ...]" >&2
    exit 1
fi
CONFIG="$(resolve_config "$1")"; shift
ROLLING=()
if [ "${1:-}" = "--rolling" ]; then ROLLING=(--rolling); shift; fi

DEVICE=()
OVERRIDES=()
if grep -qE "^ml_model:" "${CONFIG}"; then
    DEVICE=(--cpu)
    OVERRIDES=("ml_features.n_jobs=${SLURM_CPUS_PER_TASK:-8}")
fi
OVERRIDES+=("$@")
SET_ARGS=()
[ ${#OVERRIDES[@]} -gt 0 ] && SET_ARGS=(--set "${OVERRIDES[@]}")

job_header "evaluate.py --config ${CONFIG#"${CODE_DIR}"/} ${ROLLING[*]-}"
run_python ${DEVICE[@]+"${DEVICE[@]}"} evaluate.py --config "${CONFIG}" ${ROLLING[@]+"${ROLLING[@]}"} \
    ${SET_ARGS[@]+"${SET_ARGS[@]}"}
job_footer
