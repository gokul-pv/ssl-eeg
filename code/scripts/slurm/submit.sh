#!/bin/bash
# =============================================================================
# Submit a job script from scripts/slurm/ with sbatch.
#
#   bash scripts/slurm/submit.sh [sbatch options] <job>.sh [job arguments]
#
# Sets the .out/.err location (SLURM_LOG_DIR in configs/cluster/cluster.env)
# and exports SSL_EEG_CODE_DIR so the job finds the repository (sbatch runs a
# copy of the script from its spool directory). sbatch options given here
# override the #SBATCH defaults of the job script, e.g. --partition, --time.
#
# Examples (from code/):
#   bash scripts/slurm/submit.sh scripts/slurm/finetune.sh configs/finetune/brainlm_eeg_ft_adftd.yaml
#   bash scripts/slurm/submit.sh --array=0-87%16 scripts/slurm/preprocess.sh adftd
#   bash scripts/slurm/submit.sh --partition=gpu_v100 scripts/slurm/probe.sh configs/probe/labram_lp_fep_cross.yaml
#
# Any job script can also be run directly, without SLURM, for a smoke test:
#   bash scripts/slurm/finetune.sh configs/finetune/brainlm_eeg_ft_adftd.yaml train_epochs=1
# =============================================================================

set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SLURM_DIR}/../.." && pwd)"
source "${SSL_EEG_CLUSTER_ENV:-${CODE_DIR}/configs/cluster/cluster.env}"

SBATCH_OPTS=()
while [ $# -gt 0 ] && [[ "$1" != *.sh ]]; do
    SBATCH_OPTS+=("$1"); shift
done
if [ $# -lt 1 ]; then
    echo "Usage: bash $0 [sbatch options] <job>.sh [job arguments]" >&2
    exit 1
fi
JOB="$1"; shift
[ -f "${JOB}" ] || JOB="${SLURM_DIR}/$(basename "${JOB}")"
[ -f "${JOB}" ] || { echo "ERROR: job script not found: $1" >&2; exit 1; }
JOB="$(cd "$(dirname "${JOB}")" && pwd)/$(basename "${JOB}")"

LOG_DIR="${SLURM_LOG_DIR:-outputs/logs/slurm}"
[[ "${LOG_DIR}" = /* ]] || LOG_DIR="${CODE_DIR}/${LOG_DIR}"
mkdir -p "${LOG_DIR}"

if [[ " ${SBATCH_OPTS[*]-} " == *" --array"* ]]; then PATTERN="%x_%A_%a"; else PATTERN="%x_%j"; fi

export SSL_EEG_CODE_DIR="${CODE_DIR}"
sbatch --chdir="${CODE_DIR}" \
       --output="${LOG_DIR}/${PATTERN}.out" --error="${LOG_DIR}/${PATTERN}.err" \
       ${SBATCH_OPTS[@]+"${SBATCH_OPTS[@]}"} "${JOB}" "$@"
