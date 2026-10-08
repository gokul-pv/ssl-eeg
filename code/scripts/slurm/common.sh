#!/bin/bash
# Shared set-up for the job scripts in scripts/slurm/ (sourced, not run).
#
# Provides CODE_DIR (absolute path of code/; the job runs there), the values of
# configs/cluster/cluster.env, and:
#   resolve_config <path>            config path → absolute path (accepts code/-relative
#                                    or repository-relative paths, e.g. code/configs/...)
#   run_python [--cpu] <script> ...  run code/<script> inside the container
#                                    (through srun inside a SLURM job; --cpu: no --nv)
#   run_config_job [--cpu] <script> <config> [KEY=VALUE ...]
#                                    the usual job: one config plus --set overrides
#   job_header <label> / job_footer  log lines

set -euo pipefail

if [ -n "${SSL_EEG_CODE_DIR:-}" ]; then
    CODE_DIR="${SSL_EEG_CODE_DIR}"
else
    CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [ ! -f "${CODE_DIR}/pyproject.toml" ]; then
    echo "ERROR: ${CODE_DIR} is not the code/ folder of the repository." >&2
    echo "Submit jobs with scripts/slurm/submit.sh (it exports SSL_EEG_CODE_DIR)." >&2
    exit 1
fi
cd "${CODE_DIR}"

CLUSTER_ENV="${SSL_EEG_CLUSTER_ENV:-${CODE_DIR}/configs/cluster/cluster.env}"
# shellcheck source=../../configs/cluster/cluster.env
source "${CLUSTER_ENV}"

ulimit -n 65536 2>/dev/null || true   # many DataLoader workers open many files

resolve_config() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        :
    elif [ -f "${CODE_DIR}/${path}" ]; then
        path="${CODE_DIR}/${path}"
    elif [ -f "${CODE_DIR}/../${path}" ]; then
        path="$(cd "$(dirname "${CODE_DIR}/../${path}")" && pwd)/$(basename "${path}")"
    fi
    if [ ! -f "${path}" ]; then
        echo "ERROR: config not found: $1" >&2
        exit 1
    fi
    echo "${path}"
}

run_python() {
    local nv="--nv"
    if [ "${1:-}" = "--cpu" ]; then nv=""; shift; fi
    local script="$1"; shift
    local launcher=()
    [ -n "${SLURM_JOB_ID:-}" ] && launcher=(srun)

    if [ "${SSL_EEG_NO_CONTAINER:-0}" = "1" ]; then
        ${launcher[@]+"${launcher[@]}"} python "${CODE_DIR}/${script}" "$@"
        return
    fi
    local binds=(--bind "${CODE_DIR}:${CODE_DIR}")
    for p in ${BIND_PATHS:-}; do binds+=(--bind "${p}:${p}"); done
    local envs=()
    for v in HF_HOME HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_OFFLINE WANDB_API_KEY WANDB_ENTITY; do
        [ -n "${!v:-}" ] && envs+=(--env "${v}=${!v}")
    done
    # shellcheck disable=SC2086  # ${nv} is intentionally empty for CPU jobs
    ${launcher[@]+"${launcher[@]}"} singularity exec --userns ${nv} "${binds[@]}" ${envs[@]+"${envs[@]}"} \
        --pwd "${CODE_DIR}" "${SINGULARITY_IMAGE}" \
        python "${CODE_DIR}/${script}" "$@"
}

run_config_job() {
    local cpu=()
    if [ "${1:-}" = "--cpu" ]; then cpu=(--cpu); shift; fi
    local script="$1"; shift
    if [ $# -lt 1 ]; then
        echo "Usage: $(basename "$0") <config.yaml> [KEY=VALUE ...]" >&2
        exit 1
    fi
    local config; config="$(resolve_config "$1")"; shift
    local set_args=()
    [ $# -gt 0 ] && set_args=(--set "$@")
    job_header "${script} --config ${config#"${CODE_DIR}"/} ${*:-}"
    run_python ${cpu[@]+"${cpu[@]}"} "${script}" --config "${config}" ${set_args[@]+"${set_args[@]}"}
    job_footer
}

job_header() {
    echo "============================================================"
    echo "Job       : ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+  array task ${SLURM_ARRAY_TASK_ID}}"
    echo "Node      : $(hostname)"
    echo "Run       : $*"
    echo "Started   : $(date)"
    echo "============================================================"
}

job_footer() {
    echo "============================================================"
    echo "Finished  : $(date)"
    echo "============================================================"
}
