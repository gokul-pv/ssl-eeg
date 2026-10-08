#!/bin/bash
# =============================================================================
# Frozen EEG foundation model (LaBraM, CBraMod, REVE) + classification head (probe.py).
#
#   bash scripts/slurm/submit.sh scripts/slurm/probe.sh configs/probe/labram_lp_adftd.yaml
#
# Weights are downloaded from Hugging Face into HF_HOME_DIR (cluster.env). REVE is
# gated: export HF_TOKEN before submitting. On compute nodes without internet,
# download once on a login node and submit with HF_HUB_OFFLINE=1.
# =============================================================================
#SBATCH --job-name=probe
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00

source "${SSL_EEG_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/slurm/common.sh"

export HF_HOME="${HF_HOME:-${HF_HOME_DIR:-outputs/hf_cache}}"
[[ "${HF_HOME}" = /* ]] || HF_HOME="${CODE_DIR}/${HF_HOME}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
mkdir -p "${HF_HOME}"
run_config_job probe.py "$@"
