#!/bin/bash
# =============================================================================
# Masking ablation: linear probe of each ablation encoder on the three
# downstream tasks (finetune.py, five-fold rolling CV).
#
#   bash scripts/slurm/submit.sh --array=0-23 scripts/slurm/ablation_finetune.sh [KEY=VALUE ...]
#
# Array task = 3 * ABLATION + TASK
#   ABLATION  0 ratio_20   1 ratio_50   2 ratio_75   3 ratio_90
#             4 strategy_channel   5 strategy_temporal   6 strategy_brain_region
#             7 strategy_column
#   TASK      0 AD vs HC   1 FTD vs HC   2 FEP vs HC
#   e.g. ratio_75 only: --array=6-8
#
# Needs the encoders of ablation_pretrain.sh.
# =============================================================================
#SBATCH --job-name=ablation_finetune
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00

source "${SSL_EEG_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/slurm/common.sh"

ABLATIONS=(ratio_20 ratio_50 ratio_75 ratio_90
           strategy_channel strategy_temporal strategy_brain_region strategy_column)
TASKS=(lp_ad_vs_hc lp_ftd_vs_hc lp_fep)

TASK_ID="${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID (submit with --array=0-23)}"
N_TASKS=${#TASKS[@]}
if [ "${TASK_ID}" -ge $(( ${#ABLATIONS[@]} * N_TASKS )) ]; then
    echo "ERROR: array task ${TASK_ID} out of range (0-$(( ${#ABLATIONS[@]} * N_TASKS - 1 )))" >&2
    exit 1
fi
ABLATION="${ABLATIONS[$(( TASK_ID / N_TASKS ))]}"
TASK="${TASKS[$(( TASK_ID % N_TASKS ))]}"

ENCODER="outputs/checkpoints/pretrain/ablation/${ABLATION}/brainlmeeg/best_pretrain_encoder.pth"
if [ ! -f "${ENCODER}" ]; then
    echo "ERROR: encoder not found: ${CODE_DIR}/${ENCODER} (run ablation_pretrain.sh first)" >&2
    exit 1
fi

run_config_job finetune.py "configs/ablation/finetune/${TASK}.yaml" \
    "ablation_name=${ABLATION}" \
    "pretrained_encoder_path=${ENCODER}" \
    "save_dir=outputs/checkpoints/finetune/ablation/${ABLATION}" \
    "wandb_dir=outputs/wandb/finetune/ablation/${ABLATION}" \
    "terminal_log_dir=outputs/logs/finetune/ablation/${ABLATION}" \
    "$@"
