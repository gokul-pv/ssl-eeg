#!/bin/bash
# =============================================================================
# Masking ablation: pretraining of the eight ablation models (pretrain.py), one
# per array task. 200 epochs each (see the ablation configs).
#
#   bash scripts/slurm/submit.sh --array=0-7 scripts/slurm/ablation_pretrain.sh [KEY=VALUE ...]
#
#   0 ratio_20   1 ratio_50   2 ratio_75   3 ratio_90          (random masking)
#   4 strategy_channel   5 strategy_temporal   6 strategy_brain_region
#   7 strategy_column                                          (75% masking)
#
# Encoders are written to outputs/checkpoints/pretrain/ablation/<name>/brainlmeeg/.
# Then run ablation_finetune.sh.
# =============================================================================
#SBATCH --job-name=ablation_pretrain
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00

source "${SSL_EEG_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/slurm/common.sh"

ABLATIONS=(
    masking_ratio/ratio_20
    masking_ratio/ratio_50
    masking_ratio/ratio_75
    masking_ratio/ratio_90
    masking_strategy/strategy_channel
    masking_strategy/strategy_temporal
    masking_strategy/strategy_brain_region
    masking_strategy/strategy_column
)
TASK_ID="${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID (submit with --array=0-7)}"
if [ "${TASK_ID}" -ge "${#ABLATIONS[@]}" ]; then
    echo "ERROR: array task ${TASK_ID} out of range (0-$(( ${#ABLATIONS[@]} - 1 )))" >&2
    exit 1
fi

run_config_job pretrain.py "configs/ablation/${ABLATIONS[${TASK_ID}]}.yaml" "$@"
