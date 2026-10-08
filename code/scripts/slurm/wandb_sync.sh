#!/bin/bash
# =============================================================================
# Upload offline W&B runs (wandb_mode: offline in the configs). Run on a node
# with internet access after the jobs finish; needs `wandb login` once.
#
#   bash scripts/slurm/wandb_sync.sh                              # everything under outputs/wandb/
#   bash scripts/slurm/wandb_sync.sh outputs/wandb/finetune/ablation
#
# Uses the wandb CLI on PATH if there is one, otherwise the one in the container.
# =============================================================================

source "${SSL_EEG_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/slurm/common.sh"

ROOTS=("$@")
[ ${#ROOTS[@]} -gt 0 ] || ROOTS=(outputs/wandb)

if command -v wandb >/dev/null 2>&1; then
    WANDB=(wandb)
else
    WANDB=(singularity exec --userns --bind "${CODE_DIR}:${CODE_DIR}" --pwd "${CODE_DIR}" "${SINGULARITY_IMAGE}" wandb)
fi

n_ok=0; n_failed=0
while IFS= read -r -d '' run_dir; do
    echo "→ ${run_dir}"
    if "${WANDB[@]}" sync "${run_dir}"; then n_ok=$(( n_ok + 1 )); else n_failed=$(( n_failed + 1 )); fi
done < <(find "${ROOTS[@]}" -type d -name "offline-run-*" -print0 | sort -z)

echo "Synced ${n_ok} run(s), ${n_failed} failed."
[ "${n_failed}" -eq 0 ]
