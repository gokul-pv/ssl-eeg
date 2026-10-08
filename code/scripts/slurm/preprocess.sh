#!/bin/bash
# =============================================================================
# Preprocessing (preprocess.py), one subject per array task. CPU only.
#
#   bash scripts/slurm/submit.sh --array=0-<N-1>%16 scripts/slurm/preprocess.sh <dataset> [KEY=VALUE ...]
#
# <dataset> is the name of configs/preprocess/<dataset>.yaml. Array task i
# processes the i-th subject of the sorted subject list in
# metadata/<dataset_id>/master_subject_table_*.csv
# (python preprocess.py --config configs/preprocess/<dataset>.yaml --list-subjects).
#
#   dataset  dataset_id  subjects  submit with
#   adftd    ds004504        88    --array=0-87%16
#   fep      ds003944        82    --array=0-81%16
#   mdd      ds003478       122    --array=0-121%16
#   scz      scz             77    --array=0-76%16
#   lemon    LEMON          215    --array=0-214%16
#   srm      ds003775       111    --array=0-110%16
#   dvs      ds005385       608    --array=0-607%16 --mem=32G --time=04:00:00
#
# The raw-data locations come from raw_data_path in the preprocessing config;
# BIND_PATHS in configs/cluster/cluster.env must cover them. Outside SLURM, set
# SLURM_ARRAY_TASK_ID or run preprocess.py without --subject-index to process
# every subject in one go.
# =============================================================================
#SBATCH --job-name=preprocess
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

source "${SSL_EEG_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/slurm/common.sh"

if [ $# -lt 1 ]; then
    echo "Usage: $(basename "$0") <adftd|fep|mdd|scz|lemon|srm|dvs> [KEY=VALUE ...]" >&2
    exit 1
fi
DATASET="$1"; shift
CONFIG="$(resolve_config "configs/preprocess/${DATASET}.yaml")"
INDEX="${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID (submit with --array)}"

job_header "preprocess.py ${DATASET} subject index ${INDEX}"
run_python --cpu preprocess.py --config "${CONFIG}" --subject-index "${INDEX}" \
    --set "n_jobs=${SLURM_CPUS_PER_TASK:-4}" "$@"
job_footer
