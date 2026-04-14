#!/bin/bash
#SBATCH --job-name=auditops-uk-corpus-bootstrap
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00

set -euo pipefail

DEFAULT_REPO_ROOT="$HOME/work/AuditOps"
if [[ ! -d "$DEFAULT_REPO_ROOT" ]]; then
  DEFAULT_REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
fi
REPO_ROOT="${AUDITOPS_REPO_ROOT:-$DEFAULT_REPO_ROOT}"
source "$REPO_ROOT/scripts/isambard_env.sh"

LOG_FILE="$AUDITOPS_SCRATCH_ROOT/logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"
exec > >(tee -a "$LOG_FILE") 2>&1

source "$REPO_ROOT/scripts/isambard_activate_env.sh"

CORPUS_ROOT="${AUDITOPS_CORPUS_ROOT:?Set AUDITOPS_CORPUS_ROOT to the UK corpus root}"
CONSTITUENTS_CSV="${AUDITOPS_CONSTITUENTS_CSV:?Set AUDITOPS_CONSTITUENTS_CSV to the dated FTSE constituents CSV}"
SNAPSHOT_DATE="${AUDITOPS_SNAPSHOT_DATE:?Set AUDITOPS_SNAPSHOT_DATE to YYYY-MM-DD}"
CONSTITUENT_SOURCE="${AUDITOPS_CONSTITUENT_SOURCE:-$(basename "$CONSTITUENTS_CSV")}"

echo "HOST=$(hostname)"
echo "REPO=$AUDITOPS_REPO_ROOT"
echo "ENV=$AUDITOPS_ENV_ROOT"
echo "CORPUS_ROOT=$CORPUS_ROOT"
echo "CONSTITUENTS_CSV=$CONSTITUENTS_CSV"
echo "SNAPSHOT_DATE=$SNAPSHOT_DATE"
echo "LOG=$LOG_FILE"

auditops build-uk-manifest \
  --constituents "$CONSTITUENTS_CSV" \
  --corpus-root "$CORPUS_ROOT" \
  --snapshot-date "$SNAPSHOT_DATE" \
  --constituent-source "$CONSTITUENT_SOURCE"

auditops download-uk-filings --corpus-root "$CORPUS_ROOT"
