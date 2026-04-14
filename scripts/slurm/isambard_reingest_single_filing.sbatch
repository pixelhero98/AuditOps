#!/bin/bash
#SBATCH --job-name=auditops-reingest-single
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00

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

CORPUS_ROOT="${AUDITOPS_CORPUS_ROOT:?Set AUDITOPS_CORPUS_ROOT to the corpus root}"
TICKER="${AUDITOPS_TICKER:?Set AUDITOPS_TICKER to the issuer ticker}"
ACCESSION="${AUDITOPS_ACCESSION:?Set AUDITOPS_ACCESSION to the SEC accession number}"

echo "HOST=$(hostname)"
echo "REPO=$AUDITOPS_REPO_ROOT"
echo "ENV=$AUDITOPS_ENV_ROOT"
echo "CORPUS_ROOT=$CORPUS_ROOT"
echo "TICKER=$TICKER"
echo "ACCESSION=$ACCESSION"
echo "LOG=$LOG_FILE"

auditops repair-filing --corpus-root "$CORPUS_ROOT" --ticker "$TICKER" --accession "$ACCESSION"
