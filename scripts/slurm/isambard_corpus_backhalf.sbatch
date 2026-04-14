#!/bin/bash
#SBATCH --job-name=auditops-corpus-backhalf
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00

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

echo "HOST=$(hostname)"
echo "REPO=$AUDITOPS_REPO_ROOT"
echo "ENV=$AUDITOPS_ENV_ROOT"
echo "CORPUS_ROOT=$CORPUS_ROOT"
echo "LOG=$LOG_FILE"

auditops generate-corpus-datasets --corpus-root "$CORPUS_ROOT"
auditops eval-corpus --corpus-root "$CORPUS_ROOT"
