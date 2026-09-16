#!/bin/bash
#SBATCH --job-name=auditops-corpus-rebuild
#SBATCH --nodes=1
#SBATCH --time=08:00:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"

LOG_FILE="$AUDITOPS_LOG_ROOT/${SLURM_JOB_NAME:-auditops-corpus-rebuild}.${SLURM_JOB_ID:-manual}.out"
exec > >(tee -a "$LOG_FILE") 2>&1

source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

CORPUS_ROOT="${AUDITOPS_CORPUS_ROOT:?Set AUDITOPS_CORPUS_ROOT to the corpus root}"

echo "HOST=$(hostname)"
echo "REPO=$AUDITOPS_REPO_ROOT"
echo "ENV=$AUDITOPS_ENV_ROOT"
echo "CORPUS_ROOT=$CORPUS_ROOT"
echo "LOG=$LOG_FILE"

python -m auditops.cli ingest-corpus --corpus-root "$CORPUS_ROOT"
python -m auditops.cli generate-corpus-datasets --corpus-root "$CORPUS_ROOT"
python -m auditops.cli eval-corpus --corpus-root "$CORPUS_ROOT"
