#!/bin/bash
#SBATCH --job-name=auditops-corpus-smoke
#SBATCH --nodes=1
#SBATCH --time=00:30:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"

LOG_FILE="$AUDITOPS_LOG_ROOT/${SLURM_JOB_NAME:-auditops-corpus-smoke}.${SLURM_JOB_ID:-manual}.out"
exec > >(tee -a "$LOG_FILE") 2>&1

source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

CORPUS_ROOT="${AUDITOPS_CORPUS_ROOT:?Set AUDITOPS_CORPUS_ROOT to the corpus root}"
CONSTITUENTS_CSV="${AUDITOPS_CONSTITUENTS_CSV:?Set AUDITOPS_CONSTITUENTS_CSV to the dated constituents CSV}"
SNAPSHOT_DATE="${AUDITOPS_SNAPSHOT_DATE:?Set AUDITOPS_SNAPSHOT_DATE to YYYY-MM-DD}"
CONSTITUENT_SOURCE="${AUDITOPS_CONSTITUENT_SOURCE:-$(basename "$CONSTITUENTS_CSV")}"

echo "HOST=$(hostname)"
echo "REPO=$AUDITOPS_REPO_ROOT"
echo "ENV=$AUDITOPS_ENV_ROOT"
echo "CORPUS_ROOT=$CORPUS_ROOT"
echo "CONSTITUENTS_CSV=$CONSTITUENTS_CSV"
echo "SNAPSHOT_DATE=$SNAPSHOT_DATE"
echo "LOG=$LOG_FILE"

auditops build-manifest \
  --constituents "$CONSTITUENTS_CSV" \
  --corpus-root "$CORPUS_ROOT" \
  --snapshot-date "$SNAPSHOT_DATE" \
  --constituent-source "$CONSTITUENT_SOURCE"

auditops download-filings --corpus-root "$CORPUS_ROOT"
auditops ingest-corpus --corpus-root "$CORPUS_ROOT"
auditops generate-corpus-datasets --corpus-root "$CORPUS_ROOT"
auditops eval-corpus --corpus-root "$CORPUS_ROOT"
