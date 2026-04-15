#!/bin/bash
#SBATCH --job-name=auditops-reingest-single
#SBATCH --nodes=1
#SBATCH --time=01:00:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"

LOG_FILE="$AUDITOPS_LOG_ROOT/${SLURM_JOB_NAME:-auditops-reingest-single}.${SLURM_JOB_ID:-manual}.out"
exec > >(tee -a "$LOG_FILE") 2>&1

source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

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
