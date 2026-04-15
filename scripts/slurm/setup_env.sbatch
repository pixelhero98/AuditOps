#!/bin/bash
#SBATCH --job-name=auditops-setup-env
#SBATCH --nodes=1
#SBATCH --time=00:30:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"

LOG_FILE="$AUDITOPS_LOG_ROOT/${SLURM_JOB_NAME:-auditops-setup-env}.${SLURM_JOB_ID:-manual}.out"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "HOST=$(hostname)"
echo "REPO=$AUDITOPS_REPO_ROOT"
echo "ENV=$AUDITOPS_ENV_ROOT"
echo "LOG=$LOG_FILE"

bash "$AUDITOPS_REPO_ROOT/scripts/setup_env.sh"
