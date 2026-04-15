#!/bin/bash
#SBATCH --job-name=auditops-dev-check
#SBATCH --nodes=1
#SBATCH --time=00:20:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"

LOG_FILE="$AUDITOPS_LOG_ROOT/${SLURM_JOB_NAME:-auditops-dev-check}.${SLURM_JOB_ID:-manual}.out"
exec > >(tee -a "$LOG_FILE") 2>&1

source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

echo "HOST=$(hostname)"
echo "REPO=$AUDITOPS_REPO_ROOT"
echo "ENV=$AUDITOPS_ENV_ROOT"
echo "LOG=$LOG_FILE"

nvidia-smi -L || true
python -c "import auditops; print(auditops.__file__)"
auditops --help >/dev/null
python -m pytest -q
