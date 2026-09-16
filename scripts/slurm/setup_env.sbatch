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

if [[ -n "${AUDITOPS_PACKAGE_LOCK_FILE:-}" ]]; then
  if [[ ! -f "$AUDITOPS_PACKAGE_LOCK_FILE" ]]; then
    echo "Package lock not found: $AUDITOPS_PACKAGE_LOCK_FILE" >&2
    exit 2
  fi
  if [[ -e "$AUDITOPS_ENV_ROOT/bin/activate" ]]; then
    echo "Refusing to overwrite existing environment: $AUDITOPS_ENV_ROOT" >&2
    exit 2
  fi
  PYTHON_BIN=${AUDITOPS_PYTHON_BIN:-python3}
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 2
  fi
  "$PYTHON_BIN" -m venv "$AUDITOPS_ENV_ROOT"
  source "$AUDITOPS_ENV_ROOT/bin/activate"
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  export PIP_NO_INPUT=1
  python -m pip install --require-hashes --only-binary=:all: \
    -r "$AUDITOPS_PACKAGE_LOCK_FILE"
  cd "$AUDITOPS_REPO_ROOT"
  python -c "import auditops; print(auditops.__file__)"
  python -m auditops.cli --help >/dev/null
  hf --help >/dev/null
  echo "AuditOps locked environment is ready at $AUDITOPS_ENV_ROOT"
else
  bash "$AUDITOPS_REPO_ROOT/scripts/setup_env.sh"
fi
