#!/bin/bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this inside a Slurm allocation on hopper." >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/isambard_env.sh"

python3 -m venv "$AUDITOPS_ENV_ROOT"
source "$AUDITOPS_ENV_ROOT/bin/activate"

python -m pip install --upgrade pip setuptools wheel
cd "$AUDITOPS_REPO_ROOT"
python -m pip install -e '.[dev,retrieval]'

echo "AuditOps dev env is ready at $AUDITOPS_ENV_ROOT"
