#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"

PYTHON_BIN="${AUDITOPS_PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Could not find a Python interpreter. Set AUDITOPS_PYTHON_BIN to continue." >&2
    exit 1
  fi
fi

"$PYTHON_BIN" -m venv "$AUDITOPS_ENV_ROOT"
source "$AUDITOPS_ENV_ROOT/bin/activate"

python -m pip install --upgrade pip setuptools wheel
cd "$AUDITOPS_REPO_ROOT"
python -m pip install -e "${AUDITOPS_INSTALL_SPEC:-.[dev,retrieval]}"

echo "AuditOps dev env is ready at $AUDITOPS_ENV_ROOT"
