#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"

if [[ ! -f "$AUDITOPS_ENV_ROOT/bin/activate" ]]; then
  echo "AuditOps env not found at $AUDITOPS_ENV_ROOT" >&2
  echo "Run $AUDITOPS_REPO_ROOT/scripts/setup_env.sh first, or set AUDITOPS_ENV_ROOT to an existing virtualenv." >&2
  exit 1
fi

source "$AUDITOPS_ENV_ROOT/bin/activate"
cd "$AUDITOPS_REPO_ROOT"
