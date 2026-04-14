#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/isambard_env.sh"

if [[ ! -f "$AUDITOPS_ENV_ROOT/bin/activate" ]]; then
  echo "AuditOps env not found at $AUDITOPS_ENV_ROOT" >&2
  echo "Run $SCRIPT_DIR/isambard_setup_env.sh inside a Slurm allocation on hopper first." >&2
  exit 1
fi

source "$AUDITOPS_ENV_ROOT/bin/activate"
cd "$AUDITOPS_REPO_ROOT"
