#!/bin/bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this inside a Slurm allocation on your compute partition before installing the Isambard profile env." >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export AUDITOPS_PROFILE="${AUDITOPS_PROFILE:-isambard3}"

bash "$AUDITOPS_REPO_ROOT/scripts/setup_env.sh"
