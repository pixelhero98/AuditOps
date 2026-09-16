#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_DB_PATH:?Set AUDITOPS_DB_PATH to the frozen US corpus SQLite DB}"
: "${AUDITOPS_US_AUDIT_NARRATIVE_OUTPUT:?Set AUDITOPS_US_AUDIT_NARRATIVE_OUTPUT}"
SOURCE_SYSTEM=${AUDITOPS_SOURCE_SYSTEM:-SEC-EDGAR}

auditops_assert_new_path "$AUDITOPS_US_AUDIT_NARRATIVE_OUTPUT" "US audit narrative task output"
auditops_start_log auditops-us-audit-tasks
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

args=(
  build-us-audit-narrative-tasks
  --db "$AUDITOPS_DB_PATH"
  --output "$AUDITOPS_US_AUDIT_NARRATIVE_OUTPUT"
  --source-system "$SOURCE_SYSTEM"
)
if [[ -n "${AUDITOPS_BASE_NARRATIVE_JSONL:-}" ]]; then
  if [[ ! -f "$AUDITOPS_BASE_NARRATIVE_JSONL" ]]; then
    echo "Base narrative JSONL not found: $AUDITOPS_BASE_NARRATIVE_JSONL" >&2
    exit 2
  fi
  args+=(--base-narrative-jsonl "$AUDITOPS_BASE_NARRATIVE_JSONL")
fi

python -m auditops.cli "${args[@]}"
