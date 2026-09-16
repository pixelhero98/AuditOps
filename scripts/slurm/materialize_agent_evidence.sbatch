#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_QUANT_JSONL:?Set AUDITOPS_QUANT_JSONL}"
: "${AUDITOPS_NARRATIVE_JSONL:?Set AUDITOPS_NARRATIVE_JSONL}"
: "${AUDITOPS_DB_PATH:?Set AUDITOPS_DB_PATH to the jurisdiction-specific corpus SQLite DB}"
: "${AUDITOPS_EVIDENCE_OUTPUT_DIR:?Set AUDITOPS_EVIDENCE_OUTPUT_DIR}"

TOP_K=${AUDITOPS_EVIDENCE_TOP_K:-5}
CANDIDATE_K=${AUDITOPS_EVIDENCE_CANDIDATE_K:-15}
SOURCE_SYSTEM=${AUDITOPS_SOURCE_SYSTEM:-SEC-EDGAR}
if [[ "$TOP_K" != "5" || "$CANDIDATE_K" != "15" ]]; then
  echo "AuditOps v0.1 fixes evidence retrieval at top_k=5 and candidate_k=15" >&2
  exit 2
fi

auditops_assert_new_path "$AUDITOPS_EVIDENCE_OUTPUT_DIR" "evidence output"
auditops_start_log auditops-materialize-evidence
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

python -m auditops.cli materialize-agent-evidence \
  --quant-jsonl "$AUDITOPS_QUANT_JSONL" \
  --narrative-jsonl "$AUDITOPS_NARRATIVE_JSONL" \
  --db "$AUDITOPS_DB_PATH" \
  --output-dir "$AUDITOPS_EVIDENCE_OUTPUT_DIR" \
  --top-k "$TOP_K" \
  --candidate-k "$CANDIDATE_K" \
  --source-system "$SOURCE_SYSTEM"
