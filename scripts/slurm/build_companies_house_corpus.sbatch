#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_CH_BULK_ZIP:?Set AUDITOPS_CH_BULK_ZIP}"
: "${AUDITOPS_CH_CORPUS_DIR:?Set AUDITOPS_CH_CORPUS_DIR}"

auditops_assert_new_path "$AUDITOPS_CH_CORPUS_DIR" "Companies House corpus output"
auditops_start_log auditops-ch-corpus
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

args=(
  build-companies-house-corpus
  --bulk-zip "$AUDITOPS_CH_BULK_ZIP"
  --output-dir "$AUDITOPS_CH_CORPUS_DIR"
  --snapshot-date "${AUDITOPS_CH_SNAPSHOT_DATE:-2026-01-31}"
  --seed "${AUDITOPS_SEED:-20260821}"
)
if [[ -n "${AUDITOPS_CH_MAX_FILINGS:-}" ]]; then
  args+=(--max-filings "$AUDITOPS_CH_MAX_FILINGS")
fi

python -m auditops.cli "${args[@]}"
