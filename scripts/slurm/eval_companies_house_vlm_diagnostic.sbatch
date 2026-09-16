#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_CH_VLM_DIR:?Set the frozen VLM diagnostic directory}"
: "${AUDITOPS_CH_VLM_PREDICTIONS:?Set the immutable predictions JSONL}"
: "${AUDITOPS_CH_VLM_EVAL_OUTPUT_DIR:?Set a new evaluation output directory}"

auditops_assert_new_path "$AUDITOPS_CH_VLM_EVAL_OUTPUT_DIR" "VLM evaluation output"
auditops_start_log auditops-ch-vlm-eval
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

python -m auditops.cli eval-companies-house-vlm-diagnostic \
  --diagnostic-dir "$AUDITOPS_CH_VLM_DIR" \
  --predictions-jsonl "$AUDITOPS_CH_VLM_PREDICTIONS" \
  --output-dir "$AUDITOPS_CH_VLM_EVAL_OUTPUT_DIR"
