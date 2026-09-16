#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_CH_PAIR_MANIFEST:?Set AUDITOPS_CH_PAIR_MANIFEST}"
: "${AUDITOPS_CH_FACT_AUDIT:?Set AUDITOPS_CH_FACT_AUDIT}"
: "${AUDITOPS_CH_VLM_DIR:?Set AUDITOPS_CH_VLM_DIR}"

auditops_assert_new_path "$AUDITOPS_CH_VLM_DIR" "VLM diagnostic output"
auditops_start_log auditops-vlm-data
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

python -m auditops.cli build-companies-house-vlm-diagnostic \
  --pair-manifest "$AUDITOPS_CH_PAIR_MANIFEST" \
  --fact-audit "$AUDITOPS_CH_FACT_AUDIT" \
  --output-dir "$AUDITOPS_CH_VLM_DIR"
