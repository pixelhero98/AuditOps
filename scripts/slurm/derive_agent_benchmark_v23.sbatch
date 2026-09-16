#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_PARENT_BENCHMARK_DIR:?Set AUDITOPS_PARENT_BENCHMARK_DIR}"
: "${AUDITOPS_BENCHMARK_DIR:?Set AUDITOPS_BENCHMARK_DIR}"

if [[ ! -d "$AUDITOPS_PARENT_BENCHMARK_DIR" || -L "$AUDITOPS_PARENT_BENCHMARK_DIR" ]]; then
  echo "Parent benchmark must be an existing non-symlink directory" >&2
  exit 2
fi
auditops_assert_new_path "$AUDITOPS_BENCHMARK_DIR" "v2.3 benchmark output"

auditops_start_log auditops-derive-v23
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

python -m auditops.cli derive-agent-benchmark-v23 \
  --parent-benchmark-dir "$AUDITOPS_PARENT_BENCHMARK_DIR" \
  --output-dir "$AUDITOPS_BENCHMARK_DIR"
