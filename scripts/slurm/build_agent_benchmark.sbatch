#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_QUANT_JSONL:?Set AUDITOPS_QUANT_JSONL}"
: "${AUDITOPS_NARRATIVE_JSONL:?Set AUDITOPS_NARRATIVE_JSONL}"
: "${AUDITOPS_BENCHMARK_DIR:?Set AUDITOPS_BENCHMARK_DIR}"
: "${AUDITOPS_BENCHMARK_PROFILE:?Set AUDITOPS_BENCHMARK_PROFILE to an explicit frozen benchmark profile}"
: "${AUDITOPS_JURISDICTION:?Set AUDITOPS_JURISDICTION to US or UK}"
: "${AUDITOPS_CORPUS_ID:?Set AUDITOPS_CORPUS_ID}"
: "${AUDITOPS_REPORTING_FRAMEWORK:?Set AUDITOPS_REPORTING_FRAMEWORK}"
: "${AUDITOPS_STANDARDS_VERSION:?Set AUDITOPS_STANDARDS_VERSION}"
: "${AUDITOPS_SOURCE_SYSTEM:?Set AUDITOPS_SOURCE_SYSTEM}"
if [[ "$AUDITOPS_JURISDICTION" != "US" && "$AUDITOPS_JURISDICTION" != "UK" ]]; then
  echo "AUDITOPS_JURISDICTION must be US or UK" >&2
  exit 2
fi

auditops_assert_new_path "$AUDITOPS_BENCHMARK_DIR" "benchmark output"
auditops_start_log auditops-build-benchmark
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

AUDITOPS_PARENT_BENCHMARK_ARGS=()
if [[ -n "${AUDITOPS_PARENT_BENCHMARK_DIR:-}" ]]; then
  if [[ ! -d "$AUDITOPS_PARENT_BENCHMARK_DIR" || -L "$AUDITOPS_PARENT_BENCHMARK_DIR" ]]; then
    echo "Parent benchmark must be an existing non-symlink directory: $AUDITOPS_PARENT_BENCHMARK_DIR" >&2
    exit 2
  fi
  AUDITOPS_PARENT_BENCHMARK_ARGS=(
    --parent-benchmark-dir "$AUDITOPS_PARENT_BENCHMARK_DIR"
  )
fi

python -m auditops.cli build-agent-benchmark \
  --quant-jsonl "$AUDITOPS_QUANT_JSONL" \
  --narrative-jsonl "$AUDITOPS_NARRATIVE_JSONL" \
  --output-dir "$AUDITOPS_BENCHMARK_DIR" \
  --benchmark-profile "$AUDITOPS_BENCHMARK_PROFILE" \
  --jurisdiction "$AUDITOPS_JURISDICTION" \
  --corpus-id "$AUDITOPS_CORPUS_ID" \
  --reporting-framework "$AUDITOPS_REPORTING_FRAMEWORK" \
  --standards-version "$AUDITOPS_STANDARDS_VERSION" \
  --source-system "$AUDITOPS_SOURCE_SYSTEM" \
  --seed "${AUDITOPS_SEED:-20260821}" \
  --quant-count "${AUDITOPS_QUANT_COUNT:-300}" \
  --narrative-count "${AUDITOPS_NARRATIVE_COUNT:-200}" \
  "${AUDITOPS_PARENT_BENCHMARK_ARGS[@]}"
