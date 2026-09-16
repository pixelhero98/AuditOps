#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_BENCHMARK_DIR:?Set AUDITOPS_BENCHMARK_DIR}"
: "${AUDITOPS_EVAL_OUTPUT_DIR:?Set AUDITOPS_EVAL_OUTPUT_DIR}"
AUDITOPS_EVAL_COMMAND=${AUDITOPS_EVAL_COMMAND:-eval-agent-baseline}
if [[ "$AUDITOPS_EVAL_COMMAND" != "eval-agent-baseline" && \
      "$AUDITOPS_EVAL_COMMAND" != "report-agent-baseline" ]]; then
  echo "AUDITOPS_EVAL_COMMAND must be eval-agent-baseline or report-agent-baseline" >&2
  exit 2
fi
if (( $# == 0 )); then
  echo "Pass one or more immutable runs as NAME=RESULTS_JSONL" >&2
  exit 2
fi

auditops_assert_new_path "$AUDITOPS_EVAL_OUTPUT_DIR" "evaluation output"
auditops_start_log auditops-eval
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

args=("$AUDITOPS_EVAL_COMMAND" --benchmark-dir "$AUDITOPS_BENCHMARK_DIR" --output-dir "$AUDITOPS_EVAL_OUTPUT_DIR")
for named_run in "$@"; do
  if [[ "$named_run" != *=* ]]; then
    echo "Each run must use NAME=RESULTS_JSONL: $named_run" >&2
    exit 2
  fi
  args+=(--run "$named_run")
done
if [[ -n "${AUDITOPS_QUANTIZATION_PAIRS:-}" ]]; then
  IFS=';' read -r -a quantization_pairs <<< "$AUDITOPS_QUANTIZATION_PAIRS"
  for pair in "${quantization_pairs[@]}"; do
    if [[ -n "$pair" ]]; then
      args+=(--quantization-pair "$pair")
    fi
  done
fi
python -m auditops.cli "${args[@]}"
