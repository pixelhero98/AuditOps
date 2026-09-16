#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

for name in AUDITOPS_BENCHMARK_DIR AUDITOPS_GATE_RUN_DIR AUDITOPS_GATE_EVAL_INITIAL_DIR \
  AUDITOPS_GATE_PROVISIONAL_REVIEW AUDITOPS_GATE_EVAL_FINAL_DIR \
  AUDITOPS_EXPLORATORY_AUTHORIZATION; do
  if [[ -z "${!name:-}" ]]; then echo "Set $name" >&2; exit 2; fi
done
for source in "$AUDITOPS_BENCHMARK_DIR" "$AUDITOPS_GATE_RUN_DIR"; do
  [[ -d "$source" && ! -L "$source" ]] || { echo "Invalid input directory: $source" >&2; exit 2; }
done
for target in "$AUDITOPS_GATE_EVAL_INITIAL_DIR" "$AUDITOPS_GATE_PROVISIONAL_REVIEW" \
  "$AUDITOPS_GATE_EVAL_FINAL_DIR" "$AUDITOPS_EXPLORATORY_AUTHORIZATION"; do
  auditops_assert_new_path "$target" "v2.4p gate artifact"
done

auditops_start_log auditops-v24p-gate-eval
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"
python -m auditops.cli eval-agent-baseline-v24 \
  --benchmark-dir "$AUDITOPS_BENCHMARK_DIR" --run-dir "$AUDITOPS_GATE_RUN_DIR" \
  --output-dir "$AUDITOPS_GATE_EVAL_INITIAL_DIR"
python -m auditops.cli record-gate-provisional-review-v24p \
  --benchmark-dir "$AUDITOPS_BENCHMARK_DIR" --run-dir "$AUDITOPS_GATE_RUN_DIR" \
  --release-review-packet "$AUDITOPS_GATE_EVAL_INITIAL_DIR/release_review_packet.json" \
  --output "$AUDITOPS_GATE_PROVISIONAL_REVIEW"
python -m auditops.cli eval-agent-baseline-v24 \
  --benchmark-dir "$AUDITOPS_BENCHMARK_DIR" --run-dir "$AUDITOPS_GATE_RUN_DIR" \
  --provisional-release-review "$AUDITOPS_GATE_PROVISIONAL_REVIEW" \
  --output-dir "$AUDITOPS_GATE_EVAL_FINAL_DIR"
python -m auditops.cli authorize-exploratory-full-run-v24p \
  --benchmark-dir "$AUDITOPS_BENCHMARK_DIR" \
  --gate-metrics "$AUDITOPS_GATE_EVAL_FINAL_DIR/metrics.json" \
  --authorized-by "${AUDITOPS_EXPLORATORY_AUTHORIZER:?Set the explicit exploratory authorizer}" --output "$AUDITOPS_EXPLORATORY_AUTHORIZATION"
python -m auditops.cli scan-artifacts --path "$AUDITOPS_GATE_EVAL_FINAL_DIR" \
  --path "$AUDITOPS_GATE_PROVISIONAL_REVIEW" --path "$AUDITOPS_EXPLORATORY_AUTHORIZATION"
find "$AUDITOPS_GATE_EVAL_INITIAL_DIR" "$AUDITOPS_GATE_EVAL_FINAL_DIR" -type f -exec chmod 0444 {} +
find "$AUDITOPS_GATE_EVAL_INITIAL_DIR" "$AUDITOPS_GATE_EVAL_FINAL_DIR" -depth -type d -exec chmod 0555 {} +
chmod 0444 "$AUDITOPS_GATE_PROVISIONAL_REVIEW" "$AUDITOPS_EXPLORATORY_AUTHORIZATION"
