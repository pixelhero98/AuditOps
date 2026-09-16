#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_V24_CANDIDATE_DIR:?Set AUDITOPS_V24_CANDIDATE_DIR}"
: "${AUDITOPS_V24_REVIEW_HTML:?Set AUDITOPS_V24_REVIEW_HTML}"
: "${AUDITOPS_V24_PRIMARY_REVIEW:?Set AUDITOPS_V24_PRIMARY_REVIEW}"
: "${AUDITOPS_V24_SECONDARY_REVIEW:?Set AUDITOPS_V24_SECONDARY_REVIEW}"
: "${AUDITOPS_V24_APPROVAL_OUTPUT:?Set AUDITOPS_V24_APPROVAL_OUTPUT}"

if [[ ! -d "$AUDITOPS_V24_CANDIDATE_DIR" || -L "$AUDITOPS_V24_CANDIDATE_DIR" ]]; then
  echo "v2.4 candidate is not a non-symlink directory" >&2
  exit 2
fi
for source in "$AUDITOPS_V24_REVIEW_HTML" "$AUDITOPS_V24_PRIMARY_REVIEW" "$AUDITOPS_V24_SECONDARY_REVIEW"; do
  if [[ ! -f "$source" || -L "$source" ]]; then
    echo "Required human-review artifact is not a regular file: $source" >&2
    exit 2
  fi
done
if [[ -n "${AUDITOPS_V24_ADJUDICATION_REVIEW:-}" && ( ! -f "$AUDITOPS_V24_ADJUDICATION_REVIEW" || -L "$AUDITOPS_V24_ADJUDICATION_REVIEW" ) ]]; then
  echo "Adjudication review is not a regular file" >&2
  exit 2
fi
auditops_assert_new_path "$AUDITOPS_V24_APPROVAL_OUTPUT" "v2.4 benchmark approval"

auditops_start_log auditops-v24-approve
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

approval_args=(
  --candidate-dir "$AUDITOPS_V24_CANDIDATE_DIR"
  --rendered-packet "$AUDITOPS_V24_REVIEW_HTML"
  --primary-review "$AUDITOPS_V24_PRIMARY_REVIEW"
  --secondary-review "$AUDITOPS_V24_SECONDARY_REVIEW"
  --output "$AUDITOPS_V24_APPROVAL_OUTPUT"
)
if [[ -n "${AUDITOPS_V24_ADJUDICATION_REVIEW:-}" ]]; then
  approval_args+=(--adjudication-review "$AUDITOPS_V24_ADJUDICATION_REVIEW")
fi
python -m auditops.cli approve-agent-benchmark-v24 "${approval_args[@]}"
python -m auditops.cli scan-artifacts --path "$AUDITOPS_V24_APPROVAL_OUTPUT"
chmod 0444 "$AUDITOPS_V24_APPROVAL_OUTPUT"

echo "V24_APPROVAL=$AUDITOPS_V24_APPROVAL_OUTPUT"
