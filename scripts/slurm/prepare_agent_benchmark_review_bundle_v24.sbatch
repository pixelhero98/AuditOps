#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_V24_CANDIDATE_DIR:?Set AUDITOPS_V24_CANDIDATE_DIR}"
: "${AUDITOPS_V24_REVIEW_HTML:?Set AUDITOPS_V24_REVIEW_HTML}"
: "${AUDITOPS_V24_CORPUS_DB:?Set AUDITOPS_V24_CORPUS_DB}"
: "${AUDITOPS_V24_RAW_FILING_ROOT:?Set AUDITOPS_V24_RAW_FILING_ROOT}"
: "${AUDITOPS_V24_REVIEW_BUNDLE_DIR:?Set AUDITOPS_V24_REVIEW_BUNDLE_DIR}"

for source in "$AUDITOPS_V24_REVIEW_HTML" "$AUDITOPS_V24_CORPUS_DB"; do
  if [[ ! -f "$source" || -L "$source" ]]; then
    echo "Required v2.4 review source is not a regular file: $source" >&2
    exit 2
  fi
done
for source in "$AUDITOPS_V24_CANDIDATE_DIR" "$AUDITOPS_V24_RAW_FILING_ROOT"; do
  if [[ ! -d "$source" || -L "$source" ]]; then
    echo "Required v2.4 review source is not a non-symlink directory: $source" >&2
    exit 2
  fi
done
auditops_assert_new_path "$AUDITOPS_V24_REVIEW_BUNDLE_DIR" "v2.4 offline review bundle"

auditops_start_log auditops-v24-review-bundle
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

python -m auditops.cli prepare-agent-benchmark-review-bundle-v24 \
  --candidate-dir "$AUDITOPS_V24_CANDIDATE_DIR" \
  --rendered-packet "$AUDITOPS_V24_REVIEW_HTML" \
  --corpus-db "$AUDITOPS_V24_CORPUS_DB" \
  --raw-filing-root "$AUDITOPS_V24_RAW_FILING_ROOT" \
  --output-dir "$AUDITOPS_V24_REVIEW_BUNDLE_DIR"
python -m auditops.cli verify-agent-benchmark-review-bundle-v24 \
  --bundle-dir "$AUDITOPS_V24_REVIEW_BUNDLE_DIR"
python -m auditops.cli scan-artifacts --path "$AUDITOPS_V24_REVIEW_BUNDLE_DIR"

find "$AUDITOPS_V24_REVIEW_BUNDLE_DIR" -type f -exec chmod 0444 {} +
find "$AUDITOPS_V24_REVIEW_BUNDLE_DIR" -depth -type d -exec chmod 0555 {} +

echo "V24_REVIEW_BUNDLE=$AUDITOPS_V24_REVIEW_BUNDLE_DIR"
