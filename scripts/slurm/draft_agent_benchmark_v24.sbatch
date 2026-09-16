#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_V24_QUANT_JSONL:?Set AUDITOPS_V24_QUANT_JSONL}"
: "${AUDITOPS_V24_CORPUS_DB:?Set AUDITOPS_V24_CORPUS_DB}"
: "${AUDITOPS_V24_RAW_FILING_ROOT:?Set AUDITOPS_V24_RAW_FILING_ROOT}"
: "${AUDITOPS_V24_CANDIDATE_DIR:?Set AUDITOPS_V24_CANDIDATE_DIR}"
: "${AUDITOPS_V24_REVIEW_HTML:?Set AUDITOPS_V24_REVIEW_HTML}"
AUDITOPS_V24_CORPUS_ID=${AUDITOPS_V24_CORPUS_ID:-us_sec_existing20_cached20_v2.4-candidate}

for source in "$AUDITOPS_V24_QUANT_JSONL" "$AUDITOPS_V24_CORPUS_DB"; do
  if [[ ! -f "$source" || -L "$source" ]]; then
    echo "Required v2.4 source is not a regular file: $source" >&2
    exit 2
  fi
done
if [[ ! -d "$AUDITOPS_V24_RAW_FILING_ROOT" || -L "$AUDITOPS_V24_RAW_FILING_ROOT" ]]; then
  echo "Required frozen raw filing root is not a non-symlink directory: $AUDITOPS_V24_RAW_FILING_ROOT" >&2
  exit 2
fi
auditops_assert_new_path "$AUDITOPS_V24_CANDIDATE_DIR" "v2.4 candidate"
auditops_assert_new_path "$AUDITOPS_V24_REVIEW_HTML" "v2.4 review HTML"

auditops_start_log auditops-v24-draft
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

python -m auditops.cli draft-agent-benchmark-v24 \
  --quant-jsonl "$AUDITOPS_V24_QUANT_JSONL" \
  --corpus-db "$AUDITOPS_V24_CORPUS_DB" \
  --raw-filing-root "$AUDITOPS_V24_RAW_FILING_ROOT" \
  --corpus-id "$AUDITOPS_V24_CORPUS_ID" \
  --output-dir "$AUDITOPS_V24_CANDIDATE_DIR"
python -m auditops.cli render-agent-benchmark-review-v24 \
  --candidate-dir "$AUDITOPS_V24_CANDIDATE_DIR" \
  --output-html "$AUDITOPS_V24_REVIEW_HTML"
python -m auditops.cli verify-agent-benchmark-candidate-v24 \
  --candidate-dir "$AUDITOPS_V24_CANDIDATE_DIR"

find "$AUDITOPS_V24_CANDIDATE_DIR" -type f -exec chmod 0444 {} +
find "$AUDITOPS_V24_CANDIDATE_DIR" -depth -type d -exec chmod 0555 {} +
chmod 0444 "$AUDITOPS_V24_REVIEW_HTML"

echo "V24_CANDIDATE=$AUDITOPS_V24_CANDIDATE_DIR"
echo "V24_REVIEW_HTML=$AUDITOPS_V24_REVIEW_HTML"
