#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

for name in \
  AUDITOPS_V24_QUANT_JSONL AUDITOPS_V24_CORPUS_DB AUDITOPS_V24_RAW_FILING_ROOT \
  AUDITOPS_V24_R1_CANDIDATE_DIR AUDITOPS_V24_R1_REVIEW_HTML \
  AUDITOPS_V24_R1_PRIMARY_REVIEW AUDITOPS_V24_R1_SECONDARY_REVIEW \
  AUDITOPS_V24_R1_VALIDATION_SUMMARY AUDITOPS_V24R2_CANDIDATE_DIR \
  AUDITOPS_V24R2_REVIEW_HTML AUDITOPS_V24R2_REVIEW_BUNDLE_DIR \
  AUDITOPS_V24P_CRITIC_REVIEW AUDITOPS_V24P_PRIMARY_REVIEW \
  AUDITOPS_V24P_ACCEPTANCE AUDITOPS_V24P_BENCHMARK_DIR; do
  if [[ -z "${!name:-}" ]]; then
    echo "Set $name" >&2
    exit 2
  fi
done
AUDITOPS_V24R2_CORPUS_ID=${AUDITOPS_V24R2_CORPUS_ID:-us_sec_existing20_cached20_v2.4r2-candidate}

for source in \
  "$AUDITOPS_V24_QUANT_JSONL" "$AUDITOPS_V24_CORPUS_DB" \
  "$AUDITOPS_V24_R1_REVIEW_HTML" "$AUDITOPS_V24_R1_PRIMARY_REVIEW" \
  "$AUDITOPS_V24_R1_SECONDARY_REVIEW" "$AUDITOPS_V24_R1_VALIDATION_SUMMARY"; do
  if [[ ! -f "$source" || -L "$source" ]]; then
    echo "Required immutable input is not a regular file: $source" >&2
    exit 2
  fi
done
for source in "$AUDITOPS_V24_RAW_FILING_ROOT" "$AUDITOPS_V24_R1_CANDIDATE_DIR"; do
  if [[ ! -d "$source" || -L "$source" ]]; then
    echo "Required immutable input is not a directory: $source" >&2
    exit 2
  fi
done
for target in \
  "$AUDITOPS_V24R2_CANDIDATE_DIR" "$AUDITOPS_V24R2_REVIEW_HTML" \
  "$AUDITOPS_V24R2_REVIEW_BUNDLE_DIR" "$AUDITOPS_V24P_CRITIC_REVIEW" \
  "$AUDITOPS_V24P_PRIMARY_REVIEW" "$AUDITOPS_V24P_ACCEPTANCE" \
  "$AUDITOPS_V24P_BENCHMARK_DIR"; do
  auditops_assert_new_path "$target" "v2.4r2 output"
done

auditops_start_log auditops-v24r2-build
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"
R1_BEFORE=$(mktemp "$AUDITOPS_TMP_ROOT/v24r2-r1-before.XXXXXXXX")
R1_AFTER=$(mktemp "$AUDITOPS_TMP_ROOT/v24r2-r1-after.XXXXXXXX")
cleanup_hashes() { rm -f -- "$R1_BEFORE" "$R1_AFTER"; }
trap cleanup_hashes EXIT
find "$AUDITOPS_V24_R1_CANDIDATE_DIR" -type f -print0 | sort -z | xargs -0 sha256sum >"$R1_BEFORE"

python -m auditops.cli draft-agent-benchmark-v24r2 \
  --quant-jsonl "$AUDITOPS_V24_QUANT_JSONL" \
  --corpus-db "$AUDITOPS_V24_CORPUS_DB" \
  --raw-filing-root "$AUDITOPS_V24_RAW_FILING_ROOT" \
  --r1-candidate-dir "$AUDITOPS_V24_R1_CANDIDATE_DIR" \
  --r1-rendered-packet "$AUDITOPS_V24_R1_REVIEW_HTML" \
  --r1-primary-review "$AUDITOPS_V24_R1_PRIMARY_REVIEW" \
  --r1-secondary-review "$AUDITOPS_V24_R1_SECONDARY_REVIEW" \
  --r1-validation-summary "$AUDITOPS_V24_R1_VALIDATION_SUMMARY" \
  --corpus-id "$AUDITOPS_V24R2_CORPUS_ID" \
  --output-dir "$AUDITOPS_V24R2_CANDIDATE_DIR"
python -m auditops.cli render-agent-benchmark-review-v24 \
  --candidate-dir "$AUDITOPS_V24R2_CANDIDATE_DIR" \
  --output-html "$AUDITOPS_V24R2_REVIEW_HTML"
python -m auditops.cli prepare-agent-benchmark-review-bundle-v24 \
  --candidate-dir "$AUDITOPS_V24R2_CANDIDATE_DIR" \
  --rendered-packet "$AUDITOPS_V24R2_REVIEW_HTML" \
  --corpus-db "$AUDITOPS_V24_CORPUS_DB" \
  --raw-filing-root "$AUDITOPS_V24_RAW_FILING_ROOT" \
  --output-dir "$AUDITOPS_V24R2_REVIEW_BUNDLE_DIR"
python -m auditops.cli verify-agent-benchmark-candidate-v24 --candidate-dir "$AUDITOPS_V24R2_CANDIDATE_DIR"
python -m auditops.cli verify-agent-benchmark-review-bundle-v24 --bundle-dir "$AUDITOPS_V24R2_REVIEW_BUNDLE_DIR"

# The critic pass is intentionally created before the primary pass exists.
python -m auditops.cli record-provisional-ai-review-v24p \
  --candidate-dir "$AUDITOPS_V24R2_CANDIDATE_DIR" \
  --review-bundle-dir "$AUDITOPS_V24R2_REVIEW_BUNDLE_DIR" \
  --pass-kind CRITIC --output "$AUDITOPS_V24P_CRITIC_REVIEW"
python -m auditops.cli record-provisional-ai-review-v24p \
  --candidate-dir "$AUDITOPS_V24R2_CANDIDATE_DIR" \
  --review-bundle-dir "$AUDITOPS_V24R2_REVIEW_BUNDLE_DIR" \
  --pass-kind PRIMARY --output "$AUDITOPS_V24P_PRIMARY_REVIEW"
python -m auditops.cli accept-agent-benchmark-provisional-v24p \
  --candidate-dir "$AUDITOPS_V24R2_CANDIDATE_DIR" \
  --primary-review "$AUDITOPS_V24P_PRIMARY_REVIEW" \
  --critic-review "$AUDITOPS_V24P_CRITIC_REVIEW" \
  --output "$AUDITOPS_V24P_ACCEPTANCE"
python -m auditops.cli freeze-agent-benchmark-provisional-v24p \
  --candidate-dir "$AUDITOPS_V24R2_CANDIDATE_DIR" \
  --acceptance "$AUDITOPS_V24P_ACCEPTANCE" \
  --output-dir "$AUDITOPS_V24P_BENCHMARK_DIR"
python -m auditops.cli verify-agent-benchmark-v24p --benchmark-dir "$AUDITOPS_V24P_BENCHMARK_DIR"
python -m auditops.cli scan-artifacts \
  --path "$AUDITOPS_V24R2_CANDIDATE_DIR" \
  --path "$AUDITOPS_V24R2_REVIEW_BUNDLE_DIR" \
  --path "$AUDITOPS_V24P_BENCHMARK_DIR"

find "$AUDITOPS_V24_R1_CANDIDATE_DIR" -type f -print0 | sort -z | xargs -0 sha256sum >"$R1_AFTER"
cmp -s -- "$R1_BEFORE" "$R1_AFTER" || { echo "r1 candidate was mutated" >&2; exit 2; }
for target in "$AUDITOPS_V24R2_CANDIDATE_DIR" "$AUDITOPS_V24R2_REVIEW_BUNDLE_DIR" "$AUDITOPS_V24P_BENCHMARK_DIR"; do
  find "$target" -type f -exec chmod 0444 {} +
  find "$target" -depth -type d -exec chmod 0555 {} +
done
chmod 0444 "$AUDITOPS_V24R2_REVIEW_HTML" "$AUDITOPS_V24P_CRITIC_REVIEW" \
  "$AUDITOPS_V24P_PRIMARY_REVIEW" "$AUDITOPS_V24P_ACCEPTANCE"

echo "V24R2_CANDIDATE=$AUDITOPS_V24R2_CANDIDATE_DIR"
echo "V24P_BENCHMARK=$AUDITOPS_V24P_BENCHMARK_DIR"
