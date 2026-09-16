#!/bin/bash
#SBATCH --job-name=auditops-narrative-eval
#SBATCH --nodes=1
#SBATCH --time=00:20:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"

LOG_FILE="$AUDITOPS_LOG_ROOT/${SLURM_JOB_NAME:-auditops-narrative-eval}.${SLURM_JOB_ID:-manual}.out"
exec > >(tee -a "$LOG_FILE") 2>&1

source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

CORPUS_ROOT="${AUDITOPS_CORPUS_ROOT:?Set AUDITOPS_CORPUS_ROOT to the corpus root}"
DB_PATH="${AUDITOPS_DB_PATH:-$CORPUS_ROOT/db/corpus.sqlite}"
TASK_PATH="${AUDITOPS_NARRATIVE_TASK_PATH:-$CORPUS_ROOT/eval/narrative_benchmark.jsonl}"
RESULT_PATH="${AUDITOPS_NARRATIVE_RESULT_PATH:-$CORPUS_ROOT/eval/narrative_citation_summary.json}"
ANSWER_RESULT_PATH="${AUDITOPS_NARRATIVE_ANSWER_RESULT_PATH:-$CORPUS_ROOT/eval/narrative_answer_summary.json}"
TOP_K="${AUDITOPS_NARRATIVE_TOP_K:-5}"
METHOD="${AUDITOPS_NARRATIVE_METHOD:-bm25_rerank}"
CANDIDATE_K="${AUDITOPS_NARRATIVE_CANDIDATE_K:-15}"
LIMIT="${AUDITOPS_NARRATIVE_LIMIT:-200}"
PER_FILING_LIMIT="${AUDITOPS_NARRATIVE_PER_FILING_LIMIT:-1}"
INCLUDE_UNANSWERABLE="${AUDITOPS_NARRATIVE_INCLUDE_UNANSWERABLE:-1}"
UNANSWERABLE_LIMIT="${AUDITOPS_NARRATIVE_UNANSWERABLE_LIMIT:-100}"

echo "HOST=$(hostname)"
echo "REPO=$AUDITOPS_REPO_ROOT"
echo "ENV=$AUDITOPS_ENV_ROOT"
echo "CORPUS_ROOT=$CORPUS_ROOT"
echo "DB_PATH=$DB_PATH"
echo "TASK_PATH=$TASK_PATH"
echo "RESULT_PATH=$RESULT_PATH"
echo "ANSWER_RESULT_PATH=$ANSWER_RESULT_PATH"
echo "TOP_K=$TOP_K"
echo "METHOD=$METHOD"
echo "CANDIDATE_K=$CANDIDATE_K"
echo "LIMIT=$LIMIT"
echo "PER_FILING_LIMIT=$PER_FILING_LIMIT"
echo "INCLUDE_UNANSWERABLE=$INCLUDE_UNANSWERABLE"
echo "UNANSWERABLE_LIMIT=$UNANSWERABLE_LIMIT"
echo "LOG=$LOG_FILE"

BUILD_ARGS=(
  --db "$DB_PATH"
  --output "$TASK_PATH"
  --limit "$LIMIT"
  --per-filing-limit "$PER_FILING_LIMIT"
)
if [[ "$INCLUDE_UNANSWERABLE" == "1" ]]; then
  BUILD_ARGS+=(--include-unanswerable --unanswerable-limit "$UNANSWERABLE_LIMIT")
fi

python -m auditops.cli build-narrative-benchmark "${BUILD_ARGS[@]}"

python -m auditops.cli eval-narrative-citations \
  --db "$DB_PATH" \
  --tasks "$TASK_PATH" \
  --method "$METHOD" \
  --top-k "$TOP_K" \
  --candidate-k "$CANDIDATE_K" \
  --output "$RESULT_PATH"

python -m auditops.cli eval-narrative-answers \
  --db "$DB_PATH" \
  --tasks "$TASK_PATH" \
  --method "$METHOD" \
  --top-k "$TOP_K" \
  --candidate-k "$CANDIDATE_K" \
  --output "$ANSWER_RESULT_PATH"
