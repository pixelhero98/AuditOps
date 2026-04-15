#!/bin/bash
#SBATCH --job-name=auditops-retrieval-eval
#SBATCH --nodes=1
#SBATCH --time=00:45:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"

LOG_FILE="$AUDITOPS_LOG_ROOT/${SLURM_JOB_NAME:-auditops-retrieval-eval}.${SLURM_JOB_ID:-manual}.out"
exec > >(tee -a "$LOG_FILE") 2>&1

source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

CORPUS_ROOT="${AUDITOPS_CORPUS_ROOT:?Set AUDITOPS_CORPUS_ROOT to the corpus root}"
DB_PATH="${AUDITOPS_DB_PATH:-$CORPUS_ROOT/db/corpus.sqlite}"
BENCHMARK_PATH="${AUDITOPS_RETRIEVAL_BENCHMARK_PATH:-$CORPUS_ROOT/eval/retrieval_benchmark_examples.jsonl}"
RESULT_PATH="${AUDITOPS_RETRIEVAL_RESULT_PATH:-$CORPUS_ROOT/eval/retrieval_benchmark_summary.json}"
BENCHMARK_LIMIT="${AUDITOPS_RETRIEVAL_LIMIT:-250}"
PER_FILING_LIMIT="${AUDITOPS_RETRIEVAL_PER_FILING_LIMIT:-1}"
TOP_K="${AUDITOPS_RETRIEVAL_TOP_K:-5}"
METHOD="${AUDITOPS_RETRIEVAL_METHOD:-bm25_rerank}"
CANDIDATE_K="${AUDITOPS_RETRIEVAL_CANDIDATE_K:-15}"

echo "HOST=$(hostname)"
echo "REPO=$AUDITOPS_REPO_ROOT"
echo "ENV=$AUDITOPS_ENV_ROOT"
echo "CORPUS_ROOT=$CORPUS_ROOT"
echo "DB_PATH=$DB_PATH"
echo "BENCHMARK_PATH=$BENCHMARK_PATH"
echo "RESULT_PATH=$RESULT_PATH"
echo "BENCHMARK_LIMIT=$BENCHMARK_LIMIT"
echo "PER_FILING_LIMIT=$PER_FILING_LIMIT"
echo "TOP_K=$TOP_K"
echo "METHOD=$METHOD"
echo "CANDIDATE_K=$CANDIDATE_K"
echo "LOG=$LOG_FILE"

auditops build-retrieval-benchmark \
  --db "$DB_PATH" \
  --output "$BENCHMARK_PATH" \
  --limit "$BENCHMARK_LIMIT" \
  --per-filing-limit "$PER_FILING_LIMIT"

auditops eval-retrieval \
  --db "$DB_PATH" \
  --examples "$BENCHMARK_PATH" \
  --method "$METHOD" \
  --top-k "$TOP_K" \
  --candidate-k "$CANDIDATE_K" \
  --output "$RESULT_PATH"
