#!/bin/bash
#SBATCH --job-name=auditops-narrative-routing
#SBATCH --nodes=1
#SBATCH --time=00:20:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"

LOG_FILE="$AUDITOPS_LOG_ROOT/${SLURM_JOB_NAME:-auditops-narrative-routing}.${SLURM_JOB_ID:-manual}.out"
exec > >(tee -a "$LOG_FILE") 2>&1

source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

CORPUS_ROOT="${AUDITOPS_CORPUS_ROOT:?Set AUDITOPS_CORPUS_ROOT to the corpus root}"
DB_PATH="${AUDITOPS_DB_PATH:-$CORPUS_ROOT/db/corpus.sqlite}"
TASK_PATH="${AUDITOPS_NARRATIVE_TASK_PATH:?Set AUDITOPS_NARRATIVE_TASK_PATH to the narrative task JSONL path}"
OUTPUT_PATH="${AUDITOPS_NARRATIVE_ROUTING_RESULT_PATH:-$CORPUS_ROOT/eval/narrative_routing_summary.json}"
TOP_K="${AUDITOPS_NARRATIVE_TOP_K:-5}"
METHOD="${AUDITOPS_NARRATIVE_METHOD:-bm25_rerank}"
CANDIDATE_K="${AUDITOPS_NARRATIVE_CANDIDATE_K:-15}"
VARIANTS_PER_TASK="${AUDITOPS_NARRATIVE_VARIANTS_PER_TASK:-1}"

echo "HOST=$(hostname)"
echo "REPO=$AUDITOPS_REPO_ROOT"
echo "ENV=$AUDITOPS_ENV_ROOT"
echo "CORPUS_ROOT=$CORPUS_ROOT"
echo "DB_PATH=$DB_PATH"
echo "TASK_PATH=$TASK_PATH"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "TOP_K=$TOP_K"
echo "METHOD=$METHOD"
echo "CANDIDATE_K=$CANDIDATE_K"
echo "VARIANTS_PER_TASK=$VARIANTS_PER_TASK"
echo "LOG=$LOG_FILE"

python -m auditops.cli eval-narrative-routing \
  --db "$DB_PATH" \
  --tasks "$TASK_PATH" \
  --variants-per-task "$VARIANTS_PER_TASK" \
  --method "$METHOD" \
  --top-k "$TOP_K" \
  --candidate-k "$CANDIDATE_K" \
  --output "$OUTPUT_PATH"
