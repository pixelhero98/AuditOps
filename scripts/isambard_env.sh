#!/bin/bash
set -euo pipefail

source /opt/cray/pe/lmod/lmod/init/bash
module load cray-python/3.11.7

export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$HOME/work/AuditOps}"
export AUDITOPS_PROJECT_ROOT="${AUDITOPS_PROJECT_ROOT:-/projects/b35z/AuditOps}"
export AUDITOPS_ENV_ROOT="${AUDITOPS_ENV_ROOT:-$AUDITOPS_PROJECT_ROOT/envs/auditops-dev-py311}"
export AUDITOPS_SCRATCH_ROOT="${AUDITOPS_SCRATCH_ROOT:-${SCRATCHDIR:-${SCRATCH:-$HOME}}/AuditOps}"

mkdir -p \
  "$AUDITOPS_PROJECT_ROOT/data" \
  "$AUDITOPS_PROJECT_ROOT/artifacts" \
  "$AUDITOPS_PROJECT_ROOT/envs" \
  "$AUDITOPS_SCRATCH_ROOT/cache/pip" \
  "$AUDITOPS_SCRATCH_ROOT/cache/huggingface" \
  "$AUDITOPS_SCRATCH_ROOT/cache/transformers" \
  "$AUDITOPS_SCRATCH_ROOT/cache/torch" \
  "$AUDITOPS_SCRATCH_ROOT/logs" \
  "$AUDITOPS_SCRATCH_ROOT/runs" \
  "$AUDITOPS_SCRATCH_ROOT/tmp"

export PIP_CACHE_DIR="$AUDITOPS_SCRATCH_ROOT/cache/pip"
export HF_HOME="$AUDITOPS_SCRATCH_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$AUDITOPS_SCRATCH_ROOT/cache/transformers"
export TORCH_HOME="$AUDITOPS_SCRATCH_ROOT/cache/torch"
export TMPDIR="$AUDITOPS_SCRATCH_ROOT/tmp"
