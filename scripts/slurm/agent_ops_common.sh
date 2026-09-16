#!/bin/bash

# Shared fail-closed helpers for the v0.1 benchmark operations.  Callers must
# enable `set -euo pipefail` before sourcing this file.

auditops_start_log() {
  local default_name=$1
  mkdir -p "$AUDITOPS_LOG_ROOT"
  AUDITOPS_ACTIVE_LOG="$AUDITOPS_LOG_ROOT/${SLURM_JOB_NAME:-$default_name}.${SLURM_JOB_ID:-manual}.out"
  export AUDITOPS_ACTIVE_LOG
  exec > >(tee -a "$AUDITOPS_ACTIVE_LOG") 2>&1
}

auditops_require_sha256() {
  local variable_name=$1
  local value=${!variable_name:-}
  if [[ ! "$value" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "$variable_name must be a 64-character SHA-256 value" >&2
    exit 2
  fi
}

auditops_require_git_commit() {
  local variable_name=$1
  local value=${!variable_name:-}
  if [[ ! "$value" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "$variable_name must be an immutable 40-character Git commit" >&2
    exit 2
  fi
}

auditops_assert_new_path() {
  local target=$1
  local label=$2
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite $label: $target" >&2
    exit 2
  fi
}

auditops_assert_under_root() {
  local target=$1
  local root=$2
  local label=$3
  local resolved_target resolved_root
  resolved_target=$(realpath -m -- "$target")
  resolved_root=$(realpath -m -- "$root")
  case "$resolved_target/" in
    "$resolved_root"/*) ;;
    *)
      echo "$label must be under $resolved_root, got $resolved_target" >&2
      exit 2
      ;;
  esac
}
