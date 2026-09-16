#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_VLLM_SIF:?Set AUDITOPS_VLLM_SIF}"
: "${AUDITOPS_VLLM_OCI_REF:=docker://vllm/vllm-openai:v0.26.0-ubuntu2404@sha256:ef7bfc14df9233e3e5d41e733e3be0afa6abbe5ae5f14ee0758110030f6cd53e}"
: "${AUDITOPS_TARGET_ARCH:=amd64}"
if [[ "$AUDITOPS_TARGET_ARCH" != "arm64" && "$AUDITOPS_TARGET_ARCH" != "amd64" ]]; then
  echo "AUDITOPS_TARGET_ARCH must be amd64 or arm64" >&2
  exit 2
fi
PINNED_OCI_DIGEST='sha256:ef7bfc14df9233e3e5d41e733e3be0afa6abbe5ae5f14ee0758110030f6cd53e'
PINNED_OCI_REF="docker://vllm/vllm-openai:v0.26.0-ubuntu2404@${PINNED_OCI_DIGEST}"
if [[ "$AUDITOPS_VLLM_OCI_REF" != "$PINNED_OCI_REF" ]]; then
  echo "AUDITOPS_VLLM_OCI_REF must match the checked-in multi-architecture vLLM 0.26.0 OCI index digest" >&2
  exit 2
fi
# Apptainer 1.4 rejects Docker references that contain both a tag and a digest.
# Keep the tagged form above for provenance, but resolve the identical immutable
# index through the digest-only form accepted by Apptainer.
APPTAINER_PULL_REF="docker://vllm/vllm-openai@${PINNED_OCI_DIGEST}"
auditops_assert_new_path "$AUDITOPS_VLLM_SIF" "container image"
AUDITOPS_VLLM_SIF_SHA256_FILE="${AUDITOPS_VLLM_SIF_SHA256_FILE:-$AUDITOPS_VLLM_SIF.sha256}"
auditops_assert_new_path "$AUDITOPS_VLLM_SIF_SHA256_FILE" "container checksum"
auditops_assert_new_path "$AUDITOPS_VLLM_SIF.oci-ref" "container source reference"
auditops_assert_new_path "$AUDITOPS_VLLM_SIF.pull-ref" "container pull reference"
auditops_assert_new_path "$AUDITOPS_VLLM_SIF.inspect.json" "container inspection record"
auditops_assert_new_path "$AUDITOPS_VLLM_SIF.sif-list.txt" "container SIF descriptor record"
auditops_assert_new_path "$AUDITOPS_VLLM_SIF.expected-vllm-version.txt" "container expected vLLM version record"
auditops_assert_new_path "$AUDITOPS_VLLM_SIF.target-architecture.txt" "container architecture record"

auditops_start_log auditops-vllm
if ! command -v apptainer >/dev/null 2>&1; then
  echo "Apptainer is required to prepare the vLLM image" >&2
  exit 2
fi

mkdir -p "$(dirname "$AUDITOPS_VLLM_SIF")" "$AUDITOPS_TMP_ROOT" "$AUDITOPS_CACHE_ROOT/apptainer"
STAGING_DIR=$(mktemp -d "$AUDITOPS_TMP_ROOT/vllm-container.XXXXXXXX")
STAGING_SIF="$STAGING_DIR/vllm-0.26.0.sif"
APPTAINER_CACHEDIR="$AUDITOPS_CACHE_ROOT/apptainer" \
APPTAINER_TMPDIR="$AUDITOPS_TMP_ROOT" \
apptainer pull --arch "$AUDITOPS_TARGET_ARCH" "$STAGING_SIF" "$APPTAINER_PULL_REF"

SIF_SHA256=$(sha256sum "$STAGING_SIF" | awk '{print $1}')
apptainer inspect --json "$STAGING_SIF" > "$STAGING_DIR/inspect.json"
apptainer sif list "$STAGING_SIF" > "$STAGING_DIR/sif-list.txt"
if [[ "$AUDITOPS_TARGET_ARCH" == "arm64" ]]; then
  ARCH_PATTERN='arm64|aarch64'
else
  ARCH_PATTERN='amd64|x86_64'
fi
if ! grep -Eiq "$ARCH_PATTERN" "$STAGING_DIR/sif-list.txt"; then
  echo "Prepared SIF does not advertise the required $AUDITOPS_TARGET_ARCH architecture" >&2
  exit 2
fi
printf '%s\n' "$AUDITOPS_VLLM_OCI_REF" > "$STAGING_DIR/oci-ref"
printf '%s\n' "$APPTAINER_PULL_REF" > "$STAGING_DIR/pull-ref"
printf '%s\n' '0.26.0' > "$STAGING_DIR/expected-vllm-version.txt"
printf '%s\n' "$AUDITOPS_TARGET_ARCH" > "$STAGING_DIR/target-architecture.txt"
printf '%s  %s\n' "$SIF_SHA256" "$(basename "$AUDITOPS_VLLM_SIF")" > "$STAGING_DIR/image.sha256"
mv -- "$STAGING_SIF" "$AUDITOPS_VLLM_SIF"
if [[ "$(sha256sum "$AUDITOPS_VLLM_SIF" | awk '{print $1}')" != "$SIF_SHA256" ]]; then
  echo "Final container copy failed checksum verification" >&2
  exit 2
fi
mv -- "$STAGING_DIR/inspect.json" "$AUDITOPS_VLLM_SIF.inspect.json"
mv -- "$STAGING_DIR/sif-list.txt" "$AUDITOPS_VLLM_SIF.sif-list.txt"
mv -- "$STAGING_DIR/oci-ref" "$AUDITOPS_VLLM_SIF.oci-ref"
mv -- "$STAGING_DIR/pull-ref" "$AUDITOPS_VLLM_SIF.pull-ref"
mv -- "$STAGING_DIR/expected-vllm-version.txt" "$AUDITOPS_VLLM_SIF.expected-vllm-version.txt"
mv -- "$STAGING_DIR/target-architecture.txt" "$AUDITOPS_VLLM_SIF.target-architecture.txt"
mv -- "$STAGING_DIR/image.sha256" "$AUDITOPS_VLLM_SIF_SHA256_FILE"
rmdir -- "$STAGING_DIR"
chmod 0444 \
  "$AUDITOPS_VLLM_SIF" \
  "$AUDITOPS_VLLM_SIF_SHA256_FILE" \
  "$AUDITOPS_VLLM_SIF.oci-ref" \
  "$AUDITOPS_VLLM_SIF.pull-ref" \
  "$AUDITOPS_VLLM_SIF.inspect.json" \
  "$AUDITOPS_VLLM_SIF.sif-list.txt" \
  "$AUDITOPS_VLLM_SIF.expected-vllm-version.txt" \
  "$AUDITOPS_VLLM_SIF.target-architecture.txt"

echo "VLLM_IMAGE=$AUDITOPS_VLLM_SIF"
echo "VLLM_IMAGE_SHA256=$SIF_SHA256"
