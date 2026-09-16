#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_US_CONSTITUENTS_OUTPUT:?Set AUDITOPS_US_CONSTITUENTS_OUTPUT}"
: "${AUDITOPS_SNAPSHOT_DATE:?Set AUDITOPS_SNAPSHOT_DATE to YYYY-MM-DD}"
: "${AUDITOPS_SEC_USER_AGENT:?Set a contact-bearing AUDITOPS_SEC_USER_AGENT}"

auditops_assert_new_path "$AUDITOPS_US_CONSTITUENTS_OUTPUT" "US constituent snapshot"
auditops_assert_new_path "$AUDITOPS_US_CONSTITUENTS_OUTPUT.source.json" "US constituent source manifest"
auditops_assert_new_path "$AUDITOPS_US_CONSTITUENTS_OUTPUT.source.html" "US constituent raw source"

auditops_start_log auditops-fetch-us-constituents
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

# The contact-bearing identity is read only from the environment.  It is sent
# in request headers and represented in durable artifacts only by SHA-256.
python -m auditops.cli fetch-us-constituents \
  --output "$AUDITOPS_US_CONSTITUENTS_OUTPUT" \
  --snapshot-date "$AUDITOPS_SNAPSHOT_DATE"

python -m auditops.cli scan-artifacts \
  --path "$AUDITOPS_US_CONSTITUENTS_OUTPUT" \
  --path "$AUDITOPS_US_CONSTITUENTS_OUTPUT.source.json" \
  --path "$AUDITOPS_US_CONSTITUENTS_OUTPUT.source.html"
