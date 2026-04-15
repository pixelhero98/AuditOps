#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export AUDITOPS_PROFILE="${AUDITOPS_PROFILE:-isambard3}"

source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"
