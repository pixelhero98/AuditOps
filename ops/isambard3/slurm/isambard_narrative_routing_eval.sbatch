#!/bin/bash
#SBATCH --job-name=auditops-narrative-routing
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export AUDITOPS_PROFILE="${AUDITOPS_PROFILE:-isambard3}"

bash "$AUDITOPS_REPO_ROOT/scripts/slurm/narrative_routing_eval.sbatch"
