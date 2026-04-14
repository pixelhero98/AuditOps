#!/bin/bash
#SBATCH --job-name=auditops-setup-env
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00

set -euo pipefail

DEFAULT_REPO_ROOT="$HOME/work/AuditOps"
if [[ ! -d "$DEFAULT_REPO_ROOT" ]]; then
  DEFAULT_REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
fi
REPO_ROOT="${AUDITOPS_REPO_ROOT:-$DEFAULT_REPO_ROOT}"

bash "$REPO_ROOT/scripts/isambard_setup_env.sh"
