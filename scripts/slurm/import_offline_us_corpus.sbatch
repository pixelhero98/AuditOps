#!/bin/bash
#SBATCH --job-name=auditops-offline-us
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_OFFLINE_US_SOURCE_ROOT:?Set AUDITOPS_OFFLINE_US_SOURCE_ROOT to the staged, read-only US filing collection}"
: "${AUDITOPS_CORPUS_ROOT:?Set AUDITOPS_CORPUS_ROOT to a new corpus root}"
: "${AUDITOPS_SNAPSHOT_DATE:?Set AUDITOPS_SNAPSHOT_DATE to YYYY-MM-DD}"
: "${AUDITOPS_CORPUS_ID:?Set AUDITOPS_CORPUS_ID to the immutable offline corpus identifier}"

for variable_name in \
  HF_TOKEN \
  HUGGING_FACE_HUB_TOKEN \
  COMPANIES_HOUSE_API_KEY \
  AWS_SECRET_ACCESS_KEY \
  AUDITOPS_SEC_USER_AGENT; do
  if [[ -n "${!variable_name:-}" ]]; then
    echo "$variable_name must be unset for the credential-free offline US import" >&2
    exit 2
  fi
done
unset variable_name

if [[ ! "$AUDITOPS_SNAPSHOT_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "AUDITOPS_SNAPSHOT_DATE must use YYYY-MM-DD" >&2
  exit 2
fi
if [[ ! "$AUDITOPS_CORPUS_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "AUDITOPS_CORPUS_ID must contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi
for path_name in AUDITOPS_OFFLINE_US_SOURCE_ROOT AUDITOPS_CORPUS_ROOT; do
  path_value=${!path_name}
  if [[ "$path_value" != /* || "$path_value" == *$'\n'* || "$path_value" == *$'\r'* ]]; then
    echo "$path_name must be an absolute path without control characters" >&2
    exit 2
  fi
done
unset path_name path_value

if ! SOURCE_ROOT=$(realpath -e -- "$AUDITOPS_OFFLINE_US_SOURCE_ROOT"); then
  echo "Offline US source root does not exist" >&2
  exit 2
fi
if [[ ! -d "$SOURCE_ROOT" || -L "$AUDITOPS_OFFLINE_US_SOURCE_ROOT" ]]; then
  echo "Offline US source root must be a real directory, not a symlink" >&2
  exit 2
fi
CORPUS_ROOT=$(realpath -m -- "$AUDITOPS_CORPUS_ROOT")
case "$CORPUS_ROOT/" in
  "$SOURCE_ROOT"/*)
    echo "Corpus output may not be inside the offline source root" >&2
    exit 2
    ;;
esac
case "$SOURCE_ROOT/" in
  "$CORPUS_ROOT"/*)
    echo "Offline source root may not be inside the corpus output" >&2
    exit 2
    ;;
esac

auditops_assert_new_path "$CORPUS_ROOT" "offline US corpus output"

export AUDITOPS_DISABLE_NETWORK=1
auditops_start_log auditops-offline-us
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

echo "HOST=$(hostname)"
echo "REPO=$AUDITOPS_REPO_ROOT"
echo "ENV=$AUDITOPS_ENV_ROOT"
echo "SOURCE_ROOT=$SOURCE_ROOT"
echo "CORPUS_ROOT=$CORPUS_ROOT"
echo "SNAPSHOT_DATE=$AUDITOPS_SNAPSHOT_DATE"
echo "CORPUS_ID=$AUDITOPS_CORPUS_ID"
echo "NETWORK_DISABLED=$AUDITOPS_DISABLE_NETWORK"
echo "LOG=$AUDITOPS_ACTIVE_LOG"

# Scan before copying so a contaminated source never enters the corpus tree.
python -m auditops.cli scan-artifacts --path "$SOURCE_ROOT"

python -m auditops.cli import-offline-us-corpus \
  --source-root "$SOURCE_ROOT" \
  --corpus-root "$CORPUS_ROOT" \
  --snapshot-date "$AUDITOPS_SNAPSHOT_DATE" \
  --corpus-id "$AUDITOPS_CORPUS_ID"

# Verify the imported, provenance-bearing corpus before derived data is created.
python -m auditops.cli scan-artifacts --path "$CORPUS_ROOT"
python -m auditops.cli ingest-corpus --corpus-root "$CORPUS_ROOT"
python -m auditops.cli generate-corpus-datasets --corpus-root "$CORPUS_ROOT"

# Include the database and generated datasets in the release secret scan.
python -m auditops.cli scan-artifacts --path "$CORPUS_ROOT"
