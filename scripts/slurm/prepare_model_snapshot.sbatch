#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_HF_MODEL_ID:?Set AUDITOPS_HF_MODEL_ID}"
: "${AUDITOPS_HF_REVISION:?Set AUDITOPS_HF_REVISION to an immutable commit}"
: "${AUDITOPS_MODEL_SNAPSHOT_DIR:?Set AUDITOPS_MODEL_SNAPSHOT_DIR}"
auditops_require_git_commit AUDITOPS_HF_REVISION
auditops_assert_new_path "$AUDITOPS_MODEL_SNAPSHOT_DIR" "model snapshot"

auditops_start_log auditops-model
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"

if ! command -v hf >/dev/null 2>&1; then
  echo "The hf CLI is required; install the AuditOps models extra in the setup environment" >&2
  exit 2
fi
: "${AUDITOPS_MODEL_REGISTRY:?Set AUDITOPS_MODEL_REGISTRY}"
if [[ ! -f "$AUDITOPS_MODEL_REGISTRY" ]]; then
  echo "Pinned model registry not found: $AUDITOPS_MODEL_REGISTRY" >&2
  exit 2
fi

python -I -S - "$AUDITOPS_MODEL_REGISTRY" "$AUDITOPS_HF_MODEL_ID" "$AUDITOPS_HF_REVISION" <<'PY'
import json
import sys
from pathlib import Path

registry_path, model_id, revision = sys.argv[1:]
registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
matches = [row for row in registry.get("models", []) if row.get("model_id") == model_id]
if len(matches) != 1:
    raise SystemExit(f"Model is not uniquely registered: {model_id}")
if matches[0].get("revision") != revision:
    raise SystemExit(f"Revision does not match the pinned registry for {model_id}")
PY

mkdir -p \
  "$(dirname "$AUDITOPS_MODEL_SNAPSHOT_DIR")" \
  "$AUDITOPS_CACHE_ROOT/huggingface" \
  "$AUDITOPS_TMP_ROOT"
STAGING_DIR=$(mktemp -d "$AUDITOPS_TMP_ROOT/model-download.XXXXXXXX")
PUBLISH_STAGING_DIR=""
PUBLISH_MOVED=0
cleanup_model_staging() {
  local resolved_staging resolved_tmp_root resolved_publish resolved_publish_parent
  if [[ -e "$STAGING_DIR" ]]; then
    resolved_staging=$(realpath -m -- "$STAGING_DIR")
    resolved_tmp_root=$(realpath -m -- "$AUDITOPS_TMP_ROOT")
    case "$resolved_staging" in
      "$resolved_tmp_root"/model-download.*) rm -rf -- "$resolved_staging" ;;
      *) echo "Refusing to remove unexpected model download path: $resolved_staging" >&2 ;;
    esac
  fi
  if [[ "$PUBLISH_MOVED" == "0" && -n "$PUBLISH_STAGING_DIR" && -e "$PUBLISH_STAGING_DIR" ]]; then
    resolved_publish=$(realpath -m -- "$PUBLISH_STAGING_DIR")
    resolved_publish_parent=$(realpath -m -- "$(dirname "$AUDITOPS_MODEL_SNAPSHOT_DIR")")
    case "$resolved_publish" in
      "$resolved_publish_parent"/.auditops-model-publish.*)
        while IFS= read -r -d '' directory; do
          chmod 0700 -- "$directory"
        done < <(find "$resolved_publish" -depth -type d -print0)
        rm -rf -- "$resolved_publish"
        ;;
      *) echo "Refusing to remove unexpected model publish path: $resolved_publish" >&2 ;;
    esac
  fi
}
trap cleanup_model_staging EXIT

export HF_HUB_DISABLE_TELEMETRY=1
hf download "$AUDITOPS_HF_MODEL_ID" \
  --revision "$AUDITOPS_HF_REVISION" \
  --local-dir "$STAGING_DIR"

# `hf download --local-dir` creates resumable-transfer metadata that is not part
# of the model repository. Remove only that tool-owned staging subtree before
# enforcing exact remote membership and publishing the immutable snapshot.
HF_LOCAL_METADATA_DIR="$STAGING_DIR/.cache/huggingface"
if [[ -e "$HF_LOCAL_METADATA_DIR" || -L "$HF_LOCAL_METADATA_DIR" ]]; then
  rm -rf -- "$HF_LOCAL_METADATA_DIR"
fi
rmdir -- "$STAGING_DIR/.cache" 2>/dev/null || true

hf cache verify "$AUDITOPS_HF_MODEL_ID" \
  --revision "$AUDITOPS_HF_REVISION" \
  --local-dir "$STAGING_DIR" \
  --fail-on-missing-files \
  --fail-on-extra-files

python -I -S - "$AUDITOPS_MODEL_REGISTRY" "$AUDITOPS_HF_MODEL_ID" "$AUDITOPS_HF_REVISION" "$STAGING_DIR/AUDITOPS_SNAPSHOT.json" <<'PY'
import json
import sys
from pathlib import Path

registry_path, model_id, revision, output_path = sys.argv[1:]
registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
entry = next(row for row in registry["models"] if row["model_id"] == model_id)
payload = {
    "manifest_version": "auditops-model-snapshot.v1",
    "model_id": model_id,
    "revision": revision,
    "quantization": entry.get("quantization"),
}
Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

PUBLISH_STAGING_DIR=$(mktemp -d \
  "$(dirname "$AUDITOPS_MODEL_SNAPSHOT_DIR")/.auditops-model-publish.XXXXXXXX")
cp -a -- "$STAGING_DIR/." "$PUBLISH_STAGING_DIR/"

MODEL_SNAPSHOT_MANIFEST_SHA256=$(python -I -S - \
  "$AUDITOPS_REPO_ROOT" "$PUBLISH_STAGING_DIR" <<'MODEL_MANIFEST_WRITER_PY'
import sys

sys.path.insert(0, sys.argv[1])
from auditops.provenance import write_model_file_manifest

_, digest = write_model_file_manifest(sys.argv[2])
print(digest)
MODEL_MANIFEST_WRITER_PY
)
auditops_require_sha256 MODEL_SNAPSHOT_MANIFEST_SHA256

while IFS= read -r -d '' file; do
  chmod 0444 -- "$file"
done < <(find "$PUBLISH_STAGING_DIR" -type f -print0)
while IFS= read -r -d '' directory; do
  chmod 0555 -- "$directory"
done < <(find "$PUBLISH_STAGING_DIR" -depth -type d -print0)

python -I -S - "$AUDITOPS_REPO_ROOT" "$PUBLISH_STAGING_DIR" \
  "$MODEL_SNAPSHOT_MANIFEST_SHA256" <<'MODEL_MANIFEST_VERIFY_PY'
import sys

sys.path.insert(0, sys.argv[1])
from auditops.provenance import verify_model_snapshot

verify_model_snapshot(sys.argv[2], sys.argv[3], require_read_only=True)
MODEL_MANIFEST_VERIFY_PY

mv -T --no-clobber -- "$PUBLISH_STAGING_DIR" "$AUDITOPS_MODEL_SNAPSHOT_DIR"
if [[ -e "$PUBLISH_STAGING_DIR" ]]; then
  echo "Refusing to publish model snapshot because the destination appeared concurrently" >&2
  exit 2
fi
PUBLISH_MOVED=1
python -I -S - "$AUDITOPS_REPO_ROOT" "$AUDITOPS_MODEL_SNAPSHOT_DIR" \
  "$MODEL_SNAPSHOT_MANIFEST_SHA256" <<'MODEL_MANIFEST_VERIFY_PY'
import sys

sys.path.insert(0, sys.argv[1])
from auditops.provenance import verify_model_snapshot

verify_model_snapshot(sys.argv[2], sys.argv[3], require_read_only=True)
MODEL_MANIFEST_VERIFY_PY

echo "MODEL_SNAPSHOT=$AUDITOPS_MODEL_SNAPSHOT_DIR"
echo "MODEL_SNAPSHOT_MANIFEST_SHA256=$MODEL_SNAPSHOT_MANIFEST_SHA256"
