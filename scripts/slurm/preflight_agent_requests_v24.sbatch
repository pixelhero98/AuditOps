#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_BENCHMARK_DIR:?Set AUDITOPS_BENCHMARK_DIR}"
: "${AUDITOPS_MODEL_CONFIG:?Set AUDITOPS_MODEL_CONFIG}"
: "${AUDITOPS_V24_PREFLIGHT_DIR:?Set AUDITOPS_V24_PREFLIGHT_DIR}"
: "${AUDITOPS_VLLM_SIF:?Set AUDITOPS_VLLM_SIF to the pinned Apptainer image}"
: "${AUDITOPS_VLLM_SIF_SHA256:?Set AUDITOPS_VLLM_SIF_SHA256 to an external digest pin}"
: "${AUDITOPS_MODEL_CONFIG_SHA256:?Set AUDITOPS_MODEL_CONFIG_SHA256 to an external digest pin}"
: "${AUDITOPS_SOURCE_MANIFEST:?Set AUDITOPS_SOURCE_MANIFEST}"
: "${AUDITOPS_SOURCE_MANIFEST_SHA256:?Set AUDITOPS_SOURCE_MANIFEST_SHA256 to an external digest pin}"
: "${AUDITOPS_PACKAGE_LOCK_FILE:?Set AUDITOPS_PACKAGE_LOCK_FILE}"
auditops_assert_new_path "$AUDITOPS_V24_PREFLIGHT_DIR" "v2.4 preflight output"
auditops_start_log auditops-v24-preflight
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"
auditops_require_sha256 AUDITOPS_VLLM_SIF_SHA256
auditops_require_sha256 AUDITOPS_MODEL_CONFIG_SHA256
auditops_require_sha256 AUDITOPS_SOURCE_MANIFEST_SHA256
if [[ ! -f "$AUDITOPS_VLLM_SIF" || -L "$AUDITOPS_VLLM_SIF" ]]; then
  echo "Pinned container is not a regular file: $AUDITOPS_VLLM_SIF" >&2
  exit 2
fi
ACTUAL_SIF_SHA256=$(sha256sum "$AUDITOPS_VLLM_SIF" | awk '{print $1}')
if [[ "$ACTUAL_SIF_SHA256" != "$AUDITOPS_VLLM_SIF_SHA256" ]]; then
  echo "Pinned container checksum mismatch" >&2
  exit 2
fi
ACTUAL_MODEL_CONFIG_SHA256=$(sha256sum "$AUDITOPS_MODEL_CONFIG" | awk '{print $1}')
if [[ "$ACTUAL_MODEL_CONFIG_SHA256" != "$AUDITOPS_MODEL_CONFIG_SHA256" ]]; then
  echo "Pinned model-config checksum mismatch" >&2
  exit 2
fi
if [[ ! -f "$AUDITOPS_PACKAGE_LOCK_FILE" || -L "$AUDITOPS_PACKAGE_LOCK_FILE" ]]; then
  echo "Pinned package lock is not a regular file" >&2
  exit 2
fi
AUDITOPS_PACKAGE_LOCK_SHA256=$(sha256sum "$AUDITOPS_PACKAGE_LOCK_FILE" | awk '{print $1}')
auditops_require_sha256 AUDITOPS_PACKAGE_LOCK_SHA256
AUDITOPS_SOURCE_TREE_SHA256=$(python - "$AUDITOPS_REPO_ROOT" "$AUDITOPS_SOURCE_MANIFEST" "$AUDITOPS_SOURCE_MANIFEST_SHA256" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
manifest_path = pathlib.Path(sys.argv[2])
expected_manifest_sha256 = sys.argv[3]
payload = manifest_path.read_bytes()
if hashlib.sha256(payload).hexdigest() != expected_manifest_sha256:
    raise SystemExit("Pinned source-manifest checksum mismatch")
manifest = json.loads(payload)
rows = manifest.get("files")
if not isinstance(rows, list) or manifest.get("file_count") != len(rows):
    raise SystemExit("Source manifest file table is invalid")
actual_rows = []
for row in rows:
    relative = pathlib.PurePosixPath(str(row["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit("Source manifest contains an unsafe path")
    path = root.joinpath(*relative.parts)
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"Source snapshot entry is invalid: {relative}")
    data = path.read_bytes()
    actual = {
        "path": relative.as_posix(),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    if actual != row:
        raise SystemExit(f"Source snapshot checksum mismatch: {relative}")
    actual_rows.append(actual)
tree_payload = json.dumps(
    actual_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
).encode("utf-8")
tree_sha256 = hashlib.sha256(tree_payload).hexdigest()
if tree_sha256 != manifest.get("source_tree_sha256"):
    raise SystemExit("Source tree digest does not match the pinned manifest")
print(tree_sha256)
PY
)
auditops_require_sha256 AUDITOPS_SOURCE_TREE_SHA256
MODEL_PATH=$(python - "$AUDITOPS_MODEL_CONFIG" <<'PY'
import json
import pathlib
import sys

config = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
path = pathlib.Path(str(config["model_path"]))
if not path.is_absolute():
    raise SystemExit("model_path must be absolute")
print(path)
PY
)
if [[ ! -d "$MODEL_PATH" || -L "$MODEL_PATH" ]]; then
  echo "Pinned model snapshot is not a regular directory: $MODEL_PATH" >&2
  exit 2
fi

CONTAINER_RUNTIME_BIN=$(command -v apptainer)
if [[ ! -x "$CONTAINER_RUNTIME_BIN" ]]; then
  echo "Apptainer executable is unavailable" >&2
  exit 2
fi
RUNTIME_TMP=$(mktemp -d "$AUDITOPS_TMP_ROOT/v24-preflight.XXXXXXXX")
cleanup_runtime_tmp() {
  local resolved_tmp resolved_root
  resolved_tmp=$(realpath -m -- "$RUNTIME_TMP")
  resolved_root=$(realpath -m -- "$AUDITOPS_TMP_ROOT")
  case "$resolved_tmp" in
    "$resolved_root"/v24-preflight.*) rm -rf -- "$resolved_tmp" ;;
    *) echo "Refusing to remove unexpected preflight temp path: $resolved_tmp" >&2 ;;
  esac
}
trap cleanup_runtime_tmp EXIT
mkdir -p "$RUNTIME_TMP/home" "$RUNTIME_TMP/tmp"
chmod 0700 "$RUNTIME_TMP/home" "$RUNTIME_TMP/tmp"

CONTAINER_ARGS=(--cleanenv --containall --net --network none)
if "$CONTAINER_RUNTIME_BIN" exec --help | grep -q -- '--no-mount'; then
  CONTAINER_ARGS+=(--no-mount bind-paths,hostfs,cwd)
fi
env -i \
  HOME="$RUNTIME_TMP/home" \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 TMPDIR="$RUNTIME_TMP/tmp" \
  "$CONTAINER_RUNTIME_BIN" exec \
  "${CONTAINER_ARGS[@]}" \
  --bind "$AUDITOPS_REPO_ROOT:/workspace/AuditOps:ro" \
  --bind "$AUDITOPS_BENCHMARK_DIR:$AUDITOPS_BENCHMARK_DIR:ro" \
  --bind "$AUDITOPS_MODEL_CONFIG:$AUDITOPS_MODEL_CONFIG:ro" \
  --bind "$MODEL_PATH:$MODEL_PATH:ro" \
  --bind "$AUDITOPS_ARTIFACT_ROOT:$AUDITOPS_ARTIFACT_ROOT:rw" \
  --bind "$RUNTIME_TMP/tmp:/auditops-tmp:rw" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env HF_HOME=/auditops-tmp/hf \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DISABLE_TELEMETRY=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env AUDITOPS_DISABLE_NETWORK=1 \
  --env "AUDITOPS_CONTAINER_SHA256=$ACTUAL_SIF_SHA256" \
  --env "AUDITOPS_MODEL_CONFIG_FILE_SHA256=$ACTUAL_MODEL_CONFIG_SHA256" \
  --env "AUDITOPS_SOURCE_MANIFEST_SHA256=$AUDITOPS_SOURCE_MANIFEST_SHA256" \
  --env "AUDITOPS_SOURCE_TREE_SHA256=$AUDITOPS_SOURCE_TREE_SHA256" \
  --env "AUDITOPS_PACKAGE_LOCK_SHA256=$AUDITOPS_PACKAGE_LOCK_SHA256" \
  --env "SLURM_JOB_ID=$SLURM_JOB_ID" \
  --pwd /workspace/AuditOps \
  "$AUDITOPS_VLLM_SIF" \
  python3 -m auditops.agent_preflight_cli \
    --benchmark-dir "$AUDITOPS_BENCHMARK_DIR" \
    --model-config "$AUDITOPS_MODEL_CONFIG" \
    --output-dir "$AUDITOPS_V24_PREFLIGHT_DIR"
