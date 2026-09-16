#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_BENCHMARK_DIR:?Set AUDITOPS_BENCHMARK_DIR}"
: "${AUDITOPS_MODEL_CONFIG:?Set AUDITOPS_MODEL_CONFIG}"
: "${AUDITOPS_MODEL_CONFIG_SHA256:?Set AUDITOPS_MODEL_CONFIG_SHA256 to an external digest pin}"
: "${AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256:?Set AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256 to an external digest pin}"
: "${AUDITOPS_RUN_OUTPUT_DIR:?Set AUDITOPS_RUN_OUTPUT_DIR to a new directory under AUDITOPS_RUN_ROOT}"
: "${AUDITOPS_RUNTIME_MODE:?Set AUDITOPS_RUNTIME_MODE to direct, capability_agent, or safety_hybrid}"
: "${AUDITOPS_PROMPT_CONDITION:?Set AUDITOPS_PROMPT_CONDITION to zero_shot or few_shot}"
: "${AUDITOPS_VLLM_SIF:?Set AUDITOPS_VLLM_SIF to the pinned Apptainer image}"
: "${AUDITOPS_VLLM_SIF_SHA256:?Set AUDITOPS_VLLM_SIF_SHA256 to an external digest pin}"
: "${AUDITOPS_SOURCE_MANIFEST:?Set AUDITOPS_SOURCE_MANIFEST}"
: "${AUDITOPS_SOURCE_MANIFEST_SHA256:?Set AUDITOPS_SOURCE_MANIFEST_SHA256 to the externally pinned manifest digest}"
: "${AUDITOPS_PACKAGE_LOCK_FILE:?Set AUDITOPS_PACKAGE_LOCK_FILE to the frozen package lock}"

AUDITOPS_NARRATIVE_GATE_V24=${AUDITOPS_NARRATIVE_GATE_V24:-0}
AUDITOPS_FULL_RUN_V24=${AUDITOPS_FULL_RUN_V24:-0}
AUDITOPS_EXPLORATORY_FULL_RUN_V24P=${AUDITOPS_EXPLORATORY_FULL_RUN_V24P:-0}
MODE_COUNT=$((AUDITOPS_NARRATIVE_GATE_V24 + AUDITOPS_FULL_RUN_V24 + AUDITOPS_EXPLORATORY_FULL_RUN_V24P))
if [[ "$MODE_COUNT" -gt 1 ]]; then
  echo "Narrative-gate, official-full, and exploratory-full modes are mutually exclusive" >&2
  exit 2
fi
if [[ "$AUDITOPS_FULL_RUN_V24" == "1" || "$AUDITOPS_EXPLORATORY_FULL_RUN_V24P" == "1" ]]; then
  : "${AUDITOPS_FULL_RUN_AUTHORIZATION:?A v2.4-family full run requires AUDITOPS_FULL_RUN_AUTHORIZATION}"
elif [[ -n "${AUDITOPS_FULL_RUN_AUTHORIZATION:-}" ]]; then
  echo "AUDITOPS_FULL_RUN_AUTHORIZATION is legal only for a v2.4-family full run" >&2
  exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Agent inference must run inside a Slurm GPU allocation" >&2
  exit 2
fi
if [[ "$AUDITOPS_RUNTIME_MODE" != "direct" && \
      "$AUDITOPS_RUNTIME_MODE" != "capability_agent" && \
      "$AUDITOPS_RUNTIME_MODE" != "safety_hybrid" ]]; then
  echo "AUDITOPS_RUNTIME_MODE must be direct, capability_agent, or safety_hybrid" >&2
  exit 2
fi
if [[ "$AUDITOPS_PROMPT_CONDITION" != "zero_shot" && "$AUDITOPS_PROMPT_CONDITION" != "few_shot" ]]; then
  echo "AUDITOPS_PROMPT_CONDITION must be zero_shot or few_shot" >&2
  exit 2
fi
if [[ "$AUDITOPS_PROMPT_CONDITION" == "few_shot" ]]; then
  : "${AUDITOPS_FEW_SHOT_APPROVAL:?Few-shot runs require AUDITOPS_FEW_SHOT_APPROVAL}"
elif [[ -n "${AUDITOPS_FEW_SHOT_APPROVAL:-}" ]]; then
  echo "Zero-shot runs must not receive a few-shot approval artifact" >&2
  exit 2
fi
if [[ -n "${AUDITOPS_RUN_LIMIT:-}" && ! "$AUDITOPS_RUN_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "AUDITOPS_RUN_LIMIT must be a positive integer" >&2
  exit 2
fi

# Hardware is an explicit experiment binding, independent of scheduler names.
: "${AUDITOPS_EXPECTED_GPU:?Set the exact GPU name reported by nvidia-smi}"
if [[ "${AUDITOPS_COMPATIBILITY_SMOKE:-0}" == "1" && ( -z "${AUDITOPS_RUN_LIMIT:-}" || "$AUDITOPS_RUN_LIMIT" -gt 10 ) ]]; then
  echo "Compatibility smoke requires a limit between 1 and 10" >&2
  exit 2
fi
if [[ "${AUDITOPS_PARITY_SLICE:-0}" == "1" && "${AUDITOPS_RUN_LIMIT:-}" != "50" ]]; then
  echo "Parity requires the frozen 50-case slice" >&2
  exit 2
fi
if [[ "$MODE_COUNT" -gt 0 && ( -n "${AUDITOPS_RUN_LIMIT:-}" || "$AUDITOPS_RUNTIME_MODE" != "safety_hybrid" || "$AUDITOPS_PROMPT_CONDITION" != "zero_shot" ) ]]; then
  echo "Quality-gated runs require zero-shot safety_hybrid and no prefix limit" >&2
  exit 2
fi

auditops_assert_new_path "$AUDITOPS_RUN_OUTPUT_DIR" "run output"
auditops_assert_under_root "$AUDITOPS_RUN_OUTPUT_DIR" "$AUDITOPS_RUN_ROOT" "Run output"
auditops_start_log auditops-agent
cd "$AUDITOPS_REPO_ROOT"
HOST_PYTHON=${AUDITOPS_PYTHON_BIN:-python3}
if ! command -v "$HOST_PYTHON" >/dev/null 2>&1; then
  echo "Host Python interpreter not found: $HOST_PYTHON" >&2
  exit 2
fi
auditops_require_sha256 AUDITOPS_SOURCE_MANIFEST_SHA256
auditops_require_sha256 AUDITOPS_VLLM_SIF_SHA256
auditops_require_sha256 AUDITOPS_MODEL_CONFIG_SHA256
auditops_require_sha256 AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256
for digest_name in \
  AUDITOPS_SOURCE_MANIFEST_SHA256 \
  AUDITOPS_VLLM_SIF_SHA256 \
  AUDITOPS_MODEL_CONFIG_SHA256 \
  AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256; do
  if [[ ! "${!digest_name}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$digest_name must be a canonical lowercase SHA-256 value" >&2
    exit 2
  fi
done

mkdir -p "$AUDITOPS_RUN_ROOT" "$AUDITOPS_TMP_ROOT" "$AUDITOPS_CACHE_ROOT/vllm"
RUNTIME_TMP=$(mktemp -d "$AUDITOPS_TMP_ROOT/agent-runtime.XXXXXXXX")
cleanup_runtime_tmp() {
  local resolved_tmp resolved_root
  resolved_tmp=$(realpath -m -- "$RUNTIME_TMP")
  resolved_root=$(realpath -m -- "$AUDITOPS_TMP_ROOT")
  case "$resolved_tmp" in
    "$resolved_root"/agent-runtime.*)
      find "$resolved_tmp" -type d -exec chmod u+w -- {} + 2>/dev/null || true
      rm -rf -- "$resolved_tmp"
      ;;
    *) echo "Refusing to remove unexpected runtime temp path: $resolved_tmp" >&2 ;;
  esac
}
trap cleanup_runtime_tmp EXIT
SOURCE_VIEW="$RUNTIME_TMP/source"
EXPECTED_MODEL_CONFIG_SHA256_FILE="$AUDITOPS_MODEL_CONFIG.sha256"
EXPECTED_VLLM_SIF_SHA256_FILE="$AUDITOPS_VLLM_SIF.sha256"
if [[ -n "${AUDITOPS_MODEL_CONFIG_SHA256_FILE:-}" && \
      "$AUDITOPS_MODEL_CONFIG_SHA256_FILE" != "$EXPECTED_MODEL_CONFIG_SHA256_FILE" ]]; then
  echo "Model-config checksum record must be colocated as <model-config>.sha256" >&2
  exit 2
fi
if [[ -n "${AUDITOPS_VLLM_SIF_SHA256_FILE:-}" && \
      "$AUDITOPS_VLLM_SIF_SHA256_FILE" != "$EXPECTED_VLLM_SIF_SHA256_FILE" ]]; then
  echo "Container checksum record must be colocated as <image>.sha256" >&2
  exit 2
fi
AUDITOPS_MODEL_CONFIG_SHA256_FILE="$EXPECTED_MODEL_CONFIG_SHA256_FILE"
AUDITOPS_VLLM_SIF_SHA256_FILE="$EXPECTED_VLLM_SIF_SHA256_FILE"

for required_file in \
  "$AUDITOPS_MODEL_CONFIG" \
  "$AUDITOPS_MODEL_CONFIG_SHA256_FILE" \
  "$AUDITOPS_SOURCE_MANIFEST" \
  "$AUDITOPS_PACKAGE_LOCK_FILE" \
  "$AUDITOPS_VLLM_SIF" \
  "$AUDITOPS_VLLM_SIF_SHA256_FILE"; do
  if [[ ! -f "$required_file" || -L "$required_file" ]]; then
    echo "Required immutable input not found: $required_file" >&2
    exit 2
  fi
  if [[ -w "$required_file" ]]; then
    echo "Required immutable input is writable: $required_file" >&2
    exit 2
  fi
done

ACTUAL_SOURCE_MANIFEST_SHA256=$(sha256sum "$AUDITOPS_SOURCE_MANIFEST" | awk '{print $1}')
if [[ "$ACTUAL_SOURCE_MANIFEST_SHA256" != "$AUDITOPS_SOURCE_MANIFEST_SHA256" ]]; then
  echo "Source manifest checksum differs from AUDITOPS_SOURCE_MANIFEST_SHA256" >&2
  exit 2
fi

if ! cmp -s -- "$AUDITOPS_VLLM_SIF_SHA256_FILE" \
  <(printf '%s  %s\n' "$AUDITOPS_VLLM_SIF_SHA256" "$(basename -- "$AUDITOPS_VLLM_SIF")"); then
  echo "Container checksum record differs from the external digest pin" >&2
  exit 2
fi
ACTUAL_SIF_SHA256=$(sha256sum "$AUDITOPS_VLLM_SIF" | awk '{print $1}')
if [[ "$ACTUAL_SIF_SHA256" != "$AUDITOPS_VLLM_SIF_SHA256" ]]; then
  echo "Apptainer image checksum mismatch" >&2
  exit 2
fi

if ! cmp -s -- "$AUDITOPS_MODEL_CONFIG_SHA256_FILE" \
  <(printf '%s  %s\n' "$AUDITOPS_MODEL_CONFIG_SHA256" "$(basename -- "$AUDITOPS_MODEL_CONFIG")"); then
  echo "Model-config checksum record differs from the external digest pin" >&2
  exit 2
fi
ACTUAL_MODEL_CONFIG_SHA256=$(sha256sum "$AUDITOPS_MODEL_CONFIG" | awk '{print $1}')
if [[ "$ACTUAL_MODEL_CONFIG_SHA256" != "$AUDITOPS_MODEL_CONFIG_SHA256" ]]; then
  echo "Model-config checksum mismatch" >&2
  exit 2
fi
MODEL_CONFIG_VIEW="$RUNTIME_TMP/config"
mkdir -p "$MODEL_CONFIG_VIEW"
cp -- "$AUDITOPS_MODEL_CONFIG" "$MODEL_CONFIG_VIEW/model.json"
chmod 0444 "$MODEL_CONFIG_VIEW/model.json"
chmod 0555 "$MODEL_CONFIG_VIEW"
VERIFIED_MODEL_CONFIG="$MODEL_CONFIG_VIEW/model.json"
if [[ "$(sha256sum "$VERIFIED_MODEL_CONFIG" | awk '{print $1}')" != "$AUDITOPS_MODEL_CONFIG_SHA256" ]]; then
  echo "Staged model config differs from the external digest pin" >&2
  exit 2
fi
CONTAINER_ARCH_FILE="$AUDITOPS_VLLM_SIF.target-architecture.txt"
if [[ ! -f "$CONTAINER_ARCH_FILE" ]]; then
  echo "Container architecture record not found: $CONTAINER_ARCH_FILE" >&2
  exit 2
fi
EXPECTED_CONTAINER_ARCH=${AUDITOPS_TARGET_ARCH:-amd64}
if [[ "$(<"$CONTAINER_ARCH_FILE")" != "$EXPECTED_CONTAINER_ARCH" ]]; then
  echo "This run requires a $EXPECTED_CONTAINER_ARCH SIF; refusing architecture substitution" >&2
  exit 2
fi

mapfile -t SOURCE_PROVENANCE < <("$HOST_PYTHON" -I -S - \
  "$AUDITOPS_REPO_ROOT" \
  "$AUDITOPS_SOURCE_MANIFEST" \
  "$SOURCE_VIEW" \
  "$AUDITOPS_SOURCE_MANIFEST_SHA256" <<'SOURCE_MANIFEST_VALIDATOR_PY'
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath

EXPECTED_TOP_LEVEL = {
    "provenance_version",
    "base_commit",
    "branch",
    "source_tree_sha256",
    "git_diff_sha256",
    "dirty",
    "file_count",
    "files",
}
EXPECTED_FILE_KEYS = {"path", "size", "sha256"}
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".auditops",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


def fail(message: str) -> "NoReturn":
    raise SystemExit(message)


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"Source manifest contains duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        fail("Source manifest file path must be a non-empty string")
    if "\\" in value or any(ord(character) < 32 for character in value):
        fail(f"Source manifest contains a non-canonical path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        fail(f"Source manifest contains a non-canonical path: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        fail(f"Source manifest contains a non-canonical path: {value!r}")
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in path.parts):
        fail(f"Source manifest includes an excluded directory: {value}")
    return value


def iter_snapshot_files(root: Path) -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        retained_directories = []
        for name in sorted(directory_names):
            candidate = current_path / name
            if candidate.is_symlink():
                fail(f"Source snapshot contains a directory symlink: {candidate.relative_to(root).as_posix()}")
            if name in EXCLUDED_DIRECTORY_NAMES:
                continue
            if not candidate.is_dir():
                fail(f"Source snapshot contains a non-directory entry: {candidate.relative_to(root).as_posix()}")
            retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink():
                fail(f"Source snapshot contains a file symlink: {relative}")
            try:
                mode = candidate.lstat().st_mode
            except OSError as exc:
                fail(f"Source snapshot entry cannot be inspected: {relative}: {exc}")
            if not stat.S_ISREG(mode):
                fail(f"Source snapshot contains a non-regular file: {relative}")
            discovered.append((canonical_relative_path(relative), candidate))
    return sorted(discovered, key=lambda item: item[0])


root = Path(sys.argv[1]).resolve(strict=True)
manifest_path = Path(sys.argv[2]).resolve(strict=True)
stage = Path(sys.argv[3]).resolve()
pinned_manifest_sha256 = sys.argv[4]
if not root.is_dir():
    fail(f"Source snapshot root is not a directory: {root}")
if not SHA256_RE.fullmatch(pinned_manifest_sha256):
    fail("Externally pinned source-manifest digest is not canonical SHA-256")

manifest_payload = manifest_path.read_bytes()
manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
if manifest_sha256 != pinned_manifest_sha256:
    fail("Source manifest checksum differs from the external pin")
try:
    manifest = json.loads(
        manifest_payload.decode("utf-8"), object_pairs_hook=reject_duplicate_keys
    )
except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    fail(f"Source manifest is not canonical UTF-8 JSON: {exc}")
if not isinstance(manifest, dict) or set(manifest) != EXPECTED_TOP_LEVEL:
    fail("Source manifest has an unexpected top-level schema")
if manifest["provenance_version"] != "auditops-provenance.v1":
    fail("Source manifest has an unsupported provenance_version")
commit = manifest["base_commit"]
tree_hash = manifest["source_tree_sha256"]
if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
    fail("Source manifest lacks a canonical immutable base_commit")
if not isinstance(manifest["branch"], str) or any(
    ord(character) < 32 for character in manifest["branch"]
):
    fail("Source manifest branch is malformed")
if not isinstance(tree_hash, str) or not SHA256_RE.fullmatch(tree_hash):
    fail("Source manifest lacks a canonical source_tree_sha256")
if not isinstance(manifest["git_diff_sha256"], str) or not SHA256_RE.fullmatch(
    manifest["git_diff_sha256"]
):
    fail("Source manifest lacks a canonical git_diff_sha256")
if not isinstance(manifest["dirty"], bool):
    fail("Source manifest dirty field must be boolean")
if (
    not isinstance(manifest["file_count"], int)
    or isinstance(manifest["file_count"], bool)
    or manifest["file_count"] < 1
):
    fail("Source manifest file_count must be a positive integer")
files = manifest["files"]
if not isinstance(files, list) or not files:
    fail("Source manifest files must be a non-empty list")
if manifest["file_count"] != len(files):
    fail("Source manifest file_count does not match files")

validated_rows = []
seen_paths = set()
for row in files:
    if not isinstance(row, dict) or set(row) != EXPECTED_FILE_KEYS:
        fail("Source manifest contains a malformed file row")
    relative = canonical_relative_path(row["path"])
    if relative in seen_paths:
        fail(f"Source manifest contains a duplicate path: {relative}")
    seen_paths.add(relative)
    size = row["size"]
    digest = row["sha256"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        fail(f"Source manifest size is invalid: {relative}")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        fail(f"Source manifest checksum is invalid: {relative}")
    validated_rows.append({"path": relative, "size": size, "sha256": digest})
if [row["path"] for row in validated_rows] != sorted(seen_paths):
    fail("Source manifest file rows are not in canonical path order")

actual_files = iter_snapshot_files(root)
actual_paths = [relative for relative, _ in actual_files]
manifest_paths = [row["path"] for row in validated_rows]
if actual_paths != manifest_paths:
    missing = sorted(set(manifest_paths) - set(actual_paths))
    unexpected = sorted(set(actual_paths) - set(manifest_paths))
    fail(f"Source snapshot file-set mismatch; missing={missing}; unexpected={unexpected}")

if stage.exists() and (not stage.is_dir() or any(stage.iterdir())):
    fail(f"Verified source staging directory is not empty: {stage}")
stage.mkdir(parents=True, exist_ok=True)
actual_rows = []
for expected, (relative, candidate) in zip(validated_rows, actual_files, strict=True):
    before = candidate.lstat()
    payload = candidate.read_bytes()
    after = candidate.lstat()
    stable_fields_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    stable_fields_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if stable_fields_before != stable_fields_after or not stat.S_ISREG(after.st_mode):
        fail(f"Source snapshot changed while it was being verified: {relative}")
    digest = hashlib.sha256(payload).hexdigest()
    actual_row = {"path": relative, "size": len(payload), "sha256": digest}
    if actual_row != expected:
        fail(f"Source snapshot metadata or checksum mismatch: {relative}")
    destination = stage.joinpath(*PurePosixPath(relative).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(payload)
    if os.name == "posix":
        destination.chmod(0o444)
    actual_rows.append(actual_row)

tree_payload = json.dumps(
    actual_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
).encode("utf-8")
if hashlib.sha256(tree_payload).hexdigest() != tree_hash:
    fail("Source manifest source_tree_sha256 does not match the verified file set")
staged_paths = [relative for relative, _ in iter_snapshot_files(stage)]
if staged_paths != manifest_paths:
    fail("Verified source staging file set is incomplete")
if os.name == "posix":
    for directory in sorted(
        (path for path in stage.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    stage.chmod(0o555)

print(commit)
print(tree_hash)
print(manifest_sha256)
SOURCE_MANIFEST_VALIDATOR_PY
)
if (( ${#SOURCE_PROVENANCE[@]} != 3 )); then
  echo "Could not resolve source provenance" >&2
  exit 2
fi
AUDITOPS_GIT_COMMIT=${SOURCE_PROVENANCE[0]}
AUDITOPS_SOURCE_TREE_SHA256=${SOURCE_PROVENANCE[1]}
VERIFIED_SOURCE_MANIFEST_SHA256=${SOURCE_PROVENANCE[2]}
auditops_require_git_commit AUDITOPS_GIT_COMMIT
auditops_require_sha256 AUDITOPS_SOURCE_TREE_SHA256
if [[ "$VERIFIED_SOURCE_MANIFEST_SHA256" != "$AUDITOPS_SOURCE_MANIFEST_SHA256" ]]; then
  echo "Verified source-manifest digest differs from the external pin" >&2
  exit 2
fi

VERIFIED_PACKAGE_LOCK_FILE="$SOURCE_VIEW/requirements/$(basename -- "$AUDITOPS_PACKAGE_LOCK_FILE")"
if [[ ! -f "$VERIFIED_PACKAGE_LOCK_FILE" || -L "$VERIFIED_PACKAGE_LOCK_FILE" ]]; then
  echo "Package lock is absent from the verified source snapshot" >&2
  exit 2
fi
AUDITOPS_PACKAGE_LOCK_SHA256=$(sha256sum "$VERIFIED_PACKAGE_LOCK_FILE" | awk '{print $1}')
auditops_require_sha256 AUDITOPS_PACKAGE_LOCK_SHA256

MODEL_REGISTRY="$SOURCE_VIEW/config/model_revisions.json"
if [[ ! -f "$MODEL_REGISTRY" || -L "$MODEL_REGISTRY" ]]; then
  echo "Pinned model registry is absent from the verified source snapshot" >&2
  exit 2
fi
mapfile -t MODEL_INFO < <("$HOST_PYTHON" -I -S - \
  "$VERIFIED_MODEL_CONFIG" "$MODEL_REGISTRY" "$AUDITOPS_MODEL_ROOT" <<'MODEL_CONFIG_VALIDATOR_PY'
import json
import sys
from pathlib import Path


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"Model input contains a duplicate JSON key: {key}")
        result[key] = value
    return result


config_path, registry_path, root_path = sys.argv[1:]
config = json.loads(
    Path(config_path).read_text(encoding="utf-8"),
    object_pairs_hook=reject_duplicate_keys,
)
registry = json.loads(
    Path(registry_path).read_text(encoding="utf-8"),
    object_pairs_hook=reject_duplicate_keys,
)
if not isinstance(config, dict) or config.get("model_config_version") != "v2":
    raise SystemExit("Model config version is unsupported")
if not isinstance(registry, dict) or registry.get("manifest_version") != "auditops-model-revisions.v1":
    raise SystemExit("Verified model registry version is unsupported")
model_id = config.get("model_id")
revision = config.get("revision")
matches = [row for row in registry.get("models", []) if row.get("model_id") == model_id]
if len(matches) != 1:
    raise SystemExit(f"Model is not uniquely pinned in the registry: {model_id}")
entry = matches[0]
if revision != entry.get("revision") or config.get("quantization") != entry.get("quantization"):
    raise SystemExit("Model config revision or quantization differs from the pinned registry")
if config.get("backend") != "vllm_offline":
    raise SystemExit("Inference backend must be vllm_offline")
model_path = Path(str(config.get("model_path"))).resolve()
model_root = Path(root_path).resolve()
try:
    model_path.relative_to(model_root)
except ValueError as exc:
    raise SystemExit(f"Model path must be under {model_root}") from exc
if not model_path.is_dir():
    raise SystemExit(f"Pinned model snapshot not found: {model_path}")
if "\n" in str(model_path) or "\n" in str(model_id) or "\n" in str(revision):
    raise SystemExit("Model metadata must not contain newlines")
print(model_path)
print(model_id)
print(revision)
MODEL_CONFIG_VALIDATOR_PY
)
if (( ${#MODEL_INFO[@]} != 3 )); then
  echo "Could not resolve pinned model configuration" >&2
  exit 2
fi
MODEL_PATH=${MODEL_INFO[0]}
MODEL_ID=${MODEL_INFO[1]}
MODEL_REVISION=${MODEL_INFO[2]}
AUDITOPS_MODEL_SNAPSHOT_SHA256="$AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256"
"$HOST_PYTHON" -I -S - "$SOURCE_VIEW" "$MODEL_PATH" \
  "$AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256" "$MODEL_ID" "$MODEL_REVISION" \
  "$VERIFIED_MODEL_CONFIG" <<'MODEL_SNAPSHOT_VALIDATOR_PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from auditops.provenance import verify_model_snapshot

snapshot_path = Path(sys.argv[2])
verify_model_snapshot(snapshot_path, sys.argv[3], require_read_only=True)
metadata = json.loads((snapshot_path / "AUDITOPS_SNAPSHOT.json").read_text(encoding="utf-8"))
config = json.loads(Path(sys.argv[6]).read_text(encoding="utf-8"))
expected_metadata = {
    "manifest_version": "auditops-model-snapshot.v1",
    "model_id": sys.argv[4],
    "revision": sys.argv[5],
    "quantization": config.get("quantization"),
}
if metadata != expected_metadata:
    raise SystemExit("Model snapshot metadata differs from the externally pinned config")
MODEL_SNAPSHOT_VALIDATOR_PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required in the GPU allocation" >&2
  exit 2
fi
GPU_INVENTORY=$(nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader)
if [[ -z "$GPU_INVENTORY" ]]; then
  echo "No GPU is visible in the allocation" >&2
  exit 2
fi
AUDITOPS_GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';' -)
if [[ "$AUDITOPS_GPU_MODEL" != "$AUDITOPS_EXPECTED_GPU" ]]; then
  echo "Visible GPU differs from AUDITOPS_EXPECTED_GPU" >&2
  exit 2
fi
echo "$GPU_INVENTORY"

BENCHMARK_VIEW="$RUNTIME_TMP/benchmark"
mkdir -p "$BENCHMARK_VIEW" "$RUNTIME_TMP/container-tmp"
ORIGINAL_BENCHMARK_DIR=$(realpath -e -- "$AUDITOPS_BENCHMARK_DIR")
ORIGINAL_GOLD_PATH=$(realpath -m -- "$AUDITOPS_BENCHMARK_DIR/gold.jsonl")
AUDITOPS_MEMBERSHIP_ARTIFACT=parity_50.json
if [[ "${AUDITOPS_NARRATIVE_GATE_V24:-0}" == "1" ]]; then
  AUDITOPS_MEMBERSHIP_ARTIFACT=gate_50.json
  for name in benchmark_manifest.json cases.jsonl evidence.jsonl gate_50.json; do
    if [[ ! -f "$AUDITOPS_BENCHMARK_DIR/$name" || -L "$AUDITOPS_BENCHMARK_DIR/$name" ]]; then
      echo "Required inference benchmark artifact not found: $name" >&2
      exit 2
    fi
    cp -- "$AUDITOPS_BENCHMARK_DIR/$name" "$BENCHMARK_VIEW/$name"
    chmod 0444 "$BENCHMARK_VIEW/$name"
  done
elif [[ "$AUDITOPS_FULL_RUN_V24" == "1" || "$AUDITOPS_EXPLORATORY_FULL_RUN_V24P" == "1" ]]; then
  if [[ ! -f "$AUDITOPS_FULL_RUN_AUTHORIZATION" || -L "$AUDITOPS_FULL_RUN_AUTHORIZATION" || \
        -w "$AUDITOPS_FULL_RUN_AUTHORIZATION" ]]; then
    echo "Full-run authorization must be an immutable regular file" >&2
    exit 2
  fi
  for name in benchmark_manifest.json cases.jsonl evidence.jsonl; do
    if [[ ! -f "$AUDITOPS_BENCHMARK_DIR/$name" || -L "$AUDITOPS_BENCHMARK_DIR/$name" ]]; then
      echo "Required inference benchmark artifact not found: $name" >&2
      exit 2
    fi
    cp -- "$AUDITOPS_BENCHMARK_DIR/$name" "$BENCHMARK_VIEW/$name"
    chmod 0444 "$BENCHMARK_VIEW/$name"
  done
  AUTHORIZATION_TARGET=full_run_authorization.json
  if [[ "$AUDITOPS_EXPLORATORY_FULL_RUN_V24P" == "1" ]]; then
    AUTHORIZATION_TARGET=exploratory_run_authorization.v2.4p.json
  fi
  AUDITOPS_MEMBERSHIP_ARTIFACT=$AUTHORIZATION_TARGET
  cp -- "$AUDITOPS_FULL_RUN_AUTHORIZATION" "$BENCHMARK_VIEW/$AUTHORIZATION_TARGET"
  chmod 0444 "$BENCHMARK_VIEW/$AUTHORIZATION_TARGET"
else
  for name in benchmark_manifest.json cases.jsonl evidence.jsonl parity_50.json; do
    if [[ ! -f "$AUDITOPS_BENCHMARK_DIR/$name" || -L "$AUDITOPS_BENCHMARK_DIR/$name" ]]; then
      echo "Required inference benchmark artifact not found: $name" >&2
      exit 2
    fi
    cp -- "$AUDITOPS_BENCHMARK_DIR/$name" "$BENCHMARK_VIEW/$name"
    chmod 0444 "$BENCHMARK_VIEW/$name"
  done
fi
if [[ "$AUDITOPS_PROMPT_CONDITION" == "few_shot" ]]; then
  for source in "$AUDITOPS_BENCHMARK_DIR/few_shot.jsonl" "$AUDITOPS_FEW_SHOT_APPROVAL"; do
    if [[ ! -f "$source" || -L "$source" ]]; then
      echo "Required few-shot artifact not found: $source" >&2
      exit 2
    fi
  done
  cp -- "$AUDITOPS_BENCHMARK_DIR/few_shot.jsonl" "$BENCHMARK_VIEW/few_shot.jsonl"
  cp -- "$AUDITOPS_FEW_SHOT_APPROVAL" "$BENCHMARK_VIEW/few_shot_approval.json"
  chmod 0444 "$BENCHMARK_VIEW/few_shot.jsonl" "$BENCHMARK_VIEW/few_shot_approval.json"
fi
chmod 0555 "$BENCHMARK_VIEW"

binds=(
  --bind "$SOURCE_VIEW:/workspace/AuditOps:ro"
  --bind "$BENCHMARK_VIEW:/auditops-benchmark:ro"
  --bind "$MODEL_PATH:$MODEL_PATH:ro"
  --bind "$VERIFIED_MODEL_CONFIG:/auditops-config/model.json:ro"
  --bind "$AUDITOPS_RUN_ROOT:$AUDITOPS_RUN_ROOT:rw"
  --bind "$AUDITOPS_CACHE_ROOT/vllm:$AUDITOPS_CACHE_ROOT/vllm:rw"
  --bind "$RUNTIME_TMP/container-tmp:/auditops-tmp:rw"
)

cli_args=(
  --benchmark-dir /auditops-benchmark
  --model-config /auditops-config/model.json
  --runtime-mode "$AUDITOPS_RUNTIME_MODE"
  --prompt-condition "$AUDITOPS_PROMPT_CONDITION"
  --output-dir "$AUDITOPS_RUN_OUTPUT_DIR"
)
if [[ "${AUDITOPS_NARRATIVE_GATE_V24:-0}" == "1" ]]; then
  cli_args+=(--selection-mode narrative_gate)
elif [[ "$AUDITOPS_FULL_RUN_V24" == "1" ]]; then
  cli_args+=(--selection-mode full --full-run-authorization /auditops-benchmark/full_run_authorization.json)
elif [[ "$AUDITOPS_EXPLORATORY_FULL_RUN_V24P" == "1" ]]; then
  cli_args+=(--selection-mode full --full-run-authorization /auditops-benchmark/exploratory_run_authorization.v2.4p.json --expected-authorizer "${AUDITOPS_EXPLORATORY_AUTHORIZER:?Set the explicit exploratory authorizer}")
fi
if [[ "$AUDITOPS_PROMPT_CONDITION" == "few_shot" ]]; then
  cli_args+=(--few-shot-approval /auditops-benchmark/few_shot_approval.json)
fi
if [[ -n "${AUDITOPS_RUN_LIMIT:-}" ]]; then
  cli_args+=(--limit "$AUDITOPS_RUN_LIMIT")
fi

RUN_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
CONTAINER_RUNTIME_BIN=$(command -v apptainer)
if [[ ! -x "$CONTAINER_RUNTIME_BIN" ]]; then
  echo "Apptainer executable is unavailable" >&2
  exit 2
fi
mkdir -p "$RUNTIME_TMP/launcher-home" "$RUNTIME_TMP/launcher-tmp"
chmod 0700 "$RUNTIME_TMP/launcher-home" "$RUNTIME_TMP/launcher-tmp"
AUDITOPS_CONTAINER_HOSTNAME=$(hostname -f)
AUDITOPS_HOST_NETNS=$(readlink /proc/self/ns/net)

# The submission environment is not an authority for Apptainer behavior. Remove
# every inherited compatibility/control variable, then invoke the resolved
# executable through env -i with a fixed launcher allowlist.
for environment_name in $(compgen -e); do
  case "$environment_name" in
    APPTAINER_*|SINGULARITY_*) unset "$environment_name" ;;
  esac
done
CONTAINER_LAUNCH_ENV=(
  "HOME=$RUNTIME_TMP/launcher-home"
  "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  "LANG=C.UTF-8"
  "LC_ALL=C.UTF-8"
  "TMPDIR=$RUNTIME_TMP/launcher-tmp"
)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CONTAINER_LAUNCH_ENV+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi

CONTAINER_ISOLATION_ARGS=(--cleanenv --containall --net --network none --nv)
CONTAINER_EXEC_HELP=$(env -i "${CONTAINER_LAUNCH_ENV[@]}" "$CONTAINER_RUNTIME_BIN" exec --help)
if grep -q -- '--no-mount' <<<"$CONTAINER_EXEC_HELP"; then
  CONTAINER_ISOLATION_ARGS+=(--no-mount bind-paths,hostfs,cwd)
fi

CONTAINER_ENV_ARGS=(
  --env "PYTHONDONTWRITEBYTECODE=1"
  --env "HF_HOME=/auditops-tmp/hf"
  --env "HF_HUB_OFFLINE=1"
  --env "TRANSFORMERS_OFFLINE=1"
  --env "HF_HUB_DISABLE_TELEMETRY=1"
  --env "TOKENIZERS_PARALLELISM=false"
  --env "VLLM_CACHE_ROOT=$AUDITOPS_CACHE_ROOT/vllm"
  --env "VLLM_NO_USAGE_STATS=1"
  --env "DO_NOT_TRACK=1"
  --env "TMPDIR=/auditops-tmp"
  --env "AUDITOPS_DISABLE_NETWORK=1"
  --env "AUDITOPS_GIT_COMMIT=$AUDITOPS_GIT_COMMIT"
  --env "AUDITOPS_SOURCE_TREE_SHA256=$AUDITOPS_SOURCE_TREE_SHA256"
  --env "AUDITOPS_SOURCE_MANIFEST_SHA256=$AUDITOPS_SOURCE_MANIFEST_SHA256"
  --env "AUDITOPS_CONTAINER_SHA256=$ACTUAL_SIF_SHA256"
  --env "AUDITOPS_MODEL_CONFIG_SHA256=$AUDITOPS_MODEL_CONFIG_SHA256"
  --env "AUDITOPS_MODEL_SNAPSHOT_SHA256=$AUDITOPS_MODEL_SNAPSHOT_SHA256"
  --env "AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256=$AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256"
  --env "AUDITOPS_MODEL_PATH=$MODEL_PATH"
  --env "AUDITOPS_PACKAGE_LOCK_SHA256=$AUDITOPS_PACKAGE_LOCK_SHA256"
  --env "AUDITOPS_GPU_MODEL=$AUDITOPS_GPU_MODEL"
  --env "AUDITOPS_HOST_NETNS=$AUDITOPS_HOST_NETNS"
  --env "HOSTNAME=$AUDITOPS_CONTAINER_HOSTNAME"
  --env "SLURM_JOB_ID=$SLURM_JOB_ID"
)
for slurm_name in \
  SLURM_JOB_NAME \
  SLURM_JOB_PARTITION \
  SLURM_JOB_NUM_NODES \
  SLURM_JOB_NODELIST \
  SLURM_CPUS_PER_TASK \
  SLURM_JOB_GPUS; do
  if [[ -n "${!slurm_name:-}" ]]; then
    CONTAINER_ENV_ARGS+=(--env "$slurm_name=${!slurm_name}")
  fi
done
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CONTAINER_ENV_ARGS+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi

auditops_apptainer_exec() {
  env -i "${CONTAINER_LAUNCH_ENV[@]}" "$CONTAINER_RUNTIME_BIN" exec \
    "${CONTAINER_ISOLATION_ARGS[@]}" \
    "${CONTAINER_ENV_ARGS[@]}" \
    "${binds[@]}" \
    --pwd /workspace/AuditOps \
    "$AUDITOPS_VLLM_SIF" \
    "$@"
}

if ! NETWORK_PREFLIGHT_OUTPUT=$(auditops_apptainer_exec python3 -I -S - \
  "$ORIGINAL_BENCHMARK_DIR" "$ORIGINAL_GOLD_PATH" "$AUDITOPS_PROMPT_CONDITION" \
  "$AUDITOPS_MEMBERSHIP_ARTIFACT" \
  <<'NETWORK_DENIAL_PREFLIGHT_PY'
import errno
import os
import socket
import sys
from pathlib import Path

original_benchmark = Path(sys.argv[1])
original_gold = Path(sys.argv[2])
prompt_condition = sys.argv[3]
membership_artifact = sys.argv[4]
for label, path in (
    ("original benchmark", original_benchmark),
    ("original gold", original_gold),
):
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise SystemExit(f"Could not prove the {label} path is absent") from exc
    else:
        raise SystemExit(f"Container can still access the {label} path")
if Path("/auditops-benchmark/gold.jsonl").exists():
    raise SystemExit("Evaluator-only gold is visible in the inference container")

expected_benchmark_files = {
    "benchmark_manifest.json",
    "cases.jsonl",
    "evidence.jsonl",
}
if membership_artifact != "NONE":
    expected_benchmark_files.add(membership_artifact)
if prompt_condition == "few_shot":
    expected_benchmark_files.update({"few_shot.jsonl", "few_shot_approval.json"})
benchmark_root = Path("/auditops-benchmark")
actual_benchmark_files = {entry.name for entry in benchmark_root.iterdir()}
if actual_benchmark_files != expected_benchmark_files:
    raise SystemExit("Inference-visible benchmark file set is not exact")
for entry in benchmark_root.iterdir():
    if entry.is_symlink() or not entry.is_file():
        raise SystemExit(f"Inference benchmark contains a non-regular entry: {entry.name}")

forbidden_environment_names = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_CLIENT_SECRET",
    "COMPANIES_HOUSE_API_KEY",
    "GITHUB_TOKEN",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "HF_TOKEN",
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "APPTAINER_BINDPATH",
    "APPTAINER_MOUNT",
    "SINGULARITY_BINDPATH",
    "SINGULARITY_MOUNT",
}
forbidden_secret_suffixes = ("_API_KEY", "_CREDENTIAL", "_PASSWORD", "_SECRET", "_TOKEN")
for name in os.environ:
    if (
        name in forbidden_environment_names
        or name.endswith(forbidden_secret_suffixes)
        or name.startswith("APPTAINERENV_")
        or name.startswith("SINGULARITYENV_")
    ):
        raise SystemExit(f"Forbidden secret/control environment is present: {name}")
for value in os.environ.values():
    if str(original_benchmark) in value or str(original_gold) in value:
        raise SystemExit("Original benchmark or gold path leaked through the environment")

sys.path.insert(0, "/workspace/AuditOps")
from auditops.provenance import verify_model_snapshot

verify_model_snapshot(
    os.environ["AUDITOPS_MODEL_PATH"],
    os.environ["AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256"],
    require_read_only=True,
)

host_namespace = os.environ.get("AUDITOPS_HOST_NETNS")
container_namespace = os.readlink("/proc/self/ns/net")
if not host_namespace or container_namespace == host_namespace:
    raise SystemExit("Apptainer did not create an isolated network namespace")

interfaces = {name for _, name in socket.if_nameindex()}
if interfaces - {"lo"}:
    raise SystemExit(f"Network namespace exposes non-loopback interfaces: {sorted(interfaces)}")

denied_errnos = {
    errno.EACCES,
    errno.EPERM,
    errno.ENETUNREACH,
    errno.EHOSTUNREACH,
    errno.EADDRNOTAVAIL,
    errno.EAFNOSUPPORT,
    errno.EPROTONOSUPPORT,
}
probes = (
    (socket.AF_INET, ("192.0.2.1", 9)),
    (socket.AF_INET6, ("2001:db8::1", 9, 0, 0)),
)
for family, address in probes:
    probe = None
    try:
        probe = socket.socket(family, socket.SOCK_STREAM)
        probe.settimeout(0.25)
        probe.connect(address)
    except OSError as exc:
        if exc.errno not in denied_errnos:
            raise SystemExit(
                f"Network-denial probe failed with an unapproved result: {family}: {exc!r}"
            ) from exc
    else:
        raise SystemExit(f"Network-denial probe unexpectedly connected: {family}")
    finally:
        if probe is not None:
            probe.close()

print("AUDITOPS_NETWORK_DENIED")
NETWORK_DENIAL_PREFLIGHT_PY
); then
  echo "Apptainer network isolation or socket-denial preflight failed" >&2
  exit 2
fi
if [[ "$NETWORK_PREFLIGHT_OUTPUT" != "AUDITOPS_NETWORK_DENIED" ]]; then
  echo "Apptainer network isolation preflight did not return its exact success marker" >&2
  exit 2
fi

REPORTED_VLLM_VERSION=$(auditops_apptainer_exec \
  python3 -I -c 'import vllm; print(vllm.__version__)')
if [[ "$REPORTED_VLLM_VERSION" != "0.26.0" ]]; then
  echo "Inference image reported vLLM $REPORTED_VLLM_VERSION, expected 0.26.0" >&2
  exit 2
fi

AGENT_ENTRYPOINT_PY='import hashlib, os, runpy, sys
from pathlib import Path
sys.path.insert(0, "/workspace/AuditOps")
from auditops.provenance import verify_model_snapshot
verify_model_snapshot(os.environ["AUDITOPS_MODEL_PATH"], os.environ["AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256"], require_read_only=True)
config_sha256 = hashlib.sha256(Path("/auditops-config/model.json").read_bytes()).hexdigest()
if config_sha256 != os.environ["AUDITOPS_MODEL_CONFIG_SHA256"]:
    raise SystemExit("In-container model config differs from the external digest pin")
sys.argv = ["auditops.agent_cli", *sys.argv[1:]]
runpy.run_module("auditops.agent_cli", run_name="__main__")'
AUDITOPS_AGENT_ENTRYPOINT_SHA256=$(printf '%s' "$AGENT_ENTRYPOINT_PY" | sha256sum | awk '{print $1}')
auditops_apptainer_exec python3 -I -c "$AGENT_ENTRYPOINT_PY" "${cli_args[@]}"
RUN_FINISHED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)

if [[ ! -f "$AUDITOPS_RUN_OUTPUT_DIR/results.jsonl" || ! -f "$AUDITOPS_RUN_OUTPUT_DIR/run_manifest.json" ]]; then
  echo "Run completed without the required immutable result artifacts" >&2
  exit 2
fi

export AUDITOPS_RUN_STARTED_AT="$RUN_STARTED_AT"
export AUDITOPS_RUN_FINISHED_AT="$RUN_FINISHED_AT"
export AUDITOPS_GPU_INVENTORY="$GPU_INVENTORY"
export AUDITOPS_CONTAINER_SHA256="$ACTUAL_SIF_SHA256"
export AUDITOPS_MODEL_ID="$MODEL_ID"
export AUDITOPS_MODEL_REVISION="$MODEL_REVISION"
export AUDITOPS_GIT_COMMIT
export AUDITOPS_SOURCE_TREE_SHA256
export AUDITOPS_SOURCE_MANIFEST_SHA256
export AUDITOPS_MODEL_SNAPSHOT_SHA256
export AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256
export AUDITOPS_MODEL_CONFIG_SHA256
export AUDITOPS_PACKAGE_LOCK_SHA256
export AUDITOPS_AGENT_ENTRYPOINT_SHA256
export AUDITOPS_RUNTIME_MODE
export AUDITOPS_PROMPT_CONDITION
export AUDITOPS_RUN_OUTPUT_DIR
if [[ -n "${AUDITOPS_RUN_LIMIT:-}" ]]; then
  export AUDITOPS_RUN_LIMIT
fi
export AUDITOPS_NARRATIVE_GATE_V24
export AUDITOPS_FULL_RUN_V24
export AUDITOPS_EXPLORATORY_FULL_RUN_V24P
export AUDITOPS_EXPLORATORY_AUTHORIZER="${AUDITOPS_EXPLORATORY_AUTHORIZER:-}"
export AUDITOPS_REPORTED_VLLM_VERSION="$REPORTED_VLLM_VERSION"
export AUDITOPS_NETWORK_ISOLATION="apptainer-net-none+socket-preflight"
"$HOST_PYTHON" -I -S - "$AUDITOPS_RUN_OUTPUT_DIR/slurm_launch_manifest.json" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
argv = [
    "python3",
    "-I",
    "-c",
    "<externally-pinned-model revalidation wrapper>",
    "--benchmark-dir",
    "/auditops-benchmark",
    "--model-config",
    "/auditops-config/model.json",
    "--runtime-mode",
    os.environ["AUDITOPS_RUNTIME_MODE"],
    "--prompt-condition",
    os.environ["AUDITOPS_PROMPT_CONDITION"],
    "--output-dir",
    os.environ["AUDITOPS_RUN_OUTPUT_DIR"],
]
if os.environ["AUDITOPS_PROMPT_CONDITION"] == "few_shot":
    argv.extend(["--few-shot-approval", "/auditops-benchmark/few_shot_approval.json"])
if os.environ.get("AUDITOPS_RUN_LIMIT"):
    argv.extend(["--limit", os.environ["AUDITOPS_RUN_LIMIT"]])
if os.environ.get("AUDITOPS_NARRATIVE_GATE_V24") == "1":
    argv.extend(["--selection-mode", "narrative_gate"])
elif os.environ.get("AUDITOPS_FULL_RUN_V24") == "1":
    argv.extend(
        [
            "--selection-mode",
            "full",
            "--full-run-authorization",
            "/auditops-benchmark/full_run_authorization.json",
        ]
    )
elif os.environ.get("AUDITOPS_EXPLORATORY_FULL_RUN_V24P") == "1":
    argv.extend(
        [
            "--selection-mode",
            "full",
            "--full-run-authorization",
            "/auditops-benchmark/exploratory_run_authorization.v2.4p.json",
            "--expected-authorizer",
            os.environ["AUDITOPS_EXPLORATORY_AUTHORIZER"],
        ]
    )
payload = {
    "manifest_version": "auditops-slurm-launch.v1",
    "started_at": os.environ["AUDITOPS_RUN_STARTED_AT"],
    "finished_at": os.environ["AUDITOPS_RUN_FINISHED_AT"],
    "command": {
        "entrypoint": "python3 -I -c <externally-pinned-model revalidation wrapper>",
        "entrypoint_sha256": os.environ["AUDITOPS_AGENT_ENTRYPOINT_SHA256"],
        "argv": argv,
        "runtime_mode": os.environ["AUDITOPS_RUNTIME_MODE"],
        "prompt_condition": os.environ["AUDITOPS_PROMPT_CONDITION"],
        "limit": os.environ.get("AUDITOPS_RUN_LIMIT"),
        "selection_mode": (
            "narrative_gate"
            if os.environ.get("AUDITOPS_NARRATIVE_GATE_V24") == "1"
            else "full"
            if os.environ.get("AUDITOPS_FULL_RUN_V24") == "1"
            else "full"
            if os.environ.get("AUDITOPS_EXPLORATORY_FULL_RUN_V24P") == "1"
            else None
        ),
    },
    "provenance": {
        "git_commit": os.environ["AUDITOPS_GIT_COMMIT"],
        "source_tree_sha256": os.environ["AUDITOPS_SOURCE_TREE_SHA256"],
        "source_manifest_sha256": os.environ["AUDITOPS_SOURCE_MANIFEST_SHA256"],
        "container_sha256": os.environ["AUDITOPS_CONTAINER_SHA256"],
        "model_snapshot_sha256": os.environ["AUDITOPS_MODEL_SNAPSHOT_SHA256"],
        "model_snapshot_manifest_sha256": os.environ[
            "AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256"
        ],
        "model_config_sha256": os.environ["AUDITOPS_MODEL_CONFIG_SHA256"],
        "package_lock_sha256": os.environ["AUDITOPS_PACKAGE_LOCK_SHA256"],
        "model_id": os.environ["AUDITOPS_MODEL_ID"],
        "model_revision": os.environ["AUDITOPS_MODEL_REVISION"],
        "vllm_version": os.environ["AUDITOPS_REPORTED_VLLM_VERSION"],
    },
    "network_isolation": {
        "policy": os.environ["AUDITOPS_NETWORK_ISOLATION"],
        "preflight": "passed",
    },
    "slurm": {
        "job_id": os.environ["SLURM_JOB_ID"],
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "nodes": os.environ.get("SLURM_JOB_NUM_NODES"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    },
    "gpu_inventory": os.environ["AUDITOPS_GPU_INVENTORY"].splitlines(),
}
with output.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

"$HOST_PYTHON" -I -S - "$SOURCE_VIEW" "$AUDITOPS_RUN_OUTPUT_DIR" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from auditops.provenance import assert_no_secrets

assert_no_secrets([sys.argv[2]])
PY

echo "RUN_OUTPUT=$AUDITOPS_RUN_OUTPUT_DIR"
echo "RUN_LOG=$AUDITOPS_ACTIVE_LOG"
