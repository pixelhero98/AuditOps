#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_CH_VLM_DIR:?Set AUDITOPS_CH_VLM_DIR to the frozen evaluation diagnostic}"
: "${AUDITOPS_CH_PDF_ROOT:?Set AUDITOPS_CH_PDF_ROOT to the paired PDF root}"
: "${AUDITOPS_MODEL_CONFIG:?Set AUDITOPS_MODEL_CONFIG}"
: "${AUDITOPS_VLM_RUN_OUTPUT_DIR:?Set AUDITOPS_VLM_RUN_OUTPUT_DIR under AUDITOPS_RUN_ROOT}"
: "${AUDITOPS_VLM_PROMPT_CONDITION:?Set zero_shot or few_shot}"
: "${AUDITOPS_VLLM_SIF:?Set AUDITOPS_VLLM_SIF}"
: "${AUDITOPS_VLLM_SIF_SHA256:?Externally pin the VLLM SIF SHA-256}"
: "${AUDITOPS_MODEL_CONFIG_SHA256:?Externally pin the model-config SHA-256}"
: "${AUDITOPS_SOURCE_MANIFEST:?Set AUDITOPS_SOURCE_MANIFEST}"
: "${AUDITOPS_SOURCE_MANIFEST_SHA256:?Externally pin the source manifest SHA-256}"
: "${AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256:?Externally pin the model snapshot manifest SHA-256}"
: "${AUDITOPS_PACKAGE_LOCK_FILE:?Set AUDITOPS_PACKAGE_LOCK_FILE}"
: "${AUDITOPS_MODEL_REGISTRY:?Set AUDITOPS_MODEL_REGISTRY}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "VLM launcher requires a Slurm GPU allocation" >&2
  exit 2
fi
if [[ "$AUDITOPS_VLM_PROMPT_CONDITION" != "zero_shot" && "$AUDITOPS_VLM_PROMPT_CONDITION" != "few_shot" ]]; then
  echo "AUDITOPS_VLM_PROMPT_CONDITION must be zero_shot or few_shot" >&2
  exit 2
fi
if [[ "$AUDITOPS_VLM_PROMPT_CONDITION" == "few_shot" ]]; then
  : "${AUDITOPS_CH_VLM_FEW_SHOT_DIR:?Few-shot runs require the separate four-filing diagnostic}"
  : "${AUDITOPS_CH_VLM_VISUAL_EXAMPLES:?Few-shot runs require human-authored visual examples}"
  : "${AUDITOPS_CH_VLM_FEW_SHOT_APPROVAL:?Few-shot runs require the approval artifact}"
  : "${AUDITOPS_CH_VLM_FEW_SHOT_PDF_ROOT:?Few-shot runs require their PDF root}"
  for required_dir in "$AUDITOPS_CH_VLM_FEW_SHOT_DIR" "$AUDITOPS_CH_VLM_FEW_SHOT_PDF_ROOT"; do
    if [[ ! -d "$required_dir" || -L "$required_dir" ]]; then
      echo "Few-shot input directory is missing or a symlink: $required_dir" >&2
      exit 2
    fi
  done
  for required_file in "$AUDITOPS_CH_VLM_VISUAL_EXAMPLES" "$AUDITOPS_CH_VLM_FEW_SHOT_APPROVAL"; do
    if [[ ! -f "$required_file" || -L "$required_file" ]]; then
      echo "Few-shot artifact is missing or a symlink: $required_file" >&2
      exit 2
    fi
  done
else
  for name in AUDITOPS_CH_VLM_FEW_SHOT_DIR AUDITOPS_CH_VLM_VISUAL_EXAMPLES AUDITOPS_CH_VLM_FEW_SHOT_APPROVAL AUDITOPS_CH_VLM_FEW_SHOT_PDF_ROOT; do
    if [[ -n "${!name:-}" ]]; then
      echo "Zero-shot VLM runs must not receive $name" >&2
      exit 2
    fi
  done
fi

auditops_require_sha256 AUDITOPS_SOURCE_MANIFEST_SHA256
auditops_require_sha256 AUDITOPS_VLLM_SIF_SHA256
auditops_require_sha256 AUDITOPS_MODEL_CONFIG_SHA256
auditops_require_sha256 AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256
for digest_name in AUDITOPS_SOURCE_MANIFEST_SHA256 AUDITOPS_VLLM_SIF_SHA256 AUDITOPS_MODEL_CONFIG_SHA256 AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256; do
  if [[ ! "${!digest_name}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$digest_name must be a canonical lowercase SHA-256 value" >&2
    exit 2
  fi
done
auditops_assert_new_path "$AUDITOPS_VLM_RUN_OUTPUT_DIR" "VLM run output"
auditops_assert_under_root "$AUDITOPS_VLM_RUN_OUTPUT_DIR" "$AUDITOPS_RUN_ROOT" "VLM run output"
auditops_start_log auditops-ch-vlm-run
cd "$AUDITOPS_REPO_ROOT"

EXPECTED_MODEL_CONFIG_SHA256_FILE="$AUDITOPS_MODEL_CONFIG.sha256"
EXPECTED_VLLM_SIF_SHA256_FILE="$AUDITOPS_VLLM_SIF.sha256"
if [[ -n "${AUDITOPS_MODEL_CONFIG_SHA256_FILE:-}" && "$AUDITOPS_MODEL_CONFIG_SHA256_FILE" != "$EXPECTED_MODEL_CONFIG_SHA256_FILE" ]]; then
  echo "Model-config checksum record must be colocated as <model-config>.sha256" >&2
  exit 2
fi
if [[ -n "${AUDITOPS_VLLM_SIF_SHA256_FILE:-}" && "$AUDITOPS_VLLM_SIF_SHA256_FILE" != "$EXPECTED_VLLM_SIF_SHA256_FILE" ]]; then
  echo "Container checksum record must be colocated as <image>.sha256" >&2
  exit 2
fi
AUDITOPS_MODEL_CONFIG_SHA256_FILE="$EXPECTED_MODEL_CONFIG_SHA256_FILE"
AUDITOPS_VLLM_SIF_SHA256_FILE="$EXPECTED_VLLM_SIF_SHA256_FILE"

for required_file in "$AUDITOPS_MODEL_CONFIG" "$AUDITOPS_MODEL_CONFIG_SHA256_FILE" "$AUDITOPS_VLLM_SIF" "$AUDITOPS_VLLM_SIF_SHA256_FILE" "$AUDITOPS_SOURCE_MANIFEST" "$AUDITOPS_PACKAGE_LOCK_FILE" "$AUDITOPS_MODEL_REGISTRY"; do
  if [[ ! -f "$required_file" || -L "$required_file" ]]; then
    echo "Required immutable input is missing or a symlink: $required_file" >&2
    exit 2
  fi
done
for required_dir in "$AUDITOPS_CH_VLM_DIR" "$AUDITOPS_CH_PDF_ROOT"; do
  if [[ ! -d "$required_dir" || -L "$required_dir" ]]; then
    echo "Required input directory is missing or a symlink: $required_dir" >&2
    exit 2
  fi
done
if ! cmp -s -- "$AUDITOPS_VLLM_SIF_SHA256_FILE" <(printf '%s  %s\n' "$AUDITOPS_VLLM_SIF_SHA256" "$(basename -- "$AUDITOPS_VLLM_SIF")"); then
  echo "Container checksum record differs from the external digest pin" >&2
  exit 2
fi
AUDITOPS_CONTAINER_SHA256=$(sha256sum "$AUDITOPS_VLLM_SIF" | awk '{print $1}')
if [[ "$AUDITOPS_CONTAINER_SHA256" != "$AUDITOPS_VLLM_SIF_SHA256" ]]; then
  echo "Container checksum mismatch" >&2
  exit 2
fi
if ! cmp -s -- "$AUDITOPS_MODEL_CONFIG_SHA256_FILE" <(printf '%s  %s\n' "$AUDITOPS_MODEL_CONFIG_SHA256" "$(basename -- "$AUDITOPS_MODEL_CONFIG")"); then
  echo "Model-config checksum record differs from the external digest pin" >&2
  exit 2
fi
if [[ "$(sha256sum "$AUDITOPS_MODEL_CONFIG" | awk '{print $1}')" != "$AUDITOPS_MODEL_CONFIG_SHA256" ]]; then
  echo "Model-config checksum mismatch" >&2
  exit 2
fi
AUDITOPS_PACKAGE_LOCK_SHA256=$(sha256sum "$AUDITOPS_PACKAGE_LOCK_FILE" | awk '{print $1}')
auditops_require_sha256 AUDITOPS_PACKAGE_LOCK_SHA256

HOST_PYTHON=${AUDITOPS_PYTHON_BIN:-python3}
mkdir -p "$AUDITOPS_TMP_ROOT" "$AUDITOPS_RUN_ROOT" "$AUDITOPS_CACHE_ROOT/vllm"
RUNTIME_TMP=$(mktemp -d "$AUDITOPS_TMP_ROOT/ch-vlm-runtime.XXXXXXXX")
cleanup_runtime_tmp() {
  local resolved_tmp resolved_root
  resolved_tmp=$(realpath -m -- "$RUNTIME_TMP")
  resolved_root=$(realpath -m -- "$AUDITOPS_TMP_ROOT")
  case "$resolved_tmp" in
    "$resolved_root"/ch-vlm-runtime.*)
      find "$resolved_tmp" -type d -exec chmod u+w -- {} + 2>/dev/null || true
      rm -rf -- "$resolved_tmp"
      ;;
    *) echo "Refusing to remove unexpected VLM temp path: $resolved_tmp" >&2 ;;
  esac
}
trap cleanup_runtime_tmp EXIT

SOURCE_VIEW="$RUNTIME_TMP/source"
mapfile -t SOURCE_INFO < <("$HOST_PYTHON" -I -S - \
  "$AUDITOPS_REPO_ROOT" "$AUDITOPS_SOURCE_MANIFEST" "$SOURCE_VIEW" \
  "$AUDITOPS_SOURCE_MANIFEST_SHA256" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path, PurePosixPath

EXCLUDED = {".git", ".auditops", ".pytest_cache", ".ruff_cache", "__pycache__", ".venv", "venv"}


def fail(message):
    raise SystemExit(message)


def reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            fail(f"JSON contains a duplicate key: {key}")
        value[key] = item
    return value


def canonical_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or any(ord(character) < 32 for character in value):
        fail("Source manifest contains a non-canonical path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        fail("Source manifest contains an unsafe path")
    if any(part in EXCLUDED for part in path.parts):
        fail("Source manifest includes an excluded path")
    return value


def read_regular_once(path, label):
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        fail(f"{label} is not a non-symlink regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_mode, before.st_size) != (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_size):
            fail(f"{label} changed before it was opened")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    stable_before = (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns)
    stable_open = (after_open.st_dev, after_open.st_ino, after_open.st_mode, after_open.st_size, after_open.st_mtime_ns)
    stable_after = (after_path.st_dev, after_path.st_ino, after_path.st_mode, after_path.st_size, after_path.st_mtime_ns)
    if os.name == "posix":
        stable_before += (before.st_ctime_ns,)
        stable_open += (after_open.st_ctime_ns,)
        stable_after += (after_path.st_ctime_ns,)
    if stable_before != stable_open or stable_before != stable_after or len(payload) != before.st_size:
        fail(f"{label} changed while it was read")
    return payload


def snapshot_paths(root):
    discovered = []
    for current, directories, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        retained = []
        for name in sorted(directories):
            candidate = current_path / name
            if candidate.is_symlink() or not candidate.is_dir():
                fail("Source snapshot contains a non-directory entry")
            if name not in EXCLUDED:
                retained.append(name)
        directories[:] = retained
        for name in sorted(names):
            candidate = current_path / name
            relative = canonical_relative(candidate.relative_to(root).as_posix())
            if candidate.is_symlink() or not stat.S_ISREG(candidate.lstat().st_mode):
                fail(f"Source snapshot contains a non-regular file: {relative}")
            discovered.append((relative, candidate))
    return sorted(discovered, key=lambda item: item[0])

root_argument = Path(sys.argv[1])
if root_argument.is_symlink() or not root_argument.is_dir():
    fail("Source snapshot root must be a non-symlink directory")
root = root_argument.resolve(strict=True)
manifest_path = Path(sys.argv[2])
stage = Path(sys.argv[3])
external_manifest_sha256 = sys.argv[4]
manifest_payload = read_regular_once(manifest_path, "Source manifest")
manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
if manifest_sha256 != external_manifest_sha256:
    fail("Source manifest differs from its external digest pin")
try:
    expected = json.loads(manifest_payload.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    fail(f"Source manifest is invalid UTF-8 JSON: {exc}")
if not isinstance(expected, dict) or set(expected) != {"provenance_version", "base_commit", "branch", "source_tree_sha256", "git_diff_sha256", "dirty", "file_count", "files"}:
    fail("Source manifest has an unexpected schema")
expected_rows = expected["files"]
if not isinstance(expected_rows, list) or expected.get("file_count") != len(expected_rows):
    fail("Source manifest file_count differs from files")
for row in expected_rows:
    if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
        fail("Source manifest contains an invalid file row")
    canonical_relative(row["path"])

source_paths = snapshot_paths(root)
stage.mkdir()
files = []
for relative, candidate in source_paths:
    payload = read_regular_once(candidate, f"Source file {relative}")
    row = {"path": relative, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    files.append(row)
    target = stage.joinpath(*PurePosixPath(relative).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(payload)
    target.chmod(0o444)
if files != expected_rows:
    fail("Deployable source bytes differ from the externally pinned manifest")
if [relative for relative, _ in snapshot_paths(root)] != [row["path"] for row in files]:
    fail("Deployable source file set changed during verification")
tree_payload = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
if hashlib.sha256(tree_payload).hexdigest() != expected.get("source_tree_sha256"):
    fail("Deployable source tree hash differs from the manifest")
for directory in sorted((path for path in stage.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
    directory.chmod(0o555)
stage.chmod(0o555)
print(expected["base_commit"])
print(expected["source_tree_sha256"])
print(manifest_sha256)
PY
)
if (( ${#SOURCE_INFO[@]} != 3 )); then
  echo "Could not stage the verified source snapshot" >&2
  exit 2
fi
AUDITOPS_GIT_COMMIT=${SOURCE_INFO[0]}
AUDITOPS_SOURCE_TREE_SHA256=${SOURCE_INFO[1]}
VERIFIED_SOURCE_MANIFEST_SHA256=${SOURCE_INFO[2]}
auditops_require_git_commit AUDITOPS_GIT_COMMIT
auditops_require_sha256 AUDITOPS_SOURCE_TREE_SHA256
if [[ "$VERIFIED_SOURCE_MANIFEST_SHA256" != "$AUDITOPS_SOURCE_MANIFEST_SHA256" ]]; then
  echo "Staged source manifest digest differs from its external pin" >&2
  exit 2
fi

VERIFIED_MODEL_CONFIG="$RUNTIME_TMP/model.json"
cp -- "$AUDITOPS_MODEL_CONFIG" "$VERIFIED_MODEL_CONFIG"
chmod 0444 "$VERIFIED_MODEL_CONFIG"
if [[ "$(sha256sum "$VERIFIED_MODEL_CONFIG" | awk '{print $1}')" != "$AUDITOPS_MODEL_CONFIG_SHA256" ]]; then
  echo "Staged model config differs from its external pin" >&2
  exit 2
fi

mapfile -t MODEL_INFO < <("$HOST_PYTHON" -I -S - "$SOURCE_VIEW" "$VERIFIED_MODEL_CONFIG" "$SOURCE_VIEW/config/model_revisions.json" "$AUDITOPS_MODEL_ROOT" "$AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256" <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from auditops.agent_contracts import validate_model_config
from auditops.provenance import verify_model_snapshot

config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
registry = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
validate_model_config(config)
matches = [row for row in registry.get("models", []) if row.get("model_id") == config["model_id"]]
if len(matches) != 1 or matches[0].get("revision") != config["revision"] or matches[0].get("quantization") != config["quantization"]:
    raise SystemExit("Model config differs from the pinned registry")
model = Path(config["model_path"]).resolve(strict=True)
model.relative_to(Path(sys.argv[4]).resolve(strict=True))
verify_model_snapshot(model, sys.argv[5], require_read_only=True)
print(model)
print(config["model_id"])
print(config["revision"])
PY
)
if (( ${#MODEL_INFO[@]} != 3 )); then
  echo "Could not verify the pinned model snapshot" >&2
  exit 2
fi
MODEL_PATH=${MODEL_INFO[0]}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable" >&2
  exit 2
fi
AUDITOPS_GPU_INVENTORY=$(nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader | paste -sd ';' -)
: "${AUDITOPS_EXPECTED_GPU:?Set the exact GPU name reported by nvidia-smi}"
AUDITOPS_GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';' -)
if [[ -z "$AUDITOPS_GPU_INVENTORY" || "$AUDITOPS_GPU_NAME" != "$AUDITOPS_EXPECTED_GPU" ]]; then
  echo "Visible GPU differs from AUDITOPS_EXPECTED_GPU" >&2
  exit 2
fi

DIAGNOSTIC_VIEW="$RUNTIME_TMP/evaluation-diagnostic"
mkdir -p "$DIAGNOSTIC_VIEW"
for name in diagnostic_manifest.json cases.jsonl; do
  if [[ ! -f "$AUDITOPS_CH_VLM_DIR/$name" || -L "$AUDITOPS_CH_VLM_DIR/$name" ]]; then
    echo "Missing inference-visible VLM artifact: $name" >&2
    exit 2
  fi
  cp -- "$AUDITOPS_CH_VLM_DIR/$name" "$DIAGNOSTIC_VIEW/$name"
  chmod 0444 "$DIAGNOSTIC_VIEW/$name"
done
# Evaluator-only evaluation gold is intentionally neither copied nor mounted.

if [[ "$AUDITOPS_VLM_PROMPT_CONDITION" == "few_shot" ]]; then
  FEW_VIEW="$RUNTIME_TMP/few-shot"
  mkdir -p "$FEW_VIEW"
  for name in diagnostic_manifest.json cases.jsonl gold.jsonl; do
    cp -- "$AUDITOPS_CH_VLM_FEW_SHOT_DIR/$name" "$FEW_VIEW/$name"
    chmod 0444 "$FEW_VIEW/$name"
  done
  cp -- "$AUDITOPS_CH_VLM_VISUAL_EXAMPLES" "$FEW_VIEW/visual_examples.jsonl"
  cp -- "$AUDITOPS_CH_VLM_FEW_SHOT_APPROVAL" "$FEW_VIEW/approval.json"
  chmod 0444 "$FEW_VIEW/visual_examples.jsonl" "$FEW_VIEW/approval.json"
fi

stage_pdf_view() {
  local source_root=$1
  local stage_root=$2
  local manifest_output=$3
  shift 3
  "$HOST_PYTHON" -I -S - "$source_root" "$stage_root" "$manifest_output" "$@" <<'PDF_STAGE_PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path, PurePosixPath


def fail(message):
    raise SystemExit(message)


def reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            fail(f"VLM cases contain a duplicate JSON key: {key}")
        value[key] = item
    return value


def read_regular_once(path, label):
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        fail(f"{label} is not a non-symlink regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_mode, before.st_size) != (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_size):
            fail(f"{label} changed before it was opened")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    stable_before = (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns)
    stable_open = (after_open.st_dev, after_open.st_ino, after_open.st_mode, after_open.st_size, after_open.st_mtime_ns)
    stable_after = (after_path.st_dev, after_path.st_ino, after_path.st_mode, after_path.st_size, after_path.st_mtime_ns)
    if os.name == "posix":
        stable_before += (before.st_ctime_ns,)
        stable_open += (after_open.st_ctime_ns,)
        stable_after += (after_path.st_ctime_ns,)
    if stable_before != stable_open or stable_before != stable_after or len(payload) != before.st_size:
        fail(f"{label} changed while it was read")
    return payload, (after_open.st_dev, after_open.st_ino)


def checked_pdf_path(root, raw_path):
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path or any(ord(character) < 32 for character in raw_path):
        fail("VLM case pdf_path is not canonical text")
    lexical = Path(raw_path)
    if any(part in {"", ".", ".."} for part in lexical.parts):
        fail("VLM case pdf_path contains an unsafe component")
    candidate = lexical if lexical.is_absolute() else root / lexical
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        fail("VLM case PDF is outside the declared source root")
    if not relative.parts or relative.suffix.lower() != ".pdf":
        fail("VLM case input must be a PDF below the declared source root")
    cursor = root
    for index, part in enumerate(relative.parts):
        cursor = cursor / part
        status = cursor.lstat()
        if cursor.is_symlink():
            fail("VLM PDF path contains a symlink")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(status.st_mode):
            fail("VLM PDF parent path contains a non-directory entry")
        if index == len(relative.parts) - 1 and not stat.S_ISREG(status.st_mode):
            fail("VLM PDF source is not a regular file")
    return cursor, PurePosixPath(*relative.parts).as_posix()


root_argument = Path(sys.argv[1])
if root_argument.is_symlink() or not root_argument.is_dir():
    fail("VLM PDF source root must be a non-symlink directory")
root = root_argument.resolve(strict=True)
stage = Path(sys.argv[2])
manifest_output = Path(sys.argv[3])
case_files = [Path(value) for value in sys.argv[4:]]
if not case_files:
    fail("At least one frozen VLM cases file is required")

case_rows = []
seen_tasks = set()
for cases_path in case_files:
    payload, _ = read_regular_once(cases_path, "Frozen VLM cases")
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        fail(f"Frozen VLM cases are not UTF-8: {exc}")
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            fail(f"Frozen VLM case line {line_number} is invalid JSON: {exc}")
        if not isinstance(row, dict):
            fail("Frozen VLM case must be an object")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in seen_tasks:
            fail("Frozen VLM cases contain a missing or duplicate task_id")
        seen_tasks.add(task_id)
        source, relative = checked_pdf_path(root, row.get("pdf_path"))
        case_rows.append((task_id, source, relative))

relative_paths = [relative for _, _, relative in case_rows]
if len(relative_paths) != len(set(relative_paths)):
    fail("Frozen VLM cases contain duplicate PDF paths")
stage.mkdir()
manifest_rows = []
source_identities = set()
for task_id, source, relative in sorted(case_rows, key=lambda row: row[2]):
    payload, source_identity = read_regular_once(source, f"VLM PDF {relative}")
    if source_identity in source_identities:
        fail("Frozen VLM cases contain duplicate PDF source files")
    source_identities.add(source_identity)
    target = stage.joinpath(*PurePosixPath(relative).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(payload)
    target.chmod(0o444)
    manifest_rows.append({
        "task_id": task_id,
        "path": relative,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    })

actual_files = []
for current, directories, names in os.walk(stage, topdown=True, followlinks=False):
    current_path = Path(current)
    for name in sorted(directories):
        candidate = current_path / name
        if candidate.is_symlink() or not candidate.is_dir():
            fail("Staged VLM PDF tree contains a non-directory entry")
    for name in sorted(names):
        candidate = current_path / name
        if candidate.is_symlink() or not stat.S_ISREG(candidate.lstat().st_mode) or candidate.suffix.lower() != ".pdf":
            fail("Staged VLM PDF tree contains a non-PDF or special entry")
        actual_files.append(candidate.relative_to(stage).as_posix())
if sorted(actual_files) != sorted(relative_paths):
    fail("Staged VLM PDF tree does not exactly match frozen case membership")

manifest = {
    "manifest_version": "auditops-vlm-pdf-stage.v1",
    "file_count": len(manifest_rows),
    "files": manifest_rows,
}
manifest_payload = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
with manifest_output.open("xb") as handle:
    handle.write(manifest_payload)
manifest_output.chmod(0o444)
for directory in sorted((path for path in stage.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
    directory.chmod(0o555)
stage.chmod(0o555)
print(hashlib.sha256(manifest_payload).hexdigest())
PDF_STAGE_PY
}

EVALUATION_PDF_SOURCE_ROOT=$(realpath -e -- "$AUDITOPS_CH_PDF_ROOT")
EVALUATION_PDF_VIEW="$RUNTIME_TMP/evaluation-pdfs"
EVALUATION_PDF_MANIFEST="$RUNTIME_TMP/evaluation-pdf-manifest.json"
if [[ "$AUDITOPS_VLM_PROMPT_CONDITION" == "few_shot" ]]; then
  FEW_SHOT_PDF_SOURCE_ROOT=$(realpath -e -- "$AUDITOPS_CH_VLM_FEW_SHOT_PDF_ROOT")
else
  FEW_SHOT_PDF_SOURCE_ROOT=""
fi

assert_pdf_root_isolated() {
  local pdf_root=$1
  local label=$2
  local protected resolved_protected
  for protected in "$MODEL_PATH" "$AUDITOPS_RUN_ROOT" "$AUDITOPS_CACHE_ROOT" "$AUDITOPS_TMP_ROOT"; do
    resolved_protected=$(realpath -m -- "$protected")
    case "$resolved_protected/" in
      "$pdf_root"/*)
        echo "$label is too broad and would cover another required mount: $resolved_protected" >&2
        exit 2
        ;;
    esac
    case "$pdf_root/" in
      "$resolved_protected"/*)
        echo "$label is nested below a writable or separately exposed mount: $resolved_protected" >&2
        exit 2
        ;;
    esac
  done
}
assert_pdf_root_isolated "$EVALUATION_PDF_SOURCE_ROOT" "Evaluation PDF root"
if [[ -n "$FEW_SHOT_PDF_SOURCE_ROOT" ]]; then
  assert_pdf_root_isolated "$FEW_SHOT_PDF_SOURCE_ROOT" "Few-shot PDF root"
  if [[ "$FEW_SHOT_PDF_SOURCE_ROOT" != "$EVALUATION_PDF_SOURCE_ROOT" ]]; then
    case "$FEW_SHOT_PDF_SOURCE_ROOT/" in
      "$EVALUATION_PDF_SOURCE_ROOT"/*) echo "Few-shot PDF root may not be nested below the evaluation PDF root" >&2; exit 2 ;;
    esac
    case "$EVALUATION_PDF_SOURCE_ROOT/" in
      "$FEW_SHOT_PDF_SOURCE_ROOT"/*) echo "Evaluation PDF root may not be nested below the few-shot PDF root" >&2; exit 2 ;;
    esac
  fi
fi

binds=(
  --bind "$SOURCE_VIEW:/workspace/AuditOps:ro"
  --bind "$DIAGNOSTIC_VIEW:/auditops-vlm/evaluation:ro"
  --bind "$MODEL_PATH:$MODEL_PATH:ro"
  --bind "$VERIFIED_MODEL_CONFIG:/auditops-vlm/model.json:ro"
  --bind "$AUDITOPS_RUN_ROOT:$AUDITOPS_RUN_ROOT:rw"
  --bind "$AUDITOPS_CACHE_ROOT/vllm:$AUDITOPS_CACHE_ROOT/vllm:rw"
  --bind "$RUNTIME_TMP/container-tmp:/auditops-tmp:rw"
)
cli_args=(
  --diagnostic-dir /auditops-vlm/evaluation
  --model-config /auditops-vlm/model.json
  --output-dir "$AUDITOPS_VLM_RUN_OUTPUT_DIR"
  --pdf-root "$EVALUATION_PDF_SOURCE_ROOT"
  --prompt-condition "$AUDITOPS_VLM_PROMPT_CONDITION"
)

if [[ "$AUDITOPS_VLM_PROMPT_CONDITION" == "few_shot" && "$FEW_SHOT_PDF_SOURCE_ROOT" == "$EVALUATION_PDF_SOURCE_ROOT" ]]; then
  EVALUATION_PDF_MANIFEST_SHA256=$(stage_pdf_view \
    "$EVALUATION_PDF_SOURCE_ROOT" "$EVALUATION_PDF_VIEW" "$EVALUATION_PDF_MANIFEST" \
    "$DIAGNOSTIC_VIEW/cases.jsonl" "$FEW_VIEW/cases.jsonl")
  FEW_SHOT_PDF_VIEW="$EVALUATION_PDF_VIEW"
  FEW_SHOT_PDF_MANIFEST="$EVALUATION_PDF_MANIFEST"
  FEW_SHOT_PDF_MANIFEST_SHA256="$EVALUATION_PDF_MANIFEST_SHA256"
  binds+=(--bind "$EVALUATION_PDF_VIEW:$EVALUATION_PDF_SOURCE_ROOT:ro")
elif [[ "$AUDITOPS_VLM_PROMPT_CONDITION" == "few_shot" ]]; then
  EVALUATION_PDF_MANIFEST_SHA256=$(stage_pdf_view \
    "$EVALUATION_PDF_SOURCE_ROOT" "$EVALUATION_PDF_VIEW" "$EVALUATION_PDF_MANIFEST" \
    "$DIAGNOSTIC_VIEW/cases.jsonl")
  FEW_SHOT_PDF_VIEW="$RUNTIME_TMP/few-shot-pdfs"
  FEW_SHOT_PDF_MANIFEST="$RUNTIME_TMP/few-shot-pdf-manifest.json"
  FEW_SHOT_PDF_MANIFEST_SHA256=$(stage_pdf_view \
    "$FEW_SHOT_PDF_SOURCE_ROOT" "$FEW_SHOT_PDF_VIEW" "$FEW_SHOT_PDF_MANIFEST" \
    "$FEW_VIEW/cases.jsonl")
  binds+=(
    --bind "$EVALUATION_PDF_VIEW:$EVALUATION_PDF_SOURCE_ROOT:ro"
    --bind "$FEW_SHOT_PDF_VIEW:$FEW_SHOT_PDF_SOURCE_ROOT:ro"
  )
else
  EVALUATION_PDF_MANIFEST_SHA256=$(stage_pdf_view \
    "$EVALUATION_PDF_SOURCE_ROOT" "$EVALUATION_PDF_VIEW" "$EVALUATION_PDF_MANIFEST" \
    "$DIAGNOSTIC_VIEW/cases.jsonl")
  FEW_SHOT_PDF_VIEW=""
  FEW_SHOT_PDF_MANIFEST=""
  FEW_SHOT_PDF_MANIFEST_SHA256=""
  binds+=(--bind "$EVALUATION_PDF_VIEW:$EVALUATION_PDF_SOURCE_ROOT:ro")
fi
auditops_require_sha256 EVALUATION_PDF_MANIFEST_SHA256
if [[ -n "$FEW_SHOT_PDF_MANIFEST_SHA256" ]]; then
  auditops_require_sha256 FEW_SHOT_PDF_MANIFEST_SHA256
fi

if [[ "$AUDITOPS_VLM_PROMPT_CONDITION" == "few_shot" ]]; then
  binds+=(
    --bind "$FEW_VIEW:/auditops-vlm/few-shot:ro"
    --bind "$EVALUATION_PDF_MANIFEST:/auditops-vlm/evaluation-pdf-manifest.json:ro"
    --bind "$FEW_SHOT_PDF_MANIFEST:/auditops-vlm/few-shot-pdf-manifest.json:ro"
  )
  cli_args+=(
    --few-shot-diagnostic-dir /auditops-vlm/few-shot
    --visual-examples-jsonl /auditops-vlm/few-shot/visual_examples.jsonl
    --few-shot-approval /auditops-vlm/few-shot/approval.json
    --few-shot-pdf-root "$FEW_SHOT_PDF_SOURCE_ROOT"
  )
else
  binds+=(--bind "$EVALUATION_PDF_MANIFEST:/auditops-vlm/evaluation-pdf-manifest.json:ro")
fi

APPTAINER_BIN=$(command -v apptainer)
if [[ -z "$APPTAINER_BIN" ]]; then
  echo "Apptainer is unavailable" >&2
  exit 2
fi
for environment_name in $(compgen -e); do
  case "$environment_name" in APPTAINER_*|SINGULARITY_*) unset "$environment_name" ;; esac
done
mkdir -p "$RUNTIME_TMP/home" "$RUNTIME_TMP/container-tmp"
launcher_env=(
  "HOME=$RUNTIME_TMP/home"
  "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  "LANG=C.UTF-8"
  "LC_ALL=C.UTF-8"
  "TMPDIR=$RUNTIME_TMP/container-tmp"
)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then launcher_env+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"); fi
AUDITOPS_HOST_NETNS=$(readlink /proc/self/ns/net)

env_args=(
  --env PYTHONDONTWRITEBYTECODE=1
  --env HF_HUB_OFFLINE=1
  --env TRANSFORMERS_OFFLINE=1
  --env HF_HUB_DISABLE_TELEMETRY=1
  --env VLLM_NO_USAGE_STATS=1
  --env AUDITOPS_GIT_COMMIT="$AUDITOPS_GIT_COMMIT"
  --env AUDITOPS_SOURCE_TREE_SHA256="$AUDITOPS_SOURCE_TREE_SHA256"
  --env AUDITOPS_SOURCE_MANIFEST_SHA256="$AUDITOPS_SOURCE_MANIFEST_SHA256"
  --env AUDITOPS_CONTAINER_SHA256="$AUDITOPS_CONTAINER_SHA256"
  --env AUDITOPS_MODEL_CONFIG_SHA256="$AUDITOPS_MODEL_CONFIG_SHA256"
  --env AUDITOPS_PACKAGE_LOCK_SHA256="$AUDITOPS_PACKAGE_LOCK_SHA256"
  --env AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256="$AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256"
  --env AUDITOPS_MODEL_PATH="$MODEL_PATH"
  --env AUDITOPS_GPU_INVENTORY="$AUDITOPS_GPU_INVENTORY"
  --env AUDITOPS_HOST_NETNS="$AUDITOPS_HOST_NETNS"
  --env AUDITOPS_EVALUATION_PDF_ROOT="$EVALUATION_PDF_SOURCE_ROOT"
  --env AUDITOPS_EVALUATION_PDF_MANIFEST=/auditops-vlm/evaluation-pdf-manifest.json
  --env AUDITOPS_EVALUATION_PDF_MANIFEST_SHA256="$EVALUATION_PDF_MANIFEST_SHA256"
  --env AUDITOPS_FEW_SHOT_PDF_ROOT="$FEW_SHOT_PDF_SOURCE_ROOT"
  --env AUDITOPS_FEW_SHOT_PDF_MANIFEST=/auditops-vlm/few-shot-pdf-manifest.json
  --env AUDITOPS_FEW_SHOT_PDF_MANIFEST_SHA256="$FEW_SHOT_PDF_MANIFEST_SHA256"
  --env SLURM_JOB_ID="$SLURM_JOB_ID"
  --env SLURM_JOB_PARTITION="$SLURM_JOB_PARTITION"
  --env SLURMD_NODENAME="${SLURMD_NODENAME:-unknown}"
  --env TMPDIR=/auditops-tmp
)

if ! env -i "${launcher_env[@]}" "$APPTAINER_BIN" exec --help | grep -q -- '--no-mount'; then
  echo "Installed Apptainer lacks required --no-mount isolation support" >&2
  exit 2
fi
isolation_args=(
  --cleanenv --containall --net --network none --nv
  --no-mount bind-paths,hostfs,cwd
)
env -i "${launcher_env[@]}" "$APPTAINER_BIN" exec \
  "${isolation_args[@]}" "${binds[@]}" "${env_args[@]}" --pwd /workspace/AuditOps "$AUDITOPS_VLLM_SIF" \
  bash -c '
set -euo pipefail
original_diagnostic=$1
original_gold=$2
shift 2
command -v pdftoppm >/dev/null || { echo "Pinned VLM image lacks pdftoppm" >&2; exit 2; }
[[ "$(python3 -I -c "import vllm; print(vllm.__version__)")" == "0.26.0" ]] || { echo "Pinned image does not contain vLLM 0.26.0" >&2; exit 2; }
python3 -I -S - "$original_diagnostic" "$original_gold" <<"PY"
import errno
import hashlib
import json
import os
import socket
import stat
import sys
from pathlib import Path, PurePosixPath


def reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"Staged PDF manifest contains duplicate JSON key: {key}")
        value[key] = item
    return value


def read_regular_once(path, label):
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise SystemExit(f"{label} is not a non-symlink regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_mode, before.st_size) != (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_size):
            raise SystemExit(f"{label} changed before it was opened")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    stable_before = (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns)
    stable_open = (after_open.st_dev, after_open.st_ino, after_open.st_mode, after_open.st_size, after_open.st_mtime_ns)
    stable_after = (after_path.st_dev, after_path.st_ino, after_path.st_mode, after_path.st_size, after_path.st_mtime_ns)
    if os.name == "posix":
        stable_before += (before.st_ctime_ns,)
        stable_open += (after_open.st_ctime_ns,)
        stable_after += (after_path.st_ctime_ns,)
    if stable_before != stable_open or stable_before != stable_after or len(payload) != before.st_size:
        raise SystemExit(f"{label} changed while it was read")
    return payload


def validate_pdf_mount(root_value, manifest_value, expected_manifest_sha256, label):
    if not root_value or not expected_manifest_sha256:
        raise SystemExit(f"{label} PDF mount provenance is incomplete")
    root = Path(root_value)
    if root.is_symlink() or not root.is_dir():
        raise SystemExit(f"{label} PDF mount root is not a non-symlink directory")
    manifest_path = Path(manifest_value)
    manifest_payload = read_regular_once(manifest_path, f"{label} PDF manifest")
    if hashlib.sha256(manifest_payload).hexdigest() != expected_manifest_sha256:
        raise SystemExit(f"{label} PDF manifest differs from its host-stage digest")
    try:
        manifest = json.loads(manifest_payload.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} PDF manifest is invalid: {exc}") from exc
    if not isinstance(manifest, dict) or set(manifest) != {"manifest_version", "file_count", "files"} or manifest.get("manifest_version") != "auditops-vlm-pdf-stage.v1":
        raise SystemExit(f"{label} PDF manifest schema/version is invalid")
    rows = manifest.get("files")
    if not isinstance(rows, list) or manifest.get("file_count") != len(rows) or not rows:
        raise SystemExit(f"{label} PDF manifest file_count is invalid")
    expected_paths = []
    seen_tasks = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"task_id", "path", "size", "sha256"}:
            raise SystemExit(f"{label} PDF manifest contains an invalid row")
        path_value = row["path"]
        path = PurePosixPath(path_value) if isinstance(path_value, str) else None
        if path is None or path.is_absolute() or path.as_posix() != path_value or any(part in {"", ".", ".."} for part in path.parts) or path.suffix.lower() != ".pdf":
            raise SystemExit(f"{label} PDF manifest contains an unsafe/non-PDF path")
        if path_value in expected_paths or row["task_id"] in seen_tasks:
            raise SystemExit(f"{label} PDF manifest contains duplicate paths/tasks")
        expected_paths.append(path_value)
        seen_tasks.add(row["task_id"])
        candidate = root.joinpath(*path.parts)
        payload = read_regular_once(candidate, f"{label} staged PDF {path_value}")
        if len(payload) != row["size"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise SystemExit(f"{label} staged PDF bytes differ from the host-stage manifest")
    actual_paths = []
    for current, directories, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(directories):
            candidate = current_path / name
            if candidate.is_symlink() or not candidate.is_dir():
                raise SystemExit(f"{label} PDF mount exposes a symlink/special directory")
        for name in sorted(names):
            candidate = current_path / name
            if candidate.is_symlink() or not stat.S_ISREG(candidate.lstat().st_mode) or candidate.suffix.lower() != ".pdf":
                raise SystemExit(f"{label} PDF mount exposes adjacent XHTML/text/audit or special files")
            actual_paths.append(candidate.relative_to(root).as_posix())
    if sorted(actual_paths) != sorted(expected_paths):
        raise SystemExit(f"{label} PDF mount exposes files outside the exact frozen PDF set")

for raw in sys.argv[1:]:
    if Path(raw).exists():
        raise SystemExit("Original diagnostic/gold path is visible in the inference container")
if Path("/auditops-vlm/evaluation/gold.jsonl").exists():
    raise SystemExit("Evaluator-only evaluation gold is visible")
validate_pdf_mount(
    os.environ.get("AUDITOPS_EVALUATION_PDF_ROOT"),
    os.environ.get("AUDITOPS_EVALUATION_PDF_MANIFEST"),
    os.environ.get("AUDITOPS_EVALUATION_PDF_MANIFEST_SHA256"),
    "Evaluation",
)
if os.environ.get("AUDITOPS_FEW_SHOT_PDF_ROOT"):
    validate_pdf_mount(
        os.environ.get("AUDITOPS_FEW_SHOT_PDF_ROOT"),
        os.environ.get("AUDITOPS_FEW_SHOT_PDF_MANIFEST"),
        os.environ.get("AUDITOPS_FEW_SHOT_PDF_MANIFEST_SHA256"),
        "Few-shot",
    )
for name in os.environ:
    if name in {"COMPANIES_HOUSE_API_KEY", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY", "LD_PRELOAD", "PYTHONPATH"} or name.startswith(("APPTAINERENV_", "SINGULARITYENV_")):
        raise SystemExit(f"Forbidden environment reached inference: {name}")
if os.readlink("/proc/self/ns/net") == os.environ["AUDITOPS_HOST_NETNS"]:
    raise SystemExit("Apptainer did not create a separate network namespace")
if {name for _, name in socket.if_nameindex()} - {"lo"}:
    raise SystemExit("Inference network namespace exposes a non-loopback interface")
denied = {errno.EACCES, errno.EPERM, errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT, errno.EPROTONOSUPPORT}
for family, address in ((socket.AF_INET, ("192.0.2.1", 9)), (socket.AF_INET6, ("2001:db8::1", 9, 0, 0))):
    sock = None
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.settimeout(0.25)
        sock.connect(address)
    except OSError as exc:
        if exc.errno not in denied:
            raise SystemExit(f"Network probe returned an unapproved error: {exc!r}") from exc
    else:
        raise SystemExit("Network probe unexpectedly connected")
    finally:
        if sock is not None:
            sock.close()
PY
python3 -I -S - <<"PY"
import hashlib
import os
import sys
from pathlib import Path
sys.path.insert(0, "/workspace/AuditOps")
from auditops.provenance import verify_model_snapshot
verify_model_snapshot(os.environ["AUDITOPS_MODEL_PATH"], os.environ["AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256"], require_read_only=True)
if hashlib.sha256(Path("/auditops-vlm/model.json").read_bytes()).hexdigest() != os.environ["AUDITOPS_MODEL_CONFIG_SHA256"]:
    raise SystemExit("In-container model config differs from its external pin")
PY
exec python3 -m auditops.cli run-companies-house-vlm-diagnostic "$@"
' bash "$AUDITOPS_CH_VLM_DIR" "$AUDITOPS_CH_VLM_DIR/gold.jsonl" "${cli_args[@]}"
