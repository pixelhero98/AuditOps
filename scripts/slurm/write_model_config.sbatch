#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AUDITOPS_REPO_ROOT="${AUDITOPS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$AUDITOPS_REPO_ROOT/scripts/env.sh"
source "$AUDITOPS_REPO_ROOT/scripts/slurm/agent_ops_common.sh"

: "${AUDITOPS_HF_MODEL_ID:?Set AUDITOPS_HF_MODEL_ID}"
: "${AUDITOPS_HF_REVISION:?Set AUDITOPS_HF_REVISION}"
: "${AUDITOPS_MODEL_SNAPSHOT_DIR:?Set AUDITOPS_MODEL_SNAPSHOT_DIR}"
: "${AUDITOPS_MODEL_QUANTIZATION:?Set AUDITOPS_MODEL_QUANTIZATION, using none for BF16}"
: "${AUDITOPS_MODEL_CONFIG_OUTPUT:?Set AUDITOPS_MODEL_CONFIG_OUTPUT}"
EXPECTED_MODEL_CONFIG_SHA256_FILE="$AUDITOPS_MODEL_CONFIG_OUTPUT.sha256"
if [[ -n "${AUDITOPS_MODEL_CONFIG_SHA256_FILE:-}" && \
      "$AUDITOPS_MODEL_CONFIG_SHA256_FILE" != "$EXPECTED_MODEL_CONFIG_SHA256_FILE" ]]; then
  echo "Model-config checksum record must be colocated as <model-config>.sha256" >&2
  exit 2
fi
AUDITOPS_MODEL_CONFIG_SHA256_FILE="$EXPECTED_MODEL_CONFIG_SHA256_FILE"

auditops_require_git_commit AUDITOPS_HF_REVISION
auditops_assert_new_path "$AUDITOPS_MODEL_CONFIG_OUTPUT" "model config"
auditops_assert_new_path "$AUDITOPS_MODEL_CONFIG_SHA256_FILE" "model-config checksum record"
if [[ ! -f "$AUDITOPS_MODEL_SNAPSHOT_DIR/AUDITOPS_SNAPSHOT_COMPLETE" ]]; then
  echo "Model snapshot is not finalized: $AUDITOPS_MODEL_SNAPSHOT_DIR" >&2
  exit 2
fi

auditops_start_log auditops-model-config
source "$AUDITOPS_REPO_ROOT/scripts/activate_env.sh"
python -m auditops.cli write-model-config \
  --output "$AUDITOPS_MODEL_CONFIG_OUTPUT" \
  --model-id "$AUDITOPS_HF_MODEL_ID" \
  --revision "$AUDITOPS_HF_REVISION" \
  --model-path "$AUDITOPS_MODEL_SNAPSHOT_DIR" \
  --quantization "$AUDITOPS_MODEL_QUANTIZATION"

MODEL_CONFIG_SHA256=$(python -I -S - \
  "$AUDITOPS_MODEL_CONFIG_OUTPUT" "$AUDITOPS_MODEL_CONFIG_SHA256_FILE" <<'MODEL_CONFIG_DIGEST_PY'
import hashlib
import os
import sys
import tempfile
from pathlib import Path

config_path = Path(sys.argv[1]).resolve(strict=True)
output_path = Path(sys.argv[2])
digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
record = f"{digest}  {config_path.name}\n".encode("ascii")
output_path.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
)
temporary_path = Path(temporary_name)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(record)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary_path, output_path)
    except FileExistsError as exc:
        raise SystemExit(f"Refusing to overwrite model-config checksum record: {output_path}") from exc
finally:
    temporary_path.unlink(missing_ok=True)
print(digest)
MODEL_CONFIG_DIGEST_PY
)
auditops_require_sha256 MODEL_CONFIG_SHA256
chmod 0444 "$AUDITOPS_MODEL_CONFIG_OUTPUT" "$AUDITOPS_MODEL_CONFIG_SHA256_FILE"
echo "MODEL_CONFIG=$AUDITOPS_MODEL_CONFIG_OUTPUT"
echo "MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256"
