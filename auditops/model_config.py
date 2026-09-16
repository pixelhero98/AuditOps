from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .agent_contracts import build_model_config
from .provenance import assert_no_secrets

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _chat_template_bytes(model_path: Path) -> bytes:
    tokenizer_config_path = model_path / "tokenizer_config.json"
    if tokenizer_config_path.is_file():
        try:
            tokenizer_config = json.loads(
                tokenizer_config_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid tokenizer config: {tokenizer_config_path}"
            ) from exc
        template = tokenizer_config.get("chat_template")
        if isinstance(template, str) and template:
            return template.encode("utf-8")
        if isinstance(template, (list, dict)) and template:
            return json.dumps(
                template, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
    template_path = model_path / "chat_template.jinja"
    if template_path.is_file() and template_path.stat().st_size:
        return template_path.read_bytes()
    raise ValueError(f"Pinned model snapshot has no chat template: {model_path}")


def create_pinned_model_config(
    *,
    model_id: str,
    revision: str,
    model_path: str | Path,
    quantization: str | None,
) -> dict[str, Any]:
    if not _COMMIT_RE.fullmatch(revision):
        raise ValueError("Model revision must be a 40-character lowercase commit hash")
    snapshot = Path(model_path).resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(
            f"Pinned model snapshot directory not found: {snapshot}"
        )
    template_sha256 = hashlib.sha256(_chat_template_bytes(snapshot)).hexdigest()
    return build_model_config(
        model_id=model_id,
        revision=revision,
        model_path=str(snapshot),
        quantization=quantization,
        backend="vllm_offline",
        context_window_tokens=16_384,
        max_input_tokens=14_336,
        max_plan_tokens=192,
        max_answer_tokens=768,
        max_repair_tokens=768,
        temperature=0.0,
        top_p=1.0,
        seed=20_260_821,
        thinking_enabled=False,
        chat_template_sha256=template_sha256,
    )


def write_pinned_model_config(
    output_path: str | Path,
    *,
    model_id: str,
    revision: str,
    model_path: str | Path,
    quantization: str | None,
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite model config: {output}")
    config = create_pinned_model_config(
        model_id=model_id,
        revision=revision,
        model_path=model_path,
        quantization=quantization,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        assert_no_secrets([temporary])
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite model config: {output}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return config
