"""Synthetic structured-output probes for the pinned offline model backend."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .agent_contracts import semantic_model_config, validate_model_config
from .canonical_json import canonical_json_sha256
from .model_adapter import ModelAdapterError, ModelRequest, OfflineVLLMAdapter
from .provenance import assert_no_secrets

GRAMMAR_PROBE_VERSION = "auditops.structured_output_probe.v2.2"


def run_structured_output_probes(
    model_config_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Run four non-filing grammar probes and publish an immutable result."""

    config_path = Path(model_config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_model_config(config)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite probe directory: {destination}")
    probes = (
        (
            "constant",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["status"],
                "properties": {"status": {"const": "READY"}},
            },
            "Return the required synthetic status object.",
        ),
        (
            "tool_action",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["action", "tool_name", "tool_arguments"],
                "properties": {
                    "action": {"const": "CALL_TOOL"},
                    "tool_name": {"const": "synthetic_noop"},
                    "tool_arguments": {"const": {"fixture": 1}},
                },
            },
            "Return the registered synthetic tool action.",
        ),
        (
            "terminal_union",
            {
                "type": "object",
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["action", "value"],
                        "properties": {
                            "action": {"const": "ANSWER"},
                            "value": {"const": "2"},
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["action", "refusal_code"],
                        "properties": {
                            "action": {"const": "REFUSE"},
                            "refusal_code": {"const": "SYNTHETIC_REFUSAL"},
                        },
                    },
                ],
            },
            "Return the synthetic answer branch with value 2.",
        ),
        (
            "narrative_selection",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["action", "extracts"],
                "properties": {
                    "action": {"const": "ANSWER"},
                    "extracts": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["evidence_id", "exact_quote"],
                            "properties": {
                                "evidence_id": {"const": "synthetic-1"},
                                "exact_quote": {"const": "Synthetic exact quote."},
                            },
                        },
                    },
                },
            },
            "Select the synthetic exact quote from evidence synthetic-1.",
        ),
    )
    adapter = OfflineVLLMAdapter(config)
    semantic = semantic_model_config(config)
    rows: list[dict[str, Any]] = []
    for name, schema, instruction in probes:
        request = ModelRequest(
            request_id=f"probe:{name}",
            messages=(
                {
                    "role": "system",
                    "content": "This is a synthetic JSON grammar diagnostic with no filing data.",
                },
                {"role": "user", "content": instruction},
            ),
            json_schema=schema,
            max_tokens=128,
            stage="direct",
            repair_attempt=False,
            prompt_version="auditops.agent_prompt.v2.2",
            runtime_version="auditops.agent_runtime.v2.2",
            model_semantic_config=semantic,
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            seed=int(config["seed"]),
        )
        try:
            response = adapter.generate_json(request)
        except ModelAdapterError as exc:
            rows.append(
                {
                    "probe": name,
                    "passed": False,
                    "failure_type": type(exc).__name__,
                    "request_sha256": request.request_sha256,
                    "response_schema_sha256": request.response_schema_sha256,
                }
            )
            break
        rows.append(
            {
                "probe": name,
                "passed": True,
                "failure_type": None,
                "request_sha256": request.request_sha256,
                "response_schema_sha256": request.response_schema_sha256,
                "response_sha256": response.response_sha256,
                "payload": dict(response.payload),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "finish_reason": response.finish_reason,
                "metadata": dict(response.metadata),
            }
        )
    report = {
        "grammar_probe_version": GRAMMAR_PROBE_VERSION,
        "model_id": config["model_id"],
        "revision": config["revision"],
        "model_config_sha256": canonical_json_sha256(config),
        "constraint_backend": adapter.constraint_backend,
        "passed": len(rows) == len(probes) and all(row["passed"] for row in rows),
        "probes": rows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        (temporary / "probe_results.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        assert_no_secrets([temporary])
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


__all__ = ["GRAMMAR_PROBE_VERSION", "run_structured_output_probes"]
