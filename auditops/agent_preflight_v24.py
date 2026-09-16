"""CPU-only exact-token preflight for AuditOps v2.4 narrative requests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .agent_batch import (
    _artifact_path,
    _runtime_contract_version,
    _validate_v2_benchmark_manifest,
)
from .agent_context import build_context_pack
from .agent_contracts import validate_agent_task_input
from .agent_operations import validate_task_operation_binding
from .agent_prompts import build_synthesis_prompt
from .agent_tools import load_frozen_evidence
from .canonical_json import canonical_json_bytes, canonical_json_sha256
from .model_config import _chat_template_bytes
from .provenance import assert_no_secrets

PREFLIGHT_VERSION_V24 = "auditops-agent-request-preflight.v2.4"
PREFLIGHT_VERSION_V24P = "auditops-agent-request-preflight.v2.4p"
BENCHMARK_VERSION_V24P = "agent_benchmark.v2.4p"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected object in {path} line {line_number}")
            rows.append(value)
    return rows


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise TypeError("Pinned tokenizer did not return one integer token sequence")
    return value


def _runtime_sha256(name: str) -> str | None:
    value = os.environ.get(name)
    if value is not None and (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _load_verified_preflight_inputs(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Verify the frozen inference boundary without importing builder dependencies."""

    manifest = _read_json(root / "benchmark_manifest.json")
    _validate_v2_benchmark_manifest(manifest)
    if manifest.get("benchmark_version") not in {
        "agent_benchmark.v2.4",
        BENCHMARK_VERSION_V24P,
    }:
        raise ValueError("Exact v2.4 preflight requires a v2.4-family benchmark")
    if _runtime_contract_version(manifest) != "v2.3":
        raise ValueError("v2.4-family preflight requires runtime contract v2.3")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("Benchmark manifest must bind its frozen artifacts")
    for name in artifacts:
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Benchmark artifact names must be safe basenames")
        _artifact_path(root, manifest, name)
    visibility = manifest.get("artifact_visibility")
    if not isinstance(visibility, Mapping):
        raise TypeError("Benchmark must declare artifact visibility")
    inference_visible = set(visibility.get("inference_visible", []))
    evaluator_only = set(visibility.get("evaluator_only", []))
    if not {"cases.jsonl", "evidence.jsonl", "gate_50.json"}.issubset(
        inference_visible
    ) or inference_visible.intersection(evaluator_only):
        raise ValueError("Benchmark inference/evaluator visibility boundary is invalid")

    cases = _read_jsonl(root / "cases.jsonl")
    evidence_rows = _read_jsonl(root / "evidence.jsonl")
    counts = manifest.get("counts")
    if (
        not isinstance(counts, Mapping)
        or counts.get("total") != 500
        or counts.get("quant") != 300
        or counts.get("narrative") != 200
        or len(cases) != 500
        or len(evidence_rows) != 500
    ):
        raise ValueError("Frozen v2.4 benchmark counts are invalid")
    case_ids = [str(row.get("task_id")) for row in cases]
    evidence_ids = [str(row.get("task_id")) for row in evidence_rows]
    if len(set(case_ids)) != 500 or set(evidence_ids) != set(case_ids):
        raise ValueError("Cases and evidence must have identical unique membership")
    for case in cases:
        validate_agent_task_input(case)
        validate_task_operation_binding(case)
    evidence_by_id = {str(row["task_id"]): row["items"] for row in evidence_rows}
    gate = _read_json(root / "gate_50.json")
    gate_ids = gate.get("task_ids")
    if (
        gate.get("case_count") != 50
        or not isinstance(gate_ids, list)
        or len(gate_ids) != 50
        or len(set(gate_ids)) != 50
        or any(str(task_id) not in set(case_ids) for task_id in gate_ids)
    ):
        raise ValueError("Frozen narrative gate membership is invalid")
    if manifest["benchmark_version"] == BENCHMARK_VERSION_V24P and (
        manifest.get("assurance") != "PROVISIONAL_AI_REVIEW"
        or manifest.get("experiment_label") != "EXPLORATORY"
        or manifest.get("generalization_label") != "cached-20"
        or manifest.get("external_human_approval") is not False
    ):
        raise ValueError("v2.4p benchmark lacks provisional exploratory labels")
    return manifest, cases, evidence_by_id


def preflight_agent_requests_v24(
    benchmark_dir: str | Path,
    model_config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Render every exact zero-shot safety-hybrid narrative synthesis request."""

    root = Path(benchmark_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite preflight directory: {destination}"
        )
    manifest, cases, evidence_by_id = _load_verified_preflight_inputs(root)
    provisional = manifest.get("benchmark_version") == BENCHMARK_VERSION_V24P
    model_config_file = Path(model_config_path)
    model_config = _read_json(model_config_file)
    model_path = Path(str(model_config["model_path"]))
    expected_template_hash = str(model_config["chat_template_sha256"])
    if (
        hashlib.sha256(_chat_template_bytes(model_path)).hexdigest()
        != expected_template_hash
    ):
        raise ValueError("Pinned chat-template hash does not match model configuration")
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - remote dependency boundary
        raise RuntimeError(
            "Exact v2.4 preflight requires the pinned transformers package"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        revision=str(model_config["revision"]),
    )

    max_input_tokens = int(model_config["max_input_tokens"])

    def count_text(text: str) -> int:
        return len(_token_ids(tokenizer.encode(text, add_special_tokens=False)))

    records: list[dict[str, Any]] = []
    for case in cases:
        if case["task_type"] != "narrative_citation":
            continue
        task_id = str(case["task_id"])
        evidence = evidence_by_id[task_id]
        context = build_context_pack(
            case,
            evidence,
            token_counter=count_text,
            max_input_tokens=max_input_tokens,
        )
        observation = load_frozen_evidence(case, evidence, visibility="MODEL_VISIBLE")
        prompt = build_synthesis_prompt(
            context, observation, prompt_condition="zero_shot"
        )
        rendered = tokenizer.apply_chat_template(
            list(prompt.messages),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_count = len(_token_ids(rendered))
        records.append(
            {
                "task_id": task_id,
                "subtype": case["narrative_subtype"],
                "filing_id": case["filing"]["filing_id"],
                "context_sha256": context["context_sha256"],
                "prompt_sha256": prompt.prompt_sha256,
                "response_schema_sha256": canonical_json_sha256(prompt.json_schema),
                "input_tokens": input_count,
                "max_input_tokens": max_input_tokens,
                "within_limit": input_count < max_input_tokens,
            }
        )
    token_counts = sorted(int(row["input_tokens"]) for row in records)
    p95_index = max(0, min(len(token_counts) - 1, int((len(token_counts) - 1) * 0.95)))
    passed = bool(records) and all(row["within_limit"] for row in records)
    container_sha256 = _runtime_sha256("AUDITOPS_CONTAINER_SHA256")
    summary = {
        "preflight_version": (
            PREFLIGHT_VERSION_V24P if provisional else PREFLIGHT_VERSION_V24
        ),
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_manifest_sha256": _sha256_path(root / "benchmark_manifest.json"),
        "model_config_sha256": canonical_json_sha256(model_config),
        "model_id": model_config["model_id"],
        "model_revision": model_config["revision"],
        "tokenizer_runtime": {
            "transformers_version": str(transformers.__version__),
            "container_sha256": container_sha256,
            "model_config_file_sha256": _runtime_sha256(
                "AUDITOPS_MODEL_CONFIG_FILE_SHA256"
            ),
            "source_manifest_sha256": _runtime_sha256(
                "AUDITOPS_SOURCE_MANIFEST_SHA256"
            ),
            "source_tree_sha256": _runtime_sha256("AUDITOPS_SOURCE_TREE_SHA256"),
            "package_lock_sha256": _runtime_sha256("AUDITOPS_PACKAGE_LOCK_SHA256"),
            "offline": True,
            "network_disabled": os.environ.get("AUDITOPS_DISABLE_NETWORK") == "1",
        },
        "prompt_condition": "zero_shot",
        "runtime_mode": "safety_hybrid",
        "request_count": len(records),
        "input_token_limit": max_input_tokens,
        "input_tokens_max": max(token_counts) if token_counts else None,
        "input_tokens_p95": token_counts[p95_index] if token_counts else None,
        "strictly_below_limit": passed,
        "records_sha256": canonical_json_sha256(records),
        "assurance_status": (
            "PROVISIONAL_AI_REVIEW" if provisional else "EXTERNAL_HUMAN_APPROVED"
        ),
        "experiment_label": "EXPLORATORY" if provisional else "OFFICIAL_CANDIDATE",
        "corpus_scope_label": "cached-20",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        (temporary / "requests.jsonl").write_bytes(
            b"".join(canonical_json_bytes(row, newline=True) for row in records)
        )
        (temporary / "preflight.json").write_bytes(
            canonical_json_bytes(summary, newline=True)
        )
        (temporary / "report.md").write_text(
            (
                "# AuditOps v2.4p exploratory exact-token preflight\n\n"
                if provisional
                else "# AuditOps v2.4 exact-token preflight\n\n"
            )
            + (
                "- Assurance: PROVISIONAL_AI_REVIEW (not human-approved)\n"
                "- Experiment: EXPLORATORY, cached-20\n"
                if provisional
                else ""
            )
            + f"- Requests: {len(records)} narrative synthesis calls\n"
            f"- Maximum input tokens: {summary['input_tokens_max']}\n"
            f"- p95 input tokens: {summary['input_tokens_p95']}\n"
            f"- Locked limit: {max_input_tokens}\n"
            f"- Result: {'PASS' if passed else 'FAIL'}\n",
            encoding="utf-8",
            newline="\n",
        )
        assert_no_secrets([temporary])
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output_dir": str(destination),
        "passed": passed,
        "request_count": len(records),
        "input_tokens_max": summary["input_tokens_max"],
        "input_tokens_p95": summary["input_tokens_p95"],
    }


__all__ = [
    "PREFLIGHT_VERSION_V24",
    "PREFLIGHT_VERSION_V24P",
    "preflight_agent_requests_v24",
]
