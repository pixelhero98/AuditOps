from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agent_context import validate_evidence_items
from .agent_contracts import (
    semantic_model_config,
    validate_agent_case_result,
    validate_agent_proposal,
    validate_agent_task_input,
    validate_model_config,
    validate_tool_observation,
)
from .agent_operations import (
    expected_tool_arguments,
    task_operation_spec,
    validate_task_operation_binding,
)
from .agent_prompts import (
    MAX_DEMONSTRATION_TOKENS,
    render_demonstration_messages,
)
from .agent_runtime import run_agent_case
from .agent_tools import execute_deterministic_tool
from .canonical_json import canonical_json_bytes, canonical_json_sha256
from .model_adapter import ModelAdapter, OfflineVLLMAdapter
from .provenance import assert_no_secrets

BATCH_RUN_VERSION = "auditops-agent-batch.v2.2"
FEW_SHOT_APPROVAL_VERSION = "auditops-few-shot-approval.v2.2"
FEW_SHOT_REVIEW_PACKET_VERSION = "auditops-few-shot-review-packet.v2.2"
CHECKPOINT_VERSION = "auditops-agent-checkpoint.v2.2"
SUPPORTED_BENCHMARK_VERSION = "agent_benchmark.v2.2"
BENCHMARK_VERSION_V24P = "agent_benchmark.v2.4p"
SUPPORTED_BENCHMARK_VERSIONS = frozenset(
    {
        SUPPORTED_BENCHMARK_VERSION,
        "agent_benchmark.v2.3",
        "agent_benchmark.v2.4",
        BENCHMARK_VERSION_V24P,
    }
)
FULL_RUN_AUTHORIZATION_VERSION_V24 = "auditops-full-run-authorization.v2.4"
_FULL_RUN_AUTHORIZATION_FIELDS_V24 = frozenset(
    {
        "authorization_version",
        "benchmark_id",
        "gate_metrics_sha256",
        "decision",
        "authorized_by",
        "authorized_at",
    }
)

_RESULT_ROW_FIELDS = frozenset(
    {
        "agent_case_result_version",
        "task_id",
        "outcome",
        "plan_proposal",
        "final_proposal",
        "tool_observation",
        "run_record",
        "failure",
    }
)
_FEW_SHOT_ROW_FIELDS = frozenset(
    {
        "few_shot_example_version",
        "task_id",
        "example_id",
        "review_status",
        "template_id",
        "task_family",
        "task",
        "evidence_items",
        "plan_response",
        "tool_observation",
        "assistant_response",
        "source_task_sha256",
    }
)


def _validate_v2_benchmark_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("benchmark_version") not in SUPPORTED_BENCHMARK_VERSIONS:
        raise ValueError(
            "Text-agent v2 requires benchmark_version agent_benchmark.v2.2, "
            "v2.3, v2.4, or v2.4p"
        )


def _is_v24_family(manifest: Mapping[str, Any]) -> bool:
    return manifest.get("benchmark_version") in {
        "agent_benchmark.v2.4",
        BENCHMARK_VERSION_V24P,
    }


def _runtime_contract_version(manifest: Mapping[str, Any]) -> str:
    """Resolve the task/runtime contract independently of benchmark packaging."""

    benchmark_version = manifest.get("benchmark_version")
    declared = manifest.get("runtime_contract_version")
    if benchmark_version in {"agent_benchmark.v2.4", BENCHMARK_VERSION_V24P}:
        if declared != "v2.3":
            raise ValueError(
                "v2.4-family benchmarks must declare runtime_contract_version v2.3"
            )
        return "v2.3"
    if declared is not None:
        raise ValueError(
            "Legacy benchmarks cannot add a runtime_contract_version field"
        )
    return "v2.3" if benchmark_version == "agent_benchmark.v2.3" else "v2.2"


def _execution_version(manifest: Mapping[str, Any]) -> str:
    benchmark_version = str(manifest.get("benchmark_version"))
    return benchmark_version.removeprefix("agent_benchmark.")


def _validate_few_shot_row(row: Mapping[str, Any]) -> dict[str, Any]:
    version = row.get("few_shot_example_version")
    if version not in {"v2.2", "v2.3"}:
        raise ValueError("Few-shot examples must use version v2.2 or v2.3")
    if set(row) != _FEW_SHOT_ROW_FIELDS:
        raise ValueError("Few-shot example has an invalid versioned envelope")
    if row.get("review_status") != "PENDING_HUMAN_REVIEW":
        raise ValueError("Frozen few-shot review_status must be PENDING_HUMAN_REVIEW")
    for field in ("task_id", "example_id", "template_id", "task_family"):
        if not isinstance(row.get(field), str) or not str(row[field]).strip():
            raise ValueError(f"Few-shot {field} must be a non-empty string")
    task = row.get("task")
    evidence_items = row.get("evidence_items")
    response = row.get("assistant_response")
    if (
        not isinstance(task, Mapping)
        or not isinstance(evidence_items, list)
        or not all(isinstance(item, Mapping) for item in evidence_items)
        or not isinstance(response, Mapping)
    ):
        raise TypeError(
            "Few-shot rows require a task, evidence item array, and assistant response"
        )
    validate_agent_task_input(task)
    validate_task_operation_binding(task)
    validate_agent_proposal(response)
    validated_evidence = validate_evidence_items(task, evidence_items)
    plan_response = row.get("plan_response")
    tool_observation = row.get("tool_observation")
    if not isinstance(plan_response, Mapping) or not isinstance(
        tool_observation, Mapping
    ):
        raise TypeError("Few-shot plan_response and tool_observation must be objects")
    expected_plan = {
        "task_id": task["task_id"],
        "action": "CALL_TOOL",
        "tool_name": task_operation_spec(
            task["task_type"], version=str(version)
        ).operation,
        "tool_arguments": expected_tool_arguments(task),
    }
    if dict(plan_response) != expected_plan:
        raise ValueError("Few-shot plan_response does not match the operation registry")
    validate_tool_observation(tool_observation)
    if tool_observation.get("task_id") != task.get("task_id"):
        raise ValueError("Few-shot tool observation targets a different task")
    expected_observation = execute_deterministic_tool(
        expected_plan["tool_name"],
        expected_plan["tool_arguments"],
        task,
        validated_evidence,
    )
    if canonical_json_sha256(tool_observation) != canonical_json_sha256(
        expected_observation
    ):
        raise ValueError(
            "Few-shot tool observation does not match deterministic replay"
        )
    source_task_sha256 = row.get("source_task_sha256")
    if not isinstance(source_task_sha256, str) or len(source_task_sha256) != 64:
        raise ValueError("Few-shot source_task_sha256 must be a SHA-256 digest")
    task_id = task.get("task_id")
    if row.get("task_id") != task_id or response.get("task_id") != task_id:
        raise ValueError("Few-shot wrapper, task, and response IDs must match")
    if response.get("action") not in {"ANSWER", "REFUSE"}:
        raise ValueError("Few-shot assistant responses must be terminal proposals")
    evidence_ids = [item.get("evidence_id") for item in validated_evidence]
    if evidence_ids != task["evidence_scope"]["evidence_ids"]:
        raise ValueError("Few-shot evidence order must match the frozen task scope")
    task_family = str(row["task_family"])
    if task_family != task["task_type"] and not task_family.startswith(
        f"{task['task_type']}:"
    ):
        raise ValueError("Few-shot task_family must be bound to task.task_type")
    return {
        "few_shot_example_version": str(version),
        "example_id": str(row["example_id"]),
        "template_id": str(row["template_id"]),
        "task_family": task_family,
        "task": dict(task),
        "evidence_items": validated_evidence,
        "plan_response": dict(plan_response),
        "tool_observation": dict(tool_observation),
        "assistant_response": dict(response),
        "source_task_sha256": source_task_sha256,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object in {path} line {line_number}")
            task_id = row.get("task_id")
            if isinstance(task_id, str):
                if task_id in seen:
                    raise ValueError(f"Duplicate task_id {task_id!r} in {path}")
                seen.add(task_id)
            rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_bytes_durable(
    path: Path, payload: bytes, *, staging_dir: Path | None = None
) -> None:
    """Write a new file durably and refuse to replace an existing artifact."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = staging_dir or path.parent
    temporary_parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(temporary_parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        assert_no_secrets([temporary])
        try:
            # A hard-link publish is atomic and, unlike os.replace(), cannot
            # clobber a checkpoint written by a concurrent invocation. Case
            # temporaries live outside case_results, so a hard interruption can
            # leave debris without making the checkpoint unreadable.
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite immutable artifact: {path}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl_durable(path: Path, row: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(dict(row), newline=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as staged:
            staged.write(payload)
            staged.flush()
            os.fsync(staged.fileno())
        assert_no_secrets([temporary])
        with path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _checkpoint_case_name(index: int, task_id: str) -> str:
    task_digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]
    return f"{index:06d}-{task_digest}.json"


def _validate_checkpoint_row(
    row: Mapping[str, Any],
    *,
    expected_task_id: str,
    benchmark_manifest_sha256: str,
    corpus_id: str,
    runtime_mode: str,
    prompt_condition: str,
    model_config_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if set(row) != _RESULT_ROW_FIELDS:
        raise ValueError(
            f"Checkpoint result for {expected_task_id} has an invalid schema"
        )
    if row.get("task_id") != expected_task_id:
        raise ValueError(f"Checkpoint task binding mismatch for {expected_task_id}")
    validate_agent_case_result(row)
    run_record = row.get("run_record")
    if not isinstance(run_record, Mapping):
        raise TypeError(f"Checkpoint result for {expected_task_id} lacks a run record")
    if run_record.get("task_id") != expected_task_id:
        raise ValueError(f"Checkpoint run record task mismatch for {expected_task_id}")
    expected_bindings = {
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
        "corpus_id": corpus_id,
        "runtime_mode": runtime_mode,
        "prompt_condition": prompt_condition,
        "provenance": dict(provenance),
    }
    for field, expected in expected_bindings.items():
        if run_record.get(field) != expected:
            raise ValueError(
                f"Checkpoint run record {field} mismatch for {expected_task_id}"
            )
    run_model_config = run_record.get("model_config")
    if run_model_config is None and runtime_mode == "safety_hybrid":
        pass
    elif canonical_json_sha256(run_model_config) != model_config_sha256:
        raise ValueError(
            f"Checkpoint model configuration mismatch for {expected_task_id}"
        )
    return dict(row)


def _load_checkpoint_rows(
    checkpoint_dir: Path,
    selected_task_ids: Sequence[str],
    *,
    benchmark_manifest_sha256: str,
    corpus_id: str,
    runtime_mode: str,
    prompt_condition: str,
    model_config_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result_dir = checkpoint_dir / "case_results"
    if not result_dir.is_dir():
        raise ValueError(f"Checkpoint result directory is missing: {result_dir}")
    expected_names = {
        _checkpoint_case_name(index, task_id): task_id
        for index, task_id in enumerate(selected_task_ids)
    }
    entries = list(result_dir.iterdir())
    non_files = sorted(path.name for path in entries if not path.is_file())
    if non_files:
        raise ValueError(
            "Checkpoint result directory contains non-file artifacts: "
            + ", ".join(non_files)
        )
    actual_names = {path.name for path in entries}
    unexpected = sorted(actual_names - set(expected_names))
    if unexpected:
        raise ValueError(
            "Checkpoint contains unexpected result artifacts: " + ", ".join(unexpected)
        )
    rows: dict[str, dict[str, Any]] = {}
    for name, task_id in expected_names.items():
        path = result_dir / name
        if not path.exists():
            continue
        row = _read_json(path)
        rows[task_id] = _validate_checkpoint_row(
            row,
            expected_task_id=task_id,
            benchmark_manifest_sha256=benchmark_manifest_sha256,
            corpus_id=corpus_id,
            runtime_mode=runtime_mode,
            prompt_condition=prompt_condition,
            model_config_sha256=model_config_sha256,
            provenance=provenance,
        )
    return rows


def _append_batch_failure(
    checkpoint_dir: Path,
    *,
    binding_sha256: str,
    task_id: str | None,
    case_index: int | None,
    stage: str,
    exception: BaseException,
    completed_case_count: int,
    checkpoint_version: str = CHECKPOINT_VERSION,
) -> None:
    # Never persist exception messages: backend errors can contain credentials or
    # raw request material.  The exception class is sufficient for diagnosis and
    # the original exception is re-raised to the caller with its traceback.
    _append_jsonl_durable(
        checkpoint_dir / "batch_failure_manifests.jsonl",
        {
            "failure_manifest_version": checkpoint_version,
            "recorded_at": _utc_now(),
            "binding_sha256": binding_sha256,
            "task_id": task_id,
            "case_index": case_index,
            "stage": stage,
            "exception_type": type(exception).__name__,
            "completed_case_count": completed_case_count,
            "checkpoint_dir": checkpoint_dir.name,
        },
    )


def _add_checkpoint_note(exception: BaseException, checkpoint_dir: Path) -> None:
    exception.add_note(
        f"AuditOps retained resumable case checkpoints at {checkpoint_dir}"
    )


def _artifact_path(root: Path, manifest: Mapping[str, Any], name: str) -> Path:
    metadata = manifest.get("artifacts", {}).get(name)
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("sha256"), str):
        raise TypeError(
            f"Benchmark manifest does not declare inference artifact {name}"
        )
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark artifact not found: {path}")
    actual = _sha256_file(path)
    if actual != metadata["sha256"]:
        raise ValueError(f"Benchmark artifact checksum mismatch: {name}")
    return path


def write_few_shot_approval(
    benchmark_dir: str | Path,
    output_path: str | Path,
    *,
    reviewer: str,
    approved_example_ids: Sequence[str],
    reviewed_at: str,
    review_packet_path: str | Path,
) -> dict[str, Any]:
    root = Path(benchmark_dir)
    manifest = _read_json(root / "benchmark_manifest.json")
    _validate_v2_benchmark_manifest(manifest)
    contract_version = (
        "v2.3" if manifest["benchmark_version"] == "agent_benchmark.v2.3" else "v2.2"
    )
    approval_version = f"auditops-few-shot-approval.{contract_version}"
    review_packet_version = f"auditops-few-shot-review-packet.{contract_version}"
    few_shot_path = _artifact_path(root, manifest, "few_shot.jsonl")
    examples = _read_jsonl(few_shot_path)
    for row in examples:
        _validate_few_shot_row(row)
    if len(examples) != 20:
        raise ValueError("Few-shot approval requires exactly twenty versioned examples")
    expected = sorted(str(row.get("example_id")) for row in examples)
    approved = sorted(set(approved_example_ids))
    if approved != expected:
        raise ValueError(
            "Approval must list every and only the frozen few-shot example IDs"
        )
    if not reviewer.strip() or not reviewed_at.strip():
        raise ValueError("reviewer and reviewed_at must be non-empty")
    review_packet_file = Path(review_packet_path)
    review_packet = _read_json(review_packet_file)
    review_max_tokens = review_packet.get("max_demonstration_tokens")
    if (
        review_packet.get("few_shot_review_packet_version") != review_packet_version
        or review_packet.get("benchmark_id") != manifest.get("benchmark_id")
        or review_packet.get("few_shot_sha256") != _sha256_file(few_shot_path)
        or review_packet.get("passed") is not True
        or isinstance(review_max_tokens, bool)
        or not isinstance(review_max_tokens, int)
        or review_max_tokens > MAX_DEMONSTRATION_TOKENS
    ):
        raise ValueError("Review packet does not approve the frozen compact examples")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite few-shot approval: {output}")
    payload = {
        "few_shot_approval_version": approval_version,
        "benchmark_id": manifest["benchmark_id"],
        "few_shot_sha256": _sha256_file(few_shot_path),
        "review_packet_sha256": _sha256_file(review_packet_file),
        "review_binding_sha256": canonical_json_sha256(
            {
                "rendered_messages_sha256": review_packet.get(
                    "rendered_messages_sha256"
                ),
                "tokenizer_counts_sha256": review_packet.get("tokenizer_counts_sha256"),
            }
        ),
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "approved_example_ids": approved,
        "decision": "APPROVED",
    }
    _write_bytes_durable(
        output,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return payload


def _tokenizer_message_count(
    model_path: str, messages: Sequence[Mapping[str, str]]
) -> int:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - production image dependency
        raise RuntimeError(
            "transformers is required to render the few-shot review packet"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    token_ids = tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=True,
        add_generation_prompt=False,
    )
    if isinstance(token_ids, Mapping):
        token_ids = token_ids.get("input_ids")
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if (
        isinstance(token_ids, list)
        and len(token_ids) == 1
        and isinstance(token_ids[0], list)
    ):
        token_ids = token_ids[0]
    if not isinstance(token_ids, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in token_ids
    ):
        raise TypeError("Pinned tokenizer returned an invalid token sequence")
    return len(token_ids)


def write_few_shot_review_packet(
    benchmark_dir: str | Path,
    model_config_paths: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Render the exact demonstrations and count them with both pinned tokenizers."""

    root = Path(benchmark_dir)
    manifest = _read_json(root / "benchmark_manifest.json")
    _validate_v2_benchmark_manifest(manifest)
    contract_version = (
        "v2.3" if manifest["benchmark_version"] == "agent_benchmark.v2.3" else "v2.2"
    )
    few_shot_path = _artifact_path(root, manifest, "few_shot.jsonl")
    examples = [_validate_few_shot_row(row) for row in _read_jsonl(few_shot_path)]
    if len(examples) != 20:
        raise ValueError(
            "Review packet requires four quantitative and sixteen narrative examples"
        )
    configs = [_read_json(Path(path)) for path in model_config_paths]
    for config in configs:
        validate_model_config(config)
        if not isinstance(config.get("model_path"), str) or not config["model_path"]:
            raise ValueError("Review tokenizers require pinned local model paths")
    model_ids = {str(config["model_id"]) for config in configs}
    required_models = {
        "Qwen/Qwen3.5-27B-FP8",
        "google/gemma-4-31B-it-qat-w4a16-ct",
    }
    if model_ids != required_models or len(configs) != 2:
        raise ValueError("Review packet requires the pinned Qwen and Gemma tokenizers")

    rendered: dict[str, dict[str, list[dict[str, str]]]] = {}
    pack_names = ["quant_metric"] + sorted(
        {
            str(example["task"]["narrative_subtype"])
            for example in examples
            if example["task"]["task_type"] == "narrative_citation"
        }
    )
    for pack_name in pack_names:
        family_examples = [
            example
            for example in examples
            if (
                example["task"]["task_type"] == "quant_metric"
                and pack_name == "quant_metric"
            )
            or example["task"].get("narrative_subtype") == pack_name
        ]
        if len(family_examples) != 4:
            raise ValueError(f"Expected four {pack_name} demonstration examples")
        rendered[pack_name] = {
            stage: list(render_demonstration_messages(family_examples, stage=stage))
            for stage in ("direct", "plan", "synthesis")
        }

    counts: list[dict[str, Any]] = []
    for config in sorted(configs, key=lambda item: str(item["model_id"])):
        stage_counts = {
            task_type: {
                stage: _tokenizer_message_count(config["model_path"], messages)
                for stage, messages in stages.items()
            }
            for task_type, stages in rendered.items()
        }
        counts.append(
            {
                "model_id": config["model_id"],
                "revision": config["revision"],
                "chat_template_sha256": config["chat_template_sha256"],
                "semantic_model_config_sha256": canonical_json_sha256(
                    semantic_model_config(config)
                ),
                "counts": stage_counts,
            }
        )
    max_tokens = max(
        count
        for model in counts
        for family in model["counts"].values()
        for count in family.values()
    )
    packet = {
        "few_shot_review_packet_version": f"auditops-few-shot-review-packet.{contract_version}",
        "benchmark_id": manifest["benchmark_id"],
        "few_shot_sha256": _sha256_file(few_shot_path),
        "example_source_bindings": [
            {
                "example_id": row["example_id"],
                "source_task_sha256": row["source_task_sha256"],
                "task_sha256": canonical_json_sha256(row["task"]),
                "assistant_response_sha256": canonical_json_sha256(
                    row["assistant_response"]
                ),
            }
            for row in sorted(examples, key=lambda item: item["example_id"])
        ],
        "rendered_messages": rendered,
        "rendered_messages_sha256": canonical_json_sha256(rendered),
        "tokenizer_counts": counts,
        "tokenizer_counts_sha256": canonical_json_sha256(counts),
        "demonstration_token_cap": MAX_DEMONSTRATION_TOKENS,
        "max_demonstration_tokens": max_tokens,
        "passed": max_tokens <= MAX_DEMONSTRATION_TOKENS,
        "review_status": "PENDING_HUMAN_REVIEW",
    }
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite review packet: {output}")
    _write_bytes_durable(
        output,
        (json.dumps(packet, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return packet


def _validated_few_shot_examples(
    root: Path,
    manifest: Mapping[str, Any],
    approval_path: str | Path,
) -> list[dict[str, Any]]:
    few_shot_path = _artifact_path(root, manifest, "few_shot.jsonl")
    rows = _read_jsonl(few_shot_path)
    approval = _read_json(Path(approval_path))
    contract_version = (
        "v2.3" if manifest["benchmark_version"] == "agent_benchmark.v2.3" else "v2.2"
    )
    expected_ids = sorted(str(row.get("example_id")) for row in rows)
    if (
        approval.get("few_shot_approval_version")
        != f"auditops-few-shot-approval.{contract_version}"
    ):
        raise ValueError("Unsupported few-shot approval version")
    if approval.get("benchmark_id") != manifest.get("benchmark_id"):
        raise ValueError("Few-shot approval targets a different benchmark")
    if approval.get("few_shot_sha256") != _sha256_file(few_shot_path):
        raise ValueError("Few-shot approval does not match the frozen examples")
    for field in ("review_packet_sha256", "review_binding_sha256"):
        value = approval.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Few-shot approval lacks {field}")
    if approval.get("decision") != "APPROVED":
        raise ValueError("Few-shot approval decision is not APPROVED")
    if sorted(approval.get("approved_example_ids") or []) != expected_ids:
        raise ValueError("Few-shot approval does not cover every frozen example")
    examples = []
    for row in rows:
        examples.append(_validate_few_shot_row(row))
    if len(examples) != 20:
        raise ValueError("Few-shot baseline requires exactly twenty approved examples")
    return examples


def _index_task_rows(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{label} row lacks task_id")
        if task_id in result:
            raise ValueError(f"Duplicate {label} task_id: {task_id}")
        result[task_id] = dict(row)
    return result


def _validate_full_run_authorization_v24(
    authorization: Mapping[str, Any], *, benchmark_id: str
) -> dict[str, Any]:
    if set(authorization) != _FULL_RUN_AUTHORIZATION_FIELDS_V24:
        raise ValueError("Full-run authorization has an unexpected field set")
    digest = authorization.get("gate_metrics_sha256")
    if (
        authorization.get("authorization_version") != FULL_RUN_AUTHORIZATION_VERSION_V24
        or authorization.get("benchmark_id") != benchmark_id
        or authorization.get("decision") != "GO"
        or not isinstance(digest, str)
        or len(digest) != 64
        or digest.casefold() != digest
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Full-run authorization identity or GO decision is invalid")
    authorized_by = authorization.get("authorized_by")
    if (
        not isinstance(authorized_by, str)
        or not authorized_by.strip()
        or any(
            term in authorized_by.casefold() for term in ("codex", "chatgpt", "openai")
        )
    ):
        raise ValueError("Full-run authorization must come from an external human")
    authorized_at = authorization.get("authorized_at")
    if not isinstance(authorized_at, str):
        raise TypeError("Full-run authorization requires an RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(authorized_at)
    except ValueError as exc:
        raise ValueError(
            "Full-run authorization requires an RFC 3339 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError("Full-run authorization timestamp must include a timezone")
    return dict(authorization)


def run_agent_baseline(
    benchmark_dir: str | Path,
    model_config_path: str | Path,
    output_dir: str | Path,
    *,
    runtime_mode: str,
    prompt_condition: str,
    few_shot_approval: str | Path | None = None,
    limit: int | None = None,
    selection_mode: str | None = None,
    full_run_authorization: str | Path | None = None,
    expected_authorizer: str | None = None,
    adapter: ModelAdapter | None = None,
    strict_provenance: bool = True,
) -> dict[str, Any]:
    """Run or resume a frozen baseline and publish it only when it is complete.

    The sibling ``.<output-name>.checkpoint`` directory is intentionally stable
    across invocations.  Each completed case is written as a separate immutable,
    fsynced JSON object, so an interruption cannot erase earlier cases.  A rerun
    with identical bindings resumes missing cases; any changed binding is rejected.
    """

    root = Path(benchmark_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {destination}")
    manifest_path = root / "benchmark_manifest.json"
    manifest = _read_json(manifest_path)
    _validate_v2_benchmark_manifest(manifest)
    contract_version = _runtime_contract_version(manifest)
    execution_version = _execution_version(manifest)
    checkpoint_version = f"auditops-agent-checkpoint.{execution_version}"
    batch_run_version = f"auditops-agent-batch.{execution_version}"
    visibility = manifest.get("artifact_visibility")
    required_evaluator_only = {
        "gold.jsonl",
        "verifier_observations.jsonl",
        "case_lineage.jsonl",
    }
    if (
        not isinstance(visibility, Mapping)
        or not required_evaluator_only.issubset(
            set(visibility.get("evaluator_only", []))
        )
        or required_evaluator_only.intersection(
            set(visibility.get("inference_visible", []))
        )
    ):
        raise ValueError("Benchmark does not declare the evaluator-only boundary")
    cases_path = _artifact_path(root, manifest, "cases.jsonl")
    evidence_path = _artifact_path(root, manifest, "evidence.jsonl")
    parity_path: Path | None = None
    cases = _read_jsonl(cases_path)
    evidence_by_task = _index_task_rows(_read_jsonl(evidence_path), label="evidence")
    for case in cases:
        validate_agent_task_input(case)
        validate_task_operation_binding(case)
        if case.get("agent_task_input_version") != contract_version:
            raise ValueError(
                "Benchmark task version conflicts with runtime_contract_version"
            )
    if selection_mode not in {None, "full", "narrative_gate"}:
        raise ValueError("selection_mode must be full or narrative_gate")
    if selection_mode == "full" and limit is not None:
        raise ValueError("full selection cannot be combined with --limit")
    if selection_mode == "narrative_gate":
        if not _is_v24_family(manifest):
            raise ValueError(
                "narrative_gate selection requires a v2.4-family benchmark"
            )
        if limit is not None:
            raise ValueError("narrative_gate uses frozen gate_50.json, not --limit")
        parity_path = _artifact_path(root, manifest, "gate_50.json")
        gate = _read_json(parity_path)
        gate_ids = gate.get("task_ids")
        if (
            gate.get("case_count") != 50
            or not isinstance(gate_ids, list)
            or len(gate_ids) != 50
            or len(set(gate_ids)) != 50
        ):
            raise ValueError("The frozen gate_50.json must contain 50 unique task IDs")
        case_by_id = {str(case["task_id"]): case for case in cases}
        if any(str(task_id) not in case_by_id for task_id in gate_ids):
            raise ValueError("Narrative gate membership references unknown tasks")
        cases = [case_by_id[str(task_id)] for task_id in gate_ids]
    authorization_payload: bytes | None = None
    authorization_source: Path | None = None
    if _is_v24_family(manifest):
        resolved_selection = selection_mode or "full"
        if resolved_selection == "full":
            if full_run_authorization is None:
                raise ValueError(
                    "The v2.4 full 500-case run requires explicit go/no-go authorization"
                )
            authorization_source = Path(full_run_authorization)
            authorization_payload = authorization_source.read_bytes()
            authorization = json.loads(authorization_payload)
            if not isinstance(authorization, Mapping):
                raise TypeError("Full-run authorization must be a JSON object")
            if manifest["benchmark_version"] == BENCHMARK_VERSION_V24P:
                # Keep CPU-only benchmark/review dependencies out of the GPU
                # entrypoint's import graph until this branch is actually run.
                from .agent_provisional_v24p import (
                    validate_exploratory_authorization_v24p,
                )

                validate_exploratory_authorization_v24p(
                    authorization,
                    benchmark_id=str(manifest["benchmark_id"]),
                    expected_authorizer=expected_authorizer,
                )
            else:
                _validate_full_run_authorization_v24(
                    authorization, benchmark_id=str(manifest["benchmark_id"])
                )
            if authorization_source.read_bytes() != authorization_payload:
                raise ValueError("Full-run authorization changed during validation")
        elif full_run_authorization is not None:
            raise ValueError("Narrative-gate runs cannot carry full-run authorization")
    elif full_run_authorization is not None:
        raise ValueError(
            "Full-run authorization is only defined for v2.4-family benchmarks"
        )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if limit == 50:
            parity_path = _artifact_path(root, manifest, "parity_50.json")
            parity = _read_json(parity_path)
            parity_ids = parity.get("task_ids")
            if (
                parity.get("case_count") != 50
                or not isinstance(parity_ids, list)
                or len(parity_ids) != 50
                or len(set(parity_ids)) != 50
            ):
                raise ValueError(
                    "The frozen parity_50.json must contain exactly 50 unique task IDs"
                )
            case_by_id = {str(case["task_id"]): case for case in cases}
            missing_parity = sorted(set(parity_ids) - set(case_by_id))
            if missing_parity:
                raise ValueError("Parity membership references unknown benchmark tasks")
            cases = [case_by_id[str(task_id)] for task_id in parity_ids]
        else:
            cases = cases[:limit]

    approval_source: Path | None = None
    approval_payload: bytes | None = None
    if prompt_condition == "few_shot":
        if _is_v24_family(manifest):
            raise ValueError(
                "v2.4-family few-shot inference is blocked pending separate approval"
            )
        if few_shot_approval is None:
            raise ValueError(
                "few_shot runs require an explicit human approval manifest"
            )
        approval_source = Path(few_shot_approval)
        approval_payload = approval_source.read_bytes()
        examples = _validated_few_shot_examples(root, manifest, few_shot_approval)
        if approval_source.read_bytes() != approval_payload:
            raise ValueError("Few-shot approval changed while it was being validated")
    elif prompt_condition == "zero_shot":
        if few_shot_approval is not None:
            raise ValueError("zero_shot runs cannot use a few-shot approval")
        examples = []
    else:
        raise ValueError("prompt_condition must be zero_shot or few_shot")

    model_config_file = Path(model_config_path)
    model_config = _read_json(model_config_file)
    validate_model_config(model_config)
    if runtime_mode not in {"direct", "capability_agent", "safety_hybrid"}:
        raise ValueError(
            "runtime_mode must be direct, capability_agent, or safety_hybrid"
        )

    provenance = {
        "git_commit": os.environ.get("AUDITOPS_GIT_COMMIT"),
        "source_tree_sha256": os.environ.get("AUDITOPS_SOURCE_TREE_SHA256"),
        "container_digest": os.environ.get("AUDITOPS_CONTAINER_SHA256"),
        "model_snapshot_sha256": os.environ.get("AUDITOPS_MODEL_SNAPSHOT_SHA256"),
        "tokenizer_revision": model_config["revision"],
        "package_lock_sha256": os.environ.get("AUDITOPS_PACKAGE_LOCK_SHA256"),
    }
    if strict_provenance:
        missing = [key for key, value in provenance.items() if value is None]
        if missing:
            raise ValueError(f"Missing required run provenance: {', '.join(missing)}")
    resources = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_model": os.environ.get("AUDITOPS_GPU_MODEL"),
        "peak_vram_bytes": None,
        "host": os.environ.get("HOSTNAME"),
    }
    benchmark_manifest_sha256 = _sha256_file(manifest_path)

    selected_task_ids = [str(case["task_id"]) for case in cases]
    if not selected_task_ids:
        raise ValueError("The selected benchmark slice is empty")
    missing_evidence = sorted(set(selected_task_ids) - set(evidence_by_task))
    if missing_evidence:
        raise ValueError(
            f"Inference artifact coverage failure: evidence={len(missing_evidence)}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    model_config_sha256 = canonical_json_sha256(model_config)
    input_hashes = {
        "cases_sha256": _sha256_file(cases_path),
        "evidence_sha256": _sha256_file(evidence_path),
        "few_shot_sha256": _sha256_file(
            _artifact_path(root, manifest, "few_shot.jsonl")
        )
        if prompt_condition == "few_shot"
        else None,
        "few_shot_approval_sha256": hashlib.sha256(approval_payload).hexdigest()
        if approval_payload is not None
        else None,
        "parity_membership_sha256": _sha256_file(parity_path)
        if parity_path is not None
        else None,
        **(
            {
                (
                    "exploratory_run_authorization_sha256"
                    if manifest["benchmark_version"] == BENCHMARK_VERSION_V24P
                    else "full_run_authorization_sha256"
                ): hashlib.sha256(authorization_payload).hexdigest()
                if authorization_payload is not None
                else None
            }
            if _is_v24_family(manifest)
            else {}
        ),
    }
    checkpoint_binding = {
        "checkpoint_version": checkpoint_version,
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
        "corpus_id": manifest["corpus_id"],
        "runtime_mode": runtime_mode,
        "prompt_condition": prompt_condition,
        **(
            {"selection_mode": selection_mode or "full"}
            if _is_v24_family(manifest)
            else {}
        ),
        "model_config_sha256": model_config_sha256,
        "selected_task_ids": selected_task_ids,
        "task_ids_sha256": canonical_json_sha256(selected_task_ids),
        "inputs": input_hashes,
        "provenance": provenance,
    }
    binding_sha256 = canonical_json_sha256(checkpoint_binding)
    checkpoint_manifest = {
        **checkpoint_binding,
        "binding_sha256": binding_sha256,
    }
    checkpoint_dir = destination.parent / f".{destination.name}.checkpoint"
    checkpoint_manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    checkpoint_approval_path = checkpoint_dir / "few_shot_approval.json"

    if checkpoint_dir.exists():
        if not checkpoint_dir.is_dir() or not checkpoint_manifest_path.is_file():
            raise ValueError(f"Invalid or incomplete checkpoint: {checkpoint_dir}")
        existing_manifest = _read_json(checkpoint_manifest_path)
        if existing_manifest != checkpoint_manifest:
            raise ValueError(
                "Checkpoint bindings differ from this run; use a distinct output directory"
            )
        if approval_source is not None:
            if not checkpoint_approval_path.is_file():
                raise ValueError(
                    "Few-shot checkpoint lacks its immutable approval artifact"
                )
            if (
                _sha256_file(checkpoint_approval_path)
                != input_hashes["few_shot_approval_sha256"]
            ):
                raise ValueError("Few-shot checkpoint approval checksum mismatch")
        elif checkpoint_approval_path.exists():
            raise ValueError(
                "Zero-shot checkpoint unexpectedly contains an approval artifact"
            )
        assert_no_secrets([checkpoint_dir])
    else:
        checkpoint_staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.checkpoint-init.",
                dir=str(destination.parent),
            )
        )
        try:
            (checkpoint_staging / "case_results").mkdir()
            (checkpoint_staging / "write_staging").mkdir()
            _write_bytes_durable(
                checkpoint_staging / "checkpoint_manifest.json",
                canonical_json_bytes(checkpoint_manifest, newline=True),
            )
            if approval_payload is not None:
                _write_bytes_durable(
                    checkpoint_staging / "few_shot_approval.json",
                    approval_payload,
                )
            if checkpoint_dir.exists():
                raise FileExistsError(
                    f"Checkpoint was created concurrently: {checkpoint_dir}"
                )
            os.replace(checkpoint_staging, checkpoint_dir)
        finally:
            if checkpoint_staging.exists():
                shutil.rmtree(checkpoint_staging, ignore_errors=True)

    completed_rows = _load_checkpoint_rows(
        checkpoint_dir,
        selected_task_ids,
        benchmark_manifest_sha256=benchmark_manifest_sha256,
        corpus_id=str(manifest["corpus_id"]),
        runtime_mode=runtime_mode,
        prompt_condition=prompt_condition,
        model_config_sha256=model_config_sha256,
        provenance=provenance,
    )

    pending_cases = [
        (index, case)
        for index, case in enumerate(cases)
        if str(case["task_id"]) not in completed_rows
    ]
    active_adapter: ModelAdapter | None = adapter
    if pending_cases and active_adapter is None:
        first_index, first_case = pending_cases[0]
        try:
            active_adapter = OfflineVLLMAdapter(model_config)
        except BaseException as exc:
            try:
                _append_batch_failure(
                    checkpoint_dir,
                    binding_sha256=binding_sha256,
                    task_id=str(first_case["task_id"]),
                    case_index=first_index,
                    stage="adapter_initialization",
                    exception=exc,
                    completed_case_count=len(completed_rows),
                    checkpoint_version=checkpoint_version,
                )
            except OSError as journal_error:  # pragma: no cover - disk failure
                exc.add_note(
                    "AuditOps could not persist the batch failure manifest: "
                    f"{type(journal_error).__name__}"
                )
            _add_checkpoint_note(exc, checkpoint_dir)
            raise

    for case_index, case in pending_cases:
        task_id = str(case["task_id"])
        try:
            evidence_row = evidence_by_task[task_id]
            evidence_items = evidence_row.get("items")
            if not isinstance(evidence_items, list):
                raise TypeError(f"Evidence row for {task_id} lacks items")

            def execute_tool(
                tool_name: str,
                tool_arguments: Mapping[str, Any],
                task_input: Mapping[str, Any],
                *,
                frozen_evidence: Sequence[Mapping[str, Any]] = tuple(evidence_items),
            ) -> Mapping[str, Any]:
                return execute_deterministic_tool(
                    tool_name,
                    tool_arguments,
                    task_input,
                    frozen_evidence,
                )

            if active_adapter is None:  # pragma: no cover - guarded above
                raise RuntimeError("Model adapter was not initialized")
            result = run_agent_case(
                case,
                adapter=active_adapter,
                model_config=model_config,
                corpus_id=str(manifest["corpus_id"]),
                benchmark_manifest_sha256=benchmark_manifest_sha256,
                runtime_mode=runtime_mode,
                prompt_condition=prompt_condition,
                evidence_items=evidence_items,
                tool_executor=(
                    execute_tool
                    if runtime_mode in {"capability_agent", "safety_hybrid"}
                    else None
                ),
                few_shot_examples=(
                    [
                        example
                        for example in examples
                        if example["task"]["task_type"] == case["task_type"]
                        and (
                            case["task_type"] == "quant_metric"
                            or example["task"]["narrative_subtype"]
                            == case["narrative_subtype"]
                        )
                    ]
                    if prompt_condition == "few_shot"
                    else []
                ),
                resources=resources,
                provenance=provenance,
            )
            row = {"task_id": task_id, **result.to_dict()}
            validated_row = _validate_checkpoint_row(
                row,
                expected_task_id=task_id,
                benchmark_manifest_sha256=benchmark_manifest_sha256,
                corpus_id=str(manifest["corpus_id"]),
                runtime_mode=runtime_mode,
                prompt_condition=prompt_condition,
                model_config_sha256=model_config_sha256,
                provenance=provenance,
            )
            _write_bytes_durable(
                checkpoint_dir
                / "case_results"
                / _checkpoint_case_name(case_index, task_id),
                canonical_json_bytes(validated_row, newline=True),
                staging_dir=checkpoint_dir / "write_staging",
            )
            completed_rows[task_id] = validated_row
        except BaseException as exc:
            try:
                _append_batch_failure(
                    checkpoint_dir,
                    binding_sha256=binding_sha256,
                    task_id=task_id,
                    case_index=case_index,
                    stage="case_execution",
                    exception=exc,
                    completed_case_count=len(completed_rows),
                    checkpoint_version=checkpoint_version,
                )
            except OSError as journal_error:  # pragma: no cover - disk failure
                exc.add_note(
                    "AuditOps could not persist the batch failure manifest: "
                    f"{type(journal_error).__name__}"
                )
            _add_checkpoint_note(exc, checkpoint_dir)
            raise

    # Re-read every immutable case artifact before publication instead of
    # trusting only in-memory state from the current invocation.
    try:
        completed_rows = _load_checkpoint_rows(
            checkpoint_dir,
            selected_task_ids,
            benchmark_manifest_sha256=benchmark_manifest_sha256,
            corpus_id=str(manifest["corpus_id"]),
            runtime_mode=runtime_mode,
            prompt_condition=prompt_condition,
            model_config_sha256=model_config_sha256,
            provenance=provenance,
        )
        if set(completed_rows) != set(selected_task_ids) or len(completed_rows) != len(
            selected_task_ids
        ):
            missing = sorted(set(selected_task_ids) - set(completed_rows))
            raise RuntimeError(
                "Refusing to publish an incomplete batch; missing task IDs: "
                + ", ".join(missing)
            )
    except BaseException as exc:
        try:
            _append_batch_failure(
                checkpoint_dir,
                binding_sha256=binding_sha256,
                task_id=None,
                case_index=None,
                stage="checkpoint_verification",
                exception=exc,
                completed_case_count=len(completed_rows),
                checkpoint_version=checkpoint_version,
            )
        except OSError as journal_error:  # pragma: no cover - disk failure
            exc.add_note(
                "AuditOps could not persist the checkpoint failure manifest: "
                f"{type(journal_error).__name__}"
            )
        _add_checkpoint_note(exc, checkpoint_dir)
        raise

    ordered_rows = [completed_rows[task_id] for task_id in selected_task_ids]
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.publish.", dir=str(destination.parent)
        )
    )
    try:
        result_path = temporary / "results.jsonl"
        _write_bytes_durable(
            result_path,
            b"".join(canonical_json_bytes(row, newline=True) for row in ordered_rows),
        )
        if approval_source is not None:
            approval_payload = checkpoint_approval_path.read_bytes()
            if (
                hashlib.sha256(approval_payload).hexdigest()
                != input_hashes["few_shot_approval_sha256"]
            ):
                raise ValueError("Checkpoint approval changed before publication")
            _write_bytes_durable(temporary / "few_shot_approval.json", approval_payload)
        if authorization_payload is not None:
            _write_bytes_durable(
                temporary
                / (
                    "exploratory_run_authorization.v2.4p.json"
                    if manifest["benchmark_version"] == BENCHMARK_VERSION_V24P
                    else "full_run_authorization.json"
                ),
                authorization_payload,
            )
        run_manifest = {
            "batch_run_version": batch_run_version,
            "benchmark_id": manifest["benchmark_id"],
            "benchmark_manifest_sha256": benchmark_manifest_sha256,
            "runtime_mode": runtime_mode,
            "prompt_condition": prompt_condition,
            **(
                {"selection_mode": selection_mode or "full"}
                if _is_v24_family(manifest)
                else {}
            ),
            "assurance_status": (
                "PROVISIONAL_AI_REVIEW"
                if manifest["benchmark_version"] == BENCHMARK_VERSION_V24P
                else manifest.get("assurance_status")
            ),
            "experiment_label": (
                "EXPLORATORY"
                if manifest["benchmark_version"] == BENCHMARK_VERSION_V24P
                else None
            ),
            "corpus_scope_label": (
                "cached-20"
                if manifest["benchmark_version"] == BENCHMARK_VERSION_V24P
                else None
            ),
            "model_config_sha256": model_config_sha256,
            "model_id": model_config["model_id"],
            "model_revision": model_config["revision"],
            "quantization": model_config["quantization"],
            "case_count": len(cases),
            "full_benchmark_case_count": manifest["counts"]["total"],
            "complete_full_benchmark": len(cases) == manifest["counts"]["total"],
            "task_ids_sha256": canonical_json_sha256(selected_task_ids),
            "inputs": input_hashes,
            "results": {"path": "results.jsonl", "sha256": _sha256_file(result_path)},
            "provenance": provenance,
        }
        _write_bytes_durable(
            temporary / "run_manifest.json",
            canonical_json_bytes(run_manifest, newline=True),
        )
        assert_no_secrets([temporary])
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite run directory: {destination}")
        os.replace(temporary, destination)
    except BaseException as exc:
        try:
            _append_batch_failure(
                checkpoint_dir,
                binding_sha256=binding_sha256,
                task_id=None,
                case_index=None,
                stage="finalization",
                exception=exc,
                completed_case_count=len(completed_rows),
                checkpoint_version=checkpoint_version,
            )
        except OSError as journal_error:  # pragma: no cover - disk failure
            exc.add_note(
                "AuditOps could not persist the finalization failure manifest: "
                f"{type(journal_error).__name__}"
            )
        _add_checkpoint_note(exc, checkpoint_dir)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    return run_manifest
