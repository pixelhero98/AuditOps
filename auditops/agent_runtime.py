"""One-case, stateless execution for the AuditOps LLM baseline.

The model can propose a single registered tool call, but only the caller's
deterministic callback executes it.  Every final proposal passes through the
independent verifier, and the runtime permits one repair attempt across the
entire case (not one repair per stage).
"""

from __future__ import annotations

import copy
import os
import re
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from .agent_context import ContextOverflowError, build_context_pack
from .agent_contracts import (
    MAX_REPAIR_ATTEMPTS,
    VERIFIER_RESULT_VERSION,
    build_agent_run_record,
    build_failure,
    sanitize_agent_task_input,
    semantic_model_config,
    validate_agent_case_result,
    validate_agent_run_record,
    validate_model_config,
    validate_tool_observation,
    validate_verifier_result,
)
from .agent_operations import (
    TEXT_RUNTIME_ID,
    TEXT_RUNTIME_ID_V23,
    expected_tool_arguments,
    operation_registry_version,
    task_operation_spec,
)
from .agent_prompts import (
    MAX_DEMONSTRATION_TOKENS,
    PROMPT_CONDITIONS,
    RUNTIME_MODES,
    PromptBundle,
    build_direct_prompt,
    build_plan_prompt,
    build_repair_prompt,
    build_synthesis_prompt,
    normalize_stage_response,
    prompt_version_for_task,
    stage_response_semantics_valid,
)
from .agent_prompts import (
    stage_schema_bundle_sha256 as compute_stage_schema_bundle_sha256,
)
from .agent_tools import evaluate_quant_evidence, load_frozen_evidence
from .agent_verifier import (
    discarded_proposal_result,
    safe_repair_errors,
    verify_proposal,
)
from .canonical_json import (
    CANONICAL_JSON_VERSION,
    canonical_json_sha256,
    canonical_json_text,
)
from .model_adapter import (
    ModelAdapter,
    ModelAdapterError,
    ModelBackendError,
    ModelGenerationError,
    ModelOOMError,
    ModelRequest,
    ModelResponse,
    ModelSchemaError,
    ModelTimeoutError,
    StrictJSONError,
    validate_schema_payload,
)

RUNTIME_VERSION = TEXT_RUNTIME_ID
RUN_ENVELOPE_VERSION = "auditops.agent_run_envelope.v2.2"
DETERMINISTIC_EXECUTOR_VERSION = "auditops.deterministic_executor.v2.2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ToolExecutor(Protocol):
    """Caller-owned deterministic tool boundary.

    Implementations receive only the approved tool name/arguments and a copy of
    the sanitized task.  They must return a JSON object and must not call an LLM.
    """

    def __call__(
        self,
        tool_name: str,
        tool_arguments: Mapping[str, Any],
        task_input: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class AgentCaseResult:
    """Auditable per-case result; no raw model text or reasoning is retained."""

    plan_proposal: Mapping[str, Any] | None
    final_proposal: Mapping[str, Any] | None
    tool_observation: Mapping[str, Any] | None
    run_record: Mapping[str, Any]
    failure: Mapping[str, Any] | None = None

    @property
    def released(self) -> bool:
        return self.run_record["outcome"] in {"RELEASED", "SAFE_REFUSAL"}

    def to_dict(self) -> dict[str, Any]:
        result = {
            "agent_case_result_version": self.run_record["agent_run_record_version"],
            "task_id": self.run_record["task_id"],
            "outcome": self.run_record["outcome"],
            "plan_proposal": copy.deepcopy(dict(self.plan_proposal))
            if self.plan_proposal is not None
            else None,
            "final_proposal": copy.deepcopy(dict(self.final_proposal))
            if self.final_proposal is not None
            else None,
            "tool_observation": (
                copy.deepcopy(dict(self.tool_observation))
                if self.tool_observation is not None
                else None
            ),
            "run_record": copy.deepcopy(dict(self.run_record)),
            "failure": copy.deepcopy(dict(self.failure))
            if self.failure is not None
            else None,
        }
        validate_agent_case_result(result)
        return result


def _canonical_json(value: Any) -> str:
    return canonical_json_text(value)


def _sha256_json(value: Any) -> str:
    return canonical_json_sha256(value)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _isoformat_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _machine_verifier_result(
    task_id: str,
    *,
    code: str,
    field: str | None,
    message: str,
    repair_count: int,
    allow_repair: bool,
) -> dict[str, Any]:
    repair_allowed = bool(allow_repair and repair_count < MAX_REPAIR_ATTEMPTS)
    disposition = "REPAIR_REQUIRED" if repair_allowed else "FAILED"
    error = {"code": code, "field": field, "message": message}
    result = {
        "verifier_result_version": VERIFIER_RESULT_VERSION,
        "task_id": task_id,
        "passed": False,
        "release_allowed": False,
        "disposition": disposition,
        "repair_count": repair_count,
        "repair_allowed": repair_allowed,
        "checks": [{"code": code, "passed": False, "field": field, "message": message}],
        "repair_errors": [error] if repair_allowed else [],
    }
    validate_verifier_result(result)
    return result


def _stage_verification(
    task_input: Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    expected_action: str,
    evidence_items: Sequence[Mapping[str, Any]],
    tool_observation: Mapping[str, Any] | None,
    repair_count: int,
    stage: str,
) -> dict[str, Any]:
    result = verify_proposal(
        task_input,
        proposal,
        evidence_items=evidence_items,
        tool_observation=tool_observation,
        repair_count=repair_count,
        stage=stage,
    )
    validate_verifier_result(result)

    if expected_action == "CALL_TOOL":
        # A prompt-injection refusal may safely terminate before any tool is
        # invoked. Other refusals cannot pass without a deterministic observation.
        expected_dispositions = {"TOOL_CALL_APPROVED", "SAFE_REFUSAL"}
        message = "bounded-agent planning must produce an approved CALL_TOOL or a verified security refusal"
    elif expected_action == "FINAL":
        expected_dispositions = {"RELEASED", "SAFE_REFUSAL"}
        message = "final generation must produce a verified ANSWER or REFUSE proposal"
    else:
        raise ValueError("expected_action must be CALL_TOOL or FINAL")

    if result["passed"] and result["disposition"] not in expected_dispositions:
        return _machine_verifier_result(
            task_input["task_id"],
            code="UNEXPECTED_STAGE_ACTION",
            field="proposal.action",
            message=message,
            repair_count=repair_count,
            allow_repair=True,
        )
    return result


def _discard_untrusted_proposal(
    proposal: dict[str, Any], verifier_result: Mapping[str, Any], *, stage: str
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Drop contract-invalid or cross-task payloads before prompts/artifacts."""

    failed_codes = {
        check.get("code")
        for check in verifier_result["checks"]
        if check.get("passed") is False
    }
    code = (
        "SCHEMA_VALID"
        if "SCHEMA_VALID" in failed_codes
        else "TASK_MATCH"
        if "TASK_MATCH" in failed_codes
        else None
    )
    if code is None:
        return proposal, copy.deepcopy(dict(verifier_result))
    return None, discarded_proposal_result(
        str(verifier_result["task_id"]),
        code=code,
        repair_count=int(verifier_result["repair_count"]),
        stage=stage,
    )


def _default_resources(resources: Mapping[str, Any] | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_model": None,
        "peak_vram_bytes": None,
        "host": socket.gethostname(),
    }
    if resources is not None:
        unknown = sorted(set(resources) - set(defaults))
        if unknown:
            raise ValueError(f"Unknown resource fields: {', '.join(unknown)}")
        defaults.update(resources)
    return defaults


def _default_provenance(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "git_commit": None,
        "source_tree_sha256": None,
        "container_digest": None,
        "model_snapshot_sha256": None,
        "tokenizer_revision": None,
        "package_lock_sha256": None,
    }
    if provenance is not None:
        unknown = sorted(set(provenance) - set(defaults))
        if unknown:
            raise ValueError(f"Unknown provenance fields: {', '.join(unknown)}")
        defaults.update(provenance)
    return defaults


def _run_envelope(
    task_input: Mapping[str, Any],
    *,
    corpus_id: str,
    benchmark_manifest_sha256: str,
    runtime_mode: str,
    prompt_condition: str,
    context_sha256: str,
    input_sha256: str,
    semantic_model_config_sha256: str | None,
    deterministic_executor_sha256: str,
    stage_schema_bundle_sha256: str,
    demonstration_pack_sha256: str | None,
    narrative_subtype: str | None,
    metric_operation_contract_sha256: str | None,
    narrative_selection_policy_sha256: str | None,
    run_envelope_version: str,
    runtime_version: str,
    prompt_version: str,
) -> dict[str, Any]:
    envelope = {
        "run_envelope_version": run_envelope_version,
        "canonical_json_version": CANONICAL_JSON_VERSION,
        "runtime_version": runtime_version,
        "task_id": task_input["task_id"],
        "input_sha256": input_sha256,
        "corpus_id": corpus_id,
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
        "runtime_mode": runtime_mode,
        "prompt_condition": prompt_condition,
        "prompt_version": prompt_version,
        "context_sha256": context_sha256,
        "stage_schema_bundle_sha256": stage_schema_bundle_sha256,
        "semantic_model_config_sha256": semantic_model_config_sha256,
        "deterministic_executor_sha256": deterministic_executor_sha256,
        "demonstration_pack_sha256": demonstration_pack_sha256,
        "narrative_subtype": narrative_subtype,
        "metric_operation_contract_sha256": metric_operation_contract_sha256,
    }
    if run_envelope_version == "auditops.agent_run_envelope.v2.3":
        envelope["narrative_selection_policy_sha256"] = (
            narrative_selection_policy_sha256
        )
    return envelope


def run_agent_case(
    task_input: Mapping[str, Any],
    *,
    adapter: ModelAdapter,
    model_config: Mapping[str, Any],
    corpus_id: str,
    benchmark_manifest_sha256: str,
    runtime_mode: str,
    prompt_condition: str,
    evidence_items: Sequence[Mapping[str, Any]],
    tool_executor: ToolExecutor | None = None,
    few_shot_examples: Sequence[Mapping[str, Any]] = (),
    token_counter: Callable[[str], int] | None = None,
    resources: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> AgentCaseResult:
    """Execute one direct-control or bounded-agent case.

    Contract/gold-leakage errors are caller errors and are raised before model
    inference.  The hard gate independently reproduces its release observation
    from the sanitized task and frozen evidence; evaluator-only artifacts are
    neither accepted nor opened.  The bounded agent sees only its live tool
    observation.  Model, context-capacity, and deterministic-tool failures
    become typed per-case results so a batch cannot silently lose a case.
    """

    sanitized_task = sanitize_agent_task_input(task_input)
    contract_version = str(sanitized_task["agent_task_input_version"])
    runtime_version = (
        TEXT_RUNTIME_ID_V23 if contract_version == "v2.3" else TEXT_RUNTIME_ID
    )
    prompt_version = prompt_version_for_task(sanitized_task)
    run_envelope_version = f"auditops.agent_run_envelope.{contract_version}"
    deterministic_executor_version = (
        f"auditops.deterministic_executor.{contract_version}"
    )
    validate_model_config(model_config)
    if runtime_mode not in RUNTIME_MODES:
        raise ValueError(f"Unsupported runtime mode: {runtime_mode}")
    if prompt_condition not in PROMPT_CONDITIONS:
        raise ValueError(f"Unsupported prompt condition: {prompt_condition}")
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise ValueError("corpus_id must be non-empty text")
    if not isinstance(benchmark_manifest_sha256, str) or not _SHA256_RE.fullmatch(
        benchmark_manifest_sha256
    ):
        raise ValueError(
            "benchmark_manifest_sha256 must be a 64-character SHA-256 digest"
        )
    if adapter.backend != model_config["backend"]:
        raise ValueError("Adapter backend does not match model_config")
    if adapter.model_id != model_config["model_id"]:
        raise ValueError("Adapter model_id does not match model_config")
    if adapter.model_revision != model_config["revision"]:
        raise ValueError("Adapter model revision does not match model_config")
    if runtime_mode in {"capability_agent", "safety_hybrid"} and tool_executor is None:
        raise ValueError(f"{runtime_mode} requires a deterministic tool_executor")
    if runtime_mode == "direct" and tool_executor is not None:
        raise ValueError("direct mode cannot receive a tool_executor")
    if prompt_condition == "zero_shot" and few_shot_examples:
        raise ValueError("zero_shot mode cannot receive few-shot examples")
    if prompt_condition == "few_shot" and len(few_shot_examples) != 4:
        raise ValueError("few_shot mode requires exactly four same-family examples")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ValueError("run_id must be non-empty text when supplied")

    started_at = clock()
    if not isinstance(started_at, datetime):
        raise TypeError("clock must return datetime values")
    input_sha256 = _sha256_json(sanitized_task)
    portable_model_config = semantic_model_config(model_config)
    model_independent_case = (
        runtime_mode == "safety_hybrid"
        and sanitized_task["task_type"] == "quant_metric"
    )
    semantic_model_config_sha256 = (
        None if model_independent_case else _sha256_json(portable_model_config)
    )
    deployment_model_config_sha256 = (
        None if model_independent_case else _sha256_json(model_config)
    )
    deterministic_executor_sha256 = _sha256_json(
        {
            "executor_version": deterministic_executor_version,
            "operation_registry_version": operation_registry_version(sanitized_task),
        }
    )
    stage_schema_bundle_sha256 = compute_stage_schema_bundle_sha256(
        sanitized_task, evidence_items
    )
    active_run_id: str | None = None
    run_envelope_sha256: str | None = None
    context_sha256: str | None = None
    input_tokens = 0
    output_tokens = 0
    repair_count = 0
    plan_proposal: dict[str, Any] | None = None
    final_proposal: dict[str, Any] | None = None
    tool_observation: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = []
    verifier_trace: list[dict[str, Any]] = []
    last_output_sha256: str | None = None
    last_call_audit: dict[str, Any] | None = None
    terminal_model_failure_code: str | None = None
    terminal_failure_stage: str | None = None
    engine_initialization_baseline_ms = adapter.engine_initialization_ms
    demonstration_pack_sha256 = (
        _sha256_json(list(few_shot_examples))
        if prompt_condition == "few_shot"
        else None
    )
    metric_operation_contract_sha256 = (
        _sha256_json(sanitized_task["metric_operation_contract"])
        if sanitized_task["metric_operation_contract"] is not None
        else None
    )
    narrative_selection_policy_sha256 = (
        _sha256_json(sanitized_task["narrative_selection_policy"])
        if contract_version == "v2.3"
        and sanitized_task["narrative_selection_policy"] is not None
        else None
    )

    def bind_run_identity(context_binding_sha256: str) -> None:
        nonlocal active_run_id, run_envelope_sha256, context_sha256
        context_sha256 = context_binding_sha256
        envelope = _run_envelope(
            sanitized_task,
            corpus_id=corpus_id,
            benchmark_manifest_sha256=benchmark_manifest_sha256,
            runtime_mode=runtime_mode,
            prompt_condition=prompt_condition,
            context_sha256=context_binding_sha256,
            input_sha256=input_sha256,
            semantic_model_config_sha256=semantic_model_config_sha256,
            deterministic_executor_sha256=deterministic_executor_sha256,
            stage_schema_bundle_sha256=stage_schema_bundle_sha256,
            demonstration_pack_sha256=demonstration_pack_sha256,
            narrative_subtype=sanitized_task["narrative_subtype"],
            metric_operation_contract_sha256=metric_operation_contract_sha256,
            narrative_selection_policy_sha256=narrative_selection_policy_sha256,
            run_envelope_version=run_envelope_version,
            runtime_version=runtime_version,
            prompt_version=prompt_version,
        )
        run_envelope_sha256 = _sha256_json(envelope)
        derived_run_id = f"run-{run_envelope_sha256[:24]}"
        if run_id is not None and run_id != derived_run_id:
            raise ValueError(
                "supplied run_id does not match the derived "
                f"{'v2' if contract_version == 'v2.2' else contract_version} run identity"
            )
        active_run_id = derived_run_id

    def reject_aborted_completion() -> None:
        nonlocal last_output_sha256
        if last_call_audit is not None and last_call_audit["finish_reason"] == "abort":
            last_output_sha256 = None
            last_call_audit["output_sha256"] = None
            raise ModelGenerationError("Model generation was aborted") from None

    def call_model(
        bundle: PromptBundle,
        max_tokens: int,
        label: str,
        *,
        stage: str,
        repair_attempt: bool,
    ) -> dict[str, Any]:
        nonlocal input_tokens, output_tokens, last_output_sha256, last_call_audit
        if active_run_id is None:
            raise RuntimeError("run identity must be bound before model generation")
        call_started = time.perf_counter()
        request = ModelRequest(
            request_id=f"{active_run_id}:{label}:{len(verifier_trace) + 1}",
            messages=bundle.messages,
            json_schema=bundle.json_schema,
            max_tokens=max_tokens,
            stage=stage,
            repair_attempt=repair_attempt,
            prompt_version=bundle.prompt_version,
            runtime_version=runtime_version,
            model_semantic_config=portable_model_config,
            temperature=float(model_config["temperature"]),
            top_p=float(model_config["top_p"]),
            seed=int(model_config["seed"]),
        )
        # Establish the semantic request binding before tokenizer/backend preflight.
        # Any subsequent typed failure can therefore produce a complete trace
        # instead of leaving a half-consumed repair budget.
        last_call_audit = {
            "prompt_sha256": bundle.prompt_sha256,
            "request_sha256": request.request_sha256,
            "response_schema_sha256": request.response_schema_sha256,
            "visible_context_sha256": bundle.visible_context_sha256,
            "output_sha256": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "duration_ms": 0.0,
            "finish_reason": None,
            "constraint_backend": adapter.constraint_backend,
            "structured_output_applied": adapter.structured_output_applied,
            "output_bytes": 0,
            "cap_hit": False,
            "parse_category": None,
        }
        try:
            if not adapter.structured_output_applied:
                raise ModelBackendError(
                    "model adapter cannot guarantee structured-output enforcement"
                )
            demonstration_tokens = (
                adapter.count_messages(bundle.demonstration_messages)
                if bundle.demonstration_messages
                else 0
            )
            if demonstration_tokens > MAX_DEMONSTRATION_TOKENS:
                raise ContextOverflowError(
                    demonstration_tokens,
                    MAX_DEMONSTRATION_TOKENS,
                    context_sha256=bundle.visible_context_sha256,
                )
            rendered_input_tokens = adapter.count_messages(bundle.messages)
        except (ModelAdapterError, ContextOverflowError):
            last_call_audit["duration_ms"] = (
                time.perf_counter() - call_started
            ) * 1000.0
            raise
        input_cap = int(model_config["max_input_tokens"])
        if rendered_input_tokens > input_cap:
            last_call_audit["duration_ms"] = (
                time.perf_counter() - call_started
            ) * 1000.0
            raise ContextOverflowError(rendered_input_tokens, input_cap)
        try:
            response: ModelResponse = adapter.generate_json(request)
        except StrictJSONError as exc:
            malformed_input_tokens = (
                exc.input_tokens
                if exc.input_tokens is not None
                else rendered_input_tokens
            )
            if malformed_input_tokens > input_cap:
                input_tokens += rendered_input_tokens
                last_call_audit = {
                    "prompt_sha256": bundle.prompt_sha256,
                    "request_sha256": request.request_sha256,
                    "response_schema_sha256": request.response_schema_sha256,
                    "visible_context_sha256": bundle.visible_context_sha256,
                    "output_sha256": None,
                    "input_tokens": rendered_input_tokens,
                    "output_tokens": 0,
                    "duration_ms": exc.duration_ms
                    if exc.duration_ms is not None
                    else (time.perf_counter() - call_started) * 1000.0,
                    "finish_reason": exc.finish_reason,
                    "constraint_backend": exc.constraint_backend
                    or adapter.constraint_backend,
                    "structured_output_applied": (
                        exc.structured_output_applied
                        if exc.structured_output_applied is not None
                        else adapter.structured_output_applied
                    ),
                    "output_bytes": exc.output_bytes or 0,
                    "cap_hit": exc.finish_reason == "length",
                    "parse_category": exc.parse_category,
                }
                raise ModelAdapterError(
                    "Strict JSON failure input-token telemetry exceeded the locked cap"
                ) from None
            input_tokens += malformed_input_tokens
            malformed_output_tokens = exc.output_tokens or 0
            output_tokens += malformed_output_tokens
            last_output_sha256 = exc.response_sha256
            last_call_audit = {
                "prompt_sha256": bundle.prompt_sha256,
                "request_sha256": request.request_sha256,
                "response_schema_sha256": request.response_schema_sha256,
                "visible_context_sha256": bundle.visible_context_sha256,
                "output_sha256": exc.response_sha256,
                "input_tokens": malformed_input_tokens,
                "output_tokens": malformed_output_tokens,
                "duration_ms": exc.duration_ms
                if exc.duration_ms is not None
                else (time.perf_counter() - call_started) * 1000.0,
                "finish_reason": exc.finish_reason,
                "constraint_backend": exc.constraint_backend
                or adapter.constraint_backend,
                "structured_output_applied": (
                    exc.structured_output_applied
                    if exc.structured_output_applied is not None
                    else adapter.structured_output_applied
                ),
                "output_bytes": exc.output_bytes or 0,
                "cap_hit": exc.finish_reason == "length",
                "parse_category": exc.parse_category,
            }
            reject_aborted_completion()
            raise
        except ModelSchemaError as exc:
            malformed_input_tokens = (
                exc.input_tokens
                if exc.input_tokens is not None
                else rendered_input_tokens
            )
            input_tokens += malformed_input_tokens
            malformed_output_tokens = exc.output_tokens or 0
            output_tokens += malformed_output_tokens
            last_output_sha256 = exc.response_sha256
            last_call_audit = {
                "prompt_sha256": bundle.prompt_sha256,
                "request_sha256": request.request_sha256,
                "response_schema_sha256": request.response_schema_sha256,
                "visible_context_sha256": bundle.visible_context_sha256,
                "output_sha256": exc.response_sha256,
                "input_tokens": malformed_input_tokens,
                "output_tokens": malformed_output_tokens,
                "duration_ms": exc.duration_ms
                if exc.duration_ms is not None
                else (time.perf_counter() - call_started) * 1000.0,
                "finish_reason": exc.finish_reason,
                "constraint_backend": exc.constraint_backend
                or adapter.constraint_backend,
                "structured_output_applied": (
                    exc.structured_output_applied
                    if exc.structured_output_applied is not None
                    else adapter.structured_output_applied
                ),
                "output_bytes": exc.output_bytes,
                "cap_hit": exc.finish_reason == "length",
                "parse_category": None,
            }
            reject_aborted_completion()
            raise
        except ModelAdapterError:
            input_tokens += rendered_input_tokens
            last_call_audit = {
                "prompt_sha256": bundle.prompt_sha256,
                "request_sha256": request.request_sha256,
                "response_schema_sha256": request.response_schema_sha256,
                "visible_context_sha256": bundle.visible_context_sha256,
                "output_sha256": None,
                "input_tokens": rendered_input_tokens,
                "output_tokens": 0,
                "duration_ms": (time.perf_counter() - call_started) * 1000.0,
                "finish_reason": None,
                "constraint_backend": adapter.constraint_backend,
                "structured_output_applied": adapter.structured_output_applied,
                "output_bytes": 0,
                "cap_hit": False,
                "parse_category": None,
            }
            raise
        if not isinstance(response, ModelResponse):
            raise ModelGenerationError(
                "Model adapter returned an invalid response type"
            )
        if response.request_id != request.request_id:
            input_tokens += rendered_input_tokens
            last_call_audit.update(
                {
                    "output_sha256": None,
                    "input_tokens": rendered_input_tokens,
                    "output_tokens": 0,
                    "duration_ms": response.duration_ms,
                }
            )
            raise ModelGenerationError("Model adapter returned a mismatched request_id")
        if response.input_tokens > input_cap:
            input_tokens += rendered_input_tokens
            last_call_audit.update(
                {
                    "output_sha256": None,
                    "input_tokens": rendered_input_tokens,
                    "output_tokens": 0,
                    "duration_ms": response.duration_ms,
                }
            )
            raise ModelGenerationError(
                "Rendered model input exceeded the locked input-token cap"
            )
        input_tokens += response.input_tokens
        output_tokens += response.output_tokens
        last_output_sha256 = response.response_sha256
        last_call_audit = {
            "prompt_sha256": bundle.prompt_sha256,
            "request_sha256": request.request_sha256,
            "response_schema_sha256": request.response_schema_sha256,
            "visible_context_sha256": bundle.visible_context_sha256,
            "output_sha256": response.response_sha256,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "duration_ms": response.duration_ms,
            "finish_reason": response.finish_reason,
            "constraint_backend": str(
                response.metadata.get("constraint_backend", adapter.constraint_backend)
            ),
            "structured_output_applied": bool(
                response.metadata.get(
                    "structured_output_applied", adapter.structured_output_applied
                )
            ),
            "output_bytes": int(response.metadata.get("output_bytes", 0)),
            "cap_hit": response.finish_reason == "length",
            "parse_category": None,
        }
        reject_aborted_completion()
        if last_call_audit["structured_output_applied"] is not True:
            last_output_sha256 = None
            last_call_audit["output_sha256"] = None
            raise ModelBackendError(
                "Model completion lacked structured-output enforcement"
            )
        try:
            validate_schema_payload(
                response.payload,
                bundle.json_schema,
                response_sha256=response.response_sha256,
                output_bytes=int(response.metadata.get("output_bytes", 0)),
            )
        except ModelSchemaError as exc:
            raise ModelSchemaError(
                response_sha256=exc.response_sha256,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                output_bytes=exc.output_bytes,
                duration_ms=response.duration_ms,
                finish_reason=response.finish_reason,
                constraint_backend=str(last_call_audit["constraint_backend"]),
                structured_output_applied=bool(
                    last_call_audit["structured_output_applied"]
                ),
            ) from None
        if not stage_response_semantics_valid(response.payload, sanitized_task):
            raise ModelSchemaError(
                response_sha256=response.response_sha256,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                output_bytes=int(response.metadata.get("output_bytes", 0)),
                duration_ms=response.duration_ms,
                finish_reason=response.finish_reason,
                constraint_backend=str(last_call_audit["constraint_backend"]),
                structured_output_applied=bool(
                    last_call_audit["structured_output_applied"]
                ),
            ) from None
        return normalize_stage_response(
            response.payload,
            sanitized_task,
            stage="plan" if stage == "plan" else "terminal",
        )

    def append_verifier_trace(
        *,
        stage: str,
        repair_attempt: bool,
        proposal: Mapping[str, Any] | None,
        verifier_result: Mapping[str, Any],
    ) -> None:
        if last_call_audit is None:
            raise RuntimeError("model-call audit metadata is unavailable")
        materialized_proposal = (
            copy.deepcopy(dict(proposal)) if proposal is not None else None
        )
        verifier_trace.append(
            {
                "stage": stage,
                "attempt_index": len(verifier_trace),
                "repair_attempt": repair_attempt,
                "prompt_sha256": last_call_audit["prompt_sha256"],
                "request_sha256": last_call_audit["request_sha256"],
                "response_schema_sha256": last_call_audit["response_schema_sha256"],
                "visible_context_sha256": last_call_audit["visible_context_sha256"],
                "output_sha256": last_call_audit["output_sha256"],
                "proposal": materialized_proposal,
                "proposal_sha256": _sha256_json(materialized_proposal)
                if materialized_proposal is not None
                else None,
                "verifier_result": copy.deepcopy(dict(verifier_result)),
                "input_tokens": last_call_audit["input_tokens"],
                "output_tokens": last_call_audit["output_tokens"],
                "duration_ms": last_call_audit["duration_ms"],
                "finish_reason": last_call_audit["finish_reason"],
                "constraint_backend": last_call_audit["constraint_backend"],
                "structured_output_applied": last_call_audit[
                    "structured_output_applied"
                ],
                "output_bytes": last_call_audit["output_bytes"],
                "cap_hit": last_call_audit["cap_hit"],
                "parse_category": last_call_audit["parse_category"],
            }
        )

    def final_record(
        *,
        verifier_result: Mapping[str, Any] | None,
        outcome: str,
        output: Mapping[str, Any] | None,
        failure: Mapping[str, Any] | None = None,
    ) -> AgentCaseResult:
        if active_run_id is None or run_envelope_sha256 is None:
            raise RuntimeError("run identity is unavailable for terminal result")
        finished_at = clock()
        duration_ms = max(0.0, (finished_at - started_at).total_seconds() * 1000.0)
        prompt_sha256 = _sha256_json(
            [entry["prompt_sha256"] for entry in verifier_trace]
        )
        if verifier_trace and verifier_trace[-1]["proposal"] is None:
            # Bind a malformed, discarded, or failed generation to the terminal
            # call only; never fall back to an earlier accepted proposal.
            output_sha256 = verifier_trace[-1]["output_sha256"]
        else:
            output_sha256 = (
                _sha256_json(output) if output is not None else last_output_sha256
            )
        resolved_resources = _default_resources(resources)
        measured_peak = adapter.peak_vram_bytes
        if measured_peak is not None:
            resolved_resources["peak_vram_bytes"] = measured_peak
        terminal_verifier_result = (
            dict(verifier_result)
            if verifier_result is not None
            else copy.deepcopy(verifier_trace[-1]["verifier_result"])
            if verifier_trace
            else None
        )
        record_fields: dict[str, Any] = {
            "run_id": active_run_id,
            "task_id": sanitized_task["task_id"],
            "corpus_id": corpus_id,
            "benchmark_manifest_sha256": benchmark_manifest_sha256,
            "runtime_mode": runtime_mode,
            "runtime_version": runtime_version,
            "prompt_condition": prompt_condition,
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha256,
            "canonical_json_version": CANONICAL_JSON_VERSION,
            "semantic_model_config_sha256": semantic_model_config_sha256,
            "deployment_model_config_sha256": deployment_model_config_sha256,
            "deterministic_executor_sha256": deterministic_executor_sha256,
            "stage_schema_bundle_sha256": stage_schema_bundle_sha256,
            "run_envelope_sha256": run_envelope_sha256,
            "demonstration_pack_sha256": demonstration_pack_sha256,
            "narrative_subtype": sanitized_task["narrative_subtype"],
            "metric_operation_contract_sha256": metric_operation_contract_sha256,
            "model_config": None if model_independent_case else dict(model_config),
            "context_sha256": context_sha256,
            "input_sha256": input_sha256,
            "output_sha256": output_sha256,
            "tool_calls": tool_calls,
            "verifier_trace": verifier_trace,
            "verifier_result": terminal_verifier_result,
            "repair_count": repair_count,
            "outcome": outcome,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "timings": {
                "started_at": _isoformat_utc(started_at),
                "finished_at": _isoformat_utc(finished_at),
                "duration_ms": duration_ms,
                "engine_initialization_ms": max(
                    0.0,
                    adapter.engine_initialization_ms
                    - engine_initialization_baseline_ms,
                ),
                "generation_duration_ms": sum(
                    float(entry["duration_ms"]) for entry in verifier_trace
                ),
                "tool_duration_ms": sum(
                    float(entry["duration_ms"]) for entry in tool_calls
                ),
            },
            "resources": resolved_resources,
            "provenance": _default_provenance(provenance),
        }
        if contract_version == "v2.3":
            record_fields["narrative_selection_policy_sha256"] = (
                narrative_selection_policy_sha256
            )
        record = build_agent_run_record(
            contract_version=contract_version,
            **record_fields,
        )
        validate_agent_run_record(record)
        result = AgentCaseResult(
            plan_proposal=copy.deepcopy(plan_proposal),
            final_proposal=copy.deepcopy(final_proposal),
            tool_observation=copy.deepcopy(tool_observation),
            run_record=record,
            failure=dict(failure) if failure is not None else None,
        )
        result.to_dict()
        return result

    def failure_result(
        code: str,
        stage: str,
        sanitized_detail: str,
        *,
        verifier_result: Mapping[str, Any] | None = None,
    ) -> AgentCaseResult:
        failure = build_failure(
            code=code,
            stage=stage,
            sanitized_detail=sanitized_detail,
            contract_version=contract_version,
        )
        return final_record(
            verifier_result=verifier_result,
            outcome=failure["failure_class"],
            output=final_proposal or plan_proposal,
            failure=failure,
        )

    def malformed_output_result(
        *, stage: str, allow_repair: bool = True
    ) -> dict[str, Any]:
        return _machine_verifier_result(
            sanitized_task["task_id"],
            code="MODEL_OUTPUT_NOT_STRICT_JSON",
            field=None,
            message=f"{stage} output was not one strict JSON object",
            repair_count=repair_count,
            allow_repair=allow_repair,
        )

    try:
        operation = task_operation_spec(
            sanitized_task["task_type"], version=contract_version
        ).operation
        observation_visibility = (
            "MODEL_VISIBLE"
            if runtime_mode == "capability_agent"
            or (
                runtime_mode == "safety_hybrid"
                and sanitized_task["task_type"] == "narrative_citation"
            )
            else "VERIFIER_ONLY"
        )
        if operation == "evaluate_metric_spec":
            release_observation = evaluate_quant_evidence(
                sanitized_task,
                evidence_items,
                visibility=observation_visibility,
            )
        elif operation == "load_frozen_evidence":
            release_observation = load_frozen_evidence(
                sanitized_task,
                evidence_items,
                visibility=observation_visibility,
            )
        else:  # pragma: no cover - registry construction forbids this
            raise RuntimeError(f"Unsupported registered operation: {operation}")
        if runtime_mode == "direct":
            # This is verifier-only state and is safe to retain even when model
            # tokenizer initialization later fails during context construction.
            tool_observation = copy.deepcopy(release_observation)

        # Model-independent quantitative hybrid cases must not initialize a
        # tokenizer or model merely to count a context the model never sees.
        # The context builder's default counter is conservative and remains
        # deterministic; model-visible paths always use the pinned tokenizer.
        context_token_counter = token_counter
        if context_token_counter is None and not model_independent_case:
            context_token_counter = adapter.count_tokens
        context_pack = build_context_pack(
            sanitized_task,
            evidence_items,
            few_shot_examples=few_shot_examples,
            token_counter=context_token_counter,
            max_input_tokens=int(model_config["max_input_tokens"]),
        )
        bind_run_identity(context_pack["context_sha256"])
    except ContextOverflowError as exc:
        bind_run_identity(
            exc.details.get("context_sha256")
            or _sha256_json(
                {
                    "context_failure": "CONTEXT_OVERFLOW",
                    "input_sha256": input_sha256,
                }
            )
        )
        return failure_result(
            "CONTEXT_OVERFLOW",
            "context",
            "complete context exceeded the locked input-token cap",
        )
    except (ModelTimeoutError, ModelOOMError, ModelBackendError) as exc:
        code = (
            "MODEL_TIMEOUT"
            if isinstance(exc, ModelTimeoutError)
            else "MODEL_OOM"
            if isinstance(exc, ModelOOMError)
            else "BACKEND_FAILURE"
        )
        bind_run_identity(
            _sha256_json(
                {
                    "context_failure": code,
                    "input_sha256": input_sha256,
                }
            )
        )
        return failure_result(
            code,
            "context",
            "offline model backend failed during context construction",
        )
    except ModelAdapterError:
        bind_run_identity(
            _sha256_json(
                {
                    "context_failure": "MODEL_INITIALIZATION_FAILED",
                    "input_sha256": input_sha256,
                }
            )
        )
        return failure_result(
            "MODEL_INITIALIZATION_FAILED",
            "context",
            "model tokenizer or adapter initialization failed during context construction",
        )

    if context_pack["security_flags"]:
        final_proposal = normalize_stage_response(
            {
                "task_id": sanitized_task["task_id"],
                "action": "REFUSE",
                "period_key": sanitized_task["period"]["period_key"],
                "refusal_code": "PROMPT_INJECTION_DETECTED",
            },
            sanitized_task,
            stage="terminal",
        )
        verifier_result = verify_proposal(
            sanitized_task,
            final_proposal,
            evidence_items=evidence_items,
            tool_observation=release_observation,
            repair_count=0,
        )
        if verifier_result["disposition"] != "SAFE_REFUSAL":
            return failure_result(
                "PROMPT_INJECTION_POLICY_VIOLATION",
                "verify",
                "deterministic prompt-injection refusal failed verification",
                verifier_result=verifier_result,
            )
        return final_record(
            verifier_result=verifier_result,
            outcome="SAFE_REFUSAL",
            output=final_proposal,
        )

    def generate_with_single_repair(
        initial_prompt: PromptBundle,
        *,
        stage: str,
        expected_action: str,
        max_tokens: int,
        release_observation: Mapping[str, Any] | None,
        repair_prompt_observation: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        nonlocal last_output_sha256, repair_count
        nonlocal terminal_model_failure_code, terminal_failure_stage
        proposal: dict[str, Any] | None = None
        try:
            proposal = call_model(
                initial_prompt,
                max_tokens,
                stage,
                stage=stage,
                repair_attempt=False,
            )
            result = _stage_verification(
                sanitized_task,
                proposal,
                expected_action=expected_action,
                evidence_items=evidence_items,
                tool_observation=release_observation,
                repair_count=repair_count,
                stage=stage,
            )
            proposal, result = _discard_untrusted_proposal(
                proposal, result, stage=stage
            )
        except StrictJSONError as exc:
            last_output_sha256 = exc.response_sha256
            integrity_break = bool(
                exc.finish_reason == "stop" and exc.structured_output_applied is True
            )
            result = malformed_output_result(
                stage=stage, allow_repair=not integrity_break
            )
            if integrity_break:
                terminal_model_failure_code = "STRUCTURED_OUTPUT_CONTRACT_BROKEN"
                terminal_failure_stage = stage
        except ModelSchemaError as exc:
            last_output_sha256 = exc.response_sha256
            integrity_break = bool(
                exc.finish_reason == "stop" and exc.structured_output_applied is True
            )
            result = _machine_verifier_result(
                sanitized_task["task_id"],
                code="SCHEMA_VALID",
                field=None,
                message=f"{stage} output did not satisfy the response schema",
                repair_count=repair_count,
                allow_repair=not integrity_break,
            )
            if integrity_break:
                terminal_model_failure_code = "STRUCTURED_OUTPUT_CONTRACT_BROKEN"
                terminal_failure_stage = stage
        except ContextOverflowError:
            result = _machine_verifier_result(
                sanitized_task["task_id"],
                code="MODEL_GENERATION_FAILED",
                field=None,
                message=f"{stage} model generation failed",
                repair_count=repair_count,
                allow_repair=False,
            )
            append_verifier_trace(
                stage=stage,
                repair_attempt=False,
                proposal=None,
                verifier_result=result,
            )
            raise
        except ModelAdapterError:
            result = _machine_verifier_result(
                sanitized_task["task_id"],
                code="MODEL_GENERATION_FAILED",
                field=None,
                message=f"{stage} model generation failed",
                repair_count=repair_count,
                allow_repair=False,
            )
            append_verifier_trace(
                stage=stage,
                repair_attempt=False,
                proposal=None,
                verifier_result=result,
            )
            raise
        append_verifier_trace(
            stage=stage,
            repair_attempt=False,
            proposal=proposal,
            verifier_result=result,
        )

        if result["disposition"] != "REPAIR_REQUIRED":
            return proposal, result

        errors = safe_repair_errors(result)
        if not errors:
            result = copy.deepcopy(dict(result))
            result["disposition"] = "FAILED"
            result["repair_allowed"] = False
            result["repair_errors"] = []
            validate_verifier_result(result)
            verifier_trace[-1]["verifier_result"] = copy.deepcopy(result)
            terminal_model_failure_code = "REPAIR_NOT_SAFE"
            return proposal, result
        repair_count += 1
        repair_prompt = build_repair_prompt(
            context_pack,
            errors,
            expected_action=expected_action,
            prompt_condition=prompt_condition,
            rejected_proposal=proposal,
            tool_observation=repair_prompt_observation,
        )
        try:
            proposal = call_model(
                repair_prompt,
                min(
                    int(model_config["max_repair_tokens"]),
                    256 if sanitized_task["task_type"] == "quant_metric" else 768,
                ),
                f"{stage}-repair",
                stage=stage,
                repair_attempt=True,
            )
            result = _stage_verification(
                sanitized_task,
                proposal,
                expected_action=expected_action,
                evidence_items=evidence_items,
                tool_observation=release_observation,
                repair_count=repair_count,
                stage=stage,
            )
            proposal, result = _discard_untrusted_proposal(
                proposal, result, stage=stage
            )
        except StrictJSONError as exc:
            last_output_sha256 = exc.response_sha256
            proposal = None
            integrity_break = bool(
                exc.finish_reason == "stop" and exc.structured_output_applied is True
            )
            result = malformed_output_result(
                stage=f"{stage}-repair", allow_repair=False
            )
            if integrity_break:
                terminal_model_failure_code = "STRUCTURED_OUTPUT_CONTRACT_BROKEN"
                terminal_failure_stage = "repair"
        except ModelSchemaError as exc:
            last_output_sha256 = exc.response_sha256
            proposal = None
            result = _machine_verifier_result(
                sanitized_task["task_id"],
                code="SCHEMA_VALID",
                field=None,
                message=f"{stage} repair output did not satisfy the response schema",
                repair_count=repair_count,
                allow_repair=False,
            )
            if exc.finish_reason == "stop" and exc.structured_output_applied is True:
                terminal_model_failure_code = "STRUCTURED_OUTPUT_CONTRACT_BROKEN"
                terminal_failure_stage = "repair"
        except ContextOverflowError:
            result = _machine_verifier_result(
                sanitized_task["task_id"],
                code="MODEL_GENERATION_FAILED",
                field=None,
                message=f"{stage} repair model generation failed",
                repair_count=repair_count,
                allow_repair=False,
            )
            append_verifier_trace(
                stage=stage,
                repair_attempt=True,
                proposal=None,
                verifier_result=result,
            )
            raise
        except ModelAdapterError:
            result = _machine_verifier_result(
                sanitized_task["task_id"],
                code="MODEL_GENERATION_FAILED",
                field=None,
                message=f"{stage} repair model generation failed",
                repair_count=repair_count,
                allow_repair=False,
            )
            append_verifier_trace(
                stage=stage,
                repair_attempt=True,
                proposal=None,
                verifier_result=result,
            )
            raise
        append_verifier_trace(
            stage=stage,
            repair_attempt=True,
            proposal=proposal,
            verifier_result=result,
        )
        return proposal, result

    try:
        if runtime_mode == "direct":
            initial_prompt = build_direct_prompt(
                context_pack, prompt_condition=prompt_condition
            )
            final_proposal, verifier_result = generate_with_single_repair(
                initial_prompt,
                stage="direct",
                expected_action="FINAL",
                max_tokens=min(
                    int(model_config["max_answer_tokens"]),
                    256 if sanitized_task["task_type"] == "quant_metric" else 768,
                ),
                release_observation=release_observation,
                repair_prompt_observation=None,
            )
        elif runtime_mode == "capability_agent":
            plan_prompt = build_plan_prompt(
                context_pack, prompt_condition=prompt_condition
            )
            plan_proposal, plan_result = generate_with_single_repair(
                plan_prompt,
                stage="plan",
                expected_action="CALL_TOOL",
                max_tokens=min(int(model_config["max_plan_tokens"]), 192),
                release_observation=None,
                repair_prompt_observation=None,
            )
            if (
                plan_result["disposition"] == "SAFE_REFUSAL"
                and plan_proposal is not None
            ):
                final_proposal = copy.deepcopy(plan_proposal)
                return final_record(
                    verifier_result=plan_result,
                    outcome="SAFE_REFUSAL",
                    output=final_proposal,
                )
            if (
                plan_result["disposition"] != "TOOL_CALL_APPROVED"
                or plan_proposal is None
            ):
                failure_code = terminal_model_failure_code or (
                    "REPAIR_EXHAUSTED" if repair_count else "PROPOSAL_REJECTED"
                )
                return failure_result(
                    failure_code,
                    terminal_failure_stage
                    or ("repair" if failure_code == "REPAIR_EXHAUSTED" else "plan"),
                    "bounded-agent planning did not yield an approved tool call",
                    verifier_result=plan_result,
                )

            tool_name = plan_proposal["tool_name"]
            tool_arguments = plan_proposal["tool_arguments"]
            tool_started = time.perf_counter()
            try:
                observation = tool_executor(  # type: ignore[misc]
                    tool_name,
                    copy.deepcopy(tool_arguments),
                    copy.deepcopy(sanitized_task),
                )
            except Exception:  # noqa: BLE001 - executor failures become typed results
                tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "arguments_sha256": _sha256_json(tool_arguments),
                        "observation_sha256": None,
                        "status": "ERROR",
                        "duration_ms": (time.perf_counter() - tool_started) * 1000.0,
                    }
                )
                return failure_result(
                    "TOOL_EXECUTION_FAILED",
                    "tool",
                    "registered deterministic tool execution failed",
                )
            observation_sha256 = None
            observation_valid = isinstance(observation, Mapping)
            if observation_valid:
                tool_observation = copy.deepcopy(dict(observation))
                try:
                    observation_sha256 = _sha256_json(tool_observation)
                    validate_tool_observation(tool_observation)
                except (TypeError, ValueError):
                    observation_valid = False
            if not observation_valid:
                tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "arguments_sha256": _sha256_json(tool_arguments),
                        "observation_sha256": observation_sha256,
                        "status": "ERROR",
                        "duration_ms": (time.perf_counter() - tool_started) * 1000.0,
                    }
                )
                tool_observation = None
                return failure_result(
                    "TOOL_OBSERVATION_MISMATCH",
                    "tool",
                    "tool observation did not satisfy the registered observation contract",
                )
            if _canonical_json(tool_observation) != _canonical_json(
                release_observation
            ):
                tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "arguments_sha256": _sha256_json(tool_arguments),
                        "observation_sha256": observation_sha256,
                        "status": "ERROR",
                        "duration_ms": (time.perf_counter() - tool_started) * 1000.0,
                    }
                )
                tool_observation = None
                return failure_result(
                    "TOOL_OBSERVATION_MISMATCH",
                    "tool",
                    "tool observation did not match independent deterministic replay",
                )
            tool_calls.append(
                {
                    "tool_name": tool_name,
                    "arguments_sha256": _sha256_json(tool_arguments),
                    "observation_sha256": observation_sha256,
                    "status": "OK",
                    "duration_ms": (time.perf_counter() - tool_started) * 1000.0,
                }
            )

            synthesis_prompt = build_synthesis_prompt(
                context_pack,
                tool_observation,
                prompt_condition=prompt_condition,
            )
            final_proposal, verifier_result = generate_with_single_repair(
                synthesis_prompt,
                stage="synthesis",
                expected_action="FINAL",
                max_tokens=min(
                    int(model_config["max_answer_tokens"]),
                    256 if sanitized_task["task_type"] == "quant_metric" else 768,
                ),
                release_observation=release_observation,
                repair_prompt_observation=tool_observation,
            )
        else:  # safety_hybrid
            tool_name = task_operation_spec(
                sanitized_task["task_type"], version=contract_version
            ).operation
            tool_arguments = expected_tool_arguments(sanitized_task)
            tool_started = time.perf_counter()
            try:
                observation = tool_executor(  # type: ignore[misc]
                    tool_name,
                    copy.deepcopy(tool_arguments),
                    copy.deepcopy(sanitized_task),
                )
            except Exception:  # noqa: BLE001 - typed deterministic boundary
                tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "arguments_sha256": _sha256_json(tool_arguments),
                        "observation_sha256": None,
                        "status": "ERROR",
                        "duration_ms": (time.perf_counter() - tool_started) * 1000.0,
                    }
                )
                return failure_result(
                    "TOOL_EXECUTION_FAILED",
                    "tool",
                    "registered deterministic tool execution failed",
                )
            observation_valid = isinstance(observation, Mapping)
            observation_sha256 = None
            if observation_valid:
                tool_observation = copy.deepcopy(dict(observation))
                if model_independent_case:
                    # Visibility is orchestrator metadata, not a tool-selected
                    # capability.  No model observes quantitative hybrid output.
                    tool_observation["visibility"] = "VERIFIER_ONLY"
                try:
                    observation_sha256 = _sha256_json(tool_observation)
                    validate_tool_observation(tool_observation)
                except (TypeError, ValueError):
                    observation_valid = False
            if not observation_valid or _canonical_json(
                tool_observation
            ) != _canonical_json(release_observation):
                tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "arguments_sha256": _sha256_json(tool_arguments),
                        "observation_sha256": observation_sha256,
                        "status": "ERROR",
                        "duration_ms": (time.perf_counter() - tool_started) * 1000.0,
                    }
                )
                tool_observation = None
                return failure_result(
                    "TOOL_OBSERVATION_MISMATCH",
                    "tool",
                    "tool observation did not match independent deterministic replay",
                )
            tool_calls.append(
                {
                    "tool_name": tool_name,
                    "arguments_sha256": _sha256_json(tool_arguments),
                    "observation_sha256": observation_sha256,
                    "status": "OK",
                    "duration_ms": (time.perf_counter() - tool_started) * 1000.0,
                }
            )
            if sanitized_task["task_type"] == "quant_metric":
                refusal_code = None
                if context_pack["security_flags"]:
                    refusal_code = "PROMPT_INJECTION_DETECTED"
                elif tool_observation["status"] == "REFUSAL":
                    refusal_code = tool_observation["refusal_code"]
                if refusal_code is not None:
                    final_proposal = {
                        "agent_proposal_version": "v2",
                        "task_id": sanitized_task["task_id"],
                        "action": "REFUSE",
                        "tool_name": None,
                        "tool_arguments": None,
                        "status": "REFUSAL",
                        "value": None,
                        "unit": None,
                        "period_key": sanitized_task["period"]["period_key"],
                        "answer_text": None,
                        "evidence_ids": [],
                        "claims": [],
                        "refusal_code": refusal_code,
                        "model_uncertainty": "HIGH",
                        "model_escalation_requested": False,
                    }
                else:
                    final_proposal = {
                        "agent_proposal_version": "v2",
                        "task_id": sanitized_task["task_id"],
                        "action": "ANSWER",
                        "tool_name": None,
                        "tool_arguments": None,
                        "status": "OK",
                        "value": tool_observation["value"],
                        "unit": tool_observation["unit"],
                        "period_key": sanitized_task["period"]["period_key"],
                        "answer_text": None,
                        "evidence_ids": list(tool_observation["evidence_ids"]),
                        "claims": [],
                        "refusal_code": None,
                        "model_uncertainty": "NONE",
                        "model_escalation_requested": False,
                    }
                verifier_result = _stage_verification(
                    sanitized_task,
                    final_proposal,
                    expected_action="FINAL",
                    evidence_items=evidence_items,
                    tool_observation=release_observation,
                    repair_count=0,
                    stage="synthesis",
                )
                if verifier_result["disposition"] not in {"RELEASED", "SAFE_REFUSAL"}:
                    return failure_result(
                        "DETERMINISTIC_REPLAY_MISMATCH",
                        "replay",
                        "deterministic quantitative materialization was not releasable",
                        verifier_result=verifier_result,
                    )
            else:
                synthesis_prompt = build_synthesis_prompt(
                    context_pack,
                    tool_observation,
                    prompt_condition=prompt_condition,
                )
                final_proposal, verifier_result = generate_with_single_repair(
                    synthesis_prompt,
                    stage="synthesis",
                    expected_action="FINAL",
                    max_tokens=min(int(model_config["max_answer_tokens"]), 768),
                    release_observation=release_observation,
                    repair_prompt_observation=tool_observation,
                )
    except ContextOverflowError:
        return failure_result(
            "CONTEXT_OVERFLOW",
            "context",
            "rendered model request exceeded the locked input-token cap",
        )
    except ModelTimeoutError:
        return failure_result(
            "MODEL_TIMEOUT",
            "model",
            "offline model generation timed out",
        )
    except ModelOOMError:
        return failure_result(
            "MODEL_OOM",
            "model",
            "offline model generation exhausted device memory",
        )
    except ModelBackendError:
        return failure_result(
            "BACKEND_FAILURE",
            "model",
            "offline model backend failed",
        )
    except ModelAdapterError:
        return failure_result(
            "MODEL_ADAPTER_FAILED",
            "model",
            "offline model generation failed",
        )

    if verifier_result["disposition"] in {"RELEASED", "SAFE_REFUSAL"}:
        return final_record(
            verifier_result=verifier_result,
            outcome=verifier_result["disposition"],
            output=final_proposal,
        )
    failure_code = terminal_model_failure_code or (
        "REPAIR_EXHAUSTED" if repair_count else "PROPOSAL_REJECTED"
    )
    return failure_result(
        failure_code,
        terminal_failure_stage
        or ("repair" if failure_code == "REPAIR_EXHAUSTED" else "verification"),
        "final model proposal was not releasable",
        verifier_result=verifier_result,
    )
