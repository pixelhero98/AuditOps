from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from auditops.agent_context import build_context_pack, validate_context_pack
from auditops.agent_contracts import (
    VERIFIER_CHECK_CODES,
    ContractValidationError,
    semantic_model_config,
    validate_agent_case_result,
    validate_agent_proposal,
    validate_agent_run_record,
    validate_agent_task_input,
    validate_failure,
    validate_model_config,
    validate_tool_observation,
    validate_verifier_result,
)
from auditops.agent_failures import (
    FAILURE_SPECS,
    failure_spec,
    validate_failure_binding,
)
from auditops.agent_operations import (
    TASK_OPERATION_SPECS,
    expected_tool_arguments,
    task_operation_spec,
    validate_task_operation_binding,
)
from auditops.agent_prompts import (
    build_plan_prompt,
    build_synthesis_prompt,
    plan_response_schema,
    terminal_response_schema,
)
from auditops.agent_tools import load_frozen_evidence, metric_operation_contract
from auditops.canonical_json import canonical_json_sha256

jsonschema = pytest.importorskip("jsonschema")
referencing = pytest.importorskip("referencing")

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "auditops" / "specs"
SHA256 = "a" * 64

V2_SCHEMA_FILES = {
    "agent_case_result.v2.schema.json",
    "agent_proposal.v2.schema.json",
    "agent_run_record.v2.schema.json",
    "agent_task_input.v2.schema.json",
    "context_pack.v2.schema.json",
    "failure.v2.schema.json",
    "model_config.v2.schema.json",
    "narrative_answer_or_refusal.v2.schema.json",
    "quantitative_answer_or_refusal.v2.schema.json",
    "tool_arguments.v2.schema.json",
    "tool_observation.v2.schema.json",
    "verifier_result.v2.schema.json",
    "vlm_diagnostic_result.v1.schema.json",
    "vlm_diagnostic_task.v1.schema.json",
}
LEGACY_V21_SCHEMA_FILES = {
    "agent_case_result.v2.1.schema.json",
    "agent_plan_response.v2.1.schema.json",
    "agent_proposal.v2.1.schema.json",
    "agent_run_record.v2.1.schema.json",
    "agent_task_input.v2.1.schema.json",
    "agent_terminal_response.v2.1.schema.json",
    "context_pack.v2.1.schema.json",
    "failure.v2.1.schema.json",
    "tool_arguments.v2.1.schema.json",
    "tool_observation.v2.1.schema.json",
}
V22_SCHEMA_FILES = {
    "agent_case_result.v2.2.schema.json",
    "agent_plan_response.v2.2.schema.json",
    "agent_proposal.v2.2.schema.json",
    "agent_run_record.v2.2.schema.json",
    "agent_task_input.v2.2.schema.json",
    "agent_terminal_response.v2.2.schema.json",
    "context_pack.v2.2.schema.json",
    "failure.v2.2.schema.json",
    "narrative_selection_or_refusal.v2.2.schema.json",
    "narrative_selection_response.v2.2.schema.json",
    "quantitative_answer_or_refusal.v2.2.schema.json",
    "tool_arguments.v2.2.schema.json",
    "tool_observation.v2.2.schema.json",
}


def _schemas() -> dict[str, dict]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in SPECS.glob("*.schema.json")
    }


def _registry(schemas: dict[str, dict]):
    registry = referencing.Registry()
    for filename, schema in schemas.items():
        resource = referencing.Resource.from_contents(schema)
        registry = registry.with_resource(filename, resource)
        registry = registry.with_resource(schema["$id"], resource)
    return registry


def _errors(filename: str, instance: object) -> list[str]:
    schemas = _schemas()
    validator = jsonschema.Draft202012Validator(
        schemas[filename], registry=_registry(schemas)
    )
    return [error.message for error in validator.iter_errors(instance)]


def _assert_valid(filename: str, instance: object) -> None:
    assert _errors(filename, instance) == []


def _quant_task(*, evidence_ids: list[str] | None = None) -> dict:
    ids = ["fact-assets"] if evidence_ids is None else list(evidence_ids)
    return {
        "agent_task_input_version": "v2.2",
        "task_id": "us-quant-001",
        "task_type": "quant_metric",
        "jurisdiction": "US",
        "reporting_framework": "US-GAAP",
        "standards_profile": {"name": "PCAOB filing verification", "version": "2026"},
        "source_system": "SEC_EDGAR",
        "question": "Calculate the current ratio.",
        "entity": {"entity_id": "0001", "name": "Fixture Corp"},
        "filing": {"filing_id": "fixture-10k", "form_type": "10-K"},
        "period": {"period_key": "ASOF_20251231", "instant": "2025-12-31"},
        "task_parameters": {
            "metric_spec_id": "current_ratio",
            "retrieval_query": None,
        },
        "narrative_subtype": None,
        "metric_operation_contract": metric_operation_contract("current_ratio"),
        "allowed_tools": ["evaluate_metric_spec"],
        "evidence_scope": {
            "evidence_ids": ids,
            "filing_ids": ["fixture-10k"],
            "max_items": len(ids),
            "selection_method": "TYPED_FACT_MATERIALIZATION",
            "empty_reason": None if ids else "NO_VALID_FACTS",
        },
        "output_schema_id": "quantitative_answer_or_refusal.v2.2",
        "refusal_policy": {
            "allowed_codes": ["MISSING_INPUT", "PROMPT_INJECTION_DETECTED"]
        },
    }


def _narrative_task(*, evidence_ids: list[str] | None = None) -> dict:
    ids = ["chunk-1"] if evidence_ids is None else list(evidence_ids)
    task = _quant_task(evidence_ids=[])
    task.update(
        {
            "task_id": "us-narrative-001",
            "task_type": "narrative_citation",
            "question": "What audit opinion language is disclosed?",
            "task_parameters": {
                "metric_spec_id": None,
                "retrieval_query": "audit opinion language",
            },
            "narrative_subtype": "auditor_report_opinion_language",
            "metric_operation_contract": None,
            "allowed_tools": ["load_frozen_evidence"],
            "evidence_scope": {
                "evidence_ids": ids,
                "filing_ids": ["fixture-10k"],
                "max_items": len(ids),
                "selection_method": "FROZEN_BM25",
                "empty_reason": None if ids else "NO_MATCHING_EVIDENCE",
            },
            "output_schema_id": "narrative_selection_or_refusal.v2.2",
            "refusal_policy": {
                "allowed_codes": [
                    "NARRATIVE_NOT_SUPPORTED",
                    "PROMPT_INJECTION_DETECTED",
                ]
            },
        }
    )
    return task


def _quant_answer(value: str = "2") -> dict:
    return {
        "agent_proposal_version": "v2",
        "task_id": "us-quant-001",
        "action": "ANSWER",
        "tool_name": None,
        "tool_arguments": None,
        "status": "OK",
        "value": value,
        "unit": "pure",
        "period_key": "ASOF_20251231",
        "answer_text": None,
        "evidence_ids": ["fact-assets"],
        "claims": [],
        "refusal_code": None,
        "model_uncertainty": "NONE",
        "model_escalation_requested": False,
    }


def _model_config() -> dict:
    return {
        "model_config_version": "v2",
        "model_id": "auditops/mock",
        "model_path": None,
        "revision": "fixture",
        "quantization": None,
        "backend": "mock",
        "context_window_tokens": 16384,
        "max_input_tokens": 14336,
        "max_plan_tokens": 192,
        "max_answer_tokens": 768,
        "max_repair_tokens": 768,
        "temperature": 0,
        "top_p": 1,
        "seed": 20260821,
        "thinking_enabled": False,
        "chat_template_sha256": None,
    }


def _run_record(outcome: str = "INFRASTRUCTURE_FAILURE") -> dict:
    model_config = _model_config()
    run_envelope_sha256 = "b" * 64
    return {
        "agent_run_record_version": "v2.2",
        "run_id": "run-" + run_envelope_sha256[:24],
        "task_id": "us-quant-001",
        "corpus_id": "fixture-corpus",
        "benchmark_manifest_sha256": SHA256,
        "runtime_mode": "direct",
        "runtime_version": "auditops.agent_runtime.v2.2",
        "prompt_condition": "zero_shot",
        "prompt_version": "auditops.agent_prompt.v2.2",
        "prompt_sha256": canonical_json_sha256([]),
        "canonical_json_version": "auditops.canonical_json.v1",
        "semantic_model_config_sha256": canonical_json_sha256(
            semantic_model_config(model_config)
        ),
        "deployment_model_config_sha256": canonical_json_sha256(model_config),
        "deterministic_executor_sha256": SHA256,
        "stage_schema_bundle_sha256": SHA256,
        "run_envelope_sha256": run_envelope_sha256,
        "demonstration_pack_sha256": None,
        "narrative_subtype": None,
        "metric_operation_contract_sha256": canonical_json_sha256(
            metric_operation_contract("current_ratio")
        ),
        "model_config": model_config,
        "context_sha256": None,
        "input_sha256": SHA256,
        "output_sha256": None,
        "tool_calls": [],
        "verifier_trace": [],
        "verifier_result": None,
        "repair_count": 0,
        "outcome": outcome,
        "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "timings": {
            "started_at": "2026-08-23T00:00:00Z",
            "finished_at": "2026-08-23T00:00:01Z",
            "duration_ms": 1000,
            "engine_initialization_ms": 0,
            "generation_duration_ms": 0,
            "tool_duration_ms": 0,
        },
        "resources": {
            "slurm_job_id": None,
            "gpu_model": None,
            "peak_vram_bytes": None,
            "host": "fixture",
        },
        "provenance": {
            "git_commit": None,
            "source_tree_sha256": None,
            "container_digest": None,
            "model_snapshot_sha256": None,
            "tokenizer_revision": None,
            "package_lock_sha256": None,
        },
    }


def test_all_v2_and_separate_vlm_schemas_are_valid_draft_2020_12() -> None:
    schemas = _schemas()
    schema_files = V2_SCHEMA_FILES | LEGACY_V21_SCHEMA_FILES | V22_SCHEMA_FILES
    assert schema_files <= set(schemas)
    for filename in sorted(schema_files):
        schema = schemas[filename]
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        jsonschema.Draft202012Validator.check_schema(schema)


def test_representative_python_validators_and_json_schemas_reject_the_same_boundaries() -> (
    None
):
    task = _quant_task()
    proposal = _quant_answer()
    model = _model_config()
    validate_agent_task_input(task)
    validate_agent_proposal(proposal)
    validate_model_config(model)
    _assert_valid("agent_task_input.v2.2.schema.json", task)
    _assert_valid("agent_proposal.v2.schema.json", proposal)
    _assert_valid("model_config.v2.schema.json", model)

    invalid_task = copy.deepcopy(task)
    invalid_task["allowed_tools"] = ["load_frozen_evidence"]
    with pytest.raises(ContractValidationError):
        validate_agent_task_input(invalid_task)
    assert _errors("agent_task_input.v2.2.schema.json", invalid_task)

    invalid_proposal = copy.deepcopy(proposal)
    invalid_proposal["value"] = "2.0"
    with pytest.raises(ContractValidationError):
        validate_agent_proposal(invalid_proposal)
    assert _errors("agent_proposal.v2.schema.json", invalid_proposal)

    invalid_model = copy.deepcopy(model)
    invalid_model["seed"] = 1
    with pytest.raises(ContractValidationError):
        validate_model_config(invalid_model)
    assert _errors("model_config.v2.schema.json", invalid_model)


def test_operation_registry_is_exact_closed_and_builds_current_tool_interfaces() -> (
    None
):
    assert set(TASK_OPERATION_SPECS) == {"quant_metric", "narrative_citation"}
    quant = task_operation_spec("quant_metric")
    narrative = task_operation_spec("narrative_citation")
    assert quant.runtime_id == narrative.runtime_id == "auditops.agent_runtime.v2.2"
    assert quant.operation == "evaluate_metric_spec"
    assert quant.terminal_output_schema_id == "quantitative_answer_or_refusal.v2.2"
    assert narrative.operation == "load_frozen_evidence"
    assert narrative.terminal_output_schema_id == "narrative_selection_or_refusal.v2.2"
    assert expected_tool_arguments(_quant_task()) == {
        "filing_id": "fixture-10k",
        "metric_spec_id": "current_ratio",
        "period_key": "ASOF_20251231",
    }
    assert expected_tool_arguments(_narrative_task()) == {
        "filing_id": "fixture-10k",
        "evidence_scope_sha256": canonical_json_sha256(
            _narrative_task()["evidence_scope"]
        ),
    }
    validate_task_operation_binding(_quant_task())
    with pytest.raises(ValueError, match="Unregistered text-agent task type"):
        task_operation_spec("vlm_diagnostic")


def test_registry_schema_ids_arguments_and_refusal_policies_cannot_drift() -> None:
    schemas = _schemas()
    argument_defs = schemas["tool_arguments.v2.2.schema.json"]["$defs"]
    fixtures = {
        "quant_metric": (_quant_task(), "metric"),
        "narrative_citation": (_narrative_task(), "frozenEvidence"),
    }
    for task_type, (task, argument_def) in fixtures.items():
        spec = task_operation_spec(task_type)
        terminal_filename = f"{spec.terminal_output_schema_id}.schema.json"
        assert schemas[terminal_filename]["$id"] == spec.terminal_output_schema_id
        assert set(argument_defs[argument_def]["required"]) == set(
            expected_tool_arguments(task)
        )
        task["refusal_policy"]["allowed_codes"] = sorted(spec.permitted_refusal_codes)
        validate_task_operation_binding(task)
        _assert_valid("agent_task_input.v2.2.schema.json", task)

    invalid = _quant_task()
    invalid["refusal_policy"]["allowed_codes"].append("NARRATIVE_NOT_SUPPORTED")
    with pytest.raises(ValueError, match="not permitted"):
        validate_task_operation_binding(invalid)
    assert _errors("agent_task_input.v2.2.schema.json", invalid)


@pytest.mark.parametrize(
    "task",
    [
        _quant_task(),
        _narrative_task(),
        _quant_task(evidence_ids=[]),
        _narrative_task(evidence_ids=[]),
    ],
)
def test_task_schema_accepts_bound_text_tasks_and_zero_to_five_evidence(
    task: dict,
) -> None:
    _assert_valid("agent_task_input.v2.2.schema.json", task)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_type", "vlm_diagnostic"),
        ("allowed_tools", []),
        ("allowed_tools", ["load_frozen_evidence"]),
        ("allowed_tools", ["evaluate_metric_spec", "load_frozen_evidence"]),
        ("output_schema_id", "narrative_answer_or_refusal.v2"),
    ],
)
def test_task_schema_rejects_vlm_and_inexact_operation_bindings(
    field: str, value: object
) -> None:
    task = _quant_task()
    task[field] = value
    assert _errors("agent_task_input.v2.2.schema.json", task)


def test_task_schema_requires_typed_empty_scope_origin() -> None:
    task = _quant_task(evidence_ids=[])
    task["evidence_scope"]["empty_reason"] = None
    assert _errors("agent_task_input.v2.2.schema.json", task)
    task = _quant_task()
    task["evidence_scope"]["empty_reason"] = "NO_VALID_FACTS"
    assert _errors("agent_task_input.v2.2.schema.json", task)


def test_proposal_schema_enforces_action_truth_table_and_tool_pairing() -> None:
    call = {
        "agent_proposal_version": "v2",
        "task_id": "us-quant-001",
        "action": "CALL_TOOL",
        "tool_name": "evaluate_metric_spec",
        "tool_arguments": expected_tool_arguments(_quant_task()),
        "status": None,
        "value": None,
        "unit": None,
        "period_key": None,
        "answer_text": None,
        "evidence_ids": [],
        "claims": [],
        "refusal_code": None,
        "model_uncertainty": "NONE",
        "model_escalation_requested": False,
    }
    _assert_valid("agent_proposal.v2.schema.json", call)
    wrong = copy.deepcopy(call)
    wrong["tool_name"] = "load_frozen_evidence"
    assert _errors("agent_proposal.v2.2.schema.json", wrong)
    wrong = copy.deepcopy(call)
    wrong["period_key"] = "ASOF_20251231"
    assert _errors("agent_proposal.v2.schema.json", wrong)
    wrong = copy.deepcopy(call)
    wrong["model_uncertainty"] = "LOW"
    assert _errors("agent_proposal.v2.schema.json", wrong)


@pytest.mark.parametrize("value", ["2", "2.5", "0", "-0.25", "100.01"])
def test_quantitative_answer_requires_canonical_decimal_string(value: str) -> None:
    answer = _quant_answer(value)
    _assert_valid("agent_proposal.v2.schema.json", answer)
    _assert_valid("quantitative_answer_or_refusal.v2.schema.json", answer)


@pytest.mark.parametrize("value", [2, 2.0, "2.0", "02", "2e0", "-0", "0.00"])
def test_quantitative_answer_rejects_noncanonical_lexical_forms(value: object) -> None:
    assert _errors("agent_proposal.v2.schema.json", _quant_answer(value))


def test_narrative_answer_and_refusal_are_extractive_and_citation_free_respectively() -> (
    None
):
    answer = {
        **_quant_answer(),
        "task_id": "us-narrative-001",
        "value": None,
        "unit": None,
        "answer_text": "The auditor expressed an unqualified opinion.",
        "evidence_ids": ["chunk-1"],
        "claims": [
            {
                "claim_id": "claim-1",
                "text": "The auditor expressed an unqualified opinion.",
                "evidence_ids": ["chunk-1"],
                "supporting_text": "expressed an unqualified opinion",
            }
        ],
    }
    _assert_valid("narrative_answer_or_refusal.v2.schema.json", answer)
    refusal = {
        **answer,
        "action": "REFUSE",
        "status": "REFUSAL",
        "answer_text": None,
        "evidence_ids": [],
        "claims": [],
        "refusal_code": "NARRATIVE_NOT_SUPPORTED",
        "model_uncertainty": "HIGH",
    }
    _assert_valid("narrative_answer_or_refusal.v2.schema.json", refusal)
    refusal["evidence_ids"] = ["chunk-1"]
    assert _errors("agent_proposal.v2.schema.json", refusal)


def test_narrative_claim_count_has_no_schema_only_cap() -> None:
    answer = {
        **_quant_answer(),
        "task_id": "us-narrative-001",
        "value": None,
        "unit": None,
        "answer_text": "The auditor expressed an unqualified opinion.",
        "evidence_ids": ["chunk-1"],
        "claims": [
            {
                "claim_id": f"claim-{index}",
                "text": "The auditor expressed an unqualified opinion.",
                "evidence_ids": ["chunk-1"],
                "supporting_text": "expressed an unqualified opinion",
            }
            for index in range(1, 7)
        ],
    }
    validate_agent_proposal(answer)
    _assert_valid("agent_proposal.v2.schema.json", answer)


def test_stage_specific_schemas_exclude_wrong_actions() -> None:
    task = _quant_task()
    evidence = [
        {
            "evidence_id": "fact-assets",
            "filing_id": "fixture-10k",
            "content": "input_name=assets_current; value=100",
            "period_key": "ASOF_20251231",
            "unit": "USD",
            "value": "100",
            "metadata": {
                "input_name": "assets_current",
                "context_id": "ctx",
                "entity_id": "0001",
                "dimensions": {},
            },
        }
    ]
    plan = expected_tool_arguments(task)
    plan_payload = {
        "task_id": task["task_id"],
        "action": "CALL_TOOL",
        "tool_name": "evaluate_metric_spec",
        "tool_arguments": plan,
    }
    answer_payload = {
        "task_id": task["task_id"],
        "action": "ANSWER",
        "period_key": "ASOF_20251231",
        "value": "2",
        "unit": "pure",
        "evidence_ids": ["fact-assets"],
    }
    plan_validator = jsonschema.Draft202012Validator(plan_response_schema(task))
    terminal_validator = jsonschema.Draft202012Validator(
        terminal_response_schema(task, evidence)
    )
    assert not list(plan_validator.iter_errors(plan_payload))
    assert list(plan_validator.iter_errors(answer_payload))
    assert not list(terminal_validator.iter_errors(answer_payload))
    assert list(terminal_validator.iter_errors(plan_payload))


def test_narrative_stage_schema_bounds_claims_evidence_and_text() -> None:
    task = _narrative_task()
    evidence = [
        {
            "evidence_id": "chunk-1",
            "filing_id": "fixture-10k",
            "content": "The auditor expressed an unqualified opinion.",
            "rank": 1,
        }
    ]
    validator = jsonschema.Draft202012Validator(
        terminal_response_schema(task, evidence)
    )
    quote = "The auditor expressed an unqualified opinion."
    answer = {
        "task_id": task["task_id"],
        "action": "ANSWER",
        "period_key": task["period"]["period_key"],
        "extracts": [{"evidence_id": "chunk-1", "exact_quote": quote}],
    }
    assert not list(validator.iter_errors(answer))
    too_many_extracts = copy.deepcopy(answer)
    too_many_extracts["extracts"] = [
        {"evidence_id": "chunk-1", "exact_quote": f"quote-{index}"}
        for index in range(4)
    ]
    assert list(validator.iter_errors(too_many_extracts))
    oversized = copy.deepcopy(answer)
    oversized["extracts"][0]["exact_quote"] = "x" * 1001
    assert list(validator.iter_errors(oversized))
    forged = copy.deepcopy(answer)
    forged["extracts"][0]["evidence_id"] = "forged-evidence"
    assert list(validator.iter_errors(forged))


def test_capability_narrative_hides_evidence_until_one_load_observation() -> None:
    task = _narrative_task()
    marker = "UNIQUE-FROZEN-EVIDENCE-MARKER"
    evidence = [
        {
            "evidence_id": "chunk-1",
            "filing_id": "fixture-10k",
            "content": marker,
            "rank": 1,
        }
    ]
    context = build_context_pack(task, evidence, token_counter=lambda _: 100)
    plan = build_plan_prompt(context)
    assert marker not in "\n".join(message["content"] for message in plan.messages)

    observation = load_frozen_evidence(task, evidence, visibility="MODEL_VISIBLE")
    synthesis = build_synthesis_prompt(context, observation)
    rendered = "\n".join(message["content"] for message in synthesis.messages)
    assert rendered.count(marker) == 1


def test_observation_schema_separates_visibility_and_variants() -> None:
    metric = {
        "tool_observation_version": "v2.2",
        "observation_kind": "METRIC",
        "visibility": "MODEL_VISIBLE",
        "task_id": "us-quant-001",
        "filing_id": "fixture-10k",
        "metric_spec_id": "current_ratio",
        "period_key": "ASOF_20251231",
        "status": "OK",
        "value": "2",
        "unit": "pure",
        "evidence_ids": ["fact-assets"],
        "refusal_code": None,
    }
    narrative_evidence = [
        {
            "evidence_id": "chunk-1",
            "filing_id": "fixture-10k",
            "content": "The auditor expressed an unqualified opinion.",
            "rank": 1,
        }
    ]
    retrieval = load_frozen_evidence(
        _narrative_task(), narrative_evidence, visibility="VERIFIER_ONLY"
    )
    _assert_valid("tool_observation.v2.2.schema.json", metric)
    _assert_valid("tool_observation.v2.2.schema.json", retrieval)
    retrieval["evidence_scope_sha256"] = "not-a-hash"
    assert _errors("tool_observation.v2.2.schema.json", retrieval)


def test_tool_observation_python_and_schema_validators_stay_in_lockstep() -> None:
    metric = {
        "tool_observation_version": "v2.2",
        "observation_kind": "METRIC",
        "visibility": "MODEL_VISIBLE",
        "task_id": "us-quant-001",
        "filing_id": "fixture-10k",
        "metric_spec_id": "current_ratio",
        "period_key": "ASOF_20251231",
        "status": "OK",
        "value": "2",
        "unit": "pure",
        "evidence_ids": ["fact-assets"],
        "refusal_code": None,
    }
    validate_tool_observation(metric)
    _assert_valid("tool_observation.v2.2.schema.json", metric)

    for field, invalid_value in (
        ("value", "2.0"),
        ("visibility", "INTERNAL"),
        ("evidence_ids", []),
    ):
        invalid = copy.deepcopy(metric)
        invalid[field] = invalid_value
        with pytest.raises(ContractValidationError):
            validate_tool_observation(invalid)
        assert _errors("tool_observation.v2.2.schema.json", invalid)


def test_failure_registry_and_schema_agree_on_every_code() -> None:
    schema_codes = set(
        _schemas()["failure.v2.2.schema.json"]["properties"]["code"]["enum"]
    )
    assert schema_codes == set(FAILURE_SPECS)
    for code, spec in FAILURE_SPECS.items():
        stage = min(spec.stages)
        validate_failure_binding(
            code=code,
            failure_class=spec.failure_class,
            stage=stage,
            retryable=spec.retryable,
        )
        payload = {
            "failure_version": "v2.2",
            "failure_class": spec.failure_class,
            "code": code,
            "stage": stage,
            "retryable": spec.retryable,
            "sanitized_detail": "Typed failure detail.",
        }
        _assert_valid("failure.v2.2.schema.json", payload)
    with pytest.raises(ValueError, match="Unregistered AuditOps v2 failure code"):
        failure_spec("UNKNOWN_FAILURE")


def test_context_case_and_run_outer_schemas_resolve_public_references() -> None:
    task = _quant_task()
    context = {
        "context_pack_version": "v2.2",
        "instructions": "Use only frozen evidence.",
        "task": task,
        "few_shot_examples": [],
        "evidence_items": [
            {
                "evidence_id": "fact-assets",
                "filing_id": "fixture-10k",
                "content": "input_name=assets_current; value=100",
                "rank": 1,
                "period_key": "ASOF_20251231",
                "unit": "USD",
                "value": "100",
                "metadata": {
                    "input_name": "assets_current",
                    "context_id": "ctx",
                    "entity_id": "0001",
                    "dimensions": {},
                },
            }
        ],
        "security_flags": [],
        "selection_provenance": {
            "selection_method": "TYPED_FACT_MATERIALIZATION",
            "empty_reason": None,
            "ordered_evidence_ids": ["fact-assets"],
        },
        "token_count": 100,
        "max_input_tokens": 14336,
        "context_sha256": SHA256,
    }
    _assert_valid("context_pack.v2.2.schema.json", context)
    run = _run_record()
    _assert_valid("agent_run_record.v2.2.schema.json", run)
    case = {
        "agent_case_result_version": "v2.2",
        "task_id": "us-quant-001",
        "outcome": "INFRASTRUCTURE_FAILURE",
        "plan_proposal": None,
        "final_proposal": None,
        "tool_observation": None,
        "run_record": run,
        "failure": {
            "failure_version": "v2.2",
            "failure_class": "INFRASTRUCTURE_FAILURE",
            "code": "CONTEXT_OVERFLOW",
            "stage": "context",
            "retryable": False,
            "sanitized_detail": "Complete context exceeded the fixed input cap.",
        },
    }
    _assert_valid("agent_case_result.v2.2.schema.json", case)


def test_context_run_case_and_failure_python_schema_parity() -> None:
    task = _quant_task()
    evidence = [
        {
            "evidence_id": "fact-assets",
            "filing_id": "fixture-10k",
            "content": "input_name=assets_current; value=100",
            "rank": 1,
            "period_key": "ASOF_20251231",
            "unit": "USD",
            "value": "100",
            "metadata": {
                "input_name": "assets_current",
                "context_id": "ctx",
                "entity_id": "0001",
                "dimensions": {},
            },
        }
    ]
    context = build_context_pack(task, evidence, token_counter=lambda _: 100)
    validate_context_pack(context)
    _assert_valid("context_pack.v2.2.schema.json", context)

    failure = {
        "failure_version": "v2.2",
        "failure_class": "INFRASTRUCTURE_FAILURE",
        "code": "CONTEXT_OVERFLOW",
        "stage": "context",
        "retryable": False,
        "sanitized_detail": "Complete context exceeded the fixed input cap.",
    }
    validate_failure(failure)
    _assert_valid("failure.v2.2.schema.json", failure)

    run = _run_record()
    validate_agent_run_record(run)
    _assert_valid("agent_run_record.v2.2.schema.json", run)
    case = {
        "agent_case_result_version": "v2.2",
        "task_id": "us-quant-001",
        "outcome": "INFRASTRUCTURE_FAILURE",
        "plan_proposal": None,
        "final_proposal": None,
        "tool_observation": None,
        "run_record": run,
        "failure": failure,
    }
    validate_agent_case_result(case)
    _assert_valid("agent_case_result.v2.2.schema.json", case)

    mismatched_case = copy.deepcopy(case)
    mismatched_case["outcome"] = "MODEL_FAILURE"
    mismatched_case["failure"] = {
        "failure_version": "v2.2",
        "failure_class": "MODEL_FAILURE",
        "code": "MODEL_GENERATION_FAILED",
        "stage": "direct",
        "retryable": False,
        "sanitized_detail": "Model output did not satisfy the terminal contract.",
    }
    with pytest.raises(ContractValidationError, match="CASE_RESULT_BINDING_INVALID"):
        validate_agent_case_result(mismatched_case)
    assert _errors("agent_case_result.v2.2.schema.json", mismatched_case)

    tampered_context = copy.deepcopy(context)
    tampered_context["selection_provenance"]["ordered_evidence_ids"] = []
    with pytest.raises(ContractValidationError):
        validate_context_pack(tampered_context)
    assert _errors("context_pack.v2.2.schema.json", tampered_context)

    wrong_failure = copy.deepcopy(failure)
    wrong_failure["failure_class"] = "MODEL_FAILURE"
    with pytest.raises(ContractValidationError):
        validate_failure(wrong_failure)
    assert _errors("failure.v2.2.schema.json", wrong_failure)


def test_terminal_repair_state_is_rejected_by_python_and_schema() -> None:
    run = _run_record(outcome="MODEL_FAILURE")
    proposal = _quant_answer()
    proposal_sha256 = canonical_json_sha256(proposal)
    repair_result = {
        "verifier_result_version": "v2",
        "task_id": run["task_id"],
        "passed": False,
        "release_allowed": False,
        "disposition": "REPAIR_REQUIRED",
        "repair_count": 0,
        "repair_allowed": True,
        "checks": [
            {
                "code": "ANSWER_SHAPE_VALID",
                "passed": False,
                "field": "proposal.value",
                "message": "quantitative value is not in canonical lexical form",
            }
        ],
        "repair_errors": [
            {
                "code": "ANSWER_SHAPE_VALID",
                "field": "proposal.value",
                "message": "quantitative value is not in canonical lexical form",
            }
        ],
    }
    run["verifier_trace"] = [
        {
            "stage": "direct",
            "attempt_index": 0,
            "repair_attempt": False,
            "prompt_sha256": SHA256,
            "request_sha256": SHA256,
            "response_schema_sha256": SHA256,
            "visible_context_sha256": SHA256,
            "output_sha256": proposal_sha256,
            "proposal": proposal,
            "proposal_sha256": proposal_sha256,
            "verifier_result": repair_result,
            "input_tokens": 10,
            "output_tokens": 5,
            "duration_ms": 1,
            "finish_reason": "stop",
            "constraint_backend": "mock_schema",
            "structured_output_applied": True,
            "output_bytes": 100,
            "cap_hit": False,
            "parse_category": None,
        }
    ]
    run["verifier_result"] = repair_result
    run["prompt_sha256"] = canonical_json_sha256([SHA256])
    run["output_sha256"] = proposal_sha256
    run["token_usage"] = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    with pytest.raises(ContractValidationError, match="TRACE_TRANSITION_INVALID"):
        validate_agent_run_record(run)
    assert _errors("agent_run_record.v2.2.schema.json", run)

    forged_terminal = copy.deepcopy(repair_result)
    forged_terminal.update(
        disposition="FAILED",
        repair_allowed=False,
        repair_errors=[],
    )
    run["verifier_result"] = forged_terminal
    with pytest.raises(ContractValidationError, match="TRACE_TRANSITION_INVALID"):
        validate_agent_run_record(run)
    assert _errors("agent_run_record.v2.2.schema.json", run)


def test_verifier_schema_never_exposes_empty_or_terminal_repair_errors() -> None:
    result = {
        "verifier_result_version": "v2",
        "task_id": "us-quant-001",
        "passed": False,
        "release_allowed": False,
        "disposition": "REPAIR_REQUIRED",
        "repair_count": 0,
        "repair_allowed": True,
        "checks": [
            {
                "code": "SCHEMA_VALID",
                "passed": False,
                "field": "proposal",
                "message": "Proposal does not match the schema.",
            }
        ],
        "repair_errors": [
            {
                "code": "SCHEMA_VALID",
                "field": "proposal",
                "message": "Proposal does not match the schema.",
            }
        ],
    }
    _assert_valid("verifier_result.v2.schema.json", result)
    result["repair_errors"] = []
    assert _errors("verifier_result.v2.schema.json", result)
    result.update(
        {
            "disposition": "FAILED",
            "repair_allowed": False,
            "repair_count": 1,
            "repair_errors": [
                {
                    "code": "SCHEMA_VALID",
                    "field": "proposal",
                    "message": "Proposal does not match the schema.",
                }
            ],
        }
    )
    assert _errors("verifier_result.v2.schema.json", result)


def test_verifier_check_code_schema_matches_python_registry() -> None:
    schema = _schemas()["verifier_result.v2.schema.json"]
    assert set(schema["$defs"]["checkCode"]["enum"]) == set(VERIFIER_CHECK_CODES)

    result = {
        "verifier_result_version": "v2",
        "task_id": "us-quant-001",
        "passed": True,
        "release_allowed": True,
        "disposition": "RELEASED",
        "repair_count": 0,
        "repair_allowed": False,
        "checks": [
            {
                "code": "UNREGISTERED_CHECK",
                "passed": True,
                "field": None,
                "message": "This code is not registered.",
            }
        ],
        "repair_errors": [],
    }
    with pytest.raises(ContractValidationError):
        validate_verifier_result(result)
    assert _errors("verifier_result.v2.schema.json", result)


def test_separate_vlm_contracts_wrap_the_existing_diagnostic_shapes() -> None:
    task = {
        "diagnostic_version": "companies-house-vlm-diagnostic.v1",
        "task_id": "ch-vlm-1",
        "pair_id": "pair-1",
        "company_number": "01234567",
        "filing_id": "tx-1",
        "pdf_path": "01234567/tx-1.pdf",
        "page_count": 2,
        "requested_facts": ["current assets"],
        "required_output": {
            "value": True,
            "page": True,
            "bbox": True,
            "evidence_text": True,
        },
        "bbox_gold_available": False,
    }
    success = {
        "task_id": "ch-vlm-1",
        "facts": [
            {
                "fact_label": "current assets",
                "value": "1000",
                "unit": "GBP",
                "period_context": "2025-12-31",
                "page": 1,
                "bbox": [0.1, 0.2, 0.5, 0.3],
                "evidence_text": "Current assets 1,000",
            }
        ],
    }
    failure = {
        "task_id": "ch-vlm-1",
        "failure": {
            "code": "PDF_NOT_READABLE",
            "message": "The diagnostic case did not produce a releasable extraction.",
        },
    }
    _assert_valid("vlm_diagnostic_task.v1.schema.json", task)
    _assert_valid("vlm_diagnostic_result.v1.schema.json", success)
    _assert_valid("vlm_diagnostic_result.v1.schema.json", failure)
    task["task_type"] = "vlm_diagnostic"
    assert _errors("vlm_diagnostic_task.v1.schema.json", task)
