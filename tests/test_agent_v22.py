from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from auditops.agent_context import build_context_pack
from auditops.agent_contracts import (
    ContractValidationError,
    build_agent_task_input,
    build_model_config,
    validate_agent_task_input,
    validate_tool_observation,
)
from auditops.agent_operations import task_operation_spec
from auditops.agent_prompts import (
    build_direct_prompt,
    normalize_stage_response,
    terminal_response_schema,
)
from auditops.agent_runtime import run_agent_case
from auditops.agent_tools import (
    execute_deterministic_tool,
    load_frozen_evidence,
    metric_operation_contract,
)
from auditops.agent_verifier import verify_proposal
from auditops.canonical_json import canonical_json_sha256
from auditops.model_adapter import MockModelAdapter

MANIFEST_SHA256 = "a" * 64


def _narrative_task(*, subtype: str = "accounting_policy") -> dict:
    spec = task_operation_spec("narrative_citation")
    return build_agent_task_input(
        task_id=f"narrative-{subtype}",
        task_type="narrative_citation",
        jurisdiction="US",
        reporting_framework="US-GAAP",
        standards_profile={"name": "PCAOB filing verification", "version": "2026"},
        source_system="SEC-EDGAR-CACHED",
        question="When is revenue recognized?",
        entity={"entity_id": "issuer-1", "name": "Issuer One"},
        filing={"filing_id": "filing-1", "form_type": "10-K"},
        period={"period_key": "FY2025"},
        retrieval_query="revenue recognition policy",
        narrative_subtype=subtype,
        allowed_tools=[spec.operation],
        evidence_ids=["chunk-1", "chunk-2"],
        output_schema_id=spec.terminal_output_schema_id,
        refusal_codes=["NARRATIVE_NOT_SUPPORTED"],
    )


def _narrative_evidence() -> list[dict]:
    return [
        {
            "evidence_id": "chunk-1",
            "filing_id": "filing-1",
            "content": "Revenue is recognized when control of the promised goods transfers to the customer.",
            "rank": 1,
            "period_key": "FY2025",
        },
        {
            "evidence_id": "chunk-2",
            "filing_id": "filing-1",
            "content": "This note also describes contract balances.",
            "rank": 2,
            "period_key": "FY2025",
        },
    ]


def _clock():
    values = iter(
        (
            datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
            datetime(2026, 8, 24, 12, 0, tzinfo=UTC) + timedelta(milliseconds=20),
        )
    )
    return lambda: next(values)


def _config(adapter: MockModelAdapter) -> dict:
    return build_model_config(
        model_id=adapter.model_id,
        revision=adapter.model_revision,
        backend="mock",
    )


def _run_narrative(payload: dict):
    task = _narrative_task()
    evidence = _narrative_evidence()
    adapter = MockModelAdapter((payload,))

    def execute(tool_name, arguments, task_input):
        return execute_deterministic_tool(tool_name, arguments, task_input, evidence)

    return run_agent_case(
        task,
        adapter=adapter,
        model_config=_config(adapter),
        corpus_id="us-sec-existing20-v22",
        benchmark_manifest_sha256=MANIFEST_SHA256,
        runtime_mode="safety_hybrid",
        prompt_condition="zero_shot",
        evidence_items=evidence,
        tool_executor=execute,
        clock=_clock(),
    )


def test_metric_operation_contract_is_complete_non_gold_and_tamper_evident():
    contract = metric_operation_contract("roe")
    assert contract["formula"]["op"] == "divide"
    assert contract["output_unit"] == "pure"
    assert [item["name"] for item in contract["ordered_inputs"]] == [
        "net_income",
        "ending_equity",
        "beginning_equity",
    ]
    assert "target" not in json.dumps(contract).casefold()
    unhashed = dict(contract)
    digest = unhashed.pop("contract_sha256")
    assert digest == canonical_json_sha256(unhashed)


def test_rollup_metric_contract_omits_taxonomy_target_aliases():
    contract = metric_operation_contract("current_assets_rollup")

    assert contract["formula"] == {
        "op": "rollup_sum",
        "components": [
            "cash",
            "short_term_investments",
            "accounts_receivable",
            "inventory",
        ],
    }


def test_v22_task_binds_narrative_subtype_and_metric_contract():
    narrative = _narrative_task()
    assert narrative["agent_task_input_version"] == "v2.2"
    assert narrative["narrative_subtype"] == "accounting_policy"
    assert narrative["metric_operation_contract"] is None

    tampered = copy.deepcopy(narrative)
    tampered["narrative_subtype"] = None
    with pytest.raises(ContractValidationError, match="narrative_subtype"):
        validate_agent_task_input(tampered)


def test_frozen_scope_does_not_claim_answerability():
    task = _narrative_task()
    observation = load_frozen_evidence(
        task, _narrative_evidence(), visibility="MODEL_VISIBLE"
    )
    validate_tool_observation(observation)
    assert observation["retrieval_status"] == "SCOPE_LOADED"
    assert observation["answerability_assessed"] is False
    assert "status" not in observation


def test_narrative_selection_reconstructs_release_fields_deterministically():
    task = _narrative_task()
    evidence = _narrative_evidence()
    quote = "Revenue is recognized when control of the promised goods transfers to the customer."
    payload = {
        "task_id": task["task_id"],
        "action": "ANSWER",
        "period_key": "FY2025",
        "extracts": [{"evidence_id": "chunk-1", "exact_quote": quote}],
    }
    proposal = normalize_stage_response(payload, task, stage="terminal")
    observation = load_frozen_evidence(task, evidence, visibility="MODEL_VISIBLE")
    result = verify_proposal(
        task,
        proposal,
        evidence_items=evidence,
        tool_observation=observation,
        repair_count=0,
    )
    assert proposal["answer_text"] == quote
    assert proposal["claims"] == [
        {
            "claim_id": "1",
            "text": quote,
            "evidence_ids": ["chunk-1"],
            "supporting_text": quote,
        }
    ]
    assert result["disposition"] == "RELEASED"


def test_multi_extract_reconstruction_verifies_each_chunk_and_order():
    task = _narrative_task()
    evidence = _narrative_evidence()
    first_quote = "Revenue is recognized when control of the promised goods transfers to the customer."
    second_quote = "This note also describes contract balances."
    proposal = normalize_stage_response(
        {
            "task_id": task["task_id"],
            "action": "ANSWER",
            "period_key": "FY2025",
            "extracts": [
                {"evidence_id": "chunk-1", "exact_quote": first_quote},
                {"evidence_id": "chunk-2", "exact_quote": second_quote},
            ],
        },
        task,
        stage="terminal",
    )
    observation = load_frozen_evidence(task, evidence, visibility="MODEL_VISIBLE")

    released = verify_proposal(
        task,
        proposal,
        evidence_items=evidence,
        tool_observation=observation,
        repair_count=0,
    )
    assert proposal["answer_text"] == f"{first_quote}\n\n{second_quote}"
    assert released["disposition"] == "RELEASED"

    reordered = copy.deepcopy(proposal)
    reordered["answer_text"] = f"{second_quote}\n\n{first_quote}"
    rejected = verify_proposal(
        task,
        reordered,
        evidence_items=evidence,
        tool_observation=observation,
        repair_count=0,
    )
    assert rejected["disposition"] == "FAILED"
    assert any(
        check["code"] == "SUPPORT_VALID" and not check["passed"]
        for check in rejected["checks"]
    )


def test_narrative_schema_has_no_model_authored_prose_fields():
    task = _narrative_task()
    schema = terminal_response_schema(task, _narrative_evidence())
    answer_schema = schema["oneOf"][0]
    assert answer_schema["required"] == ["task_id", "action", "period_key", "extracts"]
    assert "answer_text" not in answer_schema["properties"]
    assert "claims" not in answer_schema["properties"]
    assert "uniqueItems" not in answer_schema["properties"]["extracts"]
    Draft202012Validator.check_schema(schema)


def test_safety_hybrid_releases_only_runtime_reconstructed_exact_quote():
    quote = "Revenue is recognized when control of the promised goods transfers to the customer."
    result = _run_narrative(
        {
            "task_id": "narrative-accounting_policy",
            "action": "ANSWER",
            "period_key": "FY2025",
            "extracts": [{"evidence_id": "chunk-1", "exact_quote": quote}],
        }
    )
    assert result.run_record["outcome"] == "RELEASED"
    assert result.final_proposal["answer_text"] == quote
    assert result.final_proposal["claims"][0]["supporting_text"] == quote
    assert result.run_record["repair_count"] == 0
    assert result.run_record["narrative_subtype"] == "accounting_policy"
    assert result.run_record["timings"]["generation_duration_ms"] >= 0


def test_paraphrase_support_failure_is_terminal_and_not_repaired():
    result = _run_narrative(
        {
            "task_id": "narrative-accounting_policy",
            "action": "ANSWER",
            "period_key": "FY2025",
            "extracts": [
                {
                    "evidence_id": "chunk-1",
                    "exact_quote": "The issuer records revenue after control passes.",
                }
            ],
        }
    )
    assert result.run_record["outcome"] == "MODEL_FAILURE"
    assert result.run_record["repair_count"] == 0
    assert any(
        check["code"] == "SUPPORT_VALID" and not check["passed"]
        for check in result.run_record["verifier_result"]["checks"]
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "task_id": "narrative-accounting_policy",
            "action": "ANSWER",
            "period_key": "FY2025",
            "answer_text": "The filing does not state the requested policy.",
            "evidence_ids": ["chunk-1"],
            "claims": [],
        },
        {
            "task_id": "narrative-accounting_policy",
            "action": "ANSWER",
            "period_key": "FY2025",
            "extracts": [{"evidence_id": "forged", "exact_quote": "forged"}],
        },
        {
            "task_id": "narrative-accounting_policy",
            "action": "ANSWER",
            "period_key": "FY2025",
            "extracts": [
                {"evidence_id": "chunk-1", "exact_quote": "duplicate"},
                {"evidence_id": "chunk-1", "exact_quote": "duplicate"},
            ],
        },
    ],
)
def test_grammar_claimed_invalid_selection_is_integrity_failure(payload):
    result = _run_narrative(payload)
    assert result.run_record["outcome"] == "INTEGRITY_FAILURE"
    assert result.failure["code"] == "STRUCTURED_OUTPUT_CONTRACT_BROKEN"
    assert result.run_record["repair_count"] == 0


def test_direct_prompt_exposes_metric_contract_not_a_target():
    quant_spec = task_operation_spec("quant_metric")
    task = build_agent_task_input(
        task_id="quant-roe",
        task_type="quant_metric",
        jurisdiction="US",
        reporting_framework="US-GAAP",
        standards_profile={"name": "PCAOB", "version": "2026"},
        source_system="SEC",
        question="Calculate return on equity.",
        entity={"entity_id": "e", "name": "E"},
        filing={"filing_id": "f"},
        period={"period_key": "FY2025"},
        metric_spec_id="roe",
        allowed_tools=[quant_spec.operation],
        evidence_ids=[],
        output_schema_id=quant_spec.terminal_output_schema_id,
        refusal_codes=["MISSING_INPUT"],
    )
    context = build_context_pack(task, [], token_counter=lambda _text: 1)
    prompt = build_direct_prompt(context)
    visible = "\n".join(message["content"] for message in prompt.messages)
    assert "metric_operation_contract" in visible
    assert '"op":"divide"' in visible
    assert "expected_value" not in visible
    assert "target_answer" not in visible


def test_v22_task_schema_accepts_representative_python_valid_task():
    task = _narrative_task(subtype="critical_audit_matter")
    schema_path = Path("auditops/specs/agent_task_input.v2.2.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(task)
