from __future__ import annotations

import pytest

from auditops.agent_contracts import ContractValidationError, build_agent_task_input
from auditops.agent_operations import expected_tool_arguments
from auditops.agent_tools import execute_deterministic_tool

PERIOD_KEY = "ASOF_20251231"
FILING_ID = "uk-filing-1"


def _quant_task(
    metric_spec_id: str = "current_ratio",
    *,
    evidence_ids: tuple[str, ...] = ("assets", "creditors"),
) -> dict:
    return build_agent_task_input(
        task_id=f"uk-{metric_spec_id}",
        task_type="quant_metric",
        jurisdiction="UK",
        reporting_framework="FRS_105",
        standards_profile={
            "name": "ISA (UK) public-filing profile",
            "version": "2026-08-21",
        },
        source_system="UK_COMPANIES_HOUSE",
        question=f"Calculate {metric_spec_id.replace('_', ' ')}.",
        entity={"entity_id": "01234567", "name": "Example Limited"},
        filing={"filing_id": FILING_ID, "form_type": "accounts"},
        period={"period_key": PERIOD_KEY, "instant": "2025-12-31"},
        metric_spec_id=metric_spec_id,
        allowed_tools=["evaluate_metric_spec"],
        evidence_ids=evidence_ids,
        output_schema_id="quantitative_answer_or_refusal.v2",
        refusal_codes=[
            "AMBIGUOUS_CONTEXT",
            "DIVISION_BY_ZERO",
            "INCOMPATIBLE_UNITS",
            "MISSING_INPUT",
            "PERIOD_NOT_SUPPORTED",
            "UNSUPPORTED_REQUEST",
        ],
    )


def _metric_arguments(task: dict) -> dict:
    return {
        "filing_id": task["filing"]["filing_id"],
        "metric_spec_id": task["task_parameters"]["metric_spec_id"],
        "period_key": task["period"]["period_key"],
    }


def _fact(
    evidence_id: str,
    input_name: str,
    value: str,
    *,
    unit: str | None = "gbp",
    period_key: str = PERIOD_KEY,
    rank: int = 1,
) -> dict:
    is_creditor = input_name == "liabilities_current"
    concept = (
        "uk-core:CreditorsAmountsFallingDueWithinOneYear"
        if is_creditor
        else "uk-core:CurrentAssets"
    )
    return {
        "evidence_id": evidence_id,
        "filing_id": FILING_ID,
        "content": (
            f"input_name={input_name}; concept={concept}; period_key={period_key}; "
            f"unit={unit if unit is not None else 'null'}; value={value}"
        ),
        "rank": rank,
        "period_key": period_key,
        "unit": unit,
        # Companies House preprocessing converts the ix:sign='-' presentation
        # convention for creditors to the positive liability magnitude consumed
        # by metric formulae.
        "value": value,
        "source_system": "UK_COMPANIES_HOUSE",
        "metadata": {
            "input_name": input_name,
            "concept": concept,
            "context_id": "ctx-2025",
            "entity_id": "01234567",
            "dimensions": {},
        },
    }


def _gbp_evidence(*, creditor_value: str = "500") -> list[dict]:
    # Deliberately reverse source order: formula execution, not caller order,
    # determines the stable evidence order in the observation.
    return [
        _fact("creditors", "liabilities_current", creditor_value, rank=2),
        _fact("assets", "assets_current", "1200", rank=1),
    ]


@pytest.mark.parametrize(
    ("metric_spec_id", "expected_value", "expected_unit"),
    [
        ("current_ratio", "2.4", "pure"),
        ("working_capital", "700", "gbp"),
    ],
)
def test_executes_registered_metrics_with_normalized_gbp_creditor_sign(
    metric_spec_id: str,
    expected_value: str,
    expected_unit: str,
):
    task = _quant_task(metric_spec_id)
    evidence = _gbp_evidence()
    evidence[0]["unit"] = "iso4217:GBP"
    evidence[1]["unit"] = "GBP"

    observation = execute_deterministic_tool(
        "evaluate_metric_spec",
        _metric_arguments(task),
        task,
        evidence,
    )

    assert observation == {
        "tool_observation_version": "v2.2",
        "observation_kind": "METRIC",
        "visibility": "MODEL_VISIBLE",
        "task_id": f"uk-{metric_spec_id}",
        "filing_id": FILING_ID,
        "metric_spec_id": metric_spec_id,
        "period_key": PERIOD_KEY,
        "status": "OK",
        "value": expected_value,
        "unit": expected_unit,
        "evidence_ids": ["assets", "creditors"],
        "refusal_code": None,
    }


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ({"period_key": "ASOF_20241231"}, "PERIOD_NOT_SUPPORTED"),
        ({"unit": "usd"}, "INCOMPATIBLE_UNITS"),
    ],
)
def test_refuses_wrong_period_or_incompatible_unit(mutation: dict, expected_code: str):
    task = _quant_task()
    evidence = _gbp_evidence()
    evidence[0].update(mutation)

    observation = execute_deterministic_tool(
        "evaluate_metric_spec", _metric_arguments(task), task, evidence
    )

    assert observation["status"] == "REFUSAL"
    assert observation["refusal_code"] == expected_code
    assert observation["value"] is None
    assert observation["unit"] is None
    assert observation["evidence_ids"] == []


def test_refuses_conflicting_duplicate_input():
    task = _quant_task(evidence_ids=("assets", "assets-conflict", "creditors"))
    evidence = _gbp_evidence()
    evidence.append(_fact("assets-conflict", "assets_current", "1199", rank=3))

    observation = execute_deterministic_tool(
        "evaluate_metric_spec", _metric_arguments(task), task, evidence
    )

    assert observation["status"] == "REFUSAL"
    assert observation["refusal_code"] == "AMBIGUOUS_CONTEXT"
    assert observation["evidence_ids"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        {"concept": "uk-core:Turnover"},
        {"context_id": "parent-only-context"},
        {"entity_id": "99999999"},
    ],
)
def test_refuses_wrong_taxonomy_context_or_entity_binding(mutation: dict):
    task = _quant_task()
    evidence = _gbp_evidence()
    evidence[0]["metadata"].update(mutation)

    observation = execute_deterministic_tool(
        "evaluate_metric_spec", _metric_arguments(task), task, evidence
    )

    assert observation["status"] == "REFUSAL"
    assert observation["refusal_code"] == "AMBIGUOUS_CONTEXT"


def test_zero_denominator_uses_allowed_division_by_zero_alias():
    task = _quant_task()

    observation = execute_deterministic_tool(
        "evaluate_metric_spec",
        _metric_arguments(task),
        task,
        _gbp_evidence(creditor_value="0"),
    )

    assert observation["status"] == "REFUSAL"
    assert observation["refusal_code"] == "DIVISION_BY_ZERO"
    assert "ZERO_DENOMINATOR" not in task["refusal_policy"]["allowed_codes"]


@pytest.mark.parametrize("non_finite", ["NaN", "Infinity", "-Infinity"])
def test_refuses_non_finite_numeric_evidence(non_finite: str):
    task = _quant_task()
    evidence = _gbp_evidence()
    evidence[1]["value"] = non_finite

    observation = execute_deterministic_tool(
        "evaluate_metric_spec", _metric_arguments(task), task, evidence
    )

    assert observation["status"] == "REFUSAL"
    assert observation["refusal_code"] == "MISSING_INPUT"


@pytest.mark.parametrize(
    "mutation",
    [
        {"filing_id": "other-filing"},
        {"metric_spec_id": "working_capital"},
        {"period_key": "ASOF_20241231"},
        {"unexpected": "argument"},
    ],
)
def test_rejects_mutated_metric_tool_arguments(mutation: dict):
    task = _quant_task()
    arguments = {**_metric_arguments(task), **mutation}

    with pytest.raises(ValueError, match="tool arguments do not match the frozen task"):
        execute_deterministic_tool(
            "evaluate_metric_spec", arguments, task, _gbp_evidence()
        )


def test_rejects_unknown_tool_name():
    task = _quant_task()

    with pytest.raises(ValueError, match="Unregistered text-agent operation"):
        execute_deterministic_tool(
            "execute_python", _metric_arguments(task), task, _gbp_evidence()
        )


def _us_roe_task(
    *, end_date: str = "2025-02-01", selector_periods: dict | None = None
) -> dict:
    period = {"period_key": "FY2025", "end_date": end_date}
    if selector_periods is not None:
        period["selector_periods"] = selector_periods
    return build_agent_task_input(
        task_id="us-roe",
        task_type="quant_metric",
        jurisdiction="US",
        reporting_framework="US-GAAP",
        standards_profile={
            "name": "PCAOB public-filing profile",
            "version": "2026-08-21",
        },
        source_system="SEC-EDGAR-CACHED",
        question="Calculate return on equity.",
        entity={
            "entity_id": "issuer-1",
            "name": "Example Corporation",
            "source_entity_id": "0000000001",
        },
        filing={"filing_id": "us-filing-1", "form_type": "10-K"},
        period=period,
        metric_spec_id="roe",
        allowed_tools=["evaluate_metric_spec"],
        evidence_ids=("income", "ending", "beginning"),
        output_schema_id="quantitative_answer_or_refusal.v2",
        refusal_codes=[
            "AMBIGUOUS_CONTEXT",
            "INCOMPATIBLE_UNITS",
            "MISSING_INPUT",
            "PERIOD_NOT_SUPPORTED",
        ],
    )


def _us_roe_fact(
    evidence_id: str,
    input_name: str,
    concept: str,
    period_key: str,
    value: str,
) -> dict:
    return {
        "evidence_id": evidence_id,
        "filing_id": "us-filing-1",
        "content": (
            f"input_name={input_name}; concept={concept}; period_key={period_key}; "
            f"unit=USD; value={value}"
        ),
        "period_key": period_key,
        "unit": "USD",
        "value": value,
        "source_system": "SEC-EDGAR-CACHED",
        "metadata": {
            "input_name": input_name,
            "concept": concept,
            "context_id": f"ctx-{period_key}",
            "entity_id": "0000000001",
            "dimensions": {},
        },
    }


def _us_roe_evidence(*, prior_period_key: str) -> list[dict]:
    return [
        _us_roe_fact("income", "net_income", "us-gaap_NetIncomeLoss", "FY2025", "20"),
        _us_roe_fact(
            "ending",
            "ending_equity",
            "us-gaap_StockholdersEquity",
            "ASOF_20250201",
            "100",
        ),
        _us_roe_fact(
            "beginning",
            "beginning_equity",
            "us-gaap_StockholdersEquity",
            prior_period_key,
            "100",
        ),
    ]


@pytest.mark.parametrize("prior_period_key", ["ASOF_20240125", "ASOF_20240208"])
def test_average_balance_metric_uses_exact_frozen_asof_binding(prior_period_key: str):
    task = _us_roe_task(
        selector_periods={
            "current_period_end_asof": "ASOF_20250201",
            "prior_year_period_end_asof": prior_period_key,
        }
    )

    observation = execute_deterministic_tool(
        "evaluate_metric_spec",
        _metric_arguments(task),
        task,
        _us_roe_evidence(prior_period_key=prior_period_key),
    )

    assert observation["status"] == "OK"
    assert observation["value"] == "0.2"


@pytest.mark.parametrize(
    "prior_period_key",
    ["ASOF_20240125", "ASOF_20240208", "ASOF_20240124", "ASOF_20240209"],
)
def test_average_balance_metric_refuses_unbound_asof_delta(
    prior_period_key: str,
):
    task = _us_roe_task()

    observation = execute_deterministic_tool(
        "evaluate_metric_spec",
        _metric_arguments(task),
        task,
        _us_roe_evidence(prior_period_key=prior_period_key),
    )

    assert observation["status"] == "REFUSAL"
    assert observation["refusal_code"] == "PERIOD_NOT_SUPPORTED"


def test_average_balance_metric_refuses_multiple_periods_for_one_input():
    task = _us_roe_task(
        selector_periods={
            "current_period_end_asof": "ASOF_20250201",
            "prior_year_period_end_asof": "ASOF_20240125",
        }
    )
    evidence = _us_roe_evidence(prior_period_key="ASOF_20240125")
    evidence.append(
        _us_roe_fact(
            "beginning-duplicate",
            "beginning_equity",
            "us-gaap_StockholdersEquity",
            "ASOF_20240201",
            "100",
        )
    )
    task["evidence_scope"]["evidence_ids"].append("beginning-duplicate")

    observation = execute_deterministic_tool(
        "evaluate_metric_spec", _metric_arguments(task), task, evidence
    )

    assert observation["status"] == "REFUSAL"
    assert observation["refusal_code"] == "PERIOD_NOT_SUPPORTED"


def test_average_balance_metric_handles_prior_year_from_leap_day():
    task = _us_roe_task(end_date="2024-02-29")
    task["period"]["period_key"] = "FY2024"
    evidence = _us_roe_evidence(prior_period_key="ASOF_20230228")
    evidence[0]["period_key"] = "FY2024"
    evidence[0]["content"] = evidence[0]["content"].replace("FY2025", "FY2024")
    evidence[1]["period_key"] = "ASOF_20240229"
    evidence[1]["content"] = evidence[1]["content"].replace(
        "ASOF_20250201", "ASOF_20240229"
    )

    observation = execute_deterministic_tool(
        "evaluate_metric_spec", _metric_arguments(task), task, evidence
    )

    assert observation["status"] == "OK"


def _narrative_task() -> dict:
    return build_agent_task_input(
        task_id="uk-audit-language",
        task_type="narrative_citation",
        jurisdiction="UK",
        reporting_framework="FRS_102",
        standards_profile={
            "name": "ISA (UK) public-filing profile",
            "version": "2026-08-21",
        },
        source_system="UK_COMPANIES_HOUSE",
        question="What opinion language appears in the auditor's report?",
        entity={"entity_id": "01234567", "name": "Example Limited"},
        filing={"filing_id": FILING_ID, "form_type": "accounts"},
        period={"period_key": "FY2025", "end_date": "2025-12-31"},
        retrieval_query="auditor report opinion language",
        allowed_tools=["load_frozen_evidence"],
        evidence_ids=["chunk-a", "chunk-b", "chunk-c"],
        output_schema_id="narrative_answer_or_refusal.v2",
        refusal_codes=["NARRATIVE_NOT_SUPPORTED", "UNSUPPORTED_REQUEST"],
    )


def test_narrative_tool_loads_the_exact_frozen_scope_with_provenance():
    task = _narrative_task()
    evidence = [
        {
            "evidence_id": "chunk-b",
            "filing_id": FILING_ID,
            "content": "Unrelated accounting policy.",
            "rank": 2,
            "metadata": {"heading": "Accounting policies"},
        },
        {
            "evidence_id": "chunk-a",
            "filing_id": FILING_ID,
            "content": "In our opinion, the financial statements give a true and fair view.",
            "rank": 1,
            "metadata": {"heading": "Independent auditor's report"},
        },
        {
            "evidence_id": "chunk-c",
            "filing_id": FILING_ID,
            "content": "Basis of preparation.",
            "metadata": {"heading": "Basis of preparation"},
        },
    ]
    arguments = expected_tool_arguments(task)

    observation = execute_deterministic_tool(
        "load_frozen_evidence", arguments, task, evidence
    )

    assert observation["tool_observation_version"] == "v2.2"
    assert observation["observation_kind"] == "FROZEN_EVIDENCE"
    assert observation["visibility"] == "MODEL_VISIBLE"
    assert observation["retrieval_status"] == "SCOPE_LOADED"
    assert observation["answerability_assessed"] is False
    assert "status" not in observation
    assert observation["evidence_ids"] == ["chunk-a", "chunk-b", "chunk-c"]
    assert [item["evidence_id"] for item in observation["evidence_items"]] == [
        "chunk-a",
        "chunk-b",
        "chunk-c",
    ]
    assert observation["selection_provenance"] == {
        "selection_method": "FROZEN_BM25",
        "empty_reason": None,
        "ordered_evidence_ids": ["chunk-a", "chunk-b", "chunk-c"],
    }
    assert observation["evidence_scope_sha256"] == arguments["evidence_scope_sha256"]


def test_narrative_tool_rejects_evaluator_only_evidence_metadata():
    task = _narrative_task()
    evidence = [
        {
            "evidence_id": evidence_id,
            "filing_id": FILING_ID,
            "content": "Public filing text.",
            "rank": rank,
            "metadata": {"answerability": True} if rank == 1 else {},
        }
        for rank, evidence_id in enumerate(("chunk-a", "chunk-b", "chunk-c"), 1)
    ]

    with pytest.raises(ContractValidationError, match="GOLD_FIELD_FORBIDDEN"):
        execute_deterministic_tool(
            "load_frozen_evidence",
            expected_tool_arguments(task),
            task,
            evidence,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        {"evidence_scope_sha256": "0" * 64},
        {"filing_id": "other-filing"},
        {"answerability": True},
    ],
)
def test_rejects_mutated_retrieval_tool_arguments(mutation: dict):
    task = _narrative_task()
    arguments = {**expected_tool_arguments(task), **mutation}

    with pytest.raises(ValueError, match="tool arguments do not match the frozen task"):
        execute_deterministic_tool("load_frozen_evidence", arguments, task, [])
