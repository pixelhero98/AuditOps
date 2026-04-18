from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, Mapping, Optional, Sequence

from .metrics import answer_metric_spec
from .tasks import (
    EXECUTOR_OP,
    OUTPUT_FIELDS,
    STRUCTURED_ANSWER_SCHEMA_ID,
    TASK_PLAN_SCHEMA_ID,
    TASK_SPEC_VERSION,
    TASK_TYPE,
    build_task_plan_target,
    build_task_specs,
    dedupe_task_specs,
)


def validate_task_plan(task_plan: Mapping[str, Any]) -> None:
    """Validate that a task plan matches the required schema contract.
    
    Parameters
    ----------
    task_plan : Mapping[str, Any]
        Executor-ready task-plan payload derived from a task specification.
    
    Raises
    ------
    ValueError
        TaskPlan is missing required fields: {...}.
    ValueError
        Unsupported TaskPlan version: {...}.
    ValueError
        Unsupported TaskPlan type: {...}.
    ValueError
        Unsupported executor op: {...}.
    ValueError
        TaskPlan required_output_schema must reference structured_answer.v1.
    """
    required_fields = {
        "task_plan_version",
        "task_id",
        "task_type",
        "metric_spec_id",
        "filing_id",
        "period_key",
        "executor_op",
        "required_output_schema",
        "refusal_policy",
    }
    missing = sorted(required_fields - set(task_plan))
    if missing:
        raise ValueError(f"TaskPlan is missing required fields: {', '.join(missing)}")
    if task_plan["task_plan_version"] != TASK_SPEC_VERSION:
        raise ValueError(f"Unsupported TaskPlan version: {task_plan['task_plan_version']}")
    if task_plan["task_type"] != TASK_TYPE:
        raise ValueError(f"Unsupported TaskPlan type: {task_plan['task_type']}")
    if task_plan["executor_op"] != EXECUTOR_OP:
        raise ValueError(f"Unsupported executor op: {task_plan['executor_op']}")
    if task_plan["required_output_schema"].get("schema_id") != STRUCTURED_ANSWER_SCHEMA_ID:
        raise ValueError("TaskPlan required_output_schema must reference structured_answer.v1")


def validate_structured_answer(answer: Mapping[str, Any]) -> None:
    """Validate that a structured answer matches the required schema.
    
    Parameters
    ----------
    answer : Mapping[str, Any]
        Structured answer payload to validate, route, or evaluate.
    
    Raises
    ------
    ValueError
        Structured answer is missing required fields: {...}.
    ValueError
        Unsupported structured answer version: {...}.
    ValueError
        Unsupported structured answer status: {...}.
    ValueError
        Structured answer evidence_ids must be a list.
    ValueError
        OK structured answers must include a value.
    ValueError
        OK structured answers cannot include refusal_code.
    ValueError
        OK structured answers must include evidence IDs.
    ValueError
        REFUSAL structured answers must include refusal_code.
    
    Examples
    --------
    >>> answer = {...}
    >>> validate_structured_answer(answer)  # doctest: +SKIP
    """
    required_fields = {
        "structured_answer_version",
        "task_id",
        "metric_spec_id",
        "filing_id",
        "status",
        "value",
        "unit",
        "period_key",
        "evidence_ids",
        "refusal_code",
    }
    missing = sorted(required_fields - set(answer))
    if missing:
        raise ValueError(f"Structured answer is missing required fields: {', '.join(missing)}")
    if answer["structured_answer_version"] != TASK_SPEC_VERSION:
        raise ValueError(f"Unsupported structured answer version: {answer['structured_answer_version']}")
    if answer["status"] not in {"OK", "REFUSAL"}:
        raise ValueError(f"Unsupported structured answer status: {answer['status']}")
    if not isinstance(answer["evidence_ids"], list):
        raise ValueError("Structured answer evidence_ids must be a list")
    if answer["status"] == "OK":
        if answer["value"] is None:
            raise ValueError("OK structured answers must include a value")
        if answer["refusal_code"] is not None:
            raise ValueError("OK structured answers cannot include refusal_code")
        if not answer["evidence_ids"]:
            raise ValueError("OK structured answers must include evidence IDs")
    else:
        if answer["refusal_code"] is None:
            raise ValueError("REFUSAL structured answers must include refusal_code")


def build_task_plan(task_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the target task-plan payload for executor routing.
    
    Parameters
    ----------
    task_spec : Mapping[str, Any]
        Task specification payload with routing, evidence, and target-answer metadata.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with output fields for this operation.
    """
    return build_task_plan_target(task_spec)


def _to_structured_answer(task_plan: Mapping[str, Any], metric_answer: Mapping[str, Any]) -> Dict[str, Any]:
    result = metric_answer.get("result") or {}
    structured_answer = {
        "structured_answer_version": TASK_SPEC_VERSION,
        "task_id": task_plan["task_id"],
        "metric_spec_id": task_plan["metric_spec_id"],
        "filing_id": task_plan["filing_id"],
        "status": metric_answer["status"],
        "value": result.get("value"),
        "unit": result.get("unit"),
        "period_key": metric_answer["period"]["period_key"],
        "evidence_ids": list(metric_answer.get("evidence_ids", [])),
        "refusal_code": metric_answer.get("refusal_code"),
    }
    validate_structured_answer(structured_answer)
    return structured_answer


def execute_task_plan(conn: sqlite3.Connection, task_plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Execute a task plan and return a validated structured answer.
    
    Parameters
    ----------
    conn : sqlite3.Connection
        Open SQLite connection for the corpus database.
    task_plan : Mapping[str, Any]
        Executor-ready task-plan payload derived from a task specification.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with output fields for this operation.
    """
    validate_task_plan(task_plan)
    metric_answer = answer_metric_spec(
        conn,
        filing_id=task_plan["filing_id"],
        metric_spec_id=task_plan["metric_spec_id"],
        period_key=task_plan["period_key"],
    )
    return _to_structured_answer(task_plan, metric_answer)


def _match_metric(question_lower: str, task_specs: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    exact_id_matches = []
    metric_matches = []
    for task_spec in task_specs:
        metric_id = task_spec["metric_spec_id"].lower()
        metric_label = task_spec["metric_label"].lower()
        if f"({metric_id})" in question_lower:
            exact_id_matches.append(task_spec)
            continue
        if metric_id in question_lower or metric_label in question_lower:
            metric_matches.append(task_spec)
    if exact_id_matches:
        return exact_id_matches
    if not metric_matches:
        return metric_matches

    def _metric_match_width(task_spec: Mapping[str, Any]) -> int:
        metric_id = task_spec["metric_spec_id"].lower()
        metric_label = task_spec["metric_label"].lower()
        widths = [0]
        if metric_id in question_lower:
            widths.append(len(metric_id))
        if metric_label in question_lower:
            widths.append(len(metric_label))
        return max(widths)

    max_width = max(_metric_match_width(task_spec) for task_spec in metric_matches)
    return [task_spec for task_spec in metric_matches if _metric_match_width(task_spec) == max_width]


def _match_period(question_lower: str, task_specs: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    period_matches = []
    match_widths = []
    for task_spec in task_specs:
        period_key = task_spec["period"]["period_key"].lower()
        if re.search(rf"(?<![a-z0-9_]){re.escape(period_key)}(?![a-z0-9_])", question_lower):
            period_matches.append(task_spec)
            match_widths.append(len(period_key))
            continue
        if period_key in question_lower:
            period_matches.append(task_spec)
            match_widths.append(len(period_key))
    if not period_matches:
        return period_matches
    max_width = max(match_widths)
    return [task_spec for task_spec in period_matches if len(task_spec["period"]["period_key"]) == max_width]


def route_question(question: str, filing_id: str, task_specs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Route a quant question to the best matching task specification.
    
    Parameters
    ----------
    question : str
        Natural-language user question to route and answer.
    filing_id : str
        Canonical filing identifier used by manifests and derived artifacts. e.g., '0000320193-2025-10K'
    task_specs : Sequence[Mapping[str, Any]]
        Collection of task specification payloads.
    
    Returns
    -------
    Mapping[str, Any]
        Dictionary with fields produced while a quant question to the best matching task specification.
    
    Raises
    ------
    ValueError
        No TaskSpecs available for filing {...}.
    ValueError
        Question did not match any MetricSpec-backed task.
    ValueError
        Question matched multiple MetricSpecs.
    ValueError
        Question matched multiple periods for the same MetricSpec.
    ValueError
        Question routing did not resolve to a single TaskSpec.
    """
    filing_candidates = [task_spec for task_spec in dedupe_task_specs(task_specs) if task_spec["filing_id"] == filing_id]
    if not filing_candidates:
        raise ValueError(f"No TaskSpecs available for filing {filing_id}")

    question_lower = question.lower()
    metric_matches = _match_metric(question_lower, filing_candidates)
    if not metric_matches:
        raise ValueError("Question did not match any MetricSpec-backed task")
    if len({task_spec["metric_spec_id"] for task_spec in metric_matches}) > 1:
        period_filtered = _match_period(question_lower, metric_matches)
        if len(period_filtered) == 1:
            return period_filtered[0]
        raise ValueError("Question matched multiple MetricSpecs")

    period_matches = _match_period(question_lower, metric_matches)
    if len(period_matches) == 1:
        return period_matches[0]
    if not period_matches and len(metric_matches) == 1:
        return metric_matches[0]
    if len(period_matches) > 1:
        raise ValueError("Question matched multiple periods for the same MetricSpec")
    raise ValueError("Question routing did not resolve to a single TaskSpec")


def _unsupported_task_answer(task_id: str, filing_id: str) -> Dict[str, Any]:
    return {
        "structured_answer_version": TASK_SPEC_VERSION,
        "task_id": task_id,
        "metric_spec_id": None,
        "filing_id": filing_id,
        "status": "REFUSAL",
        "value": None,
        "unit": None,
        "period_key": None,
        "evidence_ids": [],
        "refusal_code": "TASK_NOT_SUPPORTED",
    }


def answer_quant(conn: sqlite3.Connection, question: str, filing_id: str, task_specs: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """Answer a quantitative question for a filing using task specs.
    
    Parameters
    ----------
    conn : sqlite3.Connection
        SQLite connection for the corpus database.
    question : str
        Natural-language user question to route and answer.
    filing_id : str
        Canonical filing identifier used by manifests and derived artifacts. e.g., '0000320193-2025-10K'
    task_specs : Optional[Sequence[Mapping[str, Any]]], optional
        Collection of task specification payloads.
    
    Returns
    -------
    Dict[str, Any]
        Structured response payload for downstream execution or evaluation.
    """
    active_task_specs = dedupe_task_specs(task_specs or build_task_specs(conn, filing_id=filing_id))
    try:
        task_spec = route_question(question, filing_id, active_task_specs)
    except ValueError:
        return _unsupported_task_answer("unsupported", filing_id)
    task_plan = build_task_plan(task_spec)
    return execute_task_plan(conn, task_plan)
