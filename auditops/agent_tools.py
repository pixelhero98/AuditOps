"""Small deterministic tools available to the bounded audit agent.

These functions consume only the sanitized task and its frozen evidence.  They
never read evaluator gold, call a network service, or execute model-provided
code.
"""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from importlib import resources
from typing import Any

from .agent_contracts import canonical_decimal_string
from .agent_operations import operation_spec
from .canonical_json import canonical_json_sha256


@dataclass(frozen=True)
class _Value:
    value: Decimal
    unit: str | None
    evidence_ids: tuple[str, ...]


class _EvaluationRefusal(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


_UK_INPUT_ALIASES = {
    "assets_current": {"currentassets"},
    "liabilities_current": {
        "creditors",
        "creditorsamountsfallingduewithinoneyear",
        "currentliabilities",
    },
}
_CURRENT_CREDITOR_DIMENSIONS = {
    "financialinstrumentcurrentnoncurrentdimension": {"currentfinancialinstruments"},
    "maturitiesorexpirationperiodsdimension": {"withinoneyear"},
}


@lru_cache(maxsize=1)
def _metric_specs() -> dict[str, dict[str, Any]]:
    path = resources.files("auditops.specs").joinpath("metric_specs.json")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("metric_specs.json must contain an array")
    return {str(spec["id"]): dict(spec) for spec in payload}


def canonical_metric_output_unit(metric_spec_id: str) -> str:
    """Return the registry-bound output unit without evaluating case evidence."""

    spec = _metric_specs().get(metric_spec_id)
    if spec is None:
        raise ValueError(f"Unknown metric_spec_id: {metric_spec_id}")
    unit = spec.get("output_unit")
    if not isinstance(unit, str) or not unit:
        raise ValueError(f"Metric {metric_spec_id!r} lacks a canonical output unit")
    return unit


def metric_operation_contract(
    metric_spec_id: str, *, contract_version: str = "v2.2"
) -> dict[str, Any]:
    """Return inference-visible, non-gold metric semantics for one contract.

    The contract intentionally omits taxonomy aliases and case evidence.  It
    explains the registered operation to a direct-control model without
    exposing the deterministic result or evaluator target.
    """

    if contract_version not in {"v2.2", "v2.3"}:
        raise ValueError(f"Unsupported metric contract version: {contract_version}")
    spec = _metric_specs().get(metric_spec_id)
    if spec is None:
        raise ValueError(f"Unknown metric_spec_id: {metric_spec_id}")
    required_inputs: list[dict[str, str]] = []
    for raw in spec.get("required_inputs", []):
        if not isinstance(raw, Mapping):
            raise TypeError(f"Metric {metric_spec_id!r} has an invalid input spec")
        name = raw.get("name")
        unit_family = raw.get("unit_family")
        period = raw.get("period")
        if not all(
            isinstance(value, str) and value for value in (name, unit_family, period)
        ):
            raise ValueError(f"Metric {metric_spec_id!r} has an incomplete input spec")
        required_inputs.append(
            {"name": str(name), "unit_family": str(unit_family), "period": str(period)}
        )
    formula = copy.deepcopy(spec["formula"])
    if not isinstance(formula, dict):
        raise TypeError(f"Metric {metric_spec_id!r} has an invalid formula")
    # Rollup target aliases are taxonomy-resolution metadata used by the
    # deterministic executor. They are neither operation semantics nor safe
    # inference fields, and their legacy ``target_*`` name is gold-like.
    formula.pop("target_aliases", None)
    contract = {
        "metric_operation_contract_version": contract_version,
        "metric_spec_id": metric_spec_id,
        "metric_spec_version": str(spec.get("version") or "1"),
        "kind": str(spec["kind"]),
        "ordered_inputs": required_inputs,
        "formula": formula,
        "output_unit": canonical_metric_output_unit(metric_spec_id),
        "applicable_period_types": list(spec.get("applicable_period_types") or []),
        "deterministic_refusal_conditions": list(spec.get("refusal_rules") or []),
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def _decimal(value: Any) -> Decimal:
    if value is None or isinstance(value, bool):
        raise _EvaluationRefusal("MISSING_INPUT")
    try:
        materialized = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise _EvaluationRefusal("MISSING_INPUT") from exc
    if not materialized.is_finite():
        raise _EvaluationRefusal("MISSING_INPUT")
    return materialized


def _same_unit(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.strip().casefold() == right.strip().casefold()


def _canonical_unit(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().casefold()
    for prefix in ("iso4217:", "iso4217_"):
        normalized = normalized.removeprefix(prefix)
    return normalized


def _canonical_concept(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    local = re.split(r"[:_]", value.strip())[-1]
    normalized = re.sub(r"[^a-z0-9]", "", local.casefold())
    return normalized or None


def _period_type(period_key: str) -> str | None:
    if period_key.startswith("ASOF_"):
        return "ASOF"
    if period_key.startswith("FY"):
        return "FY"
    if period_key.startswith("YTD_Q"):
        return "YTD"
    if period_key.startswith("Q"):
        return "Q"
    return None


def _canonical_dimensions(value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, str] = {}
    for raw_dimension, raw_member in value.items():
        member_value = (
            raw_member.get("member") if isinstance(raw_member, Mapping) else raw_member
        )
        dimension = _canonical_concept(raw_dimension)
        member = _canonical_concept(member_value)
        if dimension is None or member is None:
            return None
        normalized[dimension] = member
    return normalized


def _allowlisted_current_creditor(
    *, input_name: str, jurisdiction: str, metadata: Mapping[str, Any]
) -> bool:
    if (
        jurisdiction != "UK"
        or input_name != "liabilities_current"
        or metadata.get("scenario") != "CURRENT_CREDITOR_ALLOWLIST"
    ):
        return False
    dimensions = _canonical_dimensions(metadata.get("dimensions"))
    if not dimensions or len(dimensions) != 1:
        return False
    dimension, member = next(iter(dimensions.items()))
    return member in _CURRENT_CREDITOR_DIMENSIONS.get(dimension, set())


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _prior_year_key(period_key: str) -> str | None:
    match = re.fullmatch(r"FY(\d{4})", period_key)
    if match:
        return f"FY{int(match.group(1)) - 1}"
    match = re.fullmatch(r"(Q[1-4]|YTD_Q[1-4])_(\d{4})", period_key)
    if match:
        return f"{match.group(1)}_{int(match.group(2)) - 1}"
    return None


def _quarter_key(period_key: str, *, previous: bool) -> str | None:
    match = re.fullmatch(r"(?:YTD_)?Q([1-4])_(\d{4})", period_key)
    if not match:
        return None
    quarter = int(match.group(1))
    year = int(match.group(2))
    if previous:
        quarter -= 1
        if quarter == 0:
            quarter = 4
            year -= 1
    return f"Q{quarter}_{year}"


def _asof_key(task_input: Mapping[str, Any], *, prior_year: bool) -> str | None:
    period = task_input.get("period")
    end_date = period.get("end_date") if isinstance(period, Mapping) else None
    if not isinstance(end_date, str):
        return None
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", end_date)
    if not match:
        return None
    year = int(match.group(1)) - (1 if prior_year else 0)
    month = int(match.group(2))
    day = int(match.group(3))
    if month == 2 and day == 29 and prior_year:
        day = 28
    return f"ASOF_{year:04d}{month:02d}{day:02d}"


def _expected_period_key(selector: Any, task_input: Mapping[str, Any]) -> str | None:
    period = task_input["period"]
    task_period = str(period["period_key"])
    selector_periods = period.get("selector_periods")
    if selector in {
        "current_period_end_asof",
        "prior_year_period_end_asof",
    } and isinstance(selector_periods, Mapping):
        frozen_period = selector_periods.get(selector)
        if isinstance(frozen_period, str):
            return frozen_period
    if selector == "current":
        return task_period
    if selector == "prior_year_same_period":
        return _prior_year_key(task_period)
    if selector == "current_quarter":
        return _quarter_key(task_period, previous=False)
    if selector == "previous_quarter":
        return _quarter_key(task_period, previous=True)
    if selector == "current_period_end_asof":
        return _asof_key(task_input, prior_year=False)
    if selector == "prior_year_period_end_asof":
        return _asof_key(task_input, prior_year=True)
    return None


def _period_key_matches_selector(
    actual_period_key: Any,
    selector: Any,
    task_input: Mapping[str, Any],
) -> tuple[bool, str | None]:
    expected_period_key = _expected_period_key(selector, task_input)
    if expected_period_key is None or not isinstance(actual_period_key, str):
        return False, expected_period_key
    return actual_period_key == expected_period_key, expected_period_key


def _evaluate_formula(
    formula: Mapping[str, Any], inputs: Mapping[str, _Value]
) -> _Value:
    operation = formula.get("op")
    if operation == "input":
        name = formula.get("name")
        if not isinstance(name, str) or name not in inputs:
            raise _EvaluationRefusal("MISSING_INPUT")
        return inputs[name]
    if operation == "abs":
        value = _evaluate_formula(formula["value"], inputs)
        return _Value(abs(value.value), value.unit, value.evidence_ids)
    if operation in {"add", "subtract"}:
        left = _evaluate_formula(formula["left"], inputs)
        right = _evaluate_formula(formula["right"], inputs)
        if not _same_unit(left.unit, right.unit):
            raise _EvaluationRefusal("INCOMPATIBLE_UNITS")
        value = (
            left.value + right.value if operation == "add" else left.value - right.value
        )
        return _Value(
            value,
            left.unit,
            _ordered_unique(left.evidence_ids + right.evidence_ids),
        )
    if operation in {"sum", "average"}:
        arguments = [_evaluate_formula(item, inputs) for item in formula["args"]]
        if not arguments:
            raise _EvaluationRefusal("PERIOD_NOT_SUPPORTED")
        unit = arguments[0].unit
        if any(not _same_unit(unit, item.unit) for item in arguments[1:]):
            raise _EvaluationRefusal("INCOMPATIBLE_UNITS")
        total = sum((item.value for item in arguments), Decimal(0))
        if operation == "average":
            total /= Decimal(len(arguments))
        return _Value(
            total,
            unit,
            _ordered_unique(
                tuple(
                    evidence_id
                    for item in arguments
                    for evidence_id in item.evidence_ids
                )
            ),
        )
    if operation == "divide":
        left = _evaluate_formula(formula["left"], inputs)
        right = _evaluate_formula(formula["right"], inputs)
        if not _same_unit(left.unit, right.unit):
            raise _EvaluationRefusal("INCOMPATIBLE_UNITS")
        if right.value == 0:
            raise _EvaluationRefusal("ZERO_DENOMINATOR")
        return _Value(
            left.value / right.value,
            str(formula.get("output_unit") or "pure"),
            _ordered_unique(left.evidence_ids + right.evidence_ids),
        )
    if operation == "rollup_sum":
        arguments = [
            inputs[name]
            for name in formula.get("components", [])
            if isinstance(name, str) and name in inputs
        ]
        if len(arguments) != len(formula.get("components", [])):
            raise _EvaluationRefusal("MISSING_INPUT")
        if not arguments:
            raise _EvaluationRefusal("MISSING_INPUT")
        unit = arguments[0].unit
        if any(not _same_unit(unit, item.unit) for item in arguments[1:]):
            raise _EvaluationRefusal("INCOMPATIBLE_UNITS")
        return _Value(
            sum((item.value for item in arguments), Decimal(0)),
            unit,
            _ordered_unique(
                tuple(
                    evidence_id
                    for item in arguments
                    for evidence_id in item.evidence_ids
                )
            ),
        )
    raise _EvaluationRefusal("PERIOD_NOT_SUPPORTED")


def _refusal_code(task_input: Mapping[str, Any], code: str) -> str:
    allowed = set(task_input["refusal_policy"]["allowed_codes"])
    if code == "ZERO_DENOMINATOR" and "DIVISION_BY_ZERO" in allowed:
        return "DIVISION_BY_ZERO"
    if code in allowed:
        return code
    for fallback in ("MISSING_INPUT", "AMBIGUOUS_CONTEXT", "UNSUPPORTED_REQUEST"):
        if fallback in allowed:
            return fallback
    return code


def evaluate_quant_evidence(
    task_input: Mapping[str, Any],
    evidence_items: Sequence[Mapping[str, Any]],
    *,
    visibility: str = "VERIFIER_ONLY",
) -> dict[str, Any]:
    """Recompute a registered metric from complete, typed evidence items."""

    if visibility not in {"MODEL_VISIBLE", "VERIFIER_ONLY"}:
        raise ValueError("visibility must be MODEL_VISIBLE or VERIFIER_ONLY")
    task_id = str(task_input["task_id"])
    filing_id = str(task_input["filing"]["filing_id"])
    allowed_filing_ids = set(
        task_input["evidence_scope"].get("filing_ids", [filing_id])
    )
    period_key = str(task_input["period"]["period_key"])
    metric_spec_id = task_input["task_parameters"]["metric_spec_id"]
    base = {
        "tool_observation_version": str(
            task_input.get("agent_task_input_version", "v2.2")
        ),
        "observation_kind": "METRIC",
        "visibility": visibility,
        "task_id": task_id,
        "filing_id": filing_id,
        "metric_spec_id": metric_spec_id,
        "period_key": period_key,
    }
    spec = _metric_specs().get(str(metric_spec_id))
    if spec is None:
        return {
            **base,
            "status": "REFUSAL",
            "value": None,
            "unit": None,
            "evidence_ids": [],
            "refusal_code": _refusal_code(task_input, "UNSUPPORTED_REQUEST"),
        }
    if _period_type(period_key) not in set(spec.get("applicable_period_types") or []):
        return {
            **base,
            "status": "REFUSAL",
            "value": None,
            "unit": None,
            "evidence_ids": [],
            "refusal_code": _refusal_code(task_input, "PERIOD_NOT_SUPPORTED"),
        }

    required = {
        str(item["name"]): dict(item)
        for item in spec.get("required_inputs", [])
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    malformed = False
    for item in evidence_items:
        if (
            not isinstance(item, Mapping)
            or item.get("filing_id") not in allowed_filing_ids
        ):
            malformed = True
            continue
        metadata = item.get("metadata")
        input_name = (
            metadata.get("input_name") if isinstance(metadata, Mapping) else None
        )
        if isinstance(input_name, str) and input_name in required:
            grouped[input_name].append(item)

    try:
        if malformed:
            raise _EvaluationRefusal("AMBIGUOUS_CONTEXT")
        inputs: dict[str, _Value] = {}
        entity_ids: set[str] = set()
        normal_contexts_by_period: dict[str, set[str]] = defaultdict(set)
        special_contexts_by_period: dict[str, set[str]] = defaultdict(set)
        resolved_periods_by_selector: dict[str, set[str]] = defaultdict(set)
        jurisdiction = str(task_input.get("jurisdiction") or "").upper()
        for name, input_spec in required.items():
            candidates = grouped.get(name, [])
            if not candidates:
                raise _EvaluationRefusal("MISSING_INPUT")
            values = {_decimal(item.get("value")) for item in candidates}
            units = {_canonical_unit(item.get("unit")) for item in candidates}
            if len(values) != 1 or len(units) != 1:
                raise _EvaluationRefusal("AMBIGUOUS_CONTEXT")
            if (
                len(units) != 1
                or next(iter(units)) is None
                or re.fullmatch(r"[a-z]{3}", str(next(iter(units)))) is None
            ):
                raise _EvaluationRefusal("INCOMPATIBLE_UNITS")
            candidate_period_keys = {
                item.get("period_key")
                for item in candidates
                if isinstance(item.get("period_key"), str)
            }
            if len(candidate_period_keys) != 1 or any(
                not isinstance(item.get("period_key"), str) for item in candidates
            ):
                raise _EvaluationRefusal("PERIOD_NOT_SUPPORTED")
            resolved_period_key = next(iter(candidate_period_keys))
            selector = input_spec.get("period")
            period_matches, expected_period_key = _period_key_matches_selector(
                resolved_period_key, selector, task_input
            )
            if not period_matches or expected_period_key is None:
                raise _EvaluationRefusal("PERIOD_NOT_SUPPORTED")
            resolved_periods_by_selector[str(selector)].add(resolved_period_key)
            aliases = {
                _canonical_concept(alias)
                for alias in input_spec.get("aliases", [])
                if _canonical_concept(alias) is not None
            }
            if jurisdiction == "UK" and name in _UK_INPUT_ALIASES:
                aliases = set(_UK_INPUT_ALIASES[name])
            for item in candidates:
                metadata = item.get("metadata")
                if not isinstance(metadata, Mapping):
                    raise _EvaluationRefusal("AMBIGUOUS_CONTEXT")
                concept = _canonical_concept(
                    metadata.get("concept_norm") or metadata.get("concept")
                )
                if concept is None or concept not in aliases:
                    raise _EvaluationRefusal("AMBIGUOUS_CONTEXT")
                context_id = metadata.get("context_id")
                entity_id = metadata.get("entity_id")
                if (
                    not isinstance(context_id, str)
                    or not context_id.strip()
                    or not isinstance(entity_id, str)
                    or not entity_id.strip()
                ):
                    raise _EvaluationRefusal("AMBIGUOUS_CONTEXT")
                entity_ids.add(entity_id.strip())
                dimensions = metadata.get("dimensions")
                canonical_dimensions = _canonical_dimensions(dimensions)
                if canonical_dimensions is None:
                    raise _EvaluationRefusal("AMBIGUOUS_CONTEXT")
                is_special = _allowlisted_current_creditor(
                    input_name=name,
                    jurisdiction=jurisdiction,
                    metadata=metadata,
                )
                if canonical_dimensions and not is_special:
                    raise _EvaluationRefusal("AMBIGUOUS_CONTEXT")
                if not canonical_dimensions and metadata.get("scenario") is not None:
                    raise _EvaluationRefusal("AMBIGUOUS_CONTEXT")
                target_contexts = (
                    special_contexts_by_period
                    if is_special
                    else normal_contexts_by_period
                )
                target_contexts[resolved_period_key].add(context_id.strip())
            inputs[name] = _Value(
                next(iter(values)),
                next(iter(units)),
                _ordered_unique(tuple(str(item["evidence_id"]) for item in candidates)),
            )
        if len(entity_ids) != 1:
            raise _EvaluationRefusal("AMBIGUOUS_CONTEXT")
        if any(
            len(period_keys) != 1
            for period_keys in resolved_periods_by_selector.values()
        ):
            raise _EvaluationRefusal("PERIOD_NOT_SUPPORTED")
        expected_entity_id = task_input["entity"].get(
            "source_entity_id", task_input["entity"]["entity_id"]
        )
        if next(iter(entity_ids)) != str(expected_entity_id).strip():
            raise _EvaluationRefusal("AMBIGUOUS_CONTEXT")
        for expected_period_key in set(normal_contexts_by_period) | set(
            special_contexts_by_period
        ):
            normal_contexts = normal_contexts_by_period[expected_period_key]
            special_contexts = special_contexts_by_period[expected_period_key]
            if len(normal_contexts) > 1 or len(special_contexts) > 1:
                raise _EvaluationRefusal("AMBIGUOUS_CONTEXT")
            if special_contexts and len(normal_contexts) != 1:
                raise _EvaluationRefusal("AMBIGUOUS_CONTEXT")
        evaluated = _evaluate_formula(spec["formula"], inputs)
    except _EvaluationRefusal as refusal:
        return {
            **base,
            "status": "REFUSAL",
            "value": None,
            "unit": None,
            "evidence_ids": [],
            "refusal_code": _refusal_code(task_input, refusal.code),
        }

    return {
        **base,
        "status": "OK",
        "value": canonical_decimal_string(evaluated.value),
        "unit": evaluated.unit,
        "evidence_ids": list(evaluated.evidence_ids),
        "refusal_code": None,
    }


def load_frozen_evidence(
    task_input: Mapping[str, Any],
    evidence_items: Sequence[Mapping[str, Any]],
    *,
    visibility: str = "VERIFIER_ONLY",
) -> dict[str, Any]:
    """Load the already-frozen BM25 scope exactly once for model synthesis."""

    if visibility not in {"MODEL_VISIBLE", "VERIFIER_ONLY"}:
        raise ValueError("visibility must be MODEL_VISIBLE or VERIFIER_ONLY")
    # Keep the standalone tool boundary as strict as the runtime boundary.  In
    # particular, never copy evaluator-only metadata into a model-visible
    # observation when this function is called outside ``build_context_pack``.
    from .agent_context import validate_evidence_items

    validated = validate_evidence_items(task_input, evidence_items)
    ordered = sorted(
        validated,
        key=lambda item: (
            item.get("rank") if isinstance(item.get("rank"), int) else 2**31 - 1,
            str(item.get("evidence_id")),
        ),
    )
    evidence_ids = [str(item["evidence_id"]) for item in ordered]
    return {
        "tool_observation_version": str(
            task_input.get("agent_task_input_version", "v2.2")
        ),
        "observation_kind": "FROZEN_EVIDENCE",
        "visibility": visibility,
        "task_id": str(task_input["task_id"]),
        "filing_id": str(task_input["filing"]["filing_id"]),
        "retrieval_status": "SCOPE_LOADED",
        "answerability_assessed": False,
        "period_key": str(task_input["period"]["period_key"]),
        "evidence_scope_sha256": canonical_json_sha256(task_input["evidence_scope"]),
        "evidence_ids": evidence_ids,
        "evidence_items": copy.deepcopy(ordered),
        "selection_provenance": {
            "selection_method": task_input["evidence_scope"]["selection_method"],
            "empty_reason": task_input["evidence_scope"]["empty_reason"],
            "ordered_evidence_ids": evidence_ids,
        },
    }


def execute_deterministic_tool(
    tool_name: str,
    tool_arguments: Mapping[str, Any],
    task_input: Mapping[str, Any],
    evidence_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Execute one registered, side-effect-free tool after exact routing checks."""

    spec = operation_spec(
        tool_name,
        version=str(task_input.get("agent_task_input_version", "v2.2")),
    )
    if task_input.get("task_type") != spec.task_type:
        raise ValueError("tool operation does not match the frozen task type")
    if not spec.arguments_match(task_input, tool_arguments):
        raise ValueError("tool arguments do not match the frozen task")
    if tool_name == "evaluate_metric_spec":
        return evaluate_quant_evidence(
            task_input, evidence_items, visibility="MODEL_VISIBLE"
        )
    if tool_name == "load_frozen_evidence":
        return load_frozen_evidence(
            task_input, evidence_items, visibility="MODEL_VISIBLE"
        )
    raise ValueError(f"Unregistered tool: {tool_name}")
