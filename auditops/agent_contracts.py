from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from .agent_failures import failure_spec, validate_failure_binding
from .agent_operations import (
    NARRATIVE_SUBTYPES,
    REGISTERED_OPERATIONS,
    TEXT_RUNTIME_ID,
    TEXT_RUNTIME_ID_V23,
    TEXT_TASK_TYPES,
    narrative_selection_contract,
    task_operation_spec,
    validate_task_operation_binding,
)
from .canonical_json import CANONICAL_JSON_VERSION, canonical_json_sha256

AGENT_TASK_INPUT_VERSION = "v2.2"
AGENT_TASK_INPUT_VERSION_V23 = "v2.3"
AGENT_PROPOSAL_VERSION = "v2"
VERIFIER_RESULT_VERSION = "v2"
AGENT_RUN_RECORD_VERSION = "v2.2"
AGENT_CASE_RESULT_VERSION = "v2.2"
FAILURE_VERSION = "v2.2"
AGENT_RUN_RECORD_VERSION_V23 = "v2.3"
AGENT_CASE_RESULT_VERSION_V23 = "v2.3"
FAILURE_VERSION_V23 = "v2.3"
MODEL_CONFIG_VERSION = "v2"

MAX_INPUT_TOKENS = 14_336
MAX_PLAN_TOKENS = 192
MAX_ANSWER_TOKENS = 768
MAX_REPAIR_TOKENS = 768
DEFAULT_SEED = 20_260_821
MAX_REPAIR_ATTEMPTS = 1

TASK_TYPES = TEXT_TASK_TYPES
JURISDICTIONS = frozenset({"US", "UK"})
REGISTERED_TOOLS = REGISTERED_OPERATIONS
PROPOSAL_ACTIONS = frozenset({"CALL_TOOL", "ANSWER", "REFUSE"})
UNCERTAINTY_LEVELS = frozenset({"NONE", "LOW", "HIGH"})
VERIFIER_DISPOSITIONS = frozenset(
    {"TOOL_CALL_APPROVED", "RELEASED", "SAFE_REFUSAL", "REPAIR_REQUIRED", "FAILED"}
)
TERMINAL_OUTCOMES = frozenset(
    {
        "RELEASED",
        "SAFE_REFUSAL",
        "MODEL_FAILURE",
        "INFRASTRUCTURE_FAILURE",
        "INTEGRITY_FAILURE",
    }
)
TOOL_OBSERVATION_KINDS = frozenset({"METRIC", "FROZEN_EVIDENCE"})
TOOL_OBSERVATION_VISIBILITIES = frozenset({"MODEL_VISIBLE", "VERIFIER_ONLY"})
VERIFIER_CHECK_CODES = frozenset(
    {
        "ANSWER_SHAPE_VALID",
        "ANSWER_TEXT_MATCH",
        "ARITHMETIC_VALID",
        "CITATIONS_MATCH",
        "EVIDENCE_VALID",
        "FILING_MATCH",
        "MODEL_GENERATION_FAILED",
        "MODEL_OUTPUT_NOT_STRICT_JSON",
        "OBSERVATION_EVIDENCE_VALID",
        "OBSERVATION_PERIOD_MATCH",
        "OBSERVATION_SCOPE_VALID",
        "OPERATION_MATCH",
        "PERIOD_MATCH",
        "PROMPT_INJECTION_CLEAR",
        "QUERY_MATCH",
        "REFUSAL_VALID",
        "RETRIEVAL_LIMITS_MATCH",
        "SCHEMA_VALID",
        "STATUS_MATCH",
        "SUPPORT_VALID",
        "TASK_MATCH",
        "TOOL_ALLOWED",
        "TOOL_ARGUMENTS_VALID",
        "TOOL_OBSERVATION_PRESENT",
        "UNEXPECTED_STAGE_ACTION",
        "UNIT_MATCH",
        "VALUE_MATCH",
    }
)

FORBIDDEN_TASK_KEYS = frozenset(
    {
        "answerability",
        "canonical_answer",
        "correct_answer",
        "expected_answer",
        "expected_chunk_ids",
        "expected_evidence_ids",
        "expected_refusal_code",
        "expected_status",
        "expected_value",
        "extractive_answer",
        "gold",
        "gold_answer",
        "gold_label",
        "gold_value",
        "ground_truth",
        "is_answerable",
        "label_answer",
        "reference_answer",
        "solution",
        "source_answer_id",
        "target",
        "target_answer",
        "target_status",
        "target_value",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_DECIMAL_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$")
_ASOF_PERIOD_KEY_RE = re.compile(r"^ASOF_(\d{4})(\d{2})(\d{2})$")
_ASOF_SELECTORS = frozenset({"current_period_end_asof", "prior_year_period_end_asof"})


class ContractValidationError(ValueError):
    """A clear, field-addressed contract validation failure."""

    def __init__(self, code: str, field: str, message: str):
        self.code = code
        self.field = field
        self.message = message
        super().__init__(f"{code} at {field}: {message}")


def _fail(code: str, field: str, message: str) -> None:
    raise ContractValidationError(code, field, message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("INVALID_TYPE", field, "must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    field: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    if missing:
        _fail("MISSING_FIELDS", field, f"missing required fields: {', '.join(missing)}")
    unknown = sorted(set(value) - allowed)
    if unknown:
        _fail("UNKNOWN_FIELDS", field, f"unknown fields: {', '.join(unknown)}")


def _string(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        _fail("INVALID_STRING", field, "must be a non-empty string")
    return value


def _enum(value: Any, choices: frozenset[str], field: str) -> str:
    if not isinstance(value, str) or value not in choices:
        _fail("INVALID_ENUM", field, f"must be one of: {', '.join(sorted(choices))}")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        _fail("INVALID_TYPE", field, "must be a boolean")
    return value


def _integer(
    value: Any, field: str, *, minimum: int = 0, nullable: bool = False
) -> int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(
            "INVALID_INTEGER",
            field,
            f"must be an integer greater than or equal to {minimum}",
        )
    return value


def _number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("INVALID_NUMBER", field, "must be a finite number")
    materialized = float(value)
    if not math.isfinite(materialized) or materialized < minimum:
        _fail(
            "INVALID_NUMBER",
            field,
            f"must be finite and greater than or equal to {minimum}",
        )
    return materialized


def _answer_value(value: Any, field: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or _CANONICAL_DECIMAL_RE.fullmatch(value) is None:
        _fail(
            "INVALID_CANONICAL_DECIMAL",
            field,
            "must be a canonical fixed-point decimal string or null",
        )
    if value == "-0":
        _fail("INVALID_CANONICAL_DECIMAL", field, "zero must be written as 0")


def canonical_decimal_string(value: Any) -> str:
    """Normalize a finite decimal value to the v2 fixed-point lexical form."""

    if isinstance(value, bool) or value is None:
        raise ValueError("canonical decimal input must be a finite number or string")
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("canonical decimal input is invalid") from exc
    if not decimal_value.is_finite():
        raise ValueError("canonical decimal input must be finite")
    if decimal_value == 0:
        return "0"
    materialized = format(decimal_value, "f")
    if "." in materialized:
        materialized = materialized.rstrip("0").rstrip(".")
    if _CANONICAL_DECIMAL_RE.fullmatch(materialized) is None:
        raise ValueError("canonical decimal normalization failed")
    return materialized


def validate_tool_observation(observation: Mapping[str, Any]) -> None:
    """Validate the strict, task-neutral ToolObservationV2.2 envelope.

    Task, filing, operation, and evidence bindings are checked separately by the
    verifier; this validator owns the public lexical and discriminated-union
    shape so malformed or augmented observations cannot enter a run record.
    """

    value = _mapping(observation, "tool_observation")
    kind = value.get("observation_kind")
    common = (
        "tool_observation_version",
        "observation_kind",
        "visibility",
        "task_id",
        "filing_id",
        "period_key",
        "evidence_ids",
    )
    if kind == "METRIC":
        _exact_keys(
            value,
            required=common
            + ("status", "metric_spec_id", "value", "unit", "refusal_code"),
            field="tool_observation",
        )
    elif kind == "FROZEN_EVIDENCE":
        _exact_keys(
            value,
            required=common
            + (
                "retrieval_status",
                "answerability_assessed",
                "evidence_items",
                "selection_provenance",
                "evidence_scope_sha256",
            ),
            field="tool_observation",
        )
    else:
        _enum(kind, TOOL_OBSERVATION_KINDS, "tool_observation.observation_kind")
        raise AssertionError("unreachable")

    observation_version = value["tool_observation_version"]
    if observation_version not in {"v2.2", "v2.3"}:
        _fail(
            "UNSUPPORTED_VERSION",
            "tool_observation.tool_observation_version",
            "must equal v2.2 or v2.3",
        )
    _enum(
        value["visibility"],
        TOOL_OBSERVATION_VISIBILITIES,
        "tool_observation.visibility",
    )
    for key in ("task_id", "filing_id", "period_key"):
        _string(value[key], f"tool_observation.{key}")
    evidence_ids = _string_list(
        value["evidence_ids"], "tool_observation.evidence_ids", max_items=5
    )

    if kind == "METRIC":
        _string(value["metric_spec_id"], "tool_observation.metric_spec_id")
        status = _enum(
            value["status"], frozenset({"OK", "REFUSAL"}), "tool_observation.status"
        )
        if status == "OK":
            _answer_value(value["value"], "tool_observation.value")
            if value["value"] is None:
                _fail(
                    "MISSING_VALUE",
                    "tool_observation.value",
                    "successful metric observations require a value",
                )
            _string(value["unit"], "tool_observation.unit")
            if not evidence_ids:
                _fail(
                    "EMPTY_ARRAY",
                    "tool_observation.evidence_ids",
                    "successful metric observations require evidence",
                )
            if value["refusal_code"] is not None:
                _fail(
                    "INCONSISTENT_OBSERVATION",
                    "tool_observation.refusal_code",
                    "must be null for a successful metric observation",
                )
        else:
            if value["value"] is not None or value["unit"] is not None:
                _fail(
                    "INCONSISTENT_OBSERVATION",
                    "tool_observation",
                    "refusal observations cannot carry a value or unit",
                )
            if evidence_ids:
                _fail(
                    "INCONSISTENT_OBSERVATION",
                    "tool_observation.evidence_ids",
                    "refusal observations cannot carry evidence IDs",
                )
            permitted = task_operation_spec("quant_metric").permitted_refusal_codes - {
                "PROMPT_INJECTION_DETECTED"
            }
            _enum(
                value["refusal_code"],
                frozenset(permitted),
                "tool_observation.refusal_code",
            )
        return

    if value["retrieval_status"] != "SCOPE_LOADED":
        _fail(
            "INVALID_ENUM",
            "tool_observation.retrieval_status",
            "must equal SCOPE_LOADED",
        )
    if value["answerability_assessed"] is not False:
        _fail(
            "INVALID_VALUE",
            "tool_observation.answerability_assessed",
            "must be false because scope loading never assesses answerability",
        )
    _sha256(value["evidence_scope_sha256"], "tool_observation.evidence_scope_sha256")
    items = value["evidence_items"]
    if not isinstance(items, (list, tuple)) or len(items) > 5:
        _fail(
            "INVALID_TYPE",
            "tool_observation.evidence_items",
            "must be an array containing at most five evidence items",
        )
    item_ids: list[str] = []
    allowed_item_keys = {
        "evidence_id",
        "filing_id",
        "content",
        "rank",
        "period_key",
        "unit",
        "value",
        "source_system",
        "metadata",
    }
    for index, item in enumerate(items):
        materialized = _mapping(item, f"tool_observation.evidence_items[{index}]")
        unknown = sorted(set(materialized) - allowed_item_keys)
        missing = sorted({"evidence_id", "filing_id", "content"} - set(materialized))
        if unknown or missing:
            _fail(
                "INVALID_EVIDENCE",
                f"tool_observation.evidence_items[{index}]",
                "must contain the strict evidence-item fields",
            )
        item_ids.append(
            _string(
                materialized["evidence_id"],
                f"tool_observation.evidence_items[{index}].evidence_id",
            )
            or ""
        )
        _string(
            materialized["filing_id"],
            f"tool_observation.evidence_items[{index}].filing_id",
        )
        _string(
            materialized["content"],
            f"tool_observation.evidence_items[{index}].content",
        )
    if item_ids != evidence_ids:
        _fail(
            "INCONSISTENT_OBSERVATION",
            "tool_observation.evidence_items",
            "ordered evidence items must exactly match evidence_ids",
        )
    provenance = _mapping(
        value["selection_provenance"], "tool_observation.selection_provenance"
    )
    _exact_keys(
        provenance,
        required=("selection_method", "empty_reason", "ordered_evidence_ids"),
        field="tool_observation.selection_provenance",
    )
    _enum(
        provenance["selection_method"],
        frozenset(
            {
                "FROZEN_BM25_V23_SECTION_SPLIT"
                if observation_version == "v2.3"
                else "FROZEN_BM25"
            }
        ),
        "tool_observation.selection_provenance.selection_method",
    )
    ordered = _string_list(
        provenance["ordered_evidence_ids"],
        "tool_observation.selection_provenance.ordered_evidence_ids",
        max_items=5,
    )
    if ordered != evidence_ids:
        _fail(
            "INCONSISTENT_OBSERVATION",
            "tool_observation.selection_provenance.ordered_evidence_ids",
            "must exactly match evidence_ids",
        )
    empty_reason = provenance["empty_reason"]
    if evidence_ids and empty_reason is not None:
        _fail(
            "INCONSISTENT_OBSERVATION",
            "tool_observation.selection_provenance.empty_reason",
            "must be null for a non-empty frozen scope",
        )
    if not evidence_ids and empty_reason != "NO_MATCHING_EVIDENCE":
        _fail(
            "INCONSISTENT_OBSERVATION",
            "tool_observation.selection_provenance.empty_reason",
            "must explain an empty frozen scope",
        )


def _string_list(
    value: Any,
    field: str,
    *,
    allow_empty: bool = True,
    max_items: int | None = None,
) -> list[str]:
    if not isinstance(value, (list, tuple)):
        _fail("INVALID_TYPE", field, "must be an array")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_string(item, f"{field}[{index}]") or "")
    if not allow_empty and not result:
        _fail("EMPTY_ARRAY", field, "must not be empty")
    if len(result) != len(set(result)):
        _fail("DUPLICATE_ITEMS", field, "must not contain duplicate values")
    if max_items is not None and len(result) > max_items:
        _fail("TOO_MANY_ITEMS", field, f"must contain at most {max_items} items")
    return result


def _optional_string_field(value: Mapping[str, Any], key: str, field: str) -> None:
    if key in value:
        _string(value[key], f"{field}.{key}", nullable=True)


def _sha256(value: Any, field: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    text = _string(value, field)
    assert text is not None
    if not _SHA256_RE.fullmatch(text):
        _fail(
            "INVALID_SHA256", field, "must be a 64-character hexadecimal SHA-256 digest"
        )


def _scan_forbidden_keys(value: Any, field: str = "task") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                _fail("INVALID_KEY", field, "object keys must be strings")
            child_field = f"{field}.{key}"
            if key.lower() in FORBIDDEN_TASK_KEYS:
                _fail(
                    "GOLD_FIELD_FORBIDDEN",
                    child_field,
                    "evaluation-only data is forbidden from inference input",
                )
            _scan_forbidden_keys(child, child_field)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_forbidden_keys(child, f"{field}[{index}]")


def validate_agent_task_input(task_input: Mapping[str, Any]) -> None:
    task = _mapping(task_input, "task")
    _scan_forbidden_keys(task)
    contract_version = task.get("agent_task_input_version")
    if contract_version not in {AGENT_TASK_INPUT_VERSION, AGENT_TASK_INPUT_VERSION_V23}:
        _fail(
            "UNSUPPORTED_VERSION",
            "task.agent_task_input_version",
            "must equal v2.2 or v2.3",
        )
    optional_v23_fields = (
        ("narrative_selection_policy",) if contract_version == "v2.3" else ()
    )
    _exact_keys(
        task,
        required=(
            "agent_task_input_version",
            "task_id",
            "task_type",
            "jurisdiction",
            "reporting_framework",
            "standards_profile",
            "source_system",
            "question",
            "entity",
            "filing",
            "period",
            "task_parameters",
            "allowed_tools",
            "evidence_scope",
            "output_schema_id",
            "refusal_policy",
            "narrative_subtype",
            "metric_operation_contract",
            *optional_v23_fields,
        ),
        field="task",
    )
    _string(task["task_id"], "task.task_id")
    _enum(task["task_type"], TASK_TYPES, "task.task_type")
    _enum(task["jurisdiction"], JURISDICTIONS, "task.jurisdiction")
    _string(task["reporting_framework"], "task.reporting_framework")
    _string(task["source_system"], "task.source_system")
    _string(task["question"], "task.question")
    _string(task["output_schema_id"], "task.output_schema_id")

    narrative_subtype = task["narrative_subtype"]
    metric_contract = task["metric_operation_contract"]
    if (
        task["task_type"] == "narrative_citation"
        and isinstance(task.get("period"), Mapping)
        and "selector_periods" in task["period"]
    ):
        _fail(
            "INVALID_TASK_PERIOD",
            "task.period.selector_periods",
            "is only allowed for quantitative metric tasks",
        )
    if task["task_type"] == "narrative_citation":
        _enum(narrative_subtype, NARRATIVE_SUBTYPES, "task.narrative_subtype")
        if metric_contract is not None:
            _fail(
                "INVALID_TASK_PARAMETERS",
                "task.metric_operation_contract",
                "narrative tasks must not carry a metric operation contract",
            )
        if contract_version == "v2.3":
            selection_policy = _mapping(
                task["narrative_selection_policy"],
                "task.narrative_selection_policy",
            )
            expected_policy = narrative_selection_contract(str(narrative_subtype))
            if dict(selection_policy) != expected_policy:
                _fail(
                    "TASK_OPERATION_BINDING_INVALID",
                    "task.narrative_selection_policy",
                    "must exactly match the registered subtype extraction policy",
                )
    else:
        if narrative_subtype is not None:
            _fail(
                "INVALID_TASK_PARAMETERS",
                "task.narrative_subtype",
                "quantitative tasks must not carry a narrative subtype",
            )
        if (
            contract_version == "v2.3"
            and task["narrative_selection_policy"] is not None
        ):
            _fail(
                "INVALID_TASK_PARAMETERS",
                "task.narrative_selection_policy",
                "quantitative tasks must not carry a narrative selection policy",
            )
        contract = _mapping(metric_contract, "task.metric_operation_contract")
        _exact_keys(
            contract,
            required=(
                "metric_operation_contract_version",
                "metric_spec_id",
                "metric_spec_version",
                "kind",
                "ordered_inputs",
                "formula",
                "output_unit",
                "applicable_period_types",
                "deterministic_refusal_conditions",
                "contract_sha256",
            ),
            field="task.metric_operation_contract",
        )
        if contract["metric_operation_contract_version"] != contract_version:
            _fail(
                "UNSUPPORTED_VERSION",
                "task.metric_operation_contract.metric_operation_contract_version",
                f"must equal {contract_version}",
            )
        for key in ("metric_spec_id", "metric_spec_version", "kind", "output_unit"):
            _string(contract[key], f"task.metric_operation_contract.{key}")
        ordered_inputs = contract["ordered_inputs"]
        if not isinstance(ordered_inputs, (list, tuple)) or not ordered_inputs:
            _fail(
                "INVALID_TYPE",
                "task.metric_operation_contract.ordered_inputs",
                "must be a non-empty ordered array",
            )
        for index, raw_input in enumerate(ordered_inputs):
            input_spec = _mapping(
                raw_input, f"task.metric_operation_contract.ordered_inputs[{index}]"
            )
            _exact_keys(
                input_spec,
                required=("name", "unit_family", "period"),
                field=f"task.metric_operation_contract.ordered_inputs[{index}]",
            )
            for key in ("name", "unit_family", "period"):
                _string(
                    input_spec[key],
                    f"task.metric_operation_contract.ordered_inputs[{index}].{key}",
                )
        _mapping(contract["formula"], "task.metric_operation_contract.formula")
        _string_list(
            contract["applicable_period_types"],
            "task.metric_operation_contract.applicable_period_types",
            allow_empty=False,
        )
        _string_list(
            contract["deterministic_refusal_conditions"],
            "task.metric_operation_contract.deterministic_refusal_conditions",
            allow_empty=False,
        )
        _sha256(
            contract["contract_sha256"],
            "task.metric_operation_contract.contract_sha256",
        )
        unhashed = dict(contract)
        expected_contract_hash = unhashed.pop("contract_sha256")
        if canonical_json_sha256(unhashed) != expected_contract_hash:
            _fail(
                "HASH_MISMATCH",
                "task.metric_operation_contract.contract_sha256",
                "does not bind the complete metric contract",
            )

    standards = _mapping(task["standards_profile"], "task.standards_profile")
    _exact_keys(standards, required=("name", "version"), field="task.standards_profile")
    _string(standards["name"], "task.standards_profile.name")
    _string(standards["version"], "task.standards_profile.version")

    entity = _mapping(task["entity"], "task.entity")
    _exact_keys(
        entity,
        required=("entity_id", "name"),
        optional=("ticker", "source_entity_id"),
        field="task.entity",
    )
    _string(entity["entity_id"], "task.entity.entity_id")
    _string(entity["name"], "task.entity.name")
    _optional_string_field(entity, "ticker", "task.entity")
    _optional_string_field(entity, "source_entity_id", "task.entity")

    filing = _mapping(task["filing"], "task.filing")
    _exact_keys(
        filing,
        required=("filing_id",),
        optional=("form_type", "accession"),
        field="task.filing",
    )
    _string(filing["filing_id"], "task.filing.filing_id")
    _optional_string_field(filing, "form_type", "task.filing")
    _optional_string_field(filing, "accession", "task.filing")

    period = _mapping(task["period"], "task.period")
    _exact_keys(
        period,
        required=("period_key",),
        optional=("start_date", "end_date", "instant", "selector_periods"),
        field="task.period",
    )
    _string(period["period_key"], "task.period.period_key")
    for key in ("start_date", "end_date", "instant"):
        _optional_string_field(period, key, "task.period")
    if "selector_periods" in period:
        if task["task_type"] != "quant_metric":
            _fail(
                "INVALID_TASK_PERIOD",
                "task.period.selector_periods",
                "is only allowed for quantitative metric tasks",
            )
        selector_periods = _mapping(
            period["selector_periods"], "task.period.selector_periods"
        )
        if not selector_periods:
            _fail(
                "EMPTY_OBJECT",
                "task.period.selector_periods",
                "must contain at least one exact ASOF selector binding",
            )
        _exact_keys(
            selector_periods,
            required=(),
            optional=tuple(sorted(_ASOF_SELECTORS)),
            field="task.period.selector_periods",
        )
        selector_dates: dict[str, date] = {}
        for selector, value in selector_periods.items():
            period_key = _string(value, f"task.period.selector_periods.{selector}")
            assert period_key is not None
            match = _ASOF_PERIOD_KEY_RE.fullmatch(period_key)
            if match is None:
                _fail(
                    "INVALID_PERIOD_KEY",
                    f"task.period.selector_periods.{selector}",
                    "must use ASOF_YYYYMMDD",
                )
            try:
                selector_dates[selector] = date(*(int(part) for part in match.groups()))
            except ValueError:
                _fail(
                    "INVALID_PERIOD_KEY",
                    f"task.period.selector_periods.{selector}",
                    "must contain a valid calendar date",
                )
        current_selector = "current_period_end_asof"
        prior_selector = "prior_year_period_end_asof"
        current_date = selector_dates.get(current_selector)
        prior_date = selector_dates.get(prior_selector)
        if prior_date is not None and current_date is None:
            _fail(
                "PERIOD_BINDING_CONFLICT",
                f"task.period.selector_periods.{prior_selector}",
                "requires a current_period_end_asof binding",
            )
        end_date = period.get("end_date")
        if (
            current_date is not None
            and isinstance(end_date, str)
            and end_date != current_date.isoformat()
        ):
            _fail(
                "PERIOD_BINDING_CONFLICT",
                "task.period.end_date",
                "must exactly match current_period_end_asof",
            )
        if current_date is not None and prior_date is not None:
            if current_date.year == 1:
                _fail(
                    "PERIOD_BINDING_CONFLICT",
                    f"task.period.selector_periods.{prior_selector}",
                    "cannot represent a prior year for year 0001",
                )
            try:
                expected_prior = current_date.replace(year=current_date.year - 1)
            except ValueError:
                expected_prior = current_date.replace(
                    year=current_date.year - 1, day=28
                )
            if abs((prior_date - expected_prior).days) > 7:
                _fail(
                    "PERIOD_BINDING_CONFLICT",
                    f"task.period.selector_periods.{prior_selector}",
                    "must be within seven days of the leap-safe prior-year period end",
                )

    task_parameters = _mapping(task["task_parameters"], "task.task_parameters")
    _exact_keys(
        task_parameters,
        required=("metric_spec_id", "retrieval_query"),
        field="task.task_parameters",
    )
    _string(
        task_parameters["metric_spec_id"],
        "task.task_parameters.metric_spec_id",
        nullable=True,
    )
    _string(
        task_parameters["retrieval_query"],
        "task.task_parameters.retrieval_query",
        nullable=True,
    )
    if task["task_type"] == "quant_metric":
        if (
            task_parameters["metric_spec_id"] is None
            or task_parameters["retrieval_query"] is not None
        ):
            _fail(
                "INVALID_TASK_PARAMETERS",
                "task.task_parameters",
                "quant_metric requires metric_spec_id and forbids retrieval_query",
            )
        from .agent_tools import metric_operation_contract

        expected_contract = metric_operation_contract(
            str(task_parameters["metric_spec_id"]),
            contract_version=str(contract_version),
        )
        if canonical_json_sha256(
            task["metric_operation_contract"]
        ) != canonical_json_sha256(expected_contract):
            _fail(
                "TASK_OPERATION_BINDING_INVALID",
                "task.metric_operation_contract",
                "must exactly match the registered non-gold metric contract",
            )
    else:
        if (
            task_parameters["metric_spec_id"] is not None
            or task_parameters["retrieval_query"] is None
        ):
            _fail(
                "INVALID_TASK_PARAMETERS",
                "task.task_parameters",
                "narrative tasks require retrieval_query and forbid metric_spec_id",
            )

    allowed_tools = _string_list(task["allowed_tools"], "task.allowed_tools")
    unknown_tools = sorted(set(allowed_tools) - REGISTERED_TOOLS)
    if unknown_tools:
        _fail(
            "UNREGISTERED_TOOL",
            "task.allowed_tools",
            f"unregistered tools: {', '.join(unknown_tools)}",
        )
    spec = task_operation_spec(task["task_type"], version=str(contract_version))
    if allowed_tools != [spec.operation]:
        _fail(
            "TASK_OPERATION_BINDING_INVALID",
            "task.allowed_tools",
            f"must be exactly [{spec.operation!r}]",
        )
    if task["output_schema_id"] != spec.terminal_output_schema_id:
        _fail(
            "TASK_OPERATION_BINDING_INVALID",
            "task.output_schema_id",
            f"must equal {spec.terminal_output_schema_id}",
        )

    evidence_scope = _mapping(task["evidence_scope"], "task.evidence_scope")
    _exact_keys(
        evidence_scope,
        required=(
            "evidence_ids",
            "max_items",
            "selection_method",
            "empty_reason",
        ),
        optional=("filing_ids",),
        field="task.evidence_scope",
    )
    evidence_ids = _string_list(
        evidence_scope["evidence_ids"],
        "task.evidence_scope.evidence_ids",
        max_items=5,
    )
    max_items = _integer(
        evidence_scope["max_items"], "task.evidence_scope.max_items", minimum=0
    )
    assert max_items is not None
    if max_items > 5:
        _fail(
            "TOO_MANY_ITEMS",
            "task.evidence_scope.max_items",
            "must be no greater than 5",
        )
    if len(evidence_ids) > max_items:
        _fail(
            "EVIDENCE_SCOPE_EXCEEDED",
            "task.evidence_scope",
            "evidence_ids exceeds max_items",
        )
    evidence_filing_ids = _string_list(
        evidence_scope.get("filing_ids", [filing["filing_id"]]),
        "task.evidence_scope.filing_ids",
        allow_empty=False,
        max_items=5,
    )
    if filing["filing_id"] not in evidence_filing_ids:
        _fail(
            "FILING_SCOPE_INVALID",
            "task.evidence_scope.filing_ids",
            "must include the primary task filing_id",
        )
    expected_selection_method = (
        "TYPED_FACT_MATERIALIZATION"
        if task["task_type"] == "quant_metric"
        else (
            "FROZEN_BM25_V23_SECTION_SPLIT"
            if contract_version == "v2.3"
            else "FROZEN_BM25"
        )
    )
    if evidence_scope["selection_method"] != expected_selection_method:
        _fail(
            "INVALID_EVIDENCE_SELECTION",
            "task.evidence_scope.selection_method",
            f"must equal {expected_selection_method}",
        )
    empty_reason = evidence_scope["empty_reason"]
    allowed_empty_reasons = (
        frozenset({"NO_VALID_FACTS"})
        if task["task_type"] == "quant_metric"
        else frozenset({"NO_MATCHING_EVIDENCE"})
    )
    if evidence_ids:
        if empty_reason is not None:
            _fail(
                "INVALID_EVIDENCE_SELECTION",
                "task.evidence_scope.empty_reason",
                "must be null when evidence_ids is non-empty",
            )
    elif empty_reason not in allowed_empty_reasons:
        _fail(
            "INVALID_EVIDENCE_SELECTION",
            "task.evidence_scope.empty_reason",
            f"must be one of: {', '.join(sorted(allowed_empty_reasons))}",
        )

    refusal_policy = _mapping(task["refusal_policy"], "task.refusal_policy")
    _exact_keys(
        refusal_policy, required=("allowed_codes",), field="task.refusal_policy"
    )
    refusal_codes = _string_list(
        refusal_policy["allowed_codes"], "task.refusal_policy.allowed_codes"
    )
    if "PROMPT_INJECTION_DETECTED" not in refusal_codes:
        _fail(
            "SECURITY_REFUSAL_REQUIRED",
            "task.refusal_policy.allowed_codes",
            "must always permit PROMPT_INJECTION_DETECTED",
        )
    try:
        validate_task_operation_binding(task)
    except ValueError as exc:
        _fail("TASK_OPERATION_BINDING_INVALID", "task", str(exc))


def sanitize_agent_task_input(task_input: Mapping[str, Any]) -> dict[str, Any]:
    """Return an isolated inference input, rejecting rather than dropping gold data."""

    validate_agent_task_input(task_input)
    return copy.deepcopy(dict(task_input))


def build_agent_task_input(
    *,
    task_id: str,
    task_type: str,
    jurisdiction: str,
    reporting_framework: str,
    standards_profile: Mapping[str, Any],
    source_system: str,
    question: str,
    entity: Mapping[str, Any],
    filing: Mapping[str, Any],
    period: Mapping[str, Any],
    metric_spec_id: str | None = None,
    retrieval_query: str | None = None,
    narrative_subtype: str | None = None,
    allowed_tools: Sequence[str],
    evidence_ids: Sequence[str],
    output_schema_id: str,
    refusal_codes: Sequence[str],
    evidence_max_items: int = 5,
    evidence_filing_ids: Sequence[str] | None = None,
    evidence_selection_method: str | None = None,
    evidence_empty_reason: str | None = None,
    contract_version: str = "v2.2",
) -> dict[str, Any]:
    if contract_version not in {"v2.2", "v2.3"}:
        raise ValueError("contract_version must be v2.2 or v2.3")
    legacy_schema_ids = {
        "quantitative_answer_or_refusal.v2": f"quantitative_answer_or_refusal.{contract_version}",
        "narrative_answer_or_refusal.v2": f"narrative_selection_or_refusal.{contract_version}",
        "quantitative_answer_or_refusal.v2.2": f"quantitative_answer_or_refusal.{contract_version}",
        "narrative_selection_or_refusal.v2.2": f"narrative_selection_or_refusal.{contract_version}",
    }
    output_schema_id = legacy_schema_ids.get(output_schema_id, output_schema_id)
    metric_contract = None
    if task_type == "quant_metric":
        if not isinstance(metric_spec_id, str) or not metric_spec_id:
            raise ValueError("quant_metric requires metric_spec_id")
        from .agent_tools import metric_operation_contract

        metric_contract = metric_operation_contract(
            metric_spec_id, contract_version=contract_version
        )
    elif task_type == "narrative_citation" and narrative_subtype is None:
        # Convenience for small programmatic fixtures. Frozen benchmark builders
        # always provide the source-derived subtype explicitly.
        narrative_subtype = "footnote_note"
    materialized_evidence_ids = list(evidence_ids)
    selection_method = evidence_selection_method or (
        "TYPED_FACT_MATERIALIZATION"
        if task_type == "quant_metric"
        else (
            "FROZEN_BM25_V23_SECTION_SPLIT"
            if contract_version == "v2.3"
            else "FROZEN_BM25"
        )
    )
    empty_reason = evidence_empty_reason
    if not materialized_evidence_ids and empty_reason is None:
        empty_reason = (
            "NO_VALID_FACTS" if task_type == "quant_metric" else "NO_MATCHING_EVIDENCE"
        )
    task = {
        "agent_task_input_version": contract_version,
        "task_id": task_id,
        "task_type": task_type,
        "jurisdiction": jurisdiction,
        "reporting_framework": reporting_framework,
        "standards_profile": dict(standards_profile),
        "source_system": source_system,
        "question": question,
        "entity": dict(entity),
        "filing": dict(filing),
        "period": dict(period),
        "task_parameters": {
            "metric_spec_id": metric_spec_id,
            "retrieval_query": retrieval_query,
        },
        "narrative_subtype": narrative_subtype,
        "metric_operation_contract": metric_contract,
        "allowed_tools": list(allowed_tools),
        "evidence_scope": {
            "evidence_ids": materialized_evidence_ids,
            "filing_ids": list(
                evidence_filing_ids
                if evidence_filing_ids is not None
                else [filing["filing_id"]]
            ),
            "max_items": evidence_max_items,
            "selection_method": selection_method,
            "empty_reason": empty_reason,
        },
        "output_schema_id": output_schema_id,
        "refusal_policy": {
            "allowed_codes": sorted({*refusal_codes, "PROMPT_INJECTION_DETECTED"})
        },
    }
    if contract_version == "v2.3":
        task["narrative_selection_policy"] = (
            narrative_selection_contract(str(narrative_subtype))
            if task_type == "narrative_citation"
            else None
        )
    validate_agent_task_input(task)
    return task


def _validate_claim(claim: Any, index: int) -> None:
    field = f"proposal.claims[{index}]"
    payload = _mapping(claim, field)
    _exact_keys(
        payload,
        required=("claim_id", "text", "evidence_ids", "supporting_text"),
        field=field,
    )
    _string(payload["claim_id"], f"{field}.claim_id")
    _string(payload["text"], f"{field}.text")
    _string_list(
        payload["evidence_ids"], f"{field}.evidence_ids", allow_empty=False, max_items=5
    )
    _string(payload["supporting_text"], f"{field}.supporting_text")


def validate_agent_proposal(proposal: Mapping[str, Any]) -> None:
    value = _mapping(proposal, "proposal")
    _exact_keys(
        value,
        required=(
            "agent_proposal_version",
            "task_id",
            "action",
            "tool_name",
            "tool_arguments",
            "status",
            "value",
            "unit",
            "period_key",
            "answer_text",
            "evidence_ids",
            "claims",
            "refusal_code",
            "model_uncertainty",
            "model_escalation_requested",
        ),
        field="proposal",
    )
    if value["agent_proposal_version"] != AGENT_PROPOSAL_VERSION:
        _fail(
            "UNSUPPORTED_VERSION",
            "proposal.agent_proposal_version",
            f"must equal {AGENT_PROPOSAL_VERSION}",
        )
    _string(value["task_id"], "proposal.task_id")
    action = _enum(value["action"], PROPOSAL_ACTIONS, "proposal.action")
    _string(value["tool_name"], "proposal.tool_name", nullable=True)
    if value["tool_arguments"] is not None:
        _mapping(value["tool_arguments"], "proposal.tool_arguments")
    if value["status"] is not None:
        _enum(value["status"], frozenset({"OK", "REFUSAL"}), "proposal.status")
    _answer_value(value["value"], "proposal.value")
    for key in ("unit", "period_key", "answer_text", "refusal_code"):
        _string(value[key], f"proposal.{key}", nullable=True)
    evidence_ids = _string_list(
        value["evidence_ids"], "proposal.evidence_ids", max_items=5
    )
    if not isinstance(value["claims"], (list, tuple)):
        _fail("INVALID_TYPE", "proposal.claims", "must be an array")
    for index, claim in enumerate(value["claims"]):
        _validate_claim(claim, index)
        claim_evidence = set(claim["evidence_ids"])
        if not claim_evidence.issubset(evidence_ids):
            _fail(
                "CLAIM_EVIDENCE_OUT_OF_SCOPE",
                f"proposal.claims[{index}].evidence_ids",
                "must be included in proposal.evidence_ids",
            )
    claim_ids = [claim["claim_id"] for claim in value["claims"]]
    if len(claim_ids) != len(set(claim_ids)):
        _fail("DUPLICATE_ITEMS", "proposal.claims", "claim_id values must be unique")
    model_uncertainty = _enum(
        value["model_uncertainty"],
        UNCERTAINTY_LEVELS,
        "proposal.model_uncertainty",
    )
    model_escalation_requested = _boolean(
        value["model_escalation_requested"],
        "proposal.model_escalation_requested",
    )

    answer_fields = ("value", "unit", "period_key", "answer_text", "refusal_code")
    if action == "CALL_TOOL":
        _string(value["tool_name"], "proposal.tool_name")
        _mapping(value["tool_arguments"], "proposal.tool_arguments")
        if value["status"] is not None or any(
            value[key] is not None for key in answer_fields
        ):
            _fail(
                "ACTION_FIELD_CONFLICT",
                "proposal",
                "CALL_TOOL cannot include answer or refusal fields",
            )
        if evidence_ids or value["claims"]:
            _fail(
                "ACTION_FIELD_CONFLICT",
                "proposal",
                "CALL_TOOL cannot include final evidence or claims",
            )
        if model_uncertainty != "NONE" or model_escalation_requested:
            _fail(
                "ACTION_FIELD_CONFLICT",
                "proposal",
                "CALL_TOOL advisory fields must be NONE and false",
            )
    elif action == "ANSWER":
        if value["tool_name"] is not None or value["tool_arguments"] is not None:
            _fail(
                "ACTION_FIELD_CONFLICT", "proposal", "ANSWER cannot include a tool call"
            )
        if value["status"] != "OK":
            _fail("INVALID_STATUS", "proposal.status", "ANSWER requires status OK")
        if value["refusal_code"] is not None:
            _fail(
                "ACTION_FIELD_CONFLICT",
                "proposal.refusal_code",
                "ANSWER cannot include a refusal code",
            )
        if value["value"] is None and value["answer_text"] is None:
            _fail("MISSING_ANSWER", "proposal", "ANSWER requires value or answer_text")
        _string(value["period_key"], "proposal.period_key")
        if not evidence_ids:
            _fail(
                "MISSING_EVIDENCE",
                "proposal.evidence_ids",
                "ANSWER requires at least one evidence ID",
            )
        if value["answer_text"] is not None and not value["claims"]:
            _fail(
                "MISSING_CLAIMS",
                "proposal.claims",
                "narrative ANSWER requires claim-level support",
            )
        if value["value"] is not None:
            _string(value["unit"], "proposal.unit")
            if value["answer_text"] is not None or value["claims"]:
                _fail(
                    "ACTION_FIELD_CONFLICT",
                    "proposal",
                    "quantitative ANSWER cannot include narrative text or claims",
                )
        elif value["unit"] is not None:
            _fail(
                "ACTION_FIELD_CONFLICT",
                "proposal.unit",
                "narrative ANSWER cannot include a unit",
            )
    else:
        if value["tool_name"] is not None or value["tool_arguments"] is not None:
            _fail(
                "ACTION_FIELD_CONFLICT", "proposal", "REFUSE cannot include a tool call"
            )
        if value["status"] != "REFUSAL":
            _fail("INVALID_STATUS", "proposal.status", "REFUSE requires status REFUSAL")
        if (
            value["value"] is not None
            or value["unit"] is not None
            or value["answer_text"] is not None
        ):
            _fail(
                "ACTION_FIELD_CONFLICT",
                "proposal",
                "REFUSE cannot include answer fields",
            )
        _string(value["refusal_code"], "proposal.refusal_code")
        _string(value["period_key"], "proposal.period_key")
        if evidence_ids:
            _fail(
                "ACTION_FIELD_CONFLICT",
                "proposal.evidence_ids",
                "REFUSE cannot include citations",
            )
        if value["claims"]:
            _fail(
                "ACTION_FIELD_CONFLICT",
                "proposal.claims",
                "REFUSE cannot include claims",
            )


def build_agent_proposal(
    *,
    task_id: str,
    action: str,
    tool_name: str | None = None,
    tool_arguments: Mapping[str, Any] | None = None,
    status: str | None = None,
    value: Any = None,
    unit: str | None = None,
    period_key: str | None = None,
    answer_text: str | None = None,
    evidence_ids: Sequence[str] = (),
    claims: Sequence[Mapping[str, Any]] = (),
    refusal_code: str | None = None,
    model_uncertainty: str = "NONE",
    model_escalation_requested: bool = False,
) -> dict[str, Any]:
    proposal = {
        "agent_proposal_version": AGENT_PROPOSAL_VERSION,
        "task_id": task_id,
        "action": action,
        "tool_name": tool_name,
        "tool_arguments": dict(tool_arguments) if tool_arguments is not None else None,
        "status": status,
        "value": value,
        "unit": unit,
        "period_key": period_key,
        "answer_text": answer_text,
        "evidence_ids": list(evidence_ids),
        "claims": [dict(claim) for claim in claims],
        "refusal_code": refusal_code,
        "model_uncertainty": model_uncertainty,
        "model_escalation_requested": model_escalation_requested,
    }
    validate_agent_proposal(proposal)
    return proposal


def _validate_check(check: Any, index: int, *, repair_error: bool = False) -> None:
    prefix = "result.repair_errors" if repair_error else "result.checks"
    field = f"{prefix}[{index}]"
    payload = _mapping(check, field)
    required = (
        ("code", "field", "message")
        if repair_error
        else ("code", "passed", "field", "message")
    )
    _exact_keys(payload, required=required, field=field)
    _enum(payload["code"], VERIFIER_CHECK_CODES, f"{field}.code")
    if not repair_error:
        _boolean(payload["passed"], f"{field}.passed")
    _string(payload["field"], f"{field}.field", nullable=True)
    _string(payload["message"], f"{field}.message")


def validate_verifier_result(result: Mapping[str, Any]) -> None:
    value = _mapping(result, "result")
    _exact_keys(
        value,
        required=(
            "verifier_result_version",
            "task_id",
            "passed",
            "release_allowed",
            "disposition",
            "repair_count",
            "repair_allowed",
            "checks",
            "repair_errors",
        ),
        field="result",
    )
    if value["verifier_result_version"] != VERIFIER_RESULT_VERSION:
        _fail(
            "UNSUPPORTED_VERSION",
            "result.verifier_result_version",
            f"must equal {VERIFIER_RESULT_VERSION}",
        )
    _string(value["task_id"], "result.task_id")
    passed = _boolean(value["passed"], "result.passed")
    release_allowed = _boolean(value["release_allowed"], "result.release_allowed")
    disposition = _enum(
        value["disposition"], VERIFIER_DISPOSITIONS, "result.disposition"
    )
    repair_count = _integer(value["repair_count"], "result.repair_count", minimum=0)
    assert repair_count is not None
    if repair_count > MAX_REPAIR_ATTEMPTS:
        _fail("REPAIR_LIMIT_EXCEEDED", "result.repair_count", "must be 0 or 1")
    repair_allowed = _boolean(value["repair_allowed"], "result.repair_allowed")
    if not isinstance(value["checks"], (list, tuple)) or not value["checks"]:
        _fail("INVALID_CHECKS", "result.checks", "must be a non-empty array")
    for index, check in enumerate(value["checks"]):
        _validate_check(check, index)
    if not isinstance(value["repair_errors"], (list, tuple)):
        _fail("INVALID_TYPE", "result.repair_errors", "must be an array")
    for index, error in enumerate(value["repair_errors"]):
        _validate_check(error, index, repair_error=True)

    failed_checks = [check for check in value["checks"] if not check["passed"]]
    if passed == bool(failed_checks):
        _fail(
            "INCONSISTENT_RESULT",
            "result.passed",
            "must be true exactly when every check passes",
        )
    release_dispositions = {"RELEASED", "SAFE_REFUSAL"}
    if disposition in release_dispositions and not release_allowed:
        _fail(
            "INCONSISTENT_RESULT",
            "result.release_allowed",
            "released dispositions require release_allowed",
        )
    if disposition not in release_dispositions and release_allowed:
        _fail(
            "INCONSISTENT_RESULT",
            "result.release_allowed",
            "non-release dispositions cannot be released",
        )
    if release_allowed and not passed:
        _fail(
            "INCONSISTENT_RESULT",
            "result.release_allowed",
            "a failed verifier result can never be released",
        )
    passed_dispositions = release_dispositions | {"TOOL_CALL_APPROVED"}
    if passed and disposition not in passed_dispositions:
        _fail(
            "INCONSISTENT_RESULT",
            "result.disposition",
            "a passed result must use a passed disposition",
        )
    if not passed and disposition not in {"REPAIR_REQUIRED", "FAILED"}:
        _fail(
            "INCONSISTENT_RESULT",
            "result.disposition",
            "a failed result must require repair or be terminal",
        )
    if disposition == "REPAIR_REQUIRED" and (not repair_allowed or repair_count != 0):
        _fail(
            "INCONSISTENT_RESULT",
            "result.repair_allowed",
            "only the initial failed attempt may request repair",
        )
    if disposition == "REPAIR_REQUIRED" and not value["repair_errors"]:
        _fail(
            "INCONSISTENT_RESULT",
            "result.repair_errors",
            "REPAIR_REQUIRED requires at least one safe diagnostic",
        )
    if disposition != "REPAIR_REQUIRED" and repair_allowed:
        _fail(
            "INCONSISTENT_RESULT",
            "result.repair_allowed",
            "repair is only allowed for REPAIR_REQUIRED",
        )
    if disposition != "REPAIR_REQUIRED" and value["repair_errors"]:
        _fail(
            "INCONSISTENT_RESULT",
            "result.repair_errors",
            "terminal and released results cannot expose repair diagnostics",
        )
    failed_error_candidates = [
        {
            "code": check["code"],
            "field": check["field"],
            "message": check["message"],
        }
        for check in failed_checks
    ]
    repair_errors = list(value["repair_errors"])
    if repair_errors and not all(
        error in failed_error_candidates for error in repair_errors
    ):
        _fail(
            "INCONSISTENT_RESULT",
            "result.repair_errors",
            "must be an ordered subset of failed checks",
        )


def validate_model_config(config: Mapping[str, Any]) -> None:
    value = _mapping(config, "model_config")
    _exact_keys(
        value,
        required=(
            "model_config_version",
            "model_id",
            "model_path",
            "revision",
            "quantization",
            "backend",
            "context_window_tokens",
            "max_input_tokens",
            "max_plan_tokens",
            "max_answer_tokens",
            "max_repair_tokens",
            "temperature",
            "top_p",
            "seed",
            "thinking_enabled",
            "chat_template_sha256",
        ),
        field="model_config",
    )
    if value["model_config_version"] != MODEL_CONFIG_VERSION:
        _fail(
            "UNSUPPORTED_VERSION",
            "model_config.model_config_version",
            f"must equal {MODEL_CONFIG_VERSION}",
        )
    _string(value["model_id"], "model_config.model_id")
    _string(value["model_path"], "model_config.model_path", nullable=True)
    _string(value["revision"], "model_config.revision")
    _string(value["quantization"], "model_config.quantization", nullable=True)
    backend = _enum(
        value["backend"], frozenset({"vllm_offline", "mock"}), "model_config.backend"
    )
    if backend == "vllm_offline" and value["model_path"] is None:
        _fail(
            "LOCAL_MODEL_REQUIRED",
            "model_config.model_path",
            "vllm_offline requires a local snapshot path",
        )
    context_window = _integer(
        value["context_window_tokens"], "model_config.context_window_tokens", minimum=1
    )
    max_input = _integer(
        value["max_input_tokens"], "model_config.max_input_tokens", minimum=1
    )
    assert context_window is not None and max_input is not None
    if max_input > MAX_INPUT_TOKENS:
        _fail(
            "INPUT_CAP_EXCEEDED",
            "model_config.max_input_tokens",
            f"must be no greater than {MAX_INPUT_TOKENS}",
        )
    limits = (
        ("max_plan_tokens", MAX_PLAN_TOKENS),
        ("max_answer_tokens", MAX_ANSWER_TOKENS),
        ("max_repair_tokens", MAX_REPAIR_TOKENS),
    )
    output_limits: list[int] = []
    for key, cap in limits:
        limit = _integer(value[key], f"model_config.{key}", minimum=1)
        assert limit is not None
        if limit > cap:
            _fail(
                "OUTPUT_CAP_EXCEEDED",
                f"model_config.{key}",
                f"must be no greater than {cap}",
            )
        output_limits.append(limit)
    if max_input + max(output_limits) > context_window:
        _fail(
            "CONTEXT_WINDOW_EXCEEDED",
            "model_config",
            "input and largest output budget exceed context window",
        )
    if _number(value["temperature"], "model_config.temperature") != 0.0:
        _fail("NONDETERMINISTIC_SAMPLING", "model_config.temperature", "must equal 0")
    if _number(value["top_p"], "model_config.top_p") != 1.0:
        _fail("NONDETERMINISTIC_SAMPLING", "model_config.top_p", "must equal 1")
    seed = _integer(value["seed"], "model_config.seed", minimum=0)
    if seed != DEFAULT_SEED:
        _fail("NONSTANDARD_SEED", "model_config.seed", f"must equal {DEFAULT_SEED}")
    if _boolean(value["thinking_enabled"], "model_config.thinking_enabled"):
        _fail("THINKING_NOT_ALLOWED", "model_config.thinking_enabled", "must be false")
    _sha256(
        value["chat_template_sha256"],
        "model_config.chat_template_sha256",
        nullable=True,
    )
    if backend == "vllm_offline" and value["chat_template_sha256"] is None:
        _fail(
            "CHAT_TEMPLATE_HASH_REQUIRED",
            "model_config.chat_template_sha256",
            "vllm_offline requires the pinned snapshot chat-template hash",
        )


def build_model_config(
    *,
    model_id: str,
    revision: str,
    model_path: str | None = None,
    quantization: str | None = None,
    backend: str = "vllm_offline",
    context_window_tokens: int = 16_384,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    max_plan_tokens: int = MAX_PLAN_TOKENS,
    max_answer_tokens: int = MAX_ANSWER_TOKENS,
    max_repair_tokens: int = MAX_REPAIR_TOKENS,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = DEFAULT_SEED,
    thinking_enabled: bool = False,
    chat_template_sha256: str | None = None,
) -> dict[str, Any]:
    config = {
        "model_config_version": MODEL_CONFIG_VERSION,
        "model_id": model_id,
        "model_path": model_path,
        "revision": revision,
        "quantization": quantization,
        "backend": backend,
        "context_window_tokens": context_window_tokens,
        "max_input_tokens": max_input_tokens,
        "max_plan_tokens": max_plan_tokens,
        "max_answer_tokens": max_answer_tokens,
        "max_repair_tokens": max_repair_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "thinking_enabled": thinking_enabled,
        "chat_template_sha256": chat_template_sha256,
    }
    validate_model_config(config)
    return config


def semantic_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the portable behavioral model identity used in v2 run hashes."""

    validate_model_config(config)
    return {
        "model_config_version": config["model_config_version"],
        "model_id": config["model_id"],
        "revision": config["revision"],
        "quantization": config["quantization"],
        "backend": config["backend"],
        "context_window_tokens": config["context_window_tokens"],
        "max_input_tokens": config["max_input_tokens"],
        "max_plan_tokens": config["max_plan_tokens"],
        "max_answer_tokens": config["max_answer_tokens"],
        "max_repair_tokens": config["max_repair_tokens"],
        "temperature": float(config["temperature"]),
        "top_p": float(config["top_p"]),
        "seed": config["seed"],
        "thinking_enabled": config["thinking_enabled"],
        "chat_template_sha256": config["chat_template_sha256"],
    }


def _validate_tool_call(value: Any, index: int) -> None:
    field = f"run.tool_calls[{index}]"
    payload = _mapping(value, field)
    _exact_keys(
        payload,
        required=(
            "tool_name",
            "arguments_sha256",
            "observation_sha256",
            "status",
            "duration_ms",
        ),
        field=field,
    )
    _enum(payload["tool_name"], REGISTERED_TOOLS, f"{field}.tool_name")
    _sha256(payload["arguments_sha256"], f"{field}.arguments_sha256")
    _sha256(payload["observation_sha256"], f"{field}.observation_sha256", nullable=True)
    _enum(payload["status"], frozenset({"OK", "ERROR"}), f"{field}.status")
    _number(payload["duration_ms"], f"{field}.duration_ms")
    if payload["status"] == "OK" and payload["observation_sha256"] is None:
        _fail(
            "MISSING_OBSERVATION_HASH",
            f"{field}.observation_sha256",
            "successful tool calls require a hash",
        )


def _validate_verifier_trace_entry(
    value: Any, index: int, *, task_id: str
) -> Mapping[str, Any]:
    field = f"run.verifier_trace[{index}]"
    entry = _mapping(value, field)
    _exact_keys(
        entry,
        required=(
            "stage",
            "attempt_index",
            "repair_attempt",
            "prompt_sha256",
            "request_sha256",
            "response_schema_sha256",
            "visible_context_sha256",
            "output_sha256",
            "proposal",
            "proposal_sha256",
            "verifier_result",
            "input_tokens",
            "output_tokens",
            "duration_ms",
            "finish_reason",
            "constraint_backend",
            "structured_output_applied",
            "output_bytes",
            "cap_hit",
            "parse_category",
        ),
        field=field,
    )
    _enum(entry["stage"], frozenset({"plan", "synthesis", "direct"}), f"{field}.stage")
    attempt_index = _integer(
        entry["attempt_index"], f"{field}.attempt_index", minimum=0
    )
    if attempt_index != index:
        _fail(
            "TRACE_SEQUENCE_INVALID",
            f"{field}.attempt_index",
            "must equal its zero-based position in the verifier trace",
        )
    _boolean(entry["repair_attempt"], f"{field}.repair_attempt")
    _sha256(entry["prompt_sha256"], f"{field}.prompt_sha256")
    _sha256(entry["request_sha256"], f"{field}.request_sha256")
    _sha256(
        entry["response_schema_sha256"],
        f"{field}.response_schema_sha256",
    )
    _sha256(entry["visible_context_sha256"], f"{field}.visible_context_sha256")
    _sha256(entry["output_sha256"], f"{field}.output_sha256", nullable=True)
    verifier = _mapping(entry["verifier_result"], f"{field}.verifier_result")
    validate_verifier_result(verifier)
    if verifier["task_id"] != task_id:
        _fail(
            "TASK_MISMATCH",
            f"{field}.verifier_result.task_id",
            "must match run.task_id",
        )
    proposal = entry["proposal"]
    if proposal is None:
        if entry["proposal_sha256"] is not None:
            _fail(
                "TRACE_PROPOSAL_HASH_MISMATCH",
                f"{field}.proposal_sha256",
                "must be null when proposal is null",
            )
        failed_checks = [check for check in verifier["checks"] if not check["passed"]]
        allowed_null_codes = {
            "MODEL_GENERATION_FAILED",
            "MODEL_OUTPUT_NOT_STRICT_JSON",
            "SCHEMA_VALID",
            "TASK_MATCH",
        }
        if (
            len(failed_checks) != 1
            or failed_checks[0]["code"] not in allowed_null_codes
        ):
            _fail(
                "TRACE_PROPOSAL_MISSING",
                f"{field}.proposal",
                "may be null only for a typed generation or discarded-proposal failure",
            )
        failure_code = failed_checks[0]["code"]
        if failure_code == "MODEL_GENERATION_FAILED":
            if entry["output_sha256"] is not None:
                _fail(
                    "TRACE_OUTPUT_HASH_MISMATCH",
                    f"{field}.output_sha256",
                    "must be null when model generation produced no completion",
                )
        elif entry["output_sha256"] is None:
            _fail(
                "TRACE_OUTPUT_HASH_MISMATCH",
                f"{field}.output_sha256",
                "must bind the discarded or malformed model completion",
            )
    else:
        proposal_mapping = _mapping(proposal, f"{field}.proposal")
        validate_agent_proposal(proposal_mapping)
        if proposal_mapping["task_id"] != task_id:
            _fail(
                "TASK_MISMATCH", f"{field}.proposal.task_id", "must match run.task_id"
            )
        _sha256(entry["proposal_sha256"], f"{field}.proposal_sha256")
        if entry["proposal_sha256"] != canonical_json_sha256(proposal_mapping):
            _fail(
                "TRACE_PROPOSAL_HASH_MISMATCH",
                f"{field}.proposal_sha256",
                "does not match the canonical proposal",
            )
        failed_codes = {
            check["code"] for check in verifier["checks"] if not check["passed"]
        }
        if failed_codes & {"SCHEMA_VALID", "TASK_MATCH"}:
            _fail(
                "TRACE_PROPOSAL_DIAGNOSTIC_MISMATCH",
                f"{field}.proposal",
                "contract-invalid and cross-task proposals must be discarded",
            )
    _integer(entry["input_tokens"], f"{field}.input_tokens", minimum=0)
    _integer(entry["output_tokens"], f"{field}.output_tokens", minimum=0)
    _number(entry["duration_ms"], f"{field}.duration_ms")
    if entry["finish_reason"] is not None:
        _enum(
            entry["finish_reason"],
            frozenset({"stop", "length", "abort"}),
            f"{field}.finish_reason",
        )
    _string(entry["constraint_backend"], f"{field}.constraint_backend")
    _boolean(entry["structured_output_applied"], f"{field}.structured_output_applied")
    _integer(entry["output_bytes"], f"{field}.output_bytes", minimum=0)
    _boolean(entry["cap_hit"], f"{field}.cap_hit")
    if entry["parse_category"] is not None:
        _enum(
            entry["parse_category"],
            frozenset(
                {"EMPTY", "NON_OBJECT", "SYNTAX", "DUPLICATE_KEY", "NON_JSON_NUMBER"}
            ),
            f"{field}.parse_category",
        )
    if entry["cap_hit"] != (entry["finish_reason"] == "length"):
        _fail(
            "TRACE_TELEMETRY_MISMATCH",
            f"{field}.cap_hit",
            "must exactly reflect finish_reason=length",
        )
    return entry


def validate_agent_run_record(run_record: Mapping[str, Any]) -> None:
    run = _mapping(run_record, "run")
    record_version = run.get("agent_run_record_version")
    if record_version not in {"v2.2", "v2.3"}:
        _fail(
            "UNSUPPORTED_VERSION",
            "run.agent_run_record_version",
            "must equal v2.2 or v2.3",
        )
    versioned_fields = (
        ("narrative_selection_policy_sha256",) if record_version == "v2.3" else ()
    )
    _exact_keys(
        run,
        required=(
            "agent_run_record_version",
            "run_id",
            "task_id",
            "corpus_id",
            "benchmark_manifest_sha256",
            "runtime_mode",
            "runtime_version",
            "prompt_condition",
            "prompt_version",
            "prompt_sha256",
            "canonical_json_version",
            "semantic_model_config_sha256",
            "deployment_model_config_sha256",
            "deterministic_executor_sha256",
            "stage_schema_bundle_sha256",
            "run_envelope_sha256",
            "demonstration_pack_sha256",
            "narrative_subtype",
            "metric_operation_contract_sha256",
            *versioned_fields,
            "model_config",
            "context_sha256",
            "input_sha256",
            "output_sha256",
            "tool_calls",
            "verifier_trace",
            "verifier_result",
            "repair_count",
            "outcome",
            "token_usage",
            "timings",
            "resources",
            "provenance",
        ),
        field="run",
    )
    for key in ("run_id", "task_id", "corpus_id", "prompt_version"):
        _string(run[key], f"run.{key}")
    expected_runtime_version = (
        TEXT_RUNTIME_ID_V23 if record_version == "v2.3" else TEXT_RUNTIME_ID
    )
    if run["runtime_version"] != expected_runtime_version:
        _fail(
            "UNSUPPORTED_VERSION",
            "run.runtime_version",
            f"must equal {expected_runtime_version}",
        )
    expected_prompt_version = f"auditops.agent_prompt.{record_version}"
    if run["prompt_version"] != expected_prompt_version:
        _fail(
            "UNSUPPORTED_VERSION",
            "run.prompt_version",
            f"must equal {expected_prompt_version}",
        )
    if run["canonical_json_version"] != CANONICAL_JSON_VERSION:
        _fail(
            "UNSUPPORTED_VERSION",
            "run.canonical_json_version",
            f"must equal {CANONICAL_JSON_VERSION}",
        )
    for key in (
        "benchmark_manifest_sha256",
        "prompt_sha256",
        "deterministic_executor_sha256",
        "stage_schema_bundle_sha256",
        "run_envelope_sha256",
    ):
        _sha256(run[key], f"run.{key}")
    for key in ("semantic_model_config_sha256", "deployment_model_config_sha256"):
        _sha256(run[key], f"run.{key}", nullable=True)
    for key in ("context_sha256", "input_sha256", "output_sha256"):
        _sha256(run[key], f"run.{key}", nullable=True)
    _sha256(
        run["demonstration_pack_sha256"],
        "run.demonstration_pack_sha256",
        nullable=True,
    )
    _sha256(
        run["metric_operation_contract_sha256"],
        "run.metric_operation_contract_sha256",
        nullable=True,
    )
    if record_version == "v2.3":
        _sha256(
            run["narrative_selection_policy_sha256"],
            "run.narrative_selection_policy_sha256",
            nullable=True,
        )
    if run["narrative_subtype"] is not None:
        _enum(run["narrative_subtype"], NARRATIVE_SUBTYPES, "run.narrative_subtype")
        if run["metric_operation_contract_sha256"] is not None:
            _fail(
                "TASK_OPERATION_BINDING_INVALID",
                "run.metric_operation_contract_sha256",
                "narrative records cannot bind a metric operation contract",
            )
        if (
            record_version == "v2.3"
            and run["narrative_selection_policy_sha256"] is None
        ):
            _fail(
                "TASK_OPERATION_BINDING_INVALID",
                "run.narrative_selection_policy_sha256",
                "v2.3 narrative records must bind their extraction policy",
            )
    elif run["metric_operation_contract_sha256"] is None:
        _fail(
            "TASK_OPERATION_BINDING_INVALID",
            "run.metric_operation_contract_sha256",
            "quantitative records must bind a metric operation contract",
        )
    elif (
        record_version == "v2.3"
        and run["narrative_selection_policy_sha256"] is not None
    ):
        _fail(
            "TASK_OPERATION_BINDING_INVALID",
            "run.narrative_selection_policy_sha256",
            "quantitative records cannot bind a narrative extraction policy",
        )
    if run["prompt_condition"] == "few_shot":
        if run["demonstration_pack_sha256"] is None:
            _fail(
                "HASH_MISMATCH",
                "run.demonstration_pack_sha256",
                "few-shot records must bind their exact demonstration pack",
            )
    elif run["demonstration_pack_sha256"] is not None:
        _fail(
            "HASH_MISMATCH",
            "run.demonstration_pack_sha256",
            "zero-shot records cannot bind a demonstration pack",
        )
    _enum(
        run["runtime_mode"],
        frozenset({"direct", "capability_agent", "safety_hybrid"}),
        "run.runtime_mode",
    )
    _enum(
        run["prompt_condition"],
        frozenset({"zero_shot", "few_shot"}),
        "run.prompt_condition",
    )
    if run["model_config"] is None:
        if run["runtime_mode"] != "safety_hybrid" or (
            run["semantic_model_config_sha256"] is not None
            or run["deployment_model_config_sha256"] is not None
        ):
            _fail(
                "MODEL_CONFIG_MISSING",
                "run.model_config",
                "may be null only for a model-independent safety-hybrid case",
            )
    else:
        model_config = _mapping(run["model_config"], "run.model_config")
        validate_model_config(model_config)
        if run["semantic_model_config_sha256"] != canonical_json_sha256(
            semantic_model_config(model_config)
        ):
            _fail(
                "MODEL_CONFIG_HASH_MISMATCH",
                "run.semantic_model_config_sha256",
                "does not match the portable semantic model configuration",
            )
        if run["deployment_model_config_sha256"] != canonical_json_sha256(model_config):
            _fail(
                "MODEL_CONFIG_HASH_MISMATCH",
                "run.deployment_model_config_sha256",
                "does not match the complete deployment model configuration",
            )
    if not isinstance(run["tool_calls"], (list, tuple)):
        _fail("INVALID_TYPE", "run.tool_calls", "must be an array")
    for index, tool_call in enumerate(run["tool_calls"]):
        _validate_tool_call(tool_call, index)
    if not isinstance(run["verifier_trace"], (list, tuple)):
        _fail("INVALID_TYPE", "run.verifier_trace", "must be an array")
    trace = [
        _validate_verifier_trace_entry(entry, index, task_id=run["task_id"])
        for index, entry in enumerate(run["verifier_trace"])
    ]
    repair_count = _integer(run["repair_count"], "run.repair_count", minimum=0)
    assert repair_count is not None
    if repair_count > MAX_REPAIR_ATTEMPTS:
        _fail("REPAIR_LIMIT_EXCEEDED", "run.repair_count", "must be 0 or 1")
    repair_entries = [entry for entry in trace if entry["repair_attempt"]]
    if len(repair_entries) != repair_count:
        _fail(
            "REPAIR_COUNT_MISMATCH",
            "run.verifier_trace",
            "repair_attempt entries must exactly match run.repair_count",
        )
    running_repairs = 0
    for index, entry in enumerate(trace):
        if entry["repair_attempt"]:
            running_repairs += 1
            if (
                index == 0
                or trace[index - 1]["stage"] != entry["stage"]
                or trace[index - 1]["verifier_result"]["disposition"]
                != "REPAIR_REQUIRED"
            ):
                _fail(
                    "TRACE_TRANSITION_INVALID",
                    f"run.verifier_trace[{index}]",
                    "a repair must immediately follow REPAIR_REQUIRED in the same stage",
                )
        if entry["verifier_result"]["repair_count"] != running_repairs:
            _fail(
                "REPAIR_COUNT_MISMATCH",
                f"run.verifier_trace[{index}].verifier_result.repair_count",
                "must match repairs already consumed",
            )
        if entry["verifier_result"]["disposition"] == "REPAIR_REQUIRED" and (
            index + 1 >= len(trace) or not trace[index + 1]["repair_attempt"]
        ):
            _fail(
                "TRACE_TRANSITION_INVALID",
                f"run.verifier_trace[{index}]",
                "REPAIR_REQUIRED must be followed by the single repair attempt",
            )
        if index + 1 < len(trace):
            next_stage = trace[index + 1]["stage"]
            disposition = entry["verifier_result"]["disposition"]
            legal_continuation = disposition == "REPAIR_REQUIRED" or (
                entry["stage"] == "plan"
                and next_stage == "synthesis"
                and disposition == "TOOL_CALL_APPROVED"
            )
            if not legal_continuation:
                _fail(
                    "TRACE_TRANSITION_INVALID",
                    f"run.verifier_trace[{index}]",
                    "a terminal verifier result cannot be followed by another model attempt",
                )
    stages = [entry["stage"] for entry in trace]
    if run["runtime_mode"] == "direct" and any(stage != "direct" for stage in stages):
        _fail(
            "TRACE_TRANSITION_INVALID",
            "run.verifier_trace",
            "direct runs may contain only direct-stage attempts",
        )
    if run["runtime_mode"] == "capability_agent":
        if any(stage == "direct" for stage in stages) or stages != sorted(
            stages, key=lambda stage: {"plan": 0, "synthesis": 1}[stage]
        ):
            _fail(
                "TRACE_TRANSITION_INVALID",
                "run.verifier_trace",
                "capability-agent traces must contain plan attempts before synthesis attempts",
            )
        if trace and trace[0]["stage"] != "plan":
            _fail(
                "TRACE_TRANSITION_INVALID",
                "run.verifier_trace",
                "capability-agent traces must begin with a plan attempt",
            )
    if run["runtime_mode"] == "safety_hybrid" and any(
        stage != "synthesis" for stage in stages
    ):
        _fail(
            "TRACE_TRANSITION_INVALID",
            "run.verifier_trace",
            "safety-hybrid model traces may contain only narrative synthesis attempts",
        )
    if sum(stage == "direct" for stage in stages) > 2 or any(
        stages.count(stage) > 2 for stage in ("plan", "synthesis")
    ):
        _fail(
            "TRACE_TRANSITION_INVALID",
            "run.verifier_trace",
            "each generation stage permits at most one initial and one repair attempt",
        )
    outcome = _enum(run["outcome"], TERMINAL_OUTCOMES, "run.outcome")
    if run["run_id"] != f"run-{run['run_envelope_sha256'][:24]}":
        _fail(
            "RUN_ID_MISMATCH",
            "run.run_id",
            "must be derived from run_envelope_sha256",
        )
    if run["verifier_result"] is None:
        if trace:
            _fail(
                "MISSING_VERIFIER_RESULT",
                "run.verifier_result",
                "a non-empty trace requires its terminal verifier result",
            )
        if outcome in {"RELEASED", "SAFE_REFUSAL"}:
            _fail(
                "MISSING_VERIFIER_RESULT",
                "run.verifier_result",
                "released outcomes require a terminal verifier result",
            )
    else:
        validate_verifier_result(
            _mapping(run["verifier_result"], "run.verifier_result")
        )
        if run["verifier_result"]["task_id"] != run["task_id"]:
            _fail(
                "TASK_MISMATCH", "run.verifier_result.task_id", "must match run.task_id"
            )
        if run["verifier_result"]["repair_count"] != repair_count:
            _fail(
                "REPAIR_COUNT_MISMATCH",
                "run.repair_count",
                "must match verifier result",
            )
        verifier_disposition = run["verifier_result"]["disposition"]
        if outcome in {"RELEASED", "SAFE_REFUSAL"} and (
            verifier_disposition != outcome
        ):
            _fail(
                "DISPOSITION_MISMATCH",
                "run.outcome",
                "released outcome must match the verifier release disposition",
            )
        if outcome == "MODEL_FAILURE" and verifier_disposition != "FAILED":
            _fail(
                "DISPOSITION_MISMATCH",
                "run.outcome",
                "model failures with a verifier result require disposition FAILED",
            )
        if outcome in {"INFRASTRUCTURE_FAILURE", "INTEGRITY_FAILURE"} and (
            verifier_disposition not in {"FAILED", "TOOL_CALL_APPROVED"}
        ):
            _fail(
                "DISPOSITION_MISMATCH",
                "run.outcome",
                "infrastructure and integrity failures may end only at FAILED or an approved plan tool call",
            )
        if verifier_disposition == "TOOL_CALL_APPROVED" and (
            outcome not in {"INFRASTRUCTURE_FAILURE", "INTEGRITY_FAILURE"}
            or run["runtime_mode"] != "capability_agent"
            or not trace
            or trace[-1]["stage"] != "plan"
        ):
            _fail(
                "DISPOSITION_MISMATCH",
                "run.verifier_result.disposition",
                "TOOL_CALL_APPROVED may terminate only an agent infrastructure or integrity failure after planning",
            )
        if trace and trace[-1]["verifier_result"] != run["verifier_result"]:
            _fail(
                "VERIFIER_TRACE_MISMATCH",
                "run.verifier_trace",
                "the final trace result must exactly match run.verifier_result",
            )
        deterministic_security_refusal = (
            not trace
            and outcome == "SAFE_REFUSAL"
            and verifier_disposition == "SAFE_REFUSAL"
            and not run["tool_calls"]
            and repair_count == 0
        )
        if (
            not trace
            and run["runtime_mode"] != "safety_hybrid"
            and not deterministic_security_refusal
        ):
            _fail(
                "VERIFIER_TRACE_MISMATCH",
                "run.verifier_trace",
                "only deterministic safety paths may end without a model trace",
            )

    if trace:
        terminal_entry = trace[-1]
        expected_output_sha256 = (
            terminal_entry["proposal_sha256"]
            if terminal_entry["proposal"] is not None
            else terminal_entry["output_sha256"]
        )
        if run["output_sha256"] != expected_output_sha256:
            _fail(
                "RUN_OUTPUT_HASH_MISMATCH",
                "run.output_sha256",
                "must bind the terminal proposal or discarded completion",
            )

    expected_prompt_sha256 = canonical_json_sha256(
        [entry["prompt_sha256"] for entry in trace]
    )
    if run["prompt_sha256"] != expected_prompt_sha256:
        _fail(
            "PROMPT_HASH_MISMATCH",
            "run.prompt_sha256",
            "must bind the ordered verifier-trace prompt hashes",
        )

    usage = _mapping(run["token_usage"], "run.token_usage")
    _exact_keys(
        usage,
        required=("input_tokens", "output_tokens", "total_tokens"),
        field="run.token_usage",
    )
    input_tokens = _integer(
        usage["input_tokens"], "run.token_usage.input_tokens", minimum=0
    )
    output_tokens = _integer(
        usage["output_tokens"], "run.token_usage.output_tokens", minimum=0
    )
    total_tokens = _integer(
        usage["total_tokens"], "run.token_usage.total_tokens", minimum=0
    )
    if total_tokens != input_tokens + output_tokens:
        _fail(
            "TOKEN_COUNT_MISMATCH",
            "run.token_usage.total_tokens",
            "must equal input_tokens plus output_tokens",
        )
    if input_tokens != sum(
        entry["input_tokens"] for entry in trace
    ) or output_tokens != sum(entry["output_tokens"] for entry in trace):
        _fail(
            "TOKEN_COUNT_MISMATCH",
            "run.token_usage",
            "token totals must equal the audited model-call trace",
        )

    timings = _mapping(run["timings"], "run.timings")
    _exact_keys(
        timings,
        required=(
            "started_at",
            "finished_at",
            "duration_ms",
            "engine_initialization_ms",
            "generation_duration_ms",
            "tool_duration_ms",
        ),
        field="run.timings",
    )
    _string(timings["started_at"], "run.timings.started_at")
    _string(timings["finished_at"], "run.timings.finished_at")
    _number(timings["duration_ms"], "run.timings.duration_ms")
    for key in (
        "engine_initialization_ms",
        "generation_duration_ms",
        "tool_duration_ms",
    ):
        measured = _number(timings[key], f"run.timings.{key}")
        if measured is not None and measured < 0:
            _fail("INVALID_NUMBER", f"run.timings.{key}", "must be non-negative")

    resources = _mapping(run["resources"], "run.resources")
    _exact_keys(
        resources,
        required=("slurm_job_id", "gpu_model", "peak_vram_bytes", "host"),
        field="run.resources",
    )
    _string(resources["slurm_job_id"], "run.resources.slurm_job_id", nullable=True)
    _string(resources["gpu_model"], "run.resources.gpu_model", nullable=True)
    _integer(
        resources["peak_vram_bytes"],
        "run.resources.peak_vram_bytes",
        minimum=0,
        nullable=True,
    )
    _string(resources["host"], "run.resources.host", nullable=True)

    provenance = _mapping(run["provenance"], "run.provenance")
    provenance_keys = (
        "git_commit",
        "source_tree_sha256",
        "container_digest",
        "model_snapshot_sha256",
        "tokenizer_revision",
        "package_lock_sha256",
    )
    _exact_keys(provenance, required=provenance_keys, field="run.provenance")
    _string(provenance["git_commit"], "run.provenance.git_commit", nullable=True)
    _sha256(
        provenance["source_tree_sha256"],
        "run.provenance.source_tree_sha256",
        nullable=True,
    )
    _string(
        provenance["container_digest"], "run.provenance.container_digest", nullable=True
    )
    _sha256(
        provenance["model_snapshot_sha256"],
        "run.provenance.model_snapshot_sha256",
        nullable=True,
    )
    _string(
        provenance["tokenizer_revision"],
        "run.provenance.tokenizer_revision",
        nullable=True,
    )
    _sha256(
        provenance["package_lock_sha256"],
        "run.provenance.package_lock_sha256",
        nullable=True,
    )


def build_agent_run_record(
    *, contract_version: str = "v2.2", **fields: Any
) -> dict[str, Any]:
    """Build and validate a run record without inventing missing provenance."""

    record = {
        "agent_run_record_version": contract_version,
        **copy.deepcopy(fields),
    }
    validate_agent_run_record(record)
    return record


def validate_failure(failure: Mapping[str, Any]) -> None:
    value = _mapping(failure, "failure")
    _exact_keys(
        value,
        required=(
            "failure_version",
            "failure_class",
            "code",
            "stage",
            "retryable",
            "sanitized_detail",
        ),
        field="failure",
    )
    if value["failure_version"] not in {FAILURE_VERSION, FAILURE_VERSION_V23}:
        _fail(
            "UNSUPPORTED_VERSION",
            "failure.failure_version",
            "must equal v2.2 or v2.3",
        )
    for key in ("failure_class", "code", "stage", "sanitized_detail"):
        _string(value[key], f"failure.{key}")
    if len(value["sanitized_detail"]) > 512:
        _fail(
            "STRING_TOO_LONG",
            "failure.sanitized_detail",
            "must contain at most 512 characters",
        )
    _boolean(value["retryable"], "failure.retryable")
    try:
        validate_failure_binding(
            code=value["code"],
            failure_class=value["failure_class"],
            stage=value["stage"],
            retryable=value["retryable"],
        )
    except ValueError as exc:
        _fail("FAILURE_BINDING_INVALID", "failure", str(exc))


def build_failure(
    *,
    code: str,
    stage: str,
    sanitized_detail: str,
    contract_version: str = "v2.2",
) -> dict[str, Any]:
    spec = failure_spec(code)
    failure = {
        "failure_version": contract_version,
        "failure_class": spec.failure_class,
        "code": code,
        "stage": stage,
        "retryable": spec.retryable,
        "sanitized_detail": sanitized_detail,
    }
    validate_failure(failure)
    return failure


def validate_agent_case_result(case_result: Mapping[str, Any]) -> None:
    value = _mapping(case_result, "case_result")
    _exact_keys(
        value,
        required=(
            "agent_case_result_version",
            "task_id",
            "outcome",
            "plan_proposal",
            "final_proposal",
            "tool_observation",
            "run_record",
            "failure",
        ),
        field="case_result",
    )
    if value["agent_case_result_version"] not in {
        AGENT_CASE_RESULT_VERSION,
        AGENT_CASE_RESULT_VERSION_V23,
    }:
        _fail(
            "UNSUPPORTED_VERSION",
            "case_result.agent_case_result_version",
            "must equal v2.2 or v2.3",
        )
    task_id = _string(value["task_id"], "case_result.task_id")
    outcome = _enum(value["outcome"], TERMINAL_OUTCOMES, "case_result.outcome")
    for key in ("plan_proposal", "final_proposal"):
        proposal = value[key]
        if proposal is not None:
            validate_agent_proposal(_mapping(proposal, f"case_result.{key}"))
            if proposal["task_id"] != task_id:
                _fail(
                    "TASK_MISMATCH",
                    f"case_result.{key}.task_id",
                    "must match case_result.task_id",
                )
    if value["tool_observation"] is not None:
        observation = _mapping(
            value["tool_observation"], "case_result.tool_observation"
        )
        validate_tool_observation(observation)
        if observation["task_id"] != task_id:
            _fail(
                "TASK_MISMATCH",
                "case_result.tool_observation.task_id",
                "must match case_result.task_id",
            )
    run_record = _mapping(value["run_record"], "case_result.run_record")
    validate_agent_run_record(run_record)
    if value["agent_case_result_version"] != run_record["agent_run_record_version"]:
        _fail(
            "CASE_RESULT_BINDING_INVALID",
            "case_result.agent_case_result_version",
            "must match run_record.agent_run_record_version",
        )
    if run_record["task_id"] != task_id or run_record["outcome"] != outcome:
        _fail(
            "CASE_RESULT_BINDING_INVALID",
            "case_result.run_record",
            "task_id and outcome must match the outer case result",
        )
    failure = value["failure"]
    if outcome in {"RELEASED", "SAFE_REFUSAL"}:
        if failure is not None:
            _fail(
                "CASE_RESULT_BINDING_INVALID",
                "case_result.failure",
                "released outcomes cannot contain a failure",
            )
        final_proposal = value["final_proposal"]
        expected_action = "ANSWER" if outcome == "RELEASED" else "REFUSE"
        if final_proposal is None or final_proposal["action"] != expected_action:
            _fail(
                "CASE_RESULT_BINDING_INVALID",
                "case_result.final_proposal",
                f"{outcome} requires a final {expected_action} proposal",
            )
    else:
        failure_mapping = _mapping(failure, "case_result.failure")
        validate_failure(failure_mapping)
        if failure_mapping["failure_class"] != outcome:
            _fail(
                "CASE_RESULT_BINDING_INVALID",
                "case_result.failure.failure_class",
                "must equal case_result.outcome",
            )
