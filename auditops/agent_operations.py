"""Authoritative operation registries for AuditOps text-agent contracts.

This module intentionally has no dependency on the v1 contracts or runtime.  Every
v2 consumer uses the same task-to-operation binding and the same exact deterministic
tool arguments.  The registry contains semantic identifiers; deployment-local paths
and evaluator targets never belong here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .canonical_json import canonical_json_sha256

AGENT_OPERATION_REGISTRY_VERSION = "v2.2"
TEXT_RUNTIME_ID = "auditops.agent_runtime.v2.2"
AGENT_OPERATION_REGISTRY_VERSION_V23 = "v2.3"
TEXT_RUNTIME_ID_V23 = "auditops.agent_runtime.v2.3"

QUANT_TASK_TYPE = "quant_metric"
NARRATIVE_TASK_TYPE = "narrative_citation"

QUANT_OPERATION = "evaluate_metric_spec"
NARRATIVE_OPERATION = "load_frozen_evidence"

QUANT_TERMINAL_SCHEMA_ID = "quantitative_answer_or_refusal.v2.2"
NARRATIVE_TERMINAL_SCHEMA_ID = "narrative_selection_or_refusal.v2.2"
QUANT_TERMINAL_SCHEMA_ID_V23 = "quantitative_answer_or_refusal.v2.3"
NARRATIVE_TERMINAL_SCHEMA_ID_V23 = "narrative_selection_or_refusal.v2.3"

NARRATIVE_SUBTYPES = frozenset(
    {
        "footnote_note",
        "accounting_policy",
        "auditor_report_opinion_language",
        "critical_audit_matter",
    }
)

TEXT_TASK_TYPES = frozenset({QUANT_TASK_TYPE, NARRATIVE_TASK_TYPE})
REGISTERED_OPERATIONS = frozenset({QUANT_OPERATION, NARRATIVE_OPERATION})

QUANT_REFUSAL_CODES = frozenset(
    {
        "AMBIGUOUS_CONTEXT",
        "DIVISION_BY_ZERO",
        "INCOMPATIBLE_UNITS",
        "MISSING_INPUT",
        "PERIOD_NOT_SUPPORTED",
        "PROMPT_INJECTION_DETECTED",
        "UNSUPPORTED_REQUEST",
        "ZERO_DENOMINATOR",
    }
)
NARRATIVE_REFUSAL_CODES = frozenset(
    {
        "NARRATIVE_CITATION_MISS",
        "NARRATIVE_NOT_SUPPORTED",
        "PROMPT_INJECTION_DETECTED",
        "TASK_NOT_SUPPORTED",
        "UNSUPPORTED_REQUEST",
    }
)
NARRATIVE_REFUSAL_CODES_V23 = frozenset(
    {
        "NO_DIRECT_EVIDENCE",
        "PROMPT_INJECTION_DETECTED",
        "TASK_NOT_SUPPORTED",
        "UNSUPPORTED_REQUEST",
    }
)


@dataclass(frozen=True, slots=True)
class NarrativeSelectionSpec:
    """Non-gold extract cardinality policy for one narrative subtype."""

    subtype: str
    min_extracts: int
    max_extracts: int
    selection_rule: str

    def as_contract(self) -> dict[str, Any]:
        material = {
            "narrative_selection_policy_version": "v2.3",
            "subtype": self.subtype,
            "min_extracts": self.min_extracts,
            "max_extracts": self.max_extracts,
            "selection_rule": self.selection_rule,
        }
        return {**material, "policy_sha256": canonical_json_sha256(material)}


_NARRATIVE_SELECTION_SPECS = {
    subtype: NarrativeSelectionSpec(
        subtype=subtype,
        min_extracts=1,
        max_extracts=3 if subtype == "critical_audit_matter" else 1,
        selection_rule=(
            "SELECT_SMALLEST_DIRECT_EXTRACT_SET;MULTIPLE_ONLY_FOR_DISTINCT_CAM_COMPONENTS"
            if subtype == "critical_audit_matter"
            else "SELECT_ONE_MINIMAL_DIRECT_EXTRACT"
        ),
    )
    for subtype in NARRATIVE_SUBTYPES
}
NARRATIVE_SELECTION_SPECS: Mapping[str, NarrativeSelectionSpec] = MappingProxyType(
    _NARRATIVE_SELECTION_SPECS
)


def _required_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _required_value(value: Mapping[str, Any], key: str, field: str) -> Any:
    if key not in value:
        raise ValueError(f"{field}.{key} is required")
    result = value[key]
    if result is None or (isinstance(result, str) and not result.strip()):
        raise ValueError(f"{field}.{key} must be non-null and non-empty")
    return result


def _quant_arguments(task_input: Mapping[str, Any]) -> dict[str, Any]:
    filing = _required_mapping(task_input.get("filing"), "task.filing")
    parameters = _required_mapping(
        task_input.get("task_parameters"), "task.task_parameters"
    )
    period = _required_mapping(task_input.get("period"), "task.period")
    return {
        "filing_id": _required_value(filing, "filing_id", "task.filing"),
        "metric_spec_id": _required_value(
            parameters, "metric_spec_id", "task.task_parameters"
        ),
        "period_key": _required_value(period, "period_key", "task.period"),
    }


def _narrative_arguments(task_input: Mapping[str, Any]) -> dict[str, Any]:
    filing = _required_mapping(task_input.get("filing"), "task.filing")
    evidence_scope = _required_mapping(
        task_input.get("evidence_scope"), "task.evidence_scope"
    )
    return {
        "filing_id": _required_value(filing, "filing_id", "task.filing"),
        "evidence_scope_sha256": canonical_json_sha256(evidence_scope),
    }


@dataclass(frozen=True, slots=True)
class TaskOperationSpec:
    """One complete, immutable text-task routing rule."""

    task_type: str
    runtime_id: str
    operation: str
    tool_argument_schema_id: str
    observation_schema_id: str
    terminal_output_schema_id: str
    permitted_refusal_codes: frozenset[str]
    _argument_builder: Callable[[Mapping[str, Any]], dict[str, Any]]

    def expected_arguments(self, task_input: Mapping[str, Any]) -> dict[str, Any]:
        actual_task_type = task_input.get("task_type")
        if actual_task_type != self.task_type:
            raise ValueError(
                f"task.task_type must be {self.task_type!r}, got {actual_task_type!r}"
            )
        return self._argument_builder(task_input)

    def arguments_match(
        self, task_input: Mapping[str, Any], proposed: Mapping[str, Any]
    ) -> bool:
        return isinstance(proposed, Mapping) and dict(
            proposed
        ) == self.expected_arguments(task_input)


_TASK_OPERATION_SPECS = {
    QUANT_TASK_TYPE: TaskOperationSpec(
        task_type=QUANT_TASK_TYPE,
        runtime_id=TEXT_RUNTIME_ID,
        operation=QUANT_OPERATION,
        tool_argument_schema_id="tool_arguments.v2.2#/$defs/metric",
        observation_schema_id="tool_observation.v2.2#/$defs/metric",
        terminal_output_schema_id=QUANT_TERMINAL_SCHEMA_ID,
        permitted_refusal_codes=QUANT_REFUSAL_CODES,
        _argument_builder=_quant_arguments,
    ),
    NARRATIVE_TASK_TYPE: TaskOperationSpec(
        task_type=NARRATIVE_TASK_TYPE,
        runtime_id=TEXT_RUNTIME_ID,
        operation=NARRATIVE_OPERATION,
        tool_argument_schema_id="tool_arguments.v2.2#/$defs/frozenEvidence",
        observation_schema_id="tool_observation.v2.2#/$defs/frozenEvidence",
        terminal_output_schema_id=NARRATIVE_TERMINAL_SCHEMA_ID,
        permitted_refusal_codes=NARRATIVE_REFUSAL_CODES,
        _argument_builder=_narrative_arguments,
    ),
}

_TASK_OPERATION_SPECS_V23 = {
    QUANT_TASK_TYPE: TaskOperationSpec(
        task_type=QUANT_TASK_TYPE,
        runtime_id=TEXT_RUNTIME_ID_V23,
        operation=QUANT_OPERATION,
        tool_argument_schema_id="tool_arguments.v2.3#/$defs/metric",
        observation_schema_id="tool_observation.v2.3#/$defs/metric",
        terminal_output_schema_id=QUANT_TERMINAL_SCHEMA_ID_V23,
        permitted_refusal_codes=QUANT_REFUSAL_CODES,
        _argument_builder=_quant_arguments,
    ),
    NARRATIVE_TASK_TYPE: TaskOperationSpec(
        task_type=NARRATIVE_TASK_TYPE,
        runtime_id=TEXT_RUNTIME_ID_V23,
        operation=NARRATIVE_OPERATION,
        tool_argument_schema_id="tool_arguments.v2.3#/$defs/frozenEvidence",
        observation_schema_id="tool_observation.v2.3#/$defs/frozenEvidence",
        terminal_output_schema_id=NARRATIVE_TERMINAL_SCHEMA_ID_V23,
        permitted_refusal_codes=NARRATIVE_REFUSAL_CODES_V23,
        _argument_builder=_narrative_arguments,
    ),
}

TASK_OPERATION_SPECS: Mapping[str, TaskOperationSpec] = MappingProxyType(
    _TASK_OPERATION_SPECS
)
OPERATION_TASK_SPECS: Mapping[str, TaskOperationSpec] = MappingProxyType(
    {spec.operation: spec for spec in _TASK_OPERATION_SPECS.values()}
)
TASK_OPERATION_SPECS_V23: Mapping[str, TaskOperationSpec] = MappingProxyType(
    _TASK_OPERATION_SPECS_V23
)
OPERATION_TASK_SPECS_V23: Mapping[str, TaskOperationSpec] = MappingProxyType(
    {spec.operation: spec for spec in _TASK_OPERATION_SPECS_V23.values()}
)


def _task_contract_version(task_input: Mapping[str, Any]) -> str:
    version = task_input.get("agent_task_input_version", "v2.2")
    if version not in {"v2.2", "v2.3"}:
        raise ValueError(f"Unsupported text-agent contract version: {version!r}")
    return str(version)


def task_operation_spec(task_type: str, *, version: str = "v2.2") -> TaskOperationSpec:
    """Return the registered rule or fail closed for non-text/unknown task types."""

    try:
        registry = (
            TASK_OPERATION_SPECS_V23 if version == "v2.3" else TASK_OPERATION_SPECS
        )
        if version not in {"v2.2", "v2.3"}:
            raise ValueError(f"Unsupported text-agent contract version: {version!r}")
        return registry[task_type]
    except KeyError as exc:
        raise ValueError(f"Unregistered text-agent task type: {task_type!r}") from exc


def operation_spec(operation: str, *, version: str = "v2.2") -> TaskOperationSpec:
    """Return the registered rule for an allowed deterministic operation."""

    try:
        registry = (
            OPERATION_TASK_SPECS_V23 if version == "v2.3" else OPERATION_TASK_SPECS
        )
        if version not in {"v2.2", "v2.3"}:
            raise ValueError(f"Unsupported text-agent contract version: {version!r}")
        return registry[operation]
    except KeyError as exc:
        raise ValueError(f"Unregistered text-agent operation: {operation!r}") from exc


def expected_tool_arguments(task_input: Mapping[str, Any]) -> dict[str, Any]:
    """Build the one exact model-visible tool argument object for a v2 task."""

    return task_operation_spec(
        str(task_input.get("task_type")), version=_task_contract_version(task_input)
    ).expected_arguments(task_input)


def validate_task_operation_binding(task_input: Mapping[str, Any]) -> TaskOperationSpec:
    """Validate exact tool and terminal-schema bindings declared by a v2 task."""

    spec = task_operation_spec(
        str(task_input.get("task_type")), version=_task_contract_version(task_input)
    )
    if task_input.get("allowed_tools") != [spec.operation]:
        raise ValueError(
            f"task.allowed_tools must be exactly [{spec.operation!r}] for {spec.task_type}"
        )
    if task_input.get("output_schema_id") != spec.terminal_output_schema_id:
        raise ValueError(
            "task.output_schema_id must equal "
            f"{spec.terminal_output_schema_id!r} for {spec.task_type}"
        )
    policy = _required_mapping(task_input.get("refusal_policy"), "task.refusal_policy")
    allowed_codes = policy.get("allowed_codes")
    if not isinstance(allowed_codes, (list, tuple)) or not allowed_codes:
        raise ValueError("task.refusal_policy.allowed_codes must be a non-empty array")
    if len(allowed_codes) != len(set(allowed_codes)):
        raise ValueError("task.refusal_policy.allowed_codes must be unique")
    unknown = sorted(set(allowed_codes) - spec.permitted_refusal_codes)
    if unknown:
        raise ValueError(
            "task.refusal_policy.allowed_codes contains codes not permitted for "
            f"{spec.task_type}: {', '.join(unknown)}"
        )
    if "PROMPT_INJECTION_DETECTED" not in allowed_codes:
        raise ValueError(
            "task.refusal_policy.allowed_codes must permit PROMPT_INJECTION_DETECTED"
        )
    return spec


def narrative_selection_spec(subtype: str) -> NarrativeSelectionSpec:
    try:
        return NARRATIVE_SELECTION_SPECS[subtype]
    except KeyError as exc:
        raise ValueError(f"Unregistered narrative subtype: {subtype!r}") from exc


def narrative_selection_contract(subtype: str) -> dict[str, Any]:
    return narrative_selection_spec(subtype).as_contract()


def operation_registry_version(task_input: Mapping[str, Any]) -> str:
    return (
        AGENT_OPERATION_REGISTRY_VERSION_V23
        if _task_contract_version(task_input) == "v2.3"
        else AGENT_OPERATION_REGISTRY_VERSION
    )


__all__ = [
    "AGENT_OPERATION_REGISTRY_VERSION",
    "AGENT_OPERATION_REGISTRY_VERSION_V23",
    "NARRATIVE_OPERATION",
    "NARRATIVE_REFUSAL_CODES",
    "NARRATIVE_REFUSAL_CODES_V23",
    "NARRATIVE_SELECTION_SPECS",
    "NARRATIVE_SUBTYPES",
    "NARRATIVE_TASK_TYPE",
    "NARRATIVE_TERMINAL_SCHEMA_ID",
    "NARRATIVE_TERMINAL_SCHEMA_ID_V23",
    "OPERATION_TASK_SPECS",
    "QUANT_OPERATION",
    "QUANT_REFUSAL_CODES",
    "QUANT_TASK_TYPE",
    "QUANT_TERMINAL_SCHEMA_ID",
    "QUANT_TERMINAL_SCHEMA_ID_V23",
    "REGISTERED_OPERATIONS",
    "TASK_OPERATION_SPECS",
    "TASK_OPERATION_SPECS_V23",
    "TEXT_RUNTIME_ID",
    "TEXT_RUNTIME_ID_V23",
    "TEXT_TASK_TYPES",
    "NarrativeSelectionSpec",
    "TaskOperationSpec",
    "expected_tool_arguments",
    "narrative_selection_contract",
    "narrative_selection_spec",
    "operation_registry_version",
    "operation_spec",
    "task_operation_spec",
    "validate_task_operation_binding",
]
