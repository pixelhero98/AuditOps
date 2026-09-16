"""Closed failure registry for terminal AuditOps text-agent v2 results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

FAILURE_REGISTRY_VERSION = "v2.2"

MODEL_FAILURE = "MODEL_FAILURE"
INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
FAILURE_CLASSES = frozenset({MODEL_FAILURE, INFRASTRUCTURE_FAILURE, INTEGRITY_FAILURE})

TERMINAL_OUTCOMES = frozenset(
    {
        "RELEASED",
        "SAFE_REFUSAL",
        MODEL_FAILURE,
        INFRASTRUCTURE_FAILURE,
        INTEGRITY_FAILURE,
    }
)


@dataclass(frozen=True, slots=True)
class FailureSpec:
    """Deterministic classification for one externally serialized failure code."""

    code: str
    failure_class: str
    stages: frozenset[str]
    retryable: bool


def _spec(
    code: str, failure_class: str, stages: tuple[str, ...], retryable: bool = False
) -> FailureSpec:
    return FailureSpec(code, failure_class, frozenset(stages), retryable)


_FAILURE_SPECS = {
    spec.code: spec
    for spec in (
        _spec(
            "MODEL_OUTPUT_NOT_STRICT_JSON",
            MODEL_FAILURE,
            ("plan", "synthesis", "direct", "repair"),
        ),
        _spec(
            "MODEL_OUTPUT_SCHEMA_INVALID",
            MODEL_FAILURE,
            ("plan", "synthesis", "direct", "repair"),
        ),
        _spec(
            "ACTION_SHAPE_INVALID",
            MODEL_FAILURE,
            ("plan", "synthesis", "direct", "repair"),
        ),
        _spec(
            "CANONICAL_FORMAT_INVALID",
            MODEL_FAILURE,
            ("synthesis", "direct", "repair"),
        ),
        _spec(
            "PROPOSAL_REJECTED",
            MODEL_FAILURE,
            ("plan", "synthesis", "direct", "repair", "verification"),
        ),
        _spec(
            "REPAIR_NOT_SAFE",
            MODEL_FAILURE,
            ("plan", "synthesis", "direct", "verification"),
        ),
        _spec(
            "REPAIR_EXHAUSTED",
            MODEL_FAILURE,
            ("repair", "verification"),
        ),
        _spec(
            "PROMPT_INJECTION_POLICY_VIOLATION",
            MODEL_FAILURE,
            ("plan", "synthesis", "direct", "repair", "verification"),
        ),
        _spec("CONTEXT_OVERFLOW", INFRASTRUCTURE_FAILURE, ("context",)),
        _spec(
            "MODEL_INITIALIZATION_FAILED",
            INFRASTRUCTURE_FAILURE,
            ("context", "model"),
            True,
        ),
        _spec(
            "MODEL_ADAPTER_FAILED",
            INFRASTRUCTURE_FAILURE,
            ("plan", "synthesis", "direct", "repair", "model"),
            True,
        ),
        _spec(
            "MODEL_TIMEOUT",
            INFRASTRUCTURE_FAILURE,
            ("context", "plan", "synthesis", "direct", "repair", "model"),
            True,
        ),
        _spec(
            "MODEL_OOM",
            INFRASTRUCTURE_FAILURE,
            ("context", "plan", "synthesis", "direct", "repair", "model"),
            True,
        ),
        _spec("BACKEND_FAILURE", INFRASTRUCTURE_FAILURE, ("context", "model"), True),
        _spec("TOOL_EXECUTION_FAILED", INFRASTRUCTURE_FAILURE, ("tool",), True),
        _spec(
            "ARTIFACT_HASH_MISMATCH",
            INTEGRITY_FAILURE,
            ("artifact", "input", "context", "output", "replay", "report"),
        ),
        _spec(
            "TASK_BINDING_MISMATCH",
            INTEGRITY_FAILURE,
            ("plan", "synthesis", "direct", "repair", "verification", "replay"),
        ),
        _spec(
            "TOOL_ARGUMENT_MISMATCH",
            INTEGRITY_FAILURE,
            ("plan", "tool", "verification", "replay"),
        ),
        _spec(
            "TOOL_OBSERVATION_MISMATCH",
            INTEGRITY_FAILURE,
            ("tool", "verification", "replay"),
        ),
        _spec(
            "DETERMINISTIC_REPLAY_MISMATCH",
            INTEGRITY_FAILURE,
            ("replay",),
        ),
        _spec(
            "STRUCTURED_OUTPUT_CONTRACT_BROKEN",
            INTEGRITY_FAILURE,
            ("plan", "synthesis", "direct", "repair"),
        ),
        _spec("OVERWRITE_DETECTED", INTEGRITY_FAILURE, ("artifact", "report")),
    )
}

FAILURE_SPECS: Mapping[str, FailureSpec] = MappingProxyType(_FAILURE_SPECS)


def failure_spec(code: str) -> FailureSpec:
    """Return a registered failure classification or fail closed."""

    try:
        return FAILURE_SPECS[code]
    except KeyError as exc:
        raise ValueError(f"Unregistered AuditOps v2 failure code: {code!r}") from exc


def validate_failure_binding(
    *, code: str, failure_class: str, stage: str, retryable: bool
) -> FailureSpec:
    """Validate serialized classification fields against the closed registry."""

    spec = failure_spec(code)
    if failure_class != spec.failure_class:
        raise ValueError(
            f"failure_class for {code} must be {spec.failure_class}, got {failure_class}"
        )
    if stage not in spec.stages:
        raise ValueError(
            f"stage for {code} must be one of {sorted(spec.stages)}, got {stage!r}"
        )
    if retryable is not spec.retryable:
        raise ValueError(f"retryable for {code} must be {spec.retryable}")
    return spec


__all__ = [
    "FAILURE_CLASSES",
    "FAILURE_REGISTRY_VERSION",
    "FAILURE_SPECS",
    "INFRASTRUCTURE_FAILURE",
    "INTEGRITY_FAILURE",
    "MODEL_FAILURE",
    "TERMINAL_OUTCOMES",
    "FailureSpec",
    "failure_spec",
    "validate_failure_binding",
]
