from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .agent_context import scan_prompt_injection
from .agent_contracts import (
    MAX_REPAIR_ATTEMPTS,
    VERIFIER_RESULT_VERSION,
    ContractValidationError,
    validate_agent_proposal,
    validate_agent_task_input,
    validate_tool_observation,
    validate_verifier_result,
)
from .agent_operations import expected_tool_arguments, task_operation_spec
from .agent_tools import evaluate_quant_evidence
from .text_support import is_contiguous_text_supported, normalize_support_text

PROMPT_INJECTION_REFUSAL_CODE = "PROMPT_INJECTION_DETECTED"
_REPAIRABLE_CODES_BY_STAGE = {
    "plan": frozenset(
        {"MODEL_OUTPUT_NOT_STRICT_JSON", "SCHEMA_VALID", "UNEXPECTED_STAGE_ACTION"}
    ),
    "direct": frozenset(
        {
            "ANSWER_SHAPE_VALID",
            "MODEL_OUTPUT_NOT_STRICT_JSON",
            "SCHEMA_VALID",
            "UNEXPECTED_STAGE_ACTION",
        }
    ),
    "synthesis": frozenset(
        {
            "ANSWER_SHAPE_VALID",
            "MODEL_OUTPUT_NOT_STRICT_JSON",
            "SCHEMA_VALID",
            "UNEXPECTED_STAGE_ACTION",
        }
    ),
}

_DISCARDED_PROPOSAL_ERRORS = {
    "SCHEMA_VALID": (
        "proposal",
        "proposal did not satisfy the AgentProposalV2 contract and was discarded",
    ),
    "TASK_MATCH": (
        "proposal.task_id",
        "proposal task_id did not match the active task and was discarded",
    ),
}


def _check(
    checks: list[dict[str, Any]],
    code: str,
    passed: bool,
    field: str | None,
    success_message: str,
    failure_message: str,
) -> None:
    checks.append(
        {
            "code": code,
            "passed": bool(passed),
            "field": field,
            "message": success_message if passed else failure_message,
        }
    )


def _result(
    task_id: str,
    checks: Sequence[Mapping[str, Any]],
    action: str | None,
    repair_count: int,
    *,
    stage: str,
    force_nonrepairable: bool = False,
) -> dict[str, Any]:
    if stage not in _REPAIRABLE_CODES_BY_STAGE:
        raise ValueError("stage must be plan, direct, or synthesis")
    failed = [check for check in checks if not check["passed"]]
    repair_errors = [
        {
            "code": check["code"],
            "field": check["field"],
            "message": check["message"],
        }
        for check in failed
        if check["code"] in _REPAIRABLE_CODES_BY_STAGE[stage]
    ]
    passed = not failed
    if failed:
        repair_allowed = bool(
            repair_errors
            and not force_nonrepairable
            and repair_count < MAX_REPAIR_ATTEMPTS
        )
        disposition = "REPAIR_REQUIRED" if repair_allowed else "FAILED"
        release_allowed = False
    elif action == "CALL_TOOL":
        repair_allowed = False
        disposition = "TOOL_CALL_APPROVED"
        release_allowed = False
    elif action == "REFUSE":
        repair_allowed = False
        disposition = "SAFE_REFUSAL"
        release_allowed = True
    else:
        repair_allowed = False
        disposition = "RELEASED"
        release_allowed = True
    result = {
        "verifier_result_version": VERIFIER_RESULT_VERSION,
        "task_id": task_id,
        "passed": passed,
        "release_allowed": release_allowed,
        "disposition": disposition,
        "repair_count": repair_count,
        "repair_allowed": repair_allowed,
        "checks": [dict(check) for check in checks],
        "repair_errors": repair_errors if repair_allowed else [],
    }
    validate_verifier_result(result)
    return result


def safe_repair_errors(verifier_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return only bounded error codes/fields/messages safe to send to a repair prompt."""

    validate_verifier_result(verifier_result)
    return [dict(error) for error in verifier_result["repair_errors"]]


def discarded_proposal_result(
    task_id: str, *, code: str, repair_count: int, stage: str
) -> dict[str, Any]:
    """Build the fixed diagnostic for a parsed payload that is not retained."""

    if code not in _DISCARDED_PROPOSAL_ERRORS:
        raise ValueError("discarded proposal code must be SCHEMA_VALID or TASK_MATCH")
    field, message = _DISCARDED_PROPOSAL_ERRORS[code]
    check = {
        "code": code,
        "passed": False,
        "field": field,
        "message": message,
    }
    return _result(task_id, [check], None, repair_count, stage=stage)


def _evidence_index(
    evidence_items: Sequence[Mapping[str, Any]], allowed_filing_ids: set[str]
) -> tuple[dict[str, Mapping[str, Any]], bool]:
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    valid = True
    for item in evidence_items:
        if not isinstance(item, Mapping):
            valid = False
            continue
        evidence_id = item.get("evidence_id")
        content = item.get("content")
        if (
            not isinstance(evidence_id, str)
            or not evidence_id.strip()
            or not isinstance(content, str)
            or item.get("filing_id") not in allowed_filing_ids
        ):
            valid = False
            continue
        if evidence_id in evidence_by_id:
            valid = False
            continue
        evidence_by_id[evidence_id] = item
    return evidence_by_id, valid


def _prompt_injection_flags(
    task_input: Mapping[str, Any], evidence_items: Sequence[Mapping[str, Any]]
) -> tuple[str, ...]:
    flags = list(scan_prompt_injection(task_input, prefix="TASK"))
    for index, item in enumerate(evidence_items):
        flags.extend(scan_prompt_injection(item, prefix=f"EVIDENCE[{index}]"))
    return tuple(sorted(set(flags)))


def _validate_tool_call(
    task_input: Mapping[str, Any],
    proposal: Mapping[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    tool_name = proposal["tool_name"]
    spec = task_operation_spec(
        task_input["task_type"],
        version=str(task_input.get("agent_task_input_version", "v2.2")),
    )
    tool_allowed = tool_name == spec.operation and task_input["allowed_tools"] == [
        spec.operation
    ]
    _check(
        checks,
        "TOOL_ALLOWED",
        tool_allowed,
        "proposal.tool_name",
        "tool is registered and allowed for this task",
        "tool is not registered or allowed for this task",
    )
    if not tool_allowed:
        return
    arguments = proposal["tool_arguments"]
    expected_arguments = spec.expected_arguments(task_input)
    expected_keys = set(expected_arguments)
    argument_shape_valid = (
        isinstance(arguments, Mapping) and set(arguments) == expected_keys
    )
    _check(
        checks,
        "TOOL_ARGUMENTS_VALID",
        argument_shape_valid,
        "proposal.tool_arguments",
        "tool arguments match the registered interface",
        "tool arguments do not match the registered interface",
    )
    if not argument_shape_valid:
        return

    filing_matches = arguments["filing_id"] == expected_arguments["filing_id"]
    _check(
        checks,
        "FILING_MATCH",
        filing_matches,
        "proposal.tool_arguments.filing_id",
        "tool call targets the task filing",
        "tool call targets a different filing",
    )
    if tool_name == "evaluate_metric_spec":
        metric_matches = (
            arguments["metric_spec_id"] == expected_arguments["metric_spec_id"]
        )
        _check(
            checks,
            "OPERATION_MATCH",
            metric_matches,
            "proposal.tool_arguments.metric_spec_id",
            "tool call uses the task metric",
            "tool call uses a different metric",
        )
        period_matches = arguments["period_key"] == expected_arguments["period_key"]
        _check(
            checks,
            "PERIOD_MATCH",
            period_matches,
            "proposal.tool_arguments.period_key",
            "tool call uses the task period",
            "tool call uses a different period",
        )
    else:
        scope_matches = (
            arguments["evidence_scope_sha256"]
            == expected_arguments["evidence_scope_sha256"]
        )
        _check(
            checks,
            "OPERATION_MATCH",
            scope_matches,
            "proposal.tool_arguments.evidence_scope_sha256",
            "tool call loads the exact frozen evidence scope",
            "tool call changes the frozen evidence scope",
        )


def _observation_field(observation: Mapping[str, Any], key: str) -> Any:
    if key in observation:
        return observation[key]
    if key in {"value", "unit"} and isinstance(observation.get("result"), Mapping):
        return observation["result"].get(key)
    if key == "period_key" and isinstance(observation.get("period"), Mapping):
        return observation["period"].get("period_key")
    if key == "evidence_ids" and "chunk_evidence_ids" in observation:
        return observation["chunk_evidence_ids"]
    return None


def _values_equal(left: Any, right: Any) -> bool:
    return isinstance(left, str) and isinstance(right, str) and left == right


def _units_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return isinstance(left, str) and isinstance(right, str) and left == right


def _evidence_ids_from_observation(observation: Mapping[str, Any]) -> list[str] | None:
    raw = _observation_field(observation, "evidence_ids")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or any(
        not isinstance(item, str) for item in raw
    ):
        return None
    return list(raw)


def _validate_support(
    proposal: Mapping[str, Any],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    *,
    reconstructed_extracts: bool = False,
) -> bool:
    answer_text = proposal["answer_text"]
    if answer_text is None:
        return True
    claims = proposal["claims"]
    if reconstructed_extracts:
        # v2.2 narrative answers are runtime-authored by joining the model's
        # ordered exact extracts.  A multi-extract answer need not itself be a
        # contiguous passage in any one chunk; its deterministic reconstruction
        # and every component extract must instead verify exactly.
        if answer_text != "\n\n".join(claim["text"] for claim in claims):
            return False
    else:
        cited_contents = [
            evidence_by_id[evidence_id]["content"]
            for evidence_id in proposal["evidence_ids"]
            if evidence_id in evidence_by_id
        ]
        if not any(
            is_contiguous_text_supported(answer_text, content)
            for content in cited_contents
        ):
            return False
    proposal_evidence_ids = set(proposal["evidence_ids"])
    for claim in claims:
        claim_text = claim["text"]
        supporting_text = claim["supporting_text"]
        claim_evidence_ids = set(claim["evidence_ids"])
        if (
            not claim_evidence_ids
            or not claim_evidence_ids.issubset(proposal_evidence_ids)
            or normalize_support_text(claim_text)
            != normalize_support_text(supporting_text)
        ):
            return False
        if not any(
            is_contiguous_text_supported(
                claim_text, evidence_by_id[evidence_id]["content"]
            )
            and is_contiguous_text_supported(
                supporting_text, evidence_by_id[evidence_id]["content"]
            )
            for evidence_id in claim_evidence_ids
            if evidence_id in evidence_by_id
        ):
            return False
    return True


def _narrative_selection_shape_valid(
    task_input: Mapping[str, Any], proposal: Mapping[str, Any]
) -> bool:
    claims = proposal.get("claims")
    if not isinstance(claims, (list, tuple)) or not claims:
        return False
    policy = task_input.get("narrative_selection_policy")
    if not isinstance(policy, Mapping):
        return len(claims) <= 3
    minimum = policy.get("min_extracts")
    maximum = policy.get("max_extracts")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not minimum <= len(claims) <= maximum
    ):
        return False
    normalized = [
        normalize_support_text(str(claim.get("text") or "")) for claim in claims
    ]
    if any(not text for text in normalized) or len(normalized) != len(set(normalized)):
        return False
    return not any(
        left != right and left in right for left in normalized for right in normalized
    )


def _validate_final_proposal(
    task_input: Mapping[str, Any],
    proposal: Mapping[str, Any],
    evidence_items: Sequence[Mapping[str, Any]],
    tool_observation: Mapping[str, Any] | None,
    checks: list[dict[str, Any]],
) -> None:
    evidence_by_id, evidence_shape_valid = _evidence_index(
        evidence_items,
        set(
            task_input["evidence_scope"].get(
                "filing_ids", [task_input["filing"]["filing_id"]]
            )
        ),
    )
    scoped_ids = set(task_input["evidence_scope"]["evidence_ids"])
    cited_ids = set(proposal["evidence_ids"])
    evidence_valid = (
        evidence_shape_valid
        and cited_ids.issubset(scoped_ids)
        and cited_ids.issubset(evidence_by_id)
    )
    _check(
        checks,
        "EVIDENCE_VALID",
        evidence_valid,
        "proposal.evidence_ids",
        "all cited evidence exists within the task scope",
        "one or more cited evidence IDs are invalid or outside the task scope",
    )

    period_valid = proposal["period_key"] in {None, task_input["period"]["period_key"]}
    if proposal["action"] == "ANSWER":
        period_valid = proposal["period_key"] == task_input["period"]["period_key"]
    _check(
        checks,
        "PERIOD_MATCH",
        period_valid,
        "proposal.period_key",
        "proposal period matches the task",
        "proposal period does not match the task",
    )

    injection_flags = _prompt_injection_flags(task_input, evidence_items)
    injection_refusal = (
        proposal["action"] == "REFUSE"
        and proposal["refusal_code"] == PROMPT_INJECTION_REFUSAL_CODE
    )
    injection_safe = not injection_flags or injection_refusal
    _check(
        checks,
        "PROMPT_INJECTION_CLEAR",
        injection_safe,
        "proposal",
        "no hostile embedded instruction can influence a released answer",
        "hostile embedded instructions require a prompt-injection refusal",
    )

    if injection_flags and injection_refusal:
        refusal_allowed = (
            PROMPT_INJECTION_REFUSAL_CODE
            in task_input["refusal_policy"]["allowed_codes"]
        )
        _check(
            checks,
            "REFUSAL_VALID",
            refusal_allowed,
            "proposal.refusal_code",
            "prompt-injection refusal is permitted",
            "prompt-injection refusal is not permitted by the task policy",
        )
        return

    observation_valid = isinstance(tool_observation, Mapping)
    _check(
        checks,
        "TOOL_OBSERVATION_PRESENT",
        observation_valid,
        "tool_observation",
        "a deterministic tool observation is available",
        "final proposals require a deterministic tool observation",
    )
    if not observation_valid:
        return
    assert tool_observation is not None

    try:
        validate_tool_observation(tool_observation)
        observation_contract_valid = True
    except ContractValidationError:
        observation_contract_valid = False

    observation_task_id = tool_observation.get("task_id")
    observation_task_valid = observation_task_id == task_input["task_id"]
    observation_filing = tool_observation.get("filing_id")
    observation_filing_valid = observation_filing == task_input["filing"]["filing_id"]
    expected_observation_kind = (
        "METRIC" if task_input["task_type"] == "quant_metric" else "FROZEN_EVIDENCE"
    )
    observation_envelope_valid = (
        observation_contract_valid
        and tool_observation.get("tool_observation_version")
        == task_input.get("agent_task_input_version", "v2.2")
        and tool_observation.get("observation_kind") == expected_observation_kind
        and tool_observation.get("visibility") in {"MODEL_VISIBLE", "VERIFIER_ONLY"}
    )
    observation_operation = tool_observation.get("metric_spec_id")
    operation_valid = (
        observation_operation == task_input["task_parameters"]["metric_spec_id"]
        if task_input["task_type"] == "quant_metric"
        else observation_operation is None
        and tool_observation.get("evidence_scope_sha256")
        == expected_tool_arguments(task_input)["evidence_scope_sha256"]
    )
    _check(
        checks,
        "OBSERVATION_SCOPE_VALID",
        observation_envelope_valid
        and observation_task_valid
        and observation_filing_valid
        and operation_valid,
        "tool_observation",
        "tool observation belongs to this task and filing",
        "tool observation belongs to a different task, filing, or operation",
    )

    task_type = task_input["task_type"]
    observation_status = tool_observation.get("status")
    retrieval_status = tool_observation.get("retrieval_status")
    observation_period = _observation_field(tool_observation, "period_key")
    observation_period_matches = (
        observation_period == task_input["period"]["period_key"]
    )
    _check(
        checks,
        "OBSERVATION_PERIOD_MATCH",
        observation_period_matches,
        "tool_observation.period_key",
        "tool observation uses the task period",
        "tool observation uses a different period",
    )

    quant_evaluation = (
        evaluate_quant_evidence(task_input, evidence_items)
        if task_type == "quant_metric"
        else None
    )
    observation_evidence_ids = _evidence_ids_from_observation(tool_observation)
    if quant_evaluation is not None:
        status_matches = (
            proposal["action"] == "ANSWER" and observation_status == "OK"
        ) or (proposal["action"] == "REFUSE" and observation_status == "REFUSAL")
        expected_evidence_ids = (
            list(quant_evaluation["evidence_ids"])
            if quant_evaluation["status"] == "OK"
            else observation_evidence_ids
        )
        citations_match = (
            expected_evidence_ids is not None
            and set(expected_evidence_ids) == cited_ids
        )
    else:
        # Narrative release is deliberately independent of evaluator gold.  The
        # deterministic observation proves only the frozen retrieval scope; the
        # verifier then gates claims by verbatim support in the cited excerpts.
        status_matches = (
            retrieval_status == "SCOPE_LOADED"
            and tool_observation.get("answerability_assessed") is False
        )
        citations_match = (
            proposal["action"] == "REFUSE"
            and not cited_ids
            or proposal["action"] == "ANSWER"
            and bool(cited_ids)
            and observation_evidence_ids is not None
            and cited_ids.issubset(set(observation_evidence_ids))
        )
    _check(
        checks,
        "STATUS_MATCH",
        status_matches,
        (
            "tool_observation.status"
            if task_type == "quant_metric"
            else "tool_observation.retrieval_status"
        ),
        "deterministic observation has the expected task-specific status",
        "deterministic observation has an invalid task-specific status",
    )
    _check(
        checks,
        "CITATIONS_MATCH",
        citations_match,
        "proposal.evidence_ids",
        "citations match the deterministic observation",
        "citations do not match the deterministic observation",
    )

    if quant_evaluation is not None:
        observation_evidence_valid = (
            observation_status == quant_evaluation["status"]
            and observation_period == quant_evaluation["period_key"]
        )
        if observation_status == "OK":
            observation_evidence_valid = (
                observation_evidence_valid
                and _values_equal(
                    _observation_field(tool_observation, "value"),
                    quant_evaluation["value"],
                )
                and _units_equal(
                    _observation_field(tool_observation, "unit"),
                    quant_evaluation["unit"],
                )
                and observation_evidence_ids is not None
                and set(observation_evidence_ids)
                == set(quant_evaluation["evidence_ids"])
            )
        _check(
            checks,
            "OBSERVATION_EVIDENCE_VALID",
            observation_evidence_valid,
            "tool_observation",
            "deterministic observation is independently reproduced from cited evidence",
            "deterministic observation conflicts with the frozen evidence or formula",
        )

    if proposal["action"] == "REFUSE":
        allowed_codes = task_input["refusal_policy"]["allowed_codes"]
        observation_refusal_code = tool_observation.get("refusal_code")
        if task_type == "quant_metric":
            refusal_valid = (
                proposal["refusal_code"] in allowed_codes
                and observation_status == "REFUSAL"
                and proposal["refusal_code"] == observation_refusal_code
            )
            refusal_success = (
                "refusal code is allowed and matches the deterministic calculation"
            )
            refusal_failure = "refusal is unsupported or uses an incorrect code"
        else:
            # A supported narrative answer is checked extractively below.  A
            # narrative refusal is safe to release when it uses the bounded
            # policy and the independently reproduced retrieval scope; whether
            # it was an over-refusal remains an evaluator-only accuracy metric.
            refusal_valid = (
                proposal["refusal_code"] in allowed_codes
                and retrieval_status == "SCOPE_LOADED"
                and tool_observation.get("answerability_assessed") is False
                and observation_period_matches
            )
            refusal_success = "refusal code is allowed for the frozen retrieval scope"
            refusal_failure = "refusal code or retrieval scope is invalid"
        _check(
            checks,
            "REFUSAL_VALID",
            refusal_valid,
            "proposal.refusal_code",
            refusal_success,
            refusal_failure,
        )
        return

    if task_type == "quant_metric":
        quant_shape_valid = (
            proposal["value"] is not None
            and proposal["answer_text"] is None
            and not proposal["claims"]
        )
        _check(
            checks,
            "ANSWER_SHAPE_VALID",
            quant_shape_valid,
            "proposal",
            "quantitative answer uses the value field",
            "quantitative answer must use value and cannot include narrative text or claims",
        )
        assert quant_evaluation is not None
        observation_value = _observation_field(tool_observation, "value")
        value_matches = (
            observation_status == "OK"
            and quant_evaluation["status"] == "OK"
            and _values_equal(proposal["value"], observation_value)
        )
        _check(
            checks,
            "VALUE_MATCH",
            quant_evaluation["status"] == "OK"
            and _values_equal(proposal["value"], quant_evaluation["value"]),
            "proposal.value",
            "value matches the deterministic executor",
            "value does not match the deterministic executor",
        )
        _check(
            checks,
            "ARITHMETIC_VALID",
            value_matches,
            "proposal.value",
            "arithmetic result matches the deterministic executor",
            "arithmetic result is not supported by the deterministic executor",
        )
        unit_matches = quant_evaluation["status"] == "OK" and _units_equal(
            proposal["unit"], quant_evaluation["unit"]
        )
        _check(
            checks,
            "UNIT_MATCH",
            unit_matches,
            "proposal.unit",
            "unit matches the deterministic executor",
            "unit does not match the deterministic executor",
        )
    else:
        narrative_shape_valid = (
            proposal["answer_text"] is not None
            and proposal["value"] is None
            and proposal["unit"] is None
            and _narrative_selection_shape_valid(task_input, proposal)
        )
        _check(
            checks,
            "ANSWER_SHAPE_VALID",
            narrative_shape_valid,
            "proposal",
            "narrative answer uses claim-level evidence",
            "narrative answer must contain text and claim-level evidence only",
        )
        support_valid = evidence_valid and _validate_support(
            proposal,
            evidence_by_id,
            reconstructed_extracts=(
                task_input.get("agent_task_input_version") in {"v2.2", "v2.3"}
            ),
        )
        _check(
            checks,
            "SUPPORT_VALID",
            support_valid,
            "proposal.claims",
            "answer and supporting excerpts occur verbatim in cited evidence",
            "answer or a supporting excerpt is not present in cited evidence",
        )


def verify_proposal(
    task_input: Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    evidence_items: Sequence[Mapping[str, Any]] = (),
    tool_observation: Mapping[str, Any] | None = None,
    repair_count: int = 0,
    stage: str = "synthesis",
) -> dict[str, Any]:
    """Deterministically approve a tool call or gate a final answer/refusal.

    Failure details contain only stable codes, fields, and generic messages suitable
    for one bounded repair prompt; expected values and evidence content are omitted.
    """

    validate_agent_task_input(task_input)
    if (
        isinstance(repair_count, bool)
        or not isinstance(repair_count, int)
        or not 0 <= repair_count <= 1
    ):
        raise ContractValidationError(
            "REPAIR_LIMIT_EXCEEDED", "repair_count", "must be 0 or 1"
        )
    task_id = task_input["task_id"]
    checks: list[dict[str, Any]] = []
    if (
        isinstance(proposal, Mapping)
        and isinstance(proposal.get("task_id"), str)
        and proposal["task_id"] != task_id
    ):
        _check(
            checks,
            "TASK_MATCH",
            False,
            "proposal.task_id",
            "proposal belongs to the task",
            "proposal belongs to a different task",
        )
        return _result(
            task_id,
            checks,
            None,
            repair_count,
            stage=stage,
            force_nonrepairable=True,
        )
    try:
        validate_agent_proposal(proposal)
    except ContractValidationError as exc:
        _check(
            checks,
            "SCHEMA_VALID",
            False,
            exc.field,
            "proposal matches agent_proposal.v2",
            f"proposal contract violation ({exc.code})",
        )
        return _result(task_id, checks, None, repair_count, stage=stage)
    except (TypeError, ValueError):
        _check(
            checks,
            "SCHEMA_VALID",
            False,
            "proposal",
            "proposal matches agent_proposal.v2",
            "proposal contract violation",
        )
        return _result(task_id, checks, None, repair_count, stage=stage)

    _check(
        checks,
        "SCHEMA_VALID",
        True,
        "proposal",
        "proposal matches agent_proposal.v2",
        "proposal contract violation",
    )
    task_matches = proposal["task_id"] == task_id
    _check(
        checks,
        "TASK_MATCH",
        task_matches,
        "proposal.task_id",
        "proposal belongs to the task",
        "proposal belongs to a different task",
    )
    if proposal["action"] == "CALL_TOOL":
        injection_clear = not _prompt_injection_flags(task_input, evidence_items)
        _check(
            checks,
            "PROMPT_INJECTION_CLEAR",
            injection_clear,
            "proposal",
            "no hostile embedded instruction can influence a tool call",
            "hostile embedded instructions prevent tool execution",
        )
        _validate_tool_call(task_input, proposal, checks)
    else:
        _validate_final_proposal(
            task_input, proposal, evidence_items, tool_observation, checks
        )
    return _result(
        task_id,
        checks,
        proposal["action"],
        repair_count,
        stage=stage,
    )
