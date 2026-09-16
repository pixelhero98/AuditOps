"""Explicitly non-human v2.4p benchmark and run authorization controls.

The normative v2.4 path remains external-human-only.  This module exists so a
research run can consume AI review feedback without rewriting an AI identity
or attestation as human approval.  Every published object carries the
``PROVISIONAL_AI_REVIEW`` and ``EXPLORATORY`` labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agent_benchmark_v24 import (
    _NEGATIVE_CATALOG_R2,
    CANDIDATE_VERSION_V24_R2,
    GATE_MEMBERSHIP_VERSION_V24,
    _anchor_topic_from_question_v24r2,
    _artifact,
    _negative_spec_matches_scope_v24r2,
    _read_json,
    _read_jsonl,
    _sha256_path,
    _write_json,
    _write_jsonl,
    validate_natural_question,
    validate_question_scope_v24,
    validate_semantic_anchor_v24r2,
    verify_agent_benchmark_candidate_v24,
)
from .agent_contracts import validate_agent_task_input
from .agent_review_v24 import verify_agent_benchmark_review_bundle_v24
from .canonical_json import canonical_json_sha256
from .provenance import assert_no_secrets
from .text_support import is_contiguous_text_supported, normalize_support_text

BENCHMARK_VERSION_V24P = "agent_benchmark.v2.4p"
PROVISIONAL_AI_REVIEW_VERSION_V24P = "auditops-benchmark-provisional-ai-review.v2.4p"
PROVISIONAL_ACCEPTANCE_VERSION_V24P = "auditops-benchmark-provisional-acceptance.v2.4p"
GATE_PROVISIONAL_REVIEW_VERSION_V24P = "auditops-gate-provisional-ai-review.v2.4p"
EXPLORATORY_AUTHORIZATION_VERSION_V24P = (
    "auditops-exploratory-run-authorization.v2.4p.1"
)

_ASSURANCE = "PROVISIONAL_AI_REVIEW"
_EXPERIMENT_LABEL = "EXPLORATORY"
_MODEL_IDENTITY = "Codex/GPT-5"
_DEPLOYMENT_STATUS = "NOT_EXPOSED_TO_AGENT"
_CHECK_NAMES = (
    "question_quality",
    "subtype",
    "scope",
    "answerability",
    "evidence",
    "semantic_anchors",
    "refusal_validity",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _review_identity(*, pass_kind: str, pass_id: str) -> dict[str, Any]:
    return {
        "reviewer_kind": "AI",
        "model_identity": _MODEL_IDENTITY,
        "deployment_identifier": None,
        "deployment_identifier_status": _DEPLOYMENT_STATUS,
        "pass_kind": pass_kind,
        "pass_id": pass_id,
        "independence_claim": False,
        "reviewed_at": _utc_now(),
        "attestation": (
            "I performed a provisional AI review. This is not human approval "
            "and makes no claim of reviewer independence."
        ),
    }


def _bundle_bindings(bundle_root: Path) -> dict[str, Mapping[str, Any]]:
    manifest = _read_json(bundle_root / "bundle_manifest.json")
    bindings = manifest.get("refusal_scope_bindings")
    if not isinstance(bindings, list):
        raise TypeError("Review bundle refusal bindings must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise TypeError("Review bundle refusal bindings must be objects")
        task_id = str(binding.get("task_id") or "")
        if not task_id or task_id in result:
            raise ValueError("Review bundle refusal binding IDs are invalid")
        result[task_id] = binding
    return result


def _negative_spec(subtype: str, claim: str) -> Mapping[str, Any] | None:
    normalized = normalize_support_text(claim).casefold()
    return next(
        (
            spec
            for spec in _NEGATIVE_CATALOG_R2.get(subtype, ())
            if normalize_support_text(str(spec.get("claim") or "")).casefold()
            == normalized
        ),
        None,
    )


def _record_checks(
    record: Mapping[str, Any],
    *,
    bundle_root: Path,
    refusal_bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, bool], list[str]]:
    checks = {name: True for name in _CHECK_NAMES}
    findings: list[str] = []
    task_id = str(record.get("task_id") or "")
    subtype = str(record.get("narrative_subtype") or "")
    answerability = str(record.get("answerability") or "")

    try:
        validate_natural_question(str(record.get("question") or ""))
    except ValueError as exc:
        checks["question_quality"] = False
        findings.append(str(exc))
    if "closed during the year" in str(record.get("question") or "").casefold():
        checks["question_quality"] = False
        findings.append("question retains the ambiguous r1 pension wording")
    try:
        validate_question_scope_v24(record)
    except ValueError as exc:
        checks["subtype"] = False
        findings.append(str(exc))

    evidence = record.get("frozen_evidence")
    if not isinstance(evidence, list) or not all(
        isinstance(item, Mapping) for item in evidence
    ):
        checks["evidence"] = False
        evidence = []
        findings.append("frozen evidence is not an object array")
    evidence_by_id = {
        str(item.get("evidence_id") or ""): item
        for item in evidence
        if isinstance(item, Mapping)
    }
    scope_description = str(record.get("scope_description") or "").casefold()
    answer_sets = record.get("acceptable_answer_sets")
    refusal_scope = record.get("refusal_scope")

    if answerability == "ANSWERABLE":
        if (
            not isinstance(answer_sets, list)
            or not answer_sets
            or refusal_scope is not None
        ):
            checks["answerability"] = False
            findings.append(
                "answerable record lacks an exclusive acceptable answer set"
            )
        else:
            for answer_set in answer_sets:
                extracts = (
                    answer_set.get("extracts")
                    if isinstance(answer_set, Mapping)
                    else None
                )
                if not isinstance(extracts, list) or not extracts:
                    checks["answerability"] = False
                    findings.append("acceptable answer set has no extracts")
                    continue
                for extract in extracts:
                    if not isinstance(extract, Mapping):
                        checks["evidence"] = False
                        findings.append("acceptable extract is not an object")
                        continue
                    evidence_id = str(extract.get("evidence_id") or "")
                    support_window = str(extract.get("support_window") or "")
                    item = evidence_by_id.get(evidence_id)
                    if item is None or not is_contiguous_text_supported(
                        support_window, str(item.get("content") or "")
                    ):
                        checks["evidence"] = False
                        checks["answerability"] = False
                        findings.append(
                            "support window is not bound to its exact evidence item"
                        )
                    if subtype == "footnote_note" and item is not None:
                        metadata = item.get("metadata")
                        item_number = (
                            str(metadata.get("item") or "")
                            if isinstance(metadata, Mapping)
                            else ""
                        )
                        if item_number not in {"8", "8A"}:
                            checks["scope"] = False
                            checks["subtype"] = False
                            findings.append("footnote support is outside Item 8/8A")
                    anchors = extract.get("required_semantic_anchors")
                    if not isinstance(anchors, list) or not anchors:
                        checks["semantic_anchors"] = False
                        findings.append("positive extract has no semantic anchors")
                    else:
                        for anchor in anchors:
                            try:
                                validate_semantic_anchor_v24r2(
                                    str(anchor),
                                    support_window=support_window,
                                    topic=_anchor_topic_from_question_v24r2(
                                        str(record.get("question") or "")
                                    ),
                                )
                            except ValueError as exc:
                                checks["semantic_anchors"] = False
                                findings.append(str(exc))
        if subtype == "auditor_report_opinion_language" and (
            "independent auditor's report, opinion on the financial statements"
            not in scope_description
        ):
            checks["scope"] = False
            findings.append("auditor-opinion scope is not the fixed opinion section")
        if refusal_scope is not None:
            checks["refusal_validity"] = False
            findings.append("answerable record unexpectedly carries refusal scope")
    elif answerability == "UNANSWERABLE":
        if answer_sets != [] or not isinstance(refusal_scope, Mapping):
            checks["answerability"] = False
            findings.append("unanswerable record has invalid answer/refusal material")
        else:
            binding = refusal_bindings.get(task_id)
            if binding is None:
                checks["scope"] = False
                checks["refusal_validity"] = False
                findings.append("refusal lacks a complete-scope bundle binding")
            else:
                scope_path = bundle_root / str(binding.get("scope_path") or "")
                try:
                    payload = scope_path.read_bytes()
                except OSError:
                    checks["scope"] = False
                    checks["refusal_validity"] = False
                    findings.append("bound refusal scope cannot be read")
                else:
                    scope_hash = _sha256_bytes(payload)
                    if scope_hash != binding.get("sha256"):
                        checks["scope"] = False
                        findings.append("bound refusal scope hash differs")
                    expected_text_hash = refusal_scope.get("full_filing_text_sha256")
                    if scope_hash != expected_text_hash:
                        checks["scope"] = False
                        findings.append("review record and bundle scope hashes differ")
                    claim = str(refusal_scope.get("absent_claim") or "")
                    spec = _negative_spec(subtype, claim)
                    if spec is None:
                        checks["refusal_validity"] = False
                        findings.append(
                            "absent claim is not registered in the r2 catalog"
                        )
                    elif _negative_spec_matches_scope_v24r2(
                        spec,
                        normalize_support_text(payload.decode("utf-8")).casefold(),
                    ):
                        checks["answerability"] = False
                        checks["refusal_validity"] = False
                        findings.append(
                            "claim or reasonable synonymous wording occurs in scope"
                        )
                if subtype == "critical_audit_matter" and (
                    binding.get("scope_kind") != "CANONICAL_PLUS_RAW_CAM"
                    or "raw cam supplement" not in scope_description
                    or "raw cam supplement"
                    not in str(refusal_scope.get("scope_description") or "").casefold()
                ):
                    checks["scope"] = False
                    findings.append(
                        "CAM refusal is not explicitly bound to augmented scope"
                    )
            if refusal_scope.get("causal_negative_type") != (
                "review_required_full_scope_absence"
            ):
                checks["refusal_validity"] = False
                findings.append("r2 refusal has the wrong provisional negative type")
        if answer_sets:
            checks["semantic_anchors"] = False
            findings.append("unanswerable record unexpectedly carries semantic anchors")
    else:
        checks["answerability"] = False
        findings.append("answerability is neither ANSWERABLE nor UNANSWERABLE")

    if not task_id or subtype not in _NEGATIVE_CATALOG_R2:
        checks["subtype"] = False
        findings.append("task identity or narrative subtype is invalid")
    return checks, sorted(set(findings))


def write_provisional_ai_review_v24p(
    candidate_dir: str | Path,
    review_bundle_dir: str | Path,
    output_path: str | Path,
    *,
    pass_kind: str,
) -> dict[str, Any]:
    """Write one fail-closed Codex review pass over an immutable r2 bundle."""

    if pass_kind not in {"PRIMARY", "CRITIC"}:
        raise ValueError("pass_kind must be PRIMARY or CRITIC")
    candidate_root = Path(candidate_dir)
    candidate_verification = verify_agent_benchmark_candidate_v24(candidate_root)
    if not candidate_verification["valid"]:
        raise ValueError(
            "Invalid r2 candidate: " + "; ".join(candidate_verification["errors"])
        )
    candidate_manifest = _read_json(candidate_root / "candidate_manifest.json")
    if candidate_manifest.get("candidate_version") != CANDIDATE_VERSION_V24_R2:
        raise ValueError("Provisional AI review accepts only a v2.4r2 candidate")
    bundle_root = Path(review_bundle_dir)
    bundle_verification = verify_agent_benchmark_review_bundle_v24(bundle_root)
    if not bundle_verification["valid"]:
        raise ValueError(
            "Invalid r2 review bundle: " + "; ".join(bundle_verification["errors"])
        )
    bundle_manifest = _read_json(bundle_root / "bundle_manifest.json")
    if bundle_manifest.get("candidate_id") != candidate_manifest.get("candidate_id"):
        raise ValueError("Review bundle targets a different r2 candidate")
    packet = _read_json(candidate_root / "review_packet.json")
    records = packet.get("records")
    if not isinstance(records, list) or len(records) != 200:
        raise ValueError("r2 provisional review requires 200 narrative records")
    if pass_kind == "CRITIC":
        sample = _read_json(candidate_root / "review_sample_40.json")
        sample_ids = [str(task_id) for task_id in sample.get("task_ids") or []]
        records_by_id = {str(record["task_id"]): record for record in records}
        if len(sample_ids) != 40 or any(
            task_id not in records_by_id for task_id in sample_ids
        ):
            raise ValueError("r2 critic membership is invalid")
        assigned = [records_by_id[task_id] for task_id in sample_ids]
    else:
        assigned = list(records)
    bindings = _bundle_bindings(bundle_root)
    pass_id = (
        "v24p-"
        + canonical_json_sha256(
            {
                "candidate_id": candidate_manifest["candidate_id"],
                "bundle_id": bundle_manifest["bundle_id"],
                "pass_kind": pass_kind,
                "task_ids": [record["task_id"] for record in assigned],
            }
        )[:24]
    )
    decisions: list[dict[str, Any]] = []
    for record in assigned:
        checks, findings = _record_checks(
            record,
            bundle_root=bundle_root,
            refusal_bindings=bindings,
        )
        approved = all(checks.values()) and not findings
        decisions.append(
            {
                "task_id": str(record["task_id"]),
                "decision": "APPROVE" if approved else "REJECT",
                "checks": checks,
                "comment": None if approved else "; ".join(findings),
            }
        )
    review_material = {
        "provisional_ai_review_version": PROVISIONAL_AI_REVIEW_VERSION_V24P,
        "candidate_id": candidate_manifest["candidate_id"],
        "candidate_manifest_sha256": _sha256_path(
            candidate_root / "candidate_manifest.json"
        ),
        "rendered_packet_sha256": bundle_manifest["rendered_packet_sha256"],
        "review_packet_sha256": _sha256_path(candidate_root / "review_packet.json"),
        "source_manifest_sha256": _sha256_path(candidate_root / "source_manifest.json"),
        "review_bundle_id": bundle_manifest["bundle_id"],
        "assurance": _ASSURANCE,
        "experiment_label": _EXPERIMENT_LABEL,
        "reviewer": _review_identity(pass_kind=pass_kind, pass_id=pass_id),
        "method": {
            "method_id": "auditops.codex_provisional_review.v2.4p",
            "complete_scope_used_for_refusals": True,
            "deterministic_preflight_used": True,
            "human_approval": False,
        },
        "decisions": decisions,
    }
    review = {
        **review_material,
        "decision_set_sha256": canonical_json_sha256(decisions),
        "review_id": canonical_json_sha256(review_material),
    }
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite provisional AI review: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, review)
    assert_no_secrets([destination])
    return {
        "review_id": review["review_id"],
        "output": str(destination),
        "pass_kind": pass_kind,
        "case_count": len(decisions),
        "approvals": sum(item["decision"] == "APPROVE" for item in decisions),
        "rejections": sum(item["decision"] == "REJECT" for item in decisions),
        "human_approval": False,
    }


def _validate_provisional_review(
    review: Mapping[str, Any],
    *,
    candidate_id: str,
    expected_kind: str,
    expected_ids: Sequence[str],
) -> dict[str, str]:
    reviewer = review.get("reviewer")
    method = review.get("method")
    if (
        review.get("provisional_ai_review_version")
        != PROVISIONAL_AI_REVIEW_VERSION_V24P
        or review.get("candidate_id") != candidate_id
        or review.get("assurance") != _ASSURANCE
        or review.get("experiment_label") != _EXPERIMENT_LABEL
        or not isinstance(reviewer, Mapping)
        or reviewer.get("reviewer_kind") != "AI"
        or reviewer.get("model_identity") != _MODEL_IDENTITY
        or reviewer.get("deployment_identifier") is not None
        or reviewer.get("deployment_identifier_status") != _DEPLOYMENT_STATUS
        or reviewer.get("pass_kind") != expected_kind
        or reviewer.get("independence_claim") is not False
        or not isinstance(method, Mapping)
        or method.get("human_approval") is not False
    ):
        raise ValueError("Provisional AI review identity is invalid")
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise TypeError("Provisional AI decisions must be an array")
    if review.get("decision_set_sha256") != canonical_json_sha256(decisions):
        raise ValueError("Provisional AI decision hash is invalid")
    material = {
        key: value
        for key, value in review.items()
        if key not in {"decision_set_sha256", "review_id"}
    }
    if review.get("review_id") != canonical_json_sha256(material):
        raise ValueError("Provisional AI review ID is invalid")
    by_id: dict[str, str] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping) or set(decision) != {
            "task_id",
            "decision",
            "checks",
            "comment",
        }:
            raise ValueError("Provisional AI decision envelope is invalid")
        task_id = str(decision.get("task_id") or "")
        checks = decision.get("checks")
        if (
            task_id in by_id
            or not isinstance(checks, Mapping)
            or set(checks) != set(_CHECK_NAMES)
            or any(type(value) is not bool for value in checks.values())
            or decision.get("decision") not in {"APPROVE", "REJECT"}
            or (
                decision.get("decision") == "APPROVE"
                and (not all(checks.values()) or decision.get("comment") is not None)
            )
            or (
                decision.get("decision") == "REJECT"
                and not str(decision.get("comment") or "").strip()
            )
        ):
            raise ValueError(f"Invalid provisional AI decision: {task_id}")
        by_id[task_id] = str(decision["decision"])
    if list(by_id) != list(expected_ids):
        raise ValueError(f"{expected_kind} provisional AI membership/order differs")
    return by_id


def write_provisional_acceptance_v24p(
    candidate_dir: str | Path,
    primary_review: str | Path,
    critic_review: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    candidate_root = Path(candidate_dir)
    verification = verify_agent_benchmark_candidate_v24(candidate_root)
    if not verification["valid"]:
        raise ValueError("Invalid r2 candidate: " + "; ".join(verification["errors"]))
    manifest = _read_json(candidate_root / "candidate_manifest.json")
    if manifest.get("candidate_version") != CANDIDATE_VERSION_V24_R2:
        raise ValueError("Provisional acceptance requires candidate v2.4r2")
    packet = _read_json(candidate_root / "review_packet.json")
    all_ids = [str(record["task_id"]) for record in packet["records"]]
    sample = _read_json(candidate_root / "review_sample_40.json")
    sample_ids = [str(task_id) for task_id in sample["task_ids"]]
    primary_path = Path(primary_review)
    critic_path = Path(critic_review)
    primary = _read_json(primary_path)
    critic = _read_json(critic_path)
    primary_decisions = _validate_provisional_review(
        primary,
        candidate_id=str(manifest["candidate_id"]),
        expected_kind="PRIMARY",
        expected_ids=all_ids,
    )
    critic_decisions = _validate_provisional_review(
        critic,
        candidate_id=str(manifest["candidate_id"]),
        expected_kind="CRITIC",
        expected_ids=sample_ids,
    )
    if primary["reviewer"]["pass_id"] == critic["reviewer"]["pass_id"]:
        raise ValueError("Primary and critic passes must have different pass IDs")
    rejected = [
        task_id
        for task_id, decision in primary_decisions.items()
        if decision != "APPROVE"
    ]
    critic_rejected = [
        task_id
        for task_id, decision in critic_decisions.items()
        if decision != "APPROVE"
    ]
    disagreements = [
        task_id
        for task_id in sample_ids
        if primary_decisions[task_id] != critic_decisions[task_id]
    ]
    if rejected or critic_rejected or disagreements:
        raise ValueError(
            "Provisional acceptance requires 200/200 primary and 40/40 critic approvals "
            f"with exact agreement; primary_rejected={len(rejected)}, "
            f"critic_rejected={len(critic_rejected)}, disagreements={len(disagreements)}"
        )
    material = {
        "provisional_acceptance_version": PROVISIONAL_ACCEPTANCE_VERSION_V24P,
        "candidate_id": manifest["candidate_id"],
        "candidate_manifest_sha256": _sha256_path(
            candidate_root / "candidate_manifest.json"
        ),
        "review_packet_sha256": _sha256_path(candidate_root / "review_packet.json"),
        "source_manifest_sha256": _sha256_path(candidate_root / "source_manifest.json"),
        "r1_feedback_report_sha256": _sha256_path(
            candidate_root / "r1_feedback_report.json"
        ),
        "primary_review": {
            "sha256": _sha256_path(primary_path),
            "review_id": primary["review_id"],
            "case_count": 200,
        },
        "critic_review": {
            "sha256": _sha256_path(critic_path),
            "review_id": critic["review_id"],
            "case_count": 40,
        },
        "approved_task_ids_sha256": canonical_json_sha256(all_ids),
        "approved_case_count": 200,
        "assurance": _ASSURANCE,
        "experiment_label": _EXPERIMENT_LABEL,
        "external_human_approval": False,
        "accepted_at": _utc_now(),
    }
    acceptance = {**material, "acceptance_id": canonical_json_sha256(material)}
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite provisional acceptance: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, acceptance)
    assert_no_secrets([destination])
    return {
        "acceptance_id": acceptance["acceptance_id"],
        "output": str(destination),
        "approved_case_count": 200,
        "external_human_approval": False,
    }


def _validate_acceptance(
    acceptance: Mapping[str, Any],
    *,
    candidate_root: Path,
    candidate_manifest: Mapping[str, Any],
) -> None:
    material = {
        key: value for key, value in acceptance.items() if key != "acceptance_id"
    }
    if (
        acceptance.get("provisional_acceptance_version")
        != PROVISIONAL_ACCEPTANCE_VERSION_V24P
        or acceptance.get("candidate_id") != candidate_manifest.get("candidate_id")
        or acceptance.get("candidate_manifest_sha256")
        != _sha256_path(candidate_root / "candidate_manifest.json")
        or acceptance.get("review_packet_sha256")
        != _sha256_path(candidate_root / "review_packet.json")
        or acceptance.get("source_manifest_sha256")
        != _sha256_path(candidate_root / "source_manifest.json")
        or acceptance.get("r1_feedback_report_sha256")
        != _sha256_path(candidate_root / "r1_feedback_report.json")
        or acceptance.get("approved_case_count") != 200
        or acceptance.get("assurance") != _ASSURANCE
        or acceptance.get("experiment_label") != _EXPERIMENT_LABEL
        or acceptance.get("external_human_approval") is not False
        or acceptance.get("acceptance_id") != canonical_json_sha256(material)
    ):
        raise ValueError("Provisional acceptance binding is invalid")


def freeze_agent_benchmark_v24p(
    candidate_dir: str | Path,
    acceptance_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    candidate_root = Path(candidate_dir)
    verification = verify_agent_benchmark_candidate_v24(candidate_root)
    if not verification["valid"]:
        raise ValueError("Invalid r2 candidate: " + "; ".join(verification["errors"]))
    candidate = _read_json(candidate_root / "candidate_manifest.json")
    if candidate.get("candidate_version") != CANDIDATE_VERSION_V24_R2:
        raise ValueError("v2.4p freeze requires candidate v2.4r2")
    acceptance_source = Path(acceptance_path)
    acceptance = _read_json(acceptance_source)
    _validate_acceptance(
        acceptance,
        candidate_root=candidate_root,
        candidate_manifest=candidate,
    )
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite v2.4p benchmark: {destination}")
    mapping = {
        "candidate_cases.jsonl": "cases.jsonl",
        "candidate_evidence.jsonl": "evidence.jsonl",
        "candidate_gold.jsonl": "gold.jsonl",
        "candidate_verifier_observations.jsonl": "verifier_observations.jsonl",
        "candidate_case_lineage.jsonl": "case_lineage.jsonl",
        "review_sample_40.json": "review_sample_40.json",
        "gate_50.json": "gate_50.json",
        "source_manifest.json": "source_manifest.json",
        "r1_feedback_report.json": "r1_feedback_report.json",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        for source_name, target_name in mapping.items():
            shutil.copyfile(candidate_root / source_name, temporary / target_name)
        shutil.copyfile(acceptance_source, temporary / "provisional_acceptance.json")
        _write_jsonl(temporary / "few_shot.jsonl", [])
        artifact_names = [
            *mapping.values(),
            "provisional_acceptance.json",
            "few_shot.jsonl",
        ]
        artifacts = {
            name: _artifact(
                temporary / name,
                0
                if name == "few_shot.jsonl"
                else (
                    len(_read_jsonl(temporary / name))
                    if name.endswith(".jsonl")
                    else None
                ),
            )
            for name in artifact_names
        }
        cases = _read_jsonl(temporary / "cases.jsonl")
        gold = _read_jsonl(temporary / "gold.jsonl")
        narrative_status = Counter(
            str(row.get("target", {}).get("status"))
            for row in gold
            if row.get("task_type") == "narrative_citation"
        )
        issuer_ids = {
            str(case.get("entity", {}).get("source_entity_id")) for case in cases
        }
        manifest_material = {
            "benchmark_version": BENCHMARK_VERSION_V24P,
            "benchmark_profile": "us_cached20_provisional_ai_review_v2.4p",
            "runtime_contract_version": "v2.3",
            "seed": candidate["seed"],
            "corpus_id": candidate["corpus_id"],
            "jurisdiction": "US",
            "reporting_framework": "US-GAAP",
            "standards_version": "PCAOB_PUBLIC_FILING_METADATA_V0.1",
            "source_system": "SEC-EDGAR-CACHED",
            "candidate_id": candidate["candidate_id"],
            "acceptance_id": acceptance["acceptance_id"],
            "assurance": _ASSURANCE,
            "experiment_label": _EXPERIMENT_LABEL,
            "generalization_label": "cached-20",
            "external_human_approval": False,
            "counts": {
                "quant": 300,
                "narrative": 200,
                "narrative_answerable": narrative_status["OK"],
                "narrative_unanswerable": narrative_status["REFUSAL"],
                "issuers": len(issuer_ids),
                "total": 500,
                "few_shot": 0,
                "narrative_gate": 50,
            },
            "review_policy": {
                "primary_ai_cases": 200,
                "critic_ai_cases": 40,
                "independence_claim": False,
                "human_approval": False,
                "any_rejection_or_disagreement_blocks": True,
            },
            "few_shot_policy": {
                "status": "BLOCKED",
                "request_pack_size": 0,
            },
            "artifact_visibility": {
                "inference_visible": [
                    "cases.jsonl",
                    "evidence.jsonl",
                    "few_shot.jsonl",
                    "gate_50.json",
                ],
                "evaluator_only": [
                    "gold.jsonl",
                    "verifier_observations.jsonl",
                    "case_lineage.jsonl",
                    "provisional_acceptance.json",
                    "review_sample_40.json",
                    "r1_feedback_report.json",
                ],
            },
            "artifacts": artifacts,
        }
        manifest = {
            **manifest_material,
            "benchmark_id": canonical_json_sha256(manifest_material),
        }
        _write_json(temporary / "benchmark_manifest.json", manifest)
        final_verification = verify_agent_benchmark_v24p(temporary)
        if not final_verification["valid"]:
            raise ValueError(
                "Invalid v2.4p benchmark: " + "; ".join(final_verification["errors"])
            )
        assert_no_secrets([temporary])
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "benchmark_id": manifest["benchmark_id"],
        "output_dir": str(destination),
        "counts": manifest["counts"],
        "assurance": _ASSURANCE,
        "experiment_label": _EXPERIMENT_LABEL,
        "external_human_approval": False,
        "verified": True,
    }


def verify_agent_benchmark_v24p(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    errors: list[str] = []
    try:
        manifest = _read_json(root / "benchmark_manifest.json")
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        return {
            "valid": False,
            "errors": [f"benchmark manifest could not be loaded: {exc}"],
        }
    expected_artifacts = {
        "cases.jsonl",
        "evidence.jsonl",
        "gold.jsonl",
        "verifier_observations.jsonl",
        "case_lineage.jsonl",
        "review_sample_40.json",
        "gate_50.json",
        "source_manifest.json",
        "r1_feedback_report.json",
        "provisional_acceptance.json",
        "few_shot.jsonl",
    }
    artifacts = manifest.get("artifacts")
    if (
        manifest.get("benchmark_version") != BENCHMARK_VERSION_V24P
        or manifest.get("benchmark_profile")
        != "us_cached20_provisional_ai_review_v2.4p"
        or manifest.get("runtime_contract_version") != "v2.3"
        or manifest.get("assurance") != _ASSURANCE
        or manifest.get("experiment_label") != _EXPERIMENT_LABEL
        or manifest.get("generalization_label") != "cached-20"
        or manifest.get("external_human_approval") is not False
    ):
        errors.append("v2.4p manifest identity or assurance is invalid")
    material = {key: value for key, value in manifest.items() if key != "benchmark_id"}
    if manifest.get("benchmark_id") != canonical_json_sha256(material):
        errors.append("v2.4p benchmark ID is invalid")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        errors.append("v2.4p artifact set is invalid")
    else:
        for name, descriptor in artifacts.items():
            path = root / name
            if (
                not path.is_file()
                or path.is_symlink()
                or not isinstance(descriptor, Mapping)
                or descriptor.get("sha256") != _sha256_path(path)
                or descriptor.get("bytes") != path.stat().st_size
            ):
                errors.append(f"v2.4p artifact binding is invalid: {name}")
    try:
        cases = _read_jsonl(root / "cases.jsonl")
        evidence = _read_jsonl(root / "evidence.jsonl")
        gold = _read_jsonl(root / "gold.jsonl")
        observations = _read_jsonl(root / "verifier_observations.jsonl")
        lineage = _read_jsonl(root / "case_lineage.jsonl")
        few_shot = _read_jsonl(root / "few_shot.jsonl")
        gate = _read_json(root / "gate_50.json")
        acceptance = _read_json(root / "provisional_acceptance.json")
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        errors.append(f"v2.4p artifacts could not be loaded: {exc}")
        cases, evidence, gold, observations, lineage, few_shot, gate, acceptance = (
            [],
            [],
            [],
            [],
            [],
            [],
            {},
            {},
        )
    task_ids = [str(row.get("task_id")) for row in cases]
    if len(task_ids) != 500 or len(set(task_ids)) != 500:
        errors.append("v2.4p benchmark must contain 500 unique tasks")
    for label, rows in (
        ("evidence", evidence),
        ("gold", gold),
        ("observations", observations),
        ("lineage", lineage),
    ):
        if [str(row.get("task_id")) for row in rows] != task_ids:
            errors.append(f"v2.4p {label} membership/order differs")
    if few_shot:
        errors.append("v2.4p few-shot examples must remain empty")
    for case in cases:
        try:
            validate_agent_task_input(case)
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid v2.4p task {case.get('task_id')}: {exc}")
        if case.get("agent_task_input_version") != "v2.3":
            errors.append(
                f"v2.4p task uses the wrong runtime contract: {case.get('task_id')}"
            )
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or counts.get("total") != 500:
        errors.append("v2.4p task counts are invalid")
    gate_ids = gate.get("task_ids") if isinstance(gate, Mapping) else None
    if (
        gate.get("gate_membership_version") != GATE_MEMBERSHIP_VERSION_V24
        or not isinstance(gate_ids, list)
        or len(gate_ids) != 50
        or len(set(gate_ids)) != 50
        or not set(gate_ids).issubset(set(task_ids))
    ):
        errors.append("v2.4p narrative gate is invalid")
    if (
        acceptance.get("provisional_acceptance_version")
        != PROVISIONAL_ACCEPTANCE_VERSION_V24P
        or acceptance.get("acceptance_id") != manifest.get("acceptance_id")
        or acceptance.get("external_human_approval") is not False
    ):
        errors.append("v2.4p provisional acceptance binding is invalid")
    try:
        assert_no_secrets([root])
    except ValueError as exc:
        errors.append(str(exc))
    return {
        "valid": not errors,
        "errors": errors,
        "benchmark_id": manifest.get("benchmark_id"),
        "assurance": manifest.get("assurance"),
        "experiment_label": manifest.get("experiment_label"),
    }


def write_gate_provisional_review_v24p(
    benchmark_dir: str | Path,
    run_dir: str | Path,
    release_review_packet: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    benchmark_root = Path(benchmark_dir)
    verification = verify_agent_benchmark_v24p(benchmark_root)
    if not verification["valid"]:
        raise ValueError(
            "Invalid v2.4p benchmark: " + "; ".join(verification["errors"])
        )
    manifest = _read_json(benchmark_root / "benchmark_manifest.json")
    run_root = Path(run_dir)
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest = _read_json(run_manifest_path)
    if (
        run_manifest.get("benchmark_id") != manifest.get("benchmark_id")
        or run_manifest.get("batch_run_version") != "auditops-agent-batch.v2.4p"
        or run_manifest.get("selection_mode") != "narrative_gate"
    ):
        raise ValueError("Gate run is not the matching v2.4p narrative gate")
    packet_path = Path(release_review_packet)
    packet = _read_json(packet_path)
    records = packet.get("records")
    if not isinstance(records, list):
        raise TypeError("Gate release review packet records must be an array")
    relevant = [
        record
        for record in records
        if isinstance(record, Mapping)
        and record.get("run_manifest_sha256") == _sha256_path(run_manifest_path)
    ]
    decisions: list[dict[str, Any]] = []
    for record in relevant:
        passed = bool(
            record.get("automatic_exact_support") is True
            and record.get("automatic_anchor_satisfaction") is True
            and record.get("matched_acceptable_answer_set")
        )
        decisions.append(
            {
                "task_id": str(record.get("task_id") or ""),
                "quote_answers_question": passed,
                "comment": (
                    None
                    if passed
                    else "Released quote did not satisfy exact support and semantic anchors."
                ),
            }
        )
    material = {
        "gate_provisional_review_version": GATE_PROVISIONAL_REVIEW_VERSION_V24P,
        "benchmark_id": manifest["benchmark_id"],
        "run_manifest_sha256": _sha256_path(run_manifest_path),
        "release_review_packet_sha256": _sha256_path(packet_path),
        "assurance": _ASSURANCE,
        "experiment_label": _EXPERIMENT_LABEL,
        "reviewer": _review_identity(
            pass_kind="GATE_RELEASES",
            pass_id="v24p-gate-"
            + canonical_json_sha256(
                {
                    "benchmark_id": manifest["benchmark_id"],
                    "run_manifest_sha256": _sha256_path(run_manifest_path),
                }
            )[:24],
        ),
        "benchmark_defect_status": (
            "NONE_FOUND"
            if all(item["quote_answers_question"] for item in decisions)
            else "DEFECT_FOUND"
        ),
        "decisions": decisions,
    }
    review = {**material, "review_id": canonical_json_sha256(material)}
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite gate provisional review: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, review)
    assert_no_secrets([destination])
    return {
        "review_id": review["review_id"],
        "output": str(destination),
        "released_case_count": len(decisions),
        "confirmed": all(item["quote_answers_question"] for item in decisions),
        "benchmark_defect_status": review["benchmark_defect_status"],
        "human_approval": False,
    }


def validate_gate_provisional_review_v24p(
    review: Mapping[str, Any],
    *,
    benchmark_id: str,
    run_manifest_sha256: str,
    release_task_ids: set[str],
) -> str:
    reviewer = review.get("reviewer")
    if (
        review.get("gate_provisional_review_version")
        != GATE_PROVISIONAL_REVIEW_VERSION_V24P
        or review.get("benchmark_id") != benchmark_id
        or review.get("run_manifest_sha256") != run_manifest_sha256
        or review.get("assurance") != _ASSURANCE
        or review.get("experiment_label") != _EXPERIMENT_LABEL
        or review.get("benchmark_defect_status") not in {"NONE_FOUND", "DEFECT_FOUND"}
        or not isinstance(reviewer, Mapping)
        or reviewer.get("reviewer_kind") != "AI"
        or reviewer.get("independence_claim") is not False
    ):
        raise ValueError("Gate provisional review identity is invalid")
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise TypeError("Gate provisional review decisions must be an array")
    by_id: dict[str, bool] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise TypeError("Gate provisional decisions must be objects")
        task_id = str(decision.get("task_id") or "")
        if task_id in by_id or type(decision.get("quote_answers_question")) is not bool:
            raise ValueError("Gate provisional decision is invalid")
        by_id[task_id] = bool(decision["quote_answers_question"])
    if set(by_id) != release_task_ids:
        raise ValueError("Gate provisional review must cover every released answer")
    material = {key: value for key, value in review.items() if key != "review_id"}
    if review.get("review_id") != canonical_json_sha256(material):
        raise ValueError("Gate provisional review ID is invalid")
    return "CONFIRMED" if all(by_id.values()) else "FAILED"


def write_exploratory_authorization_v24p(
    benchmark_dir: str | Path,
    gate_metrics: str | Path,
    output_path: str | Path,
    *,
    authorized_by: str,
) -> dict[str, Any]:
    root = Path(benchmark_dir)
    verification = verify_agent_benchmark_v24p(root)
    if not verification["valid"]:
        raise ValueError(
            "Invalid v2.4p benchmark: " + "; ".join(verification["errors"])
        )
    manifest = _read_json(root / "benchmark_manifest.json")
    metrics_path = Path(gate_metrics)
    metrics = _read_json(metrics_path)
    runs = metrics.get("runs")
    if (
        metrics.get("evaluation_version") != "auditops.text_agent_evaluation.v2.4p"
        or metrics.get("benchmark", {}).get("benchmark_id") != manifest["benchmark_id"]
        or not isinstance(runs, list)
        or len(runs) != 1
        or runs[0].get("selection_mode") != "narrative_gate"
        or runs[0].get("technical_gate_pass") is not True
        or runs[0].get("provisional_release_review_status") != "CONFIRMED"
        or runs[0].get("benchmark_defect_status") != "NONE_FOUND"
        or runs[0].get("recommend_exploratory_full_500") is not True
    ):
        raise ValueError("Gate metrics do not authorize an exploratory full run")
    if (
        not isinstance(authorized_by, str)
        or not authorized_by.strip()
        or authorized_by != authorized_by.strip()
        or "\n" in authorized_by
        or "\r" in authorized_by
    ):
        raise ValueError(
            "Exploratory authorizer must be an explicit non-empty identity"
        )
    limitations = {
        "assurance": _ASSURANCE,
        "experiment_label": _EXPERIMENT_LABEL,
        "generalization_label": "cached-20",
        "external_human_approval": False,
        "official_baseline": False,
    }
    material = {
        "authorization_version": EXPLORATORY_AUTHORIZATION_VERSION_V24P,
        "benchmark_id": manifest["benchmark_id"],
        "gate_metrics_sha256": _sha256_path(metrics_path),
        "decision": "GO_EXPLORATORY",
        "authorized_by": authorized_by,
        "authorized_at": _utc_now(),
        "assurance": _ASSURANCE,
        "experiment_label": _EXPERIMENT_LABEL,
        "known_limitations": limitations,
        "known_limitations_sha256": canonical_json_sha256(limitations),
    }
    authorization = {**material, "authorization_id": canonical_json_sha256(material)}
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite exploratory authorization: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, authorization)
    assert_no_secrets([destination])
    return {
        "authorization_id": authorization["authorization_id"],
        "output": str(destination),
        "decision": "GO_EXPLORATORY",
        "authorized_by": authorized_by,
    }


def validate_exploratory_authorization_v24p(
    authorization: Mapping[str, Any], *, benchmark_id: str, expected_authorizer: str
) -> dict[str, Any]:
    if (
        not isinstance(expected_authorizer, str)
        or not expected_authorizer.strip()
        or expected_authorizer != expected_authorizer.strip()
        or "\n" in expected_authorizer
        or "\r" in expected_authorizer
    ):
        raise ValueError(
            "Expected exploratory authorizer must be explicitly configured"
        )
    limitations = authorization.get("known_limitations")
    material = {
        key: value for key, value in authorization.items() if key != "authorization_id"
    }
    if (
        authorization.get("authorization_version")
        != EXPLORATORY_AUTHORIZATION_VERSION_V24P
        or authorization.get("benchmark_id") != benchmark_id
        or authorization.get("decision") != "GO_EXPLORATORY"
        or authorization.get("authorized_by") != expected_authorizer
        or authorization.get("assurance") != _ASSURANCE
        or authorization.get("experiment_label") != _EXPERIMENT_LABEL
        or not isinstance(limitations, Mapping)
        or limitations.get("external_human_approval") is not False
        or limitations.get("official_baseline") is not False
        or authorization.get("known_limitations_sha256")
        != canonical_json_sha256(limitations)
        or authorization.get("authorization_id") != canonical_json_sha256(material)
    ):
        raise ValueError("Exploratory full-run authorization is invalid")
    timestamp = authorization.get("authorized_at")
    if not isinstance(timestamp, str):
        raise TypeError("Exploratory authorization timestamp is required")
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None:
        raise ValueError("Exploratory authorization timestamp must include timezone")
    digest = authorization.get("gate_metrics_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Exploratory authorization gate metrics hash is invalid")
    return dict(authorization)


__all__ = [
    "BENCHMARK_VERSION_V24P",
    "EXPLORATORY_AUTHORIZATION_VERSION_V24P",
    "GATE_PROVISIONAL_REVIEW_VERSION_V24P",
    "PROVISIONAL_ACCEPTANCE_VERSION_V24P",
    "PROVISIONAL_AI_REVIEW_VERSION_V24P",
    "freeze_agent_benchmark_v24p",
    "validate_exploratory_authorization_v24p",
    "validate_gate_provisional_review_v24p",
    "verify_agent_benchmark_v24p",
    "write_exploratory_authorization_v24p",
    "write_gate_provisional_review_v24p",
    "write_provisional_acceptance_v24p",
    "write_provisional_ai_review_v24p",
]
