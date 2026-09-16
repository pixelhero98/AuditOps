"""AuditOps v2.4 benchmark and narrative-gate evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .agent_benchmark_v24 import verify_agent_benchmark_v24
from .agent_contracts import validate_agent_case_result
from .agent_provisional_v24p import (
    BENCHMARK_VERSION_V24P,
    validate_gate_provisional_review_v24p,
    verify_agent_benchmark_v24p,
)
from .canonical_json import canonical_json_bytes, canonical_json_sha256
from .provenance import assert_no_secrets
from .text_support import is_contiguous_text_supported, normalize_support_text

EVALUATION_VERSION_V24 = "auditops.text_agent_evaluation.v2.4"
EVALUATION_VERSION_V24P = "auditops.text_agent_evaluation.v2.4p"
GATE_RELEASE_REVIEW_VERSION_V24 = "auditops-gate-release-review.v2.4"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected object in {path} line {line_number}")
            rows.append(value)
    return rows


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _decimal(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return result if result.is_finite() else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _proposal_extracts(proposal: Mapping[str, Any]) -> list[dict[str, str]]:
    claims = proposal.get("claims")
    if not isinstance(claims, list):
        return []
    extracts: list[dict[str, str]] = []
    for claim in claims:
        if not isinstance(claim, Mapping):
            return []
        evidence_ids = claim.get("evidence_ids")
        quote = claim.get("supporting_text")
        if (
            not isinstance(evidence_ids, list)
            or len(evidence_ids) != 1
            or not isinstance(evidence_ids[0], str)
            or not isinstance(quote, str)
            or not quote
        ):
            return []
        extracts.append({"evidence_id": evidence_ids[0], "exact_quote": quote})
    return extracts


def _released_support(
    proposal: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]
) -> tuple[bool, bool]:
    evidence_by_id = {str(item.get("evidence_id")): item for item in evidence}
    extracts = _proposal_extracts(proposal)
    citations = proposal.get("evidence_ids")
    citation_ids = (
        [str(item) for item in citations] if isinstance(citations, list) else []
    )
    citations_valid = (
        bool(citation_ids)
        and len(citation_ids) == len(set(citation_ids))
        and all(evidence_id in evidence_by_id for evidence_id in citation_ids)
    )
    exact_support = bool(extracts) and all(
        extract["evidence_id"] in evidence_by_id
        and is_contiguous_text_supported(
            extract["exact_quote"],
            str(evidence_by_id[extract["evidence_id"]].get("content") or ""),
        )
        for extract in extracts
    )
    return citations_valid, exact_support


def _matches_answer_set(
    proposal: Mapping[str, Any], acceptable_sets: Sequence[Mapping[str, Any]]
) -> tuple[bool, str | None]:
    actual = _proposal_extracts(proposal)
    for answer_set in acceptable_sets:
        expected = answer_set.get("extracts")
        if not isinstance(expected, list) or len(actual) != len(expected):
            continue
        unmatched = list(actual)
        valid = True
        for target in expected:
            if not isinstance(target, Mapping):
                valid = False
                break
            target_id = target.get("evidence_id")
            window = target.get("support_window")
            anchors = target.get("required_semantic_anchors")
            if not isinstance(window, str) or not isinstance(anchors, list):
                valid = False
                break
            match_index = next(
                (
                    index
                    for index, extract in enumerate(unmatched)
                    if extract["evidence_id"] == target_id
                    and is_contiguous_text_supported(extract["exact_quote"], window)
                    and all(
                        normalize_support_text(str(anchor)).casefold()
                        in normalize_support_text(extract["exact_quote"]).casefold()
                        for anchor in anchors
                    )
                ),
                None,
            )
            if match_index is None:
                valid = False
                break
            unmatched.pop(match_index)
        if valid and not unmatched:
            return True, str(answer_set.get("set_id"))
    return False, None


def _scope_cases(
    root: Path,
    cases: Sequence[Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> tuple[str, list[Mapping[str, Any]]]:
    benchmark_version = _read_json(root / "benchmark_manifest.json").get(
        "benchmark_version"
    )
    expected_batch_version = (
        "auditops-agent-batch.v2.4p"
        if benchmark_version == BENCHMARK_VERSION_V24P
        else "auditops-agent-batch.v2.4"
    )
    if run_manifest.get("batch_run_version") != expected_batch_version:
        raise ValueError("Run batch version does not match the v2.4-family benchmark")
    if run_manifest.get("benchmark_manifest_sha256") != _sha256_path(
        root / "benchmark_manifest.json"
    ):
        raise ValueError("Run benchmark manifest hash mismatch")
    mode = run_manifest.get("selection_mode")
    cases_by_id = {str(case["task_id"]): case for case in cases}
    if mode == "full":
        expected = [str(case["task_id"]) for case in cases]
        scope = list(cases)
        if run_manifest.get("complete_full_benchmark") is not True:
            raise ValueError("Full v2.4 run is not marked complete")
    elif mode == "narrative_gate":
        gate = _read_json(root / "gate_50.json")
        expected = [str(task_id) for task_id in gate["task_ids"]]
        scope = [cases_by_id[task_id] for task_id in expected]
        if run_manifest.get("complete_full_benchmark") is not False:
            raise ValueError("Narrative gate cannot be marked a full benchmark run")
        inputs = run_manifest.get("inputs")
        if not isinstance(inputs, Mapping) or inputs.get(
            "parity_membership_sha256"
        ) != _sha256_path(root / "gate_50.json"):
            raise ValueError("Narrative gate run does not bind gate_50.json")
    else:
        raise ValueError(
            "v2.4 evaluation accepts only full or narrative_gate selection"
        )
    result_ids = [str(row.get("task_id")) for row in results]
    if result_ids != expected:
        raise ValueError("Run result membership/order differs from frozen scope")
    if run_manifest.get("task_ids_sha256") != canonical_json_sha256(result_ids):
        raise ValueError("Run task membership hash mismatch")
    return str(mode), scope


def _validate_human_release_review(
    review_path: Path | None,
    *,
    benchmark_id: str,
    run_manifest_sha256: str,
    release_task_ids: set[str],
) -> tuple[str, dict[str, Any] | None]:
    if review_path is None:
        return "PENDING_HUMAN_REVIEW", None
    review = _read_json(review_path)
    if (
        review.get("gate_release_review_version") != GATE_RELEASE_REVIEW_VERSION_V24
        or review.get("benchmark_id") != benchmark_id
        or review.get("run_manifest_sha256") != run_manifest_sha256
    ):
        raise ValueError(
            "Gate release review identity does not match the evaluated run"
        )
    reviewer = review.get("reviewer")
    if (
        not isinstance(reviewer, Mapping)
        or set(reviewer) != {"identity", "reviewed_at"}
        or not str(reviewer.get("identity") or "").strip()
    ):
        raise ValueError("Gate release review requires a human reviewer identity")
    if any(
        term in str(reviewer.get("identity")).casefold()
        for term in ("codex", "chatgpt", "openai")
    ):
        raise ValueError("Gate release review cannot be self-approved by Codex")
    reviewed_at = reviewer.get("reviewed_at")
    if not isinstance(reviewed_at, str):
        raise TypeError("Gate release review requires an RFC 3339 timestamp")
    try:
        parsed_reviewed_at = datetime.fromisoformat(reviewed_at)
    except ValueError as exc:
        raise ValueError("Gate release review requires an RFC 3339 timestamp") from exc
    if parsed_reviewed_at.tzinfo is None:
        raise ValueError("Gate release review timestamp must include a timezone")
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise TypeError("Gate release review decisions must be an array")
    by_id: dict[str, bool] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise TypeError("Gate release review decisions must be objects")
        task_id = str(decision.get("task_id") or "")
        if task_id in by_id:
            raise ValueError(f"Duplicate gate release review task: {task_id}")
        if not isinstance(decision.get("quote_answers_question"), bool):
            raise TypeError("quote_answers_question must be boolean")
        by_id[task_id] = bool(decision["quote_answers_question"])
    if set(by_id) != release_task_ids:
        raise ValueError(
            "Gate release review must cover every released narrative answer"
        )
    status = "CONFIRMED" if all(by_id.values()) else "FAILED"
    return status, review


def _evaluate_run(
    *,
    root: Path,
    benchmark_manifest: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    evidence_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
    run_dir: Path,
    human_release_review: Path | None,
    provisional_release_review: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = _read_json(run_manifest_path)
    results_path = run_dir / "results.jsonl"
    if _sha256_path(results_path) != run_manifest.get("results", {}).get("sha256"):
        raise ValueError("Run results hash mismatch")
    results = _read_jsonl(results_path)
    selection_mode, scoped_cases = _scope_cases(root, cases, run_manifest, results)
    result_by_id = {str(row["task_id"]): row for row in results}
    counters: Counter[str] = Counter()
    subtype_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    answerability_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    negative_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    filing_scores: defaultdict[str, list[int]] = defaultdict(list)
    latencies: list[float] = []
    invocation_tokens: list[float] = []
    input_tokens: list[float] = []
    output_tokens: list[float] = []
    peak_vram: list[int] = []
    release_review_records: list[dict[str, Any]] = []
    release_task_ids: set[str] = set()

    for case in scoped_cases:
        task_id = str(case["task_id"])
        result = result_by_id[task_id]
        validate_agent_case_result(result)
        counters["tasks"] += 1
        counters["typed"] += 1
        outcome = str(result["outcome"])
        counters[f"outcome:{outcome}"] += 1
        record = result["run_record"]
        repair_count = int(record.get("repair_count") or 0)
        counters["repairs"] += repair_count
        counters["repair_limit_violation"] += int(repair_count > 1)
        resources = record.get("resources")
        if isinstance(resources, Mapping) and isinstance(
            resources.get("peak_vram_bytes"), int
        ):
            peak_vram.append(int(resources["peak_vram_bytes"]))
        for attempt in record.get("verifier_trace") or []:
            if not isinstance(attempt, Mapping):
                continue
            latencies.append(float(attempt.get("duration_ms") or 0))
            input_count = float(attempt.get("input_tokens") or 0)
            output_count = float(attempt.get("output_tokens") or 0)
            input_tokens.append(input_count)
            output_tokens.append(output_count)
            invocation_tokens.append(input_count + output_count)
            counters["cap_hits"] += int(attempt.get("cap_hit") is True)
            counters["malformed_stopped"] += int(
                attempt.get("finish_reason") == "stop"
                and attempt.get("proposal") is None
                and attempt.get("parse_category") is not None
            )
        counters["infrastructure_failure"] += int(outcome == "INFRASTRUCTURE_FAILURE")
        counters["integrity_failure"] += int(outcome == "INTEGRITY_FAILURE")
        proposal = result.get("final_proposal")
        action = proposal.get("action") if isinstance(proposal, Mapping) else None
        target = gold_by_id[task_id]["target"]
        task_correct = False
        if case["task_type"] == "quant_metric":
            counters["quant_tasks"] += 1
            if target["status"] == "OK":
                counters["quant_answer_cases"] += 1
                task_correct = bool(
                    action == "ANSWER"
                    and _decimal(proposal.get("value")) == _decimal(target.get("value"))
                    and proposal.get("unit") == target.get("unit")
                )
                counters["quant_answer_correct"] += int(task_correct)
            else:
                counters["quant_refusal_cases"] += 1
                task_correct = bool(
                    action == "REFUSE"
                    and proposal.get("refusal_code") == "MISSING_INPUT"
                )
                counters["quant_refusal_correct"] += int(task_correct)
            counters["arithmetic_context_discrepancy"] += int(
                outcome == "RELEASED" and not task_correct
            )
        else:
            counters["narrative_tasks"] += 1
            subtype = str(case["narrative_subtype"])
            expected_answerability = (
                "ANSWERABLE" if target["status"] == "OK" else "UNANSWERABLE"
            )
            subtype_counts[subtype]["tasks"] += 1
            answerability_counts[expected_answerability]["tasks"] += 1
            expected_action = "ANSWER" if target["status"] == "OK" else "REFUSE"
            answerability_correct = action == expected_action
            counters["narrative_answerability_correct"] += int(answerability_correct)
            subtype_counts[subtype]["answerability_correct"] += int(
                answerability_correct
            )
            answerability_counts[expected_answerability]["correct"] += int(
                answerability_correct
            )
            if target["status"] == "REFUSAL":
                cause = str(
                    gold_by_id[task_id].get("evaluator_negative_cause") or "UNKNOWN"
                )
                negative_counts[cause]["tasks"] += 1
                negative_counts[cause]["correct"] += int(answerability_correct)
                task_correct = answerability_correct
                counters["narrative_refusal_correct"] += int(answerability_correct)
            elif isinstance(proposal, Mapping) and action == "ANSWER":
                citations_valid, exact_support = _released_support(
                    proposal, evidence_by_id[task_id]
                )
                anchor_match, answer_set_id = _matches_answer_set(
                    proposal,
                    gold_by_id[task_id].get("acceptable_answer_sets") or [],
                )
                counters["narrative_answer_proposals"] += 1
                counters["citation_valid"] += int(citations_valid)
                counters["exact_support"] += int(exact_support)
                counters["anchor_satisfied"] += int(anchor_match)
                subtype_counts[subtype]["answer_proposals"] += 1
                subtype_counts[subtype]["anchor_satisfied"] += int(anchor_match)
                task_correct = answerability_correct and exact_support and anchor_match
                if outcome == "RELEASED":
                    release_task_ids.add(task_id)
                    counters["unsupported_release"] += int(not exact_support)
                    counters["invalid_citation_release"] += int(not citations_valid)
                    counters["anchor_failure_release"] += int(not anchor_match)
                    release_review_records.append(
                        {
                            "task_id": task_id,
                            "question": case["question"],
                            "subtype": subtype,
                            "filing": case["filing"],
                            "released_extracts": _proposal_extracts(proposal),
                            "matched_acceptable_answer_set": answer_set_id,
                            "automatic_exact_support": exact_support,
                            "automatic_anchor_satisfaction": anchor_match,
                            "human_question_answer_check": "PENDING",
                        }
                    )
            else:
                task_correct = False
        counters["task_correct"] += int(task_correct)
        filing_scores[str(case["filing"]["filing_id"])].append(int(task_correct))
        verifier = record.get("verifier_result")
        if outcome in {"RELEASED", "SAFE_REFUSAL"} and (
            not isinstance(verifier, Mapping)
            or verifier.get("release_allowed") is not True
        ):
            counters["verifier_bypass"] += 1

    provisional = benchmark_manifest.get("benchmark_version") == BENCHMARK_VERSION_V24P
    if provisional:
        if human_release_review is not None:
            raise ValueError(
                "v2.4p cannot use the external-human gate review interface"
            )
        human_status = "NOT_APPLICABLE"
        human_review = None
        if provisional_release_review is None:
            provisional_status = "PENDING_PROVISIONAL_AI_REVIEW"
            provisional_review = None
        else:
            provisional_review = _read_json(provisional_release_review)
            provisional_status = validate_gate_provisional_review_v24p(
                provisional_review,
                benchmark_id=str(benchmark_manifest["benchmark_id"]),
                run_manifest_sha256=_sha256_path(run_manifest_path),
                release_task_ids=release_task_ids,
            )
    else:
        if provisional_release_review is not None:
            raise ValueError("Official v2.4 cannot use a provisional AI gate review")
        human_status, human_review = _validate_human_release_review(
            human_release_review,
            benchmark_id=str(benchmark_manifest["benchmark_id"]),
            run_manifest_sha256=_sha256_path(run_manifest_path),
            release_task_ids=release_task_ids,
        )
        provisional_status = "NOT_APPLICABLE"
        provisional_review = None
    hard_failures = {
        "missing_or_untyped_results": len(scoped_cases) - counters["typed"],
        "malformed_stopped_outputs": counters["malformed_stopped"],
        "cap_hits": counters["cap_hits"],
        "infrastructure_failures": counters["infrastructure_failure"],
        "integrity_failures": counters["integrity_failure"],
        "unsupported_releases": counters["unsupported_release"],
        "invalid_citation_releases": counters["invalid_citation_release"],
        "anchor_failure_releases": counters["anchor_failure_release"],
        "arithmetic_or_context_discrepancies": counters[
            "arithmetic_context_discrepancy"
        ],
        "verifier_bypass": counters["verifier_bypass"],
        "repair_limit_violations": counters["repair_limit_violation"],
    }
    technical_gate_pass = (
        selection_mode == "narrative_gate"
        and counters["tasks"] == 50
        and all(value == 0 for value in hard_failures.values())
    )
    active_review = provisional_review if provisional else human_review
    benchmark_defect_status = (
        str(active_review.get("benchmark_defect_status"))
        if active_review is not None
        else (
            "PENDING_PROVISIONAL_AI_REVIEW" if provisional else "PENDING_HUMAN_REVIEW"
        )
    )
    recommend_full = bool(
        technical_gate_pass
        and (
            provisional_status == "CONFIRMED"
            if provisional
            else human_status == "CONFIRMED"
        )
        and benchmark_defect_status == "NONE_FOUND"
    )
    filing_rates = [sum(values) / len(values) for values in filing_scores.values()]
    metrics = {
        "run_dir": str(run_dir),
        "run_manifest_sha256": _sha256_path(run_manifest_path),
        "selection_mode": selection_mode,
        "task_count": counters["tasks"],
        "technical_gate_pass": technical_gate_pass,
        "human_release_review_status": human_status,
        "provisional_release_review_status": provisional_status,
        "benchmark_defect_status": benchmark_defect_status,
        "recommend_full_500": recommend_full if not provisional else False,
        "recommend_exploratory_full_500": recommend_full if provisional else False,
        "assurance_status": (
            "PROVISIONAL_AI_REVIEW" if provisional else "EXTERNAL_HUMAN_APPROVED"
        ),
        "experiment_label": "EXPLORATORY" if provisional else None,
        "corpus_scope_label": "cached-20",
        "hard_gate_failures": hard_failures,
        "outcomes": {
            outcome: counters[f"outcome:{outcome}"]
            for outcome in (
                "RELEASED",
                "SAFE_REFUSAL",
                "MODEL_FAILURE",
                "INFRASTRUCTURE_FAILURE",
                "INTEGRITY_FAILURE",
            )
        },
        "quantitative": {
            "task_count": counters["quant_tasks"],
            "answer_accuracy": _rate(
                counters["quant_answer_correct"], counters["quant_answer_cases"]
            ),
            "refusal_accuracy": _rate(
                counters["quant_refusal_correct"], counters["quant_refusal_cases"]
            ),
        },
        "narrative": {
            "task_count": counters["narrative_tasks"],
            "answerability_accuracy": _rate(
                counters["narrative_answerability_correct"], counters["narrative_tasks"]
            ),
            "answer_anchor_satisfaction": _rate(
                counters["anchor_satisfied"], counters["narrative_answer_proposals"]
            ),
            "exact_support": _rate(
                counters["exact_support"], counters["narrative_answer_proposals"]
            ),
            "citation_validity": _rate(
                counters["citation_valid"], counters["narrative_answer_proposals"]
            ),
            "refusal_accuracy": _rate(
                counters["narrative_refusal_correct"],
                answerability_counts["UNANSWERABLE"]["tasks"],
            ),
            "by_subtype": {
                subtype: {
                    "tasks": values["tasks"],
                    "answerability_accuracy": _rate(
                        values["answerability_correct"], values["tasks"]
                    ),
                    "anchor_satisfaction": _rate(
                        values["anchor_satisfied"], values["answer_proposals"]
                    ),
                }
                for subtype, values in sorted(subtype_counts.items())
            },
            "by_answerability": {
                status: {
                    "tasks": values["tasks"],
                    "accuracy": _rate(values["correct"], values["tasks"]),
                }
                for status, values in sorted(answerability_counts.items())
            },
            "by_negative_type": {
                cause: {
                    "tasks": values["tasks"],
                    "accuracy": _rate(values["correct"], values["tasks"]),
                }
                for cause, values in sorted(negative_counts.items())
            },
        },
        "filing_clustered": {
            "filing_count": len(filing_scores),
            "macro_accuracy": sum(filing_rates) / len(filing_rates)
            if filing_rates
            else None,
            "minimum_filing_accuracy": min(filing_rates) if filing_rates else None,
            "maximum_filing_accuracy": max(filing_rates) if filing_rates else None,
            "uncertainty_note": "Task rows share 20 filing clusters; task-level rows are not independent.",
        },
        "latency_tokens_resources": {
            "model_invocations": len(latencies),
            "warm_generation_median_ms": _percentile(latencies, 0.5),
            "warm_generation_p95_ms": _percentile(latencies, 0.95),
            "input_tokens_median": _percentile(input_tokens, 0.5),
            "input_tokens_p95": _percentile(input_tokens, 0.95),
            "input_tokens_max": max(input_tokens) if input_tokens else None,
            "output_tokens_median": _percentile(output_tokens, 0.5),
            "tokens_per_invocation_median": _percentile(invocation_tokens, 0.5),
            "peak_vram_bytes_max": max(peak_vram) if peak_vram else None,
        },
    }
    return metrics, release_review_records


def _report(metrics: Mapping[str, Any]) -> str:
    provisional = metrics.get("evaluation_version") == EVALUATION_VERSION_V24P
    lines = [
        (
            "# AuditOps v2.4p exploratory cached-20 evaluation"
            if provisional
            else "# AuditOps v2.4 cached-20 evaluation"
        ),
        "",
        "This benchmark covers 20 frozen SEC filings. It is a benchmark of filing-verification behavior, not a broad S&P 500 estimate and not an audit opinion system.",
        "",
    ]
    if provisional:
        lines.extend(
            ["**EXPLORATORY — PROVISIONAL_AI_REVIEW — not human-approved.**", ""]
        )
    for run in metrics["runs"]:
        lines.extend(
            [
                f"## {Path(run['run_dir']).name}",
                "",
                f"- Selection: `{run['selection_mode']}` ({run['task_count']} tasks)",
                f"- Technical safety gate: **{'PASS' if run['technical_gate_pass'] else 'FAIL'}**",
                (
                    f"- Provisional AI released-quote review: `{run['provisional_release_review_status']}`"
                    if provisional
                    else f"- Human released-quote review: `{run['human_release_review_status']}`"
                ),
                f"- Benchmark defect review: `{run['benchmark_defect_status']}`",
                (
                    f"- Recommend exploratory full 500: **{'YES' if run['recommend_exploratory_full_500'] else 'NO'}**"
                    if provisional
                    else f"- Recommend full 500: **{'YES' if run['recommend_full_500'] else 'NO'}**"
                ),
                f"- Narrative answerability: `{run['narrative']['answerability_accuracy']}`",
                f"- Narrative anchor satisfaction: `{run['narrative']['answer_anchor_satisfaction']}`",
                f"- Quantitative answer/refusal accuracy: `{run['quantitative']['answer_accuracy']}` / `{run['quantitative']['refusal_accuracy']}`",
                "",
                "Hard-gate failures: `"
                + json.dumps(run["hard_gate_failures"], sort_keys=True)
                + "`",
                "",
            ]
        )
    authorized = (
        metrics.get("all_runs_authorized_for_exploratory_full_500")
        if provisional
        else metrics["all_runs_authorized_for_full_500"]
    )
    if not authorized:
        lines.extend(
            [
                "## Go/no-go status",
                "",
                (
                    "The exploratory full 500-case run requires a passing gate, confirmed provisional quote review, no discovered benchmark defect, and a content-bound GO_EXPLORATORY artifact."
                    if provisional
                    else "The full 500-case run is not authorized by this report. A passing narrative gate, complete human released-quote confirmation, no discovered benchmark defect, and explicit user authorization are all required."
                ),
                "",
            ]
        )
    return "\n".join(lines)


def write_evaluation_artifacts_v24(
    benchmark_dir: str | Path,
    run_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    human_release_reviews: Sequence[str | Path | None] | None = None,
    provisional_release_reviews: Sequence[str | Path | None] | None = None,
) -> dict[str, Any]:
    root = Path(benchmark_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite evaluation directory: {destination}"
        )
    manifest = _read_json(root / "benchmark_manifest.json")
    provisional = manifest.get("benchmark_version") == BENCHMARK_VERSION_V24P
    verification = (
        verify_agent_benchmark_v24p(root)
        if provisional
        else verify_agent_benchmark_v24(root)
    )
    if not verification["valid"]:
        raise ValueError("Invalid v2.4 benchmark: " + "; ".join(verification["errors"]))
    if not run_dirs:
        raise ValueError("At least one v2.4 run directory is required")
    reviews = list(human_release_reviews or [None] * len(run_dirs))
    provisional_reviews = list(provisional_release_reviews or [None] * len(run_dirs))
    if len(reviews) != len(run_dirs):
        raise ValueError("human_release_reviews must align one-for-one with run_dirs")
    if len(provisional_reviews) != len(run_dirs):
        raise ValueError(
            "provisional_release_reviews must align one-for-one with run_dirs"
        )
    cases = _read_jsonl(root / "cases.jsonl")
    evidence_by_id = {
        str(row["task_id"]): row["items"]
        for row in _read_jsonl(root / "evidence.jsonl")
    }
    gold_by_id = {str(row["task_id"]): row for row in _read_jsonl(root / "gold.jsonl")}
    runs: list[dict[str, Any]] = []
    release_records: list[dict[str, Any]] = []
    for run_dir, review, provisional_review in zip(
        run_dirs, reviews, provisional_reviews, strict=True
    ):
        run_metrics, records = _evaluate_run(
            root=root,
            benchmark_manifest=manifest,
            cases=cases,
            evidence_by_id=evidence_by_id,
            gold_by_id=gold_by_id,
            run_dir=Path(run_dir),
            human_release_review=Path(review) if review is not None else None,
            provisional_release_review=(
                Path(provisional_review) if provisional_review is not None else None
            ),
        )
        runs.append(run_metrics)
        release_records.extend(
            {"run_manifest_sha256": run_metrics["run_manifest_sha256"], **record}
            for record in records
        )
    metrics = {
        "evaluation_version": (
            EVALUATION_VERSION_V24P if provisional else EVALUATION_VERSION_V24
        ),
        "benchmark": {
            "benchmark_id": manifest["benchmark_id"],
            "benchmark_manifest_sha256": _sha256_path(root / "benchmark_manifest.json"),
            "label": "cached-20",
            "task_count": 500,
            "filing_count": 20,
            "generalization_warning": "Do not imply broad S&P 500 generalization.",
        },
        "runs": runs,
        "all_runs_authorized_for_full_500": (
            False if provisional else all(run["recommend_full_500"] for run in runs)
        ),
        "all_runs_authorized_for_exploratory_full_500": (
            all(run["recommend_exploratory_full_500"] for run in runs)
            if provisional
            else False
        ),
        "assurance_status": (
            "PROVISIONAL_AI_REVIEW" if provisional else "EXTERNAL_HUMAN_APPROVED"
        ),
        "experiment_label": "EXPLORATORY" if provisional else None,
    }
    release_packet = {
        "gate_release_review_packet_version": (
            "auditops-gate-provisional-review-packet.v2.4p"
            if provisional
            else "auditops-gate-release-review-packet.v2.4"
        ),
        "benchmark_id": manifest["benchmark_id"],
        "records": release_records,
        "instructions": (
            "Codex may record provisional AI feedback for every released quote; this is not external human approval."
            if provisional
            else "A human reviewer must confirm whether every released quote directly answers its task question and record any benchmark defect."
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        (temporary / "metrics.json").write_bytes(
            canonical_json_bytes(metrics, newline=True)
        )
        (temporary / "release_review_packet.json").write_bytes(
            canonical_json_bytes(release_packet, newline=True)
        )
        with (temporary / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "run",
                    "selection",
                    "technical_gate_pass",
                    "human_release_review",
                    "provisional_release_review",
                    "benchmark_defect_status",
                    "recommend_full_500",
                    "task_count",
                    "narrative_answerability",
                    "narrative_anchor_satisfaction",
                    "quant_answer_accuracy",
                    "quant_refusal_accuracy",
                    "filing_macro_accuracy",
                ]
            )
            for run in runs:
                writer.writerow(
                    [
                        Path(run["run_dir"]).name,
                        run["selection_mode"],
                        run["technical_gate_pass"],
                        run["human_release_review_status"],
                        run["provisional_release_review_status"],
                        run["benchmark_defect_status"],
                        run["recommend_full_500"],
                        run["task_count"],
                        run["narrative"]["answerability_accuracy"],
                        run["narrative"]["answer_anchor_satisfaction"],
                        run["quantitative"]["answer_accuracy"],
                        run["quantitative"]["refusal_accuracy"],
                        run["filing_clustered"]["macro_accuracy"],
                    ]
                )
        (temporary / "report.md").write_text(
            _report(metrics), encoding="utf-8", newline="\n"
        )
        assert_no_secrets([temporary])
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output_dir": str(destination),
        "metrics_sha256": _sha256_path(destination / "metrics.json"),
        "report_sha256": _sha256_path(destination / "report.md"),
        "all_runs_authorized_for_full_500": metrics["all_runs_authorized_for_full_500"],
        "all_runs_authorized_for_exploratory_full_500": metrics[
            "all_runs_authorized_for_exploratory_full_500"
        ],
    }


__all__ = [
    "EVALUATION_VERSION_V24",
    "EVALUATION_VERSION_V24P",
    "GATE_RELEASE_REVIEW_VERSION_V24",
    "write_evaluation_artifacts_v24",
]
