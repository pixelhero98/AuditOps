"""Evaluator and answer-first report for AuditOps v2.3 runs."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .agent_benchmark_v23 import verify_agent_benchmark_v23
from .agent_contracts import validate_agent_case_result
from .canonical_json import canonical_json_bytes, canonical_json_sha256
from .provenance import assert_no_secrets
from .text_support import is_contiguous_text_supported, normalize_support_text

EVALUATION_VERSION_V23 = "auditops.text_agent_evaluation.v2.3"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


def _sha256_path(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        materialized = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return materialized if materialized.is_finite() else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _cluster_interval(values_by_cluster: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    cluster_rates = [
        statistics.mean(values) for values in values_by_cluster.values() if values
    ]
    if not cluster_rates:
        return {"filing_count": 0, "macro_rate": None, "bootstrap_95": [None, None]}
    generator = random.Random(20_260_821)
    samples = [
        statistics.mean(generator.choice(cluster_rates) for _ in cluster_rates)
        for _ in range(2_000)
    ]
    return {
        "filing_count": len(cluster_rates),
        "macro_rate": statistics.mean(cluster_rates),
        "bootstrap_95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
    }


def _narrative_support(
    proposal: Mapping[str, Any], evidence_items: Sequence[Mapping[str, Any]]
) -> tuple[bool, bool]:
    evidence_by_id = {str(item["evidence_id"]): item for item in evidence_items}
    cited_ids = [str(item) for item in proposal.get("evidence_ids") or []]
    citations_valid = bool(cited_ids) and all(
        evidence_id in evidence_by_id for evidence_id in cited_ids
    )
    claims = proposal.get("claims")
    if not citations_valid or not isinstance(claims, list) or not claims:
        return citations_valid, False
    supported = all(
        isinstance(claim, Mapping)
        and len(claim.get("evidence_ids") or []) == 1
        and str(claim["evidence_ids"][0]) in evidence_by_id
        and normalize_support_text(str(claim.get("text") or ""))
        == normalize_support_text(str(claim.get("supporting_text") or ""))
        and is_contiguous_text_supported(
            str(claim.get("text") or ""),
            str(evidence_by_id[str(claim["evidence_ids"][0])]["content"]),
        )
        for claim in claims
    )
    return citations_valid, supported


def _run_label(manifest: Mapping[str, Any], index: int) -> str:
    quantization = manifest.get("quantization") or "bf16"
    return "/".join(
        (
            str(manifest.get("model_id") or f"run-{index}"),
            str(quantization),
            str(manifest.get("runtime_mode")),
            str(manifest.get("prompt_condition")),
        )
    )


def _bind_evaluation_scope(
    *,
    root: Path,
    benchmark_manifest: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> tuple[str, list[Mapping[str, Any]]]:
    """Bind a run to either the full benchmark or its exact frozen parity slice."""

    if run_manifest.get("benchmark_id") != benchmark_manifest.get("benchmark_id"):
        raise ValueError("Run benchmark_id does not match the evaluated benchmark")
    if run_manifest.get("benchmark_manifest_sha256") != _sha256_path(
        root / "benchmark_manifest.json"
    ):
        raise ValueError("Run benchmark manifest hash mismatch")
    full_ids = [str(case["task_id"]) for case in cases]
    result_ids = [str(row["task_id"]) for row in results]
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("Run results contain duplicate task IDs")
    if run_manifest.get("case_count") != len(result_ids):
        raise ValueError("Run case_count does not match results")
    if run_manifest.get("full_benchmark_case_count") != len(full_ids):
        raise ValueError("Run full benchmark case count mismatch")
    if run_manifest.get("task_ids_sha256") != canonical_json_sha256(result_ids):
        raise ValueError("Run task membership hash mismatch")

    inputs = run_manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError("Run inputs must be an object")
    for name in ("cases.jsonl", "evidence.jsonl"):
        field = name.removesuffix(".jsonl") + "_sha256"
        if inputs.get(field) != _sha256_path(root / name):
            raise ValueError(f"Run {field} does not match the frozen benchmark")

    complete = run_manifest.get("complete_full_benchmark")
    if not isinstance(complete, bool):
        raise TypeError("Run complete_full_benchmark must be a boolean")
    if complete != (len(result_ids) == len(full_ids)):
        raise ValueError("Run complete_full_benchmark is inconsistent with membership")
    parity_hash = inputs.get("parity_membership_sha256")
    if complete:
        if result_ids != full_ids:
            raise ValueError("Full run task IDs do not match benchmark order")
        if parity_hash is not None:
            raise ValueError("Full runs cannot bind a parity membership artifact")
        return "full", list(cases)

    parity_path = root / "parity_50.json"
    if parity_hash != _sha256_path(parity_path):
        raise ValueError("Partial run does not bind the frozen parity artifact")
    parity = _read_json(parity_path)
    parity_ids = parity.get("task_ids")
    if (
        not isinstance(parity_ids, list)
        or not parity_ids
        or len(parity_ids) != len(set(parity_ids))
        or any(not isinstance(task_id, str) or not task_id for task_id in parity_ids)
    ):
        raise ValueError("Frozen parity membership must contain unique task IDs")
    if parity.get("case_count") != len(parity_ids):
        raise ValueError("Frozen parity case_count is inconsistent")
    if parity.get("task_ids_sha256") != canonical_json_sha256(parity_ids):
        raise ValueError("Frozen parity task membership hash mismatch")
    if result_ids != parity_ids:
        raise ValueError("Run task IDs do not match frozen parity membership order")
    cases_by_id = {str(case["task_id"]): case for case in cases}
    if any(task_id not in cases_by_id for task_id in parity_ids):
        raise ValueError("Frozen parity membership contains an unknown task ID")
    return "frozen_parity", [cases_by_id[task_id] for task_id in parity_ids]


def _evaluate_one(
    *,
    root: Path,
    benchmark_manifest: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    evidence_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
    run_dir: Path,
    index: int,
) -> dict[str, Any]:
    run_manifest = _read_json(run_dir / "run_manifest.json")
    if run_manifest.get("batch_run_version") != "auditops-agent-batch.v2.3":
        raise ValueError(f"Run is not a v2.3 batch: {run_dir}")
    results_path = run_dir / "results.jsonl"
    if _sha256_path(results_path) != run_manifest.get("results", {}).get("sha256"):
        raise ValueError(f"Run results hash mismatch: {run_dir}")
    results = _read_jsonl(results_path)
    evaluation_scope, scoped_cases = _bind_evaluation_scope(
        root=root,
        benchmark_manifest=benchmark_manifest,
        cases=cases,
        run_manifest=run_manifest,
        results=results,
    )
    result_by_id = {str(row["task_id"]): row for row in results}
    expected_ids = [str(case["task_id"]) for case in scoped_cases]
    missing = sorted(set(expected_ids) - set(result_by_id))
    extra = sorted(set(result_by_id) - set(expected_ids))
    if missing or extra:
        raise ValueError(
            f"Run membership mismatch: missing={len(missing)} extra={len(extra)}"
        )

    counters: defaultdict[str, int] = defaultdict(int)
    by_subtype: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    by_negative: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    cluster_accuracy: defaultdict[str, list[int]] = defaultdict(list)
    generation_latencies: list[float] = []
    invocation_tokens: list[int] = []
    engine_initialization_ms: list[float] = []

    for case in scoped_cases:
        task_id = str(case["task_id"])
        result = result_by_id[task_id]
        validate_agent_case_result(result)
        counters["tasks"] += 1
        counters["typed"] += 1
        outcome = str(result["outcome"])
        if outcome in {"RELEASED", "SAFE_REFUSAL"}:
            counters["verifier_accepted"] += 1
        if outcome == "RELEASED":
            counters["released"] += 1
        elif outcome == "SAFE_REFUSAL":
            counters["safe_refusal"] += 1
        else:
            counters["typed_failure"] += 1
        record = result["run_record"]
        engine_initialization_ms.append(
            float(record["timings"]["engine_initialization_ms"])
        )
        for attempt in record["verifier_trace"]:
            generation_latencies.append(float(attempt["duration_ms"]))
            invocation_tokens.append(
                int(attempt["input_tokens"]) + int(attempt["output_tokens"])
            )

        gold = gold_by_id[task_id]
        target = gold["target"]
        proposal = result.get("final_proposal")
        proposal_action = (
            proposal.get("action") if isinstance(proposal, Mapping) else None
        )
        expected_action = "ANSWER" if target["status"] == "OK" else "REFUSE"
        answerability_correct = proposal_action == expected_action
        filing_id = str(case["filing"]["filing_id"])
        task_correct = False

        if case["task_type"] == "quant_metric":
            counters["quant_tasks"] += 1
            if target["status"] == "OK":
                counters["quant_answer_cases"] += 1
                task_correct = bool(
                    proposal_action == "ANSWER"
                    and _decimal(proposal.get("value")) == _decimal(target.get("value"))
                    and proposal.get("unit") == target.get("unit")
                )
                counters["quant_answer_correct"] += int(task_correct)
            else:
                counters["quant_refusal_cases"] += 1
                task_correct = bool(
                    proposal_action == "REFUSE"
                    and proposal.get("refusal_code") == target.get("refusal_code")
                )
                counters["quant_refusal_correct"] += int(task_correct)
            if outcome == "RELEASED" and not task_correct:
                counters["arithmetic_or_context_discrepancy"] += 1
        else:
            counters["narrative_tasks"] += 1
            subtype = str(case["narrative_subtype"])
            subtype_counts = by_subtype[subtype]
            subtype_counts["tasks"] += 1
            counters["narrative_answerability_correct"] += int(answerability_correct)
            subtype_counts["answerability_correct"] += int(answerability_correct)
            negative_cause = str(gold.get("evaluator_negative_cause") or "NONE")
            citations_valid = False
            exact_support = False
            if proposal_action == "ANSWER":
                citations_valid, exact_support = _narrative_support(
                    proposal, evidence_by_id[task_id]
                )
                if outcome == "RELEASED":
                    counters["invalid_citation_release"] += int(not citations_valid)
                    counters["unsupported_release"] += int(not exact_support)
            if target["status"] == "REFUSAL":
                by_negative[negative_cause]["tasks"] += 1
                by_negative[negative_cause]["answerability_correct"] += int(
                    answerability_correct
                )
                task_correct = answerability_correct
            elif proposal_action == "ANSWER":
                counters["narrative_answer_cases"] += 1
                counters["citation_valid"] += int(citations_valid)
                counters["exact_extractive_support"] += int(exact_support)
                subtype_counts["answer_cases"] += 1
                subtype_counts["exact_extractive_support"] += int(exact_support)
                gold_ids = {
                    str(item) for item in target.get("chunk_evidence_ids") or []
                }
                proposal_ids = {
                    str(item) for item in proposal.get("evidence_ids") or []
                }
                citation_agreement = proposal_ids == gold_ids
                preferred_exact = normalize_support_text(
                    str(proposal.get("answer_text") or "")
                ) == normalize_support_text(str(target.get("answer_text") or ""))
                counters["citation_agreement"] += int(citation_agreement)
                counters["preferred_wording_exact"] += int(preferred_exact)
                subtype_counts["citation_agreement"] += int(citation_agreement)
                subtype_counts["preferred_wording_exact"] += int(preferred_exact)
                task_correct = answerability_correct and exact_support
            else:
                task_correct = False
            if (
                outcome not in {"RELEASED", "SAFE_REFUSAL"}
                and isinstance(proposal, Mapping)
                and proposal.get("evidence_ids")
            ):
                counters["rejected_proposals_with_citations"] += 1

        counters["diagnostic_proposal_correct"] += int(task_correct)
        released_correct = outcome in {"RELEASED", "SAFE_REFUSAL"} and task_correct
        counters["released_output_correct"] += int(released_correct)
        cluster_accuracy[filing_id].append(int(task_correct))

    hard_gate_failures = {
        "unsupported_releases": counters["unsupported_release"],
        "invalid_citation_releases": counters["invalid_citation_release"],
        "arithmetic_or_context_discrepancies": counters[
            "arithmetic_or_context_discrepancy"
        ],
        "silent_or_untyped_results": counters["tasks"] - counters["typed"],
    }
    return {
        "run_label": _run_label(run_manifest, index),
        "run_dir": str(run_dir),
        "run_manifest_sha256": _sha256_path(run_dir / "run_manifest.json"),
        "model_id": run_manifest.get("model_id"),
        "model_revision": run_manifest.get("model_revision"),
        "quantization": run_manifest.get("quantization"),
        "runtime_mode": run_manifest.get("runtime_mode"),
        "prompt_condition": run_manifest.get("prompt_condition"),
        "evaluation_scope": evaluation_scope,
        "full_benchmark_task_count": len(cases),
        "task_count": counters["tasks"],
        "hard_safety_pass": all(value == 0 for value in hard_gate_failures.values()),
        "hard_gate_failures": hard_gate_failures,
        "core": {
            "typed_result_rate": _rate(counters["typed"], counters["tasks"]),
            "verifier_acceptance": _rate(
                counters["verifier_accepted"], counters["tasks"]
            ),
            "released_answer_coverage": _rate(counters["released"], counters["tasks"]),
            "safe_refusal_coverage": _rate(counters["safe_refusal"], counters["tasks"]),
            "typed_failure_rate": _rate(counters["typed_failure"], counters["tasks"]),
        },
        "quantitative": {
            "answer_accuracy": _rate(
                counters["quant_answer_correct"], counters["quant_answer_cases"]
            ),
            "refusal_accuracy": _rate(
                counters["quant_refusal_correct"], counters["quant_refusal_cases"]
            ),
            "answer_cases": counters["quant_answer_cases"],
            "refusal_cases": counters["quant_refusal_cases"],
        },
        "narrative": {
            "answerability_accuracy": _rate(
                counters["narrative_answerability_correct"], counters["narrative_tasks"]
            ),
            "exact_extractive_support": _rate(
                counters["exact_extractive_support"], counters["narrative_answer_cases"]
            ),
            "citation_agreement": _rate(
                counters["citation_agreement"], counters["narrative_answer_cases"]
            ),
            "preferred_wording_exactness": _rate(
                counters["preferred_wording_exact"], counters["narrative_answer_cases"]
            ),
            "rejected_proposals_with_citations": counters[
                "rejected_proposals_with_citations"
            ],
            "by_subtype": {
                subtype: {
                    "tasks": values["tasks"],
                    "answerability_accuracy": _rate(
                        values["answerability_correct"], values["tasks"]
                    ),
                    "exact_extractive_support": _rate(
                        values["exact_extractive_support"], values["answer_cases"]
                    ),
                    "citation_agreement": _rate(
                        values["citation_agreement"], values["answer_cases"]
                    ),
                    "preferred_wording_exactness": _rate(
                        values["preferred_wording_exact"], values["answer_cases"]
                    ),
                }
                for subtype, values in sorted(by_subtype.items())
            },
            "by_evaluator_negative_cause": {
                cause: {
                    "tasks": values["tasks"],
                    "answerability_accuracy": _rate(
                        values["answerability_correct"], values["tasks"]
                    ),
                }
                for cause, values in sorted(by_negative.items())
            },
        },
        "filing_clustered": _cluster_interval(cluster_accuracy),
        "latency": {
            "engine_initialization_ms_total": sum(engine_initialization_ms),
            "warm_generation_median_ms": _percentile(generation_latencies, 0.5),
            "warm_generation_p95_ms": _percentile(generation_latencies, 0.95),
            "model_invocation_count": len(generation_latencies),
            "tokens_per_invocation_median": _percentile(
                [float(value) for value in invocation_tokens], 0.5
            ),
        },
        "diagnostic_compatibility": {
            "proposal_accuracy": _rate(
                counters["diagnostic_proposal_correct"], counters["tasks"]
            ),
            "released_output_accuracy": _rate(
                counters["released_output_correct"], counters["tasks"]
            ),
            "note": "Retained for compatibility; not a headline narrative/model comparison metric.",
        },
    }


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.1f}%"


def _report(metrics: Mapping[str, Any]) -> str:
    benchmark = metrics["benchmark"]
    lines = [
        "# AuditOps v2.3 safety-hybrid evaluation",
        "",
        "v2.3 keeps quantitative safety-hybrid execution deterministic and evaluates narrative models as bounded exact-quote selectors. Aggregate proposal accuracy is diagnostic only; the primary narrative measures are answerability, exact support, citation agreement, and preferred wording.",
        "",
        f"Benchmark: `{benchmark['benchmark_id']}` — {benchmark['task_count']} frozen tasks from {benchmark['filing_count']} filings. Each row states whether it covers the full benchmark or the exact frozen parity slice. Filing-clustered results are reported because task rows from the same filing are not independent.",
        "",
        "| Run | Scope | Cases | Hard safety | Typed | Verifier | Quant answer | Quant refusal | Narrative answerability | Exact support | Citation agreement | Preferred wording |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in metrics["runs"]:
        lines.append(
            "| {label} | {scope} | {cases} | {safety} | {typed} | {verifier} | {qa} | {qr} | {na} | {support} | {citation} | {wording} |".format(
                label=run["run_label"],
                scope=run["evaluation_scope"],
                cases=run["task_count"],
                safety="PASS" if run["hard_safety_pass"] else "FAIL",
                typed=_fmt(run["core"]["typed_result_rate"]),
                verifier=_fmt(run["core"]["verifier_acceptance"]),
                qa=_fmt(run["quantitative"]["answer_accuracy"]),
                qr=_fmt(run["quantitative"]["refusal_accuracy"]),
                na=_fmt(run["narrative"]["answerability_accuracy"]),
                support=_fmt(run["narrative"]["exact_extractive_support"]),
                citation=_fmt(run["narrative"]["citation_agreement"]),
                wording=_fmt(run["narrative"]["preferred_wording_exactness"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "`NO_DIRECT_EVIDENCE` is the only model-facing absence-of-support refusal. The evaluator retains causal negative types separately; the model is never asked to distinguish retrieval miss from filing-wide absence.",
            "",
            "`capability_agent` remains a research ablation: its single legal tool call is schema-forced, so it is not presented as autonomous planning. Few-shot comparisons remain blocked until the v2.3 rendered packs receive human approval.",
            "",
            "## Latency",
            "",
            "Engine initialization is separated from warm generation. Warm medians and p95 values are per actual model invocation, not per benchmark task.",
            "",
            "## Compatibility diagnostics",
            "",
            "Proposal and released-output accuracy remain in `metrics.json` for historical comparison but are deliberately not used as the headline narrative/model score.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_artifacts_v23(
    benchmark_dir: str | Path,
    run_dirs: Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(benchmark_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite evaluation directory: {destination}"
        )
    verification = verify_agent_benchmark_v23(root)
    if not verification["valid"]:
        raise ValueError("Invalid v2.3 benchmark: " + "; ".join(verification["errors"]))
    manifest = _read_json(root / "benchmark_manifest.json")
    cases = _read_jsonl(root / "cases.jsonl")
    evidence_by_id = {
        str(row["task_id"]): row["items"]
        for row in _read_jsonl(root / "evidence.jsonl")
    }
    gold_by_id = {str(row["task_id"]): row for row in _read_jsonl(root / "gold.jsonl")}
    runs = [
        _evaluate_one(
            root=root,
            benchmark_manifest=manifest,
            cases=cases,
            evidence_by_id=evidence_by_id,
            gold_by_id=gold_by_id,
            run_dir=Path(run_dir),
            index=index,
        )
        for index, run_dir in enumerate(run_dirs, start=1)
    ]
    metrics = {
        "evaluation_version": EVALUATION_VERSION_V23,
        "benchmark": {
            "benchmark_id": manifest["benchmark_id"],
            "benchmark_manifest_sha256": _sha256_path(root / "benchmark_manifest.json"),
            "task_count": len(cases),
            "filing_count": len({str(case["filing"]["filing_id"]) for case in cases}),
            "cached_source_warning": "Tasks sharing a filing are not independent observations.",
        },
        "runs": runs,
        "all_hard_safety_pass": all(run["hard_safety_pass"] for run in runs),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        (temporary / "metrics.json").write_bytes(canonical_json_bytes(metrics) + b"\n")
        with (temporary / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "run_label",
                    "hard_safety_pass",
                    "typed_result_rate",
                    "verifier_acceptance",
                    "quant_answer_accuracy",
                    "quant_refusal_accuracy",
                    "narrative_answerability_accuracy",
                    "exact_extractive_support",
                    "citation_agreement",
                    "preferred_wording_exactness",
                    "filing_macro_rate",
                ]
            )
            for run in runs:
                writer.writerow(
                    [
                        run["run_label"],
                        run["hard_safety_pass"],
                        run["core"]["typed_result_rate"],
                        run["core"]["verifier_acceptance"],
                        run["quantitative"]["answer_accuracy"],
                        run["quantitative"]["refusal_accuracy"],
                        run["narrative"]["answerability_accuracy"],
                        run["narrative"]["exact_extractive_support"],
                        run["narrative"]["citation_agreement"],
                        run["narrative"]["preferred_wording_exactness"],
                        run["filing_clustered"]["macro_rate"],
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
        "all_hard_safety_pass": metrics["all_hard_safety_pass"],
    }


__all__ = ["EVALUATION_VERSION_V23", "write_evaluation_artifacts_v23"]
