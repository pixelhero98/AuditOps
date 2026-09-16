from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from auditops.agent_batch import BATCH_RUN_VERSION, FEW_SHOT_APPROVAL_VERSION
from auditops.agent_benchmark import build_agent_benchmark
from auditops.agent_contracts import (
    build_model_config,
)
from auditops.agent_evaluation import (
    _narrative_release_supported,
    _quant_joint_match,
    _routing_match,
    _verifier_failure_codes,
    build_matched_baseline_comparisons,
    compare_quantization_runs,
    evaluate_agent_run,
    write_evaluation_artifacts,
)
from auditops.agent_operations import expected_tool_arguments
from auditops.agent_runtime import run_agent_case
from auditops.agent_tools import execute_deterministic_tool
from auditops.model_adapter import (
    MockModelAdapter,
    ModelBackendError,
    ModelGenerationError,
    ModelOOMError,
    ModelTimeoutError,
)


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_v22_evaluator_supports_ordered_exact_extracts_across_chunks():
    first = "The financial statements present fairly in all material respects."
    second = "Internal control was effective in all material respects."
    answer = {
        "status": "OK",
        "answer_text": f"{first}\n\n{second}",
        "evidence_ids": ["chunk-1", "chunk-2"],
        "claims": [
            {
                "claim_id": "1",
                "text": first,
                "supporting_text": first,
                "evidence_ids": ["chunk-1"],
            },
            {
                "claim_id": "2",
                "text": second,
                "supporting_text": second,
                "evidence_ids": ["chunk-2"],
            },
        ],
    }
    evidence = [
        {"evidence_id": "chunk-1", "content": f"Opinion. {first}"},
        {"evidence_id": "chunk-2", "content": f"Controls. {second}"},
    ]

    assert _narrative_release_supported(answer, evidence, reconstructed_extracts=True)

    reordered = copy.deepcopy(answer)
    reordered["answer_text"] = f"{second}\n\n{first}"
    assert not _narrative_release_supported(
        reordered, evidence, reconstructed_extracts=True
    )


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _quant_fixture(index: int) -> dict:
    status = "OK" if index % 2 == 0 else "REFUSAL"
    metric_spec_id = "current_ratio" if index % 4 < 2 else "working_capital"
    assets = Decimal(1000 + index * 50)
    liabilities = Decimal(500)
    period_key = "ASOF_20251231"
    inputs = [
        {
            "input_name": "assets_current",
            "concept_norm": "uk-core:CurrentAssets",
            "period_key": period_key,
            "unit_canon": "gbp",
            "value_num_exact": str(assets),
            "fact_evidence_ids": [f"assets::{index}"],
        }
    ]
    evidence_items = [
        {
            "evidence_id": f"assets::{index}",
            "filing_id": f"q-fil-{index}",
            "content": f"Current assets: {assets}",
            "period_key": period_key,
            "unit": "gbp",
            "value": str(assets),
            "source_system": "Companies-House",
            "metadata": {
                "input_name": "assets_current",
                "concept_norm": "uk-core:CurrentAssets",
                "context_id": f"ctx-{index}",
                "entity_id": f"Q{index}",
                "dimensions": {},
            },
        }
    ]
    if status == "OK":
        inputs.append(
            {
                "input_name": "liabilities_current",
                "concept_norm": "uk-core:CreditorsAmountsFallingDueWithinOneYear",
                "period_key": period_key,
                "unit_canon": "gbp",
                "value_num_exact": str(liabilities),
                "fact_evidence_ids": [f"creditors::{index}"],
            }
        )
        evidence_items.append(
            {
                "evidence_id": f"creditors::{index}",
                "filing_id": f"q-fil-{index}",
                "content": f"Creditors falling due within one year: {liabilities}",
                "period_key": period_key,
                "unit": "gbp",
                "value": str(liabilities),
                "source_system": "Companies-House",
                "metadata": {
                    "input_name": "liabilities_current",
                    "concept_norm": "uk-core:CreditorsAmountsFallingDueWithinOneYear",
                    "context_id": f"ctx-{index}",
                    "entity_id": f"Q{index}",
                    "dimensions": {},
                },
            }
        )
    if status == "OK" and metric_spec_id == "current_ratio":
        value = str(assets / liabilities)
    elif status == "OK":
        value = str(assets - liabilities)
    else:
        value = None
    evidence_ids = [f"assets::{index}", f"creditors::{index}"] if status == "OK" else []
    return {
        "task_id": f"q-{index}",
        "template_id": f"quant-{status.lower()}-{(index // 2) % 8}",
        "task_family": f"quant_metric:{metric_spec_id}:{status.lower()}",
        "filing_id": f"q-fil-{index}",
        "ticker": f"Q{index}",
        "metric_spec_id": metric_spec_id,
        "metric_kind": "ratio" if metric_spec_id == "current_ratio" else "difference",
        "executor_op": "divide" if metric_spec_id == "current_ratio" else "subtract",
        "period": {
            "period_type": "ASOF",
            "period_key": period_key,
            "end_date": "2025-12-31",
        },
        "question": f"quant {index}",
        "canonical_inputs": inputs,
        "evidence_items": evidence_items,
        "target_status": status,
        "target_answer": {
            "task_id": f"q-{index}",
            "metric_spec_id": metric_spec_id,
            "filing_id": f"q-fil-{index}",
            "status": status,
            "value": value,
            "unit": "pure"
            if status == "OK" and metric_spec_id == "current_ratio"
            else "gbp"
            if status == "OK"
            else None,
            "period_key": period_key,
            "evidence_ids": evidence_ids,
            "refusal_code": None if status == "OK" else "MISSING_INPUT",
        },
        "negative_type": None if status == "OK" else "missing_input",
        "refusal_policy": {
            "allowed_codes": [
                "AMBIGUOUS_CONTEXT",
                "INCOMPATIBLE_UNITS",
                "MISSING_INPUT",
                "PERIOD_NOT_SUPPORTED",
                "UNSUPPORTED_REQUEST",
                "ZERO_DENOMINATOR",
            ]
        },
    }


def _narrative_fixture(index: int) -> dict:
    answerable = index % 2 == 0
    status = "OK" if answerable else "REFUSAL"
    return {
        "task_id": f"n-{index}",
        "template_id": f"narrative-{status.lower()}-{(index // 2) % 8}",
        "task_family": f"narrative_citation:footnote_note:{status.lower()}",
        "filing_id": f"n-fil-{index}",
        "ticker": f"N{index}",
        "period_key": "FY2025",
        "form_type": "10-K",
        "question": f"narrative {index}",
        "label": "footnote_note",
        "answerability": "ANSWERABLE" if answerable else "UNANSWERABLE",
        "expected_chunk_ids": [f"chunk::{index}"] if answerable else [],
        "extractive_answer": f"answer {index}" if answerable else None,
        "refusal_code": None if answerable else "NARRATIVE_NOT_SUPPORTED",
        "negative_type": None if answerable else "unsupported_attribute",
        "evidence_items": [
            {
                "evidence_id": f"chunk::{index}",
                "filing_id": f"n-fil-{index}",
                "content": f"answer {index}"
                if answerable
                else f"irrelevant text {index}",
                "period_key": "FY2025",
                "source_system": "Companies-House",
            }
        ],
    }


def _benchmark(
    tmp_path: Path, *, quant_count: int = 6, narrative_count: int = 4
) -> Path:
    quant_path = tmp_path / "quant.jsonl"
    narrative_path = tmp_path / "narrative.jsonl"
    source_count = max(80, quant_count + 20, narrative_count + 20)
    _write_jsonl(quant_path, [_quant_fixture(index) for index in range(source_count)])
    _write_jsonl(
        narrative_path,
        [_narrative_fixture(index) for index in range(source_count)],
    )
    output = tmp_path / "benchmark"
    build_agent_benchmark(
        quant_path,
        narrative_path,
        output,
        benchmark_profile="synthetic",
        jurisdiction="UK",
        corpus_id="companies-house-fixture",
        reporting_framework="UK-GAAP",
        standards_version="ISA-UK-current",
        source_system="Companies-House",
        quant_count=quant_count,
        narrative_count=narrative_count,
    )
    return output


def _proposal_for_target(task: dict, target: dict) -> dict:
    if target["status"] == "REFUSAL":
        return {
            "task_id": task["task_id"],
            "action": "REFUSE",
            "period_key": target["period_key"],
            "refusal_code": target["refusal_code"],
        }
    evidence_ids = list(
        target.get("evidence_ids") or target.get("chunk_evidence_ids") or []
    )
    answer_text = target.get("answer_text")
    if answer_text is not None:
        return {
            "task_id": task["task_id"],
            "action": "ANSWER",
            "period_key": target["period_key"],
            "extracts": [{"evidence_id": evidence_ids[0], "exact_quote": answer_text}],
        }
    response = {
        "task_id": task["task_id"],
        "action": "ANSWER",
        "period_key": target["period_key"],
        "evidence_ids": evidence_ids,
    }
    response.update(value=target.get("value"), unit=target.get("unit"))
    return response


def _perfect_rows(
    benchmark: Path, *, model_id: str = "fixture/model", quantization: str = "bf16"
) -> list[dict]:
    cases = _read_jsonl(benchmark / "cases.jsonl")
    gold = {row["task_id"]: row for row in _read_jsonl(benchmark / "gold.jsonl")}
    evidence = {
        row["task_id"]: row["items"]
        for row in _read_jsonl(benchmark / "evidence.jsonl")
    }
    benchmark_sha256 = _file_sha256(benchmark / "benchmark_manifest.json")
    model_config = build_model_config(
        model_id=model_id,
        revision="fixture-revision",
        quantization=quantization,
        backend="mock",
    )
    rows = []
    for task in cases:
        task_id = task["task_id"]
        proposal = _proposal_for_target(task, gold[task_id]["target"])
        adapter = MockModelAdapter(
            [proposal],
            model_id=model_config["model_id"],
            revision=model_config["revision"],
        )
        result = run_agent_case(
            task,
            adapter=adapter,
            model_config=model_config,
            corpus_id="companies-house-fixture",
            benchmark_manifest_sha256=benchmark_sha256,
            runtime_mode="direct",
            prompt_condition="zero_shot",
            evidence_items=evidence[task_id],
            token_counter=lambda _text: 0,
        )
        rows.append(result.to_dict())
    return rows


def _agent_rows(benchmark: Path) -> list[dict]:
    cases = {row["task_id"]: row for row in _read_jsonl(benchmark / "cases.jsonl")}
    gold = {row["task_id"]: row for row in _read_jsonl(benchmark / "gold.jsonl")}
    evidence = {
        row["task_id"]: row["items"]
        for row in _read_jsonl(benchmark / "evidence.jsonl")
    }
    benchmark_sha256 = _file_sha256(benchmark / "benchmark_manifest.json")
    model_config = build_model_config(
        model_id="fixture/model",
        revision="fixture-revision",
        quantization="bf16",
        backend="mock",
    )
    rows: list[dict] = []
    for task_id, task in cases.items():
        tool_name = task["allowed_tools"][0]
        tool_arguments = expected_tool_arguments(task)
        plan = {
            "task_id": task_id,
            "action": "CALL_TOOL",
            "tool_name": tool_name,
            "tool_arguments": tool_arguments,
        }
        final = _proposal_for_target(task, gold[task_id]["target"])
        adapter = MockModelAdapter(
            [plan, final],
            model_id=model_config["model_id"],
            revision=model_config["revision"],
        )
        result = run_agent_case(
            task,
            adapter=adapter,
            model_config=model_config,
            corpus_id="companies-house-fixture",
            benchmark_manifest_sha256=benchmark_sha256,
            runtime_mode="capability_agent",
            prompt_condition="zero_shot",
            evidence_items=evidence[task_id],
            tool_executor=lambda name, arguments, task_input, items=evidence[task_id]: (
                execute_deterministic_tool(name, arguments, task_input, items)
            ),
            token_counter=lambda _text: 0,
        )
        rows.append(result.to_dict())
    return rows


def _hybrid_rows(benchmark: Path) -> list[dict]:
    cases = _read_jsonl(benchmark / "cases.jsonl")
    gold = {row["task_id"]: row for row in _read_jsonl(benchmark / "gold.jsonl")}
    evidence = {
        row["task_id"]: row["items"]
        for row in _read_jsonl(benchmark / "evidence.jsonl")
    }
    benchmark_sha256 = _file_sha256(benchmark / "benchmark_manifest.json")
    model_config = build_model_config(
        model_id="fixture/model",
        revision="fixture-revision",
        quantization="bf16",
        backend="mock",
    )
    rows: list[dict] = []
    for task in cases:
        task_id = task["task_id"]
        proposal = _proposal_for_target(task, gold[task_id]["target"])
        adapter = MockModelAdapter(
            [proposal],
            model_id=model_config["model_id"],
            revision=model_config["revision"],
        )
        result = run_agent_case(
            task,
            adapter=adapter,
            model_config=model_config,
            corpus_id="companies-house-fixture",
            benchmark_manifest_sha256=benchmark_sha256,
            runtime_mode="safety_hybrid",
            prompt_condition="zero_shot",
            evidence_items=evidence[task_id],
            tool_executor=lambda name, arguments, task_input, items=evidence[task_id]: (
                execute_deterministic_tool(name, arguments, task_input, items)
            ),
            token_counter=lambda _text: 0,
        )
        rows.append(result.to_dict())
    return rows


def _few_shot_rows(benchmark: Path) -> list[dict]:
    cases = {row["task_id"]: row for row in _read_jsonl(benchmark / "cases.jsonl")}
    gold = {row["task_id"]: row for row in _read_jsonl(benchmark / "gold.jsonl")}
    evidence = {
        row["task_id"]: row["items"]
        for row in _read_jsonl(benchmark / "evidence.jsonl")
    }
    examples = [
        {
            key: row[key]
            for key in (
                "few_shot_example_version",
                "example_id",
                "template_id",
                "task_family",
                "task",
                "evidence_items",
                "plan_response",
                "tool_observation",
                "assistant_response",
                "source_task_sha256",
            )
        }
        for row in _read_jsonl(benchmark / "few_shot.jsonl")
    ]
    benchmark_sha256 = _file_sha256(benchmark / "benchmark_manifest.json")
    model_config = build_model_config(
        model_id="fixture/model",
        revision="fixture-revision",
        quantization="bf16",
        backend="mock",
    )
    rows: list[dict] = []
    for task_id, task in cases.items():
        proposal = _proposal_for_target(task, gold[task_id]["target"])
        adapter = MockModelAdapter(
            [proposal],
            model_id=model_config["model_id"],
            revision=model_config["revision"],
        )
        result = run_agent_case(
            task,
            adapter=adapter,
            model_config=model_config,
            corpus_id="companies-house-fixture",
            benchmark_manifest_sha256=benchmark_sha256,
            runtime_mode="direct",
            prompt_condition="few_shot",
            evidence_items=evidence[task_id],
            few_shot_examples=[
                example
                for example in examples
                if example["task"]["task_type"] == task["task_type"]
                and (
                    task["task_type"] == "quant_metric"
                    or example["task"]["narrative_subtype"] == task["narrative_subtype"]
                )
            ],
            token_counter=lambda _text: 0,
        )
        rows.append(result.to_dict())
    return rows


def _runtime_model_failure_rows(
    benchmark: Path,
    *,
    runtime_mode: str,
    repair_task_id: str | None = None,
) -> list[dict]:
    cases = _read_jsonl(benchmark / "cases.jsonl")
    evidence = {
        row["task_id"]: row["items"]
        for row in _read_jsonl(benchmark / "evidence.jsonl")
    }
    benchmark_sha256 = _file_sha256(benchmark / "benchmark_manifest.json")
    model_config = build_model_config(
        model_id="fixture/model",
        revision="fixture-revision",
        quantization="bf16",
        backend="mock",
    )
    rows = []
    for task in cases:
        task_id = task["task_id"]
        scripted_responses: list[object] = []
        if task_id == repair_task_id:
            rejected = '{"task_id":"' + task_id + '","action":'
            scripted_responses.append(rejected)
        scripted_responses.append(ModelGenerationError("fixture backend failure"))
        adapter = MockModelAdapter(
            scripted_responses,
            model_id=model_config["model_id"],
            revision=model_config["revision"],
        )

        def tool_executor(
            tool_name,
            tool_arguments,
            task_input,
            task_evidence=evidence[task_id],
        ):
            return execute_deterministic_tool(
                tool_name,
                tool_arguments,
                task_input,
                task_evidence,
            )

        result = run_agent_case(
            task,
            adapter=adapter,
            model_config=model_config,
            corpus_id="companies-house-fixture",
            benchmark_manifest_sha256=benchmark_sha256,
            runtime_mode=runtime_mode,
            prompt_condition="zero_shot",
            evidence_items=evidence[task_id],
            tool_executor=(
                tool_executor if runtime_mode == "capability_agent" else None
            ),
        )
        rows.append({"task_id": task_id, **result.to_dict()})
    return rows


def _write_run(benchmark: Path, run_dir: Path, rows: list[dict]) -> Path:
    run_dir.mkdir()
    results_path = run_dir / "results.jsonl"
    _write_jsonl(results_path, rows)
    benchmark_manifest = _read_json(benchmark / "benchmark_manifest.json")
    task_ids = [row["run_record"]["task_id"] for row in rows]
    first_record = rows[0]["run_record"]
    model_record = next(
        (
            row["run_record"]
            for row in rows
            if row["run_record"].get("model_config") is not None
        ),
        None,
    )
    if model_record is None:
        raise ValueError("Fixture run requires at least one model-backed record")
    approval_path = run_dir / "few_shot_approval.json"
    if first_record["prompt_condition"] == "few_shot":
        few_shot_rows = _read_jsonl(benchmark / "few_shot.jsonl")
        approval_path.write_text(
            json.dumps(
                {
                    "few_shot_approval_version": FEW_SHOT_APPROVAL_VERSION,
                    "benchmark_id": benchmark_manifest["benchmark_id"],
                    "few_shot_sha256": _file_sha256(benchmark / "few_shot.jsonl"),
                    "review_packet_sha256": "a" * 64,
                    "review_binding_sha256": "b" * 64,
                    "reviewer": "fixture-reviewer",
                    "reviewed_at": "2026-08-21T00:00:00Z",
                    "approved_example_ids": sorted(
                        row["example_id"] for row in few_shot_rows
                    ),
                    "decision": "APPROVED",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    full_count = benchmark_manifest["counts"]["total"]
    is_parity_slice = len(rows) == 50 and full_count != 50
    manifest = {
        "batch_run_version": BATCH_RUN_VERSION,
        "benchmark_id": benchmark_manifest["benchmark_id"],
        "benchmark_manifest_sha256": _file_sha256(
            benchmark / "benchmark_manifest.json"
        ),
        "runtime_mode": first_record["runtime_mode"],
        "prompt_condition": first_record["prompt_condition"],
        "model_config_sha256": _canonical_sha256(model_record["model_config"]),
        "model_id": model_record["model_config"]["model_id"],
        "model_revision": model_record["model_config"]["revision"],
        "quantization": model_record["model_config"]["quantization"],
        "case_count": len(rows),
        "full_benchmark_case_count": full_count,
        "complete_full_benchmark": len(rows) == full_count,
        "task_ids_sha256": _canonical_sha256(task_ids),
        "inputs": {
            "cases_sha256": _file_sha256(benchmark / "cases.jsonl"),
            "evidence_sha256": _file_sha256(benchmark / "evidence.jsonl"),
            "few_shot_sha256": _file_sha256(benchmark / "few_shot.jsonl")
            if first_record["prompt_condition"] == "few_shot"
            else None,
            "few_shot_approval_sha256": _file_sha256(approval_path)
            if first_record["prompt_condition"] == "few_shot"
            else None,
            "parity_membership_sha256": _file_sha256(benchmark / "parity_50.json")
            if is_parity_slice
            else None,
        },
        "results": {"path": "results.jsonl", "sha256": _file_sha256(results_path)},
        "provenance": first_record["provenance"],
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results_path


def _legacy_smoke_rows(
    benchmark: Path, *, start: int = 0, count: int = 10
) -> list[dict]:
    rows = _perfect_rows(benchmark)[start : start + count]
    provenance = {
        "git_commit": "a" * 40,
        "source_tree_sha256": "1" * 64,
        "container_digest": "2" * 64,
        "model_snapshot_sha256": "3" * 64,
        "tokenizer_revision": "fixture-revision",
        "package_lock_sha256": "4" * 64,
    }
    for row in rows:
        row["run_record"]["provenance"] = copy.deepcopy(provenance)
        row["run_record"]["resources"]["gpu_model"] = "NVIDIA A100-SXM4-40GB"
        row["run_record"]["resources"]["slurm_job_id"] = "fixture-job"
    return rows


def _write_legacy_smoke_launch(
    run_path: Path,
    *,
    partition: str = "gpu-test",
    declared_limit: int | None = None,
    argv_limit: int | None = None,
    source_tree_sha256: str | None = None,
) -> Path:
    run_manifest = _read_json(run_path.parent / "run_manifest.json")
    first_row = _read_jsonl(run_path)[0]
    case_count = run_manifest["case_count"]
    declared = case_count if declared_limit is None else declared_limit
    argv_value = case_count if argv_limit is None else argv_limit
    run_provenance = run_manifest["provenance"]
    resources = first_row["run_record"]["resources"]
    launch = {
        "command": {
            "argv": [
                "python3",
                "-I",
                "-c",
                "<fixture wrapper>",
                "--runtime-mode",
                run_manifest["runtime_mode"],
                "--prompt-condition",
                run_manifest["prompt_condition"],
                "--limit",
                str(argv_value),
            ],
            "entrypoint": "python3 -I -c <fixture wrapper>",
            "entrypoint_sha256": "5" * 64,
            "limit": str(declared),
            "prompt_condition": run_manifest["prompt_condition"],
            "runtime_mode": run_manifest["runtime_mode"],
        },
        "finished_at": "2026-08-22T00:00:10Z",
        "gpu_inventory": ["NVIDIA A100-SXM4-40GB, fixture, 40960 MiB"],
        "manifest_version": "auditops-slurm-launch.v1",
        "network_isolation": {
            "policy": "apptainer-net-none+socket-preflight",
            "preflight": "passed",
        },
        "provenance": {
            "container_sha256": run_provenance["container_digest"],
            "git_commit": run_provenance["git_commit"],
            "model_config_sha256": "6" * 64,
            "model_id": run_manifest["model_id"],
            "model_revision": run_manifest["model_revision"],
            "model_snapshot_manifest_sha256": run_provenance["model_snapshot_sha256"],
            "model_snapshot_sha256": run_provenance["model_snapshot_sha256"],
            "package_lock_sha256": run_provenance["package_lock_sha256"],
            "source_manifest_sha256": "7" * 64,
            "source_tree_sha256": source_tree_sha256
            or run_provenance["source_tree_sha256"],
            "vllm_version": "0.26.0",
        },
        "slurm": {
            "cpus_per_task": "16",
            "job_gpus": "3",
            "job_id": resources["slurm_job_id"],
            "job_name": "auditops-agent-compat",
            "node_list": "fixture-node",
            "nodes": "1",
            "partition": partition,
        },
        "started_at": "2026-08-22T00:00:00Z",
    }
    path = run_path.parent / "slurm_launch_manifest.json"
    path.write_text(
        json.dumps(launch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def test_evaluate_perfect_run_scores_all_metrics_and_gates(tmp_path):
    benchmark = _benchmark(tmp_path)
    run_path = _write_run(benchmark, tmp_path / "perfect", _perfect_rows(benchmark))

    result = evaluate_agent_run(benchmark, run_path, run_name="perfect")

    assert result["summary"]["expected_task_count"] == 10
    assert result["summary"]["schema_compliance"] == 1.0
    assert result["summary"]["routing_accuracy"] is None
    assert result["summary"]["routing_evaluable_task_count"] == 0
    assert result["summary"]["quant_joint_accuracy"] == 1.0
    assert result["summary"]["narrative_answerability_accuracy"] == 1.0
    assert result["summary"]["citation_exactness"] == 1.0
    assert result["summary"]["citation_precision"] == 1.0
    assert result["summary"]["citation_coverage"] == 1.0
    assert result["summary"]["refusal_accuracy"] == 1.0
    assert result["hard_safety_gates"]["passed"] is True
    assert result["failures"] == []


def test_evaluate_perfect_safety_hybrid_replays_model_independent_quant(tmp_path):
    benchmark = _benchmark(tmp_path)
    task_types = {
        row["task_id"]: row["task_type"]
        for row in _read_jsonl(benchmark / "cases.jsonl")
    }
    rows = _hybrid_rows(benchmark)
    run_path = _write_run(benchmark, tmp_path / "hybrid-perfect", rows)

    result = evaluate_agent_run(benchmark, run_path, run_name="hybrid-perfect")

    assert result["summary"]["schema_compliance"] == 1.0
    assert result["summary"]["quant_joint_accuracy"] == 1.0
    assert result["summary"]["narrative_answerability_accuracy"] == 1.0
    assert result["summary"]["citation_exactness"] == 1.0
    assert result["summary"]["refusal_accuracy"] == 1.0
    assert result["hard_safety_gates"]["passed"] is True
    assert result["failures"] == []

    quant_rows = [row for row in rows if task_types[row["task_id"]] == "quant_metric"]
    narrative_rows = [
        row for row in rows if task_types[row["task_id"]] == "narrative_citation"
    ]
    assert quant_rows and narrative_rows
    for row in quant_rows:
        record = row["run_record"]
        assert record["model_config"] is None
        assert record["verifier_trace"] == []
        assert record["token_usage"] == {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        assert row["tool_observation"]["visibility"] == "VERIFIER_ONLY"
    for row in narrative_rows:
        assert row["run_record"]["model_config"] is not None
        assert row["run_record"]["verifier_trace"]
        assert row["tool_observation"]["visibility"] == "MODEL_VISIBLE"


def test_safety_hybrid_quantitative_tamper_fails_deterministic_replay(tmp_path):
    benchmark = _benchmark(tmp_path)
    task_types = {
        row["task_id"]: row["task_type"]
        for row in _read_jsonl(benchmark / "cases.jsonl")
    }
    rows = _hybrid_rows(benchmark)
    forged = next(
        row
        for row in rows
        if task_types[row["task_id"]] == "quant_metric"
        and row["final_proposal"]["status"] == "OK"
    )
    forged["final_proposal"]["value"] = "999"
    forged["run_record"]["output_sha256"] = _canonical_sha256(forged["final_proposal"])
    run_path = _write_run(benchmark, tmp_path / "hybrid-forged", rows)

    result = evaluate_agent_run(benchmark, run_path)

    assert result["hard_safety_gates"]["passed"] is False
    affected = next(
        item for item in result["per_task"] if item["task_id"] == forged["task_id"]
    )
    assert "DETERMINISTIC_FINAL_PROPOSAL_MISMATCH" in affected["issue_codes"]
    assert "VERIFIER_INTEGRITY_FAILURE" in affected["safety_codes"]


def test_supported_narrative_variant_is_accuracy_error_not_safety_failure(tmp_path):
    benchmark = _benchmark(tmp_path)
    rows = _perfect_rows(benchmark)
    cases = {row["task_id"]: row for row in _read_jsonl(benchmark / "cases.jsonl")}
    evidence = {
        row["task_id"]: row["items"]
        for row in _read_jsonl(benchmark / "evidence.jsonl")
    }
    benchmark_sha256 = _file_sha256(benchmark / "benchmark_manifest.json")
    replaced = next(
        row
        for row in rows
        if cases[row["task_id"]]["task_type"] == "narrative_citation"
        and row["final_proposal"]["status"] == "OK"
    )
    task_id = replaced["task_id"]
    task = cases[task_id]
    evidence_id = task["evidence_scope"]["evidence_ids"][0]
    proposal = {
        "task_id": task_id,
        "action": "ANSWER",
        "period_key": task["period"]["period_key"],
        "extracts": [{"evidence_id": evidence_id, "exact_quote": "answer"}],
    }
    model_config = replaced["run_record"]["model_config"]
    adapter = MockModelAdapter(
        [proposal],
        model_id=model_config["model_id"],
        revision=model_config["revision"],
    )
    variant = run_agent_case(
        task,
        adapter=adapter,
        model_config=model_config,
        corpus_id="companies-house-fixture",
        benchmark_manifest_sha256=benchmark_sha256,
        runtime_mode="direct",
        prompt_condition="zero_shot",
        evidence_items=evidence[task_id],
        token_counter=lambda _text: 0,
    ).to_dict()
    assert variant["outcome"] == "RELEASED"
    rows = [variant if row["task_id"] == task_id else row for row in rows]
    run_path = _write_run(benchmark, tmp_path / "supported-variant", rows)

    result = evaluate_agent_run(benchmark, run_path)

    assert result["summary"]["narrative_answerability_accuracy"] == 1.0
    assert result["summary"]["narrative_answer_exactness"] < 1.0
    assert result["hard_safety_gates"]["passed"] is True
    assert (
        result["hard_safety_gates"]["violation_counts"]["unsupported_released_claim"]
        == 0
    )
    affected = next(item for item in result["per_task"] if item["task_id"] == task_id)
    assert "NARRATIVE_ANSWER_MISMATCH" in affected["issue_codes"]
    assert "UNSUPPORTED_RELEASED_CLAIM" not in affected["safety_codes"]


@pytest.mark.parametrize(
    ("field", "expected_code"),
    [
        ("prompt", "PROMPT_HASH_MISMATCH[0]"),
        ("context", "CONTEXT_HASH_MISMATCH"),
    ],
)
def test_evaluator_reconstructs_prompt_and_context_hashes(
    tmp_path, field, expected_code
):
    benchmark = _benchmark(tmp_path)
    rows = _perfect_rows(benchmark)
    forged = rows[0]
    if field == "prompt":
        forged["run_record"]["verifier_trace"][0]["prompt_sha256"] = "0" * 64
        forged["run_record"]["prompt_sha256"] = _canonical_sha256(["0" * 64])
    else:
        forged["run_record"]["context_sha256"] = "0" * 64
    run_path = _write_run(benchmark, tmp_path / f"forged-{field}", rows)

    result = evaluate_agent_run(benchmark, run_path)

    affected = next(
        item for item in result["per_task"] if item["task_id"] == forged["task_id"]
    )
    assert expected_code in affected["issue_codes"]
    assert result["hard_safety_gates"]["passed"] is False


def test_evaluator_reconstructs_repair_prompt_from_prior_attempt(tmp_path):
    benchmark = _benchmark(tmp_path)
    rows = _perfect_rows(benchmark)
    cases = {row["task_id"]: row for row in _read_jsonl(benchmark / "cases.jsonl")}
    evidence = {
        row["task_id"]: row["items"]
        for row in _read_jsonl(benchmark / "evidence.jsonl")
    }
    gold = {row["task_id"]: row for row in _read_jsonl(benchmark / "gold.jsonl")}
    repaired = next(
        row
        for row in rows
        if row["final_proposal"]["status"] == "OK"
        and row["final_proposal"]["value"] is not None
    )
    task_id = repaired["task_id"]
    task = cases[task_id]
    final_proposal = _proposal_for_target(task, gold[task_id]["target"])
    rejected = '{"task_id":"' + task_id + '","action":'
    model_config = repaired["run_record"]["model_config"]
    adapter = MockModelAdapter(
        [rejected, final_proposal],
        model_id=model_config["model_id"],
        revision=model_config["revision"],
    )
    runtime_result = run_agent_case(
        task,
        adapter=adapter,
        model_config=model_config,
        corpus_id="companies-house-fixture",
        benchmark_manifest_sha256=_file_sha256(benchmark / "benchmark_manifest.json"),
        runtime_mode="direct",
        prompt_condition="zero_shot",
        evidence_items=evidence[task_id],
        token_counter=lambda _text: 0,
    )
    replacement = runtime_result.to_dict()
    trace = replacement["run_record"]["verifier_trace"]
    assert [entry["verifier_result"]["disposition"] for entry in trace] == [
        "REPAIR_REQUIRED",
        "RELEASED",
    ]
    assert trace[0]["proposal"] is None
    rows = [replacement if row["task_id"] == task_id else row for row in rows]
    run_path = _write_run(benchmark, tmp_path / "repaired", rows)

    result = evaluate_agent_run(benchmark, run_path)

    assert result["hard_safety_gates"]["passed"] is True
    assert result["summary"]["repair_rate"] > 0

    # A self-consistent forged replacement must still fail independent replay.
    trace[1]["prompt_sha256"] = "0" * 64
    replacement["run_record"]["prompt_sha256"] = _canonical_sha256(
        [trace[0]["prompt_sha256"], "0" * 64]
    )
    forged_path = _write_run(benchmark, tmp_path / "forged-repair", rows)
    forged_result = evaluate_agent_run(benchmark, forged_path)
    affected = next(
        item for item in forged_result["per_task"] if item["task_id"] == task_id
    )
    assert "PROMPT_HASH_MISMATCH[1]" in affected["issue_codes"]


def test_evaluator_requires_exact_prompt_version(tmp_path):
    benchmark = _benchmark(tmp_path)
    rows = _perfect_rows(benchmark)
    rows[0]["run_record"]["prompt_version"] = "unfrozen-prompt"
    run_path = _write_run(benchmark, tmp_path / "wrong-prompt-version", rows)

    with pytest.raises(ValueError, match="prompt_version"):
        evaluate_agent_run(benchmark, run_path)


def test_few_shot_run_requires_retained_approval_and_replays_prompts(tmp_path):
    benchmark = _benchmark(tmp_path)
    run_path = _write_run(benchmark, tmp_path / "few-shot", _few_shot_rows(benchmark))
    assert evaluate_agent_run(benchmark, run_path)["hard_safety_gates"]["passed"]

    approval_path = run_path.parent / "few_shot_approval.json"
    original_approval = approval_path.read_bytes()
    approval_path.write_bytes(original_approval + b" ")
    with pytest.raises(ValueError, match="approval SHA-256 mismatch"):
        evaluate_agent_run(benchmark, run_path)
    approval_path.write_bytes(original_approval)
    approval_path.unlink()
    with pytest.raises(ValueError, match="few_shot_approval.json"):
        evaluate_agent_run(benchmark, run_path)


def test_zero_shot_run_rejects_retained_few_shot_approval(tmp_path):
    benchmark = _benchmark(tmp_path)
    run_path = _write_run(
        benchmark, tmp_path / "zero-with-approval", _perfect_rows(benchmark)
    )
    (run_path.parent / "few_shot_approval.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Zero-shot"):
        evaluate_agent_run(benchmark, run_path)


def test_quant_joint_citations_are_order_independent_but_unique():
    target = {
        "status": "OK",
        "value": "2",
        "unit": "pure",
        "period_key": "P",
        "refusal_code": None,
        "evidence_ids": ["e1", "e2"],
    }
    answer = {**target, "evidence_ids": ["e2", "e1"]}
    assert _quant_joint_match(answer, target) is True
    assert _quant_joint_match({**answer, "evidence_ids": ["e1", "e1"]}, target) is False


def test_routing_is_na_for_direct_and_exact_for_agent():
    case = {
        "task_id": "n1",
        "task_type": "narrative_citation",
        "filing": {"filing_id": "f1"},
        "period": {"period_key": "FY2025"},
        "task_parameters": {"metric_spec_id": None, "retrieval_query": "audit status"},
        "evidence_scope": {
            "evidence_ids": ["e1"],
            "filing_ids": ["f1"],
            "max_items": 1,
            "selection_method": "FROZEN_BM25",
            "empty_reason": None,
        },
    }
    exact_plan = {
        "task_id": "n1",
        "action": "CALL_TOOL",
        "tool_name": "load_frozen_evidence",
        "tool_arguments": {
            "filing_id": "f1",
            "evidence_scope_sha256": _canonical_sha256(case["evidence_scope"]),
        },
    }
    assert (
        _routing_match(
            case,
            {"run_record": {"runtime_mode": "direct"}},
            {},
            {},
        )
        is None
    )
    assert (
        _routing_match(
            case,
            {
                "run_record": {"runtime_mode": "capability_agent"},
                "plan_proposal": exact_plan,
            },
            {},
            {},
        )
        is True
    )
    incomplete = copy.deepcopy(exact_plan)
    incomplete["tool_arguments"].pop("evidence_scope_sha256")
    assert (
        _routing_match(
            case,
            {
                "run_record": {"runtime_mode": "capability_agent"},
                "plan_proposal": incomplete,
            },
            {},
            {},
        )
        is False
    )


def test_unsupported_proposal_codes_include_every_trace_attempt():
    row = {
        "run_record": {
            "verifier_trace": [
                {
                    "verifier_result": {
                        "checks": [{"code": "SUPPORT_VALID", "passed": False}]
                    }
                },
                {
                    "verifier_result": {
                        "checks": [{"code": "SUPPORT_VALID", "passed": True}]
                    }
                },
            ],
            "verifier_result": {"checks": [{"code": "SUPPORT_VALID", "passed": True}]},
        }
    }
    assert "SUPPORT_VALID" in _verifier_failure_codes(row)


def test_evaluation_detects_silent_drop_and_released_safety_failures(tmp_path):
    benchmark = _benchmark(tmp_path)
    rows = _perfect_rows(benchmark)
    rows.pop()
    bad = next(
        row
        for row in rows
        if row["final_proposal"]["status"] == "OK"
        and row["final_proposal"]["answer_text"] is not None
    )
    bad["final_proposal"]["answer_text"] = "unsupported text"
    bad["final_proposal"]["evidence_ids"] = ["chunk::fabricated"]
    bad["final_proposal"]["claims"] = [
        {
            "claim_id": "claim-1",
            "text": "unsupported text",
            "evidence_ids": ["chunk::fabricated"],
            "supporting_text": "unsupported text",
        }
    ]
    bad["run_record"]["output_sha256"] = _canonical_sha256(bad["final_proposal"])
    bad["run_record"]["repair_count"] = 2
    run_path = _write_run(benchmark, tmp_path / "bad", rows)

    result = evaluate_agent_run(benchmark, run_path, run_name="bad")

    assert result["summary"]["missing_task_count"] == 1
    assert result["hard_safety_gates"]["passed"] is False
    violations = result["hard_safety_gates"]["violation_counts"]
    assert violations["unsupported_released_claim"] == 1
    assert violations["invalid_released_citation"] == 1
    assert violations["repair_limit_exceeded"] == 1
    assert (
        result["hard_safety_gates"]["checks"]["complete_no_silent_drops"]["passed"]
        is False
    )


def test_forged_minimal_verifier_and_empty_citations_are_hard_failures(tmp_path):
    benchmark = _benchmark(tmp_path)
    rows = _perfect_rows(benchmark)
    forged = next(
        row
        for row in rows
        if row["final_proposal"]["status"] == "OK"
        and row["final_proposal"]["answer_text"] is not None
    )
    forged["final_proposal"]["evidence_ids"] = []
    forged["final_proposal"]["claims"] = []
    forged["run_record"]["output_sha256"] = _canonical_sha256(forged["final_proposal"])
    forged["run_record"]["verifier_result"] = {
        "release_allowed": True,
        "checks": [],
    }
    run_path = _write_run(benchmark, tmp_path / "forged", rows)

    result = evaluate_agent_run(benchmark, run_path, run_name="forged")

    violations = result["hard_safety_gates"]["violation_counts"]
    assert result["hard_safety_gates"]["passed"] is False
    assert violations["invalid_run_contract"] == 1
    assert violations["verifier_bypass"] == 1
    assert violations["invalid_released_citation"] == 1


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("results", "Run results SHA-256 does not match"),
        ("benchmark_binding", "benchmark input hashes do not match"),
    ],
)
def test_evaluation_rejects_tampered_run_hash_or_benchmark_binding(
    tmp_path, tamper: str, message: str
):
    benchmark = _benchmark(tmp_path)
    run_path = _write_run(
        benchmark, tmp_path / "tampered-run", _perfect_rows(benchmark)
    )
    if tamper == "results":
        with run_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n")
    else:
        manifest_path = run_path.parent / "run_manifest.json"
        manifest = _read_json(manifest_path)
        manifest["inputs"]["cases_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        evaluate_agent_run(benchmark, run_path)


def test_evaluator_recomputes_and_rejects_omitted_verifier_check(tmp_path):
    benchmark = _benchmark(tmp_path)
    rows = _perfect_rows(benchmark)
    forged = rows[0]
    forged_result = copy.deepcopy(forged["run_record"]["verifier_result"])
    assert len(forged_result["checks"]) > 1
    forged_result["checks"].pop()
    forged["run_record"]["verifier_result"] = forged_result
    forged["run_record"]["verifier_trace"][-1]["verifier_result"] = forged_result
    run_path = _write_run(benchmark, tmp_path / "omitted-check", rows)

    result = evaluate_agent_run(benchmark, run_path)

    assert result["hard_safety_gates"]["passed"] is False
    assert (
        result["hard_safety_gates"]["violation_counts"]["verifier_integrity_failure"]
        == 1
    )
    affected = next(
        item for item in result["per_task"] if item["task_id"] == forged["task_id"]
    )
    assert "VERIFIER_TRACE_MISMATCH[0]" in affected["issue_codes"]


def test_evaluator_rejects_mixed_run_metadata_and_unknown_manifest_fields(tmp_path):
    benchmark = _benchmark(tmp_path)
    mixed_rows = _perfect_rows(benchmark)
    mixed_rows[1]["run_record"]["runtime_mode"] = "capability_agent"
    mixed_path = _write_run(benchmark, tmp_path / "mixed", mixed_rows)
    with pytest.raises(ValueError, match="do not match the frozen run manifest"):
        evaluate_agent_run(benchmark, mixed_path)

    valid_path = _write_run(
        benchmark, tmp_path / "unknown-field", _perfect_rows(benchmark)
    )
    manifest_path = valid_path.parent / "run_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["unexpected"] = "fail-closed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid schema"):
        evaluate_agent_run(benchmark, valid_path)


def test_agent_tool_observation_is_recomputed_from_frozen_evidence(tmp_path):
    benchmark = _benchmark(tmp_path)
    rows = _agent_rows(benchmark)
    forged = rows[0]
    forged["tool_observation"] = {"status": "FORGED"}
    run_path = _write_run(benchmark, tmp_path / "forged-tool", rows)

    result = evaluate_agent_run(benchmark, run_path)

    assert result["hard_safety_gates"]["passed"] is False
    affected = next(
        item for item in result["per_task"] if item["task_id"] == forged["task_id"]
    )
    assert "TOOL_OBSERVATION_MISMATCH" in affected["issue_codes"]
    assert "INTEGRITY_FAILURE" in affected["safety_codes"]


@pytest.mark.parametrize(
    ("field", "expected_code"),
    [
        ("request", "REQUEST_HASH_MISMATCH[0]"),
        ("response_schema", "RESPONSE_SCHEMA_HASH_MISMATCH[0]"),
        ("run_envelope", "RUN_ENVELOPE_HASH_MISMATCH"),
    ],
)
def test_evaluator_replays_complete_v2_request_and_run_identity(
    tmp_path, field, expected_code
):
    benchmark = _benchmark(tmp_path)
    rows = _perfect_rows(benchmark)
    forged = rows[0]
    record = forged["run_record"]
    trace = record["verifier_trace"]
    if field == "request":
        trace[0]["request_sha256"] = "0" * 64
    elif field == "response_schema":
        trace[0]["response_schema_sha256"] = "0" * 64
    else:
        record["run_envelope_sha256"] = "0" * 64
        record["run_id"] = "run-" + "0" * 24
    run_path = _write_run(benchmark, tmp_path / f"forged-{field}", rows)

    result = evaluate_agent_run(benchmark, run_path)

    affected = next(
        item for item in result["per_task"] if item["task_id"] == forged["task_id"]
    )
    assert expected_code in affected["issue_codes"]
    assert "INTEGRITY_FAILURE" in affected["safety_codes"]
    assert result["hard_safety_gates"]["passed"] is False


def test_local_model_path_change_preserves_portable_run_identity(tmp_path):
    benchmark = _benchmark(tmp_path)
    rows = _perfect_rows(benchmark)
    original_run_ids = [row["run_record"]["run_id"] for row in rows]
    original_request_ids = [
        row["run_record"]["verifier_trace"][0]["request_sha256"] for row in rows
    ]
    for row in rows:
        record = row["run_record"]
        record["model_config"]["model_path"] = "D:/deployment-only/model-copy"
        record["deployment_model_config_sha256"] = _canonical_sha256(
            record["model_config"]
        )
    run_path = _write_run(benchmark, tmp_path / "portable-path", rows)

    result = evaluate_agent_run(benchmark, run_path)

    assert result["hard_safety_gates"]["passed"] is True
    assert [row["run_record"]["run_id"] for row in rows] == original_run_ids
    assert [
        row["run_record"]["verifier_trace"][0]["request_sha256"] for row in rows
    ] == original_request_ids


def test_tool_observation_mismatch_is_a_typed_integrity_outcome(tmp_path):
    benchmark = _benchmark(tmp_path)
    rows = _agent_rows(benchmark)
    cases = {row["task_id"]: row for row in _read_jsonl(benchmark / "cases.jsonl")}
    evidence = {
        row["task_id"]: row["items"]
        for row in _read_jsonl(benchmark / "evidence.jsonl")
    }
    replaced = rows[0]
    task_id = replaced["task_id"]
    task = cases[task_id]
    tool_name = task["allowed_tools"][0]
    tool_arguments = expected_tool_arguments(task)
    plan = {
        "task_id": task_id,
        "action": "CALL_TOOL",
        "tool_name": tool_name,
        "tool_arguments": tool_arguments,
    }
    model_config = replaced["run_record"]["model_config"]
    adapter = MockModelAdapter(
        [plan],
        model_id=model_config["model_id"],
        revision=model_config["revision"],
    )

    def mismatched_tool(name, arguments, task_input):
        observation = execute_deterministic_tool(
            name, arguments, task_input, evidence[task_id]
        )
        observation["visibility"] = "VERIFIER_ONLY"
        return observation

    runtime_result = run_agent_case(
        task,
        adapter=adapter,
        model_config=model_config,
        corpus_id="companies-house-fixture",
        benchmark_manifest_sha256=_file_sha256(benchmark / "benchmark_manifest.json"),
        runtime_mode="capability_agent",
        prompt_condition="zero_shot",
        evidence_items=evidence[task_id],
        tool_executor=mismatched_tool,
        token_counter=lambda _text: 0,
    ).to_dict()
    assert runtime_result["outcome"] == "INTEGRITY_FAILURE"
    assert runtime_result["failure"]["code"] == "TOOL_OBSERVATION_MISMATCH"
    rows[0] = runtime_result
    run_path = _write_run(benchmark, tmp_path / "typed-integrity", rows)

    result = evaluate_agent_run(benchmark, run_path)

    violations = result["hard_safety_gates"]["violation_counts"]
    assert result["summary"]["typed_integrity_failure_count"] == 1
    assert violations["integrity_failure"] == 1
    assert violations["invalid_run_contract"] == 0
    assert violations["verifier_integrity_failure"] == 0
    assert result["hard_safety_gates"]["passed"] is False


@pytest.mark.parametrize("runtime_mode", ["direct", "capability_agent"])
def test_runtime_model_generation_failure_is_typed_infrastructure(
    tmp_path, runtime_mode
):
    benchmark = _benchmark(tmp_path)
    rows = _runtime_model_failure_rows(benchmark, runtime_mode=runtime_mode)
    run_path = _write_run(
        benchmark,
        tmp_path / f"typed-model-failure-{runtime_mode}",
        rows,
    )

    result = evaluate_agent_run(benchmark, run_path)

    violations = result["hard_safety_gates"]["violation_counts"]
    assert result["summary"]["typed_infrastructure_failure_count"] == len(rows)
    assert violations["untyped_infrastructure_failure"] == 0
    assert violations["invalid_run_contract"] == 0
    assert violations["verifier_bypass"] == 0
    assert result["hard_safety_gates"]["passed"] is True
    assert result["failures"] == []


def test_evaluator_replays_context_time_backend_failures(tmp_path):
    benchmark = _benchmark(tmp_path)
    rows = _perfect_rows(benchmark)
    cases = _read_jsonl(benchmark / "cases.jsonl")
    evidence = {
        row["task_id"]: row["items"]
        for row in _read_jsonl(benchmark / "evidence.jsonl")
    }
    benchmark_sha256 = _file_sha256(benchmark / "benchmark_manifest.json")
    model_config = build_model_config(
        model_id="fixture/model",
        revision="fixture-revision",
        quantization="bf16",
        backend="mock",
    )
    failures = (
        ModelTimeoutError("secret timeout detail"),
        ModelOOMError("secret OOM detail"),
        ModelBackendError("secret backend detail"),
    )
    for index, exception in enumerate(failures):
        task = cases[index]
        adapter = MockModelAdapter(
            [{}],
            model_id=model_config["model_id"],
            revision=model_config["revision"],
        )

        def fail_count(_: str, failure=exception) -> int:
            raise failure

        result = run_agent_case(
            task,
            adapter=adapter,
            model_config=model_config,
            corpus_id="companies-house-fixture",
            benchmark_manifest_sha256=benchmark_sha256,
            runtime_mode="direct",
            prompt_condition="zero_shot",
            evidence_items=evidence[task["task_id"]],
            token_counter=fail_count,
        )
        assert result.failure["stage"] == "context"
        rows[index] = result.to_dict()

    run_path = _write_run(benchmark, tmp_path / "context-backend-failures", rows)
    result = evaluate_agent_run(benchmark, run_path)

    violations = result["hard_safety_gates"]["violation_counts"]
    assert result["summary"]["typed_infrastructure_failure_count"] == len(failures)
    assert violations["invalid_run_contract"] == 0
    assert violations["verifier_integrity_failure"] == 0
    assert violations["verifier_bypass"] == 0
    assert result["failures"] == []


def test_runtime_model_generation_failure_after_repair_is_typed(tmp_path):
    benchmark = _benchmark(tmp_path)
    cases = _read_jsonl(benchmark / "cases.jsonl")
    gold = {row["task_id"]: row for row in _read_jsonl(benchmark / "gold.jsonl")}
    repair_task_id = next(
        case["task_id"]
        for case in cases
        if case["task_type"] == "quant_metric"
        and gold[case["task_id"]]["target"]["status"] == "OK"
    )
    rows = _runtime_model_failure_rows(
        benchmark,
        runtime_mode="direct",
        repair_task_id=repair_task_id,
    )
    repaired_failure = next(row for row in rows if row["task_id"] == repair_task_id)
    assert [
        entry["verifier_result"]["disposition"]
        for entry in repaired_failure["run_record"]["verifier_trace"]
    ] == ["REPAIR_REQUIRED", "FAILED"]
    assert (
        repaired_failure["run_record"]["verifier_trace"][-1]["repair_attempt"] is True
    )
    run_path = _write_run(benchmark, tmp_path / "typed-repair-model-failure", rows)

    result = evaluate_agent_run(benchmark, run_path)

    assert result["summary"]["typed_infrastructure_failure_count"] == len(rows)
    assert result["hard_safety_gates"]["passed"] is True


def test_evaluator_accepts_terminal_discarded_schema_failure_and_replays_trace(
    tmp_path,
):
    benchmark = _benchmark(tmp_path)
    rows = _perfect_rows(benchmark)
    cases = {row["task_id"]: row for row in _read_jsonl(benchmark / "cases.jsonl")}
    gold = {row["task_id"]: row for row in _read_jsonl(benchmark / "gold.jsonl")}
    evidence = {
        row["task_id"]: row["items"]
        for row in _read_jsonl(benchmark / "evidence.jsonl")
    }
    task_id = next(
        task_id
        for task_id, task in cases.items()
        if task["task_type"] == "quant_metric"
        and gold[task_id]["target"]["status"] == "OK"
    )
    task = cases[task_id]
    invalid = _proposal_for_target(task, gold[task_id]["target"])
    invalid["status"] = "REFUSAL"
    invalid["value"] = "987654321"
    model_config = rows[0]["run_record"]["model_config"]
    adapter = MockModelAdapter(
        (invalid, invalid),
        model_id=model_config["model_id"],
        revision=model_config["revision"],
    )
    runtime_result = run_agent_case(
        task,
        adapter=adapter,
        model_config=model_config,
        corpus_id="companies-house-fixture",
        benchmark_manifest_sha256=_file_sha256(benchmark / "benchmark_manifest.json"),
        runtime_mode="direct",
        prompt_condition="zero_shot",
        evidence_items=evidence[task_id],
    )
    replacement = {"task_id": task_id, **runtime_result.to_dict()}
    rows = [replacement if row["task_id"] == task_id else row for row in rows]
    run_path = _write_run(benchmark, tmp_path / "discarded-schema-failure", rows)

    result = evaluate_agent_run(benchmark, run_path)

    violations = result["hard_safety_gates"]["violation_counts"]
    assert violations["invalid_run_contract"] == 0
    assert violations["verifier_bypass"] == 0
    assert violations["verifier_integrity_failure"] == 0
    affected = next(item for item in result["per_task"] if item["task_id"] == task_id)
    assert "SCHEMA_INVALID" in affected["issue_codes"]
    assert replacement["final_proposal"] is None
    assert "987654321" not in str(replacement)


@pytest.mark.parametrize("tamper", ["diagnostic_code", "output_hash"])
def test_forged_model_generation_failure_trace_is_not_typed(tmp_path, tamper):
    benchmark = _benchmark(tmp_path)
    rows = _runtime_model_failure_rows(benchmark, runtime_mode="direct")
    forged = rows[0]
    terminal = forged["run_record"]["verifier_trace"][-1]
    if tamper == "diagnostic_code":
        terminal["verifier_result"]["checks"][0]["code"] = (
            "MODEL_OUTPUT_NOT_STRICT_JSON"
        )
    else:
        terminal["output_sha256"] = "0" * 64
    run_path = _write_run(benchmark, tmp_path / f"forged-model-failure-{tamper}", rows)

    result = evaluate_agent_run(benchmark, run_path)

    violations = result["hard_safety_gates"]["violation_counts"]
    assert result["summary"]["typed_infrastructure_failure_count"] == len(rows) - 1
    assert violations["untyped_infrastructure_failure"] == 1
    assert violations["invalid_run_contract"] == 1
    assert violations["verifier_bypass"] == 1
    affected = next(
        item for item in result["per_task"] if item["task_id"] == forged["task_id"]
    )
    assert "INVALID_CASE_RESULT" in affected["issue_codes"]


def test_unknown_infrastructure_failure_code_is_not_typed(tmp_path):
    benchmark = _benchmark(tmp_path)
    rows = _perfect_rows(benchmark)
    forged = rows[0]
    forged["outcome"] = "INFRASTRUCTURE_FAILURE"
    forged["run_record"]["outcome"] = "INFRASTRUCTURE_FAILURE"
    forged["run_record"]["verifier_result"] = None
    forged["failure"] = {
        "failure_version": "v2",
        "failure_class": "INFRASTRUCTURE_FAILURE",
        "code": "BANANA",
        "stage": "model",
        "retryable": True,
        "sanitized_detail": "unknown fixture failure",
    }
    run_path = _write_run(benchmark, tmp_path / "unknown-infra", rows)

    result = evaluate_agent_run(benchmark, run_path)

    violations = result["hard_safety_gates"]["violation_counts"]
    assert violations["untyped_infrastructure_failure"] == 1
    assert violations["invalid_run_contract"] == 1


def test_frozen_50_case_parity_membership_is_the_evaluation_universe(tmp_path):
    benchmark = _benchmark(tmp_path, quant_count=36, narrative_count=24)
    all_rows = {row["task_id"]: row for row in _perfect_rows(benchmark)}
    parity = _read_json(benchmark / "parity_50.json")
    rows = [all_rows[task_id] for task_id in parity["task_ids"]]
    run_path = _write_run(benchmark, tmp_path / "parity", rows)

    result = evaluate_agent_run(benchmark, run_path)

    assert result["summary"]["expected_task_count"] == 50
    assert result["summary"]["evaluation_scope"] == "parity_50"
    assert result["summary"]["scope_complete"] is True
    assert result["summary"]["missing_task_count"] == 0
    assert result["hard_safety_gates"]["passed"] is True


def test_parity_run_rejects_a_different_ordered_set_of_50(tmp_path):
    benchmark = _benchmark(tmp_path, quant_count=36, narrative_count=24)
    rows = _perfect_rows(benchmark)[:50]
    run_path = _write_run(benchmark, tmp_path / "wrong-parity", rows)

    with pytest.raises(ValueError, match="frozen parity membership order"):
        evaluate_agent_run(benchmark, run_path)


def test_strict_ten_case_launch_is_a_compatibility_smoke_universe(tmp_path):
    benchmark = _benchmark(tmp_path, quant_count=30, narrative_count=20)
    rows = _legacy_smoke_rows(benchmark)
    run_path = _write_run(benchmark, tmp_path / "compatibility-smoke", rows)
    _write_legacy_smoke_launch(run_path)

    result = evaluate_agent_run(benchmark, run_path)

    summary = result["summary"]
    assert result["run"]["evaluation_scope"] == "compatibility_smoke"
    assert result["run"]["case_count"] == 10
    assert result["run"]["full_benchmark_case_count"] == 50
    assert result["run"]["complete_full_benchmark"] is False
    assert summary["evaluation_scope"] == "compatibility_smoke"
    assert summary["scope_complete"] is True
    assert summary["expected_task_count"] == 10
    assert summary["full_benchmark_task_count"] == 50
    assert summary["completed_task_count"] == 10
    assert summary["missing_task_count"] == 0
    assert summary["schema_compliance"] == 1.0
    assert summary["repair_rate"] == 0.0
    assert summary["mean_tokens_per_task"] > 0
    assert summary["mean_latency_ms"] >= 0
    assert summary["throughput_tasks_per_second"] is not None
    assert result["hard_safety_gates"]["passed"] is True


@pytest.mark.parametrize(
    "launch_tamper",
    ["missing", "missing_partition", "wrong_limit", "wrong_provenance"],
)
def test_unproven_ten_case_prefix_remains_full_benchmark_incomplete(
    tmp_path, launch_tamper
):
    benchmark = _benchmark(tmp_path, quant_count=30, narrative_count=20)
    rows = _legacy_smoke_rows(benchmark)
    run_path = _write_run(benchmark, tmp_path / f"unproven-{launch_tamper}", rows)
    if launch_tamper == "missing_partition":
        _write_legacy_smoke_launch(run_path, partition="")
    elif launch_tamper == "wrong_limit":
        _write_legacy_smoke_launch(run_path, declared_limit=9)
    elif launch_tamper == "wrong_provenance":
        _write_legacy_smoke_launch(run_path, source_tree_sha256="8" * 64)

    result = evaluate_agent_run(benchmark, run_path)

    summary = result["summary"]
    assert summary["evaluation_scope"] == "full_benchmark_incomplete"
    assert summary["scope_complete"] is False
    assert summary["expected_task_count"] == 50
    assert summary["completed_task_count"] == 10
    assert summary["missing_task_count"] == 40
    assert result["hard_safety_gates"]["passed"] is False


def test_proven_smoke_rejects_a_nonprefix_selection(tmp_path):
    benchmark = _benchmark(tmp_path, quant_count=30, narrative_count=20)
    rows = _legacy_smoke_rows(benchmark, start=1)
    run_path = _write_run(benchmark, tmp_path / "wrong-smoke", rows)
    _write_legacy_smoke_launch(run_path)

    with pytest.raises(ValueError, match="frozen benchmark prefix"):
        evaluate_agent_run(benchmark, run_path)


def test_larger_partial_run_still_fails_full_benchmark_completeness(tmp_path):
    benchmark = _benchmark(tmp_path, quant_count=30, narrative_count=20)
    rows = _perfect_rows(benchmark)[:11]
    run_path = _write_run(benchmark, tmp_path / "incomplete-full", rows)

    result = evaluate_agent_run(benchmark, run_path)

    summary = result["summary"]
    assert summary["evaluation_scope"] == "full_benchmark_incomplete"
    assert summary["scope_complete"] is False
    assert summary["expected_task_count"] == 50
    assert summary["completed_task_count"] == 11
    assert summary["missing_task_count"] == 39
    assert result["hard_safety_gates"]["passed"] is False


def test_quantization_comparison_disqualifies_only_introduced_safety_failure(
    tmp_path,
):
    benchmark = _benchmark(tmp_path, quant_count=30, narrative_count=20)
    bf16_rows = _perfect_rows(
        benchmark, model_id="Qwen/Qwen3.5-27B", quantization="bf16"
    )
    quantized_rows = _perfect_rows(
        benchmark, model_id="Qwen/Qwen3.5-27B-FP8", quantization="fp8"
    )
    bad = next(
        row
        for row in quantized_rows
        if row["final_proposal"]["status"] == "OK"
        and row["final_proposal"]["value"] is not None
    )
    bad["final_proposal"]["value"] = "999999"
    bad["run_record"]["output_sha256"] = _canonical_sha256(bad["final_proposal"])
    bf16_path = _write_run(benchmark, tmp_path / "bf16", bf16_rows)
    quantized_path = _write_run(benchmark, tmp_path / "fp8", quantized_rows)

    bf16 = evaluate_agent_run(benchmark, bf16_path, run_name="bf16")
    quantized = evaluate_agent_run(benchmark, quantized_path, run_name="fp8")
    comparison = compare_quantization_runs(bf16, quantized)

    assert comparison["hard_safety_regression"] is True
    assert comparison["quantized_disqualified"] is True
    assert comparison["introduced_hard_safety_failures"]
    assert (
        comparison["metric_deltas"]["quantitative_answer_accuracy_percentage_points"]
        < 0
    )


def _comparison_stub(*, model_id: str, quantization: str) -> dict:
    task_ids = [f"parity-{index:02d}" for index in range(50)]
    return {
        "benchmark_id": "benchmark-fixture",
        "run": {
            "run_name": quantization,
            "model_id": model_id,
            "quantization": quantization,
            "prompt_condition": "zero_shot",
            "runtime_mode": "direct",
            "corpus_id": "fixture-corpus",
            "task_ids_sha256": _canonical_sha256(task_ids),
        },
        "summary": {
            "schema_compliance": 1.0,
            "routing_accuracy": 1.0,
            "quant_joint_accuracy": 1.0,
            "narrative_answerability_accuracy": 1.0,
            "citation_exactness": 1.0,
            "refusal_accuracy": 1.0,
            "repair_rate": 0.0,
        },
        "per_task": [
            {
                "task_id": task_id,
                "result_present": True,
                "safety_codes": [],
            }
            for task_id in task_ids
        ],
    }


def test_matched_baseline_comparisons_cover_required_experimental_axes():
    direct = _comparison_stub(model_id="Qwen/Qwen3.5-27B-FP8", quantization="fp8")
    direct["run"].update({"run_name": "qwen-zero-direct", "runtime_mode": "direct"})
    agent = copy.deepcopy(direct)
    agent["run"].update(
        {"run_name": "qwen-zero-agent", "runtime_mode": "capability_agent"}
    )
    few = copy.deepcopy(agent)
    few["run"].update({"run_name": "qwen-few-agent", "prompt_condition": "few_shot"})
    gemma = copy.deepcopy(agent)
    gemma["run"].update(
        {
            "run_name": "gemma-zero-agent",
            "model_id": "google/gemma-4-31B-it-qat-w4a16-ct",
            "quantization": "w4a16",
        }
    )

    comparisons = build_matched_baseline_comparisons([direct, agent, few, gemma])

    assert {item["comparison_type"] for item in comparisons} == {
        "direct_vs_agent",
        "zero_shot_vs_few_shot",
        "model_vs_model",
    }
    assert all(item["task_count"] == 50 for item in comparisons)
    model_comparison = next(
        item for item in comparisons if item["comparison_type"] == "model_vs_model"
    )
    assert model_comparison["challenger_run"] == "qwen-zero-agent"
    assert model_comparison["baseline_run"] == "gemma-zero-agent"


def test_matched_comparisons_reject_same_hash_with_different_task_order():
    direct = _comparison_stub(model_id="Qwen/Qwen3.5-27B-FP8", quantization="fp8")
    direct["run"].update({"run_name": "direct", "runtime_mode": "direct"})
    agent = copy.deepcopy(direct)
    agent["run"].update({"run_name": "agent", "runtime_mode": "capability_agent"})
    agent["per_task"].reverse()

    with pytest.raises(ValueError, match="identical ordered task membership"):
        build_matched_baseline_comparisons([direct, agent])


def test_quantization_comparison_rejects_unaligned_or_non_50_runs():
    bf16 = _comparison_stub(model_id="google/gemma-4-31B-it", quantization="bf16")
    quantized = _comparison_stub(
        model_id="google/gemma-4-31B-it-qat-w4a16-ct",
        quantization="qat-w4a16-ct",
    )
    assert compare_quantization_runs(bf16, quantized)["task_count"] == 50

    wrong_prompt = copy.deepcopy(quantized)
    wrong_prompt["run"]["prompt_condition"] = "few_shot"
    with pytest.raises(ValueError, match="matching prompt_condition"):
        compare_quantization_runs(bf16, wrong_prompt)

    wrong_family = copy.deepcopy(quantized)
    wrong_family["run"]["model_id"] = "Qwen/Qwen3.5-27B-FP8"
    with pytest.raises(ValueError, match="same model family"):
        compare_quantization_runs(bf16, wrong_family)

    wrong_role = copy.deepcopy(quantized)
    wrong_role["run"]["quantization"] = "bf16"
    with pytest.raises(ValueError, match="challenger run must use a quantized model"):
        compare_quantization_runs(bf16, wrong_role)

    only_49 = copy.deepcopy(quantized)
    only_49["per_task"].pop()
    with pytest.raises(ValueError, match="identical task IDs|exactly 50 task IDs"):
        compare_quantization_runs(bf16, only_49)


def test_writes_stable_json_csv_markdown_and_refuses_overwrite(tmp_path):
    benchmark = _benchmark(tmp_path, quant_count=30, narrative_count=20)
    bf16_path = _write_run(
        benchmark,
        tmp_path / "bf16",
        _perfect_rows(benchmark, quantization="bf16"),
    )
    fp8_path = _write_run(
        benchmark,
        tmp_path / "fp8",
        _perfect_rows(benchmark, quantization="fp8"),
    )
    agent_path = _write_run(
        benchmark,
        tmp_path / "agent",
        _agent_rows(benchmark),
    )

    result = write_evaluation_artifacts(
        benchmark,
        {"agent": agent_path, "bf16": bf16_path, "fp8": fp8_path},
        tmp_path / "evaluation",
        quantization_pairs=[("bf16", "fp8")],
    )

    assert result["run_count"] == 3
    assert result["all_hard_safety_gates_passed"] is True
    assert (tmp_path / "evaluation" / "metrics.json").is_file()
    assert (
        (tmp_path / "evaluation" / "metrics.csv")
        .read_text(encoding="utf-8")
        .startswith("run_name,")
    )
    report = (tmp_path / "evaluation" / "report.md").read_text(encoding="utf-8")
    assert "# AuditOps Agent Baseline Evaluation" in report
    assert "not an audit opinion" in report
    assert "Direct vs agent comparisons" in report
    assert "Quantization comparisons" in report
    assert "Scope complete" in report
    metrics = _read_json(tmp_path / "evaluation" / "metrics.json")
    assert metrics["all_runs_complete_full_benchmark"] is True
    assert {
        item["comparison_type"] for item in metrics["matched_baseline_comparisons"]
    } == {"direct_vs_agent"}

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_evaluation_artifacts(
            benchmark, {"bf16": bf16_path}, tmp_path / "evaluation"
        )


def test_smoke_report_labels_scope_and_full_benchmark_incompleteness(tmp_path):
    benchmark = _benchmark(tmp_path, quant_count=30, narrative_count=20)
    run_path = _write_run(
        benchmark,
        tmp_path / "compatibility-smoke",
        _legacy_smoke_rows(benchmark),
    )
    _write_legacy_smoke_launch(run_path)

    result = write_evaluation_artifacts(
        benchmark,
        {"gemma-smoke": run_path},
        tmp_path / "smoke-evaluation",
    )

    assert result["all_hard_safety_gates_passed"] is True
    assert result["all_runs_complete_full_benchmark"] is False
    report = (tmp_path / "smoke-evaluation" / "report.md").read_text(encoding="utf-8")
    assert "compatibility_smoke" in report
    assert "10/50" in report
    assert "not a complete benchmark run" in report
    metrics = _read_json(tmp_path / "smoke-evaluation" / "metrics.json")
    assert metrics["all_runs_complete_full_benchmark"] is False
    manifest = _read_json(tmp_path / "smoke-evaluation" / "evaluation_manifest.json")
    assert manifest["all_runs_complete_full_benchmark"] is False


def test_secret_scan_prevents_evaluation_report_publication(tmp_path, monkeypatch):
    synthetic_secret = "synthetic-evaluation-secret-123456"
    monkeypatch.setenv("HF_TOKEN", synthetic_secret)
    benchmark = _benchmark(tmp_path)
    run_path = _write_run(benchmark, tmp_path / "run", _perfect_rows(benchmark))
    output = tmp_path / "secret-evaluation"

    with pytest.raises(ValueError, match="Secret scan failed"):
        write_evaluation_artifacts(
            benchmark,
            {synthetic_secret: run_path},
            output,
        )

    assert not output.exists()
