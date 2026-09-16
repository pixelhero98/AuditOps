from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from auditops import agent_batch
from auditops.agent_batch import (
    BATCH_RUN_VERSION,
    CHECKPOINT_VERSION,
    FEW_SHOT_APPROVAL_VERSION,
    SUPPORTED_BENCHMARK_VERSION,
    run_agent_baseline,
    write_few_shot_approval,
    write_few_shot_review_packet,
)
from auditops.agent_contracts import (
    build_agent_proposal,
    build_agent_task_input,
    build_model_config,
)
from auditops.agent_tools import load_frozen_evidence
from auditops.model_adapter import MockModelAdapter


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quant_evidence(task_id: str, filing_id: str, entity_id: str):
    context_id = f"ctx-{task_id}"
    return {
        "task_id": task_id,
        "items": [
            {
                "evidence_id": f"{task_id}-fact-1",
                "filing_id": filing_id,
                "content": "input_name=assets_current; concept=us-gaap_AssetsCurrent; period_key=ASOF_20251231; unit=usd; value=200",
                "period_key": "ASOF_20251231",
                "unit": "usd",
                "value": "200",
                "metadata": {
                    "input_name": "assets_current",
                    "concept": "us-gaap_AssetsCurrent",
                    "context_id": context_id,
                    "entity_id": entity_id,
                    "dimensions": {},
                },
            },
            {
                "evidence_id": f"{task_id}-fact-2",
                "filing_id": filing_id,
                "content": "input_name=liabilities_current; concept=us-gaap_LiabilitiesCurrent; period_key=ASOF_20251231; unit=usd; value=100",
                "period_key": "ASOF_20251231",
                "unit": "usd",
                "value": "100",
                "metadata": {
                    "input_name": "liabilities_current",
                    "concept": "us-gaap_LiabilitiesCurrent",
                    "context_id": context_id,
                    "entity_id": entity_id,
                    "dimensions": {},
                },
            },
        ],
    }


def _quant_task(index: int):
    task_id = f"task-{index}"
    filing_id = f"filing-{index}"
    entity_id = f"ACME-{index}"
    task = build_agent_task_input(
        task_id=task_id,
        task_type="quant_metric",
        jurisdiction="US",
        reporting_framework="US_GAAP",
        standards_profile={"name": "PCAOB public-filing profile", "version": "2026"},
        source_system="SEC_EDGAR",
        question="What is the current ratio?",
        entity={"entity_id": entity_id, "name": f"Acme {index}"},
        filing={"filing_id": filing_id},
        period={"period_key": "ASOF_20251231"},
        metric_spec_id="current_ratio",
        allowed_tools=["evaluate_metric_spec"],
        evidence_ids=[f"{task_id}-fact-1", f"{task_id}-fact-2"],
        output_schema_id="quantitative_answer_or_refusal.v2",
        refusal_codes=["MISSING_INPUT"],
    )
    observation = {
        "tool_observation_version": "v2.2",
        "observation_kind": "METRIC",
        "visibility": "MODEL_VISIBLE",
        "task_id": task_id,
        "filing_id": filing_id,
        "metric_spec_id": "current_ratio",
        "status": "OK",
        "value": "2",
        "unit": "pure",
        "period_key": "ASOF_20251231",
        "evidence_ids": [f"{task_id}-fact-1", f"{task_id}-fact-2"],
        "refusal_code": None,
    }
    return task, _quant_evidence(task_id, filing_id, entity_id), observation


def _benchmark(tmp_path: Path, *, task_count: int = 1, few_shot_rows=None):
    root = tmp_path / "benchmark"
    root.mkdir()
    built = [_quant_task(index) for index in range(1, task_count + 1)]
    tasks = [item[0] for item in built]
    evidence = [item[1] for item in built]
    observations = [item[2] for item in built]
    files = {
        "cases.jsonl": tasks,
        "evidence.jsonl": evidence,
        "verifier_observations.jsonl": [
            {
                "task_id": task["task_id"],
                "tool_name": "evaluate_metric_spec",
                "observation": observation,
            }
            for task, observation in zip(tasks, observations)
        ],
        "few_shot.jsonl": list(few_shot_rows or []),
    }
    artifacts = {}
    for name, rows in files.items():
        _write_jsonl(root / name, rows)
        artifacts[name] = {"sha256": _sha(root / name), "records": len(rows)}
    manifest = {
        "benchmark_version": SUPPORTED_BENCHMARK_VERSION,
        "benchmark_id": "benchmark-1",
        "corpus_id": "corpus-1",
        "counts": {"total": task_count},
        "artifact_visibility": {
            "inference_visible": ["cases.jsonl", "evidence.jsonl", "few_shot.jsonl"],
            "evaluator_only": [
                "gold.jsonl",
                "verifier_observations.jsonl",
                "case_lineage.jsonl",
            ],
        },
        "artifacts": artifacts,
    }
    (root / "benchmark_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root, tasks[0], observations[0]


def _review_packet(root: Path, output: Path) -> Path:
    manifest = json.loads(
        (root / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    output.write_text(
        json.dumps(
            {
                "few_shot_review_packet_version": "auditops-few-shot-review-packet.v2.2",
                "benchmark_id": manifest["benchmark_id"],
                "few_shot_sha256": _sha(root / "few_shot.jsonl"),
                "rendered_messages_sha256": "a" * 64,
                "tokenizer_counts_sha256": "b" * 64,
                "max_demonstration_tokens": 100,
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    return output


def _answer(task_id: str):
    return {
        "task_id": task_id,
        "action": "ANSWER",
        "value": "2",
        "unit": "pure",
        "period_key": "ASOF_20251231",
        "evidence_ids": [f"{task_id}-fact-1", f"{task_id}-fact-2"],
    }


def _full_answer(task_id: str):
    return build_agent_proposal(
        task_id=task_id,
        action="ANSWER",
        status="OK",
        value="2",
        unit="pure",
        period_key="ASOF_20251231",
        evidence_ids=[f"{task_id}-fact-1", f"{task_id}-fact-2"],
        model_uncertainty="NONE",
    )


def _few_shot_rows():
    rows = []
    for index in range(10, 14):
        task, evidence, _ = _quant_task(index)
        rows.append(
            {
                "few_shot_example_version": "v2.2",
                "task_id": task["task_id"],
                "example_id": f"example-{index}",
                "review_status": "PENDING_HUMAN_REVIEW",
                "template_id": f"quant-template-{index}",
                "task_family": "quant_metric:current_ratio:OK",
                "task": task,
                "evidence_items": evidence["items"],
                "plan_response": {
                    "task_id": task["task_id"],
                    "action": "CALL_TOOL",
                    "tool_name": "evaluate_metric_spec",
                    "tool_arguments": {
                        "filing_id": task["filing"]["filing_id"],
                        "metric_spec_id": "current_ratio",
                        "period_key": "ASOF_20251231",
                    },
                },
                "tool_observation": {
                    **_quant_task(index)[2],
                    "tool_observation_version": "v2.2",
                    "observation_kind": "METRIC",
                    "visibility": "MODEL_VISIBLE",
                },
                "assistant_response": _full_answer(task["task_id"]),
                "source_task_sha256": f"{index:064x}",
            }
        )
    narrative_subtypes = (
        "footnote_note",
        "accounting_policy",
        "auditor_report_opinion_language",
        "critical_audit_matter",
    )
    for offset in range(16):
        index = 14 + offset
        subtype = narrative_subtypes[offset // 4]
        task_id = f"narrative-{index}"
        filing_id = f"filing-{index}"
        evidence_item = {
            "evidence_id": f"{task_id}-chunk-1",
            "filing_id": filing_id,
            "content": "The auditor expressed an unqualified opinion.",
            "rank": 1,
            "period_key": "FY2025",
            "metadata": {"heading": "auditor report"},
        }
        task = build_agent_task_input(
            task_id=task_id,
            task_type="narrative_citation",
            jurisdiction="US",
            reporting_framework="US_GAAP",
            standards_profile={
                "name": "PCAOB public-filing profile",
                "version": "2026",
            },
            source_system="SEC_EDGAR",
            question="What opinion language is disclosed?",
            entity={"entity_id": f"ACME-{index}", "name": f"Acme {index}"},
            filing={"filing_id": filing_id},
            period={"period_key": "FY2025"},
            retrieval_query="auditor opinion language",
            narrative_subtype=subtype,
            allowed_tools=["load_frozen_evidence"],
            evidence_ids=[evidence_item["evidence_id"]],
            output_schema_id="narrative_answer_or_refusal.v2",
            refusal_codes=["NARRATIVE_NOT_SUPPORTED"],
        )
        observation = load_frozen_evidence(
            task, [evidence_item], visibility="MODEL_VISIBLE"
        )
        if offset % 4 < 2:
            response = build_agent_proposal(
                task_id=task_id,
                action="ANSWER",
                status="OK",
                period_key="FY2025",
                answer_text="The auditor expressed an unqualified opinion.",
                evidence_ids=[evidence_item["evidence_id"]],
                claims=[
                    {
                        "claim_id": "claim-1",
                        "text": "The auditor expressed an unqualified opinion.",
                        "evidence_ids": [evidence_item["evidence_id"]],
                        "supporting_text": "The auditor expressed an unqualified opinion.",
                    }
                ],
            )
            status = "OK"
        else:
            response = build_agent_proposal(
                task_id=task_id,
                action="REFUSE",
                status="REFUSAL",
                period_key="FY2025",
                refusal_code="NARRATIVE_NOT_SUPPORTED",
            )
            status = "REFUSAL"
        rows.append(
            {
                "few_shot_example_version": "v2.2",
                "task_id": task_id,
                "example_id": f"example-{index}",
                "review_status": "PENDING_HUMAN_REVIEW",
                "template_id": f"narrative-template-{index}",
                "task_family": f"narrative_citation:{subtype}:{status}",
                "task": task,
                "evidence_items": [evidence_item],
                "plan_response": {
                    "task_id": task_id,
                    "action": "CALL_TOOL",
                    "tool_name": "load_frozen_evidence",
                    "tool_arguments": {
                        "filing_id": filing_id,
                        "evidence_scope_sha256": observation["evidence_scope_sha256"],
                    },
                },
                "tool_observation": observation,
                "assistant_response": response,
                "source_task_sha256": f"{index:064x}",
            }
        )
    return rows


def test_run_direct_batch_never_passes_observation_to_prompt(tmp_path):
    root, task, observation = _benchmark(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = build_model_config(
        model_id="mock/model", revision="test-v1", model_path=None, backend="mock"
    )
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    proposal = _answer(task["task_id"])
    adapter = MockModelAdapter([proposal], model_id="mock/model", revision="test-v1")
    manifest = run_agent_baseline(
        root,
        config_path,
        tmp_path / "run",
        runtime_mode="direct",
        prompt_condition="zero_shot",
        adapter=adapter,
        strict_provenance=False,
    )
    assert manifest["case_count"] == 1
    prompt = "\n".join(message["content"] for message in adapter.requests[0].messages)
    assert json.dumps(observation, sort_keys=True) not in prompt
    row = json.loads((tmp_path / "run" / "results.jsonl").read_text(encoding="utf-8"))
    assert row["agent_case_result_version"] == "v2.2"
    assert row["outcome"] == "RELEASED"
    assert row["run_record"]["outcome"] == "RELEASED"
    assert manifest["batch_run_version"] == BATCH_RUN_VERSION


def test_v2_batch_rejects_legacy_benchmark_without_creating_checkpoint(tmp_path):
    root, task, _ = _benchmark(tmp_path)
    manifest_path = root / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark_version"] = "agent_benchmark.v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = build_model_config(
        model_id="mock/model", revision="test-v1", model_path=None, backend="mock"
    )
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Text-agent v2 requires benchmark_version"):
        run_agent_baseline(
            root,
            config_path,
            tmp_path / "legacy-run",
            runtime_mode="direct",
            prompt_condition="zero_shot",
            adapter=MockModelAdapter(
                [_answer(task["task_id"])],
                model_id="mock/model",
                revision="test-v1",
            ),
            strict_provenance=False,
        )

    assert not (tmp_path / ".legacy-run.checkpoint").exists()


def test_batch_revalidates_exact_registry_binding_before_adapter_use(tmp_path):
    root, task, _ = _benchmark(tmp_path)
    task["allowed_tools"] = ["evaluate_metric_spec", "load_frozen_evidence"]
    cases_path = root / "cases.jsonl"
    _write_jsonl(cases_path, [task])
    manifest_path = root / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["cases.jsonl"]["sha256"] = _sha(cases_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = build_model_config(
        model_id="mock/model", revision="test-v1", model_path=None, backend="mock"
    )
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    adapter = MockModelAdapter(
        [_answer(task["task_id"])], model_id="mock/model", revision="test-v1"
    )

    with pytest.raises(ValueError, match="must be exactly"):
        run_agent_baseline(
            root,
            config_path,
            tmp_path / "wrong-binding-run",
            runtime_mode="capability_agent",
            prompt_condition="zero_shot",
            adapter=adapter,
            strict_provenance=False,
        )

    assert adapter.requests == []
    assert not (tmp_path / ".wrong-binding-run.checkpoint").exists()


def test_agent_batch_never_opens_evaluator_oracle_for_tool_prompt(tmp_path):
    root, task, verifier_observation = _benchmark(tmp_path)
    oracle_marker = "EVALUATOR_ONLY_DO_NOT_OPEN_8472"
    _write_jsonl(
        root / "verifier_observations.jsonl",
        [
            {
                "task_id": task["task_id"],
                "tool_name": "evaluate_metric_spec",
                "observation": {**verifier_observation, "oracle_marker": oracle_marker},
            }
        ],
    )
    config = build_model_config(
        model_id="mock/model", revision="test-v1", model_path=None, backend="mock"
    )
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    plan = {
        "task_id": task["task_id"],
        "action": "CALL_TOOL",
        "tool_name": "evaluate_metric_spec",
        "tool_arguments": {
            "filing_id": "filing-1",
            "metric_spec_id": "current_ratio",
            "period_key": "ASOF_20251231",
        },
    }
    answer = _answer(task["task_id"])
    adapter = MockModelAdapter(
        [plan, answer], model_id="mock/model", revision="test-v1"
    )

    run_agent_baseline(
        root,
        config_path,
        tmp_path / "agent-run",
        runtime_mode="capability_agent",
        prompt_condition="zero_shot",
        adapter=adapter,
        strict_provenance=False,
    )

    synthesis = "\n".join(
        message["content"] for message in adapter.requests[1].messages
    )
    assert '"status":"OK"' in synthesis
    assert oracle_marker not in synthesis


def test_few_shot_requires_exact_human_approval(tmp_path):
    root, _, _ = _benchmark(tmp_path)
    with pytest.raises(ValueError):
        write_few_shot_approval(
            root,
            tmp_path / "approval.json",
            reviewer="reviewer",
            approved_example_ids=["not-a-frozen-example"],
            reviewed_at="2026-08-21T00:00:00Z",
            review_packet_path=tmp_path / "missing-review.json",
        )


def test_review_packet_binds_exact_messages_and_both_pinned_tokenizers(
    tmp_path, monkeypatch
):
    examples = _few_shot_rows()
    root, _, _ = _benchmark(tmp_path, few_shot_rows=examples)
    configs = []
    for filename, model_id in (
        ("qwen.json", "Qwen/Qwen3.5-27B-FP8"),
        ("gemma.json", "google/gemma-4-31B-it-qat-w4a16-ct"),
    ):
        model_path = tmp_path / filename.removesuffix(".json")
        model_path.mkdir()
        config = build_model_config(
            model_id=model_id,
            revision="pinned-test-revision",
            model_path=str(model_path),
            backend="mock",
            chat_template_sha256="c" * 64,
        )
        config_path = tmp_path / filename
        config_path.write_text(json.dumps(config), encoding="utf-8")
        configs.append(config_path)

    monkeypatch.setattr(
        agent_batch,
        "_tokenizer_message_count",
        lambda _path, messages: len(messages) * 10,
    )
    packet_path = tmp_path / "review-packet.json"
    packet = write_few_shot_review_packet(root, configs, packet_path)

    assert packet["review_status"] == "PENDING_HUMAN_REVIEW"
    assert packet["passed"] is True
    assert packet["max_demonstration_tokens"] == 80
    assert len(packet["example_source_bindings"]) == 20
    assert {row["model_id"] for row in packet["tokenizer_counts"]} == {
        "Qwen/Qwen3.5-27B-FP8",
        "google/gemma-4-31B-it-qat-w4a16-ct",
    }
    assert set(packet["rendered_messages"]) == {
        "quant_metric",
        "footnote_note",
        "accounting_policy",
        "auditor_report_opinion_language",
        "critical_audit_matter",
    }
    for family in packet["rendered_messages"].values():
        assert set(family) == {"direct", "plan", "synthesis"}
        assert all(len(messages) == 8 for messages in family.values())

    approval_path = tmp_path / "approval.json"
    approval = write_few_shot_approval(
        root,
        approval_path,
        reviewer="human-reviewer",
        approved_example_ids=[row["example_id"] for row in examples],
        reviewed_at="2026-08-23T00:00:00Z",
        review_packet_path=packet_path,
    )
    assert approval["review_packet_sha256"] == _sha(packet_path)
    assert len(approval["review_binding_sha256"]) == 64


def test_review_token_count_reads_batch_encoding_input_ids(monkeypatch):
    class FakeTokenizer:
        @staticmethod
        def apply_chat_template(*_args, **_kwargs):
            return {"input_ids": [[1, 2, 3, 4]], "attention_mask": [[1, 1, 1, 1]]}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )

    assert (
        agent_batch._tokenizer_message_count(
            "/pinned/model", [{"role": "user", "content": "synthetic"}]
        )
        == 4
    )


def test_late_case_exception_preserves_checkpoint_and_resumes(tmp_path, monkeypatch):
    root, _, _ = _benchmark(tmp_path, task_count=2)
    config = build_model_config(
        model_id="mock/model", revision="test-v1", model_path=None, backend="mock"
    )
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    destination = tmp_path / "run"
    original_run_agent_case = agent_batch.run_agent_case

    def fail_second_case(case, **kwargs):
        if case["task_id"] == "task-2":
            raise RuntimeError("backend details must not be persisted")
        return original_run_agent_case(case, **kwargs)

    monkeypatch.setattr(agent_batch, "run_agent_case", fail_second_case)
    first_adapter = MockModelAdapter(
        [_answer("task-1")], model_id="mock/model", revision="test-v1"
    )
    with pytest.raises(RuntimeError, match="backend details"):
        run_agent_baseline(
            root,
            config_path,
            destination,
            runtime_mode="direct",
            prompt_condition="zero_shot",
            adapter=first_adapter,
            strict_provenance=False,
        )

    checkpoint = tmp_path / ".run.checkpoint"
    assert checkpoint.is_dir()
    assert not destination.exists()
    checkpoint_manifest = json.loads(
        (checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    assert checkpoint_manifest["checkpoint_version"] == CHECKPOINT_VERSION
    case_files = list((checkpoint / "case_results").glob("*.json"))
    assert len(case_files) == 1
    assert json.loads(case_files[0].read_text(encoding="utf-8"))["task_id"] == "task-1"
    failures = [
        json.loads(line)
        for line in (checkpoint / "batch_failure_manifests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert failures[-1]["task_id"] == "task-2"
    assert failures[-1]["completed_case_count"] == 1
    assert failures[-1]["exception_type"] == "RuntimeError"
    assert "backend details" not in json.dumps(failures)

    checkpoint_bytes = case_files[0].read_bytes()
    changed_config = build_model_config(
        model_id="mock/model", revision="test-v2", model_path=None, backend="mock"
    )
    config_path.write_text(json.dumps(changed_config), encoding="utf-8")
    with pytest.raises(ValueError, match="Checkpoint bindings differ"):
        run_agent_baseline(
            root,
            config_path,
            destination,
            runtime_mode="direct",
            prompt_condition="zero_shot",
            adapter=MockModelAdapter(
                [_answer("task-2")], model_id="mock/model", revision="test-v2"
            ),
            strict_provenance=False,
        )
    assert case_files[0].read_bytes() == checkpoint_bytes
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr(agent_batch, "run_agent_case", original_run_agent_case)
    resumed_adapter = MockModelAdapter(
        [_answer("task-2")], model_id="mock/model", revision="test-v1"
    )
    manifest = run_agent_baseline(
        root,
        config_path,
        destination,
        runtime_mode="direct",
        prompt_condition="zero_shot",
        adapter=resumed_adapter,
        strict_provenance=False,
    )

    assert manifest["case_count"] == 2
    assert len(resumed_adapter.requests) == 1
    rows = [
        json.loads(line)
        for line in (destination / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["task_id"] for row in rows] == ["task-1", "task-2"]
    assert all(row["outcome"] == "RELEASED" for row in rows)
    assert all(row["run_record"]["outcome"] == "RELEASED" for row in rows)
    assert not checkpoint.exists()


def test_few_shot_run_retains_full_examples_and_approval_artifact(tmp_path):
    examples = _few_shot_rows()
    root, task, _ = _benchmark(tmp_path, few_shot_rows=examples)
    approval_path = tmp_path / "human-approval.json"
    review_packet_path = _review_packet(root, tmp_path / "review-packet.json")
    write_few_shot_approval(
        root,
        approval_path,
        reviewer="human-reviewer",
        approved_example_ids=[row["example_id"] for row in examples],
        reviewed_at="2026-08-21T00:00:00Z",
        review_packet_path=review_packet_path,
    )
    assert (
        json.loads(approval_path.read_text(encoding="utf-8"))[
            "few_shot_approval_version"
        ]
        == FEW_SHOT_APPROVAL_VERSION
    )
    config = build_model_config(
        model_id="mock/model", revision="test-v1", model_path=None, backend="mock"
    )
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    adapter = MockModelAdapter(
        [_answer(task["task_id"])], model_id="mock/model", revision="test-v1"
    )
    destination = tmp_path / "few-shot-run"

    manifest = run_agent_baseline(
        root,
        config_path,
        destination,
        runtime_mode="direct",
        prompt_condition="few_shot",
        few_shot_approval=approval_path,
        adapter=adapter,
        strict_provenance=False,
    )

    retained_approval = destination / "few_shot_approval.json"
    assert retained_approval.read_bytes() == approval_path.read_bytes()
    assert manifest["inputs"]["few_shot_approval_sha256"] == _sha(retained_approval)
    assert set(manifest["inputs"]) == {
        "cases_sha256",
        "evidence_sha256",
        "few_shot_sha256",
        "few_shot_approval_sha256",
        "parity_membership_sha256",
    }
    prompt = "\n".join(message["content"] for message in adapter.requests[0].messages)
    selected_examples = [
        example
        for example in examples
        if example["task"]["task_type"] == "quant_metric"
    ]
    assert len(selected_examples) == 4
    for example in selected_examples:
        assert example["task"]["task_id"] in prompt
        assert example["evidence_items"][0]["evidence_id"] in prompt
        assert example["assistant_response"]["task_id"] in prompt
    assert all(
        example["task"]["task_id"] not in prompt
        for example in examples
        if example["task"]["task_type"] == "narrative_citation"
    )


def test_secret_scan_blocks_checkpoint_case_and_run_publication(tmp_path, monkeypatch):
    synthetic_secret = "synthetic-batch-case-secret-123456"
    monkeypatch.setenv("HF_TOKEN", synthetic_secret)
    root, task, _ = _benchmark(tmp_path)
    config = build_model_config(
        model_id=f"mock/{synthetic_secret}",
        revision="test-v1",
        model_path=None,
        backend="mock",
    )
    config_path = tmp_path / "secret-model.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    adapter = MockModelAdapter(
        [_answer(task["task_id"])],
        model_id=f"mock/{synthetic_secret}",
        revision="test-v1",
    )
    destination = tmp_path / "secret-run"

    with pytest.raises(ValueError, match="Secret scan failed"):
        run_agent_baseline(
            root,
            config_path,
            destination,
            runtime_mode="direct",
            prompt_condition="zero_shot",
            adapter=adapter,
            strict_provenance=False,
        )

    assert not destination.exists()
    checkpoint = tmp_path / ".secret-run.checkpoint"
    assert checkpoint.is_dir()
    assert not list((checkpoint / "case_results").glob("*.json"))


def test_secret_scan_blocks_copied_few_shot_approval(tmp_path, monkeypatch):
    synthetic_secret = "synthetic-approval-secret-123456"
    monkeypatch.setenv("HF_TOKEN", synthetic_secret)
    examples = _few_shot_rows()
    root, task, _ = _benchmark(tmp_path, few_shot_rows=examples)
    benchmark_manifest = json.loads(
        (root / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    approval_path = tmp_path / "untrusted-approval.json"
    approval_path.write_text(
        json.dumps(
            {
                "few_shot_approval_version": FEW_SHOT_APPROVAL_VERSION,
                "benchmark_id": benchmark_manifest["benchmark_id"],
                "few_shot_sha256": _sha(root / "few_shot.jsonl"),
                "review_packet_sha256": "a" * 64,
                "review_binding_sha256": "b" * 64,
                "reviewer": synthetic_secret,
                "reviewed_at": "2026-08-21T00:00:00Z",
                "approved_example_ids": sorted(row["example_id"] for row in examples),
                "decision": "APPROVED",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = build_model_config(
        model_id="mock/model", revision="test-v1", model_path=None, backend="mock"
    )
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    destination = tmp_path / "secret-approval-run"

    with pytest.raises(ValueError, match="Secret scan failed"):
        run_agent_baseline(
            root,
            config_path,
            destination,
            runtime_mode="direct",
            prompt_condition="few_shot",
            few_shot_approval=approval_path,
            adapter=MockModelAdapter(
                [_answer(task["task_id"])],
                model_id="mock/model",
                revision="test-v1",
            ),
            strict_provenance=False,
        )

    assert not destination.exists()
    assert not (tmp_path / ".secret-approval-run.checkpoint").exists()


def test_human_approval_writer_scans_before_publication(tmp_path, monkeypatch):
    synthetic_secret = "synthetic-reviewer-secret-123456"
    monkeypatch.setenv("HF_TOKEN", synthetic_secret)
    examples = _few_shot_rows()
    root, _, _ = _benchmark(tmp_path, few_shot_rows=examples)
    output = tmp_path / "approval.json"
    review_packet_path = _review_packet(root, tmp_path / "review-packet.json")

    with pytest.raises(ValueError, match="Secret scan failed"):
        write_few_shot_approval(
            root,
            output,
            reviewer=synthetic_secret,
            approved_example_ids=[row["example_id"] for row in examples],
            reviewed_at="2026-08-21T00:00:00Z",
            review_packet_path=review_packet_path,
        )

    assert not output.exists()


def test_complete_batch_scans_final_staging_tree_before_publication(
    tmp_path, monkeypatch
):
    root, task, _ = _benchmark(tmp_path)
    config = build_model_config(
        model_id="mock/model", revision="test-v1", model_path=None, backend="mock"
    )
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    destination = tmp_path / "scanned-run"
    observed_scans: list[tuple[str, bool]] = []
    real_scan = agent_batch.assert_no_secrets

    def observe_scan(paths):
        resolved = [Path(path) for path in paths]
        observed_scans.extend((path.name, path.is_dir()) for path in resolved)
        return real_scan(paths)

    monkeypatch.setattr(agent_batch, "assert_no_secrets", observe_scan)
    run_agent_baseline(
        root,
        config_path,
        destination,
        runtime_mode="direct",
        prompt_condition="zero_shot",
        adapter=MockModelAdapter(
            [_answer(task["task_id"])],
            model_id="mock/model",
            revision="test-v1",
        ),
        strict_provenance=False,
    )

    assert destination.is_dir()
    assert any(
        is_directory and name.startswith(".scanned-run.publish.")
        for name, is_directory in observed_scans
    )
