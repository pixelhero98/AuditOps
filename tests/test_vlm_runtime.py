from __future__ import annotations

import csv
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

import auditops.vlm_runtime as vlm_runtime
from auditops.agent_contracts import build_model_config
from auditops.companies_house import build_companies_house_vlm_diagnostic
from auditops.model_adapter import ModelGenerationError
from auditops.vlm_evaluation import evaluate_companies_house_vlm_diagnostic
from auditops.vlm_runtime import (
    VLM_PREDICTION_JSON_SCHEMA,
    MockVisionModelAdapter,
    OfflineVLLMVisionAdapter,
    PageImage,
    VisionModelRequest,
    run_companies_house_vlm_diagnostic,
    write_vlm_few_shot_approval,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _build_diagnostic(
    root: Path,
    *,
    name: str,
    company_numbers: list[str],
) -> tuple[Path, list[dict]]:
    source = root / f"{name}-source"
    source.mkdir()
    pairs = source / "pairs.csv"
    with pairs.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "company_number",
                "transaction_id",
                "document_id",
                "pdf_path",
                "pdf_page_count",
            ],
        )
        writer.writeheader()
        for index, company_number in enumerate(company_numbers, start=1):
            pdf = source / f"filing-{index}.pdf"
            pdf.write_bytes(b"%PDF-1.4\nfixture\n%%EOF\n")
            writer.writerow(
                {
                    "pair_id": f"{name}-{index}",
                    "company_number": company_number,
                    "transaction_id": f"tx-{name}-{index}",
                    "document_id": f"doc-{name}-{index}",
                    "pdf_path": str(pdf.resolve()),
                    "pdf_page_count": "1",
                }
            )
    facts = source / "facts.csv"
    with facts.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "fact_label",
                "ixbrl_tag_concept",
                "ixbrl_value",
                "unit",
                "period_context",
                "pdf_page",
                "pdf_evidence",
                "usable_as_benchmark",
            ],
        )
        writer.writeheader()
        for index, _ in enumerate(company_numbers, start=1):
            writer.writerow(
                {
                    "pair_id": f"{name}-{index}",
                    "fact_label": "Current assets",
                    "ixbrl_tag_concept": "uk-core:CurrentAssets",
                    "ixbrl_value": str(index * 100),
                    "unit": "GBP",
                    "period_context": "Current year",
                    "pdf_page": "page 1",
                    "pdf_evidence": f"Current assets {index * 100}",
                    "usable_as_benchmark": "yes",
                }
            )
    output = root / name
    build_companies_house_vlm_diagnostic(
        pairs,
        facts,
        output,
        expected_pair_count=len(company_numbers),
        expected_fact_count=len(company_numbers),
    )
    cases = [
        json.loads(line)
        for line in (output / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return output, cases


def _model_config(path: Path) -> Path:
    config = build_model_config(
        model_id="auditops/mock-vlm",
        revision="test-v1",
        model_path=None,
        quantization=None,
        backend="mock",
        chat_template_sha256=None,
    )
    path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _page_provider(case: dict, pdf_root: Path, output: Path) -> tuple[PageImage, ...]:
    pdf = Path(case["pdf_path"]).resolve(strict=True)
    pdf.relative_to(pdf_root.resolve(strict=True))
    output.mkdir(parents=True, exist_ok=True)
    image = output / "page-1.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfixture-image")
    return (PageImage(1, image),)


def _prediction(task_id: str, value: str = "100") -> dict:
    return {
        "task_id": task_id,
        "facts": [
            {
                "fact_label": "Current assets",
                "value": value,
                "unit": "GBP",
                "period_context": "Current year",
                "page": 1,
                "bbox": [0.1, 0.2, 0.8, 0.3],
                "evidence_text": f"Current assets {value}",
            }
        ],
    }


def test_mock_zero_shot_run_is_exact_immutable_and_evaluator_compatible(
    tmp_path: Path,
) -> None:
    diagnostic, cases = _build_diagnostic(
        tmp_path, name="evaluation", company_numbers=["00000001"]
    )
    config = _model_config(tmp_path / "model.json")
    adapter = MockVisionModelAdapter([_prediction(cases[0]["task_id"])])
    output = tmp_path / "run"

    result = run_companies_house_vlm_diagnostic(
        diagnostic,
        config,
        output,
        pdf_root=tmp_path / "evaluation-source",
        prompt_condition="zero_shot",
        adapter=adapter,
        page_image_provider=_page_provider,
    )

    manifest = result["run_manifest"]
    assert manifest["case_count"] == manifest["result_count"] == 1
    assert manifest["typed_infrastructure_failure_count"] == 0
    assert manifest["isolated_from_xhtml_score"] is True
    assert not (output.parent / ".run.vlm-inprogress").exists()
    assert len(list((output / "cases").glob("*.json"))) == 1
    assert "gold" not in json.dumps(adapter.requests[0].prompt).lower()
    assert len(adapter.requests[0].images) == 1
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_companies_house_vlm_diagnostic(
            diagnostic,
            config,
            output,
            pdf_root=tmp_path / "evaluation-source",
            adapter=MockVisionModelAdapter([_prediction(cases[0]["task_id"])]),
            page_image_provider=_page_provider,
        )

    evaluation = evaluate_companies_house_vlm_diagnostic(
        diagnostic, output / "predictions.jsonl", tmp_path / "evaluation-report"
    )
    assert evaluation["metrics"]["summary"]["joint_fact_accuracy"] == 1.0


def test_zero_shot_runner_never_opens_evaluator_gold(
    tmp_path: Path, monkeypatch
) -> None:
    diagnostic, cases = _build_diagnostic(
        tmp_path, name="evaluation", company_numbers=["00000001"]
    )
    original = vlm_runtime._read_jsonl

    def gold_tripwire(path: Path):
        if Path(path).name == "gold.jsonl":
            raise AssertionError("evaluation gold reached the inference runner")
        return original(path)

    monkeypatch.setattr(vlm_runtime, "_read_jsonl", gold_tripwire)
    run_companies_house_vlm_diagnostic(
        diagnostic,
        _model_config(tmp_path / "model.json"),
        tmp_path / "run",
        pdf_root=tmp_path / "evaluation-source",
        adapter=MockVisionModelAdapter([_prediction(cases[0]["task_id"])]),
        page_image_provider=_page_provider,
    )


def test_secret_echo_is_rejected_before_checkpoint_publication(
    tmp_path: Path, monkeypatch
) -> None:
    diagnostic, cases = _build_diagnostic(
        tmp_path, name="evaluation", company_numbers=["00000001"]
    )
    secret = "hf_fixture_secret_value_123456789"
    monkeypatch.setenv("HF_TOKEN", secret)
    prediction = _prediction(cases[0]["task_id"])
    prediction["facts"][0]["evidence_text"] = secret
    with pytest.raises(ValueError, match="Secret scan failed"):
        run_companies_house_vlm_diagnostic(
            diagnostic,
            _model_config(tmp_path / "model.json"),
            tmp_path / "run",
            pdf_root=tmp_path / "evaluation-source",
            adapter=MockVisionModelAdapter([prediction]),
            page_image_provider=_page_provider,
        )
    assert not (tmp_path / "run").exists()
    assert not list((tmp_path / ".run.vlm-inprogress" / "cases").glob("*.json"))


def test_runtime_rechecks_external_model_config_pin(
    tmp_path: Path, monkeypatch
) -> None:
    diagnostic, cases = _build_diagnostic(
        tmp_path, name="evaluation", company_numbers=["00000001"]
    )
    monkeypatch.setenv("AUDITOPS_MODEL_CONFIG_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="differs from AUDITOPS_MODEL_CONFIG_SHA256"):
        run_companies_house_vlm_diagnostic(
            diagnostic,
            _model_config(tmp_path / "model.json"),
            tmp_path / "run",
            pdf_root=tmp_path / "evaluation-source",
            adapter=MockVisionModelAdapter([_prediction(cases[0]["task_id"])]),
            page_image_provider=_page_provider,
        )
    assert not (tmp_path / "run").exists()


def test_model_failure_and_invalid_bbox_become_typed_complete_rows(
    tmp_path: Path,
) -> None:
    diagnostic, cases = _build_diagnostic(
        tmp_path, name="evaluation", company_numbers=["00000001", "00000002"]
    )
    config = _model_config(tmp_path / "model.json")
    invalid = _prediction(cases[1]["task_id"], value="200")
    invalid["facts"][0]["bbox"] = [0.8, 0.2, 0.1, 0.3]
    adapter = MockVisionModelAdapter(
        [ModelGenerationError("GPU failure details must not persist"), invalid]
    )

    result = run_companies_house_vlm_diagnostic(
        diagnostic,
        config,
        tmp_path / "run",
        pdf_root=tmp_path / "evaluation-source",
        adapter=adapter,
        page_image_provider=_page_provider,
    )
    assert result["run_manifest"]["result_count"] == 2
    assert result["run_manifest"]["typed_infrastructure_failure_count"] == 2
    predictions = [
        json.loads(line)
        for line in (tmp_path / "run" / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["failure"]["code"] for row in predictions] == [
        "MODEL_GENERATION_FAILED",
        "MODEL_OUTPUT_INVALID",
    ]
    assert "GPU failure details" not in (
        tmp_path / "run" / "run_records.jsonl"
    ).read_text(encoding="utf-8")
    evaluation = evaluate_companies_house_vlm_diagnostic(
        diagnostic,
        tmp_path / "run" / "predictions.jsonl",
        tmp_path / "evaluation-report",
    )
    assert evaluation["metrics"]["summary"]["typed_failure_task_count"] == 2


def test_interrupted_run_resumes_without_reexecuting_checkpointed_case(
    tmp_path: Path,
) -> None:
    diagnostic, cases = _build_diagnostic(
        tmp_path, name="evaluation", company_numbers=["00000001", "00000002"]
    )
    config = _model_config(tmp_path / "model.json")
    first_adapter = MockVisionModelAdapter(
        [_prediction(cases[0]["task_id"]), _prediction(cases[1]["task_id"], "200")]
    )
    calls = 0

    def interrupted_provider(
        case: dict, root: Path, output: Path
    ) -> tuple[PageImage, ...]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return _page_provider(case, root, output)

    with pytest.raises(KeyboardInterrupt):
        run_companies_house_vlm_diagnostic(
            diagnostic,
            config,
            tmp_path / "run",
            pdf_root=tmp_path / "evaluation-source",
            adapter=first_adapter,
            page_image_provider=interrupted_provider,
        )
    assert len(first_adapter.requests) == 1
    assert len(list((tmp_path / ".run.vlm-inprogress" / "cases").glob("*.json"))) == 1

    resumed_adapter = MockVisionModelAdapter(
        [_prediction(cases[1]["task_id"], value="200")]
    )
    result = run_companies_house_vlm_diagnostic(
        diagnostic,
        config,
        tmp_path / "run",
        pdf_root=tmp_path / "evaluation-source",
        adapter=resumed_adapter,
        page_image_provider=_page_provider,
    )
    assert result["run_manifest"]["result_count"] == 2
    assert [request.request_id for request in resumed_adapter.requests] == [
        cases[1]["task_id"]
    ]


def test_four_entity_disjoint_approved_visual_examples_reach_few_shot_prompt(
    tmp_path: Path,
) -> None:
    evaluation, evaluation_cases = _build_diagnostic(
        tmp_path, name="evaluation", company_numbers=["00000001"]
    )
    few_shot, few_cases = _build_diagnostic(
        tmp_path,
        name="few-shot",
        company_numbers=["00000101", "00000102", "00000103", "00000104"],
    )
    visual_examples = tmp_path / "visual-examples.jsonl"
    _write_jsonl(
        visual_examples,
        [
            _prediction(case["task_id"], value=str(index * 100))
            for index, case in enumerate(few_cases, start=1)
        ],
    )
    approval = tmp_path / "visual-approval.json"
    approved = write_vlm_few_shot_approval(
        few_shot,
        evaluation,
        visual_examples,
        approval,
        reviewer="fixture-reviewer",
        reviewed_at="2026-08-21T12:00:00Z",
    )
    assert len(approved["approved_example_ids"]) == 4

    config = _model_config(tmp_path / "model.json")
    adapter = MockVisionModelAdapter([_prediction(evaluation_cases[0]["task_id"])])
    result = run_companies_house_vlm_diagnostic(
        evaluation,
        config,
        tmp_path / "run",
        pdf_root=tmp_path / "evaluation-source",
        prompt_condition="few_shot",
        few_shot_diagnostic_dir=few_shot,
        visual_examples_jsonl=visual_examples,
        few_shot_approval=approval,
        few_shot_pdf_root=tmp_path / "few-shot-source",
        adapter=adapter,
        page_image_provider=_page_provider,
    )
    assert len(adapter.requests) == 1
    assert len(adapter.requests[0].examples) == 4
    assert all(len(example.images) == 1 for example in adapter.requests[0].examples)
    assert (
        result["run_manifest"]["few_shot_approval_sha256"]
        == approved["approval_sha256"]
    )

    visual_examples.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum differs"):
        run_companies_house_vlm_diagnostic(
            evaluation,
            config,
            tmp_path / "tampered-run",
            pdf_root=tmp_path / "evaluation-source",
            prompt_condition="few_shot",
            few_shot_diagnostic_dir=few_shot,
            visual_examples_jsonl=visual_examples,
            few_shot_approval=approval,
            few_shot_pdf_root=tmp_path / "few-shot-source",
            adapter=MockVisionModelAdapter(
                [_prediction(evaluation_cases[0]["task_id"])]
            ),
            page_image_provider=_page_provider,
        )


def test_few_shot_approval_rejects_evaluation_entity_overlap(tmp_path: Path) -> None:
    evaluation, _ = _build_diagnostic(
        tmp_path, name="evaluation", company_numbers=["00000001"]
    )
    few_shot, few_cases = _build_diagnostic(
        tmp_path,
        name="few-shot",
        company_numbers=["00000001", "00000102", "00000103", "00000104"],
    )
    examples = tmp_path / "examples.jsonl"
    _write_jsonl(
        examples,
        [
            _prediction(case["task_id"], value=str(index * 100))
            for index, case in enumerate(few_cases, start=1)
        ],
    )
    with pytest.raises(ValueError, match="overlap evaluation"):
        write_vlm_few_shot_approval(
            few_shot,
            evaluation,
            examples,
            tmp_path / "approval.json",
            reviewer="reviewer",
            reviewed_at="2026-08-21T12:00:00Z",
        )


def test_offline_adapter_accepts_only_real_local_image_bytes(tmp_path: Path) -> None:
    invalid = tmp_path / "fake.png"
    invalid.write_text("https://example.invalid/image.png", encoding="utf-8")
    with pytest.raises(ModelGenerationError, match="not PNG"):
        OfflineVLLMVisionAdapter._image_data_uri(invalid)


def test_offline_adapter_sends_only_local_data_uris_to_in_process_vllm(
    tmp_path: Path, monkeypatch
) -> None:
    calls: dict[str, list] = {"engine": [], "chat": [], "sampling": [], "schema": []}

    class FakeStructuredOutputsParams:
        def __init__(self, **kwargs):
            calls["schema"].append(kwargs)

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            calls["sampling"].append(kwargs)

    class FakeCompletion:
        text = json.dumps(_prediction("task-1"), separators=(",", ":"))
        token_ids = [1, 2]
        finish_reason = "stop"

    class FakeOutput:
        prompt_token_ids = [1, 2, 3]
        outputs = [FakeCompletion()]

    class FakeLLM:
        def __init__(self, **kwargs):
            calls["engine"].append(kwargs)

        def chat(self, messages, **kwargs):
            calls["chat"].append((messages, kwargs))
            return [FakeOutput()]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    fake_sampling = types.ModuleType("vllm.sampling_params")
    fake_sampling.StructuredOutputsParams = FakeStructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_sampling)
    monkeypatch.setattr(
        "auditops.model_adapter.importlib_metadata.version", lambda _: "0.26.0"
    )

    model = tmp_path / "model"
    model.mkdir()
    template = "fixture {{ messages }}"
    (model / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": template}), encoding="utf-8"
    )
    config = build_model_config(
        model_id="fixture/vlm",
        revision="a" * 40,
        model_path=str(model),
        quantization=None,
        backend="vllm_offline",
        chat_template_sha256=hashlib.sha256(template.encode()).hexdigest(),
    )
    image = tmp_path / "page.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfixture-image")
    adapter = OfflineVLLMVisionAdapter(config)
    response = adapter.generate_vision_json(
        VisionModelRequest(
            request_id="task-1",
            prompt="Extract Current assets.",
            images=(PageImage(1, image),),
            json_schema=VLM_PREDICTION_JSON_SCHEMA,
            max_tokens=128,
        )
    )

    assert response.payload["task_id"] == "task-1"
    assert calls["engine"][0]["limit_mm_per_prompt"] == {"image": 96}
    user_content = calls["chat"][0][0][-1]["content"]
    image_url = next(item for item in user_content if item["type"] == "image_url")
    assert image_url["image_url"]["url"].startswith("data:image/png;base64,")
    assert "http://" not in image_url["image_url"]["url"]
    assert "https://" not in image_url["image_url"]["url"]
    assert str(image) not in json.dumps(calls["chat"][0][0])
    assert calls["schema"] == [{"json": VLM_PREDICTION_JSON_SCHEMA}]
