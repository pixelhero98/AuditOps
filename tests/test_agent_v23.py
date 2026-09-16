from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from auditops.agent_batch import run_agent_baseline
from auditops.agent_benchmark_v23 import (
    derive_agent_benchmark_v23,
    split_narrative_evidence_item_v23,
    verify_agent_benchmark_v23,
)
from auditops.agent_context import build_context_pack
from auditops.agent_contracts import (
    ContractValidationError,
    build_agent_task_input,
    build_failure,
    build_model_config,
    validate_agent_task_input,
)
from auditops.agent_evaluation_v23 import (
    _bind_evaluation_scope,
    write_evaluation_artifacts_v23,
)
from auditops.agent_operations import narrative_selection_contract, task_operation_spec
from auditops.agent_prompts import (
    stage_response_semantics_valid,
    terminal_response_schema,
)
from auditops.agent_runtime import run_agent_case
from auditops.agent_tools import execute_deterministic_tool, load_frozen_evidence
from auditops.canonical_json import canonical_json_bytes, canonical_json_sha256
from auditops.model_adapter import MockModelAdapter

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "auditops" / "specs"


def _narrative_task(subtype: str = "auditor_report_opinion_language") -> dict:
    spec = task_operation_spec("narrative_citation", version="v2.3")
    return build_agent_task_input(
        task_id=f"v23-{subtype}",
        task_type="narrative_citation",
        jurisdiction="US",
        reporting_framework="US-GAAP",
        standards_profile={"name": "PCAOB filing verification", "version": "2026"},
        source_system="SEC-EDGAR-CACHED",
        question="What opinion did the auditor express on the financial statements?",
        entity={"entity_id": "issuer-1", "name": "Issuer One"},
        filing={"filing_id": "filing-1", "form_type": "10-K"},
        period={"period_key": "FY2025"},
        retrieval_query="auditor opinion financial statements present fairly",
        narrative_subtype=subtype,
        allowed_tools=[spec.operation],
        evidence_ids=["chunk-1"],
        output_schema_id=spec.terminal_output_schema_id,
        refusal_codes=["NO_DIRECT_EVIDENCE"],
        contract_version="v2.3",
    )


def _evidence() -> list[dict]:
    return [
        {
            "evidence_id": "chunk-1",
            "filing_id": "filing-1",
            "content": "In our opinion, the financial statements present fairly, in all material respects.",
            "rank": 1,
            "period_key": "FY2025",
        }
    ]


def _schemas() -> tuple[dict[str, dict], Registry]:
    schemas = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in SPECS.glob("*.schema.json")
    }
    registry = Registry()
    for name, schema in schemas.items():
        resource = Resource.from_contents(schema)
        registry = registry.with_resource(name, resource)
        registry = registry.with_resource(schema["$id"], resource)
    return schemas, registry


def _clock():
    values = iter(
        [
            datetime(2026, 8, 25, 10, 0, tzinfo=UTC),
            datetime(2026, 8, 25, 10, 0, tzinfo=UTC) + timedelta(milliseconds=10),
        ]
    )
    return lambda: next(values)


def test_v23_task_binds_one_model_facing_evidence_refusal_and_policy():
    task = _narrative_task()
    assert task["refusal_policy"]["allowed_codes"] == [
        "NO_DIRECT_EVIDENCE",
        "PROMPT_INJECTION_DETECTED",
    ]
    assert task["narrative_selection_policy"] == narrative_selection_contract(
        "auditor_report_opinion_language"
    )
    tampered = copy.deepcopy(task)
    tampered["narrative_selection_policy"]["max_extracts"] = 3
    with pytest.raises(ContractValidationError, match="extraction policy"):
        validate_agent_task_input(tampered)


@pytest.mark.parametrize(
    ("subtype", "maximum"),
    [
        ("footnote_note", 1),
        ("accounting_policy", 1),
        ("auditor_report_opinion_language", 1),
        ("critical_audit_matter", 3),
    ],
)
def test_v23_dynamic_schema_enforces_subtype_extract_cardinality(subtype, maximum):
    task = _narrative_task(subtype)
    schema = terminal_response_schema(task, _evidence())
    extracts = schema["oneOf"][0]["properties"]["extracts"]
    assert extracts["minItems"] == 1
    assert extracts["maxItems"] == maximum
    assert schema["oneOf"][1]["properties"]["refusal_code"] == {
        "type": "string",
        "enum": ["NO_DIRECT_EVIDENCE"],
        "maxLength": 128,
    }
    Draft202012Validator.check_schema(schema)


def test_v23_semantic_guard_rejects_redundant_cam_extracts():
    task = _narrative_task("critical_audit_matter")
    assert not stage_response_semantics_valid(
        {
            "action": "ANSWER",
            "extracts": [
                {"evidence_id": "chunk-1", "exact_quote": "material respects"},
                {
                    "evidence_id": "chunk-1",
                    "exact_quote": "present fairly, in all material respects",
                },
            ],
        },
        task,
    )


def test_v23_section_split_separates_financial_statement_and_icfr_opinions():
    item = {
        "evidence_id": "parent-1",
        "filing_id": "filing-1",
        "content": (
            "Opinion on the Financial Statements\n"
            "In our opinion, the financial statements present fairly.\n\n"
            "Opinion on Internal Control Over Financial Reporting\n"
            "In our opinion, the company maintained effective internal control."
        ),
        "rank": 1,
        "metadata": {"char_start": 100},
    }
    segments = split_narrative_evidence_item_v23(item)
    assert len(segments) == 2
    assert "internal control" not in segments[0]["content"].casefold()
    assert "financial statements" not in segments[1]["content"].casefold()
    assert [segment["metadata"]["segment_index"] for segment in segments] == [1, 2]
    assert len({segment["evidence_id"] for segment in segments}) == 2


def test_v23_section_split_does_not_treat_inline_cam_phrase_as_heading():
    item = {
        "evidence_id": "parent-inline",
        "filing_id": "filing-1",
        "content": (
            "The matter required especially challenging auditor judgment.\n"
            "Critical Audit Matter considerations were integrated into our response.\n"
            "We evaluated the relevant controls and substantive evidence."
        ),
        "rank": 1,
    }
    segments = split_narrative_evidence_item_v23(item)
    assert len(segments) == 1
    assert segments[0]["content"] == item["content"]


def test_v23_runtime_releases_exact_quote_and_binds_policy_identity():
    task = _narrative_task()
    evidence = _evidence()
    payload = {
        "task_id": task["task_id"],
        "action": "ANSWER",
        "period_key": "FY2025",
        "extracts": [{"evidence_id": "chunk-1", "exact_quote": evidence[0]["content"]}],
    }
    adapter = MockModelAdapter((payload,))
    config = build_model_config(
        model_id=adapter.model_id,
        revision=adapter.model_revision,
        backend="mock",
    )
    result = run_agent_case(
        task,
        adapter=adapter,
        model_config=config,
        corpus_id="fixture-v23",
        benchmark_manifest_sha256="a" * 64,
        runtime_mode="safety_hybrid",
        prompt_condition="zero_shot",
        evidence_items=evidence,
        tool_executor=lambda name, arguments, frozen_task: execute_deterministic_tool(
            name, arguments, frozen_task, evidence
        ),
        clock=_clock(),
    )
    assert result.run_record["outcome"] == "RELEASED"
    assert result.run_record["runtime_version"] == "auditops.agent_runtime.v2.3"
    assert result.run_record["prompt_version"] == "auditops.agent_prompt.v2.3"
    assert result.run_record[
        "narrative_selection_policy_sha256"
    ] == canonical_json_sha256(task["narrative_selection_policy"])
    assert result.to_dict()["agent_case_result_version"] == "v2.3"
    schemas, registry = _schemas()
    for filename, value in (
        ("agent_run_record.v2.3.schema.json", result.run_record),
        ("agent_case_result.v2.3.schema.json", result.to_dict()),
    ):
        errors = list(
            Draft202012Validator(schemas[filename], registry=registry).iter_errors(
                value
            )
        )
        assert errors == []


def test_v23_failure_schema_validates_registered_sanitized_failure():
    schemas, registry = _schemas()
    failure = build_failure(
        code="MODEL_OUTPUT_SCHEMA_INVALID",
        stage="synthesis",
        sanitized_detail="The structured response did not match the bound schema.",
        contract_version="v2.3",
    )
    errors = list(
        Draft202012Validator(
            schemas["failure.v2.3.schema.json"], registry=registry
        ).iter_errors(failure)
    )
    assert errors == []


def test_v23_public_schema_snapshots_validate_runtime_objects():
    schemas, registry = _schemas()
    task = _narrative_task()
    context = build_context_pack(task, _evidence(), token_counter=lambda _: 1)
    observation = load_frozen_evidence(task, _evidence(), visibility="MODEL_VISIBLE")
    for filename, value in (
        ("agent_task_input.v2.3.schema.json", task),
        ("context_pack.v2.3.schema.json", context),
        ("tool_observation.v2.3.schema.json", observation),
    ):
        errors = list(
            Draft202012Validator(schemas[filename], registry=registry).iter_errors(
                value
            )
        )
        assert errors == []


def _write_parent(path: Path) -> None:
    path.mkdir()
    task = copy.deepcopy(_narrative_task())
    task["agent_task_input_version"] = "v2.2"
    task.pop("narrative_selection_policy")
    task["output_schema_id"] = "narrative_selection_or_refusal.v2.2"
    task["evidence_scope"]["selection_method"] = "FROZEN_BM25"
    task["refusal_policy"] = {
        "allowed_codes": ["NARRATIVE_NOT_SUPPORTED", "PROMPT_INJECTION_DETECTED"]
    }
    validate_agent_task_input(task)
    evidence = [
        {
            "evidence_id": "parent-chunk",
            "filing_id": "filing-1",
            "content": (
                "Opinion on the Financial Statements\n"
                "In our opinion, the financial statements present fairly.\n\n"
                "Opinion on Internal Control Over Financial Reporting\n"
                "In our opinion, the company maintained effective internal control."
            ),
            "rank": 1,
            "period_key": "FY2025",
        }
    ]
    task["evidence_scope"]["evidence_ids"] = ["parent-chunk"]
    files: dict[str, list[dict] | dict] = {
        "cases.jsonl": [task],
        "evidence.jsonl": [{"task_id": task["task_id"], "items": evidence}],
        "gold.jsonl": [
            {
                "gold_record_version": "v2.2",
                "schema_id": "agent_gold_record.v2",
                "task_id": task["task_id"],
                "task_type": "narrative_citation",
                "template_id": "opinion-template",
                "task_family": "narrative_citation:opinion",
                "strata": {"negative_type": "NONE"},
                "jurisdiction": "US",
                "corpus_id": "fixture",
                "source_task_sha256": "b" * 64,
                "target": {
                    "status": "OK",
                    "answer_text": "In our opinion, the financial statements present fairly.",
                    "chunk_evidence_ids": ["parent-chunk"],
                    "refusal_code": None,
                    "filing_id": "filing-1",
                    "period_key": "FY2025",
                },
            }
        ],
        "verifier_observations.jsonl": [
            {
                "task_id": task["task_id"],
                "tool_name": "load_frozen_evidence",
                "observation": load_frozen_evidence(task, evidence),
            }
        ],
        "few_shot.jsonl": [],
        "case_lineage.jsonl": [
            {"task_id": task["task_id"], "parent_task_id": "source-task"}
        ],
        "source_manifest.json": {"source_manifest_version": "v1", "sources": {}},
        "parity_50.json": {
            "parity_slice_version": "fixture",
            "requested_count": 1,
            "case_count": 1,
            "seed": 20260821,
            "task_ids": [task["task_id"]],
            "task_ids_sha256": canonical_json_sha256([task["task_id"]]),
            "breakdown": {},
        },
    }
    artifact_meta = {}
    for name, value in files.items():
        target = path / name
        if isinstance(value, list):
            target.write_bytes(
                b"".join(canonical_json_bytes(row) + b"\n" for row in value)
            )
            records = len(value)
        else:
            target.write_bytes(canonical_json_bytes(value) + b"\n")
            records = None
        import hashlib

        meta = {
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "bytes": target.stat().st_size,
        }
        if records is not None:
            meta["records"] = records
        artifact_meta[name] = meta
    material = {
        "benchmark_version": "agent_benchmark.v2.2",
        "benchmark_profile": "synthetic",
        "seed": 20260821,
        "corpus_id": "fixture",
        "jurisdiction": "US",
        "reporting_framework": "US-GAAP",
        "standards_version": "2026",
        "source_system": "SEC-EDGAR-CACHED",
        "counts": {
            "quant": 0,
            "narrative": 1,
            "narrative_answerable": 1,
            "narrative_unanswerable": 0,
            "total": 1,
            "evidence_records": 1,
            "evidence_items": 1,
            "verifier_observations": 1,
            "few_shot": 0,
            "parity_slice": 1,
        },
        "few_shot_policy": {"requires_human_review": True},
        "stratification": {},
        "artifact_visibility": {
            "inference_visible": [
                "cases.jsonl",
                "evidence.jsonl",
                "few_shot.jsonl",
                "parity_50.json",
            ],
            "evaluator_only": [
                "gold.jsonl",
                "verifier_observations.jsonl",
                "case_lineage.jsonl",
            ],
        },
        "artifacts": artifact_meta,
    }
    manifest = {**material, "benchmark_id": canonical_json_sha256(material)}
    (path / "benchmark_manifest.json").write_bytes(
        canonical_json_bytes(manifest) + b"\n"
    )


def test_v23_derivation_preserves_membership_and_rebinds_gold_after_split(tmp_path):
    parent = tmp_path / "parent"
    derived = tmp_path / "derived"
    _write_parent(parent)
    result = derive_agent_benchmark_v23(parent, derived)
    assert result["verified"] is True
    assert verify_agent_benchmark_v23(derived)["valid"] is True
    case = json.loads((derived / "cases.jsonl").read_text(encoding="utf-8"))
    evidence = json.loads((derived / "evidence.jsonl").read_text(encoding="utf-8"))
    gold = json.loads((derived / "gold.jsonl").read_text(encoding="utf-8"))
    assert case["task_id"].startswith("v23-")
    assert case["evidence_scope"]["selection_method"] == "FROZEN_BM25_V23_SECTION_SPLIT"
    assert len(evidence["items"]) == 2
    assert gold["target"]["chunk_evidence_ids"] == [evidence["items"][0]["evidence_id"]]
    assert all(
        item["metadata"]["parent_evidence_id"] == "parent-chunk"
        for item in evidence["items"]
    )


def test_v23_derivation_is_byte_reproducible_from_one_frozen_parent(tmp_path):
    parent = tmp_path / "parent"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_parent(parent)
    derive_agent_benchmark_v23(parent, first)
    derive_agent_benchmark_v23(parent, second)
    first_files = sorted(path.name for path in first.iterdir())
    assert first_files == sorted(path.name for path in second.iterdir())
    assert {name: (first / name).read_bytes() for name in first_files} == {
        name: (second / name).read_bytes() for name in first_files
    }


def test_v23_batch_executes_derived_zero_shot_case(tmp_path):
    parent = tmp_path / "parent"
    derived = tmp_path / "derived"
    output = tmp_path / "run"
    _write_parent(parent)
    derive_agent_benchmark_v23(parent, derived)
    case = json.loads((derived / "cases.jsonl").read_text(encoding="utf-8"))
    evidence = json.loads((derived / "evidence.jsonl").read_text(encoding="utf-8"))
    exact_quote = "In our opinion, the financial statements present fairly."
    supporting = next(
        item for item in evidence["items"] if exact_quote in item["content"]
    )
    adapter = MockModelAdapter(
        (
            {
                "task_id": case["task_id"],
                "action": "ANSWER",
                "period_key": "FY2025",
                "extracts": [
                    {
                        "evidence_id": supporting["evidence_id"],
                        "exact_quote": exact_quote,
                    }
                ],
            },
        )
    )
    config = build_model_config(
        model_id=adapter.model_id,
        revision=adapter.model_revision,
        backend="mock",
    )
    config_path = tmp_path / "model.json"
    config_path.write_bytes(canonical_json_bytes(config) + b"\n")
    manifest = run_agent_baseline(
        derived,
        config_path,
        output,
        runtime_mode="safety_hybrid",
        prompt_condition="zero_shot",
        limit=1,
        adapter=adapter,
        strict_provenance=False,
    )
    assert manifest["batch_run_version"] == "auditops-agent-batch.v2.3"
    result = json.loads((output / "results.jsonl").read_text(encoding="utf-8"))
    assert result["outcome"] == "RELEASED"
    assert result["run_record"]["agent_run_record_version"] == "v2.3"
    evaluation = tmp_path / "evaluation"
    summary = write_evaluation_artifacts_v23(derived, [output], evaluation)
    assert summary["all_hard_safety_pass"] is True
    report = (evaluation / "report.md").read_text(encoding="utf-8")
    assert "Narrative answerability" in report
    assert "Proposal and released-output accuracy" in report
    assert "| Run | Scope | Cases | Hard safety | Typed | Verifier |" in report
    metrics = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["runs"][0]["evaluation_scope"] == "full"


def test_v23_evaluator_accepts_only_exact_frozen_parity_membership(tmp_path):
    cases = [{"task_id": "task-a"}, {"task_id": "task-b"}]
    (tmp_path / "cases.jsonl").write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in cases)
    )
    (tmp_path / "evidence.jsonl").write_bytes(b"{}\n")
    benchmark_manifest = {"benchmark_id": "benchmark-v23", "counts": {"total": 2}}
    (tmp_path / "benchmark_manifest.json").write_bytes(
        canonical_json_bytes(benchmark_manifest) + b"\n"
    )
    parity_ids = ["task-b"]
    parity = {
        "case_count": 1,
        "task_ids": parity_ids,
        "task_ids_sha256": canonical_json_sha256(parity_ids),
    }
    (tmp_path / "parity_50.json").write_bytes(canonical_json_bytes(parity) + b"\n")

    def file_sha(name: str) -> str:
        return hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()

    run_manifest = {
        "benchmark_id": "benchmark-v23",
        "benchmark_manifest_sha256": file_sha("benchmark_manifest.json"),
        "case_count": 1,
        "full_benchmark_case_count": 2,
        "complete_full_benchmark": False,
        "task_ids_sha256": canonical_json_sha256(parity_ids),
        "inputs": {
            "cases_sha256": file_sha("cases.jsonl"),
            "evidence_sha256": file_sha("evidence.jsonl"),
            "parity_membership_sha256": file_sha("parity_50.json"),
        },
    }
    scope, scoped_cases = _bind_evaluation_scope(
        root=tmp_path,
        benchmark_manifest=benchmark_manifest,
        cases=cases,
        run_manifest=run_manifest,
        results=[{"task_id": "task-b"}],
    )
    assert scope == "frozen_parity"
    assert scoped_cases == [cases[1]]

    tampered = copy.deepcopy(run_manifest)
    tampered["task_ids_sha256"] = canonical_json_sha256(["task-a"])
    with pytest.raises(ValueError, match="frozen parity membership order"):
        _bind_evaluation_scope(
            root=tmp_path,
            benchmark_manifest=benchmark_manifest,
            cases=cases,
            run_manifest=tampered,
            results=[{"task_id": "task-a"}],
        )
