from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from auditops.agent_benchmark import (
    BENCHMARK_PROFILE_SYNTHETIC,
    BENCHMARK_PROFILE_US_OFFLINE_SAMPLE_V0_1,
    BENCHMARK_PROFILE_US_V0_1,
    BENCHMARK_VERSION,
    DEFAULT_NARRATIVE_COUNT,
    DEFAULT_QUANT_COUNT,
    _build_case,
    _build_evidence_items,
    _build_verifier_observation,
    _select_few_shots_and_evaluation,
    build_agent_benchmark,
    verify_benchmark_artifacts,
)


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _quant_rows(count: int):
    rows = []
    for index in range(count):
        status = "OK" if index % 2 == 0 else "REFUSAL"
        task_id = f"q-{index:04d}"
        filing_id = f"q-filing-{index:04d}"
        asset_id = f"fact::{index:04d}:assets"
        liability_id = f"fact::{index:04d}:liabilities"
        canonical_inputs = [
            {
                "input_name": "assets_current",
                "concept_norm": "us-gaap_AssetsCurrent",
                "period_key": "ASOF_20251231",
                "unit_canon": "USD",
                "value_num_exact": str(2 * (index + 1)),
                "fact_evidence_ids": [asset_id],
                "context_id": f"ctx-{index:04d}",
                "entity_identifier": f"Q{index:04d}",
                "dimensions": {},
            }
        ]
        if status == "OK":
            canonical_inputs.append(
                {
                    "input_name": "liabilities_current",
                    "concept_norm": "us-gaap_LiabilitiesCurrent",
                    "period_key": "ASOF_20251231",
                    "unit_canon": "USD",
                    "value_num_exact": str(index + 1),
                    "fact_evidence_ids": [liability_id],
                    "context_id": f"ctx-{index:04d}",
                    "entity_identifier": f"Q{index:04d}",
                    "dimensions": {},
                }
            )
        rows.append(
            {
                "task_spec_version": "v1",
                "task_id": task_id,
                "task_type": "quant_metric",
                "template_id": f"quant:{status.casefold()}:{'heldout' if index < 4 else 'eval'}:fixture-v1",
                "task_family": f"quant_metric:current_ratio:{status.casefold()}",
                "source_answer_id": f"source-{task_id}",
                "metric_spec_id": "current_ratio",
                "metric_kind": "ratio",
                "filing_id": filing_id,
                "ticker": f"Q{index:04d}",
                "filing_metadata": {"form_type": "10-K", "report_date": "2025-12-31"},
                "period": {
                    "period_key": "ASOF_20251231",
                    "end_date": "2025-12-31",
                },
                "question": f"Calculate metric {index}",
                "target_status": status,
                "target_answer": {
                    "task_id": task_id,
                    "metric_spec_id": "current_ratio",
                    "filing_id": filing_id,
                    "status": status,
                    "value": "2" if status == "OK" else None,
                    "unit": "pure" if status == "OK" else None,
                    "period_key": "ASOF_20251231",
                    "evidence_ids": [asset_id, liability_id] if status == "OK" else [],
                    "refusal_code": None if status == "OK" else "MISSING_INPUT",
                },
                "canonical_inputs": canonical_inputs,
                "negative_type": None if status == "OK" else "missing_input",
                "refusal_policy": {
                    "allowed_codes": ["MISSING_INPUT"],
                    "default_code": "MISSING_INPUT",
                },
            }
        )
    return rows


def _narrative_rows(count: int):
    rows = []
    subtypes = (
        "footnote_note",
        "accounting_policy",
        "auditor_report_opinion_language",
        "critical_audit_matter",
    )
    for index in range(count):
        answerable = index % 2 == 0
        subtype = subtypes[(index // 2) % len(subtypes)]
        task_id = f"n-{index:04d}"
        filing_id = f"n-filing-{index:04d}"
        chunk_id = f"chunk::{index:04d}"
        rows.append(
            {
                "narrative_task_spec_version": "v1",
                "task_id": task_id,
                "task_type": "narrative_citation",
                "template_id": f"narrative:{subtype}:{'answerable' if answerable else 'unanswerable'}:{'heldout' if index < 16 else 'eval'}:fixture-v1",
                "task_family": f"narrative_citation:{subtype}:{'answerable' if answerable else 'unanswerable'}",
                "filing_id": filing_id,
                "ticker": f"N{index:04d}",
                "form_type": "10-K",
                "period_key": "FY2025",
                "question": f"What does note {index} say?",
                "retrieval_query": f"note {index}",
                "label": subtype,
                "answerability": "ANSWERABLE" if answerable else "UNANSWERABLE",
                "expected_chunk_ids": [chunk_id] if answerable else [],
                "extractive_answer": f"Supported answer {index}."
                if answerable
                else None,
                "refusal_code": None if answerable else "NARRATIVE_NOT_SUPPORTED",
                "negative_type": None if answerable else "unsupported_attribute",
                "citation_policy": {"top_k": 5, "candidate_k": 15},
                "evidence_items": [
                    {
                        "evidence_id": chunk_id,
                        "filing_id": filing_id,
                        "content": (
                            f"Supported answer {index}."
                            if answerable
                            else f"Irrelevant filing text {index}."
                        ),
                        "period_key": "FY2025",
                        "source_system": "SEC-EDGAR",
                    }
                ],
            }
        )
    return rows


def _build_sources(
    tmp_path: Path,
    quant_count: int,
    narrative_count: int,
    *,
    materialization_manifest: bool = False,
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    quant_path = source_dir / "quant.jsonl"
    narrative_path = source_dir / "narrative.jsonl"
    _write_jsonl(quant_path, _quant_rows(quant_count))
    _write_jsonl(narrative_path, _narrative_rows(narrative_count))
    if materialization_manifest:

        def artifact(path: Path, records: int) -> dict:
            return {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "records": records,
            }

        (source_dir / "materialization_manifest.json").write_text(
            json.dumps(
                {
                    "materialization_version": "agent_evidence.v1",
                    "artifacts": {
                        "enriched_quant.jsonl": artifact(quant_path, quant_count),
                        "enriched_narrative.jsonl": artifact(
                            narrative_path, narrative_count
                        ),
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return quant_path, narrative_path


def _build(quant_path: Path, narrative_path: Path, output_dir: Path, **overrides):
    benchmark_profile = overrides.pop("benchmark_profile", BENCHMARK_PROFILE_SYNTHETIC)
    return build_agent_benchmark(
        quant_path,
        narrative_path,
        output_dir,
        benchmark_profile=benchmark_profile,
        jurisdiction="US",
        corpus_id="fixture-corpus",
        reporting_framework="US-GAAP",
        standards_version="PCAOB-current",
        source_system="SEC-EDGAR",
        **overrides,
    )


def _write_minimal_v2_parent(
    path: Path,
    quant_rows: list[dict],
    narrative_rows: list[dict],
    *,
    quant_count: int,
    narrative_count: int,
) -> tuple[set[str], set[str], str]:
    examples, selected_quant, selected_narrative, _, _ = (
        _select_few_shots_and_evaluation(
            quant_rows,
            narrative_rows,
            quant_count,
            narrative_count,
            20260821,
        )
    )
    path.mkdir()
    cases = [
        {"task_id": row["task_id"], "task_type": "quant_metric"}
        for row in selected_quant
    ] + [
        {"task_id": row["task_id"], "task_type": "narrative_citation"}
        for row in selected_narrative
    ]
    few_shot = [
        {
            "task_id": row["task_id"],
            "task": {
                "task_id": row["task_id"],
                "task_type": (
                    "quant_metric" if task_type == "quant" else "narrative_citation"
                ),
            },
        }
        for task_type, row in examples
    ]
    _write_jsonl(path / "cases.jsonl", cases)
    _write_jsonl(path / "few_shot.jsonl", few_shot)

    def metadata(name: str) -> dict:
        artifact = path / name
        return {
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "bytes": artifact.stat().st_size,
        }

    material = {
        "benchmark_version": "agent_benchmark.v2",
        "artifacts": {
            "cases.jsonl": metadata("cases.jsonl"),
            "few_shot.jsonl": metadata("few_shot.jsonl"),
        },
    }
    benchmark_id = hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = {**material, "benchmark_id": benchmark_id}
    (path / "benchmark_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return (
        {row["task_id"] for row in cases},
        {row["task_id"] for row in few_shot},
        benchmark_id,
    )


def _historical_roe_row() -> dict:
    row = _quant_rows(1)[0]
    row.update(
        {
            "task_id": "historical-roe",
            "metric_spec_id": "roe",
            "period": {
                "period_key": "FY2024",
                "fiscal_year": 2024,
            },
            "required_inputs": [
                {"name": "net_income", "period": "current"},
                {
                    "name": "ending_equity",
                    "period": "current_period_end_asof",
                },
                {
                    "name": "beginning_equity",
                    "period": "prior_year_period_end_asof",
                },
            ],
            "canonical_inputs": [
                {"input_name": "net_income", "period_key": "FY2024"},
                {
                    "input_name": "ending_equity",
                    "period_key": "ASOF_20241231",
                },
                {
                    "input_name": "beginning_equity",
                    "period_key": "ASOF_20231231",
                },
            ],
        }
    )
    return row


def _period_evidence(evidence_id: str, input_name: str, period_key: str) -> dict:
    return {
        "evidence_id": evidence_id,
        "filing_id": "q-filing-0000",
        "period_key": period_key,
        "metadata": {"input_name": input_name},
    }


def test_quant_case_uses_frozen_current_asof_evidence_for_historical_period_end():
    row = _historical_roe_row()
    evidence = [
        _period_evidence("ending", "ending_equity", "ASOF_20241231"),
        _period_evidence("beginning", "beginning_equity", "ASOF_20231231"),
    ]

    case = _build_case(
        row,
        "quant_metric",
        jurisdiction="US",
        corpus_id="fixture-corpus",
        source_system="SEC-EDGAR-CACHED",
        reporting_framework="US-GAAP",
        standards_version="PCAOB-current",
        evidence_items=evidence,
    )

    assert case["period"] == {
        "period_key": "FY2024",
        "end_date": "2024-12-31",
        "selector_periods": {
            "current_period_end_asof": "ASOF_20241231",
            "prior_year_period_end_asof": "ASOF_20231231",
        },
    }
    assert case["period"]["end_date"] != row["filing_metadata"]["report_date"]


def test_ambiguous_evidence_cannot_change_frozen_source_period_binding():
    row = _historical_roe_row()
    evidence = [
        _period_evidence("ending-a", "ending_equity", "ASOF_20241229"),
        _period_evidence("ending-b", "ending_equity", "ASOF_20241231"),
    ]

    case = _build_case(
        row,
        "quant_metric",
        jurisdiction="US",
        corpus_id="fixture-corpus",
        source_system="SEC-EDGAR-CACHED",
        reporting_framework="US-GAAP",
        standards_version="PCAOB-current",
        evidence_items=evidence,
    )

    assert case["period"] == {
        "period_key": "FY2024",
        "end_date": "2024-12-31",
        "selector_periods": {
            "current_period_end_asof": "ASOF_20241231",
            "prior_year_period_end_asof": "ASOF_20231231",
        },
    }


def test_quant_case_accepts_cross_calendar_fiscal_year_binding():
    row = _historical_roe_row()
    row["canonical_inputs"][1]["period_key"] = "ASOF_20250201"
    row["canonical_inputs"][2]["period_key"] = "ASOF_20240127"
    evidence = [
        _period_evidence("ending", "ending_equity", "ASOF_20250201"),
        _period_evidence("beginning", "beginning_equity", "ASOF_20240127"),
    ]

    case = _build_case(
        row,
        "quant_metric",
        jurisdiction="US",
        corpus_id="fixture-corpus",
        source_system="SEC-EDGAR-CACHED",
        reporting_framework="US-GAAP",
        standards_version="PCAOB-current",
        evidence_items=evidence,
    )

    assert case["period"]["end_date"] == "2025-02-01"
    assert case["period"]["selector_periods"] == {
        "current_period_end_asof": "ASOF_20250201",
        "prior_year_period_end_asof": "ASOF_20240127",
    }


def test_malformed_evidence_input_name_cannot_change_frozen_source_binding():
    row = _historical_roe_row()
    evidence = [
        {
            "evidence_id": "malformed",
            "filing_id": "q-filing-0000",
            "period_key": "ASOF_20241231",
            "metadata": {"input_name": []},
        }
    ]

    case = _build_case(
        row,
        "quant_metric",
        jurisdiction="US",
        corpus_id="fixture-corpus",
        source_system="SEC-EDGAR-CACHED",
        reporting_framework="US-GAAP",
        standards_version="PCAOB-current",
        evidence_items=evidence,
    )

    assert case["period"]["selector_periods"] == {
        "current_period_end_asof": "ASOF_20241231",
        "prior_year_period_end_asof": "ASOF_20231231",
    }


def test_builds_default_500_case_manifest_without_target_leakage(tmp_path):
    quant_path, narrative_path = _build_sources(
        tmp_path,
        320,
        220,
        materialization_manifest=True,
    )

    result = _build(
        quant_path,
        narrative_path,
        tmp_path / "benchmark",
        benchmark_profile=BENCHMARK_PROFILE_US_V0_1,
    )

    assert result["counts"]["quant"] == DEFAULT_QUANT_COUNT == 300
    assert result["counts"]["narrative"] == DEFAULT_NARRATIVE_COUNT == 200
    assert result["counts"]["total"] == 500
    verification = verify_benchmark_artifacts(tmp_path / "benchmark")
    assert verification == {
        "valid": True,
        "benchmark_id": result["benchmark_id"],
        "errors": [],
        "case_count": 500,
        "evidence_record_count": 500,
        "gold_count": 500,
        "verifier_observation_count": 500,
        "few_shot_count": 20,
    }

    cases = [
        json.loads(line)
        for line in (tmp_path / "benchmark" / "cases.jsonl").read_text().splitlines()
    ]
    gold = [
        json.loads(line)
        for line in (tmp_path / "benchmark" / "gold.jsonl").read_text().splitlines()
    ]
    evidence = [
        json.loads(line)
        for line in (tmp_path / "benchmark" / "evidence.jsonl").read_text().splitlines()
    ]
    observations = [
        json.loads(line)
        for line in (tmp_path / "benchmark" / "verifier_observations.jsonl")
        .read_text()
        .splitlines()
    ]
    examples = [
        json.loads(line)
        for line in (tmp_path / "benchmark" / "few_shot.jsonl").read_text().splitlines()
    ]
    assert [row["task_id"] for row in cases] == [row["task_id"] for row in gold]
    assert [row["task_id"] for row in cases] == [row["task_id"] for row in evidence]
    assert [row["task_id"] for row in cases] == [row["task_id"] for row in observations]
    narrative_observations = [
        observation["observation"]
        for case, observation in zip(cases, observations)
        if case["task_type"] == "narrative_citation"
    ]
    assert all(
        row["retrieval_status"] == "SCOPE_LOADED"
        and row["answerability_assessed"] is False
        for row in narrative_observations
    )
    assert all("answer_text" not in row for row in narrative_observations)
    assert all("refusal_code" not in row for row in narrative_observations)
    for case, evidence_record in zip(cases, evidence):
        assert case["evidence_scope"]["evidence_ids"] == [
            item["evidence_id"] for item in evidence_record["items"]
        ]
    quant_evidence = next(
        record
        for case, record in zip(cases, evidence)
        if case["task_type"] == "quant_metric"
    )
    assert quant_evidence["items"][0]["value"] is not None
    assert "concept=us-gaap_AssetsCurrent" in quant_evidence["items"][0]["content"]
    assert all("target_answer" not in json.dumps(case) for case in cases)
    assert all("expected_chunk_ids" not in json.dumps(case) for case in cases)
    assert all(case["jurisdiction"] == "US" for case in cases)
    assert {example["assistant_response"]["status"] for example in examples} == {
        "OK",
        "REFUSAL",
    }
    assert {example["review_status"] for example in examples} == {
        "PENDING_HUMAN_REVIEW"
    }
    assert {example["task"]["entity"]["entity_id"] for example in examples}.isdisjoint(
        {case["entity"]["entity_id"] for case in cases}
    )
    manifest = json.loads(
        (tmp_path / "benchmark" / "benchmark_manifest.json").read_text()
    )
    assert manifest["benchmark_profile"] == BENCHMARK_PROFILE_US_V0_1
    source_manifest = json.loads(
        (tmp_path / "benchmark" / "source_manifest.json").read_text()
    )
    materialization_source = source_manifest["sources"]["materialization_manifest"]
    assert materialization_source["file_name"] == "materialization_manifest.json"
    assert (
        materialization_source["sha256"]
        == hashlib.sha256(
            (tmp_path / "source" / "materialization_manifest.json").read_bytes()
        ).hexdigest()
    )

    copied_materialization = tmp_path / "benchmark" / "materialization_manifest.json"
    copied_materialization.write_text(
        copied_materialization.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    tampered = verify_benchmark_artifacts(tmp_path / "benchmark")
    assert tampered["valid"] is False
    assert "sha256 mismatch: materialization_manifest.json" in tampered["errors"]
    assert (
        "source manifest sha256 mismatch: materialization_manifest.json"
        in tampered["errors"]
    )


def test_us_v01_profile_requires_materialization_manifest(tmp_path):
    quant_path, narrative_path = _build_sources(tmp_path, 320, 220)

    with pytest.raises(
        ValueError,
        match="requires materialization_manifest.json beside both enriched inputs",
    ):
        _build(
            quant_path,
            narrative_path,
            tmp_path / "benchmark",
            benchmark_profile=BENCHMARK_PROFILE_US_V0_1,
        )


def test_us_offline_sample_v01_builds_and_verifies_exact_real_data_shape(tmp_path):
    quant_path, narrative_path = _build_sources(
        tmp_path,
        320,
        220,
        materialization_manifest=True,
    )
    output = tmp_path / "benchmark"

    result = build_agent_benchmark(
        quant_path,
        narrative_path,
        output,
        benchmark_profile=BENCHMARK_PROFILE_US_OFFLINE_SAMPLE_V0_1,
        jurisdiction="US",
        corpus_id="us-sec-existing20-realdata-v1",
        reporting_framework="US-GAAP",
        standards_version="PCAOB-current",
        source_system="SEC-EDGAR-CACHED",
    )

    asserted_counts = {
        key: result["counts"][key]
        for key in (
            "quant",
            "narrative",
            "narrative_answerable",
            "narrative_unanswerable",
            "total",
        )
    }
    assert asserted_counts == {
        "quant": 300,
        "narrative": 200,
        "narrative_answerable": 100,
        "narrative_unanswerable": 100,
        "total": 500,
    }
    manifest = json.loads((output / "benchmark_manifest.json").read_text())
    assert manifest["benchmark_profile"] == BENCHMARK_PROFILE_US_OFFLINE_SAMPLE_V0_1
    assert manifest["jurisdiction"] == "US"
    assert manifest["source_system"] == "SEC-EDGAR-CACHED"
    assert manifest["seed"] == 20260821
    assert (
        manifest["stratification"]["quant"]["requirements"]["axes"]["issuer"][
            "profile_cap"
        ]
        == 50
    )
    assert (
        manifest["stratification"]["narrative"]["requirements"]["axes"]["issuer"][
            "profile_cap"
        ]
        == 50
    )
    verification = verify_benchmark_artifacts(output)
    assert verification["valid"] is True
    assert verification["errors"] == []
    assert verification["case_count"] == 500


def test_us_offline_sample_v01_requires_cached_sec_source_system(tmp_path):
    quant_path, narrative_path = _build_sources(tmp_path, 320, 220)

    with pytest.raises(
        ValueError,
        match=("us_offline_sample_v0.1 requires source_system='SEC-EDGAR-CACHED'"),
    ):
        build_agent_benchmark(
            quant_path,
            narrative_path,
            tmp_path / "benchmark",
            benchmark_profile=BENCHMARK_PROFILE_US_OFFLINE_SAMPLE_V0_1,
            jurisdiction="US",
            corpus_id="us-sec-existing20-realdata-v1",
            reporting_framework="US-GAAP",
            standards_version="PCAOB-current",
            source_system="SEC-EDGAR",
        )


def test_us_offline_sample_v01_requires_materialization_manifest(tmp_path):
    quant_path, narrative_path = _build_sources(tmp_path, 320, 220)

    with pytest.raises(
        ValueError,
        match="requires materialization_manifest.json beside both enriched inputs",
    ):
        build_agent_benchmark(
            quant_path,
            narrative_path,
            tmp_path / "benchmark",
            benchmark_profile=BENCHMARK_PROFILE_US_OFFLINE_SAMPLE_V0_1,
            jurisdiction="US",
            corpus_id="us-sec-existing20-realdata-v1",
            reporting_framework="US-GAAP",
            standards_version="PCAOB-current",
            source_system="SEC-EDGAR-CACHED",
        )


def test_us_v01_profile_rejects_arbitrary_materialization_manifest(tmp_path):
    quant_path, narrative_path = _build_sources(tmp_path, 320, 220)
    (tmp_path / "source" / "materialization_manifest.json").write_text(
        json.dumps({"fixture": True}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="materialization_version='agent_evidence.v1'"):
        _build(
            quant_path,
            narrative_path,
            tmp_path / "benchmark",
            benchmark_profile=BENCHMARK_PROFILE_US_V0_1,
        )


def test_us_v01_profile_rejects_materialization_input_hash_mismatch(tmp_path):
    quant_path, narrative_path = _build_sources(
        tmp_path,
        320,
        220,
        materialization_manifest=True,
    )
    manifest_path = tmp_path / "source" / "materialization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["enriched_quant.jsonl"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="enriched_quant.jsonl binding mismatch: sha256",
    ):
        _build(
            quant_path,
            narrative_path,
            tmp_path / "benchmark",
            benchmark_profile=BENCHMARK_PROFILE_US_V0_1,
        )


def test_narrative_verifier_observation_is_independent_of_gold_fields():
    row = _narrative_rows(1)[0]
    evidence_items = _build_evidence_items(row, "narrative_citation", "SEC-EDGAR")
    case = _build_case(
        row,
        "narrative_citation",
        jurisdiction="US",
        corpus_id="fixture-corpus",
        source_system="SEC-EDGAR",
        reporting_framework="US-GAAP",
        standards_version="PCAOB-current",
        evidence_items=evidence_items,
    )
    original = _build_verifier_observation(
        row,
        "narrative_citation",
        evidence_items,
        task_input=case,
    )
    changed_gold = copy.deepcopy(row)
    changed_gold.update(
        {
            "answerability": "UNANSWERABLE",
            "expected_chunk_ids": [],
            "extractive_answer": None,
            "refusal_code": "NARRATIVE_NOT_SUPPORTED",
        }
    )

    changed = _build_verifier_observation(
        changed_gold,
        "narrative_citation",
        evidence_items,
        task_input=case,
    )

    assert changed == original
    assert original["observation"]["retrieval_status"] == "SCOPE_LOADED"
    assert original["observation"]["answerability_assessed"] is False


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [
        ({"jurisdiction": "UK"}, "requires jurisdiction='US'"),
        ({"source_system": "Companies-House"}, "requires source_system='SEC-EDGAR'"),
        ({"seed": DEFAULT_QUANT_COUNT}, "requires seed=20260821"),
        (
            {"quant_count": 6, "narrative_count": 4},
            "requires quant_count=300",
        ),
    ],
)
def test_us_v01_profile_rejects_nonproduction_configuration(
    tmp_path, overrides, expected_error
):
    quant_path, narrative_path = _build_sources(tmp_path, 320, 220)
    arguments = {
        "benchmark_profile": BENCHMARK_PROFILE_US_V0_1,
        "jurisdiction": "US",
        "corpus_id": "fixture-corpus",
        "reporting_framework": "US-GAAP",
        "standards_version": "PCAOB-current",
        "source_system": "SEC-EDGAR",
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=expected_error):
        build_agent_benchmark(
            quant_path,
            narrative_path,
            tmp_path / "invalid-profile",
            **arguments,
        )

    assert not (tmp_path / "invalid-profile").exists()


@pytest.mark.parametrize("authoritative_key", ["cik", "source_entity_id"])
def test_few_shot_split_excludes_ticker_aliases_for_one_authoritative_issuer(
    authoritative_key,
):
    quant_rows = _quant_rows(30)
    narrative_rows = _narrative_rows(30)
    quant_rows[0][authoritative_key] = "shared-quant-issuer"
    quant_rows[4][authoritative_key] = "shared-quant-issuer"
    quant_rows[0]["ticker"] = "OLD"
    quant_rows[4]["ticker"] = "NEW"
    narrative_rows[0][authoritative_key] = "shared-narrative-issuer"
    narrative_rows[16][authoritative_key] = "shared-narrative-issuer"
    narrative_rows[0]["ticker"] = "OLDN"
    narrative_rows[16]["ticker"] = "NEWN"

    _, _, _, eligible_quant, eligible_narrative = _select_few_shots_and_evaluation(
        quant_rows,
        narrative_rows,
        quant_count=6,
        narrative_count=4,
        seed=20260821,
    )

    assert "q-0004" not in {row["task_id"] for row in eligible_quant}
    assert "n-0016" not in {row["task_id"] for row in eligible_narrative}


def test_stratification_accepts_repeated_issuers_across_status_strata(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    quant_rows = _quant_rows(20)
    narrative_rows = _narrative_rows(40)
    for index, row in enumerate(quant_rows):
        source_entity_id = (
            f"quant-dev-{index}" if index < 4 else f"quant-eval-{(index - 4) % 3}"
        )
        row["source_entity_id"] = source_entity_id
        for item in row["canonical_inputs"]:
            item["entity_identifier"] = source_entity_id
    for index, row in enumerate(narrative_rows):
        row["source_entity_id"] = (
            f"narrative-dev-{index}"
            if index < 16
            else f"narrative-eval-{(index - 16) % 2}"
        )
    quant_path = source_dir / "quant.jsonl"
    narrative_path = source_dir / "narrative.jsonl"
    _write_jsonl(quant_path, quant_rows)
    _write_jsonl(narrative_path, narrative_rows)
    output = tmp_path / "benchmark"

    result = _build(
        quant_path,
        narrative_path,
        output,
        quant_count=6,
        narrative_count=8,
    )

    assert result["verified"] is True
    cases = [
        json.loads(line) for line in (output / "cases.jsonl").read_text().splitlines()
    ]
    quant_entities = {
        case["entity"]["source_entity_id"]
        for case in cases
        if case["task_type"] == "quant_metric"
    }
    narrative_entities = {
        case["entity"]["source_entity_id"]
        for case in cases
        if case["task_type"] == "narrative_citation"
    }
    assert len(quant_entities) == 3
    assert len(narrative_entities) == 2


def test_verifier_rejects_authoritative_entity_overlap_despite_ticker_alias(
    tmp_path,
):
    quant_path, narrative_path = _build_sources(tmp_path, 20, 20)
    output = tmp_path / "benchmark"
    _build(quant_path, narrative_path, output, quant_count=6, narrative_count=4)
    examples = [
        json.loads(line)
        for line in (output / "few_shot.jsonl").read_text().splitlines()
    ]
    shared_source_id = examples[0]["task"]["entity"]["source_entity_id"]
    cases_path = output / "cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    cases[0]["entity"]["source_entity_id"] = shared_source_id
    cases_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in cases
        ),
        encoding="utf-8",
    )
    manifest_path = output / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["cases.jsonl"].update(
        {
            "sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            "bytes": cases_path.stat().st_size,
        }
    )
    material = {key: value for key, value in manifest.items() if key != "benchmark_id"}
    manifest["benchmark_id"] = hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    verification = verify_benchmark_artifacts(output)

    assert verification["valid"] is False
    assert "few-shot and evaluation entities overlap" in verification["errors"]


def test_verifier_rejects_us_profile_claim_on_small_fixture(tmp_path):
    quant_path, narrative_path = _build_sources(tmp_path, 20, 20)
    output = tmp_path / "benchmark"
    _build(quant_path, narrative_path, output, quant_count=6, narrative_count=4)
    manifest_path = output / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["benchmark_profile"] = BENCHMARK_PROFILE_US_V0_1
    material = {key: value for key, value in manifest.items() if key != "benchmark_id"}
    manifest["benchmark_id"] = hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    verification = verify_benchmark_artifacts(output)

    assert verification["valid"] is False
    assert any("requires quant_count=300" in error for error in verification["errors"])


def test_secret_scan_prevents_benchmark_directory_publication(tmp_path, monkeypatch):
    synthetic_secret = "synthetic-benchmark-secret-123456"
    monkeypatch.setenv("HF_TOKEN", synthetic_secret)
    quant_path, narrative_path = _build_sources(tmp_path, 24, 24)
    output = tmp_path / "secret-benchmark"

    with pytest.raises(ValueError, match="Secret scan failed"):
        build_agent_benchmark(
            quant_path,
            narrative_path,
            output,
            benchmark_profile=BENCHMARK_PROFILE_SYNTHETIC,
            jurisdiction="US",
            corpus_id=synthetic_secret,
            reporting_framework="US-GAAP",
            standards_version="PCAOB-current",
            source_system="SEC-EDGAR",
            quant_count=6,
            narrative_count=4,
        )

    assert not output.exists()


def test_build_is_stable_across_destinations_and_refuses_overwrite(tmp_path):
    quant_path, narrative_path = _build_sources(tmp_path, 20, 20)
    first = _build(
        quant_path, narrative_path, tmp_path / "one", quant_count=6, narrative_count=4
    )
    second = _build(
        quant_path, narrative_path, tmp_path / "two", quant_count=6, narrative_count=4
    )

    assert first["benchmark_id"] == second["benchmark_id"]
    for name in (
        "cases.jsonl",
        "evidence.jsonl",
        "gold.jsonl",
        "verifier_observations.jsonl",
        "few_shot.jsonl",
        "source_manifest.json",
        "benchmark_manifest.json",
    ):
        assert (tmp_path / "one" / name).read_bytes() == (
            tmp_path / "two" / name
        ).read_bytes()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _build(
            quant_path,
            narrative_path,
            tmp_path / "one",
            quant_count=6,
            narrative_count=4,
        )


def test_derived_v22_build_preserves_verified_v2_evaluation_membership(tmp_path):
    quant_rows = _quant_rows(24)
    narrative_rows = _narrative_rows(24)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    quant_path = source_dir / "quant.jsonl"
    narrative_path = source_dir / "narrative.jsonl"
    _write_jsonl(quant_path, quant_rows)
    _write_jsonl(narrative_path, narrative_rows)
    parent_ids, _, parent_benchmark_id = _write_minimal_v2_parent(
        tmp_path / "parent-v2",
        quant_rows,
        narrative_rows,
        quant_count=6,
        narrative_count=4,
    )

    result = _build(
        quant_path,
        narrative_path,
        tmp_path / "derived-v22",
        quant_count=6,
        narrative_count=4,
        parent_benchmark_dir=tmp_path / "parent-v2",
    )

    assert result["verified"] is True
    lineage = [
        json.loads(line)
        for line in (tmp_path / "derived-v22" / "case_lineage.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["parent_task_id"] for row in lineage} == parent_ids
    parent_case_order = [
        json.loads(line)["task_id"]
        for line in (tmp_path / "parent-v2" / "cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["parent_task_id"] for row in lineage] == parent_case_order
    assert all(row["task_id"].startswith("v22-") for row in lineage)
    derived_few_shot = [
        json.loads(line)
        for line in (tmp_path / "derived-v22" / "few_shot.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(derived_few_shot) == 20
    assert not ({row["task_id"] for row in derived_few_shot} & parent_ids)
    assert all(len(row["source_task_sha256"]) == 64 for row in derived_few_shot)
    subtype_counts = {}
    for row in derived_few_shot:
        subtype = row["task"].get("narrative_subtype") or "quant_metric"
        subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
    assert subtype_counts == {
        "quant_metric": 4,
        "footnote_note": 4,
        "accounting_policy": 4,
        "auditor_report_opinion_language": 4,
        "critical_audit_matter": 4,
    }
    source_manifest = json.loads(
        (tmp_path / "derived-v22" / "source_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        source_manifest["sources"]["parent_benchmark"]["benchmark_id"]
        == parent_benchmark_id
    )


def test_derived_v21_build_rejects_tampered_parent_membership(tmp_path):
    quant_rows = _quant_rows(24)
    narrative_rows = _narrative_rows(24)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    quant_path = source_dir / "quant.jsonl"
    narrative_path = source_dir / "narrative.jsonl"
    _write_jsonl(quant_path, quant_rows)
    _write_jsonl(narrative_path, narrative_rows)
    parent_dir = tmp_path / "parent-v2"
    _write_minimal_v2_parent(
        parent_dir,
        quant_rows,
        narrative_rows,
        quant_count=6,
        narrative_count=4,
    )
    with (parent_dir / "cases.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="artifact (hash|byte) mismatch"):
        _build(
            quant_path,
            narrative_path,
            tmp_path / "derived-v21",
            quant_count=6,
            narrative_count=4,
            parent_benchmark_dir=parent_dir,
        )


def test_few_shot_narrative_excerpts_are_compacted_before_freeze(tmp_path):
    quant_rows = _quant_rows(24)
    narrative_rows = _narrative_rows(24)
    for row in narrative_rows:
        original = row["evidence_items"][0]["content"]
        row["evidence_items"][0]["content"] = original + (" filing context" * 500)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    quant_path = source_dir / "quant.jsonl"
    narrative_path = source_dir / "narrative.jsonl"
    _write_jsonl(quant_path, quant_rows)
    _write_jsonl(narrative_path, narrative_rows)

    _build(
        quant_path,
        narrative_path,
        tmp_path / "benchmark",
        quant_count=6,
        narrative_count=4,
    )

    examples = [
        json.loads(line)
        for line in (tmp_path / "benchmark" / "few_shot.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    narrative_examples = [
        row for row in examples if row["task"]["task_type"] == "narrative_citation"
    ]
    assert all(
        len(item["content"]) <= 512
        for row in narrative_examples
        for item in row["evidence_items"]
    )
    for row in narrative_examples:
        answer_text = row["assistant_response"].get("answer_text")
        if answer_text is not None:
            assert all(item["content"] == answer_text for item in row["evidence_items"])


def test_benchmark_verifier_accepts_declared_cross_filing_evidence(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    quant_rows = _quant_rows(20)
    for row in quant_rows:
        comparison_filing = f"comparison-{row['filing_id']}"
        for item in row["canonical_inputs"]:
            item["filing_id"] = comparison_filing
    quant_path = source_dir / "quant.jsonl"
    narrative_path = source_dir / "narrative.jsonl"
    _write_jsonl(quant_path, quant_rows)
    _write_jsonl(narrative_path, _narrative_rows(20))

    result = _build(
        quant_path,
        narrative_path,
        tmp_path / "benchmark",
        quant_count=6,
        narrative_count=4,
    )

    assert result["verified"] is True
    verification = verify_benchmark_artifacts(tmp_path / "benchmark")
    assert verification["valid"] is True
    cases = [
        json.loads(line)
        for line in (tmp_path / "benchmark" / "cases.jsonl").read_text().splitlines()
    ]
    quant_cases = [case for case in cases if case["task_type"] == "quant_metric"]
    assert all(len(case["evidence_scope"]["filing_ids"]) == 2 for case in quant_cases)


def test_verifier_recomputes_benchmark_id_after_manifest_tampering(tmp_path):
    quant_path, narrative_path = _build_sources(tmp_path, 20, 20)
    output = tmp_path / "benchmark"
    original = _build(
        quant_path, narrative_path, output, quant_count=6, narrative_count=4
    )
    gold_path = output / "gold.jsonl"
    rows = [json.loads(line) for line in gold_path.read_text().splitlines()]
    quant = next(row for row in rows if row["task_type"] == "quant_metric")
    quant["target"]["value"] = "999999"
    gold_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest_path = output / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["gold.jsonl"].update(
        {
            "sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
            "bytes": gold_path.stat().st_size,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    verification = verify_benchmark_artifacts(output)

    assert verification["valid"] is False
    assert verification["benchmark_id"] == original["benchmark_id"]
    assert any("benchmark_id" in error for error in verification["errors"])


def test_verifier_rejects_relabelled_legacy_benchmark_version(tmp_path):
    quant_path, narrative_path = _build_sources(tmp_path, 20, 20)
    output = tmp_path / "benchmark"
    _build(quant_path, narrative_path, output, quant_count=6, narrative_count=4)
    manifest_path = output / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark_version"] = "agent_benchmark.v1"
    material = {key: value for key, value in manifest.items() if key != "benchmark_id"}
    manifest["benchmark_id"] = hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    verification = verify_benchmark_artifacts(output)

    assert verification["valid"] is False
    assert (
        f"unsupported benchmark_version: expected {BENCHMARK_VERSION}"
        in verification["errors"]
    )


def test_verifier_reports_exact_profile_quota_shortage(tmp_path):
    quant_path, narrative_path = _build_sources(tmp_path, 20, 20)
    output = tmp_path / "benchmark"
    _build(quant_path, narrative_path, output, quant_count=6, narrative_count=4)
    manifest_path = output / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stratification"]["quant"]["eligible"]["axes"]["operation"] = {
        "current_ratio": 14,
        **{f"unselected-operation-{index}": 1 for index in range(6)},
    }
    material = {key: value for key, value in manifest.items() if key != "benchmark_id"}
    manifest["benchmark_id"] = hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    verification = verify_benchmark_artifacts(output)

    assert verification["valid"] is False
    assert (
        "quant operation distinct-strata shortage: requires 4, selected 1, eligible 7"
        in verification["errors"]
    )


def test_verification_detects_artifact_tampering(tmp_path):
    quant_path, narrative_path = _build_sources(tmp_path, 20, 20)
    _build(
        quant_path,
        narrative_path,
        tmp_path / "benchmark",
        quant_count=6,
        narrative_count=4,
    )
    cases_path = tmp_path / "benchmark" / "cases.jsonl"
    cases_path.write_text(
        cases_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    result = verify_benchmark_artifacts(tmp_path / "benchmark")

    assert result["valid"] is False
    assert "sha256 mismatch: cases.jsonl" in result["errors"]
    assert "byte count mismatch: cases.jsonl" in result["errors"]
