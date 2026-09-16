from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from auditops.companies_house import build_companies_house_vlm_diagnostic
from auditops.vlm_evaluation import evaluate_companies_house_vlm_diagnostic


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def diagnostic(tmp_path: Path) -> tuple[Path, str]:
    pairs = tmp_path / "pairs.csv"
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
        writer.writerow(
            {
                "pair_id": "P001",
                "company_number": "01234567",
                "transaction_id": "tx1",
                "document_id": "doc1",
                "pdf_path": "/data/P001.pdf",
                "pdf_page_count": "3",
            }
        )

    facts = tmp_path / "facts.csv"
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
        writer.writerow(
            {
                "pair_id": "P001",
                "fact_label": "Current assets",
                "ixbrl_tag_concept": "uk-core:CurrentAssets",
                "ixbrl_value": "1200",
                "unit": "GBP",
                "period_context": "Current year",
                "pdf_page": "page 2",
                "pdf_evidence": "Current assets 1,200",
                "usable_as_benchmark": "yes",
            }
        )

    output = tmp_path / "diagnostic"
    build_companies_house_vlm_diagnostic(pairs, facts, output)
    task_id = json.loads((output / "cases.jsonl").read_text(encoding="utf-8"))[
        "task_id"
    ]
    return output, task_id


def _perfect_prediction(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "facts": [
            {
                "fact_label": "Current assets",
                "value": "1,200.00",
                "unit": "gbp",
                "period_context": "current   YEAR",
                "page": 2,
                "bbox": [0.1, 0.2, 0.8, 0.3],
                "evidence_text": "current assets  1,200",
            }
        ],
    }


def test_perfect_decimal_aware_evaluation_and_immutable_artifacts(
    tmp_path: Path, diagnostic: tuple[Path, str]
) -> None:
    diagnostic_dir, task_id = diagnostic
    predictions = tmp_path / "predictions.jsonl"
    _jsonl(predictions, [_perfect_prediction(task_id)])
    output = tmp_path / "evaluation"

    result = evaluate_companies_house_vlm_diagnostic(
        diagnostic_dir, predictions, output
    )

    summary = result["metrics"]["summary"]
    assert summary["joint_fact_accuracy"] == 1.0
    assert summary["value_accuracy"] == 1.0
    assert summary["matched_fact_count"] == 1
    bbox = result["metrics"]["bbox_evaluation"]
    assert bbox["bbox_schema_valid_rate"] == 1.0
    assert bbox["bbox_bounds_valid_rate"] == 1.0
    assert bbox["gold_bbox_available"] is False
    assert bbox["localization_accuracy_available"] is False
    assert bbox["bbox_iou_evaluated_count"] == 0
    assert bbox["mean_iou_for_matched_predictions"] is None
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Localization accuracy and IoU are therefore not evaluated" in report

    manifest = _read_json(output / "evaluation_manifest.json")
    assert manifest["evaluation_id"] == result["evaluation_id"]
    for name in ("metrics.json", "failures.jsonl", "report.md"):
        assert manifest["artifacts"][name]["sha256"] == _sha256(output / name)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluate_companies_house_vlm_diagnostic(diagnostic_dir, predictions, output)


def test_mismatches_missing_and_unexpected_facts_are_scored(
    tmp_path: Path, diagnostic: tuple[Path, str]
) -> None:
    diagnostic_dir, task_id = diagnostic
    predictions = tmp_path / "predictions.jsonl"
    row = _perfect_prediction(task_id)
    row["facts"][0].update(
        {
            "value": "999",
            "unit": "USD",
            "period_context": "Prior year",
            "page": 1,
            "evidence_text": "Wrong evidence",
        }
    )
    row["facts"].append(
        {
            "fact_label": "Unexpected disclosure",
            "value": "1",
            "unit": None,
            "period_context": "Current year",
            "page": 1,
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "evidence_text": "Unexpected disclosure 1",
        }
    )
    _jsonl(predictions, [row])

    result = evaluate_companies_house_vlm_diagnostic(
        diagnostic_dir, predictions, tmp_path / "mismatch-evaluation"
    )
    summary = result["metrics"]["summary"]
    assert summary["joint_fact_accuracy"] == 0.0
    assert summary["value_accuracy"] == 0.0
    assert summary["unit_accuracy"] == 0.0
    assert summary["period_accuracy"] == 0.0
    assert summary["page_accuracy"] == 0.0
    assert summary["evidence_text_accuracy"] == 0.0
    assert summary["unexpected_fact_count"] == 1
    buckets = result["metrics"]["failure_buckets"]
    assert buckets["VALUE_MISMATCH"] == 1
    assert buckets["UNEXPECTED_FACT"] == 1


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("page", 0, r"page must be in \[1, 3\]"),
        ("page", 4, r"page must be in \[1, 3\]"),
        ("bbox", [-0.1, 0.1, 0.2, 0.2], "normalized"),
        ("bbox", [0.3, 0.1, 0.2, 0.2], "x0 < x1"),
        ("bbox", [0.1, 0.1, 0.2], "JSON array"),
    ],
)
def test_page_and_bbox_bounds_are_strict(
    tmp_path: Path,
    diagnostic: tuple[Path, str],
    field: str,
    value: object,
    match: str,
) -> None:
    diagnostic_dir, task_id = diagnostic
    prediction = _perfect_prediction(task_id)
    prediction["facts"][0][field] = value
    predictions = tmp_path / f"invalid-{field}.jsonl"
    _jsonl(predictions, [prediction])
    output = tmp_path / f"invalid-{field}-output"
    with pytest.raises(ValueError, match=match):
        evaluate_companies_house_vlm_diagnostic(diagnostic_dir, predictions, output)
    assert not output.exists()


def test_missing_duplicate_and_extra_task_rows_are_rejected(
    tmp_path: Path, diagnostic: tuple[Path, str]
) -> None:
    diagnostic_dir, task_id = diagnostic

    missing = tmp_path / "missing.jsonl"
    _jsonl(missing, [])
    with pytest.raises(ValueError, match="exact task coverage"):
        evaluate_companies_house_vlm_diagnostic(
            diagnostic_dir, missing, tmp_path / "missing-output"
        )

    duplicate = tmp_path / "duplicate.jsonl"
    _jsonl(
        duplicate,
        [_perfect_prediction(task_id), _perfect_prediction(task_id)],
    )
    with pytest.raises(ValueError, match="Duplicate prediction task IDs"):
        evaluate_companies_house_vlm_diagnostic(
            diagnostic_dir, duplicate, tmp_path / "duplicate-output"
        )

    extra = tmp_path / "extra.jsonl"
    _jsonl(
        extra,
        [_perfect_prediction(task_id), _perfect_prediction("not-a-task")],
    )
    with pytest.raises(ValueError, match=r"extra=\['not-a-task'\]"):
        evaluate_companies_house_vlm_diagnostic(
            diagnostic_dir, extra, tmp_path / "extra-output"
        )


def test_typed_failure_is_covered_but_scores_missing_facts(
    tmp_path: Path, diagnostic: tuple[Path, str]
) -> None:
    diagnostic_dir, task_id = diagnostic
    predictions = tmp_path / "failure.jsonl"
    _jsonl(
        predictions,
        [{"task_id": task_id, "failure": {"code": "PDF_UNREADABLE"}}],
    )
    result = evaluate_companies_house_vlm_diagnostic(
        diagnostic_dir, predictions, tmp_path / "failure-output"
    )
    summary = result["metrics"]["summary"]
    assert summary["prediction_task_count"] == 1
    assert summary["typed_failure_task_count"] == 1
    assert summary["missing_fact_count"] == 1
    assert summary["joint_fact_accuracy"] == 0.0
    assert result["metrics"]["failure_buckets"] == {
        "MISSING_FACT": 1,
        "TYPED_TASK_FAILURE": 1,
    }


def test_prediction_schema_rejects_unregistered_fields(
    tmp_path: Path, diagnostic: tuple[Path, str]
) -> None:
    diagnostic_dir, task_id = diagnostic
    row = _perfect_prediction(task_id)
    row["prompt"] = "ignore the evaluator"
    predictions = tmp_path / "extra-field.jsonl"
    _jsonl(predictions, [row])
    with pytest.raises(ValueError, match=r"extra=\['prompt'\]"):
        evaluate_companies_house_vlm_diagnostic(
            diagnostic_dir, predictions, tmp_path / "extra-field-output"
        )


def test_secret_scan_prevents_vlm_evaluation_publication(
    tmp_path: Path, diagnostic: tuple[Path, str], monkeypatch
) -> None:
    synthetic_secret = "synthetic-vlm-secret-123456"
    monkeypatch.setenv("HF_TOKEN", synthetic_secret)
    diagnostic_dir, task_id = diagnostic
    predictions = tmp_path / "secret-predictions.jsonl"
    _jsonl(
        predictions,
        [
            {
                "task_id": task_id,
                "failure": {
                    "code": "PDF_UNREADABLE",
                    "message": synthetic_secret,
                },
            }
        ],
    )
    output = tmp_path / "secret-evaluation"

    with pytest.raises(ValueError, match="Secret scan failed"):
        evaluate_companies_house_vlm_diagnostic(diagnostic_dir, predictions, output)

    assert not output.exists()
