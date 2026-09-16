"""Evaluation for the isolated Companies House image-PDF diagnostic.

The diagnostic deliberately remains separate from the XHTML benchmark.  In
particular, a model-emitted bounding box is validated for shape and page bounds,
but localization accuracy is reported only when the evaluator-only gold data
actually contains a bounding box.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .canonical_json import canonical_json_bytes as _canonical_json_bytes
from .provenance import assert_no_secrets

VLM_EVALUATION_VERSION = "companies-house-vlm-evaluation.v1"

_PREDICTION_FIELDS = {"task_id", "facts", "failure"}
_FACT_FIELDS = {
    "fact_label",
    "value",
    "unit",
    "period_context",
    "page",
    "bbox",
    "evidence_text",
}
_FAILURE_FIELDS = {"code", "message"}
_FAILURE_CODE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    handle = path.open("r", encoding="utf-8")
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise TypeError(f"Expected a JSON object in {path} line {line_number}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_canonical_json_bytes(dict(value), newline=True))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_json_bytes(dict(row), newline=True))
            count += 1
    return count


def _normalized_text(value: Any, *, casefold: bool = True) -> str:
    rendered = " ".join(unicodedata.normalize("NFKC", str(value or "")).split())
    return rendered.casefold() if casefold else rendered


def _decimal_value(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    rendered = unicodedata.normalize("NFKC", str(value)).strip()
    if not rendered:
        return None
    negative_parentheses = rendered.startswith("(") and rendered.endswith(")")
    if negative_parentheses:
        rendered = rendered[1:-1]
    rendered = rendered.replace(",", "").replace(" ", "")
    for prefix in ("GBP", "USD", "EUR", "£", "$", "€"):
        if rendered.upper().startswith(prefix.upper()):
            rendered = rendered[len(prefix) :]
            break
    try:
        parsed = Decimal(rendered)
    except (InvalidOperation, ValueError):
        return None
    return -abs(parsed) if negative_parentheses else parsed


def _value_equal(predicted: Any, expected: Any) -> bool:
    predicted_decimal = _decimal_value(predicted)
    expected_decimal = _decimal_value(expected)
    if predicted_decimal is not None and expected_decimal is not None:
        return predicted_decimal == expected_decimal
    return _normalized_text(predicted) == _normalized_text(expected)


def _nullable_text_equal(predicted: Any, expected: Any) -> bool:
    if predicted is None or expected is None:
        return predicted is expected
    return _normalized_text(predicted) == _normalized_text(expected)


def _bbox_error(value: Any) -> str | None:
    if not isinstance(value, list) or len(value) != 4:
        return "must be a JSON array [x0, y0, x1, y1]"
    coordinates: list[float] = []
    for coordinate in value:
        if (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(float(coordinate))
        ):
            return "coordinates must be finite JSON numbers"
        coordinates.append(float(coordinate))
    if any(coordinate < 0.0 or coordinate > 1.0 for coordinate in coordinates):
        return "coordinates must be normalized to the inclusive range [0, 1]"
    x0, y0, x1, y1 = coordinates
    if not x0 < x1 or not y0 < y1:
        return "coordinates must satisfy x0 < x1 and y0 < y1"
    return None


def _validate_failure(value: Any, *, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{location}.failure must be an object")
    extra = set(value) - _FAILURE_FIELDS
    missing = {"code"} - set(value)
    if extra or missing:
        raise ValueError(
            f"{location}.failure has invalid fields; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    code = value.get("code")
    if not isinstance(code, str) or not _FAILURE_CODE.fullmatch(code):
        raise ValueError(f"{location}.failure.code must match {_FAILURE_CODE.pattern}")
    message = value.get("message")
    if message is not None and not isinstance(message, str):
        raise ValueError(f"{location}.failure.message must be a string when present")
    return dict(value)


def _validate_fact(value: Any, *, page_count: int, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{location} must be an object")
    missing = _FACT_FIELDS - set(value)
    extra = set(value) - _FACT_FIELDS
    if missing or extra:
        raise ValueError(
            f"{location} has invalid fields; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    fact_label = value.get("fact_label")
    if not isinstance(fact_label, str) or not fact_label.strip():
        raise ValueError(f"{location}.fact_label must be a non-empty string")
    raw_value = value.get("value")
    if isinstance(raw_value, bool) or not isinstance(
        raw_value, (str, int, float, type(None))
    ):
        raise TypeError(f"{location}.value must be a string, number, or null")
    unit = value.get("unit")
    if unit is not None and not isinstance(unit, str):
        raise ValueError(f"{location}.unit must be a string or null")
    if not isinstance(value.get("period_context"), str):
        raise TypeError(f"{location}.period_context must be a string")
    page = value.get("page")
    if isinstance(page, bool) or not isinstance(page, int):
        raise TypeError(f"{location}.page must be an integer")
    if page < 1 or page > page_count:
        raise ValueError(
            f"{location}.page must be in [1, {page_count}], received {page}"
        )
    bbox_error = _bbox_error(value.get("bbox"))
    if bbox_error:
        raise ValueError(f"{location}.bbox {bbox_error}")
    if not isinstance(value.get("evidence_text"), str):
        raise TypeError(f"{location}.evidence_text must be a string")
    return dict(value)


def _validate_prediction(
    value: Mapping[str, Any], *, page_count: int, row_number: int
) -> dict[str, Any]:
    location = f"prediction row {row_number}"
    extra = set(value) - _PREDICTION_FIELDS
    missing = {"task_id"} - set(value)
    if extra or missing:
        raise ValueError(
            f"{location} has invalid fields; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    task_id = value.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"{location}.task_id must be a non-empty string")

    has_failure = "failure" in value
    has_facts = "facts" in value
    if has_failure:
        failure = _validate_failure(value["failure"], location=location)
        if has_facts and value["facts"] not in ([], None):
            raise ValueError(f"{location} cannot contain both facts and a failure")
        return {"task_id": task_id, "facts": [], "failure": failure}
    if not has_facts or not isinstance(value.get("facts"), list):
        raise ValueError(f"{location}.facts must be an array when failure is absent")

    facts = [
        _validate_fact(
            fact,
            page_count=page_count,
            location=f"{location}.facts[{index}]",
        )
        for index, fact in enumerate(value["facts"])
    ]
    labels = [str(fact["fact_label"]) for fact in facts]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"{location} has duplicate fact labels: {duplicates}")
    return {"task_id": task_id, "facts": facts}


def _validate_diagnostic(
    root: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    manifest_path = root / "diagnostic_manifest.json"
    cases_path = root / "cases.jsonl"
    gold_path = root / "gold.jsonl"
    manifest = _read_json(manifest_path)
    cases = _read_jsonl(cases_path)
    gold = _read_jsonl(gold_path)

    declared_files = manifest.get("files")
    if not isinstance(declared_files, Mapping):
        raise TypeError("diagnostic_manifest.json must contain a files object")
    for name, path in (("cases.jsonl", cases_path), ("gold.jsonl", gold_path)):
        declared = declared_files.get(name)
        if isinstance(declared, Mapping):
            declared = declared.get("sha256")
        actual = _sha256_file(path)
        if declared != actual:
            raise ValueError(
                f"Diagnostic artifact hash mismatch for {name}: "
                f"declared={declared!r}, actual={actual!r}"
            )

    case_by_id: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases, start=1):
        task_id = case.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"Diagnostic case {index} has no valid task_id")
        if task_id in case_by_id:
            raise ValueError(f"Duplicate task_id in diagnostic cases: {task_id}")
        page_count = case.get("page_count")
        if isinstance(page_count, bool) or not isinstance(page_count, int):
            raise TypeError(f"Diagnostic case {task_id} has invalid page_count")
        requested = case.get("requested_facts")
        if not isinstance(requested, list) or any(
            not isinstance(label, str) or not label for label in requested
        ):
            raise ValueError(f"Diagnostic case {task_id} has invalid requested_facts")
        if len(requested) != len(set(requested)):
            raise ValueError(f"Diagnostic case {task_id} has duplicate requested_facts")
        case_by_id[task_id] = case

    gold_by_id: dict[str, dict[str, Any]] = {}
    gold_fact_count = 0
    for index, row in enumerate(gold, start=1):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"Diagnostic gold row {index} has no valid task_id")
        if task_id in gold_by_id:
            raise ValueError(f"Duplicate task_id in diagnostic gold: {task_id}")
        facts = row.get("facts")
        if not isinstance(facts, list):
            raise TypeError(f"Diagnostic gold row {task_id} has invalid facts")
        labels: list[str] = []
        for fact_index, fact in enumerate(facts):
            if not isinstance(fact, Mapping):
                raise TypeError(
                    f"Diagnostic gold row {task_id} fact {fact_index} is not an object"
                )
            label = fact.get("fact_label")
            if not isinstance(label, str) or not label:
                raise ValueError(
                    f"Diagnostic gold row {task_id} fact {fact_index} has no label"
                )
            labels.append(label)
            bbox = fact.get("bbox")
            if bbox is not None:
                bbox_error = _bbox_error(bbox)
                if bbox_error:
                    raise ValueError(
                        f"Diagnostic gold row {task_id} fact {fact_index} bbox "
                        f"{bbox_error}"
                    )
        if len(labels) != len(set(labels)):
            raise ValueError(f"Diagnostic gold row {task_id} has duplicate fact labels")
        gold_fact_count += len(facts)
        gold_by_id[task_id] = row

    if set(case_by_id) != set(gold_by_id):
        raise ValueError("Diagnostic cases and gold must contain identical task IDs")
    for task_id, case in case_by_id.items():
        requested = set(case["requested_facts"])
        gold_labels = {str(fact["fact_label"]) for fact in gold_by_id[task_id]["facts"]}
        if requested != gold_labels:
            raise ValueError(
                f"Diagnostic requested/gold fact mismatch for task {task_id}"
            )
    if manifest.get("case_count") != len(cases):
        raise ValueError("Diagnostic manifest case_count does not match cases.jsonl")
    if manifest.get("gold_fact_count") != gold_fact_count:
        raise ValueError(
            "Diagnostic manifest gold_fact_count does not match gold.jsonl"
        )
    actual_bbox_available = any(
        fact.get("bbox") is not None for row in gold for fact in row["facts"]
    )
    if bool(manifest.get("bbox_gold_available")) != actual_bbox_available:
        raise ValueError(
            "Diagnostic manifest bbox_gold_available does not match gold.jsonl"
        )
    return manifest, cases, gold, case_by_id


def _bbox_iou(predicted: Sequence[float], expected: Sequence[float]) -> float:
    px0, py0, px1, py1 = (float(value) for value in predicted)
    ex0, ey0, ex1, ey1 = (float(value) for value in expected)
    intersection_width = max(0.0, min(px1, ex1) - max(px0, ex0))
    intersection_height = max(0.0, min(py1, ey1) - max(py0, ey0))
    intersection = intersection_width * intersection_height
    predicted_area = (px1 - px0) * (py1 - py0)
    expected_area = (ex1 - ex0) * (ey1 - ey0)
    union = predicted_area + expected_area - intersection
    return intersection / union if union else 0.0


def _safe_rate(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _markdown_report(metrics: Mapping[str, Any]) -> str:
    summary = metrics["summary"]
    bbox = metrics["bbox_evaluation"]
    lines = [
        "# Companies House image-PDF VLM diagnostic",
        "",
        (
            "This isolated diagnostic measures extraction and citations from image PDFs. "
            "It is not part of the XHTML benchmark score and is not an audit opinion."
        ),
        "",
        "## Extraction results",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Tasks | {summary['expected_task_count']} |",
        f"| Typed task failures | {summary['typed_failure_task_count']} |",
        f"| Gold facts | {summary['expected_fact_count']} |",
        f"| Matched facts | {summary['matched_fact_count']} |",
        f"| Value accuracy | {_format_rate(summary['value_accuracy'])} |",
        f"| Unit accuracy | {_format_rate(summary['unit_accuracy'])} |",
        f"| Period-context accuracy | {_format_rate(summary['period_accuracy'])} |",
        f"| Page accuracy | {_format_rate(summary['page_accuracy'])} |",
        f"| Evidence-text accuracy | {_format_rate(summary['evidence_text_accuracy'])} |",
        f"| Joint fact accuracy | {_format_rate(summary['joint_fact_accuracy'])} |",
        "",
        "## Bounding-box evaluation",
        "",
        (
            f"Prediction boxes passed schema and bounds checks: "
            f"{bbox['bbox_bounds_valid_count']}/{bbox['prediction_bbox_count']}."
        ),
        "",
    ]
    if bbox["gold_bbox_available"]:
        lines.extend(
            [
                (
                    f"Gold boxes: {bbox['gold_bbox_count']}; IoU evaluated: "
                    f"{bbox['bbox_iou_evaluated_count']}; mean matched-box IoU: "
                    f"{_format_rate(bbox['mean_iou_for_matched_predictions'])}."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                (
                    "Gold bounding boxes are unavailable. Localization accuracy and IoU "
                    "are therefore not evaluated; valid model boxes must not be interpreted "
                    "as accurate localization."
                ),
                "",
            ]
        )
    lines.extend(["## Failure buckets", ""])
    if metrics["failure_buckets"]:
        lines.extend(
            f"- `{code}`: {count}"
            for code, count in sorted(metrics["failure_buckets"].items())
        )
    else:
        lines.append("No extraction mismatches or typed failures.")
    lines.append("")
    return "\n".join(lines)


def _format_rate(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def evaluate_companies_house_vlm_diagnostic(
    diagnostic_dir: str | Path,
    predictions_jsonl: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate, score, and atomically persist one VLM diagnostic run.

    A successful prediction row has exactly ``task_id`` and ``facts``.  Each
    fact has exactly ``fact_label``, ``value``, ``unit``, ``period_context``,
    ``page``, ``bbox``, and ``evidence_text``.  Alternatively, a task may have
    ``failure={"code": "TYPED_CODE", "message": "..."}`` and no facts.
    Missing, duplicate, or extra task rows are rejected before evaluation.
    """

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite VLM evaluation directory: {destination}"
        )

    diagnostic_root = Path(diagnostic_dir)
    manifest, cases, gold_rows, case_by_id = _validate_diagnostic(diagnostic_root)
    prediction_path = Path(predictions_jsonl)
    raw_predictions = _read_jsonl(prediction_path)

    raw_by_id: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for row_number, row in enumerate(raw_predictions, start=1):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(
                f"prediction row {row_number}.task_id must be a non-empty string"
            )
        raw_by_id.setdefault(task_id, []).append((row_number, row))
    duplicate_ids = sorted(
        task_id for task_id, rows in raw_by_id.items() if len(rows) > 1
    )
    if duplicate_ids:
        raise ValueError(f"Duplicate prediction task IDs: {duplicate_ids}")
    expected_ids = [str(case["task_id"]) for case in cases]
    missing_ids = sorted(set(expected_ids) - set(raw_by_id))
    extra_ids = sorted(set(raw_by_id) - set(expected_ids))
    if missing_ids or extra_ids:
        raise ValueError(
            "Predictions must have exact task coverage; "
            f"missing={missing_ids}, extra={extra_ids}"
        )

    predictions: dict[str, dict[str, Any]] = {}
    for task_id in expected_ids:
        row_number, raw = raw_by_id[task_id][0]
        predictions[task_id] = _validate_prediction(
            raw,
            page_count=int(case_by_id[task_id]["page_count"]),
            row_number=row_number,
        )

    gold_by_id = {str(row["task_id"]): row for row in gold_rows}
    expected_fact_count = sum(len(row["facts"]) for row in gold_rows)
    counters = {
        "typed_failure_task_count": 0,
        "predicted_fact_count": 0,
        "matched_fact_count": 0,
        "missing_fact_count": 0,
        "unexpected_fact_count": 0,
        "value_correct_count": 0,
        "unit_correct_count": 0,
        "period_correct_count": 0,
        "page_correct_count": 0,
        "evidence_text_correct_count": 0,
        "joint_fact_correct_count": 0,
        "bbox_schema_valid_count": 0,
        "bbox_bounds_valid_count": 0,
        "gold_bbox_count": 0,
        "bbox_iou_evaluated_count": 0,
        "bbox_iou_at_0_5_count": 0,
    }
    iou_values: list[float] = []
    failures: list[dict[str, Any]] = []
    failure_buckets: dict[str, int] = {}

    def add_failure(row: dict[str, Any]) -> None:
        failures.append(row)
        for code in row["codes"]:
            failure_buckets[code] = failure_buckets.get(code, 0) + 1

    for task_id in expected_ids:
        prediction = predictions[task_id]
        predicted_facts = prediction["facts"]
        counters["predicted_fact_count"] += len(predicted_facts)
        counters["bbox_schema_valid_count"] += len(predicted_facts)
        counters["bbox_bounds_valid_count"] += len(predicted_facts)
        typed_failure = prediction.get("failure")
        if typed_failure is not None:
            counters["typed_failure_task_count"] += 1
            add_failure(
                {
                    "task_id": task_id,
                    "fact_label": None,
                    "codes": ["TYPED_TASK_FAILURE"],
                    "typed_failure": typed_failure,
                }
            )

        predicted_by_label = {str(fact["fact_label"]): fact for fact in predicted_facts}
        expected_facts = gold_by_id[task_id]["facts"]
        expected_labels = {str(fact["fact_label"]) for fact in expected_facts}
        for label in sorted(set(predicted_by_label) - expected_labels):
            counters["unexpected_fact_count"] += 1
            add_failure(
                {
                    "task_id": task_id,
                    "fact_label": label,
                    "codes": ["UNEXPECTED_FACT"],
                    "predicted": predicted_by_label[label],
                }
            )

        for expected in expected_facts:
            label = str(expected["fact_label"])
            if expected.get("bbox") is not None:
                counters["gold_bbox_count"] += 1
            predicted = predicted_by_label.get(label)
            if predicted is None:
                counters["missing_fact_count"] += 1
                missing_row: dict[str, Any] = {
                    "task_id": task_id,
                    "fact_label": label,
                    "codes": ["MISSING_FACT"],
                    "expected": expected,
                }
                if typed_failure is not None:
                    missing_row["typed_failure_code"] = typed_failure["code"]
                add_failure(missing_row)
                continue

            counters["matched_fact_count"] += 1
            checks = {
                "value": _value_equal(predicted["value"], expected.get("value")),
                "unit": _nullable_text_equal(predicted["unit"], expected.get("unit")),
                "period": _normalized_text(predicted["period_context"])
                == _normalized_text(expected.get("period_context")),
                "page": predicted["page"] == expected.get("page"),
                "evidence_text": _normalized_text(predicted["evidence_text"])
                == _normalized_text(expected.get("evidence_text")),
            }
            for key, counter in (
                ("value", "value_correct_count"),
                ("unit", "unit_correct_count"),
                ("period", "period_correct_count"),
                ("page", "page_correct_count"),
                ("evidence_text", "evidence_text_correct_count"),
            ):
                if checks[key]:
                    counters[counter] += 1
            if all(checks.values()):
                counters["joint_fact_correct_count"] += 1
            else:
                mismatch_codes = [
                    f"{field.upper()}_MISMATCH"
                    for field, passed in checks.items()
                    if not passed
                ]
                add_failure(
                    {
                        "task_id": task_id,
                        "fact_label": label,
                        "codes": mismatch_codes,
                        "expected": expected,
                        "predicted": predicted,
                    }
                )

            expected_bbox = expected.get("bbox")
            if expected_bbox is not None:
                iou = _bbox_iou(predicted["bbox"], expected_bbox)
                iou_values.append(iou)
                counters["bbox_iou_evaluated_count"] += 1
                if iou >= 0.5:
                    counters["bbox_iou_at_0_5_count"] += 1

    summary = {
        "expected_task_count": len(expected_ids),
        "prediction_task_count": len(predictions),
        "typed_failure_task_count": counters["typed_failure_task_count"],
        "successful_task_count": len(expected_ids)
        - counters["typed_failure_task_count"],
        "expected_fact_count": expected_fact_count,
        "predicted_fact_count": counters["predicted_fact_count"],
        "matched_fact_count": counters["matched_fact_count"],
        "missing_fact_count": counters["missing_fact_count"],
        "unexpected_fact_count": counters["unexpected_fact_count"],
        "value_correct_count": counters["value_correct_count"],
        "value_accuracy": _safe_rate(
            counters["value_correct_count"], expected_fact_count
        ),
        "unit_correct_count": counters["unit_correct_count"],
        "unit_accuracy": _safe_rate(
            counters["unit_correct_count"], expected_fact_count
        ),
        "period_correct_count": counters["period_correct_count"],
        "period_accuracy": _safe_rate(
            counters["period_correct_count"], expected_fact_count
        ),
        "page_correct_count": counters["page_correct_count"],
        "page_accuracy": _safe_rate(
            counters["page_correct_count"], expected_fact_count
        ),
        "evidence_text_correct_count": counters["evidence_text_correct_count"],
        "evidence_text_accuracy": _safe_rate(
            counters["evidence_text_correct_count"], expected_fact_count
        ),
        "joint_fact_correct_count": counters["joint_fact_correct_count"],
        "joint_fact_accuracy": _safe_rate(
            counters["joint_fact_correct_count"], expected_fact_count
        ),
    }
    bbox_gold_available = counters["gold_bbox_count"] > 0
    bbox_evaluation = {
        "prediction_bbox_count": counters["predicted_fact_count"],
        "bbox_schema_valid_count": counters["bbox_schema_valid_count"],
        "bbox_schema_valid_rate": _safe_rate(
            counters["bbox_schema_valid_count"], counters["predicted_fact_count"]
        ),
        "bbox_bounds_valid_count": counters["bbox_bounds_valid_count"],
        "bbox_bounds_valid_rate": _safe_rate(
            counters["bbox_bounds_valid_count"], counters["predicted_fact_count"]
        ),
        "gold_bbox_available": bbox_gold_available,
        "gold_bbox_count": counters["gold_bbox_count"],
        "bbox_iou_evaluated_count": counters["bbox_iou_evaluated_count"],
        "bbox_iou_coverage": _safe_rate(
            counters["bbox_iou_evaluated_count"], counters["gold_bbox_count"]
        ),
        "mean_iou_for_matched_predictions": (
            sum(iou_values) / len(iou_values) if iou_values else None
        ),
        "iou_at_0_5_rate_for_matched_predictions": _safe_rate(
            counters["bbox_iou_at_0_5_count"],
            counters["bbox_iou_evaluated_count"],
        ),
        "localization_accuracy_available": bbox_gold_available,
        "note": (
            "IoU is evaluated only for facts with labelled gold boxes."
            if bbox_gold_available
            else "Gold bounding boxes are unavailable; schema and normalized bounds "
            "are checked, but localization accuracy and IoU are not reported."
        ),
    }
    metrics = {
        "evaluation_version": VLM_EVALUATION_VERSION,
        "diagnostic_version": manifest.get("diagnostic_version"),
        "summary": summary,
        "bbox_evaluation": bbox_evaluation,
        "failure_buckets": dict(sorted(failure_buckets.items())),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        _write_json(temporary / "metrics.json", metrics)
        _write_jsonl(temporary / "failures.jsonl", failures)
        (temporary / "report.md").write_text(
            _markdown_report(metrics), encoding="utf-8", newline="\n"
        )
        source_artifacts = {
            "diagnostic_manifest.json": _sha256_file(
                diagnostic_root / "diagnostic_manifest.json"
            ),
            "cases.jsonl": _sha256_file(diagnostic_root / "cases.jsonl"),
            "gold.jsonl": _sha256_file(diagnostic_root / "gold.jsonl"),
            "predictions.jsonl": _sha256_file(prediction_path),
        }
        artifacts = {
            name: {
                "sha256": _sha256_file(temporary / name),
                "bytes": (temporary / name).stat().st_size,
            }
            for name in ("metrics.json", "failures.jsonl", "report.md")
        }
        evaluation_material = {
            "evaluation_version": VLM_EVALUATION_VERSION,
            "diagnostic_version": manifest.get("diagnostic_version"),
            "source_artifacts": source_artifacts,
            "artifacts": artifacts,
            "task_count": len(expected_ids),
            "fact_count": expected_fact_count,
        }
        evaluation_id = hashlib.sha256(
            _canonical_json_bytes(evaluation_material)
        ).hexdigest()
        evaluation_manifest = {
            **evaluation_material,
            "evaluation_id": evaluation_id,
        }
        _write_json(temporary / "evaluation_manifest.json", evaluation_manifest)
        assert_no_secrets([temporary])
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite VLM evaluation directory: {destination}"
            )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "evaluation_id": evaluation_id,
        "output_dir": str(destination),
        "metrics": metrics,
        "paths": {
            "metrics": str(destination / "metrics.json"),
            "failures": str(destination / "failures.jsonl"),
            "report": str(destination / "report.md"),
            "manifest": str(destination / "evaluation_manifest.json"),
        },
    }


__all__ = [
    "VLM_EVALUATION_VERSION",
    "evaluate_companies_house_vlm_diagnostic",
]
