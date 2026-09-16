"""Deterministic, leakage-safe benchmark packaging for AuditOps agents.

The corpus builders deliberately keep inference-visible cases separate from the
evaluator-only gold records.  This module does not download data or invoke a
model; it packages existing quantitative and narrative JSONL task files into an
immutable benchmark directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .agent_context import build_context_pack, canonical_quant_evidence_content
from .agent_contracts import (
    AGENT_TASK_INPUT_VERSION,
    build_agent_task_input,
    canonical_decimal_string,
    validate_agent_proposal,
    validate_agent_task_input,
    validate_tool_observation,
)
from .agent_operations import (
    NARRATIVE_SUBTYPES,
    expected_tool_arguments,
    task_operation_spec,
)
from .agent_tools import evaluate_quant_evidence, load_frozen_evidence
from .agent_verifier import verify_proposal
from .canonical_json import canonical_json_bytes as _canonical_json_bytes
from .canonical_json import canonical_json_sha256
from .provenance import assert_no_secrets, assert_no_secrets_in_value
from .text_support import is_contiguous_text_supported

BENCHMARK_VERSION = "agent_benchmark.v2.2"
GOLD_RECORD_VERSION = "v2.2"
FEW_SHOT_EXAMPLE_VERSION = "v2.2"
DEFAULT_SEED = 20260821
DEFAULT_QUANT_COUNT = 300
DEFAULT_NARRATIVE_COUNT = 200
DEFAULT_FEW_SHOT_PER_TASK_TYPE = 4
DEFAULT_FEW_SHOT_PER_NARRATIVE_SUBTYPE = 4
DEFAULT_NARRATIVE_FEW_SHOT_COUNT = 16
FEW_SHOT_NARRATIVE_EXCERPT_MAX_CHARS = 512
BENCHMARK_PROFILE_US_V0_1 = "us_v0.1"
BENCHMARK_PROFILE_US_OFFLINE_SAMPLE_V0_1 = "us_offline_sample_v0.1"
BENCHMARK_PROFILE_SYNTHETIC = "synthetic"
BENCHMARK_PROFILES = (
    BENCHMARK_PROFILE_US_V0_1,
    BENCHMARK_PROFILE_US_OFFLINE_SAMPLE_V0_1,
    BENCHMARK_PROFILE_SYNTHETIC,
)
STRATIFICATION_REQUIREMENTS_VERSION = "auditops-agent-stratification-requirements.v2.2"
_STRATIFICATION_AXES = {
    "quant": {
        "status",
        "operation",
        "issuer",
        "period",
        "unit_profile",
        "context_profile",
        "dimension_profile",
        "filing_profile",
        "negative_type",
    },
    "narrative": {
        "status",
        "task_family",
        "issuer",
        "period",
        "filing_profile",
        "negative_type",
    },
}
_US_STRATIFICATION_DISTINCT_CAPS = {
    "quant": {
        "operation": 12,
        "issuer": 50,
        "period": 4,
        "unit_profile": 6,
        "context_profile": 6,
        "dimension_profile": 4,
        "filing_profile": 6,
        "negative_type": 8,
    },
    "narrative": {
        "task_family": 12,
        "issuer": 50,
        "period": 4,
        "filing_profile": 6,
        "negative_type": 8,
    },
}
_STRATIFICATION_DISTINCT_CAPS = {
    BENCHMARK_PROFILE_US_V0_1: _US_STRATIFICATION_DISTINCT_CAPS,
    BENCHMARK_PROFILE_US_OFFLINE_SAMPLE_V0_1: _US_STRATIFICATION_DISTINCT_CAPS,
    BENCHMARK_PROFILE_SYNTHETIC: {
        "quant": {
            "operation": 4,
            "issuer": 8,
            "period": 3,
            "unit_profile": 4,
            "context_profile": 4,
            "dimension_profile": 3,
            "filing_profile": 4,
            "negative_type": 4,
        },
        "narrative": {
            "task_family": 4,
            "issuer": 8,
            "period": 3,
            "filing_profile": 4,
            "negative_type": 4,
        },
    },
}
_MATERIALIZED_BENCHMARK_PROFILES = frozenset(
    {
        BENCHMARK_PROFILE_US_V0_1,
        BENCHMARK_PROFILE_US_OFFLINE_SAMPLE_V0_1,
    }
)
PARITY_SLICE_VERSION = "agent_parity_slice.v2"
PARITY_SLICE_COUNT = 50
MATERIALIZATION_MANIFEST_NAME = "materialization_manifest.json"

_GOLD_ONLY_KEYS = {
    "answerability",
    "code_target",
    "correct_answer",
    "default_code",
    "donor_task_id",
    "expected_answer",
    "expected_chunk_ids",
    "extractive_answer",
    "gold",
    "gold_answer",
    "gold_label",
    "ground_truth",
    "is_answerable",
    "negative_type",
    "reference_answer",
    "refusal_code",
    "source_answer_id",
    "solution",
    "target",
    "target_answer",
    "target_status",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_rank(seed: int, namespace: str, row: Mapping[str, Any]) -> str:
    identity = row.get("task_id") or canonical_json_sha256(row)
    return _sha256_bytes(f"{seed}|{namespace}|{identity}".encode())


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source_path = Path(path)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {source_path} line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError(
                    f"Expected an object in {source_path} line {line_number}"
                )
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(
                    f"Missing non-empty task_id in {source_path} line {line_number}"
                )
            if task_id in seen:
                raise ValueError(f"Duplicate task_id {task_id!r} in {source_path}")
            seen.add(task_id)
            records.append(row)
    return records


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_json_bytes(dict(row), newline=True))
            count += 1
    return count


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_canonical_json_bytes(dict(value), newline=True))


def _first_text(row: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _filing_id(row: Mapping[str, Any]) -> str:
    filing_id = _first_text(row, "filing_id", "accession_number", "document_id")
    if filing_id is None:
        raise ValueError(f"Task {row.get('task_id')!r} has no filing identifier")
    return filing_id


def _entity_id(row: Mapping[str, Any]) -> str:
    nested_metadata = row.get("filing_metadata")
    metadata = nested_metadata if isinstance(nested_metadata, Mapping) else {}
    entity_id = _first_text(
        row,
        "entity_id",
        "ticker",
        "company_number",
        "cik",
        "lei",
        "issuer_id",
    ) or _first_text(
        metadata, "entity_id", "ticker", "company_number", "cik", "lei", "issuer_id"
    )
    return entity_id or _filing_id(row)


def _source_entity_id(row: Mapping[str, Any]) -> str:
    """Return the strongest available issuer identifier for split isolation."""

    metadata = _as_mapping(row.get("filing_metadata"))
    source_entity_id = _first_text(
        row,
        "source_entity_id",
        "cik",
        "company_number",
        "lei",
        "issuer_id",
    ) or _first_text(
        metadata,
        "source_entity_id",
        "cik",
        "company_number",
        "lei",
        "issuer_id",
    )
    return source_entity_id or _entity_id(row)


def _case_source_entity_id(case: Mapping[str, Any]) -> str:
    entity = _as_mapping(case.get("entity"))
    return (
        _first_text(entity, "source_entity_id", "cik", "company_number", "lei")
        or _first_text(entity, "entity_id", "ticker")
        or str(case.get("task_id") or "")
    )


def _benchmark_profile_errors(
    benchmark_profile: Any,
    *,
    jurisdiction: Any,
    source_system: Any,
    seed: Any,
    quant_count: Any,
    narrative_count: Any,
    narrative_answerable: Any,
    narrative_unanswerable: Any,
    total_count: Any,
) -> list[str]:
    """Validate immutable profile claims independently of caller-supplied labels."""

    if benchmark_profile not in BENCHMARK_PROFILES:
        return ["benchmark_profile must be one of " + ", ".join(BENCHMARK_PROFILES)]
    if benchmark_profile == BENCHMARK_PROFILE_SYNTHETIC:
        return []

    expected = {
        "jurisdiction": "US",
        "source_system": (
            "SEC-EDGAR-CACHED"
            if benchmark_profile == BENCHMARK_PROFILE_US_OFFLINE_SAMPLE_V0_1
            else "SEC-EDGAR"
        ),
        "seed": DEFAULT_SEED,
        "quant_count": DEFAULT_QUANT_COUNT,
        "narrative_count": DEFAULT_NARRATIVE_COUNT,
        "narrative_answerable": DEFAULT_NARRATIVE_COUNT // 2,
        "narrative_unanswerable": DEFAULT_NARRATIVE_COUNT // 2,
        "total_count": DEFAULT_QUANT_COUNT + DEFAULT_NARRATIVE_COUNT,
    }
    actual = {
        "jurisdiction": jurisdiction,
        "source_system": source_system,
        "seed": seed,
        "quant_count": quant_count,
        "narrative_count": narrative_count,
        "narrative_answerable": narrative_answerable,
        "narrative_unanswerable": narrative_unanswerable,
        "total_count": total_count,
    }
    return [
        f"{benchmark_profile} requires {key}={expected_value!r}, got {actual[key]!r}"
        for key, expected_value in expected.items()
        if actual[key] != expected_value
    ]


def _period_key(row: Mapping[str, Any]) -> str | None:
    period = row.get("period")
    if isinstance(period, Mapping):
        value = _first_text(period, "period_key")
        if value:
            return value
    return _first_text(row, "period_key", "report_date")


def _quant_status(row: Mapping[str, Any]) -> str:
    status = _first_text(row, "target_status")
    target = row.get("target_answer")
    if status is None and isinstance(target, Mapping):
        status = _first_text(target, "status")
    if status not in {"OK", "REFUSAL"}:
        raise ValueError(
            f"Quant task {row.get('task_id')!r} has unsupported target status {status!r}"
        )
    return status


def _narrative_status(row: Mapping[str, Any]) -> str:
    answerability = _first_text(row, "answerability")
    if answerability == "ANSWERABLE":
        return "OK"
    if answerability == "UNANSWERABLE":
        return "REFUSAL"
    raise ValueError(
        f"Narrative task {row.get('task_id')!r} has unsupported answerability {answerability!r}"
    )


def _filing_profile(row: Mapping[str, Any]) -> tuple[str, ...]:
    metadata = _as_mapping(row.get("filing_metadata"))

    def value(*keys: str) -> str:
        return _first_text(row, *keys) or _first_text(metadata, *keys) or ""

    taxonomy_refs = row.get("taxonomy_refs") or metadata.get("taxonomy_refs") or []
    if isinstance(taxonomy_refs, str):
        taxonomy_profile = taxonomy_refs
    elif isinstance(taxonomy_refs, Sequence):
        taxonomy_profile = "|".join(
            sorted(str(item) for item in taxonomy_refs if str(item).strip())
        )
    else:
        taxonomy_profile = ""
    return (
        value("form_type", "account_type", "report_type"),
        value("audit_status", "audit_exemption_status"),
        value("reporting_framework"),
        taxonomy_profile,
    )


def _profile_value(row: Mapping[str, Any], key: str) -> str | None:
    metadata = _as_mapping(row.get("filing_metadata"))
    return _first_text(row, key) or _first_text(metadata, key)


def _resolved_profile_value(
    row: Mapping[str, Any],
    key: str,
    configured: str,
    *,
    allow_mixed: bool = False,
) -> str:
    source_value = _profile_value(row, key)
    if configured == "MIXED" and allow_mixed:
        if source_value is None:
            raise ValueError(
                f"Task {row.get('task_id')!r} must declare {key} when the benchmark profile is MIXED"
            )
        return source_value
    if source_value is not None and source_value != configured:
        raise ValueError(
            f"Task {row.get('task_id')!r} declares {key}={source_value!r}, "
            f"which conflicts with configured {configured!r}"
        )
    return source_value or configured


def _task_family(row: Mapping[str, Any], task_type: str) -> str:
    explicit = _first_text(row, "task_family")
    if explicit is not None:
        return explicit
    if task_type in {"quant", "quant_metric"}:
        family = _first_text(row, "metric_spec_id", "operation") or "unknown_metric"
        return f"quant_metric:{family}:{_quant_status(row)}"
    family = (
        _first_text(row, "label", "audit_task_type", "negative_type") or "narrative"
    )
    return f"narrative_citation:{family}:{_narrative_status(row)}"


def _narrative_subtype(row: Mapping[str, Any]) -> str:
    """Resolve one public v2.2 narrative subtype from frozen source metadata."""

    material = " ".join(
        str(value)
        for value in (
            row.get("task_family"),
            row.get("template_id"),
            row.get("label"),
            row.get("audit_task_type"),
        )
        if value is not None
    ).casefold()
    aliases = (
        (
            "critical_audit_matter",
            ("critical_audit_matter", "critical audit matter", "cam"),
        ),
        (
            "auditor_report_opinion_language",
            (
                "auditor_report_opinion_language",
                "auditor report opinion",
                "opinion_language",
                "opinion",
            ),
        ),
        ("accounting_policy", ("accounting_policy", "accounting policy")),
        ("footnote_note", ("footnote_note", "footnote", "note")),
    )
    for subtype, patterns in aliases:
        if any(pattern in material for pattern in patterns):
            return subtype
    raise ValueError(
        f"Narrative task {row.get('task_id')!r} has no registered v2.2 subtype"
    )


def _template_id(row: Mapping[str, Any], task_type: str) -> str:
    explicit = _first_text(
        row,
        "template_id",
        "task_template_id",
        "question_template_id",
    )
    if explicit is not None:
        return explicit
    raise ValueError(
        f"Task {row.get('task_id')!r} must declare an explicit stable template_id"
    )


def _quant_unit_dimension_profile(row: Mapping[str, Any]) -> tuple[str, str, str]:
    target = _as_mapping(row.get("target_answer"))
    units: set[str] = set()
    if target.get("unit") is not None:
        units.add(str(target["unit"]))
    contexts: set[str] = set()
    dimensions: list[Any] = []
    sources: list[Any] = []
    for field in ("canonical_inputs", "evidence_items"):
        candidates = row.get(field)
        if isinstance(candidates, Sequence) and not isinstance(
            candidates, (str, bytes, bytearray)
        ):
            sources.extend(candidates)
    for item in sources:
        if not isinstance(item, Mapping):
            continue
        metadata = _as_mapping(item.get("metadata"))
        unit = item.get("unit_canon") or item.get("unit")
        if unit is not None:
            units.add(str(unit))
        context_id = item.get("context_id") or metadata.get("context_id")
        if context_id is not None:
            contexts.add(str(context_id))
        if "dimensions" in item:
            dimension = item.get("dimensions")
        elif "dimensions" in metadata:
            dimension = metadata.get("dimensions")
        else:
            dimension = (
                item.get("dimension") or item.get("segment") or metadata.get("scenario")
            )
        if dimension is not None:
            dimensions.append(dimension)
    unit_profile = "|".join(sorted(units)) or "UNSPECIFIED"
    context_profile = (
        canonical_json_sha256(sorted(contexts))[:16] if contexts else "UNSPECIFIED"
    )
    if not dimensions:
        dimension_profile = "UNSPECIFIED"
    elif all(not value for value in dimensions):
        dimension_profile = "DIMENSIONLESS"
    else:
        dimension_profile = canonical_json_sha256(dimensions)[:16]
    return unit_profile, context_profile, dimension_profile


def _stratification_axes(row: Mapping[str, Any], task_type: str) -> dict[str, str]:
    filing_profile = "|".join(_filing_profile(row)) or "UNSPECIFIED"
    if task_type == "quant":
        unit_profile, context_profile, dimension_profile = (
            _quant_unit_dimension_profile(row)
        )
        return {
            "status": _quant_status(row),
            "operation": _first_text(row, "metric_spec_id", "operation")
            or "UNSPECIFIED",
            "issuer": _source_entity_id(row),
            "period": _period_key(row) or "UNSPECIFIED",
            "unit_profile": unit_profile,
            "context_profile": context_profile,
            "dimension_profile": dimension_profile,
            "filing_profile": filing_profile,
            "negative_type": _first_text(row, "negative_type") or "NONE",
        }
    return {
        "status": _narrative_status(row),
        "task_family": _task_family(row, "narrative"),
        "issuer": _source_entity_id(row),
        "period": _period_key(row) or "UNSPECIFIED",
        "filing_profile": filing_profile,
        "negative_type": _first_text(row, "negative_type") or "NONE",
    }


def _stratification_profile(
    rows: Sequence[Mapping[str, Any]], task_type: str
) -> dict[str, Any]:
    return _axes_profile([_stratification_axes(row, task_type) for row in rows])


def _axes_profile(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for axis, value in row.items():
            counters[str(axis)][str(value)] += 1
    return {
        "row_count": len(rows),
        "axes": {
            axis: dict(sorted(counter.items()))
            for axis, counter in sorted(counters.items())
        },
    }


def _stratification_requirements(
    eligible: Mapping[str, Any],
    *,
    task_type: str,
    selected_count: int,
    benchmark_profile: str,
) -> dict[str, Any]:
    """Materialize auditable, jointly conservative coverage quotas."""

    profile_caps = _STRATIFICATION_DISTINCT_CAPS.get(benchmark_profile)
    if profile_caps is None or task_type not in profile_caps:
        raise ValueError(f"Unsupported benchmark profile {benchmark_profile!r}")
    eligible_axes = _as_mapping(eligible.get("axes"))
    expected_statuses = {
        "OK": selected_count // 2 + selected_count % 2,
        "REFUSAL": selected_count // 2,
    }
    axis_requirements: dict[str, Any] = {
        "status": {
            "kind": "exact_counts",
            "required_counts": expected_statuses,
        }
    }
    for axis, cap in profile_caps[task_type].items():
        eligible_distinct = len(_as_mapping(eligible_axes.get(axis)))
        selection_capacity = (
            max(expected_statuses.values()) if axis == "issuer" else selected_count
        )
        axis_requirements[axis] = {
            "kind": "minimum_distinct",
            "eligible_distinct": eligible_distinct,
            "minimum_distinct": min(eligible_distinct, selection_capacity, cap),
            "profile_cap": cap,
            "selection_capacity": selection_capacity,
        }
    return {
        "requirements_version": STRATIFICATION_REQUIREMENTS_VERSION,
        "benchmark_profile": benchmark_profile,
        "eligible_row_count": eligible.get("row_count"),
        "selected_row_count": selected_count,
        "axes": axis_requirements,
    }


def _stratification_errors(
    stratification: Mapping[str, Any],
    *,
    benchmark_profile: str,
    quant_count: int,
    narrative_count: int,
) -> list[str]:
    errors: list[str] = []
    if benchmark_profile not in BENCHMARK_PROFILES:
        return [f"unsupported stratification benchmark profile {benchmark_profile!r}"]
    for task_type, selected_count in (
        ("quant", quant_count),
        ("narrative", narrative_count),
    ):
        section = _as_mapping(stratification.get(task_type))
        eligible = _as_mapping(section.get("eligible"))
        selected = _as_mapping(section.get("selected"))
        eligible_axes = _as_mapping(eligible.get("axes"))
        selected_axes = _as_mapping(selected.get("axes"))
        if selected.get("row_count") != selected_count:
            errors.append(f"{task_type} stratification row count is invalid")
        expected_axes = _STRATIFICATION_AXES[task_type]
        if set(eligible_axes) != expected_axes or set(selected_axes) != expected_axes:
            errors.append(f"{task_type} stratification axes are missing or unexpected")
            continue
        expected_requirements = _stratification_requirements(
            eligible,
            task_type=task_type,
            selected_count=selected_count,
            benchmark_profile=benchmark_profile,
        )
        requirements = _as_mapping(section.get("requirements"))
        if requirements != expected_requirements:
            errors.append(
                f"{task_type} stratification requirements do not match the deterministic profile"
            )
        requirement_axes = _as_mapping(expected_requirements.get("axes"))
        statuses = _as_mapping(selected_axes.get("status"))
        expected_statuses = _as_mapping(
            _as_mapping(requirement_axes.get("status")).get("required_counts")
        )
        for status, required_count in expected_statuses.items():
            actual_count = statuses.get(status, 0)
            if actual_count != required_count:
                errors.append(
                    f"{task_type} status {status} count mismatch: "
                    f"requires {required_count}, selected {actual_count}"
                )
        unexpected_statuses = sorted(set(statuses) - set(expected_statuses))
        if unexpected_statuses:
            errors.append(
                f"{task_type} status strata contain unexpected values: "
                + ", ".join(unexpected_statuses)
            )
        for axis in expected_axes - {"status"}:
            available = set(_as_mapping(eligible_axes.get(axis)))
            chosen = set(_as_mapping(selected_axes.get(axis)))
            if not chosen or not chosen.issubset(available):
                errors.append(f"{task_type} {axis} strata are invalid")
                continue
            required_distinct = _as_mapping(requirement_axes.get(axis)).get(
                "minimum_distinct"
            )
            if not isinstance(required_distinct, int):
                errors.append(f"{task_type} {axis} requirement is invalid")
                continue
            if len(chosen) < required_distinct:
                errors.append(
                    f"{task_type} {axis} distinct-strata shortage: requires "
                    f"{required_distinct}, selected {len(chosen)}, eligible {len(available)}"
                )
        refusal_count = int(expected_statuses["REFUSAL"])
        negative_counts = _as_mapping(selected_axes.get("negative_type"))
        try:
            hard_negative_count = sum(
                int(count)
                for label, count in negative_counts.items()
                if label != "NONE"
            )
        except (TypeError, ValueError):
            errors.append(f"{task_type} hard-negative counts are invalid")
            hard_negative_count = -1
        if hard_negative_count < refusal_count:
            errors.append(
                f"{task_type} hard-negative shortage: requires {refusal_count}, "
                f"selected {hard_negative_count}"
            )
    return errors


def _find_forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            lower_key = key.lower()
            if lower_key in _GOLD_ONLY_KEYS or lower_key.startswith(
                ("target_", "expected_")
            ):
                found.append(path)
            found.extend(_find_forbidden_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_keys(child, f"{prefix}[{index}]"))
    return found


def _balanced_sample(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    *,
    seed: int,
    namespace: str,
    strata: Callable[[Mapping[str, Any]], Sequence[Any]],
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("Sample count cannot be negative")
    if len(rows) < count:
        raise ValueError(
            f"Need {count} {namespace} tasks but only {len(rows)} eligible tasks are available"
        )

    remaining = [dict(row) for row in rows]
    axis_counts: list[Counter[str]] = []
    materialized_strata: dict[str, tuple[str, ...]] = {}
    for row in remaining:
        task_id = str(row["task_id"])
        values = tuple("" if value is None else str(value) for value in strata(row))
        materialized_strata[task_id] = values
        while len(axis_counts) < len(values):
            axis_counts.append(Counter())

    selected: list[dict[str, Any]] = []
    while len(selected) < count:

        def candidate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
            values = materialized_strata[str(row["task_id"])]
            unseen = sum(
                1
                for index, value in enumerate(values)
                if axis_counts[index][value] == 0
            )
            balance = sum(
                1_000_000 // (axis_counts[index][value] + 1)
                for index, value in enumerate(values)
            )
            return (
                -unseen,
                -balance,
                _stable_rank(seed, namespace, row),
                str(row["task_id"]),
            )

        chosen = min(remaining, key=candidate_key)
        remaining.remove(chosen)
        selected.append(chosen)
        for index, value in enumerate(materialized_strata[str(chosen["task_id"])]):
            axis_counts[index][value] += 1
    selected.sort(key=lambda row: str(row["task_id"]))
    return selected


def _select_quant(
    rows: Sequence[Mapping[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    by_status = {
        status: [row for row in rows if _quant_status(row) == status]
        for status in ("OK", "REFUSAL")
    }
    desired = {"OK": count // 2 + count % 2, "REFUSAL": count // 2}
    shortages = {
        status: desired[status] - len(by_status[status])
        for status in by_status
        if len(by_status[status]) < desired[status]
    }
    if shortages:
        detail = ", ".join(
            f"{status} short by {shortage}"
            for status, shortage in sorted(shortages.items())
        )
        raise ValueError(
            f"Need a balanced {count}-case quantitative benchmark: {detail}"
        )
    quotas = desired

    selected: list[dict[str, Any]] = []
    for status in ("OK", "REFUSAL"):
        selected.extend(
            _balanced_sample(
                by_status[status],
                quotas[status],
                seed=seed,
                namespace=f"quant-{status.lower()}",
                strata=lambda row: (
                    row.get("metric_spec_id") or row.get("operation"),
                    row.get("negative_type"),
                    _period_key(row),
                    *_quant_unit_dimension_profile(row),
                    *_filing_profile(row),
                    _source_entity_id(row),
                ),
            )
        )
    selected.sort(key=lambda row: str(row["task_id"]))
    return selected


def _select_narrative(
    rows: Sequence[Mapping[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if count % 2:
        raise ValueError(
            "Narrative benchmark count must be even to remain answerability-balanced"
        )
    per_status = count // 2
    selected: list[dict[str, Any]] = []
    for status in ("OK", "REFUSAL"):
        eligible = [row for row in rows if _narrative_status(row) == status]
        selected.extend(
            _balanced_sample(
                eligible,
                per_status,
                seed=seed,
                namespace=f"narrative-{status.lower()}",
                strata=lambda row: (
                    row.get("label"),
                    row.get("negative_type"),
                    row.get("form_type") or row.get("account_type"),
                    _period_key(row),
                    *_filing_profile(row),
                    _source_entity_id(row),
                ),
            )
        )
    selected.sort(key=lambda row: str(row["task_id"]))
    return selected


def _few_shot_candidates(
    quant_rows: Sequence[Mapping[str, Any]],
    narrative_rows: Sequence[Mapping[str, Any]],
    seed: int,
    salt: int,
    *,
    require_all_subtypes: bool = False,
) -> list[tuple[str, dict[str, Any]]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    categories: list[tuple[str, str, list[Mapping[str, Any]], str]] = [
        (
            "quant",
            "OK",
            [row for row in quant_rows if _quant_status(row) == "OK"],
            "quant",
        ),
        (
            "quant",
            "REFUSAL",
            [row for row in quant_rows if _quant_status(row) == "REFUSAL"],
            "quant",
        ),
    ]
    available_subtypes = sorted({_narrative_subtype(row) for row in narrative_rows})
    narrative_subtypes = (
        sorted(NARRATIVE_SUBTYPES) if require_all_subtypes else available_subtypes
    )
    for subtype in narrative_subtypes:
        for status in ("OK", "REFUSAL"):
            categories.append(
                (
                    "narrative",
                    status,
                    [
                        row
                        for row in narrative_rows
                        if _narrative_subtype(row) == subtype
                        and _narrative_status(row) == status
                    ],
                    subtype,
                )
            )
    for task_type, status, rows, family in categories:
        by_template: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_template[_template_id(row, task_type)].append(row)
        eligible_templates = sorted(
            (
                template_id
                for template_id, group in by_template.items()
                if len(group) >= 2
            ),
            key=lambda template_id: (
                len(by_template[template_id]),
                _sha256_bytes(
                    f"{seed + salt}|few-shot-template|{task_type}|{family}|{status}|{template_id}".encode()
                ),
            ),
        )
        if not eligible_templates:
            raise ValueError(
                f"Need at least two {status} {task_type}/{family} tasks sharing a held-out template"
            )
        held_out_template = eligible_templates[0]
        ranked = sorted(
            by_template[held_out_template],
            key=lambda row: (
                _stable_rank(
                    seed + salt, f"few-shot|{task_type}|{family}|{status}", row
                ),
                str(row["task_id"]),
            ),
        )
        selected.extend((task_type, dict(row)) for row in ranked[:2])
    return selected


def _select_few_shots_and_evaluation(
    quant_rows: Sequence[Mapping[str, Any]],
    narrative_rows: Sequence[Mapping[str, Any]],
    quant_count: int,
    narrative_count: int,
    seed: int,
    *,
    require_all_subtypes: bool = False,
) -> tuple[
    list[tuple[str, dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
]:
    for salt in range(1024):
        examples = _few_shot_candidates(
            quant_rows,
            narrative_rows,
            seed,
            salt,
            require_all_subtypes=require_all_subtypes,
        )
        reserved_entities = {_source_entity_id(row) for _, row in examples}
        reserved_filings = {_filing_id(row) for _, row in examples}
        eligible_quant = [
            row
            for row in quant_rows
            if not row.get("development_only")
            and _source_entity_id(row) not in reserved_entities
            and _filing_id(row) not in reserved_filings
        ]
        eligible_narrative = [
            row
            for row in narrative_rows
            if not row.get("development_only")
            and _source_entity_id(row) not in reserved_entities
            and _filing_id(row) not in reserved_filings
        ]
        try:
            selected_quant = _select_quant(eligible_quant, quant_count, seed)
            selected_narrative = _select_narrative(
                eligible_narrative, narrative_count, seed
            )
        except ValueError:
            continue
        return (
            examples,
            selected_quant,
            selected_narrative,
            eligible_quant,
            eligible_narrative,
        )
    raise ValueError(
        "Could not reserve the v2.2 entity/filing/template-disjoint few-shot packs while retaining the requested evaluation counts"
    )


def _augment_v22_opinion_refusal_development_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add transparent development-only opinion-refusal variants.

    The cached corpus has opinion-language positives but no negative opinion
    template. These variants never enter evaluation membership; they support
    the fixed two-answer/two-refusal demonstration pack.
    """

    materialized = [dict(row) for row in rows]
    for row in rows:
        if (
            _narrative_subtype(row) != "auditor_report_opinion_language"
            or _narrative_status(row) != "OK"
        ):
            continue
        source_task_id = str(row["task_id"])
        variant = copy.deepcopy(dict(row))
        variant.update(
            {
                "task_id": f"{source_task_id}:v22-dev-refusal",
                "template_id": "v22-dev:auditor_report_opinion_language:refusal:1",
                "task_family": "narrative_citation:auditor_report_opinion_language:unanswerable",
                "question": "Quote the auditor's opinion guaranteeing that the issuer will increase revenue by 10 percent next year.",
                "retrieval_query": "auditor opinion guarantee 10 percent revenue increase next year",
                "answerability": "UNANSWERABLE",
                "expected_chunk_ids": [],
                "extractive_answer": None,
                "refusal_code": "NARRATIVE_NOT_SUPPORTED",
                "negative_type": "DEVELOPMENT_ONLY_UNSUPPORTED_OPINION_CLAIM",
                "development_only": True,
                "development_source_task_id": source_task_id,
            }
        )
        materialized.append(variant)
    return materialized


def _select_v22_development_examples(
    quant_rows: Sequence[Mapping[str, Any]],
    narrative_rows: Sequence[Mapping[str, Any]],
    selected_quant: Sequence[Mapping[str, Any]],
    selected_narrative: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    require_all_subtypes: bool,
) -> list[tuple[str, dict[str, Any]]]:
    """Select four quantitative and four examples per narrative subtype."""

    evaluation_entities = {
        _source_entity_id(row) for row in (*selected_quant, *selected_narrative)
    }
    evaluation_filings = {
        _filing_id(row) for row in (*selected_quant, *selected_narrative)
    }
    evaluation_task_ids = {
        str(row["task_id"]) for row in (*selected_quant, *selected_narrative)
    }

    def development_only(
        rows: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        return [
            row
            for row in rows
            if str(row["task_id"]) not in evaluation_task_ids
            and _source_entity_id(row) not in evaluation_entities
            and _filing_id(row) not in evaluation_filings
        ]

    quant_pool = development_only(quant_rows)
    narrative_pool = development_only(narrative_rows)
    selected: list[tuple[str, dict[str, Any]]] = []

    def choose(
        rows: Sequence[Mapping[str, Any]],
        *,
        task_type: str,
        status: str,
        namespace: str,
    ) -> list[dict[str, Any]]:
        ranked = sorted(
            rows,
            key=lambda row: (
                _stable_rank(seed, namespace, row),
                str(row["task_id"]),
            ),
        )
        if len(ranked) < 2:
            raise ValueError(
                f"Need two development-only {status} {namespace} examples; found {len(ranked)}"
            )
        return [dict(row) for row in ranked[:2]]

    for status in ("OK", "REFUSAL"):
        chosen = choose(
            [row for row in quant_pool if _quant_status(row) == status],
            task_type="quant",
            status=status,
            namespace=f"v22-demo-quant-{status}",
        )
        selected.extend(("quant", row) for row in chosen)

    available_subtypes = sorted({_narrative_subtype(row) for row in narrative_pool})
    subtypes = (
        sorted(NARRATIVE_SUBTYPES) if require_all_subtypes else available_subtypes
    )
    for subtype in subtypes:
        for status in ("OK", "REFUSAL"):
            chosen = choose(
                [
                    row
                    for row in narrative_pool
                    if _narrative_subtype(row) == subtype
                    and _narrative_status(row) == status
                ],
                task_type="narrative",
                status=status,
                namespace=f"v22-demo-narrative-{subtype}-{status}",
            )
            selected.extend(("narrative", row) for row in chosen)
    return selected


def _quant_gold(row: Mapping[str, Any]) -> dict[str, Any]:
    target = row.get("target_answer")
    if not isinstance(target, Mapping):
        raise TypeError(
            f"Quant task {row.get('task_id')!r} has no target_answer object"
        )
    gold = dict(target)
    gold.setdefault("status", _quant_status(row))
    return gold


def _narrative_gold(row: Mapping[str, Any]) -> dict[str, Any]:
    status = _narrative_status(row)
    return {
        "status": status,
        "answer_text": row.get("extractive_answer") if status == "OK" else None,
        "chunk_evidence_ids": list(row.get("expected_chunk_ids") or []),
        "refusal_code": row.get("refusal_code") if status == "REFUSAL" else None,
        "filing_id": _filing_id(row),
        "period_key": _period_key(row),
    }


def _quant_asof_period_bindings(row: Mapping[str, Any]) -> dict[str, str]:
    """Bind ASOF selectors to exact periods chosen by frozen materialization.

    Historical task rows can share a later filing whose report date is not the
    task's fiscal-period end, and 52/53-week issuers need not use the same
    month/day in consecutive years.  These exact bindings retain the source
    resolver's selected canonical-input periods without applying a permissive
    runtime window or letting evidence authorize itself.  The source task and
    its materialization manifest are content-addressed before benchmark build.
    Missing, malformed, or ambiguous source bindings yield no binding so the
    deterministic tool can return a typed refusal.
    """

    required_inputs = row.get("required_inputs")
    if required_inputs is None:
        return {}
    if not isinstance(required_inputs, (list, tuple)):
        raise TypeError(
            f"Quant task {row.get('task_id')!r} required_inputs must be an array"
        )

    selector_names: dict[str, set[str]] = {
        "current_period_end_asof": set(),
        "prior_year_period_end_asof": set(),
    }
    for index, item in enumerate(required_inputs):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"Quant task {row.get('task_id')!r} required_inputs[{index}] must be an object"
            )
        selector = item.get("period")
        if selector not in selector_names:
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Quant task {row.get('task_id')!r} has an unnamed ASOF input"
            )
        selector_names[str(selector)].add(name.strip())

    period_keys_by_selector: dict[str, set[str]] = {
        selector: set() for selector, names in selector_names.items() if names
    }
    invalid_selectors: set[str] = set()
    canonical_inputs = row.get("canonical_inputs")
    if canonical_inputs is None:
        return {}
    if not isinstance(canonical_inputs, (list, tuple)):
        raise TypeError(
            f"Quant task {row.get('task_id')!r} canonical_inputs must be an array"
        )
    for item in canonical_inputs:
        if not isinstance(item, Mapping):
            continue
        input_name = item.get("input_name")
        if not isinstance(input_name, str) or not input_name.strip():
            continue
        matching_selectors = [
            selector
            for selector, names in selector_names.items()
            if input_name.strip() in names
        ]
        if not matching_selectors:
            continue
        period_key = item.get("period_key")
        if not isinstance(period_key, str):
            invalid_selectors.update(matching_selectors)
            continue
        match = re.fullmatch(r"ASOF_(\d{4})(\d{2})(\d{2})", period_key)
        if match is None:
            invalid_selectors.update(matching_selectors)
            continue
        try:
            date(*(int(part) for part in match.groups()))
        except ValueError:
            invalid_selectors.update(matching_selectors)
            continue
        for selector in matching_selectors:
            period_keys_by_selector[selector].add(period_key)

    bindings = {
        selector: next(iter(period_keys))
        for selector, period_keys in sorted(period_keys_by_selector.items())
        if selector not in invalid_selectors and len(period_keys) == 1
    }
    if (
        "prior_year_period_end_asof" in bindings
        and "current_period_end_asof" not in bindings
    ):
        bindings.pop("prior_year_period_end_asof")
    return bindings


def _build_case(
    row: Mapping[str, Any],
    task_type: str,
    *,
    jurisdiction: str,
    corpus_id: str,
    source_system: str,
    reporting_framework: str,
    standards_version: str,
    evidence_items: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_jurisdiction = _resolved_profile_value(
        row, "jurisdiction", jurisdiction
    ).upper()
    resolved_source_system = _resolved_profile_value(
        row, "source_system", source_system
    )
    resolved_reporting_framework = _resolved_profile_value(
        row,
        "reporting_framework",
        reporting_framework,
        allow_mixed=True,
    )
    resolved_standards_version = _resolved_profile_value(
        row,
        "standards_version",
        standards_version,
        allow_mixed=True,
    )
    filing_id = _filing_id(row)
    filing_metadata = row.get("filing_metadata")
    metadata = filing_metadata if isinstance(filing_metadata, Mapping) else {}
    ticker = _first_text(row, "ticker") or _first_text(metadata, "ticker")
    entity_id = _entity_id(row)
    entity_name = (
        _first_text(row, "entity_name", "company_name", "issuer_name", "name")
        or ticker
        or entity_id
    )
    entity: dict[str, Any] = {"entity_id": entity_id, "name": entity_name}
    if ticker:
        entity["ticker"] = ticker

    filing: dict[str, Any] = {"filing_id": filing_id}
    form_type = _first_text(
        row, "form_type", "account_type", "report_type"
    ) or _first_text(metadata, "form_type", "account_type", "report_type")
    accession = _first_text(row, "accession", "accession_number") or _first_text(
        metadata, "accession", "accession_number"
    )
    if form_type:
        filing["form_type"] = form_type
    if accession:
        filing["accession"] = accession

    scoped_evidence_items = (
        list(evidence_items)
        if evidence_items is not None
        else _build_evidence_items(row, task_type, resolved_source_system)
    )
    quant_asof_bindings = (
        _quant_asof_period_bindings(row) if task_type == "quant_metric" else {}
    )

    source_period = row.get("period")
    period_source = source_period if isinstance(source_period, Mapping) else row
    period: dict[str, Any] = {"period_key": _period_key(row) or "UNKNOWN_PERIOD"}
    for key in ("start_date", "end_date", "instant"):
        value = _first_text(period_source, key)
        if key == "end_date" and value is None:
            value = _first_text(period_source, "period_end")
            if value is None and task_type == "quant_metric":
                current_asof = quant_asof_bindings.get("current_period_end_asof")
                if current_asof is not None:
                    value = date(
                        int(current_asof[5:9]),
                        int(current_asof[9:11]),
                        int(current_asof[11:13]),
                    ).isoformat()
            if value is None and task_type != "quant_metric":
                value = _first_text(metadata, "report_date", "period_end")
        if key == "start_date" and value is None:
            value = _first_text(period_source, "period_start")
        if value:
            period[key] = value
    if quant_asof_bindings:
        period["selector_periods"] = quant_asof_bindings
    scoped_evidence_ids = [str(item["evidence_id"]) for item in scoped_evidence_items]
    evidence_filing_ids = list(
        dict.fromkeys(
            [filing_id] + [str(item["filing_id"]) for item in scoped_evidence_items]
        )
    )
    trusted_source_entity_id = _source_entity_id(row)
    entity["source_entity_id"] = trusted_source_entity_id
    operation_spec = task_operation_spec(task_type)
    if task_type == "quant_metric":
        metric_spec_id = _first_text(row, "metric_spec_id")
        if metric_spec_id is None:
            raise ValueError(f"Quant task {row.get('task_id')!r} has no metric_spec_id")
        retrieval_query = None
        allowed_tools = [operation_spec.operation]
        output_schema_id = operation_spec.terminal_output_schema_id
        refusal_codes = _refusal_codes(row, task_type)
        narrative_subtype = None
    else:
        metric_spec_id = None
        retrieval_query = _first_text(row, "retrieval_query", "question")
        if retrieval_query is None:
            raise ValueError(
                f"Narrative task {row.get('task_id')!r} has no retrieval query"
            )
        allowed_tools = [operation_spec.operation]
        output_schema_id = operation_spec.terminal_output_schema_id
        refusal_codes = _refusal_codes(row, task_type)
        narrative_subtype = _narrative_subtype(row)

    result = build_agent_task_input(
        task_id=str(row["task_id"]),
        task_type=task_type,
        jurisdiction=resolved_jurisdiction,
        reporting_framework=resolved_reporting_framework,
        standards_profile={
            "name": "PCAOB public-filing profile"
            if resolved_jurisdiction == "US"
            else "ISA (UK) public-filing profile",
            "version": resolved_standards_version,
        },
        source_system=resolved_source_system,
        question=str(row.get("question") or retrieval_query or metric_spec_id),
        entity=entity,
        filing=filing,
        period=period,
        metric_spec_id=metric_spec_id,
        retrieval_query=retrieval_query,
        narrative_subtype=narrative_subtype,
        allowed_tools=allowed_tools,
        evidence_ids=scoped_evidence_ids,
        evidence_filing_ids=evidence_filing_ids,
        output_schema_id=output_schema_id,
        refusal_codes=refusal_codes,
        evidence_max_items=5,
    )
    validate_agent_task_input(result)
    return result


def _evidence_content(item: Mapping[str, Any], *, fallback_prefix: str) -> str:
    direct = _first_text(
        item, "content", "retrieval_text", "text", "supporting_text", "value_text"
    )
    if direct:
        return direct
    parts = []
    for key in (
        "input_name",
        "concept_norm",
        "period_key",
        "unit_canon",
        "unit",
        "value_num_exact",
        "value",
    ):
        value = item.get(key)
        if value is not None and str(value).strip():
            parts.append(f"{key}={value}")
    return f"{fallback_prefix}: " + (
        ", ".join(parts) if parts else "source metadata only"
    )


def _normalize_evidence_item(
    item: Mapping[str, Any],
    *,
    filing_id: str,
    source_system: str,
    fallback_prefix: str,
) -> dict[str, Any] | None:
    evidence_id = _first_text(
        item, "evidence_id", "chunk_evidence_id", "fact_evidence_id"
    )
    if evidence_id is None:
        return None
    item_filing_id = _first_text(item, "filing_id") or filing_id
    normalized: dict[str, Any] = {
        "evidence_id": evidence_id,
        "filing_id": item_filing_id,
        "content": _evidence_content(item, fallback_prefix=fallback_prefix),
        "source_system": _first_text(item, "source_system") or source_system,
    }
    period_key = _first_text(item, "period_key")
    unit = _first_text(item, "unit", "unit_canon")
    value = _first_text(item, "value", "value_num_exact", "value_text")
    if period_key:
        normalized["period_key"] = period_key
    if unit:
        normalized["unit"] = unit
    if value is not None:
        normalized["value"] = value
    metadata = item.get("metadata")
    clean_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    for key in (
        "input_name",
        "concept_norm",
        "derived",
        "heading",
        "heading_path",
        "source_file",
        "context_id",
        "entity_id",
        "entity_identifier",
        "dimensions",
        "scenario",
    ):
        if item.get(key) is not None:
            normalized_key = "entity_id" if key == "entity_identifier" else key
            clean_metadata[normalized_key] = item[key]
    if clean_metadata:
        normalized["metadata"] = clean_metadata
    canonical_content = canonical_quant_evidence_content(normalized)
    if canonical_content is not None:
        normalized["content"] = canonical_content
    return normalized


def _build_evidence_items(
    row: Mapping[str, Any], task_type: str, source_system: str
) -> list[dict[str, Any]]:
    """Package at most five complete inference-visible items in stable source order."""

    filing_id = _filing_id(row)
    candidates: list[dict[str, Any]] = []

    direct_items = row.get("evidence_items")
    if isinstance(direct_items, Sequence) and not isinstance(
        direct_items, (str, bytes, bytearray)
    ):
        for item in direct_items:
            if isinstance(item, Mapping):
                normalized = _normalize_evidence_item(
                    item,
                    filing_id=filing_id,
                    source_system=source_system,
                    fallback_prefix="Evidence",
                )
                if normalized is not None:
                    candidates.append(normalized)

    retrieval_results = row.get("retrieval_results")
    if isinstance(retrieval_results, Sequence) and not isinstance(
        retrieval_results, (str, bytes, bytearray)
    ):
        for item in retrieval_results:
            if isinstance(item, Mapping):
                normalized = _normalize_evidence_item(
                    item,
                    filing_id=filing_id,
                    source_system=source_system,
                    fallback_prefix="Retrieved filing evidence",
                )
                if normalized is not None:
                    candidates.append(normalized)

    if task_type == "quant_metric":
        canonical_inputs = row.get("canonical_inputs")
        if isinstance(canonical_inputs, Sequence) and not isinstance(
            canonical_inputs, (str, bytes, bytearray)
        ):
            for canonical_input in canonical_inputs:
                if not isinstance(canonical_input, Mapping):
                    continue
                evidence_ids = canonical_input.get("fact_evidence_ids")
                if isinstance(evidence_ids, str):
                    evidence_ids = [evidence_ids]
                if not isinstance(evidence_ids, Sequence):
                    continue
                for evidence_id in evidence_ids:
                    materialized = {**canonical_input, "evidence_id": evidence_id}
                    normalized = _normalize_evidence_item(
                        materialized,
                        filing_id=filing_id,
                        source_system=source_system,
                        fallback_prefix="Canonical filing fact",
                    )
                    if normalized is not None:
                        candidates.append(normalized)

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        evidence_id = item["evidence_id"]
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        item["rank"] = len(deduped) + 1
        deduped.append(item)
        if len(deduped) == 5:
            break
    return deduped


def _refusal_codes(row: Mapping[str, Any], task_type: str) -> list[str]:
    permitted = task_operation_spec(task_type).permitted_refusal_codes
    policy = row.get("refusal_policy")
    if isinstance(policy, Mapping):
        codes = policy.get("allowed_codes")
        if isinstance(codes, Sequence) and not isinstance(
            codes, (str, bytes, bytearray)
        ):
            materialized = sorted({str(code) for code in codes if str(code).strip()})
            if materialized:
                selected = set(materialized) & permitted
                selected.add("PROMPT_INJECTION_DETECTED")
                return sorted(selected)
    return sorted(permitted)


def _few_shot_assistant_response(
    gold_record: Mapping[str, Any],
    task_input: Mapping[str, Any],
    tool_observation: Mapping[str, Any],
) -> dict[str, Any]:
    target = gold_record["target"]
    task_id = str(gold_record["task_id"])
    if task_input.get("task_type") == "quant_metric":
        status = str(tool_observation["status"])
        evidence_ids = list(tool_observation.get("evidence_ids") or [])
        value = tool_observation.get("value")
        unit = tool_observation.get("unit")
        refusal_code = tool_observation.get("refusal_code")
        answer_text = None
    else:
        status = str(target["status"])
        evidence_ids = list(
            target.get("evidence_ids") or target.get("chunk_evidence_ids") or []
        )
        value = None
        unit = None
        refusal_code = target.get("refusal_code")
        answer_text = target.get("answer_text") if status == "OK" else None
        if answer_text is not None:
            supporting_ids = [
                str(item["evidence_id"])
                for item in tool_observation.get("evidence_items", [])
                if is_contiguous_text_supported(
                    str(answer_text), str(item.get("content") or "")
                )
            ]
            if not supporting_ids:
                raise ValueError(
                    f"Few-shot narrative task {task_id!r} has no exact supporting extract"
                )
            evidence_ids = supporting_ids[:1]
    if status == "REFUSAL":
        evidence_ids = []
    claims = []
    if answer_text is not None:
        claims = [
            {
                "claim_id": "claim-1",
                "text": str(answer_text),
                "evidence_ids": evidence_ids,
                "supporting_text": str(answer_text),
            }
        ]
    response = {
        "agent_proposal_version": "v2",
        "task_id": task_id,
        "action": "ANSWER" if status == "OK" else "REFUSE",
        "tool_name": None,
        "tool_arguments": None,
        "status": status,
        "value": (
            canonical_decimal_string(value)
            if status == "OK" and value is not None
            else None
        ),
        "unit": unit if status == "OK" else None,
        "period_key": task_input["period"]["period_key"],
        "answer_text": answer_text,
        "evidence_ids": evidence_ids,
        "claims": claims,
        "refusal_code": refusal_code if status == "REFUSAL" else None,
        "model_uncertainty": "NONE" if status == "OK" else "HIGH",
        "model_escalation_requested": False,
    }
    validate_agent_proposal(response)
    return response


def _compact_narrative_example_evidence(
    evidence_items: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Freeze minimal source excerpts for demonstrations before token preflight."""

    status = str(target.get("status") or "")
    answer_text = target.get("answer_text")
    if status == "OK" and (
        not isinstance(answer_text, str)
        or not answer_text.strip()
        or len(answer_text) > FEW_SHOT_NARRATIVE_EXCERPT_MAX_CHARS
    ):
        raise ValueError("Answerable narrative demonstration lacks a compact answer")
    compact: list[dict[str, Any]] = []
    for rank, item in enumerate(evidence_items, start=1):
        content = str(item["content"])
        if status == "OK":
            content = str(answer_text)
        else:
            content = content[:FEW_SHOT_NARRATIVE_EXCERPT_MAX_CHARS].rstrip()
        if not content:
            raise ValueError("Narrative demonstration excerpt cannot be empty")
        projected = {
            "evidence_id": str(item["evidence_id"]),
            "filing_id": str(item["filing_id"]),
            "content": content,
            "rank": rank,
        }
        for key in ("period_key", "source_system"):
            if item.get(key) is not None:
                projected[key] = item[key]
        compact.append(projected)
    return compact


def _build_gold(
    row: Mapping[str, Any],
    task_type: str,
    *,
    jurisdiction: str,
    corpus_id: str,
) -> dict[str, Any]:
    return {
        "gold_record_version": GOLD_RECORD_VERSION,
        "schema_id": "agent_gold_record.v2",
        "task_id": str(row["task_id"]),
        "task_type": task_type,
        "template_id": _template_id(row, task_type),
        "task_family": _task_family(row, task_type),
        "strata": _stratification_axes(
            row, "quant" if task_type == "quant_metric" else "narrative"
        ),
        "jurisdiction": jurisdiction,
        "corpus_id": corpus_id,
        "source_task_sha256": canonical_json_sha256(row),
        "target": _quant_gold(row)
        if task_type == "quant_metric"
        else _narrative_gold(row),
    }


def _build_verifier_observation(
    row: Mapping[str, Any],
    task_type: str,
    evidence_items: Sequence[Mapping[str, Any]],
    *,
    task_input: Mapping[str, Any],
) -> dict[str, Any]:
    task_id = str(row["task_id"])
    operation = task_operation_spec(task_type).operation
    if task_type == "quant_metric":
        target = _quant_gold(row)
        observation = evaluate_quant_evidence(task_input, evidence_items)
        if not _quant_target_matches_observation(target, observation):
            raise ValueError(
                f"Quant task {task_id!r} gold target does not match deterministic evaluation of frozen evidence"
            )
        return {
            "task_id": task_id,
            "tool_name": operation,
            "observation": observation,
        }

    return {
        "task_id": task_id,
        "tool_name": operation,
        "observation": load_frozen_evidence(task_input, evidence_items),
    }


def _normalized_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        materialized = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return materialized if materialized.is_finite() else None


def _normalized_unit(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().casefold()
    for prefix in ("iso4217:", "iso4217_"):
        normalized = normalized.removeprefix(prefix)
    return normalized


def _quant_target_matches_observation(
    target: Mapping[str, Any], observation: Mapping[str, Any]
) -> bool:
    if target.get("status") != observation.get("status"):
        return False
    for key in ("filing_id", "metric_spec_id", "period_key"):
        if target.get(key) is not None and target.get(key) != observation.get(key):
            return False
    if {str(item) for item in target.get("evidence_ids") or []} != {
        str(item) for item in observation.get("evidence_ids") or []
    }:
        return False
    if target.get("status") == "OK":
        return (
            _normalized_decimal(target.get("value"))
            == _normalized_decimal(observation.get("value"))
            and _normalized_unit(target.get("unit"))
            == _normalized_unit(observation.get("unit"))
            and observation.get("refusal_code") is None
        )
    return (
        target.get("refusal_code") == observation.get("refusal_code")
        and observation.get("value") is None
        and observation.get("unit") is None
    )


def _validate_narrative_target_support(
    row: Mapping[str, Any], evidence_items: Sequence[Mapping[str, Any]]
) -> None:
    if _narrative_status(row) != "OK":
        return
    answer = row.get("extractive_answer")
    expected_ids = [str(item) for item in row.get("expected_chunk_ids") or []]
    if not isinstance(answer, str) or not answer or not expected_ids:
        raise ValueError(
            f"Narrative task {row.get('task_id')!r} lacks an extractive answer or citation"
        )
    evidence_by_id = {str(item["evidence_id"]): item for item in evidence_items}
    if any(evidence_id not in evidence_by_id for evidence_id in expected_ids):
        raise ValueError(
            f"Narrative task {row.get('task_id')!r} cites evidence outside the frozen context"
        )
    if not any(
        is_contiguous_text_supported(
            answer, str(evidence_by_id[evidence_id].get("content") or "")
        )
        for evidence_id in expected_ids
    ):
        raise ValueError(
            f"Narrative task {row.get('task_id')!r} extractive answer is absent from cited evidence"
        )


def _artifact_metadata(path: Path, records: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sha256": _sha256_path(path),
        "bytes": path.stat().st_size,
    }
    if records is not None:
        result["records"] = records
    return result


def _source_metadata(path: Path, records: int) -> dict[str, Any]:
    return {
        "file_name": path.name,
        "sha256": _sha256_path(path),
        "bytes": path.stat().st_size,
        "records": records,
    }


def _required_materialization_manifest(
    quant_path: Path,
    narrative_path: Path,
    benchmark_profile: str,
    *,
    quant_records: int,
    narrative_records: int,
) -> Path | None:
    if benchmark_profile not in _MATERIALIZED_BENCHMARK_PROFILES:
        return None
    quant_parent = quant_path.parent.resolve()
    narrative_parent = narrative_path.parent.resolve()
    if quant_parent != narrative_parent:
        raise ValueError(
            f"{benchmark_profile} requires both enriched inputs to share "
            "one materialization directory"
        )
    manifest_path = quant_parent / MATERIALIZATION_MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(
            f"{benchmark_profile} requires "
            f"{MATERIALIZATION_MANIFEST_NAME} beside both enriched inputs"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{MATERIALIZATION_MANIFEST_NAME} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise TypeError(f"{MATERIALIZATION_MANIFEST_NAME} must contain an object")
    if manifest.get("materialization_version") != "agent_evidence.v1":
        raise ValueError(
            f"{MATERIALIZATION_MANIFEST_NAME} must declare "
            "materialization_version='agent_evidence.v1'"
        )
    artifacts = _as_mapping(manifest.get("artifacts"))
    expected_artifacts = {
        "enriched_quant.jsonl": (quant_path, quant_records),
        "enriched_narrative.jsonl": (narrative_path, narrative_records),
    }
    for name, (path, records) in expected_artifacts.items():
        declared = _as_mapping(artifacts.get(name))
        expected = _source_metadata(path, records)
        mismatches = [
            key
            for key in ("sha256", "bytes", "records")
            if declared.get(key) != expected.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"{MATERIALIZATION_MANIFEST_NAME} {name} binding mismatch: "
                + ", ".join(mismatches)
            )
    return manifest_path


def _build_parity_slice(
    cases: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    target_count = min(PARITY_SLICE_COUNT, len(cases))
    quant_target = min(
        sum(case.get("task_type") == "quant_metric" for case in cases),
        round(target_count * 0.6),
    )
    narrative_target = target_count - quant_target
    by_id = {str(case["task_id"]): case for case in cases}
    gold_by_id = {str(row["task_id"]): row for row in gold}
    selected_ids: list[str] = []
    breakdown: dict[str, int] = {}
    for task_type, type_target in (
        ("quant_metric", quant_target),
        ("narrative_citation", narrative_target),
    ):
        status_targets = {
            "OK": type_target // 2 + type_target % 2,
            "REFUSAL": type_target // 2,
        }
        for status, count in status_targets.items():
            candidates = [
                {
                    "task_id": task_id,
                    "form_type": _as_mapping(case.get("filing")).get("form_type"),
                    "period_key": _as_mapping(case.get("period")).get("period_key"),
                    "operation": _as_mapping(case.get("task_parameters")).get(
                        "metric_spec_id"
                    ),
                    "entity_id": _as_mapping(case.get("entity")).get("entity_id"),
                }
                for task_id, case in by_id.items()
                if case.get("task_type") == task_type
                and _as_mapping(gold_by_id.get(task_id)).get("target", {}).get("status")
                == status
            ]
            chosen = _balanced_sample(
                candidates,
                count,
                seed=seed,
                namespace=f"parity-50-{task_type}-{status}",
                strata=lambda row: (
                    row.get("operation"),
                    row.get("form_type"),
                    row.get("period_key"),
                    row.get("entity_id"),
                ),
            )
            selected_ids.extend(str(row["task_id"]) for row in chosen)
            breakdown[f"{task_type}:{status}"] = len(chosen)
    selected_ids.sort()
    return {
        "parity_slice_version": PARITY_SLICE_VERSION,
        "requested_count": PARITY_SLICE_COUNT,
        "case_count": len(selected_ids),
        "seed": seed,
        "task_ids": selected_ids,
        "task_ids_sha256": canonical_json_sha256(selected_ids),
        "breakdown": dict(sorted(breakdown.items())),
    }


def _load_parent_benchmark_membership(
    parent_dir: Path,
    *,
    quant_count: int,
    narrative_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify and load frozen evaluation membership for a derived v2.2 build."""

    manifest_path = parent_dir / "benchmark_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load parent benchmark manifest: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise TypeError("Parent benchmark manifest must contain an object")
    if manifest.get("benchmark_version") not in {
        "agent_benchmark.v2",
        "agent_benchmark.v2.1",
    }:
        raise ValueError(
            "Derived v2.2 membership requires an agent_benchmark.v2 or v2.1 parent"
        )
    material = {key: value for key, value in manifest.items() if key != "benchmark_id"}
    if manifest.get("benchmark_id") != canonical_json_sha256(material):
        raise ValueError("Parent benchmark_id does not bind its canonical manifest")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TypeError("Parent benchmark artifact registry is missing")
    bound_artifacts: dict[str, dict[str, Any]] = {}
    required_artifacts = ["cases.jsonl"]
    if "case_lineage.jsonl" in artifacts:
        required_artifacts.append("case_lineage.jsonl")
    for name in required_artifacts:
        expected = artifacts.get(name)
        path = parent_dir / name
        if not isinstance(expected, Mapping) or not path.is_file():
            raise ValueError(f"Parent benchmark is missing bound artifact {name}")
        if _sha256_path(path) != expected.get("sha256"):
            raise ValueError(f"Parent benchmark artifact hash mismatch: {name}")
        if path.stat().st_size != expected.get("bytes"):
            raise ValueError(f"Parent benchmark artifact byte mismatch: {name}")
        bound_artifacts[name] = {
            "sha256": expected["sha256"],
            "bytes": expected["bytes"],
        }

    cases = _read_jsonl(parent_dir / "cases.jsonl")
    membership: dict[str, Any] = {
        "cases": {"quant_metric": [], "narrative_citation": []},
    }
    source_task_id_by_case: dict[str, str] = {}
    if (parent_dir / "case_lineage.jsonl").is_file():
        source_task_id_by_case = {
            str(row["task_id"]): str(row["parent_task_id"])
            for row in _read_jsonl(parent_dir / "case_lineage.jsonl")
        }
    case_order: list[str] = []
    for row in cases:
        task_type = str(row.get("task_type") or "")
        task_id = str(row.get("task_id") or "")
        if task_type not in membership["cases"] or not task_id:
            raise ValueError("Parent benchmark contains an invalid text-agent case")
        source_task_id = source_task_id_by_case.get(task_id, task_id)
        membership["cases"][task_type].append(source_task_id)
        case_order.append(source_task_id)

    expected_counts = {
        "cases": {
            "quant_metric": quant_count,
            "narrative_citation": narrative_count,
        },
    }
    for group, task_types in expected_counts.items():
        for task_type, expected_count in task_types.items():
            task_ids = membership[group][task_type]
            if len(task_ids) != expected_count:
                raise ValueError(
                    f"Parent benchmark {group} {task_type} count is "
                    f"{len(task_ids)}, expected {expected_count}"
                )
            if len(task_ids) != len(set(task_ids)):
                raise ValueError(
                    f"Parent benchmark {group} {task_type} task IDs are not unique"
                )
    binding = {
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_version": manifest["benchmark_version"],
        "benchmark_manifest_sha256": _sha256_path(manifest_path),
        "artifacts": bound_artifacts,
    }
    membership["case_order"] = case_order
    return membership, binding


def build_agent_benchmark(
    quant_jsonl: str | Path,
    narrative_jsonl: str | Path,
    output_dir: str | Path,
    *,
    benchmark_profile: str,
    jurisdiction: str,
    corpus_id: str,
    reporting_framework: str,
    standards_version: str,
    source_system: str,
    seed: int = DEFAULT_SEED,
    quant_count: int = DEFAULT_QUANT_COUNT,
    narrative_count: int = DEFAULT_NARRATIVE_COUNT,
    parent_benchmark_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build an immutable, deterministic agent benchmark from existing JSONL.

    The destination must not already exist. Four quantitative examples and four
    examples per available narrative subtype are reserved before evaluation
    selection. Their entities, filings, and templates cannot occur in cases.
    """

    quant_path = Path(quant_jsonl)
    narrative_path = Path(narrative_jsonl)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite benchmark directory: {destination}"
        )
    if (
        not jurisdiction.strip()
        or not corpus_id.strip()
        or not reporting_framework.strip()
        or not standards_version.strip()
    ):
        raise ValueError(
            "jurisdiction, corpus_id, reporting_framework, and standards_version must be non-empty"
        )
    if not source_system.strip():
        raise ValueError("source_system must be non-empty")
    if quant_count <= 0 or narrative_count <= 0:
        raise ValueError("quant_count and narrative_count must be positive")
    assert_no_secrets_in_value(
        {
            "benchmark_profile": benchmark_profile,
            "jurisdiction": jurisdiction,
            "corpus_id": corpus_id,
            "reporting_framework": reporting_framework,
            "standards_version": standards_version,
            "source_system": source_system,
        },
        label="benchmark configuration",
    )
    profile_errors = _benchmark_profile_errors(
        benchmark_profile,
        jurisdiction=jurisdiction.upper(),
        source_system=source_system,
        seed=seed,
        quant_count=quant_count,
        narrative_count=narrative_count,
        narrative_answerable=narrative_count // 2,
        narrative_unanswerable=narrative_count // 2,
        total_count=quant_count + narrative_count,
    )
    if profile_errors:
        raise ValueError(
            "Invalid benchmark profile configuration: " + "; ".join(profile_errors)
        )
    raw_quant_rows = _read_jsonl(quant_path)
    raw_narrative_source_rows = _read_jsonl(narrative_path)
    raw_narrative_rows = _augment_v22_opinion_refusal_development_rows(
        raw_narrative_source_rows
    )

    def version_rows(
        rows: Sequence[Mapping[str, Any]], task_type: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        materialized: list[dict[str, Any]] = []
        lineage: list[dict[str, Any]] = []
        for row in rows:
            parent_task_id = str(row.get("task_id") or "")
            if not parent_task_id:
                raise ValueError("Source benchmark row lacks task_id")
            source_sha256 = canonical_json_sha256(row)
            task_id = (
                "v22-"
                + canonical_json_sha256(
                    {
                        "benchmark_version": BENCHMARK_VERSION,
                        "task_type": task_type,
                        "parent_task_id": parent_task_id,
                        "source_sha256": source_sha256,
                    }
                )[:24]
            )
            updated = dict(row)
            updated["task_id"] = task_id
            materialized.append(updated)
            lineage.append(
                {
                    "task_id": task_id,
                    "parent_task_id": parent_task_id,
                    "source_task_sha256": source_sha256,
                }
            )
        return materialized, lineage

    quant_rows, quant_lineage = version_rows(raw_quant_rows, "quant_metric")
    narrative_rows, narrative_lineage = version_rows(
        raw_narrative_rows, "narrative_citation"
    )
    case_lineage = quant_lineage + narrative_lineage
    lineage_by_id = {row["task_id"]: row for row in case_lineage}
    materialization_manifest_path = _required_materialization_manifest(
        quant_path,
        narrative_path,
        benchmark_profile,
        quant_records=len(quant_rows),
        narrative_records=len(raw_narrative_source_rows),
    )
    duplicate_ids = sorted(
        {str(row["task_id"]) for row in quant_rows}
        & {str(row["task_id"]) for row in narrative_rows}
    )
    if duplicate_ids:
        raise ValueError(
            f"Task IDs must be unique across task types; duplicate: {duplicate_ids[0]}"
        )

    parent_binding: dict[str, Any] | None = None
    parent_case_position: dict[str, int] | None = None
    parent_example_position: dict[str, int] | None = None
    if parent_benchmark_dir is None:
        (
            examples,
            selected_quant,
            selected_narrative,
            eligible_quant,
            eligible_narrative,
        ) = _select_few_shots_and_evaluation(
            quant_rows,
            narrative_rows,
            quant_count,
            narrative_count,
            seed,
            require_all_subtypes=(benchmark_profile != BENCHMARK_PROFILE_SYNTHETIC),
        )
    else:
        parent_membership, parent_binding = _load_parent_benchmark_membership(
            Path(parent_benchmark_dir),
            quant_count=quant_count,
            narrative_count=narrative_count,
        )
        parent_case_position = {
            task_id: index
            for index, task_id in enumerate(parent_membership["case_order"])
        }
        quant_by_parent = {
            lineage["parent_task_id"]: row
            for row, lineage in zip(quant_rows, quant_lineage, strict=True)
        }
        narrative_by_parent = {
            lineage["parent_task_id"]: row
            for row, lineage in zip(narrative_rows, narrative_lineage, strict=True)
        }

        def resolve_rows(
            task_ids: Sequence[str],
            rows_by_parent: Mapping[str, dict[str, Any]],
            *,
            label: str,
        ) -> list[dict[str, Any]]:
            missing = [task_id for task_id in task_ids if task_id not in rows_by_parent]
            if missing:
                raise ValueError(
                    f"Parent benchmark {label} task is absent from frozen source: {missing[0]}"
                )
            return [rows_by_parent[task_id] for task_id in task_ids]

        selected_quant = resolve_rows(
            parent_membership["cases"]["quant_metric"],
            quant_by_parent,
            label="quantitative evaluation",
        )
        selected_narrative = resolve_rows(
            parent_membership["cases"]["narrative_citation"],
            narrative_by_parent,
            label="narrative evaluation",
        )
        examples = _select_v22_development_examples(
            quant_rows,
            narrative_rows,
            selected_quant,
            selected_narrative,
            seed=seed,
            require_all_subtypes=(benchmark_profile != BENCHMARK_PROFILE_SYNTHETIC),
        )
        reserved_entities = {_source_entity_id(row) for _, row in examples}
        reserved_filings = {_filing_id(row) for _, row in examples}
        eligible_quant = [
            row
            for row in quant_rows
            if not row.get("development_only")
            and _source_entity_id(row) not in reserved_entities
            and _filing_id(row) not in reserved_filings
        ]
        eligible_narrative = [
            row
            for row in narrative_rows
            if not row.get("development_only")
            and _source_entity_id(row) not in reserved_entities
            and _filing_id(row) not in reserved_filings
        ]
    narrative_status_counts = Counter(
        _narrative_status(row) for row in selected_narrative
    )
    profile_errors = _benchmark_profile_errors(
        benchmark_profile,
        jurisdiction=jurisdiction.upper(),
        source_system=source_system,
        seed=seed,
        quant_count=len(selected_quant),
        narrative_count=len(selected_narrative),
        narrative_answerable=narrative_status_counts["OK"],
        narrative_unanswerable=narrative_status_counts["REFUSAL"],
        total_count=len(selected_quant) + len(selected_narrative),
    )
    if profile_errors:
        raise ValueError(
            "Selected tasks violate benchmark profile: " + "; ".join(profile_errors)
        )
    stratification = {
        "stratification_version": "auditops-agent-stratification.v1",
        "quant": {
            "eligible": _stratification_profile(eligible_quant, "quant"),
            "selected": _stratification_profile(selected_quant, "quant"),
        },
        "narrative": {
            "eligible": _stratification_profile(eligible_narrative, "narrative"),
            "selected": _stratification_profile(selected_narrative, "narrative"),
        },
    }
    for task_type, selected_count in (
        ("quant", quant_count),
        ("narrative", narrative_count),
    ):
        section = stratification[task_type]
        section["requirements"] = _stratification_requirements(
            section["eligible"],
            task_type=task_type,
            selected_count=selected_count,
            benchmark_profile=benchmark_profile,
        )
    coverage_errors = _stratification_errors(
        stratification,
        benchmark_profile=benchmark_profile,
        quant_count=quant_count,
        narrative_count=narrative_count,
    )
    if coverage_errors:
        raise ValueError(
            "Benchmark stratification requirements were not met: "
            + "; ".join(coverage_errors)
        )
    typed_rows = [("quant_metric", row) for row in selected_quant] + [
        ("narrative_citation", row) for row in selected_narrative
    ]
    if parent_case_position is None:
        typed_rows.sort(key=lambda item: str(item[1]["task_id"]))
    else:
        typed_rows.sort(
            key=lambda item: parent_case_position[
                lineage_by_id[str(item[1]["task_id"])]["parent_task_id"]
            ]
        )

    cases: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    verifier_observations: list[dict[str, Any]] = []
    for task_type, row in typed_rows:
        resolved_source_system = _resolved_profile_value(
            row, "source_system", source_system
        )
        evidence_items = _build_evidence_items(row, task_type, resolved_source_system)
        case = _build_case(
            row,
            task_type,
            jurisdiction=jurisdiction.upper(),
            corpus_id=corpus_id,
            source_system=source_system,
            reporting_framework=reporting_framework,
            standards_version=standards_version,
            evidence_items=evidence_items,
        )
        build_context_pack(case, evidence_items, token_counter=lambda _: 0)
        if task_type == "narrative_citation":
            _validate_narrative_target_support(row, evidence_items)
        cases.append(case)
        evidence_records.append(
            {"task_id": str(row["task_id"]), "items": evidence_items}
        )
        gold.append(
            _build_gold(
                row,
                task_type,
                jurisdiction=jurisdiction.upper(),
                corpus_id=corpus_id,
            )
        )
        verifier_observations.append(
            _build_verifier_observation(
                row,
                task_type,
                evidence_items,
                task_input=case,
            )
        )
    few_shot = []
    if parent_example_position is None:
        ordered_examples = sorted(
            examples, key=lambda item: (item[0], str(item[1]["task_id"]))
        )
    else:
        ordered_examples = sorted(
            examples,
            key=lambda item: parent_example_position[
                lineage_by_id[str(item[1]["task_id"])]["parent_task_id"]
            ],
        )
    for task_type, row in ordered_examples:
        normalized_type = (
            "quant_metric" if task_type == "quant" else "narrative_citation"
        )
        resolved_source_system = _resolved_profile_value(
            row, "source_system", source_system
        )
        full_example_evidence = _build_evidence_items(
            row, normalized_type, resolved_source_system
        )
        gold_record = _build_gold(
            row,
            normalized_type,
            jurisdiction=jurisdiction.upper(),
            corpus_id=corpus_id,
        )
        target = gold_record["target"]
        if normalized_type == "narrative_citation":
            cited = set(
                target.get("evidence_ids") or target.get("chunk_evidence_ids") or []
            )
            if len(cited) > 2:
                raise ValueError(
                    f"Few-shot narrative task {row.get('task_id')!r} requires more than two excerpts"
                )
            example_evidence = (
                [item for item in full_example_evidence if item["evidence_id"] in cited]
                if cited
                else full_example_evidence[:2]
            )
            _validate_narrative_target_support(row, full_example_evidence)
            example_evidence = _compact_narrative_example_evidence(
                example_evidence, target
            )
        else:
            cited = set(
                target.get("evidence_ids") or target.get("chunk_evidence_ids") or []
            )
            example_evidence = (
                [item for item in full_example_evidence if item["evidence_id"] in cited]
                if cited
                else full_example_evidence[:2]
            )
        for rank, item in enumerate(example_evidence, start=1):
            item["rank"] = rank
        case = _build_case(
            row,
            normalized_type,
            jurisdiction=jurisdiction.upper(),
            corpus_id=corpus_id,
            source_system=source_system,
            reporting_framework=reporting_framework,
            standards_version=standards_version,
            evidence_items=example_evidence,
        )
        build_context_pack(case, example_evidence, token_counter=lambda _: 0)
        if normalized_type == "narrative_citation":
            _validate_narrative_target_support(row, example_evidence)
        elif not _quant_target_matches_observation(
            _quant_gold(row), evaluate_quant_evidence(case, example_evidence)
        ):
            # Some refusal demonstrations require the complete negative fact scope.
            example_evidence = full_example_evidence
            case = _build_case(
                row,
                normalized_type,
                jurisdiction=jurisdiction.upper(),
                corpus_id=corpus_id,
                source_system=source_system,
                reporting_framework=reporting_framework,
                standards_version=standards_version,
                evidence_items=example_evidence,
            )
            if not _quant_target_matches_observation(
                _quant_gold(row), evaluate_quant_evidence(case, example_evidence)
            ):
                raise ValueError(
                    f"Few-shot quant task {row.get('task_id')!r} target does not match frozen evidence"
                )
        operation = task_operation_spec(normalized_type).operation
        tool_observation = (
            evaluate_quant_evidence(case, example_evidence, visibility="MODEL_VISIBLE")
            if normalized_type == "quant_metric"
            else load_frozen_evidence(
                case, example_evidence, visibility="MODEL_VISIBLE"
            )
        )
        assistant_response = _few_shot_assistant_response(
            gold_record, case, tool_observation
        )
        few_shot.append(
            {
                "few_shot_example_version": FEW_SHOT_EXAMPLE_VERSION,
                "task_id": str(row["task_id"]),
                "example_id": _sha256_bytes(
                    f"{corpus_id}|{jurisdiction.upper()}|{normalized_type}|{row['task_id']}".encode()
                )[:24],
                "review_status": "PENDING_HUMAN_REVIEW",
                "template_id": (
                    "v22-development:"
                    + (
                        str(case["task_parameters"]["metric_spec_id"])
                        if normalized_type == "quant_metric"
                        else str(case["narrative_subtype"])
                    )
                    + ":"
                    + str(target["status"]).casefold()
                    + ":"
                    + canonical_json_sha256(
                        {
                            "source_template_id": _template_id(row, task_type),
                            "task_id": row["task_id"],
                        }
                    )[:12]
                ),
                "task_family": _task_family(row, task_type),
                "task": case,
                "evidence_items": example_evidence,
                "plan_response": {
                    "task_id": case["task_id"],
                    "action": "CALL_TOOL",
                    "tool_name": operation,
                    "tool_arguments": expected_tool_arguments(case),
                },
                "tool_observation": tool_observation,
                "assistant_response": assistant_response,
                "source_task_sha256": lineage_by_id[str(row["task_id"])][
                    "source_task_sha256"
                ],
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        case_count = _write_jsonl(temporary / "cases.jsonl", cases)
        evidence_count = _write_jsonl(temporary / "evidence.jsonl", evidence_records)
        gold_count = _write_jsonl(temporary / "gold.jsonl", gold)
        observation_count = _write_jsonl(
            temporary / "verifier_observations.jsonl", verifier_observations
        )
        few_shot_count = _write_jsonl(temporary / "few_shot.jsonl", few_shot)
        selected_lineage = [lineage_by_id[case["task_id"]] for case in cases]
        lineage_count = _write_jsonl(temporary / "case_lineage.jsonl", selected_lineage)
        parity_slice = _build_parity_slice(cases, gold, seed=seed)
        _write_json(temporary / "parity_50.json", parity_slice)
        materialization_source = None
        if materialization_manifest_path is not None:
            materialization_source = {
                "file_name": MATERIALIZATION_MANIFEST_NAME,
                **_artifact_metadata(materialization_manifest_path),
            }
            materialization_copy = temporary / MATERIALIZATION_MANIFEST_NAME
            shutil.copyfile(materialization_manifest_path, materialization_copy)
            copied_metadata = _artifact_metadata(materialization_copy)
            if copied_metadata != {
                key: materialization_source[key] for key in ("sha256", "bytes")
            }:
                raise ValueError(
                    f"{MATERIALIZATION_MANIFEST_NAME} changed while the benchmark was being built"
                )
        source_manifest = {
            "source_manifest_version": "v1",
            "benchmark_profile": benchmark_profile,
            "corpus_id": corpus_id,
            "jurisdiction": jurisdiction.upper(),
            "sources": {
                "quant": _source_metadata(quant_path, len(quant_rows)),
                "narrative": _source_metadata(
                    narrative_path, len(raw_narrative_source_rows)
                ),
                **(
                    {"materialization_manifest": materialization_source}
                    if materialization_source is not None
                    else {}
                ),
                **(
                    {"parent_benchmark": parent_binding}
                    if parent_binding is not None
                    else {}
                ),
            },
        }
        _write_json(temporary / "source_manifest.json", source_manifest)
        artifact_files = {
            "cases.jsonl": _artifact_metadata(temporary / "cases.jsonl", case_count),
            "evidence.jsonl": _artifact_metadata(
                temporary / "evidence.jsonl", evidence_count
            ),
            "gold.jsonl": _artifact_metadata(temporary / "gold.jsonl", gold_count),
            "verifier_observations.jsonl": _artifact_metadata(
                temporary / "verifier_observations.jsonl", observation_count
            ),
            "few_shot.jsonl": _artifact_metadata(
                temporary / "few_shot.jsonl", few_shot_count
            ),
            "case_lineage.jsonl": _artifact_metadata(
                temporary / "case_lineage.jsonl", lineage_count
            ),
            "source_manifest.json": _artifact_metadata(
                temporary / "source_manifest.json"
            ),
            "parity_50.json": _artifact_metadata(temporary / "parity_50.json"),
        }
        if materialization_manifest_path is not None:
            artifact_files[MATERIALIZATION_MANIFEST_NAME] = _artifact_metadata(
                temporary / MATERIALIZATION_MANIFEST_NAME
            )
        benchmark_material = {
            "benchmark_version": BENCHMARK_VERSION,
            "benchmark_profile": benchmark_profile,
            "seed": seed,
            "corpus_id": corpus_id,
            "jurisdiction": jurisdiction.upper(),
            "reporting_framework": reporting_framework,
            "standards_version": standards_version,
            "source_system": source_system,
            "counts": {
                "quant": quant_count,
                "narrative": narrative_count,
                "narrative_answerable": narrative_status_counts["OK"],
                "narrative_unanswerable": narrative_status_counts["REFUSAL"],
                "total": quant_count + narrative_count,
                "evidence_records": len(evidence_records),
                "evidence_items": sum(
                    len(record["items"]) for record in evidence_records
                ),
                "verifier_observations": len(verifier_observations),
                "few_shot": len(few_shot),
                "parity_slice": parity_slice["case_count"],
            },
            "few_shot_policy": {
                "quantitative_examples": DEFAULT_FEW_SHOT_PER_TASK_TYPE,
                "narrative_examples_per_subtype": DEFAULT_FEW_SHOT_PER_NARRATIVE_SUBTYPE,
                "answerable_per_pack": 2,
                "refusal_per_pack": 2,
                "narrative_subtypes": sorted(
                    {
                        str(row["task"]["narrative_subtype"])
                        for row in few_shot
                        if row["task"]["task_type"] == "narrative_citation"
                    }
                ),
                "request_pack_size": 4,
                "entity_and_filing_disjoint": True,
                "entity_key_precedence": [
                    "source_entity_id",
                    "cik",
                    "company_number",
                    "lei",
                    "issuer_id",
                    "entity_id",
                    "ticker",
                ],
                "evidence_included": True,
                "template_disjoint": True,
                "requires_human_review": True,
            },
            "stratification": stratification,
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
            "artifacts": artifact_files,
        }
        benchmark_id = canonical_json_sha256(benchmark_material)
        manifest = {**benchmark_material, "benchmark_id": benchmark_id}
        _write_json(temporary / "benchmark_manifest.json", manifest)
        preflight = verify_benchmark_artifacts(temporary)
        if not preflight["valid"]:
            raise ValueError(
                "Refusing to publish an invalid benchmark: "
                + "; ".join(preflight["errors"])
            )
        assert_no_secrets([temporary])
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite benchmark directory: {destination}"
            )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    verification = verify_benchmark_artifacts(destination)
    return {
        "benchmark_id": benchmark_id,
        "output_dir": str(destination),
        "manifest_path": str(destination / "benchmark_manifest.json"),
        "counts": manifest["counts"],
        "verified": verification["valid"],
    }


def _collect_values(value: Any, keys: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if str(raw_key) in keys:
                if isinstance(child, list):
                    found.update(str(item) for item in child)
                elif child is not None:
                    found.add(str(child))
            found.update(_collect_values(child, keys))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect_values(child, keys))
    return found


def verify_benchmark_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Verify hashes, counts, leakage boundaries, and example disjointness."""

    root = Path(output_dir)
    manifest_path = root / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    expected_manifest_fields = {
        "benchmark_version",
        "benchmark_profile",
        "seed",
        "corpus_id",
        "jurisdiction",
        "reporting_framework",
        "standards_version",
        "source_system",
        "counts",
        "few_shot_policy",
        "stratification",
        "artifact_visibility",
        "artifacts",
        "benchmark_id",
    }
    if set(manifest) != expected_manifest_fields:
        errors.append("benchmark manifest fields are missing or unexpected")
    if manifest.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append(f"unsupported benchmark_version: expected {BENCHMARK_VERSION}")
    benchmark_material = {
        key: value for key, value in manifest.items() if key != "benchmark_id"
    }
    recomputed_benchmark_id = canonical_json_sha256(benchmark_material)
    if manifest.get("benchmark_id") != recomputed_benchmark_id:
        errors.append("benchmark_id does not bind the canonical manifest material")
    expected_artifact_names = {
        "cases.jsonl",
        "evidence.jsonl",
        "gold.jsonl",
        "verifier_observations.jsonl",
        "few_shot.jsonl",
        "case_lineage.jsonl",
        "parity_50.json",
        "source_manifest.json",
    }
    if manifest.get("benchmark_profile") in _MATERIALIZED_BENCHMARK_PROFILES:
        expected_artifact_names.add(MATERIALIZATION_MANIFEST_NAME)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifact_names:
        errors.append("benchmark artifact set is missing or unexpected")
    visibility = manifest.get("artifact_visibility")
    expected_visibility = {
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
    }
    if visibility != expected_visibility:
        errors.append("artifact visibility boundary is missing or invalid")
    for name, expected in artifacts.items() if isinstance(artifacts, Mapping) else ():
        path = root / name
        if not path.is_file():
            errors.append(f"missing artifact: {name}")
            continue
        if _sha256_path(path) != expected.get("sha256"):
            errors.append(f"sha256 mismatch: {name}")
        if path.stat().st_size != expected.get("bytes"):
            errors.append(f"byte count mismatch: {name}")
        expected_records = expected.get("records")
        if expected_records is not None:
            try:
                actual_records = len(_read_jsonl(path))
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"invalid JSONL {name}: {exc}")
            else:
                if actual_records != expected_records:
                    errors.append(f"record count mismatch: {name}")

    try:
        source_manifest = json.loads(
            (root / "source_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"could not load source_manifest.json: {exc}")
        source_manifest = {}
    if not isinstance(source_manifest, Mapping):
        errors.append("source_manifest.json must contain an object")
        source_manifest = {}
    else:
        for key in ("benchmark_profile", "corpus_id", "jurisdiction"):
            if source_manifest.get(key) != manifest.get(key):
                errors.append(
                    f"source_manifest.json {key} conflicts with benchmark manifest"
                )
        if manifest.get("benchmark_profile") in _MATERIALIZED_BENCHMARK_PROFILES:
            materialization_source = _as_mapping(
                _as_mapping(source_manifest.get("sources")).get(
                    "materialization_manifest"
                )
            )
            expected_source_fields = {"file_name", "sha256", "bytes"}
            if (
                set(materialization_source) != expected_source_fields
                or materialization_source.get("file_name")
                != MATERIALIZATION_MANIFEST_NAME
            ):
                errors.append(
                    f"source_manifest.json must bind {MATERIALIZATION_MANIFEST_NAME}"
                )
            else:
                materialization_path = root / MATERIALIZATION_MANIFEST_NAME
                if not materialization_path.is_file():
                    errors.append(f"missing artifact: {MATERIALIZATION_MANIFEST_NAME}")
                else:
                    if _sha256_path(materialization_path) != materialization_source.get(
                        "sha256"
                    ):
                        errors.append(
                            f"source manifest sha256 mismatch: {MATERIALIZATION_MANIFEST_NAME}"
                        )
                    if (
                        materialization_path.stat().st_size
                        != materialization_source.get("bytes")
                    ):
                        errors.append(
                            f"source manifest byte count mismatch: {MATERIALIZATION_MANIFEST_NAME}"
                        )
                    artifact_record = _as_mapping(
                        _as_mapping(artifacts).get(MATERIALIZATION_MANIFEST_NAME)
                    )
                    if artifact_record != {
                        "sha256": materialization_source.get("sha256"),
                        "bytes": materialization_source.get("bytes"),
                    }:
                        errors.append(
                            f"artifact metadata conflicts with source binding: {MATERIALIZATION_MANIFEST_NAME}"
                        )
                    try:
                        materialization_manifest = json.loads(
                            materialization_path.read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError) as exc:
                        errors.append(
                            f"could not load {MATERIALIZATION_MANIFEST_NAME}: {exc}"
                        )
                        materialization_manifest = {}
                    if not isinstance(materialization_manifest, Mapping):
                        errors.append(
                            f"{MATERIALIZATION_MANIFEST_NAME} must contain an object"
                        )
                        materialization_manifest = {}
                    if (
                        materialization_manifest.get("materialization_version")
                        != "agent_evidence.v1"
                    ):
                        errors.append(
                            f"{MATERIALIZATION_MANIFEST_NAME} has an invalid materialization_version"
                        )
                    materialization_artifacts = _as_mapping(
                        materialization_manifest.get("artifacts")
                    )
                    benchmark_sources = _as_mapping(source_manifest.get("sources"))
                    for artifact_name, source_name in (
                        ("enriched_quant.jsonl", "quant"),
                        ("enriched_narrative.jsonl", "narrative"),
                    ):
                        declared_binding = _as_mapping(
                            materialization_artifacts.get(artifact_name)
                        )
                        benchmark_binding = _as_mapping(
                            benchmark_sources.get(source_name)
                        )
                        mismatches = [
                            key
                            for key in ("sha256", "bytes", "records")
                            if declared_binding.get(key) != benchmark_binding.get(key)
                        ]
                        if mismatches:
                            errors.append(
                                f"{MATERIALIZATION_MANIFEST_NAME} {artifact_name} "
                                "binding conflicts with benchmark source: "
                                + ", ".join(mismatches)
                            )

    try:
        cases = _read_jsonl(root / "cases.jsonl")
        evidence_records = _read_jsonl(root / "evidence.jsonl")
        gold = _read_jsonl(root / "gold.jsonl")
        verifier_observations = _read_jsonl(root / "verifier_observations.jsonl")
        few_shot = _read_jsonl(root / "few_shot.jsonl")
        parity_slice = json.loads((root / "parity_50.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"could not load benchmark records: {exc}")
        cases, evidence_records, gold, verifier_observations, few_shot, parity_slice = (
            [],
            [],
            [],
            [],
            [],
            {},
        )

    case_ids = [str(row["task_id"]) for row in cases]
    evidence_task_ids = [str(row["task_id"]) for row in evidence_records]
    gold_ids = [str(row["task_id"]) for row in gold]
    observation_task_ids = [str(row["task_id"]) for row in verifier_observations]
    if case_ids != gold_ids:
        errors.append("cases.jsonl and gold.jsonl task IDs/order differ")
    if case_ids != evidence_task_ids:
        errors.append("cases.jsonl and evidence.jsonl task IDs/order differ")
    if case_ids != observation_task_ids:
        errors.append(
            "cases.jsonl and verifier_observations.jsonl task IDs/order differ"
        )
    expected_total = manifest.get("counts", {}).get("total")
    if expected_total is not None and len(cases) != expected_total:
        errors.append(f"expected {expected_total} cases, found {len(cases)}")
    stratification = _as_mapping(manifest.get("stratification"))
    if (
        stratification.get("stratification_version")
        != "auditops-agent-stratification.v1"
    ):
        errors.append("benchmark stratification version is missing or invalid")
    declared_counts = _as_mapping(manifest.get("counts"))

    def declared_count(name: str) -> int:
        value = declared_counts.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"benchmark count {name} is missing or invalid")
            return 0
        return value

    quant_count = declared_count("quant")
    narrative_count = declared_count("narrative")
    narrative_answerable_count = declared_count("narrative_answerable")
    narrative_unanswerable_count = declared_count("narrative_unanswerable")
    total_count = declared_count("total")
    errors.extend(
        _benchmark_profile_errors(
            manifest.get("benchmark_profile"),
            jurisdiction=manifest.get("jurisdiction"),
            source_system=manifest.get("source_system"),
            seed=manifest.get("seed"),
            quant_count=quant_count,
            narrative_count=narrative_count,
            narrative_answerable=narrative_answerable_count,
            narrative_unanswerable=narrative_unanswerable_count,
            total_count=total_count,
        )
    )
    actual_quant_count = sum(case.get("task_type") == "quant_metric" for case in cases)
    actual_narrative_count = sum(
        case.get("task_type") == "narrative_citation" for case in cases
    )
    actual_narrative_answerable = sum(
        row.get("task_type") == "narrative_citation"
        and _as_mapping(row.get("target")).get("status") == "OK"
        for row in gold
    )
    actual_narrative_unanswerable = sum(
        row.get("task_type") == "narrative_citation"
        and _as_mapping(row.get("target")).get("status") == "REFUSAL"
        for row in gold
    )
    actual_counts = {
        "quant": actual_quant_count,
        "narrative": actual_narrative_count,
        "narrative_answerable": actual_narrative_answerable,
        "narrative_unanswerable": actual_narrative_unanswerable,
        "total": len(cases),
    }
    for name, actual_count in actual_counts.items():
        if declared_counts.get(name) != actual_count:
            errors.append(
                f"benchmark count {name} does not match packaged task membership"
            )
    errors.extend(
        _stratification_errors(
            stratification,
            benchmark_profile=str(manifest.get("benchmark_profile")),
            quant_count=quant_count,
            narrative_count=narrative_count,
        )
    )
    for task_type, manifest_key in (
        ("quant_metric", "quant"),
        ("narrative_citation", "narrative"),
    ):
        gold_strata = [
            _as_mapping(row.get("strata"))
            for row in gold
            if row.get("task_type") == task_type
        ]
        recomputed_profile = _axes_profile(gold_strata)
        declared_profile = _as_mapping(
            _as_mapping(stratification.get(manifest_key)).get("selected")
        )
        if recomputed_profile != declared_profile:
            errors.append(
                f"{manifest_key} selected stratification does not match evaluator-only gold strata"
            )
    expected_parity_count = min(PARITY_SLICE_COUNT, len(cases))
    expected_parity_fields = {
        "parity_slice_version",
        "requested_count",
        "case_count",
        "seed",
        "task_ids",
        "task_ids_sha256",
        "breakdown",
    }
    if (
        not isinstance(parity_slice, Mapping)
        or set(parity_slice) != expected_parity_fields
    ):
        errors.append("parity_50.json fields are missing or unexpected")
    else:
        parity_ids = parity_slice.get("task_ids")
        if (
            parity_slice.get("parity_slice_version") != PARITY_SLICE_VERSION
            or parity_slice.get("requested_count") != PARITY_SLICE_COUNT
            or parity_slice.get("case_count") != expected_parity_count
            or manifest.get("counts", {}).get("parity_slice") != expected_parity_count
            or not isinstance(parity_ids, list)
            or len(parity_ids) != expected_parity_count
            or len(set(parity_ids)) != len(parity_ids)
            or not set(parity_ids).issubset(set(case_ids))
            or parity_slice.get("task_ids_sha256") != canonical_json_sha256(parity_ids)
        ):
            errors.append("parity_50.json membership or hash is invalid")
        elif len(cases) >= PARITY_SLICE_COUNT:
            parity_statuses = {
                str(row["task_id"]): _as_mapping(row.get("target")).get("status")
                for row in gold
            }
            parity_cases = {str(row["task_id"]): row for row in cases}
            actual_breakdown: dict[str, int] = defaultdict(int)
            for task_id in parity_ids:
                actual_breakdown[
                    f"{parity_cases[task_id].get('task_type')}:{parity_statuses.get(task_id)}"
                ] += 1
            if dict(sorted(actual_breakdown.items())) != parity_slice.get("breakdown"):
                errors.append("parity_50.json breakdown is invalid")
    for case in cases:
        forbidden = _find_forbidden_keys(case)
        if forbidden:
            errors.append(
                f"gold leakage in task {case.get('task_id')}: {', '.join(forbidden)}"
            )
        try:
            validate_agent_task_input(case)
        except ValueError as exc:
            errors.append(f"invalid AgentTaskInputV2 {case.get('task_id')}: {exc}")

    evidence_by_task = {str(row["task_id"]): row for row in evidence_records}
    for case in cases:
        task_id = str(case["task_id"])
        record = evidence_by_task.get(task_id)
        if record is None:
            continue
        if set(record) != {"task_id", "items"}:
            errors.append(
                f"evidence record {task_id} must contain exactly task_id and items"
            )
            continue
        items = record.get("items")
        if not isinstance(items, list) or len(items) > 5:
            errors.append(
                f"evidence record {task_id} must contain an items array of at most five entries"
            )
            continue
        evidence_ids: list[str] = []
        filing_id = _as_mapping(case.get("filing")).get("filing_id")
        evidence_scope = _as_mapping(case.get("evidence_scope"))
        allowed_filing_ids = {
            str(item) for item in evidence_scope.get("filing_ids", [])
        }
        if not allowed_filing_ids and filing_id is not None:
            allowed_filing_ids = {str(filing_id)}
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                errors.append(
                    f"evidence record {task_id} item {index} is not an object"
                )
                continue
            missing = {"evidence_id", "filing_id", "content"} - set(item)
            if missing:
                errors.append(
                    f"evidence record {task_id} item {index} missing {', '.join(sorted(missing))}"
                )
                continue
            evidence_id = str(item["evidence_id"])
            evidence_ids.append(evidence_id)
            if str(item.get("filing_id")) not in allowed_filing_ids:
                errors.append(
                    f"evidence record {task_id} item {index} has an out-of-scope filing_id"
                )
            if not isinstance(item.get("content"), str) or not item["content"].strip():
                errors.append(
                    f"evidence record {task_id} item {index} has empty content"
                )
        if len(evidence_ids) != len(set(evidence_ids)):
            errors.append(f"evidence record {task_id} contains duplicate evidence IDs")
        scoped_ids = evidence_scope.get("evidence_ids")
        if evidence_ids != scoped_ids:
            errors.append(f"evidence_scope mismatch for task {task_id}")
        forbidden = _find_forbidden_keys(record)
        if forbidden:
            errors.append(
                f"gold leakage in evidence for task {task_id}: {', '.join(forbidden)}"
            )
        try:
            build_context_pack(case, items, token_counter=lambda _: 0)
        except ValueError as exc:
            errors.append(f"invalid inference context for task {task_id}: {exc}")

    for case, verifier_record in zip(cases, verifier_observations):
        task_id = str(case["task_id"])
        if set(verifier_record) != {"task_id", "tool_name", "observation"}:
            errors.append(
                f"verifier observation {task_id} must contain exactly task_id, tool_name, and observation"
            )
            continue
        try:
            expected_tool = task_operation_spec(str(case.get("task_type"))).operation
        except ValueError:
            errors.append(
                f"verifier observation {task_id} has an unregistered task type"
            )
            continue
        if verifier_record.get("tool_name") != expected_tool:
            errors.append(f"verifier observation {task_id} uses the wrong tool")
        observation = verifier_record.get("observation")
        if not isinstance(observation, Mapping):
            errors.append(
                f"verifier observation {task_id} observation must be an object"
            )
            continue
        if observation.get("task_id") != task_id:
            errors.append(
                f"verifier observation {task_id} contains a mismatched task_id"
            )
        filing_id = _as_mapping(case.get("filing")).get("filing_id")
        if observation.get("filing_id") != filing_id:
            errors.append(
                f"verifier observation {task_id} contains a mismatched filing_id"
            )
        packaged_ids = {
            str(item.get("evidence_id"))
            for item in _as_mapping(evidence_by_task.get(task_id)).get("items", [])
            if isinstance(item, Mapping)
        }
        observation_ids = observation.get("evidence_ids")
        if not isinstance(observation_ids, list) or not {
            str(item) for item in observation_ids
        }.issubset(packaged_ids):
            errors.append(
                f"verifier observation {task_id} references unpackaged evidence"
            )

    gold_by_task = {str(row.get("task_id")): row for row in gold}
    observation_by_task = {
        str(row.get("task_id")): row for row in verifier_observations
    }
    for case in cases:
        task_id = str(case["task_id"])
        evidence_items = _as_mapping(evidence_by_task.get(task_id)).get("items", [])
        gold_target = _as_mapping(gold_by_task.get(task_id)).get("target")
        observation = _as_mapping(observation_by_task.get(task_id)).get("observation")
        if not isinstance(gold_target, Mapping) or not isinstance(observation, Mapping):
            errors.append(
                f"task {task_id} lacks a valid gold target or verifier observation"
            )
            continue
        if case.get("task_type") == "quant_metric":
            deterministic = evaluate_quant_evidence(case, evidence_items)
            if not _quant_target_matches_observation(gold_target, deterministic):
                errors.append(
                    f"quant gold for task {task_id} conflicts with deterministic evidence evaluation"
                )
            if _canonical_json_bytes(deterministic) != _canonical_json_bytes(
                observation
            ):
                errors.append(
                    f"quant verifier observation for task {task_id} was not independently reproduced"
                )
        else:
            status = gold_target.get("status")
            expected_ids = [
                str(item) for item in gold_target.get("chunk_evidence_ids") or []
            ]
            answer_text = gold_target.get("answer_text")
            evidence_index = {
                str(item.get("evidence_id")): item
                for item in evidence_items
                if isinstance(item, Mapping)
            }
            if status == "OK":
                supported = (
                    isinstance(answer_text, str)
                    and bool(answer_text)
                    and bool(expected_ids)
                    and all(item in evidence_index for item in expected_ids)
                    and any(
                        is_contiguous_text_supported(
                            answer_text,
                            str(evidence_index[item].get("content") or ""),
                        )
                        for item in expected_ids
                    )
                )
                if not supported:
                    errors.append(
                        f"narrative gold for task {task_id} is not extractively supported"
                    )
            else:
                allowed = set(
                    _as_mapping(case.get("refusal_policy")).get("allowed_codes", [])
                )
                if gold_target.get("refusal_code") not in allowed:
                    errors.append(
                        f"narrative refusal for task {task_id} is not allowed by its task policy"
                    )
            deterministic = load_frozen_evidence(case, evidence_items)
            if _canonical_json_bytes(deterministic) != _canonical_json_bytes(
                observation
            ):
                errors.append(
                    f"narrative verifier observation for task {task_id} was not independently reproduced"
                )

    example_cases = [
        row.get("task") for row in few_shot if isinstance(row.get("task"), Mapping)
    ]
    example_task_ids = {str(case.get("task_id")) for case in example_cases}
    if example_task_ids & set(case_ids):
        errors.append("few-shot and evaluation task IDs overlap")
    example_entities = {_case_source_entity_id(case) for case in example_cases}
    evaluation_entities = {_case_source_entity_id(case) for case in cases}
    if example_entities & evaluation_entities:
        errors.append("few-shot and evaluation entities overlap")
    example_filings = {
        str(_as_mapping(case.get("filing")).get("filing_id")) for case in example_cases
    }
    evaluation_filings = {
        str(_as_mapping(case.get("filing")).get("filing_id")) for case in cases
    }
    if example_filings & evaluation_filings:
        errors.append("few-shot and evaluation filings overlap")
    evidence_keys = {
        "evidence_ids",
        "fact_evidence_ids",
        "chunk_evidence_ids",
        "fact_evidence_id",
        "chunk_evidence_id",
    }
    example_evidence = _collect_values(few_shot, evidence_keys)
    evaluation_evidence = _collect_values(cases, evidence_keys)
    if example_evidence & evaluation_evidence:
        errors.append("few-shot and evaluation evidence IDs overlap")
    example_templates = {
        str(row.get("template_id")) for row in few_shot if row.get("template_id")
    }
    evaluation_templates = {
        str(row.get("template_id"))
        for row in gold
        if row.get("template_id") is not None
    }
    if example_templates & evaluation_templates:
        errors.append("few-shot and evaluation template IDs overlap")

    expected_few_shot = manifest.get("counts", {}).get("few_shot")
    if expected_few_shot is not None and len(few_shot) != expected_few_shot:
        errors.append(
            f"expected {expected_few_shot} few-shot examples, found {len(few_shot)}"
        )
    few_shot_breakdown: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in few_shot:
        case = row.get("task") or {}
        answer = row.get("assistant_response") or {}
        expected_fields = {
            "few_shot_example_version",
            "task_id",
            "example_id",
            "review_status",
            "template_id",
            "task_family",
            "task",
            "evidence_items",
            "plan_response",
            "tool_observation",
            "assistant_response",
            "source_task_sha256",
        }
        if set(row) != expected_fields:
            errors.append(
                f"few-shot example {row.get('example_id')} fields are missing or unexpected"
            )
        try:
            validate_agent_task_input(case)
            validate_agent_proposal(answer)
            evidence_items = row.get("evidence_items")
            if not isinstance(evidence_items, list):
                raise TypeError("evidence_items must be an array")
            build_context_pack(case, evidence_items, token_counter=lambda _: 0)
            if row.get("task_id") != case.get("task_id"):
                raise ValueError("wrapper task_id must match task.task_id")
            if not isinstance(row.get("template_id"), str) or not row["template_id"]:
                raise ValueError("template_id must be non-empty")
            if not isinstance(row.get("task_family"), str) or not row["task_family"]:
                raise ValueError("task_family must be non-empty")
            if case.get("task_type") == "quant_metric":
                example_observation = evaluate_quant_evidence(
                    case, evidence_items, visibility="MODEL_VISIBLE"
                )
            else:
                example_observation = load_frozen_evidence(
                    case, evidence_items, visibility="MODEL_VISIBLE"
                )
            validate_tool_observation(row.get("tool_observation"))
            if _canonical_json_bytes(example_observation) != _canonical_json_bytes(
                row.get("tool_observation")
            ):
                raise ValueError("tool_observation does not match deterministic replay")
            expected_plan = {
                "task_id": case["task_id"],
                "action": "CALL_TOOL",
                "tool_name": task_operation_spec(case["task_type"]).operation,
                "tool_arguments": expected_tool_arguments(case),
            }
            if row.get("plan_response") != expected_plan:
                raise ValueError("plan_response does not match the operation registry")
            verifier_result = verify_proposal(
                case,
                answer,
                evidence_items=evidence_items,
                tool_observation=example_observation,
                repair_count=0,
            )
            if not verifier_result["release_allowed"]:
                raise ValueError(
                    "assistant_response is not deterministically releasable: "
                    + ",".join(
                        str(check.get("code"))
                        for check in verifier_result.get("checks", [])
                        if not check.get("passed")
                    )
                )
        except ValueError as exc:
            errors.append(f"invalid few-shot example {row.get('example_id')}: {exc}")
        pack = (
            "quant_metric"
            if case.get("task_type") == "quant_metric"
            else str(case.get("narrative_subtype"))
        )
        few_shot_breakdown[
            (str(case.get("task_type")), pack, str(answer.get("status")))
        ] += 1
    for status in ("OK", "REFUSAL"):
        if few_shot_breakdown[("quant_metric", "quant_metric", status)] != 2:
            errors.append(
                f"few-shot balance must include two {status} quantitative examples"
            )
    manifest_subtypes = set(
        manifest.get("few_shot_policy", {}).get("narrative_subtypes", [])
    )
    required_subtypes = (
        set(NARRATIVE_SUBTYPES)
        if manifest.get("benchmark_profile") != BENCHMARK_PROFILE_SYNTHETIC
        else manifest_subtypes
    )
    if manifest_subtypes != required_subtypes:
        errors.append("few-shot narrative subtype policy is incomplete")
    for subtype in sorted(required_subtypes):
        for status in ("OK", "REFUSAL"):
            if few_shot_breakdown[("narrative_citation", subtype, status)] != 2:
                errors.append(
                    f"few-shot balance must include two {status} {subtype} examples"
                )

    return {
        "valid": not errors,
        "benchmark_id": manifest.get("benchmark_id"),
        "errors": errors,
        "case_count": len(cases),
        "evidence_record_count": len(evidence_records),
        "gold_count": len(gold),
        "verifier_observation_count": len(verifier_observations),
        "few_shot_count": len(few_shot),
    }


__all__ = [
    "AGENT_TASK_INPUT_VERSION",
    "BENCHMARK_PROFILES",
    "BENCHMARK_PROFILE_SYNTHETIC",
    "BENCHMARK_PROFILE_US_OFFLINE_SAMPLE_V0_1",
    "BENCHMARK_PROFILE_US_V0_1",
    "BENCHMARK_VERSION",
    "DEFAULT_NARRATIVE_COUNT",
    "DEFAULT_QUANT_COUNT",
    "DEFAULT_SEED",
    "build_agent_benchmark",
    "verify_benchmark_artifacts",
]
