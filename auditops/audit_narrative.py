"""Deterministic US auditor-report narrative task construction.

This module creates public-filing *verification* tasks.  It extracts language
that is already present in SEC filing chunks; it never classifies an opinion,
forms an audit conclusion, or invents a Critical Audit Matter (CAM).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical_json import canonical_json_bytes as _canonical_json_bytes
from .narrative_tasks import validate_narrative_task_spec
from .provenance import assert_no_secrets

BUILD_VERSION = "us_audit_narrative_build.v1"
MANIFEST_SUFFIX = ".manifest.json"
TASK_SPEC_VERSION = "v1"
TASK_SCHEMA_ID = "narrative_task_spec.v1"
TASK_TYPE = "narrative_citation"
REFUSAL_CODE = "NARRATIVE_NOT_SUPPORTED"
SOURCE_SYSTEM = "SEC-EDGAR"
SUPPORTED_SOURCE_SYSTEMS = frozenset({SOURCE_SYSTEM, "SEC-EDGAR-CACHED"})

_CHUNK_COLUMNS = frozenset(
    {
        "chunk_evidence_id",
        "filing_id",
        "period_key",
        "item",
        "heading",
        "subheading",
        "heading_path",
        "source_file",
        "char_start",
        "char_end",
        "retrieval_text",
        "text_masked",
    }
)
_FILING_COLUMNS = frozenset({"filing_id", "ticker", "form_type", "report_date"})

_REPORT_MARKERS = (
    "report of independent registered public accounting firm",
    "report of independent auditors",
    "report of independent auditor",
    "opinion on the financial statements",
    "basis for opinion",
)
_FINANCIAL_STATEMENT_TERMS = (
    "financial statements",
    "consolidated balance sheets",
    "consolidated statements of operations",
    "consolidated statements of income",
)

_OPINION_RE = re.compile(r"\bin\s+our\s+opinion\b", re.IGNORECASE)
_DISCLAIMER_RE = re.compile(
    r"\bwe\s+do\s+not\s+express\s+an\s+opinion\b", re.IGNORECASE
)
_AUDITED_RE = re.compile(r"\bwe\s+have\s+audited\s+the\s+accompanying\b", re.IGNORECASE)
_CAM_PATTERNS: tuple[tuple[int, re.Pattern[str]], ...] = (
    (
        40,
        re.compile(
            r"\bprincipal\s+considerations\s+for\s+our\s+determination\s+that"
            r".{0,260}?\b(?:is|was|were)\s+(?:a\s+)?critical\s+audit\s+matter\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        35,
        re.compile(
            r"\bwe\s+(?:identified|determined)\b.{0,260}?\b(?:as|to\s+be)\s+"
            r"(?:a\s+)?critical\s+audit\s+matter\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        30,
        re.compile(
            r"\bwe\s+determined\s+that\b.{0,260}?\b(?:is|was|were)\s+"
            r"(?:a\s+)?critical\s+audit\s+matter\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        15,
        re.compile(
            r"\bhow\s+the\s+critical\s+audit\s+matter\s+was\s+addressed\s+"
            r"in\s+the\s+audit\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class _Candidate:
    score: int
    row: Mapping[str, Any]
    answer: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_fingerprint(path: Path, label: str) -> dict[str, int | str]:
    """Hash a stable file image and retain non-manifest mutation indicators."""

    try:
        before = path.stat()
        sha256 = _sha256_file(path)
        after = path.stat()
    except OSError as exc:
        raise RuntimeError(
            f"{label} changed while it was being hashed: {path}"
        ) from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"{label} changed while it was being hashed: {path}")
    return {
        "sha256": sha256,
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def _assert_file_unchanged(
    path: Path,
    label: str,
    before: Mapping[str, int | str],
) -> None:
    after = _file_fingerprint(path, label)
    if dict(before) != after:
        raise RuntimeError(f"{label} changed while narrative tasks were built: {path}")


def _source_metadata(
    path: Path,
    fingerprint: Mapping[str, int | str],
) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": str(fingerprint["sha256"]),
        "bytes": int(fingerprint["bytes"]),
    }


def _stable_id(*parts: Any) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()
    if table is None:
        raise ValueError(f"Canonical database is missing required table {table_name!r}")
    return {
        str(row["name"])
        for row in connection.execute(f'PRAGMA table_info("{table_name}")')
    }


def _validate_database_schema(connection: sqlite3.Connection) -> None:
    for table_name, required in (
        ("chunk_canon", _CHUNK_COLUMNS),
        ("filings", _FILING_COLUMNS),
    ):
        missing = sorted(required - _table_columns(connection, table_name))
        if missing:
            raise ValueError(
                f"Canonical database table {table_name!r} is missing columns: "
                + ", ".join(missing)
            )


def _connect_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _chunk_text(row: Mapping[str, Any]) -> str:
    return str(row.get("text_masked") or row.get("retrieval_text") or "").strip()


def _searchable_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in (
            "heading",
            "subheading",
            "heading_path",
            "retrieval_text",
            "text_masked",
        )
    ).casefold()


def _sentence_bounds(text: str, match: re.Match[str]) -> tuple[int, int]:
    start = 0
    for marker in ".!?":
        start = max(start, text.rfind(marker, 0, match.start()) + 1)
    while start < len(text) and text[start].isspace():
        start += 1

    endings = [
        position
        for marker in ".!?"
        if (position := text.find(marker, match.end())) >= 0
    ]
    end = min(endings) + 1 if endings else len(text)
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _exact_excerpt(text: str, match: re.Match[str], max_chars: int) -> str:
    """Return a bounded verbatim substring that retains the matched phrase."""

    sentence_start, sentence_end = _sentence_bounds(text, match)
    if sentence_end - sentence_start <= max_chars:
        return text[sentence_start:sentence_end]

    match_length = match.end() - match.start()
    if match_length >= max_chars:
        return text[match.start() : match.start() + max_chars].rstrip()

    left_budget = min(max_chars // 3, match.start() - sentence_start)
    start = match.start() - left_budget
    end = min(sentence_end, start + max_chars)
    if end - start < max_chars:
        start = max(sentence_start, end - max_chars)

    if start > sentence_start:
        next_space = text.find(" ", start, match.start())
        if next_space >= 0:
            start = next_space + 1
    end = min(sentence_end, start + max_chars)
    if end < sentence_end:
        previous_space = text.rfind(" ", match.end(), end)
        if previous_space > match.end():
            end = previous_space
    return text[start:end].strip()


def _report_candidate(
    row: Mapping[str, Any], *, filing_has_report_marker: bool, max_chars: int
) -> _Candidate | None:
    text = _chunk_text(row)
    if not text:
        return None

    disclaimer = _DISCLAIMER_RE.search(text)
    if disclaimer:
        excerpt = _exact_excerpt(text, disclaimer, max_chars)
        if any(term in excerpt.casefold() for term in _FINANCIAL_STATEMENT_TERMS):
            return _Candidate(45, row, excerpt)

    opinion = _OPINION_RE.search(text)
    if opinion:
        excerpt = _exact_excerpt(text, opinion, max_chars)
        lowered = excerpt.casefold()
        if (
            any(term in lowered for term in _FINANCIAL_STATEMENT_TERMS)
            and ("present fairly" in lowered or "do not present fairly" in lowered)
            and not (
                "internal control over financial reporting" in lowered
                and "financial statements" not in lowered
            )
        ):
            return _Candidate(50, row, excerpt)

    audited = _AUDITED_RE.search(text)
    if audited and filing_has_report_marker:
        return _Candidate(20, row, _exact_excerpt(text, audited, max_chars))
    return None


def _cam_candidate(row: Mapping[str, Any], *, max_chars: int) -> _Candidate | None:
    text = _chunk_text(row)
    if not text:
        return None
    lowered = text.casefold()
    if "no critical audit matters" in lowered or "no critical audit matter" in lowered:
        return None
    for score, pattern in _CAM_PATTERNS:
        match = pattern.search(text)
        if match:
            return _Candidate(score, row, _exact_excerpt(text, match, max_chars))
    return None


def _best_candidate(candidates: Iterable[_Candidate]) -> _Candidate | None:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            str(candidate.row.get("source_file") or ""),
            int(candidate.row.get("char_start") or 0),
            str(candidate.row.get("chunk_evidence_id") or ""),
        ),
    )
    return ranked[0] if ranked else None


def _validate_source_system(source_system: str) -> str:
    if source_system not in SUPPORTED_SOURCE_SYSTEMS:
        raise ValueError(
            "source_system must be one of "
            + ", ".join(sorted(SUPPORTED_SOURCE_SYSTEMS))
        )
    return source_system


def _with_source_system(row: Mapping[str, Any], source_system: str) -> dict[str, Any]:
    declared = row.get("source_system")
    if declared is not None and declared != source_system:
        raise ValueError(
            f"Task {row.get('task_id')!r} declares source_system={declared!r}, "
            f"which conflicts with configured {source_system!r}"
        )
    return {**row, "source_system": source_system}


def _task_row(
    filing_rows: Sequence[Mapping[str, Any]],
    *,
    task_class: str,
    candidate: _Candidate | None,
    source_system: str,
) -> dict[str, Any]:
    first = filing_rows[0]
    filing_id = str(first["filing_id"])
    ticker = str(first.get("filing_ticker") or filing_id)
    form_type = str(first.get("form_type") or "UNKNOWN")
    period_key = next(
        (
            str(row["period_key"])
            for row in filing_rows
            if row.get("period_key") is not None and str(row["period_key"]).strip()
        ),
        str(first.get("report_date") or "UNKNOWN_PERIOD"),
    )

    answerability = "ANSWERABLE" if candidate is not None else "UNANSWERABLE"
    template_variant = (
        int(_stable_id("audit-narrative-template", filing_id, task_class)[:8], 16) % 2
    )
    if task_class == "auditor_report_opinion_language":
        question = (
            (
                "Quote the exact auditor-report or opinion language stated in this "
                "filing. Do not infer or classify an audit opinion."
            )
            if template_variant == 0
            else (
                "What exact auditor-report language does this filing contain? Quote "
                "the filing. Do not infer or classify an audit opinion."
            )
        )
        retrieval_query = (
            "independent registered public accounting firm opinion financial "
            "statements present fairly audited accompanying"
        )
        label = "auditor_report_opinion_language"
        item_fallback = "auditor_report"
    elif task_class == "critical_audit_matter":
        question = (
            (
                "Quote the exact filing language that identifies a Critical Audit "
                "Matter. Do not infer or create a matter."
            )
            if template_variant == 0
            else (
                "What exact passage identifies a Critical Audit Matter in this filing? "
                "Quote only supported filing language."
            )
        )
        retrieval_query = (
            "critical audit matter principal considerations determination auditor"
        )
        label = "critical_audit_matter"
        item_fallback = "critical_audit_matter"
    else:  # pragma: no cover - all callers use the closed set above.
        raise ValueError(f"Unsupported audit narrative task class: {task_class}")

    source_row = candidate.row if candidate is not None else None
    task_id = _stable_id(
        "us-audit-narrative-task",
        TASK_SPEC_VERSION,
        filing_id,
        task_class,
        source_row.get("chunk_evidence_id") if source_row is not None else "absent",
    )
    task = {
        "narrative_task_spec_version": TASK_SPEC_VERSION,
        "narrative_task_schema_id": TASK_SCHEMA_ID,
        "task_id": task_id,
        "task_type": TASK_TYPE,
        "template_id": (
            f"us-audit-narrative:{task_class}:{answerability.casefold()}:"
            f"{template_variant}:v1"
        ),
        "task_family": f"narrative_citation:{task_class}:{answerability.casefold()}",
        "filing_id": filing_id,
        "ticker": ticker,
        "form_type": form_type,
        "period_key": str(source_row.get("period_key") or period_key)
        if source_row is not None
        else period_key,
        "item": source_row.get("item") if source_row is not None else item_fallback,
        "heading": source_row.get("heading") if source_row is not None else None,
        "subheading": source_row.get("subheading") if source_row is not None else None,
        "heading_path": source_row.get("heading_path")
        if source_row is not None
        else None,
        "question": question,
        "retrieval_query": retrieval_query,
        "label": label,
        "scope_type": "filing",
        "scope_key": filing_id,
        "answerability": answerability,
        "expected_chunk_ids": [str(source_row["chunk_evidence_id"])]
        if source_row is not None
        else [],
        "extractive_answer": candidate.answer if candidate is not None else None,
        "refusal_code": None if candidate is not None else REFUSAL_CODE,
        "negative_type": None
        if candidate is not None
        else "absent_after_full_filing_scan",
        "donor_task_id": None,
        "citation_policy": {
            "require_chunk_evidence_ids": candidate is not None,
            "retrieval_method": "bm25_rerank",
            "top_k": 5,
            "candidate_k": 15,
        },
        "jurisdiction": "US",
        "source_system": source_system,
        "reporting_framework": "US-GAAP",
        "standards_version": "PCAOB_PUBLIC_FILING_METADATA_V0.1",
        "filing_metadata": {
            "ticker": first.get("filing_ticker"),
            "form_type": first.get("form_type"),
            "report_date": first.get("report_date"),
        },
    }
    validate_narrative_task_spec(task)
    return task


def _tasks_for_filing(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_answer_chars: int,
    source_system: str,
) -> list[dict[str, Any]]:
    if not any(_chunk_text(row) for row in rows):
        return []
    filing_has_report_marker = any(
        marker in _searchable_text(row) for row in rows for marker in _REPORT_MARKERS
    )
    report_candidate = _best_candidate(
        candidate
        for row in rows
        if (
            candidate := _report_candidate(
                row,
                filing_has_report_marker=filing_has_report_marker,
                max_chars=max_answer_chars,
            )
        )
        is not None
    )
    cam_candidate = _best_candidate(
        candidate
        for row in rows
        if (candidate := _cam_candidate(row, max_chars=max_answer_chars)) is not None
    )
    return [
        _task_row(
            rows,
            task_class="auditor_report_opinion_language",
            candidate=report_candidate,
            source_system=source_system,
        ),
        _task_row(
            rows,
            task_class="critical_audit_matter",
            candidate=cam_candidate,
            source_system=source_system,
        ),
    ]


def _read_base_tasks(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        raise FileNotFoundError(f"Base narrative JSONL not found: {path}")
    unique: dict[str, dict[str, Any]] = {}
    raw_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw_count += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object in {path} line {line_number}")
            task_id = value.get("task_id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(
                    f"Base narrative task in {path} line {line_number} has no "
                    "non-empty task_id"
                )
            try:
                validate_narrative_task_spec(value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid base narrative task in {path} line {line_number}: {exc}"
                ) from exc
            previous = unique.get(task_id)
            if previous is not None and _canonical_json_bytes(
                previous
            ) != _canonical_json_bytes(value):
                raise ValueError(f"Conflicting duplicate task_id {task_id!r} in {path}")
            unique.setdefault(task_id, value)
    return list(unique.values()), raw_count


def _merge_tasks(
    base_tasks: Sequence[Mapping[str, Any]],
    generated_tasks: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    merged: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for source in itertools.chain(base_tasks, generated_tasks):
        task = dict(source)
        task_id = str(task["task_id"])
        previous = merged.get(task_id)
        if previous is None:
            merged[task_id] = task
            continue
        if _canonical_json_bytes(previous) != _canonical_json_bytes(task):
            raise ValueError(
                f"Conflicting duplicate task_id {task_id!r} across task sources"
            )
        duplicate_count += 1
    return [merged[task_id] for task_id in sorted(merged)], duplicate_count


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + MANIFEST_SUFFIX)


def _publish_immutable_bundle(
    output_path: Path,
    output_payload: bytes,
    manifest: Mapping[str, Any],
) -> None:
    """Publish an output and manifest without overwriting either destination."""

    manifest_path = _manifest_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (output_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable output: {path}")

    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    temporary_output = temporary_dir / output_path.name
    temporary_manifest = temporary_dir / manifest_path.name
    published: list[Path] = []
    try:
        for path, payload in (
            (temporary_output, output_payload),
            (temporary_manifest, _canonical_json_bytes(manifest, newline=True)),
        ):
            with path.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        assert_no_secrets([temporary_dir])
        try:
            os.link(temporary_output, output_path)
            published.append(output_path)
            os.link(temporary_manifest, manifest_path)
            published.append(manifest_path)
        except FileExistsError as exc:
            raise FileExistsError(
                "Refusing to overwrite immutable output or manifest: "
                f"{output_path}, {manifest_path}"
            ) from exc
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def build_us_audit_narrative_tasks(
    db_path: str | Path,
    output_path: str | Path,
    *,
    base_narrative_jsonl: str | Path | None = None,
    max_answer_chars: int = 320,
    source_system: str = SOURCE_SYSTEM,
) -> dict[str, Any]:
    """Build immutable US audit-report/CAM verification TaskSpecs.

    The database is opened in SQLite read-only/query-only mode.  Every refusal
    represents absence of the requested conservative pattern after all canonical
    chunks for that filing have been scanned.  Empty filings are excluded because
    absence cannot be established from missing evidence.
    """

    if isinstance(max_answer_chars, bool) or not isinstance(max_answer_chars, int):
        raise TypeError("max_answer_chars must be a positive integer")
    if max_answer_chars <= 0:
        raise ValueError("max_answer_chars must be a positive integer")
    source_system = _validate_source_system(source_system)

    database = Path(db_path)
    destination = Path(output_path)
    if not database.is_file():
        raise FileNotFoundError(f"Canonical database not found: {database}")
    destination_manifest = _manifest_path(destination)
    for path in (destination, destination_manifest):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable output: {path}")

    base_path = Path(base_narrative_jsonl) if base_narrative_jsonl is not None else None
    if base_path is not None and not base_path.is_file():
        raise FileNotFoundError(f"Base narrative JSONL not found: {base_path}")
    database_fingerprint = _file_fingerprint(database, "Canonical database")
    base_fingerprint = (
        _file_fingerprint(base_path, "Base narrative JSONL")
        if base_path is not None
        else None
    )
    base_tasks: list[dict[str, Any]] = []
    base_raw_count = 0
    if base_path is not None:
        base_tasks, base_raw_count = _read_base_tasks(base_path)
        base_tasks = [_with_source_system(task, source_system) for task in base_tasks]

    generated: list[dict[str, Any]] = []
    source_rows_digest = hashlib.sha256()
    source_chunk_count = 0
    scanned_filing_count = 0
    excluded_empty_filing_count = 0

    connection = _connect_read_only(database)
    try:
        _validate_database_schema(connection)
        total_filing_count = int(
            connection.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
        )
        for filing_row in connection.execute(
            """
            SELECT filing_id, ticker, form_type, report_date
            FROM filings
            ORDER BY filing_id
            """
        ):
            source_rows_digest.update(
                _canonical_json_bytes(
                    {"row": dict(filing_row), "table": "filings"}, newline=True
                )
            )
        cursor = connection.execute(
            """
            SELECT
              c.chunk_evidence_id,
              c.filing_id,
              c.period_key,
              c.item,
              c.heading,
              c.subheading,
              c.heading_path,
              c.source_file,
              c.char_start,
              c.char_end,
              c.retrieval_text,
              c.text_masked,
              f.ticker AS filing_ticker,
              f.form_type,
              f.report_date
            FROM chunk_canon c
            JOIN filings f ON f.filing_id = c.filing_id
            ORDER BY c.filing_id, c.source_file, c.char_start, c.chunk_evidence_id
            """
        )
        for _, group in itertools.groupby(
            cursor, key=lambda row: str(row["filing_id"])
        ):
            filing_rows = [dict(row) for row in group]
            scanned_filing_count += 1
            for row in filing_rows:
                source_rows_digest.update(
                    _canonical_json_bytes(
                        {"row": row, "table": "chunk_canon_join_filings"},
                        newline=True,
                    )
                )
                source_chunk_count += 1
            filing_tasks = _tasks_for_filing(
                filing_rows,
                max_answer_chars=max_answer_chars,
                source_system=source_system,
            )
            if not filing_tasks:
                excluded_empty_filing_count += 1
            generated.extend(filing_tasks)
    finally:
        connection.close()

    _assert_file_unchanged(database, "Canonical database", database_fingerprint)
    if base_path is not None and base_fingerprint is not None:
        _assert_file_unchanged(base_path, "Base narrative JSONL", base_fingerprint)

    merged, cross_source_duplicate_count = _merge_tasks(base_tasks, generated)
    class_counts: dict[str, dict[str, int]] = {}
    for task in generated:
        bucket = class_counts.setdefault(
            str(task["label"]), {"ANSWERABLE": 0, "UNANSWERABLE": 0, "total": 0}
        )
        bucket[str(task["answerability"])] += 1
        bucket["total"] += 1

    output_payload = b"".join(
        _canonical_json_bytes(row, newline=True) for row in merged
    )
    manifest = {
        "manifest_version": BUILD_VERSION,
        "manifest_path": str(destination_manifest.resolve()),
        "source_system": source_system,
        "sources": {
            "database": {
                **_source_metadata(database, database_fingerprint),
                "logical_rows_sha256": source_rows_digest.hexdigest(),
                "logical_rows": {
                    "sha256": source_rows_digest.hexdigest(),
                    "tables": {
                        "filings": total_filing_count,
                        "chunk_canon": source_chunk_count,
                    },
                },
                "chunk_count": source_chunk_count,
                "filing_count": total_filing_count,
            },
            "base_narrative_jsonl": None
            if base_path is None
            else {
                **_source_metadata(base_path, base_fingerprint),
                "raw_record_count": base_raw_count,
                "unique_record_count": len(base_tasks),
            },
        },
        "output": {
            "path": str(destination.resolve()),
            "sha256": hashlib.sha256(output_payload).hexdigest(),
            "bytes": len(output_payload),
            "record_count": len(merged),
        },
        "counts": {
            "database_filings": total_filing_count,
            "scanned_filings_with_chunks": scanned_filing_count,
            "filings_without_chunks": total_filing_count - scanned_filing_count,
            "excluded_empty_filings": excluded_empty_filing_count,
            "generated": len(generated),
            "generated_answerable": sum(
                task["answerability"] == "ANSWERABLE" for task in generated
            ),
            "generated_unanswerable": sum(
                task["answerability"] == "UNANSWERABLE" for task in generated
            ),
            "base_unique": len(base_tasks),
            "cross_source_duplicates_removed": cross_source_duplicate_count,
            "output": len(merged),
            "by_generated_class": class_counts,
        },
        "policy": {
            "scope": "public_filing_verification_only",
            "forms_audit_opinion": False,
            "professional_judgment": False,
            "refusal_requires_full_filing_scan": True,
            "max_answer_chars": max_answer_chars,
        },
    }
    _publish_immutable_bundle(destination, output_payload, manifest)
    return manifest
