"""Materialize inference-visible evidence from an AuditOps canonical database.

The materializer is deliberately independent of optional retrieval libraries.  It
uses only task-visible query and scope fields when ranking narrative chunks, and
copies evaluation-only fields through without consulting them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .canonical_json import canonical_json_bytes as _canonical_json_bytes
from .provenance import assert_no_secrets

MATERIALIZATION_VERSION = "agent_evidence.v1"
MATERIALIZATION_MANIFEST = "materialization_manifest.json"
SOURCE_SYSTEM = "SEC-EDGAR"
SUPPORTED_SOURCE_SYSTEMS = frozenset({SOURCE_SYSTEM, "SEC-EDGAR-CACHED"})
MAX_EVIDENCE_ITEMS = 5
BM25_K1 = 1.5
BM25_B = 0.75

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'-]*")
_FACT_COLUMNS = frozenset(
    {
        "canon_id",
        "fact_evidence_id",
        "filing_id",
        "concept_norm",
        "period_key",
        "unit_canon",
        "value_num_exact",
        "value_text",
        "source_anchor",
        "context_id",
        "entity_identifier",
        "dimensions_json",
    }
)
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
_CORPUS_PROVENANCE_PATHS = {
    "source_manifest": Path("manifest/source_manifest.json"),
    "download_ledger": Path("manifest/download_ledger.jsonl"),
    "ingest_ledger": Path("logs/ingest_ledger.jsonl"),
}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_metadata(
    path: Path,
    *,
    records: int | None = None,
    reported_path: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str((reported_path or path).resolve()),
        "sha256": _sha256_path(path),
        "bytes": path.stat().st_size,
    }
    if records is not None:
        result["records"] = records
    return result


def _file_fingerprint(path: Path, label: str) -> dict[str, int | str]:
    """Hash a stable file image and retain non-manifest mutation indicators."""

    try:
        before = path.stat()
        sha256 = _sha256_path(path)
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
        raise RuntimeError(
            f"{label} changed while evidence was being materialized: {path}"
        )


def _source_metadata(
    path: Path,
    fingerprint: Mapping[str, int | str],
    *,
    records: int | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": str(fingerprint["sha256"]),
        "bytes": int(fingerprint["bytes"]),
    }
    if records is not None:
        metadata["records"] = records
    return metadata


def _require_file(path: str | Path, label: str) -> Path:
    materialized = Path(path)
    if not materialized.is_file():
        raise FileNotFoundError(
            f"{label} does not exist or is not a file: {materialized}"
        )
    return materialized


def _corpus_provenance_paths(db_path: Path) -> tuple[Path, dict[str, Path]] | None:
    """Resolve the required chain for a canonical production corpus layout."""

    if db_path.name != "corpus.sqlite" or db_path.parent.name != "db":
        return None
    corpus_root = db_path.parent.parent
    paths = {
        name: corpus_root / relative_path
        for name, relative_path in _CORPUS_PROVENANCE_PATHS.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Canonical corpus provenance chain is incomplete; missing required files: "
            + ", ".join(missing)
        )
    return corpus_root, paths


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {label} line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object in {label} line {line_number}")
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(
                    f"Missing non-empty task_id in {label} line {line_number}"
                )
            if task_id in seen_task_ids:
                raise ValueError(f"Duplicate task_id {task_id!r} in {label}")
            if "evidence_items" in row or "retrieval_results" in row:
                raise ValueError(
                    f"Task {task_id!r} in {label} already contains materialized evidence fields"
                )
            seen_task_ids.add(task_id)
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_json_bytes(row, newline=True))


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _require_table_columns(
    connection: sqlite3.Connection,
    table_name: str,
    required_columns: frozenset[str],
) -> None:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    if table is None:
        raise ValueError(f"Canonical database is missing required table {table_name!r}")
    available = {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table_name})")
    }
    missing = sorted(required_columns - available)
    if missing:
        raise ValueError(
            f"Canonical database table {table_name!r} is missing columns: {', '.join(missing)}"
        )


def _non_empty_text(value: Any, *, field: str, task_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Task {task_id!r} requires non-empty {field}")
    return value.strip()


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


def _sequence(value: Any, *, field: str, task_id: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"Task {task_id!r} requires {field} to be an array")
    return value


def _fact_content(
    *,
    input_name: str,
    concept: str,
    period_key: str,
    unit: str | None,
    value: str | None,
) -> str:
    return "; ".join(
        (
            f"input_name={input_name}",
            f"concept={concept}",
            f"period_key={period_key}",
            f"unit={unit if unit is not None else 'null'}",
            f"value={value if value is not None else 'null'}",
        )
    )


def _trusted_source_entity_id(
    connection: sqlite3.Connection,
    *,
    filing_id: str,
    task_id: str,
) -> str:
    source_entities = {
        str(item["entity_identifier"]).strip()
        for item in connection.execute(
            """
            SELECT DISTINCT entity_identifier
            FROM facts_canon
            WHERE filing_id=? AND entity_identifier IS NOT NULL
              AND TRIM(entity_identifier) <> ''
            """,
            (filing_id,),
        ).fetchall()
    }
    if len(source_entities) != 1:
        raise ValueError(
            f"Task {task_id!r} primary filing does not have one unambiguous source entity identifier"
        )
    return next(iter(source_entities))


def _materialize_quant_row(
    connection: sqlite3.Connection,
    row: Mapping[str, Any],
    *,
    evidence_limit: int,
    source_system: str,
) -> dict[str, Any]:
    task_id = str(row["task_id"])
    filing_id = _non_empty_text(
        row.get("filing_id"), field="filing_id", task_id=task_id
    )
    task_filing = connection.execute(
        "SELECT ticker FROM filings WHERE filing_id=?", (filing_id,)
    ).fetchone()
    if task_filing is None:
        raise ValueError(f"Task {task_id!r} references unknown filing {filing_id!r}")
    task_ticker = task_filing["ticker"]
    trusted_source_entity_id = _trusted_source_entity_id(
        connection,
        filing_id=filing_id,
        task_id=task_id,
    )
    canonical_inputs = _sequence(
        row.get("canonical_inputs"), field="canonical_inputs", task_id=task_id
    )

    references: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for input_index, raw_input in enumerate(canonical_inputs):
        if not isinstance(raw_input, Mapping):
            raise TypeError(
                f"Task {task_id!r} canonical_inputs[{input_index}] must be an object"
            )
        input_name = _non_empty_text(
            raw_input.get("input_name"),
            field=f"canonical_inputs[{input_index}].input_name",
            task_id=task_id,
        )
        evidence_ids = _sequence(
            raw_input.get("fact_evidence_ids"),
            field=f"canonical_inputs[{input_index}].fact_evidence_ids",
            task_id=task_id,
        )
        for evidence_index, raw_evidence_id in enumerate(evidence_ids):
            evidence_id = _non_empty_text(
                raw_evidence_id,
                field=(
                    f"canonical_inputs[{input_index}].fact_evidence_ids[{evidence_index}]"
                ),
                task_id=task_id,
            )
            if evidence_id not in seen_ids:
                references.append((evidence_id, input_name))
                seen_ids.add(evidence_id)

    if len(references) > evidence_limit:
        raise ValueError(
            f"Task {task_id!r} references {len(references)} canonical facts; "
            f"the evidence limit is {evidence_limit}. Refusing to emit partial evidence."
        )

    evidence_items: list[dict[str, Any]] = []
    for rank, (evidence_id, input_name) in enumerate(references, start=1):
        matches = connection.execute(
            """
            SELECT fc.fact_evidence_id, fc.filing_id, fc.concept_norm, fc.period_key, fc.unit_canon,
                   fc.value_num_exact, fc.value_text, fc.source_anchor, fc.context_id,
                   fc.entity_identifier, fc.dimensions_json, f.ticker AS fact_ticker
            FROM facts_canon AS fc
            JOIN filings AS f ON f.filing_id = fc.filing_id
            WHERE fc.fact_evidence_id=?
            ORDER BY fc.canon_id
            """,
            (evidence_id,),
        ).fetchall()
        if not matches:
            raise ValueError(
                f"Task {task_id!r} references missing facts_canon evidence ID {evidence_id!r}"
            )
        if len(matches) != 1:
            raise ValueError(
                f"Task {task_id!r} evidence ID {evidence_id!r} is ambiguous in facts_canon"
            )
        fact = dict(matches[0])
        if str(fact.get("entity_identifier") or "").strip() != trusted_source_entity_id:
            raise ValueError(
                f"Task {task_id!r} evidence ID {evidence_id!r} belongs to a different source entity"
            )
        if fact["filing_id"] != filing_id and (
            not task_ticker
            or not fact["fact_ticker"]
            or str(fact["fact_ticker"]).upper() != str(task_ticker).upper()
        ):
            raise ValueError(
                f"Task {task_id!r} evidence ID {evidence_id!r} belongs to filing "
                f"{fact['filing_id']!r} outside the task issuer scope"
            )
        value = fact["value_num_exact"]
        if value is None:
            value = fact["value_text"]
        value = None if value is None else str(value)
        unit = None if fact["unit_canon"] is None else str(fact["unit_canon"])
        period_key = str(fact["period_key"])
        concept = str(fact["concept_norm"])
        try:
            dimensions = json.loads(fact["dimensions_json"] or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Task {task_id!r} evidence ID {evidence_id!r} has invalid dimensions_json"
            ) from exc
        if not isinstance(dimensions, dict):
            raise TypeError(
                f"Task {task_id!r} evidence ID {evidence_id!r} dimensions must be an object"
            )
        evidence_items.append(
            {
                "evidence_id": evidence_id,
                "filing_id": str(fact["filing_id"]),
                "content": _fact_content(
                    input_name=input_name,
                    concept=concept,
                    period_key=period_key,
                    unit=unit,
                    value=value,
                ),
                "rank": rank,
                "period_key": period_key,
                "unit": unit,
                "value": value,
                "source_system": source_system,
                "metadata": {
                    "input_name": input_name,
                    "concept": concept,
                    "source_anchor": fact["source_anchor"],
                    "context_id": fact["context_id"],
                    "entity_id": fact["entity_identifier"],
                    "dimensions": dimensions,
                },
            }
        )

    enriched = _with_source_system(row, source_system)
    enriched["source_entity_id"] = trusted_source_entity_id
    enriched["evidence_items"] = evidence_items
    return enriched


def _normalize_heading_text(text: str) -> str:
    normalized = re.sub(r"[ \t]+", " ", text.replace("\xa0", " ")).strip()
    if not normalized:
        return normalized
    words = normalized.split()
    merged: list[str] = []
    index = 0
    while index < len(words):
        if index + 1 < len(words):
            left, right = words[index], words[index + 1]
            if (
                left.isupper()
                and right.isupper()
                and (
                    (len(left) == 1 and len(right) > 1)
                    or (len(left) > 3 and 1 < len(right) <= 3)
                )
            ):
                merged.append(left + right)
                index += 2
                continue
        merged.append(words[index])
        index += 1
    return " ".join(merged)


def _scope_key_for_parts(
    filing_id: str,
    *,
    item: Any,
    heading: Any,
    subheading: Any,
    heading_path: Any,
) -> str:
    if isinstance(heading_path, str) and heading_path.strip():
        return f"note:{filing_id}:{_normalize_heading_text(heading_path).lower()}"
    parts = [
        _normalize_heading_text(part).lower()
        for part in (item, heading, subheading)
        if isinstance(part, str) and part.strip()
    ]
    suffix = "|".join(parts) if parts else "unknown"
    return f"note:{filing_id}:{suffix}"


def _task_scope_key(row: Mapping[str, Any], filing_id: str) -> str:
    scope_key = row.get("scope_key")
    if isinstance(scope_key, str) and scope_key.strip():
        return scope_key.strip().lower()
    return _scope_key_for_parts(
        filing_id,
        item=row.get("item"),
        heading=row.get("heading"),
        subheading=row.get("subheading"),
        heading_path=row.get("heading_path"),
    )


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _chunk_document(row: Mapping[str, Any]) -> str:
    return "\n".join(
        str(value).strip()
        for value in (
            row.get("item"),
            row.get("heading"),
            row.get("subheading"),
            row.get("heading_path"),
            row.get("retrieval_text"),
            row.get("text_masked"),
        )
        if value is not None and str(value).strip()
    )


def _bm25_rank(
    query: str, rows: Sequence[Mapping[str, Any]]
) -> list[tuple[float, Mapping[str, Any]]]:
    query_terms = Counter(_tokenize(query))
    tokenized_documents = [_tokenize(_chunk_document(row)) for row in rows]
    document_count = len(tokenized_documents)
    average_length = (
        sum(len(tokens) for tokens in tokenized_documents) / document_count
        if document_count
        else 1.0
    )
    if average_length == 0:
        average_length = 1.0

    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_documents:
        document_frequency.update(set(tokens))

    ranked: list[tuple[float, Mapping[str, Any]]] = []
    for row, tokens in zip(rows, tokenized_documents):
        term_frequency = Counter(tokens)
        document_length = len(tokens)
        score = 0.0
        for term, query_frequency in query_terms.items():
            frequency = term_frequency.get(term, 0)
            if not frequency:
                continue
            frequency_in_documents = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0
                + (document_count - frequency_in_documents + 0.5)
                / (frequency_in_documents + 0.5)
            )
            denominator = frequency + BM25_K1 * (
                1.0 - BM25_B + BM25_B * document_length / average_length
            )
            score += (
                query_frequency
                * inverse_document_frequency
                * frequency
                * (BM25_K1 + 1.0)
                / denominator
            )
        ranked.append((score, row))

    ranked.sort(
        key=lambda item: (
            -item[0],
            int(item[1]["char_start"]),
            str(item[1]["chunk_evidence_id"]),
        )
    )
    return ranked


def _chunk_evidence_item(
    row: Mapping[str, Any],
    *,
    rank: int,
    source_system: str,
) -> dict[str, Any]:
    content = str(row.get("text_masked") or row.get("retrieval_text") or "").strip()
    if not content:
        raise ValueError(
            f"Chunk evidence ID {row['chunk_evidence_id']!r} has empty content"
        )
    period_key = row.get("period_key")
    return {
        "evidence_id": str(row["chunk_evidence_id"]),
        "filing_id": str(row["filing_id"]),
        "content": content,
        "rank": rank,
        "period_key": None if period_key is None else str(period_key),
        "source_system": source_system,
        "metadata": {
            "item": row.get("item"),
            "heading": row.get("heading"),
            "subheading": row.get("subheading"),
            "heading_path": row.get("heading_path"),
            "source_file": row.get("source_file"),
            "char_start": row.get("char_start"),
            "char_end": row.get("char_end"),
        },
    }


def _materialize_narrative_row(
    connection: sqlite3.Connection,
    row: Mapping[str, Any],
    *,
    top_k: int,
    candidate_k: int,
    source_system: str,
) -> dict[str, Any]:
    task_id = str(row["task_id"])
    filing_id = _non_empty_text(
        row.get("filing_id"), field="filing_id", task_id=task_id
    )
    trusted_source_entity_id = _trusted_source_entity_id(
        connection,
        filing_id=filing_id,
        task_id=task_id,
    )
    raw_query = row.get("retrieval_query")
    query_field = "retrieval_query"
    if not isinstance(raw_query, str) or not raw_query.strip():
        raw_query = row.get("question")
        query_field = "question"
    query = _non_empty_text(raw_query, field=query_field, task_id=task_id)

    scope_type = row.get("scope_type") or "filing"
    if scope_type not in {"filing", "note"}:
        raise ValueError(f"Task {task_id!r} has unsupported scope_type {scope_type!r}")
    chunk_rows = [
        dict(chunk)
        for chunk in connection.execute(
            """
            SELECT chunk_evidence_id, filing_id, period_key, item, heading, subheading,
                   heading_path, source_file, char_start, char_end, retrieval_text,
                   text_masked
            FROM chunk_canon
            WHERE filing_id=?
            ORDER BY char_start, chunk_evidence_id
            """,
            (filing_id,),
        ).fetchall()
    ]
    if not chunk_rows:
        raise ValueError(
            f"Task {task_id!r} has no chunk_canon evidence for filing {filing_id!r}"
        )

    if scope_type == "note":
        scope_key = _task_scope_key(row, filing_id)
        chunk_rows = [
            chunk
            for chunk in chunk_rows
            if _scope_key_for_parts(
                filing_id,
                item=chunk.get("item"),
                heading=chunk.get("heading"),
                subheading=chunk.get("subheading"),
                heading_path=chunk.get("heading_path"),
            ).lower()
            == scope_key
        ]
        if not chunk_rows:
            raise ValueError(
                f"Task {task_id!r} has no chunk_canon evidence in note scope {scope_key!r}"
            )

    candidates = _bm25_rank(query, chunk_rows)[:candidate_k]
    selected = candidates[:top_k]
    evidence_items: list[dict[str, Any]] = []
    retrieval_results: list[dict[str, Any]] = []
    for rank, (score, chunk) in enumerate(selected, start=1):
        item = _chunk_evidence_item(
            chunk,
            rank=rank,
            source_system=source_system,
        )
        evidence_items.append(item)
        retrieval_results.append({**item, "bm25_score": round(score, 12)})

    enriched = _with_source_system(row, source_system)
    enriched["source_entity_id"] = trusted_source_entity_id
    enriched["evidence_items"] = evidence_items
    enriched["retrieval_results"] = retrieval_results
    return enriched


def _validate_limits(top_k: int, candidate_k: int) -> None:
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= MAX_EVIDENCE_ITEMS
    ):
        raise ValueError(f"top_k must be an integer between 1 and {MAX_EVIDENCE_ITEMS}")
    if (
        isinstance(candidate_k, bool)
        or not isinstance(candidate_k, int)
        or candidate_k < top_k
    ):
        raise ValueError(
            "candidate_k must be an integer greater than or equal to top_k"
        )


def materialize_agent_source_evidence(
    quant_jsonl: str | Path,
    narrative_jsonl: str | Path,
    db_path: str | Path,
    output_dir: str | Path,
    *,
    top_k: int = 5,
    candidate_k: int = 15,
    source_system: str = SOURCE_SYSTEM,
) -> dict[str, Any]:
    """Create immutable, inference-ready SEC evidence JSONL files.

    Quantitative evidence is resolved by canonical evidence ID.  Narrative
    evidence is ranked with deterministic, task-scoped Okapi BM25.  Ranking
    consults only ``retrieval_query`` (or ``question`` as its fallback), filing
    scope, and canonical chunk text; evaluator targets are never read.
    """

    _validate_limits(top_k, candidate_k)
    source_system = _validate_source_system(source_system)
    quant_path = _require_file(quant_jsonl, "Quant JSONL")
    narrative_path = _require_file(narrative_jsonl, "Narrative JSONL")
    database_path = _require_file(db_path, "Canonical database")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {destination}"
        )

    corpus_provenance = _corpus_provenance_paths(database_path)
    upstream_paths = {} if corpus_provenance is None else corpus_provenance[1]
    source_fingerprints = {
        "quant_jsonl": _file_fingerprint(quant_path, "Quant JSONL"),
        "narrative_jsonl": _file_fingerprint(narrative_path, "Narrative JSONL"),
        "database": _file_fingerprint(database_path, "Canonical database"),
    }
    upstream_fingerprints = {
        name: _file_fingerprint(path, f"Corpus provenance {name}")
        for name, path in upstream_paths.items()
    }
    quant_rows = _read_jsonl(quant_path, "quant JSONL")
    narrative_rows = _read_jsonl(narrative_path, "narrative JSONL")
    connection = _connect_read_only(database_path)
    try:
        _require_table_columns(connection, "facts_canon", _FACT_COLUMNS)
        _require_table_columns(connection, "chunk_canon", _CHUNK_COLUMNS)
        enriched_quant = [
            _materialize_quant_row(
                connection,
                row,
                evidence_limit=top_k,
                source_system=source_system,
            )
            for row in quant_rows
        ]
        enriched_narrative = [
            _materialize_narrative_row(
                connection,
                row,
                top_k=top_k,
                candidate_k=candidate_k,
                source_system=source_system,
            )
            for row in narrative_rows
        ]
    finally:
        connection.close()

    _assert_file_unchanged(
        quant_path, "Quant JSONL", source_fingerprints["quant_jsonl"]
    )
    _assert_file_unchanged(
        narrative_path,
        "Narrative JSONL",
        source_fingerprints["narrative_jsonl"],
    )
    _assert_file_unchanged(
        database_path, "Canonical database", source_fingerprints["database"]
    )
    for name, path in upstream_paths.items():
        _assert_file_unchanged(
            path,
            f"Corpus provenance {name}",
            upstream_fingerprints[name],
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        quant_output = temporary / "enriched_quant.jsonl"
        narrative_output = temporary / "enriched_narrative.jsonl"
        _write_jsonl(quant_output, enriched_quant)
        _write_jsonl(narrative_output, enriched_narrative)
        manifest_path = temporary / MATERIALIZATION_MANIFEST
        published_manifest_path = destination / MATERIALIZATION_MANIFEST
        manifest = {
            "materialization_version": MATERIALIZATION_VERSION,
            "manifest_path": str(published_manifest_path.resolve()),
            "output_dir": str(destination.resolve()),
            "parameters": {
                "top_k": top_k,
                "candidate_k": candidate_k,
                "retrieval_method": "stdlib_okapi_bm25",
                "source_system": source_system,
            },
            "counts": {
                "quant": len(enriched_quant),
                "narrative": len(enriched_narrative),
                "total": len(enriched_quant) + len(enriched_narrative),
                "quant_evidence_items": sum(
                    len(row["evidence_items"]) for row in enriched_quant
                ),
                "narrative_evidence_items": sum(
                    len(row["evidence_items"]) for row in enriched_narrative
                ),
            },
            "sources": {
                "quant_jsonl": _source_metadata(
                    quant_path,
                    source_fingerprints["quant_jsonl"],
                    records=len(quant_rows),
                ),
                "narrative_jsonl": _source_metadata(
                    narrative_path,
                    source_fingerprints["narrative_jsonl"],
                    records=len(narrative_rows),
                ),
                "database": _source_metadata(
                    database_path, source_fingerprints["database"]
                ),
                "corpus_provenance": None
                if corpus_provenance is None
                else {
                    "corpus_root": str(corpus_provenance[0].resolve()),
                    "required_artifacts": sorted(_CORPUS_PROVENANCE_PATHS),
                    "missing_artifacts": [],
                    "artifacts": {
                        name: _source_metadata(path, upstream_fingerprints[name])
                        for name, path in sorted(upstream_paths.items())
                    },
                },
            },
            "artifacts": {
                "enriched_quant.jsonl": _file_metadata(
                    quant_output,
                    records=len(enriched_quant),
                    reported_path=destination / "enriched_quant.jsonl",
                ),
                "enriched_narrative.jsonl": _file_metadata(
                    narrative_output,
                    records=len(enriched_narrative),
                    reported_path=destination / "enriched_narrative.jsonl",
                ),
            },
        }
        manifest_path.write_bytes(_canonical_json_bytes(manifest, newline=True))
        assert_no_secrets([temporary])
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output directory: {destination}"
            )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return manifest
