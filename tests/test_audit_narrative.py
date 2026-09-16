from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

import auditops.audit_narrative as audit_narrative_module
from auditops.audit_narrative import build_us_audit_narrative_tasks
from auditops.narrative_tasks import validate_narrative_task_spec


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _build_database(path: Path) -> tuple[Path, dict[str, str]]:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE filings (
              filing_id TEXT PRIMARY KEY,
              ticker TEXT,
              form_type TEXT,
              report_date TEXT
            );
            CREATE TABLE chunk_canon (
              chunk_evidence_id TEXT PRIMARY KEY,
              filing_id TEXT NOT NULL,
              period_key TEXT,
              item TEXT,
              heading TEXT,
              subheading TEXT,
              heading_path TEXT,
              source_file TEXT NOT NULL,
              char_start INTEGER NOT NULL,
              char_end INTEGER NOT NULL,
              retrieval_text TEXT,
              text_masked TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO filings(filing_id, ticker, form_type, report_date) VALUES (?, ?, ?, ?)",
            [
                ("supported-filing", "SUP", "10-K", "2025-12-31"),
                ("absent-filing", "ABS", "10-Q", "2025-09-30"),
                ("empty-filing", "EMP", "10-K", "2025-12-31"),
            ],
        )
        texts = {
            "opinion-chunk": (
                "In our opinion, the consolidated financial statements present "
                "fairly, in all material respects, the financial position of Example Corp."
            ),
            "cam-chunk": (
                "The principal considerations for our determination that revenue "
                "recognition is a critical audit matter were the significant judgment "
                "and audit effort required."
            ),
            "generic-chunk": (
                "Basis for Opinion. Critical audit matters are matters arising from "
                "the current-period audit that were communicated with the audit committee."
            ),
            "empty-chunk": "   ",
        }
        connection.executemany(
            """
            INSERT INTO chunk_canon(
              chunk_evidence_id, filing_id, period_key, item, heading, subheading,
              heading_path, source_file, char_start, char_end, retrieval_text,
              text_masked
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "opinion-chunk",
                    "supported-filing",
                    "FY2025",
                    "8",
                    "Report of Independent Registered Public Accounting Firm",
                    "Opinion on the Financial Statements",
                    "Item 8 > Auditor Report > Opinion on the Financial Statements",
                    "report.htm",
                    10,
                    170,
                    "independent auditor opinion financial statements present fairly",
                    texts["opinion-chunk"],
                ),
                (
                    "cam-chunk",
                    "supported-filing",
                    "FY2025",
                    "8",
                    "Critical Audit Matter",
                    "Revenue Recognition",
                    "Item 8 > Auditor Report > Critical Audit Matter > Revenue Recognition",
                    "report.htm",
                    200,
                    390,
                    "critical audit matter principal considerations revenue recognition",
                    texts["cam-chunk"],
                ),
                (
                    "generic-chunk",
                    "absent-filing",
                    "Q3_2025",
                    "1",
                    "Financial Statements",
                    "Basis for Opinion",
                    "Item 1 > Basis for Opinion",
                    "quarterly.htm",
                    1,
                    170,
                    "basis opinion definition critical audit matters",
                    texts["generic-chunk"],
                ),
                (
                    "empty-chunk",
                    "empty-filing",
                    "FY2025",
                    "8",
                    None,
                    None,
                    None,
                    "empty.htm",
                    1,
                    4,
                    None,
                    texts["empty-chunk"],
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return path, texts


def _base_accounting_policy_task(task_id: str = "base-accounting-policy") -> dict:
    return {
        "narrative_task_spec_version": "v1",
        "narrative_task_schema_id": "narrative_task_spec.v1",
        "task_id": task_id,
        "task_type": "narrative_citation",
        "filing_id": "base-filing",
        "ticker": "BASE",
        "form_type": "10-K",
        "period_key": "FY2025",
        "item": "8",
        "heading": "Notes",
        "subheading": "Revenue Recognition",
        "heading_path": "Item 8 > Notes > Revenue Recognition",
        "question": "What revenue-recognition policy does the filing state?",
        "retrieval_query": "revenue recognition policy control transfer",
        "label": "accounting_policy",
        "scope_type": "note",
        "scope_key": "note:base-filing:revenue-recognition",
        "answerability": "ANSWERABLE",
        "expected_chunk_ids": ["base-chunk"],
        "extractive_answer": "Revenue is recognized when control transfers.",
        "refusal_code": None,
        "negative_type": None,
        "donor_task_id": None,
        "citation_policy": {
            "require_chunk_evidence_ids": True,
            "retrieval_method": "bm25_rerank",
            "top_k": 5,
            "candidate_k": 15,
        },
    }


def test_builds_supported_opinion_cam_and_full_scan_refusals(tmp_path):
    database, source_texts = _build_database(tmp_path / "canon.sqlite")
    output = tmp_path / "audit-narrative.jsonl"

    manifest = build_us_audit_narrative_tasks(database, output)
    rows = _read_jsonl(output)

    assert len(rows) == 4
    assert manifest["counts"]["generated"] == 4
    assert manifest["counts"]["generated_answerable"] == 2
    assert manifest["counts"]["generated_unanswerable"] == 2
    assert manifest["counts"]["excluded_empty_filings"] == 1
    assert manifest["counts"]["filings_without_chunks"] == 0
    assert manifest["policy"]["forms_audit_opinion"] is False
    assert manifest["policy"]["professional_judgment"] is False
    assert manifest["source_system"] == "SEC-EDGAR"

    by_key = {(row["filing_id"], row["label"]): row for row in rows}
    opinion = by_key[("supported-filing", "auditor_report_opinion_language")]
    cam = by_key[("supported-filing", "critical_audit_matter")]
    assert opinion["answerability"] == "ANSWERABLE"
    assert opinion["expected_chunk_ids"] == ["opinion-chunk"]
    assert opinion["extractive_answer"] in source_texts["opinion-chunk"]
    assert "Do not infer or classify" in opinion["question"]
    assert cam["answerability"] == "ANSWERABLE"
    assert cam["expected_chunk_ids"] == ["cam-chunk"]
    assert cam["extractive_answer"] in source_texts["cam-chunk"]
    assert "Do not infer or create" in cam["question"]

    for label in ("auditor_report_opinion_language", "critical_audit_matter"):
        refusal = by_key[("absent-filing", label)]
        assert refusal["answerability"] == "UNANSWERABLE"
        assert refusal["expected_chunk_ids"] == []
        assert refusal["extractive_answer"] is None
        assert refusal["refusal_code"] == "NARRATIVE_NOT_SUPPORTED"
        assert refusal["negative_type"] == "absent_after_full_filing_scan"

    for row in rows:
        validate_narrative_task_spec(row)
        assert row["source_system"] == "SEC-EDGAR"

    assert manifest["sources"]["database"]["chunk_count"] == 4
    assert len(manifest["sources"]["database"]["sha256"]) == 64
    assert len(manifest["sources"]["database"]["logical_rows_sha256"]) == 64
    assert (
        manifest["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted_manifest == manifest
    assert manifest_path.read_bytes() == (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert manifest["sources"]["database"]["logical_rows"]["tables"] == {
        "filings": 3,
        "chunk_canon": 4,
    }


def test_cached_source_system_is_stamped_on_generated_and_base_tasks(tmp_path):
    database, _ = _build_database(tmp_path / "canon.sqlite")
    base = tmp_path / "base.jsonl"
    _write_jsonl(base, [_base_accounting_policy_task()])
    output = tmp_path / "cached.jsonl"

    manifest = build_us_audit_narrative_tasks(
        database,
        output,
        base_narrative_jsonl=base,
        source_system="SEC-EDGAR-CACHED",
    )

    assert manifest["source_system"] == "SEC-EDGAR-CACHED"
    assert {row["source_system"] for row in _read_jsonl(output)} == {"SEC-EDGAR-CACHED"}


def test_base_merge_deduplicates_identical_task_ids(tmp_path):
    database, _ = _build_database(tmp_path / "canon.sqlite")
    base = tmp_path / "base.jsonl"
    base_task = _base_accounting_policy_task()
    _write_jsonl(base, [base_task, base_task])

    output = tmp_path / "merged.jsonl"
    manifest = build_us_audit_narrative_tasks(
        database, output, base_narrative_jsonl=base
    )
    rows = _read_jsonl(output)

    assert sum(row["task_id"] == base_task["task_id"] for row in rows) == 1
    assert {row["label"] for row in rows} >= {
        "accounting_policy",
        "auditor_report_opinion_language",
        "critical_audit_matter",
    }
    assert manifest["sources"]["base_narrative_jsonl"]["raw_record_count"] == 2
    assert manifest["sources"]["base_narrative_jsonl"]["unique_record_count"] == 1
    assert manifest["counts"]["base_unique"] == 1
    assert manifest["counts"]["output"] == 5


def test_output_is_deterministic_and_immutable(tmp_path):
    database, _ = _build_database(tmp_path / "canon.sqlite")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    first_manifest = build_us_audit_narrative_tasks(database, first)
    second_manifest = build_us_audit_narrative_tasks(database, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_manifest["output"]["sha256"] == second_manifest["output"]["sha256"]
    before = first.read_bytes()
    with pytest.raises(FileExistsError, match="immutable output"):
        build_us_audit_narrative_tasks(database, first)
    assert first.read_bytes() == before


def test_refuses_preexisting_sibling_manifest(tmp_path):
    database, _ = _build_database(tmp_path / "canon.sqlite")
    output = tmp_path / "audit-narrative.jsonl"
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text("reserved\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable output"):
        build_us_audit_narrative_tasks(database, output)

    assert not output.exists()
    assert manifest_path.read_text(encoding="utf-8") == "reserved\n"


def test_rejects_base_mutation_during_build(tmp_path, monkeypatch):
    database, _ = _build_database(tmp_path / "canon.sqlite")
    base = tmp_path / "base.jsonl"
    _write_jsonl(base, [_base_accounting_policy_task()])
    output = tmp_path / "audit-narrative.jsonl"
    original_read_base_tasks = audit_narrative_module._read_base_tasks

    def read_then_mutate(path):
        rows, count = original_read_base_tasks(path)
        path.write_bytes(path.read_bytes() + b"\n")
        return rows, count

    monkeypatch.setattr(audit_narrative_module, "_read_base_tasks", read_then_mutate)
    with pytest.raises(RuntimeError, match="Base narrative JSONL changed"):
        build_us_audit_narrative_tasks(database, output, base_narrative_jsonl=base)

    assert not output.exists()
    assert not output.with_suffix(output.suffix + ".manifest.json").exists()


def test_secret_scan_prevents_narrative_file_publication(tmp_path, monkeypatch):
    synthetic_secret = "synthetic-narrative-secret-123456"
    monkeypatch.setenv("HF_TOKEN", synthetic_secret)
    database, _ = _build_database(tmp_path / "canon.sqlite")
    base = tmp_path / "base.jsonl"
    base_task = _base_accounting_policy_task()
    base_task["question"] = synthetic_secret
    _write_jsonl(base, [base_task])
    output = tmp_path / "audit-narrative.jsonl"

    with pytest.raises(ValueError, match="Secret scan failed"):
        build_us_audit_narrative_tasks(database, output, base_narrative_jsonl=base)

    assert not output.exists()
    assert not output.with_suffix(output.suffix + ".manifest.json").exists()


@pytest.mark.parametrize(
    "contents, message",
    [
        ("{not json}\n", "Invalid JSON"),
        ('{"task_id":"incomplete"}\n', "Invalid base narrative task"),
    ],
)
def test_rejects_malformed_base_jsonl(tmp_path, contents, message):
    database, _ = _build_database(tmp_path / "canon.sqlite")
    base = tmp_path / "base.jsonl"
    base.write_text(contents, encoding="utf-8")
    output = tmp_path / "output.jsonl"

    with pytest.raises(ValueError, match=message):
        build_us_audit_narrative_tasks(database, output, base_narrative_jsonl=base)
    assert not output.exists()


def test_rejects_missing_canonical_schema(tmp_path):
    database = tmp_path / "broken.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE filings (
              filing_id TEXT PRIMARY KEY,
              ticker TEXT,
              form_type TEXT,
              report_date TEXT
            );
            CREATE TABLE chunk_canon (
              chunk_evidence_id TEXT PRIMARY KEY,
              filing_id TEXT NOT NULL
            );
            """
        )
        connection.commit()
    finally:
        connection.close()

    output = tmp_path / "output.jsonl"
    with pytest.raises(ValueError, match="missing columns"):
        build_us_audit_narrative_tasks(database, output)
    assert not output.exists()


def test_rejects_conflicting_duplicate_base_task_ids(tmp_path):
    database, _ = _build_database(tmp_path / "canon.sqlite")
    first = _base_accounting_policy_task()
    second = {**first, "question": "A conflicting question"}
    base = tmp_path / "base.jsonl"
    _write_jsonl(base, [first, second])

    with pytest.raises(ValueError, match="Conflicting duplicate task_id"):
        build_us_audit_narrative_tasks(
            database, tmp_path / "output.jsonl", base_narrative_jsonl=base
        )
