from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

import auditops.agent_evidence as agent_evidence_module
from auditops.agent_benchmark import _build_case
from auditops.agent_evidence import materialize_agent_source_evidence
from auditops.agent_tools import evaluate_quant_evidence
from auditops.pipeline import connect_db, process_zip
from auditops.tasks import build_task_specs

from .conftest import build_fixture_zip


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _build_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE facts_canon (
              canon_id TEXT PRIMARY KEY,
              fact_evidence_id TEXT NOT NULL,
              filing_id TEXT NOT NULL,
              concept_norm TEXT NOT NULL,
              period_key TEXT NOT NULL,
              unit_canon TEXT,
              value_num_exact TEXT,
              value_text TEXT,
              source_anchor TEXT,
              context_id TEXT,
              entity_identifier TEXT,
              dimensions_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE filings (
              filing_id TEXT PRIMARY KEY,
              ticker TEXT
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
            "INSERT INTO filings(filing_id, ticker) VALUES (?, ?)",
            [("filing-a", "FIX"), ("filing-b", "OTHER")],
        )
        connection.executemany(
            """
            INSERT INTO facts_canon(
              canon_id, fact_evidence_id, filing_id, concept_norm, period_key,
              unit_canon, value_num_exact, value_text, source_anchor,
              context_id, entity_identifier, dimensions_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "canon-a",
                    "fact-a",
                    "filing-a",
                    "AssetsCurrent",
                    "instant:2025-12-31",
                    "USD",
                    "120",
                    None,
                    "report.htm#fact-a",
                    "ctx-2025",
                    "0000000001",
                    "{}",
                ),
                (
                    "canon-b",
                    "fact-b",
                    "filing-a",
                    "LiabilitiesCurrent",
                    "instant:2025-12-31",
                    "USD",
                    "60",
                    None,
                    "report.htm#fact-b",
                    "ctx-2025",
                    "0000000001",
                    "{}",
                ),
            ],
        )
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
                    "chunk-revenue",
                    "filing-a",
                    "FY2025",
                    "8",
                    "Financial Statements",
                    "Revenue Recognition",
                    "Item 8 > Financial Statements > Revenue Recognition",
                    "report.htm",
                    10,
                    90,
                    "revenue recognition contract liabilities",
                    "Revenue is recognized when control transfers to the customer.",
                ),
                (
                    "chunk-contract",
                    "filing-a",
                    "FY2025",
                    "8",
                    "Financial Statements",
                    "Revenue Recognition",
                    "Item 8 > Financial Statements > Revenue Recognition",
                    "report.htm",
                    100,
                    180,
                    "remaining performance obligations contracts",
                    "Contract liabilities are included in deferred revenue.",
                ),
                (
                    "chunk-inventory",
                    "filing-a",
                    "FY2025",
                    "8",
                    "Financial Statements",
                    "Inventory",
                    "Item 8 > Financial Statements > Inventory",
                    "report.htm",
                    200,
                    270,
                    "inventory valuation obsolescence",
                    "Inventory is measured at the lower of cost and net realizable value.",
                ),
                (
                    "chunk-other-filing",
                    "filing-b",
                    "FY2025",
                    "8",
                    "Financial Statements",
                    "Revenue Recognition",
                    "Item 8 > Financial Statements > Revenue Recognition",
                    "other.htm",
                    1,
                    80,
                    "revenue recognition revenue recognition",
                    "Revenue recognition in another filing must remain out of scope.",
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return path


def _build_corpus_layout(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    corpus_root = tmp_path / "corpus"
    database = _build_db(corpus_root / "db" / "corpus.sqlite")
    paths = {
        "source_manifest": corpus_root / "manifest" / "source_manifest.json",
        "download_ledger": corpus_root / "manifest" / "download_ledger.jsonl",
        "ingest_ledger": corpus_root / "logs" / "ingest_ledger.jsonl",
    }
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"artifact": name}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return database, paths


def _quant_task(*, evidence_ids=("fact-b", "fact-a")):
    return {
        "task_id": "quant-1",
        "task_type": "quant_metric",
        "filing_id": "filing-a",
        "metric_spec_id": "current_ratio",
        "target_answer": {"status": "OK", "value": "2"},
        "canonical_inputs": [
            {
                "input_name": "current_liabilities",
                "concept_norm": "LiabilitiesCurrent",
                "fact_evidence_ids": [evidence_ids[0]],
            },
            {
                "input_name": "current_assets",
                "concept_norm": "AssetsCurrent",
                "fact_evidence_ids": list(evidence_ids[1:]),
            },
        ],
    }


def _narrative_task(**overrides):
    row = {
        "task_id": "narrative-1",
        "task_type": "narrative_citation",
        "filing_id": "filing-a",
        "question": "What is the revenue recognition policy?",
        "retrieval_query": "revenue recognition control transfer",
        "scope_type": "note",
        "scope_key": "note:filing-a:item 8 > financial statements > revenue recognition",
        "answerability": "ANSWERABLE",
        "expected_chunk_ids": ["chunk-revenue"],
        "extractive_answer": "Revenue is recognized when control transfers.",
    }
    row.update(overrides)
    return row


def _materialize(
    tmp_path: Path,
    *,
    quant_rows=None,
    narrative_rows=None,
    output_name="output",
    db_path=None,
    source_system="SEC-EDGAR",
):
    quant_path = tmp_path / f"{output_name}-quant.jsonl"
    narrative_path = tmp_path / f"{output_name}-narrative.jsonl"
    _write_jsonl(quant_path, [_quant_task()] if quant_rows is None else quant_rows)
    _write_jsonl(
        narrative_path,
        [_narrative_task()] if narrative_rows is None else narrative_rows,
    )
    database = db_path or _build_db(tmp_path / f"{output_name}.sqlite")
    return materialize_agent_source_evidence(
        quant_path,
        narrative_path,
        database,
        tmp_path / output_name,
        source_system=source_system,
    )


def test_materializes_canonical_facts_and_scoped_bm25_evidence(tmp_path):
    summary = _materialize(tmp_path)

    quant = _read_jsonl(tmp_path / "output" / "enriched_quant.jsonl")[0]
    narrative = _read_jsonl(tmp_path / "output" / "enriched_narrative.jsonl")[0]

    assert [item["evidence_id"] for item in quant["evidence_items"]] == [
        "fact-b",
        "fact-a",
    ]
    assert quant["source_entity_id"] == "0000000001"
    assert narrative["source_entity_id"] == "0000000001"
    first_fact = quant["evidence_items"][0]
    assert first_fact == {
        "evidence_id": "fact-b",
        "filing_id": "filing-a",
        "content": (
            "input_name=current_liabilities; concept=LiabilitiesCurrent; "
            "period_key=instant:2025-12-31; unit=USD; value=60"
        ),
        "rank": 1,
        "period_key": "instant:2025-12-31",
        "unit": "USD",
        "value": "60",
        "source_system": "SEC-EDGAR",
        "metadata": {
            "input_name": "current_liabilities",
            "concept": "LiabilitiesCurrent",
            "source_anchor": "report.htm#fact-b",
            "context_id": "ctx-2025",
            "entity_id": "0000000001",
            "dimensions": {},
        },
    }
    assert [item["evidence_id"] for item in narrative["evidence_items"]] == [
        "chunk-revenue",
        "chunk-contract",
    ]
    assert (
        narrative["retrieval_results"][0]["bm25_score"]
        > narrative["retrieval_results"][1]["bm25_score"]
    )
    assert "chunk-inventory" not in {
        item["evidence_id"] for item in narrative["retrieval_results"]
    }
    assert "chunk-other-filing" not in {
        item["evidence_id"] for item in narrative["retrieval_results"]
    }
    assert quant["target_answer"] == {"status": "OK", "value": "2"}
    assert summary["counts"] == {
        "quant": 1,
        "narrative": 1,
        "total": 2,
        "quant_evidence_items": 2,
        "narrative_evidence_items": 2,
    }
    for artifact in summary["artifacts"].values():
        assert len(artifact["sha256"]) == 64
        assert artifact["records"] == 1

    manifest_path = tmp_path / "output" / "materialization_manifest.json"
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted_manifest == summary
    assert manifest_path.read_bytes() == (
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert summary["sources"]["quant_jsonl"]["records"] == 1
    assert summary["sources"]["narrative_jsonl"]["records"] == 1
    assert len(summary["sources"]["database"]["sha256"]) == 64
    assert summary["parameters"]["source_system"] == "SEC-EDGAR"
    assert quant["source_system"] == "SEC-EDGAR"
    assert narrative["source_system"] == "SEC-EDGAR"


def test_source_entity_identity_is_authoritative_for_all_task_types(tmp_path):
    result = _materialize(
        tmp_path,
        quant_rows=[{**_quant_task(), "source_entity_id": "FIX"}],
        narrative_rows=[{**_narrative_task(), "source_entity_id": "FIX"}],
        output_name="authoritative-entity",
    )
    quant = _read_jsonl(Path(result["output_dir"]) / "enriched_quant.jsonl")[0]
    narrative = _read_jsonl(Path(result["output_dir"]) / "enriched_narrative.jsonl")[0]

    assert quant["source_entity_id"] == "0000000001"
    assert narrative["source_entity_id"] == "0000000001"


@pytest.mark.parametrize("ambiguous", [False, True])
def test_narrative_requires_one_unambiguous_canonical_source_entity(
    tmp_path, ambiguous
):
    database = _build_db(tmp_path / "entity.sqlite")
    filing_id = "filing-b"
    if ambiguous:
        filing_id = "filing-a"
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                """
                INSERT INTO facts_canon(
                  canon_id, fact_evidence_id, filing_id, concept_norm, period_key,
                  unit_canon, value_num_exact, value_text, source_anchor,
                  context_id, entity_identifier, dimensions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "canon-ambiguous",
                    "fact-ambiguous",
                    filing_id,
                    "AssetsCurrent",
                    "instant:2025-12-31",
                    "USD",
                    "120",
                    None,
                    "report.htm#fact-ambiguous",
                    "ctx-ambiguous",
                    "0000000002",
                    "{}",
                ),
            )
            connection.commit()
        finally:
            connection.close()

    with pytest.raises(ValueError, match="one unambiguous source entity identifier"):
        _materialize(
            tmp_path,
            quant_rows=[],
            narrative_rows=[
                _narrative_task(
                    filing_id=filing_id,
                    scope_type="filing",
                    scope_key=None,
                )
            ],
            output_name=f"entity-{ambiguous}",
            db_path=database,
        )


def test_cached_source_system_is_stamped_on_tasks_and_evidence(tmp_path):
    summary = _materialize(tmp_path, source_system="SEC-EDGAR-CACHED")
    quant = _read_jsonl(tmp_path / "output" / "enriched_quant.jsonl")[0]
    narrative = _read_jsonl(tmp_path / "output" / "enriched_narrative.jsonl")[0]

    assert summary["parameters"]["source_system"] == "SEC-EDGAR-CACHED"
    assert quant["source_system"] == "SEC-EDGAR-CACHED"
    assert narrative["source_system"] == "SEC-EDGAR-CACHED"
    assert {
        item["source_system"]
        for row in (quant, narrative)
        for item in row["evidence_items"]
    } == {"SEC-EDGAR-CACHED"}


def test_real_cross_accession_task_remains_same_issuer_and_verifiable(tmp_path):
    database = tmp_path / "real.sqlite"
    for index, fixture in enumerate(("filing_10k", "filing_q1", "filing_q2")):
        process_zip(
            str(build_fixture_zip(tmp_path, fixture)),
            str(database),
            extract_narrative=True,
            reset_db=index == 0,
        )
    connection = connect_db(str(database))
    try:
        source_task = next(
            row
            for row in build_task_specs(connection)
            if row["metric_spec_id"] == "qoq_revenue_growth"
            and row["filing_id"] == "acme-20250630"
            and row["period"]["period_key"] == "Q2_2025"
            and row["target_status"] == "OK"
        )
    finally:
        connection.close()

    quant_path = tmp_path / "real-quant.jsonl"
    narrative_path = tmp_path / "real-narrative.jsonl"
    _write_jsonl(quant_path, [source_task])
    _write_jsonl(narrative_path, [])
    materialize_agent_source_evidence(
        quant_path, narrative_path, database, tmp_path / "real-materialized"
    )
    enriched = _read_jsonl(tmp_path / "real-materialized" / "enriched_quant.jsonl")[0]
    evidence = enriched["evidence_items"]

    assert {item["filing_id"] for item in evidence} == {
        "acme-20250331",
        "acme-20250630",
    }
    assert len({item["metadata"]["entity_id"] for item in evidence}) == 1
    task = _build_case(
        enriched,
        "quant_metric",
        jurisdiction="US",
        corpus_id="fixture-real",
        source_system="SEC-EDGAR",
        reporting_framework="US-GAAP",
        standards_version="PCAOB-current",
        evidence_items=evidence,
    )
    assert set(task["evidence_scope"]["filing_ids"]) == {
        "acme-20250331",
        "acme-20250630",
    }
    observation = evaluate_quant_evidence(task, evidence)
    assert observation["status"] == "OK"
    assert observation["evidence_ids"]


def test_narrative_retrieval_is_independent_of_gold_fields(tmp_path):
    database = _build_db(tmp_path / "corpus.sqlite")
    first = _materialize(
        tmp_path,
        quant_rows=[],
        narrative_rows=[_narrative_task()],
        output_name="first",
        db_path=database,
    )
    second = _materialize(
        tmp_path,
        quant_rows=[],
        narrative_rows=[
            _narrative_task(
                expected_chunk_ids=["chunk-inventory", "invented-gold-id"],
                extractive_answer="A deliberately changed evaluator target.",
                gold={"preferred_chunk": "chunk-other-filing"},
                target_answer={"evidence_ids": ["chunk-other-filing"]},
            )
        ],
        output_name="second",
        db_path=database,
    )

    first_row = _read_jsonl(Path(first["output_dir"]) / "enriched_narrative.jsonl")[0]
    second_row = _read_jsonl(Path(second["output_dir"]) / "enriched_narrative.jsonl")[0]
    assert first_row["evidence_items"] == second_row["evidence_items"]
    assert first_row["retrieval_results"] == second_row["retrieval_results"]


def test_refuses_overwrite_and_partial_quant_evidence(tmp_path):
    _materialize(tmp_path)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _materialize(tmp_path, db_path=tmp_path / "output.sqlite")

    too_many = _quant_task(evidence_ids=("fact-b", "fact-a", "x", "y", "z", "six"))
    with pytest.raises(ValueError, match="Refusing to emit partial evidence"):
        _materialize(
            tmp_path,
            quant_rows=[too_many],
            narrative_rows=[],
            output_name="too-many",
        )


def test_fails_clearly_for_missing_database_table_and_evidence_id(tmp_path):
    quant_path = tmp_path / "quant.jsonl"
    narrative_path = tmp_path / "narrative.jsonl"
    _write_jsonl(quant_path, [_quant_task()])
    _write_jsonl(narrative_path, [])

    with pytest.raises(FileNotFoundError, match="Canonical database"):
        materialize_agent_source_evidence(
            quant_path,
            narrative_path,
            tmp_path / "missing.sqlite",
            tmp_path / "missing-output",
        )

    empty_db = tmp_path / "empty.sqlite"
    sqlite3.connect(empty_db).close()
    with pytest.raises(ValueError, match="missing required table 'facts_canon'"):
        materialize_agent_source_evidence(
            quant_path,
            narrative_path,
            empty_db,
            tmp_path / "empty-output",
        )

    database = _build_db(tmp_path / "corpus.sqlite")
    _write_jsonl(quant_path, [_quant_task(evidence_ids=("missing-fact",))])
    with pytest.raises(
        ValueError, match="missing facts_canon evidence ID 'missing-fact'"
    ):
        materialize_agent_source_evidence(
            quant_path,
            narrative_path,
            database,
            tmp_path / "missing-id-output",
        )


def test_falls_back_to_question_and_validates_retrieval_limits(tmp_path):
    row = _narrative_task(retrieval_query=None, scope_type="filing", scope_key=None)
    result = _materialize(
        tmp_path,
        quant_rows=[],
        narrative_rows=[row],
        output_name="fallback",
    )
    enriched = _read_jsonl(Path(result["output_dir"]) / "enriched_narrative.jsonl")[0]
    assert enriched["retrieval_results"][0]["evidence_id"] == "chunk-revenue"

    with pytest.raises(ValueError, match="top_k"):
        materialize_agent_source_evidence(
            tmp_path / "fallback-quant.jsonl",
            tmp_path / "fallback-narrative.jsonl",
            tmp_path / "fallback.sqlite",
            tmp_path / "invalid-top-k",
            top_k=6,
        )
    with pytest.raises(ValueError, match="candidate_k"):
        materialize_agent_source_evidence(
            tmp_path / "fallback-quant.jsonl",
            tmp_path / "fallback-narrative.jsonl",
            tmp_path / "fallback.sqlite",
            tmp_path / "invalid-candidate-k",
            top_k=5,
            candidate_k=4,
        )


def test_binds_complete_corpus_provenance_chain(tmp_path):
    database, upstream_paths = _build_corpus_layout(tmp_path)
    summary = _materialize(tmp_path, db_path=database)

    provenance = summary["sources"]["corpus_provenance"]
    assert provenance["missing_artifacts"] == []
    assert provenance["required_artifacts"] == [
        "download_ledger",
        "ingest_ledger",
        "source_manifest",
    ]
    for name, path in upstream_paths.items():
        assert (
            provenance["artifacts"][name]["sha256"]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_rejects_incomplete_corpus_provenance_chain(tmp_path):
    corpus_root = tmp_path / "corpus"
    database = _build_db(corpus_root / "db" / "corpus.sqlite")
    quant_path = tmp_path / "quant.jsonl"
    narrative_path = tmp_path / "narrative.jsonl"
    _write_jsonl(quant_path, [_quant_task()])
    _write_jsonl(narrative_path, [_narrative_task()])

    with pytest.raises(
        FileNotFoundError, match="provenance chain is incomplete"
    ) as exc_info:
        materialize_agent_source_evidence(
            quant_path, narrative_path, database, tmp_path / "materialized"
        )

    assert "source_manifest.json" in str(exc_info.value)
    assert "download_ledger.jsonl" in str(exc_info.value)
    assert "ingest_ledger.jsonl" in str(exc_info.value)


def test_rejects_corpus_provenance_mutation_during_materialization(
    tmp_path, monkeypatch
):
    database, upstream_paths = _build_corpus_layout(tmp_path)
    quant_path = tmp_path / "quant.jsonl"
    narrative_path = tmp_path / "narrative.jsonl"
    _write_jsonl(quant_path, [_quant_task()])
    _write_jsonl(narrative_path, [_narrative_task()])
    original_read_jsonl = agent_evidence_module._read_jsonl

    def read_then_mutate(path, label):
        rows = original_read_jsonl(path, label)
        if path == quant_path:
            source_manifest = upstream_paths["source_manifest"]
            source_manifest.write_bytes(source_manifest.read_bytes() + b"\n")
        return rows

    monkeypatch.setattr(agent_evidence_module, "_read_jsonl", read_then_mutate)
    with pytest.raises(RuntimeError, match="Corpus provenance source_manifest changed"):
        materialize_agent_source_evidence(
            quant_path, narrative_path, database, tmp_path / "materialized"
        )

    assert not (tmp_path / "materialized").exists()


def test_rejects_source_mutation_during_materialization(tmp_path, monkeypatch):
    quant_path = tmp_path / "quant.jsonl"
    narrative_path = tmp_path / "narrative.jsonl"
    database = _build_db(tmp_path / "corpus.sqlite")
    output = tmp_path / "materialized"
    _write_jsonl(quant_path, [_quant_task()])
    _write_jsonl(narrative_path, [_narrative_task()])

    original_read_jsonl = agent_evidence_module._read_jsonl

    def read_then_mutate(path, label):
        rows = original_read_jsonl(path, label)
        if path == quant_path:
            path.write_bytes(path.read_bytes() + b"\n")
        return rows

    monkeypatch.setattr(agent_evidence_module, "_read_jsonl", read_then_mutate)
    with pytest.raises(RuntimeError, match="Quant JSONL changed"):
        materialize_agent_source_evidence(quant_path, narrative_path, database, output)

    assert not output.exists()


def test_secret_scan_prevents_evidence_directory_publication(tmp_path, monkeypatch):
    synthetic_secret = "synthetic-evidence-secret-123456"
    monkeypatch.setenv("HF_TOKEN", synthetic_secret)
    output = tmp_path / "secret-output"

    with pytest.raises(ValueError, match="Secret scan failed"):
        _materialize(
            tmp_path,
            quant_rows=[],
            narrative_rows=[_narrative_task(retrieval_query=synthetic_secret)],
            output_name=output.name,
        )

    assert not output.exists()
