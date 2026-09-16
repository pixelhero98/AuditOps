from __future__ import annotations

import hashlib
import zipfile
from datetime import date
from decimal import Decimal

import pytest

from auditops.metrics import generate_answer_objects
from auditops.narrative import build_narrative_chunks, extract_narrative_text
from auditops.pipeline import (
    FilingMetadata,
    _classify_period,
    connect_db,
    ensure_schema,
    infer_filing_id,
    load_zip_members,
    pick_main_html,
    process_zip,
)
from auditops.tasks import build_task_specs

from .conftest import build_fixture_zip


def test_infer_filing_id_recognizes_sec_and_corpus_zip_names():
    accession = "0000320193-25-000079"

    assert infer_filing_id(f"{accession}-xbrl.zip", "issuer.htm") == accession
    assert (
        infer_filing_id(
            f"AAPL_10-K_{accession}.zip",
            "aapl-20250927.htm",
        )
        == accession
    )
    assert infer_filing_id("fixture.zip", "issuer-report.htm") == "issuer-report"


def test_schema_migration_adds_filing_lineage_without_losing_rows(tmp_path):
    db_path = tmp_path / "legacy.sqlite"
    conn = connect_db(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE filings (
              filing_id TEXT PRIMARY KEY,
              ticker TEXT,
              zip_name TEXT,
              main_html TEXT,
              processed_at TEXT,
              form_type TEXT,
              fiscal_year_focus INTEGER,
              fiscal_period_focus TEXT,
              report_date TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO filings VALUES (
              'legacy-filing', 'OLD', 'old.zip', 'old.htm',
              '2026-08-21T00:00:00+00:00', '10-K', 2025, 'FY', '2025-12-31'
            )
            """
        )
        conn.commit()

        ensure_schema(conn)

        filing = conn.execute(
            "SELECT * FROM filings WHERE filing_id='legacy-filing'"
        ).fetchone()
        assert filing["ticker"] == "OLD"
        assert filing["cik"] is None
        assert filing["accession"] is None
        assert filing["source_sha256"] is None
        assert filing["source_size_bytes"] is None
        assert filing["taxonomy_refs_json"] == "[]"
        assert filing["taxonomy_identifiers_json"] == "[]"
    finally:
        conn.close()


def test_process_zip_persists_and_exposes_sec_lineage(tmp_path):
    source_zip = build_fixture_zip(tmp_path, "filing_10k")
    accession = "0000320193-25-000079"
    sec_zip = tmp_path / f"AAPL_10-K_{accession}.zip"
    schema_ref = "https://www.example.com/aapl/2025/aapl-20250927.xsd"
    with (
        zipfile.ZipFile(source_zip, "r") as source,
        zipfile.ZipFile(sec_zip, "w") as target,
    ):
        for member in source.infolist():
            payload = source.read(member.filename)
            if member.filename == "acme-20241231.htm":
                payload = payload.replace(b"0000000001", b"0000320193")
                payload = payload.replace(
                    b"  <body>",
                    (
                        f'  <head><schemaRef href="{schema_ref}" /></head>\n  <body>'
                    ).encode("utf-8"),
                )
            target.writestr(member, payload)

    source_sha256 = hashlib.sha256(sec_zip.read_bytes()).hexdigest()
    source_size_bytes = sec_zip.stat().st_size
    db_path = tmp_path / "lineage.sqlite"
    summary = process_zip(
        str(sec_zip),
        str(db_path),
        reset_db=True,
        metadata_overrides={
            "ticker": "AAPL",
            "cik": 320193,
            "accession": accession,
            "source_sha256": source_sha256,
            "source_size_bytes": source_size_bytes,
        },
    )

    assert summary["filing_id"] == accession
    assert summary["cik"] == "320193"
    assert summary["accession"] == accession
    assert summary["source_sha256"] == source_sha256
    assert summary["source_size_bytes"] == source_size_bytes
    assert summary["taxonomy_refs"] == [schema_ref]
    assert "http://fasb.org/us-gaap/2024" in summary["taxonomy_identifiers"]
    assert "http://xbrl.sec.gov/dei/2024" in summary["taxonomy_identifiers"]

    conn = connect_db(str(db_path))
    try:
        filing = conn.execute(
            "SELECT * FROM filings WHERE filing_id=?", (accession,)
        ).fetchone()
        assert filing["cik"] == "320193"
        assert filing["accession"] == accession
        assert filing["source_sha256"] == source_sha256
        assert filing["source_size_bytes"] == source_size_bytes

        task_metadata = build_task_specs(conn, filing_id=accession)[0][
            "filing_metadata"
        ]
        assert task_metadata["cik"] == "320193"
        assert task_metadata["accession"] == accession
        assert task_metadata["source_sha256"] == source_sha256
        assert task_metadata["source_size_bytes"] == source_size_bytes
        assert task_metadata["taxonomy_refs"] == [schema_ref]
        assert "http://fasb.org/us-gaap/2024" in task_metadata["taxonomy_identifiers"]
    finally:
        conn.close()


def test_process_zip_rejects_source_lineage_mismatch(tmp_path):
    zip_path = build_fixture_zip(tmp_path, "filing_q1")

    with pytest.raises(ValueError, match="source SHA-256"):
        process_zip(
            str(zip_path),
            str(tmp_path / "mismatch.sqlite"),
            metadata_overrides={"source_sha256": "0" * 64},
        )


def test_ingest_materializes_canonical_layers_and_chunks(tmp_path):
    db_path = tmp_path / "auditops.sqlite"
    zip_path = build_fixture_zip(tmp_path, "filing_10k")

    summary = process_zip(
        str(zip_path), str(db_path), extract_narrative=True, reset_db=True
    )

    assert summary["facts"] > 0
    conn = connect_db(str(db_path))
    try:
        filing = conn.execute(
            "SELECT * FROM filings WHERE filing_id='acme-20241231'"
        ).fetchone()
        assert filing["form_type"] == "10-K"
        assert filing["fiscal_year_focus"] == 2024
        assert filing["fiscal_period_focus"] == "FY"

        fact = conn.execute(
            "SELECT fact_evidence_id, source_anchor FROM facts WHERE filing_id='acme-20241231' AND concept_norm='us-gaap_Assets' AND period_key='ASOF_20241231' LIMIT 1"
        ).fetchone()
        assert fact["fact_evidence_id"].startswith("acme-20241231::")
        assert fact["source_anchor"] == "acme-20241231.htm#fact_assets_2024"

        chunk = conn.execute(
            """
          SELECT period_key, item, heading, subheading, heading_path, retrieval_text, text_masked, text_sha1
          FROM chunk_canon
          WHERE filing_id='acme-20241231'
          ORDER BY char_start
          LIMIT 1
          """
        ).fetchone()
        assert chunk["period_key"] == "FY2024"
        assert chunk["item"] == "1"
        assert chunk["heading"] == "Business"
        assert chunk["subheading"] == "Company Background"
        assert chunk["heading_path"] == "Item 1 > Business > Company Background"
        assert "Item: 1" in chunk["retrieval_text"]
        assert "Section: Business" in chunk["retrieval_text"]
        assert "Subsection: Company Background" in chunk["retrieval_text"]
        assert "<NUM>" in chunk["text_masked"]
        assert "Content:" in chunk["retrieval_text"]
        assert len(chunk["text_sha1"]) == 40

        assert (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='text_chunks'"
            ).fetchone()
            is None
        )
    finally:
        conn.close()


def test_stable_synthetic_evidence_id_for_idless_facts(tmp_path):
    zip_path = build_fixture_zip(tmp_path, "filing_q1")
    db_one = tmp_path / "one.sqlite"
    db_two = tmp_path / "two.sqlite"

    process_zip(str(zip_path), str(db_one), extract_narrative=False, reset_db=True)
    process_zip(str(zip_path), str(db_two), extract_narrative=False, reset_db=True)

    conn_one = connect_db(str(db_one))
    conn_two = connect_db(str(db_two))
    try:
        row_one = conn_one.execute(
            "SELECT fact_id, fact_evidence_id, source_anchor FROM facts WHERE filing_id='acme-20250331' AND concept_norm='us-gaap_OperatingIncomeLoss'"
        ).fetchone()
        row_two = conn_two.execute(
            "SELECT fact_id, fact_evidence_id, source_anchor FROM facts WHERE filing_id='acme-20250331' AND concept_norm='us-gaap_OperatingIncomeLoss'"
        ).fetchone()
        assert row_one["fact_id"] == row_two["fact_id"]
        assert row_one["fact_evidence_id"] == row_two["fact_evidence_id"]
        assert row_one["fact_id"].startswith("synthetic-")
        assert row_one["source_anchor"] is None
    finally:
        conn_one.close()
        conn_two.close()


def test_split_ixbrl_html_parts_share_contexts_and_units(tmp_path):
    db_path = tmp_path / "split.sqlite"
    zip_path = build_fixture_zip(tmp_path, "filing_split")

    summary = process_zip(
        str(zip_path), str(db_path), extract_narrative=False, reset_db=True
    )

    assert summary["facts"] >= 8
    conn = connect_db(str(db_path))
    try:
        filing = conn.execute(
            "SELECT * FROM filings WHERE filing_id='splitco-20241231'"
        ).fetchone()
        assert filing["ticker"] == "SPLT"

        revenue = conn.execute(
            "SELECT fact_id, source_file FROM facts WHERE filing_id='splitco-20241231' AND concept_norm='us-gaap_Revenues'"
        ).fetchone()
        cogs = conn.execute(
            "SELECT fact_id, source_file FROM facts WHERE filing_id='splitco-20241231' AND concept_norm='us-gaap_CostOfGoodsSold'"
        ).fetchone()
        net_income = conn.execute(
            "SELECT fact_id, source_file FROM facts WHERE filing_id='splitco-20241231' AND concept_norm='us-gaap_NetIncomeLoss'"
        ).fetchone()

        assert revenue["source_file"] == "splitco-20241231.htm"
        assert cogs["source_file"] == "splitco-20241231_d2.htm"
        assert net_income["source_file"] == "splitco-20241231_d2.htm"
    finally:
        conn.close()


def test_preferred_split_part_normalizes_to_base_main_html(tmp_path):
    zip_path = build_fixture_zip(tmp_path, "filing_split")
    members = load_zip_members(str(zip_path))

    assert (
        pick_main_html(members, preferred_name="splitco-20241231_d2.htm")
        == "splitco-20241231.htm"
    )


def test_period_types_and_validators(populated_db):
    conn = connect_db(str(populated_db))
    try:
        period_types = {
            row["period_type"]
            for row in conn.execute(
                "SELECT DISTINCT period_type FROM facts_canon"
            ).fetchall()
        }
        assert {"FY", "Q", "YTD", "ASOF"}.issubset(period_types)

        validator_codes = [
            row["validator_code"]
            for row in conn.execute(
                "SELECT validator_code FROM validators_v0 WHERE filing_id='acme-20250630' ORDER BY validator_code"
            ).fetchall()
        ]
        assert "UNIT_SCALE_MISMATCH" in validator_codes
        assert "CONTEXT_SELECTION_AMBIGUOUS" in validator_codes
    finally:
        conn.close()


def test_narrative_chunks_allow_repeated_item_chunk_indexes(tmp_path):
    db_path = tmp_path / "auditops.sqlite"
    zip_path = build_fixture_zip(tmp_path, "filing_q2")

    summary = process_zip(
        str(zip_path), str(db_path), extract_narrative=True, reset_db=True
    )

    conn = connect_db(str(db_path))
    try:
        raw_count = conn.execute(
            "SELECT COUNT(*) AS c FROM narrative_chunks WHERE filing_id='acme-20250630'"
        ).fetchone()["c"]
        canon_count = conn.execute(
            "SELECT COUNT(*) AS c FROM chunk_canon WHERE filing_id='acme-20250630'"
        ).fetchone()["c"]
        duplicate_keys = conn.execute(
            """
          SELECT item, chunk_index, COUNT(*) AS collisions
          FROM narrative_chunks
          WHERE filing_id='acme-20250630'
          GROUP BY item, chunk_index
          HAVING COUNT(*) > 1
          ORDER BY item, chunk_index
          """
        ).fetchall()

        assert raw_count == summary["narrative_chunks"]
        assert canon_count == summary["narrative_chunks"]
        assert duplicate_keys
    finally:
        conn.close()


def test_narrative_filters_headers_footers_and_structural_markers():
    html = """
    <html>
      <body>
        <h1>Item 1. Business</h1>
        <div>Example Corp. | 2025 Form 10-K | 2</div>
        <div>PART I</div>
        <h2>Company Background</h2>
        <p>Revenue was 100 and cash was 40.</p>
        <div>Item 1A</div>
        <div>(Continued)</div>
        <h2>Intellectual Property</h2>
        <p>Patents and trademarks are important.</p>
        <h1>ITEM1A. RISK FACTORS</h1>
        <div>(Unaudited)</div>
        <p>Supply chain disruptions could affect operations, create sourcing bottlenecks, and reduce customer demand across multiple regions.</p>
      </body>
    </html>
    """

    text = extract_narrative_text(html)
    assert "Example Corp. | 2025 Form 10-K | 2" not in text
    assert "PART I" not in text
    assert "(Continued)" not in text
    assert "(Unaudited)" not in text

    sections, chunks = build_narrative_chunks(text, chunk_chars=110, overlap=0)
    assert [section.item for section in sections] == ["1", "1A"]

    subheadings = {chunk.subheading for chunk in chunks if chunk.subheading}
    assert "Company Background" in subheadings
    assert "Intellectual Property" in subheadings
    assert "PART I" not in subheadings
    assert "Item 1A" not in subheadings
    assert "(Continued)" not in subheadings
    assert "(Unaudited)" not in subheadings
    assert all("Form 10-K" not in chunk.text_raw for chunk in chunks)


def test_off_calendar_comparative_fy_classification():
    filing = FilingMetadata(
        filing_id="aapl-like",
        ticker="AAPL",
        zip_name="aapl.zip",
        main_html="aapl.htm",
        processed_at="2026-03-17T00:00:00+00:00",
        form_type="10-K",
        fiscal_year_focus=2025,
        fiscal_period_focus="FY",
        report_date=date(2025, 9, 27),
    )

    current_period = _classify_period(
        filing, date(2024, 9, 29), date(2025, 9, 27), None
    )
    comparative_period = _classify_period(
        filing, date(2023, 10, 1), date(2024, 9, 28), None
    )

    assert current_period["period_key"] == "FY2025"
    assert comparative_period["period_key"] == "FY2024"


def test_answer_objects_end_to_end(db_conn):
    answers = generate_answer_objects(db_conn)

    current_ratio = next(
        answer
        for answer in answers
        if answer["metric_spec_id"] == "current_ratio"
        and answer["filing_id"] == "acme-20250630"
        and answer["period"]["period_key"] == "ASOF_20250630"
        and answer["status"] == "OK"
    )
    assert Decimal(current_ratio["result"]["value"]) == Decimal("2")
    assert current_ratio["evidence_ids"]

    yoy_revenue = next(
        answer
        for answer in answers
        if answer["metric_spec_id"] == "yoy_revenue_growth"
        and answer["filing_id"] == "acme-20250630"
        and answer["period"]["period_key"] == "Q2_2025"
        and answer["status"] == "OK"
    )
    assert Decimal(yoy_revenue["result"]["value"]).quantize(
        Decimal("0.000001")
    ) == Decimal("0.192308")

    qoq_revenue = next(
        answer
        for answer in answers
        if answer["metric_spec_id"] == "qoq_revenue_growth"
        and answer["filing_id"] == "acme-20250630"
        and answer["period"]["period_key"] == "Q2_2025"
        and answer["status"] == "OK"
    )
    assert Decimal(qoq_revenue["result"]["value"]).quantize(
        Decimal("0.000001")
    ) == Decimal("0.192308")

    operating_cash_flow_margin = next(
        answer
        for answer in answers
        if answer["metric_spec_id"] == "operating_cash_flow_margin"
        and answer["filing_id"] == "acme-20250630"
        and answer["period"]["period_key"] == "Q2_2025"
        and answer["status"] == "OK"
    )
    assert Decimal(operating_cash_flow_margin["result"]["value"]).quantize(
        Decimal("0.000001")
    ) == Decimal("0.241935")

    free_cash_flow = next(
        answer
        for answer in answers
        if answer["metric_spec_id"] == "free_cash_flow"
        and answer["filing_id"] == "acme-20241231"
        and answer["period"]["period_key"] == "FY2024"
        and answer["status"] == "OK"
    )
    assert Decimal(free_cash_flow["result"]["value"]) == Decimal("130")

    quality_of_earnings = next(
        answer
        for answer in answers
        if answer["metric_spec_id"] == "quality_of_earnings_ratio"
        and answer["filing_id"] == "acme-20241231"
        and answer["period"]["period_key"] == "FY2024"
        and answer["status"] == "OK"
    )
    assert Decimal(quality_of_earnings["result"]["value"]).quantize(
        Decimal("0.000001")
    ) == Decimal("1.400000")

    roa = next(
        answer
        for answer in answers
        if answer["metric_spec_id"] == "roa"
        and answer["filing_id"] == "acme-20241231"
        and answer["period"]["period_key"] == "FY2024"
        and answer["status"] == "OK"
    )
    assert Decimal(roa["result"]["value"]).quantize(Decimal("0.000001")) == Decimal(
        "0.111111"
    )

    equity_multiplier = next(
        answer
        for answer in answers
        if answer["metric_spec_id"] == "equity_multiplier"
        and answer["filing_id"] == "acme-20241231"
        and answer["period"]["period_key"] == "FY2024"
        and answer["status"] == "OK"
    )
    assert Decimal(equity_multiplier["result"]["value"]).quantize(
        Decimal("0.000001")
    ) == Decimal("2.160000")

    refusal = next(
        answer
        for answer in answers
        if answer["metric_spec_id"] == "current_assets_rollup"
        and answer["filing_id"] == "acme-20250630"
        and answer["period"]["period_key"] == "ASOF_20250630"
        and answer["status"] == "REFUSAL"
    )
    assert refusal["refusal_code"] == "MISSING_INPUT"
