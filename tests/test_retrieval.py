from __future__ import annotations

import pytest

pytest.importorskip("haystack")
from haystack import Document

from auditops.pipeline import connect_db
from auditops.retrieval import (
    RetrievalExample,
    _is_benchmarkable_row,
    _build_index_document_text,
    _build_query_from_chunk,
    _rerank_documents,
    build_retrieval_benchmark_examples,
    evaluate_bm25_retrieval,
)


def _chunk_id(conn, filing_id: str, chunk_evidence_id: str | None = None, heading_path: str | None = None, text_like: str | None = None):
    clauses = ["filing_id = ?"]
    params = [filing_id]
    if chunk_evidence_id is not None:
        clauses.append("chunk_evidence_id = ?")
        params.append(chunk_evidence_id)
    if heading_path is not None:
        clauses.append("heading_path = ?")
        params.append(heading_path)
    if text_like is not None:
        clauses.append("text_masked LIKE ?")
        params.append(f"%{text_like}%")
    row = conn.execute(
        f"SELECT chunk_evidence_id FROM chunk_canon WHERE {' AND '.join(clauses)} LIMIT 1",
        params,
    ).fetchone()
    assert row is not None
    return row["chunk_evidence_id"]


def test_bm25_retrieval_hits_expected_fixture_chunks(populated_db):
    conn = connect_db(str(populated_db))
    try:
        examples = [
            RetrievalExample(
                query="company background revenue current assets",
                expected_chunk_ids=[
                    _chunk_id(
                        conn,
                        "acme-20241231",
                        heading_path="Item 1 > Business > Company Background",
                    )
                ],
                filing_id="acme-20241231",
                label="company_background",
            ),
            RetrievalExample(
                query="quarterly revenue cash balances net income",
                expected_chunk_ids=[
                    _chunk_id(
                        conn,
                        "acme-20250331",
                        heading_path="Item 2 > Management's Discussion and Analysis",
                    )
                ],
                filing_id="acme-20250331",
                label="q1_mda",
            ),
            RetrievalExample(
                query="revenue recognition contract assets deferred revenue",
                expected_chunk_ids=[
                    _chunk_id(
                        conn,
                        "acme-20250630",
                        heading_path="Item 1 > Financial Statements > Revenue Recognition",
                    )
                ],
                filing_id="acme-20250630",
                label="revenue_recognition",
            ),
            RetrievalExample(
                query="liquidity capital resources cash equivalents capital expenditures operating cash flow",
                expected_chunk_ids=[
                    _chunk_id(
                        conn,
                        "acme-20250630",
                        heading_path="Item 2 > Management's Discussion and Analysis",
                        text_like="Liquidity and Capital Resources",
                    )
                ],
                filing_id="acme-20250630",
                label="liquidity",
            ),
        ]
    finally:
        conn.close()

    result = evaluate_bm25_retrieval(str(populated_db), examples, top_k=3)

    assert result["summary"]["query_count"] == 4
    assert result["summary"]["hit_rate_at_k"] == 1.0
    assert result["summary"]["top1_hit_rate"] == 1.0
    assert result["summary"]["mrr_at_k"] == 1.0


def test_benchmark_builder_produces_diverse_fixture_examples(populated_db):
    examples = build_retrieval_benchmark_examples(str(populated_db), limit=4, per_filing_limit=1)

    assert len(examples) == 3
    assert len({example.filing_id for example in examples}) == 3
    assert all(example.query for example in examples)
    assert all(example.expected_chunk_ids for example in examples)


def test_query_builder_skips_company_banner_and_note_boilerplate():
    row = {
        "item": "1",
        "heading": "Financial Statements (Unaudited)",
        "subheading": "THE GOLDMAN SACHS GROUP, INC.AND SUBSIDIARIES",
        "text_masked": (
            "THE GOLDMAN SACHS GROUP, INC. AND SUBSIDIARIES\n\n"
            "Notes to Consolidated Financial Statements\n\n"
            "For qualifying interest rate fair value hedges, gains or losses on derivatives are included in interest income/expense."
        ),
    }

    query = _build_query_from_chunk(row)

    assert query is not None
    lowered = query.lower()
    assert "goldman" not in lowered
    assert "subsidiaries" not in lowered
    assert "notes to consolidated financial statements" not in lowered
    assert "qualifying" in lowered
    assert "interest" in lowered
    assert "hedges" in lowered


def test_index_document_text_strips_table_of_contents_noise():
    row = {
        "item": "8",
        "heading": "Financial Statements and Supplementary Data.",
        "subheading": "Forged Wheels",
        "text_masked": (
            "Table of Contents\n\n"
            "(<NUM>) Segment Adjusted cost of goods sold is exclusive of Provision for depreciation and amortization.\n\n"
            "The following table reconciles Total segment capital expenditures with Capital expenditures as presented in the Statement of Consolidated Cash Flows."
        ),
    }

    content = _build_index_document_text(row)
    lowered = content.lower()

    assert lowered.startswith("item 8\nfinancial statements and supplementary data.\nforged wheels")
    assert "table of contents" not in lowered
    assert "segment adjusted" in lowered
    assert "capital expenditures" in lowered


def test_query_builder_skips_parenthetical_tabular_disclaimer():
    row = {
        "item": "1",
        "heading": "Condensed Consolidated Financial Statements",
        "subheading": "Notes to Condensed Consolidated Financial Statements",
        "text_masked": (
            "THE CHARLES SCHWAB CORPORATION\n\n"
            "Notes to Condensed Consolidated Financial Statements\n\n"
            "(Tabular Amounts in Millions, Except Per Share Data, Ratios, or as Noted)\n\n"
            "Legal contingencies: Schwab is subject to claims and lawsuits in the ordinary course of business."
        ),
    }

    query = _build_query_from_chunk(row)

    assert query is not None
    lowered = query.lower()
    assert "tabular amounts" not in lowered
    assert "per share" not in lowered
    assert "legal" in lowered
    assert "contingencies" in lowered


def test_generic_line_catches_compact_notes_header():
    row = {
        "item": "8",
        "heading": "FINANCIAL STATEMENTSAND SUPPLEMENTARY DATA",
        "subheading": "NOTESTO CONSOLIDATED FINANCIAL STATEMENTS- (Continued)",
        "text_masked": (
            "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS- (Continued)\n\n"
            "December <NUM>, <NUM> and <NUM>\n\n"
            "Hedge of Net Investment in Foreign Operations – The Company has no outstanding derivatives."
        ),
    }

    query = _build_query_from_chunk(row)

    assert query is not None
    lowered = query.lower()
    assert "notesto" not in lowered
    assert "hedge" in lowered
    assert "foreign" in lowered


def test_benchmarkable_row_rejects_cross_section_chunk():
    row = {
        "item": "1A",
        "heading": "Risk Factors",
        "subheading": "Risks Related to Our Contracts",
        "text_masked": (
            "Item 1B. Unresolved Staff Comments\n\n"
            "None.\n\n"
            "Item 1C. Cybersecurity\n\n"
            "Risk Management and Strategy\n\n"
            "Our cybersecurity strategy prioritizes detection, analysis and response."
        ),
    }

    assert _is_benchmarkable_row(row) is False


def test_benchmarkable_row_rejects_headingless_date_stub():
    row = {
        "item": None,
        "heading": None,
        "subheading": "September 30, 2025",
        "text_masked": (
            "Part I - Item <NUM>. Management's Discussion and Analysis of Financial Condition and Results of Operations\n\n"
            "Loss and loss adjustment expense ratio information follows."
        ),
    }

    assert _is_benchmarkable_row(row) is False


def test_reranker_prefers_matching_heading_and_subheading():
    documents = [
        Document(
            id="mda",
            content="Item 7\nManagement’s Discussion and Analysis\nForged Wheels\nThird-party sales for the Forged Wheels segment decreased.",
            meta={
                "heading": "Management’s Discussion and Analysis of Financial Condition and Results of Operations.",
                "subheading": "Forged Wheels",
                "lead_text": "Third-party sales for the Forged Wheels segment decreased.",
            },
            score=12.0,
        ),
        Document(
            id="footnote",
            content="Item 8\nFinancial Statements and Supplementary Data.\nForged Wheels\nSegment Adjusted cost of goods sold is exclusive of Provision for depreciation and amortization.",
            meta={
                "heading": "Financial Statements and Supplementary Data.",
                "subheading": "Forged Wheels",
                "lead_text": "Segment Adjusted cost of goods sold is exclusive of Provision for depreciation and amortization.",
            },
            score=9.0,
        ),
    ]

    reranked = _rerank_documents(
        "financial statements Forged Wheels segment adjusted cost goods sold exclusive",
        documents,
        top_k=2,
    )

    assert [document.id for document in reranked] == ["footnote", "mda"]


def test_eval_retrieval_reports_rerank_method(populated_db):
    conn = connect_db(str(populated_db))
    try:
        examples = [
            RetrievalExample(
                query="revenue recognition contract assets deferred revenue",
                expected_chunk_ids=[
                    _chunk_id(
                        conn,
                        "acme-20250630",
                        heading_path="Item 1 > Financial Statements > Revenue Recognition",
                    )
                ],
                filing_id="acme-20250630",
                label="revenue_recognition",
            ),
        ]
    finally:
        conn.close()

    result = evaluate_bm25_retrieval(str(populated_db), examples, top_k=3, method="bm25_rerank", candidate_k=9)

    assert result["summary"]["method"] == "bm25_rerank"
    assert result["summary"]["candidate_k"] == 9
    assert result["summary"]["top1_hit_rate"] == 1.0
