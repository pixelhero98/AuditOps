from __future__ import annotations

import csv
import json
from pathlib import Path

import requests

import auditops.uk_corpus as uk_corpus
from auditops.corpus import ensure_corpus_layout
from auditops.tasks import read_jsonl
from auditops.uk_corpus import build_uk_manifest, download_uk_filings


def _write_uk_constituents_csv(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ticker",
                "company_name",
                "lei",
                "nsm_keyword",
                "issuer_country",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_build_uk_manifest_resolves_ftse_reports_and_prefers_tagged_annual(
    tmp_path, monkeypatch
):
    constituents_path = tmp_path / "ftse100.csv"
    _write_uk_constituents_csv(
        constituents_path,
        [
            {
                "ticker": "BNZL",
                "company_name": "Bunzl plc",
                "lei": "213800Q1Q9DV4L78UM09",
                "nsm_keyword": "",
                "issuer_country": "UK",
            },
            {
                "ticker": "REL",
                "company_name": "RELX PLC",
                "lei": "",
                "nsm_keyword": "RELX PLC",
                "issuer_country": "UK",
            },
        ],
    )

    annual_hits = [
        {
            "_source": {
                "company": "BUNZL PUBLIC LIMITED COMPANY;",
                "disclosure_id": "NI-ANNUAL-TAGGED",
                "document_date": "2025-12-31",
                "download_link": "NSM/DirectUpload/NI-ANNUAL-TAGGED/report.zip",
                "headline": "Annual Report and Accounts 2025 (ESEF)",
                "html_link": "NSM/DirectUpload/NI-ANNUAL-TAGGED/report.xhtml",
                "lei": "213800Q1Q9DV4L78UM09",
                "publication_date": "2026-03-17T12:37:00.000Z",
                "ProcessType": "ESEF",
                "seq_id": "NI-ANNUAL-TAGGED",
                "source": "Direct Upload",
                "tag_esef": "Tagged",
                "type": "Annual Financial Report",
            }
        },
        {
            "_source": {
                "company": "BUNZL PUBLIC LIMITED COMPANY;",
                "disclosure_id": "NI-ANNUAL-PDF",
                "document_date": "2025-12-31",
                "download_link": "NSM/DirectUpload/NI-ANNUAL-PDF/report.pdf",
                "headline": "Annual Report and Accounts 2025",
                "html_link": "",
                "lei": "213800Q1Q9DV4L78UM09",
                "publication_date": "2026-03-17T12:37:00.000Z",
                "ProcessType": "NSM",
                "seq_id": "NI-ANNUAL-PDF",
                "source": "Direct Upload",
                "tag_esef": "",
                "type": "Annual Financial Report",
            }
        },
        {
            "_source": {
                "company": "RELX PLC;",
                "disclosure_id": "NI-REL-ANNUAL",
                "document_date": "2025-12-31",
                "download_link": "NSM/DirectUpload/NI-REL-ANNUAL/report.zip",
                "headline": "Annual Report and Accounts 2025 (ESEF)",
                "html_link": "NSM/DirectUpload/NI-REL-ANNUAL/report.xhtml",
                "lei": "549300WSM0VTB7JZAN22",
                "publication_date": "2026-03-20T09:00:00.000Z",
                "ProcessType": "ESEF",
                "seq_id": "NI-REL-ANNUAL",
                "source": "Direct Upload",
                "tag_esef": "Tagged",
                "type": "Annual Financial Report",
            }
        },
    ]
    half_year_hits = [
        {
            "_source": {
                "company": "BUNZL PUBLIC LIMITED COMPANY;",
                "disclosure_id": "NI-HALF-PDF",
                "document_date": "2025-06-30",
                "download_link": "NSM/DirectUpload/NI-HALF-PDF/report.pdf",
                "headline": "Half-year Financial Report",
                "html_link": "",
                "lei": "213800Q1Q9DV4L78UM09",
                "publication_date": "2025-08-25T07:00:00.000Z",
                "ProcessType": "NSM",
                "seq_id": "NI-HALF-PDF",
                "source": "Direct Upload",
                "tag_esef": "",
                "type": "Half-year Financial Report",
            }
        }
    ]

    def fake_search(session, keyword, **kwargs):
        if "Half-year Financial Report" in keyword:
            return half_year_hits
        if keyword == "213800Q1Q9DV4L78UM09":
            return annual_hits + half_year_hits
        if "RELX PLC Annual Financial Report" in keyword:
            return annual_hits
        return []

    monkeypatch.setattr(uk_corpus, "_get_session", lambda user_agent=None: object())
    monkeypatch.setattr(uk_corpus, "_nsm_search", fake_search)

    summary = build_uk_manifest(
        constituents_path=str(constituents_path),
        corpus_root=str(tmp_path / "uk_corpus"),
        snapshot_date="2026-03-21",
        constituent_source="unit-test",
    )

    layout = ensure_corpus_layout(tmp_path / "uk_corpus")
    issuer_rows = read_jsonl(layout.manifest / "issuer_manifest.jsonl")
    filing_rows = read_jsonl(layout.manifest / "filing_manifest.jsonl")

    assert summary["snapshot_id"] == "uk_ftse100_nsm_latest_2026-03-21"
    assert summary["issuer_count"] == 2
    assert summary["resolved_filing_count"] == 3
    assert summary["annual_resolved_count"] == 2
    assert summary["half_yearly_resolved_count"] == 1

    bunzl_annual = next(
        row
        for row in filing_rows
        if row["ticker"] == "BNZL" and row["report_type"] == "annual_financial_report"
    )
    assert bunzl_annual["disclosure_id"] == "NI-ANNUAL-TAGGED"
    assert bunzl_annual["tag_esef"] == "Tagged"
    assert bunzl_annual["resolution_kind"] == "lei_exact"

    bunzl_half = next(
        row
        for row in filing_rows
        if row["ticker"] == "BNZL"
        and row["report_type"] == "half_yearly_financial_report"
    )
    assert bunzl_half["status"] == "resolved"
    assert bunzl_half["disclosure_id"] == "NI-HALF-PDF"

    rel_half = next(
        row
        for row in filing_rows
        if row["ticker"] == "REL"
        and row["report_type"] == "half_yearly_financial_report"
    )
    assert rel_half["status"] == "missing_latest_report"

    assert issuer_rows[0]["source_system"] == "FCA_NSM"
    assert issuer_rows[0]["forms_expected"] == [
        "annual_financial_report",
        "half_yearly_financial_report",
    ]


def test_download_uk_filings_archives_details_and_sets_local_path_only_for_zip(
    tmp_path, monkeypatch
):
    layout = ensure_corpus_layout(tmp_path / "uk_corpus")
    filing_rows = [
        {
            "snapshot_id": "uk_ftse100_nsm_latest_2026-03-21",
            "snapshot_date": "2026-03-21",
            "ticker": "BNZL",
            "company_name": "Bunzl plc",
            "lei": "213800Q1Q9DV4L78UM09",
            "issuer_country": "UK",
            "source_system": "FCA_NSM",
            "report_type": "annual_financial_report",
            "form_type": "UK_AFR",
            "nsm_type": "Annual Financial Report",
            "disclosure_id": "NI-ANNUAL-TAGGED",
            "seq_id": "NI-ANNUAL-TAGGED",
            "filed_at": "2026-03-17T12:37:00.000Z",
            "document_date": "2025-12-31",
            "publication_date": "2026-03-17T12:37:00.000Z",
            "headline": "Annual Report and Accounts 2025 (ESEF)",
            "download_link": "NSM/DirectUpload/NI-ANNUAL-TAGGED/report.zip",
            "html_link": "NSM/DirectUpload/NI-ANNUAL-TAGGED/report.xhtml",
            "tag_esef": "Tagged",
            "process_type": "ESEF",
            "source": "Direct Upload",
            "resolution_kind": "lei_exact",
            "search_keyword_used": "213800Q1Q9DV4L78UM09",
            "status": "resolved",
            "error": None,
        },
        {
            "snapshot_id": "uk_ftse100_nsm_latest_2026-03-21",
            "snapshot_date": "2026-03-21",
            "ticker": "REL",
            "company_name": "RELX PLC",
            "lei": "549300WSM0VTB7JZAN22",
            "issuer_country": "UK",
            "source_system": "FCA_NSM",
            "report_type": "annual_financial_report",
            "form_type": "UK_AFR",
            "nsm_type": "Annual Financial Report",
            "disclosure_id": "NI-REL-PDF",
            "seq_id": "NI-REL-PDF",
            "filed_at": "2026-03-20T09:00:00.000Z",
            "document_date": "2025-12-31",
            "publication_date": "2026-03-20T09:00:00.000Z",
            "headline": "Annual Report and Accounts 2025",
            "download_link": "NSM/DirectUpload/NI-REL-PDF/report.pdf",
            "html_link": "",
            "tag_esef": "",
            "process_type": "NSM",
            "source": "Direct Upload",
            "resolution_kind": "company_exact",
            "search_keyword_used": "RELX PLC Annual Financial Report",
            "status": "resolved",
            "error": None,
        },
    ]
    uk_corpus.write_jsonl(layout.manifest / "filing_manifest.jsonl", filing_rows)

    monkeypatch.setattr(uk_corpus, "_get_session", lambda user_agent=None: object())
    monkeypatch.setattr(
        uk_corpus,
        "_fetch_nsm_details",
        lambda session, disclosure_id: {
            "_id": disclosure_id,
            "_source": {"disclosure_id": disclosure_id},
        },
    )

    def fake_download(session, asset_path, output_path):
        if asset_path.endswith(".zip"):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"zip-bytes")
            return 200, len(b"zip-bytes")
        response = requests.Response()
        response.status_code = 403
        raise requests.HTTPError("HTTP 403", response=response)

    monkeypatch.setattr(uk_corpus, "_download_nsm_asset", fake_download)

    summary = download_uk_filings(str(layout.root))
    ledger_rows = read_jsonl(layout.manifest / "download_ledger.jsonl")

    assert summary["details_success_count"] == 2
    assert summary["raw_success_count"] == 1
    assert summary["details_only_count"] == 1

    bunzl = next(row for row in ledger_rows if row["ticker"] == "BNZL")
    assert bunzl["status"] == "downloaded"
    assert bunzl["raw_status"] == "downloaded"
    assert bunzl["local_path"].endswith(".zip")
    assert Path(bunzl["details_path"]).exists()
    assert Path(bunzl["raw_local_path"]).read_bytes() == b"zip-bytes"

    rel = next(row for row in ledger_rows if row["ticker"] == "REL")
    assert rel["status"] == "details_only"
    assert rel["details_status"] == "downloaded"
    assert rel["raw_status"] == "download_error"
    assert rel["local_path"] is None
    assert Path(rel["details_path"]).exists()
    details_payload = json.loads(Path(rel["details_path"]).read_text(encoding="utf-8"))
    assert details_payload["_id"] == "NI-REL-PDF"


def test_nsm_search_uses_post_with_expected_payload(monkeypatch):
    calls = []

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"hits": {"hits": [{"_source": {"company": "RELX PLC;"}}]}}

    class DummySession:
        def post(self, url, params=None, json=None, timeout=None):
            calls.append(
                {
                    "url": url,
                    "params": params,
                    "json": json,
                    "timeout": timeout,
                }
            )
            return DummyResponse()

    monkeypatch.setenv("AUDITOPS_FCA_MIN_INTERVAL_SECONDS", "0")
    hits = uk_corpus._nsm_search(
        DummySession(), "RELX PLC Annual Financial Report", size=25
    )

    assert hits == [{"_source": {"company": "RELX PLC;"}}]
    assert len(calls) == 1
    assert calls[0]["url"] == uk_corpus.FCA_NSM_SEARCH_API
    assert calls[0]["params"] == {"index": uk_corpus.FCA_NSM_INDEX_NAME}
    assert calls[0]["json"]["keyword"] == "RELX PLC Annual Financial Report"
    assert calls[0]["json"]["size"] == 25


def test_candidate_matches_issuer_accepts_core_company_name_variants():
    admiral_issuer = {"company_name": "Admiral Group", "lei": None}
    admiral_source = {"company": "ADMIRAL GROUP PLC;", "lei": "213800FGVM7Z9EJB2685"}
    assert (
        uk_corpus._candidate_matches_issuer(admiral_source, admiral_issuer)
        == "company_core"
    )

    threei_issuer = {"company_name": "3i", "lei": None}
    threei_source = {"company": "3I GROUP PLC", "lei": "35GDVHRBMFE7NWATNM84"}
    assert (
        uk_corpus._candidate_matches_issuer(threei_source, threei_issuer)
        == "company_core"
    )
