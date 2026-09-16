from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import auditops.bootstrap_inputs as bootstrap_inputs
import auditops.corpus as corpus
import auditops.uk_corpus as uk_corpus
from auditops.bootstrap_inputs import fetch_uk_constituents, fetch_us_constituents
from auditops.corpus import build_manifest, ensure_corpus_layout
from auditops.tasks import read_jsonl
from auditops.uk_corpus import build_uk_manifest

US_HTML = """
<html>
  <body>
    <table class="wikitable sortable">
      <tr>
        <th>Symbol</th>
        <th>Security</th>
        <th>GICS Sector</th>
      </tr>
      <tr><td>MMM</td><td>3M</td><td>Industrials</td></tr>
      <tr><td>AOS</td><td>A. O. Smith</td><td>Industrials</td></tr>
      <tr><td>ABT</td><td>Abbott Laboratories</td><td>Health Care</td></tr>
    </table>
  </body>
</html>
"""


UK_HTML = """
<html>
  <body>
    <table class="wikitable sortable">
      <tr>
        <th>Company</th>
        <th>Ticker</th>
        <th>FTSE industry classification benchmark sector</th>
      </tr>
      <tr><td>3i</td><td>III</td><td>Financial services</td></tr>
      <tr><td>Admiral Group</td><td>ADM</td><td>Insurance</td></tr>
      <tr><td>Airtel Africa</td><td>AAF</td><td>Telecommunications services</td></tr>
    </table>
  </body>
</html>
"""


class _FakeResponse:
    def __init__(self, text: str = "", payload: dict | None = None):
        self.text = text
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        assert self._payload is not None
        return self._payload


def _fake_us_as_of_get(*args, **kwargs):
    url = args[0]
    if url == bootstrap_inputs.WIKIPEDIA_API_URL:
        return _FakeResponse(
            payload={
                "query": {
                    "pages": [
                        {
                            "pageid": 2676045,
                            "title": bootstrap_inputs.S_AND_P_500_WIKI_TITLE,
                            "revisions": [
                                {
                                    "revid": 123456789,
                                    "timestamp": "2026-04-15T20:30:00Z",
                                }
                            ],
                        }
                    ]
                }
            }
        )
    assert url == bootstrap_inputs.WIKIPEDIA_OLDID_URL
    assert kwargs["params"]["oldid"] == "123456789"
    return _FakeResponse(US_HTML)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_fetch_us_constituents_writes_expected_schema_and_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap_inputs.requests, "get", _fake_us_as_of_get)

    output_path = tmp_path / "sp500.csv"
    summary = fetch_us_constituents(
        output_path,
        snapshot_date="2026-04-15",
        limit=2,
        user_agent="AuditOps test research@example.org",
    )
    rows = _read_csv(output_path)

    assert summary["row_count"] == 2
    assert summary["snapshot_date"] == "2026-04-15"
    assert summary["retrieved_at"].endswith("Z")
    assert summary["source_sha256"] == hashlib.sha256(US_HTML.encode()).hexdigest()
    source_manifest = json.loads(Path(summary["source_manifest"]).read_text())
    assert source_manifest["output_sha256"] == summary["output_sha256"]
    assert (
        source_manifest["source_manifest_version"]
        == "auditops-us-constituents-source.v2"
    )
    assert source_manifest["capture_scope"] == "limited"
    assert source_manifest["row_limit"] == 2
    assert source_manifest["revision_id"] == 123456789
    assert source_manifest["revision_timestamp"] == "2026-04-15T20:30:00Z"
    assert Path(summary["raw_source"]).read_text(encoding="utf-8") == US_HTML
    assert (
        source_manifest["raw_source_sha256"]
        == hashlib.sha256(US_HTML.encode()).hexdigest()
    )
    assert "research@example.org" not in Path(summary["source_manifest"]).read_text()
    assert rows == [
        {"ticker": "MMM", "company_name": "3M"},
        {"ticker": "AOS", "company_name": "A. O. Smith"},
    ]


def test_fetch_us_constituents_rejects_revision_newer_than_snapshot(
    tmp_path, monkeypatch
):
    def fake_get(*args, **kwargs):
        return _FakeResponse(
            payload={
                "query": {
                    "pages": [
                        {
                            "revisions": [
                                {"revid": 7, "timestamp": "2026-04-16T00:00:00Z"}
                            ]
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(bootstrap_inputs.requests, "get", fake_get)
    with pytest.raises(ValueError, match="newer than the requested snapshot"):
        fetch_us_constituents(
            tmp_path / "sp500.csv",
            snapshot_date="2026-04-15",
            user_agent="AuditOps test research@example.org",
        )


def test_fetch_uk_constituents_writes_expected_schema_and_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bootstrap_inputs.requests, "get", lambda *args, **kwargs: _FakeResponse(UK_HTML)
    )

    output_path = tmp_path / "ftse100.csv"
    summary = fetch_uk_constituents(output_path, snapshot_date="2026-04-15", limit=2)
    rows = _read_csv(output_path)

    assert summary["row_count"] == 2
    assert summary["snapshot_date"] == "2026-04-15"
    assert rows == [
        {
            "ticker": "III",
            "company_name": "3i",
            "lei": "",
            "nsm_keyword": "3i",
            "issuer_country": "UK",
        },
        {
            "ticker": "ADM",
            "company_name": "Admiral Group",
            "lei": "",
            "nsm_keyword": "Admiral Group",
            "issuer_country": "UK",
        },
    ]


def test_bootstrapped_us_csv_feeds_build_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap_inputs.requests, "get", _fake_us_as_of_get)
    sp500_path = tmp_path / "sp500.csv"
    fetch_us_constituents(
        sp500_path,
        snapshot_date="2026-04-15",
        limit=2,
        user_agent="AuditOps test research@example.org",
    )

    sec_tickers = {
        "MMM": {"ticker": "MMM", "company_name": "3M", "cik": 66740},
        "AOS": {"ticker": "AOS", "company_name": "A. O. Smith", "cik": 91142},
    }
    submissions = {
        66740: {
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q"],
                    "accessionNumber": ["mmm-k", "mmm-q"],
                    "filingDate": ["2026-02-20", "2025-10-25"],
                    "reportDate": ["2025-12-31", "2025-09-30"],
                    "primaryDocument": ["mmm-10k.htm", "mmm-10q.htm"],
                }
            }
        },
        91142: {
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q"],
                    "accessionNumber": ["aos-k", "aos-q"],
                    "filingDate": ["2026-02-21", "2025-11-01"],
                    "reportDate": ["2025-12-31", "2025-09-30"],
                    "primaryDocument": ["aos-10k.htm", "aos-10q.htm"],
                }
            }
        },
    }

    monkeypatch.setattr(corpus, "_get_session", lambda user_agent=None: object())
    monkeypatch.setattr(
        corpus, "_fetch_sec_company_tickers", lambda session: sec_tickers
    )
    monkeypatch.setattr(
        corpus, "_fetch_submission_json", lambda session, cik: submissions[cik]
    )

    summary = build_manifest(
        constituents_path=str(sp500_path),
        corpus_root=str(tmp_path / "corpus"),
        snapshot_date="2026-04-15",
        constituent_source="bootstrap-test",
        allow_limited_constituents=True,
    )

    layout = ensure_corpus_layout(tmp_path / "corpus")
    issuer_rows = read_jsonl(layout.manifest / "issuer_manifest.jsonl")
    filing_rows = read_jsonl(layout.manifest / "filing_manifest.jsonl")

    assert summary["issuer_count"] == 2
    assert summary["resolved_filing_count"] == 4
    source_manifest = json.loads(Path(summary["source_manifest"]).read_text())
    assert source_manifest["snapshot_date"] == "2026-04-15"
    assert {item["role"] for item in source_manifest["sources"]} == {
        "constituents_snapshot",
        "constituents_source_capture",
        "sec_company_tickers",
        "sec_submission",
    }
    source_capture = next(
        item
        for item in source_manifest["sources"]
        if item["role"] == "constituents_source_capture"
    )
    bootstrap_capture = json.loads(
        sp500_path.with_suffix(".csv.source.json").read_text(encoding="utf-8")
    )
    assert source_capture["retrieved_at"] == bootstrap_capture["retrieved_at"]
    assert source_capture["source_payload_sha256"] == bootstrap_capture["source_sha256"]
    assert {row["ticker"] for row in issuer_rows} == {"MMM", "AOS"}
    assert {row["form_type"] for row in filing_rows} == {"10-K", "10-Q"}


def test_bootstrapped_uk_csv_feeds_build_uk_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bootstrap_inputs.requests, "get", lambda *args, **kwargs: _FakeResponse(UK_HTML)
    )
    ftse_path = tmp_path / "ftse100.csv"
    fetch_uk_constituents(ftse_path, limit=2)

    annual_hits = [
        {
            "_source": {
                "company": "3I;",
                "disclosure_id": "III-AFR",
                "document_date": "2025-12-31",
                "download_link": "NSM/DirectUpload/III-AFR/report.zip",
                "headline": "Annual Report and Accounts 2025 (ESEF)",
                "html_link": "NSM/DirectUpload/III-AFR/report.xhtml",
                "lei": "",
                "publication_date": "2026-03-20T09:00:00.000Z",
                "ProcessType": "ESEF",
                "seq_id": "III-AFR",
                "source": "Direct Upload",
                "tag_esef": "Tagged",
                "type": "Annual Financial Report",
            }
        }
    ]
    half_year_hits = [
        {
            "_source": {
                "company": "3I;",
                "disclosure_id": "III-HYR",
                "document_date": "2025-06-30",
                "download_link": "NSM/DirectUpload/III-HYR/report.pdf",
                "headline": "Half-year Financial Report",
                "html_link": "",
                "lei": "",
                "publication_date": "2025-08-15T09:00:00.000Z",
                "ProcessType": "NSM",
                "seq_id": "III-HYR",
                "source": "Direct Upload",
                "tag_esef": "",
                "type": "Half-year Financial Report",
            }
        }
    ]

    def fake_search(session, keyword, **kwargs):
        if "Half-year Financial Report" in keyword:
            return half_year_hits
        if "Annual Financial Report" in keyword:
            return annual_hits
        return []

    monkeypatch.setattr(uk_corpus, "_get_session", lambda user_agent=None: object())
    monkeypatch.setattr(uk_corpus, "_nsm_search", fake_search)

    summary = build_uk_manifest(
        constituents_path=str(ftse_path),
        corpus_root=str(tmp_path / "uk_corpus"),
        snapshot_date="2026-04-15",
        constituent_source="bootstrap-test",
    )

    layout = ensure_corpus_layout(tmp_path / "uk_corpus")
    issuer_rows = read_jsonl(layout.manifest / "issuer_manifest.jsonl")
    filing_rows = read_jsonl(layout.manifest / "filing_manifest.jsonl")

    assert summary["issuer_count"] == 2
    assert summary["resolved_filing_count"] == 2
    assert issuer_rows[0]["source_system"] == "FCA_NSM"
    assert {row["ticker"] for row in filing_rows} == {"III", "ADM"}
