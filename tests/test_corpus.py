from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
import requests

import auditops.corpus as corpus
from auditops import cli as auditops_cli
from auditops.corpus import (
    _download_with_resume,
    _get_session,
    _issuer_split,
    build_manifest,
    download_filings,
    ensure_corpus_layout,
    eval_corpus,
    generate_corpus_datasets,
    ingest_corpus,
    repair_corpus_filing,
)
from auditops.tasks import read_jsonl, write_jsonl

from .conftest import build_fixture_zip


def test_sec_session_requires_contact_bearing_identity(monkeypatch):
    monkeypatch.delenv("AUDITOPS_SEC_USER_AGENT", raising=False)
    with pytest.raises(ValueError, match="contact email"):
        _get_session()
    with pytest.raises(ValueError, match="contact email"):
        _get_session("AuditOps contact omitted")

    session = _get_session("AuditOps test audit@example.org")
    assert session.headers["User-Agent"] == "AuditOps test audit@example.org"


@pytest.mark.parametrize(
    ("command", "summary", "attribute"),
    [
        (
            ["download-filings", "--corpus-root", "fixture"],
            {"download_ledger": "ledger.jsonl", "error_count": 1},
            "download_filings",
        ),
        (
            ["ingest-corpus", "--corpus-root", "fixture"],
            {"ingest_ledger": "ingest.jsonl", "error_count": 1, "published": False},
            "ingest_corpus",
        ),
    ],
)
def test_corpus_cli_returns_nonzero_but_prints_typed_error_summary(
    monkeypatch,
    capsys,
    command,
    summary,
    attribute,
):
    monkeypatch.setattr(auditops_cli, attribute, lambda *args, **kwargs: summary)

    assert auditops_cli.main(command) == 1
    assert json.loads(capsys.readouterr().out) == summary


def _write_constituents_csv(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "company_name"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class FakeResponse:
    def __init__(self, status_code: int, body: bytes = b"", headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = dict(headers or {})
        if status_code in {200, 206} and "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(body))

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def iter_content(self, chunk_size: int):
        for index in range(0, len(self._body), chunk_size):
            yield self._body[index : index + chunk_size]


class FakeSession:
    def __init__(self, outcomes):
        self._outcomes = {url: list(queue) for url, queue in outcomes.items()}
        self.calls = []

    def get(self, url, headers=None, stream=False, timeout=None, **kwargs):
        self.calls.append(
            {"url": url, "headers": dict(headers or {}), "kwargs": dict(kwargs)}
        )
        queue = self._outcomes[url]
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _zip_payload(name: str = "filing.txt", content: bytes = b"filing") -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, content)
    return payload.getvalue()


def _ledger_integrity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _sec_zip_url(cik: int, accession: str) -> str:
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/"
        f"{accession.replace('-', '')}/{accession}-xbrl.zip"
    )


def test_build_manifest_freezes_resolution_and_retains_unresolved(
    tmp_path, monkeypatch
):
    constituents_path = tmp_path / "constituents.csv"
    _write_constituents_csv(
        constituents_path,
        [
            {"ticker": "MSFT", "company_name": "Microsoft Corporation"},
            {"ticker": "BRK.B", "company_name": "Berkshire Hathaway Inc."},
            {"ticker": "GOOG", "company_name": "Alphabet Inc. Class C"},
            {"ticker": "GOOGL", "company_name": "Alphabet Inc. Class A"},
            {"ticker": "ZZZZ", "company_name": "Missing Co"},
        ],
    )

    sec_tickers = {
        "MSFT": {
            "ticker": "MSFT",
            "company_name": "Microsoft Corporation",
            "cik": 789019,
        },
        "BRK-B": {
            "ticker": "BRK-B",
            "company_name": "Berkshire Hathaway Inc.",
            "cik": 1067983,
        },
        "GOOG": {"ticker": "GOOG", "company_name": "Alphabet Inc.", "cik": 1652044},
        "GOOGL": {"ticker": "GOOGL", "company_name": "Alphabet Inc.", "cik": 1652044},
    }
    submissions = {
        789019: {
            "filings": {
                "recent": {
                    "form": ["10-K", "8-K", "10-Q", "10-K"],
                    "accessionNumber": [
                        "future-10k",
                        "8k",
                        "0001-0001-0001",
                        "0002-0002-0002",
                    ],
                    "filingDate": [
                        "2026-03-21",
                        "2026-03-01",
                        "2026-01-28",
                        "2025-07-30",
                    ],
                    "reportDate": [
                        "2026-03-01",
                        "2026-02-28",
                        "2025-12-31",
                        "2025-06-30",
                    ],
                    "primaryDocument": [
                        "future.htm",
                        "foo.htm",
                        "msft-10q.htm",
                        "msft-10k.htm",
                    ],
                }
            }
        },
        1067983: {
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "accessionNumber": ["0003-0003-0003"],
                    "filingDate": ["2026-02-20"],
                    "reportDate": ["2025-12-31"],
                    "primaryDocument": ["brk-10k.htm"],
                }
            }
        },
        1652044: {
            "filings": {
                "recent": {
                    "form": ["10-Q", "10-K"],
                    "accessionNumber": ["0004-0004-0004", "0005-0005-0005"],
                    "filingDate": ["2026-02-04", "2025-10-24"],
                    "reportDate": ["2025-12-31", "2025-09-30"],
                    "primaryDocument": ["goog-10q.htm", "goog-10k.htm"],
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
        constituents_path=str(constituents_path),
        corpus_root=str(tmp_path / "corpus"),
        snapshot_date="2026-03-20",
        constituent_source="unit-test",
    )

    layout = ensure_corpus_layout(tmp_path / "corpus")
    issuer_rows = read_jsonl(layout.manifest / "issuer_manifest.jsonl")
    filing_rows = read_jsonl(layout.manifest / "filing_manifest.jsonl")

    assert summary["issuer_count"] == 5
    assert summary["resolved_issuer_count"] == 4
    assert summary["ingest_enabled_issuer_count"] == 3
    assert summary["resolved_filing_count"] == 5

    brk = next(row for row in issuer_rows if row["ticker"] == "BRK.B")
    assert brk["status"] == "resolved"
    assert brk["cik"] == 1067983

    unresolved = next(row for row in issuer_rows if row["ticker"] == "ZZZZ")
    assert unresolved["status"] == "unresolved_sec_ticker"

    goog = next(row for row in issuer_rows if row["ticker"] == "GOOG")
    googl = next(row for row in issuer_rows if row["ticker"] == "GOOGL")
    assert goog["ingest_enabled"] is True
    assert googl["ingest_enabled"] is False
    assert googl["duplicate_cik_of"] == "GOOG"

    brk_10q = next(
        row
        for row in filing_rows
        if row["ticker"] == "BRK.B" and row["form_type"] == "10-Q"
    )
    assert brk_10q["status"] == "missing_latest_form"

    googl_10k = next(
        row
        for row in filing_rows
        if row["ticker"] == "GOOGL" and row["form_type"] == "10-K"
    )
    assert googl_10k["status"] == "duplicate_cik_constituent"
    assert googl_10k["canonical_ticker"] == "GOOG"

    msft_10k = next(
        row
        for row in filing_rows
        if row["ticker"] == "MSFT" and row["form_type"] == "10-K"
    )
    assert msft_10k["accession"] == "0002-0002-0002"
    assert msft_10k["zip_url"].endswith("0002-0002-0002-xbrl.zip")
    assert msft_10k["report_date"] == "2025-06-30"
    assert msft_10k["selection_rank"] == 1

    assert (layout.raw_submissions / "CIK0000789019.json").exists()
    assert (
        json.loads(
            (layout.manifest / "sec_company_tickers.json").read_text(encoding="utf-8")
        )["MSFT"]["cik"]
        == 789019
    )
    source_manifest = json.loads(
        Path(summary["source_manifest"]).read_text(encoding="utf-8")
    )
    assert source_manifest["source_manifest_version"] == "auditops-sec-source.v2"
    assert (
        source_manifest["artifacts"]["issuer_manifest"]["sha256"]
        == hashlib.sha256(
            (layout.manifest / "issuer_manifest.jsonl").read_bytes()
        ).hexdigest()
    )
    assert (
        source_manifest["artifacts"]["filing_manifest"]["sha256"]
        == hashlib.sha256(
            (layout.manifest / "filing_manifest.jsonl").read_bytes()
        ).hexdigest()
    )


def _write_constituent_source_manifest(
    constituents_path: Path,
    *,
    snapshot_date: str = "2026-03-20",
    row_count: int = 1,
    capture_scope: str = "full",
    row_limit=None,
) -> None:
    raw_source_path = constituents_path.with_name(
        f"{constituents_path.name}.source.html"
    )
    raw_source_path.write_text("<html>pinned source</html>", encoding="utf-8")
    raw_sha256 = hashlib.sha256(raw_source_path.read_bytes()).hexdigest()
    payload = {
        "source_manifest_version": "auditops-us-constituents-source.v2",
        "snapshot_date": snapshot_date,
        "output_sha256": hashlib.sha256(constituents_path.read_bytes()).hexdigest(),
        "row_count": row_count,
        "capture_scope": capture_scope,
        "row_limit": row_limit,
        "requested_as_of": f"{snapshot_date}T23:59:59Z",
        "revision_id": 12345,
        "revision_timestamp": f"{snapshot_date}T12:00:00Z",
        "raw_source_path": raw_source_path.name,
        "raw_source_sha256": raw_sha256,
        "source_sha256": raw_sha256,
    }
    constituents_path.with_name(f"{constituents_path.name}.source.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("snapshot_date", "row_count", "capture_scope", "row_limit", "error"),
    [
        ("2026-03-19", 1, "full", None, "snapshot_date"),
        ("2026-03-20", 2, "full", None, "row_count"),
        ("2026-03-20", 1, "limited", 1, "Limited constituent snapshots"),
        ("2026-03-20", 1, "unknown", None, "capture_scope"),
    ],
)
def test_build_manifest_validates_constituent_source_binding(
    tmp_path,
    monkeypatch,
    snapshot_date,
    row_count,
    capture_scope,
    row_limit,
    error,
):
    constituents_path = tmp_path / "constituents.csv"
    _write_constituents_csv(
        constituents_path,
        [{"ticker": "MSFT", "company_name": "Microsoft Corporation"}],
    )
    _write_constituent_source_manifest(
        constituents_path,
        snapshot_date=snapshot_date,
        row_count=row_count,
        capture_scope=capture_scope,
        row_limit=row_limit,
    )
    monkeypatch.setattr(
        corpus,
        "_get_session",
        lambda user_agent=None: pytest.fail("validation must happen before SEC access"),
    )

    with pytest.raises(ValueError, match=error):
        build_manifest(
            constituents_path=str(constituents_path),
            corpus_root=str(tmp_path / "corpus"),
            snapshot_date="2026-03-20",
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("source_manifest_version", "auditops-us-constituents-source.v1", "v2 format"),
        ("requested_as_of", "2026-03-19T23:59:59Z", "requested_as_of"),
        ("revision_id", 0, "revision_id"),
        ("revision_timestamp", "2026-03-21T00:00:00Z", "newer than"),
        ("raw_source_sha256", "0" * 64, "raw source SHA-256"),
    ],
)
def test_build_manifest_rejects_unbound_historical_source_fields(
    tmp_path,
    monkeypatch,
    field,
    value,
    error,
):
    constituents_path = tmp_path / "constituents.csv"
    _write_constituents_csv(
        constituents_path,
        [{"ticker": "MSFT", "company_name": "Microsoft Corporation"}],
    )
    _write_constituent_source_manifest(constituents_path)
    source_path = constituents_path.with_name(f"{constituents_path.name}.source.json")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source[field] = value
    source_path.write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(
        corpus,
        "_get_session",
        lambda user_agent=None: pytest.fail("validation must happen before SEC access"),
    )

    with pytest.raises(ValueError, match=error):
        build_manifest(
            constituents_path=str(constituents_path),
            corpus_root=str(tmp_path / "corpus"),
            snapshot_date="2026-03-20",
        )


def test_build_manifest_allows_explicit_limited_smoke_snapshot(tmp_path, monkeypatch):
    constituents_path = tmp_path / "constituents.csv"
    _write_constituents_csv(
        constituents_path,
        [{"ticker": "MISS", "company_name": "Missing Corporation"}],
    )
    _write_constituent_source_manifest(
        constituents_path,
        capture_scope="limited",
        row_limit=1,
    )
    monkeypatch.setattr(corpus, "_get_session", lambda user_agent=None: object())
    monkeypatch.setattr(corpus, "_fetch_sec_company_tickers", lambda session: {})

    summary = build_manifest(
        constituents_path=str(constituents_path),
        corpus_root=str(tmp_path / "corpus"),
        snapshot_date="2026-03-20",
        allow_limited_constituents=True,
    )

    assert summary["issuer_count"] == 1
    source_manifest = json.loads(
        Path(summary["source_manifest"]).read_text(encoding="utf-8")
    )
    raw_artifact = source_manifest["artifacts"]["constituents_raw_source"]
    copied_raw = Path(summary["source_manifest"]).parents[1] / raw_artifact["path"]
    assert copied_raw.is_file()
    assert raw_artifact["sha256"] == hashlib.sha256(copied_raw.read_bytes()).hexdigest()


@pytest.mark.parametrize("snapshot_date", ["2026-3-20", "not-a-date", "9999-01-01"])
def test_build_manifest_rejects_invalid_or_future_snapshot_date(
    tmp_path,
    monkeypatch,
    snapshot_date,
):
    constituents_path = tmp_path / "constituents.csv"
    _write_constituents_csv(
        constituents_path,
        [{"ticker": "MSFT", "company_name": "Microsoft Corporation"}],
    )
    monkeypatch.setattr(
        corpus,
        "_get_session",
        lambda user_agent=None: pytest.fail("validation must happen before SEC access"),
    )

    with pytest.raises(ValueError, match="Snapshot date"):
        build_manifest(
            constituents_path=str(constituents_path),
            corpus_root=str(tmp_path / "corpus"),
            snapshot_date=snapshot_date,
        )


def test_build_manifest_refuses_existing_corpus_state_before_sec_access(
    tmp_path, monkeypatch
):
    constituents_path = tmp_path / "constituents.csv"
    _write_constituents_csv(
        constituents_path,
        [{"ticker": "MSFT", "company_name": "Microsoft Corporation"}],
    )
    stale_path = tmp_path / "corpus" / "raw" / "submissions" / "CIK0000000001.json"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        corpus,
        "_get_session",
        lambda user_agent=None: pytest.fail(
            "state validation must happen before SEC access"
        ),
    )

    with pytest.raises(FileExistsError, match="new or file-empty corpus root"):
        build_manifest(
            constituents_path=str(constituents_path),
            corpus_root=str(tmp_path / "corpus"),
            snapshot_date="2026-03-20",
        )


def test_build_manifest_can_select_trailing_two_fiscal_years(tmp_path, monkeypatch):
    constituents_path = tmp_path / "constituents.csv"
    _write_constituents_csv(
        constituents_path, [{"ticker": "MSFT", "company_name": "Microsoft Corporation"}]
    )

    sec_tickers = {
        "MSFT": {
            "ticker": "MSFT",
            "company_name": "Microsoft Corporation",
            "cik": 789019,
        },
    }
    submissions = {
        789019: {
            "filings": {
                "recent": {
                    "form": [
                        "10-Q",
                        "10-Q",
                        "10-Q",
                        "10-Q",
                        "10-Q",
                        "10-K",
                        "10-K",
                        "10-K",
                    ],
                    "accessionNumber": [
                        "q1",
                        "q2",
                        "q3",
                        "q4",
                        "q5",
                        "k1",
                        "k2",
                        "k3",
                    ],
                    "filingDate": [
                        "2026-01-28",
                        "2025-10-24",
                        "2025-07-25",
                        "2025-04-25",
                        "2024-10-24",
                        "2025-07-30",
                        "2024-07-30",
                        "2023-07-30",
                    ],
                    "reportDate": [
                        "2025-12-31",
                        "2025-09-30",
                        "2025-06-30",
                        "2025-03-31",
                        "2024-09-30",
                        "2025-06-30",
                        "2024-06-30",
                        "2023-06-30",
                    ],
                    "primaryDocument": [
                        "msft-q1.htm",
                        "msft-q2.htm",
                        "msft-q3.htm",
                        "msft-q4.htm",
                        "msft-q5.htm",
                        "msft-k1.htm",
                        "msft-k2.htm",
                        "msft-k3.htm",
                    ],
                }
            }
        }
    }

    monkeypatch.setattr(corpus, "_get_session", lambda user_agent=None: object())
    monkeypatch.setattr(
        corpus, "_fetch_sec_company_tickers", lambda session: sec_tickers
    )
    monkeypatch.setattr(
        corpus, "_fetch_submission_json", lambda session, cik: submissions[cik]
    )

    summary = build_manifest(
        constituents_path=str(constituents_path),
        corpus_root=str(tmp_path / "corpus"),
        snapshot_date="2026-03-20",
        trailing_fiscal_years=2,
        constituent_source="unit-test",
    )

    layout = ensure_corpus_layout(tmp_path / "corpus")
    filing_rows = read_jsonl(layout.manifest / "filing_manifest.jsonl")

    assert summary["snapshot_id"] == "sp500_trailing_2fy_2026-03-20"
    assert summary["trailing_fiscal_years"] == 2
    assert summary["resolved_filing_count"] == 7

    ten_ks = [row for row in filing_rows if row["form_type"] == "10-K"]
    ten_qs = [row for row in filing_rows if row["form_type"] == "10-Q"]
    assert [row["accession"] for row in ten_ks] == ["k1", "k2"]
    assert [row["selection_rank"] for row in ten_ks] == [1, 2]
    assert [row["accession"] for row in ten_qs] == ["q1", "q2", "q3", "q4", "q5"]
    assert all(row["filing_span"] == "trailing_fiscal_years" for row in filing_rows)


def test_issuer_holdout_is_deterministic_and_disjoint():
    issuer_manifest = [
        {
            "snapshot_id": "sp500_latest_2026-03-20",
            "snapshot_date": "2026-03-20",
            "ticker": ticker,
            "status": "resolved",
            "ingest_enabled": ticker != "GOOGL",
        }
        for ticker in ["AAPL", "MSFT", "NVDA", "TSLA", "BRK.B", "AMZN", "GOOGL", "META"]
    ]

    split_one = _issuer_split(issuer_manifest)
    split_two = _issuer_split(issuer_manifest)

    assert split_one == split_two
    assert any(row["split"] == "eval_holdout" for row in split_one)
    assert all(row["ticker"] != "GOOGL" for row in split_one)
    assert {row["ticker"] for row in split_one if row["split"] == "train"}.isdisjoint(
        {row["ticker"] for row in split_one if row["split"] == "eval_holdout"}
    )


def test_download_with_resume_overwrites_if_server_ignores_range(tmp_path):
    output_path = tmp_path / "resume.zip"
    output_path.write_bytes(b"partial-")
    session = FakeSession(
        {"https://example.com/file.zip": [FakeResponse(200, b"full-file")]}
    )

    http_status, size = _download_with_resume(
        session, "https://example.com/file.zip", output_path
    )

    assert http_status == 200
    assert size == len(b"full-file")
    assert output_path.read_bytes() == b"full-file"
    assert session.calls[0]["headers"]["Range"] == "bytes=8-"
    assert session.calls[0]["kwargs"]["allow_redirects"] is False


def test_download_with_resume_requires_exact_content_range(tmp_path):
    output_path = tmp_path / "resume.zip"
    output_path.write_bytes(b"partial-")
    session = FakeSession(
        {
            "https://example.com/file.zip": [
                FakeResponse(
                    206,
                    b"rest",
                    headers={
                        "Content-Length": "4",
                        "Content-Range": "bytes 7-10/11",
                    },
                )
            ]
        }
    )

    with pytest.raises(corpus.DownloadIntegrityError, match="Content-Range"):
        _download_with_resume(session, "https://example.com/file.zip", output_path)

    assert output_path.read_bytes() == b"partial-"


def test_download_with_resume_accepts_complete_validated_suffix(tmp_path):
    output_path = tmp_path / "resume.zip"
    output_path.write_bytes(b"partial-")
    session = FakeSession(
        {
            "https://example.com/file.zip": [
                FakeResponse(
                    206,
                    b"rest",
                    headers={
                        "Content-Length": "4",
                        "Content-Range": "bytes 8-11/12",
                    },
                )
            ]
        }
    )

    http_status, size = _download_with_resume(
        session,
        "https://example.com/file.zip",
        output_path,
    )

    assert http_status == 206
    assert size == 12
    assert output_path.read_bytes() == b"partial-rest"


def test_download_with_resume_refetches_full_object_after_416(tmp_path):
    output_path = tmp_path / "resume.zip"
    output_path.write_bytes(b"untrusted-partial")
    full_payload = b"complete-object"
    session = FakeSession(
        {
            "https://example.com/file.zip": [
                FakeResponse(416, headers={"Content-Range": "bytes */15"}),
                FakeResponse(200, full_payload),
            ]
        }
    )

    http_status, size = _download_with_resume(
        session,
        "https://example.com/file.zip",
        output_path,
    )

    assert http_status == 200
    assert size == len(full_payload)
    assert output_path.read_bytes() == full_payload
    assert session.calls[0]["headers"]["Range"] == "bytes=17-"
    assert session.calls[1]["headers"] == {}


def test_download_filings_records_success_errors_and_retry(tmp_path, monkeypatch):
    layout = ensure_corpus_layout(tmp_path / "corpus")
    write_jsonl(
        layout.manifest / "filing_manifest.jsonl",
        [
            {
                "snapshot_id": "sp500_latest_2026-03-20",
                "snapshot_date": "2026-03-20",
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "cik": 320193,
                "form_type": "10-K",
                "accession": "0000320193-25-000079",
                "filed_at": "2025-10-31",
                "primary_doc": "aapl-10k.htm",
                "zip_url": _sec_zip_url(320193, "0000320193-25-000079"),
                "status": "resolved",
                "error": None,
            },
            {
                "snapshot_id": "sp500_latest_2026-03-20",
                "snapshot_date": "2026-03-20",
                "ticker": "MSFT",
                "company_name": "Microsoft Corporation",
                "cik": 789019,
                "form_type": "10-Q",
                "accession": "0000789019-26-000010",
                "filed_at": "2026-01-28",
                "primary_doc": "msft-10q.htm",
                "zip_url": _sec_zip_url(789019, "0000789019-26-000010"),
                "status": "resolved",
                "error": None,
            },
            {
                "snapshot_id": "sp500_latest_2026-03-20",
                "snapshot_date": "2026-03-20",
                "ticker": "NVDA",
                "company_name": "NVIDIA Corporation",
                "cik": 1045810,
                "form_type": "10-K",
                "accession": "0001045810-26-000020",
                "filed_at": "2026-02-25",
                "primary_doc": "nvda-10k.htm",
                "zip_url": _sec_zip_url(1045810, "0001045810-26-000020"),
                "status": "resolved",
                "error": None,
            },
            {
                "snapshot_id": "sp500_latest_2026-03-20",
                "snapshot_date": "2026-03-20",
                "ticker": "TSLA",
                "company_name": "Tesla, Inc.",
                "cik": 1318605,
                "form_type": "10-Q",
                "accession": "0001318605-25-000030",
                "filed_at": "2025-10-23",
                "primary_doc": "tsla-10q.htm",
                "zip_url": _sec_zip_url(1318605, "0001318605-25-000030"),
                "status": "resolved",
                "error": None,
            },
            {
                "snapshot_id": "sp500_latest_2026-03-20",
                "snapshot_date": "2026-03-20",
                "ticker": "BAD",
                "company_name": "Bad Co",
                "cik": None,
                "form_type": "10-Q",
                "accession": None,
                "filed_at": None,
                "primary_doc": None,
                "zip_url": None,
                "status": "unresolved_issuer",
                "error": "ticker_not_found",
            },
        ],
    )

    aapl_zip = _zip_payload("aapl.txt", b"aapl-data")
    tsla_zip = _zip_payload("tsla.txt", b"tsla-data")
    session = FakeSession(
        {
            _sec_zip_url(320193, "0000320193-25-000079"): [FakeResponse(200, aapl_zip)],
            _sec_zip_url(789019, "0000789019-26-000010"): [
                FakeResponse(404, b"missing")
            ],
            _sec_zip_url(1045810, "0001045810-26-000020"): [
                FakeResponse(403, b"blocked")
            ],
            _sec_zip_url(1318605, "0001318605-25-000030"): [
                requests.Timeout("slow"),
                FakeResponse(200, tsla_zip),
            ],
        }
    )
    monkeypatch.setattr(corpus, "_get_session", lambda user_agent=None: session)

    summary = download_filings(str(layout.root))
    ledger_rows = read_jsonl(layout.manifest / "download_ledger.jsonl")

    assert summary["success_count"] == 2
    assert summary["error_count"] == 2

    statuses = {(row["ticker"], row["status"]) for row in ledger_rows}
    assert ("AAPL", "downloaded") in statuses
    assert ("MSFT", "download_error") in statuses
    assert ("NVDA", "download_error") in statuses
    assert ("TSLA", "downloaded") in statuses
    assert ("BAD", "unresolved_issuer") in statuses

    msft_row = next(row for row in ledger_rows if row["ticker"] == "MSFT")
    nvda_row = next(row for row in ledger_rows if row["ticker"] == "NVDA")
    tsla_row = next(row for row in ledger_rows if row["ticker"] == "TSLA")
    assert msft_row["http_status"] == 404
    assert nvda_row["http_status"] == 403
    assert tsla_row["attempts"] == 2
    assert Path(tsla_row["local_path"]).read_bytes() == tsla_zip


def test_download_filings_only_reuses_hash_bound_existing_zip(tmp_path, monkeypatch):
    layout = ensure_corpus_layout(tmp_path / "corpus")
    filing = {
        "snapshot_id": "sp500_latest_2026-03-20",
        "snapshot_date": "2026-03-20",
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "cik": 320193,
        "form_type": "10-K",
        "accession": "0000320193-25-000079",
        "filed_at": "2025-10-31",
        "primary_doc": "aapl-10k.htm",
        "zip_url": _sec_zip_url(320193, "0000320193-25-000079"),
        "status": "resolved",
        "error": None,
    }
    write_jsonl(layout.manifest / "filing_manifest.jsonl", [filing])
    local_path = layout.raw_xbrl_zip / "AAPL" / "AAPL_10-K_0000320193-25-000079.zip"
    local_path.parent.mkdir(parents=True)
    local_path.write_bytes(b"untrusted-partial")
    complete_zip = _zip_payload("aapl.txt", b"complete")
    first_session = FakeSession(
        {
            filing["zip_url"]: [
                FakeResponse(416, headers={"Content-Range": "bytes */17"}),
                FakeResponse(200, complete_zip),
            ]
        }
    )
    monkeypatch.setattr(corpus, "_get_session", lambda user_agent=None: first_session)

    first = download_filings(str(layout.root), max_attempts=1)

    assert first["success_count"] == 1
    assert len(first_session.calls) == 2
    assert local_path.read_bytes() == complete_zip

    second_session = FakeSession({})
    monkeypatch.setattr(corpus, "_get_session", lambda user_agent=None: second_session)
    second = download_filings(str(layout.root), max_attempts=1)
    second_row = read_jsonl(layout.manifest / "download_ledger.jsonl")[0]

    assert second["success_count"] == 1
    assert second_row["status"] == "existing_local"
    assert second_session.calls == []


def test_download_filings_refuses_modified_bound_filing_manifest(tmp_path, monkeypatch):
    layout = ensure_corpus_layout(tmp_path / "corpus")
    filing_manifest_path = layout.manifest / "filing_manifest.jsonl"
    write_jsonl(filing_manifest_path, [{"ticker": "AAPL", "form_type": "10-K"}])
    source_manifest = {
        "artifacts": {
            "filing_manifest": {
                "path": str(filing_manifest_path.relative_to(layout.root)),
                "bytes": filing_manifest_path.stat().st_size,
                "sha256": hashlib.sha256(filing_manifest_path.read_bytes()).hexdigest(),
            }
        }
    }
    (layout.manifest / "source_manifest.json").write_text(
        json.dumps(source_manifest),
        encoding="utf-8",
    )
    filing_manifest_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(corpus, "_get_session", lambda user_agent=None: object())

    with pytest.raises(ValueError, match="filing_manifest .* does not match"):
        download_filings(str(layout.root))


def test_download_filings_refuses_stale_ledger_from_other_manifest(
    tmp_path, monkeypatch
):
    layout = ensure_corpus_layout(tmp_path / "corpus")
    write_jsonl(
        layout.manifest / "filing_manifest.jsonl",
        [
            {
                "snapshot_id": "snapshot-a",
                "ticker": "AAPL",
                "cik": 320193,
                "form_type": "10-K",
                "accession": "0000320193-25-000079",
                "zip_url": _sec_zip_url(320193, "0000320193-25-000079"),
                "status": "resolved",
            }
        ],
    )
    write_jsonl(
        layout.manifest / "download_ledger.jsonl",
        [
            {
                "snapshot_id": "snapshot-b",
                "ticker": "MSFT",
                "cik": 789019,
                "form_type": "10-K",
                "accession": "0000789019-25-000010",
                "zip_url": _sec_zip_url(789019, "0000789019-25-000010"),
                "status": "downloaded",
            }
        ],
    )
    monkeypatch.setattr(corpus, "_get_session", lambda user_agent=None: object())

    with pytest.raises(ValueError, match="different filing manifest"):
        download_filings(str(layout.root))


@pytest.mark.parametrize(
    ("accession", "zip_url", "error"),
    [
        (
            "0000320193-25-000079",
            "https://evil.example/Archives/edgar/data/320193/000032019325000079/"
            "0000320193-25-000079-xbrl.zip",
            "www.sec.gov",
        ),
        (
            "0001",
            _sec_zip_url(320193, "0000320193-25-000079"),
            "malformed accession",
        ),
    ],
)
def test_download_filings_rejects_poisoned_sec_target(
    tmp_path,
    monkeypatch,
    accession,
    zip_url,
    error,
):
    layout = ensure_corpus_layout(tmp_path / "corpus")
    write_jsonl(
        layout.manifest / "filing_manifest.jsonl",
        [
            {
                "snapshot_id": "snapshot-a",
                "ticker": "AAPL",
                "cik": 320193,
                "form_type": "10-K",
                "accession": accession,
                "zip_url": zip_url,
                "status": "resolved",
            }
        ],
    )
    monkeypatch.setattr(
        corpus,
        "_get_session",
        lambda user_agent=None: pytest.fail(
            "URL validation must happen before SEC access"
        ),
    )

    with pytest.raises(ValueError, match=error):
        download_filings(str(layout.root))


def test_sec_target_accepts_accession_submitted_by_a_different_edgar_identity():
    accession = "0000950170-25-100235"
    issuer_cik = 789019

    corpus._validate_sec_filing_target(
        {
            "status": "resolved",
            "ticker": "MSFT",
            "cik": issuer_cik,
            "form_type": "10-K",
            "accession": accession,
            "zip_url": _sec_zip_url(issuer_cik, accession),
        }
    )


def test_ingest_corpus_rejects_ledger_sha_mismatch_before_processing(tmp_path):
    layout = ensure_corpus_layout(tmp_path / "corpus")
    zip_path = build_fixture_zip(tmp_path, "filing_10k")
    integrity = _ledger_integrity(zip_path)
    write_jsonl(
        layout.manifest / "download_ledger.jsonl",
        [
            {
                "ticker": "ALFA",
                "form_type": "10-K",
                "accession": "0001",
                "status": "downloaded",
                "local_path": str(zip_path),
                **integrity,
            }
        ],
    )
    payload = bytearray(zip_path.read_bytes())
    payload[-1] ^= 1
    zip_path.write_bytes(payload)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ingest_corpus(str(layout.root))

    assert not layout.db_path.exists()
    assert not layout.db_path.with_name(f"{layout.db_path.name}.building").exists()


def test_ingest_corpus_requires_explicit_atomic_db_replacement(tmp_path):
    layout = ensure_corpus_layout(tmp_path / "corpus")
    zip_path = build_fixture_zip(tmp_path, "filing_10k")
    write_jsonl(
        layout.manifest / "download_ledger.jsonl",
        [
            {
                "ticker": "ALFA",
                "form_type": "10-K",
                "accession": "0001",
                "status": "downloaded",
                "local_path": str(zip_path),
                **_ledger_integrity(zip_path),
            }
        ],
    )
    layout.db_path.write_bytes(b"existing-db-sentinel")

    with pytest.raises(FileExistsError, match="replace_existing=True"):
        ingest_corpus(str(layout.root))
    assert layout.db_path.read_bytes() == b"existing-db-sentinel"

    summary = ingest_corpus(str(layout.root), replace_existing=True)

    assert summary["published"] is True
    assert summary["ingested_count"] == 1
    assert layout.db_path.read_bytes() != b"existing-db-sentinel"


def test_ingest_failure_preserves_existing_db(tmp_path, monkeypatch):
    layout = ensure_corpus_layout(tmp_path / "corpus")
    zip_path = build_fixture_zip(tmp_path, "filing_10k")
    write_jsonl(
        layout.manifest / "download_ledger.jsonl",
        [
            {
                "ticker": "ALFA",
                "form_type": "10-K",
                "accession": "0001",
                "status": "downloaded",
                "local_path": str(zip_path),
                **_ledger_integrity(zip_path),
            }
        ],
    )
    sentinel = b"existing-db-sentinel"
    layout.db_path.write_bytes(sentinel)

    def fail_after_staging(*, out_db, **kwargs):
        Path(out_db).write_bytes(b"partial-staged-db")
        raise RuntimeError("fixture ingest failure")

    monkeypatch.setattr(corpus, "process_zip", fail_after_staging)

    summary = ingest_corpus(str(layout.root), replace_existing=True)

    assert summary["published"] is False
    assert summary["error_count"] == 1
    assert layout.db_path.read_bytes() == sentinel
    assert not layout.db_path.with_name(f"{layout.db_path.name}.building").exists()


def test_corpus_pipeline_generates_shared_db_and_eval_reports(tmp_path):
    layout = ensure_corpus_layout(tmp_path / "corpus")
    zip_10k = build_fixture_zip(tmp_path, "filing_10k")
    zip_q1 = build_fixture_zip(tmp_path, "filing_q1")
    zip_q2 = build_fixture_zip(tmp_path, "filing_q2")

    write_jsonl(
        layout.manifest / "issuer_manifest.jsonl",
        [
            {
                "snapshot_id": "sp500_latest_2026-03-20",
                "snapshot_date": "2026-03-20",
                "ticker": "ALFA",
                "company_name": "Alpha Corp",
                "cik": 1,
                "forms_expected": ["10-K", "10-Q"],
                "constituent_source": "fixtures",
                "sec_submission_url": "https://example.com/alpha.json",
                "status": "resolved",
                "error": None,
            },
            {
                "snapshot_id": "sp500_latest_2026-03-20",
                "snapshot_date": "2026-03-20",
                "ticker": "BETA",
                "company_name": "Beta Corp",
                "cik": 2,
                "forms_expected": ["10-K", "10-Q"],
                "constituent_source": "fixtures",
                "sec_submission_url": "https://example.com/beta.json",
                "status": "resolved",
                "error": None,
            },
        ],
    )
    write_jsonl(
        layout.manifest / "filing_manifest.jsonl",
        [
            {
                "snapshot_id": "sp500_latest_2026-03-20",
                "snapshot_date": "2026-03-20",
                "ticker": "ALFA",
                "company_name": "Alpha Corp",
                "cik": 1,
                "form_type": "10-K",
                "accession": "0001",
                "filed_at": "2025-01-31",
                "primary_doc": "alpha-10k.htm",
                "zip_url": "https://example.com/alpha-10k.zip",
                "status": "resolved",
                "error": None,
            },
            {
                "snapshot_id": "sp500_latest_2026-03-20",
                "snapshot_date": "2026-03-20",
                "ticker": "ALFA",
                "company_name": "Alpha Corp",
                "cik": 1,
                "form_type": "10-Q",
                "accession": "0002",
                "filed_at": "2025-07-31",
                "primary_doc": "alpha-10q.htm",
                "zip_url": "https://example.com/alpha-10q.zip",
                "status": "resolved",
                "error": None,
            },
            {
                "snapshot_id": "sp500_latest_2026-03-20",
                "snapshot_date": "2026-03-20",
                "ticker": "BETA",
                "company_name": "Beta Corp",
                "cik": 2,
                "form_type": "10-Q",
                "accession": "0003",
                "filed_at": "2025-04-30",
                "primary_doc": "beta-10q.htm",
                "zip_url": "https://example.com/beta-10q.zip",
                "status": "resolved",
                "error": None,
            },
        ],
    )
    write_jsonl(
        layout.manifest / "download_ledger.jsonl",
        [
            {
                "ticker": "ALFA",
                "form_type": "10-K",
                "accession": "0001",
                "status": "existing_local",
                "local_path": str(zip_10k),
                **_ledger_integrity(zip_10k),
            },
            {
                "ticker": "ALFA",
                "form_type": "10-Q",
                "accession": "0002",
                "status": "existing_local",
                "local_path": str(zip_q1),
                **_ledger_integrity(zip_q1),
            },
            {
                "ticker": "BETA",
                "form_type": "10-Q",
                "accession": "0003",
                "status": "existing_local",
                "local_path": str(zip_q2),
                **_ledger_integrity(zip_q2),
            },
        ],
    )

    ingest_summary = ingest_corpus(str(layout.root), extract_narrative=True)
    dataset_summary = generate_corpus_datasets(str(layout.root))
    eval_summary = eval_corpus(str(layout.root))

    assert ingest_summary["ingested_count"] == 3
    assert dataset_summary["answers"]["count"] > 0
    assert dataset_summary["counts"]["task_specs"] > 0
    assert dataset_summary["counts"]["eval_holdout"] > 0

    task_specs = read_jsonl(layout.derived_tasks / "task_specs_quant.jsonl")
    split_manifest = read_jsonl(layout.eval_dir / "split_manifest.jsonl")
    assert task_specs
    assert split_manifest
    assert {
        row["ticker"] for row in split_manifest if row["split"] == "train"
    }.isdisjoint(
        {row["ticker"] for row in split_manifest if row["split"] == "eval_holdout"}
    )

    assert Path(eval_summary["data_quality_summary"]).exists()
    assert Path(eval_summary["runtime_eval_summary"]).exists()
    assert Path(eval_summary["coverage_by_metric"]).exists()
    assert Path(eval_summary["coverage_by_issuer"]).exists()
    assert Path(eval_summary["validator_distribution"]).exists()
    assert Path(eval_summary["runtime_failures"]).exists()
    assert Path(eval_summary["runtime_failure_buckets"]).exists()

    runtime_summary = json.loads(
        Path(eval_summary["runtime_eval_summary"]).read_text(encoding="utf-8")
    )
    failure_buckets = json.loads(
        Path(eval_summary["runtime_failure_buckets"]).read_text(encoding="utf-8")
    )
    assert runtime_summary["eval_task_count"] > 0
    assert runtime_summary["status_accuracy"] == pytest.approx(1.0)
    assert failure_buckets["failure_count"] == 0
    assert failure_buckets["failure_buckets"] == {}


def test_repair_corpus_filing_updates_ingest_ledger(tmp_path):
    layout = ensure_corpus_layout(tmp_path / "corpus")
    zip_split = build_fixture_zip(tmp_path, "filing_split")

    write_jsonl(
        layout.manifest / "download_ledger.jsonl",
        [
            {
                "ticker": "SPLT",
                "form_type": "10-K",
                "accession": "0009",
                "status": "existing_local",
                "local_path": str(zip_split),
                "primary_doc": "splitco-20241231_d2.htm",
                **_ledger_integrity(zip_split),
            }
        ],
    )
    write_jsonl(
        layout.logs_dir / "ingest_ledger.jsonl",
        [
            {
                "ticker": "SPLT",
                "form_type": "10-K",
                "accession": "0009",
                "local_path": str(zip_split),
                "status": "ingest_error",
                "error": "FOREIGN KEY constraint failed",
            }
        ],
    )

    summary = repair_corpus_filing(str(layout.root), ticker="SPLT", accession="0009")

    assert summary["repaired"]["status"] == "ingested"
    conn = corpus.connect_db(str(layout.db_path))
    try:
        filing = conn.execute(
            "SELECT ticker FROM filings WHERE filing_id='splitco-20241231'"
        ).fetchone()
        assert filing["ticker"] == "SPLT"
    finally:
        conn.close()

    ingest_rows = read_jsonl(layout.logs_dir / "ingest_ledger.jsonl")
    assert len(ingest_rows) == 1
    assert ingest_rows[0]["status"] == "ingested"
    assert ingest_rows[0]["main_html"] == "splitco-20241231.htm"
