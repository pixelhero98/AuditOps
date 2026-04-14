from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import requests

import auditops.corpus as corpus
from auditops.corpus import (
    _download_with_resume,
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


def _write_constituents_csv(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "company_name"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class FakeResponse:
    def __init__(self, status_code: int, body: bytes = b""):
        self.status_code = status_code
        self._body = body

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

    def get(self, url, headers=None, stream=False, timeout=None):
        self.calls.append({"url": url, "headers": dict(headers or {})})
        queue = self._outcomes[url]
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_build_manifest_freezes_resolution_and_retains_unresolved(tmp_path, monkeypatch):
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
        "MSFT": {"ticker": "MSFT", "company_name": "Microsoft Corporation", "cik": 789019},
        "BRK-B": {"ticker": "BRK-B", "company_name": "Berkshire Hathaway Inc.", "cik": 1067983},
        "GOOG": {"ticker": "GOOG", "company_name": "Alphabet Inc.", "cik": 1652044},
        "GOOGL": {"ticker": "GOOGL", "company_name": "Alphabet Inc.", "cik": 1652044},
    }
    submissions = {
        789019: {
            "filings": {
                "recent": {
                    "form": ["8-K", "10-Q", "10-K"],
                    "accessionNumber": ["8k", "0001-0001-0001", "0002-0002-0002"],
                    "filingDate": ["2026-03-01", "2026-01-28", "2025-07-30"],
                    "reportDate": ["2026-02-28", "2025-12-31", "2025-06-30"],
                    "primaryDocument": ["foo.htm", "msft-10q.htm", "msft-10k.htm"],
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
    monkeypatch.setattr(corpus, "_fetch_sec_company_tickers", lambda session: sec_tickers)
    monkeypatch.setattr(corpus, "_fetch_submission_json", lambda session, cik: submissions[cik])

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

    brk_10q = next(row for row in filing_rows if row["ticker"] == "BRK.B" and row["form_type"] == "10-Q")
    assert brk_10q["status"] == "missing_latest_form"

    googl_10k = next(row for row in filing_rows if row["ticker"] == "GOOGL" and row["form_type"] == "10-K")
    assert googl_10k["status"] == "duplicate_cik_constituent"
    assert googl_10k["canonical_ticker"] == "GOOG"

    msft_10k = next(row for row in filing_rows if row["ticker"] == "MSFT" and row["form_type"] == "10-K")
    assert msft_10k["accession"] == "0002-0002-0002"
    assert msft_10k["zip_url"].endswith("0002-0002-0002-xbrl.zip")
    assert msft_10k["report_date"] == "2025-06-30"
    assert msft_10k["selection_rank"] == 1

    assert (layout.raw_submissions / "CIK0000789019.json").exists()
    assert json.loads((layout.manifest / "sec_company_tickers.json").read_text(encoding="utf-8"))["MSFT"]["cik"] == 789019


def test_build_manifest_can_select_trailing_two_fiscal_years(tmp_path, monkeypatch):
    constituents_path = tmp_path / "constituents.csv"
    _write_constituents_csv(constituents_path, [{"ticker": "MSFT", "company_name": "Microsoft Corporation"}])

    sec_tickers = {
        "MSFT": {"ticker": "MSFT", "company_name": "Microsoft Corporation", "cik": 789019},
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
    monkeypatch.setattr(corpus, "_fetch_sec_company_tickers", lambda session: sec_tickers)
    monkeypatch.setattr(corpus, "_fetch_submission_json", lambda session, cik: submissions[cik])

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
    session = FakeSession({"https://example.com/file.zip": [FakeResponse(200, b"full-file")]})

    http_status, size = _download_with_resume(session, "https://example.com/file.zip", output_path)

    assert http_status == 200
    assert size == len(b"full-file")
    assert output_path.read_bytes() == b"full-file"
    assert session.calls[0]["headers"]["Range"] == "bytes=8-"


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
                "accession": "0001",
                "filed_at": "2025-10-31",
                "primary_doc": "aapl-10k.htm",
                "zip_url": "https://example.com/aapl.zip",
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
                "accession": "0002",
                "filed_at": "2026-01-28",
                "primary_doc": "msft-10q.htm",
                "zip_url": "https://example.com/msft.zip",
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
                "accession": "0003",
                "filed_at": "2026-02-25",
                "primary_doc": "nvda-10k.htm",
                "zip_url": "https://example.com/nvda.zip",
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
                "accession": "0004",
                "filed_at": "2025-10-23",
                "primary_doc": "tsla-10q.htm",
                "zip_url": "https://example.com/tsla.zip",
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

    session = FakeSession(
        {
            "https://example.com/aapl.zip": [FakeResponse(200, b"aapl-data")],
            "https://example.com/msft.zip": [FakeResponse(404, b"missing")],
            "https://example.com/nvda.zip": [FakeResponse(403, b"blocked")],
            "https://example.com/tsla.zip": [requests.Timeout("slow"), FakeResponse(200, b"tsla-data")],
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
    assert Path(tsla_row["local_path"]).read_bytes() == b"tsla-data"


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
            },
            {
                "ticker": "ALFA",
                "form_type": "10-Q",
                "accession": "0002",
                "status": "existing_local",
                "local_path": str(zip_q1),
            },
            {
                "ticker": "BETA",
                "form_type": "10-Q",
                "accession": "0003",
                "status": "existing_local",
                "local_path": str(zip_q2),
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
    assert {row["ticker"] for row in split_manifest if row["split"] == "train"}.isdisjoint(
        {row["ticker"] for row in split_manifest if row["split"] == "eval_holdout"}
    )

    assert Path(eval_summary["data_quality_summary"]).exists()
    assert Path(eval_summary["runtime_eval_summary"]).exists()
    assert Path(eval_summary["coverage_by_metric"]).exists()
    assert Path(eval_summary["coverage_by_issuer"]).exists()
    assert Path(eval_summary["validator_distribution"]).exists()
    assert Path(eval_summary["runtime_failures"]).exists()
    assert Path(eval_summary["runtime_failure_buckets"]).exists()

    runtime_summary = json.loads(Path(eval_summary["runtime_eval_summary"]).read_text(encoding="utf-8"))
    failure_buckets = json.loads(Path(eval_summary["runtime_failure_buckets"]).read_text(encoding="utf-8"))
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
        filing = conn.execute("SELECT ticker FROM filings WHERE filing_id='splitco-20241231'").fetchone()
        assert filing["ticker"] == "SPLT"
    finally:
        conn.close()

    ingest_rows = read_jsonl(layout.logs_dir / "ingest_ledger.jsonl")
    assert len(ingest_rows) == 1
    assert ingest_rows[0]["status"] == "ingested"
    assert ingest_rows[0]["main_html"] == "splitco-20241231.htm"
