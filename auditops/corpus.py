from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import requests

from .metrics import generate_answer_objects
from .pipeline import _stable_digest, connect_db, fetch_validators, process_zip
from .runtime import answer_quant
from .tasks import (
    RENDER_VERSION,
    _build_code_target,
    _build_question,
    build_hard_negatives,
    build_task_plan_target,
    build_task_specs,
    build_task_specs_from_answers,
    read_jsonl,
    write_jsonl,
)


SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_SEC_USER_AGENT = "AuditOps/0.1 (research automation)"
ISSUER_HOLDOUT_VERSION = "issuer_holdout_v1"
SUCCESS_DOWNLOAD_STATUSES = {"downloaded", "existing_local"}
RETRYABLE_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class CorpusLayout:
    root: Path
    manifest: Path
    raw_submissions: Path
    raw_xbrl_zip: Path
    db_dir: Path
    derived_answers: Path
    derived_tasks: Path
    derived_datasets: Path
    eval_dir: Path
    logs_dir: Path

    @property
    def db_path(self) -> Path:
        """Return the absolute path to the corpus SQLite database file.
        
        Returns
        -------
        Path
            Text or path value produced by this operation.
        
        Examples
        --------
        >>> layout = ensure_corpus_layout('corpora/sp500_latest_2026-03-20')
        >>> layout.db_path
        PosixPath('corpora/sp500_latest_2026-03-20/db/corpus.sqlite')
        """
        return self.db_dir / "corpus.sqlite"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_jsonl_line(handle, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
    handle.write("\n")


def _iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_corpus_layout(corpus_root: str | Path) -> CorpusLayout:
    """Create corpus directory and return the required corpus directory layout.
    
    Parameters
    ----------
    corpus_root : str | Path
        Root directory of the corpus workspace (including manifest, raw, db, derived, eval). e.g., 'corpora/sp500_latest_2026-03-20'
    
    Returns
    -------
    CorpusLayout
        Return value for create and return the required corpus directory layout.
    """
    root = Path(corpus_root)
    layout = CorpusLayout(
        root=root,
        manifest=root / "manifest",
        raw_submissions=root / "raw" / "submissions",
        raw_xbrl_zip=root / "raw" / "xbrl_zip",
        db_dir=root / "db",
        derived_answers=root / "derived" / "answers",
        derived_tasks=root / "derived" / "tasks",
        derived_datasets=root / "derived" / "datasets",
        eval_dir=root / "eval",
        logs_dir=root / "logs",
    )
    for path in [
        layout.root,
        layout.manifest,
        layout.raw_submissions,
        layout.raw_xbrl_zip,
        layout.db_dir,
        layout.derived_answers,
        layout.derived_tasks,
        layout.derived_datasets,
        layout.eval_dir,
        layout.logs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _get_session(user_agent: Optional[str] = None) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent or os.environ.get("AUDITOPS_SEC_USER_AGENT", DEFAULT_SEC_USER_AGENT),
            "Accept": "application/json, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
    )
    return session


def _session_get(session: requests.Session, url: str, **kwargs):
    min_interval = float(os.environ.get("AUDITOPS_SEC_MIN_INTERVAL_SECONDS", "0.12"))
    last_request_at = getattr(session, "_auditops_last_request_at", None)
    if last_request_at is not None and min_interval > 0:
        elapsed = time.monotonic() - last_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
    response = session.get(url, **kwargs)
    session._auditops_last_request_at = time.monotonic()
    return response


def _snapshot_id(snapshot_date: str, *, trailing_fiscal_years: int = 1) -> str:
    if trailing_fiscal_years <= 1:
        return f"sp500_latest_{snapshot_date}"
    return f"sp500_trailing_{trailing_fiscal_years}fy_{snapshot_date}"


def _read_constituents_snapshot(path: str | Path) -> List[Dict[str, str]]:
    snapshot_path = Path(path)
    with snapshot_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Constituent snapshot CSV is missing a header row.")

        fields = {field.lower(): field for field in reader.fieldnames}
        ticker_field = next((fields[name] for name in ("ticker", "symbol") if name in fields), None)
        name_field = next((fields[name] for name in ("company_name", "name", "security") if name in fields), None)
        if ticker_field is None:
            raise ValueError("Constituent snapshot CSV must include a ticker or symbol column.")

        rows: List[Dict[str, str]] = []
        for raw_row in reader:
            ticker = (raw_row.get(ticker_field) or "").strip().upper()
            company_name = (raw_row.get(name_field) or "").strip() if name_field else ""
            if ticker:
                rows.append({"ticker": ticker, "company_name": company_name})
        return rows


def _ticker_variants(ticker: str) -> List[str]:
    variants: List[str] = []
    for candidate in [ticker, ticker.replace(".", "-"), ticker.replace("/", "-")]:
        normalized = candidate.strip().upper()
        if normalized and normalized not in variants:
            variants.append(normalized)
    return variants


def _fetch_sec_company_tickers(session: requests.Session) -> Dict[str, Dict[str, Any]]:
    response = _session_get(session, SEC_COMPANY_TICKERS_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()
    iterable = payload.values() if isinstance(payload, dict) else payload

    mapping: Dict[str, Dict[str, Any]] = {}
    for row in iterable:
        ticker = str(row.get("ticker", "")).upper()
        cik = row.get("cik_str")
        if ticker and cik is not None:
            mapping[ticker] = {
                "ticker": ticker,
                "company_name": row.get("title", ""),
                "cik": int(cik),
            }
    return mapping


def _fetch_submission_json(session: requests.Session, cik: int) -> Dict[str, Any]:
    response = _session_get(session, f"https://data.sec.gov/submissions/CIK{cik:010d}.json", timeout=30)
    response.raise_for_status()
    return response.json()


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _recent_filings_from_submission(submission_json: Mapping[str, Any], form_type: str) -> List[Dict[str, Any]]:
    recent = submission_json.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])
    report_dates = recent.get("reportDate", [])
    filings: List[Dict[str, Any]] = []
    for index, form in enumerate(forms):
        if form != form_type:
            continue
        accession = accessions[index] if index < len(accessions) else None
        if accession:
            filings.append(
                {
                    "accession": accession,
                    "filed_at": filing_dates[index] if index < len(filing_dates) else None,
                    "report_date": report_dates[index] if index < len(report_dates) else None,
                    "primary_doc": primary_docs[index] if index < len(primary_docs) else None,
                }
            )
    filings.sort(
        key=lambda filing: (
            _parse_iso_date(filing.get("report_date")) or date.min,
            _parse_iso_date(filing.get("filed_at")) or date.min,
            filing.get("accession") or "",
        ),
        reverse=True,
    )
    return filings


def _select_recent_filings(
    submission_json: Mapping[str, Any],
    form_type: str,
    *,
    trailing_fiscal_years: int = 1,
) -> List[Dict[str, Any]]:
    filings = _recent_filings_from_submission(submission_json, form_type)
    if trailing_fiscal_years <= 1:
        return filings[:1]
    if not filings:
        return []

    anchor_date = _parse_iso_date(filings[0].get("report_date")) or _parse_iso_date(filings[0].get("filed_at"))
    if anchor_date is None:
        return filings[:1]
    allowed_years = {anchor_date.year - offset for offset in range(max(1, trailing_fiscal_years))}
    selected = []
    for filing in filings:
        effective_date = _parse_iso_date(filing.get("report_date")) or _parse_iso_date(filing.get("filed_at"))
        if effective_date is None or effective_date.year not in allowed_years:
            continue
        selected.append(filing)
    return selected or filings[:1]


def _zip_url(cik: int, accession: str) -> str:
    compact = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{compact}/{accession}-xbrl.zip"


def _copy_snapshot_file(source_path: str | Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(source_path), str(destination_path))


def build_manifest(
    constituents_path: str,
    corpus_root: str,
    snapshot_date: str,
    trailing_fiscal_years: int = 1,
    constituent_source: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """Build issuer and filing manifests for the configured corpus snapshot.

    The issuer manifest records ticker-to-CIK resolution and ingest eligibility.
    The filing manifest records selected 10-K/10-Q accessions and download metadata.
    
    Parameters
    ----------
    constituents_path : str
        Path to the constituents snapshot CSV file. e.g., 'constituents_snapshot.csv'
    corpus_root : str
        Root directory of the corpus workspace (including manifest, raw, db, derived, eval). e.g., 'corpora/sp500_latest_2026-03-20'
    snapshot_date : str
        Snapshot date in ISO format YYYY-MM-DD.
    trailing_fiscal_years : int, optional
        Number of trailing fiscal years to include when selecting filings.
    constituent_source : Optional[str], optional
        Constituent source selector (for example, local snapshot or fetched source).
    user_agent : Optional[str], optional
        HTTP User-Agent header value for outbound requests.
    
    Returns
    -------
    Dict[str, Any]
        Summary containing manifest output paths and coverage counters, including
        ``snapshot_id``, ``snapshot_date``, ``filing_span``, ``trailing_fiscal_years``,
        ``issuer_manifest``, ``filing_manifest``, ``issuer_count``,
        ``resolved_issuer_count``, ``ingest_enabled_issuer_count``, and
        ``resolved_filing_count``.
    """
    layout = ensure_corpus_layout(corpus_root)
    snapshot_id = _snapshot_id(snapshot_date, trailing_fiscal_years=trailing_fiscal_years)
    session = _get_session(user_agent=user_agent)

    constituents = _read_constituents_snapshot(constituents_path)
    sec_tickers = _fetch_sec_company_tickers(session)
    (layout.manifest / "sec_company_tickers.json").write_text(_json_dumps(sec_tickers), encoding="utf-8")
    _copy_snapshot_file(constituents_path, layout.manifest / "constituents_snapshot.csv")

    issuer_manifest: List[Dict[str, Any]] = []
    filing_manifest: List[Dict[str, Any]] = []
    source_label = constituent_source or Path(constituents_path).name
    canonical_ticker_by_cik: Dict[int, str] = {}
    submission_cache: Dict[int, Dict[str, Any]] = {}

    for constituent in constituents:
        ticker = constituent["ticker"]
        sec_row = next((sec_tickers.get(candidate) for candidate in _ticker_variants(ticker) if sec_tickers.get(candidate)), None)
        canonical_ticker = canonical_ticker_by_cik.get(sec_row["cik"]) if sec_row else None
        ingest_enabled = bool(sec_row and canonical_ticker is None)
        if sec_row and canonical_ticker is None:
            canonical_ticker = ticker
            canonical_ticker_by_cik[sec_row["cik"]] = ticker
        issuer_row = {
            "snapshot_id": snapshot_id,
            "snapshot_date": snapshot_date,
            "filing_span": "latest" if trailing_fiscal_years <= 1 else "trailing_fiscal_years",
            "trailing_fiscal_years": trailing_fiscal_years,
            "ticker": ticker,
            "company_name": constituent["company_name"] or (sec_row["company_name"] if sec_row else ""),
            "cik": sec_row["cik"] if sec_row else None,
            "canonical_ticker": canonical_ticker,
            "duplicate_cik_of": canonical_ticker if sec_row and not ingest_enabled else None,
            "ingest_enabled": ingest_enabled,
            "forms_expected": ["10-K", "10-Q"],
            "constituent_source": source_label,
            "sec_submission_url": f"https://data.sec.gov/submissions/CIK{sec_row['cik']:010d}.json" if sec_row else None,
            "status": "resolved" if sec_row else "unresolved_sec_ticker",
            "error": None if sec_row else "ticker_not_found_in_sec_company_tickers",
        }
        issuer_manifest.append(issuer_row)

        if not sec_row:
            for form_type in ("10-K", "10-Q"):
                filing_manifest.append(
                    {
                        "snapshot_id": snapshot_id,
                        "snapshot_date": snapshot_date,
                        "filing_span": issuer_row["filing_span"],
                        "trailing_fiscal_years": trailing_fiscal_years,
                        "ticker": ticker,
                        "company_name": issuer_row["company_name"],
                        "cik": None,
                        "canonical_ticker": None,
                        "form_type": form_type,
                        "accession": None,
                        "filed_at": None,
                        "primary_doc": None,
                        "report_date": None,
                        "zip_url": None,
                        "ingest_enabled": False,
                        "selection_rank": None,
                        "status": "unresolved_issuer",
                        "error": issuer_row["error"],
                    }
                )
            continue

        try:
            if sec_row["cik"] not in submission_cache:
                submission_json = _fetch_submission_json(session, sec_row["cik"])
                submission_cache[sec_row["cik"]] = submission_json
                submission_path = layout.raw_submissions / f"CIK{sec_row['cik']:010d}.json"
                submission_path.write_text(_json_dumps(submission_json), encoding="utf-8")
            submission_json = submission_cache[sec_row["cik"]]
            for form_type in ("10-K", "10-Q"):
                selected_filings = _select_recent_filings(
                    submission_json,
                    form_type,
                    trailing_fiscal_years=trailing_fiscal_years,
                )
                if not selected_filings:
                    filing_manifest.append(
                        {
                            "snapshot_id": snapshot_id,
                            "snapshot_date": snapshot_date,
                            "filing_span": issuer_row["filing_span"],
                            "trailing_fiscal_years": trailing_fiscal_years,
                            "ticker": ticker,
                            "company_name": issuer_row["company_name"],
                            "cik": sec_row["cik"],
                            "canonical_ticker": canonical_ticker,
                            "form_type": form_type,
                            "accession": None,
                            "filed_at": None,
                            "primary_doc": None,
                            "report_date": None,
                            "zip_url": None,
                            "ingest_enabled": False,
                            "selection_rank": None,
                            "status": "missing_latest_form",
                            "error": f"no_recent_{form_type}",
                        }
                    )
                    continue
                for selection_rank, filing in enumerate(selected_filings, start=1):
                    filing_manifest.append(
                        {
                            "snapshot_id": snapshot_id,
                            "snapshot_date": snapshot_date,
                            "filing_span": issuer_row["filing_span"],
                            "trailing_fiscal_years": trailing_fiscal_years,
                            "ticker": ticker,
                            "company_name": issuer_row["company_name"],
                            "cik": sec_row["cik"],
                            "canonical_ticker": canonical_ticker,
                            "form_type": form_type,
                            "accession": filing["accession"],
                            "filed_at": filing["filed_at"],
                            "primary_doc": filing["primary_doc"],
                            "report_date": filing.get("report_date"),
                            "zip_url": _zip_url(sec_row["cik"], filing["accession"]),
                            "ingest_enabled": ingest_enabled,
                            "selection_rank": selection_rank,
                            "status": "resolved" if ingest_enabled else "duplicate_cik_constituent",
                            "error": None if ingest_enabled else f"duplicate_cik_of:{canonical_ticker}",
                        }
                    )
        except requests.RequestException as error:
            for form_type in ("10-K", "10-Q"):
                filing_manifest.append(
                    {
                        "snapshot_id": snapshot_id,
                        "snapshot_date": snapshot_date,
                        "filing_span": issuer_row["filing_span"],
                        "trailing_fiscal_years": trailing_fiscal_years,
                        "ticker": ticker,
                        "company_name": issuer_row["company_name"],
                        "cik": sec_row["cik"],
                        "canonical_ticker": canonical_ticker,
                        "form_type": form_type,
                        "accession": None,
                        "filed_at": None,
                        "primary_doc": None,
                        "report_date": None,
                        "zip_url": None,
                        "ingest_enabled": False,
                        "selection_rank": None,
                        "status": "submission_fetch_error",
                        "error": str(error),
                    }
                )

    issuer_manifest_path = layout.manifest / "issuer_manifest.jsonl"
    filing_manifest_path = layout.manifest / "filing_manifest.jsonl"
    write_jsonl(issuer_manifest_path, issuer_manifest)
    write_jsonl(filing_manifest_path, filing_manifest)

    return {
        "snapshot_id": snapshot_id,
        "snapshot_date": snapshot_date,
        "filing_span": "latest" if trailing_fiscal_years <= 1 else "trailing_fiscal_years",
        "trailing_fiscal_years": trailing_fiscal_years,
        "issuer_manifest": str(issuer_manifest_path),
        "filing_manifest": str(filing_manifest_path),
        "issuer_count": len(issuer_manifest),
        "resolved_issuer_count": sum(1 for row in issuer_manifest if row["status"] == "resolved"),
        "ingest_enabled_issuer_count": sum(1 for row in issuer_manifest if row.get("ingest_enabled")),
        "resolved_filing_count": sum(1 for row in filing_manifest if row["status"] == "resolved"),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_with_resume(session: requests.Session, url: str, output_path: Path) -> tuple[int, Optional[int]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers: Dict[str, str] = {}
    local_size = 0
    if output_path.exists():
        local_size = output_path.stat().st_size
        if local_size > 0:
            headers["Range"] = f"bytes={local_size}-"

    response = _session_get(session, url, headers=headers, stream=True, timeout=60)
    if response.status_code == 416:
        return response.status_code, None
    response.raise_for_status()
    mode = "ab" if headers.get("Range") and response.status_code == 206 else "wb"
    if mode == "wb" and output_path.exists() and local_size > 0:
        output_path.unlink()
    with output_path.open(mode, buffering=1024 * 1024) as handle:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                handle.write(chunk)
    return response.status_code, output_path.stat().st_size


def _is_retryable_error(error: requests.RequestException) -> bool:
    if isinstance(error, requests.Timeout):
        return True
    if isinstance(error, requests.HTTPError):
        status_code = error.response.status_code if error.response is not None else None
        return status_code in RETRYABLE_HTTP_STATUS_CODES
    return True


def download_filings(corpus_root: str, user_agent: Optional[str] = None, max_attempts: int = 3) -> Dict[str, Any]:
    """Download filing packages referenced by the corpus manifest.
    
    Parameters
    ----------
    corpus_root : str
        Root directory of the corpus workspace (manifest/raw/db/derived/eval). e.g., 'corpora/sp500_latest_2026-03-20'
    user_agent : Optional[str], optional
        HTTP User-Agent header value for outbound requests.
    max_attempts : int, optional
        Maximum retry attempts for transient failures.
    
    Returns
    -------
    Dict[str, Any]
        Summary dictionary containing generated outputs, counters, and run metadata.
    
    Raises
    ------
    Exception
        Raised when required inputs or runtime state do not satisfy function preconditions.
    """
    layout = ensure_corpus_layout(corpus_root)
    session = _get_session(user_agent=user_agent)
    filing_manifest = read_jsonl(layout.manifest / "filing_manifest.jsonl")
    ledger_rows: List[Dict[str, Any]] = []

    for filing in filing_manifest:
        ledger_row = dict(filing)
        local_path = None
        attempts = 0
        if filing["status"] == "resolved" and filing["accession"]:
            local_path = layout.raw_xbrl_zip / filing["ticker"] / f"{filing['ticker']}_{filing['form_type']}_{filing['accession']}.zip"
            ledger_row["local_path"] = str(local_path)
            try:
                if local_path.exists() and local_path.stat().st_size > 0:
                    ledger_row.update(
                        {
                            "status": "existing_local",
                            "http_status": None,
                            "sha256": _sha256_file(local_path),
                            "bytes": local_path.stat().st_size,
                            "error": None,
                            "attempts": 0,
                        }
                    )
                else:
                    attempts = 0
                    while True:
                        attempts += 1
                        try:
                            http_status, byte_count = _download_with_resume(session, filing["zip_url"], local_path)
                            if http_status == 416 and local_path.exists():
                                ledger_row.update(
                                    {
                                        "status": "existing_local",
                                        "http_status": http_status,
                                        "sha256": _sha256_file(local_path),
                                        "bytes": local_path.stat().st_size,
                                        "error": None,
                                        "attempts": attempts,
                                    }
                                )
                            else:
                                ledger_row.update(
                                    {
                                        "status": "downloaded",
                                        "http_status": http_status,
                                        "sha256": _sha256_file(local_path),
                                        "bytes": byte_count,
                                        "error": None,
                                        "attempts": attempts,
                                    }
                                )
                            break
                        except requests.RequestException as error:
                            if attempts >= max_attempts or not _is_retryable_error(error):
                                raise
            except requests.HTTPError as error:
                http_status = error.response.status_code if error.response is not None else None
                ledger_row.update(
                    {
                        "status": "download_error",
                        "http_status": http_status,
                        "sha256": None,
                        "bytes": local_path.stat().st_size if local_path and local_path.exists() else None,
                        "error": str(error),
                        "attempts": max(attempts, 1),
                    }
                )
            except requests.RequestException as error:
                ledger_row.update(
                    {
                        "status": "download_error",
                        "http_status": None,
                        "sha256": None,
                        "bytes": local_path.stat().st_size if local_path and local_path.exists() else None,
                        "error": str(error),
                        "attempts": max(attempts, 1),
                    }
                )
        else:
            ledger_row.update(
                {
                    "local_path": None,
                    "http_status": None,
                    "sha256": None,
                    "bytes": None,
                    "error": filing.get("error"),
                    "attempts": 0,
                }
            )
        ledger_rows.append(ledger_row)

    ledger_path = layout.manifest / "download_ledger.jsonl"
    write_jsonl(ledger_path, ledger_rows)
    return {
        "download_ledger": str(ledger_path),
        "attempted": len(ledger_rows),
        "success_count": sum(1 for row in ledger_rows if row["status"] in SUCCESS_DOWNLOAD_STATUSES),
        "error_count": sum(1 for row in ledger_rows if row["status"] == "download_error"),
    }


def ingest_corpus(corpus_root: str, extract_narrative: bool = True) -> Dict[str, Any]:
    """Ingest downloaded filing packages into the corpus database.
    
    Parameters
    ----------
    corpus_root : str
        Root directory of the corpus workspace (manifest/raw/db/derived/eval). e.g., 'corpora/sp500_latest_2026-03-20'
    extract_narrative : bool, optional
        Whether narrative chunks should be extracted during ingestion.
    
    Returns
    -------
    Dict[str, Any]
        Summary dictionary containing generated outputs, counters, and run metadata.
    """
    layout = ensure_corpus_layout(corpus_root)
    ledger_rows = read_jsonl(layout.manifest / "download_ledger.jsonl")
    ingest_rows: List[Dict[str, Any]] = []

    if layout.db_path.exists():
        layout.db_path.unlink()

    success_rows = [row for row in ledger_rows if row["status"] in SUCCESS_DOWNLOAD_STATUSES and row.get("local_path")]
    for index, row in enumerate(success_rows):
        try:
            summary = process_zip(
                zip_path=row["local_path"],
                out_db=str(layout.db_path),
                ticker=row["ticker"],
                extract_narrative=extract_narrative,
                reset_db=(index == 0),
                metadata_overrides={
                    "form_type": row["form_type"],
                    "preferred_main_html": row.get("primary_doc"),
                },
            )
            ingest_rows.append(
                {
                    "ticker": row["ticker"],
                    "form_type": row["form_type"],
                    "accession": row["accession"],
                    "local_path": row["local_path"],
                    "status": "ingested",
                    "error": None,
                    **summary,
                }
            )
        except Exception as error:  # pragma: no cover - defensive path
            ingest_rows.append(
                {
                    "ticker": row["ticker"],
                    "form_type": row["form_type"],
                    "accession": row["accession"],
                    "local_path": row["local_path"],
                    "status": "ingest_error",
                    "error": str(error),
                }
            )

    ingest_ledger_path = layout.logs_dir / "ingest_ledger.jsonl"
    write_jsonl(ingest_ledger_path, ingest_rows)
    return {
        "db": str(layout.db_path),
        "ingest_ledger": str(ingest_ledger_path),
        "ingested_count": sum(1 for row in ingest_rows if row["status"] == "ingested"),
        "error_count": sum(1 for row in ingest_rows if row["status"] == "ingest_error"),
    }


def repair_corpus_filing(
    corpus_root: str,
    ticker: str,
    accession: str,
    extract_narrative: bool = True,
) -> Dict[str, Any]:
    """Reprocess one filing and refresh derived artifacts in place.
    
    Parameters
    ----------
    corpus_root : str
        Root directory of the corpus workspace (manifest/raw/db/derived/eval). e.g., 'corpora/sp500_latest_2026-03-20'
    ticker : str
        Issuer ticker symbol used in manifests and derived outputs. e.g., 'AAPL'
    accession : str
        SEC accession number identifying a filing package.
    extract_narrative : bool, optional
        Whether narrative chunks should be extracted during ingestion.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with fields produced while reprocess one filing and refresh derived artifacts in place.
    
    Raises
    ------
    ValueError
        No download ledger row found for ticker={...} accession={...}.
    ValueError
        Download ledger row for ticker={...} accession={...} is not locally available.
    Exception
        Raised when required inputs or runtime state do not satisfy function preconditions.
    """
    layout = ensure_corpus_layout(corpus_root)
    ledger_rows = read_jsonl(layout.manifest / "download_ledger.jsonl")
    download_row = next(
        (
            row
            for row in ledger_rows
            if row.get("ticker") == ticker and row.get("accession") == accession
        ),
        None,
    )
    if download_row is None:
        raise ValueError(f"No download ledger row found for ticker={ticker} accession={accession}.")
    if download_row["status"] not in SUCCESS_DOWNLOAD_STATUSES or not download_row.get("local_path"):
        raise ValueError(
            f"Download ledger row for ticker={ticker} accession={accession} is not locally available."
        )

    ingest_ledger_path = layout.logs_dir / "ingest_ledger.jsonl"
    existing_ingest_rows = read_jsonl(ingest_ledger_path) if ingest_ledger_path.exists() else []
    keep_rows = [
        row
        for row in existing_ingest_rows
        if not (
            row.get("ticker") == ticker
            and row.get("form_type") == download_row.get("form_type")
            and row.get("accession") == accession
        )
    ]

    try:
        summary = process_zip(
            zip_path=download_row["local_path"],
            out_db=str(layout.db_path),
            ticker=download_row["ticker"],
            extract_narrative=extract_narrative,
            reset_db=not layout.db_path.exists(),
            metadata_overrides={
                "form_type": download_row["form_type"],
                "preferred_main_html": download_row.get("primary_doc"),
            },
        )
        repaired_row = {
            "ticker": download_row["ticker"],
            "form_type": download_row["form_type"],
            "accession": download_row["accession"],
            "local_path": download_row["local_path"],
            "status": "ingested",
            "error": None,
            **summary,
        }
        keep_rows.append(repaired_row)
        write_jsonl(ingest_ledger_path, keep_rows)
        return {
            "db": str(layout.db_path),
            "ingest_ledger": str(ingest_ledger_path),
            "repaired": repaired_row,
        }
    except Exception as error:
        keep_rows.append(
            {
                "ticker": download_row["ticker"],
                "form_type": download_row["form_type"],
                "accession": download_row["accession"],
                "local_path": download_row["local_path"],
                "status": "ingest_error",
                "error": str(error),
            }
        )
        write_jsonl(ingest_ledger_path, keep_rows)
        raise


def _issuer_split(issuer_manifest: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    split_rows: List[Dict[str, Any]] = []
    for issuer in issuer_manifest:
        if not issuer.get("ingest_enabled", issuer.get("status") == "resolved"):
            continue
        digest = int(_stable_digest("issuer-holdout", ISSUER_HOLDOUT_VERSION, issuer["ticker"])[:8], 16)
        split = "eval_holdout" if digest % 5 == 0 else "train"
        split_rows.append(
            {
                "split_manifest_version": ISSUER_HOLDOUT_VERSION,
                "ticker": issuer["ticker"],
                "snapshot_id": issuer["snapshot_id"],
                "snapshot_date": issuer["snapshot_date"],
                "issuer_status": issuer["status"],
                "split": split,
            }
        )
    if split_rows and all(row["split"] == "train" for row in split_rows):
        split_rows[-1]["split"] = "eval_holdout"
    return split_rows


def _render_corpus_datasets(
    task_specs: Sequence[Dict[str, Any]],
    split_manifest: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    split_by_ticker = {row["ticker"]: row["split"] for row in split_manifest}
    output_dir.mkdir(parents=True, exist_ok=True)

    task_specs_rows = []
    qa_rows: List[Dict[str, Any]] = []
    code_rows: List[Dict[str, Any]] = []
    refusal_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []

    for task_spec in task_specs:
        split = split_by_ticker.get(task_spec["ticker"], "train")
        enriched_task_spec = dict(task_spec)
        enriched_task_spec["split"] = split
        enriched_task_spec["split_manifest_version"] = ISSUER_HOLDOUT_VERSION
        task_specs_rows.append(enriched_task_spec)

        record = {
            "render_version": RENDER_VERSION,
            "split": split,
            "task_id": task_spec["task_id"],
            "source_answer_id": task_spec["source_answer_id"],
            "filing_id": task_spec["filing_id"],
            "ticker": task_spec["ticker"],
            "metric_spec_id": task_spec["metric_spec_id"],
            "period_key": task_spec["period"]["period_key"],
            "question": _build_question(task_spec),
            "task_plan": build_task_plan_target(task_spec),
            "target_answer": dict(task_spec["target_answer"]),
            "evidence_requirements": dict(task_spec["evidence_requirements"]),
            "negative_type": task_spec["negative_type"],
        }
        if split == "eval_holdout":
            eval_rows.append(record)
        elif task_spec["target_status"] == "OK":
            qa_rows.append(record)
            code_rows.append({**record, "code_target": _build_code_target(task_spec)})
        else:
            refusal_rows.append(record)

    hard_negatives = build_hard_negatives(task_specs_rows)

    task_specs_path = output_dir.parent / "tasks" / "task_specs_quant.jsonl"
    task_specs_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "task_specs": task_specs_path,
        "train_quant_qa": output_dir / "train_quant_qa.jsonl",
        "train_quant_code": output_dir / "train_quant_code.jsonl",
        "train_refusal": output_dir / "train_refusal.jsonl",
        "hard_negatives": output_dir / "hard_negatives_quant.jsonl",
        "eval_holdout": output_dir / "eval_holdout.jsonl",
    }

    counts = {
        "task_specs": write_jsonl(paths["task_specs"], task_specs_rows),
        "train_quant_qa": write_jsonl(paths["train_quant_qa"], qa_rows),
        "train_quant_code": write_jsonl(paths["train_quant_code"], code_rows),
        "train_refusal": write_jsonl(paths["train_refusal"], refusal_rows),
        "hard_negatives": write_jsonl(paths["hard_negatives"], hard_negatives),
        "eval_holdout": write_jsonl(paths["eval_holdout"], eval_rows),
    }
    return {"counts": counts, "paths": {name: str(path) for name, path in paths.items()}}


def generate_corpus_datasets(corpus_root: str) -> Dict[str, Any]:
    """Generate derived answers, tasks, and benchmark datasets.
    
    Parameters
    ----------
    corpus_root : str
        Root directory of the corpus workspace (including manifest, raw, db, derived, eval). e.g., 'corpora/sp500_latest_2026-03-20'
    
    Returns
    -------
    Dict[str, Any]
        Summary dictionary containing generated outputs, counters, and run metadata.
    """
    layout = ensure_corpus_layout(corpus_root)
    issuer_manifest = read_jsonl(layout.manifest / "issuer_manifest.jsonl")
    split_manifest = _issuer_split(issuer_manifest)
    split_manifest_path = layout.eval_dir / "split_manifest.jsonl"
    write_jsonl(split_manifest_path, split_manifest)
    split_by_ticker = {row["ticker"]: row["split"] for row in split_manifest}

    answer_path = layout.derived_answers / "answer_objects_quant.jsonl"
    task_specs_path = layout.derived_tasks / "task_specs_quant.jsonl"
    dataset_paths = {
        "train_quant_qa": layout.derived_datasets / "train_quant_qa.jsonl",
        "train_quant_code": layout.derived_datasets / "train_quant_code.jsonl",
        "train_refusal": layout.derived_datasets / "train_refusal.jsonl",
        "hard_negatives": layout.derived_datasets / "hard_negatives_quant.jsonl",
        "eval_holdout": layout.derived_datasets / "eval_holdout.jsonl",
    }
    counts = {
        "task_specs": 0,
        "train_quant_qa": 0,
        "train_quant_code": 0,
        "train_refusal": 0,
        "hard_negatives": 0,
        "eval_holdout": 0,
    }
    answer_count = 0

    conn = connect_db(str(layout.db_path))
    try:
        filing_rows = [
            dict(row)
            for row in conn.execute("SELECT filing_id FROM filings ORDER BY filing_id").fetchall()
        ]
        with answer_path.open("w", encoding="utf-8") as answer_handle, \
            task_specs_path.open("w", encoding="utf-8") as task_handle, \
            dataset_paths["train_quant_qa"].open("w", encoding="utf-8") as qa_handle, \
            dataset_paths["train_quant_code"].open("w", encoding="utf-8") as code_handle, \
            dataset_paths["train_refusal"].open("w", encoding="utf-8") as refusal_handle, \
            dataset_paths["hard_negatives"].open("w", encoding="utf-8") as negatives_handle, \
            dataset_paths["eval_holdout"].open("w", encoding="utf-8") as eval_handle:
            for filing_row in filing_rows:
                current_filing_id = filing_row["filing_id"]
                answers = generate_answer_objects(conn, filing_id=current_filing_id)
                for answer in answers:
                    _write_jsonl_line(answer_handle, answer)
                    answer_count += 1

                task_specs = build_task_specs_from_answers(conn, answers, filing_id=current_filing_id)
                for task_spec in task_specs:
                    split = split_by_ticker.get(task_spec["ticker"], "train")
                    enriched_task_spec = dict(task_spec)
                    enriched_task_spec["split"] = split
                    enriched_task_spec["split_manifest_version"] = ISSUER_HOLDOUT_VERSION
                    _write_jsonl_line(task_handle, enriched_task_spec)
                    counts["task_specs"] += 1

                    record = {
                        "render_version": RENDER_VERSION,
                        "split": split,
                        "task_id": task_spec["task_id"],
                        "source_answer_id": task_spec["source_answer_id"],
                        "filing_id": task_spec["filing_id"],
                        "ticker": task_spec["ticker"],
                        "metric_spec_id": task_spec["metric_spec_id"],
                        "period_key": task_spec["period"]["period_key"],
                        "question": _build_question(task_spec),
                        "task_plan": build_task_plan_target(task_spec),
                        "target_answer": dict(task_spec["target_answer"]),
                        "evidence_requirements": dict(task_spec["evidence_requirements"]),
                        "negative_type": task_spec["negative_type"],
                    }
                    if split == "eval_holdout":
                        _write_jsonl_line(eval_handle, record)
                        counts["eval_holdout"] += 1
                    elif task_spec["target_status"] == "OK":
                        _write_jsonl_line(qa_handle, record)
                        counts["train_quant_qa"] += 1
                        _write_jsonl_line(code_handle, {**record, "code_target": _build_code_target(task_spec)})
                        counts["train_quant_code"] += 1
                    else:
                        _write_jsonl_line(refusal_handle, record)
                        counts["train_refusal"] += 1

                hard_negatives = build_hard_negatives(task_specs)
                for hard_negative in hard_negatives:
                    _write_jsonl_line(negatives_handle, hard_negative)
                    counts["hard_negatives"] += 1
    finally:
        conn.close()

    return {
        "answers": {"path": str(answer_path), "count": answer_count},
        "split_manifest": str(split_manifest_path),
        "counts": counts,
        "paths": {
            "task_specs": str(task_specs_path),
            "train_quant_qa": str(dataset_paths["train_quant_qa"]),
            "train_quant_code": str(dataset_paths["train_quant_code"]),
            "train_refusal": str(dataset_paths["train_refusal"]),
            "hard_negatives": str(dataset_paths["hard_negatives"]),
            "eval_holdout": str(dataset_paths["eval_holdout"]),
        },
    }


def _decimal_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left == right
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, ValueError):
        return str(left) == str(right)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _runtime_failure_reasons(
    target: Mapping[str, Any],
    predicted: Mapping[str, Any],
    *,
    status_match: bool,
    period_match: bool,
    unit_match: bool,
    value_match: bool,
    evidence_match: bool,
    refusal_match: bool,
) -> List[str]:
    reasons: List[str] = []
    if predicted.get("refusal_code") == "TASK_NOT_SUPPORTED":
        reasons.append("router_task_not_supported")
    if not status_match:
        reasons.append("status_mismatch")
    if not period_match:
        reasons.append("period_mismatch")
    if not unit_match:
        reasons.append("unit_mismatch")
    if target["status"] == "OK":
        if not value_match:
            reasons.append("value_mismatch")
    elif not refusal_match:
        reasons.append("refusal_code_mismatch")
    if not evidence_match:
        reasons.append("evidence_mismatch")
    if not reasons:
        reasons.append("other")
    return reasons


def _primary_failure_bucket(reasons: Sequence[str]) -> str:
    priority = [
        "router_task_not_supported",
        "status_mismatch",
        "period_mismatch",
        "unit_mismatch",
        "refusal_code_mismatch",
        "evidence_mismatch",
        "value_mismatch",
        "other",
    ]
    for bucket in priority:
        if bucket in reasons:
            return bucket
    return "other"


def eval_corpus(corpus_root: str) -> Dict[str, Any]:
    """Evaluate corpus coverage, quality, and runtime baseline metrics.
    
    Parameters
    ----------
    corpus_root : str
        Root directory of the corpus workspace (including manifest, raw, db, derived, eval). e.g., 'corpora/sp500_latest_2026-03-20'
    
    Returns
    -------
    Dict[str, Any]
        Summary dictionary containing generated outputs, counters, and run metadata.
    """
    layout = ensure_corpus_layout(corpus_root)
    issuer_manifest = read_jsonl(layout.manifest / "issuer_manifest.jsonl")
    filing_manifest = read_jsonl(layout.manifest / "filing_manifest.jsonl")
    download_ledger = read_jsonl(layout.manifest / "download_ledger.jsonl")
    ingest_ledger = read_jsonl(layout.logs_dir / "ingest_ledger.jsonl")
    split_manifest = read_jsonl(layout.eval_dir / "split_manifest.jsonl")
    answer_objects_path = layout.derived_answers / "answer_objects_quant.jsonl"
    task_specs_path = layout.derived_tasks / "task_specs_quant.jsonl"
    eval_holdout_path = layout.derived_datasets / "eval_holdout.jsonl"
    data_quality_path = layout.eval_dir / "data_quality_summary.json"
    runtime_eval_path = layout.eval_dir / "runtime_eval_summary.json"
    metric_csv_path = layout.eval_dir / "coverage_by_metric.csv"
    issuer_csv_path = layout.eval_dir / "coverage_by_issuer.csv"
    validator_csv_path = layout.eval_dir / "validator_distribution.csv"
    failures_path = layout.eval_dir / "runtime_failures.jsonl"
    failure_buckets_path = layout.eval_dir / "runtime_failure_buckets.json"

    answers_by_status = Counter()
    for answer in _iter_jsonl(answer_objects_path):
        answers_by_status[answer["status"]] += 1

    metric_counter: Counter[tuple[str, str]] = Counter()
    issuer_counter: Counter[tuple[str, str]] = Counter()
    for task_spec in _iter_jsonl(task_specs_path):
        split = task_spec.get("split", "unspecified")
        metric_counter[(task_spec["metric_spec_id"], split)] += 1
        issuer_counter[(task_spec["ticker"], split)] += 1

    eval_holdout_count = 0
    conn = connect_db(str(layout.db_path))
    try:
        validator_distribution = Counter(row["validator_code"] for row in fetch_validators(conn))
        failure_bucket_counter = Counter()
        runtime_counters = Counter()
        current_filing_id: Optional[str] = None
        filing_task_specs: List[Dict[str, Any]] = []
        failure_count = 0
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        with failures_path.open("w", encoding="utf-8") as failures_handle:
            for row in _iter_jsonl(eval_holdout_path):
                eval_holdout_count += 1
                filing_id = row["filing_id"]
                if filing_id != current_filing_id:
                    filing_task_specs = build_task_specs(conn, filing_id=filing_id)
                    current_filing_id = filing_id

                predicted = answer_quant(conn, row["question"], row["filing_id"], task_specs=filing_task_specs)
                target = row["target_answer"]

                status_match = predicted["status"] == target["status"]
                period_match = predicted["period_key"] == target["period_key"]
                unit_match = predicted["unit"] == target["unit"]
                value_match = _decimal_equal(predicted["value"], target["value"])
                evidence_match = sorted(predicted["evidence_ids"]) == sorted(target["evidence_ids"])
                refusal_match = predicted["refusal_code"] == target["refusal_code"]

                runtime_counters["total"] += 1
                runtime_counters["status_match"] += int(status_match)
                runtime_counters["period_match"] += int(period_match)
                runtime_counters["unit_match"] += int(unit_match)
                runtime_counters["evidence_match"] += int(evidence_match)

                if target["status"] == "OK":
                    runtime_counters["ok_total"] += 1
                    runtime_counters["numeric_match"] += int(status_match and value_match)
                    runtime_counters["context_match"] += int(evidence_match)
                else:
                    runtime_counters["refusal_total"] += 1
                    runtime_counters["refusal_match"] += int(status_match and refusal_match)

                unsupported = predicted["status"] == "OK" and (target["status"] != "OK" or not predicted["evidence_ids"])
                runtime_counters["unsupported_claims"] += int(unsupported)

                fully_correct = status_match and period_match and unit_match and evidence_match and (
                    value_match if target["status"] == "OK" else refusal_match
                )
                if not fully_correct:
                    failure_reasons = _runtime_failure_reasons(
                        target,
                        predicted,
                        status_match=status_match,
                        period_match=period_match,
                        unit_match=unit_match,
                        value_match=value_match,
                        evidence_match=evidence_match,
                        refusal_match=refusal_match,
                    )
                    failure_bucket = _primary_failure_bucket(failure_reasons)
                    failure_bucket_counter[failure_bucket] += 1
                    _write_jsonl_line(
                        failures_handle,
                        {
                            "task_id": row["task_id"],
                            "filing_id": row["filing_id"],
                            "ticker": row["ticker"],
                            "metric_spec_id": row["metric_spec_id"],
                            "question": row["question"],
                            "failure_bucket": failure_bucket,
                            "failure_reasons": failure_reasons,
                            "target": target,
                            "predicted": predicted,
                        },
                    )
                    failure_count += 1
    finally:
        conn.close()

    coverage_by_metric = [
        {"metric_spec_id": metric_spec_id, "split": split, "task_count": count}
        for (metric_spec_id, split), count in sorted(metric_counter.items())
    ]
    coverage_by_issuer = [
        {"ticker": ticker, "split": split, "task_count": count}
        for (ticker, split), count in sorted(issuer_counter.items())
    ]

    data_quality_summary = {
        "snapshot_id": issuer_manifest[0]["snapshot_id"] if issuer_manifest else None,
        "manifest_coverage": {
            "issuer_count": len(issuer_manifest),
            "resolved_issuer_count": sum(1 for row in issuer_manifest if row["status"] == "resolved"),
            "ingest_enabled_issuer_count": sum(1 for row in issuer_manifest if row.get("ingest_enabled")),
            "filing_target_count": len(filing_manifest),
            "resolved_filing_count": sum(1 for row in filing_manifest if row["status"] == "resolved"),
        },
        "sec_resolution_rate": (
            sum(1 for row in issuer_manifest if row.get("cik") is not None) / len(issuer_manifest)
            if issuer_manifest
            else 0.0
        ),
        "download_success_rate": (
            sum(1 for row in download_ledger if row["status"] in SUCCESS_DOWNLOAD_STATUSES)
            / max(1, sum(1 for row in filing_manifest if row["status"] == "resolved"))
        ),
        "ingest_success_rate": (
            sum(1 for row in ingest_ledger if row["status"] == "ingested")
            / max(1, sum(1 for row in download_ledger if row["status"] in SUCCESS_DOWNLOAD_STATUSES))
        ),
        "validator_distribution": dict(sorted(validator_distribution.items())),
        "answer_object_status_mix": dict(sorted(answers_by_status.items())),
        "dataset_counts": {
            "task_specs": sum(metric_counter.values()),
            "eval_holdout": eval_holdout_count,
            "split_manifest": len(split_manifest),
        },
    }

    runtime_eval_summary = {
        "split_manifest_version": ISSUER_HOLDOUT_VERSION,
        "eval_task_count": runtime_counters["total"],
        "numeric_accuracy": runtime_counters["numeric_match"] / max(1, runtime_counters["ok_total"]),
        "refusal_correctness": runtime_counters["refusal_match"] / max(1, runtime_counters["refusal_total"]),
        "unit_accuracy": runtime_counters["unit_match"] / max(1, runtime_counters["total"]),
        "period_accuracy": runtime_counters["period_match"] / max(1, runtime_counters["total"]),
        "context_accuracy": runtime_counters["context_match"] / max(1, runtime_counters["ok_total"]),
        "unsupported_claim_rate": runtime_counters["unsupported_claims"] / max(1, runtime_counters["total"]),
        "evidence_id_exactness": runtime_counters["evidence_match"] / max(1, runtime_counters["total"]),
        "status_accuracy": runtime_counters["status_match"] / max(1, runtime_counters["total"]),
        "failure_count": failure_count,
        "failure_buckets": dict(sorted(failure_bucket_counter.items())),
    }

    data_quality_path.write_text(_json_dumps(data_quality_summary), encoding="utf-8")
    runtime_eval_path.write_text(_json_dumps(runtime_eval_summary), encoding="utf-8")
    _write_csv(metric_csv_path, coverage_by_metric, ["metric_spec_id", "split", "task_count"])
    _write_csv(issuer_csv_path, coverage_by_issuer, ["ticker", "split", "task_count"])
    _write_csv(
        validator_csv_path,
        [{"validator_code": code, "count": count} for code, count in sorted(validator_distribution.items())],
        ["validator_code", "count"],
    )
    failure_buckets_path.write_text(
        _json_dumps(
            {
                "split_manifest_version": ISSUER_HOLDOUT_VERSION,
                "failure_count": failure_count,
                "failure_buckets": dict(sorted(failure_bucket_counter.items())),
            }
        ),
        encoding="utf-8",
    )

    return {
        "data_quality_summary": str(data_quality_path),
        "runtime_eval_summary": str(runtime_eval_path),
        "coverage_by_metric": str(metric_csv_path),
        "coverage_by_issuer": str(issuer_csv_path),
        "validator_distribution": str(validator_csv_path),
        "runtime_failures": str(failures_path),
        "runtime_failure_buckets": str(failure_buckets_path),
        "metrics": runtime_eval_summary,
    }
