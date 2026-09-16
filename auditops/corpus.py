from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

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
SEC_USER_AGENT_ENV = "AUDITOPS_SEC_USER_AGENT"
_SEC_CONTACT_RE = re.compile(
    r"(?i)(?<![a-z0-9.!#$%&'*+/=?^_`{|}~-])"
    r"[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+"
)
ISSUER_HOLDOUT_VERSION = "issuer_holdout_v1"
SUCCESS_DOWNLOAD_STATUSES = {"downloaded", "existing_local"}
RETRYABLE_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


class DownloadIntegrityError(requests.RequestException):
    """Raised when an HTTP response cannot prove a complete filing download."""


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


def resolve_sec_user_agent(user_agent: Optional[str] = None) -> str:
    """Return an explicit, contact-bearing SEC identity or fail closed."""

    candidate = user_agent or os.environ.get(SEC_USER_AGENT_ENV)
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError(
            f"SEC access requires --user-agent or {SEC_USER_AGENT_ENV} with a contact email"
        )
    candidate = " ".join(candidate.split())
    if not _SEC_CONTACT_RE.search(candidate):
        raise ValueError("SEC user agent must include a contact email address")
    return candidate


def _get_session(user_agent: Optional[str] = None) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": resolve_sec_user_agent(user_agent),
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


def _validated_snapshot_date(value: str) -> date:
    if not isinstance(value, str):
        raise ValueError("Snapshot date must use YYYY-MM-DD format.")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise ValueError("Snapshot date must use YYYY-MM-DD format.") from error
    if parsed.isoformat() != value:
        raise ValueError("Snapshot date must use YYYY-MM-DD format.")
    if parsed > datetime.now(timezone.utc).date():
        raise ValueError("Snapshot date cannot be in the future.")
    return parsed


def _ensure_new_corpus_state(corpus_root: str | Path) -> None:
    root = Path(corpus_root)
    if not root.exists():
        return
    existing_files = sorted(
        path for path in root.rglob("*") if path.is_file() or path.is_symlink()
    )
    if existing_files:
        first = existing_files[0]
        raise FileExistsError(
            "Corpus manifest construction requires a new or file-empty corpus root; "
            f"existing state starts at {first}"
        )


def _read_constituent_source_manifest(
    constituents_path: str | Path,
    *,
    snapshot_date: str,
    constituent_count: int,
    allow_limited_constituents: bool,
) -> tuple[Path, Dict[str, Any], Path] | None:
    snapshot_path = Path(constituents_path)
    source_path = snapshot_path.with_name(f"{snapshot_path.name}.source.json")
    if not source_path.exists():
        return None
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Constituent source manifest is not valid JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError("Constituent source manifest must be a JSON object.")
    if payload.get("source_manifest_version") != "auditops-us-constituents-source.v2":
        raise ValueError("Constituent source manifest must use the pinned v2 format.")
    if payload.get("output_sha256") != _sha256_file(snapshot_path):
        raise ValueError(
            "Constituent source manifest does not match the supplied snapshot."
        )
    if payload.get("snapshot_date") != snapshot_date:
        raise ValueError(
            "Constituent source manifest snapshot_date does not match the requested snapshot."
        )
    if payload.get("row_count") != constituent_count:
        raise ValueError(
            "Constituent source manifest row_count does not match the supplied snapshot."
        )
    capture_scope = payload.get("capture_scope")
    if capture_scope not in {"full", "limited"}:
        raise ValueError(
            "Constituent source manifest must declare capture_scope as full or limited."
        )
    row_limit = payload.get("row_limit")
    if capture_scope == "full" and row_limit is not None:
        raise ValueError("A full constituent snapshot cannot declare a row_limit.")
    if capture_scope == "limited":
        if (
            isinstance(row_limit, bool)
            or not isinstance(row_limit, int)
            or row_limit <= 0
        ):
            raise ValueError(
                "A limited constituent snapshot must declare a positive row_limit."
            )
        if not allow_limited_constituents:
            raise ValueError(
                "Limited constituent snapshots are disabled; set "
                "allow_limited_constituents=True only for smoke tests."
            )
    expected_as_of = f"{snapshot_date}T23:59:59Z"
    if payload.get("requested_as_of") != expected_as_of:
        raise ValueError(
            "Constituent source manifest requested_as_of is not snapshot-bound."
        )
    revision_id = payload.get("revision_id")
    if (
        isinstance(revision_id, bool)
        or not isinstance(revision_id, int)
        or revision_id <= 0
    ):
        raise ValueError("Constituent source manifest has no valid revision_id.")
    revision_timestamp = payload.get("revision_timestamp")
    try:
        revision_instant = datetime.fromisoformat(
            str(revision_timestamp).replace("Z", "+00:00")
        )
        snapshot_boundary = datetime.fromisoformat(
            expected_as_of.replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError(
            "Constituent source manifest has no valid revision_timestamp."
        ) from error
    if revision_instant.tzinfo is None or revision_instant > snapshot_boundary:
        raise ValueError(
            "Constituent source revision is newer than the snapshot boundary."
        )

    raw_source_name = payload.get("raw_source_path")
    if (
        not isinstance(raw_source_name, str)
        or not raw_source_name
        or Path(raw_source_name).is_absolute()
        or Path(raw_source_name).name != raw_source_name
    ):
        raise ValueError("Constituent source manifest has an unsafe raw_source_path.")
    raw_source_path = source_path.parent / raw_source_name
    if not raw_source_path.is_file():
        raise ValueError("Constituent source manifest raw source file does not exist.")
    raw_source_sha256 = payload.get("raw_source_sha256")
    if (
        not isinstance(raw_source_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", raw_source_sha256) is None
        or payload.get("source_sha256") != raw_source_sha256
        or _sha256_file(raw_source_path) != raw_source_sha256
    ):
        raise ValueError("Constituent raw source SHA-256 binding does not match.")
    return source_path, payload, raw_source_path


def _read_constituents_snapshot(path: str | Path) -> List[Dict[str, str]]:
    snapshot_path = Path(path)
    with snapshot_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Constituent snapshot CSV is missing a header row.")

        fields = {field.lower(): field for field in reader.fieldnames}
        ticker_field = next(
            (fields[name] for name in ("ticker", "symbol") if name in fields), None
        )
        name_field = next(
            (
                fields[name]
                for name in ("company_name", "name", "security")
                if name in fields
            ),
            None,
        )
        if ticker_field is None:
            raise ValueError(
                "Constituent snapshot CSV must include a ticker or symbol column."
            )

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
    response = _session_get(
        session, f"https://data.sec.gov/submissions/CIK{cik:010d}.json", timeout=30
    )
    response.raise_for_status()
    return response.json()


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _recent_filings_from_submission(
    submission_json: Mapping[str, Any],
    form_type: str,
    *,
    filed_on_or_before: Optional[date] = None,
) -> List[Dict[str, Any]]:
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
        filed_at = filing_dates[index] if index < len(filing_dates) else None
        parsed_filed_at = _parse_iso_date(filed_at)
        if filed_on_or_before is not None and (
            parsed_filed_at is None or parsed_filed_at > filed_on_or_before
        ):
            continue
        if accession:
            filings.append(
                {
                    "accession": accession,
                    "filed_at": filed_at,
                    "report_date": report_dates[index]
                    if index < len(report_dates)
                    else None,
                    "primary_doc": primary_docs[index]
                    if index < len(primary_docs)
                    else None,
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
    filed_on_or_before: Optional[date] = None,
) -> List[Dict[str, Any]]:
    filings = _recent_filings_from_submission(
        submission_json,
        form_type,
        filed_on_or_before=filed_on_or_before,
    )
    if trailing_fiscal_years <= 1:
        return filings[:1]
    if not filings:
        return []

    anchor_date = _parse_iso_date(filings[0].get("report_date")) or _parse_iso_date(
        filings[0].get("filed_at")
    )
    if anchor_date is None:
        return filings[:1]
    allowed_years = {
        anchor_date.year - offset for offset in range(max(1, trailing_fiscal_years))
    }
    selected = []
    for filing in filings:
        effective_date = _parse_iso_date(filing.get("report_date")) or _parse_iso_date(
            filing.get("filed_at")
        )
        if effective_date is None or effective_date.year not in allowed_years:
            continue
        selected.append(filing)
    return selected or filings[:1]


def _zip_url(cik: int, accession: str) -> str:
    compact = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{compact}/{accession}-xbrl.zip"
    )


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
    allow_limited_constituents: bool = False,
) -> Dict[str, Any]:
    snapshot_cutoff = _validated_snapshot_date(snapshot_date)
    if (
        isinstance(trailing_fiscal_years, bool)
        or not isinstance(trailing_fiscal_years, int)
        or trailing_fiscal_years <= 0
    ):
        raise ValueError("trailing_fiscal_years must be a positive integer.")
    _ensure_new_corpus_state(corpus_root)
    constituents = _read_constituents_snapshot(constituents_path)
    if not constituents:
        raise ValueError("Constituent snapshot must contain at least one ticker.")
    source_capture = _read_constituent_source_manifest(
        constituents_path,
        snapshot_date=snapshot_date,
        constituent_count=len(constituents),
        allow_limited_constituents=allow_limited_constituents,
    )
    bootstrap_source_path, bootstrap_source, bootstrap_raw_source_path = (
        source_capture if source_capture is not None else (None, None, None)
    )

    layout = ensure_corpus_layout(corpus_root)
    snapshot_id = _snapshot_id(
        snapshot_date, trailing_fiscal_years=trailing_fiscal_years
    )
    session = _get_session(user_agent=user_agent)
    retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    sec_tickers = _fetch_sec_company_tickers(session)
    sec_tickers_path = layout.manifest / "sec_company_tickers.json"
    sec_tickers_path.write_text(_json_dumps(sec_tickers), encoding="utf-8")
    constituents_snapshot_path = layout.manifest / "constituents_snapshot.csv"
    _copy_snapshot_file(constituents_path, constituents_snapshot_path)
    if bootstrap_source_path is not None:
        _copy_snapshot_file(
            bootstrap_source_path,
            layout.manifest / "constituents_source_manifest.json",
        )
        _copy_snapshot_file(
            bootstrap_raw_source_path,
            layout.manifest / bootstrap_raw_source_path.name,
        )

    issuer_manifest: List[Dict[str, Any]] = []
    filing_manifest: List[Dict[str, Any]] = []
    source_label = constituent_source or Path(constituents_path).name
    canonical_ticker_by_cik: Dict[int, str] = {}
    submission_cache: Dict[int, Dict[str, Any]] = {}
    submission_paths: List[Path] = []

    for constituent in constituents:
        ticker = constituent["ticker"]
        sec_row = next(
            (
                sec_tickers.get(candidate)
                for candidate in _ticker_variants(ticker)
                if sec_tickers.get(candidate)
            ),
            None,
        )
        canonical_ticker = (
            canonical_ticker_by_cik.get(sec_row["cik"]) if sec_row else None
        )
        ingest_enabled = bool(sec_row and canonical_ticker is None)
        if sec_row and canonical_ticker is None:
            canonical_ticker = ticker
            canonical_ticker_by_cik[sec_row["cik"]] = ticker
        issuer_row = {
            "snapshot_id": snapshot_id,
            "snapshot_date": snapshot_date,
            "filing_span": "latest"
            if trailing_fiscal_years <= 1
            else "trailing_fiscal_years",
            "trailing_fiscal_years": trailing_fiscal_years,
            "ticker": ticker,
            "company_name": constituent["company_name"]
            or (sec_row["company_name"] if sec_row else ""),
            "cik": sec_row["cik"] if sec_row else None,
            "canonical_ticker": canonical_ticker,
            "duplicate_cik_of": canonical_ticker
            if sec_row and not ingest_enabled
            else None,
            "ingest_enabled": ingest_enabled,
            "forms_expected": ["10-K", "10-Q"],
            "constituent_source": source_label,
            "sec_submission_url": f"https://data.sec.gov/submissions/CIK{sec_row['cik']:010d}.json"
            if sec_row
            else None,
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
                submission_path = (
                    layout.raw_submissions / f"CIK{sec_row['cik']:010d}.json"
                )
                submission_path.write_text(
                    _json_dumps(submission_json), encoding="utf-8"
                )
                submission_paths.append(submission_path)
            submission_json = submission_cache[sec_row["cik"]]
            for form_type in ("10-K", "10-Q"):
                selected_filings = _select_recent_filings(
                    submission_json,
                    form_type,
                    trailing_fiscal_years=trailing_fiscal_years,
                    filed_on_or_before=snapshot_cutoff,
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
                            "status": "resolved"
                            if ingest_enabled
                            else "duplicate_cik_constituent",
                            "error": None
                            if ingest_enabled
                            else f"duplicate_cik_of:{canonical_ticker}",
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

    source_records: List[Dict[str, Any]] = [
        {
            "role": "constituents_snapshot",
            "url": bootstrap_source.get("source_url") if bootstrap_source else None,
            "path": str(constituents_snapshot_path.relative_to(layout.root)),
            "sha256": _sha256_file(constituents_snapshot_path),
            "bytes": constituents_snapshot_path.stat().st_size,
        },
        {
            "role": "sec_company_tickers",
            "url": SEC_COMPANY_TICKERS_URL,
            "path": str(sec_tickers_path.relative_to(layout.root)),
            "sha256": _sha256_file(sec_tickers_path),
            "bytes": sec_tickers_path.stat().st_size,
        },
    ]
    if bootstrap_source is not None:
        copied_bootstrap_manifest = (
            layout.manifest / "constituents_source_manifest.json"
        )
        source_records.append(
            {
                "role": "constituents_source_capture",
                "url": bootstrap_source.get("source_url"),
                "path": str(copied_bootstrap_manifest.relative_to(layout.root)),
                "sha256": _sha256_file(copied_bootstrap_manifest),
                "bytes": copied_bootstrap_manifest.stat().st_size,
                "retrieved_at": bootstrap_source.get("retrieved_at"),
                "source_payload_sha256": bootstrap_source.get("source_sha256"),
            }
        )
    for submission_path in sorted(submission_paths):
        source_records.append(
            {
                "role": "sec_submission",
                "url": (f"https://data.sec.gov/submissions/{submission_path.name}"),
                "path": str(submission_path.relative_to(layout.root)),
                "sha256": _sha256_file(submission_path),
                "bytes": submission_path.stat().st_size,
            }
        )
    artifact_records: Dict[str, Dict[str, Any]] = {
        "issuer_manifest": {
            "path": str(issuer_manifest_path.relative_to(layout.root)),
            "sha256": _sha256_file(issuer_manifest_path),
            "bytes": issuer_manifest_path.stat().st_size,
            "rows": len(issuer_manifest),
        },
        "filing_manifest": {
            "path": str(filing_manifest_path.relative_to(layout.root)),
            "sha256": _sha256_file(filing_manifest_path),
            "bytes": filing_manifest_path.stat().st_size,
            "rows": len(filing_manifest),
        },
    }
    if bootstrap_source is not None:
        copied_raw_source = layout.manifest / bootstrap_raw_source_path.name
        artifact_records["constituents_raw_source"] = {
            "path": str(copied_raw_source.relative_to(layout.root)),
            "sha256": _sha256_file(copied_raw_source),
            "bytes": copied_raw_source.stat().st_size,
            "revision_id": bootstrap_source["revision_id"],
            "revision_timestamp": bootstrap_source["revision_timestamp"],
        }

    source_manifest = {
        "source_manifest_version": "auditops-sec-source.v2",
        "snapshot_id": snapshot_id,
        "snapshot_date": snapshot_date,
        "retrieved_at": retrieved_at,
        "user_agent_sha256": hashlib.sha256(
            session.headers["User-Agent"].encode("utf-8")
        ).hexdigest()
        if hasattr(session, "headers")
        else None,
        "sources": source_records,
        "artifacts": artifact_records,
    }
    source_manifest_path = layout.manifest / "source_manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )

    return {
        "snapshot_id": snapshot_id,
        "snapshot_date": snapshot_date,
        "filing_span": "latest"
        if trailing_fiscal_years <= 1
        else "trailing_fiscal_years",
        "trailing_fiscal_years": trailing_fiscal_years,
        "issuer_manifest": str(issuer_manifest_path),
        "filing_manifest": str(filing_manifest_path),
        "issuer_count": len(issuer_manifest),
        "resolved_issuer_count": sum(
            1 for row in issuer_manifest if row["status"] == "resolved"
        ),
        "ingest_enabled_issuer_count": sum(
            1 for row in issuer_manifest if row.get("ingest_enabled")
        ),
        "resolved_filing_count": sum(
            1 for row in filing_manifest if row["status"] == "resolved"
        ),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256_file(source_manifest_path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_artifact_binding(
    layout: CorpusLayout,
    *,
    artifact_name: str,
    artifact_path: Path,
) -> None:
    source_manifest_path = layout.manifest / "source_manifest.json"
    if not source_manifest_path.exists():
        return
    try:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("SEC source manifest is not valid JSON.") from error
    artifact = source_manifest.get("artifacts", {}).get(artifact_name)
    if not isinstance(artifact, dict):
        raise ValueError(f"SEC source manifest is missing the {artifact_name} binding.")
    expected_relative_path = str(artifact_path.relative_to(layout.root))
    if artifact.get("path") != expected_relative_path:
        raise ValueError(f"SEC source manifest {artifact_name} path does not match.")
    if not artifact_path.is_file():
        raise ValueError(f"Bound {artifact_name} does not exist.")
    if artifact.get("bytes") != artifact_path.stat().st_size:
        raise ValueError(f"Bound {artifact_name} byte count does not match.")
    if artifact.get("sha256") != _sha256_file(artifact_path):
        raise ValueError(f"Bound {artifact_name} SHA-256 does not match.")


def _response_content_length(response: requests.Response) -> int:
    raw_length = getattr(response, "headers", {}).get("Content-Length")
    try:
        length = int(raw_length)
    except (TypeError, ValueError) as error:
        raise DownloadIntegrityError(
            "Response is missing a valid Content-Length."
        ) from error
    if length <= 0:
        raise DownloadIntegrityError("Response Content-Length must be positive.")
    return length


def _write_complete_response(
    response: requests.Response,
    *,
    output_path: Path,
    prefix_path: Optional[Path] = None,
    expected_response_bytes: int,
    expected_final_bytes: int,
) -> None:
    temporary_path = output_path.with_name(f"{output_path.name}.download")
    if temporary_path.exists():
        temporary_path.unlink()
    received = 0
    try:
        with temporary_path.open("wb", buffering=1024 * 1024) as handle:
            if prefix_path is not None:
                with prefix_path.open("rb") as prefix:
                    shutil.copyfileobj(prefix, handle, length=1024 * 1024)
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    received += len(chunk)
        if received != expected_response_bytes:
            raise DownloadIntegrityError(
                "Downloaded response length does not match Content-Length."
            )
        if temporary_path.stat().st_size != expected_final_bytes:
            raise DownloadIntegrityError(
                "Completed download length does not match the server-declared total."
            )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _download_full_response(
    session: requests.Session,
    url: str,
    output_path: Path,
) -> tuple[int, int]:
    response = _session_get(
        session,
        url,
        headers={},
        stream=True,
        timeout=60,
        allow_redirects=False,
    )
    response.raise_for_status()
    if response.status_code != 200:
        raise DownloadIntegrityError(
            f"Full filing download returned unexpected HTTP {response.status_code}."
        )
    content_length = _response_content_length(response)
    _write_complete_response(
        response,
        output_path=output_path,
        expected_response_bytes=content_length,
        expected_final_bytes=content_length,
    )
    return response.status_code, content_length


def _download_with_resume(
    session: requests.Session,
    url: str,
    output_path: Path,
) -> tuple[int, Optional[int]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers: Dict[str, str] = {}
    local_size = 0
    if output_path.exists():
        local_size = output_path.stat().st_size
        if local_size > 0:
            headers["Range"] = f"bytes={local_size}-"

    response = _session_get(
        session,
        url,
        headers=headers,
        stream=True,
        timeout=60,
        allow_redirects=False,
    )
    if response.status_code == 416:
        # A 416 only says the requested range is unsatisfiable. It does not prove
        # that an unbound local file is complete, so fetch the full object again.
        return _download_full_response(session, url, output_path)
    response.raise_for_status()
    content_length = _response_content_length(response)
    if response.status_code == 206:
        if not headers.get("Range"):
            raise DownloadIntegrityError(
                "Unexpected partial response without a range request."
            )
        content_range = getattr(response, "headers", {}).get("Content-Range", "")
        match = _CONTENT_RANGE_RE.fullmatch(content_range)
        if match is None:
            raise DownloadIntegrityError(
                "Partial response is missing a valid Content-Range."
            )
        start, end, total = (int(value) for value in match.groups())
        if start != local_size or end < start or end + 1 != total:
            raise DownloadIntegrityError(
                "Partial response Content-Range is not a complete suffix."
            )
        if content_length != end - start + 1:
            raise DownloadIntegrityError(
                "Partial response Content-Length does not match Content-Range."
            )
        _write_complete_response(
            response,
            output_path=output_path,
            prefix_path=output_path,
            expected_response_bytes=content_length,
            expected_final_bytes=total,
        )
        return response.status_code, total
    if response.status_code != 200:
        raise DownloadIntegrityError(
            f"Filing download returned unexpected HTTP {response.status_code}."
        )
    _write_complete_response(
        response,
        output_path=output_path,
        expected_response_bytes=content_length,
        expected_final_bytes=content_length,
    )
    return response.status_code, content_length


def _validate_zip_archive(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            corrupt_member = archive.testzip()
    except (OSError, zipfile.BadZipFile) as error:
        raise DownloadIntegrityError(
            f"Downloaded filing is not a valid ZIP: {path}"
        ) from error
    if corrupt_member is not None:
        raise DownloadIntegrityError(
            f"Downloaded filing ZIP failed CRC validation at member {corrupt_member}."
        )


def _filing_row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("snapshot_id"),
        row.get("snapshot_date"),
        row.get("ticker"),
        row.get("cik"),
        row.get("form_type"),
        row.get("accession"),
        row.get("zip_url"),
    )


def _validate_sec_filing_target(row: Mapping[str, Any]) -> None:
    if row.get("status") != "resolved" or not row.get("accession"):
        return
    accession = row.get("accession")
    cik = row.get("cik")
    url = row.get("zip_url")
    if (
        not isinstance(accession, str)
        or re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession) is None
    ):
        raise ValueError("Resolved SEC filing has a malformed accession number.")
    if isinstance(cik, bool) or not isinstance(cik, int) or cik <= 0:
        raise ValueError("Resolved SEC filing has an invalid CIK.")
    if not isinstance(url, str):
        raise ValueError("Resolved SEC filing has no ZIP URL.")
    parsed = urlparse(url)
    # The accession prefix identifies the EDGAR submitting identity and is not
    # guaranteed to equal the issuer CIK (filing agents submit valid reports for
    # some issuers).  Bind the issuer and accession independently through the
    # canonical SEC archive path below instead of rejecting those filings.
    expected_path = (
        f"/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{accession}-xbrl.zip"
    )
    if (
        parsed.scheme != "https"
        or parsed.netloc != "www.sec.gov"
        or parsed.path != expected_path
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Resolved SEC filing ZIP URL must exactly match its CIK and accession on www.sec.gov."
        )


def _ledger_file_matches(row: Mapping[str, Any], expected_path: Path) -> bool:
    if row.get("status") not in SUCCESS_DOWNLOAD_STATUSES:
        return False
    recorded_path = row.get("local_path")
    if not isinstance(recorded_path, str) or Path(recorded_path) != expected_path:
        return False
    expected_bytes = row.get("bytes")
    expected_sha256 = row.get("sha256")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or not expected_path.is_file()
        or expected_path.stat().st_size != expected_bytes
        or _sha256_file(expected_path) != expected_sha256
    ):
        return False
    try:
        _validate_zip_archive(expected_path)
    except DownloadIntegrityError:
        return False
    return True


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    write_jsonl(temporary_path, rows)
    os.replace(temporary_path, path)


def _verify_download_ledger_manifest_binding(
    layout: CorpusLayout,
    filing_manifest: Sequence[Mapping[str, Any]],
    ledger_rows: Sequence[Mapping[str, Any]],
) -> None:
    if not (layout.manifest / "source_manifest.json").exists():
        return
    manifest_keys = [_filing_row_key(row) for row in filing_manifest]
    ledger_keys = [_filing_row_key(row) for row in ledger_rows]
    if (
        len(manifest_keys) != len(set(manifest_keys))
        or len(ledger_keys) != len(set(ledger_keys))
        or set(manifest_keys) != set(ledger_keys)
    ):
        raise ValueError(
            "Download ledger does not exactly match the bound filing manifest."
        )


def _validated_download_ledger_file(row: Mapping[str, Any]) -> Path:
    identity = f"{row.get('ticker')}/{row.get('form_type')}/{row.get('accession')}"
    raw_path = row.get("local_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Download ledger row {identity} has no local_path.")
    path = Path(raw_path)
    if not path.is_file():
        raise ValueError(f"Download ledger file does not exist for {identity}: {path}")
    expected_bytes = row.get("bytes")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
    ):
        raise ValueError(f"Download ledger row {identity} has no valid byte count.")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Download ledger byte count mismatch for {identity}: "
            f"expected {expected_bytes}, found {actual_bytes}."
        )
    expected_sha256 = row.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError(f"Download ledger row {identity} has no valid SHA-256.")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Download ledger SHA-256 mismatch for {identity}: "
            f"expected {expected_sha256}, found {actual_sha256}."
        )
    try:
        _validate_zip_archive(path)
    except DownloadIntegrityError as error:
        raise ValueError(
            f"Download ledger ZIP integrity check failed for {identity}."
        ) from error
    return path


def _is_retryable_error(error: requests.RequestException) -> bool:
    if isinstance(error, requests.Timeout):
        return True
    if isinstance(error, requests.HTTPError):
        status_code = error.response.status_code if error.response is not None else None
        return status_code in RETRYABLE_HTTP_STATUS_CODES
    return True


def download_filings(
    corpus_root: str, user_agent: Optional[str] = None, max_attempts: int = 3
) -> Dict[str, Any]:
    layout = ensure_corpus_layout(corpus_root)
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts <= 0
    ):
        raise ValueError("max_attempts must be a positive integer.")
    filing_manifest_path = layout.manifest / "filing_manifest.jsonl"
    _verify_source_artifact_binding(
        layout,
        artifact_name="filing_manifest",
        artifact_path=filing_manifest_path,
    )
    filing_manifest = read_jsonl(filing_manifest_path)
    for filing in filing_manifest:
        _validate_sec_filing_target(filing)
    manifest_keys = [_filing_row_key(row) for row in filing_manifest]
    if len(manifest_keys) != len(set(manifest_keys)):
        raise ValueError("Filing manifest contains duplicate filing identities.")

    ledger_path = layout.manifest / "download_ledger.jsonl"
    previous_rows = read_jsonl(ledger_path) if ledger_path.exists() else []
    previous_by_key: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    manifest_key_set = set(manifest_keys)
    for row in previous_rows:
        key = _filing_row_key(row)
        if key not in manifest_key_set:
            raise ValueError(
                "Existing download ledger belongs to a different filing manifest; "
                "refusing to mix corpus state."
            )
        if key in previous_by_key:
            raise ValueError(
                "Existing download ledger contains duplicate filing identities."
            )
        previous_by_key[key] = row

    session = _get_session(user_agent=user_agent)
    ledger_rows: List[Dict[str, Any]] = []

    for filing in filing_manifest:
        ledger_row = dict(filing)
        local_path = None
        attempts = 0
        if filing["status"] == "resolved" and filing["accession"]:
            local_path = (
                layout.raw_xbrl_zip
                / filing["ticker"]
                / f"{filing['ticker']}_{filing['form_type']}_{filing['accession']}.zip"
            )
            ledger_row["local_path"] = str(local_path)
            try:
                previous_row = previous_by_key.get(_filing_row_key(filing))
                if previous_row is not None and _ledger_file_matches(
                    previous_row, local_path
                ):
                    ledger_row.update(
                        {
                            "status": "existing_local",
                            "http_status": None,
                            "sha256": previous_row["sha256"],
                            "bytes": previous_row["bytes"],
                            "error": None,
                            "attempts": 0,
                        }
                    )
                else:
                    attempts = 0
                    while True:
                        attempts += 1
                        try:
                            http_status, byte_count = _download_with_resume(
                                session, filing["zip_url"], local_path
                            )
                            _validate_zip_archive(local_path)
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
                            if attempts >= max_attempts or not _is_retryable_error(
                                error
                            ):
                                raise
            except requests.HTTPError as error:
                http_status = (
                    error.response.status_code if error.response is not None else None
                )
                ledger_row.update(
                    {
                        "status": "download_error",
                        "http_status": http_status,
                        "sha256": None,
                        "bytes": local_path.stat().st_size
                        if local_path and local_path.exists()
                        else None,
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
                        "bytes": local_path.stat().st_size
                        if local_path and local_path.exists()
                        else None,
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
        _write_jsonl_atomic(ledger_path, ledger_rows)

    return {
        "download_ledger": str(ledger_path),
        "attempted": len(ledger_rows),
        "success_count": sum(
            1 for row in ledger_rows if row["status"] in SUCCESS_DOWNLOAD_STATUSES
        ),
        "error_count": sum(
            1 for row in ledger_rows if row["status"] == "download_error"
        ),
    }


def ingest_corpus(
    corpus_root: str,
    extract_narrative: bool = True,
    *,
    replace_existing: bool = False,
) -> Dict[str, Any]:
    layout = ensure_corpus_layout(corpus_root)
    filing_manifest_path = layout.manifest / "filing_manifest.jsonl"
    _verify_source_artifact_binding(
        layout,
        artifact_name="filing_manifest",
        artifact_path=filing_manifest_path,
    )
    ledger_rows = read_jsonl(layout.manifest / "download_ledger.jsonl")
    if (layout.manifest / "source_manifest.json").exists():
        filing_manifest = read_jsonl(filing_manifest_path)
        _verify_download_ledger_manifest_binding(layout, filing_manifest, ledger_rows)
    ingest_rows: List[Dict[str, Any]] = []

    if layout.db_path.exists() and not replace_existing:
        raise FileExistsError(
            "Corpus DB already exists; pass replace_existing=True to build and atomically replace it."
        )

    staging_db_path = layout.db_path.with_name(f"{layout.db_path.name}.building")
    if staging_db_path.exists():
        raise FileExistsError(
            f"Staged corpus DB already exists and was not removed: {staging_db_path}"
        )

    success_rows = [
        row
        for row in ledger_rows
        if row["status"] in SUCCESS_DOWNLOAD_STATUSES and row.get("local_path")
    ]
    if not success_rows:
        raise ValueError(
            "Download ledger contains no successfully downloaded filing ZIPs."
        )
    validated_paths = [_validated_download_ledger_file(row) for row in success_rows]
    for index, row in enumerate(success_rows):
        try:
            summary = process_zip(
                zip_path=str(validated_paths[index]),
                out_db=str(staging_db_path),
                ticker=row["ticker"],
                extract_narrative=extract_narrative,
                reset_db=(index == 0),
                metadata_overrides={
                    "form_type": row["form_type"],
                    "preferred_main_html": row.get("primary_doc"),
                    "cik": row.get("cik"),
                    "accession": row.get("accession"),
                    "source_sha256": row.get("sha256"),
                    "source_size_bytes": row.get("bytes"),
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
    _write_jsonl_atomic(ingest_ledger_path, ingest_rows)
    error_count = sum(1 for row in ingest_rows if row["status"] == "ingest_error")
    if error_count:
        if staging_db_path.exists():
            staging_db_path.unlink()
    else:
        os.replace(staging_db_path, layout.db_path)
    return {
        "db": str(layout.db_path),
        "ingest_ledger": str(ingest_ledger_path),
        "ingested_count": sum(1 for row in ingest_rows if row["status"] == "ingested"),
        "error_count": error_count,
        "published": error_count == 0,
    }


def repair_corpus_filing(
    corpus_root: str,
    ticker: str,
    accession: str,
    extract_narrative: bool = True,
) -> Dict[str, Any]:
    layout = ensure_corpus_layout(corpus_root)
    filing_manifest_path = layout.manifest / "filing_manifest.jsonl"
    _verify_source_artifact_binding(
        layout,
        artifact_name="filing_manifest",
        artifact_path=filing_manifest_path,
    )
    ledger_rows = read_jsonl(layout.manifest / "download_ledger.jsonl")
    if (layout.manifest / "source_manifest.json").exists():
        filing_manifest = read_jsonl(filing_manifest_path)
        _verify_download_ledger_manifest_binding(layout, filing_manifest, ledger_rows)
    download_row = next(
        (
            row
            for row in ledger_rows
            if row.get("ticker") == ticker and row.get("accession") == accession
        ),
        None,
    )
    if download_row is None:
        raise ValueError(
            f"No download ledger row found for ticker={ticker} accession={accession}."
        )
    if download_row["status"] not in SUCCESS_DOWNLOAD_STATUSES or not download_row.get(
        "local_path"
    ):
        raise ValueError(
            f"Download ledger row for ticker={ticker} accession={accession} is not locally available."
        )
    validated_zip_path = _validated_download_ledger_file(download_row)

    ingest_ledger_path = layout.logs_dir / "ingest_ledger.jsonl"
    existing_ingest_rows = (
        read_jsonl(ingest_ledger_path) if ingest_ledger_path.exists() else []
    )
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
            zip_path=str(validated_zip_path),
            out_db=str(layout.db_path),
            ticker=download_row["ticker"],
            extract_narrative=extract_narrative,
            reset_db=not layout.db_path.exists(),
            metadata_overrides={
                "form_type": download_row["form_type"],
                "preferred_main_html": download_row.get("primary_doc"),
                "cik": download_row.get("cik"),
                "accession": download_row.get("accession"),
                "source_sha256": download_row.get("sha256"),
                "source_size_bytes": download_row.get("bytes"),
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
        digest = int(
            _stable_digest("issuer-holdout", ISSUER_HOLDOUT_VERSION, issuer["ticker"])[
                :8
            ],
            16,
        )
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
    return {
        "counts": counts,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def generate_corpus_datasets(corpus_root: str) -> Dict[str, Any]:
    layout = ensure_corpus_layout(corpus_root)
    issuer_manifest_path = layout.manifest / "issuer_manifest.jsonl"
    filing_manifest_path = layout.manifest / "filing_manifest.jsonl"
    _verify_source_artifact_binding(
        layout,
        artifact_name="issuer_manifest",
        artifact_path=issuer_manifest_path,
    )
    _verify_source_artifact_binding(
        layout,
        artifact_name="filing_manifest",
        artifact_path=filing_manifest_path,
    )
    issuer_manifest = read_jsonl(issuer_manifest_path)
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
            for row in conn.execute(
                "SELECT filing_id FROM filings ORDER BY filing_id"
            ).fetchall()
        ]
        with (
            answer_path.open("w", encoding="utf-8") as answer_handle,
            task_specs_path.open("w", encoding="utf-8") as task_handle,
            dataset_paths["train_quant_qa"].open("w", encoding="utf-8") as qa_handle,
            dataset_paths["train_quant_code"].open(
                "w", encoding="utf-8"
            ) as code_handle,
            dataset_paths["train_refusal"].open(
                "w", encoding="utf-8"
            ) as refusal_handle,
            dataset_paths["hard_negatives"].open(
                "w", encoding="utf-8"
            ) as negatives_handle,
            dataset_paths["eval_holdout"].open("w", encoding="utf-8") as eval_handle,
        ):
            for filing_row in filing_rows:
                current_filing_id = filing_row["filing_id"]
                answers = generate_answer_objects(conn, filing_id=current_filing_id)
                for answer in answers:
                    _write_jsonl_line(answer_handle, answer)
                    answer_count += 1

                task_specs = build_task_specs_from_answers(
                    conn, answers, filing_id=current_filing_id
                )
                for task_spec in task_specs:
                    split = split_by_ticker.get(task_spec["ticker"], "train")
                    enriched_task_spec = dict(task_spec)
                    enriched_task_spec["split"] = split
                    enriched_task_spec["split_manifest_version"] = (
                        ISSUER_HOLDOUT_VERSION
                    )
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
                        "evidence_requirements": dict(
                            task_spec["evidence_requirements"]
                        ),
                        "negative_type": task_spec["negative_type"],
                    }
                    if split == "eval_holdout":
                        _write_jsonl_line(eval_handle, record)
                        counts["eval_holdout"] += 1
                    elif task_spec["target_status"] == "OK":
                        _write_jsonl_line(qa_handle, record)
                        counts["train_quant_qa"] += 1
                        _write_jsonl_line(
                            code_handle,
                            {**record, "code_target": _build_code_target(task_spec)},
                        )
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


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
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
    layout = ensure_corpus_layout(corpus_root)
    issuer_manifest_path = layout.manifest / "issuer_manifest.jsonl"
    filing_manifest_path = layout.manifest / "filing_manifest.jsonl"
    _verify_source_artifact_binding(
        layout,
        artifact_name="issuer_manifest",
        artifact_path=issuer_manifest_path,
    )
    _verify_source_artifact_binding(
        layout,
        artifact_name="filing_manifest",
        artifact_path=filing_manifest_path,
    )
    issuer_manifest = read_jsonl(issuer_manifest_path)
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
        validator_distribution = Counter(
            row["validator_code"] for row in fetch_validators(conn)
        )
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

                predicted = answer_quant(
                    conn,
                    row["question"],
                    row["filing_id"],
                    task_specs=filing_task_specs,
                )
                target = row["target_answer"]

                status_match = predicted["status"] == target["status"]
                period_match = predicted["period_key"] == target["period_key"]
                unit_match = predicted["unit"] == target["unit"]
                value_match = _decimal_equal(predicted["value"], target["value"])
                evidence_match = sorted(predicted["evidence_ids"]) == sorted(
                    target["evidence_ids"]
                )
                refusal_match = predicted["refusal_code"] == target["refusal_code"]

                runtime_counters["total"] += 1
                runtime_counters["status_match"] += int(status_match)
                runtime_counters["period_match"] += int(period_match)
                runtime_counters["unit_match"] += int(unit_match)
                runtime_counters["evidence_match"] += int(evidence_match)

                if target["status"] == "OK":
                    runtime_counters["ok_total"] += 1
                    runtime_counters["numeric_match"] += int(
                        status_match and value_match
                    )
                    runtime_counters["context_match"] += int(evidence_match)
                else:
                    runtime_counters["refusal_total"] += 1
                    runtime_counters["refusal_match"] += int(
                        status_match and refusal_match
                    )

                unsupported = predicted["status"] == "OK" and (
                    target["status"] != "OK" or not predicted["evidence_ids"]
                )
                runtime_counters["unsupported_claims"] += int(unsupported)

                fully_correct = (
                    status_match
                    and period_match
                    and unit_match
                    and evidence_match
                    and (value_match if target["status"] == "OK" else refusal_match)
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
            "resolved_issuer_count": sum(
                1 for row in issuer_manifest if row["status"] == "resolved"
            ),
            "ingest_enabled_issuer_count": sum(
                1 for row in issuer_manifest if row.get("ingest_enabled")
            ),
            "filing_target_count": len(filing_manifest),
            "resolved_filing_count": sum(
                1 for row in filing_manifest if row["status"] == "resolved"
            ),
        },
        "sec_resolution_rate": (
            sum(1 for row in issuer_manifest if row.get("cik") is not None)
            / len(issuer_manifest)
            if issuer_manifest
            else 0.0
        ),
        "download_success_rate": (
            sum(
                1
                for row in download_ledger
                if row["status"] in SUCCESS_DOWNLOAD_STATUSES
            )
            / max(1, sum(1 for row in filing_manifest if row["status"] == "resolved"))
        ),
        "ingest_success_rate": (
            sum(1 for row in ingest_ledger if row["status"] == "ingested")
            / max(
                1,
                sum(
                    1
                    for row in download_ledger
                    if row["status"] in SUCCESS_DOWNLOAD_STATUSES
                ),
            )
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
        "numeric_accuracy": runtime_counters["numeric_match"]
        / max(1, runtime_counters["ok_total"]),
        "refusal_correctness": runtime_counters["refusal_match"]
        / max(1, runtime_counters["refusal_total"]),
        "unit_accuracy": runtime_counters["unit_match"]
        / max(1, runtime_counters["total"]),
        "period_accuracy": runtime_counters["period_match"]
        / max(1, runtime_counters["total"]),
        "context_accuracy": runtime_counters["context_match"]
        / max(1, runtime_counters["ok_total"]),
        "unsupported_claim_rate": runtime_counters["unsupported_claims"]
        / max(1, runtime_counters["total"]),
        "evidence_id_exactness": runtime_counters["evidence_match"]
        / max(1, runtime_counters["total"]),
        "status_accuracy": runtime_counters["status_match"]
        / max(1, runtime_counters["total"]),
        "failure_count": failure_count,
        "failure_buckets": dict(sorted(failure_bucket_counter.items())),
    }

    data_quality_path.write_text(_json_dumps(data_quality_summary), encoding="utf-8")
    runtime_eval_path.write_text(_json_dumps(runtime_eval_summary), encoding="utf-8")
    _write_csv(
        metric_csv_path, coverage_by_metric, ["metric_spec_id", "split", "task_count"]
    )
    _write_csv(issuer_csv_path, coverage_by_issuer, ["ticker", "split", "task_count"])
    _write_csv(
        validator_csv_path,
        [
            {"validator_code": code, "count": count}
            for code, count in sorted(validator_distribution.items())
        ],
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
