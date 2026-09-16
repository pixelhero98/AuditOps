from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import quote, urljoin

import requests

from .corpus import (
    RETRYABLE_HTTP_STATUS_CODES,
    _copy_snapshot_file,
    _download_with_resume,
    _get_session,
    _json_dumps,
    _session_get,
    ensure_corpus_layout,
)
from .tasks import read_jsonl, write_jsonl

FCA_NSM_SEARCH_API = "https://api.data.fca.org.uk/search"
FCA_NSM_DETAILS_API = "https://api.data.fca.org.uk/details"
FCA_NSM_INDEX_NAME = "fca-nsm-searchdata"
FCA_NSM_ASSET_BASE_URL = "https://data.fca.org.uk/"

UK_REPORT_DEFS: Sequence[Dict[str, str]] = (
    {
        "report_type": "annual_financial_report",
        "form_type": "UK_AFR",
        "nsm_type": "Annual Financial Report",
    },
    {
        "report_type": "half_yearly_financial_report",
        "form_type": "UK_HYR",
        "nsm_type": "Half-year Financial Report",
    },
)

UK_RAW_DOWNLOAD_SUCCESS_STATUSES = {"downloaded", "existing_local"}
FCA_RETRYABLE_HTTP_STATUS_CODES = {403, 408, 425, 429, 500, 502, 503, 504}
COMPANY_SUFFIX_TOKENS = {
    "PLC",
    "P",
    "L",
    "C",
    "LTD",
    "LIMITED",
    "PUBLIC",
    "COMPANY",
    "CO",
    "GROUP",
    "HOLDINGS",
    "HOLDING",
    "SA",
    "N",
    "V",
}


def _uk_snapshot_id(snapshot_date: str) -> str:
    return f"uk_ftse100_nsm_latest_{snapshot_date}"


def _read_uk_constituents_snapshot(path: str | Path) -> List[Dict[str, Optional[str]]]:
    snapshot_path = Path(path)
    with snapshot_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("UK constituents CSV is missing a header row.")

        fields = {field.lower(): field for field in reader.fieldnames}
        ticker_field = next(
            (fields[name] for name in ("ticker", "symbol", "epic") if name in fields),
            None,
        )
        name_field = next(
            (
                fields[name]
                for name in ("company_name", "name", "security", "issuer")
                if name in fields
            ),
            None,
        )
        lei_field = next(
            (
                fields[name]
                for name in ("lei", "legal_entity_identifier")
                if name in fields
            ),
            None,
        )
        keyword_field = next(
            (
                fields[name]
                for name in ("nsm_keyword", "search_keyword", "keyword")
                if name in fields
            ),
            None,
        )
        country_field = next(
            (fields[name] for name in ("issuer_country", "country") if name in fields),
            None,
        )
        if ticker_field is None or name_field is None:
            raise ValueError(
                "UK constituents CSV must include ticker/symbol and company_name/name columns."
            )

        rows: List[Dict[str, Optional[str]]] = []
        for raw_row in reader:
            ticker = (raw_row.get(ticker_field) or "").strip().upper()
            company_name = (raw_row.get(name_field) or "").strip()
            if not ticker or not company_name:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "company_name": company_name,
                    "lei": (raw_row.get(lei_field) or "").strip().upper() or None
                    if lei_field
                    else None,
                    "nsm_keyword": (raw_row.get(keyword_field) or "").strip() or None
                    if keyword_field
                    else None,
                    "issuer_country": (raw_row.get(country_field) or "").strip() or "UK"
                    if country_field
                    else "UK",
                }
            )
        return rows


def _normalize_company_name(value: Optional[str]) -> str:
    if not value:
        return ""
    text = value.upper().replace("&", " AND ")
    text = text.replace("P.L.C.", " PLC ")
    text = text.replace("P.L.C", " PLC ")
    text = text.replace("PLC.", " PLC ")
    text = re.sub(r"\bPUBLIC LIMITED COMPANY\b", " PLC ", text)
    text = re.sub(r"\bLIMITED\b", " LTD ", text)
    text = re.sub(r"\bINCORPORATED\b", " INC ", text)
    text = re.sub(r"\bCOMPANY\b", " CO ", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    tokens = [token for token in text.split() if token not in {"THE"}]
    return " ".join(tokens)


def _split_company_field(company_value: Optional[str]) -> List[str]:
    if not company_value:
        return []
    parts = re.split(r"[;|]", company_value)
    normalized = []
    for part in parts:
        key = _normalize_company_name(part)
        if key and key not in normalized:
            normalized.append(key)
    return normalized


def _company_core_key(value: Optional[str]) -> str:
    tokens = _normalize_company_name(value).split()
    while tokens and tokens[-1] in COMPANY_SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens)


def _company_search_variants(company_name: str) -> List[str]:
    variants: List[str] = []
    candidates = [
        company_name,
        _normalize_company_name(company_name).replace(" ", " "),
    ]
    normalized = _normalize_company_name(company_name)
    if normalized and "PLC" not in normalized.split():
        candidates.append(f"{company_name} plc")
    core = _company_core_key(company_name)
    if core and core not in {company_name, normalized}:
        candidates.append(core)
        candidates.append(f"{core} plc")
        candidates.append(f"{core} group plc")

    for candidate in candidates:
        cleaned = " ".join(str(candidate).split()).strip()
        if cleaned and cleaned not in variants:
            variants.append(cleaned)
    return variants


def _apply_fca_headers(session: Any) -> Any:
    headers = getattr(session, "headers", None)
    if headers is not None:
        headers.update(
            {"Origin": "https://data.fca.org.uk", "Referer": "https://data.fca.org.uk/"}
        )
    return session


def _parse_iso_sort_key(value: Optional[str]) -> str:
    return value or ""


def _nsm_search(
    session: requests.Session,
    keyword: str,
    *,
    size: int = 200,
    sort: str = "publication_date",
    sortorder: str = "desc",
) -> List[Dict[str, Any]]:
    payload = {
        "from": 0,
        "size": size,
        "sort": sort,
        "sortorder": sortorder,
        "keyword": keyword,
        "criteriaObj": None,
    }
    min_interval = float(os.environ.get("AUDITOPS_FCA_MIN_INTERVAL_SECONDS", "0.5"))
    max_attempts = int(os.environ.get("AUDITOPS_FCA_MAX_ATTEMPTS", "4"))
    for attempt in range(1, max_attempts + 1):
        last_request_at = getattr(session, "_auditops_last_request_at", None)
        if last_request_at is not None and min_interval > 0:
            elapsed = time.monotonic() - last_request_at
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        response = session.post(
            FCA_NSM_SEARCH_API,
            params={"index": FCA_NSM_INDEX_NAME},
            json=payload,
            timeout=60,
        )
        session._auditops_last_request_at = time.monotonic()
        try:
            response.raise_for_status()
        except requests.HTTPError:
            if (
                response.status_code not in FCA_RETRYABLE_HTTP_STATUS_CODES
                or attempt >= max_attempts
            ):
                raise
            time.sleep(min(8.0, attempt * 2.0))
            continue
        data = response.json()
        return data.get("hits", {}).get("hits", [])
    return []


def _fetch_nsm_details(session: requests.Session, disclosure_id: str) -> Dict[str, Any]:
    response = _session_get(
        session,
        f"{FCA_NSM_DETAILS_API}/{quote(disclosure_id)}",
        params={"index": FCA_NSM_INDEX_NAME},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _nsm_asset_url(asset_path: str) -> str:
    return urljoin(FCA_NSM_ASSET_BASE_URL, asset_path.lstrip("/"))


def _download_nsm_asset(
    session: requests.Session, asset_path: str, output_path: Path
) -> tuple[int, Optional[int]]:
    return _download_with_resume(session, _nsm_asset_url(asset_path), output_path)


def _is_retryable_error(error: requests.RequestException) -> bool:
    if isinstance(error, requests.Timeout):
        return True
    if isinstance(error, requests.HTTPError):
        status_code = error.response.status_code if error.response is not None else None
        return status_code in RETRYABLE_HTTP_STATUS_CODES
    return True


def _candidate_matches_issuer(
    source: Mapping[str, Any], issuer: Mapping[str, Any]
) -> Optional[str]:
    issuer_lei = (issuer.get("lei") or "").strip().upper()
    if issuer_lei and (source.get("lei") or "").strip().upper() == issuer_lei:
        return "lei_exact"

    issuer_name_key = _normalize_company_name(str(issuer["company_name"]))
    source_name_keys = _split_company_field(source.get("company"))
    if issuer_name_key and issuer_name_key in source_name_keys:
        return "company_exact"
    issuer_core_key = _company_core_key(str(issuer["company_name"]))
    if issuer_core_key:
        source_core_keys = [_company_core_key(name) for name in source_name_keys]
        if issuer_core_key in source_core_keys:
            return "company_core"
    return None


def _candidate_sort_key(
    source: Mapping[str, Any], report_type: str
) -> tuple[int, int, str, str]:
    tag_esef = (source.get("tag_esef") or "").strip()
    tagged_score = 2 if tag_esef == "Tagged" else 1 if tag_esef == "Untagged" else 0
    direct_upload_score = 1 if source.get("source") == "Direct Upload" else 0
    publication_key = _parse_iso_sort_key(source.get("publication_date"))
    document_key = _parse_iso_sort_key(source.get("document_date"))
    if report_type == "annual_financial_report":
        return tagged_score, direct_upload_score, publication_key, document_key
    return direct_upload_score, tagged_score, publication_key, document_key


def _resolve_nsm_report(
    session: requests.Session,
    issuer: Mapping[str, Any],
    report_def: Mapping[str, str],
    *,
    search_size: int = 200,
) -> Dict[str, Any]:
    search_terms: List[str] = []
    if issuer.get("lei"):
        search_terms.append(str(issuer["lei"]))
    base_keyword = str(issuer.get("nsm_keyword") or issuer["company_name"]).strip()
    company_variants = _company_search_variants(base_keyword)
    if report_def["report_type"] == "annual_financial_report":
        report_queries = ["Annual Financial Report", "annual report"]
    else:
        report_queries = [
            "Half-year Financial Report",
            "half year report",
            "interim report",
        ]
    for variant in company_variants:
        for query in report_queries:
            search_terms.append(f"{variant} {query}")
        search_terms.append(variant)

    seen_terms = set()
    candidates: List[Dict[str, Any]] = []
    chosen_term = None
    for term in search_terms:
        if not term or term in seen_terms:
            continue
        seen_terms.add(term)
        hits = _nsm_search(session, term, size=search_size)
        current = []
        for hit in hits:
            source = dict(hit.get("_source", hit))
            if source.get("type") != report_def["nsm_type"]:
                continue
            resolution_kind = _candidate_matches_issuer(source, issuer)
            if not resolution_kind:
                continue
            current.append(
                {
                    "resolution_kind": resolution_kind,
                    "search_keyword_used": term,
                    **source,
                }
            )
        if current:
            candidates = current
            chosen_term = term
            break

    if not candidates:
        return {
            "status": "missing_latest_report",
            "error": f"no_matching_{report_def['report_type']}",
            "search_keyword_used": chosen_term,
        }

    candidates.sort(
        key=lambda row: _candidate_sort_key(row, report_def["report_type"]),
        reverse=True,
    )
    chosen = dict(candidates[0])
    chosen["status"] = "resolved"
    chosen["search_keyword_used"] = chosen_term
    return chosen


def build_uk_manifest(
    constituents_path: str,
    corpus_root: str,
    snapshot_date: str,
    constituent_source: Optional[str] = None,
    user_agent: Optional[str] = None,
    search_size: int = 200,
) -> Dict[str, Any]:
    layout = ensure_corpus_layout(corpus_root)
    snapshot_id = _uk_snapshot_id(snapshot_date)
    session = _apply_fca_headers(_get_session(user_agent=user_agent))

    constituents = _read_uk_constituents_snapshot(constituents_path)
    _copy_snapshot_file(
        constituents_path, layout.manifest / "constituents_snapshot.csv"
    )

    issuer_manifest: List[Dict[str, Any]] = []
    filing_manifest: List[Dict[str, Any]] = []
    source_label = constituent_source or Path(constituents_path).name

    for issuer in constituents:
        issuer_row = {
            "snapshot_id": snapshot_id,
            "snapshot_date": snapshot_date,
            "ticker": issuer["ticker"],
            "company_name": issuer["company_name"],
            "lei": issuer.get("lei"),
            "nsm_keyword": issuer.get("nsm_keyword"),
            "issuer_country": issuer.get("issuer_country") or "UK",
            "source_system": "FCA_NSM",
            "forms_expected": [
                report_def["report_type"] for report_def in UK_REPORT_DEFS
            ],
            "constituent_source": source_label,
            "status": "resolved",
            "error": None,
        }
        issuer_manifest.append(issuer_row)

        for report_def in UK_REPORT_DEFS:
            try:
                resolved = _resolve_nsm_report(
                    session, issuer_row, report_def, search_size=search_size
                )
                filing_manifest.append(
                    {
                        "snapshot_id": snapshot_id,
                        "snapshot_date": snapshot_date,
                        "ticker": issuer["ticker"],
                        "company_name": issuer["company_name"],
                        "lei": issuer.get("lei"),
                        "issuer_country": issuer.get("issuer_country") or "UK",
                        "source_system": "FCA_NSM",
                        "report_type": report_def["report_type"],
                        "form_type": report_def["form_type"],
                        "nsm_type": report_def["nsm_type"],
                        "disclosure_id": resolved.get("disclosure_id"),
                        "seq_id": resolved.get("seq_id"),
                        "filed_at": resolved.get("publication_date"),
                        "document_date": resolved.get("document_date"),
                        "publication_date": resolved.get("publication_date"),
                        "headline": resolved.get("headline"),
                        "download_link": resolved.get("download_link"),
                        "html_link": resolved.get("html_link"),
                        "tag_esef": resolved.get("tag_esef"),
                        "process_type": resolved.get("ProcessType"),
                        "source": resolved.get("source"),
                        "resolution_kind": resolved.get("resolution_kind"),
                        "search_keyword_used": resolved.get("search_keyword_used"),
                        "status": resolved["status"],
                        "error": resolved.get("error"),
                    }
                )
            except requests.RequestException as error:
                issuer_row["status"] = "search_error"
                issuer_row["error"] = str(error)
                filing_manifest.append(
                    {
                        "snapshot_id": snapshot_id,
                        "snapshot_date": snapshot_date,
                        "ticker": issuer["ticker"],
                        "company_name": issuer["company_name"],
                        "lei": issuer.get("lei"),
                        "issuer_country": issuer.get("issuer_country") or "UK",
                        "source_system": "FCA_NSM",
                        "report_type": report_def["report_type"],
                        "form_type": report_def["form_type"],
                        "nsm_type": report_def["nsm_type"],
                        "disclosure_id": None,
                        "seq_id": None,
                        "filed_at": None,
                        "document_date": None,
                        "publication_date": None,
                        "headline": None,
                        "download_link": None,
                        "html_link": None,
                        "tag_esef": None,
                        "process_type": None,
                        "source": None,
                        "resolution_kind": None,
                        "search_keyword_used": None,
                        "status": "search_error",
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
        "issuer_manifest": str(issuer_manifest_path),
        "filing_manifest": str(filing_manifest_path),
        "issuer_count": len(issuer_manifest),
        "resolved_issuer_count": sum(
            1 for row in issuer_manifest if row["status"] == "resolved"
        ),
        "resolved_filing_count": sum(
            1 for row in filing_manifest if row["status"] == "resolved"
        ),
        "annual_resolved_count": sum(
            1
            for row in filing_manifest
            if row["report_type"] == "annual_financial_report"
            and row["status"] == "resolved"
        ),
        "half_yearly_resolved_count": sum(
            1
            for row in filing_manifest
            if row["report_type"] == "half_yearly_financial_report"
            and row["status"] == "resolved"
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_uk_filings(
    corpus_root: str, user_agent: Optional[str] = None, max_attempts: int = 3
) -> Dict[str, Any]:
    layout = ensure_corpus_layout(corpus_root)
    session = _apply_fca_headers(_get_session(user_agent=user_agent))
    filing_manifest = read_jsonl(layout.manifest / "filing_manifest.jsonl")
    details_root = layout.root / "raw" / "details"
    documents_root = layout.root / "raw" / "documents"
    details_root.mkdir(parents=True, exist_ok=True)
    documents_root.mkdir(parents=True, exist_ok=True)

    ledger_rows: List[Dict[str, Any]] = []

    for filing in filing_manifest:
        ledger_row = dict(filing)
        ledger_row.update(
            {
                "details_path": None,
                "details_status": None,
                "details_http_status": None,
                "raw_local_path": None,
                "raw_status": None,
                "raw_http_status": None,
                "local_path": None,
                "sha256": None,
                "bytes": None,
                "attempts": 0,
            }
        )
        if filing["status"] != "resolved" or not filing.get("disclosure_id"):
            ledger_row["status"] = filing["status"]
            ledger_rows.append(ledger_row)
            continue

        details_path = (
            details_root / filing["ticker"] / f"{filing['disclosure_id']}.json"
        )
        details_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if details_path.exists() and details_path.stat().st_size > 0:
                details_payload = json.loads(details_path.read_text(encoding="utf-8"))
                ledger_row["details_status"] = "existing_local"
                ledger_row["details_http_status"] = None
            else:
                details_payload = _fetch_nsm_details(session, filing["disclosure_id"])
                details_path.write_text(_json_dumps(details_payload), encoding="utf-8")
                ledger_row["details_status"] = "downloaded"
                ledger_row["details_http_status"] = 200
            ledger_row["details_path"] = str(details_path)
        except requests.RequestException as error:
            ledger_row.update(
                {
                    "status": "download_error",
                    "details_status": "download_error",
                    "details_http_status": error.response.status_code
                    if isinstance(error, requests.HTTPError) and error.response
                    else None,
                    "error": str(error),
                }
            )
            ledger_rows.append(ledger_row)
            continue

        asset_candidates = [
            path
            for path in [filing.get("download_link"), filing.get("html_link")]
            if path
        ]
        raw_download_error = None
        raw_attempts = 0
        for asset_path in asset_candidates:
            raw_attempts += 1
            file_name = os.path.basename(asset_path)
            output_path = documents_root / filing["ticker"] / file_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if output_path.exists() and output_path.stat().st_size > 0:
                    http_status = None
                    byte_count = output_path.stat().st_size
                    raw_status = "existing_local"
                else:
                    http_status, byte_count = _download_nsm_asset(
                        session, asset_path, output_path
                    )
                    raw_status = (
                        "existing_local" if http_status == 416 else "downloaded"
                    )
                ledger_row.update(
                    {
                        "status": raw_status,
                        "raw_status": raw_status,
                        "raw_http_status": http_status,
                        "raw_local_path": str(output_path),
                        "sha256": _sha256_file(output_path),
                        "bytes": byte_count,
                        "attempts": raw_attempts,
                    }
                )
                if output_path.suffix.lower() == ".zip":
                    ledger_row["local_path"] = str(output_path)
                raw_download_error = None
                break
            except requests.HTTPError as error:
                raw_download_error = error
                if raw_attempts >= max_attempts or not _is_retryable_error(error):
                    break
            except requests.RequestException as error:
                raw_download_error = error
                if raw_attempts >= max_attempts or not _is_retryable_error(error):
                    break

        if raw_download_error is not None or not asset_candidates:
            ledger_row.update(
                {
                    "status": "details_only"
                    if ledger_row["details_status"] in {"downloaded", "existing_local"}
                    else "download_error",
                    "raw_status": "download_error"
                    if asset_candidates
                    else "not_available",
                    "raw_http_status": raw_download_error.response.status_code
                    if isinstance(raw_download_error, requests.HTTPError)
                    and raw_download_error.response
                    else None,
                    "error": str(raw_download_error) if raw_download_error else None,
                    "attempts": raw_attempts,
                }
            )

        ledger_rows.append(ledger_row)

    ledger_path = layout.manifest / "download_ledger.jsonl"
    write_jsonl(ledger_path, ledger_rows)
    return {
        "download_ledger": str(ledger_path),
        "attempted": len(ledger_rows),
        "details_success_count": sum(
            1
            for row in ledger_rows
            if row.get("details_status") in {"downloaded", "existing_local"}
        ),
        "raw_success_count": sum(
            1
            for row in ledger_rows
            if row.get("raw_status") in UK_RAW_DOWNLOAD_SUCCESS_STATUSES
        ),
        "details_only_count": sum(
            1 for row in ledger_rows if row.get("status") == "details_only"
        ),
        "error_count": sum(
            1 for row in ledger_rows if row.get("status") == "download_error"
        ),
    }
