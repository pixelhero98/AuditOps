from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

from .corpus import resolve_sec_user_agent

S_AND_P_500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
S_AND_P_500_WIKI_TITLE = "List of S&P 500 companies"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_OLDID_URL = "https://en.wikipedia.org/w/index.php"
FTSE_100_WIKI_URL = "https://en.wikipedia.org/wiki/FTSE_100_Index"
DEFAULT_BOOTSTRAP_USER_AGENT = "AuditOps/0.1 (public constituent bootstrap)"


def _validate_snapshot_date(snapshot_date: Optional[str]) -> Optional[str]:
    if snapshot_date is None:
        return None
    datetime.strptime(snapshot_date, "%Y-%m-%d")
    return snapshot_date


def _normalize_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split()).strip()


def _fetch_html(
    url: str,
    *,
    user_agent: Optional[str] = None,
    params: Optional[Dict[str, str]] = None,
) -> str:
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": user_agent or DEFAULT_BOOTSTRAP_USER_AGENT},
        params=params,
    )
    response.raise_for_status()
    return response.text


def _resolve_wikipedia_revision(
    *,
    title: str,
    snapshot_date: str,
    user_agent: str,
) -> Dict[str, object]:
    """Resolve the last published revision at or before the UTC snapshot boundary."""

    as_of = f"{snapshot_date}T23:59:59Z"
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "revisions",
        "titles": title,
        "rvdir": "older",
        "rvlimit": "1",
        "rvprop": "ids|timestamp",
        "rvstart": as_of,
    }
    response = requests.get(
        WIKIPEDIA_API_URL,
        timeout=30,
        headers={"User-Agent": user_agent},
        params=params,
    )
    response.raise_for_status()
    payload = response.json()
    pages = payload.get("query", {}).get("pages", [])
    if len(pages) != 1 or pages[0].get("missing"):
        raise ValueError(f"Wikipedia page could not be resolved: {title}")
    revisions = pages[0].get("revisions", [])
    if len(revisions) != 1:
        raise ValueError(
            f"No Wikipedia revision exists at or before snapshot boundary {as_of}"
        )
    revision = revisions[0]
    revision_id = revision.get("revid")
    revision_timestamp = revision.get("timestamp")
    if not isinstance(revision_id, int) or not isinstance(revision_timestamp, str):
        raise ValueError(
            "Wikipedia revision response omitted a revision id or timestamp"
        )
    revision_instant = datetime.fromisoformat(revision_timestamp.replace("Z", "+00:00"))
    snapshot_boundary = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    if revision_instant > snapshot_boundary:
        raise ValueError(
            "Resolved Wikipedia revision is newer than the requested snapshot"
        )

    oldid_params = {"title": title, "oldid": str(revision_id)}
    return {
        "as_of": as_of,
        "revision_id": revision_id,
        "revision_timestamp": revision_timestamp,
        "revision_api_url": f"{WIKIPEDIA_API_URL}?{urlencode(params)}",
        "source_url": f"{WIKIPEDIA_OLDID_URL}?{urlencode(oldid_params)}",
        "source_params": oldid_params,
    }


def _find_table_rows(
    html: str, *, required_headers: Sequence[str]
) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    expected = [_normalize_text(header).lower() for header in required_headers]

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        header_cells = rows[0].find_all("th")
        headers = [
            _normalize_text(cell.get_text(" ", strip=True)) for cell in header_cells
        ]
        lowered = [header.lower() for header in headers]
        if not all(header in lowered for header in expected):
            continue

        records: List[Dict[str, str]] = []
        for row in rows[1:]:
            cells = row.find_all(["th", "td"])
            if not cells:
                continue
            values = [_normalize_text(cell.get_text(" ", strip=True)) for cell in cells]
            if len(values) < len(headers):
                values.extend([""] * (len(headers) - len(values)))
            record = {headers[index]: values[index] for index in range(len(headers))}
            records.append(record)
        return records

    raise ValueError(
        f"Could not find a table with headers: {', '.join(required_headers)}"
    )


def _truncate_rows(
    rows: Iterable[Dict[str, str]], *, limit: Optional[int]
) -> List[Dict[str, str]]:
    if limit is None:
        return list(rows)
    if limit <= 0:
        raise ValueError("--limit must be a positive integer.")
    output: List[Dict[str, str]] = []
    for row in rows:
        output.append(row)
        if len(output) >= limit:
            break
    return output


def _write_csv(
    path: str | Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def fetch_us_constituents(
    output_path: str | Path,
    *,
    snapshot_date: Optional[str] = None,
    limit: Optional[int] = None,
    user_agent: Optional[str] = None,
) -> Dict[str, object]:
    normalized_snapshot_date = _validate_snapshot_date(snapshot_date)
    resolved_user_agent = resolve_sec_user_agent(user_agent)
    output = Path(output_path)
    source_manifest_path = output.with_name(f"{output.name}.source.json")
    raw_source_path = output.with_name(f"{output.name}.source.html")
    if output.exists() or source_manifest_path.exists() or raw_source_path.exists():
        raise FileExistsError(
            "US constituent snapshot and source capture are immutable; choose a new output path"
        )
    retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    revision: Optional[Dict[str, object]] = None
    source_url = S_AND_P_500_WIKI_URL
    source_params: Optional[Dict[str, str]] = None
    if normalized_snapshot_date is not None:
        revision = _resolve_wikipedia_revision(
            title=S_AND_P_500_WIKI_TITLE,
            snapshot_date=normalized_snapshot_date,
            user_agent=resolved_user_agent,
        )
        source_url = str(revision["source_url"])
        source_params = dict(revision["source_params"])
    html = _fetch_html(
        WIKIPEDIA_OLDID_URL if source_params is not None else source_url,
        user_agent=resolved_user_agent,
        params=source_params,
    )
    raw_rows = _find_table_rows(html, required_headers=("Symbol", "Security"))
    rows = _truncate_rows(
        (
            {
                "ticker": row["Symbol"].upper(),
                "company_name": row["Security"],
            }
            for row in raw_rows
            if row.get("Symbol") and row.get("Security")
        ),
        limit=limit,
    )
    if not rows:
        raise ValueError("S&P 500 constituent source produced no usable rows")
    raw_source_path.parent.mkdir(parents=True, exist_ok=True)
    raw_source_path.write_bytes(html.encode("utf-8"))
    _write_csv(output, ("ticker", "company_name"), rows)
    source_manifest = {
        "source_manifest_version": "auditops-us-constituents-source.v2",
        "source_url": source_url,
        "retrieved_at": retrieved_at,
        "snapshot_date": normalized_snapshot_date,
        "source_sha256": _sha256_bytes(html.encode("utf-8")),
        "raw_source_path": raw_source_path.name,
        "raw_source_sha256": _sha256_file(raw_source_path),
        "output_sha256": _sha256_file(output),
        "row_count": len(rows),
        "capture_scope": "full" if limit is None else "limited",
        "row_limit": limit,
        "user_agent_sha256": _sha256_bytes(resolved_user_agent.encode("utf-8")),
    }
    if revision is not None:
        source_manifest.update(
            {
                "requested_as_of": revision["as_of"],
                "revision_id": revision["revision_id"],
                "revision_timestamp": revision["revision_timestamp"],
                "revision_api_url": revision["revision_api_url"],
            }
        )
    source_manifest_path.write_text(
        json.dumps(source_manifest, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(output),
        "row_count": len(rows),
        "source_url": source_url,
        "snapshot_date": normalized_snapshot_date,
        "limit": limit,
        "retrieved_at": retrieved_at,
        "source_sha256": source_manifest["source_sha256"],
        "raw_source": str(raw_source_path),
        "raw_source_sha256": source_manifest["raw_source_sha256"],
        "output_sha256": source_manifest["output_sha256"],
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256_file(source_manifest_path),
    }


def fetch_uk_constituents(
    output_path: str | Path,
    *,
    snapshot_date: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    normalized_snapshot_date = _validate_snapshot_date(snapshot_date)
    html = _fetch_html(FTSE_100_WIKI_URL)
    raw_rows = _find_table_rows(html, required_headers=("Company", "Ticker"))
    rows = _truncate_rows(
        (
            {
                "ticker": row["Ticker"].upper(),
                "company_name": row["Company"],
                "lei": "",
                "nsm_keyword": row["Company"],
                "issuer_country": "UK",
            }
            for row in raw_rows
            if row.get("Ticker") and row.get("Company")
        ),
        limit=limit,
    )
    _write_csv(
        output_path,
        ("ticker", "company_name", "lei", "nsm_keyword", "issuer_country"),
        rows,
    )
    return {
        "output": str(output_path),
        "row_count": len(rows),
        "source_url": FTSE_100_WIKI_URL,
        "snapshot_date": normalized_snapshot_date,
        "limit": limit,
    }
