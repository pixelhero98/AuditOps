from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import requests
from bs4 import BeautifulSoup


S_AND_P_500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FTSE_100_WIKI_URL = "https://en.wikipedia.org/wiki/FTSE_100_Index"
DEFAULT_BOOTSTRAP_USER_AGENT = "AuditOps/0.1 (public constituent bootstrap)"


def _validate_snapshot_date(snapshot_date: Optional[str]) -> Optional[str]:
    if snapshot_date is None:
        return None
    datetime.strptime(snapshot_date, "%Y-%m-%d")
    return snapshot_date


def _normalize_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split()).strip()


def _fetch_html(url: str, *, user_agent: Optional[str] = None) -> str:
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": user_agent or DEFAULT_BOOTSTRAP_USER_AGENT},
    )
    response.raise_for_status()
    return response.text


def _find_table_rows(html: str, *, required_headers: Sequence[str]) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    expected = [_normalize_text(header).lower() for header in required_headers]

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        header_cells = rows[0].find_all("th")
        headers = [_normalize_text(cell.get_text(" ", strip=True)) for cell in header_cells]
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

    raise ValueError(f"Could not find a table with headers: {', '.join(required_headers)}")


def _truncate_rows(rows: Iterable[Dict[str, str]], *, limit: Optional[int]) -> List[Dict[str, str]]:
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


def _write_csv(path: str | Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def fetch_us_constituents(output_path: str | Path, *, snapshot_date: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, object]:
    """Fetch and write a normalized US constituent snapshot CSV.
    
    Parameters
    ----------
    output_path : str | Path
        Destination path for generated output artifacts. e.g., 'auditops-output.jsonl'
    snapshot_date : Optional[str], optional
        Snapshot date in ISO format YYYY-MM-DD.
    limit : Optional[int], optional
        Maximum number of records to process.
    
    Returns
    -------
    Dict[str, object]
        Dictionary with output fields for this operation.
    """
    normalized_snapshot_date = _validate_snapshot_date(snapshot_date)
    html = _fetch_html(S_AND_P_500_WIKI_URL)
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
    _write_csv(output_path, ("ticker", "company_name"), rows)
    return {
        "output": str(output_path),
        "row_count": len(rows),
        "source_url": S_AND_P_500_WIKI_URL,
        "snapshot_date": normalized_snapshot_date,
        "limit": limit,
    }


def fetch_uk_constituents(output_path: str | Path, *, snapshot_date: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, object]:
    """Fetch and write a normalized UK constituent snapshot CSV.
    
    Parameters
    ----------
    output_path : str | Path
        Destination path for generated output artifacts. e.g., auditops-output.jsonl
    snapshot_date : Optional[str], optional
        Snapshot date in ISO format YYYY-MM-DD.
    limit : Optional[int], optional
        Maximum number of records to process.
    
    Returns
    -------
    Dict[str, object]
        Dictionary with output fields for this operation.
    """
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
    _write_csv(output_path, ("ticker", "company_name", "lei", "nsm_keyword", "issuer_country"), rows)
    return {
        "output": str(output_path),
        "row_count": len(rows),
        "source_url": FTSE_100_WIKI_URL,
        "snapshot_date": normalized_snapshot_date,
        "limit": limit,
    }
