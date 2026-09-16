from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from lxml import etree

from .canonical_json import canonical_json_bytes
from .corpus import ensure_corpus_layout
from .tasks import write_jsonl

OFFLINE_US_IMPORT_VERSION = "auditops-sec-offline-import.v1"
SEC_SOURCE_MANIFEST_VERSION = "auditops-sec-source.v2"

_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_SNAPSHOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,15}$")
_SAFE_HTML_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.html?$", re.IGNORECASE)
_SAFE_XSD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.xsd$", re.IGNORECASE)
_SAFE_INDEX_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_MAX_METADATA_BYTES = 2 * 1024 * 1024
_MAX_INDEX_BYTES = 32 * 1024 * 1024
_MAX_PRIMARY_BYTES = 256 * 1024 * 1024
_MAX_SCHEMA_BYTES = 64 * 1024 * 1024
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        return (text + "\n").encode("utf-8")
    return canonical_json_bytes(value, newline=True)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, root: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, _canonical_json_bytes(value, pretty=True))


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}.")
        result[key] = value
    return result


def _read_stable_regular_file(path: Path, label: str, *, max_bytes: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"Missing {label}: {path}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular, non-symlink file: {path}")
    if before.st_size <= 0:
        raise ValueError(f"{label} must not be empty: {path}")
    if before.st_size > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes}-byte import limit: {path}")
    try:
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise ValueError(f"Could not read {label}: {path}") from error
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(payload) != before.st_size:
        raise RuntimeError(f"{label} changed while it was being imported: {path}")
    return payload


def _load_json_object(
    path: Path, label: str, *, max_bytes: int
) -> tuple[dict[str, Any], bytes]:
    payload = _read_stable_regular_file(path, label, max_bytes=max_bytes)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: _raise_invalid_json_constant(token),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value, payload


def _raise_invalid_json_constant(token: str) -> None:
    raise ValueError(f"JSON constant {token!r} is not permitted.")


def _required_text(
    value: Any,
    field: str,
    *,
    max_length: int = 512,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string.")
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > max_length:
        raise ValueError(
            f"{field} must be non-empty and at most {max_length} characters."
        )
    if any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{field} contains a control character.")
    return normalized


def _optional_text(value: Any, field: str, *, max_length: int = 512) -> str | None:
    if value is None or value == "":
        return None
    return _required_text(value, field, max_length=max_length)


def _parse_date(value: Any, field: str) -> date:
    text = _required_text(value, field, max_length=10)
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as error:
        raise ValueError(f"{field} must use YYYY-MM-DD format.") from error
    if parsed.isoformat() != text:
        raise ValueError(f"{field} must use YYYY-MM-DD format.")
    return parsed


def _normalize_cik(value: Any) -> tuple[int, str]:
    if isinstance(value, bool):
        raise ValueError(
            "cik must be a positive integer or a string of at most ten digits."
        )
    if isinstance(value, int):
        digits = str(value)
    elif isinstance(value, str) and value.isdigit():
        digits = value
    else:
        raise ValueError(
            "cik must be a positive integer or a string of at most ten digits."
        )
    if not digits or len(digits) > 10 or int(digits) <= 0:
        raise ValueError(
            "cik must be a positive integer or a string of at most ten digits."
        )
    cik = int(digits)
    return cik, f"{cik:010d}"


def _safe_filename(value: Any, field: str, pattern: re.Pattern[str]) -> str:
    name = _required_text(value, field, max_length=255)
    if (
        pattern.fullmatch(name) is None
        or Path(name).name != name
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
    ):
        raise ValueError(f"{field} must be a safe basename of the expected file type.")
    return name


def _validate_https_sec_url(value: Any, field: str, expected_path: str) -> str:
    url = _required_text(value, field, max_length=2048)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "www.sec.gov"
        or parsed.path != expected_path
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or url != f"https://www.sec.gov{expected_path}"
    ):
        raise ValueError(f"{field} does not match the filing CIK/accession metadata.")
    return url


def _normalized_index_size(value: Any, field: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer or an empty value.")
    if isinstance(value, int):
        size = value
    elif isinstance(value, str) and value.isdigit():
        size = int(value)
    else:
        raise ValueError(f"{field} must be a non-negative integer or an empty value.")
    if size < 0:
        raise ValueError(f"{field} must be a non-negative integer or an empty value.")
    return size


def _validate_source_directory(path: Path, label: str) -> list[Path]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"Missing {label}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a regular, non-symlink directory: {path}")
    entries = sorted(path.iterdir(), key=lambda item: (item.name.casefold(), item.name))
    for entry in entries:
        entry_metadata = entry.lstat()
        if stat.S_ISLNK(entry_metadata.st_mode) or not stat.S_ISREG(
            entry_metadata.st_mode
        ):
            raise ValueError(
                f"{label} may contain only regular, non-symlink files: {entry}"
            )
    return entries


def _validate_source_root(path: Path) -> list[Path]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"Missing offline US source root: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(
            f"Offline US source root must be a regular, non-symlink directory: {path}"
        )
    entries = sorted(path.iterdir(), key=lambda item: (item.name.casefold(), item.name))
    for entry in entries:
        entry_metadata = entry.lstat()
        if stat.S_ISLNK(entry_metadata.st_mode) or not stat.S_ISDIR(
            entry_metadata.st_mode
        ):
            raise ValueError(
                "Offline US source root may contain only regular, non-symlink "
                f"issuer directories: {entry}"
            )
    return entries


def _validate_primary_10k(payload: bytes, ticker: str) -> None:
    try:
        root = etree.parse(
            io.BytesIO(payload),
            etree.XMLParser(
                recover=True,
                huge_tree=True,
                no_network=True,
                resolve_entities=False,
            ),
        ).getroot()
    except (OSError, etree.XMLSyntaxError) as error:
        raise ValueError(
            f"Primary document for {ticker} is not parseable iXBRL."
        ) from error
    document_types: set[str] = set()
    for element in root.iter():
        fact_name = element.get("name")
        if (
            isinstance(fact_name, str)
            and fact_name.rsplit(":", 1)[-1].casefold() == "documenttype"
        ):
            document_types.add(" ".join("".join(element.itertext()).split()).upper())
    if "10-K" not in document_types:
        raise ValueError(
            f"Primary document for {ticker} does not contain a dei:DocumentType of 10-K."
        )


def _normalize_index(
    raw_index: Mapping[str, Any],
    *,
    ticker: str,
    directory_path: str,
    parent_path: str,
    primary_name: str,
    schema_name: str,
    zip_name: str,
    primary_size: int,
    schema_size: int,
) -> dict[str, Any]:
    directory = raw_index.get("directory")
    if not isinstance(directory, Mapping):
        raise ValueError(f"index.json for {ticker} is missing its directory object.")
    if directory.get("name") != directory_path:
        raise ValueError(
            f"index.json directory name does not match metadata for {ticker}."
        )
    if directory.get("parent-dir") != parent_path:
        raise ValueError(
            f"index.json parent directory does not match metadata for {ticker}."
        )
    raw_items = directory.get("item")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError(f"index.json for {ticker} has no directory items.")

    normalized_items: list[dict[str, Any]] = []
    items_by_name: dict[str, dict[str, Any]] = {}
    for position, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, Mapping):
            raise ValueError(
                f"index.json item {position} for {ticker} must be an object."
            )
        name = _safe_filename(
            raw_item.get("name"),
            f"index.json item {position} name",
            _SAFE_INDEX_NAME_RE,
        )
        identity = name.casefold()
        if identity in items_by_name:
            raise ValueError(
                f"index.json for {ticker} contains duplicate item {name!r}."
            )
        normalized_item = {
            "last_modified": _optional_text(
                raw_item.get("last-modified"),
                f"index.json item {name} last-modified",
                max_length=64,
            ),
            "name": name,
            "size": _normalized_index_size(
                raw_item.get("size"), f"index.json item {name} size"
            ),
            "type": _optional_text(
                raw_item.get("type"),
                f"index.json item {name} type",
                max_length=64,
            ),
        }
        items_by_name[identity] = normalized_item
        normalized_items.append(normalized_item)

    for expected_name, expected_size, role in (
        (primary_name, primary_size, "primary document"),
        (schema_name, schema_size, "schema"),
    ):
        item = items_by_name.get(expected_name.casefold())
        if item is None:
            raise ValueError(f"index.json for {ticker} does not list the {role}.")
        if item["name"] != expected_name:
            raise ValueError(
                f"index.json {role} case does not match the local filename."
            )
        if item["size"] != expected_size:
            raise ValueError(f"index.json {role} size does not match the local file.")
    if zip_name.casefold() not in items_by_name:
        raise ValueError(f"index.json for {ticker} does not list the SEC XBRL ZIP.")

    normalized_items.sort(key=lambda item: (item["name"].casefold(), item["name"]))
    return {
        "directory": {
            "items": normalized_items,
            "name": directory_path,
            "parent_dir": parent_path,
        },
        "index_version": OFFLINE_US_IMPORT_VERSION,
    }


def _write_deterministic_zip(path: Path, members: Mapping[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        with zipfile.ZipFile(
            output, mode="w", compression=zipfile.ZIP_STORED
        ) as archive:
            for name in sorted(members, key=lambda item: (item.casefold(), item)):
                info = zipfile.ZipInfo(filename=name, date_time=_ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.extra = b""
                info.comment = b""
                archive.writestr(info, members[name])
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if names != sorted(members, key=lambda item: (item.casefold(), item)):
            raise RuntimeError(f"Deterministic ZIP member order is invalid: {path}")
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise RuntimeError(
                f"Deterministic ZIP CRC failed at {corrupt_member}: {path}"
            )


def _tree_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(rows)))


def _parse_snapshot_date(value: str) -> date:
    parsed = _parse_date(value, "snapshot_date")
    if parsed > datetime.now(timezone.utc).date():
        raise ValueError("snapshot_date cannot be in the future.")
    return parsed


def _prepare_filing(
    issuer_dir: Path,
    *,
    snapshot_id: str,
    snapshot_date: str,
    snapshot_cutoff: date,
) -> dict[str, Any]:
    entries = _validate_source_directory(issuer_dir, "issuer directory")
    by_name = {entry.name: entry for entry in entries}
    if "filing_meta.json" not in by_name or "index.json" not in by_name:
        raise ValueError(
            f"Issuer directory {issuer_dir.name} must contain filing_meta.json and index.json."
        )

    raw_meta, meta_bytes = _load_json_object(
        by_name["filing_meta.json"],
        "filing metadata",
        max_bytes=_MAX_METADATA_BYTES,
    )
    raw_index, index_bytes = _load_json_object(
        by_name["index.json"],
        "SEC directory index",
        max_bytes=_MAX_INDEX_BYTES,
    )

    ticker = _required_text(raw_meta.get("ticker"), "ticker", max_length=16).upper()
    if _TICKER_RE.fullmatch(ticker) is None or issuer_dir.name != ticker:
        raise ValueError(
            f"Issuer directory {issuer_dir.name!r} does not exactly match ticker {ticker!r}."
        )
    company_name = _required_text(raw_meta.get("company"), "company", max_length=512)
    sector = _optional_text(raw_meta.get("sector"), "sector", max_length=256)
    cik, cik_padded = _normalize_cik(raw_meta.get("cik"))
    accession = _required_text(raw_meta.get("accession"), "accession", max_length=20)
    if _ACCESSION_RE.fullmatch(accession) is None:
        raise ValueError(f"Malformed SEC accession for {ticker}: {accession!r}.")

    filing_date = _parse_date(raw_meta.get("filing_date"), "filing_date")
    report_date = _parse_date(raw_meta.get("period_end"), "period_end")
    if filing_date > snapshot_cutoff:
        raise ValueError(
            f"Filing date {filing_date.isoformat()} for {ticker} is after snapshot_date."
        )
    if report_date > filing_date:
        raise ValueError(f"period_end for {ticker} cannot be after filing_date.")
    if int(accession[11:13]) != filing_date.year % 100:
        raise ValueError(f"Accession year does not match filing_date for {ticker}.")

    primary_name = _safe_filename(
        raw_meta.get("primary_document"), "primary_document", _SAFE_HTML_RE
    )
    html_files = [entry for entry in entries if _SAFE_HTML_RE.fullmatch(entry.name)]
    schema_files = [entry for entry in entries if _SAFE_XSD_RE.fullmatch(entry.name)]
    expected_names = {"filing_meta.json", "index.json", primary_name}
    if len(html_files) != 1 or html_files[0].name != primary_name:
        raise ValueError(
            f"Issuer directory {ticker} must contain exactly its declared primary HTML document."
        )
    if len(schema_files) != 1:
        raise ValueError(
            f"Issuer directory {ticker} must contain exactly one XSD schema."
        )
    schema_name = schema_files[0].name
    expected_names.add(schema_name)
    if {entry.name for entry in entries} != expected_names:
        unexpected = sorted({entry.name for entry in entries} - expected_names)
        raise ValueError(
            f"Issuer directory {ticker} contains unexpected files: {unexpected}."
        )
    if Path(primary_name).stem.casefold() != Path(schema_name).stem.casefold():
        raise ValueError(
            f"Primary document and schema stems do not match for {ticker}."
        )

    primary_bytes = _read_stable_regular_file(
        html_files[0], "primary iXBRL document", max_bytes=_MAX_PRIMARY_BYTES
    )
    schema_bytes = _read_stable_regular_file(
        schema_files[0], "XBRL schema", max_bytes=_MAX_SCHEMA_BYTES
    )
    primary_lower = primary_bytes[: 4 * 1024 * 1024].lower()
    if b"<html" not in primary_lower or (
        b"inlinexbrl" not in primary_lower and b"<ix:" not in primary_lower
    ):
        raise ValueError(
            f"Primary document for {ticker} is not recognizable inline XBRL."
        )
    _validate_primary_10k(primary_bytes, ticker)
    if b"<" not in schema_bytes[:4096] or b"schema" not in schema_bytes[:4096].lower():
        raise ValueError(f"XSD for {ticker} is not recognizable XML Schema content.")

    compact_accession = accession.replace("-", "")
    directory_path = f"/Archives/edgar/data/{cik}/{compact_accession}"
    parent_path = f"/Archives/edgar/data/{cik}"
    filing_path = f"{directory_path}/{primary_name}"
    directory_url = _validate_https_sec_url(
        raw_meta.get("directory_url"), "directory_url", directory_path
    )
    filing_url = _validate_https_sec_url(
        raw_meta.get("filing_url"), "filing_url", filing_path
    )
    zip_name = f"{accession}-xbrl.zip"
    normalized_index = _normalize_index(
        raw_index,
        ticker=ticker,
        directory_path=directory_path,
        parent_path=parent_path,
        primary_name=primary_name,
        schema_name=schema_name,
        zip_name=zip_name,
        primary_size=len(primary_bytes),
        schema_size=len(schema_bytes),
    )

    inline_value = raw_meta.get("is_inline_xbrl")
    if inline_value not in {1, "1", True, "true", "True"}:
        raise ValueError(f"Filing metadata for {ticker} is not marked as inline XBRL.")
    zip_url = f"{directory_url}/{zip_name}"
    member_records = [
        {
            "bytes": len(payload),
            "name": name,
            "sha256": _sha256_bytes(payload),
        }
        for name, payload in sorted(
            ((primary_name, primary_bytes), (schema_name, schema_bytes)),
            key=lambda item: (item[0].casefold(), item[0]),
        )
    ]
    normalized_meta = {
        "accession": accession,
        "cik": cik,
        "cik_padded": cik_padded,
        "company_name": company_name,
        "directory_url": directory_url,
        "filing_date": filing_date.isoformat(),
        "filing_url": filing_url,
        "form_type": "10-K",
        "is_inline_xbrl": True,
        "metadata_version": OFFLINE_US_IMPORT_VERSION,
        "period_end": report_date.isoformat(),
        "primary_document": primary_name,
        "schema_document": schema_name,
        "sector": sector,
        "source_files": member_records,
        "ticker": ticker,
        "zip_url": zip_url,
    }
    issuer_row = {
        "canonical_ticker": ticker,
        "cik": cik,
        "company_name": company_name,
        "constituent_source": OFFLINE_US_IMPORT_VERSION,
        "duplicate_cik_of": None,
        "filing_span": "offline_snapshot",
        "forms_expected": ["10-K"],
        "ingest_enabled": True,
        "sec_submission_url": f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
        "snapshot_date": snapshot_date,
        "snapshot_id": snapshot_id,
        "status": "resolved",
        "ticker": ticker,
        "trailing_fiscal_years": 1,
        "error": None,
    }
    filing_row = {
        "accession": accession,
        "canonical_ticker": ticker,
        "cik": cik,
        "company_name": company_name,
        "error": None,
        "filed_at": filing_date.isoformat(),
        "filing_span": "offline_snapshot",
        "form_type": "10-K",
        "ingest_enabled": True,
        "primary_doc": primary_name,
        "report_date": report_date.isoformat(),
        "selection_rank": 1,
        "snapshot_date": snapshot_date,
        "snapshot_id": snapshot_id,
        "status": "resolved",
        "ticker": ticker,
        "trailing_fiscal_years": 1,
        "zip_url": zip_url,
    }
    input_records = [
        {
            "bytes": len(meta_bytes),
            "role": "legacy_filing_metadata",
            "sha256": _sha256_bytes(meta_bytes),
            "ticker": ticker,
        },
        {
            "bytes": len(index_bytes),
            "role": "sec_directory_index",
            "sha256": _sha256_bytes(index_bytes),
            "ticker": ticker,
        },
        *(
            {
                "bytes": member["bytes"],
                "name": member["name"],
                "role": "filing_member",
                "sha256": member["sha256"],
                "ticker": ticker,
            }
            for member in member_records
        ),
    ]
    return {
        "accession": accession,
        "filing_row": filing_row,
        "index": normalized_index,
        "input_records": input_records,
        "issuer_row": issuer_row,
        "members": {primary_name: primary_bytes, schema_name: schema_bytes},
        "metadata": normalized_meta,
        "ticker": ticker,
    }


def import_offline_us_cache(
    source_root: str | Path,
    corpus_root: str | Path,
    *,
    snapshot_id: str = "us_sec_existing20_realdata_v1",
    snapshot_date: str = "2026-08-21",
) -> dict[str, Any]:
    """Import a pre-existing SEC filing cache into an immutable AuditOps corpus.

    The source is never modified. The destination is built in a sibling staging
    directory and published atomically only after all inputs and output hashes
    have been validated. Existing destinations are always refused.
    """

    if (
        not isinstance(snapshot_id, str)
        or _SNAPSHOT_ID_RE.fullmatch(snapshot_id) is None
    ):
        raise ValueError("snapshot_id must be a stable ASCII identifier.")
    snapshot_cutoff = _parse_snapshot_date(snapshot_date)

    source = Path(source_root)
    _validate_source_root(source)
    source = source.resolve(strict=True)
    target = Path(corpus_root).resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Offline corpus destination already exists: {target}")
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and corpus roots must not contain one another.")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.building")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(
            f"Offline corpus staging destination already exists: {staging}"
        )

    try:
        issuer_dirs = _validate_source_root(source)
        if not issuer_dirs:
            raise ValueError("Offline US source root contains no issuer directories.")
        prepared = [
            _prepare_filing(
                issuer_dir,
                snapshot_id=snapshot_id,
                snapshot_date=snapshot_date,
                snapshot_cutoff=snapshot_cutoff,
            )
            for issuer_dir in issuer_dirs
        ]

        seen_tickers: set[str] = set()
        seen_ciks: set[int] = set()
        seen_accessions: set[str] = set()
        for filing in prepared:
            ticker = filing["ticker"]
            cik = filing["issuer_row"]["cik"]
            accession = filing["accession"]
            if ticker in seen_tickers:
                raise ValueError(
                    f"Duplicate ticker identity in offline source: {ticker}"
                )
            if cik in seen_ciks:
                raise ValueError(f"Duplicate CIK identity in offline source: {cik}")
            if accession in seen_accessions:
                raise ValueError(
                    f"Duplicate accession identity in offline source: {accession}"
                )
            seen_tickers.add(ticker)
            seen_ciks.add(cik)
            seen_accessions.add(accession)

        layout = ensure_corpus_layout(staging)
        normalized_source_root = staging / "raw" / "offline_source"
        source_records: list[dict[str, Any]] = []
        ledger_rows: list[dict[str, Any]] = []
        all_input_records: list[dict[str, Any]] = []

        for filing in prepared:
            ticker = filing["ticker"]
            accession = filing["accession"]
            normalized_dir = normalized_source_root / ticker
            meta_path = normalized_dir / "filing_meta.json"
            index_path = normalized_dir / "index.json"
            _write_json(meta_path, filing["metadata"])
            _write_json(index_path, filing["index"])

            zip_path = layout.raw_xbrl_zip / ticker / f"{ticker}_10-K_{accession}.zip"
            _write_deterministic_zip(zip_path, filing["members"])
            zip_sha256 = _sha256_file(zip_path)
            zip_bytes = zip_path.stat().st_size
            if _SHA256_RE.fullmatch(zip_sha256) is None or zip_bytes <= 0:
                raise RuntimeError(f"Invalid synthesized ZIP lineage for {ticker}.")

            source_records.extend(
                [
                    {
                        **_file_record(meta_path, staging),
                        "input_bytes": filing["input_records"][0]["bytes"],
                        "input_sha256": filing["input_records"][0]["sha256"],
                        "role": "normalized_filing_metadata",
                        "ticker": ticker,
                    },
                    {
                        **_file_record(index_path, staging),
                        "input_bytes": filing["input_records"][1]["bytes"],
                        "input_sha256": filing["input_records"][1]["sha256"],
                        "role": "normalized_sec_directory_index",
                        "ticker": ticker,
                    },
                    {
                        "accession": accession,
                        "bytes": zip_bytes,
                        "members": filing["metadata"]["source_files"],
                        "path": zip_path.relative_to(staging).as_posix(),
                        "role": "offline_filing_package",
                        "sha256": zip_sha256,
                        "ticker": ticker,
                        "url": filing["filing_row"]["zip_url"],
                    },
                ]
            )
            all_input_records.extend(filing["input_records"])
            ledger_rows.append(
                {
                    **filing["filing_row"],
                    "attempts": 0,
                    "bytes": zip_bytes,
                    "http_status": None,
                    "local_path": str(
                        target
                        / "raw"
                        / "xbrl_zip"
                        / ticker
                        / f"{ticker}_10-K_{accession}.zip"
                    ),
                    "sha256": zip_sha256,
                    "source_mode": "offline_import",
                    "status": "existing_local",
                }
            )

        issuer_rows = [filing["issuer_row"] for filing in prepared]
        filing_rows = [filing["filing_row"] for filing in prepared]
        issuer_manifest_path = layout.manifest / "issuer_manifest.jsonl"
        filing_manifest_path = layout.manifest / "filing_manifest.jsonl"
        ledger_path = layout.manifest / "download_ledger.jsonl"
        write_jsonl(issuer_manifest_path, issuer_rows)
        write_jsonl(filing_manifest_path, filing_rows)
        write_jsonl(ledger_path, ledger_rows)

        artifacts = {
            "download_ledger": _file_record(
                ledger_path, staging, rows=len(ledger_rows)
            ),
            "filing_manifest": _file_record(
                filing_manifest_path, staging, rows=len(filing_rows)
            ),
            "issuer_manifest": _file_record(
                issuer_manifest_path, staging, rows=len(issuer_rows)
            ),
        }
        source_records.sort(
            key=lambda row: (
                str(row.get("ticker", "")).casefold(),
                str(row.get("role", "")),
                str(row.get("path", "")),
            )
        )
        all_input_records.sort(
            key=lambda row: (
                str(row.get("ticker", "")).casefold(),
                str(row.get("role", "")),
                str(row.get("name", "")),
            )
        )
        source_manifest = {
            "artifacts": artifacts,
            "capture_mode": "preexisting_offline_sec_files",
            "input_file_count": len(all_input_records),
            "input_tree_sha256": _tree_digest(all_input_records),
            "offline_import_version": OFFLINE_US_IMPORT_VERSION,
            "retrieved_at": None,
            "snapshot_date": snapshot_date,
            "snapshot_id": snapshot_id,
            "source_manifest_version": SEC_SOURCE_MANIFEST_VERSION,
            "sources": source_records,
            "user_agent_sha256": None,
        }
        source_manifest_path = layout.manifest / "source_manifest.json"
        _write_json(source_manifest_path, source_manifest)

        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"Offline corpus destination appeared during import: {target}"
            )
        os.replace(staging, target)
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise

    return {
        "corpus_root": str(target),
        "download_ledger": str(target / "manifest" / "download_ledger.jsonl"),
        "filing_count": len(prepared),
        "filing_manifest": str(target / "manifest" / "filing_manifest.jsonl"),
        "issuer_count": len(prepared),
        "issuer_manifest": str(target / "manifest" / "issuer_manifest.jsonl"),
        "snapshot_date": snapshot_date,
        "snapshot_id": snapshot_id,
        "source_manifest": str(target / "manifest" / "source_manifest.json"),
        "source_manifest_sha256": _sha256_file(
            target / "manifest" / "source_manifest.json"
        ),
    }
