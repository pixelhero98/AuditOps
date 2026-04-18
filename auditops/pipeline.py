from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import zipfile
import calendar
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lxml import etree

from .narrative import NarrativeChunk, Section, build_narrative_chunks, extract_narrative_text


ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NUM_MASK_RE = re.compile(r"\b\d[\d,.\-%]*\b")


@dataclass(frozen=True)
class FilingMetadata:
    filing_id: str
    ticker: Optional[str]
    zip_name: str
    main_html: str
    processed_at: str
    form_type: Optional[str]
    fiscal_year_focus: Optional[int]
    fiscal_period_focus: Optional[str]
    report_date: Optional[date]


@dataclass(frozen=True)
class ContextRec:
    context_id: str
    entity_identifier: str
    start_date: Optional[date]
    end_date: Optional[date]
    instant_date: Optional[date]
    period_key: str
    days: Optional[int]
    dimensions: Dict[str, Any]
    dimension_count: int
    is_consolidated: bool


@dataclass(frozen=True)
class UnitRec:
    unit_id: str
    unit_json: Dict[str, Any]


@dataclass(frozen=True)
class FactRec:
    fact_id: str
    fact_ordinal: int
    fact_evidence_id: str
    concept_qname: str
    concept_norm: str
    context_id: Optional[str]
    unit_id: Optional[str]
    decimals: Optional[str]
    scale: Optional[str]
    sign: Optional[str]
    fmt: Optional[str]
    value_type: str
    value_text: Optional[str]
    value_lexical: Optional[str]
    value_num_reported: Optional[Decimal]
    value_num_unscaled: Optional[Decimal]
    period_key: str
    dimensions: Dict[str, Any]
    is_consolidated: bool
    source_file: str
    source_anchor: Optional[str]


@dataclass(frozen=True)
class ValidationRecord:
    validator_code: str
    entity_kind: str
    entity_key: str
    message: str
    details: Dict[str, Any]


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS filings (
  filing_id TEXT PRIMARY KEY,
  ticker TEXT,
  zip_name TEXT,
  main_html TEXT,
  processed_at TEXT,
  form_type TEXT,
  fiscal_year_focus INTEGER,
  fiscal_period_focus TEXT,
  report_date TEXT
);

CREATE TABLE IF NOT EXISTS contexts (
  filing_id TEXT,
  context_id TEXT,
  entity_identifier TEXT,
  start_date TEXT,
  end_date TEXT,
  instant_date TEXT,
  period_key TEXT,
  days INTEGER,
  dimensions_json TEXT,
  dimension_count INTEGER,
  is_consolidated INTEGER,
  PRIMARY KEY (filing_id, context_id),
  FOREIGN KEY (filing_id) REFERENCES filings(filing_id)
);

CREATE TABLE IF NOT EXISTS units (
  filing_id TEXT,
  unit_id TEXT,
  unit_json TEXT,
  PRIMARY KEY (filing_id, unit_id),
  FOREIGN KEY (filing_id) REFERENCES filings(filing_id)
);

CREATE TABLE IF NOT EXISTS facts (
  filing_id TEXT,
  fact_id TEXT,
  fact_ordinal INTEGER,
  fact_evidence_id TEXT,
  concept_qname TEXT,
  concept_norm TEXT,
  context_id TEXT,
  unit_id TEXT,
  decimals TEXT,
  scale TEXT,
  sign TEXT,
  fmt TEXT,
  value_type TEXT,
  value_text TEXT,
  value_lexical TEXT,
  value_num_reported TEXT,
  value_num_unscaled TEXT,
  period_key TEXT,
  dimensions_json TEXT,
  is_consolidated INTEGER,
  source_file TEXT,
  source_anchor TEXT,
  PRIMARY KEY (filing_id, source_file, fact_id),
  FOREIGN KEY (filing_id, context_id) REFERENCES contexts(filing_id, context_id),
  FOREIGN KEY (filing_id, unit_id) REFERENCES units(filing_id, unit_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_concept_period ON facts(filing_id, concept_norm, period_key);
CREATE INDEX IF NOT EXISTS idx_facts_evidence ON facts(filing_id, fact_evidence_id);

CREATE TABLE IF NOT EXISTS labels (
  filing_id TEXT,
  concept_norm TEXT,
  label_role TEXT,
  label_lang TEXT,
  label_text TEXT,
  PRIMARY KEY (filing_id, concept_norm, label_role, label_lang, label_text),
  FOREIGN KEY (filing_id) REFERENCES filings(filing_id)
);

CREATE TABLE IF NOT EXISTS pre_edges (
  filing_id TEXT,
  role TEXT,
  parent_concept_norm TEXT,
  child_concept_norm TEXT,
  ord REAL,
  preferred_label TEXT,
  FOREIGN KEY (filing_id) REFERENCES filings(filing_id)
);

CREATE TABLE IF NOT EXISTS cal_edges (
  filing_id TEXT,
  role TEXT,
  parent_concept_norm TEXT,
  child_concept_norm TEXT,
  weight REAL,
  ord REAL,
  FOREIGN KEY (filing_id) REFERENCES filings(filing_id)
);

CREATE TABLE IF NOT EXISTS def_edges (
  filing_id TEXT,
  role TEXT,
  arcrole TEXT,
  parent_concept_norm TEXT,
  child_concept_norm TEXT,
  ord REAL,
  FOREIGN KEY (filing_id) REFERENCES filings(filing_id)
);

CREATE TABLE IF NOT EXISTS narrative_sections (
  filing_id TEXT NOT NULL,
  source_file TEXT NOT NULL,
  item TEXT,
  heading TEXT,
  start_char INTEGER NOT NULL,
  end_char INTEGER NOT NULL,
  PRIMARY KEY (filing_id, source_file, start_char),
  FOREIGN KEY (filing_id) REFERENCES filings(filing_id)
);

CREATE TABLE IF NOT EXISTS narrative_chunks (
  filing_id TEXT NOT NULL,
  source_file TEXT NOT NULL,
  item TEXT,
  heading TEXT,
  subheading TEXT,
  chunk_index INTEGER NOT NULL,
  char_start INTEGER NOT NULL,
  char_end INTEGER NOT NULL,
  heading_path TEXT,
  retrieval_text TEXT,
  text_raw TEXT NOT NULL,
  text_masked TEXT NOT NULL,
  text_sha1 TEXT NOT NULL,
  PRIMARY KEY (filing_id, source_file, char_start),
  FOREIGN KEY (filing_id) REFERENCES filings(filing_id)
);

CREATE INDEX IF NOT EXISTS idx_narrative_chunks_item ON narrative_chunks(filing_id, source_file, item, chunk_index);

CREATE TABLE IF NOT EXISTS facts_canon (
  canon_id TEXT PRIMARY KEY,
  filing_id TEXT NOT NULL,
  ticker TEXT,
  fact_evidence_id TEXT NOT NULL,
  raw_fact_ref TEXT NOT NULL,
  concept_norm TEXT NOT NULL,
  value_type TEXT NOT NULL,
  period_type TEXT NOT NULL,
  period_key TEXT NOT NULL,
  period_start TEXT,
  period_end TEXT,
  fiscal_year INTEGER,
  fiscal_quarter INTEGER,
  is_ytd INTEGER NOT NULL,
  unit_canon TEXT,
  value_num_exact TEXT,
  value_text TEXT,
  selection_rank INTEGER NOT NULL,
  selection_reason TEXT NOT NULL,
  source_anchor TEXT,
  FOREIGN KEY (filing_id) REFERENCES filings(filing_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_canon_lookup ON facts_canon(filing_id, concept_norm, period_key, unit_canon);

CREATE TABLE IF NOT EXISTS chunk_canon (
  chunk_evidence_id TEXT PRIMARY KEY,
  filing_id TEXT NOT NULL,
  period_key TEXT,
  item TEXT,
  heading TEXT,
  subheading TEXT,
  heading_path TEXT,
  source_file TEXT NOT NULL,
  char_start INTEGER NOT NULL,
  char_end INTEGER NOT NULL,
  retrieval_text TEXT,
  text_masked TEXT NOT NULL,
  text_sha1 TEXT NOT NULL,
  FOREIGN KEY (filing_id) REFERENCES filings(filing_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_canon_lookup ON chunk_canon(filing_id, period_key, item);

CREATE TABLE IF NOT EXISTS validators_v0 (
  validator_id TEXT PRIMARY KEY,
  filing_id TEXT NOT NULL,
  validator_code TEXT NOT NULL,
  entity_kind TEXT NOT NULL,
  entity_key TEXT NOT NULL,
  message TEXT NOT NULL,
  details_json TEXT NOT NULL,
  FOREIGN KEY (filing_id) REFERENCES filings(filing_id)
);
"""


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def connect_db(path: str) -> sqlite3.Connection:
    """Open a SQLite connection configured for the AuditOps schema.
    
    Parameters
    ----------
    path : str
        Database path. e.g., auditops.sqlite
    
    Returns
    -------
    sqlite3.Connection
        SQLite connection configured with row_factory and foreign-key enforcement.
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def ensure_schema(conn: sqlite3.Connection, reset: bool = False) -> None:
    """Create or reset database tables required by the pipeline.
    
    Parameters
    ----------
    conn : sqlite3.Connection
        Open SQLite connection for the corpus database.
    reset : bool, optional
        Whether to drop or reset existing structures before rebuilding.
    
    Examples
    --------
    >>> conn = connect_db('corpora/sp500_latest_2026-03-20/db/corpus.sqlite')
    >>> ensure_schema(conn=conn)  # doctest: +SKIP
    """
    if reset:
        conn.executescript(
            """
            PRAGMA foreign_keys=OFF;
            DROP TABLE IF EXISTS validators_v0;
            DROP TABLE IF EXISTS chunk_canon;
            DROP TABLE IF EXISTS facts_canon;
            DROP TABLE IF EXISTS narrative_chunks;
            DROP TABLE IF EXISTS narrative_sections;
            DROP TABLE IF EXISTS def_edges;
            DROP TABLE IF EXISTS cal_edges;
            DROP TABLE IF EXISTS pre_edges;
            DROP TABLE IF EXISTS labels;
            DROP TABLE IF EXISTS facts;
            DROP TABLE IF EXISTS units;
            DROP TABLE IF EXISTS contexts;
            DROP TABLE IF EXISTS filings;
            PRAGMA foreign_keys=ON;
            """
        )
        conn.commit()
    conn.executescript(SCHEMA_SQL)
    narrative_chunks_sql_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='narrative_chunks'"
    ).fetchone()
    narrative_chunks_sql = (narrative_chunks_sql_row["sql"] if narrative_chunks_sql_row else "") or ""
    if "PRIMARY KEY (filing_id, source_file, item, chunk_index)" in narrative_chunks_sql:
        conn.executescript(
            """
            ALTER TABLE narrative_chunks RENAME TO narrative_chunks_legacy;
            CREATE TABLE narrative_chunks (
              filing_id TEXT NOT NULL,
              source_file TEXT NOT NULL,
              item TEXT,
              heading TEXT,
              subheading TEXT,
              chunk_index INTEGER NOT NULL,
              char_start INTEGER NOT NULL,
              char_end INTEGER NOT NULL,
              heading_path TEXT,
              retrieval_text TEXT,
              text_raw TEXT NOT NULL,
              text_masked TEXT NOT NULL,
              text_sha1 TEXT NOT NULL,
              PRIMARY KEY (filing_id, source_file, char_start),
              FOREIGN KEY (filing_id) REFERENCES filings(filing_id)
            );
            INSERT INTO narrative_chunks(
              filing_id, source_file, item, heading, subheading, chunk_index, char_start, char_end,
              heading_path, retrieval_text, text_raw, text_masked, text_sha1
            )
            SELECT
              filing_id, source_file, item, heading, subheading, chunk_index, char_start, char_end,
              heading_path, retrieval_text, text_raw, text_masked, text_sha1
            FROM narrative_chunks_legacy;
            DROP TABLE narrative_chunks_legacy;
            CREATE INDEX IF NOT EXISTS idx_narrative_chunks_item ON narrative_chunks(filing_id, source_file, item, chunk_index);
            """
        )
    for table_name, required_columns in {
        "narrative_chunks": {
            "heading": "TEXT",
            "subheading": "TEXT",
            "heading_path": "TEXT",
            "retrieval_text": "TEXT",
        },
        "chunk_canon": {
            "subheading": "TEXT",
            "heading_path": "TEXT",
            "retrieval_text": "TEXT",
        },
    }.items():
        existing = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for column_name, column_type in required_columns.items():
            if column_name not in existing:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
    conn.commit()


def safe_text(el: Optional[etree._Element]) -> str:
    """Return stripped text content for an XML element, or an empty string.
    
    Parameters
    ----------
    el : Optional[etree._Element]
        XML element node under inspection.
    
    Returns
    -------
    str
        Text or path value produced by this function.
    
    Examples
    --------
    >>> result = safe_text(el=None)  # doctest: +SKIP
    """
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def resolve_continued_text(root: etree._Element, fact_el: etree._Element) -> str:
    """Execute resolve continued text.
    
    Parameters
    ----------
    root : etree._Element
        Root XML element for the parsed document tree.
    fact_el : etree._Element
        XML element corresponding to the current fact node.
    
    Returns
    -------
    str
        Text or path value produced by this function.
    
    Examples
    --------
    >>> result = resolve_continued_text(root=None, fact_el=None)  # doctest: +SKIP
    """
    if fact_el is None:
        return ""
    parts = [safe_text(fact_el)]
    cont_id = fact_el.get("continuedAt")
    if not cont_id:
        return parts[0].strip()

    seen = set()
    while cont_id and cont_id not in seen:
        seen.add(cont_id)
        els = root.xpath(f"//*[@id='{cont_id}' or @xml:id='{cont_id}'][local-name()='continuation']")
        if not els:
            break
        cont_el = els[0]
        parts.append(safe_text(cont_el))
        cont_id = cont_el.get("continuedAt")
    return " ".join(part.strip() for part in parts if part and part.strip())


def parse_date(text: str) -> Optional[date]:
    """Parse an ISO date string into a date object when possible.
    
    Parameters
    ----------
    text : str
        Text content to normalize, segment, classify, or tokenize.
    
    Returns
    -------
    Optional[date]
        Value returned by this operation.
    
    Examples
    --------
    >>> result = parse_date(text='example-value')  # doctest: +SKIP
    """
    value = (text or "").strip()
    if not value or not ISO_DATE_RE.match(value):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def norm_concept_for_linkbase(qname: str) -> str:
    """Normalize a concept name for linkbase lookup.
    
    Parameters
    ----------
    qname : str
        Qualified concept name from taxonomy/linkbase references.
    
    Returns
    -------
    str
        String/path value produced while a concept name for linkbase lookup.
    
    Examples
    --------
    >>> result = norm_concept_for_linkbase(qname='example-value')  # doctest: +SKIP
    """
    return (qname or "").strip().replace(":", "_")


def get_attr_any_ns(el: etree._Element, attr_local: str) -> Optional[str]:
    """Read an XML attribute by local name across namespaces.
    
    Parameters
    ----------
    el : etree._Element
        XML element node under inspection.
    attr_local : str
        Local XML attribute name to resolve across namespaces.
    
    Returns
    -------
    Optional[str]
        Value returned by this operation.
    
    Examples
    --------
    >>> result = get_attr_any_ns(el=None, attr_local='example-value')  # doctest: +SKIP
    """
    for key, value in el.attrib.items():
        if key.split("}")[-1] == attr_local:
            return value
    return None


def parse_numeric(text: str) -> Optional[Decimal]:
    """Parse numeric text into a Decimal value when possible.
    
    Parameters
    ----------
    text : str
        Text content to normalize, segment, classify, or tokenize.
    
    Returns
    -------
    Optional[Decimal]
        Value returned by this operation.
    
    Examples
    --------
    >>> result = parse_numeric(text='example-value')  # doctest: +SKIP
    """
    if text is None:
        return None
    value = text.strip()
    if not value:
        return None
    value = value.replace("$", "").replace("€", "").replace("£", "")
    value = value.replace("\u2212", "-")
    value = re.sub(r"\s+", "", value)
    negative = False
    if value.startswith("(") and value.endswith(")"):
        negative = True
        value = value[1:-1]
    value = value.replace(",", "")
    if value in {"--", "—", "N/A", "NA"}:
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    return -parsed if negative else parsed


def apply_scale(value: Optional[Decimal], scale: Optional[str]) -> Optional[Decimal]:
    """Apply iXBRL scale semantics to a numeric value.
    
    Parameters
    ----------
    value : Optional[Decimal]
        Input parameter for value.
    scale : Optional[str]
        Inline XBRL scale attribute used to adjust numeric values.
    
    Returns
    -------
    Optional[Decimal]
        Value returned by this operation.
    
    Examples
    --------
    >>> result = apply_scale(value=None, scale='example-value')  # doctest: +SKIP
    """
    if value is None or not scale:
        return value
    try:
        return value * (Decimal(10) ** Decimal(int(scale)))
    except Exception:
        return value


def apply_sign(value: Optional[Decimal], sign: Optional[str]) -> Optional[Decimal]:
    """Apply iXBRL sign semantics to a numeric value.
    
    Parameters
    ----------
    value : Optional[Decimal]
        Input parameter for value.
    sign : Optional[str]
        Inline XBRL sign attribute applied to numeric values.
    
    Returns
    -------
    Optional[Decimal]
        Value returned by this operation.
    
    Examples
    --------
    >>> result = apply_sign(value=None, sign='example-value')  # doctest: +SKIP
    """
    if value is None:
        return None
    return -value if sign and sign.strip() == "-" else value


_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _norm_fmt(fmt: Optional[str]) -> str:
    if not fmt:
        return ""
    value = fmt.strip()
    value = value.split("#")[-1]
    value = value.split(":")[-1]
    return value.lower()


def _words_to_int(text: str) -> Optional[int]:
    cleaned = re.sub(r"[^a-z\s-]", " ", (text or "").lower()).strip().replace("-", " ")
    if not cleaned:
        return None
    total = 0
    current = 0
    for token in [part for part in cleaned.split() if part]:
        if token in _NUM_WORDS:
            current += _NUM_WORDS[token]
        elif token == "hundred":
            current = max(1, current) * 100
        elif token == "thousand":
            total += max(1, current) * 1_000
            current = 0
        elif token == "million":
            total += max(1, current) * 1_000_000
            current = 0
        elif token == "billion":
            total += max(1, current) * 1_000_000_000
            current = 0
        else:
            return None
    return total + current


def _parse_monthname_date(text: str) -> Optional[str]:
    match = re.search(r"([A-Za-z]{3,9})\.?\s+(\d{1,2})\s*,\s*(\d{4})", (text or "").strip())
    if not match:
        return None
    month = _MONTHS.get(match.group(1).lower())
    if not month:
        return None
    try:
        return date(int(match.group(3)), month, int(match.group(2))).isoformat()
    except Exception:
        return None


def apply_ix_transform(fmt: Optional[str], raw_text: str) -> Tuple[str, Optional[str]]:
    """Apply an inline XBRL value transform and return normalized text/date.
    
    Parameters
    ----------
    fmt : Optional[str]
        Inline XBRL transform format identifier.
    raw_text : str
        Raw text value before normalization or transformation.
    
    Returns
    -------
    Tuple[str, Optional[str]]
        Tuple with outputs produced while apply an inline xbrl value transform and return normalized text/date.
    
    Examples
    --------
    >>> result = apply_ix_transform(fmt='example-value', raw_text='example-value')  # doctest: +SKIP
    >>> len(result)  # doctest: +SKIP
    """
    normalized = _norm_fmt(fmt)
    text = (raw_text or "").strip()
    if normalized == "fixed-zero":
        return "0", "numeric"
    if normalized == "fixed-true":
        return "true", "bool"
    if normalized == "fixed-false":
        return "false", "bool"
    if normalized == "numwordsen":
        number = _words_to_int(text)
        return (str(number) if number is not None else text), ("numeric" if number is not None else None)
    if normalized in {"durday", "durmonth", "duryear", "durwordsen"}:
        number = None
        match = re.search(r"(-?\d+)", text)
        if match:
            number = int(match.group(1))
        else:
            number = _words_to_int(text)
        if number is None:
            return text, None
        if "month" in normalized:
            return f"P{number}M", "duration"
        if "year" in normalized:
            return f"P{number}Y", "duration"
        return f"P{number}D", "duration"
    if normalized.startswith("date-monthname-day-year"):
        iso = _parse_monthname_date(text)
        return (iso if iso else text), ("date" if iso else None)
    return text, None


def period_key_from_context(start: Optional[date], end: Optional[date], instant: Optional[date]) -> Tuple[str, Optional[int]]:
    """Build a canonical period key from context start/end/instant dates.
    
    Parameters
    ----------
    start : Optional[date]
        Context period start date.
    end : Optional[date]
        Context period end date.
    instant : Optional[date]
        Context instant date for as-of facts.
    
    Returns
    -------
    Tuple[str, Optional[int]]
        Tuple of outputs produced by this function.
    
    Examples
    --------
    >>> result = period_key_from_context(start=None, end=None, instant=None)  # doctest: +SKIP
    >>> len(result)  # doctest: +SKIP
    """
    if instant:
        return f"ASOF_{instant.strftime('%Y%m%d')}", None
    if start and end:
        days = (end - start).days
        if 330 <= days <= 370:
            return f"FY{end.year}", days
        if 80 <= days <= 110 and (end.month, end.day) in {(3, 31), (6, 30), (9, 30), (12, 31)}:
            quarter = {3: 1, 6: 2, 9: 3, 12: 4}[end.month]
            return f"Q{quarter}_{end.year}", days
        return f"DUR_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}", days
    return "UNKNOWN_PERIOD", None


def _stable_digest(*parts: Any) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def _make_fact_identity(
    filing_id: str,
    source_file: str,
    ordinal: int,
    xml_fact_id: Optional[str],
    qname: str,
    context_id: Optional[str],
    unit_id: Optional[str],
    raw_text_lexical: str,
) -> Tuple[str, str, Optional[str]]:
    if xml_fact_id:
        evidence_id = f"{filing_id}::{source_file}#{xml_fact_id}"
        return xml_fact_id, evidence_id, f"{source_file}#{xml_fact_id}"
    digest = _stable_digest(source_file, ordinal, qname, context_id, unit_id, raw_text_lexical)[:12]
    synthetic_id = f"synthetic-{ordinal:06d}-{digest}"
    evidence_id = f"{filing_id}::{source_file}::ord={ordinal:06d}::{digest}"
    return synthetic_id, evidence_id, None


def parse_ixbrl_html(html_bytes: bytes, source_file: str, filing_id: str) -> Tuple[Dict[str, ContextRec], Dict[str, UnitRec], List[FactRec]]:
    """Parse iXBRL HTML into context, unit, and fact records.
    
    Parameters
    ----------
    html_bytes : bytes
        Raw HTML bytes to parse into filing facts and contexts.
    source_file : str
        Source filename recorded for parsed filing artifacts.
    filing_id : str
        Canonical filing identifier used by manifests and derived artifacts.
    
    Returns
    -------
    Tuple[Dict[str, ContextRec], Dict[str, UnitRec], List[FactRec]]
        Tuple with outputs produced while ixbrl html into context, unit, and fact records.
    
    Examples
    --------
    >>> result = parse_ixbrl_html(html_bytes=None, source_file='example-value', filing_id='0000320193-2025-10K')  # doctest: +SKIP
    >>> len(result)  # doctest: +SKIP
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.parse(io.BytesIO(html_bytes), parser).getroot()

    contexts: Dict[str, ContextRec] = {}
    for context_el in root.xpath("//*[local-name()='context']"):
        context_id = context_el.get("id") or context_el.get("{http://www.w3.org/XML/1998/namespace}id")
        if not context_id:
            continue

        identifier = ""
        for candidate in context_el.xpath(".//*[local-name()='entity']//*[local-name()='identifier']"):
            identifier = safe_text(candidate)
            break

        start = end = instant = None
        period_el = next(iter(context_el.xpath(".//*[local-name()='period']")), None)
        if period_el is not None:
            instant_el = next(iter(period_el.xpath(".//*[local-name()='instant']")), None)
            if instant_el is not None:
                instant = parse_date(safe_text(instant_el))
            else:
                start = parse_date(safe_text(next(iter(period_el.xpath(".//*[local-name()='startDate']")), None)))
                end = parse_date(safe_text(next(iter(period_el.xpath(".//*[local-name()='endDate']")), None)))

        dimensions: Dict[str, Any] = {}
        for member in context_el.xpath(".//*[local-name()='explicitMember']"):
            dimension = member.get("dimension") or member.get("{http://www.xbrl.org/2006/xbrldi}dimension")
            value = safe_text(member)
            if dimension and value:
                dimensions[dimension] = value
        for member in context_el.xpath(".//*[local-name()='typedMember']"):
            dimension = member.get("dimension") or member.get("{http://www.xbrl.org/2006/xbrldi}dimension")
            value = safe_text(member)
            if dimension and value:
                dimensions[dimension] = value

        period_key, days = period_key_from_context(start, end, instant)
        dimension_count = len(dimensions)
        contexts[context_id] = ContextRec(
            context_id=context_id,
            entity_identifier=identifier,
            start_date=start,
            end_date=end,
            instant_date=instant,
            period_key=period_key,
            days=days,
            dimensions=dimensions,
            dimension_count=dimension_count,
            is_consolidated=(dimension_count == 0),
        )

    units: Dict[str, UnitRec] = {}
    for unit_el in root.xpath("//*[local-name()='unit']"):
        unit_id = unit_el.get("id") or unit_el.get("{http://www.w3.org/XML/1998/namespace}id")
        if not unit_id:
            continue
        unit_obj: Dict[str, Any] = {}
        measures = [safe_text(measure) for measure in unit_el.xpath(".//*[local-name()='measure']")]
        measures = [measure for measure in measures if measure]
        if measures:
            unit_obj["measures"] = measures
        divide_el = next(iter(unit_el.xpath(".//*[local-name()='divide']")), None)
        if divide_el is not None:
            numerator = [safe_text(measure) for measure in divide_el.xpath(".//*[local-name()='unitNumerator']//*[local-name()='measure']")]
            denominator = [safe_text(measure) for measure in divide_el.xpath(".//*[local-name()='unitDenominator']//*[local-name()='measure']")]
            unit_obj["divide"] = {
                "numerator": [value for value in numerator if value],
                "denominator": [value for value in denominator if value],
            }
        units[unit_id] = UnitRec(unit_id=unit_id, unit_json=unit_obj)

    facts: List[FactRec] = []
    for ordinal, fact_el in enumerate(root.xpath("//*[local-name()='nonFraction' or local-name()='nonNumeric']"), start=1):
        local_name = fact_el.tag.split("}")[-1]
        xml_fact_id = fact_el.get("id") or fact_el.get("{http://www.w3.org/XML/1998/namespace}id")
        qname = fact_el.get("name") or ""
        concept_norm = norm_concept_for_linkbase(qname)
        context_id = fact_el.get("contextRef")
        unit_id = fact_el.get("unitRef")
        decimals = fact_el.get("decimals")
        scale = fact_el.get("scale")
        sign = fact_el.get("sign")
        fmt = fact_el.get("format")

        nil_value = fact_el.get("{http://www.w3.org/2001/XMLSchema-instance}nil")
        if nil_value and nil_value.lower() == "true":
            raw_text_lexical = ""
        else:
            raw_text_lexical = resolve_continued_text(root, fact_el) if local_name == "nonNumeric" else safe_text(fact_el)

        fact_id, fact_evidence_id, source_anchor = _make_fact_identity(
            filing_id=filing_id,
            source_file=source_file,
            ordinal=ordinal,
            xml_fact_id=xml_fact_id,
            qname=qname,
            context_id=context_id,
            unit_id=unit_id,
            raw_text_lexical=raw_text_lexical,
        )

        canonical_text, forced_value_type = apply_ix_transform(fmt, raw_text_lexical)
        period_key = "UNKNOWN_PERIOD"
        dimensions: Dict[str, Any] = {}
        is_consolidated = True
        if context_id and context_id in contexts:
            period_key = contexts[context_id].period_key
            dimensions = contexts[context_id].dimensions
            is_consolidated = contexts[context_id].is_consolidated

        if local_name == "nonFraction":
            reported = apply_sign(parse_numeric(canonical_text), sign)
            unscaled = apply_scale(reported, scale)
            facts.append(
                FactRec(
                    fact_id=fact_id,
                    fact_ordinal=ordinal,
                    fact_evidence_id=fact_evidence_id,
                    concept_qname=qname,
                    concept_norm=concept_norm,
                    context_id=context_id,
                    unit_id=unit_id,
                    decimals=decimals,
                    scale=scale,
                    sign=sign,
                    fmt=fmt,
                    value_type="numeric",
                    value_text=canonical_text or None,
                    value_lexical=raw_text_lexical or None,
                    value_num_reported=reported,
                    value_num_unscaled=unscaled,
                    period_key=period_key,
                    dimensions=dimensions,
                    is_consolidated=is_consolidated,
                    source_file=source_file,
                    source_anchor=source_anchor,
                )
            )
            continue

        value_text = canonical_text
        value_type = forced_value_type or "string"
        lowered_qname = qname.lower()
        if value_type == "string":
            if lowered_qname.endswith("flag") or lowered_qname.endswith("boolean"):
                value_type = "bool"
            elif lowered_qname.endswith("date"):
                value_type = "date"
            elif lowered_qname.endswith("duration"):
                value_type = "duration"
        if value_type == "bool" and value_text is not None:
            lowered = value_text.strip().lower()
            if lowered in {"☒", "x", "true", "t", "yes", "1"}:
                value_text = "true"
            elif lowered in {"☐", "false", "f", "no", "0"}:
                value_text = "false"
            else:
                value_text = lowered

        facts.append(
            FactRec(
                fact_id=fact_id,
                fact_ordinal=ordinal,
                fact_evidence_id=fact_evidence_id,
                concept_qname=qname,
                concept_norm=concept_norm,
                context_id=context_id,
                unit_id=unit_id,
                decimals=decimals,
                scale=scale,
                sign=sign,
                fmt=fmt,
                value_type=value_type,
                value_text=value_text or None,
                value_lexical=raw_text_lexical or None,
                value_num_reported=None,
                value_num_unscaled=None,
                period_key=period_key,
                dimensions=dimensions,
                is_consolidated=is_consolidated,
                source_file=source_file,
                source_anchor=source_anchor,
            )
        )

    return contexts, units, facts


def concept_from_href(href: str) -> Optional[str]:
    """Extract a normalized concept name from a linkbase href.
    
    Parameters
    ----------
    href : str
        Linkbase href reference to normalize into a concept name.
    
    Returns
    -------
    Optional[str]
        Value returned by this operation.
    
    Examples
    --------
    >>> result = concept_from_href(href='example-value')  # doctest: +SKIP
    """
    if not href or "#" not in href:
        return None
    return href.split("#", 1)[1].strip() or None


def parse_label_linkbase(xml_bytes: bytes) -> List[Tuple[str, str, str, str]]:
    """Parse label linkbase arcs into normalized relationship tuples.
    
    Parameters
    ----------
    xml_bytes : bytes
        Raw XML bytes to parse from linkbase or taxonomy files.
    
    Returns
    -------
    List[Tuple[str, str, str, str]]
        List of records for label linkbase arcs into normalized relationship tuples.
    
    Examples
    --------
    >>> result = parse_label_linkbase(xml_bytes=None)  # doctest: +SKIP
    >>> len(result)  # doctest: +SKIP
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.parse(io.BytesIO(xml_bytes), parser).getroot()

    loc_map: Dict[str, str] = {}
    for loc in root.xpath("//*[local-name()='loc']"):
        label = get_attr_any_ns(loc, "label")
        href = get_attr_any_ns(loc, "href")
        concept = concept_from_href(href) if href else None
        if label and concept:
            loc_map[label] = concept

    label_map: Dict[str, Tuple[str, str, str]] = {}
    for label_el in root.xpath("//*[local-name()='label']"):
        label_id = get_attr_any_ns(label_el, "label")
        if not label_id:
            continue
        role = get_attr_any_ns(label_el, "role") or ""
        lang = get_attr_any_ns(label_el, "lang") or label_el.get("{http://www.w3.org/XML/1998/namespace}lang") or ""
        label_map[label_id] = (role, lang, safe_text(label_el))

    rows = []
    for arc in root.xpath("//*[local-name()='labelArc']"):
        origin = get_attr_any_ns(arc, "from")
        target = get_attr_any_ns(arc, "to")
        concept = loc_map.get(origin)
        label_rec = label_map.get(target)
        if concept and label_rec:
            role, lang, text = label_rec
            rows.append((concept, role, lang, text))
    return rows


def parse_presentation_linkbase(xml_bytes: bytes) -> List[Tuple[str, str, str, float, str]]:
    """Parse presentation linkbase arcs into normalized tuples.
    
    Parameters
    ----------
    xml_bytes : bytes
        Raw XML bytes to parse from linkbase or taxonomy files.
    
    Returns
    -------
    List[Tuple[str, str, str, float, str]]
        List of records for presentation linkbase arcs into normalized tuples.
    
    Examples
    --------
    >>> result = parse_presentation_linkbase(xml_bytes=None)  # doctest: +SKIP
    >>> len(result)  # doctest: +SKIP
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.parse(io.BytesIO(xml_bytes), parser).getroot()
    edges = []
    for presentation_link in root.xpath("//*[local-name()='presentationLink']"):
        role = get_attr_any_ns(presentation_link, "role") or ""
        loc_map: Dict[str, str] = {}
        for loc in presentation_link.xpath(".//*[local-name()='loc']"):
            label = get_attr_any_ns(loc, "label")
            href = get_attr_any_ns(loc, "href")
            concept = concept_from_href(href) if href else None
            if label and concept:
                loc_map[label] = concept
        for arc in presentation_link.xpath(".//*[local-name()='presentationArc']"):
            origin = get_attr_any_ns(arc, "from")
            target = get_attr_any_ns(arc, "to")
            preferred = get_attr_any_ns(arc, "preferredLabel") or ""
            order_s = arc.get("order") or "0"
            try:
                order = float(order_s)
            except ValueError:
                order = 0.0
            if origin in loc_map and target in loc_map:
                edges.append((role, loc_map[origin], loc_map[target], order, preferred))
    return edges


def parse_calculation_linkbase(xml_bytes: bytes) -> List[Tuple[str, str, str, float, float]]:
    """Parse calculation linkbase arcs into normalized tuples.
    
    Parameters
    ----------
    xml_bytes : bytes
        Raw XML bytes to parse from linkbase or taxonomy files.
    
    Returns
    -------
    List[Tuple[str, str, str, float, float]]
        List of records for calculation linkbase arcs into normalized tuples.
    
    Examples
    --------
    >>> result = parse_calculation_linkbase(xml_bytes=None)  # doctest: +SKIP
    >>> len(result)  # doctest: +SKIP
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.parse(io.BytesIO(xml_bytes), parser).getroot()
    edges = []
    for calculation_link in root.xpath("//*[local-name()='calculationLink']"):
        role = get_attr_any_ns(calculation_link, "role") or ""
        loc_map: Dict[str, str] = {}
        for loc in calculation_link.xpath(".//*[local-name()='loc']"):
            label = get_attr_any_ns(loc, "label")
            href = get_attr_any_ns(loc, "href")
            concept = concept_from_href(href) if href else None
            if label and concept:
                loc_map[label] = concept
        for arc in calculation_link.xpath(".//*[local-name()='calculationArc']"):
            origin = get_attr_any_ns(arc, "from")
            target = get_attr_any_ns(arc, "to")
            weight_s = arc.get("weight") or "1"
            order_s = arc.get("order") or "0"
            try:
                weight = float(weight_s)
            except ValueError:
                weight = 1.0
            try:
                order = float(order_s)
            except ValueError:
                order = 0.0
            if origin in loc_map and target in loc_map:
                edges.append((role, loc_map[origin], loc_map[target], weight, order))
    return edges


def parse_definition_linkbase(xml_bytes: bytes) -> List[Tuple[str, str, str, str, float]]:
    """Parse definition linkbase arcs into normalized tuples.
    
    Parameters
    ----------
    xml_bytes : bytes
        Raw XML bytes to parse from linkbase or taxonomy files.
    
    Returns
    -------
    List[Tuple[str, str, str, str, float]]
        List of records for definition linkbase arcs into normalized tuples.
    
    Examples
    --------
    >>> result = parse_definition_linkbase(xml_bytes=None)  # doctest: +SKIP
    >>> len(result)  # doctest: +SKIP
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.parse(io.BytesIO(xml_bytes), parser).getroot()
    edges = []
    for definition_link in root.xpath("//*[local-name()='definitionLink']"):
        role = get_attr_any_ns(definition_link, "role") or ""
        loc_map: Dict[str, str] = {}
        for loc in definition_link.xpath(".//*[local-name()='loc']"):
            label = get_attr_any_ns(loc, "label")
            href = get_attr_any_ns(loc, "href")
            concept = concept_from_href(href) if href else None
            if label and concept:
                loc_map[label] = concept
        for arc in definition_link.xpath(".//*[local-name()='definitionArc']"):
            origin = get_attr_any_ns(arc, "from")
            target = get_attr_any_ns(arc, "to")
            arcrole = get_attr_any_ns(arc, "arcrole") or ""
            order_s = arc.get("order") or "0"
            try:
                order = float(order_s)
            except ValueError:
                order = 0.0
            if origin in loc_map and target in loc_map:
                edges.append((role, arcrole, loc_map[origin], loc_map[target], order))
    return edges


def load_zip_members(zip_path: str) -> Dict[str, bytes]:
    """Load all members from a zip package into memory.
    
    Parameters
    ----------
    zip_path : str
        Path to a filing ZIP package containing iXBRL artifacts.
    
    Returns
    -------
    Dict[str, bytes]
        Dictionary with fields produced while load all members from a zip package into memory.
    
    Examples
    --------
    >>> result = load_zip_members(zip_path='/tmp/file.txt')  # doctest: +SKIP
    >>> sorted(result.keys())[:3]  # doctest: +SKIP
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        return {info.filename: zf.read(info.filename) for info in zf.infolist()}


def _root_level_html_members(members: Dict[str, bytes]) -> List[str]:
    return sorted(
        name
        for name in members
        if "/" not in name and name.lower().endswith((".htm", ".html"))
    )


def _normalize_preferred_main_html(candidates: Sequence[str], preferred_name: Optional[str]) -> Optional[str]:
    if not preferred_name:
        return None

    candidate_map = {candidate.lower(): candidate for candidate in candidates}
    preferred = candidate_map.get(preferred_name.lower())
    if not preferred:
        return None

    stem, ext = os.path.splitext(os.path.basename(preferred))
    match = re.match(r"^(?P<base>.+)_d\d+$", stem, flags=re.IGNORECASE)
    if match:
        base_name = f"{match.group('base')}{ext}".lower()
        if base_name in candidate_map:
            return candidate_map[base_name]
    return preferred


def pick_main_html(members: Dict[str, bytes], preferred_name: Optional[str] = None) -> str:
    """Select the primary HTML member from package zip contents.
    
    Parameters
    ----------
    members : Dict[str, bytes]
        ZIP member mapping from member name to raw bytes.
    preferred_name : Optional[str], optional
        Preferred HTML member name when multiple candidates exist.
    
    Returns
    -------
    str
        String/path value for select the primary html member from package zip contents.
    
    Raises
    ------
    RuntimeError
        No root-level .htm/.html found in zip (expected iXBRL HTML).
    
    Examples
    --------
    >>> result = pick_main_html(members={})  # doctest: +SKIP
    """
    candidates = _root_level_html_members(members)
    if not candidates:
        raise RuntimeError("No root-level .htm/.html found in zip (expected iXBRL HTML).")

    preferred_candidate = _normalize_preferred_main_html(candidates, preferred_name)

    xsd_stems = {
        os.path.splitext(os.path.basename(name))[0].lower()
        for name in members
        if "/" not in name and name.lower().endswith(".xsd")
    }
    exact_matches = [
        candidate
        for candidate in candidates
        if os.path.splitext(os.path.basename(candidate))[0].lower() in xsd_stems
    ]
    if exact_matches:
        if preferred_candidate:
            preferred_stem = os.path.splitext(os.path.basename(preferred_candidate))[0].lower()
            for candidate in exact_matches:
                if os.path.splitext(os.path.basename(candidate))[0].lower() == preferred_stem:
                    return candidate
        exact_matches.sort(key=lambda name: (len(name), name.lower()))
        return exact_matches[0]

    if preferred_candidate:
        return preferred_candidate

    ranked = sorted(
        ((len(members[name]), name) for name in candidates),
        reverse=True,
    )
    return ranked[0][1]


def pick_ixbrl_html_parts(members: Dict[str, bytes], main_html: str) -> List[str]:
    """Select HTML parts to parse for inline XBRL facts.
    
    Parameters
    ----------
    members : Dict[str, bytes]
        ZIP member mapping from member name to raw bytes.
    main_html : str
        Primary HTML member name selected within a filing package.
    
    Returns
    -------
    List[str]
        List of output records for this operation.
    
    Examples
    --------
    >>> result = pick_ixbrl_html_parts(members={}, main_html='example-value')  # doctest: +SKIP
    >>> len(result)  # doctest: +SKIP
    """
    candidates = _root_level_html_members(members)
    main_stem = os.path.splitext(os.path.basename(main_html))[0]
    part_pattern = re.compile(rf"^{re.escape(main_stem)}_d\d+$", flags=re.IGNORECASE)

    parts = []
    for candidate in candidates:
        stem = os.path.splitext(os.path.basename(candidate))[0]
        if candidate == main_html or part_pattern.fullmatch(stem):
            parts.append(candidate)

    if main_html not in parts:
        parts.insert(0, main_html)

    parts = sorted(dict.fromkeys(parts), key=lambda name: (0 if name == main_html else 1, name.lower()))
    return parts


def parse_ixbrl_package(
    members: Dict[str, bytes],
    filing_id: str,
    html_members: Sequence[str],
) -> Tuple[Dict[str, ContextRec], Dict[str, UnitRec], List[FactRec]]:
    """Parse an iXBRL package zip into normalized pipeline artifacts.
    
    Parameters
    ----------
    members : Dict[str, bytes]
        ZIP member mapping from member name to raw bytes.
    filing_id : str
        Canonical filing identifier used by manifests and derived artifacts.
    html_members : Sequence[str]
        Candidate HTML member names discovered in the package.
    
    Returns
    -------
    Tuple[Dict[str, ContextRec], Dict[str, UnitRec], List[FactRec]]
        Tuple containing outputs for an ixbrl package zip into normalized pipeline artifacts.
    
    Examples
    --------
    >>> result = parse_ixbrl_package(members={}, filing_id='0000320193-2025-10K', html_members=[])  # doctest: +SKIP
    >>> len(result)  # doctest: +SKIP
    """
    contexts: Dict[str, ContextRec] = {}
    units: Dict[str, UnitRec] = {}
    facts: List[FactRec] = []

    for html_name in html_members:
        part_contexts, part_units, part_facts = parse_ixbrl_html(members[html_name], html_name, filing_id)
        contexts.update(part_contexts)
        units.update(part_units)
        facts.extend(part_facts)

    return contexts, units, facts


def infer_filing_id(zip_path: str, html_main: str) -> str:
    """Infer a stable filing identifier from package metadata.
    
    Parameters
    ----------
    zip_path : str
        Path to a filing ZIP package containing iXBRL artifacts.
    html_main : str
        Selected primary HTML member name used for metadata inference.
    
    Returns
    -------
    str
        String/path value produced while infer a stable filing identifier from package metadata.
    
    Examples
    --------
    >>> result = infer_filing_id(zip_path='/tmp/file.txt', html_main='example-value')  # doctest: +SKIP
    """
    base = os.path.basename(zip_path)
    match = re.match(r"^(\d{10}-\d{2}-\d{6})-xbrl\.zip$", base, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return os.path.splitext(os.path.basename(html_main))[0]


def infer_ticker(html_main: str) -> Optional[str]:
    """Infer a ticker symbol from the filing main HTML path.
    
    Parameters
    ----------
    html_main : str
        Selected primary HTML member name used for metadata inference.
    
    Returns
    -------
    Optional[str]
        Value returned by this operation.
    
    Examples
    --------
    >>> result = infer_ticker(html_main='example-value')  # doctest: +SKIP
    """
    stem = os.path.splitext(os.path.basename(html_main))[0]
    if "-" in stem:
        left = stem.split("-", 1)[0]
        if 1 <= len(left) <= 8 and left.isalnum():
            return left.upper()
    return None


def _best_fact_value(facts: Sequence[FactRec], aliases: Sequence[str]) -> Optional[str]:
    alias_set = set(aliases)
    filtered = [fact for fact in facts if fact.concept_norm in alias_set and fact.value_text]
    if not filtered:
        return None
    filtered.sort(key=lambda fact: (0 if fact.is_consolidated else 1, 0 if fact.source_anchor else 1, fact.fact_ordinal))
    return filtered[0].value_text


def _best_date_value(facts: Sequence[FactRec], aliases: Sequence[str]) -> Optional[date]:
    value = _best_fact_value(facts, aliases)
    if not value:
        return None
    return parse_date(value)


def extract_filing_metadata(
    filing_id: str,
    zip_name: str,
    main_html: str,
    inferred_ticker: Optional[str],
    facts: Sequence[FactRec],
    overrides: Optional[Dict[str, Any]] = None,
) -> FilingMetadata:
    """Extract filing-level metadata from parsed facts and contexts.
    
    Parameters
    ----------
    filing_id : str
        Canonical filing identifier used by manifests and derived artifacts.
    zip_name : str
        ZIP filename used as fallback metadata source.
    main_html : str
        Primary HTML member name selected within a filing package.
    inferred_ticker : Optional[str]
        Ticker inferred from package naming conventions.
    facts : Sequence[FactRec]
        Parsed fact records extracted from iXBRL content.
    overrides : Optional[Dict[str, Any]], optional
        Optional metadata override mapping applied during extraction.
    
    Returns
    -------
    FilingMetadata
        Value returned by this operation.
    
    Examples
    --------
    >>> result = extract_filing_metadata(filing_id='0000320193-2025-10K', zip_name='example-value', main_html='example-value')  # doctest: +SKIP
    """
    overrides = overrides or {}
    form_type = overrides.get("form_type") or _best_fact_value(facts, ["dei_DocumentType"])

    fiscal_year_focus = overrides.get("fiscal_year_focus")
    if fiscal_year_focus is None:
        raw_fy = _best_fact_value(facts, ["dei_DocumentFiscalYearFocus"])
        if raw_fy and raw_fy.isdigit():
            fiscal_year_focus = int(raw_fy)

    fiscal_period_focus = overrides.get("fiscal_period_focus") or _best_fact_value(
        facts, ["dei_DocumentFiscalPeriodFocus"]
    )
    report_date = overrides.get("report_date") or _best_date_value(facts, ["dei_DocumentPeriodEndDate"])
    if isinstance(report_date, str):
        report_date = parse_date(report_date)

    ticker = overrides.get("ticker") or _best_fact_value(facts, ["dei_TradingSymbol"]) or inferred_ticker

    return FilingMetadata(
        filing_id=filing_id,
        ticker=ticker,
        zip_name=zip_name,
        main_html=main_html,
        processed_at=datetime.now(timezone.utc).isoformat(),
        form_type=form_type,
        fiscal_year_focus=fiscal_year_focus,
        fiscal_period_focus=fiscal_period_focus,
        report_date=report_date,
    )


def _insert_raw_tables(
    conn: sqlite3.Connection,
    filing: FilingMetadata,
    contexts: Dict[str, ContextRec],
    units: Dict[str, UnitRec],
    facts: Sequence[FactRec],
    labels: Sequence[Tuple[str, str, str, str]],
    pre_edges: Sequence[Tuple[str, str, str, float, str]],
    cal_edges: Sequence[Tuple[str, str, str, float, float]],
    def_edges: Sequence[Tuple[str, str, str, str, float]],
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO filings(
          filing_id, ticker, zip_name, main_html, processed_at,
          form_type, fiscal_year_focus, fiscal_period_focus, report_date
        ) VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (
            filing.filing_id,
            filing.ticker,
            filing.zip_name,
            filing.main_html,
            filing.processed_at,
            filing.form_type,
            filing.fiscal_year_focus,
            filing.fiscal_period_focus,
            filing.report_date.isoformat() if filing.report_date else None,
        ),
    )

    conn.executemany(
        """
        INSERT OR REPLACE INTO contexts(
          filing_id, context_id, entity_identifier,
          start_date, end_date, instant_date,
          period_key, days, dimensions_json, dimension_count, is_consolidated
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                filing.filing_id,
                context.context_id,
                context.entity_identifier,
                context.start_date.isoformat() if context.start_date else None,
                context.end_date.isoformat() if context.end_date else None,
                context.instant_date.isoformat() if context.instant_date else None,
                context.period_key,
                context.days,
                _json_dumps(context.dimensions),
                context.dimension_count,
                1 if context.is_consolidated else 0,
            )
            for context in contexts.values()
        ],
    )

    conn.executemany(
        "INSERT OR REPLACE INTO units(filing_id, unit_id, unit_json) VALUES (?,?,?)",
        [(filing.filing_id, unit.unit_id, _json_dumps(unit.unit_json)) for unit in units.values()],
    )

    conn.executemany(
        """
        INSERT OR REPLACE INTO facts(
          filing_id, fact_id, fact_ordinal, fact_evidence_id,
          concept_qname, concept_norm, context_id, unit_id,
          decimals, scale, sign, fmt,
          value_type, value_text, value_lexical,
          value_num_reported, value_num_unscaled,
          period_key, dimensions_json, is_consolidated,
          source_file, source_anchor
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                filing.filing_id,
                fact.fact_id,
                fact.fact_ordinal,
                fact.fact_evidence_id,
                fact.concept_qname,
                fact.concept_norm,
                fact.context_id,
                fact.unit_id,
                fact.decimals,
                fact.scale,
                fact.sign,
                fact.fmt,
                fact.value_type,
                fact.value_text,
                fact.value_lexical,
                str(fact.value_num_reported) if fact.value_num_reported is not None else None,
                str(fact.value_num_unscaled) if fact.value_num_unscaled is not None else None,
                fact.period_key,
                _json_dumps(fact.dimensions),
                1 if fact.is_consolidated else 0,
                fact.source_file,
                fact.source_anchor,
            )
            for fact in facts
        ],
    )

    if labels:
        conn.executemany(
            """
            INSERT OR IGNORE INTO labels(
              filing_id, concept_norm, label_role, label_lang, label_text
            ) VALUES (?,?,?,?,?)
            """,
            [(filing.filing_id, concept, role, lang, text) for concept, role, lang, text in labels],
        )

    if pre_edges:
        conn.executemany(
            "INSERT INTO pre_edges(filing_id, role, parent_concept_norm, child_concept_norm, ord, preferred_label) VALUES (?,?,?,?,?,?)",
            [(filing.filing_id, role, parent, child, order, preferred) for role, parent, child, order, preferred in pre_edges],
        )

    if cal_edges:
        conn.executemany(
            "INSERT INTO cal_edges(filing_id, role, parent_concept_norm, child_concept_norm, weight, ord) VALUES (?,?,?,?,?,?)",
            [(filing.filing_id, role, parent, child, weight, order) for role, parent, child, weight, order in cal_edges],
        )

    if def_edges:
        conn.executemany(
            "INSERT INTO def_edges(filing_id, role, arcrole, parent_concept_norm, child_concept_norm, ord) VALUES (?,?,?,?,?,?)",
            [(filing.filing_id, role, arcrole, parent, child, order) for role, arcrole, parent, child, order in def_edges],
        )

    conn.commit()


def _store_narrative(
    conn: sqlite3.Connection,
    filing_id: str,
    source_file: str,
    html_bytes: bytes,
    chunk_chars: int = 3500,
    overlap: int = 450,
) -> Tuple[Sequence[Section], Sequence[NarrativeChunk]]:
    html = html_bytes.decode("utf-8", errors="replace")
    text = extract_narrative_text(html)
    sections, chunks = build_narrative_chunks(text, chunk_chars=chunk_chars, overlap=overlap)

    conn.execute("DELETE FROM narrative_sections WHERE filing_id=? AND source_file=?", (filing_id, source_file))
    conn.execute("DELETE FROM narrative_chunks WHERE filing_id=? AND source_file=?", (filing_id, source_file))

    if sections:
        conn.executemany(
            """
            INSERT OR REPLACE INTO narrative_sections(
              filing_id, source_file, item, heading, start_char, end_char
            ) VALUES (?,?,?,?,?,?)
            """,
            [(filing_id, source_file, section.item, section.heading, section.start, section.end) for section in sections],
        )

    if chunks:
        conn.executemany(
            """
            INSERT OR REPLACE INTO narrative_chunks(
              filing_id, source_file, item, heading, subheading, chunk_index, char_start, char_end,
              heading_path, retrieval_text, text_raw, text_masked, text_sha1
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    filing_id,
                    source_file,
                    chunk.item,
                    chunk.heading,
                    chunk.subheading,
                    chunk.chunk_index,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.heading_path,
                    chunk.retrieval_text,
                    chunk.text_raw,
                    chunk.text_masked,
                    chunk.text_sha1,
                )
                for chunk in chunks
            ],
        )
    conn.commit()
    return sections, chunks


def process_zip(
    zip_path: str,
    out_db: str,
    ticker: Optional[str] = None,
    extract_narrative: bool = False,
    reset_db: bool = False,
    metadata_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ingest (process and store) one filing zip into the corpus database.
    
    Parameters
    ----------
    zip_path : str
        Path to a filing ZIP package containing iXBRL artifacts.
    out_db : str
        Destination path for the output SQLite database.
    ticker : Optional[str], optional
        Issuer ticker symbol used in manifests and derived outputs.
    extract_narrative : bool, optional
        Whether narrative chunks should be extracted during ingestion.
    reset_db : bool, optional
        Whether to reset database tables before ingestion.
    metadata_overrides : Optional[Dict[str, Any]], optional
        Optional filing metadata values that override inferred package metadata.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with fields produced while and process one filing zip into the corpus database.
    """
    metadata_overrides = dict(metadata_overrides or {})
    if ticker:
        metadata_overrides["ticker"] = ticker

    members = load_zip_members(zip_path)
    preferred_main_html = metadata_overrides.get("preferred_main_html")
    html_main = pick_main_html(members, preferred_name=preferred_main_html)
    html_parts = pick_ixbrl_html_parts(members, html_main)
    filing_id = infer_filing_id(zip_path, html_main)
    inferred_ticker = infer_ticker(html_main)

    contexts, units, facts = parse_ixbrl_package(members, filing_id, html_parts)
    labels = []
    pre_edges = []
    cal_edges = []
    def_edges = []

    lab_name = next((name for name in members if name.lower().endswith("_lab.xml")), None)
    pre_name = next((name for name in members if name.lower().endswith("_pre.xml")), None)
    cal_name = next((name for name in members if name.lower().endswith("_cal.xml")), None)
    def_name = next((name for name in members if name.lower().endswith("_def.xml")), None)
    if lab_name:
        labels = parse_label_linkbase(members[lab_name])
    if pre_name:
        pre_edges = parse_presentation_linkbase(members[pre_name])
    if cal_name:
        cal_edges = parse_calculation_linkbase(members[cal_name])
    if def_name:
        def_edges = parse_definition_linkbase(members[def_name])

    filing = extract_filing_metadata(
        filing_id=filing_id,
        zip_name=os.path.basename(zip_path),
        main_html=html_main,
        inferred_ticker=inferred_ticker,
        facts=facts,
        overrides=metadata_overrides,
    )

    conn = connect_db(out_db)
    try:
        ensure_schema(conn, reset=reset_db)
        _insert_raw_tables(
            conn=conn,
            filing=filing,
            contexts=contexts,
            units=units,
            facts=facts,
            labels=labels,
            pre_edges=pre_edges,
            cal_edges=cal_edges,
            def_edges=def_edges,
        )

        sections = chunks = ()
        if extract_narrative:
            sections, chunks = _store_narrative(conn, filing_id, html_main, members[html_main])

        rebuild_canonical_layers(conn, filing_id)
        return {
            "db": out_db,
            "filing_id": filing_id,
            "ticker": filing.ticker,
            "main_html": html_main,
            "facts": len(facts),
            "contexts": len(contexts),
            "units": len(units),
            "labels": len(labels),
            "pre_edges": len(pre_edges),
            "cal_edges": len(cal_edges),
            "def_edges": len(def_edges),
            "narrative_sections": len(sections),
            "narrative_chunks": len(chunks),
        }
    finally:
        conn.close()


def _quarter_from_focus(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    normalized = value.upper()
    return {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 4}.get(normalized)


def _quarter_from_end(end: date) -> Optional[int]:
    if (end.month, end.day) in {(3, 31), (6, 30), (9, 30), (12, 31)}:
        return {3: 1, 6: 2, 9: 3, 12: 4}[end.month]
    return None


def filing_period_key(metadata: FilingMetadata) -> str:
    """Build a canonical period key for filing metadata.
    
    Parameters
    ----------
    metadata : FilingMetadata
        Filing metadata record used for period classification and normalization.
    
    Returns
    -------
    str
        Text or path value produced by this function.
    
    Examples
    --------
    >>> result = filing_period_key(metadata=None)  # doctest: +SKIP
    """
    if metadata.fiscal_year_focus and metadata.fiscal_period_focus:
        period = metadata.fiscal_period_focus.upper()
        if period == "FY":
            return f"FY{metadata.fiscal_year_focus}"
        if period in {"Q1", "Q2", "Q3", "Q4"}:
            return f"{period}_{metadata.fiscal_year_focus}"
    if metadata.report_date:
        return f"ASOF_{metadata.report_date.strftime('%Y%m%d')}"
    return "UNKNOWN_PERIOD"


def _add_months_preserve_eom(value: date, months: int) -> date:
    month_index = (value.month - 1) + months
    year = value.year + (month_index // 12)
    month = (month_index % 12) + 1
    source_last_day = calendar.monthrange(value.year, value.month)[1]
    target_last_day = calendar.monthrange(year, month)[1]
    if value.day == source_last_day:
        day = target_last_day
    else:
        day = min(value.day, target_last_day)
    return date(year, month, day)


def _infer_fiscal_year_end_date(metadata: FilingMetadata) -> Optional[date]:
    if not metadata.report_date or not metadata.fiscal_period_focus:
        return None
    offsets = {"FY": 0, "Q4": 0, "Q3": 3, "Q2": 6, "Q1": 9}
    focus = metadata.fiscal_period_focus.upper()
    if focus not in offsets:
        return None
    return _add_months_preserve_eom(metadata.report_date, offsets[focus])


def _infer_fiscal_year_for_date(as_of_date: date, metadata: FilingMetadata) -> Optional[int]:
    fy_end = _infer_fiscal_year_end_date(metadata)
    if fy_end is None:
        return None
    fy_end_md = (fy_end.month, fy_end.day)
    return as_of_date.year if (as_of_date.month, as_of_date.day) <= fy_end_md else as_of_date.year + 1


def _infer_historical_fiscal_year_for_date(as_of_date: date, metadata: FilingMetadata) -> Optional[int]:
    if metadata.report_date is None or metadata.fiscal_year_focus is None:
        return None
    if as_of_date > metadata.report_date:
        return None
    delta_days = (metadata.report_date - as_of_date).days
    return metadata.fiscal_year_focus - int(round(delta_days / 364.25))


def _infer_fiscal_quarter_for_date(as_of_date: date, metadata: FilingMetadata, fiscal_year: Optional[int]) -> Optional[int]:
    fy_end = _infer_fiscal_year_end_date(metadata)
    if fy_end is None or fiscal_year is None:
        return None
    q4 = date(fiscal_year, fy_end.month, min(fy_end.day, calendar.monthrange(fiscal_year, fy_end.month)[1]))
    quarter_ends = {
        1: _add_months_preserve_eom(q4, -9),
        2: _add_months_preserve_eom(q4, -6),
        3: _add_months_preserve_eom(q4, -3),
        4: q4,
    }
    for quarter, quarter_end in quarter_ends.items():
        if as_of_date == quarter_end:
            return quarter
    return None


def _classify_period(
    metadata: FilingMetadata,
    start_date: Optional[date],
    end_date: Optional[date],
    instant_date: Optional[date],
) -> Dict[str, Any]:
    reference_date = instant_date or end_date
    historical_fiscal_year = _infer_historical_fiscal_year_for_date(reference_date, metadata) if reference_date else None
    inferred_fiscal_year = historical_fiscal_year or (
        _infer_fiscal_year_for_date(reference_date, metadata) if reference_date else None
    )
    inferred_fiscal_quarter = (
        _infer_fiscal_quarter_for_date(reference_date, metadata, inferred_fiscal_year) if reference_date else None
    )

    if instant_date:
        quarter = inferred_fiscal_quarter or _quarter_from_focus(metadata.fiscal_period_focus) or _quarter_from_end(instant_date)
        fiscal_year = inferred_fiscal_year or instant_date.year
        return {
            "period_type": "ASOF",
            "period_key": f"ASOF_{instant_date.strftime('%Y%m%d')}",
            "period_start": instant_date.isoformat(),
            "period_end": instant_date.isoformat(),
            "fiscal_year": fiscal_year,
            "fiscal_quarter": quarter,
            "is_ytd": 0,
        }

    if not start_date or not end_date:
        return {
            "period_type": "DUR_OTHER",
            "period_key": "UNKNOWN_PERIOD",
            "period_start": start_date.isoformat() if start_date else None,
            "period_end": end_date.isoformat() if end_date else None,
            "fiscal_year": end_date.year if end_date else None,
            "fiscal_quarter": None,
            "is_ytd": 0,
        }

    days = (end_date - start_date).days
    fiscal_year = inferred_fiscal_year or end_date.year
    focus_quarter = _quarter_from_focus(metadata.fiscal_period_focus)
    end_matches_report = bool(
        metadata.report_date and (end_date.month, end_date.day) == (metadata.report_date.month, metadata.report_date.day)
    )
    quarter = inferred_fiscal_quarter or (focus_quarter if end_matches_report and focus_quarter else _quarter_from_end(end_date))

    if 330 <= days <= 370:
        return {
            "period_type": "FY",
            "period_key": f"FY{fiscal_year}",
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "fiscal_year": fiscal_year,
            "fiscal_quarter": 4,
            "is_ytd": 0,
        }
    if 80 <= days <= 110:
        if quarter is None:
            quarter = 1
        return {
            "period_type": "Q",
            "period_key": f"Q{quarter}_{fiscal_year}",
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "fiscal_year": fiscal_year,
            "fiscal_quarter": quarter,
            "is_ytd": 0,
        }
    if 170 <= days <= 205 or 260 <= days <= 300:
        if quarter is None:
            quarter = 2 if days <= 205 else 3
        return {
            "period_type": "YTD",
            "period_key": f"YTD_Q{quarter}_{fiscal_year}",
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "fiscal_year": fiscal_year,
            "fiscal_quarter": quarter,
            "is_ytd": 1,
        }
    return {
        "period_type": "DUR_OTHER",
        "period_key": f"DUR_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
        "period_start": start_date.isoformat(),
        "period_end": end_date.isoformat(),
        "fiscal_year": fiscal_year,
        "fiscal_quarter": quarter,
        "is_ytd": 0,
    }


def normalize_unit_family(unit_json_text: Optional[str]) -> Optional[str]:
    """Normalize a unit JSON payload to a canonical unit family.
    
    Parameters
    ----------
    unit_json_text : Optional[str]
        Serialized unit payload to normalize into a unit family.
    
    Returns
    -------
    Optional[str]
        Value returned by this operation.
    
    Examples
    --------
    >>> result = normalize_unit_family(unit_json_text='example-value')  # doctest: +SKIP
    """
    if not unit_json_text:
        return None
    try:
        unit_json = json.loads(unit_json_text)
    except json.JSONDecodeError:
        return "other"

    def _norm(values: Sequence[str]) -> List[str]:
        return [value.split(":")[-1].lower() for value in values if value]

    measures = _norm(unit_json.get("measures", []))
    divide = unit_json.get("divide") or {}
    numerator = _norm(divide.get("numerator", []))
    denominator = _norm(divide.get("denominator", []))

    if numerator or denominator:
        if numerator == ["usd"] and denominator in (["shares"], ["share"]):
            return "per_share"
        if numerator == ["usd"] and denominator == ["pure"]:
            return "usd"
        return "other"

    if measures in (["usd"], ["usdollar"]):
        return "usd"
    if measures in (["shares"], ["share"]):
        return "shares"
    if measures == ["pure"]:
        return "pure"
    if len(set(measures)) == 1 and measures:
        return measures[0]
    return "other"


def _fetch_filing(conn: sqlite3.Connection, filing_id: str) -> FilingMetadata:
    row = conn.execute("SELECT * FROM filings WHERE filing_id=?", (filing_id,)).fetchone()
    if row is None:
        raise KeyError(f"Unknown filing_id: {filing_id}")
    return FilingMetadata(
        filing_id=row["filing_id"],
        ticker=row["ticker"],
        zip_name=row["zip_name"],
        main_html=row["main_html"],
        processed_at=row["processed_at"],
        form_type=row["form_type"],
        fiscal_year_focus=row["fiscal_year_focus"],
        fiscal_period_focus=row["fiscal_period_focus"],
        report_date=parse_date(row["report_date"]) if row["report_date"] else None,
    )


def _build_candidate(row: sqlite3.Row, filing: FilingMetadata) -> Dict[str, Any]:
    start = parse_date(row["start_date"]) if row["start_date"] else None
    end = parse_date(row["end_date"]) if row["end_date"] else None
    instant = parse_date(row["instant_date"]) if row["instant_date"] else None
    period = _classify_period(filing, start, end, instant)
    unit_canon = normalize_unit_family(row["unit_json"])
    raw_fact_ref = row["source_anchor"] or f"{row['source_file']}::ord={row['fact_ordinal']:06d}"
    return {
        "filing_id": filing.filing_id,
        "ticker": filing.ticker,
        "fact_evidence_id": row["fact_evidence_id"],
        "raw_fact_ref": raw_fact_ref,
        "fact_id": row["fact_id"],
        "fact_ordinal": row["fact_ordinal"],
        "concept_norm": row["concept_norm"],
        "value_type": row["value_type"],
        "unit_canon": unit_canon,
        "value_num_exact": row["value_num_unscaled"],
        "value_text": row["value_text"],
        "source_anchor": row["source_anchor"],
        "context_id": row["context_id"],
        "dimension_count": row["dimension_count"] if row["dimension_count"] is not None else 0,
        "start_date": row["start_date"],
        "end_date": row["end_date"],
        "instant_date": row["instant_date"],
        **period,
    }


def _validation_key(filing_id: str, code: str, entity_kind: str, entity_key: str, details: Dict[str, Any]) -> str:
    return _stable_digest(filing_id, code, entity_kind, entity_key, _json_dumps(details))


def _add_validator(
    bucket: Dict[str, ValidationRecord],
    filing_id: str,
    code: str,
    entity_kind: str,
    entity_key: str,
    message: str,
    details: Dict[str, Any],
) -> None:
    key = _validation_key(filing_id, code, entity_kind, entity_key, details)
    bucket[key] = ValidationRecord(code, entity_kind, entity_key, message, details)


def _selection_reason(candidate: Dict[str, Any]) -> str:
    return "; ".join(
        [
            "dimensionless-first",
            "anchored" if candidate["source_anchor"] else "synthetic-locator",
            "context-bound" if candidate["context_id"] else "context-missing",
        ]
    )


def _build_facts_canon_and_validators(conn: sqlite3.Connection, filing: FilingMetadata) -> Tuple[List[Dict[str, Any]], List[ValidationRecord]]:
    rows = conn.execute(
        """
        SELECT
          f.*,
          c.start_date, c.end_date, c.instant_date, c.dimension_count,
          u.unit_json
        FROM facts f
        LEFT JOIN contexts c
          ON c.filing_id = f.filing_id AND c.context_id = f.context_id
        LEFT JOIN units u
          ON u.filing_id = f.filing_id AND u.unit_id = f.unit_id
        WHERE f.filing_id=?
        ORDER BY f.fact_ordinal
        """,
        (filing.filing_id,),
    ).fetchall()

    validators: Dict[str, ValidationRecord] = {}
    candidates = [_build_candidate(row, filing) for row in rows if row["concept_norm"]]

    for candidate in candidates:
        if not candidate["fact_evidence_id"]:
            _add_validator(
                validators,
                filing.filing_id,
                "MISSING_EVIDENCE_ID",
                "fact",
                candidate["raw_fact_ref"],
                "Raw fact is missing a deterministic evidence identifier.",
                {"concept_norm": candidate["concept_norm"]},
            )
        if candidate["period_type"] == "DUR_OTHER" or candidate["period_key"] == "UNKNOWN_PERIOD":
            _add_validator(
                validators,
                filing.filing_id,
                "PERIOD_ALIGNMENT_FAIL",
                "context",
                candidate["context_id"] or candidate["raw_fact_ref"],
                "Context could not be classified into FY/Q/YTD/ASOF.",
                {
                    "concept_norm": candidate["concept_norm"],
                    "period_start": candidate["period_start"],
                    "period_end": candidate["period_end"],
                },
            )

    grouped_numeric: Dict[Tuple[str, str, Optional[str]], set[str]] = defaultdict(set)
    for candidate in candidates:
        if candidate["dimension_count"] == 0 and candidate["value_type"] == "numeric" and candidate["value_num_exact"] is not None:
            grouped_numeric[(candidate["concept_norm"], candidate["period_key"], candidate["unit_canon"])].add(
                str(candidate["value_num_exact"])
            )
    for (concept_norm, period_key, unit_canon), values in grouped_numeric.items():
        if len(values) > 1:
            _add_validator(
                validators,
                filing.filing_id,
                "UNIT_SCALE_MISMATCH",
                "fact_group",
                f"{concept_norm}|{period_key}|{unit_canon or 'none'}",
                "Normalized numeric values disagree inside the same concept/period/unit family.",
                {
                    "concept_norm": concept_norm,
                    "period_key": period_key,
                    "unit_canon": unit_canon,
                    "values": sorted(values),
                },
            )

    groups: Dict[Tuple[str, str, str, Optional[str]], List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        key = (
            candidate["concept_norm"],
            candidate["value_type"],
            candidate["period_key"],
            candidate["unit_canon"],
        )
        groups[key].append(candidate)

    canon_rows: List[Dict[str, Any]] = []
    for key, group in groups.items():
        dimensionless = [candidate for candidate in group if candidate["dimension_count"] == 0]
        if not dimensionless:
            _add_validator(
                validators,
                filing.filing_id,
                "CONTEXT_SELECTION_AMBIGUOUS",
                "fact_group",
                "|".join("" if part is None else str(part) for part in key),
                "No dimensionless candidate exists for this concept-period-unit group.",
                {"group": key, "raw_fact_refs": [candidate["raw_fact_ref"] for candidate in group]},
            )
            continue

        scored: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        for candidate in dimensionless:
            business_score = (0 if candidate["source_anchor"] else 1, 0 if candidate["context_id"] else 1)
            scored[business_score].append(candidate)
        top_score = sorted(scored)[0]
        top_candidates = scored[top_score]
        if len(top_candidates) != 1:
            signatures = {
                (
                    candidate["context_id"],
                    candidate["unit_canon"],
                    str(candidate["value_num_exact"]) if candidate["value_num_exact"] is not None else None,
                    candidate["value_text"],
                )
                for candidate in top_candidates
            }
            if len(signatures) == 1:
                top_candidates = sorted(
                    top_candidates,
                    key=lambda candidate: (candidate["source_anchor"] or "", candidate["fact_evidence_id"]),
                )
                chosen = top_candidates[0]
            else:
                _add_validator(
                    validators,
                    filing.filing_id,
                    "CONTEXT_SELECTION_AMBIGUOUS",
                    "fact_group",
                    "|".join("" if part is None else str(part) for part in key),
                    "Multiple top-ranked dimensionless candidates remain after deterministic selection rules.",
                    {"group": key, "raw_fact_refs": [candidate["raw_fact_ref"] for candidate in top_candidates]},
                )
                continue
        else:
            chosen = top_candidates[0]
        canon_rows.append(
            {
                "canon_id": _stable_digest(
                    filing.filing_id,
                    chosen["concept_norm"],
                    chosen["period_key"],
                    chosen["value_type"],
                    chosen["unit_canon"],
                ),
                "filing_id": filing.filing_id,
                "ticker": filing.ticker,
                "fact_evidence_id": chosen["fact_evidence_id"],
                "raw_fact_ref": chosen["raw_fact_ref"],
                "concept_norm": chosen["concept_norm"],
                "value_type": chosen["value_type"],
                "period_type": chosen["period_type"],
                "period_key": chosen["period_key"],
                "period_start": chosen["period_start"],
                "period_end": chosen["period_end"],
                "fiscal_year": chosen["fiscal_year"],
                "fiscal_quarter": chosen["fiscal_quarter"],
                "is_ytd": chosen["is_ytd"],
                "unit_canon": chosen["unit_canon"],
                "value_num_exact": chosen["value_num_exact"],
                "value_text": chosen["value_text"],
                "selection_rank": 1,
                "selection_reason": _selection_reason(chosen),
                "source_anchor": chosen["source_anchor"],
            }
        )

    return canon_rows, list(validators.values())


def _build_chunk_canon(conn: sqlite3.Connection, filing: FilingMetadata) -> Tuple[List[Dict[str, Any]], List[ValidationRecord]]:
    rows = conn.execute(
        """
        SELECT
          nc.filing_id,
          nc.source_file,
          nc.item,
          nc.heading,
          nc.subheading,
          nc.chunk_index,
          nc.char_start,
          nc.char_end,
          nc.heading_path,
          nc.retrieval_text,
          nc.text_masked,
          nc.text_sha1
        FROM narrative_chunks nc
        WHERE nc.filing_id=?
        ORDER BY nc.source_file, nc.char_start
        """,
        (filing.filing_id,),
    ).fetchall()

    validators: Dict[str, ValidationRecord] = {}
    chunk_rows = []
    period_key = filing_period_key(filing)
    for row in rows:
        chunk_evidence_id = f"chunk::{_stable_digest(row['filing_id'], row['source_file'], row['char_start'], row['char_end'], row['text_sha1'])}"
        if not chunk_evidence_id:
            _add_validator(
                validators,
                filing.filing_id,
                "MISSING_EVIDENCE_ID",
                "chunk",
                f"{row['source_file']}:{row['char_start']}-{row['char_end']}",
                "Canonical chunk is missing an evidence identifier.",
                {},
            )
        chunk_rows.append(
            {
                "chunk_evidence_id": chunk_evidence_id,
                "filing_id": row["filing_id"],
                "period_key": period_key,
                "item": row["item"],
                "heading": row["heading"],
                "subheading": row["subheading"],
                "heading_path": row["heading_path"],
                "source_file": row["source_file"],
                "char_start": row["char_start"],
                "char_end": row["char_end"],
                "retrieval_text": row["retrieval_text"] or row["text_masked"],
                "text_masked": row["text_masked"],
                "text_sha1": row["text_sha1"],
            }
        )
    return chunk_rows, list(validators.values())


def rebuild_canonical_layers(conn: sqlite3.Connection, filing_id: Optional[str] = None) -> None:
    """Rebuild canonical fact and chunk layers from raw tables.
    
    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection bound to the AuditOps corpus database.
    filing_id : Optional[str], optional
        Canonical filing identifier used across manifests, facts, and task records.
    """
    filing_ids = [filing_id] if filing_id else [row["filing_id"] for row in conn.execute("SELECT filing_id FROM filings ORDER BY filing_id")]
    for current_filing_id in filing_ids:
        filing = _fetch_filing(conn, current_filing_id)
        facts_canon, fact_validators = _build_facts_canon_and_validators(conn, filing)
        chunk_canon, chunk_validators = _build_chunk_canon(conn, filing)
        validators = fact_validators + chunk_validators

        conn.execute("DELETE FROM facts_canon WHERE filing_id=?", (current_filing_id,))
        conn.execute("DELETE FROM chunk_canon WHERE filing_id=?", (current_filing_id,))
        conn.execute("DELETE FROM validators_v0 WHERE filing_id=?", (current_filing_id,))

        if facts_canon:
            conn.executemany(
                """
                INSERT INTO facts_canon(
                  canon_id, filing_id, ticker, fact_evidence_id, raw_fact_ref,
                  concept_norm, value_type, period_type, period_key, period_start, period_end,
                  fiscal_year, fiscal_quarter, is_ytd, unit_canon,
                  value_num_exact, value_text, selection_rank, selection_reason, source_anchor
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        row["canon_id"],
                        row["filing_id"],
                        row["ticker"],
                        row["fact_evidence_id"],
                        row["raw_fact_ref"],
                        row["concept_norm"],
                        row["value_type"],
                        row["period_type"],
                        row["period_key"],
                        row["period_start"],
                        row["period_end"],
                        row["fiscal_year"],
                        row["fiscal_quarter"],
                        row["is_ytd"],
                        row["unit_canon"],
                        row["value_num_exact"],
                        row["value_text"],
                        row["selection_rank"],
                        row["selection_reason"],
                        row["source_anchor"],
                    )
                    for row in facts_canon
                ],
            )

        if chunk_canon:
            conn.executemany(
                """
                INSERT INTO chunk_canon(
                  chunk_evidence_id, filing_id, period_key, item, heading, subheading, heading_path,
                  source_file, char_start, char_end, retrieval_text, text_masked, text_sha1
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        row["chunk_evidence_id"],
                        row["filing_id"],
                        row["period_key"],
                        row["item"],
                        row["heading"],
                        row["subheading"],
                        row["heading_path"],
                        row["source_file"],
                        row["char_start"],
                        row["char_end"],
                        row["retrieval_text"],
                        row["text_masked"],
                        row["text_sha1"],
                    )
                    for row in chunk_canon
                ],
            )

        if validators:
            conn.executemany(
                """
                INSERT INTO validators_v0(
                  validator_id, filing_id, validator_code, entity_kind, entity_key, message, details_json
                ) VALUES (?,?,?,?,?,?,?)
                """,
                [
                    (
                        _validation_key(current_filing_id, record.validator_code, record.entity_kind, record.entity_key, record.details),
                        current_filing_id,
                        record.validator_code,
                        record.entity_kind,
                        record.entity_key,
                        record.message,
                        _json_dumps(record.details),
                    )
                    for record in validators
                ],
            )

        conn.commit()


def fetch_validators(conn: sqlite3.Connection, filing_id: Optional[str] = None) -> List[sqlite3.Row]:
    """Fetch validator records for one filing or the full corpus. A validator record is something like entity_key, etc., which identifies object.
    
    Parameters
    ----------
    conn : sqlite3.Connection
        Open SQLite connection for the corpus database.
    filing_id : Optional[str], optional
        Canonical filing identifier used by manifests and derived artifacts.
    
    Returns
    -------
    List[sqlite3.Row]
        Validator rows from validators_v0 for one filing or the full corpus.
    """
    if filing_id:
        return conn.execute(
            "SELECT * FROM validators_v0 WHERE filing_id=? ORDER BY validator_code, entity_key",
            (filing_id,),
        ).fetchall()
    return conn.execute("SELECT * FROM validators_v0 ORDER BY filing_id, validator_code, entity_key").fetchall()


def inspect_chunk_canon(conn: sqlite3.Connection, filing_id: str, limit: int = 5) -> List[sqlite3.Row]:
    """Inspect canonical chunk rows for a filing.
    
    Parameters
    ----------
    conn : sqlite3.Connection
        SQLite connection for the corpus database.
    filing_id : str
        Canonical filing identifier used by manifests and derived artifacts. e.g., '0000320193-2025-10K'
    limit : int, optional
        Maximum number of records to process.
    
    Returns
    -------
    List[sqlite3.Row]
        List of records for inspect canonical chunk rows for a filing.
    """
    return conn.execute(
        """
        SELECT item, heading, char_start, char_end, substr(text_masked, 1, 240) AS preview
        FROM chunk_canon
        WHERE filing_id=?
        ORDER BY source_file, char_start
        LIMIT ?
        """,
        (filing_id, limit),
    ).fetchall()
