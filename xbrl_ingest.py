#!/usr/bin/env python3
"""
xbrl_ingest.py

Ingest an SEC EDGAR iXBRL/XBRL ZIP package into a SQLite database suitable for:
- Type A: deterministic facts (numeric + non-numeric) with raw/unscaled numeric values
- Linkbases: labels / presentation / calculation / definition graphs for hierarchy + synthesis
- Type B (optional): narrative chunks (tables removed, numbers masked) for later vector indexing

Usage:
  pip install lxml
  python xbrl_ingest.py --zip path/to/000xxxxxx-yy-zzzzzz-xbrl.zip --out_db facts.sqlite --extract_text --reset_db

Notes:
- This parser is designed for SEC iXBRL HTML packages (facts inline in .htm) plus *_lab.xml, *_pre.xml, *_cal.xml, *_def.xml.
- It stores both reported and unscaled numeric values (scale + sign applied) for deterministic computation.
- Period keys are heuristics (FY, Q, ASOF). You can refine for fiscal calendars later.
"""

import argparse
import io
import json
import os
import re
import sqlite3
import zipfile
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from lxml import etree
from lxml import html as lxml_html


# ----------------------------
# Helpers
# ----------------------------

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_DURATION_RE = re.compile(r"^P(?!$)")  # loose: starts with P
NUM_MASK_RE = re.compile(r"\b\d[\d,.\-%]*\b")


def safe_text(el: Optional[etree._Element]) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def resolve_continued_text(root: etree._Element, fact_el: etree._Element) -> str:
    """
    Resolve ix:nonNumeric text that may be continued via @continuedAt + ix:continuation.
    SEC iXBRL frequently uses this for large *TextBlock facts.
    """
    if fact_el is None:
        return ""
    parts = [safe_text(fact_el)]
    # continuedAt can chain
    cont_id = fact_el.get("continuedAt")
    if not cont_id:
        return parts[0].strip()

    seen = set()
    while cont_id and cont_id not in seen:
        seen.add(cont_id)
        # Match either @id or @xml:id, and local-name()='continuation'
        els = root.xpath(f"//*[@id='{cont_id}' or @xml:id='{cont_id}'][local-name()='continuation']")
        if not els:
            break
        cont_el = els[0]
        parts.append(safe_text(cont_el))
        cont_id = cont_el.get("continuedAt")

    # Join with single spaces to avoid accidental token glue
    return " ".join([p.strip() for p in parts if p and p.strip()])



def parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s or not ISO_DATE_RE.match(s):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def norm_concept_for_linkbase(qname: str) -> str:
    # Facts use 'us-gaap:Revenues'; linkbases often use 'us-gaap_Revenues'
    return (qname or "").strip().replace(":", "_")


def get_attr_any_ns(el: etree._Element, attr_local: str) -> Optional[str]:
    for k, v in el.attrib.items():
        if k.split("}")[-1] == attr_local:
            return v
    return None


def parse_numeric(text: str) -> Optional[Decimal]:
    """
    Parse numeric strings common in filings:
    - 1,234
    - (1,234) => -1234
    - $1,234.56
    """
    if text is None:
        return None
    t = text.strip()
    if not t:
        return None

    t = t.replace("$", "").replace("€", "").replace("£", "")
    t = t.replace("\u2212", "-")  # unicode minus
    t = re.sub(r"\s+", "", t)

    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg = True
        t = t[1:-1]

    t = t.replace(",", "")

    if t in {"--", "—", "N/A", "NA"}:
        return None

    try:
        val = Decimal(t)
        return -val if neg else val
    except InvalidOperation:
        return None


def apply_scale(val: Optional[Decimal], scale: Optional[str]) -> Optional[Decimal]:
    """
    In iXBRL/XBRL, @scale means multiply by 10^scale.
    """
    if val is None:
        return None
    if not scale:
        return val
    try:
        s = int(scale)
        return val * (Decimal(10) ** Decimal(s))
    except Exception:
        return val


def apply_sign(val: Optional[Decimal], sign: Optional[str]) -> Optional[Decimal]:
    if val is None:
        return None
    if sign and sign.strip() == "-":
        return -val
    return val



# --- iXBRL transformation support (SEC/EDGAR iXBRL) -----------------------------

_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

def _norm_fmt(fmt: Optional[str]) -> str:
    if not fmt:
        return ""
    f = fmt.strip()
    # handle prefixed forms (ixt:fixed-zero) and URI forms (…#fixed-zero)
    f = f.split("#")[-1]
    f = f.split(":")[-1]
    return f.lower()

def _words_to_int(s: str) -> Optional[int]:
    """Parse simple English number words (SEC ixt-sec:numwordsen)."""
    if not s:
        return None
    t = re.sub(r"[^a-z\s-]", " ", s.lower()).strip()
    t = t.replace("-", " ")
    if not t:
        return None
    toks = [x for x in t.split() if x]
    total = 0
    current = 0
    for tok in toks:
        if tok in _NUM_WORDS:
            current += _NUM_WORDS[tok]
        elif tok == "hundred":
            current = max(1, current) * 100
        elif tok == "thousand":
            total += max(1, current) * 1000
            current = 0
        elif tok == "million":
            total += max(1, current) * 1_000_000
            current = 0
        elif tok == "billion":
            total += max(1, current) * 1_000_000_000
            current = 0
        else:
            # unknown token -> bail
            return None
    return total + current

def _parse_monthname_date(s: str) -> Optional[str]:
    """Parse dates like 'December 31, 2025' into ISO YYYY-MM-DD."""
    if not s:
        return None
    t = s.strip()
    # Common SEC date strings: "December 31, 2025" or "Dec. 31, 2025"
    m = re.search(r"([A-Za-z]{3,9})\.?\s+(\d{1,2})\s*,\s*(\d{4})", t)
    if not m:
        return None
    mon = _MONTHS.get(m.group(1).lower())
    if not mon:
        return None
    day = int(m.group(2))
    yr = int(m.group(3))
    try:
        return date(yr, mon, day).isoformat()
    except Exception:
        return None

def apply_ix_transform(fmt: Optional[str], raw_text: str) -> Tuple[str, Optional[str]]:
    """
    Apply common iXBRL transforms so the DB stores deterministic canonical values.
    Returns (canonical_text, forced_value_type_or_None).
    """
    f = _norm_fmt(fmt)
    t = (raw_text or "").strip()

    if f == "fixed-zero":
        return "0", "numeric"
    if f == "fixed-true":
        return "true", "bool"
    if f == "fixed-false":
        return "false", "bool"
    if f == "numwordsen":
        n = _words_to_int(t)
        return (str(n) if n is not None else t), ("numeric" if n is not None else None)

    # SEC duration helpers: normalize to ISO 8601 durations (P..D/P..M/P..Y)
    if f in ("durday", "durmonth", "duryear", "durwordsen"):
        # Try numeric first
        n = None
        m = re.search(r"(-?\d+)", t)
        if m:
            n = int(m.group(1))
        else:
            n = _words_to_int(t)
        if n is None:
            return t, None
        if "month" in f:
            return f"P{n}M", "duration"
        if "year" in f:
            return f"P{n}Y", "duration"
        # default days
        return f"P{n}D", "duration"

    # Common monthname date transform family
    if f.startswith("date-monthname-day-year"):
        iso = _parse_monthname_date(t)
        return (iso if iso else t), ("date" if iso else None)

    return t, None


def period_key_from_context(
    start: Optional[date], end: Optional[date], instant: Optional[date]
) -> Tuple[str, Optional[int]]:
    """
    Returns (period_key, days).
    Heuristic:
      - instant => ASOF_YYYYMMDD
      - ~12 months duration => FY{end.year}
      - ~3 months duration => Q{n}_{end.year} if end is a standard quarter end
      - otherwise => DUR_YYYYMMDD_YYYYMMDD
    """
    if instant:
        return f"ASOF_{instant.strftime('%Y%m%d')}", None

    if start and end:
        days = (end - start).days

        if 330 <= days <= 370:
            return f"FY{end.year}", days

        if 80 <= days <= 110:
            if (end.month, end.day) in {(3, 31), (6, 30), (9, 30), (12, 31)}:
                q = {3: 1, 6: 2, 9: 3, 12: 4}[end.month]
                return f"Q{q}_{end.year}", days

        return f"DUR_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}", days

    return "UNKNOWN_PERIOD", None


def mask_numbers(text: str) -> str:
    return NUM_MASK_RE.sub("<NUM>", text)


# ----------------------------
# Records
# ----------------------------

@dataclass
class ContextRec:
    context_id: str
    entity_identifier: str
    start_date: Optional[date]
    end_date: Optional[date]
    instant_date: Optional[date]
    period_key: str
    days: Optional[int]
    dimensions: Dict[str, Any]
    is_consolidated: bool


@dataclass
class UnitRec:
    unit_id: str
    unit_json: Dict[str, Any]


@dataclass
class FactRec:
    fact_id: str
    concept_qname: str
    concept_norm: str
    context_id: Optional[str]
    unit_id: Optional[str]
    decimals: Optional[str]
    scale: Optional[str]
    sign: Optional[str]
    fmt: Optional[str]
    value_type: str  # numeric/string/date/duration/bool
    value_text: Optional[str]
    value_lexical: Optional[str]
    value_num_reported: Optional[Decimal]
    value_num_unscaled: Optional[Decimal]
    period_key: str
    dimensions: Dict[str, Any]
    is_consolidated: bool
    source_file: str
    source_anchor: Optional[str]


# ----------------------------
# Parsing: iXBRL HTML
# ----------------------------

def parse_ixbrl_html(
    html_bytes: bytes, source_file: str
) -> Tuple[Dict[str, ContextRec], Dict[str, UnitRec], List[FactRec]]:
    """
    Extracts:
      - xbrli:context
      - xbrli:unit
      - ix:nonFraction / ix:nonNumeric facts
    Works on most SEC iXBRL HTML packages.
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.parse(io.BytesIO(html_bytes), parser).getroot()

    # Contexts
    contexts: Dict[str, ContextRec] = {}
    for c in root.xpath("//*[local-name()='context']"):
        cid = c.get("id") or c.get("{http://www.w3.org/XML/1998/namespace}id")
        if not cid:
            continue

        ident_el = None
        for cand in c.xpath(".//*[local-name()='entity']//*[local-name()='identifier']"):
            ident_el = cand
            break
        entity_identifier = safe_text(ident_el)

        start = end = instant = None
        period_el = None
        for cand in c.xpath(".//*[local-name()='period']"):
            period_el = cand
            break
        if period_el is not None:
            inst_el = None
            for cand in period_el.xpath(".//*[local-name()='instant']"):
                inst_el = cand
                break
            if inst_el is not None:
                instant = parse_date(safe_text(inst_el))
            else:
                sd_el = None
                ed_el = None
                for cand in period_el.xpath(".//*[local-name()='startDate']"):
                    sd_el = cand
                    break
                for cand in period_el.xpath(".//*[local-name()='endDate']"):
                    ed_el = cand
                    break
                start = parse_date(safe_text(sd_el))
                end = parse_date(safe_text(ed_el))

        dims: Dict[str, Any] = {}
        for em in c.xpath(".//*[local-name()='explicitMember']"):
            dim = em.get("dimension") or em.get("{http://www.xbrl.org/2006/xbrldi}dimension")
            val = safe_text(em)
            if dim and val:
                dims[dim] = val
        for tm in c.xpath(".//*[local-name()='typedMember']"):
            dim = tm.get("dimension") or tm.get("{http://www.xbrl.org/2006/xbrldi}dimension")
            val = safe_text(tm)
            if dim and val:
                dims[dim] = val

        pkey, days = period_key_from_context(start, end, instant)
        contexts[cid] = ContextRec(
            context_id=cid,
            entity_identifier=entity_identifier,
            start_date=start,
            end_date=end,
            instant_date=instant,
            period_key=pkey,
            days=days,
            dimensions=dims,
            is_consolidated=(len(dims) == 0),
        )

    # Units
    units: Dict[str, UnitRec] = {}
    for u in root.xpath("//*[local-name()='unit']"):
        uid = u.get("id") or u.get("{http://www.w3.org/XML/1998/namespace}id")
        if not uid:
            continue

        unit_obj: Dict[str, Any] = {}
        measures = [safe_text(m) for m in u.xpath(".//*[local-name()='measure']")]
        measures = [m for m in measures if m]
        if measures:
            unit_obj["measures"] = measures

        div_el = None
        for cand in u.xpath(".//*[local-name()='divide']"):
            div_el = cand
            break
        if div_el is not None:
            num = [safe_text(m) for m in div_el.xpath(".//*[local-name()='unitNumerator']//*[local-name()='measure']")]
            den = [safe_text(m) for m in div_el.xpath(".//*[local-name()='unitDenominator']//*[local-name()='measure']")]
            unit_obj["divide"] = {"numerator": [x for x in num if x], "denominator": [x for x in den if x]}

        units[uid] = UnitRec(unit_id=uid, unit_json=unit_obj)

    # Facts
    facts: List[FactRec] = []
    for fact_el in root.xpath("//*[local-name()='nonFraction' or local-name()='nonNumeric']"):
        lname = fact_el.tag.split("}")[-1]

        fid = fact_el.get("id") or fact_el.get("{http://www.w3.org/XML/1998/namespace}id")
        anchorable = True
        if not fid:
            # stable-ish synthetic id (not anchorable)
            fid = f"fact_{abs(hash(etree.tostring(fact_el)[:256]))}"
            anchorable = False

        qname = fact_el.get("name") or ""
        concept_norm = norm_concept_for_linkbase(qname)

        context_id = fact_el.get("contextRef")
        unit_id = fact_el.get("unitRef")
        decimals = fact_el.get("decimals")
        scale = fact_el.get("scale")
        sign = fact_el.get("sign")
        fmt = fact_el.get("format")

        nil_val = fact_el.get("{http://www.w3.org/2001/XMLSchema-instance}nil")
        # Preserve lexical text for debugging/tick-tie; canonical value_text is after transforms.
        if nil_val and nil_val.lower() == "true":
            raw_text_lexical = ""
        else:
            # For ix:nonNumeric, resolve continuation chains; for nonFraction safe_text is sufficient.
            raw_text_lexical = resolve_continued_text(root, fact_el) if lname == "nonNumeric" else safe_text(fact_el)

        canon_text, forced_vtype = apply_ix_transform(fmt, raw_text_lexical)
        raw_text = canon_text

        period_key = "UNKNOWN_PERIOD"
        dims: Dict[str, Any] = {}
        is_consolidated = True
        if context_id and context_id in contexts:
            period_key = contexts[context_id].period_key
            dims = contexts[context_id].dimensions
            is_consolidated = contexts[context_id].is_consolidated

        source_anchor = f"{source_file}#{fid}" if anchorable else None

        if lname == "nonFraction":
            val_rep = parse_numeric(raw_text)
            val_rep = apply_sign(val_rep, sign)
            val_unscaled = apply_scale(val_rep, scale)
            facts.append(FactRec(
                fact_id=fid,
                concept_qname=qname,
                concept_norm=concept_norm,
                context_id=context_id,
                unit_id=unit_id,
                decimals=decimals,
                scale=scale,
                sign=sign,
                fmt=fmt,
                value_type="numeric",
                value_text=raw_text if raw_text else None,
                value_lexical=raw_text_lexical if raw_text_lexical else None,
                value_num_reported=val_rep,
                value_num_unscaled=val_unscaled,
                period_key=period_key,
                dimensions=dims,
                is_consolidated=is_consolidated,
                source_file=source_file,
                source_anchor=source_anchor,
            ))
        else:
            t = raw_text.strip()
            # Determine value type. forced_vtype comes from iXBRL transforms (e.g., fixed-true/false).
            vtype = "string"
            t = raw_text

            if forced_vtype in ("bool", "date", "duration"):
                vtype = forced_vtype
            elif forced_vtype == "numeric":
                # rare, but nonNumeric can be numeric via transforms
                vtype = "numeric"
            else:
                # Heuristics from concept naming
                if qname.lower().endswith("flag") or qname.lower().endswith("boolean"):
                    vtype = "bool"
                elif qname.lower().endswith("date"):
                    vtype = "date"
                elif qname.lower().endswith("duration"):
                    vtype = "duration"

            # Normalize bool strings
            if vtype == "bool" and t is not None:
                tt = t.strip().lower()
                if tt in ("☒", "x", "true", "t", "yes", "1"):
                    t = "true"
                elif tt in ("☐", "false", "f", "no", "0"):
                    t = "false"
                else:
                    # keep as-is; downstream can refuse if not canonical
                    t = tt
            facts.append(FactRec(
                fact_id=fid,
                concept_qname=qname,
                concept_norm=concept_norm,
                context_id=context_id,
                unit_id=unit_id,
                decimals=decimals,
                scale=scale,
                sign=sign,
                fmt=fmt,
                value_type=vtype,
                value_text=t if t else None,
                value_lexical=raw_text_lexical if raw_text_lexical else None,
                value_num_reported=None,
                value_num_unscaled=None,
                period_key=period_key,
                dimensions=dims,
                is_consolidated=is_consolidated,
                source_file=source_file,
                source_anchor=source_anchor,
            ))

    return contexts, units, facts


# ----------------------------
# Parsing: linkbases
# ----------------------------

def concept_from_href(href: str) -> Optional[str]:
    if not href or "#" not in href:
        return None
    frag = href.split("#", 1)[1].strip()
    return frag or None


def parse_label_linkbase(xml_bytes: bytes) -> List[Tuple[str, str, str, str]]:
    """
    Returns (concept_norm, label_role, label_lang, label_text)
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.parse(io.BytesIO(xml_bytes), parser).getroot()

    loc_map: Dict[str, str] = {}
    for loc in root.xpath("//*[local-name()='loc']"):
        lab = get_attr_any_ns(loc, "label")
        href = get_attr_any_ns(loc, "href")
        concept = concept_from_href(href) if href else None
        if lab and concept:
            loc_map[lab] = concept

    label_map: Dict[str, Tuple[str, str, str]] = {}  # label_id -> (role, lang, text)
    for lab_el in root.xpath("//*[local-name()='label']"):
        lab = get_attr_any_ns(lab_el, "label")
        role = get_attr_any_ns(lab_el, "role") or ""
        lang = lab_el.get("{http://www.w3.org/XML/1998/namespace}lang") or ""
        text = safe_text(lab_el)
        if lab and text:
            label_map[lab] = (role, lang, text)

    out: List[Tuple[str, str, str, str]] = []
    for arc in root.xpath("//*[local-name()='labelArc']"):
        fr = get_attr_any_ns(arc, "from")
        to = get_attr_any_ns(arc, "to")
        if not fr or not to:
            continue
        concept = loc_map.get(fr)
        if not concept:
            continue
        role, lang, text = label_map.get(to, ("", "", ""))
        if text:
            out.append((concept, role, lang, text))
    return out


def parse_presentation_linkbase(xml_bytes: bytes) -> List[Tuple[str, str, str, float, str]]:
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.parse(io.BytesIO(xml_bytes), parser).getroot()
    edges: List[Tuple[str, str, str, float, str]] = []

    for ext in root.xpath("//*[local-name()='presentationLink']"):
        role = get_attr_any_ns(ext, "role") or ""

        loc_map: Dict[str, str] = {}
        for loc in ext.xpath(".//*[local-name()='loc']"):
            lab = get_attr_any_ns(loc, "label")
            href = get_attr_any_ns(loc, "href")
            concept = concept_from_href(href) if href else None
            if lab and concept:
                loc_map[lab] = concept

        for arc in ext.xpath(".//*[local-name()='presentationArc']"):
            fr = get_attr_any_ns(arc, "from")
            to = get_attr_any_ns(arc, "to")
            ord_s = arc.get("order") or "0"
            pref = arc.get("preferredLabel") or ""
            try:
                ord_v = float(ord_s)
            except ValueError:
                ord_v = 0.0
            if fr in loc_map and to in loc_map:
                edges.append((role, loc_map[fr], loc_map[to], ord_v, pref))

    return edges


def parse_calculation_linkbase(xml_bytes: bytes) -> List[Tuple[str, str, str, float, float]]:
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.parse(io.BytesIO(xml_bytes), parser).getroot()
    edges: List[Tuple[str, str, str, float, float]] = []

    for ext in root.xpath("//*[local-name()='calculationLink']"):
        role = get_attr_any_ns(ext, "role") or ""

        loc_map: Dict[str, str] = {}
        for loc in ext.xpath(".//*[local-name()='loc']"):
            lab = get_attr_any_ns(loc, "label")
            href = get_attr_any_ns(loc, "href")
            concept = concept_from_href(href) if href else None
            if lab and concept:
                loc_map[lab] = concept

        for arc in ext.xpath(".//*[local-name()='calculationArc']"):
            fr = get_attr_any_ns(arc, "from")
            to = get_attr_any_ns(arc, "to")
            w_s = arc.get("weight") or "0"
            o_s = arc.get("order") or "0"
            try:
                w = float(w_s)
            except ValueError:
                w = 0.0
            try:
                o = float(o_s)
            except ValueError:
                o = 0.0
            if fr in loc_map and to in loc_map:
                edges.append((role, loc_map[fr], loc_map[to], w, o))

    return edges


def parse_definition_linkbase(xml_bytes: bytes) -> List[Tuple[str, str, str, str, float]]:
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.parse(io.BytesIO(xml_bytes), parser).getroot()
    edges: List[Tuple[str, str, str, str, float]] = []

    for ext in root.xpath("//*[local-name()='definitionLink']"):
        role = get_attr_any_ns(ext, "role") or ""

        loc_map: Dict[str, str] = {}
        for loc in ext.xpath(".//*[local-name()='loc']"):
            lab = get_attr_any_ns(loc, "label")
            href = get_attr_any_ns(loc, "href")
            concept = concept_from_href(href) if href else None
            if lab and concept:
                loc_map[lab] = concept

        for arc in ext.xpath(".//*[local-name()='definitionArc']"):
            fr = get_attr_any_ns(arc, "from")
            to = get_attr_any_ns(arc, "to")
            arcrole = get_attr_any_ns(arc, "arcrole") or ""
            o_s = arc.get("order") or "0"
            try:
                o = float(o_s)
            except ValueError:
                o = 0.0
            if fr in loc_map and to in loc_map:
                edges.append((role, arcrole, loc_map[fr], loc_map[to], o))

    return edges


# ----------------------------
# Narrative extraction (Type B staging)
# ----------------------------

def extract_text_chunks(html_bytes: bytes, source_file: str, max_chars: int = 1800) -> List[Dict[str, Any]]:
    """
    Type B narrative chunker (SEC 10-K/10-Q friendly).

    Goals:
      - Remove <table> blocks to prevent numeric leakage (Type A stores deterministic numbers)
      - Detect sections using SEC "Item" headings (more reliable than <h1>-<h6> in iXBRL HTML)
      - Mask numbers (<NUM>) for training / retrieval robustness
      - Drop obvious taxonomy/noise blobs (e.g., FASB taxonomy dumps)
      - Enforce reasonable chunk sizes

    Returns chunks with:
      chunk_id, section, chunk_index, text_raw, text_masked, source_file
    """
    doc = lxml_html.fromstring(html_bytes)

    # Remove common non-narrative content
    for el in doc.xpath("//table|//script|//style|//noscript"):
        el.drop_tree()

    # Gather paragraph-ish blocks in document order
    blocks: List[str] = []
    for el in doc.xpath("//p|//li|//div|//span"):
        t = safe_text(el)
        if not t:
            continue
        if len(t) < 40:
            continue
        blocks.append(t)

    # Fallback: if block extraction yields too little, use full text content
    if len(blocks) < 20:
        full = doc.text_content()
        full = re.sub(r"\s+", " ", full).strip()
        if full:
            blocks = [full[i:i+1200] for i in range(0, len(full), 1200)]

    item_re = re.compile(r"^\s*(?:ITEM|Item)\s+(\d{1,2}[A-Za-z]?)\s*[\.\-:]?\s*(.*)$")

    def is_noise_blob(t: str) -> bool:
        low = t.lower()
        if "fasb.org" in low or "xbrl.org" in low:
            return True
        if low.count("us-gaap:") + low.count("us-gaap_") > 15:
            return True
        if low.count("dei:") + low.count("xbrli:") > 20:
            return True
        if len(t) > 2000:
            letters = sum(ch.isalpha() for ch in t)
            if letters / max(len(t), 1) < 0.35:
                return True
        return False

    chunks: List[Dict[str, Any]] = []
    cur_section = "UNKNOWN_SECTION"
    buf: List[str] = []
    chunk_idx = 0

    def flush():
        nonlocal buf, chunk_idx
        if not buf:
            return
        raw = "\n".join([b for b in buf if b]).strip()
        buf = []
        if not raw:
            return
        if len(raw) > 8000 and is_noise_blob(raw):
            return
        chunks.append({
            "chunk_id": f"{source_file}::{cur_section}::{chunk_idx}",
            "section": cur_section,
            "chunk_index": chunk_idx,
            "text_raw": raw,
            "text_masked": mask_numbers(raw),
            "source_file": source_file,
        })
        chunk_idx += 1

    for t in blocks:
        t = re.sub(r"\s+", " ", t).strip()
        if not t:
            continue

        m = item_re.match(t[:220])
        if m:
            flush()
            item_no = m.group(1).upper()
            title = (m.group(2) or "").strip()
            cur_section = f"ITEM {item_no} - {title[:140]}" if title else f"ITEM {item_no}"
            continue

        if is_noise_blob(t):
            continue

        buf.append(t)
        if sum(len(x) for x in buf) > max_chars:
            flush()

    flush()

    # If headings were not detected, still filter out garbage-like chunks
    if chunks and all(ch["section"] == "UNKNOWN_SECTION" for ch in chunks):
        chunks = [ch for ch in chunks if not is_noise_blob(ch["text_raw"]) and len(ch["text_raw"]) <= 8000]

    return chunks


# ----------------------------
# DB schema (multi-filing)
# ----------------------------

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS filings (
  filing_id TEXT PRIMARY KEY,
  ticker TEXT,
  zip_name TEXT,
  main_html TEXT,
  processed_at TEXT
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

CREATE TABLE IF NOT EXISTS text_chunks (
  filing_id TEXT,
  chunk_id TEXT,
  section TEXT,
  chunk_index INTEGER,
  text_masked TEXT,
  text_raw TEXT,
  source_file TEXT,
  PRIMARY KEY (filing_id, chunk_id),
  FOREIGN KEY (filing_id) REFERENCES filings(filing_id)
);
"""


def ensure_schema(conn: sqlite3.Connection, reset: bool = False) -> None:
    cur = conn.cursor()
    if reset:
        cur.executescript("""
        PRAGMA foreign_keys=OFF;
        DROP TABLE IF EXISTS text_chunks;
        DROP TABLE IF EXISTS def_edges;
        DROP TABLE IF EXISTS cal_edges;
        DROP TABLE IF EXISTS pre_edges;
        DROP TABLE IF EXISTS labels;
        DROP TABLE IF EXISTS facts;
        DROP TABLE IF EXISTS units;
        DROP TABLE IF EXISTS contexts;
        DROP TABLE IF EXISTS filings;
        PRAGMA foreign_keys=ON;
        """)
        conn.commit()

    conn.executescript(SCHEMA_SQL)
    conn.commit()


# ----------------------------
# ZIP orchestrator
# ----------------------------

def load_zip_members(zip_path: str) -> Dict[str, bytes]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return {info.filename: zf.read(info.filename) for info in zf.infolist()}


def pick_main_html(members: Dict[str, bytes]) -> str:
    # pick the largest root-level .htm/.html as main iXBRL doc
    candidates: List[Tuple[int, str]] = []
    for name, b in members.items():
        low = name.lower()
        if "/" in name:
            continue
        if low.endswith((".htm", ".html")):
            candidates.append((len(b), name))
    if not candidates:
        raise RuntimeError("No root-level .htm/.html found in zip (expected iXBRL HTML).")
    candidates.sort(reverse=True)
    return candidates[0][1]


def infer_filing_id(zip_path: str, html_main: str) -> str:
    base = os.path.basename(zip_path)
    m = re.match(r"^(\d{10}-\d{2}-\d{6})-xbrl\.zip$", base, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(html_main))[0]


def infer_ticker(html_main: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(html_main))[0]
    if "-" in stem:
        left = stem.split("-", 1)[0]
        if 1 <= len(left) <= 8 and left.isalnum():
            return left.upper()
    return None


def process_zip(zip_path: str, out_db: str, ticker: Optional[str], extract_text: bool, reset_db: bool) -> None:
    members = load_zip_members(zip_path)

    html_main = pick_main_html(members)
    lab = next((n for n in members if n.lower().endswith("_lab.xml")), None)
    pre = next((n for n in members if n.lower().endswith("_pre.xml")), None)
    cal = next((n for n in members if n.lower().endswith("_cal.xml")), None)
    deff = next((n for n in members if n.lower().endswith("_def.xml")), None)

    filing_id = infer_filing_id(zip_path, html_main)
    ticker_final = ticker or infer_ticker(html_main) or ""

    contexts, units, facts = parse_ixbrl_html(members[html_main], source_file=html_main)

    labels = parse_label_linkbase(members[lab]) if lab else []
    pre_edges = parse_presentation_linkbase(members[pre]) if pre else []
    cal_edges = parse_calculation_linkbase(members[cal]) if cal else []
    def_edges = parse_definition_linkbase(members[deff]) if deff else []

    text_chunks = extract_text_chunks(members[html_main], html_main) if extract_text else []

    conn = sqlite3.connect(out_db)
    try:
        ensure_schema(conn, reset=reset_db)
        conn.execute("BEGIN")

        conn.execute(
            "INSERT OR REPLACE INTO filings(filing_id, ticker, zip_name, main_html, processed_at) VALUES (?,?,?,?,?)",
            (filing_id, ticker_final, os.path.basename(zip_path), html_main, datetime.utcnow().isoformat(timespec="seconds") + "Z")
        )

        ctx_rows = []
        for c in contexts.values():
            ctx_rows.append((
                filing_id, c.context_id, c.entity_identifier,
                c.start_date.isoformat() if c.start_date else None,
                c.end_date.isoformat() if c.end_date else None,
                c.instant_date.isoformat() if c.instant_date else None,
                c.period_key, c.days,
                json.dumps(c.dimensions, ensure_ascii=False, sort_keys=True, separators=(',', ':')),
                1 if c.is_consolidated else 0
            ))
        conn.executemany(
            """
            INSERT OR REPLACE INTO contexts(
              filing_id, context_id, entity_identifier,
              start_date, end_date, instant_date,
              period_key, days, dimensions_json, is_consolidated
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            ctx_rows
        )

        unit_rows = [(filing_id, u.unit_id, json.dumps(u.unit_json, ensure_ascii=False, sort_keys=True, separators=(',', ':'))) for u in units.values()]
        conn.executemany(
            "INSERT OR REPLACE INTO units(filing_id, unit_id, unit_json) VALUES (?,?,?)",
            unit_rows
        )

        fact_rows = []
        for f in facts:
            fact_rows.append((
                filing_id, f.fact_id, f.concept_qname, f.concept_norm,
                f.context_id, f.unit_id, f.decimals, f.scale, f.sign, f.fmt,
                f.value_type, f.value_text, f.value_lexical,
                str(f.value_num_reported) if f.value_num_reported is not None else None,
                str(f.value_num_unscaled) if f.value_num_unscaled is not None else None,
                f.period_key,
                json.dumps(f.dimensions, ensure_ascii=False, sort_keys=True, separators=(',', ':')),
                1 if f.is_consolidated else 0,
                f.source_file, f.source_anchor
            ))
        conn.executemany(
            """
            INSERT OR REPLACE INTO facts(
              filing_id, fact_id, concept_qname, concept_norm,
              context_id, unit_id, decimals, scale, sign, fmt,
              value_type, value_text, value_lexical,
              value_num_reported, value_num_unscaled,
              period_key, dimensions_json, is_consolidated,
              source_file, source_anchor
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            fact_rows
        )

        if labels:
            conn.executemany(
                """
                INSERT OR IGNORE INTO labels(
                  filing_id, concept_norm, label_role, label_lang, label_text
                ) VALUES (?,?,?,?,?)
                """,
                [(filing_id, c, r, lang, t) for (c, r, lang, t) in labels]
            )

        if pre_edges:
            conn.executemany(
                "INSERT INTO pre_edges(filing_id, role, parent_concept_norm, child_concept_norm, ord, preferred_label) VALUES (?,?,?,?,?,?)",
                [(filing_id, role, parent, child, ord_v, pref) for (role, parent, child, ord_v, pref) in pre_edges]
            )
        if cal_edges:
            conn.executemany(
                "INSERT INTO cal_edges(filing_id, role, parent_concept_norm, child_concept_norm, weight, ord) VALUES (?,?,?,?,?,?)",
                [(filing_id, role, parent, child, w, o) for (role, parent, child, w, o) in cal_edges]
            )
        if def_edges:
            conn.executemany(
                "INSERT INTO def_edges(filing_id, role, arcrole, parent_concept_norm, child_concept_norm, ord) VALUES (?,?,?,?,?,?)",
                [(filing_id, role, arcrole, parent, child, ord_v) for (role, arcrole, parent, child, ord_v) in def_edges]
            )

        if text_chunks:
            conn.executemany(
                """
                INSERT OR REPLACE INTO text_chunks(
                  filing_id, chunk_id, section, chunk_index, text_masked, text_raw, source_file
                ) VALUES (?,?,?,?,?,?,?)
                """,
                [(filing_id, ch["chunk_id"], ch["section"], ch["chunk_index"], ch["text_masked"], ch["text_raw"], ch["source_file"])
                 for ch in text_chunks]
            )

        conn.commit()

        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM facts WHERE filing_id=?", (filing_id,))
        n_facts = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM contexts WHERE filing_id=?", (filing_id,))
        n_ctx = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM units WHERE filing_id=?", (filing_id,))
        n_units = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM labels WHERE filing_id=?", (filing_id,))
        n_lab = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM pre_edges WHERE filing_id=?", (filing_id,))
        n_pre = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM cal_edges WHERE filing_id=?", (filing_id,))
        n_cal = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM def_edges WHERE filing_id=?", (filing_id,))
        n_def = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM text_chunks WHERE filing_id=?", (filing_id,))
        n_txt = cur.fetchone()[0]

        print("Done.")
        print(f"DB: {out_db}")
        print(f"Filing ID: {filing_id} | Ticker: {ticker_final} | Main HTML: {html_main}")
        print(f"Facts: {n_facts} | Contexts: {n_ctx} | Units: {n_units}")
        print(f"Labels: {n_lab} | Presentation edges: {n_pre} | Calculation edges: {n_cal} | Definition edges: {n_def}")
        if extract_text:
            print(f"Text chunks (masked): {n_txt}")

        print("\\nSample facts (5):")
        cur.execute(
            """
            SELECT f.concept_qname,
                   COALESCE(
                     (SELECT label_text FROM labels l
                      WHERE l.filing_id=f.filing_id AND l.concept_norm=f.concept_norm
                      ORDER BY CASE WHEN l.label_role LIKE '%terseLabel%' THEN 0 ELSE 1 END
                      LIMIT 1),
                     f.concept_qname
                   ) AS label,
                   f.period_key,
                   f.value_type,
                   COALESCE(f.value_num_unscaled, f.value_text) AS val,
                   f.source_anchor
            FROM facts f
            WHERE f.filing_id=?
            LIMIT 5
            """,
            (filing_id,)
        )
        for row in cur.fetchall():
            print(row)

    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Ingest an EDGAR iXBRL/XBRL zip into a SQLite facts DB.")
    ap.add_argument("--zip", required=True, help="Path to EDGAR XBRL/iXBRL zip package (e.g., 000xxxxxx-yy-zzzzzz-xbrl.zip)")
    ap.add_argument("--out_db", default="xbrl_facts.sqlite", help="Output SQLite DB path")
    ap.add_argument("--ticker", default=None, help="Optional ticker override")
    ap.add_argument("--extract_text", action="store_true", help="Also extract narrative chunks (tables removed, numbers masked)")
    ap.add_argument("--reset_db", action="store_true", help="Drop and recreate tables in out_db before ingesting")
    args = ap.parse_args()

    process_zip(args.zip, args.out_db, args.ticker, args.extract_text, args.reset_db)


if __name__ == "__main__":
    main()
