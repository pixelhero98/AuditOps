#!/usr/bin/env python3
"""
narrative_chunker.py

Extract narrative text from an SEC iXBRL HTML filing, remove tables, mask numbers,
chunk by SEC "Item" sections when possible, and store chunks into the SAME SQLite DB
as your XBRL facts.

This is intentionally "audit-friendly":
- no table parsing (tables removed)
- numbers masked (prevents model from "learning to guess")
- provenance: chunk -> (source_file, item, char range)

Usage:
  python narrative_chunker.py --zip /path/to/xbrl.zip --db xbrl.sqlite --filing_id 000... --main_html pypl-20251231.htm
  python narrative_chunker.py --db xbrl.sqlite --filing_id 000... --inspect 5
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sqlite3
import textwrap
import zipfile
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup


ITEM_RE = re.compile(
    r"^\s*item\s+(\d{1,2}[a]?)\s*[\.\-:]\s*(.*)$",
    flags=re.IGNORECASE,
)

# A conservative number masker: masks any digit sequence (including commas/decimals)
# while leaving surrounding punctuation/words intact.
NUM_RE = re.compile(
    r"""
    (?:
        (?<![A-Za-z])            # don't start in the middle of a word
        [\$\€\£]?\s*             # optional currency symbol
        \d{1,3}(?:,\d{3})+(?:\.\d+)?  # 1,234 or 1,234.56
        (?![A-Za-z])             # don't end in the middle of a word
    )
    |
    (?:
        (?<![A-Za-z])
        \d+\.\d+
        (?![A-Za-z])
    )
    |
    (?:
        (?<![A-Za-z])
        \d+
        (?![A-Za-z])
    )
    """,
    flags=re.VERBOSE,
)


@dataclass
class Section:
    item: str
    heading: str
    start: int
    end: int


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def pick_main_html_from_zip(z: zipfile.ZipFile) -> str:
    """Pick a likely main HTML by size among .htm/.html files (excluding exhibits when obvious)."""
    htmls = [n for n in z.namelist() if n.lower().endswith((".htm", ".html"))]
    if not htmls:
        raise FileNotFoundError("No .htm/.html found in ZIP")

    # Prefer root-level and larger files
    scored = []
    for n in htmls:
        try:
            info = z.getinfo(n)
            size = info.file_size
        except KeyError:
            size = 0
        penalty = 0
        low = n.lower()
        if "exhibit" in low or low.startswith("ex"):
            penalty += 2000000
        if "/xbrl" in low or "calculation" in low or "definition" in low:
            penalty += 2000000
        scored.append((size - penalty, size, n))
    scored.sort(reverse=True)
    return scored[0][2]


def read_html_from_zip(zip_path: str, main_html: Optional[str]) -> Tuple[str, str]:
    """Return (filename, html_string)."""
    with zipfile.ZipFile(zip_path, "r") as z:
        fname = main_html or pick_main_html_from_zip(z)
        with z.open(fname) as f:
            raw = f.read()
        # EDGAR is usually utf-8, but occasionally windows-1252-ish.
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                return fname, raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return fname, raw.decode("utf-8", errors="replace")


def unwrap_inline_xbrl_tags(soup: BeautifulSoup) -> None:
    """
    Unwrap ix:* tags so their text remains in the narrative stream, but the tags are removed.
    This keeps narrative readable while allowing you to later mask numbers globally.
    """
    # Any tag with a colon in the name may be treated oddly by bs4, but it preserves them.
    # We'll look for tag names that start with 'ix:' OR have local-name 'nonNumeric/nonFraction/...'
    for tag in list(soup.find_all()):
        name = (tag.name or "").lower()
        if name.startswith("ix:") or name in {
            "nonfraction", "nonnumeric", "fraction", "continuation"
        }:
            tag.unwrap()


def extract_narrative_text(html: str) -> str:
    # EDGAR "HTML" is sometimes XHTML (XML-serialized HTML). BeautifulSoup warns if it
    # looks like XML but we still want readable narrative text.
    from bs4 import XMLParsedAsHTMLWarning
    import warnings
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    parser = "lxml"
    if html.lstrip().startswith("<?xml"):
        # Prefer XML parser for XHTML documents; text extraction still works the same.
        parser = "lxml-xml"
    soup = BeautifulSoup(html, parser)

    # Drop scripts/styles
    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    # Remove tables completely to avoid numeric leakage
    for t in soup.find_all("table"):
        t.decompose()

    # Remove hidden content (common EDGAR patterns)
    for t in soup.select('[style*="display:none"], [style*="visibility:hidden"]'):
        t.decompose()

    # Unwrap inline XBRL tags but keep their text
    unwrap_inline_xbrl_tags(soup)

    body = soup.body or soup
    blocks = []

    # Collect text in block-ish order; preserve paragraph breaks
    for el in body.find_all(["h1","h2","h3","h4","h5","h6","p","li","div"]):
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        # Skip boilerplate navigation fragments
        if len(txt) <= 2:
            continue
        blocks.append(txt)

    # Collapse repeated whitespace and keep one blank line between blocks
    out = "\n\n".join(blocks)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def detect_item_sections(text: str) -> List[Section]:
    """
    Find 'Item X.' headings from the text stream.
    Works best when the HTML contains explicit headings; otherwise returns [].
    """
    # We'll scan line-by-line while tracking character offsets in the joined text
    lines = text.splitlines(True)  # keepends=True
    offsets = []
    cur = 0
    for ln in lines:
        offsets.append(cur)
        cur += len(ln)

    hits = []
    for i, ln in enumerate(lines):
        m = ITEM_RE.match(ln.strip())
        if not m:
            continue
        item = m.group(1).upper()
        heading = m.group(2).strip()
        hits.append((item, heading, offsets[i]))

    if not hits:
        return []

    # Build non-overlapping sections
    sections: List[Section] = []
    for idx, (item, heading, start) in enumerate(hits):
        end = hits[idx + 1][2] if idx + 1 < len(hits) else len(text)
        # Guard against junk headings that are too close together
        if end - start < 200:
            continue
        sections.append(Section(item=item, heading=heading, start=start, end=end))
    return sections


def mask_numbers(s: str) -> str:
    return NUM_RE.sub("<NUM>", s)


def chunk_text(s: str, chunk_chars: int, overlap: int) -> List[Tuple[int, int, str]]:
    """
    Return [(start, end, chunk)] in character offsets within `s`.
    """
    chunks = []
    n = len(s)
    if n == 0:
        return chunks

    step = max(1, chunk_chars - overlap)
    start = 0
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = s[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end == n:
            break
        start += step
    return chunks


def ensure_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS narrative_sections (
        filing_id TEXT NOT NULL,
        source_file TEXT NOT NULL,
        item TEXT,
        heading TEXT,
        start_char INTEGER NOT NULL,
        end_char INTEGER NOT NULL,
        PRIMARY KEY (filing_id, source_file, start_char)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS narrative_chunks (
        filing_id TEXT NOT NULL,
        source_file TEXT NOT NULL,
        item TEXT,
        chunk_index INTEGER NOT NULL,
        char_start INTEGER NOT NULL,
        char_end INTEGER NOT NULL,
        text_raw TEXT NOT NULL,
        text_masked TEXT NOT NULL,
        text_sha1 TEXT NOT NULL,
        PRIMARY KEY (filing_id, source_file, item, chunk_index)
    )
    """)
    conn.commit()


def upsert_chunks(
    conn: sqlite3.Connection,
    filing_id: str,
    source_file: str,
    text: str,
    chunk_chars: int,
    overlap: int,
) -> int:
    ensure_tables(conn)
    cur = conn.cursor()

    # Build sections (Item-based if possible)
    sections = detect_item_sections(text)
    if sections:
        cur.execute("DELETE FROM narrative_sections WHERE filing_id=? AND source_file=?", (filing_id, source_file))
        for sec in sections:
            cur.execute(
                "INSERT OR REPLACE INTO narrative_sections (filing_id, source_file, item, heading, start_char, end_char) VALUES (?,?,?,?,?,?)",
                (filing_id, source_file, sec.item, sec.heading, sec.start, sec.end),
            )
        conn.commit()
        section_spans = [(sec.item, sec.start, sec.end) for sec in sections]
    else:
        section_spans = [(None, 0, len(text))]

    # Delete existing chunks for this filing/source to keep deterministic regeneration
    cur.execute("DELETE FROM narrative_chunks WHERE filing_id=? AND source_file=?", (filing_id, source_file))

    total = 0
    for item, s0, s1 in section_spans:
        section_text = text[s0:s1]
        raw_chunks = chunk_text(section_text, chunk_chars=chunk_chars, overlap=overlap)
        for idx, (c0, c1, raw) in enumerate(raw_chunks):
            # offsets are within section; convert to doc offsets
            doc_start = s0 + c0
            doc_end = s0 + c1
            masked = mask_numbers(raw)
            h = sha1(masked)
            cur.execute(
                """
                INSERT OR REPLACE INTO narrative_chunks
                  (filing_id, source_file, item, chunk_index, char_start, char_end, text_raw, text_masked, text_sha1)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (filing_id, source_file, item, idx, doc_start, doc_end, raw, masked, h),
            )
            total += 1

    conn.commit()
    return total


def inspect_chunks(conn: sqlite3.Connection, filing_id: str, n: int) -> None:
    cur = conn.cursor()

    # Determine if chunks exist
    cur.execute("""
      SELECT name FROM sqlite_master
      WHERE type='table' AND name='narrative_chunks'
    """)
    if not cur.fetchone():
        print("No table narrative_chunks found. Run extraction first.")
        return

    cur.execute("""
      SELECT item, chunk_index, LENGTH(text_masked) AS len_chars,
             SUBSTR(text_masked, 1, 240) AS preview
      FROM narrative_chunks
      WHERE filing_id=?
      ORDER BY item IS NULL, item, chunk_index
      LIMIT ?
    """, (filing_id, n))
    rows = cur.fetchall()
    if not rows:
        print("No chunks found for filing_id =", filing_id)
        return

    for (item, idx, ln, preview) in rows:
        label = f"Item {item}" if item else "No-Item"
        print("=" * 90)
        print(f"{label} | chunk {idx} | {ln} chars")
        print(textwrap.fill(preview.replace("\n", " "), width=90))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", dest="zip_path", help="Path to SEC XBRL ZIP package")
    ap.add_argument("--db", required=True, help="SQLite DB created by your XBRL ingestion step")
    ap.add_argument("--filing_id", required=True, help="Accession-like id, e.g. 0001633917-26-000024")
    ap.add_argument("--main_html", default=None, help="Main HTML filename inside the ZIP (optional)")
    ap.add_argument("--chunk_chars", type=int, default=3500, help="Chunk size in characters (default 3500)")
    ap.add_argument("--overlap", type=int, default=450, help="Overlap in characters (default 450)")
    ap.add_argument("--inspect", type=int, default=0, help="Print N chunk previews from DB and exit")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        raise FileNotFoundError(f"DB not found: {args.db}")

    conn = sqlite3.connect(args.db)

    if args.inspect and args.inspect > 0:
        inspect_chunks(conn, args.filing_id, args.inspect)
        return

    if not args.zip_path:
        raise ValueError("--zip is required unless --inspect is used")

    source_file, html = read_html_from_zip(args.zip_path, args.main_html)
    text = extract_narrative_text(html)
    n_chunks = upsert_chunks(
        conn=conn,
        filing_id=args.filing_id,
        source_file=source_file,
        text=text,
        chunk_chars=args.chunk_chars,
        overlap=args.overlap,
    )

    print(f"Saved narrative chunks: {n_chunks}")
    print("You can inspect with:")
    print(f"  python narrative_chunker.py --db {args.db} --filing_id {args.filing_id} --inspect 5")


if __name__ == "__main__":
    main()
