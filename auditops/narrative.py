from __future__ import annotations

import hashlib
import re
import warnings
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

ITEM_RE = re.compile(
    r"^\s*item\s*(\d{1,2}[a]?)\s*(?:[\.\-:]\s*(.*))?$", flags=re.IGNORECASE
)
ITEM_MARKER_RE = re.compile(
    r"^\s*item\s*\d[\dA-Za-z,\s]*(?:[\.\-:]\s*.*)?$", flags=re.IGNORECASE
)
NUM_RE = re.compile(
    r"""
    (?:
        (?<![A-Za-z])
        [\$\u20ac\u00a3]?\s*
        \d{1,3}(?:,\d{3})+(?:\.\d+)?
        (?![A-Za-z])
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

BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "div")
HEADING_CONNECTORS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
PART_RE = re.compile(r"^part\s*[ivx]+$", flags=re.IGNORECASE)
PAGE_ONLY_RE = re.compile(r"^page\s+\d+$", flags=re.IGNORECASE)
STATUS_PAREN_RE = re.compile(r"^\((?:continued|unaudited)\)$", flags=re.IGNORECASE)
FORM_TYPE_RE = re.compile(r"\bform\s+10-[kq]\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class Section:
    item: str
    heading: str
    start: int
    end: int


@dataclass(frozen=True)
class Paragraph:
    start: int
    end: int
    text: str
    subheading: Optional[str]


@dataclass(frozen=True)
class NarrativeChunk:
    item: Optional[str]
    heading: Optional[str]
    subheading: Optional[str]
    heading_path: Optional[str]
    retrieval_text: str
    chunk_index: int
    char_start: int
    char_end: int
    text_raw: str
    text_masked: str
    text_sha1: str


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def pick_main_html_from_zip(zf: zipfile.ZipFile) -> str:
    htmls = [name for name in zf.namelist() if name.lower().endswith((".htm", ".html"))]
    if not htmls:
        raise FileNotFoundError("No .htm/.html found in ZIP")

    scored = []
    for name in htmls:
        try:
            size = zf.getinfo(name).file_size
        except KeyError:
            size = 0
        penalty = 0
        lowered = name.lower()
        if "exhibit" in lowered or lowered.startswith("ex"):
            penalty += 2_000_000
        if "/xbrl" in lowered or "calculation" in lowered or "definition" in lowered:
            penalty += 2_000_000
        scored.append((size - penalty, size, name))

    scored.sort(reverse=True)
    return scored[0][2]


def read_html_from_zip(zip_path: str, main_html: Optional[str]) -> Tuple[str, str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        filename = main_html or pick_main_html_from_zip(zf)
        with zf.open(filename) as handle:
            raw = handle.read()
        for encoding in ("utf-8", "cp1252", "latin-1"):
            try:
                return filename, raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return filename, raw.decode("utf-8", errors="replace")


def _normalize_space(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_heading_text(text: str) -> str:
    text = _normalize_space(text)
    if not text:
        return text

    words = text.split()
    merged: List[str] = []
    index = 0
    while index < len(words):
        if index + 1 < len(words):
            left = words[index]
            right = words[index + 1]
            if left.isupper() and right.isupper():
                if (len(left) == 1 and len(right) > 1) or (
                    len(left) > 3 and 1 < len(right) <= 3
                ):
                    merged.append(left + right)
                    index += 2
                    continue
        merged.append(words[index])
        index += 1
    return " ".join(merged)


def unwrap_inline_xbrl_tags(soup: BeautifulSoup) -> None:
    for tag in list(soup.find_all()):
        name = (tag.name or "").lower()
        if name.startswith("ix:") or name in {
            "nonfraction",
            "nonnumeric",
            "fraction",
            "continuation",
        }:
            tag.unwrap()


def _is_leaf_block(element) -> bool:
    return element.find(BLOCK_TAGS, recursive=False) is None


def extract_narrative_text(html: str) -> str:
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    parser = "lxml-xml" if html.lstrip().startswith("<?xml") else "lxml"
    soup = BeautifulSoup(html, parser)

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.find_all("table"):
        tag.decompose()
    for tag in soup.select('[style*="display:none"], [style*="visibility:hidden"]'):
        tag.decompose()

    unwrap_inline_xbrl_tags(soup)

    body = soup.body or soup
    blocks: List[str] = []
    for element in body.find_all(BLOCK_TAGS):
        if element.name == "div" and not _is_leaf_block(element):
            continue
        text = _normalize_space(element.get_text(" ", strip=True))
        if len(text) <= 2:
            continue
        if is_noise_paragraph(text):
            continue
        blocks.append(text)

    out = "\n\n".join(blocks)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def detect_item_sections(text: str) -> List[Section]:
    lines = text.splitlines(True)
    offsets = []
    current = 0
    for line in lines:
        offsets.append(current)
        current += len(line)

    hits = []
    for index, line in enumerate(lines):
        match = ITEM_RE.match(line.strip())
        if not match:
            continue
        heading = normalize_heading_text((match.group(2) or "").strip())
        hits.append((match.group(1).upper(), heading, offsets[index]))

    if not hits:
        return []

    sections: List[Section] = []
    for index, (item, heading, start) in enumerate(hits):
        end = hits[index + 1][2] if index + 1 < len(hits) else len(text)
        if end - start < 80:
            continue
        sections.append(
            Section(item=item, heading=heading or f"Item {item}", start=start, end=end)
        )
    return sections


def mask_numbers(text: str) -> str:
    return NUM_RE.sub("<NUM>", text)


def split_paragraphs(text: str) -> List[Tuple[int, int, str]]:
    paragraphs: List[Tuple[int, int, str]] = []
    cursor = 0
    size = len(text)
    while cursor < size:
        while cursor < size and text[cursor] == "\n":
            cursor += 1
        if cursor >= size:
            break
        boundary = text.find("\n\n", cursor)
        if boundary == -1:
            boundary = size
        start = cursor
        end = boundary
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            paragraphs.append((start, end, text[start:end]))
        cursor = boundary + 2
    return paragraphs


def looks_like_page_header_footer(text: str) -> bool:
    normalized = normalize_heading_text(text)
    if not normalized:
        return False

    words = normalized.split()
    if len(words) > 14:
        return False

    has_page_counter = bool(
        re.search(r"(?:\||page\s*)\s*\d+\s*$", normalized, flags=re.IGNORECASE)
    )
    if FORM_TYPE_RE.search(normalized) and (
        has_page_counter or "|" in normalized or re.search(r"\b20\d{2}\b", normalized)
    ):
        return True
    if normalized.count("|") >= 2 and has_page_counter:
        return True
    return False


def is_structural_marker(text: str) -> bool:
    normalized = normalize_heading_text(text)
    if not normalized:
        return False

    compact = re.sub(r"\s+", "", normalized)
    if PAGE_ONLY_RE.match(normalized):
        return True
    if STATUS_PAREN_RE.match(normalized):
        return True
    if PART_RE.match(normalized) or PART_RE.match(compact):
        return True
    item_match = ITEM_RE.match(normalized)
    if item_match:
        heading = normalize_heading_text((item_match.group(2) or "").strip())
        if not heading:
            return True
    return False


def is_noise_paragraph(text: str) -> bool:
    normalized = normalize_heading_text(text)
    if not normalized:
        return True
    if looks_like_page_header_footer(normalized):
        return True
    if is_structural_marker(normalized):
        return True
    return False


def is_subheading(text: str) -> bool:
    normalized = normalize_heading_text(text)
    if (
        not normalized
        or is_noise_paragraph(normalized)
        or ITEM_MARKER_RE.match(normalized)
    ):
        return False
    if len(normalized) < 3 or len(normalized) > 120:
        return False
    if normalized.endswith((".", ";", "?", "!")):
        return False

    words = normalized.split()
    if not 1 <= len(words) <= 12:
        return False

    alpha_words = re.findall(r"[A-Za-z][A-Za-z&'/\.-]*", normalized)
    if not alpha_words:
        return False

    connector_ratio = sum(
        1 for word in alpha_words if word.lower() in HEADING_CONNECTORS
    ) / len(alpha_words)
    titleish_ratio = sum(1 for word in alpha_words if word[0].isupper()) / len(
        alpha_words
    )
    uppercase_ratio = sum(
        1 for word in alpha_words if word.isupper() and len(word) > 1
    ) / len(alpha_words)

    if connector_ratio > 0.45 and len(alpha_words) > 4:
        return False
    if len(words) > 8 and uppercase_ratio < 0.6:
        return False
    return titleish_ratio >= 0.8 or uppercase_ratio >= 0.6


def annotate_paragraphs(section_text: str) -> List[Paragraph]:
    annotated: List[Paragraph] = []
    current_subheading: Optional[str] = None
    for start, end, paragraph_text in split_paragraphs(section_text):
        normalized = normalize_heading_text(paragraph_text)
        if is_noise_paragraph(normalized):
            continue
        paragraph_subheading = current_subheading
        if normalized and is_subheading(normalized):
            current_subheading = normalized
            paragraph_subheading = current_subheading
        annotated.append(
            Paragraph(
                start=start,
                end=end,
                text=paragraph_text,
                subheading=paragraph_subheading,
            )
        )
    return annotated


def chunk_paragraphs(
    section_text: str,
    paragraphs: Sequence[Paragraph],
    chunk_chars: int,
    overlap: int,
) -> List[Tuple[int, int, str, Optional[str]]]:
    if not paragraphs:
        return []

    chunks: List[Tuple[int, int, str, Optional[str]]] = []
    start_index = 0
    while start_index < len(paragraphs):
        end_index = start_index
        chunk_start = paragraphs[start_index].start
        while end_index < len(paragraphs):
            proposed_end = paragraphs[end_index].end
            if end_index > start_index and proposed_end - chunk_start > chunk_chars:
                break
            end_index += 1
        if end_index == start_index:
            end_index += 1

        chunk_end = paragraphs[end_index - 1].end
        raw = "\n\n".join(
            paragraph.text.strip()
            for paragraph in paragraphs[start_index:end_index]
            if paragraph.text.strip()
        ).strip()
        subheading = None
        for paragraph in paragraphs[start_index:end_index]:
            if paragraph.subheading:
                subheading = paragraph.subheading
                break
        if subheading is None:
            for prev_index in range(start_index - 1, -1, -1):
                if paragraphs[prev_index].subheading:
                    subheading = paragraphs[prev_index].subheading
                    break

        if raw:
            chunks.append((chunk_start, chunk_end, raw, subheading))

        if end_index >= len(paragraphs):
            break

        next_start = end_index
        overlap_chars = 0
        while next_start > start_index:
            candidate = paragraphs[next_start - 1]
            candidate_len = candidate.end - candidate.start
            if overlap_chars + candidate_len > overlap and next_start < end_index:
                break
            overlap_chars += candidate_len
            next_start -= 1
            if overlap_chars >= overlap:
                break

        start_index = max(start_index + 1, next_start)
    return chunks


def build_heading_path(
    item: Optional[str], heading: Optional[str], subheading: Optional[str]
) -> Optional[str]:
    parts: List[str] = []
    if item:
        parts.append(f"Item {item}")
    if heading:
        parts.append(normalize_heading_text(heading))
    if subheading:
        normalized_subheading = normalize_heading_text(subheading)
        if normalized_subheading and normalized_subheading != normalize_heading_text(
            heading or ""
        ):
            parts.append(normalized_subheading)
    return " > ".join(parts) if parts else None


def build_retrieval_text(
    item: Optional[str],
    heading: Optional[str],
    subheading: Optional[str],
    text_masked: str,
) -> str:
    lines: List[str] = []
    if item:
        lines.append(f"Item: {item}")
    if heading:
        lines.append(f"Section: {normalize_heading_text(heading)}")
    if subheading:
        normalized_subheading = normalize_heading_text(subheading)
        if normalized_subheading and normalized_subheading != normalize_heading_text(
            heading or ""
        ):
            lines.append(f"Subsection: {normalized_subheading}")
    lines.append("Content:")
    lines.append(text_masked)
    return "\n".join(lines)


def build_narrative_chunks(
    text: str, chunk_chars: int = 3500, overlap: int = 450
) -> Tuple[Sequence[Section], Sequence[NarrativeChunk]]:
    sections = detect_item_sections(text)
    spans = [
        (section.item, section.heading, section.start, section.end)
        for section in sections
    ] or [(None, None, 0, len(text))]

    chunks: List[NarrativeChunk] = []
    for item, heading, start, end in spans:
        section_text = text[start:end]
        paragraphs = annotate_paragraphs(section_text)
        raw_chunks = chunk_paragraphs(
            section_text, paragraphs, chunk_chars=chunk_chars, overlap=overlap
        )
        for index, (chunk_start, chunk_end, raw, subheading) in enumerate(raw_chunks):
            doc_start = start + chunk_start
            doc_end = start + chunk_end
            masked = mask_numbers(raw)
            heading_path = build_heading_path(item, heading, subheading)
            retrieval_text = build_retrieval_text(item, heading, subheading, masked)
            chunks.append(
                NarrativeChunk(
                    item=item,
                    heading=heading,
                    subheading=subheading,
                    heading_path=heading_path,
                    retrieval_text=retrieval_text,
                    chunk_index=index,
                    char_start=doc_start,
                    char_end=doc_end,
                    text_raw=raw,
                    text_masked=masked,
                    text_sha1=sha1(masked),
                )
            )
    return sections, chunks
