from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .narrative import ITEM_MARKER_RE, normalize_heading_text
from .pipeline import connect_db

BENCHMARK_STOPWORDS = {
    "also",
    "about",
    "after",
    "against",
    "analysis",
    "basis",
    "capital",
    "cash",
    "certain",
    "change",
    "changes",
    "company",
    "compared",
    "content",
    "current",
    "detail",
    "details",
    "discussion",
    "during",
    "further",
    "financial",
    "filing",
    "fiscal",
    "generated",
    "group",
    "includes",
    "including",
    "include",
    "increased",
    "income",
    "item",
    "items",
    "management",
    "operations",
    "primarily",
    "quarter",
    "reported",
    "results",
    "resources",
    "revenue",
    "risk",
    "section",
    "see",
    "section",
    "second",
    "statements",
    "subsection",
    "those",
    "this",
    "these",
    "while",
    "year",
}
GENERIC_NOTE_PHRASES = {
    "notes to consolidated financial statements",
    "notes to condensed consolidated financial statements",
    "notes to financial statements",
    "table of contents",
    "index to financial statements",
}
GENERIC_SECTION_PHRASES = {
    "management discussion and analysis of financial condition and results of operations",
    "management s discussion and analysis of financial condition and results of operations",
    "financial statements and supplementary data",
    "condensed consolidated financial statements",
    "risk factors",
    "forward looking statements",
}
COMPANY_SUFFIX_TOKENS = {
    "co",
    "company",
    "corp",
    "corporation",
    "group",
    "holding",
    "holdings",
    "inc",
    "limited",
    "ltd",
    "plc",
    "public",
    "subsidiaries",
    "subsidiary",
}
RETRIEVAL_QUERY_STOPWORDS = {
    "accounts",
    "and",
    "as",
    "by",
    "condensed",
    "consolidated",
    "continued",
    "for",
    "from",
    "its",
    "our",
    "corporation",
    "corporations",
    "corp",
    "expense",
    "expenses",
    "general",
    "inc",
    "index",
    "net",
    "note",
    "notes",
    "quarterly",
    "statement",
    "statements",
    "subsidiaries",
    "subsidiary",
    "table",
    "that",
    "the",
    "their",
    "there",
    "they",
    "was",
    "were",
    "which",
    "with",
}
BENCHMARK_HEADING_ALLOWLIST = {
    "business",
    "financial statements",
    "management's discussion and analysis",
    "management discussion and analysis",
    "risk factors",
}
GENERIC_PARENTHESES_RE = re.compile(
    r"^\(\s*tabular amounts .*?(?:as noted|per share data|ratios).*?\)\s*$",
    flags=re.IGNORECASE,
)
DATE_ONLY_RE = re.compile(
    r"^(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
    flags=re.IGNORECASE,
)
PART_ITEM_RE = re.compile(r"^part\s+[ivx]+\s*item\b", flags=re.IGNORECASE)
PLACEHOLDER_LINES = {"none", "none.", "unaudited", "continued"}


def _import_haystack():
    try:
        from haystack import Document
        from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
        from haystack.document_stores.in_memory import InMemoryDocumentStore
    except (
        ImportError
    ) as error:  # pragma: no cover - exercised when optional dep missing
        raise RuntimeError(
            "Haystack is not installed. Install AuditOps with the retrieval extra."
        ) from error
    return Document, InMemoryBM25Retriever, InMemoryDocumentStore


@dataclass(frozen=True)
class RetrievalExample:
    query: str
    expected_chunk_ids: Sequence[str]
    filing_id: Optional[str] = None
    label: Optional[str] = None


def _stable_text_hash(*parts: Any) -> str:
    import hashlib

    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def load_retrieval_examples(path: str | Path) -> List[RetrievalExample]:
    examples: List[RetrievalExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(
                RetrievalExample(
                    query=row["query"],
                    expected_chunk_ids=tuple(row["expected_chunk_ids"]),
                    filing_id=row.get("filing_id"),
                    label=row.get("label"),
                )
            )
    return examples


def write_retrieval_examples(
    path: str | Path, examples: Sequence[RetrievalExample]
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    {
                        "query": example.query,
                        "expected_chunk_ids": list(example.expected_chunk_ids),
                        "filing_id": example.filing_id,
                        "label": example.label,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")


def _fetch_chunk_rows(
    db_path: str, filing_ids: Optional[Sequence[str]] = None
) -> List[Mapping[str, Any]]:
    conn = connect_db(db_path)
    try:
        query = """
            SELECT
              chunk_evidence_id,
              filing_id,
              period_key,
              item,
              heading,
              subheading,
              heading_path,
              retrieval_text,
              text_masked
            FROM chunk_canon
        """
        params: List[Any] = []
        if filing_ids:
            placeholders = ",".join("?" for _ in filing_ids)
            query += f" WHERE filing_id IN ({placeholders})"
            params.extend(filing_ids)
        query += " ORDER BY filing_id, char_start"
        return [dict(row) for row in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


def _tokenize_for_benchmark(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z&'-]{2,}", text.lower())


def _normalized_alpha_text(text: str) -> str:
    normalized = normalize_heading_text(text)
    lowered = normalized.lower().replace("&", " and ")
    lowered = re.sub(r"[^a-z\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _is_generic_line(text: Optional[str]) -> bool:
    normalized = normalize_heading_text(text or "").strip()
    if not normalized:
        return True
    lowered = normalized.lower()
    alpha_only = _normalized_alpha_text(normalized)
    compact = alpha_only.replace(" ", "")
    if lowered in GENERIC_NOTE_PHRASES or alpha_only in GENERIC_NOTE_PHRASES:
        return True
    if compact.startswith("notesto") and "financialstatements" in compact:
        return True
    if (
        lowered.startswith("notes to ") or alpha_only.startswith("notes to ")
    ) and "financial statements" in alpha_only:
        return True
    if "table of contents" in lowered or "index to financial statements" in alpha_only:
        return True
    if GENERIC_PARENTHESES_RE.match(normalized):
        return True
    if alpha_only in PLACEHOLDER_LINES:
        return True
    if DATE_ONLY_RE.match(alpha_only):
        return True
    if PART_ITEM_RE.match(alpha_only):
        return True
    if alpha_only in GENERIC_SECTION_PHRASES:
        return True
    if ITEM_MARKER_RE.match(normalized):
        return True
    tokens = normalized.split()
    if len(tokens) <= 10 and any(
        token.lower().rstrip(".,") in COMPANY_SUFFIX_TOKENS for token in tokens
    ):
        return True
    return False


def _meaningful_content_lines(text: str) -> List[str]:
    lines: List[str] = []
    seen = set()
    for part in text.split("\n\n"):
        normalized = normalize_heading_text(part.strip())
        if not normalized or _is_generic_line(normalized):
            continue
        dedupe_key = normalized.lower()
        if dedupe_key in seen:
            continue
        lines.append(normalized)
        seen.add(dedupe_key)
    return lines


def _has_cross_section_markers(text: str) -> bool:
    marker_count = 0
    for part in text.split("\n\n"):
        normalized = normalize_heading_text(part.strip())
        if ITEM_MARKER_RE.match(normalized):
            marker_count += 1
            if marker_count >= 2:
                return True
    return False


def _query_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    seen = set()
    for token in _tokenize_for_benchmark(text):
        if (
            token in RETRIEVAL_QUERY_STOPWORDS
            or token in COMPANY_SUFFIX_TOKENS
            or token == "num"
        ):
            continue
        if token not in seen:
            tokens.append(token)
            seen.add(token)
    return tokens


def _normalize_query_text(query: str) -> str:
    tokens = _query_tokens(query)
    if len(tokens) >= 2:
        return " ".join(tokens)
    return " ".join(query.split())


def _query_keywords(row: Mapping[str, Any], *, max_keywords: int = 6) -> List[str]:
    heading_tokens = set(_tokenize_for_benchmark(row.get("heading") or ""))
    subheading_tokens = set(_tokenize_for_benchmark(row.get("subheading") or ""))
    blocked = (
        BENCHMARK_STOPWORDS
        | RETRIEVAL_QUERY_STOPWORDS
        | COMPANY_SUFFIX_TOKENS
        | heading_tokens
        | subheading_tokens
    )

    keywords: List[str] = []
    seen = set()
    source_text = "\n\n".join(
        _meaningful_content_lines(row.get("text_masked") or "")
    ) or (row.get("text_masked") or "")
    for token in _tokenize_for_benchmark(source_text):
        if token in blocked or token == "num":
            continue
        if token not in seen:
            keywords.append(token)
            seen.add(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def _meaningful_title_candidate(row: Mapping[str, Any]) -> Optional[str]:
    subheading = (row.get("subheading") or "").strip()
    if subheading and not _is_generic_line(subheading):
        return subheading

    lines = _meaningful_content_lines(row.get("text_masked") or "")
    for line in lines:
        token_count = len(_tokenize_for_benchmark(line))
        if 2 <= token_count <= 10:
            return line
    return None


def _heading_prefix(row: Mapping[str, Any], label: str) -> Optional[str]:
    heading = normalize_heading_text((row.get("heading") or "").strip())
    if not heading:
        return None
    if label == "footnote_note":
        return "financial statements"
    if label == "mda":
        return "management discussion analysis"
    if label == "risk_factors":
        return "risk factors"
    if heading.lower() in BENCHMARK_HEADING_ALLOWLIST:
        return heading
    return None


def _content_lines_for_row(row: Mapping[str, Any]) -> List[str]:
    content_lines = _meaningful_content_lines(row.get("text_masked") or "")
    if content_lines:
        return content_lines
    fallback = normalize_heading_text((row.get("text_masked") or "").strip())
    return [fallback] if fallback else []


def _build_index_document_text(row: Mapping[str, Any]) -> str:
    lines: List[str] = []

    item = (row.get("item") or "").strip()
    heading = normalize_heading_text((row.get("heading") or "").strip())
    subheading = normalize_heading_text((row.get("subheading") or "").strip())

    if item:
        lines.append(f"Item {item}")
    if heading:
        lines.append(heading)
    if subheading and not _is_generic_line(subheading):
        lines.append(subheading)

    lines.extend(_content_lines_for_row(row))
    return "\n".join(lines)


def _benchmark_label(row: Mapping[str, Any]) -> Optional[str]:
    heading = (row.get("heading") or "").strip().lower()
    if row.get("subheading"):
        if "financial statements" in heading:
            return "footnote_note"
        return "subheading_chunk"
    if heading in BENCHMARK_HEADING_ALLOWLIST:
        if "management" in heading:
            return "mda"
        if "risk" in heading:
            return "risk_factors"
        return "section_chunk"
    return None


def _build_query_from_chunk(row: Mapping[str, Any]) -> Optional[str]:
    label = _benchmark_label(row)
    if label is None:
        return None
    keywords = _query_keywords(row)
    if not keywords:
        return None

    heading = (row.get("heading") or "").strip()
    title_candidate = _meaningful_title_candidate(row)
    heading_prefix = _heading_prefix(row, label)

    if title_candidate and heading_prefix:
        pieces = [heading_prefix, title_candidate, *keywords[:6]]
    elif title_candidate:
        pieces = [title_candidate, *keywords[:6]]
    elif heading_prefix:
        pieces = [heading_prefix, *keywords[:6]]
    elif heading:
        pieces = [heading, *keywords[:6]]
    else:
        pieces = keywords[:5]
    query = " ".join(piece for piece in pieces if piece).strip()
    return query or None


def _is_benchmarkable_row(row: Mapping[str, Any]) -> bool:
    text_masked = row.get("text_masked") or ""
    if _has_cross_section_markers(text_masked):
        return False
    meaningful_lines = _meaningful_content_lines(text_masked)
    if not meaningful_lines:
        return False
    if not row.get("heading"):
        subheading = normalize_heading_text((row.get("subheading") or "").strip())
        if not subheading or _is_generic_line(subheading):
            return False
    return True


def _benchmark_candidate_quality(row: Mapping[str, Any], query: str) -> tuple[int, int]:
    heading = normalize_heading_text((row.get("heading") or "").strip())
    subheading = normalize_heading_text((row.get("subheading") or "").strip())
    meaningful_lines = _meaningful_content_lines(row.get("text_masked") or "")
    query_tokens = _query_tokens(query)

    quality = 0
    if heading:
        quality += 4
    if subheading and not _is_generic_line(subheading):
        quality += 3
    if meaningful_lines:
        quality += min(4, len(_query_tokens(meaningful_lines[0])))
    quality += min(4, len(query_tokens))
    if heading and heading.lower() in BENCHMARK_HEADING_ALLOWLIST:
        quality += 1
    if not heading:
        quality -= 2
    if subheading and DATE_ONLY_RE.match(_normalized_alpha_text(subheading)):
        quality -= 6
    return quality, len(query)


def build_retrieval_benchmark_examples(
    db_path: str,
    *,
    limit: int = 250,
    per_filing_limit: int = 1,
    filing_ids: Optional[Sequence[str]] = None,
) -> List[RetrievalExample]:
    rows = _fetch_chunk_rows(db_path, filing_ids=filing_ids)

    candidates_by_filing: Dict[
        str, List[tuple[tuple[int, int], str, RetrievalExample]]
    ] = {}
    for row in rows:
        if not _is_benchmarkable_row(row):
            continue
        query = _build_query_from_chunk(row)
        label = _benchmark_label(row)
        if not query or not label:
            continue
        example = RetrievalExample(
            query=query,
            expected_chunk_ids=(row["chunk_evidence_id"],),
            filing_id=row["filing_id"],
            label=label,
        )
        quality = _benchmark_candidate_quality(row, query)
        order_key = _stable_text_hash(label, row["filing_id"], row["chunk_evidence_id"])
        candidates_by_filing.setdefault(row["filing_id"], []).append(
            (quality, order_key, example)
        )

    filing_candidates: List[tuple[str, RetrievalExample]] = []
    for filing_id, candidates in candidates_by_filing.items():
        candidates.sort(key=lambda item: (-item[0][0], -item[0][1], item[1]))
        for quality, order_key, example in candidates[:per_filing_limit]:
            filing_candidates.append((order_key, example))

    filing_candidates.sort(key=lambda item: item[0])
    selected: List[RetrievalExample] = []
    seen_queries = set()

    for _, example in filing_candidates:
        filing_id = example.filing_id or ""
        dedupe_key = (filing_id, example.query.lower())
        if dedupe_key in seen_queries:
            continue
        selected.append(example)
        seen_queries.add(dedupe_key)
        if len(selected) >= limit:
            break
    return selected


def build_bm25_retriever(db_path: str, filing_ids: Optional[Sequence[str]] = None):
    rows = _fetch_chunk_rows(db_path, filing_ids=filing_ids)
    return _build_bm25_retriever_from_rows(rows)


def _build_bm25_retriever_from_rows(rows: Sequence[Mapping[str, Any]]):
    Document, InMemoryBM25Retriever, InMemoryDocumentStore = _import_haystack()

    document_store = InMemoryDocumentStore()
    document_store.write_documents(
        [
            (
                lambda content_lines: Document(
                    id=row["chunk_evidence_id"],
                    content=_build_index_document_text(row),
                    meta={
                        "filing_id": row["filing_id"],
                        "period_key": row["period_key"],
                        "item": row["item"],
                        "heading": row["heading"],
                        "subheading": row["subheading"],
                        "heading_path": row["heading_path"],
                        "lead_text": content_lines[0] if content_lines else None,
                    },
                )
            )(_content_lines_for_row(row))
            for row in rows
        ]
    )
    return InMemoryBM25Retriever(document_store=document_store)


def _heading_cue_bonus(query: str, heading: str) -> float:
    query_lower = normalize_heading_text(query).lower()
    heading_lower = normalize_heading_text(heading).lower()
    query_tokens = set(_query_tokens(query))

    if (
        "financial statements" in query_lower
        and "financial statements" in heading_lower
    ):
        return 4.0
    if (
        {"management", "discussion", "analysis"}.issubset(query_tokens)
        and "management" in heading_lower
        and "analysis" in heading_lower
    ):
        return 4.0
    if "risk factors" in query_lower and "risk factors" in heading_lower:
        return 4.0
    if "business" in query_tokens and heading_lower == "business":
        return 2.5
    return 0.0


def _rerank_documents(query: str, documents: Sequence[Any], *, top_k: int) -> List[Any]:
    query_normalized = normalize_heading_text(query)
    query_lower = query_normalized.lower()
    query_tokens = set(_query_tokens(query_normalized))
    ranked: List[tuple[float, float, int, Any]] = []

    for index, document in enumerate(documents):
        meta = getattr(document, "meta", {}) or {}
        heading = normalize_heading_text(meta.get("heading") or "")
        subheading = normalize_heading_text(meta.get("subheading") or "")
        lead_text = normalize_heading_text(meta.get("lead_text") or "")

        heading_tokens = set(_query_tokens(heading))
        subheading_tokens = set(_query_tokens(subheading))
        lead_tokens = set(_query_tokens(lead_text))

        heading_bonus = _heading_cue_bonus(query_normalized, heading)
        exact_subheading_bonus = (
            2.0 if subheading and subheading.lower() in query_lower else 0.0
        )
        subheading_overlap = len(query_tokens & subheading_tokens)
        lead_overlap = len(query_tokens & lead_tokens)
        heading_overlap = len(query_tokens & heading_tokens)
        base_score = float(getattr(document, "score", 0.0) or 0.0)

        rerank_score = (
            heading_bonus
            + exact_subheading_bonus
            + 2.5 * subheading_overlap
            + 1.25 * lead_overlap
            + 0.5 * heading_overlap
        )
        combined_score = rerank_score + 0.2 * base_score
        ranked.append((combined_score, rerank_score, base_score, -index, document))

    ranked.sort(reverse=True)
    return [document for _, _, _, _, document in ranked[:top_k]]


def _retrieve_documents_for_example(
    retriever: Any,
    example: RetrievalExample,
    *,
    top_k: int,
    method: str,
    candidate_k: int,
):
    filters = None
    if example.filing_id:
        filters = {
            "field": "meta.filing_id",
            "operator": "==",
            "value": example.filing_id,
        }
    response = retriever.run(
        query=_normalize_query_text(example.query),
        top_k=top_k if method == "bm25" else candidate_k,
        filters=filters,
    )
    documents = response["documents"]
    if method == "bm25_rerank":
        return _rerank_documents(example.query, documents, top_k=top_k)
    return documents


def evaluate_bm25_retrieval(
    db_path: str,
    examples: Sequence[RetrievalExample],
    *,
    top_k: int = 5,
    method: str = "bm25",
    candidate_k: Optional[int] = None,
) -> Dict[str, Any]:
    if method not in {"bm25", "bm25_rerank"}:
        raise ValueError(f"Unsupported retrieval method: {method}")
    candidate_k = max(top_k, candidate_k or max(10, top_k * 3))

    retriever = None
    filing_retrievers: Dict[str, Any] = {}
    if all(example.filing_id for example in examples):
        filing_ids = sorted(
            {example.filing_id for example in examples if example.filing_id}
        )
        grouped_rows: Dict[str, List[Mapping[str, Any]]] = {
            filing_id: [] for filing_id in filing_ids
        }
        for row in _fetch_chunk_rows(db_path, filing_ids=filing_ids):
            grouped_rows[row["filing_id"]].append(row)
        filing_retrievers = {
            filing_id: _build_bm25_retriever_from_rows(rows)
            for filing_id, rows in grouped_rows.items()
            if rows
        }
    else:
        retriever = build_bm25_retriever(db_path)

    results: List[Dict[str, Any]] = []
    hit_count = 0
    top1_hit_count = 0
    reciprocal_rank_total = 0.0

    for example in examples:
        active_retriever = (
            filing_retrievers.get(example.filing_id) if example.filing_id else retriever
        )
        if active_retriever is None:
            raise ValueError(
                f"No retriever available for filing_id={example.filing_id!r}"
            )
        documents = _retrieve_documents_for_example(
            active_retriever,
            example,
            top_k=top_k,
            method=method,
            candidate_k=candidate_k,
        )
        retrieved_ids = [document.id for document in documents]
        matched_rank = next(
            (
                index + 1
                for index, chunk_id in enumerate(retrieved_ids)
                if chunk_id in set(example.expected_chunk_ids)
            ),
            None,
        )
        hit = matched_rank is not None
        top1_hit = matched_rank == 1
        reciprocal_rank = 0.0 if matched_rank is None else 1.0 / matched_rank

        if hit:
            hit_count += 1
        if top1_hit:
            top1_hit_count += 1
        reciprocal_rank_total += reciprocal_rank

        results.append(
            {
                "query": example.query,
                "label": example.label,
                "filing_id": example.filing_id,
                "expected_chunk_ids": list(example.expected_chunk_ids),
                "retrieved_chunk_ids": retrieved_ids,
                "matched_rank": matched_rank,
                "hit": hit,
                "top1_hit": top1_hit,
                "reciprocal_rank": reciprocal_rank,
            }
        )

    query_count = len(examples)
    summary = {
        "query_count": query_count,
        "top_k": top_k,
        "method": method,
        "candidate_k": candidate_k if method == "bm25_rerank" else top_k,
        "hit_rate_at_k": 0.0 if query_count == 0 else hit_count / query_count,
        "top1_hit_rate": 0.0 if query_count == 0 else top1_hit_count / query_count,
        "mrr_at_k": 0.0 if query_count == 0 else reciprocal_rank_total / query_count,
    }
    return {"summary": summary, "results": results}
