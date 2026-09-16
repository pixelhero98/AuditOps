from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .narrative import normalize_heading_text
from .pipeline import _stable_digest, connect_db
from .retrieval import (
    RetrievalExample,
    _benchmark_candidate_quality,
    _benchmark_label,
    _build_bm25_retriever_from_rows,
    _build_query_from_chunk,
    _fetch_chunk_rows,
    _is_benchmarkable_row,
    _meaningful_content_lines,
    _normalize_query_text,
    _query_tokens,
    _retrieve_documents_for_example,
)

NARRATIVE_TASK_SPEC_VERSION = "v1"
NARRATIVE_TASK_SPEC_SCHEMA_ID = "narrative_task_spec.v1"
NARRATIVE_EXTRACTION_POLICY_VERSION = "source_paragraph_v3"
NARRATIVE_TASK_TYPE = "narrative_citation"
NARRATIVE_ANSWER_VERSION = "v1"
NARRATIVE_ANSWER_SCHEMA_ID = "narrative_structured_answer.v1"
NARRATIVE_REFUSAL_CODE = "NARRATIVE_NOT_SUPPORTED"
NARRATIVE_ROUTING_REFUSAL_CODE = "TASK_NOT_SUPPORTED"
NARRATIVE_CITATION_MISS_CODE = "NARRATIVE_CITATION_MISS"
NARRATIVE_RETRIEVAL_METHOD = "bm25_rerank"
NARRATIVE_RETRIEVAL_TOP_K = 5
NARRATIVE_RETRIEVAL_CANDIDATE_K = 15
_ANSWER_MAX_CHARS = 320
_MIN_UNANSWERABLE_UNSEEN_TOKENS = 3
_MIN_UNSUPPORTED_ATTRIBUTE_TOKENS = 2
_EXCLUDED_NARRATIVE_PHRASES = {
    "assessment of internal control over financial reporting",
    "definition and limitations of internal control over financial reporting",
    "forward-looking statements",
    "independent registered public accounting firm",
    "internal control over financial reporting",
    "index to consolidated financial statements",
    "liquidity and capital resources",
    "management report on internal control over financial reporting",
    "management's report on internal control over financial reporting",
    "opinion on internal control over financial reporting",
    "report of independent registered public accounting firm",
    "we have audited the accompanying",
}
_ANSWERABILITY_OPTIONS = {"ANSWERABLE", "UNANSWERABLE"}
_NARRATIVE_SCOPE_OPTIONS = {"filing", "note"}
_ROUTING_FOCUS_STOPWORDS = {
    "about",
    "accounting",
    "condensed",
    "consolidated",
    "data",
    "disclose",
    "disclosed",
    "does",
    "financial",
    "in",
    "is",
    "item",
    "note",
    "notes",
    "say",
    "statements",
    "tell",
    "the",
    "what",
}
_ROUTING_MIN_SCORE = 2.0
_ROUTING_AMBIGUITY_MARGIN = 0.15
_SAFE_NARRATIVE_REFUSAL_CODES = {
    NARRATIVE_REFUSAL_CODE,
    NARRATIVE_ROUTING_REFUSAL_CODE,
    NARRATIVE_CITATION_MISS_CODE,
}
_UNANSWERABLE_NEGATIVE_TYPE_ORDER = (
    "same_filing_wrong_note",
    "same_issuer_wrong_period",
    "unsupported_attribute",
    "cross_label_query_transfer",
    "cross_filing_query_transfer",
)


def _iter_candidate_rows(db_path: str, filing_ids: Optional[Sequence[str]] = None):
    conn = connect_db(db_path)
    try:
        query = """
            SELECT
              c.chunk_evidence_id,
              c.filing_id,
              c.period_key,
              c.item,
              c.heading,
              c.subheading,
              c.heading_path,
              c.retrieval_text,
              c.text_masked,
              f.ticker,
              f.form_type
            FROM chunk_canon c
            JOIN filings f ON f.filing_id = c.filing_id
        """
        params: List[Any] = []
        if filing_ids:
            placeholders = ",".join("?" for _ in filing_ids)
            query += f" WHERE c.filing_id IN ({placeholders})"
            params.extend(filing_ids)
        query += " ORDER BY c.filing_id, c.char_start"
        for row in conn.execute(query, params):
            yield dict(row)
    finally:
        conn.close()


def _truncate_text(text: str, *, max_chars: int = _ANSWER_MAX_CHARS) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cutoff = text.rfind(" ", 0, max_chars)
    if cutoff <= 0:
        cutoff = max_chars
    return text[:cutoff].rstrip()


def _extractive_answer(row: Mapping[str, Any]) -> Optional[str]:
    source_text = str(row.get("text_masked") or "")
    lines = _meaningful_content_lines(source_text)
    if not lines:
        return None
    # ``_meaningful_content_lines`` is useful for filtering navigation and
    # boilerplate, but its heading normalizer may join adjacent uppercase
    # tokens.  Map the chosen normalized line back to its original paragraph
    # so the answer preserves the filing's words and punctuation exactly,
    # modulo whitespace collapse.
    selected = lines[0]
    for paragraph in source_text.split("\n\n"):
        if normalize_heading_text(paragraph.strip()) != selected:
            continue
        answer = re.sub(r"\s+", " ", paragraph).strip()
        return _truncate_text(answer) if answer else None
    return None


def _narrative_label(row: Mapping[str, Any]) -> Optional[str]:
    benchmark_label = _benchmark_label(row)
    if benchmark_label != "footnote_note":
        return None
    combined_text = " ".join(
        part
        for part in [
            row.get("heading") or "",
            row.get("subheading") or "",
            row.get("heading_path") or "",
            row.get("retrieval_text") or "",
        ]
        if part
    ).lower()
    if any(phrase in combined_text for phrase in _EXCLUDED_NARRATIVE_PHRASES):
        return None
    if "accounting polic" in combined_text:
        return "accounting_policy"
    return "footnote_note"


def _is_excluded_narrative_candidate(
    row: Mapping[str, Any], extractive_answer: str
) -> bool:
    combined_text = " ".join(
        part
        for part in [
            row.get("heading") or "",
            row.get("subheading") or "",
            row.get("heading_path") or "",
            row.get("retrieval_text") or "",
            extractive_answer,
        ]
        if part
    ).lower()
    return any(phrase in combined_text for phrase in _EXCLUDED_NARRATIVE_PHRASES)


def _scope_type_for_label(label: str) -> str:
    if label in {"footnote_note", "accounting_policy"}:
        return "note"
    return "filing"


def _scope_key_for_parts(
    *,
    scope_type: str,
    filing_id: str,
    item: Optional[str],
    heading: Optional[str],
    subheading: Optional[str],
    heading_path: Optional[str],
) -> Optional[str]:
    if scope_type != "note":
        return None
    if heading_path:
        return f"note:{filing_id}:{normalize_heading_text(heading_path).lower()}"
    parts = [
        normalize_heading_text(part).lower()
        for part in (item or "", heading or "", subheading or "")
        if part
    ]
    return (
        f"note:{filing_id}:{'|'.join(parts)}" if parts else f"note:{filing_id}:unknown"
    )


def _scope_key_for_task_spec(task_spec: Mapping[str, Any]) -> Optional[str]:
    scope_type = str(task_spec.get("scope_type") or "filing")
    scope_key = task_spec.get("scope_key")
    if scope_key or scope_type != "note":
        return scope_key
    return _scope_key_for_parts(
        scope_type=scope_type,
        filing_id=str(task_spec["filing_id"]),
        item=task_spec.get("item"),
        heading=task_spec.get("heading"),
        subheading=task_spec.get("subheading"),
        heading_path=task_spec.get("heading_path"),
    )


def _scope_key_for_row(row: Mapping[str, Any], *, scope_type: str) -> Optional[str]:
    return _scope_key_for_parts(
        scope_type=scope_type,
        filing_id=str(row["filing_id"]),
        item=row.get("item"),
        heading=row.get("heading"),
        subheading=row.get("subheading"),
        heading_path=row.get("heading_path"),
    )


def _base_narrative_task_spec(
    *,
    task_id: str,
    filing_id: str,
    ticker: str,
    form_type: str,
    period_key: Optional[str],
    item: Optional[str],
    heading: Optional[str],
    subheading: Optional[str],
    heading_path: Optional[str],
    question: str,
    retrieval_query: str,
    label: str,
    scope_type: str,
    scope_key: Optional[str],
    answerability: str,
    expected_chunk_ids: Sequence[str],
    extractive_answer: Optional[str],
    refusal_code: Optional[str],
    negative_type: Optional[str] = None,
    donor_task_id: Optional[str] = None,
) -> Dict[str, Any]:
    template_variant = int(_stable_digest("narrative-template", task_id)[:8], 16) % 2
    status = answerability.casefold()
    template_family = negative_type or label
    return {
        "narrative_task_spec_version": NARRATIVE_TASK_SPEC_VERSION,
        "narrative_task_schema_id": NARRATIVE_TASK_SPEC_SCHEMA_ID,
        "task_id": task_id,
        "task_type": NARRATIVE_TASK_TYPE,
        "template_id": (f"narrative:{status}:{template_family}:{template_variant}:v1"),
        "task_family": f"narrative_citation:{label}:{status}",
        "filing_id": filing_id,
        "ticker": ticker,
        "form_type": form_type,
        "period_key": period_key,
        "item": item,
        "heading": heading,
        "subheading": subheading,
        "heading_path": heading_path,
        "question": question,
        "retrieval_query": retrieval_query,
        "label": label,
        "scope_type": scope_type,
        "scope_key": scope_key,
        "answerability": answerability,
        "expected_chunk_ids": list(expected_chunk_ids),
        "extractive_answer": extractive_answer,
        "refusal_code": refusal_code,
        "negative_type": negative_type,
        "donor_task_id": donor_task_id,
        "citation_policy": {
            "require_chunk_evidence_ids": answerability == "ANSWERABLE",
            "retrieval_method": NARRATIVE_RETRIEVAL_METHOD,
            "top_k": NARRATIVE_RETRIEVAL_TOP_K,
            "candidate_k": NARRATIVE_RETRIEVAL_CANDIDATE_K,
        },
    }


def validate_narrative_task_spec(task_spec: Mapping[str, Any]) -> None:
    required_fields = {
        "narrative_task_spec_version",
        "narrative_task_schema_id",
        "task_id",
        "task_type",
        "filing_id",
        "ticker",
        "form_type",
        "period_key",
        "item",
        "heading",
        "subheading",
        "heading_path",
        "question",
        "retrieval_query",
        "label",
        "scope_type",
        "scope_key",
        "answerability",
        "expected_chunk_ids",
        "extractive_answer",
        "refusal_code",
        "citation_policy",
    }
    missing = sorted(required_fields - set(task_spec))
    if missing:
        raise ValueError(
            f"Narrative TaskSpec is missing required fields: {', '.join(missing)}"
        )
    if task_spec["narrative_task_spec_version"] != NARRATIVE_TASK_SPEC_VERSION:
        raise ValueError(
            f"Unsupported narrative task spec version: {task_spec['narrative_task_spec_version']}"
        )
    if task_spec["narrative_task_schema_id"] != NARRATIVE_TASK_SPEC_SCHEMA_ID:
        raise ValueError(
            f"Unsupported narrative task schema id: {task_spec['narrative_task_schema_id']}"
        )
    if task_spec["task_type"] != NARRATIVE_TASK_TYPE:
        raise ValueError(f"Unsupported narrative task type: {task_spec['task_type']}")
    if task_spec["scope_type"] not in _NARRATIVE_SCOPE_OPTIONS:
        raise ValueError(f"Unsupported narrative scope type: {task_spec['scope_type']}")
    if task_spec["answerability"] not in _ANSWERABILITY_OPTIONS:
        raise ValueError(
            f"Unsupported narrative answerability: {task_spec['answerability']}"
        )
    if task_spec["answerability"] == "ANSWERABLE":
        if not task_spec["expected_chunk_ids"]:
            raise ValueError(
                "Answerable narrative TaskSpecs must include expected_chunk_ids"
            )
        if not task_spec["extractive_answer"]:
            raise ValueError(
                "Answerable narrative TaskSpecs must include a non-empty extractive_answer"
            )
        if task_spec["refusal_code"] is not None:
            raise ValueError("Answerable narrative TaskSpecs cannot set refusal_code")
    else:
        if task_spec["expected_chunk_ids"]:
            raise ValueError(
                "Unanswerable narrative TaskSpecs must not include expected_chunk_ids"
            )
        if task_spec["extractive_answer"] is not None:
            raise ValueError(
                "Unanswerable narrative TaskSpecs must not include extractive_answer"
            )
        if task_spec["refusal_code"] is None:
            raise ValueError(
                "Unanswerable narrative TaskSpecs must include refusal_code"
            )


def _update_filing_token_sets(
    token_sets: Dict[str, set[str]],
    row: Mapping[str, Any],
) -> None:
    filing_tokens = token_sets.setdefault(str(row["filing_id"]), set())
    filing_tokens.update(_query_tokens(row.get("retrieval_text") or ""))
    filing_tokens.update(_query_tokens(row.get("heading") or ""))
    filing_tokens.update(_query_tokens(row.get("subheading") or ""))


def _select_answerable_candidates_for_filing(
    filing_id: str,
    candidates: Sequence[tuple[tuple[int, int], str, Dict[str, Any]]],
    *,
    selection_limit: int,
) -> List[tuple[str, Dict[str, Any]]]:
    ranked = sorted(candidates, key=lambda item: (-item[0][0], -item[0][1], item[1]))
    return [
        (f"{filing_id}:{order_key}", task_spec)
        for _, order_key, task_spec in ranked[:selection_limit]
    ]


def _build_answerable_narrative_tasks(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int,
    per_filing_limit: int,
) -> tuple[List[Dict[str, Any]], Dict[str, set[str]], List[Dict[str, Any]]]:
    filing_candidates: List[tuple[str, Dict[str, Any]]] = []
    donor_candidates: List[tuple[str, Dict[str, Any]]] = []
    filing_token_sets: Dict[str, set[str]] = {}
    current_filing_id: Optional[str] = None
    current_candidates: List[tuple[tuple[int, int], str, Dict[str, Any]]] = []

    for row in rows:
        filing_id = str(row["filing_id"])
        _update_filing_token_sets(filing_token_sets, row)
        if current_filing_id is None:
            current_filing_id = filing_id
        elif filing_id != current_filing_id:
            filing_candidates.extend(
                _select_answerable_candidates_for_filing(
                    current_filing_id,
                    current_candidates,
                    selection_limit=per_filing_limit,
                )
            )
            donor_candidates.extend(
                _select_answerable_candidates_for_filing(
                    current_filing_id,
                    current_candidates,
                    selection_limit=max(per_filing_limit + 2, 3),
                )
            )
            current_filing_id = filing_id
            current_candidates = []

        if not _is_benchmarkable_row(row):
            continue
        label = _narrative_label(row)
        if label is None:
            continue
        question = _build_query_from_chunk(row)
        extractive_answer = _extractive_answer(row)
        if not question or not extractive_answer:
            continue
        if _is_excluded_narrative_candidate(row, extractive_answer):
            continue
        task_spec = _base_narrative_task_spec(
            task_id=_stable_digest(
                "narrative-task",
                row["filing_id"],
                row["chunk_evidence_id"],
                NARRATIVE_TASK_SPEC_VERSION,
                NARRATIVE_EXTRACTION_POLICY_VERSION,
            ),
            filing_id=row["filing_id"],
            ticker=row["ticker"],
            form_type=row["form_type"],
            period_key=row["period_key"],
            item=row["item"],
            heading=row["heading"],
            subheading=row["subheading"],
            heading_path=row["heading_path"],
            question=question,
            retrieval_query=_normalize_query_text(question),
            label=label,
            scope_type=_scope_type_for_label(label),
            scope_key=_scope_key_for_row(row, scope_type=_scope_type_for_label(label)),
            answerability="ANSWERABLE",
            expected_chunk_ids=[row["chunk_evidence_id"]],
            extractive_answer=extractive_answer,
            refusal_code=None,
        )
        validate_narrative_task_spec(task_spec)
        quality = _benchmark_candidate_quality(row, question)
        order_key = row["chunk_evidence_id"]
        current_candidates.append((quality, order_key, task_spec))

    if current_filing_id is not None:
        filing_candidates.extend(
            _select_answerable_candidates_for_filing(
                current_filing_id,
                current_candidates,
                selection_limit=per_filing_limit,
            )
        )
        donor_candidates.extend(
            _select_answerable_candidates_for_filing(
                current_filing_id,
                current_candidates,
                selection_limit=max(per_filing_limit + 2, 3),
            )
        )

    selected: List[Dict[str, Any]] = []
    seen_queries = set()
    filing_candidates.sort(key=lambda item: item[0])
    for _, task_spec in filing_candidates:
        dedupe_key = (task_spec["filing_id"], task_spec["question"].lower())
        if dedupe_key in seen_queries:
            continue
        selected.append(task_spec)
        seen_queries.add(dedupe_key)
        if len(selected) >= limit:
            break
    filtered_token_sets = {
        filing_id: filing_token_sets.get(str(filing_id), set())
        for filing_id in {task_spec["filing_id"] for task_spec in selected}
    }
    donor_task_specs = [
        task_spec for _, task_spec in sorted(donor_candidates, key=lambda item: item[0])
    ]
    return selected, filtered_token_sets, donor_task_specs


def _task_sort_key(task_spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(task_spec["filing_id"]), str(task_spec["task_id"])


def _distinctive_query_tokens(
    task_spec: Mapping[str, Any], target_filing_tokens: set[str]
) -> List[str]:
    return [
        token
        for token in _query_tokens(task_spec.get("retrieval_query") or "")
        if token not in target_filing_tokens
    ]


def _scope_title(task_spec: Mapping[str, Any]) -> str:
    for value in (
        task_spec.get("subheading"),
        task_spec.get("heading_path"),
        task_spec.get("heading"),
        task_spec.get("item"),
    ):
        normalized = normalize_heading_text(str(value or "").strip())
        if normalized:
            return normalized
    return "this note"


def _routing_focus_terms(
    task_spec: Mapping[str, Any], *, max_terms: int = 6
) -> List[str]:
    scope_tokens = set(_query_tokens(_scope_title(task_spec)))
    candidate_tokens = [
        token
        for token in _query_tokens(
            str(task_spec.get("retrieval_query") or task_spec.get("question") or "")
        )
        if token not in _ROUTING_FOCUS_STOPWORDS and token not in scope_tokens
    ]
    if not candidate_tokens:
        candidate_tokens = [
            token for token in scope_tokens if token not in _ROUTING_FOCUS_STOPWORDS
        ]
    return candidate_tokens[:max_terms]


def _variant_question_body(task_spec: Mapping[str, Any]) -> str:
    question = str(task_spec.get("question") or "").strip()
    if not question:
        return "this disclosure"
    body = re.sub(
        r"^for this period,\s*does the .*? note discuss\s*",
        "",
        question,
        flags=re.IGNORECASE,
    )
    if body != question:
        body = body.rstrip(" ?")
        return body or "this disclosure"
    body = re.sub(r"^in the .*? note,\s*", "", question, flags=re.IGNORECASE)
    body = re.sub(r"^does the .*? note discuss\s*", "", body, flags=re.IGNORECASE)
    body = body.rstrip(" ?")
    return body or "this disclosure"


def _build_narrative_variant_question(
    task_spec: Mapping[str, Any], variant_index: int
) -> str:
    scope_title = _scope_title(task_spec)
    focus_phrase = " ".join(_routing_focus_terms(task_spec)) or scope_title
    refusal_body = _variant_question_body(task_spec)
    templates = (
        f"What does the {scope_title} note say about {focus_phrase}?",
        f"In the {scope_title} note, what is disclosed about {focus_phrase}?",
    )
    refusal_templates = (
        f"Does the {scope_title} note discuss {refusal_body}?",
        f"In the {scope_title} note, is anything disclosed about {refusal_body}?",
    )
    same_period_refusal_templates = (
        f"For this period, does the {scope_title} note discuss {refusal_body}?",
        f"In this period's {scope_title} note, is anything disclosed about {refusal_body}?",
    )
    if task_spec["answerability"] == "ANSWERABLE":
        active_templates = templates
    elif task_spec.get("negative_type") == "same_issuer_wrong_period":
        active_templates = same_period_refusal_templates
    else:
        active_templates = refusal_templates
    return active_templates[variant_index % len(active_templates)]


def _unsupported_attribute_phrase(
    source_task: Mapping[str, Any],
    donor_task: Mapping[str, Any],
    source_tokens: set[str],
) -> Optional[str]:
    source_scope_tokens = set(_query_tokens(_scope_title(source_task)))
    donor_scope = _scope_title(donor_task)
    donor_scope_tokens = [
        token
        for token in _query_tokens(donor_scope)
        if token not in source_tokens and token not in source_scope_tokens
    ]
    donor_focus_tokens = [
        token
        for token in _routing_focus_terms(donor_task, max_terms=6)
        if token not in source_tokens and token not in source_scope_tokens
    ]
    distinctive_tokens = (
        donor_scope_tokens[:3]
        + [token for token in donor_focus_tokens if token not in donor_scope_tokens][:3]
    )
    if len(distinctive_tokens) < _MIN_UNSUPPORTED_ATTRIBUTE_TOKENS:
        distinctive_tokens = _distinctive_query_tokens(donor_task, source_tokens)[:4]
    if len(distinctive_tokens) < _MIN_UNSUPPORTED_ATTRIBUTE_TOKENS:
        return None
    return " ".join(distinctive_tokens[:4])


def _cross_filing_transfer_phrase(
    source_task: Mapping[str, Any],
    donor_task: Mapping[str, Any],
    source_tokens: set[str],
) -> Optional[str]:
    source_scope_tokens = set(_query_tokens(_scope_title(source_task)))
    donor_focus_tokens = [
        token
        for token in _routing_focus_terms(donor_task, max_terms=8)
        if token not in source_scope_tokens
    ]
    donor_distinctive_tokens = [
        token
        for token in _distinctive_query_tokens(donor_task, source_tokens)
        if token not in source_scope_tokens
    ]
    phrase_tokens = donor_focus_tokens + [
        token for token in donor_distinctive_tokens if token not in donor_focus_tokens
    ]
    if len(phrase_tokens) < _MIN_UNANSWERABLE_UNSEEN_TOKENS:
        return None
    return " ".join(phrase_tokens[:6])


def _same_issuer_wrong_period_phrase(
    source_task: Mapping[str, Any],
    donor_task: Mapping[str, Any],
    source_tokens: set[str],
) -> Optional[str]:
    source_scope_tokens = set(_query_tokens(_scope_title(source_task)))
    donor_focus_tokens = [
        token
        for token in _routing_focus_terms(donor_task, max_terms=8)
        if token not in source_scope_tokens
    ]
    donor_distinctive_tokens = [
        token
        for token in _distinctive_query_tokens(donor_task, source_tokens)
        if token not in source_scope_tokens
    ]
    phrase_tokens = donor_focus_tokens + [
        token for token in donor_distinctive_tokens if token not in donor_focus_tokens
    ]
    if len(phrase_tokens) < _MIN_UNANSWERABLE_UNSEEN_TOKENS:
        return None
    return " ".join(phrase_tokens[:6])


def build_narrative_routing_variants(
    task_specs: Sequence[Mapping[str, Any]],
    *,
    variants_per_task: int = 1,
) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    for task_spec in _dedupe_narrative_task_specs(task_specs):
        for variant_index in range(max(1, variants_per_task)):
            question = _build_narrative_variant_question(task_spec, variant_index)
            variants.append(
                {
                    "variant_id": _stable_digest(
                        "narrative-variant", task_spec["task_id"], variant_index
                    ),
                    "source_task_id": task_spec["task_id"],
                    "variant_index": variant_index,
                    "filing_id": task_spec["filing_id"],
                    "ticker": task_spec["ticker"],
                    "answerability": task_spec["answerability"],
                    "negative_type": task_spec.get("negative_type"),
                    "question": question,
                    "scope_type": task_spec.get("scope_type"),
                    "scope_key": _scope_key_for_task_spec(task_spec),
                    "expected_chunk_ids": list(
                        task_spec.get("expected_chunk_ids") or []
                    ),
                    "expected_answer_text": task_spec.get("extractive_answer"),
                    "expected_refusal_code": task_spec.get("refusal_code"),
                }
            )
    return variants


def _append_unanswerable_candidate(
    candidate_pools: Dict[str, List[Dict[str, Any]]],
    seen_candidate_keys: set[tuple[str, str]],
    filing_query_keys: Mapping[str, set[str]],
    source_task: Mapping[str, Any],
    *,
    question: str,
    retrieval_query: str,
    negative_type: str,
    donor_task_id: Optional[str],
) -> bool:
    normalized_query = _normalize_question_key(question)
    candidate_key = (str(source_task["filing_id"]), normalized_query)
    if candidate_key in seen_candidate_keys:
        return False
    if normalized_query in filing_query_keys.get(str(source_task["filing_id"]), set()):
        return False
    unanswerable_task = _base_narrative_task_spec(
        task_id=_stable_digest(
            "narrative-task",
            source_task["filing_id"],
            normalized_query,
            negative_type,
            donor_task_id or "self",
            NARRATIVE_TASK_SPEC_VERSION,
        ),
        filing_id=source_task["filing_id"],
        ticker=source_task["ticker"],
        form_type=source_task["form_type"],
        period_key=source_task["period_key"],
        item=source_task["item"],
        heading=source_task["heading"],
        subheading=source_task["subheading"],
        heading_path=source_task["heading_path"],
        question=question,
        retrieval_query=retrieval_query,
        label=source_task["label"],
        scope_type=str(source_task.get("scope_type") or "filing"),
        scope_key=_scope_key_for_task_spec(source_task),
        answerability="UNANSWERABLE",
        expected_chunk_ids=[],
        extractive_answer=None,
        refusal_code=NARRATIVE_REFUSAL_CODE,
        negative_type=negative_type,
        donor_task_id=donor_task_id,
    )
    validate_narrative_task_spec(unanswerable_task)
    candidate_pools.setdefault(negative_type, []).append(unanswerable_task)
    seen_candidate_keys.add(candidate_key)
    return True


def _build_unanswerable_narrative_tasks(
    answerable_task_specs: Sequence[Mapping[str, Any]],
    filing_token_sets: Mapping[str, set[str]],
    *,
    limit: int,
    donor_task_specs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    donor_universe = donor_task_specs or answerable_task_specs
    by_label: Dict[str, List[Mapping[str, Any]]] = {}
    by_ticker: Dict[str, List[Mapping[str, Any]]] = {}
    by_filing: Dict[str, List[Mapping[str, Any]]] = {}
    for task_spec in donor_universe:
        by_label.setdefault(task_spec["label"], []).append(task_spec)
        by_ticker.setdefault(str(task_spec["ticker"]), []).append(task_spec)
        by_filing.setdefault(str(task_spec["filing_id"]), []).append(task_spec)

    ordered_tasks = sorted(answerable_task_specs, key=_task_sort_key)
    for donor_tasks in by_label.values():
        donor_tasks.sort(key=_task_sort_key)
    for donor_tasks in by_ticker.values():
        donor_tasks.sort(key=_task_sort_key)
    for donor_tasks in by_filing.values():
        donor_tasks.sort(key=_task_sort_key)

    filing_query_keys: Dict[str, set[str]] = {}
    for task_spec in answerable_task_specs:
        filing_query_keys.setdefault(str(task_spec["filing_id"]), set()).add(
            _normalize_question_key(str(task_spec["question"]))
        )

    candidate_pools: Dict[str, List[Dict[str, Any]]] = {
        negative_type: [] for negative_type in _UNANSWERABLE_NEGATIVE_TYPE_ORDER
    }
    seen_candidate_keys: set[tuple[str, str]] = set()

    for source_task in ordered_tasks:
        source_filing_id = str(source_task["filing_id"])
        source_tokens = filing_token_sets.get(source_filing_id, set())
        source_scope_key = _scope_key_for_task_spec(source_task)

        same_filing_donors = sorted(
            (
                donor_task
                for donor_task in by_filing.get(source_filing_id, [])
                if donor_task["task_id"] != source_task["task_id"]
                and _scope_key_for_task_spec(donor_task) != source_scope_key
            ),
            key=_task_sort_key,
        )
        for donor_task in same_filing_donors:
            source_scope_title = _scope_title(source_task)
            donor_question = str(donor_task["question"]).rstrip(" ?")
            donor_retrieval_query = str(donor_task["retrieval_query"]).strip()
            question = f"In the {source_scope_title} note, {donor_question}?"
            retrieval_query = _normalize_query_text(
                f"{source_scope_title} {donor_retrieval_query}"
            )
            if _append_unanswerable_candidate(
                candidate_pools,
                seen_candidate_keys,
                filing_query_keys,
                source_task,
                question=question,
                retrieval_query=retrieval_query,
                negative_type="same_filing_wrong_note",
                donor_task_id=str(donor_task["task_id"]),
            ):
                break

        same_issuer_donors = [
            donor_task
            for donor_task in by_ticker.get(str(source_task["ticker"]), [])
            if donor_task["filing_id"] != source_task["filing_id"]
            and donor_task["period_key"] != source_task["period_key"]
        ]
        for donor_task in same_issuer_donors:
            period_phrase = _same_issuer_wrong_period_phrase(
                source_task, donor_task, source_tokens
            )
            if not period_phrase:
                continue
            source_scope_title = _scope_title(source_task)
            question = f"For this period, does the {source_scope_title} note discuss {period_phrase}?"
            retrieval_query = _normalize_query_text(
                f"{source_scope_title} {period_phrase}"
            )
            if _append_unanswerable_candidate(
                candidate_pools,
                seen_candidate_keys,
                filing_query_keys,
                source_task,
                question=question,
                retrieval_query=retrieval_query,
                negative_type="same_issuer_wrong_period",
                donor_task_id=str(donor_task["task_id"]),
            ):
                break

        attribute_donors = sorted(
            (
                donor_task
                for donor_task in donor_universe
                if donor_task["filing_id"] != source_task["filing_id"]
                and donor_task["label"] == source_task["label"]
            ),
            key=_task_sort_key,
        )
        for donor_task in attribute_donors:
            attribute_phrase = _unsupported_attribute_phrase(
                source_task, donor_task, source_tokens
            )
            if not attribute_phrase:
                continue
            source_scope_title = _scope_title(source_task)
            question = f"Does the {source_scope_title} note discuss {attribute_phrase}?"
            retrieval_query = _normalize_query_text(
                f"{source_scope_title} {attribute_phrase}"
            )
            if _append_unanswerable_candidate(
                candidate_pools,
                seen_candidate_keys,
                filing_query_keys,
                source_task,
                question=question,
                retrieval_query=retrieval_query,
                negative_type="unsupported_attribute",
                donor_task_id=str(donor_task["task_id"]),
            ):
                break

        cross_label_donors = sorted(
            (
                donor_task
                for donor_task in donor_universe
                if donor_task["filing_id"] != source_task["filing_id"]
                and donor_task["label"] != source_task["label"]
            ),
            key=_task_sort_key,
        )
        for donor_task in cross_label_donors:
            donor_tokens = _distinctive_query_tokens(donor_task, source_tokens)
            if len(donor_tokens) < _MIN_UNANSWERABLE_UNSEEN_TOKENS:
                continue
            if _append_unanswerable_candidate(
                candidate_pools,
                seen_candidate_keys,
                filing_query_keys,
                source_task,
                question=str(donor_task["question"]),
                retrieval_query=str(donor_task["retrieval_query"]),
                negative_type="cross_label_query_transfer",
                donor_task_id=str(donor_task["task_id"]),
            ):
                break

        cross_filing_donors = sorted(
            (
                donor_task
                for donor_task in by_label.get(source_task["label"], [])
                if donor_task["filing_id"] != source_task["filing_id"]
            ),
            key=_task_sort_key,
        )
        for donor_task in cross_filing_donors:
            transfer_phrase = _cross_filing_transfer_phrase(
                source_task, donor_task, source_tokens
            )
            if not transfer_phrase:
                continue
            source_scope_title = _scope_title(source_task)
            question = f"Does the {source_scope_title} note discuss {transfer_phrase}?"
            retrieval_query = _normalize_query_text(
                f"{source_scope_title} {transfer_phrase}"
            )
            if _append_unanswerable_candidate(
                candidate_pools,
                seen_candidate_keys,
                filing_query_keys,
                source_task,
                question=question,
                retrieval_query=retrieval_query,
                negative_type="cross_filing_query_transfer",
                donor_task_id=str(donor_task["task_id"]),
            ):
                break

    selected: List[Dict[str, Any]] = []
    next_index_by_type = {
        negative_type: 0 for negative_type in _UNANSWERABLE_NEGATIVE_TYPE_ORDER
    }
    while len(selected) < limit:
        added_in_round = False
        for negative_type in _UNANSWERABLE_NEGATIVE_TYPE_ORDER:
            candidate_pool = candidate_pools[negative_type]
            next_index = next_index_by_type[negative_type]
            if next_index >= len(candidate_pool):
                continue
            selected.append(candidate_pool[next_index])
            next_index_by_type[negative_type] = next_index + 1
            added_in_round = True
            if len(selected) >= limit:
                break
        if not added_in_round:
            break
    return selected


def build_narrative_benchmark(
    db_path: str,
    *,
    limit: int = 200,
    per_filing_limit: int = 1,
    filing_ids: Optional[Sequence[str]] = None,
    include_unanswerable: bool = False,
    unanswerable_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    answerable_limit = limit
    if include_unanswerable:
        answerable_limit = max(1, limit // 2)
    answerable_task_specs, filing_token_sets, donor_task_specs = (
        _build_answerable_narrative_tasks(
            _iter_candidate_rows(db_path, filing_ids=filing_ids),
            limit=answerable_limit,
            per_filing_limit=per_filing_limit,
        )
    )
    if not include_unanswerable:
        return answerable_task_specs[:limit]

    target_unanswerable_limit = (
        unanswerable_limit
        if unanswerable_limit is not None
        else max(1, limit - len(answerable_task_specs))
    )
    unanswerable_task_specs = _build_unanswerable_narrative_tasks(
        answerable_task_specs,
        filing_token_sets,
        limit=target_unanswerable_limit,
        donor_task_specs=donor_task_specs,
    )
    combined = answerable_task_specs + unanswerable_task_specs
    combined.sort(
        key=lambda task_spec: (
            task_spec["answerability"],
            task_spec["filing_id"],
            task_spec["task_id"],
        )
    )
    return combined[:limit]


def write_narrative_task_specs(
    path: str | Path, task_specs: Sequence[Mapping[str, Any]]
) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for task_spec in task_specs:
            validate_narrative_task_spec(task_spec)
            handle.write(
                json.dumps(dict(task_spec), ensure_ascii=False, sort_keys=True)
            )
            handle.write("\n")
            count += 1
    return count


def read_narrative_task_specs(path: str | Path) -> List[Dict[str, Any]]:
    task_specs: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            task_spec = json.loads(line)
            validate_narrative_task_spec(task_spec)
            task_specs.append(task_spec)
    return task_specs


def validate_narrative_answer(answer: Mapping[str, Any]) -> None:
    required_fields = {
        "narrative_answer_version",
        "task_id",
        "filing_id",
        "status",
        "answer_text",
        "chunk_evidence_ids",
        "refusal_code",
        "matched_rank",
    }
    missing = sorted(required_fields - set(answer))
    if missing:
        raise ValueError(
            f"Narrative answer is missing required fields: {', '.join(missing)}"
        )
    if answer["narrative_answer_version"] != NARRATIVE_ANSWER_VERSION:
        raise ValueError(
            f"Unsupported narrative answer version: {answer['narrative_answer_version']}"
        )
    if answer["status"] not in {"OK", "REFUSAL"}:
        raise ValueError(f"Unsupported narrative answer status: {answer['status']}")
    if not isinstance(answer["chunk_evidence_ids"], list):
        raise ValueError("Narrative answer chunk_evidence_ids must be a list")
    if answer["status"] == "OK":
        if not answer["answer_text"]:
            raise ValueError("OK narrative answers must include answer_text")
        if not answer["chunk_evidence_ids"]:
            raise ValueError("OK narrative answers must include chunk_evidence_ids")
        if answer["refusal_code"] is not None:
            raise ValueError("OK narrative answers cannot include refusal_code")
    else:
        if answer["refusal_code"] is None:
            raise ValueError("REFUSAL narrative answers must include refusal_code")


def _normalize_question_key(question: str) -> str:
    return " ".join(question.lower().split())


def _dedupe_narrative_task_specs(
    task_specs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for task_spec in task_specs:
        materialized = dict(task_spec)
        if materialized["task_id"] in seen:
            continue
        seen.add(materialized["task_id"])
        deduped.append(materialized)
    return deduped


def _routing_score(question: str, task_spec: Mapping[str, Any]) -> float:
    normalized_question = _normalize_question_key(question)
    question_tokens = set(_query_tokens(question))
    if not normalized_question or not question_tokens:
        return 0.0
    task_question_tokens = set(_query_tokens(str(task_spec.get("question") or "")))
    retrieval_tokens = set(_query_tokens(str(task_spec.get("retrieval_query") or "")))
    scope_tokens = set(_query_tokens(_scope_title(task_spec)))
    heading_tokens = set(
        _query_tokens(
            str(task_spec.get("heading_path") or task_spec.get("heading") or "")
        )
    )
    score = 0.0
    if task_question_tokens:
        score += 2.5 * (
            len(question_tokens & task_question_tokens) / len(task_question_tokens)
        )
    if retrieval_tokens:
        score += 2.0 * (len(question_tokens & retrieval_tokens) / len(retrieval_tokens))
    if scope_tokens:
        score += 1.5 * (len(question_tokens & scope_tokens) / len(scope_tokens))
    if heading_tokens:
        score += 1.0 * (len(question_tokens & heading_tokens) / len(heading_tokens))
    score += (
        0.5
        * SequenceMatcher(
            None,
            normalized_question,
            _normalize_question_key(str(task_spec.get("question") or "")),
        ).ratio()
    )
    scope_title = _scope_title(task_spec).lower()
    if scope_title and scope_title in normalized_question:
        score += 0.5
    return score


def route_narrative_question(
    question: str,
    filing_id: str,
    task_specs: Sequence[Mapping[str, Any]],
    *,
    allow_fuzzy: bool = True,
) -> Mapping[str, Any]:
    normalized_question = _normalize_question_key(question)
    filing_candidates = [
        task_spec
        for task_spec in _dedupe_narrative_task_specs(task_specs)
        if task_spec["filing_id"] == filing_id
    ]
    if not filing_candidates:
        raise ValueError(f"No narrative TaskSpecs available for filing {filing_id}")
    matches = [
        task_spec
        for task_spec in filing_candidates
        if _normalize_question_key(task_spec["question"]) == normalized_question
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        if not allow_fuzzy:
            raise ValueError("Question did not match any Narrative TaskSpec")
        scored_matches = sorted(
            (
                (
                    _routing_score(question, task_spec),
                    _task_sort_key(task_spec),
                    task_spec,
                )
                for task_spec in filing_candidates
            ),
            key=lambda item: (-item[0], item[1]),
        )
        best_score, _, best_task = scored_matches[0]
        if best_score < _ROUTING_MIN_SCORE:
            raise ValueError("Question did not match any Narrative TaskSpec")
        if (
            len(scored_matches) > 1
            and (best_score - scored_matches[1][0]) < _ROUTING_AMBIGUITY_MARGIN
        ):
            raise ValueError("Question matched multiple Narrative TaskSpecs")
        return best_task
    raise ValueError("Question matched multiple Narrative TaskSpecs")


def _build_chunk_rows_by_filing(
    db_path: str, filing_ids: Sequence[str]
) -> Dict[str, List[Mapping[str, Any]]]:
    grouped_rows: Dict[str, List[Mapping[str, Any]]] = {
        filing_id: [] for filing_id in filing_ids
    }
    for row in _fetch_chunk_rows(db_path, filing_ids=filing_ids):
        grouped_rows[row["filing_id"]].append(row)
    return grouped_rows


def _rows_for_task_scope(
    rows_by_filing: Mapping[str, Sequence[Mapping[str, Any]]],
    task_spec: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    filing_rows = list(rows_by_filing.get(str(task_spec["filing_id"]), []))
    scope_type = str(task_spec.get("scope_type") or "filing")
    if scope_type != "note":
        return filing_rows
    scope_key = _scope_key_for_task_spec(task_spec)
    if scope_key is None:
        return filing_rows
    scoped_rows = [
        row
        for row in filing_rows
        if _scope_key_for_row(row, scope_type="note") == scope_key
    ]
    return scoped_rows or filing_rows


def _get_scope_retriever(
    rows_by_filing: Mapping[str, Sequence[Mapping[str, Any]]],
    retriever_cache: Dict[tuple[str, str], Any],
    task_spec: Mapping[str, Any],
):
    scope_type = str(task_spec.get("scope_type") or "filing")
    scope_key = _scope_key_for_task_spec(task_spec) or "filing"
    cache_key = (str(task_spec["filing_id"]), f"{scope_type}:{scope_key}")
    if cache_key in retriever_cache:
        return retriever_cache[cache_key]
    scoped_rows = _rows_for_task_scope(rows_by_filing, task_spec)
    if not scoped_rows:
        return None
    retriever_cache[cache_key] = _build_bm25_retriever_from_rows(scoped_rows)
    return retriever_cache[cache_key]


def _unsupported_narrative_answer(
    task_id: str, filing_id: str, refusal_code: str
) -> Dict[str, Any]:
    answer = {
        "narrative_answer_version": NARRATIVE_ANSWER_VERSION,
        "task_id": task_id,
        "filing_id": filing_id,
        "status": "REFUSAL",
        "answer_text": None,
        "chunk_evidence_ids": [],
        "refusal_code": refusal_code,
        "matched_rank": None,
    }
    validate_narrative_answer(answer)
    return answer


def _retrieve_for_task_spec(
    retriever: Any,
    task_spec: Mapping[str, Any],
    *,
    top_k: int,
    method: str,
    candidate_k: int,
) -> List[Any]:
    example = RetrievalExample(
        query=task_spec["retrieval_query"],
        expected_chunk_ids=tuple(task_spec["expected_chunk_ids"]),
        filing_id=task_spec["filing_id"],
        label=task_spec["label"],
    )
    return _retrieve_documents_for_example(
        retriever,
        example,
        top_k=top_k,
        method=method,
        candidate_k=candidate_k,
    )


def answer_narrative(
    db_path: str,
    question: str,
    filing_id: str,
    *,
    task_specs: Sequence[Mapping[str, Any]],
    top_k: int = NARRATIVE_RETRIEVAL_TOP_K,
    method: str = NARRATIVE_RETRIEVAL_METHOD,
    candidate_k: Optional[int] = NARRATIVE_RETRIEVAL_CANDIDATE_K,
    rows_by_filing: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    retriever_cache: Optional[Dict[tuple[str, str], Any]] = None,
) -> Dict[str, Any]:
    active_task_specs = _dedupe_narrative_task_specs(task_specs)
    try:
        task_spec = route_narrative_question(question, filing_id, active_task_specs)
    except ValueError:
        return _unsupported_narrative_answer(
            "unsupported", filing_id, NARRATIVE_ROUTING_REFUSAL_CODE
        )

    if task_spec["answerability"] == "UNANSWERABLE":
        return _unsupported_narrative_answer(
            task_spec["task_id"], filing_id, task_spec["refusal_code"]
        )

    active_rows_by_filing = rows_by_filing or _build_chunk_rows_by_filing(
        db_path, [filing_id]
    )
    active_retriever_cache = retriever_cache if retriever_cache is not None else {}
    retriever = _get_scope_retriever(
        active_rows_by_filing, active_retriever_cache, task_spec
    )
    if retriever is None:
        return _unsupported_narrative_answer(
            task_spec["task_id"], filing_id, NARRATIVE_CITATION_MISS_CODE
        )

    documents = _retrieve_for_task_spec(
        retriever,
        task_spec,
        top_k=top_k,
        method=method,
        candidate_k=max(top_k, candidate_k or NARRATIVE_RETRIEVAL_CANDIDATE_K),
    )
    expected_chunk_ids = set(task_spec["expected_chunk_ids"])
    matched_rank = next(
        (
            index + 1
            for index, document in enumerate(documents)
            if document.id in expected_chunk_ids
        ),
        None,
    )
    if matched_rank is None:
        return _unsupported_narrative_answer(
            task_spec["task_id"], filing_id, NARRATIVE_CITATION_MISS_CODE
        )

    answer = {
        "narrative_answer_version": NARRATIVE_ANSWER_VERSION,
        "task_id": task_spec["task_id"],
        "filing_id": filing_id,
        "status": "OK",
        "answer_text": task_spec["extractive_answer"],
        "chunk_evidence_ids": [documents[matched_rank - 1].id],
        "refusal_code": None,
        "matched_rank": matched_rank,
    }
    validate_narrative_answer(answer)
    return answer


def evaluate_narrative_citations(
    db_path: str,
    task_specs: Sequence[Mapping[str, Any]],
    *,
    top_k: int = NARRATIVE_RETRIEVAL_TOP_K,
    method: str = NARRATIVE_RETRIEVAL_METHOD,
    candidate_k: Optional[int] = NARRATIVE_RETRIEVAL_CANDIDATE_K,
) -> Dict[str, Any]:
    validated_task_specs = [dict(task_spec) for task_spec in task_specs]
    for task_spec in validated_task_specs:
        validate_narrative_task_spec(task_spec)

    answerable_task_specs = [
        task_spec
        for task_spec in validated_task_specs
        if task_spec["answerability"] == "ANSWERABLE"
    ]
    rows_by_filing = _build_chunk_rows_by_filing(
        db_path,
        sorted({task_spec["filing_id"] for task_spec in answerable_task_specs}),
    )
    retriever_cache: Dict[tuple[str, str], Any] = {}
    citation_hits = 0
    citation_top1 = 0
    citation_mrr = 0.0

    detailed_results: List[Dict[str, Any]] = []
    negative_type_counts: Dict[str, int] = {}
    for task_spec in answerable_task_specs:
        retriever = _get_scope_retriever(rows_by_filing, retriever_cache, task_spec)
        result = {
            "retrieved_chunk_ids": [],
            "matched_rank": None,
            "top1_hit": False,
            "hit": False,
        }
        if retriever is not None:
            documents = _retrieve_for_task_spec(
                retriever,
                task_spec,
                top_k=top_k,
                method=method,
                candidate_k=max(top_k, candidate_k or NARRATIVE_RETRIEVAL_CANDIDATE_K),
            )
            retrieved_chunk_ids = [document.id for document in documents]
            expected_chunk_ids = set(task_spec["expected_chunk_ids"])
            matched_rank = next(
                (
                    index + 1
                    for index, chunk_id in enumerate(retrieved_chunk_ids)
                    if chunk_id in expected_chunk_ids
                ),
                None,
            )
            result = {
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "matched_rank": matched_rank,
                "top1_hit": matched_rank == 1,
                "hit": matched_rank is not None,
            }
        citation_hits += int(result["hit"])
        citation_top1 += int(result["top1_hit"])
        citation_mrr += (
            0.0 if result["matched_rank"] is None else 1.0 / result["matched_rank"]
        )
        detailed_results.append(
            {
                "task_id": task_spec["task_id"],
                "filing_id": task_spec["filing_id"],
                "ticker": task_spec["ticker"],
                "label": task_spec["label"],
                "question": task_spec["question"],
                "answerability": task_spec["answerability"],
                "scope_type": task_spec.get("scope_type"),
                "scope_key": _scope_key_for_task_spec(task_spec),
                "negative_type": task_spec.get("negative_type"),
                "expected_chunk_ids": list(task_spec["expected_chunk_ids"]),
                "retrieved_chunk_ids": result["retrieved_chunk_ids"],
                "matched_rank": result["matched_rank"],
                "citation_exact": result["top1_hit"],
                "citation_covered": result["hit"],
                "extractive_answer": task_spec["extractive_answer"],
            }
        )
    for task_spec in validated_task_specs:
        if task_spec["answerability"] == "UNANSWERABLE":
            negative_type = task_spec.get("negative_type")
            if negative_type:
                negative_type_counts[negative_type] = (
                    negative_type_counts.get(negative_type, 0) + 1
                )
            detailed_results.append(
                {
                    "task_id": task_spec["task_id"],
                    "filing_id": task_spec["filing_id"],
                    "ticker": task_spec["ticker"],
                    "label": task_spec["label"],
                    "question": task_spec["question"],
                    "answerability": task_spec["answerability"],
                    "negative_type": negative_type,
                    "expected_chunk_ids": [],
                    "retrieved_chunk_ids": [],
                    "matched_rank": None,
                    "citation_exact": None,
                    "citation_covered": None,
                    "extractive_answer": None,
                }
            )

    return {
        "summary": {
            "task_count": len(validated_task_specs),
            "answerable_task_count": len(answerable_task_specs),
            "unanswerable_task_count": sum(
                1
                for task_spec in validated_task_specs
                if task_spec["answerability"] == "UNANSWERABLE"
            ),
            "retrieval_method": method,
            "top_k": top_k,
            "candidate_k": candidate_k if method == "bm25_rerank" else top_k,
            "citation_precision_at_1": citation_top1
            / max(1, len(answerable_task_specs)),
            "citation_coverage_at_k": citation_hits
            / max(1, len(answerable_task_specs)),
            "citation_mrr_at_k": citation_mrr / max(1, len(answerable_task_specs)),
            "answerability_accuracy": None,
            "refusal_correctness": None,
            "negative_type_counts": negative_type_counts,
        },
        "results": detailed_results,
    }


def evaluate_narrative_answers(
    db_path: str,
    task_specs: Sequence[Mapping[str, Any]],
    *,
    top_k: int = NARRATIVE_RETRIEVAL_TOP_K,
    method: str = NARRATIVE_RETRIEVAL_METHOD,
    candidate_k: Optional[int] = NARRATIVE_RETRIEVAL_CANDIDATE_K,
) -> Dict[str, Any]:
    validated_task_specs = [dict(task_spec) for task_spec in task_specs]
    for task_spec in validated_task_specs:
        validate_narrative_task_spec(task_spec)

    rows_by_filing = _build_chunk_rows_by_filing(
        db_path,
        sorted({task_spec["filing_id"] for task_spec in validated_task_specs}),
    )
    retriever_cache: Dict[tuple[str, str], Any] = {}
    detailed_results: List[Dict[str, Any]] = []
    counters = {
        "task_count": len(validated_task_specs),
        "answerable_total": 0,
        "unanswerable_total": 0,
        "status_match": 0,
        "citation_exact": 0,
        "answer_text_exact": 0,
        "refusal_correct": 0,
    }
    negative_type_counts: Dict[str, int] = {}

    for task_spec in validated_task_specs:
        predicted = answer_narrative(
            db_path,
            task_spec["question"],
            task_spec["filing_id"],
            task_specs=validated_task_specs,
            top_k=top_k,
            method=method,
            candidate_k=candidate_k,
            rows_by_filing=rows_by_filing,
            retriever_cache=retriever_cache,
        )
        expected_status = (
            "OK" if task_spec["answerability"] == "ANSWERABLE" else "REFUSAL"
        )
        status_match = predicted["status"] == expected_status
        counters["status_match"] += int(status_match)
        if task_spec["answerability"] == "ANSWERABLE":
            counters["answerable_total"] += 1
            citation_exact = predicted["chunk_evidence_ids"] == list(
                task_spec["expected_chunk_ids"]
            )
            answer_text_exact = (
                predicted["answer_text"] == task_spec["extractive_answer"]
            )
            counters["citation_exact"] += int(status_match and citation_exact)
            counters["answer_text_exact"] += int(status_match and answer_text_exact)
            detailed_results.append(
                {
                    "task_id": task_spec["task_id"],
                    "filing_id": task_spec["filing_id"],
                    "ticker": task_spec["ticker"],
                    "label": task_spec["label"],
                    "question": task_spec["question"],
                    "answerability": task_spec["answerability"],
                    "scope_type": task_spec.get("scope_type"),
                    "scope_key": _scope_key_for_task_spec(task_spec),
                    "predicted": predicted,
                    "expected_chunk_ids": list(task_spec["expected_chunk_ids"]),
                    "expected_answer_text": task_spec["extractive_answer"],
                    "citation_exact": citation_exact,
                    "answer_text_exact": answer_text_exact,
                }
            )
        else:
            counters["unanswerable_total"] += 1
            negative_type = task_spec.get("negative_type")
            if negative_type:
                negative_type_counts[negative_type] = (
                    negative_type_counts.get(negative_type, 0) + 1
                )
            refusal_correct = (
                status_match and predicted["refusal_code"] == task_spec["refusal_code"]
            )
            counters["refusal_correct"] += int(refusal_correct)
            detailed_results.append(
                {
                    "task_id": task_spec["task_id"],
                    "filing_id": task_spec["filing_id"],
                    "ticker": task_spec["ticker"],
                    "label": task_spec["label"],
                    "question": task_spec["question"],
                    "answerability": task_spec["answerability"],
                    "scope_type": task_spec.get("scope_type"),
                    "scope_key": _scope_key_for_task_spec(task_spec),
                    "negative_type": negative_type,
                    "predicted": predicted,
                    "expected_refusal_code": task_spec["refusal_code"],
                    "refusal_correct": refusal_correct,
                }
            )

    return {
        "summary": {
            "task_count": counters["task_count"],
            "answerable_task_count": counters["answerable_total"],
            "unanswerable_task_count": counters["unanswerable_total"],
            "retrieval_method": method,
            "top_k": top_k,
            "candidate_k": candidate_k if method == "bm25_rerank" else top_k,
            "answerability_accuracy": counters["status_match"]
            / max(1, counters["task_count"]),
            "citation_exactness": counters["citation_exact"]
            / max(1, counters["answerable_total"]),
            "answer_text_exactness": counters["answer_text_exact"]
            / max(1, counters["answerable_total"]),
            "refusal_correctness": counters["refusal_correct"]
            / max(1, counters["unanswerable_total"]),
            "negative_type_counts": negative_type_counts,
        },
        "results": detailed_results,
    }


def evaluate_narrative_routing(
    db_path: str,
    task_specs: Sequence[Mapping[str, Any]],
    *,
    variants_per_task: int = 1,
    top_k: int = NARRATIVE_RETRIEVAL_TOP_K,
    method: str = NARRATIVE_RETRIEVAL_METHOD,
    candidate_k: Optional[int] = NARRATIVE_RETRIEVAL_CANDIDATE_K,
) -> Dict[str, Any]:
    validated_task_specs = [dict(task_spec) for task_spec in task_specs]
    task_specs_by_id: Dict[str, Dict[str, Any]] = {}
    for task_spec in validated_task_specs:
        validate_narrative_task_spec(task_spec)
        task_specs_by_id[str(task_spec["task_id"])] = task_spec

    variants = build_narrative_routing_variants(
        validated_task_specs, variants_per_task=variants_per_task
    )
    rows_by_filing = _build_chunk_rows_by_filing(
        db_path,
        sorted({task_spec["filing_id"] for task_spec in validated_task_specs}),
    )
    retriever_cache: Dict[tuple[str, str], Any] = {}
    detailed_results: List[Dict[str, Any]] = []
    counters = {
        "variant_count": len(variants),
        "answerable_total": 0,
        "unanswerable_total": 0,
        "route_correct": 0,
        "status_match": 0,
        "citation_exact": 0,
        "answer_text_exact": 0,
        "refusal_correct": 0,
        "safe_refusal": 0,
    }
    negative_type_counts: Dict[str, int] = {}

    for variant in variants:
        source_task = task_specs_by_id[str(variant["source_task_id"])]
        predicted = answer_narrative(
            db_path,
            str(variant["question"]),
            str(variant["filing_id"]),
            task_specs=validated_task_specs,
            top_k=top_k,
            method=method,
            candidate_k=candidate_k,
            rows_by_filing=rows_by_filing,
            retriever_cache=retriever_cache,
        )
        route_correct = predicted["task_id"] == source_task["task_id"]
        counters["route_correct"] += int(route_correct)

        expected_status = (
            "OK" if source_task["answerability"] == "ANSWERABLE" else "REFUSAL"
        )
        status_match = predicted["status"] == expected_status
        counters["status_match"] += int(status_match)

        result = {
            "variant_id": variant["variant_id"],
            "source_task_id": source_task["task_id"],
            "filing_id": source_task["filing_id"],
            "ticker": source_task["ticker"],
            "label": source_task["label"],
            "scope_type": source_task.get("scope_type"),
            "scope_key": _scope_key_for_task_spec(source_task),
            "question": variant["question"],
            "answerability": source_task["answerability"],
            "negative_type": source_task.get("negative_type"),
            "predicted": predicted,
            "route_correct": route_correct,
            "status_correct": status_match,
        }

        if source_task["answerability"] == "ANSWERABLE":
            counters["answerable_total"] += 1
            citation_exact = predicted["chunk_evidence_ids"] == list(
                source_task["expected_chunk_ids"]
            )
            answer_text_exact = (
                predicted["answer_text"] == source_task["extractive_answer"]
            )
            counters["citation_exact"] += int(
                route_correct and status_match and citation_exact
            )
            counters["answer_text_exact"] += int(
                route_correct and status_match and answer_text_exact
            )
            result.update(
                {
                    "expected_chunk_ids": list(source_task["expected_chunk_ids"]),
                    "expected_answer_text": source_task["extractive_answer"],
                    "citation_exact": citation_exact,
                    "answer_text_exact": answer_text_exact,
                }
            )
        else:
            counters["unanswerable_total"] += 1
            negative_type = source_task.get("negative_type")
            if negative_type:
                negative_type_counts[negative_type] = (
                    negative_type_counts.get(negative_type, 0) + 1
                )
            refusal_correct = (
                route_correct
                and status_match
                and predicted["refusal_code"] == source_task["refusal_code"]
            )
            safe_refusal = (
                predicted["status"] == "REFUSAL"
                and predicted["refusal_code"] in _SAFE_NARRATIVE_REFUSAL_CODES
            )
            counters["refusal_correct"] += int(refusal_correct)
            counters["safe_refusal"] += int(safe_refusal)
            result.update(
                {
                    "expected_refusal_code": source_task["refusal_code"],
                    "refusal_correct": refusal_correct,
                    "safe_refusal": safe_refusal,
                }
            )
        detailed_results.append(result)

    return {
        "summary": {
            "variant_count": counters["variant_count"],
            "variants_per_task": max(1, variants_per_task),
            "answerable_variant_count": counters["answerable_total"],
            "unanswerable_variant_count": counters["unanswerable_total"],
            "retrieval_method": method,
            "top_k": top_k,
            "candidate_k": candidate_k if method == "bm25_rerank" else top_k,
            "routing_accuracy": counters["route_correct"]
            / max(1, counters["variant_count"]),
            "answerability_accuracy": counters["status_match"]
            / max(1, counters["variant_count"]),
            "citation_exactness": counters["citation_exact"]
            / max(1, counters["answerable_total"]),
            "answer_text_exactness": counters["answer_text_exact"]
            / max(1, counters["answerable_total"]),
            "refusal_correctness": counters["refusal_correct"]
            / max(1, counters["unanswerable_total"]),
            "safe_refusal_accuracy": counters["safe_refusal"]
            / max(1, counters["unanswerable_total"]),
            "negative_type_counts": negative_type_counts,
        },
        "results": detailed_results,
    }
