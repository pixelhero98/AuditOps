"""Human-gated cached-20 benchmark construction for AuditOps v2.4.

v2.4 deliberately keeps the proven v2.3 text-agent contract.  This module
changes only benchmark construction and evaluation gold: natural questions,
issuer/subtype quotas, bounded acceptable answer sets, and an approval gate
that cannot be satisfied by the benchmark builder itself.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import warnings
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from .agent_benchmark import (
    _build_case,
    _build_evidence_items,
    _build_gold,
    _quant_gold,
    _quant_status,
    _read_jsonl,
    _source_entity_id,
)
from .agent_benchmark_v23 import (
    _convert_gold,
    _convert_task,
    split_narrative_evidence_item_v23,
)
from .agent_context import build_context_pack
from .agent_contracts import validate_agent_task_input
from .agent_operations import task_operation_spec
from .agent_tools import evaluate_quant_evidence, load_frozen_evidence
from .canonical_json import canonical_json_bytes, canonical_json_sha256
from .provenance import assert_no_secrets, assert_no_secrets_in_value
from .text_support import is_contiguous_text_supported, normalize_support_text

BENCHMARK_VERSION_V24 = "agent_benchmark.v2.4"
CANDIDATE_VERSION_V24 = "auditops-agent-benchmark-candidate.v2.4"
CANDIDATE_VERSION_V24_R2 = "auditops-agent-benchmark-candidate.v2.4r2"
NARRATIVE_GOLD_VERSION_V24 = "v2.4"
REVIEW_PACKET_VERSION_V24 = "auditops-benchmark-review-packet.v2.4"
REVIEW_PACKET_VERSION_V24_R2 = "auditops-benchmark-review-packet.v2.4r2"
REVIEW_DECISION_VERSION_V24 = "auditops-benchmark-review-decision.v2.4"
APPROVAL_VERSION_V24 = "auditops-benchmark-approval.v2.4"
GATE_MEMBERSHIP_VERSION_V24 = "auditops-agent-narrative-gate.v2.4"
SECONDARY_SAMPLE_VERSION_V24 = "auditops-agent-secondary-review-sample.v2.4"
BM25_POLICY_VERSION_V24 = "auditops.frozen_bm25.v2.4"
RAW_CAM_SUPPLEMENT_VERSION_V24 = "auditops.raw_cam_supplement.v2.4"
DEFAULT_SEED_V24 = 20260821
QUANT_COUNT_V24 = 300
NARRATIVE_COUNT_V24 = 200
ISSUER_COUNT_V24 = 20
MAX_EVIDENCE_ITEMS = 5
MAX_SUPPORT_WINDOW_CHARS = 1_000

_SUBTYPE_QUOTAS: Mapping[str, tuple[int, int]] = {
    "footnote_note": (2, 2),
    "accounting_policy": (1, 1),
    "auditor_report_opinion_language": (1, 1),
    "critical_audit_matter": (1, 1),
}
_HARDENED_R2_SUBTYPE_ORDER = (
    "auditor_report_opinion_language",
    "critical_audit_matter",
    "accounting_policy",
    "footnote_note",
)
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’.-]*")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_RE = re.compile(r"(?s)(?:^|(?<=[.!?])\s+)([^\n].*?[.!?])(?=\s+|$)")
_SPACE_RE = re.compile(r"\s+")
_ABBREVIATION_SENTINEL = "\ue000"
_ABBREVIATION_PATTERNS = (
    re.compile(r"\bU\.S\.(?=\s+\S)", re.IGNORECASE),
    re.compile(r"\bU\.K\.(?=\s+\S)", re.IGNORECASE),
    re.compile(r"\b(?:Inc|Corp|Co|Ltd)\.(?=\s+[a-z])"),
)
_HUMAN_ATTESTATION = (
    "I independently reviewed the assigned AuditOps cases as a human reviewer."
)
_ADJUDICATION_ATTESTATION = "Both named reviewers independently re-reviewed every case in each affected subtype."
_FORBIDDEN_REVIEWER_TERMS = (
    "codex",
    "chatgpt",
    "language model",
    "not_a_human_reviewer",
    "openai",
    "replace_with",
    "template",
)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "these",
        "this",
        "to",
        "was",
        "we",
        "were",
        "which",
        "with",
        "year",
        "years",
    }
)

_NON_DISCLOSURE_SECTION_MARKERS = (
    "capital resources",
    "consolidated balance sheet",
    "consolidated statement",
    "critical accounting estimates",
    "exhibits and financial statement schedules",
    "financial statements and supplementary data",
    "independent registered public accounting",
    "internal control over financial reporting",
    "liquidity and uses of cash",
    "management's discussion",
    "market risk",
    "report of independent",
    "risk factors",
    "table of contents",
)

_POLICY_TOPIC_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("revenue", "recogn"), "revenue recognition"),
    (("basis of presentation",), "basis of presentation"),
    (("cash equivalent",), "cash and cash equivalents"),
    (("inventor",), "inventory valuation"),
    (("use of estimate",), "use of estimates"),
    (("goodwill", "impair"), "goodwill impairment"),
    (("lease",), "lease accounting"),
    (("share-based",), "share-based compensation"),
    (("stock-based",), "share-based compensation"),
    (("income tax",), "income taxes"),
    (("foreign currency", "derivative"), "foreign-currency derivatives"),
    (("foreign currency", "translat"), "foreign-currency translation"),
    (("research and development",), "research and development costs"),
    (("property", "equipment", "depreciat"), "property and equipment"),
    (("property", "equipment", "stated at"), "property and equipment"),
    (("property", "equipment", "capitaliz"), "property and equipment"),
    (("depreciat",), "depreciation"),
)

# These headings identify accounting-note disclosures rather than merely
# mentioning an accounting noun in MD&A.  The list is intentionally explicit:
# construction must fail when a filing cannot supply two reviewable notes.
_FOOTNOTE_HEADING_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("nature of business",), "nature of business"),
    (("significant accounting polic",), "significant accounting policies"),
    (("revenue recognition",), "revenue recognition"),
    (("cash and cash equivalent",), "cash and cash equivalents"),
    (("inventor",), "inventory"),
    (("property", "equipment"), "property and equipment"),
    (("internal-use software",), "internal-use software"),
    (("capitalized software",), "capitalized software"),
    (("goodwill",), "goodwill"),
    (("intangible asset",), "intangible assets"),
    (("advertising cost",), "advertising costs"),
    (("fair value measurement",), "fair value measurements"),
    (("financial instrument",), "financial instruments"),
    (("hedging activit",), "hedging activities"),
    (("derivative",), "derivative instruments"),
    (("debt obligation",), "debt obligations"),
    (("long-term debt",), "long-term debt"),
    (("lease",), "leases"),
    (("income tax",), "income taxes"),
    (("share-based compensation",), "share-based compensation"),
    (("stock-based compensation",), "share-based compensation"),
    (("employee benefit",), "employee benefits"),
    (("pension",), "pension obligations"),
    (("commitments and contingencies",), "commitments and contingencies"),
    (("segment", "geographic"), "segment and geographic information"),
    (("earnings per share",), "earnings per share"),
    (("acquisition",), "acquisitions"),
    (("credit loss",), "credit losses"),
    (("receivable",), "receivables"),
    (("supplier finance",), "supplier-finance arrangements"),
    (("subsequent event",), "subsequent events"),
    (("related part",), "related-party transactions"),
)

_FOOTNOTE_SENTENCE_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("share-based compensation",), "share-based compensation"),
    (("stock-based compensation",), "share-based compensation"),
    (("defined benefit", "pension"), "pension obligations"),
    (("pension",), "pension obligations"),
    (("employee benefit",), "employee benefits"),
    (("revenue", "recogn"), "revenue recognition"),
    (("cash equivalent",), "cash and cash equivalents"),
    (("receivable",), "receivables"),
    (("inventor",), "inventory"),
    (("property", "equipment"), "property and equipment"),
    (("internal-use software",), "internal-use software"),
    (("capitalized software",), "capitalized software"),
    (("goodwill",), "goodwill"),
    (("intangible asset",), "intangible assets"),
    (("advertising cost",), "advertising costs"),
    (("derivative",), "derivative instruments"),
    (("hedg",), "hedging activities"),
    (("fair value",), "fair value measurements"),
    (("financial instrument",), "financial instruments"),
    (("long-term debt",), "long-term debt"),
    (("debt obligation",), "debt obligations"),
    (("borrowings",), "borrowings"),
    (("lease",), "leases"),
    (("deferred", "tax"), "income taxes"),
    (("tax position",), "income taxes"),
    (("taxable income",), "income taxes"),
    (("income tax", "expense"), "income taxes"),
    (("income tax", "liabil"), "income taxes"),
    (("income tax", "benefit"), "income taxes"),
    (("commitment", "contingenc"), "commitments and contingencies"),
    (("loss contingenc",), "loss contingencies"),
    (("reportable segment",), "segment information"),
    (("earnings per share",), "earnings per share"),
    (("credit loss",), "credit losses"),
    (("supplier finance",), "supplier-finance arrangements"),
    (("subsequent event",), "subsequent events"),
    (("related part",), "related-party transactions"),
    (("unpaid losses",), "insurance claim liabilities"),
    (("claim liabilit",), "insurance claim liabilities"),
    (("insurance benefit",), "insurance-benefit liabilities"),
)

_DISCLOSURE_PREDICATE_RE = re.compile(
    r"\b(?:account(?:s|ed)?|allocat(?:e|es|ed)|amortiz(?:e|es|ed)|are|"
    r"capitaliz(?:e|es|ed)|categor(?:ize|izes|ized)|classif(?:y|ies|ied)|"
    r"consist(?:s|ed)?|depreciat(?:e|es|ed)|enter(?:s|ed)?|has|have|include(?:s|d)?|"
    r"is|measure(?:s|d)?|offer(?:s|ed)?|provide(?:s|d)?|qualif(?:y|ies|ied)|"
    r"present(?:s|ed)?|"
    r"recogniz(?:e|es|ed)|record(?:s|ed)?|report(?:s|ed)?|represent(?:s|ed)?|"
    r"require(?:s|d)?|sponsor(?:s|ed)?|test(?:s|ed)?|use(?:s|d)?|were|was)\b",
    re.IGNORECASE,
)

_CAM_TOPIC_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("rebate",), "rebate estimates"),
    (("revenue", "contract", "estimate"), "revenue recognition and contract estimates"),
    (("program accounting",), "program accounting estimates"),
    (("aircraft",), "aircraft program estimates"),
    (("medical care",), "medical care cost estimates"),
    (("warranty",), "product warranty liabilities"),
    (("oil", "reserve"), "oil and gas reserve estimates"),
    (("natural gas", "reserve"), "oil and gas reserve estimates"),
    (("proved reserve",), "oil and gas reserve estimates"),
    (("fair value",), "fair value estimates"),
    (("environmental", "accrual"), "environmental remediation accruals"),
    (("contingenc",), "loss contingencies"),
    (("operating propert", "holding period"), "operating-property holding periods"),
    (("claim liabilit",), "insurance claim liabilities"),
    (("unpaid losses",), "insurance claim liabilities"),
    (("goodwill",), "goodwill impairment"),
    (("credit loss",), "the allowance for credit losses"),
    (("tax",), "income tax matters"),
    (("regulat", "rate"), "regulated rates"),
    (("inventory",), "inventory valuation"),
    (("litigation",), "litigation contingencies"),
    (("acquisition",), "acquisition accounting"),
    (("valuation",), "valuation estimates"),
)

# Canonical question topics intentionally use compact labels.  Filings often
# answer the same question with a reasonable accounting/audit synonym, so r2b
# uses these aliases only to validate exact, in-scope support.  They are not
# inference-visible and they never permit cross-topic or cross-section gold.
_TOPIC_ANCHOR_ALIASES_V24R2: Mapping[str, tuple[str, ...]] = {
    "foreign-currency translation": (
        "foreign currency",
        "functional currency",
        "exchange rate",
        "translation adjustment",
        "other comprehensive income",
    ),
    "goodwill": ("goodwill", "impairment"),
    "goodwill impairment": ("goodwill", "impairment"),
    "insurance claim liabilities": (
        "claim liabilities",
        "unpaid losses",
        "insurance reserves",
    ),
    "inventory valuation": (
        "inventory",
        "inventories",
        "obsolete",
        "obsolescence",
        "provision",
        "valuation",
    ),
    "lease accounting": (
        "lease",
        "right-of-use",
        "lease liability",
        "borrowing rate",
    ),
    "litigation contingencies": (
        "litigation",
        "lawsuit",
        "proceeding",
        "accrual",
        "MDL",
        "contingency",
    ),
    "loss contingencies": (
        "contingency",
        "claim",
        "lawsuit",
        "proceeding",
        "consent order",
    ),
    "research and development costs": (
        "research and development",
        "R&D",
        "IPR&D",
        "expensed as incurred",
    ),
}

# Retrieval aliases are fixed task semantics, not phrases selected from a gold
# answer.  They disambiguate broad labels such as ``recognition`` while leaving
# the BM25 candidate set and ordering deterministic.
_TOPIC_RETRIEVAL_ALIASES_V24R2: Mapping[str, tuple[str, ...]] = {
    "basis of presentation": (
        "consolidated financial statements",
        "generally accepted accounting principles",
    ),
    "cash and cash equivalents": (
        "highly liquid investments",
        "original maturity",
        "three months",
    ),
    "depreciation": ("straight line", "useful lives", "cost"),
    "foreign-currency derivatives": (
        "foreign exchange contracts",
        "fair value",
        "gains and losses",
    ),
    "foreign-currency translation": (
        "functional currency",
        "exchange rates",
        "other comprehensive income",
    ),
    "goodwill impairment": (
        "reporting units",
        "fair value",
        "carrying amount",
        "annual impairment test",
    ),
    "income taxes": (
        "uncertain tax positions",
        "more likely than not",
        "tax authorities",
    ),
    "inventory valuation": (
        "inventories",
        "materials and supplies",
        "cost",
        "lower market",
    ),
    "lease accounting": (
        "right-of-use assets",
        "lease liabilities",
        "discount rate",
    ),
    "property and equipment": (
        "stated at cost",
        "depreciation",
        "useful lives",
        "straight line",
    ),
    "research and development costs": (
        "research and development",
        "expensed as incurred",
    ),
    "revenue recognition": (
        "customer control",
        "title transfer",
        "risks and rewards",
    ),
    "share-based compensation": (
        "grant date fair value",
        "vesting",
        "compensation expense",
    ),
    "use of estimates": ("management estimates", "assumptions", "actual results"),
    "opinion on the financial statements": (
        "present fairly",
        "all material respects",
        "financial position",
        "results of operations",
        "cash flows",
    ),
    "rebate estimates": (
        "sales rebate",
        "discount accruals",
        "rebate liabilities",
        "assumptions",
    ),
    "product warranty liabilities": (
        "warranty liability",
        "claim rates",
        "field population",
    ),
}

_NEGATIVE_CATALOG: Mapping[str, tuple[tuple[str, str], ...]] = {
    "footnote_note": (
        ("inventory is measured using the retail inventory method", "inventory"),
        (
            "the company has a material supplier-finance program classified as bank debt",
            "supplier finance",
        ),
        ("all legal claims were settled without payment", "legal claims"),
        ("a defined-benefit pension plan was closed during the year", "pension plan"),
        (
            "all cloud-computing implementation costs are capitalized as software",
            "cloud-computing costs",
        ),
        (
            "the company has a receivables factoring program with full recourse",
            "receivables factoring",
        ),
        ("goodwill is amortized over exactly ten years", "goodwill amortized"),
        (
            "a debt-covenant violation occurred after the reporting date",
            "debt covenant",
        ),
    ),
    "accounting_policy": (
        (
            "all research and development costs are capitalized when incurred",
            "research and development",
        ),
        ("revenue is recognized only when cash is collected", "revenue recognition"),
        ("goodwill is amortized on a straight-line basis", "goodwill"),
        (
            "inventory is measured exclusively at replacement cost",
            "inventory measurement",
        ),
    ),
    "auditor_report_opinion_language": (
        (
            "the financial statements were prepared on the cash basis of accounting",
            "cash basis",
        ),
        (
            "the auditor disclaims every opinion on the financial statements",
            "disclaimer of opinion",
        ),
        (
            "the audit provides absolute assurance against all misstatement",
            "absolute assurance",
        ),
    ),
    "critical_audit_matter": (
        (
            "a supplier-finance obligation was identified as a critical audit matter",
            "supplier finance",
        ),
        (
            "a cybersecurity-incident liability was identified as a critical audit matter",
            "cybersecurity liability",
        ),
        (
            "a pension-plan termination was identified as a critical audit matter",
            "pension termination",
        ),
        (
            "a debt-covenant violation was identified as a critical audit matter",
            "debt covenant",
        ),
    ),
}

# r2 keeps the public question text separate from the phrases that disqualify a
# proposed negative.  A match requires every concept group to be present, while
# alternatives inside one group represent reasonable synonymous wording.  The
# mechanical screen is intentionally conservative: a possible match rejects the
# candidate and lets the deterministic round-robin try another claim.
_NEGATIVE_CATALOG_R2: Mapping[str, tuple[Mapping[str, Any], ...]] = {
    "footnote_note": (
        {
            "claim": "inventory is measured using the retail inventory method",
            "focus": "inventory",
            "concept_groups": (("inventory",), ("retail inventory method", "rim")),
        },
        {
            "claim": "a material supplier-finance program is classified as bank debt",
            "focus": "supplier finance",
            "concept_groups": (
                ("supplier finance", "supplier-finance"),
                ("bank debt", "borrowings", "debt obligation"),
            ),
        },
        {
            "claim": "all legal claims were settled without any payment",
            "focus": "legal claims",
            "concept_groups": (
                ("legal claim", "litigation", "lawsuit"),
                ("settled", "settlement"),
                ("without payment", "no payment", "zero payment"),
            ),
        },
        {
            "claim": "every defined-benefit pension plan was terminated during the fiscal year",
            "focus": "pension plan termination",
            "concept_groups": (
                ("defined benefit", "defined-benefit"),
                ("pension", "retirement plan", "defined benefit plan"),
                ("terminat", "closed", "frozen", "wound up", "settled"),
            ),
        },
        {
            "claim": "all cloud-computing implementation costs are capitalized as software",
            "focus": "cloud-computing costs",
            "concept_groups": (
                ("cloud computing", "cloud-computing"),
                ("implementation cost",),
                ("capitaliz",),
                ("software",),
            ),
        },
        {
            "claim": "a receivables factoring program transfers receivables with full recourse",
            "focus": "receivables factoring",
            "concept_groups": (
                ("receivable",),
                ("factor", "factoring"),
                ("full recourse", "with recourse"),
            ),
        },
        {
            "claim": "goodwill is amortized over exactly ten years",
            "focus": "goodwill amortization",
            "concept_groups": (
                ("goodwill",),
                ("amortiz",),
                ("ten years", "10 years", "ten-year", "10-year"),
            ),
        },
        {
            "claim": "a debt-covenant violation occurred after the reporting date",
            "focus": "debt covenant",
            "concept_groups": (
                ("debt covenant", "covenant"),
                ("violation", "default", "breach"),
                ("after the reporting date", "subsequent", "after year-end"),
            ),
        },
    ),
    "accounting_policy": (
        {
            "claim": "all research and development costs are capitalized when incurred",
            "focus": "research and development",
            "concept_groups": (
                ("research and development", "r&d"),
                ("capitaliz",),
                ("when incurred", "as incurred"),
            ),
        },
        {
            "claim": "revenue is recognized only when cash is collected",
            "focus": "revenue recognition",
            "concept_groups": (
                ("revenue",),
                ("recogniz",),
                ("cash is collected", "cash collection", "upon collection"),
            ),
        },
        {
            "claim": "goodwill is amortized on a straight-line basis",
            "focus": "goodwill",
            "concept_groups": (
                ("goodwill",),
                ("amortiz",),
                ("straight-line", "straight line"),
            ),
        },
        {
            "claim": "inventory is measured exclusively at replacement cost",
            "focus": "inventory measurement",
            "concept_groups": (
                ("inventory",),
                ("replacement cost",),
                ("exclusively", "only"),
            ),
        },
    ),
    "auditor_report_opinion_language": (
        {
            "claim": "the financial statements were prepared on the cash basis of accounting",
            "focus": "cash basis",
            "concept_groups": (("financial statement",), ("cash basis",)),
        },
        {
            "claim": "the auditor disclaims every opinion on the financial statements",
            "focus": "disclaimer of opinion",
            "concept_groups": (
                ("disclaim", "disclaimer"),
                ("opinion",),
                ("financial statement",),
            ),
        },
        {
            "claim": "the audit provides absolute assurance against all misstatement",
            "focus": "absolute assurance",
            "concept_groups": (
                ("absolute assurance",),
                ("misstatement",),
            ),
        },
    ),
    "critical_audit_matter": (
        {
            "claim": "a supplier-finance obligation was identified as a critical audit matter",
            "focus": "supplier finance",
            "concept_groups": (
                ("supplier finance", "supplier-finance"),
                ("critical audit matter",),
            ),
        },
        {
            "claim": "a cybersecurity-incident liability was identified as a critical audit matter",
            "focus": "cybersecurity liability",
            "concept_groups": (
                ("cybersecurity", "cyber security"),
                ("liability", "incident"),
                ("critical audit matter",),
            ),
        },
        {
            "claim": "a pension-plan termination was identified as a critical audit matter",
            "focus": "pension termination",
            "concept_groups": (
                ("pension",),
                ("terminat",),
                ("critical audit matter",),
            ),
        },
        {
            "claim": "a debt-covenant violation was identified as a critical audit matter",
            "focus": "debt covenant",
            "concept_groups": (
                ("covenant",),
                ("violation", "breach", "default"),
                ("critical audit matter",),
            ),
        },
    ),
}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, records: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sha256": _sha256_path(path),
        "bytes": path.stat().st_size,
    }
    if records is not None:
        result["records"] = records
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value, newline=True))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row, newline=True))
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _stable_rank(seed: int, namespace: str, value: Any) -> str:
    return hashlib.sha256(
        f"{seed}|{namespace}|{canonical_json_sha256(value)}".encode()
    ).hexdigest()


def _narrative_scope_items_v24r2(
    *,
    subtype: str,
    canonical_segments: Sequence[Mapping[str, Any]],
    raw_cam_segments: Sequence[Mapping[str, Any]],
    hardened_r2: bool,
) -> list[Mapping[str, Any]]:
    if hardened_r2 and subtype == "auditor_report_opinion_language":
        scoped = [
            item
            for item in canonical_segments
            if _is_auditor_opinion_scope_item_v24r2(item)
        ]
        if not scoped:
            raise ValueError("Filing lacks a bounded auditor-opinion scope")
        return scoped
    if subtype != "critical_audit_matter":
        return list(canonical_segments)
    if hardened_r2:
        # The checksum-bound supplement has explicit CAM boundaries.  Mixing it
        # with filing-wide chunks lets duplicate Item 7 text outrank the
        # auditor-report source and makes exact source binding unstable.
        return list(raw_cam_segments)
    return [*canonical_segments, *raw_cam_segments]


def _is_auditor_opinion_scope_item_v24r2(item: Mapping[str, Any]) -> bool:
    metadata = item.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    structural_scope = " ".join(
        str(metadata.get(key) or "")
        for key in ("heading", "subheading", "heading_path", "section_label")
    ).casefold()
    content = str(item.get("content") or "").casefold()
    structural_match = any(
        marker in structural_scope
        for marker in (
            "independent registered public accounting",
            "independent auditor",
            "opinion on the financial statements",
            "report of independent",
        )
    )
    opinion_match = "in our opinion" in content and "financial statement" in content
    if opinion_match:
        return True
    return str(metadata.get("item") or "") in {"8", "8A"} and structural_match


def _tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


def _bm25_select(
    query: str, items: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Rank only from the inference-visible query and source chunks."""

    documents = [_tokenize(str(item.get("content") or "")) for item in items]
    average_length = sum(map(len, documents)) / len(documents) if documents else 1.0
    average_length = average_length or 1.0
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))
    query_terms = Counter(_tokenize(query))
    ranked: list[tuple[float, int, Mapping[str, Any]]] = []
    for index, (item, document) in enumerate(zip(items, documents, strict=True)):
        frequencies = Counter(document)
        score = 0.0
        for term, query_frequency in query_terms.items():
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            inverse = math.log(
                1.0
                + (len(documents) - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * len(document) / average_length
            )
            score += query_frequency * inverse * frequency * 2.2 / denominator
        ranked.append((score, index, item))
    ranked.sort(key=lambda value: (-value[0], value[1], str(value[2]["evidence_id"])))
    selected: list[dict[str, Any]] = []
    for rank, (score, _, item) in enumerate(ranked[:MAX_EVIDENCE_ITEMS], start=1):
        materialized = copy.deepcopy(dict(item))
        materialized["rank"] = rank
        metadata = materialized.setdefault("metadata", {})
        if isinstance(metadata, dict):
            # Evidence remains an AgentTaskInputV2.3 object. Bind the v2.4
            # retrieval policy in the candidate source manifest and retain the
            # runtime contract's registered prompt-visible score field.
            metadata["v23_bm25_score"] = round(score, 12)
        selected.append(materialized)
    return selected


def validate_natural_question(question: str) -> None:
    if not isinstance(question, str) or question != question.strip():
        raise ValueError("Narrative question must be a stripped string")
    words = _WORD_RE.findall(question)
    if not 8 <= len(words) <= 40:
        raise ValueError("Narrative question must contain 8-40 words")
    if question.count("?") != 1 or not question.endswith("?"):
        raise ValueError("Narrative question must be one question ending in '?'")
    lowered = question.casefold()
    if lowered.startswith("financial statements ") or "..." in question:
        raise ValueError("Narrative question cannot be a keyword string or fragment")
    if "\n" in question or "  " in question:
        raise ValueError("Narrative question must be a single normalized sentence")


def validate_question_scope_v24(case: Mapping[str, Any]) -> None:
    """Require visible issuer, period, and subtype wording to agree with the task."""

    question = str(case.get("question") or "")
    validate_natural_question(question)
    lowered = question.casefold()
    entity = case.get("entity") if isinstance(case.get("entity"), Mapping) else {}
    ticker = str(entity.get("ticker") or "").casefold()
    name = str(entity.get("name") or "").casefold()
    if not ((ticker and ticker in lowered) or (name and name in lowered)):
        raise ValueError("Narrative question must name the task issuer")
    period = case.get("period") if isinstance(case.get("period"), Mapping) else {}
    period_key = str(period.get("period_key") or "")
    year_match = re.search(r"(?:19|20)\d{2}", period_key)
    if year_match is None or year_match.group(0) not in question:
        raise ValueError("Narrative question must name the task fiscal year")
    subtype = case.get("narrative_subtype")
    markers: Mapping[str, tuple[str, ...]] = {
        "footnote_note": ("footnote",),
        "accounting_policy": ("accounting", "policy"),
        "auditor_report_opinion_language": ("auditor", "opinion"),
        "critical_audit_matter": ("critical audit matter", "auditor"),
    }
    required = markers.get(str(subtype))
    if required is None or any(marker not in lowered for marker in required):
        raise ValueError("Narrative question wording conflicts with its subtype")


def _validate_complete_sentence(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    if len(value) > MAX_SUPPORT_WINDOW_CHARS:
        raise ValueError(f"{field} exceeds {MAX_SUPPORT_WINDOW_CHARS} characters")
    if value.rstrip()[-1] not in ".!?":
        raise ValueError(f"{field} must end with sentence punctuation")
    if value.rstrip().endswith("..."):
        raise ValueError(f"{field} cannot be a truncated fragment")


def validate_narrative_gold_v24(
    gold: Mapping[str, Any], case: Mapping[str, Any]
) -> None:
    if gold.get("gold_record_version") != NARRATIVE_GOLD_VERSION_V24:
        raise ValueError("Narrative gold must use version v2.4")
    if gold.get("task_id") != case.get("task_id"):
        raise ValueError("Narrative gold and case task IDs differ")
    target = gold.get("target")
    if not isinstance(target, Mapping) or target.get("status") not in {"OK", "REFUSAL"}:
        raise ValueError("Narrative gold target is invalid")
    answer_sets = gold.get("acceptable_answer_sets")
    refusal_scope = gold.get("refusal_scope")
    if not isinstance(answer_sets, list):
        raise TypeError("acceptable_answer_sets must be an array")
    if target["status"] == "OK":
        if not answer_sets or refusal_scope is not None:
            raise ValueError(
                "Answerable gold requires answer sets and no refusal scope"
            )
        seen_set_ids: set[str] = set()
        for answer_set in answer_sets:
            if not isinstance(answer_set, Mapping):
                raise TypeError("Acceptable answer sets must contain objects")
            set_id = answer_set.get("set_id")
            extracts = answer_set.get("extracts")
            if not isinstance(set_id, str) or not set_id or set_id in seen_set_ids:
                raise ValueError("Acceptable answer-set IDs must be unique strings")
            seen_set_ids.add(set_id)
            if not isinstance(extracts, list) or not 1 <= len(extracts) <= 3:
                raise ValueError("Acceptable answer sets require one to three extracts")
            for extract in extracts:
                if not isinstance(extract, Mapping):
                    raise TypeError("Acceptable extracts must be objects")
                window = extract.get("support_window")
                anchors = extract.get("required_semantic_anchors")
                if not isinstance(window, str):
                    raise TypeError("support_window must be a string")
                _validate_complete_sentence(window, field="support_window")
                if not isinstance(extract.get("evidence_id"), str):
                    raise TypeError("Acceptable extract evidence_id must be a string")
                if (
                    not isinstance(anchors, list)
                    or not anchors
                    or not all(
                        isinstance(anchor, str) and anchor.strip() for anchor in anchors
                    )
                ):
                    raise ValueError("Acceptable extracts require semantic anchors")
                normalized_window = normalize_support_text(window).casefold()
                if any(
                    normalize_support_text(anchor).casefold() not in normalized_window
                    for anchor in anchors
                ):
                    raise ValueError(
                        "Every semantic anchor must occur in its support window"
                    )
    else:
        if answer_sets or not isinstance(refusal_scope, Mapping):
            raise ValueError(
                "Refusal gold requires an empty answer set and refusal scope"
            )
        required = {
            "scope_description",
            "causal_negative_type",
            "absent_claim",
            "full_filing_chunk_count",
            "full_filing_text_sha256",
            "mechanical_absence_check",
        }
        if set(refusal_scope) != required:
            raise ValueError("Refusal scope fields are missing or unexpected")
        if refusal_scope.get("mechanical_absence_check") is not True:
            raise ValueError("Refusal scope must pass mechanical absence before review")


def _connect_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"Canonical corpus database not found: {path}")
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _load_corpus(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    connection = _connect_read_only(path)
    try:
        required = {"filings", "chunk_canon", "facts_canon"}
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not required.issubset(tables):
            raise ValueError("Corpus database lacks required canonical tables")
        filings = [
            dict(row)
            for row in connection.execute(
                """
                SELECT filing_id, ticker, form_type, fiscal_year_focus, report_date,
                       cik, accession, source_sha256, source_size_bytes
                FROM filings
                ORDER BY printf('%010d', CAST(cik AS INTEGER)), filing_id
                """
            )
        ]
        chunks_by_filing: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in connection.execute(
            """
            SELECT chunk_evidence_id, filing_id, period_key, item, heading,
                   subheading, heading_path, source_file, char_start, char_end,
                   retrieval_text, text_masked, text_sha1
            FROM chunk_canon
            ORDER BY filing_id, source_file, char_start, chunk_evidence_id
            """
        ):
            chunks_by_filing[str(row["filing_id"])].append(dict(row))
        identities: dict[str, set[str]] = defaultdict(set)
        for row in connection.execute(
            "SELECT filing_id, entity_identifier FROM facts_canon ORDER BY filing_id"
        ):
            if row["entity_identifier"] is not None:
                identities[str(row["filing_id"])].add(str(row["entity_identifier"]))
        for filing in filings:
            filing_id = str(filing["filing_id"])
            if len(identities[filing_id]) != 1:
                raise ValueError(
                    f"Filing {filing_id} must have exactly one canonical entity identifier"
                )
            filing["source_entity_id"] = next(iter(identities[filing_id]))
    finally:
        connection.close()
    if len(filings) != ISSUER_COUNT_V24:
        raise ValueError(
            f"v2.4 cached-20 requires exactly 20 filings, found {len(filings)}"
        )
    if any(not chunks_by_filing[str(filing["filing_id"])] for filing in filings):
        raise ValueError("Every cached-20 filing must have canonical narrative chunks")
    return filings, dict(chunks_by_filing)


def _segment_items(
    chunks: Sequence[Mapping[str, Any]], *, hardened_r2: bool = False
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for chunk in chunks:
        content = str(chunk.get("retrieval_text") or chunk.get("text_masked") or "")
        if not content.strip():
            continue
        base = {
            "evidence_id": str(chunk["chunk_evidence_id"]),
            "filing_id": str(chunk["filing_id"]),
            "content": content,
            "source_system": "SEC-EDGAR-CACHED",
            "period_key": str(chunk.get("period_key") or "UNKNOWN_PERIOD"),
            "metadata": {
                key: chunk[key]
                for key in (
                    "item",
                    "heading",
                    "subheading",
                    "heading_path",
                    "source_item",
                    "source_heading",
                    "source_heading_path",
                    "item_binding_method",
                    "item_binding_scope_sha256",
                    "source_file",
                    "char_start",
                    "char_end",
                )
                if chunk.get(key) is not None
            },
        }
        chunk_segments = split_narrative_evidence_item_v23(base)
        if hardened_r2:
            for segment in chunk_segments:
                source_evidence_id = str(segment["evidence_id"])
                metadata = segment.get("metadata")
                if not isinstance(metadata, dict):
                    raise TypeError("Segment metadata must be an object")
                identity = {
                    "identity_version": "auditops.narrative_evidence_id.v2.4r2",
                    "filing_id": segment["filing_id"],
                    "source_evidence_id": source_evidence_id,
                    "parent_evidence_id": metadata.get("parent_evidence_id"),
                    "source_file": metadata.get("source_file"),
                    "char_start": metadata.get("char_start"),
                    "char_end": metadata.get("char_end"),
                    "relative_char_start": metadata.get("relative_char_start"),
                    "relative_char_end": metadata.get("relative_char_end"),
                    "segment_index": metadata.get("segment_index"),
                    "content_sha256": hashlib.sha256(
                        str(segment.get("content") or "").encode("utf-8")
                    ).hexdigest(),
                }
                segment["evidence_id"] = (
                    "v24r2e-" + canonical_json_sha256(identity)[:24]
                )
        segments.extend(chunk_segments)
    if hardened_r2:
        identifiers = [str(item["evidence_id"]) for item in segments]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("r2 narrative evidence identifiers must be unique")
    return segments


def _bind_item8_note_chunks_v24r2(
    chunks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Correct two narrowly validated canonical Item 8 metadata omissions.

    Certain inline filings put the complete annual report after Item 15 even though
    Item 8 explicitly incorporates the audited statements and notes by reference.
    Other filings preserve explicit Note 1..N headings but lose every SEC item label.
    This r2-only correction handles those two structures and preserves source labels.
    """

    rows = [dict(chunk) for chunk in chunks]

    def compact_text(row: Mapping[str, Any]) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(row.get("text_masked") or "").casefold())

    item8_references = [
        row
        for row in rows
        if str(row.get("item") or "") in {"8", "8A"}
        and "financialstatement" in compact_text(row)
        and "note" in compact_text(row)
    ]
    item8_note_headings = [
        " ".join(
            str(row.get(key) or "") for key in ("heading", "subheading", "heading_path")
        ).casefold()
        for row in rows
        if str(row.get("item") or "") in {"8", "8A"}
    ]
    if any(
        "notestoconsolidatedfinancialstatements" in re.sub(r"[^a-z0-9]+", "", label)
        or re.search(r"\bnote\s+\d+\b", label) is not None
        for label in item8_note_headings
    ):
        return rows

    phrase_rows = [
        row
        for row in rows
        if "notestoconsolidatedfinancialstatements" in compact_text(row)
    ]
    note_number_pattern = re.compile(r"\bnote\s+(\d+)\b", re.IGNORECASE)
    note_markers: list[tuple[dict[str, Any], int]] = []
    for row in rows:
        label = " ".join(
            str(row.get(key) or "") for key in ("subheading", "heading_path")
        )
        note_markers.extend(
            (row, int(value)) for value in note_number_pattern.findall(label)
        )

    def topic_series_binding(
        reference_rows: Sequence[Mapping[str, Any]],
    ) -> tuple[int, int, list[str]] | None:
        if not reference_rows:
            return None
        lower_bound = min(int(row["char_start"]) for row in reference_rows)
        topic_markers: list[tuple[dict[str, Any], str]] = []
        for row in rows:
            if int(row["char_start"]) < lower_bound:
                continue
            label = " ".join(
                str(row.get(key) or "")
                for key in ("heading", "subheading", "heading_path")
            ).casefold()
            if any(
                marker in label
                for marker in (
                    "auditor",
                    "critical audit matter",
                    "management's discussion",
                    "management’s discussion",
                    "report of independent",
                    "risk factor",
                )
            ):
                continue
            topic = _topic_rule(label, _FOOTNOTE_HEADING_RULES)
            if topic is not None:
                topic_markers.append((row, topic))
        clusters: list[list[tuple[dict[str, Any], str]]] = []
        for marker in sorted(
            topic_markers,
            key=lambda value: (
                int(value[0]["char_start"]),
                str(value[0]["chunk_evidence_id"]),
            ),
        ):
            if (
                not clusters
                or int(marker[0]["char_start"]) - int(clusters[-1][-1][0]["char_start"])
                > 35_000
            ):
                clusters.append([marker])
            else:
                clusters[-1].append(marker)
        eligible = [
            cluster
            for cluster in clusters
            if len(cluster) >= 6
            and len({topic for _, topic in cluster}) >= 4
            and len({str(row["source_file"]) for row, _ in cluster}) == 1
        ]
        if not eligible:
            return None
        cluster = min(
            eligible,
            key=lambda value: (
                -len({topic for _, topic in value}),
                -len(value),
                int(value[0][0]["char_start"]),
            ),
        )
        cluster_start = int(cluster[0][0]["char_start"])
        cluster_end = max(int(row["char_end"]) for row, _ in cluster)
        note_one_rows = [
            row
            for row, number in note_markers
            if number == 1
            and lower_bound <= int(row["char_start"]) <= cluster_end
            and str(row["source_file"]) == str(cluster[0][0]["source_file"])
        ]
        start = (
            min(int(row["char_start"]) for row in note_one_rows)
            if note_one_rows
            else cluster_start
        )
        reference_ids = sorted(
            {str(row["chunk_evidence_id"]) for row in reference_rows}
            | {str(row["chunk_evidence_id"]) for row, _ in cluster}
            | {str(row["chunk_evidence_id"]) for row in note_one_rows}
        )
        return start, cluster_end, reference_ids

    binding_version: str
    reference_ids: list[str]
    if item8_references:
        note_start_pattern = re.compile(
            r"notes to consolidated financial statements\s+note\s+<num>\s*[-–]\s*"
            r"basis of presentation",
            re.IGNORECASE,
        )
        start_rows = [
            row
            for row in rows
            if str(row.get("item") or "") not in {"8", "8A"}
            and note_start_pattern.search(str(row.get("text_masked") or "")) is not None
        ]
        ends = (
            [
                int(row["char_start"])
                for row in rows
                if start_rows
                and int(row.get("char_start") or 0)
                > max(int(item["char_start"]) for item in start_rows)
                and "glossary of terms and acronyms"
                in str(row.get("text_masked") or "").casefold()
            ]
            if start_rows
            else []
        )
        if (
            start_rows
            and len({str(row["source_file"]) for row in start_rows}) == 1
            and max(int(row["char_start"]) for row in start_rows)
            < min(int(row["char_end"]) for row in start_rows)
            and ends
        ):
            start = max(int(row["char_start"]) for row in start_rows)
            end = min(ends)
            binding_version = "auditops.item8_note_binding.v2.4r2"
            reference_ids = sorted(
                str(row["chunk_evidence_id"]) for row in item8_references
            )
        else:
            topic_binding = topic_series_binding(item8_references)
            if topic_binding is None:
                return rows
            start, end, reference_ids = topic_binding
            binding_version = "auditops.item8_topic_series_binding.v2.4r2"
    else:
        if any(str(row.get("item") or "") for row in rows):
            return rows
        note_numbers = {number for _, number in note_markers}
        start_rows = [row for row, number in note_markers if number == 1]
        if (
            len(note_numbers) >= 8
            and {1, 2}.issubset(note_numbers)
            and max(note_numbers, default=0) >= 15
            and len(phrase_rows) >= 5
            and start_rows
            and len({str(row["source_file"]) for row, _ in note_markers}) == 1
        ):
            start = min(int(row["char_start"]) for row in start_rows)
            end = max(
                int(row["char_end"])
                for row in [*[row for row, _ in note_markers], *phrase_rows]
                if int(row["char_end"]) > start
            )
            binding_version = "auditops.missing_item_note_binding.v2.4r2"
            reference_ids = sorted(
                {str(row["chunk_evidence_id"]) for row, _ in note_markers}
                | {str(row["chunk_evidence_id"]) for row in phrase_rows}
            )
        else:
            topic_binding = topic_series_binding(phrase_rows)
            if topic_binding is None:
                return rows
            start, end, reference_ids = topic_binding
            binding_version = "auditops.missing_item_topic_series_binding.v2.4r2"
    if end <= start:
        return rows

    binding_material = {
        "binding_version": binding_version,
        "scope_reference_ids": reference_ids,
        "note_start": start,
        "note_end": end,
        "source_file": str(
            next(row["source_file"] for row in rows if int(row["char_start"]) == start)
        ),
    }
    scope_sha256 = canonical_json_sha256(binding_material)
    promoted: list[dict[str, Any]] = []
    for row in rows:
        char_start = int(row.get("char_start") or 0)
        char_end = int(row.get("char_end") or char_start)
        if char_end <= start or char_start >= end:
            promoted.append(row)
            continue
        if str(row.get("item") or "") in {"8", "8A"}:
            promoted.append(row)
            continue
        corrected = dict(row)
        corrected["source_item"] = row.get("item")
        corrected["source_heading"] = row.get("heading")
        corrected["source_heading_path"] = row.get("heading_path")
        corrected["item"] = "8"
        corrected["heading"] = "Financial Statements and Supplementary Data."
        corrected["heading_path"] = (
            "Item 8 > Financial Statements and Supplementary Data. > "
            "Notes to consolidated financial statements"
        )
        corrected["item_binding_method"] = binding_material["binding_version"]
        corrected["item_binding_scope_sha256"] = scope_sha256
        promoted.append(corrected)
    return promoted


def _sentence_spans(value: str) -> Iterable[tuple[str, int, int]]:
    """Yield sentence text and exact offsets without splitting common abbreviations."""

    protected = value
    for pattern in _ABBREVIATION_PATTERNS:
        protected = pattern.sub(
            lambda match: match.group(0).replace(".", _ABBREVIATION_SENTINEL),
            protected,
        )
    for match in _SENTENCE_RE.finditer(protected):
        start, end = match.span(1)
        yield value[start:end].strip(), start, end


def _raw_cam_segments_v24(
    raw_root: Path, filing: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover a sentence-bounded CAM scope from the same frozen filing ZIP.

    Some canonical chunks omit table-laid-out CAM bodies.  This supplement is
    deliberately narrow: it reads one checksum-bound filing, extracts only the
    auditor CAM section, and creates immutable sentence evidence before any
    gold answer is selected.
    """

    if not raw_root.is_dir() or raw_root.is_symlink():
        raise ValueError("Raw filing root must be a non-symlink directory")
    accession = str(filing["accession"])
    ticker = str(filing["ticker"])
    matches = sorted((raw_root / ticker).glob(f"*{accession}*.zip"))
    if len(matches) != 1:
        raise ValueError(
            f"Filing {ticker} must bind exactly one frozen raw ZIP, found {len(matches)}"
        )
    archive = matches[0]
    if not archive.is_file() or archive.is_symlink():
        raise ValueError(f"Raw filing archive is not a regular file: {archive}")
    archive_sha256 = _sha256_path(archive)
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        if any(
            info.is_dir()
            or (info.external_attr >> 16) & 0o170000 == 0o120000
            or Path(info.filename).name != info.filename
            for info in members
        ):
            raise ValueError(f"Raw filing archive has an unsafe member: {archive.name}")
        html_members = [
            info
            for info in members
            if info.filename.casefold().endswith((".htm", ".html"))
        ]
        if len(html_members) != 1:
            raise ValueError("Raw filing archive must contain exactly one HTML filing")
        html_info = html_members[0]
        if html_info.file_size <= 0 or html_info.file_size > 100_000_000:
            raise ValueError("Raw filing HTML size is outside the approved range")
        html_payload = handle.read(html_info)
    html_sha256 = hashlib.sha256(html_payload).hexdigest()
    expected_source_sha256 = str(filing["source_sha256"])
    if expected_source_sha256 not in {archive_sha256, html_sha256}:
        raise ValueError(f"Raw filing checksum conflicts with corpus source: {ticker}")

    # Inline-XBRL documents commonly carry an XML declaration while remaining
    # HTML.  The filing is deliberately parsed as HTML; silence only this
    # parser-classification warning, not parsing errors.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html_payload, "lxml")
    for element in soup.find_all(("script", "style")):
        element.decompose()
    text = _SPACE_RE.sub(" ", soup.get_text(" ")).strip()
    lowered = text.casefold()
    positions = [
        match.start() for match in re.finditer("critical audit matter", lowered)
    ]
    section_start: int | None = None
    section_end: int | None = None
    matter_markers = (
        "description of the matter",
        "critical audit matter description",
        "principal considerations",
        "auditing the",
        "as described in note",
        "as discussed in note",
    )
    for position in positions:
        window_end = min(len(text), position + 20_000)
        window = lowered[position:window_end]
        if not any(marker in window for marker in matter_markers):
            continue
        signature_positions = [
            candidate
            for marker in (" /s/ ", " we have served as the")
            if (candidate := lowered.find(marker, position + 1_000, window_end)) >= 0
        ]
        section_start = position
        section_end = min(signature_positions) if signature_positions else window_end
        break
    if section_start is None or section_end is None:
        raise ValueError(f"Filing {ticker} lacks a recoverable raw CAM section")
    section = text[section_start:section_end].strip()
    parent_evidence_id = (
        "v24rawcam-parent-"
        + canonical_json_sha256(
            {
                "version": RAW_CAM_SUPPLEMENT_VERSION_V24,
                "archive_sha256": archive_sha256,
                "html_sha256": html_sha256,
                "section_start": section_start,
                "section_end": section_end,
            }
        )[:24]
    )
    segments: list[dict[str, Any]] = []
    for index, (sentence, relative_start, relative_end) in enumerate(
        _sentence_spans(section), start=1
    ):
        words = _WORD_RE.findall(sentence)
        if (
            not 40 <= len(sentence) <= MAX_SUPPORT_WINDOW_CHARS
            or not 7 <= len(words) <= 140
        ):
            continue
        char_start = section_start + relative_start
        char_end = section_start + relative_end
        material = {
            "version": RAW_CAM_SUPPLEMENT_VERSION_V24,
            "parent_evidence_id": parent_evidence_id,
            "segment_index": index,
            "content": sentence,
            "char_start": char_start,
            "char_end": char_end,
        }
        segments.append(
            {
                "evidence_id": "v24rawcam-" + canonical_json_sha256(material)[:24],
                "filing_id": str(filing["filing_id"]),
                "content": sentence,
                "source_system": "SEC-EDGAR-CACHED",
                "period_key": f"FY{int(filing['fiscal_year_focus'])}",
                "metadata": {
                    "item": "8",
                    "section_label": "Critical Audit Matters",
                    "source_file": html_info.filename,
                    "char_start": char_start,
                    "char_end": char_end,
                    "parent_evidence_id": parent_evidence_id,
                    "boundary_policy_version": RAW_CAM_SUPPLEMENT_VERSION_V24,
                    "segment_index": index,
                    "relative_char_start": relative_start,
                    "relative_char_end": relative_end,
                },
            }
        )
    if not segments:
        raise ValueError(
            f"Filing {ticker} raw CAM section produced no bounded sentences"
        )
    provenance = {
        "archive_name": archive.name,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive.stat().st_size,
        "html_member": html_info.filename,
        "html_sha256": html_sha256,
        "html_size_bytes": len(html_payload),
        "supplement_version": RAW_CAM_SUPPLEMENT_VERSION_V24,
        "section_text_sha256": hashlib.sha256(section.encode("utf-8")).hexdigest(),
        "sentence_count": len(segments),
    }
    return segments, provenance


def _content_sentences(content: str) -> list[str]:
    content_only = content.split("Content:\n", 1)[-1]
    candidates: list[str] = []
    for sentence, _, _ in _sentence_spans(content_only):
        if 40 <= len(sentence) <= MAX_SUPPORT_WINDOW_CHARS:
            words = _WORD_RE.findall(sentence)
            if 7 <= len(words) <= 140 and not sentence.endswith("..."):
                candidates.append(sentence)
    return candidates


def _topic_rule(
    sentence: str, rules: Sequence[tuple[tuple[str, ...], str]]
) -> str | None:
    lowered = sentence.casefold()
    for markers, topic in rules:
        if all(marker in lowered for marker in markers):
            return topic
    return None


def _section_topic(metadata: Mapping[str, Any]) -> str | None:
    generic_heading_markers = (
        "accounting polic",
        "critical audit matter",
        "financial statements",
        "independent registered public accounting",
        "notes to consolidated",
        "table of contents",
    )
    for key in ("subheading", "heading"):
        value = metadata.get(key)
        if isinstance(value, str):
            cleaned = _SPACE_RE.sub(" ", value).strip(" .:-")
            if (
                2 <= len(_WORD_RE.findall(cleaned)) <= 10
                and not any(
                    marker in cleaned.casefold()
                    for marker in (
                        *generic_heading_markers,
                        *_NON_DISCLOSURE_SECTION_MARKERS,
                    )
                )
                and not re.fullmatch(r"[A-Z]-?\s*\d+", cleaned, re.IGNORECASE)
            ):
                cleaned = re.sub(
                    r"(?i)^(?:note\s*)?\d+[A-Za-z]?[. :\-]+", "", cleaned
                ).strip()
                if 1 <= len(_WORD_RE.findall(cleaned)) <= 10:
                    return cleaned
    return None


def _footnote_topic(metadata: Mapping[str, Any]) -> str | None:
    scope = " ".join(
        str(metadata.get(key) or "")
        for key in ("heading", "subheading", "heading_path", "section_label")
    ).casefold()
    if any(
        marker in scope
        for marker in ("critical audit matter", "report of independent", "auditor")
    ):
        return None
    for key in ("subheading", "heading", "heading_path"):
        value = str(metadata.get(key) or "").casefold()
        if not value or any(
            marker in value for marker in _NON_DISCLOSURE_SECTION_MARKERS
        ):
            continue
        topic = _topic_rule(value, _FOOTNOTE_HEADING_RULES)
        if topic is not None:
            return topic
    return None


def _footnote_topic_v24r2(metadata: Mapping[str, Any], sentence: str) -> str | None:
    """Resolve a note topic only inside compatible Item 8/8A metadata."""

    if str(metadata.get("item") or "") not in {"8", "8A"}:
        return None
    scope = " ".join(
        str(metadata.get(key) or "")
        for key in ("heading", "subheading", "heading_path", "section_label")
    ).casefold()
    compact_scope = re.sub(r"[^a-z0-9]+", "", scope)
    if not any(
        marker in compact_scope for marker in ("financialstatement", "note")
    ) or any(
        marker in scope
        for marker in (
            "critical audit matter",
            "opinion on the financial statements",
            "report of independent",
        )
    ):
        return None
    sentence_topic = _topic_rule(
        _substantive_sentence_text(sentence), _FOOTNOTE_SENTENCE_RULES
    )
    metadata_topic = _recognized_heading_topic_v24r2(metadata)
    if metadata_topic in {"nature of business", "significant accounting policies"}:
        metadata_topic = None
    if sentence_topic is None:
        return None
    if metadata_topic is not None and metadata_topic != sentence_topic:
        return None
    if metadata_topic is None and not _generic_note_or_policy_heading_v24r2(metadata):
        return None
    return metadata_topic or sentence_topic


def _recognized_heading_topic_v24r2(metadata: Mapping[str, Any]) -> str | None:
    """Return a canonical topic for a substantive note/policy heading."""

    heading_rules = (
        *_FOOTNOTE_HEADING_RULES,
        (("debt",), "debt obligations"),
        (("revenue",), "revenue recognition"),
        (("tax",), "income taxes"),
        (("postretirement",), "pension obligations"),
        (("retirement",), "pension obligations"),
        (("benefit obligation",), "pension obligations"),
        (("fair value",), "fair value measurements"),
        (("software",), "capitalized software"),
    )
    for key in ("subheading", "heading_path", "heading"):
        value = str(metadata.get(key) or "")
        if not value:
            continue
        topic = _topic_rule(value, heading_rules)
        if topic is not None:
            return topic
    return None


def _generic_note_or_policy_heading_v24r2(metadata: Mapping[str, Any]) -> bool:
    scope = " ".join(
        str(metadata.get(key) or "")
        for key in ("heading", "subheading", "heading_path")
    ).casefold()
    return any(
        marker in scope
        for marker in (
            "notes to consolidated financial statements",
            "summary of significant accounting policies",
            "significant accounting policies",
        )
    )


def _policy_topic_is_direct_v24r2(sentence: str, *, topic: str) -> bool:
    """Require a direct accounting treatment, not an incidental topic mention."""

    substantive = normalize_support_text(_substantive_sentence_text(sentence))
    lowered = substantive.casefold()
    patterns: Mapping[str, tuple[str, ...]] = {
        "revenue recognition": (
            r"^(?:unearned\s+)?revenue\b.{0,100}\b(?:is|are)\s+(?:recorded|recognized)\b",
            r"^(?:the\s+company|we|our\s+business)\b.{0,80}\brecogniz(?:e|es)\s+revenue\b",
            r"^for\s+revenue\b.{0,120}\brecogniz(?:e|es|ed)\b",
        ),
        "lease accounting": (
            r"^(?:the\s+company|the\s+firm|we)\b.{0,120}\b(?:account\w*|recogniz\w*|record\w*|measur\w*|impair\w*|amortiz\w*)\b.{0,100}\blease",
            r"^(?:the\s+company|the\s+firm|we)\b.{0,100}\blease\b.{0,120}\b(?:account\w*|recogniz\w*|record\w*|measur\w*|impair\w*|amortiz\w*)",
            r"^(?:operating|finance|leased|leasehold|sale-leaseback)\b.{0,160}\b(?:account\w*|recogniz\w*|record\w*|measur\w*|impair\w*|amortiz\w*)",
        ),
        "income taxes": (
            r"^(?:the\s+company|the\s+corporation|we|income\s+tax|current\s+income\s+tax|deferred\s+income\s+tax|the\s+benefits\s+of\s+uncertain\s+tax)\b.{0,180}\b(?:account\w*|recogniz\w*|record\w*|measur\w*|stated)",
        ),
        "foreign-currency translation": (
            r"^(?:the\s+company|we|our|a\s+portion|foreign\s+currency|translation)\b.{0,180}\b(?:record\w*|recogniz\w*|translat\w*)",
        ),
        "research and development costs": (
            r"^research\s+and\s+development\s+costs\b.{0,100}\b(?:expens\w*|capitaliz\w*|record\w*|recogniz\w*)",
        ),
    }
    topic_patterns = patterns.get(topic)
    if topic_patterns is not None:
        return any(
            re.search(pattern, lowered) is not None for pattern in topic_patterns
        )
    topic_terms = _topic_anchor_terms(topic)
    early = " ".join(_WORD_RE.findall(substantive)[:24]).casefold()
    return (
        any(term in early for term in topic_terms)
        and _DISCLOSURE_PREDICATE_RE.search(substantive) is not None
    )


def _policy_heading_is_compatible_v24r2(
    heading_topic: str | None, *, policy_topic: str
) -> bool:
    if heading_topic is None:
        return False
    equivalents = {
        "lease accounting": {"leases"},
        "inventory valuation": {"inventory"},
        "goodwill impairment": {"goodwill"},
        "property and equipment": {"property and equipment"},
        "share-based compensation": {"share-based compensation"},
        "income taxes": {"income taxes"},
        "revenue recognition": {"revenue recognition"},
        "research and development costs": {"research and development costs"},
    }
    return heading_topic == policy_topic or heading_topic in equivalents.get(
        policy_topic, set()
    )


def _disclosure_attribute_v24r2(sentence: str, *, subtype: str) -> str:
    lowered = normalize_support_text(_substantive_sentence_text(sentence)).casefold()
    if subtype == "auditor_report_opinion_language":
        return "fair-presentation conclusion"
    if subtype == "critical_audit_matter":
        return "underlying facts and estimates"
    rules = (
        (("impair",), "impairment treatment"),
        (("amortiz",), "amortization treatment"),
        (("depreciat",), "depreciation treatment"),
        (("capitaliz",), "capitalization treatment"),
        (("classif",), "classification"),
        (("measur", "valu"), "measurement or valuation"),
        (("recogniz",), "recognition"),
        (("record",), "recording"),
        (("accounted for", "account for"), "accounting treatment"),
        (("allocat",), "allocation"),
        (("consist", "include", "represent"), "composition or nature"),
        (("fund",), "funding"),
        (("use",), "use or method"),
    )
    for markers, label in rules:
        if any(marker in lowered for marker in markers):
            return label
    return "reported treatment"


def _substantive_sentence_text(sentence: str) -> str:
    paragraphs = [
        part.strip() for part in re.split(r"\n\s*\n", sentence) if part.strip()
    ]
    return paragraphs[-1] if paragraphs else sentence.strip()


def _topic(
    metadata: Mapping[str, Any], sentence: str, *, subtype: str | None = None
) -> str:
    if subtype == "footnote_note":
        sentence_topic = _topic_rule(
            _substantive_sentence_text(sentence), _FOOTNOTE_SENTENCE_RULES
        )
        if sentence_topic is not None:
            return sentence_topic
        footnote_topic = _footnote_topic(metadata)
        if footnote_topic is not None:
            return footnote_topic
    elif subtype == "accounting_policy":
        policy_topic = _topic_rule(
            _substantive_sentence_text(sentence), _POLICY_TOPIC_RULES
        )
        if policy_topic is not None:
            return policy_topic
    elif subtype == "critical_audit_matter":
        cam_topic = _topic_rule(sentence, _CAM_TOPIC_RULES)
        if cam_topic is not None:
            return cam_topic
    section_topic = _section_topic(metadata)
    if section_topic is not None:
        return section_topic
    meaningful = [
        token
        for token in _WORD_RE.findall(sentence)
        if token.casefold() not in _STOPWORDS
    ]
    return " ".join(meaningful[:6]) or "the disclosed matter"


def _anchors(sentence: str) -> list[str]:
    matches = list(_WORD_RE.finditer(sentence))
    for width in (4, 3, 2):
        for index in range(max(0, len(matches) - width + 1)):
            phrase_matches = matches[index : index + width]
            words = [match.group(0) for match in phrase_matches]
            if sum(word.casefold() not in _STOPWORDS for word in words) >= 2:
                # Preserve intervening punctuation. Joining regex tokens with
                # spaces can manufacture a phrase that never occurs in the
                # approved support window (for example across a comma).
                anchor = sentence[phrase_matches[0].start() : phrase_matches[-1].end()]
                if normalize_support_text(anchor) in normalize_support_text(sentence):
                    return [anchor]
    meaningful = [
        match.group(0)
        for match in matches
        if match.group(0).casefold() not in _STOPWORDS
    ]
    if not meaningful:
        raise ValueError("Could not derive a semantic anchor")
    return [meaningful[0]]


def _anchor_token_key(value: str) -> str:
    token = value.casefold().strip(".'’-_")
    for suffix in ("ments", "ment", "ities", "ity", "ing", "ions", "ion", "es", "s"):
        if len(token) > len(suffix) + 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _topic_anchor_terms(topic: str) -> set[str]:
    terms = {
        _anchor_token_key(token)
        for token in _WORD_RE.findall(topic)
        if token.casefold() not in _STOPWORDS
        and token.casefold() != "num"
        and len(_anchor_token_key(token)) >= 3
    }
    for alias in _TOPIC_ANCHOR_ALIASES_V24R2.get(topic.casefold(), ()):
        terms.update(
            _anchor_token_key(token)
            for token in _WORD_RE.findall(alias)
            if token.casefold() not in _STOPWORDS
            and token.casefold() != "num"
            and len(_anchor_token_key(token)) >= 3
        )
    return terms


def _is_heading_like_anchor_v24r2(anchor: str) -> bool:
    lowered = normalize_support_text(anchor).casefold().strip(" .:-")
    if any(
        marker in lowered
        for marker in (
            "table of contents",
            "annual report",
            "critical audit matter description",
            "description of",
            "description of the matter",
            "matter as described",
            "part ii item",
        )
    ):
        return True
    if lowered.endswith(" description"):
        return True
    return bool(re.match(r"^\d+\s+", lowered)) or (
        lowered.startswith(("valuation of ", "critical audit matter"))
        or lowered.endswith(" critical audit matter")
    )


def _semantic_anchors_v24r2(sentence: str, *, topic: str) -> list[str]:
    """Return one exact, topic-bearing anchor or reject the source sentence.

    r1 selected the first superficially meaningful phrase in a sentence.  That
    produced anchors such as ``We also have other`` and page headings.  r2
    instead centers a compact exact substring on the requested topic.
    """

    matches = list(_WORD_RE.finditer(sentence))
    topic_terms = _topic_anchor_terms(topic)
    if not topic_terms:
        raise ValueError("Narrative topic has no informative anchor terms")
    normalized_tokens = [_anchor_token_key(match.group(0)) for match in matches]
    topic_positions = [
        index
        for index, token in enumerate(normalized_tokens)
        if token != "num"
        and any(
            token == term or token.startswith(term) or term.startswith(token)
            for term in topic_terms
        )
    ]
    candidates: list[tuple[int, int, int, str]] = []
    for position in topic_positions:
        for width in range(3, min(12, len(matches)) + 1):
            minimum_start = max(0, position - width + 1)
            maximum_start = min(position, len(matches) - width)
            for start in range(minimum_start, maximum_start + 1):
                window = matches[start : start + width]
                words = [match.group(0) for match in window]
                lowered = [word.casefold() for word in words]
                informative = [
                    word for word in lowered if word not in _STOPWORDS and word != "num"
                ]
                if len(informative) < 2 or lowered.count("num") > 0:
                    continue
                anchor = sentence[window[0].start() : window[-1].end()]
                anchor_lowered = normalize_support_text(anchor).casefold()
                if _is_heading_like_anchor_v24r2(anchor_lowered):
                    continue
                if _DISCLOSURE_PREDICATE_RE.search(anchor) is None:
                    continue
                covered_terms = sum(
                    any(
                        token == term
                        or token.startswith(term)
                        or term.startswith(token)
                        for token in (_anchor_token_key(word) for word in informative)
                    )
                    for term in topic_terms
                )
                # Prefer the shortest window that captures the most topic terms,
                # then the most informative wording and earliest exact location.
                candidates.append((-covered_terms, width, -len(informative), anchor))
    if not candidates:
        raise ValueError("Could not derive a topic-bearing semantic anchor")
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3].casefold()))
    anchor = candidates[0][3]
    validate_semantic_anchor_v24r2(anchor, support_window=sentence, topic=topic)
    return [anchor]


def validate_semantic_anchor_v24r2(
    anchor: str, *, support_window: str, topic: str
) -> None:
    words = _WORD_RE.findall(anchor)
    if not 3 <= len(words) <= 12:
        raise ValueError("r2 semantic anchors must contain 3-12 words")
    if (
        normalize_support_text(anchor).casefold()
        not in normalize_support_text(support_window).casefold()
    ):
        raise ValueError("r2 semantic anchor is not an exact support substring")
    informative = [
        _anchor_token_key(word)
        for word in words
        if word.casefold() not in _STOPWORDS and word.casefold() != "num"
    ]
    if len(informative) < 2 or any(word.casefold() == "num" for word in words):
        raise ValueError("r2 semantic anchor lacks informative lexical content")
    topic_terms = _topic_anchor_terms(topic)
    if not any(
        token == term or token.startswith(term) or term.startswith(token)
        for token in informative
        for term in topic_terms
    ):
        raise ValueError("r2 semantic anchor does not contain a topic term")
    if _is_heading_like_anchor_v24r2(anchor):
        raise ValueError("r2 semantic anchor contains a page or section heading")
    if _DISCLOSURE_PREDICATE_RE.search(anchor) is None:
        raise ValueError("r2 semantic anchor lacks an answer-bearing predicate")


def _positive_candidate_score(
    subtype: str,
    item: Mapping[str, Any],
    sentence: str,
    sentence_index: int,
    *,
    hardened_r2: bool = False,
) -> int | None:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    scope = " ".join(
        str(value or "")
        for value in (
            metadata.get("item"),
            metadata.get("heading"),
            metadata.get("subheading"),
            metadata.get("heading_path"),
            metadata.get("section_label"),
        )
    ).casefold()
    lowered = sentence.casefold()
    if (
        subtype != "auditor_report_opinion_language" and sentence.count("<NUM>") > 2
    ) or "table of co ntents" in lowered:
        return None
    if subtype == "auditor_report_opinion_language":
        if "in our opinion" not in lowered or "financial statement" not in lowered:
            return None
        return 600 + 30 * int("present fairly" in lowered) - sentence_index
    if subtype == "critical_audit_matter":
        if "critical audit matter" not in scope:
            return None
        boilerplate = (
            "audit procedures",
            "communication of critical audit matter",
            "critical audit matters are matters",
            "does not alter our opinion",
            "how the critical audit matter was addressed",
            "our audit included",
            "procedures included",
            "testing controls",
            "testing the effectiveness",
            "principal considerations",
            "those standards require",
            "we are a public accounting firm",
            "we assessed",
            "we conducted our audit",
            "we evaluated",
            "we involved",
            "we obtained",
            "we performed",
            "we tested",
        )
        if any(marker in lowered for marker in boilerplate):
            return None
        topic = _topic_rule(sentence, _CAM_TOPIC_RULES)
        if topic is None or len(_WORD_RE.findall(sentence)) < 10:
            return None
        description_bonus = 60 * int(
            any(
                marker in lowered
                for marker in (
                    "as described in note",
                    "as discussed in note",
                    "critical audit matter description",
                    "description of the matter",
                    "refer to note",
                )
            )
        )
        substance_bonus = 15 * int(
            any(
                marker in lowered
                for marker in ("company", "firm", "management", "corporation")
            )
        )
        return 500 + description_bonus + substance_bonus - sentence_index
    if subtype == "accounting_policy":
        if any(
            marker in scope
            for marker in ("critical audit matter", "independent registered")
        ):
            return None
        substantive = _substantive_sentence_text(sentence)
        substantive_lowered = substantive.casefold()
        topic = _topic_rule(substantive, _POLICY_TOPIC_RULES)
        policy_verbs = (
            "accounted for",
            "are expensed",
            "are measured",
            "are recorded",
            "are recognized",
            "are valued",
            "is expensed",
            "is measured",
            "is recorded",
            "is recognized",
            "is valued",
            "recognize revenue",
            "prepared in accordance",
            "prepared in conformity",
            "recognizes ",
            "records ",
            "we recognize",
        )
        heading_topic = (
            _recognized_heading_topic_v24r2(metadata) if hardened_r2 else None
        )
        if topic is None or not any(
            marker in substantive_lowered for marker in policy_verbs
        ):
            return None
        if hardened_r2 and (
            str(metadata.get("item") or "") not in {"8", "8A"}
            or (
                heading_topic is not None
                and heading_topic
                not in {"significant accounting policies", "nature of business"}
                and not _policy_heading_is_compatible_v24r2(
                    heading_topic, policy_topic=topic
                )
            )
            or (
                heading_topic is None
                and not _generic_note_or_policy_heading_v24r2(metadata)
            )
            or not _policy_topic_is_direct_v24r2(substantive, topic=topic)
        ):
            return None
        return 400 + 20 * int(_section_topic(metadata) is not None) - sentence_index
    if subtype == "footnote_note":
        metadata_topic = _footnote_topic(metadata)
        substantive = _substantive_sentence_text(sentence)
        sentence_topic = _topic_rule(substantive, _FOOTNOTE_SENTENCE_RULES)
        hardened_topic = (
            _footnote_topic_v24r2(metadata, sentence) if hardened_r2 else None
        )
        item_number = str(metadata.get("item") or "")
        explicit_note_heading = any(
            re.search(r"(?i)\bnote\s+\d+", str(metadata.get(key) or ""))
            for key in ("subheading", "heading", "heading_path")
        )
        if (
            sentence_topic is None
            or _DISCLOSURE_PREDICATE_RE.search(substantive) is None
            or "in our opinion" in lowered
            or "critical audit matter" in lowered
            or (
                hardened_r2
                and (item_number not in {"8", "8A"} or hardened_topic is None)
            )
            or (
                metadata_topic is None
                and item_number not in {"8", "8A"}
                and not explicit_note_heading
            )
        ):
            return None
        return 300 + 20 * int(bool(metadata.get("subheading"))) - sentence_index
    raise ValueError(f"Unsupported narrative subtype: {subtype}")


def _question_for_positive(
    *,
    subtype: str,
    ticker: str,
    fiscal_year: int,
    topic: str,
    attribute: str | None = None,
) -> str:
    topic = " ".join(_WORD_RE.findall(topic)[:10])
    if subtype == "footnote_note" and attribute is not None:
        question = (
            f"What does {ticker}'s fiscal {fiscal_year} {topic} footnote disclose "
            f"about its {attribute}?"
        )
    elif subtype == "footnote_note":
        question = (
            f"What exact statement does {ticker}'s fiscal {fiscal_year} {topic} "
            "footnote disclose?"
        )
    elif subtype == "accounting_policy" and attribute is not None:
        question = (
            f"What accounting policy does {ticker}'s fiscal {fiscal_year} Form 10-K "
            f"disclose about the {attribute} of {topic}?"
        )
    elif subtype == "accounting_policy":
        question = (
            f"What accounting policy does {ticker} disclose for {topic} in its fiscal "
            f"{fiscal_year} Form 10-K?"
        )
    elif subtype == "auditor_report_opinion_language":
        question = (
            f"What exact opinion language does the independent auditor use for "
            f"{ticker}'s fiscal {fiscal_year} financial statements?"
        )
    elif subtype == "critical_audit_matter" and attribute is not None:
        question = (
            f"What {attribute} does {ticker}'s fiscal {fiscal_year} auditor report cite "
            f"for the critical audit matter concerning {topic}?"
        )
    else:
        question = (
            f"What exact language in {ticker}'s fiscal {fiscal_year} auditor report "
            f"describes the critical audit matter concerning {topic}?"
        )
    validate_natural_question(question)
    return question


def _retrieval_query_v24r2(*, subtype: str, topic: str, attribute: str | None) -> str:
    """Build a deterministic query only from inference-visible task semantics."""

    aliases = (
        _TOPIC_RETRIEVAL_ALIASES_V24R2.get(topic.casefold(), ())
        if subtype in {"accounting_policy", "auditor_report_opinion_language"}
        else ()
    )
    return " ".join(
        [
            topic,
            *([attribute] if attribute else []),
            *aliases,
        ]
    )


def _resolved_positive_topic_v24r2(
    subtype: str, item: Mapping[str, Any], sentence: str
) -> str | None:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    if subtype == "footnote_note":
        return _footnote_topic_v24r2(metadata, sentence)
    if subtype == "accounting_policy":
        return _topic_rule(_substantive_sentence_text(sentence), _POLICY_TOPIC_RULES)
    if subtype == "auditor_report_opinion_language":
        lowered = sentence.casefold()
        return (
            "opinion on the financial statements"
            if "in our opinion" in lowered and "financial statement" in lowered
            else None
        )
    if subtype == "critical_audit_matter":
        return _topic_rule(sentence, _CAM_TOPIC_RULES)
    raise ValueError(f"Unsupported narrative subtype: {subtype}")


def _sentence_mentions_topic_v24r2(sentence: str, *, topic: str) -> bool:
    sentence_terms = {
        _anchor_token_key(token)
        for token in _WORD_RE.findall(sentence)
        if token.casefold() not in _STOPWORDS and token.casefold() != "num"
    }
    return any(
        token == term or token.startswith(term) or term.startswith(token)
        for token in sentence_terms
        for term in _topic_anchor_terms(topic)
    )


def _anchor_topic_from_question_v24r2(question: str) -> str:
    lowered = normalize_support_text(question).casefold()
    matching_topics = [
        topic for topic in _TOPIC_ANCHOR_ALIASES_V24R2 if topic in lowered
    ]
    if "opinion language" in lowered:
        matching_topics.append("opinion on the financial statements")
    return max(matching_topics, key=len) if matching_topics else question


def _expanded_support_window_v24r2(
    *,
    subtype: str,
    item: Mapping[str, Any],
    sentence_index: int,
    topic: str,
) -> str:
    """Expand a scored sentence only through contiguous same-topic sentences."""

    sentences = _content_sentences(str(item.get("content") or ""))
    if not 0 <= sentence_index < len(sentences):
        raise IndexError("Positive support sentence index is outside its evidence item")

    def substantive(index: int) -> str:
        sentence = sentences[index]
        return (
            _substantive_sentence_text(sentence)
            if subtype in {"footnote_note", "accounting_policy"}
            else sentence
        )

    def compatible(index: int) -> bool:
        sentence = sentences[index]
        resolved = _resolved_positive_topic_v24r2(subtype, item, sentence)
        return resolved == topic or (
            resolved is None
            and _sentence_mentions_topic_v24r2(substantive(index), topic=topic)
        )

    def contiguous(indexes: range) -> bool:
        proposed = " ".join(substantive(index) for index in indexes)
        return is_contiguous_text_supported(proposed, str(item.get("content") or ""))

    start = end = sentence_index
    # At most five complete sentences.  Expansion remains contiguous and stops
    # at the first topic boundary in either direction.
    for _ in range(2):
        candidate = start - 1
        if candidate < 0 or not compatible(candidate):
            break
        proposed = " ".join(substantive(index) for index in range(candidate, end + 1))
        if len(proposed) > MAX_SUPPORT_WINDOW_CHARS or not contiguous(
            range(candidate, end + 1)
        ):
            break
        start = candidate
    for _ in range(2):
        candidate = end + 1
        if candidate >= len(sentences) or not compatible(candidate):
            break
        proposed = " ".join(substantive(index) for index in range(start, candidate + 1))
        if len(proposed) > MAX_SUPPORT_WINDOW_CHARS or not contiguous(
            range(start, candidate + 1)
        ):
            break
        end = candidate
    window = " ".join(substantive(index) for index in range(start, end + 1))
    _validate_complete_sentence(window, field="support_window")
    if not is_contiguous_text_supported(window, str(item.get("content") or "")):
        raise ValueError(
            "Expanded support window is not contiguous in its evidence item"
        )
    return window


def _acceptable_answer_sets_v24r2(
    *,
    subtype: str,
    topic: str,
    evidence: Sequence[Mapping[str, Any]],
    required_source_evidence_id: str,
    required_source_sentence: str,
) -> list[dict[str, Any]]:
    """Bind one exact answer set to the builder-selected source evidence item."""

    matching_items = [
        item
        for item in evidence
        if str(item.get("evidence_id") or "") == required_source_evidence_id
    ]
    if len(matching_items) != 1:
        raise ValueError(
            "Frozen evidence must contain the selected source item exactly once"
        )
    item = matching_items[0]
    source_sentences = _content_sentences(str(item.get("content") or ""))
    normalized_required = normalize_support_text(required_source_sentence).casefold()
    sentence_indexes = [
        index
        for index, candidate in enumerate(source_sentences)
        if normalize_support_text(candidate).casefold() == normalized_required
        or normalize_support_text(candidate).casefold().endswith(normalized_required)
    ]
    if len(sentence_indexes) != 1:
        raise ValueError("Selected source sentence must occur exactly once in its item")
    sentence_index = sentence_indexes[0]
    substantive = (
        _substantive_sentence_text(required_source_sentence)
        if subtype in {"footnote_note", "accounting_policy"}
        else required_source_sentence
    )
    window = _expanded_support_window_v24r2(
        subtype=subtype,
        item=item,
        sentence_index=sentence_index,
        topic=topic,
    )
    anchors = _semantic_anchors_v24r2(substantive, topic=topic)
    return [
        {
            "set_id": "set-1",
            "extracts": [
                {
                    "evidence_id": required_source_evidence_id,
                    "support_window": window,
                    "required_semantic_anchors": anchors,
                }
            ],
        }
    ]


def _negative_question(
    *, subtype: str, ticker: str, fiscal_year: int, claim: str
) -> str:
    if subtype == "footnote_note":
        prefix = f"Does {ticker}'s fiscal {fiscal_year} footnote disclosure state that {claim}"
    elif subtype == "accounting_policy":
        prefix = f"Does {ticker}'s fiscal {fiscal_year} accounting-policy disclosure state that {claim}"
    elif subtype == "auditor_report_opinion_language":
        prefix = (
            f"Does the independent auditor's opinion language for {ticker}'s fiscal "
            f"{fiscal_year} statements state that {claim}"
        )
    else:
        prefix = (
            f"Does {ticker}'s fiscal {fiscal_year} auditor report state that {claim}"
        )
    question = prefix + "?"
    validate_natural_question(question)
    return question


def _negative_spec_matches_scope_v24r2(
    spec: Mapping[str, Any], normalized_scope: str
) -> bool:
    claim = normalize_support_text(str(spec.get("claim") or "")).casefold()
    if claim and claim in normalized_scope:
        return True
    groups = spec.get("concept_groups")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise TypeError("r2 negative concept_groups must be an array")
    units = [
        unit.strip()
        for unit in re.split(r"(?<=[.!?])\s+|\n+", normalized_scope)
        if unit.strip()
    ]
    # Reasonable-equivalent concepts must describe one local disclosure, not
    # merely occur somewhere in a long filing. Adjacent sentences cover the
    # common subject-then-predicate layout while avoiding filing-wide joins.
    windows = [*units]
    windows.extend(
        f"{units[index]} {units[index + 1]}"
        for index in range(len(units) - 1)
        if len(units[index]) + len(units[index + 1]) <= 2_000
    )

    def negative_token_key(value: str) -> str:
        token = value.casefold().strip(".'’-_ ")
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        token = _anchor_token_key(token)
        if len(token) > 6 and token.endswith("ed"):
            token = token[:-2]
        return token

    window_records = [
        (
            window,
            [negative_token_key(token) for token in _WORD_RE.findall(window)],
        )
        for window in windows
    ]

    def alternative_matches(
        value: Any, *, window: str, window_tokens: list[str]
    ) -> bool:
        alternative = normalize_support_text(str(value)).casefold()
        if alternative in window:
            return True
        alternative_tokens = [
            negative_token_key(token) for token in _WORD_RE.findall(alternative)
        ]

        def lexical_match(left: str, right: str) -> bool:
            # ``right`` is an explicit catalog term and may intentionally be a
            # stem (for example ``capitaliz``).  A shorter source token must not
            # match a longer catalog term, and a shared prefix is not a synonym.
            return left == right or (len(right) >= 7 and left.startswith(right))

        return bool(alternative_tokens) and all(
            any(lexical_match(scope_token, token) for scope_token in window_tokens)
            for token in alternative_tokens
        )

    return bool(groups) and any(
        all(
            isinstance(group, Sequence)
            and not isinstance(group, (str, bytes))
            and any(
                alternative_matches(
                    alternative,
                    window=window,
                    window_tokens=window_tokens,
                )
                for alternative in group
            )
            for group in groups
        )
        for window, window_tokens in window_records
    )


def _task_source_row(
    *,
    filing: Mapping[str, Any],
    task_id: str,
    subtype: str,
    question: str,
    retrieval_query: str,
    evidence_items: Sequence[Mapping[str, Any]],
    answer: str | None,
    expected_ids: Sequence[str],
    negative_type: str | None,
    hardened_r2: bool = False,
) -> dict[str, Any]:
    fiscal_year = int(filing["fiscal_year_focus"])
    inference_evidence: list[dict[str, Any]] = []
    internal_metadata_fields = {
        "source_item",
        "source_heading",
        "source_heading_path",
        "item_binding_method",
        "item_binding_scope_sha256",
    }
    for item in evidence_items:
        materialized = copy.deepcopy(dict(item))
        metadata = materialized.get("metadata")
        if isinstance(metadata, dict):
            content_lowered = str(materialized.get("content") or "").casefold()
            if (
                hardened_r2
                and subtype == "auditor_report_opinion_language"
                and "in our opinion" in content_lowered
                and "financial statement" in content_lowered
            ):
                source_item = str(metadata.get("item") or "AUDITOR_REPORT")
                metadata["heading"] = "Independent auditor's report"
                metadata["subheading"] = "Opinion on the financial statements"
                metadata["heading_path"] = (
                    f"Item {source_item} > Independent auditor's report > "
                    "Opinion on the financial statements"
                )
                metadata["section_label"] = "AUDITOR_OPINION_FINANCIAL_STATEMENTS"
            for field in internal_metadata_fields:
                metadata.pop(field, None)
        inference_evidence.append(materialized)
    return {
        "task_id": task_id,
        "task_type": "narrative_citation",
        "template_id": f"v24:{subtype}:{'answer' if answer else 'refusal'}",
        "task_family": f"narrative_citation:{subtype}:{'answerable' if answer else 'unanswerable'}",
        "filing_id": str(filing["filing_id"]),
        "ticker": str(filing["ticker"]),
        "entity_name": str(filing["ticker"]),
        "source_entity_id": str(filing["source_entity_id"]),
        "form_type": str(filing["form_type"]),
        "period_key": f"FY{fiscal_year}",
        "period_end": str(filing["report_date"]),
        "question": question,
        "retrieval_query": retrieval_query,
        "label": subtype,
        "scope_type": "filing",
        "scope_key": str(filing["filing_id"]),
        "answerability": "ANSWERABLE" if answer else "UNANSWERABLE",
        "expected_chunk_ids": list(expected_ids),
        "extractive_answer": answer,
        "refusal_code": None if answer else "NARRATIVE_NOT_SUPPORTED",
        "negative_type": negative_type,
        "evidence_items": inference_evidence,
        "jurisdiction": "US",
        "source_system": "SEC-EDGAR-CACHED",
        "reporting_framework": "US-GAAP",
        "standards_version": "PCAOB_PUBLIC_FILING_METADATA_V0.1",
        "filing_metadata": {
            "ticker": filing["ticker"],
            "form_type": filing["form_type"],
            "report_date": filing["report_date"],
            "accession": filing["accession"],
        },
    }


def _package_task(
    row: Mapping[str, Any], task_type: str, *, corpus_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    source_system = "SEC-EDGAR-CACHED"
    evidence = _build_evidence_items(row, task_type, source_system)
    base_task = _build_case(
        row,
        task_type,
        jurisdiction="US",
        corpus_id=corpus_id,
        source_system=source_system,
        reporting_framework="US-GAAP",
        standards_version="PCAOB_PUBLIC_FILING_METADATA_V0.1",
        evidence_items=evidence,
    )
    task = _convert_task(
        base_task, task_id=str(row["task_id"]), evidence_items=evidence
    )
    base_gold = _build_gold(row, task_type, jurisdiction="US", corpus_id=corpus_id)
    gold = _convert_gold(
        base_gold,
        task_id=str(row["task_id"]),
        task=task,
        evidence_items=evidence,
    )
    operation = task_operation_spec(task_type, version="v2.3")
    observation = (
        evaluate_quant_evidence(task, evidence)
        if task_type == "quant_metric"
        else load_frozen_evidence(task, evidence)
    )
    build_context_pack(task, evidence, token_counter=lambda _: 0)
    return (
        task,
        evidence,
        gold,
        {
            "task_id": task["task_id"],
            "tool_name": operation.operation,
            "observation": observation,
        },
    )


def _select_quant_rows(
    rows: Sequence[Mapping[str, Any]],
    filings: Sequence[Mapping[str, Any]],
    seed: int,
    *,
    identity_namespace: str = BENCHMARK_VERSION_V24,
) -> list[dict[str, Any]]:
    by_entity_status: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        status = _quant_status(row)
        target = _quant_gold(row)
        if status == "REFUSAL" and target.get("refusal_code") != "MISSING_INPUT":
            continue
        by_entity_status[(_source_entity_id(row), status)].append(row)
    selected: list[dict[str, Any]] = []
    for issuer_index, filing in enumerate(filings):
        entity_id = str(filing["source_entity_id"])
        quotas = {
            "OK": 8 if issuer_index < 10 else 7,
            "REFUSAL": 7 if issuer_index < 10 else 8,
        }
        for status in ("OK", "REFUSAL"):
            candidates = by_entity_status[(entity_id, status)]
            if len(candidates) < quotas[status]:
                raise ValueError(
                    f"Issuer {filing['ticker']} lacks {quotas[status]} quantitative {status} cases"
                )
            by_metric: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in candidates:
                by_metric[
                    str(row.get("metric_spec_id") or row.get("operation"))
                ].append(row)
            metric_ids = sorted(by_metric)
            offset = issuer_index % max(1, len(metric_ids))
            ordered_metrics = metric_ids[offset:] + metric_ids[:offset]
            chosen: list[Mapping[str, Any]] = []
            for metric_id in ordered_metrics:
                options = sorted(
                    by_metric[metric_id],
                    key=lambda row: (
                        _stable_rank(seed, f"quant-{entity_id}-{status}", row),
                        str(row["task_id"]),
                    ),
                )
                chosen.append(options[0])
                if len(chosen) == quotas[status]:
                    break
            if len(chosen) < quotas[status]:
                used = {str(row["task_id"]) for row in chosen}
                remaining = sorted(
                    (row for row in candidates if str(row["task_id"]) not in used),
                    key=lambda row: (
                        _stable_rank(seed, f"quant-fill-{entity_id}-{status}", row),
                        str(row["task_id"]),
                    ),
                )
                chosen.extend(remaining[: quotas[status] - len(chosen)])
            for row in chosen:
                materialized = copy.deepcopy(dict(row))
                parent_id = str(materialized["task_id"])
                materialized["task_id"] = (
                    "v24q-"
                    + canonical_json_sha256(
                        {
                            "version": identity_namespace,
                            "source_task_id": parent_id,
                            "source_sha256": canonical_json_sha256(row),
                        }
                    )[:24]
                )
                materialized["v24_parent_task_id"] = parent_id
                selected.append(materialized)
    if len(selected) != QUANT_COUNT_V24:
        raise AssertionError(
            "Quantitative quota construction did not produce 300 cases"
        )
    return selected


def _build_positive_narrative(
    *,
    filing: Mapping[str, Any],
    subtype: str,
    ordinal: int,
    all_segments: Sequence[Mapping[str, Any]],
    seed: int,
    excluded_support_windows: frozenset[str] = frozenset(),
    identity_namespace: str = BENCHMARK_VERSION_V24,
    hardened_r2: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ranked: list[tuple[int, str, Mapping[str, Any], str]] = []
    for item in all_segments:
        for sentence_index, sentence in enumerate(
            _content_sentences(str(item.get("content") or ""))
        ):
            score = _positive_candidate_score(
                subtype,
                item,
                sentence,
                sentence_index,
                hardened_r2=hardened_r2,
            )
            if score is None:
                continue
            support_window = (
                _substantive_sentence_text(sentence)
                if subtype in {"footnote_note", "accounting_policy"}
                else sentence
            )
            _validate_complete_sentence(support_window, field="support_window")
            ranked.append(
                (
                    -score,
                    _stable_rank(
                        seed,
                        f"positive-{filing['filing_id']}-{subtype}",
                        {"id": item["evidence_id"], "sentence": support_window},
                    ),
                    item,
                    support_window,
                )
            )
    ranked.sort(key=lambda value: (value[0], value[1], str(value[2]["evidence_id"])))
    used_topics: set[str] = set()
    accepted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for _, _, source_item, sentence in ranked:
        normalized_sentence = normalize_support_text(sentence).casefold()
        if any(
            normalized_sentence == excluded
            or normalized_sentence in excluded
            or excluded in normalized_sentence
            for excluded in excluded_support_windows
        ):
            continue
        metadata = (
            source_item.get("metadata")
            if isinstance(source_item.get("metadata"), Mapping)
            else {}
        )
        topic = (
            _resolved_positive_topic_v24r2(subtype, source_item, sentence)
            if hardened_r2
            else _topic(metadata, sentence, subtype=subtype)
        )
        if topic is None:
            continue
        topic_key = topic.casefold()
        if topic_key in used_topics:
            continue
        attribute = (
            _disclosure_attribute_v24r2(sentence, subtype=subtype)
            if hardened_r2
            else None
        )
        question = _question_for_positive(
            subtype=subtype,
            ticker=str(filing["ticker"]),
            fiscal_year=int(filing["fiscal_year_focus"]),
            topic=topic,
            attribute=attribute,
        )
        try:
            anchors = (
                _semantic_anchors_v24r2(sentence, topic=topic)
                if hardened_r2
                else _anchors(sentence)
            )
        except ValueError:
            continue
        retrieval_query = (
            _retrieval_query_v24r2(subtype=subtype, topic=topic, attribute=attribute)
            if hardened_r2
            else " ".join(
                [
                    subtype.replace("_", " "),
                    topic,
                    *([attribute] if attribute else []),
                    *anchors,
                ]
            )
        )
        evidence = _bm25_select(retrieval_query, all_segments)
        source_evidence_id = str(source_item["evidence_id"])
        supporting = (
            [
                str(item["evidence_id"])
                for item in evidence
                if str(item["evidence_id"]) == source_evidence_id
                and is_contiguous_text_supported(
                    sentence, str(item.get("content") or "")
                )
            ]
            if hardened_r2
            else [
                str(item["evidence_id"])
                for item in evidence
                if is_contiguous_text_supported(
                    sentence, str(item.get("content") or "")
                )
            ]
        )
        if not supporting:
            continue
        try:
            acceptable_answer_sets = (
                _acceptable_answer_sets_v24r2(
                    subtype=subtype,
                    topic=topic,
                    evidence=evidence,
                    required_source_evidence_id=source_evidence_id,
                    required_source_sentence=sentence,
                )
                if hardened_r2
                else [
                    {
                        "set_id": "set-1",
                        "extracts": [
                            {
                                "evidence_id": supporting[0],
                                "support_window": sentence,
                                "required_semantic_anchors": anchors,
                            }
                        ],
                    }
                ]
            )
        except ValueError:
            continue
        if not acceptable_answer_sets:
            continue
        used_topics.add(topic_key)
        task_id = (
            "v24n-"
            + canonical_json_sha256(
                {
                    "version": identity_namespace,
                    "filing_id": filing["filing_id"],
                    "subtype": subtype,
                    "status": "OK",
                    "ordinal": len(accepted),
                    "question": question,
                    "retrieval_query": retrieval_query,
                    "source_evidence_id": supporting[0],
                    "support_window": sentence,
                }
            )[:24]
        )
        source_row = _task_source_row(
            filing=filing,
            task_id=task_id,
            subtype=subtype,
            question=question,
            retrieval_query=retrieval_query,
            evidence_items=evidence,
            answer=sentence,
            expected_ids=[supporting[0]],
            negative_type=None,
            hardened_r2=hardened_r2,
        )
        review = {
            "task_id": task_id,
            "subtype": subtype,
            "answerability": "ANSWERABLE",
            "scope_description": (
                f"{filing['ticker']} {filing['form_type']} fiscal {filing['fiscal_year_focus']} "
                + (
                    "independent auditor's report, opinion on the financial statements"
                    if hardened_r2 and subtype == "auditor_report_opinion_language"
                    else f"filing, {topic} disclosure"
                )
            ),
            "acceptable_answer_sets": acceptable_answer_sets,
            "refusal_scope": None,
            "item_binding": (
                {
                    "binding_version": metadata["item_binding_method"],
                    "binding_scope_sha256": metadata["item_binding_scope_sha256"],
                    "source_item": metadata.get("source_item"),
                    "source_heading": metadata.get("source_heading"),
                    "source_heading_path": metadata.get("source_heading_path"),
                    "corrected_item": metadata.get("item"),
                    "corrected_heading": metadata.get("heading"),
                }
                if hardened_r2
                and subtype == "footnote_note"
                and metadata.get("item_binding_method")
                else None
            ),
        }
        accepted.append((source_row, review))
        if len(accepted) > ordinal:
            return accepted[ordinal]
    raise ValueError(
        f"Filing {filing['ticker']} lacks supported {subtype} positive candidate {ordinal + 1}"
    )


def _build_negative_narrative(
    *,
    filing: Mapping[str, Any],
    subtype: str,
    ordinal: int,
    all_segments: Sequence[Mapping[str, Any]],
    seed: int,
    identity_namespace: str = BENCHMARK_VERSION_V24,
    hardened_r2: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    full_text = "\n".join(str(item.get("content") or "") for item in all_segments)
    normalized_full = normalize_support_text(full_text).casefold()
    catalog: Sequence[Any] = (
        _NEGATIVE_CATALOG_R2[subtype] if hardened_r2 else _NEGATIVE_CATALOG[subtype]
    )
    start = int(
        _stable_rank(seed, f"negative-{filing['filing_id']}-{subtype}", filing)[:8],
        16,
    ) % len(catalog)
    valid_position = 0
    for offset in range(len(catalog)):
        raw_spec = catalog[(start + offset) % len(catalog)]
        if hardened_r2:
            if not isinstance(raw_spec, Mapping):
                raise TypeError("r2 negative catalog entries must be objects")
            claim = str(raw_spec["claim"])
            focus = str(raw_spec["focus"])
            if _negative_spec_matches_scope_v24r2(raw_spec, normalized_full):
                continue
        else:
            claim, focus = raw_spec
        if normalize_support_text(claim).casefold() in normalized_full:
            continue
        question = _negative_question(
            subtype=subtype,
            ticker=str(filing["ticker"]),
            fiscal_year=int(filing["fiscal_year_focus"]),
            claim=claim,
        )
        retrieval_query = f"{subtype.replace('_', ' ')} {focus} {claim}"
        evidence = _bm25_select(retrieval_query, all_segments)
        if any(
            normalize_support_text(claim).casefold()
            in normalize_support_text(str(item.get("content") or "")).casefold()
            for item in evidence
        ):
            continue
        if valid_position < ordinal:
            valid_position += 1
            continue
        task_id = (
            "v24n-"
            + canonical_json_sha256(
                {
                    "version": identity_namespace,
                    "filing_id": filing["filing_id"],
                    "subtype": subtype,
                    "status": "REFUSAL",
                    "ordinal": ordinal,
                    "question": question,
                    "claim": claim,
                }
            )[:24]
        )
        source_row = _task_source_row(
            filing=filing,
            task_id=task_id,
            subtype=subtype,
            question=question,
            retrieval_query=retrieval_query,
            evidence_items=evidence,
            answer=None,
            expected_ids=[],
            negative_type=(
                "review_required_full_scope_absence"
                if hardened_r2
                else "human_verified_full_filing_absence"
            ),
            hardened_r2=hardened_r2,
        )
        scope_kind = (
            "canonical narrative scope plus checksum-bound raw CAM supplement"
            if hardened_r2 and subtype == "critical_audit_matter"
            else "canonical narrative scope"
        )
        review = {
            "task_id": task_id,
            "subtype": subtype,
            "answerability": "UNANSWERABLE",
            "scope_description": (
                f"Complete {filing['ticker']} {filing['form_type']} fiscal "
                f"{filing['fiscal_year_focus']} {scope_kind}"
            ),
            "acceptable_answer_sets": [],
            "refusal_scope": {
                "scope_description": f"All {scope_kind} chunks for the frozen filing",
                "causal_negative_type": (
                    "review_required_full_scope_absence"
                    if hardened_r2
                    else "human_verified_full_filing_absence"
                ),
                "absent_claim": claim,
                "full_filing_chunk_count": len(all_segments),
                "full_filing_text_sha256": hashlib.sha256(
                    full_text.encode()
                ).hexdigest(),
                "mechanical_absence_check": True,
            },
        }
        return source_row, review
    raise ValueError(
        f"Filing {filing['ticker']} has no mechanically absent {subtype} claim"
    )


def _secondary_sample(
    narrative_reviews: Sequence[Mapping[str, Any]], filings: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    issuer_order = [str(filing["source_entity_id"]) for filing in filings]
    by_stratum_issuer: dict[tuple[str, str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for review in narrative_reviews:
        by_stratum_issuer[
            (
                str(review["subtype"]),
                str(review["answerability"]),
                str(review["source_entity_id"]),
            )
        ].append(review)
    strata = [
        (subtype, answerability)
        for subtype in sorted(_SUBTYPE_QUOTAS)
        for answerability in ("ANSWERABLE", "UNANSWERABLE")
    ]
    selected: list[str] = []
    for stratum_index, (subtype, answerability) in enumerate(strata):
        block_start = (stratum_index * 5) % len(issuer_order)
        issuers = [
            issuer_order[(block_start + offset) % len(issuer_order)]
            for offset in range(5)
        ]
        for entity_id in issuers:
            options = sorted(
                by_stratum_issuer[(subtype, answerability, entity_id)],
                key=lambda row: str(row["task_id"]),
            )
            if not options:
                raise ValueError(
                    "Secondary-review sample cannot satisfy its stratum grid"
                )
            selected.append(str(options[0]["task_id"]))
    issuer_counts = Counter(
        str(review["source_entity_id"])
        for review in narrative_reviews
        if review["task_id"] in set(selected)
    )
    if len(selected) != 40 or set(issuer_counts.values()) != {2}:
        raise AssertionError(
            "Secondary-review sample must contain 40 cases and two per issuer"
        )
    return {
        "secondary_sample_version": SECONDARY_SAMPLE_VERSION_V24,
        "task_ids": selected,
        "task_ids_sha256": canonical_json_sha256(selected),
        "policy": {
            "per_subtype_answerability_stratum": 5,
            "per_issuer": 2,
            "selection": "canonical-five-case-stratum-blocks",
        },
    }


def _gate_membership(
    narrative_reviews: Sequence[Mapping[str, Any]], filings: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_key: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in narrative_reviews:
        by_key[
            (
                str(row["source_entity_id"]),
                str(row["subtype"]),
                str(row["answerability"]),
            )
        ].append(str(row["task_id"]))
    issuer_order = [str(filing["source_entity_id"]) for filing in filings]
    subtype_pairs = (
        ["footnote_note"] * 5
        + ["accounting_policy"] * 5
        + ["auditor_report_opinion_language"] * 5
        + ["critical_audit_matter"] * 5
    )
    selected: list[str] = []
    for entity_id, subtype in zip(issuer_order, subtype_pairs, strict=True):
        for answerability in ("ANSWERABLE", "UNANSWERABLE"):
            options = sorted(by_key[(entity_id, subtype, answerability)])
            if not options:
                raise ValueError("Narrative gate cannot satisfy issuer/subtype pair")
            selected.append(options[0])
    for index, entity_id in enumerate(issuer_order[:10]):
        answerability = "ANSWERABLE" if index < 5 else "UNANSWERABLE"
        options = [
            task_id
            for task_id in sorted(by_key[(entity_id, "footnote_note", answerability)])
            if task_id not in selected
        ]
        if not options:
            raise ValueError("Narrative gate lacks the extra balanced footnote cases")
        selected.append(options[0])
    reviews_by_id = {str(row["task_id"]): row for row in narrative_reviews}
    breakdown = Counter(
        f"{reviews_by_id[task_id]['subtype']}:{reviews_by_id[task_id]['answerability']}"
        for task_id in selected
    )
    issuer_counts = Counter(
        str(reviews_by_id[task_id]["source_entity_id"]) for task_id in selected
    )
    expected = {
        "accounting_policy:ANSWERABLE": 5,
        "accounting_policy:UNANSWERABLE": 5,
        "auditor_report_opinion_language:ANSWERABLE": 5,
        "auditor_report_opinion_language:UNANSWERABLE": 5,
        "critical_audit_matter:ANSWERABLE": 5,
        "critical_audit_matter:UNANSWERABLE": 5,
        "footnote_note:ANSWERABLE": 10,
        "footnote_note:UNANSWERABLE": 10,
    }
    if (
        len(selected) != 50
        or dict(sorted(breakdown.items())) != expected
        or min(issuer_counts.values()) < 2
    ):
        raise AssertionError(
            "Narrative gate membership violates its frozen quota contract"
        )
    return {
        "gate_membership_version": GATE_MEMBERSHIP_VERSION_V24,
        "case_count": 50,
        "task_ids": selected,
        "task_ids_sha256": canonical_json_sha256(selected),
        "breakdown": expected,
        "issuer_counts": dict(sorted(issuer_counts.items())),
    }


def _review_packet(
    narrative_reviews: Sequence[Mapping[str, Any]],
    cases_by_id: Mapping[str, Mapping[str, Any]],
    evidence_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    packet_version: str = REVIEW_PACKET_VERSION_V24,
) -> dict[str, Any]:
    records = []
    for review in narrative_reviews:
        task_id = str(review["task_id"])
        case = cases_by_id[task_id]
        record = {
            "task_id": task_id,
            "question": case["question"],
            "narrative_subtype": case["narrative_subtype"],
            "answerability": review["answerability"],
            "entity": case["entity"],
            "filing": case["filing"],
            "period": case["period"],
            "retrieval_query": case["task_parameters"]["retrieval_query"],
            "frozen_evidence": list(evidence_by_id[task_id]),
            "scope_description": review["scope_description"],
            "acceptable_answer_sets": review["acceptable_answer_sets"],
            "refusal_scope": review["refusal_scope"],
            "required_checks": [
                "question_quality",
                "subtype",
                "scope",
                "answerability",
                "evidence",
                "semantic_anchors",
                "refusal_validity",
            ],
        }
        if packet_version == REVIEW_PACKET_VERSION_V24_R2:
            record["item_binding"] = review.get("item_binding")
        records.append(record)
    return {
        "review_packet_version": packet_version,
        "case_count": len(records),
        "records": records,
    }


def _provisional_feedback_decisions(
    review: Mapping[str, Any],
    *,
    expected_role: str,
    expected_ids: set[str],
    candidate_id: str,
    packet_sha256: str,
) -> dict[str, Mapping[str, Any]]:
    if (
        review.get("review_decision_version") != REVIEW_DECISION_VERSION_V24
        or review.get("candidate_id") != candidate_id
        or review.get("rendered_packet_sha256") != packet_sha256
    ):
        raise ValueError("r1 provisional feedback is not bound to the frozen r1 packet")
    reviewer = review.get("reviewer")
    if (
        not isinstance(reviewer, Mapping)
        or reviewer.get("role") != expected_role
        or "not a human reviewer"
        not in str(reviewer.get("attestation") or "").casefold()
        or "ai" not in str(reviewer.get("identity") or "").casefold()
    ):
        raise ValueError("r1 feedback must remain explicitly non-human")
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise TypeError("r1 provisional feedback decisions must be an array")
    by_id: dict[str, Mapping[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping) or set(decision) != {
            "task_id",
            "decision",
            "checks",
            "comment",
        }:
            raise ValueError("r1 provisional feedback has an invalid decision envelope")
        task_id = str(decision.get("task_id") or "")
        if task_id in by_id:
            raise ValueError(f"Duplicate r1 provisional feedback decision: {task_id}")
        checks = decision.get("checks")
        if (
            not isinstance(checks, Mapping)
            or set(checks)
            != {
                "question_quality",
                "subtype",
                "scope",
                "answerability",
                "evidence",
                "semantic_anchors",
                "refusal_validity",
            }
            or any(type(value) is not bool for value in checks.values())
            or decision.get("decision") not in {"APPROVE", "REJECT"}
        ):
            raise ValueError(f"Invalid r1 provisional feedback checks: {task_id}")
        by_id[task_id] = decision
    if set(by_id) != expected_ids:
        raise ValueError(f"{expected_role} r1 provisional feedback membership differs")
    return by_id


def _feedback_stratum_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    entity = record.get("entity") if isinstance(record.get("entity"), Mapping) else {}
    return (
        str(entity.get("source_entity_id") or ""),
        str(record.get("narrative_subtype") or ""),
        str(record.get("answerability") or ""),
    )


def build_r1_to_r2_defect_report_v24(
    *,
    r1_candidate_dir: str | Path,
    r1_rendered_packet: str | Path,
    primary_review: str | Path,
    secondary_review: str | Path,
    validation_summary: str | Path,
    r2_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind supplied non-human r1 feedback to deterministic r2 replacements."""

    r1_root = Path(r1_candidate_dir)
    verification = verify_agent_benchmark_candidate_v24(r1_root)
    if not verification["valid"]:
        raise ValueError(
            "Invalid frozen r1 candidate: " + "; ".join(verification["errors"])
        )
    r1_manifest = _read_json(r1_root / "candidate_manifest.json")
    if r1_manifest.get("candidate_version") != CANDIDATE_VERSION_V24:
        raise ValueError("r1 feedback must target the original v2.4 candidate version")
    rendered = Path(r1_rendered_packet)
    primary_path = Path(primary_review)
    secondary_path = Path(secondary_review)
    summary_path = Path(validation_summary)
    rendered_sha256 = _sha256_path(rendered)
    r1_packet = _read_json(r1_root / "review_packet.json")
    r1_records = r1_packet.get("records")
    r2_records = r2_packet.get("records")
    if not isinstance(r1_records, list) or not isinstance(r2_records, list):
        raise TypeError("r1 and r2 review packets must contain record arrays")
    r1_by_id = {str(record["task_id"]): record for record in r1_records}
    primary = _read_json(primary_path)
    secondary = _read_json(secondary_path)
    sample = _read_json(r1_root / "review_sample_40.json")
    primary_by_id = _provisional_feedback_decisions(
        primary,
        expected_role="PRIMARY",
        expected_ids=set(r1_by_id),
        candidate_id=str(r1_manifest["candidate_id"]),
        packet_sha256=rendered_sha256,
    )
    secondary_by_id = _provisional_feedback_decisions(
        secondary,
        expected_role="SECONDARY",
        expected_ids={str(task_id) for task_id in sample["task_ids"]},
        candidate_id=str(r1_manifest["candidate_id"]),
        packet_sha256=rendered_sha256,
    )
    summary = _read_json(summary_path)
    expected_summary_fields = {
        "primary",
        "secondary",
        "sampled_disagreements",
        "primary_sha256",
        "secondary_sha256",
        "rendered_packet_sha256_verified",
    }
    if (
        set(summary) != expected_summary_fields
        or summary.get("primary_sha256") != _sha256_path(primary_path)
        or summary.get("secondary_sha256") != _sha256_path(secondary_path)
        or summary.get("rendered_packet_sha256_verified") != rendered_sha256
        or summary.get("primary") != {"cases": 200, "approve": 130, "reject": 70}
        or summary.get("secondary") != {"cases": 40, "approve": 27, "reject": 13}
        or summary.get("sampled_disagreements") != []
    ):
        raise ValueError("r1 provisional validation summary is invalid")
    if sum(item.get("decision") == "REJECT" for item in primary_by_id.values()) != 70:
        raise ValueError("r1 primary provisional feedback must contain 70 rejections")
    if sum(item.get("decision") == "REJECT" for item in secondary_by_id.values()) != 13:
        raise ValueError("r1 secondary provisional feedback must contain 13 rejections")

    def indexed(
        records: Sequence[Mapping[str, Any]],
    ) -> dict[tuple[str, str, str, int], Mapping[str, Any]]:
        grouped: defaultdict[tuple[str, str, str], list[Mapping[str, Any]]] = (
            defaultdict(list)
        )
        for record in records:
            grouped[_feedback_stratum_key(record)].append(record)
        result: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
        for key, values in grouped.items():
            for ordinal, record in enumerate(
                sorted(values, key=lambda value: str(value["task_id"]))
            ):
                result[(*key, ordinal)] = record
        return result

    r1_index = indexed(r1_records)
    r2_index = indexed(r2_records)
    mappings: list[dict[str, Any]] = []
    for key, r1_record in sorted(r1_index.items()):
        decision = primary_by_id[str(r1_record["task_id"])]
        if decision.get("decision") != "REJECT":
            continue
        replacement = r2_index.get(key)
        if replacement is None:
            raise ValueError(
                f"r2 lacks replacement stratum for r1 task {r1_record['task_id']}"
            )
        checks = decision["checks"]
        mappings.append(
            {
                "r1_task_id": str(r1_record["task_id"]),
                "r2_task_id": str(replacement["task_id"]),
                "mapping_key": {
                    "source_entity_id": key[0],
                    "narrative_subtype": key[1],
                    "answerability": key[2],
                    "ordinal": key[3],
                },
                "failed_checks": sorted(
                    name for name, passed in checks.items() if passed is False
                ),
                "corrective_comment": str(decision.get("comment") or ""),
                "replacement_changed": str(r1_record["task_id"])
                != str(replacement["task_id"]),
            }
        )
    if len(mappings) != 70 or any(not item["replacement_changed"] for item in mappings):
        raise ValueError("r1-to-r2 defect mapping must replace all 70 rejected tasks")
    material = {
        "defect_report_version": "auditops-r1-r2-defect-report.v2.4r2",
        "r1_candidate_id": r1_manifest["candidate_id"],
        "r1_candidate_manifest_sha256": _sha256_path(
            r1_root / "candidate_manifest.json"
        ),
        "r1_rendered_packet_sha256": rendered_sha256,
        "r1_primary_review_sha256": _sha256_path(primary_path),
        "r1_secondary_review_sha256": _sha256_path(secondary_path),
        "r1_validation_summary_sha256": _sha256_path(summary_path),
        "r2_review_packet_value_sha256": canonical_json_sha256(r2_packet),
        "counts": {
            "r1_primary_approvals": 130,
            "r1_primary_rejections": 70,
            "r1_secondary_approvals": 27,
            "r1_secondary_rejections": 13,
            "mapped_rejections": len(mappings),
        },
        "mappings": mappings,
        "assurance": "PROVISIONAL_AI_FEEDBACK_NOT_HUMAN_APPROVAL",
    }
    return {**material, "report_id": canonical_json_sha256(material)}


def _candidate_identity_namespace_v24(
    *, candidate_version: str, corpus_id: str, hardened_r2: bool
) -> str:
    return f"{candidate_version}:{corpus_id}" if hardened_r2 else candidate_version


def draft_agent_benchmark_v24(
    quant_jsonl: str | Path,
    corpus_db: str | Path,
    output_dir: str | Path,
    *,
    raw_filing_root: str | Path,
    corpus_id: str = "us_sec_existing20_cached20_v2.4-candidate",
    seed: int = DEFAULT_SEED_V24,
    candidate_revision: str = "r1",
    r1_candidate_dir: str | Path | None = None,
    r1_rendered_packet: str | Path | None = None,
    r1_primary_review: str | Path | None = None,
    r1_secondary_review: str | Path | None = None,
    r1_validation_summary: str | Path | None = None,
) -> dict[str, Any]:
    """Create an immutable candidate and review packet, never a benchmark."""

    if candidate_revision not in {"r1", "r2"}:
        raise ValueError("candidate_revision must be r1 or r2")
    hardened_r2 = candidate_revision == "r2"
    candidate_version = (
        CANDIDATE_VERSION_V24_R2 if hardened_r2 else CANDIDATE_VERSION_V24
    )
    # Hardened rebuilds bind membership identity to the candidate corpus name.
    # This preserves every prior r2 artifact while ensuring a defect-corrected
    # candidate receives 500 new task identities instead of masquerading as a
    # byte-compatible replay.
    identity_namespace = _candidate_identity_namespace_v24(
        candidate_version=candidate_version,
        corpus_id=corpus_id,
        hardened_r2=hardened_r2,
    )
    r1_feedback_paths = (
        r1_candidate_dir,
        r1_rendered_packet,
        r1_primary_review,
        r1_secondary_review,
        r1_validation_summary,
    )
    if hardened_r2 and any(value is None for value in r1_feedback_paths):
        raise ValueError("r2 construction requires every frozen r1 feedback input")
    if not hardened_r2 and any(value is not None for value in r1_feedback_paths):
        raise ValueError("r1 construction cannot receive r2 feedback inputs")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite candidate directory: {destination}"
        )
    quant_path = Path(quant_jsonl)
    database = Path(corpus_db)
    raw_root = Path(raw_filing_root)
    assert_no_secrets_in_value(
        {"corpus_id": corpus_id}, label="v2.4 candidate configuration"
    )
    filings, chunks_by_filing = _load_corpus(database)
    quant_rows = _select_quant_rows(
        _read_jsonl(quant_path),
        filings,
        seed,
        identity_namespace=identity_namespace,
    )

    cases: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    narrative_reviews: list[dict[str, Any]] = []
    raw_cam_sources: dict[str, dict[str, Any]] = {}

    filing_position = {
        str(filing["source_entity_id"]): index for index, filing in enumerate(filings)
    }
    for row in quant_rows:
        task, evidence, gold, observation = _package_task(
            row, "quant_metric", corpus_id=corpus_id
        )
        gold["gold_record_version"] = NARRATIVE_GOLD_VERSION_V24
        gold["schema_id"] = "agent_gold_record.v2.4"
        cases.append(task)
        evidence_rows.append({"task_id": task["task_id"], "items": evidence})
        gold_rows.append(gold)
        observations.append(observation)
        lineage.append(
            {
                "task_id": task["task_id"],
                "source_kind": "enriched_quantitative_jsonl",
                "source_task_id": row["v24_parent_task_id"],
                "source_task_sha256": canonical_json_sha256(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"task_id", "v24_parent_task_id"}
                    }
                ),
            }
        )

    for filing in filings:
        filing_chunks = chunks_by_filing[str(filing["filing_id"])]
        if hardened_r2:
            filing_chunks = _bind_item8_note_chunks_v24r2(filing_chunks)
        segments = _segment_items(filing_chunks, hardened_r2=hardened_r2)
        raw_cam_segments, raw_cam_source = _raw_cam_segments_v24(raw_root, filing)
        raw_cam_sources[str(filing["filing_id"])] = raw_cam_source
        filing_positive_support: set[str] = set()
        subtype_order = (
            _HARDENED_R2_SUBTYPE_ORDER if hardened_r2 else tuple(_SUBTYPE_QUOTAS)
        )
        for subtype in subtype_order:
            answer_quota, refusal_quota = _SUBTYPE_QUOTAS[subtype]
            subtype_segments = _narrative_scope_items_v24r2(
                subtype=subtype,
                canonical_segments=segments,
                raw_cam_segments=raw_cam_segments,
                hardened_r2=hardened_r2,
            )
            subtype_positive_support: list[str] = []
            for ordinal in range(answer_quota):
                source_row, review = _build_positive_narrative(
                    filing=filing,
                    subtype=subtype,
                    ordinal=ordinal,
                    all_segments=subtype_segments,
                    seed=seed,
                    excluded_support_windows=frozenset(filing_positive_support),
                    identity_namespace=identity_namespace,
                    hardened_r2=hardened_r2,
                )
                support_window = review["acceptable_answer_sets"][0]["extracts"][0][
                    "support_window"
                ]
                subtype_positive_support.append(
                    normalize_support_text(str(support_window)).casefold()
                )
                subtype_positive_support.append(
                    normalize_support_text(
                        str(source_row.get("extractive_answer") or "")
                    ).casefold()
                )
                task, evidence, gold, observation = _package_task(
                    source_row, "narrative_citation", corpus_id=corpus_id
                )
                gold["gold_record_version"] = NARRATIVE_GOLD_VERSION_V24
                gold["schema_id"] = "agent_gold_record.v2.4"
                gold["acceptable_answer_sets"] = review["acceptable_answer_sets"]
                gold["refusal_scope"] = None
                validate_narrative_gold_v24(gold, task)
                cases.append(task)
                evidence_rows.append({"task_id": task["task_id"], "items": evidence})
                gold_rows.append(gold)
                observations.append(observation)
                lineage.append(
                    {
                        "task_id": task["task_id"],
                        "source_kind": "canonical_narrative_chunk",
                        "source_task_sha256": canonical_json_sha256(source_row),
                    }
                )
                narrative_reviews.append(
                    {
                        **review,
                        "source_entity_id": filing["source_entity_id"],
                        "filing_id": filing["filing_id"],
                    }
                )
            filing_positive_support.update(subtype_positive_support)
            for ordinal in range(refusal_quota):
                source_row, review = _build_negative_narrative(
                    filing=filing,
                    subtype=subtype,
                    ordinal=ordinal,
                    all_segments=subtype_segments,
                    seed=seed,
                    identity_namespace=identity_namespace,
                    hardened_r2=hardened_r2,
                )
                task, evidence, gold, observation = _package_task(
                    source_row, "narrative_citation", corpus_id=corpus_id
                )
                gold["gold_record_version"] = NARRATIVE_GOLD_VERSION_V24
                gold["schema_id"] = "agent_gold_record.v2.4"
                gold["acceptable_answer_sets"] = []
                gold["refusal_scope"] = review["refusal_scope"]
                gold["evaluator_negative_cause"] = (
                    "provisional_ai_reviewed_full_scope_absence"
                    if hardened_r2
                    else "human_verified_full_filing_absence"
                )
                validate_narrative_gold_v24(gold, task)
                cases.append(task)
                evidence_rows.append({"task_id": task["task_id"], "items": evidence})
                gold_rows.append(gold)
                observations.append(observation)
                lineage.append(
                    {
                        "task_id": task["task_id"],
                        "source_kind": (
                            "synonym_screened_full_scope_absent_claim"
                            if hardened_r2
                            else "mechanically_absent_filing_claim"
                        ),
                        "source_task_sha256": canonical_json_sha256(source_row),
                    }
                )
                narrative_reviews.append(
                    {
                        **review,
                        "source_entity_id": filing["source_entity_id"],
                        "filing_id": filing["filing_id"],
                    }
                )

    order = sorted(
        range(len(cases)),
        key=lambda index: (
            filing_position[str(cases[index]["entity"]["source_entity_id"])],
            0 if cases[index]["task_type"] == "quant_metric" else 1,
            str(cases[index].get("narrative_subtype") or ""),
            str(gold_rows[index]["target"]["status"]),
            str(cases[index]["task_id"]),
        ),
    )
    cases = [cases[index] for index in order]
    evidence_rows = [evidence_rows[index] for index in order]
    gold_rows = [gold_rows[index] for index in order]
    observations = [observations[index] for index in order]
    lineage = [lineage[index] for index in order]
    if len(cases) != QUANT_COUNT_V24 + NARRATIVE_COUNT_V24:
        raise AssertionError("v2.4 candidate must contain exactly 500 tasks")

    cases_by_id = {str(row["task_id"]): row for row in cases}
    evidence_by_id = {str(row["task_id"]): row["items"] for row in evidence_rows}
    packet = _review_packet(
        narrative_reviews,
        cases_by_id,
        evidence_by_id,
        packet_version=(
            REVIEW_PACKET_VERSION_V24_R2 if hardened_r2 else REVIEW_PACKET_VERSION_V24
        ),
    )
    secondary = _secondary_sample(narrative_reviews, filings)
    gate = _gate_membership(narrative_reviews, filings)
    feedback_report = None
    if hardened_r2:
        feedback_report = build_r1_to_r2_defect_report_v24(
            r1_candidate_dir=str(r1_candidate_dir),
            r1_rendered_packet=str(r1_rendered_packet),
            primary_review=str(r1_primary_review),
            secondary_review=str(r1_secondary_review),
            validation_summary=str(r1_validation_summary),
            r2_packet=packet,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        counts = {
            "candidate_cases.jsonl": _write_jsonl(
                temporary / "candidate_cases.jsonl", cases
            ),
            "candidate_evidence.jsonl": _write_jsonl(
                temporary / "candidate_evidence.jsonl", evidence_rows
            ),
            "candidate_gold.jsonl": _write_jsonl(
                temporary / "candidate_gold.jsonl", gold_rows
            ),
            "candidate_verifier_observations.jsonl": _write_jsonl(
                temporary / "candidate_verifier_observations.jsonl", observations
            ),
            "candidate_case_lineage.jsonl": _write_jsonl(
                temporary / "candidate_case_lineage.jsonl", lineage
            ),
        }
        _write_json(temporary / "review_packet.json", packet)
        _write_json(temporary / "review_sample_40.json", secondary)
        _write_json(temporary / "gate_50.json", gate)
        if feedback_report is not None:
            _write_json(temporary / "r1_feedback_report.json", feedback_report)
        source_manifest = {
            "source_manifest_version": (
                "v2.4-candidate-r2" if hardened_r2 else "v2.4-candidate"
            ),
            "corpus_id": corpus_id,
            "sources": {
                "quant_jsonl": _artifact(quant_path, len(_read_jsonl(quant_path))),
                "corpus_sqlite": _artifact(database),
                "filings": [
                    {
                        "filing_id": filing["filing_id"],
                        "ticker": filing["ticker"],
                        "source_entity_id": filing["source_entity_id"],
                        "source_sha256": filing["source_sha256"],
                        "source_size_bytes": filing["source_size_bytes"],
                        "raw_cam_source": raw_cam_sources[str(filing["filing_id"])],
                    }
                    for filing in filings
                ],
            },
            "retrieval": {
                "policy_version": BM25_POLICY_VERSION_V24,
                "top_k": 5,
                "gold_used_for_ranking": False,
                "answer_support_checked_after_ranking": True,
            },
        }
        _write_json(temporary / "source_manifest.json", source_manifest)
        artifact_names = [
            *counts,
            "review_packet.json",
            "review_sample_40.json",
            "gate_50.json",
            "source_manifest.json",
            *(["r1_feedback_report.json"] if hardened_r2 else []),
        ]
        artifacts = {
            name: _artifact(temporary / name, counts.get(name))
            for name in artifact_names
        }
        candidate_material = {
            "candidate_version": candidate_version,
            "benchmark_version": BENCHMARK_VERSION_V24,
            "runtime_contract_version": "v2.3",
            "seed": seed,
            "corpus_id": corpus_id,
            "counts": {
                "quant": 300,
                "narrative": 200,
                "narrative_answerable": 100,
                "narrative_unanswerable": 100,
                "issuers": 20,
                "total": 500,
                "secondary_review": 40,
                "narrative_gate": 50,
            },
            "issuer_order": [str(filing["source_entity_id"]) for filing in filings],
            "review_status": "PENDING_EXTERNAL_HUMAN_APPROVAL",
            "artifacts": artifacts,
        }
        candidate = {
            **candidate_material,
            "candidate_id": canonical_json_sha256(candidate_material),
        }
        _write_json(temporary / "candidate_manifest.json", candidate)
        verification = verify_agent_benchmark_candidate_v24(temporary)
        if not verification["valid"]:
            raise ValueError(
                "Invalid v2.4 candidate: " + "; ".join(verification["errors"])
            )
        assert_no_secrets([temporary])
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "candidate_id": candidate["candidate_id"],
        "output_dir": str(destination),
        "review_status": candidate["review_status"],
        "verified": True,
    }


def draft_agent_benchmark_v24r2(
    quant_jsonl: str | Path,
    corpus_db: str | Path,
    output_dir: str | Path,
    *,
    raw_filing_root: str | Path,
    r1_candidate_dir: str | Path,
    r1_rendered_packet: str | Path,
    r1_primary_review: str | Path,
    r1_secondary_review: str | Path,
    r1_validation_summary: str | Path,
    corpus_id: str = "us_sec_existing20_cached20_v2.4r2-candidate",
    seed: int = DEFAULT_SEED_V24,
) -> dict[str, Any]:
    return draft_agent_benchmark_v24(
        quant_jsonl,
        corpus_db,
        output_dir,
        raw_filing_root=raw_filing_root,
        corpus_id=corpus_id,
        seed=seed,
        candidate_revision="r2",
        r1_candidate_dir=r1_candidate_dir,
        r1_rendered_packet=r1_rendered_packet,
        r1_primary_review=r1_primary_review,
        r1_secondary_review=r1_secondary_review,
        r1_validation_summary=r1_validation_summary,
    )


def _verify_artifacts(root: Path, artifacts: Any, *, errors: list[str]) -> None:
    if not isinstance(artifacts, Mapping):
        errors.append("artifact manifest must be an object")
        return
    for name, raw_expected in artifacts.items():
        if not isinstance(name, str) or Path(name).name != name:
            errors.append("artifact name is unsafe")
            continue
        expected = raw_expected if isinstance(raw_expected, Mapping) else {}
        path = root / name
        if not path.is_file():
            errors.append(f"missing artifact: {name}")
            continue
        if _sha256_path(path) != expected.get("sha256"):
            errors.append(f"sha256 mismatch: {name}")
        if path.stat().st_size != expected.get("bytes"):
            errors.append(f"byte count mismatch: {name}")
        if expected.get("records") is not None:
            try:
                count = len(_read_jsonl(path))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"invalid JSONL {name}: {exc}")
            else:
                if count != expected["records"]:
                    errors.append(f"record count mismatch: {name}")


def _source_manifest_errors(source_manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(source_manifest) != {
        "source_manifest_version",
        "corpus_id",
        "sources",
        "retrieval",
    }:
        errors.append("source manifest fields are missing or unexpected")
    if source_manifest.get("source_manifest_version") not in {
        "v2.4-candidate",
        "v2.4-candidate-r2",
    }:
        errors.append("source manifest version is invalid")
    sources = source_manifest.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "quant_jsonl",
        "corpus_sqlite",
        "filings",
    }:
        return [*errors, "source manifest sources are missing or unexpected"]
    sha256_pattern = re.compile(r"[0-9a-f]{64}")
    for label, expected_fields in (
        ("quant_jsonl", {"sha256", "bytes", "records"}),
        ("corpus_sqlite", {"sha256", "bytes"}),
    ):
        descriptor = sources.get(label)
        if not isinstance(descriptor, Mapping) or set(descriptor) != expected_fields:
            errors.append(f"source manifest {label} descriptor is invalid")
            continue
        if sha256_pattern.fullmatch(str(descriptor.get("sha256") or "")) is None:
            errors.append(f"source manifest {label} sha256 is invalid")
        if (
            isinstance(descriptor.get("bytes"), bool)
            or not isinstance(descriptor.get("bytes"), int)
            or descriptor["bytes"] <= 0
        ):
            errors.append(f"source manifest {label} byte count is invalid")
        if "records" in descriptor and (
            isinstance(descriptor.get("records"), bool)
            or not isinstance(descriptor.get("records"), int)
            or descriptor["records"] <= 0
        ):
            errors.append(f"source manifest {label} record count is invalid")
    filings = sources.get("filings")
    filing_fields = {
        "filing_id",
        "ticker",
        "source_entity_id",
        "source_sha256",
        "source_size_bytes",
        "raw_cam_source",
    }
    raw_fields = {
        "archive_name",
        "archive_sha256",
        "archive_size_bytes",
        "html_member",
        "html_sha256",
        "html_size_bytes",
        "supplement_version",
        "section_text_sha256",
        "sentence_count",
    }
    if not isinstance(filings, list) or len(filings) != ISSUER_COUNT_V24:
        errors.append("source manifest must contain exactly 20 filing descriptors")
        filings = []
    seen_filing_ids: set[str] = set()
    seen_entities: set[str] = set()
    for index, filing in enumerate(filings):
        if not isinstance(filing, Mapping) or set(filing) != filing_fields:
            errors.append(f"source filing {index} descriptor is invalid")
            continue
        filing_id = str(filing.get("filing_id") or "")
        entity_id = str(filing.get("source_entity_id") or "")
        ticker = str(filing.get("ticker") or "")
        if not filing_id or filing_id in seen_filing_ids:
            errors.append(f"source filing {index} filing_id is empty or duplicated")
        if not entity_id or entity_id in seen_entities:
            errors.append(f"source filing {index} entity is empty or duplicated")
        if not ticker:
            errors.append(f"source filing {index} ticker is empty")
        seen_filing_ids.add(filing_id)
        seen_entities.add(entity_id)
        source_sha256 = str(filing.get("source_sha256") or "")
        source_size = filing.get("source_size_bytes")
        if sha256_pattern.fullmatch(source_sha256) is None:
            errors.append(f"source filing {index} source sha256 is invalid")
        if (
            isinstance(source_size, bool)
            or not isinstance(source_size, int)
            or source_size <= 0
        ):
            errors.append(f"source filing {index} source byte count is invalid")
        raw = filing.get("raw_cam_source")
        if not isinstance(raw, Mapping) or set(raw) != raw_fields:
            errors.append(f"source filing {index} raw CAM provenance is invalid")
            continue
        archive_name = str(raw.get("archive_name") or "")
        html_member = str(raw.get("html_member") or "")
        if Path(
            archive_name
        ).name != archive_name or not archive_name.casefold().endswith(".zip"):
            errors.append(f"source filing {index} raw archive name is unsafe")
        if Path(html_member).name != html_member or not html_member.casefold().endswith(
            (".htm", ".html")
        ):
            errors.append(f"source filing {index} raw HTML member is unsafe")
        for key in ("archive_sha256", "html_sha256", "section_text_sha256"):
            if sha256_pattern.fullmatch(str(raw.get(key) or "")) is None:
                errors.append(f"source filing {index} {key} is invalid")
        for key in ("archive_size_bytes", "html_size_bytes", "sentence_count"):
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(f"source filing {index} {key} is invalid")
        if raw.get("supplement_version") != RAW_CAM_SUPPLEMENT_VERSION_V24:
            errors.append(f"source filing {index} CAM supplement version is invalid")
        if (
            raw.get("archive_sha256") != source_sha256
            or raw.get("archive_size_bytes") != source_size
        ):
            errors.append(
                f"source filing {index} raw archive does not bind the corpus source"
            )
    retrieval = source_manifest.get("retrieval")
    expected_retrieval = {
        "policy_version": BM25_POLICY_VERSION_V24,
        "top_k": 5,
        "gold_used_for_ranking": False,
        "answer_support_checked_after_ranking": True,
    }
    if retrieval != expected_retrieval:
        errors.append("source manifest retrieval policy is invalid")
    return errors


def _narrative_quota_errors(
    cases: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]]
) -> list[str]:
    errors: list[str] = []
    gold_by_id = {str(row.get("task_id")): row for row in gold}
    by_issuer: defaultdict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    questions_by_issuer_subtype: defaultdict[tuple[str, str], list[str]] = defaultdict(
        list
    )
    positive_support_by_issuer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        if case.get("task_type") != "narrative_citation":
            continue
        task_id = str(case.get("task_id"))
        target = gold_by_id.get(task_id, {}).get("target")
        status = target.get("status") if isinstance(target, Mapping) else None
        answerability = "ANSWERABLE" if status == "OK" else "UNANSWERABLE"
        entity = case.get("entity") if isinstance(case.get("entity"), Mapping) else {}
        entity_id = str(entity.get("source_entity_id"))
        by_issuer[entity_id][(str(case.get("narrative_subtype")), answerability)] += 1
        questions_by_issuer_subtype[
            (entity_id, str(case.get("narrative_subtype")))
        ].append(normalize_support_text(str(case.get("question") or "")).casefold())
        if answerability == "ANSWERABLE":
            answer_sets = gold_by_id.get(task_id, {}).get("acceptable_answer_sets")
            if isinstance(answer_sets, list) and answer_sets:
                extracts = (
                    answer_sets[0].get("extracts")
                    if isinstance(answer_sets[0], Mapping)
                    else None
                )
                if isinstance(extracts, list):
                    signature = canonical_json_sha256(
                        [
                            normalize_support_text(
                                str(extract.get("support_window") or "")
                            ).casefold()
                            for extract in extracts
                            if isinstance(extract, Mapping)
                        ]
                    )
                    positive_support_by_issuer[entity_id][signature] += 1
    if len(by_issuer) != ISSUER_COUNT_V24:
        errors.append("narrative membership must contain exactly 20 issuers")
    for entity_id, counts in sorted(by_issuer.items()):
        if sum(counts.values()) != 10:
            errors.append(f"issuer {entity_id} does not have ten narrative cases")
        for subtype, (answers, refusals) in _SUBTYPE_QUOTAS.items():
            if counts[(subtype, "ANSWERABLE")] != answers:
                errors.append(f"issuer {entity_id} has wrong {subtype} answer quota")
            if counts[(subtype, "UNANSWERABLE")] != refusals:
                errors.append(f"issuer {entity_id} has wrong {subtype} refusal quota")
    for entity_id, support_counts in positive_support_by_issuer.items():
        if any(count > 1 for count in support_counts.values()):
            errors.append(f"issuer {entity_id} reuses positive support across subtypes")
    for (entity_id, subtype), questions in sorted(questions_by_issuer_subtype.items()):
        if len(questions) != len(set(questions)):
            errors.append(f"issuer {entity_id} repeats a {subtype} question")
    return errors


def verify_agent_benchmark_candidate_v24(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    errors: list[str] = []
    try:
        manifest = _read_json(root / "candidate_manifest.json")
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        return {
            "valid": False,
            "errors": [f"candidate manifest could not be loaded: {exc}"],
        }
    expected_fields = {
        "candidate_version",
        "benchmark_version",
        "runtime_contract_version",
        "seed",
        "corpus_id",
        "counts",
        "issuer_order",
        "review_status",
        "artifacts",
        "candidate_id",
    }
    if set(manifest) != expected_fields:
        errors.append("candidate manifest fields are missing or unexpected")
    candidate_version = manifest.get("candidate_version")
    if candidate_version not in {CANDIDATE_VERSION_V24, CANDIDATE_VERSION_V24_R2}:
        errors.append("unsupported candidate_version")
    hardened_r2 = candidate_version == CANDIDATE_VERSION_V24_R2
    if manifest.get("benchmark_version") != BENCHMARK_VERSION_V24:
        errors.append("unsupported benchmark_version")
    if manifest.get("runtime_contract_version") != "v2.3":
        errors.append("v2.4 must bind the v2.3 runtime contract")
    material = {key: value for key, value in manifest.items() if key != "candidate_id"}
    if manifest.get("candidate_id") != canonical_json_sha256(material):
        errors.append("candidate_id does not bind the complete manifest")
    expected_names = {
        "candidate_cases.jsonl",
        "candidate_evidence.jsonl",
        "candidate_gold.jsonl",
        "candidate_verifier_observations.jsonl",
        "candidate_case_lineage.jsonl",
        "review_packet.json",
        "review_sample_40.json",
        "gate_50.json",
        "source_manifest.json",
    }
    if hardened_r2:
        expected_names.add("r1_feedback_report.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_names:
        errors.append("candidate artifact set is missing or unexpected")
    _verify_artifacts(root, artifacts, errors=errors)
    try:
        cases = _read_jsonl(root / "candidate_cases.jsonl")
        evidence = _read_jsonl(root / "candidate_evidence.jsonl")
        gold = _read_jsonl(root / "candidate_gold.jsonl")
        observations = _read_jsonl(root / "candidate_verifier_observations.jsonl")
        lineage = _read_jsonl(root / "candidate_case_lineage.jsonl")
        packet = _read_json(root / "review_packet.json")
        sample = _read_json(root / "review_sample_40.json")
        gate = _read_json(root / "gate_50.json")
        source_manifest = _read_json(root / "source_manifest.json")
        feedback_report = (
            _read_json(root / "r1_feedback_report.json") if hardened_r2 else None
        )
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        errors.append(f"candidate records could not be loaded: {exc}")
        (
            cases,
            evidence,
            gold,
            observations,
            lineage,
            packet,
            sample,
            gate,
            source_manifest,
            feedback_report,
        ) = [], [], [], [], [], {}, {}, {}, {}, None
    errors.extend(_source_manifest_errors(source_manifest))
    if (
        hardened_r2
        and source_manifest.get("source_manifest_version") != "v2.4-candidate-r2"
    ):
        errors.append("r2 candidate must use the r2 source manifest version")
    if (
        not hardened_r2
        and source_manifest.get("source_manifest_version") != "v2.4-candidate"
    ):
        errors.append("r1 candidate must use the original source manifest version")
    source_filings = (
        source_manifest.get("sources", {}).get("filings", [])
        if isinstance(source_manifest.get("sources"), Mapping)
        else []
    )
    if isinstance(source_filings, list) and manifest.get("issuer_order") != [
        str(filing.get("source_entity_id"))
        for filing in source_filings
        if isinstance(filing, Mapping)
    ]:
        errors.append("candidate issuer order differs from source filing order")
    ids = [str(row.get("task_id")) for row in cases]
    for label, rows in (
        ("evidence", evidence),
        ("gold", gold),
        ("observations", observations),
        ("lineage", lineage),
    ):
        if [str(row.get("task_id")) for row in rows] != ids:
            errors.append(f"candidate {label} membership/order differs from cases")
    if len(ids) != 500 or len(set(ids)) != 500:
        errors.append("candidate must contain exactly 500 unique task IDs")
    if sum(row.get("task_type") == "quant_metric" for row in cases) != 300:
        errors.append("candidate must contain 300 quantitative tasks")
    if sum(row.get("task_type") == "narrative_citation" for row in cases) != 200:
        errors.append("candidate must contain 200 narrative tasks")
    evidence_by_id = {
        str(row.get("task_id")): row.get("items")
        for row in evidence
        if isinstance(row.get("items"), list)
    }
    gold_by_id = {str(row.get("task_id")): row for row in gold}
    for case in cases:
        try:
            validate_agent_task_input(case)
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid task {case.get('task_id')}: {exc}")
            continue
        if case.get("agent_task_input_version") != "v2.3":
            errors.append(
                f"task {case.get('task_id')} does not preserve v2.3 runtime contract"
            )
        if case.get("task_type") == "narrative_citation":
            try:
                validate_question_scope_v24(case)
                validate_narrative_gold_v24(gold_by_id[str(case["task_id"])], case)
            except (TypeError, ValueError) as exc:
                errors.append(f"invalid narrative case {case.get('task_id')}: {exc}")
            items = evidence_by_id.get(str(case["task_id"]), [])
            target = gold_by_id.get(str(case["task_id"]), {}).get("target", {})
            if hardened_r2:
                frozen_ids = [
                    str(item.get("evidence_id") or "")
                    for item in items
                    if isinstance(item, Mapping)
                ]
                if len(frozen_ids) != len(set(frozen_ids)):
                    errors.append(
                        f"r2 frozen evidence IDs are not unique: {case.get('task_id')}"
                    )
            if isinstance(target, Mapping) and target.get("status") == "OK":
                answer = str(target.get("answer_text") or "")
                if not any(
                    is_contiguous_text_supported(answer, str(item.get("content") or ""))
                    for item in items
                    if isinstance(item, Mapping)
                ):
                    errors.append(
                        f"answer support missing after frozen retrieval: {case.get('task_id')}"
                    )
                if hardened_r2:
                    target_ids = target.get("chunk_evidence_ids")
                    if not isinstance(target_ids, list) or len(target_ids) != 1:
                        errors.append(
                            f"r2 positive must bind one source evidence ID: {case.get('task_id')}"
                        )
                        target_ids = []
                    answer_sets = gold_by_id.get(str(case["task_id"]), {}).get(
                        "acceptable_answer_sets"
                    )
                    evidence_by_eid = {
                        str(item.get("evidence_id")): item
                        for item in items
                        if isinstance(item, Mapping)
                    }
                    for answer_set in (
                        answer_sets if isinstance(answer_sets, list) else []
                    ):
                        extracts = (
                            answer_set.get("extracts")
                            if isinstance(answer_set, Mapping)
                            else []
                        )
                        validated_extracts = (
                            extracts if isinstance(extracts, list) else []
                        )
                        for extract in validated_extracts:
                            if not isinstance(extract, Mapping):
                                continue
                            evidence_id = str(extract.get("evidence_id") or "")
                            if evidence_id not in target_ids:
                                errors.append(
                                    f"r2 acceptable extract is not source-bound: {case.get('task_id')}"
                                )
                            if evidence_id not in evidence_by_eid:
                                errors.append(
                                    f"r2 positive source evidence is not frozen: {case.get('task_id')}"
                                )
                                continue
                            support_window = str(extract.get("support_window") or "")
                            if not is_contiguous_text_supported(
                                support_window,
                                str(evidence_by_eid[evidence_id].get("content") or ""),
                            ):
                                errors.append(
                                    f"r2 support window is not exact frozen evidence: {case.get('task_id')}"
                                )
                            if case.get("narrative_subtype") == "footnote_note":
                                metadata = evidence_by_eid[evidence_id].get("metadata")
                                item_number = (
                                    str(metadata.get("item") or "")
                                    if isinstance(metadata, Mapping)
                                    else ""
                                )
                                if item_number not in {"8", "8A"}:
                                    errors.append(
                                        f"r2 footnote evidence is outside Item 8/8A: {case.get('task_id')}"
                                    )
                            if (
                                case.get("narrative_subtype")
                                == "auditor_report_opinion_language"
                            ):
                                metadata = evidence_by_eid[evidence_id].get("metadata")
                                if (
                                    not isinstance(metadata, Mapping)
                                    or metadata.get("section_label")
                                    != "AUDITOR_OPINION_FINANCIAL_STATEMENTS"
                                ):
                                    errors.append(
                                        f"r2 auditor opinion scope label is not fixed: {case.get('task_id')}"
                                    )
                            for anchor in (
                                extract.get("required_semantic_anchors") or []
                            ):
                                try:
                                    validate_semantic_anchor_v24r2(
                                        str(anchor),
                                        support_window=support_window,
                                        topic=_anchor_topic_from_question_v24r2(
                                            str(case.get("question") or "")
                                        ),
                                    )
                                except ValueError as exc:
                                    errors.append(
                                        f"invalid r2 anchor {case.get('task_id')}: {exc}"
                                    )
    errors.extend(_narrative_quota_errors(cases, gold))
    review_records = packet.get("records") if isinstance(packet, Mapping) else None
    if (
        packet.get("review_packet_version")
        != (REVIEW_PACKET_VERSION_V24_R2 if hardened_r2 else REVIEW_PACKET_VERSION_V24)
        or not isinstance(review_records, list)
        or len(review_records) != 200
    ):
        errors.append("review packet must contain all 200 narrative cases")
    narrative_ids = {
        str(row["task_id"])
        for row in cases
        if row.get("task_type") == "narrative_citation"
    }
    if (
        isinstance(review_records, list)
        and {str(row.get("task_id")) for row in review_records} != narrative_ids
    ):
        errors.append("review packet narrative membership is incomplete")
    if hardened_r2 and isinstance(review_records, list):
        for record in review_records:
            subtype = record.get("narrative_subtype")
            answerability = record.get("answerability")
            scope_description = str(record.get("scope_description") or "").casefold()
            refusal_scope = record.get("refusal_scope")
            item_binding = record.get("item_binding")
            binding_version = (
                item_binding.get("binding_version")
                if isinstance(item_binding, Mapping)
                else None
            )
            source_item = (
                str(item_binding.get("source_item") or "")
                if isinstance(item_binding, Mapping)
                else ""
            )
            if item_binding is not None and (
                subtype != "footnote_note"
                or answerability != "ANSWERABLE"
                or not isinstance(item_binding, Mapping)
                or set(item_binding)
                != {
                    "binding_version",
                    "binding_scope_sha256",
                    "source_item",
                    "source_heading",
                    "source_heading_path",
                    "corrected_item",
                    "corrected_heading",
                }
                or binding_version
                not in {
                    "auditops.item8_note_binding.v2.4r2",
                    "auditops.item8_topic_series_binding.v2.4r2",
                    "auditops.missing_item_note_binding.v2.4r2",
                    "auditops.missing_item_topic_series_binding.v2.4r2",
                }
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(item_binding.get("binding_scope_sha256") or ""),
                )
                or (
                    binding_version == "auditops.item8_note_binding.v2.4r2"
                    and source_item != "15"
                )
                or (
                    binding_version == "auditops.missing_item_note_binding.v2.4r2"
                    and source_item
                )
                or (
                    binding_version
                    == "auditops.missing_item_topic_series_binding.v2.4r2"
                    and source_item
                )
                or (
                    binding_version == "auditops.item8_topic_series_binding.v2.4r2"
                    and source_item not in {"15", "16"}
                )
                or str(item_binding.get("corrected_item") or "") != "8"
                or "financial statements"
                not in str(item_binding.get("corrected_heading") or "").casefold()
            ):
                errors.append(
                    f"r2 Item 8 note binding is invalid: {record.get('task_id')}"
                )
            if (
                subtype == "auditor_report_opinion_language"
                and answerability == "ANSWERABLE"
                and "independent auditor's report, opinion on the financial statements"
                not in scope_description
            ):
                errors.append(f"r2 opinion scope is invalid: {record.get('task_id')}")
            if subtype == "critical_audit_matter" and answerability == "UNANSWERABLE":
                nested = (
                    str(refusal_scope.get("scope_description") or "").casefold()
                    if isinstance(refusal_scope, Mapping)
                    else ""
                )
                if (
                    "raw cam supplement" not in scope_description
                    or "raw cam supplement" not in nested
                ):
                    errors.append(
                        f"r2 CAM refusal scope is invalid: {record.get('task_id')}"
                    )
    if hardened_r2:
        if (
            not isinstance(feedback_report, Mapping)
            or feedback_report.get("defect_report_version")
            != "auditops-r1-r2-defect-report.v2.4r2"
            or feedback_report.get("assurance")
            != "PROVISIONAL_AI_FEEDBACK_NOT_HUMAN_APPROVAL"
            or feedback_report.get("counts", {}).get("mapped_rejections") != 70
            or not isinstance(feedback_report.get("mappings"), list)
            or len(feedback_report["mappings"]) != 70
        ):
            errors.append("r2 feedback report is invalid")
        else:
            report_material = {
                key: value
                for key, value in feedback_report.items()
                if key != "report_id"
            }
            if feedback_report.get("report_id") != canonical_json_sha256(
                report_material
            ):
                errors.append("r2 feedback report ID is invalid")
    sample_ids = sample.get("task_ids") if isinstance(sample, Mapping) else None
    if (
        sample.get("secondary_sample_version") != SECONDARY_SAMPLE_VERSION_V24
        or not isinstance(sample_ids, list)
        or len(sample_ids) != 40
        or len(set(sample_ids)) != 40
        or not set(sample_ids).issubset(narrative_ids)
        or sample.get("task_ids_sha256") != canonical_json_sha256(sample_ids or [])
    ):
        errors.append("secondary review sample is invalid")
    gate_ids = gate.get("task_ids") if isinstance(gate, Mapping) else None
    if (
        gate.get("gate_membership_version") != GATE_MEMBERSHIP_VERSION_V24
        or not isinstance(gate_ids, list)
        or len(gate_ids) != 50
        or len(set(gate_ids)) != 50
        or not set(gate_ids).issubset(narrative_ids)
        or gate.get("task_ids_sha256") != canonical_json_sha256(gate_ids or [])
    ):
        errors.append("narrative gate membership is invalid")
    try:
        assert_no_secrets([root])
    except ValueError as exc:
        errors.append(str(exc))
    return {
        "valid": not errors,
        "errors": errors,
        "candidate_id": manifest.get("candidate_id"),
    }


def render_agent_benchmark_review_v24(
    candidate_dir: str | Path, output_html: str | Path
) -> dict[str, Any]:
    """Render the bound JSON packet as a self-contained, no-script HTML file."""

    root = Path(candidate_dir)
    verification = verify_agent_benchmark_candidate_v24(root)
    if not verification["valid"]:
        raise ValueError("Invalid v2.4 candidate: " + "; ".join(verification["errors"]))
    destination = Path(output_html)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite review packet: {destination}")
    manifest = _read_json(root / "candidate_manifest.json")
    packet = _read_json(root / "review_packet.json")
    sections: list[str] = []
    for index, record in enumerate(packet["records"], start=1):
        evidence_html = "".join(
            "<details><summary>"
            + html.escape(f"Rank {item.get('rank')} — {item.get('evidence_id')}")
            + "</summary><pre>"
            + html.escape(str(item.get("content") or ""))
            + "</pre></details>"
            for item in record["frozen_evidence"]
        )
        answer_material = (
            json.dumps(record["acceptable_answer_sets"], ensure_ascii=False, indent=2)
            if record["answerability"] == "ANSWERABLE"
            else json.dumps(record["refusal_scope"], ensure_ascii=False, indent=2)
        )
        sections.append(
            f"<section id='{html.escape(record['task_id'])}'>"
            f"<h2>{index}. {html.escape(record['entity']['name'])} — "
            f"{html.escape(record['narrative_subtype'])} — {html.escape(record['answerability'])}</h2>"
            f"<p><strong>Task:</strong> {html.escape(record['task_id'])}</p>"
            f"<p><strong>Question:</strong> {html.escape(record['question'])}</p>"
            f"<p><strong>Scope:</strong> {html.escape(record['scope_description'])}</p>"
            f"<p><strong>Retrieval query:</strong> {html.escape(record['retrieval_query'])}</p>"
            f"<h3>Frozen top-five evidence</h3>{evidence_html}"
            f"<h3>Evaluator-only review material</h3><pre>{html.escape(answer_material)}</pre>"
            "<p class='checks'>Review: question quality □ subtype □ scope □ answerability □ "
            "evidence □ anchors □ refusal validity □</p></section>"
        )
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>AuditOps v2.4 narrative review packet</title><style>
body{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17202a}
header,section{border:1px solid #d5d8dc;border-radius:8px;padding:1rem 1.25rem;margin:1rem 0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:.8rem;border-radius:6px}
summary{cursor:pointer;font-weight:600}.checks{background:#fff8dc;padding:.75rem}code{overflow-wrap:anywhere}
</style></head><body>""" + (
        "<header><h1>AuditOps v2.4 cached-20 narrative review</h1>"
        f"<p>Candidate <code>{html.escape(manifest['candidate_id'])}</code></p>"
        "<p>This packet is review material, not an approved benchmark. It contains evaluator-only gold and must never be supplied to a model.</p></header>"
        + "".join(sections)
        + "</body></html>"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(document, encoding="utf-8", newline="\n")
        assert_no_secrets([temporary])
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "candidate_id": manifest["candidate_id"],
        "output": str(destination),
        "sha256": _sha256_path(destination),
        "case_count": len(packet["records"]),
    }


def _validate_human_review(
    review: Mapping[str, Any],
    *,
    candidate_id: str,
    packet_sha256: str,
    expected_role: str,
    expected_ids: set[str],
) -> dict[str, str]:
    if review.get("review_decision_version") != REVIEW_DECISION_VERSION_V24:
        raise ValueError("Human review uses an unsupported version")
    if (
        review.get("candidate_id") != candidate_id
        or review.get("rendered_packet_sha256") != packet_sha256
    ):
        raise ValueError("Human review is not bound to this candidate and packet")
    reviewer = review.get("reviewer")
    if (
        not isinstance(reviewer, Mapping)
        or set(reviewer) != {"identity", "role", "reviewed_at", "attestation"}
        or reviewer.get("role") != expected_role
    ):
        raise ValueError(f"Human review must have role {expected_role}")
    identity = str(reviewer.get("identity") or "").strip()
    if not identity or any(
        term in identity.casefold() for term in _FORBIDDEN_REVIEWER_TERMS
    ):
        raise ValueError("Reviewer identity must identify an external human reviewer")
    if reviewer.get("attestation") != _HUMAN_ATTESTATION:
        raise ValueError("Human reviewer attestation is missing or incorrect")
    try:
        reviewed_at = datetime.fromisoformat(str(reviewer.get("reviewed_at")))
    except ValueError as exc:
        raise ValueError("reviewed_at must be an RFC 3339 timestamp") from exc
    if reviewed_at.tzinfo is None:
        raise ValueError("reviewed_at must include a timezone")
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise TypeError("Human review decisions must be an array")
    by_id: dict[str, str] = {}
    required_checks = {
        "question_quality",
        "subtype",
        "scope",
        "answerability",
        "evidence",
        "semantic_anchors",
        "refusal_validity",
    }
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise TypeError("Human review decisions must be objects")
        if set(decision) != {"task_id", "decision", "checks", "comment"}:
            raise ValueError("Human review decision fields are missing or unexpected")
        task_id = str(decision.get("task_id") or "")
        if task_id in by_id:
            raise ValueError(f"Duplicate review decision: {task_id}")
        disposition = decision.get("decision")
        if disposition not in {"APPROVE", "REJECT"}:
            raise ValueError("Review decision must be APPROVE or REJECT")
        checks = decision.get("checks")
        if not isinstance(checks, Mapping) or set(checks) != required_checks:
            raise ValueError(f"Review checks are incomplete for {task_id}")
        if any(type(value) is not bool for value in checks.values()):
            raise TypeError(f"Review checks must be booleans for {task_id}")
        if disposition == "APPROVE" and any(
            value is not True for value in checks.values()
        ):
            raise ValueError(f"Approved case {task_id} must pass every review check")
        comment = decision.get("comment")
        if comment is not None and not isinstance(comment, str):
            raise TypeError("Human review comments must be strings or null")
        if disposition == "REJECT" and not str(comment or "").strip():
            raise ValueError(f"Rejected case {task_id} requires a corrective comment")
        by_id[task_id] = str(disposition)
    if set(by_id) != expected_ids:
        raise ValueError(
            f"{expected_role} review membership differs: expected {len(expected_ids)}, found {len(by_id)}"
        )
    return by_id


def write_agent_benchmark_approval_v24(
    candidate_dir: str | Path,
    rendered_packet: str | Path,
    primary_review: str | Path,
    secondary_review: str | Path,
    output_path: str | Path,
    *,
    adjudication_review: str | Path | None = None,
) -> dict[str, Any]:
    """Validate independent human decisions and write a freeze-enabling approval."""

    root = Path(candidate_dir)
    verification = verify_agent_benchmark_candidate_v24(root)
    if not verification["valid"]:
        raise ValueError("Invalid v2.4 candidate: " + "; ".join(verification["errors"]))
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite approval: {destination}")
    manifest = _read_json(root / "candidate_manifest.json")
    packet_path = Path(rendered_packet)
    packet_sha256 = _sha256_path(packet_path)
    packet_json = _read_json(root / "review_packet.json")
    all_ids = {str(row["task_id"]) for row in packet_json["records"]}
    sample = _read_json(root / "review_sample_40.json")
    sample_ids = {str(task_id) for task_id in sample["task_ids"]}
    primary_path = Path(primary_review)
    secondary_path = Path(secondary_review)
    primary = _read_json(primary_path)
    secondary = _read_json(secondary_path)
    primary_decisions = _validate_human_review(
        primary,
        candidate_id=manifest["candidate_id"],
        packet_sha256=packet_sha256,
        expected_role="PRIMARY",
        expected_ids=all_ids,
    )
    secondary_decisions = _validate_human_review(
        secondary,
        candidate_id=manifest["candidate_id"],
        packet_sha256=packet_sha256,
        expected_role="SECONDARY",
        expected_ids=sample_ids,
    )
    primary_identity = (
        _SPACE_RE.sub(" ", str(primary["reviewer"]["identity"])).strip().casefold()
    )
    secondary_identity = (
        _SPACE_RE.sub(" ", str(secondary["reviewer"]["identity"])).strip().casefold()
    )
    if primary_identity == secondary_identity:
        raise ValueError(
            "Primary and secondary reviews require different human identities"
        )
    disagreements = {
        task_id
        for task_id in sample_ids
        if primary_decisions[task_id] != secondary_decisions[task_id]
    }
    records_by_id = {str(row["task_id"]): row for row in packet_json["records"]}
    affected_subtypes = sorted(
        {str(records_by_id[task_id]["narrative_subtype"]) for task_id in disagreements}
    )
    adjudication_binding = None
    final_decisions = dict(primary_decisions)
    if affected_subtypes:
        if adjudication_review is None:
            raise ValueError(
                "Reviewer disagreement requires subtype-wide adjudication: "
                + ", ".join(affected_subtypes)
            )
        adjudication_path = Path(adjudication_review)
        adjudication = _read_json(adjudication_path)
        expected_adjudication_fields = {
            "adjudication_version",
            "candidate_id",
            "rendered_packet_sha256",
            "affected_subtypes",
            "reviewers",
            "decisions",
        }
        if (
            set(adjudication) != expected_adjudication_fields
            or not isinstance(adjudication.get("reviewers"), Mapping)
            or adjudication.get("adjudication_version")
            != "auditops-benchmark-adjudication.v2.4"
            or adjudication.get("candidate_id") != manifest["candidate_id"]
            or adjudication.get("rendered_packet_sha256") != packet_sha256
            or adjudication.get("affected_subtypes") != affected_subtypes
        ):
            raise ValueError("Adjudication identity or affected subtypes are invalid")
        reviewer_binding = adjudication["reviewers"]
        if (
            set(reviewer_binding)
            != {"primary_identity", "secondary_identity", "attestation"}
            or _SPACE_RE.sub(" ", str(reviewer_binding.get("primary_identity") or ""))
            .strip()
            .casefold()
            != primary_identity
            or _SPACE_RE.sub(" ", str(reviewer_binding.get("secondary_identity") or ""))
            .strip()
            .casefold()
            != secondary_identity
            or reviewer_binding.get("attestation") != _ADJUDICATION_ATTESTATION
        ):
            raise ValueError("Adjudication reviewer binding or attestation is invalid")
        expected_adjudication_ids = {
            task_id
            for task_id, record in records_by_id.items()
            if record["narrative_subtype"] in affected_subtypes
        }
        decisions = adjudication.get("decisions")
        if not isinstance(decisions, list):
            raise TypeError("Adjudication decisions must be an array")
        adjudicated_ids: set[str] = set()
        for decision in decisions:
            if not isinstance(decision, Mapping):
                raise TypeError("Adjudication decisions must be objects")
            if set(decision) != {
                "task_id",
                "primary_re_reviewed",
                "secondary_re_reviewed",
                "final_decision",
                "comment",
            }:
                raise ValueError(
                    "Adjudication decision fields are missing or unexpected"
                )
            task_id = str(decision.get("task_id") or "")
            if task_id in adjudicated_ids:
                raise ValueError(f"Duplicate adjudication task: {task_id}")
            if (
                decision.get("primary_re_reviewed") is not True
                or decision.get("secondary_re_reviewed") is not True
            ):
                raise ValueError(
                    "Both reviewers must re-review every affected subtype case"
                )
            final = decision.get("final_decision")
            if final not in {"APPROVE", "REJECT"}:
                raise ValueError("Adjudication final_decision is invalid")
            comment = decision.get("comment")
            if not isinstance(comment, str) or not comment.strip():
                raise ValueError(
                    "Every adjudication decision requires a recorded rationale"
                )
            final_decisions[task_id] = str(final)
            adjudicated_ids.add(task_id)
        if adjudicated_ids != expected_adjudication_ids:
            raise ValueError("Adjudication does not cover every affected subtype case")
        adjudication_binding = {
            "sha256": _sha256_path(adjudication_path),
            "affected_subtypes": affected_subtypes,
            "records": len(adjudicated_ids),
        }
    elif adjudication_review is not None:
        raise ValueError("Adjudication must not be supplied when reviewers agree")
    rejected = sorted(
        task_id
        for task_id, decision in final_decisions.items()
        if decision != "APPROVE"
    )
    if rejected:
        raise ValueError(
            f"Benchmark cannot be approved while {len(rejected)} narrative cases are rejected"
        )
    source_manifest = _read_json(root / "source_manifest.json")
    approval_material = {
        "approval_version": APPROVAL_VERSION_V24,
        "candidate_id": manifest["candidate_id"],
        "candidate_manifest_sha256": _sha256_path(root / "candidate_manifest.json"),
        "rendered_packet_sha256": packet_sha256,
        "review_packet_json_sha256": _sha256_path(root / "review_packet.json"),
        "source_manifest_sha256": _sha256_path(root / "source_manifest.json"),
        "source_hashes": source_manifest["sources"],
        "primary_review": {
            "sha256": _sha256_path(primary_path),
            "reviewer": primary["reviewer"],
            "case_count": len(primary_decisions),
        },
        "secondary_review": {
            "sha256": _sha256_path(secondary_path),
            "reviewer": secondary["reviewer"],
            "case_count": len(secondary_decisions),
            "sample_sha256": _sha256_path(root / "review_sample_40.json"),
        },
        "adjudication": adjudication_binding,
        "approved_task_ids_sha256": canonical_json_sha256(sorted(final_decisions)),
        "approved_case_count": len(final_decisions),
        "external_human_approval": True,
    }
    approval = {
        **approval_material,
        "approval_id": canonical_json_sha256(approval_material),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        _write_json(temporary, approval)
        assert_no_secrets([temporary])
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "approval_id": approval["approval_id"],
        "output": str(destination),
        "approved_case_count": approval["approved_case_count"],
    }


def _validate_approval_for_freeze(
    root: Path, approval_path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    approval = _read_json(approval_path)
    if approval.get("approval_version") != APPROVAL_VERSION_V24:
        raise ValueError("Unsupported v2.4 approval version")
    if approval.get("external_human_approval") is not True:
        raise ValueError("Benchmark approval is not externally human-approved")
    if approval.get("candidate_id") != manifest.get("candidate_id"):
        raise ValueError("Approval candidate binding mismatch")
    if approval.get("candidate_manifest_sha256") != _sha256_path(
        root / "candidate_manifest.json"
    ):
        raise ValueError("Approval candidate manifest hash mismatch")
    if approval.get("review_packet_json_sha256") != _sha256_path(
        root / "review_packet.json"
    ):
        raise ValueError("Approval review packet hash mismatch")
    if approval.get("source_manifest_sha256") != _sha256_path(
        root / "source_manifest.json"
    ):
        raise ValueError("Approval source manifest hash mismatch")
    material = {key: value for key, value in approval.items() if key != "approval_id"}
    if approval.get("approval_id") != canonical_json_sha256(material):
        raise ValueError("Approval ID does not bind the complete approval")
    if approval.get("approved_case_count") != 200:
        raise ValueError("Approval must cover all 200 narrative cases")
    return approval


def freeze_agent_benchmark_v24(
    candidate_dir: str | Path,
    approval_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze an approved candidate as a new immutable v2.4 benchmark."""

    root = Path(candidate_dir)
    verification = verify_agent_benchmark_candidate_v24(root)
    if not verification["valid"]:
        raise ValueError("Invalid v2.4 candidate: " + "; ".join(verification["errors"]))
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite benchmark directory: {destination}"
        )
    candidate = _read_json(root / "candidate_manifest.json")
    approval_source = Path(approval_path)
    approval = _validate_approval_for_freeze(root, approval_source, candidate)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    mapping = {
        "candidate_cases.jsonl": "cases.jsonl",
        "candidate_evidence.jsonl": "evidence.jsonl",
        "candidate_gold.jsonl": "gold.jsonl",
        "candidate_verifier_observations.jsonl": "verifier_observations.jsonl",
        "candidate_case_lineage.jsonl": "case_lineage.jsonl",
        "review_sample_40.json": "review_sample_40.json",
        "gate_50.json": "gate_50.json",
        "source_manifest.json": "source_manifest.json",
    }
    try:
        for source_name, target_name in mapping.items():
            shutil.copyfile(root / source_name, temporary / target_name)
        shutil.copyfile(approval_source, temporary / "benchmark_approval.json")
        _write_jsonl(temporary / "few_shot.jsonl", [])
        artifacts = {
            name: _artifact(
                temporary / name,
                0
                if name == "few_shot.jsonl"
                else (
                    len(_read_jsonl(temporary / name))
                    if name.endswith(".jsonl")
                    else None
                ),
            )
            for name in [*mapping.values(), "benchmark_approval.json", "few_shot.jsonl"]
        }
        cases = _read_jsonl(temporary / "cases.jsonl")
        gold = _read_jsonl(temporary / "gold.jsonl")
        narrative_counts = Counter(
            str(row["target"]["status"])
            for row in gold
            if row.get("task_type") == "narrative_citation"
        )
        issuer_counts = Counter(
            str(case["entity"]["source_entity_id"]) for case in cases
        )
        manifest_material = {
            "benchmark_version": BENCHMARK_VERSION_V24,
            "benchmark_profile": "us_cached20_human_reviewed_v2.4",
            "runtime_contract_version": "v2.3",
            "seed": candidate["seed"],
            "corpus_id": candidate["corpus_id"],
            "jurisdiction": "US",
            "reporting_framework": "US-GAAP",
            "standards_version": "PCAOB_PUBLIC_FILING_METADATA_V0.1",
            "source_system": "SEC-EDGAR-CACHED",
            "candidate_id": candidate["candidate_id"],
            "approval_id": approval["approval_id"],
            "counts": {
                "quant": 300,
                "narrative": 200,
                "narrative_answerable": narrative_counts["OK"],
                "narrative_unanswerable": narrative_counts["REFUSAL"],
                "issuers": len(issuer_counts),
                "total": 500,
                "few_shot": 0,
                "narrative_gate": 50,
            },
            "quantitative_quota_policy": {
                "per_issuer": 15,
                "global_answers": 150,
                "global_refusals": 150,
                "per_issuer_answer_split": "first-ten-8/7;second-ten-7/8",
                "refusal_code": "MISSING_INPUT",
            },
            "narrative_quota_policy": {
                "per_issuer": 10,
                "answers_per_issuer": 5,
                "refusals_per_issuer": 5,
                "subtype_quotas": {
                    subtype: {"answerable": quota[0], "refusal": quota[1]}
                    for subtype, quota in sorted(_SUBTYPE_QUOTAS.items())
                },
            },
            "review_policy": {
                "primary_cases": 200,
                "secondary_cases": 40,
                "secondary_per_stratum": 5,
                "secondary_per_issuer": 2,
                "disagreement_requires_subtype_wide_re_review": True,
                "codex_self_approval_permitted": False,
            },
            "few_shot_policy": {
                "status": "BLOCKED_PENDING_ENTITY_DISJOINT_REVIEW",
                "request_pack_size": 0,
            },
            "artifact_visibility": {
                "inference_visible": [
                    "cases.jsonl",
                    "evidence.jsonl",
                    "few_shot.jsonl",
                    "gate_50.json",
                ],
                "evaluator_only": [
                    "gold.jsonl",
                    "verifier_observations.jsonl",
                    "case_lineage.jsonl",
                    "benchmark_approval.json",
                    "review_sample_40.json",
                ],
            },
            "artifacts": artifacts,
        }
        manifest = {
            **manifest_material,
            "benchmark_id": canonical_json_sha256(manifest_material),
        }
        _write_json(temporary / "benchmark_manifest.json", manifest)
        final_verification = verify_agent_benchmark_v24(temporary)
        if not final_verification["valid"]:
            raise ValueError(
                "Invalid v2.4 benchmark: " + "; ".join(final_verification["errors"])
            )
        assert_no_secrets([temporary])
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "benchmark_id": manifest["benchmark_id"],
        "output_dir": str(destination),
        "counts": manifest["counts"],
        "verified": True,
    }


def verify_agent_benchmark_v24(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    errors: list[str] = []
    try:
        manifest = _read_json(root / "benchmark_manifest.json")
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        return {
            "valid": False,
            "errors": [f"benchmark manifest could not be loaded: {exc}"],
        }
    expected_fields = {
        "benchmark_version",
        "benchmark_profile",
        "runtime_contract_version",
        "seed",
        "corpus_id",
        "jurisdiction",
        "reporting_framework",
        "standards_version",
        "source_system",
        "candidate_id",
        "approval_id",
        "counts",
        "quantitative_quota_policy",
        "narrative_quota_policy",
        "review_policy",
        "few_shot_policy",
        "artifact_visibility",
        "artifacts",
        "benchmark_id",
    }
    if set(manifest) != expected_fields:
        errors.append("benchmark manifest fields are missing or unexpected")
    if manifest.get("benchmark_version") != BENCHMARK_VERSION_V24:
        errors.append("unsupported benchmark version")
    if manifest.get("runtime_contract_version") != "v2.3":
        errors.append("v2.4 benchmark must use runtime contract v2.3")
    material = {key: value for key, value in manifest.items() if key != "benchmark_id"}
    if manifest.get("benchmark_id") != canonical_json_sha256(material):
        errors.append("benchmark_id does not bind complete manifest material")
    expected_artifacts = {
        "cases.jsonl",
        "evidence.jsonl",
        "gold.jsonl",
        "verifier_observations.jsonl",
        "case_lineage.jsonl",
        "review_sample_40.json",
        "gate_50.json",
        "source_manifest.json",
        "benchmark_approval.json",
        "few_shot.jsonl",
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        errors.append("benchmark artifact set is missing or unexpected")
    _verify_artifacts(root, artifacts, errors=errors)
    expected_visibility = {
        "inference_visible": [
            "cases.jsonl",
            "evidence.jsonl",
            "few_shot.jsonl",
            "gate_50.json",
        ],
        "evaluator_only": [
            "gold.jsonl",
            "verifier_observations.jsonl",
            "case_lineage.jsonl",
            "benchmark_approval.json",
            "review_sample_40.json",
        ],
    }
    if manifest.get("artifact_visibility") != expected_visibility:
        errors.append("benchmark inference/evaluator visibility boundary is invalid")
    try:
        cases = _read_jsonl(root / "cases.jsonl")
        evidence = _read_jsonl(root / "evidence.jsonl")
        gold = _read_jsonl(root / "gold.jsonl")
        observations = _read_jsonl(root / "verifier_observations.jsonl")
        lineage = _read_jsonl(root / "case_lineage.jsonl")
        gate = _read_json(root / "gate_50.json")
        approval = _read_json(root / "benchmark_approval.json")
        few_shot = _read_jsonl(root / "few_shot.jsonl")
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        errors.append(f"benchmark artifacts could not be loaded: {exc}")
        cases, evidence, gold, observations, lineage, gate, approval, few_shot = (
            [],
            [],
            [],
            [],
            [],
            {},
            {},
            [],
        )
    ids = [str(row.get("task_id")) for row in cases]
    for label, rows in (
        ("evidence", evidence),
        ("gold", gold),
        ("observations", observations),
        ("lineage", lineage),
    ):
        if [str(row.get("task_id")) for row in rows] != ids:
            errors.append(f"{label} membership/order differs from cases")
    if len(ids) != 500 or len(set(ids)) != 500:
        errors.append("benchmark must contain 500 unique tasks")
    if few_shot:
        errors.append("v2.4 benchmark must not contain unapproved few-shot examples")
    gold_by_id = {str(row.get("task_id")): row for row in gold}
    for case in cases:
        try:
            validate_agent_task_input(case)
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid task {case.get('task_id')}: {exc}")
            continue
        if case.get("agent_task_input_version") != "v2.3":
            errors.append(
                f"task {case.get('task_id')} does not use v2.3 runtime contract"
            )
        if case.get("task_type") == "narrative_citation":
            try:
                validate_question_scope_v24(case)
                validate_narrative_gold_v24(gold_by_id[str(case["task_id"])], case)
            except (TypeError, ValueError) as exc:
                errors.append(f"invalid narrative task {case.get('task_id')}: {exc}")
    errors.extend(_narrative_quota_errors(cases, gold))
    quant_gold = [row for row in gold if row.get("task_type") == "quant_metric"]
    quant_status = Counter(
        str(row.get("target", {}).get("status")) for row in quant_gold
    )
    refusal_codes = {
        str(row.get("target", {}).get("refusal_code"))
        for row in quant_gold
        if row.get("target", {}).get("status") == "REFUSAL"
    }
    if len(quant_gold) != 300 or quant_status != Counter({"OK": 150, "REFUSAL": 150}):
        errors.append("quantitative membership is not the required 150/150 split")
    if refusal_codes != {"MISSING_INPUT"}:
        errors.append("v2.4 filing refusals must use only MISSING_INPUT")
    gate_ids = gate.get("task_ids") if isinstance(gate, Mapping) else None
    if (
        gate.get("gate_membership_version") != GATE_MEMBERSHIP_VERSION_V24
        or not isinstance(gate_ids, list)
        or len(gate_ids) != 50
        or gate.get("task_ids_sha256") != canonical_json_sha256(gate_ids or [])
        or not set(gate_ids or []).issubset(set(ids))
    ):
        errors.append("frozen narrative gate is invalid")
    if approval.get("approval_version") != APPROVAL_VERSION_V24 or approval.get(
        "approval_id"
    ) != manifest.get("approval_id"):
        errors.append("benchmark approval binding is invalid")
    approval_material = {
        key: value for key, value in approval.items() if key != "approval_id"
    }
    if approval and approval.get("approval_id") != canonical_json_sha256(
        approval_material
    ):
        errors.append("benchmark approval ID is invalid")
    try:
        assert_no_secrets([root])
    except ValueError as exc:
        errors.append(str(exc))
    return {
        "valid": not errors,
        "errors": errors,
        "benchmark_id": manifest.get("benchmark_id"),
    }


__all__ = [
    "APPROVAL_VERSION_V24",
    "BENCHMARK_VERSION_V24",
    "CANDIDATE_VERSION_V24",
    "CANDIDATE_VERSION_V24_R2",
    "GATE_MEMBERSHIP_VERSION_V24",
    "NARRATIVE_GOLD_VERSION_V24",
    "REVIEW_DECISION_VERSION_V24",
    "REVIEW_PACKET_VERSION_V24",
    "REVIEW_PACKET_VERSION_V24_R2",
    "build_r1_to_r2_defect_report_v24",
    "draft_agent_benchmark_v24",
    "draft_agent_benchmark_v24r2",
    "freeze_agent_benchmark_v24",
    "render_agent_benchmark_review_v24",
    "validate_narrative_gold_v24",
    "validate_natural_question",
    "validate_question_scope_v24",
    "validate_semantic_anchor_v24r2",
    "verify_agent_benchmark_candidate_v24",
    "verify_agent_benchmark_v24",
    "write_agent_benchmark_approval_v24",
]
