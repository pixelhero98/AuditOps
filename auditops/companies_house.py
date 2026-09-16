from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import zipfile
import zlib
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import strptime
from typing import Any

from lxml import etree

from .provenance import assert_no_secrets

COMPANIES_HOUSE_CORPUS_VERSION = "companies-house-corpus.v1"
COMPANIES_HOUSE_VLM_VERSION = "companies-house-vlm-diagnostic.v1"
COMPANIES_HOUSE_SOURCE_MANIFEST_VERSION = "companies-house-source-manifest.v1"
COMPANIES_HOUSE_PARSE_FAILURE_VERSION = "companies-house-parse-failure.v1"
_SUPPORTED_XHTML_SUFFIXES = (".html", ".htm", ".xhtml")
_SUPPORTED_ZIP_COMPRESSION_METHODS = frozenset(
    {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
)
_PARSE_FAILURE_CODES = frozenset(
    {
        "UNSAFE_MEMBER_NAME",
        "DUPLICATE_MEMBER_NAME",
        "ENCRYPTED_MEMBER",
        "UNSUPPORTED_COMPRESSION",
        "NON_REGULAR_MEMBER",
        "INVALID_MEMBER_METADATA",
        "MEMBER_SIZE_LIMIT_EXCEEDED",
        "COMPRESSION_RATIO_LIMIT_EXCEEDED",
        "ARCHIVE_MEMBER_READ_ERROR",
        "MEMBER_SIZE_MISMATCH",
        "XHTML_PARSE_ERROR",
    }
)
_PARSE_FAILURE_STAGES = frozenset({"ZIP_INFO", "READ", "PARSE"})
_MISSING_NUMERIC_TEXT = {"", "-", "--", "–", "—", "nil", "n/a"}
_ZERO_DASH_TRANSFORMS = {"numdash", "zerodash", "numzerodash"}
_DOT_DECIMAL_TRANSFORMS = {
    "numdotdecimal",
    "numcommadot",
    "numspacedot",
}
_COMMA_DECIMAL_TRANSFORMS = {
    "numcommadecimal",
    "numdotcomma",
    "numspacecomma",
}

_CURRENT_ASSET_CONCEPTS = {
    "currentassets",
}
_CURRENT_LIABILITY_CONCEPTS = {
    "creditorsamountsfallingduewithinoneyear",
    "currentliabilities",
    "creditors",
}
_CURRENT_CREDITOR_DIMENSION_MEMBERS = {
    "financialinstrumentcurrentnoncurrentdimension": {"currentfinancialinstruments"},
    "maturitiesorexpirationperiodsdimension": {"withinoneyear"},
}
_NONCURRENT_CREDITOR_DIMENSION_MEMBERS = {
    "financialinstrumentcurrentnoncurrentdimension": {"noncurrentfinancialinstruments"},
    "maturitiesorexpirationperiodsdimension": {"afteroneyear"},
}
_COMPANY_NUMBER_CONCEPTS = {
    "ukcompanieshouseregisterednumber",
    "companieshouseregisterednumber",
}
_COMPANY_NAME_CONCEPTS = {
    "entitycurrentlegalorregisteredname",
}
_PERIOD_END_CONCEPTS = {
    "balancesheetdate",
    "enddateforperiodcoveredbyreport",
}


@dataclass(frozen=True)
class CompaniesHouseZipLimits:
    """Resource and path limits applied to ZIP metadata before member reads."""

    max_archive_members: int = 500_000
    max_member_uncompressed_bytes: int = 64 * 1024 * 1024
    max_total_uncompressed_bytes: int = 256 * 1024 * 1024 * 1024
    max_compression_ratio: int = 250
    max_member_name_chars: int = 1_024
    max_path_depth: int = 32

    def __post_init__(self) -> None:
        for field_name in (
            "max_archive_members",
            "max_member_uncompressed_bytes",
            "max_total_uncompressed_bytes",
            "max_compression_ratio",
            "max_member_name_chars",
            "max_path_depth",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

    def manifest_row(self) -> dict[str, int]:
        return {
            "max_archive_members": self.max_archive_members,
            "max_member_uncompressed_bytes": self.max_member_uncompressed_bytes,
            "max_total_uncompressed_bytes": self.max_total_uncompressed_bytes,
            "max_compression_ratio": self.max_compression_ratio,
            "max_member_name_chars": self.max_member_name_chars,
            "max_path_depth": self.max_path_depth,
        }


class CompaniesHouseArchiveValidationError(ValueError):
    """A typed archive-wide rejection raised before any member is read."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _stable_digest(*parts: Any) -> str:
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in materialized:
            handle.write(_json_dumps(dict(row)) + "\n")
    return len(materialized)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_concept(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]", "", name.split(":")[-1].lower())


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_date(value: str | None) -> str | None:
    """Return an ISO calendar date when an iXBRL/display date is recognisable."""

    if value is None:
        return None
    cleaned = _clean_text(value).replace("\u00a0", " ")
    if not cleaned:
        return None
    cleaned = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace(",", " ")
    cleaned = _clean_text(cleaned)
    iso_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})(?:[T ].*)?", cleaned)
    if iso_match:
        try:
            return date.fromisoformat(iso_match.group(0)[:10]).isoformat()
        except ValueError:
            return None
    formats = (
        "%Y%m%d",
        "%d%m%Y",
        "%d%m%y",
        "%d %B %Y",
        "%d %b %Y",
        "%d %m %Y",
        "%d %m %y",
        "%B %d %Y",
        "%b %d %Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%d/%m/%y",
        "%d-%m-%y",
        "%d.%m.%y",
    )
    for date_format in formats:
        try:
            parsed = strptime(cleaned, date_format)
            return date(parsed.tm_year, parsed.tm_mon, parsed.tm_mday).isoformat()
        except ValueError:
            continue
    return None


def _decimal_text(value: Decimal) -> str:
    normalized = value.normalize()
    rendered = format(normalized, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _parse_numeric(text: str, *, scale: str | None, sign: str | None) -> Decimal | None:
    cleaned = _clean_text(text).lower()
    if cleaned in _MISSING_NUMERIC_TEXT:
        return None
    parenthesized = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = (
        cleaned.strip("()")
        .replace(",", "")
        .replace("£", "")
        .replace("−", "-")
        .replace("\u00a0", "")
        .replace(" ", "")
    )
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if scale:
        try:
            value *= Decimal(10) ** int(scale)
        except (InvalidOperation, ValueError):
            return None
    if sign == "-" or parenthesized:
        value = -abs(value)
    return value


def _attribute_by_local_name(node: etree._Element, local_name: str) -> str | None:
    for qname, value in node.attrib.items():
        if etree.QName(qname).localname.lower() == local_name.lower():
            return value
    return None


def _element_local_name(node: etree._Element) -> str:
    return etree.QName(node).localname.lower() if isinstance(node.tag, str) else ""


def _node_text_without_ix_excludes(node: etree._Element) -> str:
    pieces: list[str] = []

    def visit(current: etree._Element) -> None:
        if current.text:
            pieces.append(current.text)
        for child in current:
            child_local_name = _element_local_name(child)
            if child_local_name and child_local_name != "exclude":
                visit(child)
            if child.tail:
                pieces.append(child.tail)

    visit(node)
    return _clean_text(" ".join(pieces))


def _visible_document_text(root: etree._Element) -> str:
    body_nodes = root.xpath("//*[local-name()='body']")
    visible_root = body_nodes[0] if body_nodes else root
    excluded_nodes = {
        "exclude",
        "header",
        "hidden",
        "references",
        "resources",
        "script",
        "style",
    }
    pieces: list[str] = []

    def visit(current: etree._Element) -> None:
        if current.text:
            pieces.append(current.text)
        for child in current:
            child_local_name = _element_local_name(child)
            if child_local_name and child_local_name not in excluded_nodes:
                visit(child)
            if child.tail:
                pieces.append(child.tail)

    visit(visible_root)
    return _clean_text(" ".join(pieces))


def _continued_fact_text(
    node: etree._Element, continuations: Mapping[str, etree._Element]
) -> tuple[str, bool, tuple[str, ...]]:
    pieces = [_node_text_without_ix_excludes(node)]
    continuation_ids: list[str] = []
    seen: set[str] = set()
    continued_at = _attribute_by_local_name(node, "continuedAt")
    while continued_at:
        if continued_at in seen or continued_at not in continuations:
            return _clean_text(" ".join(pieces)), False, tuple(continuation_ids)
        seen.add(continued_at)
        continuation_ids.append(continued_at)
        continuation = continuations[continued_at]
        pieces.append(_node_text_without_ix_excludes(continuation))
        continued_at = _attribute_by_local_name(continuation, "continuedAt")
    return _clean_text(" ".join(pieces)), True, tuple(continuation_ids)


def _apply_ix_numeric_transform(
    text: str, format_name: str | None
) -> tuple[str | None, bool]:
    """Apply the common SEC/FRC numeric Inline XBRL transformation families."""

    if not format_name:
        return text, True
    transform = _local_concept(format_name)
    cleaned = _clean_text(text).replace("\u00a0", " ")
    if transform in _ZERO_DASH_TRANSFORMS:
        if cleaned.lower() in _MISSING_NUMERIC_TEXT:
            return "0", True
        return cleaned, True
    if transform in _DOT_DECIMAL_TRANSFORMS:
        return (
            cleaned.replace(" ", "").replace("'", "").replace(",", ""),
            True,
        )
    if transform in _COMMA_DECIMAL_TRANSFORMS:
        return (
            cleaned.replace(" ", "")
            .replace("'", "")
            .replace(".", "")
            .replace(",", "."),
            True,
        )
    # Date, boolean, fixed-empty and other transformations must not be guessed as numbers.
    return None, False


def _first_text_by_concept(
    facts: Sequence[Mapping[str, Any]], concepts: set[str]
) -> str | None:
    for fact in facts:
        if fact["concept_local"] in concepts and fact.get("value_text"):
            return str(fact["value_text"])
    return None


def _detect_account_type(text_lower: str) -> str:
    for phrase, label in (
        ("micro-entity", "micro_entity"),
        ("micro entity", "micro_entity"),
        ("total exemption full", "total_exemption_full"),
        ("small companies regime", "small_company"),
        ("abridged accounts", "abridged"),
        ("dormant company", "dormant"),
    ):
        if phrase in text_lower:
            return label
    return "unknown"


def _detect_audit_status(text_lower: str) -> str:
    text_lower = text_lower.replace("’", "'").replace("‘", "'")
    if (
        "independent auditor's report" in text_lower
        or "independent auditors' report" in text_lower
    ):
        return "AUDITED"
    if (
        "exemption from audit" in text_lower
        or "exempt from the requirements relating to the audit" in text_lower
    ):
        return "AUDIT_EXEMPT"
    if "unaudited" in text_lower:
        return "UNAUDITED"
    return "UNKNOWN"


def _detect_reporting_framework(text_lower: str) -> str:
    if "frs 105" in text_lower:
        return "FRS_105"
    if "frs 102" in text_lower:
        return "FRS_102"
    if (
        "uk-adopted international accounting standards" in text_lower
        or "uk adopted international accounting standards" in text_lower
    ):
        return "UK_ADOPTED_IFRS"
    if "international financial reporting standards" in text_lower:
        return "IFRS"
    return "UNKNOWN"


def _extract_contexts(root: etree._Element) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for node in root.xpath("//*[local-name()='context']"):
        context_id = node.get("id")
        if not context_id:
            continue
        values: dict[str, Any] = {
            "instant": None,
            "start_date": None,
            "end_date": None,
            "entity_identifier": None,
            "entity_scheme": None,
            "dimensions": {},
            "has_segment": False,
            "has_scenario": False,
        }
        for field, local_name in (
            ("instant", "instant"),
            ("start_date", "startDate"),
            ("end_date", "endDate"),
        ):
            matches = node.xpath(f".//*[local-name()='{local_name}']/text()")
            if matches:
                raw_date = _clean_text(str(matches[0]))
                values[field] = _normalize_date(raw_date) or raw_date
        identifiers = node.xpath(
            ".//*[local-name()='entity']/*[local-name()='identifier']"
        )
        if identifiers:
            identifier = identifiers[0]
            values["entity_identifier"] = _clean_text(" ".join(identifier.itertext()))
            values["entity_scheme"] = identifier.get("scheme")

        dimensions: dict[str, dict[str, str]] = {}
        for container_name in ("segment", "scenario"):
            containers = node.xpath(f".//*[local-name()='{container_name}']")
            values[f"has_{container_name}"] = bool(containers)
            for container in containers:
                for member in container.xpath(
                    ".//*[local-name()='explicitMember' or local-name()='typedMember']"
                ):
                    dimension = _attribute_by_local_name(member, "dimension")
                    if not dimension:
                        continue
                    kind = etree.QName(member).localname
                    dimensions[dimension] = {
                        "kind": "explicit" if kind == "explicitMember" else "typed",
                        "member": _clean_text(" ".join(member.itertext())),
                    }
        values["dimensions"] = dimensions
        contexts[context_id] = values
    return contexts


def _canonical_unit(measure: str) -> tuple[str | None, str]:
    cleaned = _clean_text(measure)
    local = cleaned.rsplit(":", 1)[-1].rsplit("}", 1)[-1].lower()
    if re.fullmatch(r"[a-z]{3}", local) and (
        "iso4217" in cleaned.lower() or ":" in cleaned or "}" in cleaned
    ):
        return local, "currency"
    if local in {"pure", "shares"}:
        return local, local
    return None, "unknown"


def _extract_units(root: etree._Element) -> dict[str, dict[str, str | None]]:
    units: dict[str, dict[str, str | None]] = {}
    for node in root.xpath("//*[local-name()='unit']"):
        unit_id = node.get("id")
        if not unit_id:
            continue
        measures = [
            _clean_text(str(value))
            for value in node.xpath(".//*[local-name()='measure']/text()")
        ]
        measure = " / ".join(measures)
        canonical, unit_type = (
            _canonical_unit(measures[0]) if len(measures) == 1 else (None, "complex")
        )
        units[unit_id] = {
            "measure": measure or None,
            "canonical": canonical,
            "type": unit_type,
        }
    return units


def _split_text_chunks(
    text: str, source_name: str, *, max_chars: int = 1200
) -> list[dict[str, Any]]:
    sentences = [
        part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()
    ]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current and current_len + len(sentence) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        if len(sentence) > max_chars:
            for offset in range(0, len(sentence), max_chars):
                if current:
                    chunks.append(" ".join(current))
                    current = []
                    current_len = 0
                chunks.append(sentence[offset : offset + max_chars])
            continue
        current.append(sentence)
        current_len += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))

    result = []
    for index, chunk in enumerate(chunks):
        evidence_id = (
            f"chunk::{_stable_digest('companies-house', source_name, index, chunk)}"
        )
        result.append(
            {
                "evidence_id": evidence_id,
                "chunk_evidence_id": evidence_id,
                "source_file": source_name,
                "chunk_index": index,
                "text": chunk,
                "text_sha256": _stable_digest(chunk),
            }
        )
    return result


@dataclass(frozen=True)
class ParsedCompaniesHouseFiling:
    filing_id: str
    source_name: str
    source_sha256: str
    company_number: str | None
    company_name: str | None
    period_end: str | None
    account_type: str
    audit_status: str
    reporting_framework: str
    taxonomy_refs: tuple[str, ...]
    facts: tuple[dict[str, Any], ...]
    chunks: tuple[dict[str, Any], ...]

    def manifest_row(self) -> dict[str, Any]:
        return {
            "filing_id": self.filing_id,
            "source_system": "UK_COMPANIES_HOUSE",
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "company_number": self.company_number,
            "company_name": self.company_name,
            "period_end": self.period_end,
            "account_type": self.account_type,
            "audit_status": self.audit_status,
            "reporting_framework": self.reporting_framework,
            "taxonomy_refs": list(self.taxonomy_refs),
            "fact_count": len(self.facts),
            "chunk_count": len(self.chunks),
        }


def parse_companies_house_xhtml(
    payload: bytes, *, source_name: str
) -> ParsedCompaniesHouseFiling:
    parser = etree.XMLParser(
        resolve_entities=False, no_network=True, recover=True, huge_tree=False
    )
    try:
        root = etree.fromstring(payload, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ValueError(
            f"Invalid Companies House XHTML in {source_name}: {exc}"
        ) from exc
    if root is None:
        raise ValueError(
            f"Companies House XHTML produced no document root: {source_name}"
        )

    contexts = _extract_contexts(root)
    units = _extract_units(root)
    continuation_nodes: dict[str, etree._Element] = {}
    duplicate_continuation_ids: set[str] = set()
    for continuation in root.xpath("//*[local-name()='continuation']"):
        continuation_id = _attribute_by_local_name(continuation, "id")
        if not continuation_id or continuation_id in duplicate_continuation_ids:
            continue
        if continuation_id in continuation_nodes:
            continuation_nodes.pop(continuation_id)
            duplicate_continuation_ids.add(continuation_id)
        else:
            continuation_nodes[continuation_id] = continuation
    facts: list[dict[str, Any]] = []
    fact_nodes = root.xpath(
        "//*[local-name()='nonFraction' or local-name()='nonNumeric']"
    )
    for ordinal, node in enumerate(fact_nodes):
        name = node.get("name")
        if not name:
            continue
        text, continuation_valid, continuation_ids = _continued_fact_text(
            node, continuation_nodes
        )
        concept_local = _local_concept(name)
        context_ref = _attribute_by_local_name(node, "contextRef")
        unit_ref = _attribute_by_local_name(node, "unitRef")
        context = contexts.get(context_ref or "", {})
        unit = units.get(unit_ref or "", {})
        nil_value = (_attribute_by_local_name(node, "nil") or "").lower() in {
            "true",
            "1",
        }
        numeric_value = None
        transform_supported = True
        if etree.QName(node).localname.lower() == "nonfraction":
            transformed_text, transform_supported = _apply_ix_numeric_transform(
                text, _attribute_by_local_name(node, "format")
            )
            if (
                not nil_value
                and continuation_valid
                and transform_supported
                and transformed_text is not None
            ):
                numeric_value = _parse_numeric(
                    transformed_text,
                    scale=_attribute_by_local_name(node, "scale"),
                    sign=_attribute_by_local_name(node, "sign"),
                )
        evidence_id = f"fact::{_stable_digest('companies-house', source_name, ordinal, name, context_ref, text)}"
        facts.append(
            {
                "evidence_id": evidence_id,
                "fact_evidence_id": evidence_id,
                "concept": name,
                "concept_local": concept_local,
                "context_id": context_ref,
                "period": {
                    key: context.get(key)
                    for key in ("instant", "start_date", "end_date")
                },
                "entity_identifier": context.get("entity_identifier"),
                "entity_scheme": context.get("entity_scheme"),
                "dimensions": dict(context.get("dimensions") or {}),
                "has_segment": bool(context.get("has_segment")),
                "has_scenario": bool(context.get("has_scenario")),
                "unit_ref": unit_ref,
                "unit": unit.get("measure"),
                "unit_canon": unit.get("canonical"),
                "unit_type": unit.get("type"),
                "value_num_exact": _decimal_text(numeric_value)
                if numeric_value is not None
                else None,
                "value_text": text,
                "is_nil": nil_value,
                "transform": _attribute_by_local_name(node, "format"),
                "transform_supported": transform_supported,
                "continuation_ids": list(continuation_ids),
                "continuation_valid": continuation_valid,
                "source_file": source_name,
                "ordinal": ordinal,
            }
        )

    visible_text = _visible_document_text(root)
    text_lower = visible_text.lower()
    company_number = _first_text_by_concept(facts, _COMPANY_NUMBER_CONCEPTS)
    company_name = _first_text_by_concept(facts, _COMPANY_NAME_CONCEPTS)
    reported_period_ends = {
        normalized
        for fact in facts
        if fact["concept_local"] in _PERIOD_END_CONCEPTS
        and (normalized := _normalize_date(str(fact.get("value_text") or "")))
    }
    period_end = max(reported_period_ends) if reported_period_ends else None
    taxonomy_refs = tuple(
        sorted(
            {
                str(value)
                for value in root.xpath(
                    "//*[local-name()='schemaRef']/@*[local-name()='href']"
                )
                if value
            }
        )
    )
    source_sha256 = _sha256_bytes(payload)
    filing_id = f"ch::{company_number or 'unknown'}::{period_end or source_sha256[:12]}::{source_sha256[:12]}"
    return ParsedCompaniesHouseFiling(
        filing_id=filing_id,
        source_name=source_name,
        source_sha256=source_sha256,
        company_number=company_number,
        company_name=company_name,
        period_end=period_end,
        account_type=_detect_account_type(text_lower),
        audit_status=_detect_audit_status(text_lower),
        reporting_framework=_detect_reporting_framework(text_lower),
        taxonomy_refs=taxonomy_refs,
        facts=tuple(facts),
        chunks=tuple(_split_text_chunks(visible_text, source_name)),
    )


@dataclass(frozen=True)
class _CurrentFactSelection:
    fact: Mapping[str, Any] | None
    refusal_code: str | None
    evidence_facts: tuple[Mapping[str, Any], ...] = ()


def _normalized_entity_identifier(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _fact_period_end(fact: Mapping[str, Any]) -> str | None:
    period = fact.get("period") or {}
    # Balance-sheet inputs are ASOF facts. A duration ending on the report date is
    # not an interchangeable context, even when its end date happens to match.
    return _normalize_date(period.get("instant"))


def _fact_period_key(fact: Mapping[str, Any]) -> str:
    period = fact.get("period") or {}
    instant = _normalize_date(period.get("instant"))
    if instant:
        return f"ASOF_{instant.replace('-', '')}"
    start_date = _normalize_date(period.get("start_date"))
    end_date = _normalize_date(period.get("end_date"))
    if start_date or end_date:
        return f"DURATION_{(start_date or 'UNKNOWN').replace('-', '')}_{(end_date or 'UNKNOWN').replace('-', '')}"
    return "UNKNOWN_CONTEXT_PERIOD"


def _bounded_diagnostic_facts(
    facts: Sequence[Mapping[str, Any]], *, limit: int = 2
) -> tuple[Mapping[str, Any], ...]:
    selected: list[Mapping[str, Any]] = []
    signatures: set[str] = set()
    for fact in sorted(facts, key=lambda item: int(item["ordinal"])):
        signature = _json_dumps(
            {
                "context_id": fact.get("context_id"),
                "dimensions": fact.get("dimensions") or {},
                "entity_identifier": fact.get("entity_identifier"),
                "is_nil": bool(fact.get("is_nil")),
                "period": fact.get("period") or {},
                "unit": fact.get("unit_canon"),
                "value": fact.get("value_num_exact"),
            }
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        selected.append(fact)
        if len(selected) == limit:
            break
    return tuple(selected)


def _is_default_context(fact: Mapping[str, Any]) -> bool:
    return (
        not fact.get("dimensions")
        and not fact.get("has_segment")
        and not fact.get("has_scenario")
    )


def _creditor_dimension_is_in(
    fact: Mapping[str, Any], allowed: Mapping[str, set[str]]
) -> bool:
    if fact.get("concept_local") != "creditors" or fact.get("has_scenario"):
        return False
    dimensions = fact.get("dimensions") or {}
    if len(dimensions) != 1:
        return False
    dimension_name, member = next(iter(dimensions.items()))
    if not isinstance(member, Mapping):
        return False
    allowed_members = allowed.get(_local_concept(str(dimension_name)))
    return bool(
        allowed_members
        and _local_concept(str(member.get("member") or "")) in allowed_members
    )


def _is_allowlisted_current_creditor_context(fact: Mapping[str, Any]) -> bool:
    return _creditor_dimension_is_in(fact, _CURRENT_CREDITOR_DIMENSION_MEMBERS)


def _is_known_noncurrent_creditor_context(fact: Mapping[str, Any]) -> bool:
    return _creditor_dimension_is_in(fact, _NONCURRENT_CREDITOR_DIMENSION_MEMBERS)


def _select_current_fact(
    filing: ParsedCompaniesHouseFiling, concepts: set[str]
) -> _CurrentFactSelection:
    concept_candidates = [
        fact for fact in filing.facts if fact["concept_local"] in concepts
    ]
    if not concept_candidates:
        return _CurrentFactSelection(None, "MISSING_INPUT")

    filing_period_end = _normalize_date(filing.period_end)
    if not filing_period_end:
        return _CurrentFactSelection(
            None,
            "PERIOD_NOT_SUPPORTED",
            _bounded_diagnostic_facts(concept_candidates),
        )
    period_candidates = [
        fact
        for fact in concept_candidates
        if _fact_period_end(fact) == filing_period_end
    ]
    if not period_candidates:
        return _CurrentFactSelection(
            None,
            "PERIOD_NOT_SUPPORTED",
            _bounded_diagnostic_facts(concept_candidates),
        )

    filing_entity = _normalized_entity_identifier(filing.company_number)
    if filing_entity:
        entity_candidates = [
            fact
            for fact in period_candidates
            if _normalized_entity_identifier(fact.get("entity_identifier"))
            == filing_entity
        ]
        if not entity_candidates:
            return _CurrentFactSelection(
                None,
                "AMBIGUOUS_CONTEXT",
                _bounded_diagnostic_facts(period_candidates),
            )
    else:
        entity_candidates = [
            fact for fact in period_candidates if fact.get("entity_identifier")
        ]
        entity_ids = {
            _normalized_entity_identifier(fact.get("entity_identifier"))
            for fact in entity_candidates
        }
        if len(entity_ids) != 1:
            return _CurrentFactSelection(
                None,
                "AMBIGUOUS_CONTEXT",
                _bounded_diagnostic_facts(period_candidates),
            )

    # The first UK baseline accepts default filing-entity contexts. The sole narrow
    # exception is a generic Creditors fact explicitly dimensioned as current or
    # within one year. Group/company, scenario, revised and unknown dimensions are
    # never inferred or mixed.
    direct_default_candidates = [
        fact
        for fact in entity_candidates
        if fact.get("concept_local") != "creditors" and _is_default_context(fact)
    ]
    if direct_default_candidates:
        eligible_context_candidates = direct_default_candidates
    else:
        eligible_context_candidates = [
            fact
            for fact in entity_candidates
            if (fact.get("concept_local") != "creditors" and _is_default_context(fact))
            or _is_allowlisted_current_creditor_context(fact)
        ]
    if not eligible_context_candidates:
        if entity_candidates and all(
            _is_known_noncurrent_creditor_context(fact) for fact in entity_candidates
        ):
            return _CurrentFactSelection(None, "MISSING_INPUT")
        return _CurrentFactSelection(
            None,
            "AMBIGUOUS_CONTEXT",
            _bounded_diagnostic_facts(entity_candidates),
        )

    numeric_candidates = [
        fact
        for fact in eligible_context_candidates
        if fact.get("value_num_exact") is not None
        and not fact.get("is_nil")
        and fact.get("continuation_valid", True)
        and fact.get("transform_supported", True)
    ]
    if not numeric_candidates:
        return _CurrentFactSelection(
            None,
            "MISSING_INPUT",
            _bounded_diagnostic_facts(eligible_context_candidates),
        )
    if len(numeric_candidates) != len(eligible_context_candidates):
        return _CurrentFactSelection(
            None,
            "MISSING_INPUT",
            _bounded_diagnostic_facts(eligible_context_candidates),
        )

    distinct_values = {
        (str(fact["value_num_exact"]), str(fact.get("unit_canon") or ""))
        for fact in numeric_candidates
    }
    if len(distinct_values) != 1:
        return _CurrentFactSelection(
            None,
            "AMBIGUOUS_CONTEXT",
            _bounded_diagnostic_facts(numeric_candidates),
        )
    numeric_candidates.sort(key=lambda fact: int(fact["ordinal"]))
    return _CurrentFactSelection(numeric_candidates[0], None, (numeric_candidates[0],))


def _current_fact(
    filing: ParsedCompaniesHouseFiling, concepts: set[str]
) -> Mapping[str, Any] | None:
    """Compatibility wrapper for callers that only need the selected fact."""

    return _select_current_fact(filing, concepts).fact


def _quant_task(
    filing: ParsedCompaniesHouseFiling,
    *,
    metric_spec_id: str,
    operation: str,
    question: str,
    left_name: str,
    left_fact: Mapping[str, Any] | None,
    right_name: str,
    right_fact: Mapping[str, Any] | None,
    left_refusal_code: str | None = None,
    right_refusal_code: str | None = None,
    left_evidence_facts: Sequence[Mapping[str, Any]] = (),
    right_evidence_facts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    period_key = f"ASOF_{re.sub(r'[^0-9]', '', filing.period_end or '') or 'UNKNOWN'}"
    task_id = _stable_digest("ch-quant", filing.filing_id, metric_spec_id, period_key)
    available: list[tuple[str, Mapping[str, Any]]] = []
    for input_name, selected_fact, diagnostic_facts in (
        (left_name, left_fact, left_evidence_facts),
        (right_name, right_fact, right_evidence_facts),
    ):
        facts = tuple(diagnostic_facts) or (
            (selected_fact,) if selected_fact is not None else ()
        )
        seen_evidence_ids: set[str] = set()
        for fact in facts:
            evidence_id = str(fact["fact_evidence_id"])
            if evidence_id not in seen_evidence_ids:
                seen_evidence_ids.add(evidence_id)
                available.append((input_name, fact))
    canonical_inputs = []
    evidence_items = []
    for input_name, fact in available:
        raw_value = fact.get("value_num_exact")
        value = str(raw_value) if raw_value is not None else None
        if (
            value is not None
            and input_name == right_name
            and fact["concept_local"] in _CURRENT_LIABILITY_CONCEPTS
        ):
            value = _decimal_text(abs(Decimal(value)))
        unit_canon = fact.get("unit_canon")
        fact_period_key = _fact_period_key(fact)
        canonical_inputs.append(
            {
                "input_name": input_name,
                "concept_norm": fact["concept"],
                "period_key": fact_period_key,
                "unit_canon": unit_canon,
                "value_num_exact": value,
                "fact_evidence_ids": [fact["fact_evidence_id"]],
                "derived": False,
            }
        )
        evidence_metadata: dict[str, Any] = {
            "input_name": input_name,
            "concept": fact["concept"],
            "source_file": filing.source_name,
            "context_id": fact.get("context_id"),
            "dimensions": fact.get("dimensions") or {},
        }
        if fact.get("entity_identifier"):
            evidence_metadata["entity_id"] = fact["entity_identifier"]
        fact_annotations: list[str] = []
        if fact.get("transform"):
            fact_annotations.append(f"IX_FORMAT:{fact['transform']}")
        if fact.get("is_nil"):
            fact_annotations.append("XSI_NIL")
        if not fact.get("continuation_valid", True):
            fact_annotations.append("CONTINUATION_REFERENCE_INVALID")
        if fact_annotations:
            evidence_metadata["transformation"] = ";".join(fact_annotations)
        if _is_allowlisted_current_creditor_context(fact):
            evidence_metadata["scenario"] = "CURRENT_CREDITOR_ALLOWLIST"
        elif fact.get("has_scenario"):
            evidence_metadata["scenario"] = "XBRL_SCENARIO_PRESENT"
        elif fact.get("concept_local") == "creditors" and not fact.get("dimensions"):
            evidence_metadata["scenario"] = "GENERIC_CREDITOR_DEFAULT_CONTEXT"
        evidence_items.append(
            {
                "evidence_id": fact["fact_evidence_id"],
                "filing_id": filing.filing_id,
                "period_key": fact_period_key,
                "unit": unit_canon,
                "value": value,
                "content": f"{fact['concept']}: {fact['value_text']}",
                "source_system": "UK_COMPANIES_HOUSE",
                "metadata": evidence_metadata,
            }
        )

    if left_fact is None or right_fact is None:
        status = "REFUSAL"
        value = None
        unit = None
        refusal_code = next(
            (
                code
                for code in (left_refusal_code, right_refusal_code)
                if code in {"AMBIGUOUS_CONTEXT", "PERIOD_NOT_SUPPORTED"}
            ),
            "MISSING_INPUT",
        )
        evidence_ids: list[str] = []
    else:
        left = Decimal(
            next(
                item["value_num_exact"]
                for item in canonical_inputs
                if item["input_name"] == left_name
            )
        )
        right = Decimal(
            next(
                item["value_num_exact"]
                for item in canonical_inputs
                if item["input_name"] == right_name
            )
        )
        left_unit = str(left_fact.get("unit_canon") or "")
        right_unit = str(right_fact.get("unit_canon") or "")
        monetary_units_valid = (
            left_fact.get("unit_type") == "currency"
            and right_fact.get("unit_type") == "currency"
            and bool(left_unit)
            and left_unit == right_unit
        )
        if not monetary_units_valid:
            status = "REFUSAL"
            value = None
            unit = None
            refusal_code = "INCOMPATIBLE_UNITS"
            evidence_ids = []
        elif operation == "divide" and right == 0:
            status = "REFUSAL"
            value = None
            unit = None
            refusal_code = "DIVISION_BY_ZERO"
            evidence_ids = []
        else:
            status = "OK"
            result = left / right if operation == "divide" else left - right
            value = _decimal_text(result)
            unit = "pure" if operation == "divide" else left_unit
            refusal_code = None
            evidence_ids = [
                str(left_fact["fact_evidence_id"]),
                str(right_fact["fact_evidence_id"]),
            ]

    target_answer = {
        "structured_answer_version": "v1",
        "task_id": task_id,
        "metric_spec_id": metric_spec_id,
        "filing_id": filing.filing_id,
        "status": status,
        "value": value,
        "unit": unit,
        "period_key": period_key,
        "evidence_ids": evidence_ids,
        "refusal_code": refusal_code,
    }
    return {
        "task_spec_version": "v1",
        "task_id": task_id,
        "task_type": "quant_metric",
        "question": question,
        "source_answer_id": _stable_digest("ch-answer", task_id),
        "metric_spec_id": metric_spec_id,
        "metric_kind": "ratio" if operation == "divide" else "difference",
        "executor_op": operation,
        "filing_id": filing.filing_id,
        "ticker": filing.company_number or filing.filing_id,
        "filing_metadata": filing.manifest_row(),
        "period": {
            "period_type": "ASOF",
            "period_key": period_key,
            "period_end": filing.period_end,
        },
        "target_status": status,
        "target_answer": target_answer,
        "required_inputs": [{"name": left_name}, {"name": right_name}],
        "canonical_inputs": canonical_inputs,
        "evidence_items": evidence_items,
        "distractors": [],
        "negative_type": {
            "MISSING_INPUT": "missing_input",
            "AMBIGUOUS_CONTEXT": "ambiguous_context",
            "PERIOD_NOT_SUPPORTED": "period_not_supported",
            "INCOMPATIBLE_UNITS": "incompatible_units",
        }.get(refusal_code),
        "evidence_requirements": {
            "require_evidence_ids": status == "OK",
            "min_evidence_ids": 2 if status == "OK" else 0,
        },
        "output_schema": {"schema_id": "structured_answer.v1"},
        "refusal_policy": {
            "allowed_codes": [
                "MISSING_INPUT",
                "DIVISION_BY_ZERO",
                "AMBIGUOUS_CONTEXT",
                "INCOMPATIBLE_UNITS",
                "PERIOD_NOT_SUPPORTED",
            ],
            "default_code": refusal_code,
        },
        "jurisdiction": "UK",
        "source_system": "UK_COMPANIES_HOUSE",
        "reporting_framework": filing.reporting_framework,
        "standards_version": "ISA_UK_METADATA_ONLY",
    }


def build_companies_house_quant_tasks(
    filing: ParsedCompaniesHouseFiling,
) -> list[dict[str, Any]]:
    assets = _select_current_fact(filing, _CURRENT_ASSET_CONCEPTS)
    liabilities = _select_current_fact(filing, _CURRENT_LIABILITY_CONCEPTS)
    period = filing.period_end or "the reported period"
    return [
        _quant_task(
            filing,
            metric_spec_id="current_ratio",
            operation="divide",
            question=f"What is the current ratio for {filing.company_name or filing.company_number or 'the company'} at {period}?",
            left_name="assets_current",
            left_fact=assets.fact,
            left_refusal_code=assets.refusal_code,
            left_evidence_facts=assets.evidence_facts,
            right_name="liabilities_current",
            right_fact=liabilities.fact,
            right_refusal_code=liabilities.refusal_code,
            right_evidence_facts=liabilities.evidence_facts,
        ),
        _quant_task(
            filing,
            metric_spec_id="working_capital",
            operation="subtract",
            question=f"What is working capital for {filing.company_name or filing.company_number or 'the company'} at {period}?",
            left_name="assets_current",
            left_fact=assets.fact,
            left_refusal_code=assets.refusal_code,
            left_evidence_facts=assets.evidence_facts,
            right_name="liabilities_current",
            right_fact=liabilities.fact,
            right_refusal_code=liabilities.refusal_code,
            right_evidence_facts=liabilities.evidence_facts,
        ),
    ]


def _find_chunk(
    filing: ParsedCompaniesHouseFiling, phrases: Sequence[str]
) -> Mapping[str, Any] | None:
    for chunk in filing.chunks:
        lowered = str(chunk["text"]).lower().replace("’", "'").replace("‘", "'")
        if any(phrase in lowered for phrase in phrases):
            return chunk
    return None


def _bm25_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _rank_narrative_chunks(
    filing: ParsedCompaniesHouseFiling,
    query: str,
    *,
    candidate_k: int = 15,
    top_k: int = 5,
) -> list[Mapping[str, Any]]:
    if not filing.chunks:
        return []
    query_terms = Counter(_bm25_tokens(query))
    documents = [_bm25_tokens(str(chunk["text"])) for chunk in filing.chunks]
    document_frequencies = {
        term: sum(term in document for document in documents) for term in query_terms
    }
    average_length = sum(len(document) for document in documents) / len(documents)
    k1 = 1.5
    b = 0.75

    scored: list[tuple[float, Mapping[str, Any]]] = []
    for chunk, document in zip(filing.chunks, documents):
        frequencies = Counter(document)
        score = 0.0
        for term, query_frequency in query_terms.items():
            frequency = frequencies[term]
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1
                + (len(documents) - document_frequencies[term] + 0.5)
                / (document_frequencies[term] + 0.5)
            )
            length_normalizer = 1 - b + b * len(document) / (average_length or 1)
            score += (
                query_frequency
                * inverse_document_frequency
                * frequency
                * (k1 + 1)
                / (frequency + k1 * length_normalizer)
            )
        scored.append((score, chunk))
    scored.sort(
        key=lambda item: (
            -item[0],
            int(item[1]["chunk_index"]),
            str(item[1]["evidence_id"]),
        )
    )
    candidates = [chunk for _, chunk in scored[:candidate_k]]
    return candidates[:top_k]


def _build_companies_house_narrative_task(
    filing: ParsedCompaniesHouseFiling,
    *,
    slug: str,
    item: str,
    label: str,
    question: str,
    retrieval_query: str,
    answer_phrases: Sequence[str],
) -> dict[str, Any]:
    source_answer_chunk = _find_chunk(filing, answer_phrases)
    ranked_chunks = _rank_narrative_chunks(filing, retrieval_query)
    ranked_ids = {str(chunk["evidence_id"]) for chunk in ranked_chunks}
    answer_chunk = (
        source_answer_chunk
        if source_answer_chunk is not None
        and str(source_answer_chunk["evidence_id"]) in ranked_ids
        else None
    )
    task_id = _stable_digest("ch-narrative", filing.filing_id, slug)
    return {
        "narrative_task_spec_version": "v1",
        "narrative_task_schema_id": "narrative_task_spec.v1",
        "task_id": task_id,
        "task_type": "narrative_citation",
        "filing_id": filing.filing_id,
        "ticker": filing.company_number or filing.filing_id,
        "form_type": filing.account_type,
        "period_key": filing.period_end,
        "item": item,
        "heading": None,
        "subheading": None,
        "heading_path": None,
        "question": question,
        "retrieval_query": retrieval_query,
        "label": label,
        "scope_type": "filing",
        "scope_key": filing.filing_id,
        "answerability": "ANSWERABLE" if answer_chunk is not None else "UNANSWERABLE",
        "expected_chunk_ids": (
            [answer_chunk["chunk_evidence_id"]] if answer_chunk is not None else []
        ),
        "extractive_answer": answer_chunk["text"] if answer_chunk is not None else None,
        "refusal_code": None if answer_chunk is not None else "NARRATIVE_NOT_SUPPORTED",
        "negative_type": None if answer_chunk is not None else "unsupported_attribute",
        "donor_task_id": None,
        "citation_policy": {
            "require_chunk_evidence_ids": answer_chunk is not None,
            "retrieval_method": "bm25",
            "top_k": 5,
            "candidate_k": 15,
        },
        "jurisdiction": "UK",
        "source_system": "UK_COMPANIES_HOUSE",
        "reporting_framework": filing.reporting_framework,
        "standards_version": "ISA_UK_METADATA_ONLY",
        "evidence_items": [
            {
                "evidence_id": chunk["evidence_id"],
                "filing_id": filing.filing_id,
                "period_key": filing.period_end,
                "content": chunk["text"],
                "rank": index + 1,
                "source_system": "UK_COMPANIES_HOUSE",
                "metadata": {"source_file": filing.source_name},
            }
            for index, chunk in enumerate(ranked_chunks)
        ],
    }


def build_companies_house_narrative_tasks(
    filing: ParsedCompaniesHouseFiling,
) -> list[dict[str, Any]]:
    return [
        _build_companies_house_narrative_task(
            filing,
            slug="audit-status",
            item="audit_status",
            label="audit_status",
            question="What does the filing state about whether these accounts were audited or exempt from audit?",
            retrieval_query="audit auditor exemption unaudited section 477",
            answer_phrases=(
                "exemption from audit",
                "exempt from the requirements relating to the audit",
                "independent auditor's report",
                "independent auditors' report",
                "unaudited",
            ),
        ),
        _build_companies_house_narrative_task(
            filing,
            slug="audit-opinion",
            item="audit_opinion",
            label="audit_opinion_language",
            question="What audit opinion language does the auditor use in this filing?",
            retrieval_query="auditor opinion true fair qualified adverse disclaimer financial statements",
            answer_phrases=(
                "in our opinion",
                "qualified opinion",
                "adverse opinion",
                "disclaimer of opinion",
            ),
        ),
        _build_companies_house_narrative_task(
            filing,
            slug="kam",
            item="key_audit_matter",
            label="key_audit_matter",
            question="Which key or critical audit matter did the auditor identify for this filing?",
            retrieval_query="key audit matter critical audit matter auditor",
            answer_phrases=("key audit matter", "critical audit matter"),
        ),
    ]


@dataclass(frozen=True)
class _SelectedZipMember:
    info: zipfile.ZipInfo
    member_index: int
    selection_rank: int
    member_id: str
    duplicate_name: bool

    def manifest_fields(self) -> dict[str, Any]:
        unix_mode = (
            (self.info.external_attr >> 16) & 0xFFFF
            if self.info.create_system == 3
            else None
        )
        return {
            "archive_member_id": self.member_id,
            "archive_member_index": self.member_index,
            "selection_rank": self.selection_rank,
            "source_name": self.info.filename,
            "compressed_size_bytes": self.info.compress_size,
            "uncompressed_size_bytes": self.info.file_size,
            "compression_method": self.info.compress_type,
            "compression_ratio": _zip_compression_ratio_text(self.info),
            "crc32": f"{self.info.CRC & 0xFFFFFFFF:08x}",
            "flag_bits": self.info.flag_bits,
            "unix_mode": f"{unix_mode:06o}" if unix_mode is not None else None,
        }


@dataclass(frozen=True)
class _CompaniesHouseParseFailure:
    failure_id: str
    archive_sha256: str
    member: _SelectedZipMember
    failure_stage: str
    failure_code: str
    error_type: str
    detail: str
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.failure_stage not in _PARSE_FAILURE_STAGES:
            raise ValueError(
                f"Unknown Companies House failure stage: {self.failure_stage}"
            )
        if self.failure_code not in _PARSE_FAILURE_CODES:
            raise ValueError(
                f"Unknown Companies House failure code: {self.failure_code}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.archive_sha256):
            raise ValueError("archive_sha256 must be a lowercase SHA-256 digest")
        if self.source_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.source_sha256
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")

    def manifest_row(self) -> dict[str, Any]:
        return {
            "parse_failure_version": COMPANIES_HOUSE_PARSE_FAILURE_VERSION,
            "failure_id": self.failure_id,
            "bulk_archive_sha256": self.archive_sha256,
            **self.member.manifest_fields(),
            "failure_stage": self.failure_stage,
            "failure_code": self.failure_code,
            "error_type": self.error_type,
            "detail": self.detail,
            "detail_sha256": _sha256_bytes(self.detail.encode("utf-8")),
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class _BulkMemberOutcome:
    member: _SelectedZipMember
    filing: ParsedCompaniesHouseFiling | None = None
    failure: _CompaniesHouseParseFailure | None = None

    def __post_init__(self) -> None:
        if (self.filing is None) == (self.failure is None):
            raise ValueError(
                "A bulk member outcome must contain one filing or one failure"
            )


@dataclass(frozen=True)
class _BulkArchivePlan:
    archive_sha256: str
    archive_size_bytes: int
    archive_member_count: int
    archive_uncompressed_bytes: int
    xhtml_candidate_count: int
    selected_members: tuple[_SelectedZipMember, ...]


def _zip_compression_ratio_text(info: zipfile.ZipInfo) -> str:
    if info.file_size <= 0:
        return "0"
    if info.compress_size <= 0:
        return "INFINITE"
    return _decimal_text(Decimal(info.file_size) / Decimal(info.compress_size))


def _clean_failure_detail(value: str, *, fallback: str) -> str:
    detail = _clean_text(value) or fallback
    return detail[:512]


def _make_parse_failure(
    *,
    archive_sha256: str,
    member: _SelectedZipMember,
    failure_stage: str,
    failure_code: str,
    error_type: str,
    detail: str,
    source_sha256: str | None = None,
) -> _CompaniesHouseParseFailure:
    cleaned_detail = _clean_failure_detail(detail, fallback=failure_code)
    failure_id = f"ch-failure::{_stable_digest(archive_sha256, member.member_id, failure_stage, failure_code)}"
    return _CompaniesHouseParseFailure(
        failure_id=failure_id,
        archive_sha256=archive_sha256,
        member=member,
        failure_stage=failure_stage,
        failure_code=failure_code,
        error_type=error_type,
        detail=cleaned_detail,
        source_sha256=source_sha256,
    )


def _unsafe_zip_member_name_reason(
    name: str, *, limits: CompaniesHouseZipLimits
) -> str | None:
    if not name:
        return "member name is empty"
    if len(name) > limits.max_member_name_chars:
        return "member name exceeds the configured character limit"
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        return "member name contains a control character"
    if "\\" in name:
        return "member name contains a backslash"
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return "member name is absolute"
    parts = name.split("/")
    if len(parts) > limits.max_path_depth:
        return "member path exceeds the configured depth limit"
    if any(part in {"", ".", ".."} for part in parts):
        return "member name contains an empty, dot, or parent segment"
    if any(":" in part for part in parts):
        return "member name contains a colon"
    return None


def _validate_selected_zip_member(
    member: _SelectedZipMember,
    *,
    archive_size_bytes: int,
    limits: CompaniesHouseZipLimits,
) -> tuple[str, str] | None:
    info = member.info
    unsafe_reason = _unsafe_zip_member_name_reason(info.filename, limits=limits)
    if unsafe_reason is not None:
        return "UNSAFE_MEMBER_NAME", unsafe_reason
    if member.duplicate_name:
        return (
            "DUPLICATE_MEMBER_NAME",
            "archive contains this member name more than once",
        )
    if info.flag_bits & 0x41:
        return "ENCRYPTED_MEMBER", "encrypted ZIP members are not accepted"
    if info.compress_type not in _SUPPORTED_ZIP_COMPRESSION_METHODS:
        return (
            "UNSUPPORTED_COMPRESSION",
            f"ZIP compression method {info.compress_type} is not allowed",
        )
    if info.create_system == 3:
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(unix_mode)
        if file_type not in {0, stat.S_IFREG}:
            return "NON_REGULAR_MEMBER", "ZIP member is not a regular file"
    metadata_values = (
        info.file_size,
        info.compress_size,
        info.header_offset,
        info.CRC,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in metadata_values
    ):
        return "INVALID_MEMBER_METADATA", "ZIP member metadata is not integral"
    if (
        info.file_size < 0
        or info.compress_size < 0
        or info.header_offset < 0
        or info.header_offset >= archive_size_bytes
        or info.compress_size > archive_size_bytes
        or not 0 <= info.CRC <= 0xFFFFFFFF
    ):
        return "INVALID_MEMBER_METADATA", "ZIP member metadata is outside valid bounds"
    if info.file_size > limits.max_member_uncompressed_bytes:
        return (
            "MEMBER_SIZE_LIMIT_EXCEEDED",
            "ZIP member exceeds the configured uncompressed-size limit",
        )
    if info.file_size > 0 and (
        info.compress_size == 0
        or info.file_size > limits.max_compression_ratio * info.compress_size
    ):
        return (
            "COMPRESSION_RATIO_LIMIT_EXCEEDED",
            "ZIP member exceeds the configured compression-ratio limit",
        )
    return None


def _plan_companies_house_archive(
    archive: zipfile.ZipFile,
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    seed: int,
    max_filings: int | None,
    limits: CompaniesHouseZipLimits,
) -> _BulkArchivePlan:
    infos = archive.infolist()
    if len(infos) > limits.max_archive_members:
        raise CompaniesHouseArchiveValidationError(
            "ARCHIVE_MEMBER_COUNT_LIMIT_EXCEEDED",
            f"archive has {len(infos)} members; limit is {limits.max_archive_members}",
        )
    archive_uncompressed_bytes = sum(max(info.file_size, 0) for info in infos)
    if archive_uncompressed_bytes > limits.max_total_uncompressed_bytes:
        raise CompaniesHouseArchiveValidationError(
            "ARCHIVE_UNCOMPRESSED_SIZE_LIMIT_EXCEEDED",
            "archive exceeds the configured total uncompressed-size limit",
        )

    indexed_candidates = [
        (member_index, info)
        for member_index, info in enumerate(infos)
        if not info.is_dir()
        and info.filename.lower().endswith(_SUPPORTED_XHTML_SUFFIXES)
    ]
    duplicate_names = {
        name
        for name, count in Counter(
            info.filename for _, info in indexed_candidates
        ).items()
        if count > 1
    }
    indexed_candidates.sort(
        key=lambda item: (
            _stable_digest(seed, item[1].filename),
            item[1].filename,
            item[1].header_offset,
            item[0],
        )
    )
    selected_candidates = (
        indexed_candidates if max_filings is None else indexed_candidates[:max_filings]
    )
    selected_members = tuple(
        _SelectedZipMember(
            info=info,
            member_index=member_index,
            selection_rank=selection_rank,
            member_id=(
                "ch-member::"
                + _stable_digest(
                    archive_sha256,
                    member_index,
                    info.filename,
                    info.header_offset,
                    info.CRC,
                    info.file_size,
                    info.compress_size,
                )
            ),
            duplicate_name=info.filename in duplicate_names,
        )
        for selection_rank, (member_index, info) in enumerate(selected_candidates)
    )
    return _BulkArchivePlan(
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
        archive_member_count=len(infos),
        archive_uncompressed_bytes=archive_uncompressed_bytes,
        xhtml_candidate_count=len(indexed_candidates),
        selected_members=selected_members,
    )


def _iter_companies_house_bulk_outcomes(
    bulk_zip: str | Path,
    *,
    seed: int,
    max_filings: int | None,
    zip_limits: CompaniesHouseZipLimits,
    plan_sink: list[_BulkArchivePlan] | None = None,
) -> Iterator[_BulkMemberOutcome]:
    path = Path(bulk_zip)
    if not path.is_file():
        raise FileNotFoundError(f"Companies House bulk archive not found: {path}")
    if max_filings is not None and max_filings <= 0:
        raise ValueError("max_filings must be positive")
    archive_sha256 = _sha256_file(path)
    archive_size_bytes = path.stat().st_size
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise CompaniesHouseArchiveValidationError(
            "INVALID_ZIP_ARCHIVE",
            _clean_failure_detail(str(exc), fallback="invalid ZIP"),
        ) from exc

    with archive:
        plan = _plan_companies_house_archive(
            archive,
            archive_sha256=archive_sha256,
            archive_size_bytes=archive_size_bytes,
            seed=seed,
            max_filings=max_filings,
            limits=zip_limits,
        )
        if plan_sink is not None:
            plan_sink.append(plan)

        policy_failures: dict[int, _CompaniesHouseParseFailure] = {}
        for member in plan.selected_members:
            rejection = _validate_selected_zip_member(
                member,
                archive_size_bytes=archive_size_bytes,
                limits=zip_limits,
            )
            if rejection is None:
                continue
            code, detail = rejection
            policy_failures[member.selection_rank] = _make_parse_failure(
                archive_sha256=archive_sha256,
                member=member,
                failure_stage="ZIP_INFO",
                failure_code=code,
                error_type="ZIP_POLICY",
                detail=detail,
            )

        for member in plan.selected_members:
            policy_failure = policy_failures.get(member.selection_rank)
            if policy_failure is not None:
                yield _BulkMemberOutcome(member=member, failure=policy_failure)
                continue
            try:
                payload = archive.read(member.info)
            except (
                EOFError,
                KeyError,
                NotImplementedError,
                OSError,
                RuntimeError,
                zipfile.BadZipFile,
                zlib.error,
            ) as exc:
                yield _BulkMemberOutcome(
                    member=member,
                    failure=_make_parse_failure(
                        archive_sha256=archive_sha256,
                        member=member,
                        failure_stage="READ",
                        failure_code="ARCHIVE_MEMBER_READ_ERROR",
                        error_type=type(exc).__name__,
                        detail=str(exc),
                    ),
                )
                continue
            source_sha256 = _sha256_bytes(payload)
            if len(payload) != member.info.file_size:
                yield _BulkMemberOutcome(
                    member=member,
                    failure=_make_parse_failure(
                        archive_sha256=archive_sha256,
                        member=member,
                        failure_stage="READ",
                        failure_code="MEMBER_SIZE_MISMATCH",
                        error_type="ZIP_SIZE_MISMATCH",
                        detail=(
                            f"read {len(payload)} bytes; ZIP metadata declares "
                            f"{member.info.file_size}"
                        ),
                        source_sha256=source_sha256,
                    ),
                )
                continue
            try:
                filing = parse_companies_house_xhtml(
                    payload, source_name=member.info.filename
                )
            except (ValueError, etree.Error) as exc:
                yield _BulkMemberOutcome(
                    member=member,
                    failure=_make_parse_failure(
                        archive_sha256=archive_sha256,
                        member=member,
                        failure_stage="PARSE",
                        failure_code="XHTML_PARSE_ERROR",
                        error_type=type(exc).__name__,
                        detail=str(exc),
                        source_sha256=source_sha256,
                    ),
                )
                continue
            yield _BulkMemberOutcome(member=member, filing=filing)


def iter_companies_house_bulk(
    bulk_zip: str | Path,
    *,
    seed: int = 20260821,
    max_filings: int | None = None,
    zip_limits: CompaniesHouseZipLimits | None = None,
) -> Iterator[ParsedCompaniesHouseFiling]:
    """Yield selected parseable filings after applying bounded ZIP validation.

    Selection happens before validation. A rejected selected member is not replaced
    by a later member, so filtering cannot bias seeded corpus membership.
    """

    limits = zip_limits or CompaniesHouseZipLimits()
    for outcome in _iter_companies_house_bulk_outcomes(
        bulk_zip,
        seed=seed,
        max_filings=max_filings,
        zip_limits=limits,
    ):
        if outcome.filing is not None:
            yield outcome.filing


def build_companies_house_corpus(
    bulk_zip: str | Path,
    output_dir: str | Path,
    *,
    snapshot_date: str = "2026-01-31",
    seed: int = 20260821,
    max_filings: int | None = None,
    zip_limits: CompaniesHouseZipLimits | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite corpus directory: {output}")
    limits = zip_limits or CompaniesHouseZipLimits()
    plans: list[_BulkArchivePlan] = []
    outcomes = list(
        _iter_companies_house_bulk_outcomes(
            bulk_zip,
            seed=seed,
            max_filings=max_filings,
            zip_limits=limits,
            plan_sink=plans,
        )
    )
    if len(plans) != 1:
        raise RuntimeError(
            "Companies House archive planning did not complete exactly once"
        )
    plan = plans[0]
    filings = [outcome.filing for outcome in outcomes if outcome.filing is not None]
    failures = [outcome.failure for outcome in outcomes if outcome.failure is not None]
    if len(outcomes) != len(filings) + len(failures):
        raise RuntimeError(
            "Companies House source membership accounting is inconsistent"
        )
    source_rows = []
    for outcome in outcomes:
        common = {
            "source_manifest_version": COMPANIES_HOUSE_SOURCE_MANIFEST_VERSION,
            "source_system": "UK_COMPANIES_HOUSE",
            "bulk_archive_sha256": plan.archive_sha256,
            **outcome.member.manifest_fields(),
        }
        if outcome.filing is not None:
            source_rows.append(
                {
                    **common,
                    "source_status": "PARSED",
                    "parse_failure_id": None,
                    **outcome.filing.manifest_row(),
                }
            )
        else:
            assert outcome.failure is not None
            source_rows.append(
                {
                    **common,
                    "source_status": "FAILED",
                    "source_sha256": outcome.failure.source_sha256,
                    "filing_id": None,
                    "parse_failure_id": outcome.failure.failure_id,
                    "parse_failure_code": outcome.failure.failure_code,
                }
            )
    failure_rows = [failure.manifest_row() for failure in failures]
    failure_counts = dict(
        sorted(Counter(row["failure_code"] for row in failure_rows).items())
    )
    quant_rows = [
        task for filing in filings for task in build_companies_house_quant_tasks(filing)
    ]
    narrative_rows = [
        task
        for filing in filings
        for task in build_companies_house_narrative_tasks(filing)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        _write_jsonl(temporary / "source_manifest.jsonl", source_rows)
        _write_jsonl(temporary / "parse_failures.jsonl", failure_rows)
        _write_jsonl(temporary / "task_specs_quant.jsonl", quant_rows)
        _write_jsonl(temporary / "narrative_task_specs.jsonl", narrative_rows)
        manifest = {
            "corpus_version": COMPANIES_HOUSE_CORPUS_VERSION,
            "corpus_id": "companies_house_bulk_2026-01",
            "snapshot_date": snapshot_date,
            "source_system": "UK_COMPANIES_HOUSE",
            "jurisdiction": "UK",
            "bulk_archive": str(Path(bulk_zip).resolve()),
            "bulk_archive_sha256": plan.archive_sha256,
            "bulk_archive_size_bytes": plan.archive_size_bytes,
            "bulk_archive_member_count": plan.archive_member_count,
            "bulk_archive_uncompressed_bytes": plan.archive_uncompressed_bytes,
            "xhtml_candidate_member_count": plan.xhtml_candidate_count,
            "selected_member_count": len(plan.selected_members),
            "source_manifest_count": len(source_rows),
            "parse_failure_count": len(failure_rows),
            "parse_failure_counts": failure_counts,
            "parse_failure_version": COMPANIES_HOUSE_PARSE_FAILURE_VERSION,
            "source_manifest_version": COMPANIES_HOUSE_SOURCE_MANIFEST_VERSION,
            "zip_limits": limits.manifest_row(),
            "max_filings": max_filings,
            "seed": seed,
            "build_status": ("READY" if filings else "FAILED_NO_PARSEABLE_FILINGS"),
            "filing_count": len(filings),
            "quant_task_count": len(quant_rows),
            "narrative_task_count": len(narrative_rows),
            "files": {
                name: _sha256_file(temporary / name)
                for name in (
                    "source_manifest.jsonl",
                    "parse_failures.jsonl",
                    "task_specs_quant.jsonl",
                    "narrative_task_specs.jsonl",
                )
            },
        }
        _write_json(temporary / "corpus_manifest.json", manifest)
        assert_no_secrets([temporary])
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if not filings:
        raise ValueError(
            "No parseable XHTML filings were found in the Companies House bulk archive; "
            f"selected={len(outcomes)}, failures={failure_counts}, "
            f"failure_ledger={output / 'parse_failures.jsonl'}"
        )
    return manifest


def build_companies_house_vlm_diagnostic(
    pair_manifest_csv: str | Path,
    fact_audit_csv: str | Path,
    output_dir: str | Path,
    *,
    expected_pair_count: int | None = None,
    expected_fact_count: int | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic directory: {output}")
    with Path(pair_manifest_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    with Path(fact_audit_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        facts = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("usable_as_benchmark", "")).lower() == "yes"
        ]
    facts_by_pair: dict[str, list[dict[str, str]]] = {}
    for fact in facts:
        facts_by_pair.setdefault(str(fact["pair_id"]), []).append(dict(fact))

    cases = []
    gold = []
    for pair in sorted(pairs, key=lambda row: str(row["pair_id"])):
        pair_id = str(pair["pair_id"])
        pair_facts = facts_by_pair.get(pair_id, [])
        if not pair_facts:
            continue
        task_id = _stable_digest("ch-vlm", pair_id, pair.get("document_id"))
        cases.append(
            {
                "diagnostic_version": COMPANIES_HOUSE_VLM_VERSION,
                "task_id": task_id,
                "pair_id": pair_id,
                "company_number": pair.get("company_number"),
                "filing_id": pair.get("transaction_id"),
                "pdf_path": pair.get("pdf_path"),
                "page_count": int(pair.get("pdf_page_count") or 0),
                "requested_facts": [row["fact_label"] for row in pair_facts],
                "required_output": {
                    "value": True,
                    "page": True,
                    "bbox": True,
                    "evidence_text": True,
                },
                "bbox_gold_available": False,
            }
        )
        gold.append(
            {
                "task_id": task_id,
                "pair_id": pair_id,
                "facts": [
                    {
                        "fact_label": row["fact_label"],
                        "concept": row["ixbrl_tag_concept"],
                        "value": row["ixbrl_value"],
                        "unit": row["unit"] or None,
                        "period_context": row["period_context"],
                        "page": int(re.sub(r"[^0-9]", "", row["pdf_page"]) or 0),
                        "evidence_text": row["pdf_evidence"],
                        "bbox": None,
                    }
                    for row in pair_facts
                ],
            }
        )
    if expected_pair_count is not None and len(cases) != expected_pair_count:
        raise ValueError(
            f"Expected {expected_pair_count} VLM filing cases, materialized {len(cases)}"
        )
    gold_fact_count = sum(len(row["facts"]) for row in gold)
    if expected_fact_count is not None and gold_fact_count != expected_fact_count:
        raise ValueError(
            f"Expected {expected_fact_count} VLM gold facts, materialized {gold_fact_count}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        _write_jsonl(temporary / "cases.jsonl", cases)
        _write_jsonl(temporary / "gold.jsonl", gold)
        manifest = {
            "diagnostic_version": COMPANIES_HOUSE_VLM_VERSION,
            "case_count": len(cases),
            "gold_fact_count": gold_fact_count,
            "bbox_gold_available": False,
            "bbox_policy": "Models must emit normalized in-page boxes; only bounds/schema can be verified until boxes are labelled.",
            "files": {
                "cases.jsonl": _sha256_file(temporary / "cases.jsonl"),
                "gold.jsonl": _sha256_file(temporary / "gold.jsonl"),
            },
        }
        _write_json(temporary / "diagnostic_manifest.json", manifest)
        assert_no_secrets([temporary])
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest
