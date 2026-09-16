"""Deterministic exact support checks for extractive narrative claims."""

from __future__ import annotations


def normalize_support_text(text: str) -> str:
    """Collapse Unicode and ASCII whitespace without changing other characters."""

    return " ".join(text.split())


def is_contiguous_text_supported(claim: object, evidence: object) -> bool:
    """Return whether the complete normalized claim occurs in normalized evidence."""

    if not isinstance(claim, str) or not isinstance(evidence, str):
        return False
    normalized_claim = normalize_support_text(claim)
    return bool(normalized_claim) and normalized_claim in normalize_support_text(
        evidence
    )


__all__ = ["is_contiguous_text_supported", "normalize_support_text"]
