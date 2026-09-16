from auditops.narrative_tasks import _extractive_answer
from auditops.text_support import (
    is_contiguous_text_supported,
    normalize_support_text,
)


def test_support_normalizes_unicode_and_ascii_whitespace():
    evidence = "Revenue\u00a0is\n\trecognized when control transfers."

    assert normalize_support_text(evidence) == (
        "Revenue is recognized when control transfers."
    )
    assert is_contiguous_text_supported(
        "Revenue is recognized when control transfers.", evidence
    )


def test_support_rejects_noncontiguous_joined_excerpts():
    evidence = "First paragraph.\n\nIntervening disclosure.\n\nSecond paragraph."

    assert not is_contiguous_text_supported(
        "First paragraph. Second paragraph.", evidence
    )


def test_support_rejects_synthetic_ellipsis():
    assert not is_contiguous_text_supported(
        "Revenue is recognized...",
        "Revenue is recognized when control transfers.",
    )


def test_extractive_answer_uses_one_paragraph_without_synthetic_ellipsis():
    first = "Revenue " + "recognition policy " * 30
    answer = _extractive_answer(
        {"text_masked": first + "\n\nThis second paragraph must not be joined."}
    )

    assert answer
    assert "second paragraph" not in answer
    assert not answer.endswith("...")
    assert is_contiguous_text_supported(answer, first)


def test_extractive_answer_preserves_adjacent_uppercase_source_tokens():
    source = "CONSOLIDATED STATEMENTS OF EQUITY\n\nSecond paragraph."

    answer = _extractive_answer({"text_masked": source})

    assert answer == "CONSOLIDATED STATEMENTS OF EQUITY"
    assert is_contiguous_text_supported(answer, source)
