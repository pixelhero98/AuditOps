from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from auditops.agent_batch import _execution_version, _runtime_contract_version
from auditops.agent_benchmark_v24 import (
    _HARDENED_R2_SUBTYPE_ORDER,
    _NEGATIVE_CATALOG_R2,
    _acceptable_answer_sets_v24r2,
    _anchor_topic_from_question_v24r2,
    _bind_item8_note_chunks_v24r2,
    _build_negative_narrative,
    _build_positive_narrative,
    _candidate_identity_namespace_v24,
    _expanded_support_window_v24r2,
    _footnote_topic_v24r2,
    _is_auditor_opinion_scope_item_v24r2,
    _narrative_scope_items_v24r2,
    _negative_spec_matches_scope_v24r2,
    _positive_candidate_score,
    _retrieval_query_v24r2,
    _segment_items,
    _semantic_anchors_v24r2,
    _task_source_row,
    validate_semantic_anchor_v24r2,
)
from auditops.agent_provisional_v24p import (
    _record_checks,
    _validate_provisional_review,
    validate_exploratory_authorization_v24p,
    validate_gate_provisional_review_v24p,
)
from auditops.canonical_json import canonical_json_sha256

SPEC_ROOT = Path(__file__).parents[1] / "auditops" / "specs"
CHECKS = {
    "question_quality": True,
    "subtype": True,
    "scope": True,
    "answerability": True,
    "evidence": True,
    "semantic_anchors": True,
    "refusal_validity": True,
}


def test_v24r2_reserves_support_for_constrained_subtypes_first() -> None:
    assert _HARDENED_R2_SUBTYPE_ORDER.index("accounting_policy") < (
        _HARDENED_R2_SUBTYPE_ORDER.index("footnote_note")
    )


def test_v24r2_cam_retrieval_uses_only_bound_raw_supplement() -> None:
    canonical = [{"evidence_id": "canonical"}]
    raw = [{"evidence_id": "raw-cam"}]

    assert (
        _narrative_scope_items_v24r2(
            subtype="critical_audit_matter",
            canonical_segments=canonical,
            raw_cam_segments=raw,
            hardened_r2=True,
        )
        == raw
    )
    assert _narrative_scope_items_v24r2(
        subtype="critical_audit_matter",
        canonical_segments=canonical,
        raw_cam_segments=raw,
        hardened_r2=False,
    ) == [*canonical, *raw]


def test_v24r2_opinion_retrieval_uses_only_item8_auditor_scope() -> None:
    opinion = {
        "evidence_id": "opinion",
        "content": "In our opinion, the financial statements present fairly.",
        "metadata": {"item": "8", "subheading": "Report of Independent Auditor"},
    }
    item15_opinion = copy.deepcopy(opinion)
    item15_opinion["metadata"]["item"] = "15"
    mda = {
        "evidence_id": "mda",
        "content": "Management discusses the financial statements.",
        "metadata": {"item": "7", "subheading": "MD&A"},
    }

    assert _is_auditor_opinion_scope_item_v24r2(opinion) is True
    assert _is_auditor_opinion_scope_item_v24r2(item15_opinion) is True
    assert _is_auditor_opinion_scope_item_v24r2(mda) is False
    assert _narrative_scope_items_v24r2(
        subtype="auditor_report_opinion_language",
        canonical_segments=[mda, opinion],
        raw_cam_segments=[],
        hardened_r2=True,
    ) == [opinion]


def test_v24r2_retrieval_query_is_source_and_gold_independent() -> None:
    query = _retrieval_query_v24r2(
        subtype="critical_audit_matter",
        topic="rebate estimates",
        attribute="underlying facts and estimates",
    )

    assert query == "rebate estimates underlying facts and estimates"
    assert "critical audit matter" not in query
    assert "required to estimate the rebate" not in query

    policy_query = _retrieval_query_v24r2(
        subtype="accounting_policy",
        topic="revenue recognition",
        attribute="recognition",
    )
    assert "customer control" in policy_query

    opinion_query = _retrieval_query_v24r2(
        subtype="auditor_report_opinion_language",
        topic="opinion on the financial statements",
        attribute="fair-presentation conclusion",
    )
    assert "present fairly" in opinion_query
    assert "all material respects" in opinion_query


def _review(*, pass_kind: str = "CRITIC", count: int = 40) -> dict:
    decisions = [
        {
            "task_id": f"task-{index:03d}",
            "decision": "APPROVE",
            "checks": dict(CHECKS),
            "comment": None,
        }
        for index in range(count)
    ]
    material = {
        "provisional_ai_review_version": "auditops-benchmark-provisional-ai-review.v2.4p",
        "candidate_id": "a" * 64,
        "candidate_manifest_sha256": "b" * 64,
        "rendered_packet_sha256": "c" * 64,
        "review_packet_sha256": "d" * 64,
        "source_manifest_sha256": "e" * 64,
        "review_bundle_id": "f" * 64,
        "assurance": "PROVISIONAL_AI_REVIEW",
        "experiment_label": "EXPLORATORY",
        "reviewer": {
            "reviewer_kind": "AI",
            "model_identity": "Codex/GPT-5",
            "deployment_identifier": None,
            "deployment_identifier_status": "NOT_EXPOSED_TO_AGENT",
            "pass_kind": pass_kind,
            "pass_id": f"pass-{pass_kind.casefold()}",
            "independence_claim": False,
            "reviewed_at": "2026-08-26T12:00:00Z",
            "attestation": "This is provisional AI feedback, not human approval.",
        },
        "method": {
            "method_id": "auditops.codex_provisional_review.v2.4p",
            "complete_scope_used_for_refusals": True,
            "deterministic_preflight_used": True,
            "human_approval": False,
        },
        "decisions": decisions,
    }
    return {
        **material,
        "decision_set_sha256": canonical_json_sha256(decisions),
        "review_id": canonical_json_sha256(material),
    }


def test_v24p_runtime_identity_is_separate_from_v23_contract() -> None:
    manifest = {
        "benchmark_version": "agent_benchmark.v2.4p",
        "runtime_contract_version": "v2.3",
    }
    assert _runtime_contract_version(manifest) == "v2.3"
    assert _execution_version(manifest) == "v2.4p"


def test_v24r2_anchor_is_exact_topic_bearing_and_not_generic() -> None:
    sentence = (
        "We recognize revenue when control of the promised service transfers "
        "to the customer."
    )
    anchor = _semantic_anchors_v24r2(sentence, topic="revenue recognition")[0]
    assert anchor in sentence
    assert "revenue" in anchor.casefold()
    assert 3 <= len(anchor.split()) <= 12
    with pytest.raises(ValueError, match="topic"):
        validate_semantic_anchor_v24r2(
            "We also have other",
            support_window="We also have other significant revenue arrangements.",
            topic="revenue recognition",
        )
    with pytest.raises(ValueError, match="heading"):
        validate_semantic_anchor_v24r2(
            "Table of Contents revenue",
            support_window="Table of Contents revenue disclosures",
            topic="revenue",
        )
    with pytest.raises(ValueError, match="heading"):
        validate_semantic_anchor_v24r2(
            "Loss Contingencies Description",
            support_window="Loss Contingencies Description.",
            topic="loss contingencies",
        )
    with pytest.raises(ValueError, match="heading"):
        validate_semantic_anchor_v24r2(
            "Valuation of Inventories",
            support_window="Valuation of Inventories - Provisions for Excess Inventory.",
            topic="inventory valuation",
        )
    with pytest.raises(ValueError, match="predicate"):
        validate_semantic_anchor_v24r2(
            "accumulated impairment losses",
            support_window="Goodwill is net of accumulated impairment losses.",
            topic="goodwill",
        )
    assert (
        _anchor_topic_from_question_v24r2(
            "What exact language describes the critical audit matter concerning loss contingencies?"
        )
        == "loss contingencies"
    )


def _policy_segment(evidence_id: str, content: str, *, subheading: str) -> dict:
    return {
        "evidence_id": evidence_id,
        "filing_id": "filing-1",
        "content": f"Content:\n{content}",
        "source_system": "SEC-EDGAR-CACHED",
        "period_key": "FY2025",
        "metadata": {
            "item": "8",
            "heading": "Financial Statements and Supplementary Data",
            "subheading": subheading,
            "source_file": "filing.htm",
            "char_start": 100,
            "char_end": 100 + len(content),
            "parent_evidence_id": f"parent-{evidence_id}",
            "boundary_policy_version": "auditops.test",
            "segment_index": 1,
            "relative_char_start": 0,
            "relative_char_end": len(content),
        },
    }


def test_v24r2_evidence_ids_bind_location_not_duplicate_text() -> None:
    base = {
        "filing_id": "filing-1",
        "period_key": "FY2025",
        "item": "8",
        "heading": "Financial Statements and Supplementary Data",
        "subheading": "Leases",
        "heading_path": "Item 8 > Notes > Leases",
        "source_file": "filing.htm",
        "retrieval_text": "Operating leases are recognized as right-of-use assets.",
        "text_masked": "Operating leases are recognized as right-of-use assets.",
    }
    segments = _segment_items(
        [
            {
                **base,
                "chunk_evidence_id": "chunk-a",
                "char_start": 100,
                "char_end": 160,
            },
            {
                **base,
                "chunk_evidence_id": "chunk-b",
                "char_start": 500,
                "char_end": 560,
            },
        ],
        hardened_r2=True,
    )
    assert len(segments) == 2
    assert len({item["evidence_id"] for item in segments}) == 2
    assert all(item["evidence_id"].startswith("v24r2e-") for item in segments)


def test_v24r2_opinion_evidence_uses_fixed_scope_label() -> None:
    filing = {
        "filing_id": "filing-1",
        "accession": "0000000001-26-000001",
        "ticker": "ACME",
        "source_entity_id": "0000000001",
        "form_type": "10-K",
        "fiscal_year_focus": 2025,
        "report_date": "2025-12-31",
    }
    evidence = {
        "evidence_id": "opinion-1",
        "filing_id": "filing-1",
        "period_key": "FY2025",
        "source_system": "SEC-EDGAR-CACHED",
        "content": (
            "In our opinion, the financial statements present fairly, in all material "
            "respects, the financial position of the Company."
        ),
        "metadata": {
            "item": "15",
            "heading": "Unrelated inherited heading",
            "subheading": "Exhibits",
            "heading_path": "Item 15 > Exhibits",
        },
    }
    row = _task_source_row(
        filing=filing,
        task_id="task-1",
        subtype="auditor_report_opinion_language",
        question="What exact opinion language does the independent auditor use for ACME's fiscal 2025 financial statements?",
        retrieval_query="auditor opinion financial statements",
        evidence_items=[evidence],
        answer=evidence["content"],
        expected_ids=["opinion-1"],
        negative_type=None,
        hardened_r2=True,
    )
    metadata = row["evidence_items"][0]["metadata"]
    assert metadata["item"] == "15"
    assert metadata["section_label"] == "AUDITOR_OPINION_FINANCIAL_STATEMENTS"
    assert metadata["subheading"] == "Opinion on the financial statements"


def test_v24r2_rejects_cam_procedure_sentence_and_incidental_policy() -> None:
    cam = {
        "metadata": {
            "heading": "Critical Audit Matter",
            "subheading": "Critical Audit Matter Description",
        }
    }
    assert (
        _positive_candidate_score(
            "critical_audit_matter",
            cam,
            "These procedures included testing the effectiveness of controls over oil and gas reserve estimates.",
            0,
            hardened_r2=True,
        )
        is None
    )
    incidental = _policy_segment(
        "revenue-incidental",
        "Unbilled receivables arise when the Company recognizes revenue that cannot yet be billed.",
        subheading="Revenue Recognition",
    )
    assert (
        _positive_candidate_score(
            "accounting_policy",
            incidental,
            "Unbilled receivables arise when the Company recognizes revenue that cannot yet be billed.",
            0,
            hardened_r2=True,
        )
        is None
    )


def test_v24r2_support_expands_but_gold_remains_exactly_source_bound() -> None:
    primary = _policy_segment(
        "lease-policy-1",
        "Operating leases are accounted for as right-of-use assets and lease liabilities. "
        "Lease liabilities are measured at the present value of future lease payments. "
        "The lease discount rate is the Company's incremental borrowing rate.",
        subheading="Leases",
    )
    alternative = _policy_segment(
        "lease-policy-2",
        "The Company recognizes lease liabilities at the present value of fixed lease "
        "payments and records corresponding right-of-use assets.",
        subheading="Leases",
    )
    unrelated = _policy_segment(
        "inventory-policy-1",
        "Inventories are valued at the lower of cost or net realizable value.",
        subheading="Inventories",
    )
    expanded = _expanded_support_window_v24r2(
        subtype="accounting_policy",
        item=primary,
        sentence_index=1,
        topic="lease accounting",
    )
    assert expanded.startswith("Operating leases are accounted for")
    assert expanded.endswith("incremental borrowing rate.")
    assert len(expanded) <= 1_000

    answer_sets = _acceptable_answer_sets_v24r2(
        subtype="accounting_policy",
        topic="lease accounting",
        evidence=[primary, alternative, unrelated],
        required_source_evidence_id="lease-policy-1",
        required_source_sentence=(
            "Lease liabilities are measured at the present value of future lease "
            "payments."
        ),
    )
    memberships = [
        tuple(extract["evidence_id"] for extract in answer_set["extracts"])
        for answer_set in answer_sets
    ]
    assert memberships == [("lease-policy-1",)]


def test_v24r2_support_expansion_stops_across_filtered_sentence_gap() -> None:
    selected = (
        "Lease liabilities are measured at the present value of future lease payments."
    )
    segment = _policy_segment(
        "lease-policy-gap",
        "Operating leases are accounted for as right-of-use assets and lease liabilities. "
        "This is omitted. " + selected,
        subheading="Leases",
    )

    expanded = _expanded_support_window_v24r2(
        subtype="accounting_policy",
        item=segment,
        sentence_index=1,
        topic="lease accounting",
    )

    assert expanded == selected


def test_v24r2_expanded_support_cannot_be_reused_across_subtypes() -> None:
    segment = _policy_segment(
        "lease-policy-1",
        "Operating leases are accounted for as right-of-use assets and lease liabilities. "
        "Lease liabilities are measured at the present value of future lease payments.",
        subheading="Leases",
    )
    filing = {
        "filing_id": "filing-1",
        "accession": "0000000001-26-000001",
        "ticker": "ACME",
        "source_entity_id": "0000000001",
        "form_type": "10-K",
        "fiscal_year_focus": 2025,
        "report_date": "2025-12-31",
    }
    source_row, review = _build_positive_narrative(
        filing=filing,
        subtype="accounting_policy",
        ordinal=0,
        all_segments=[segment],
        seed=20260821,
        identity_namespace="auditops-agent-benchmark-candidate.v2.4r2:test",
        hardened_r2=True,
    )
    excluded = frozenset(
        {
            str(source_row["extractive_answer"]).casefold(),
            str(
                review["acceptable_answer_sets"][0]["extracts"][0]["support_window"]
            ).casefold(),
        }
    )
    with pytest.raises(ValueError, match="lacks supported footnote_note"):
        _build_positive_narrative(
            filing=filing,
            subtype="footnote_note",
            ordinal=0,
            all_segments=[segment],
            seed=20260821,
            excluded_support_windows=excluded,
            identity_namespace="auditops-agent-benchmark-candidate.v2.4r2:test",
            hardened_r2=True,
        )


def test_v24r2_negative_ordinals_select_distinct_valid_claims() -> None:
    filing = {
        "filing_id": "filing-1",
        "accession": "0000000001-26-000001",
        "ticker": "ACME",
        "source_entity_id": "0000000001",
        "form_type": "10-K",
        "fiscal_year_focus": 2025,
        "report_date": "2025-12-31",
    }
    evidence = [
        _policy_segment(
            "ordinary-note-1",
            "The Company measures inventory at the lower of cost or net realizable value.",
            subheading="Inventories",
        )
    ]
    rows = [
        _build_negative_narrative(
            filing=filing,
            subtype="footnote_note",
            ordinal=ordinal,
            all_segments=evidence,
            seed=20260821,
            identity_namespace="auditops-agent-benchmark-candidate.v2.4r2:test",
            hardened_r2=True,
        )
        for ordinal in (0, 1)
    ]
    questions = [row[0]["question"] for row in rows]
    claims = [row[1]["refusal_scope"]["absent_claim"] for row in rows]
    assert len(set(questions)) == 2
    assert len(set(claims)) == 2


def test_v24r2_candidate_identity_changes_with_corpus_namespace() -> None:
    first = _candidate_identity_namespace_v24(
        candidate_version="auditops-agent-benchmark-candidate.v2.4r2",
        corpus_id="cached20-r2",
        hardened_r2=True,
    )
    second = _candidate_identity_namespace_v24(
        candidate_version="auditops-agent-benchmark-candidate.v2.4r2",
        corpus_id="cached20-r2b",
        hardened_r2=True,
    )
    assert first != second
    assert (
        _candidate_identity_namespace_v24(
            candidate_version="auditops-agent-benchmark-candidate.v2.4",
            corpus_id="ignored-for-r1",
            hardened_r2=False,
        )
        == "auditops-agent-benchmark-candidate.v2.4"
    )


def test_v24p_review_validates_reasonable_topic_alias_anchor(tmp_path: Path) -> None:
    quote = (
        "The Company is subject to claims, lawsuits, regulatory inquiries, other "
        "proceedings, and consent orders."
    )
    record = {
        "task_id": "v24n-alias",
        "narrative_subtype": "critical_audit_matter",
        "answerability": "ANSWERABLE",
        "question": (
            "What exact language in ACME's fiscal 2025 auditor report describes the "
            "critical audit matter concerning loss contingencies?"
        ),
        "entity": {
            "ticker": "ACME",
            "name": "ACME",
            "source_entity_id": "0000000001",
        },
        "period": {"period_key": "FY2025", "end_date": "2025-12-31"},
        "scope_description": (
            "ACME 10-K fiscal 2025 filing, loss contingencies disclosure"
        ),
        "frozen_evidence": [
            {
                "evidence_id": "cam-1",
                "content": quote,
                "metadata": {"item": "8"},
            }
        ],
        "acceptable_answer_sets": [
            {
                "set_id": "set-1",
                "extracts": [
                    {
                        "evidence_id": "cam-1",
                        "support_window": quote,
                        "required_semantic_anchors": ["Company is subject to claims"],
                    }
                ],
            }
        ],
        "refusal_scope": None,
    }
    checks, findings = _record_checks(record, bundle_root=tmp_path, refusal_bindings={})
    assert checks["semantic_anchors"] is True
    assert findings == []


def test_v24r2_footnotes_require_item_8_or_8a_and_compatible_topic() -> None:
    sentence = (
        "The Company measures inventory at the lower of cost or net realizable value."
    )
    item_8 = {
        "metadata": {
            "item": "8",
            "heading": "Financial Statements and Supplementary Data",
            "subheading": "Inventories",
        }
    }
    generic_item_8 = {
        "metadata": {
            "item": "8",
            "heading": "Financial Statements and Supplementary Data",
        }
    }
    incompatible_item_8 = {
        "metadata": {
            "item": "8",
            "heading": "Financial Statements and Supplementary Data",
            "subheading": "Property and Equipment",
        }
    }
    item_15 = {
        "metadata": {
            "item": "15",
            "heading": "Exhibits and Financial Statement Schedules",
            "subheading": "Note 3 - Inventories",
        }
    }
    assert (
        _positive_candidate_score(
            "footnote_note", item_8, sentence, 0, hardened_r2=True
        )
        is not None
    )
    assert (
        _positive_candidate_score(
            "footnote_note", generic_item_8, sentence, 0, hardened_r2=True
        )
        is None
    )
    assert (
        _positive_candidate_score(
            "footnote_note", incompatible_item_8, sentence, 0, hardened_r2=True
        )
        is None
    )
    assert (
        _positive_candidate_score(
            "footnote_note", item_15, sentence, 0, hardened_r2=True
        )
        is None
    )


def test_v24r2_binds_appended_notes_to_item8_with_source_provenance() -> None:
    chunks = [
        {
            "chunk_evidence_id": "item8-reference",
            "filing_id": "filing-1",
            "period_key": "FY2025",
            "item": "8",
            "heading": "Financial Statements and Supplementary Data.",
            "subheading": None,
            "heading_path": "Item 8 > Financial Statements and Supplementary Data.",
            "source_file": "filing.htm",
            "char_start": 100,
            "char_end": 200,
            "retrieval_text": "The financial statements together with the Notes thereto appear below.",
            "text_masked": "The financial statements together with the Notes thereto appear below.",
            "text_sha1": "a" * 40,
        },
        {
            "chunk_evidence_id": "notes-start",
            "filing_id": "filing-1",
            "period_key": "FY2025",
            "item": "15",
            "heading": "Exhibits, Financial Statement Schedules.",
            "subheading": "February 13, 2026",
            "heading_path": "Item 15 > Exhibits, Financial Statement Schedules.",
            "source_file": "filing.htm",
            "char_start": 1_000,
            "char_end": 1_400,
            "retrieval_text": (
                "Notes to consolidated financial statements Note <NUM> – "
                "Basis of presentation."
            ),
            "text_masked": (
                "Notes to consolidated financial statements Note <NUM> – "
                "Basis of presentation."
            ),
            "text_sha1": "b" * 40,
        },
        {
            "chunk_evidence_id": "note-body",
            "filing_id": "filing-1",
            "period_key": "FY2025",
            "item": "15",
            "heading": "Exhibits, Financial Statement Schedules.",
            "subheading": "Consolidation",
            "heading_path": "Item 15 > Exhibits, Financial Statement Schedules.",
            "source_file": "filing.htm",
            "char_start": 1_401,
            "char_end": 1_800,
            "retrieval_text": (
                "The Company measures inventory at the lower of cost or net "
                "realizable value."
            ),
            "text_masked": (
                "The Company measures inventory at the lower of cost or net "
                "realizable value."
            ),
            "text_sha1": "c" * 40,
        },
        {
            "chunk_evidence_id": "glossary",
            "filing_id": "filing-1",
            "period_key": "FY2025",
            "item": "15",
            "heading": "Exhibits, Financial Statement Schedules.",
            "subheading": "AFS",
            "heading_path": "Item 15 > Exhibits, Financial Statement Schedules.",
            "source_file": "filing.htm",
            "char_start": 2_000,
            "char_end": 2_200,
            "retrieval_text": "Glossary of Terms and Acronyms. AFS means available-for-sale.",
            "text_masked": "Glossary of Terms and Acronyms. AFS means available-for-sale.",
            "text_sha1": "d" * 40,
        },
    ]
    overlapping_start = copy.deepcopy(chunks[1])
    overlapping_start.update(
        {
            "chunk_evidence_id": "notes-start-overlap",
            "char_start": 1_200,
            "char_end": 1_500,
        }
    )
    chunks.insert(2, overlapping_start)
    bound = _bind_item8_note_chunks_v24r2(chunks)
    body = next(row for row in bound if row["chunk_evidence_id"] == "note-body")
    assert body["item"] == "8"
    assert body["source_item"] == "15"
    assert body["source_heading"] == "Exhibits, Financial Statement Schedules."
    assert body["item_binding_method"] == "auditops.item8_note_binding.v2.4r2"
    assert len(body["item_binding_scope_sha256"]) == 64
    assert (
        next(row for row in bound if row["chunk_evidence_id"] == "glossary")["item"]
        == "15"
    )

    segment = next(
        row
        for row in _segment_items(bound)
        if row["metadata"]["parent_evidence_id"] == "note-body"
    )
    assert (
        _positive_candidate_score(
            "footnote_note",
            segment,
            segment["content"],
            0,
            hardened_r2=True,
        )
        is not None
    )
    source_row, review = _build_positive_narrative(
        filing={
            "filing_id": "filing-1",
            "accession": "0000000001-26-000001",
            "ticker": "ACME",
            "source_entity_id": "0000000001",
            "form_type": "10-K",
            "fiscal_year_focus": 2025,
            "report_date": "2025-12-31",
        },
        subtype="footnote_note",
        ordinal=0,
        all_segments=_segment_items(bound),
        seed=20260821,
        identity_namespace="auditops-agent-benchmark-candidate.v2.4r2",
        hardened_r2=True,
    )
    assert review["item_binding"]["source_item"] == "15"
    assert review["item_binding"]["corrected_item"] == "8"
    selected_metadata = source_row["evidence_items"][0]["metadata"]
    assert "item_binding_method" not in selected_metadata
    assert "source_item" not in selected_metadata

    disjoint = copy.deepcopy(chunks)
    disjoint_start = copy.deepcopy(chunks[1])
    disjoint_start.update(
        {
            "chunk_evidence_id": "notes-start-disjoint",
            "char_start": 3_000,
            "char_end": 3_400,
        }
    )
    disjoint.append(disjoint_start)
    assert not any(
        "item_binding_method" in row for row in _bind_item8_note_chunks_v24r2(disjoint)
    )


def test_v24r2_binds_explicit_note_series_when_all_item_labels_are_missing() -> None:
    chunks = []
    for number in range(1, 16):
        prefix = "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS. " if number <= 5 else ""
        disclosure = (
            "The Company measures inventory at the lower of cost or net realizable value."
            if number == 2
            else "The accompanying schedules provide the relevant note disclosure."
        )
        content = prefix + disclosure
        chunks.append(
            {
                "chunk_evidence_id": f"note-{number}",
                "filing_id": "filing-1",
                "period_key": "FY2025",
                "item": None,
                "heading": None,
                "subheading": f"Note {number}. Example",
                "heading_path": f"Note {number}. Example",
                "source_file": "filing.htm",
                "char_start": number * 1_000,
                "char_end": number * 1_000 + len(content),
                "retrieval_text": content,
                "text_masked": content,
                "text_sha1": f"{number:040d}",
            }
        )
    bound = _bind_item8_note_chunks_v24r2(chunks)
    inventory = next(row for row in bound if row["chunk_evidence_id"] == "note-2")
    assert inventory["item"] == "8"
    assert inventory["source_item"] is None
    assert (
        inventory["item_binding_method"] == "auditops.missing_item_note_binding.v2.4r2"
    )
    inventory_segment = next(
        row
        for row in _segment_items(bound)
        if row["metadata"]["parent_evidence_id"] == "note-2"
    )
    assert (
        _positive_candidate_score(
            "footnote_note",
            inventory_segment,
            inventory_segment["content"],
            0,
            hardened_r2=True,
        )
        is not None
    )


def _topic_series_chunk(
    *,
    evidence_id: str,
    start: int,
    heading: str,
    item: str | None,
    text: str,
) -> dict:
    return {
        "chunk_evidence_id": evidence_id,
        "filing_id": "filing-1",
        "period_key": "FY2025",
        "item": item,
        "heading": heading,
        "subheading": heading,
        "heading_path": heading,
        "source_file": "filing.htm",
        "char_start": start,
        "char_end": start + len(text),
        "retrieval_text": text,
        "text_masked": text,
        "text_sha1": f"{start:040d}"[-40:],
    }


def test_v24r2_binds_dense_topic_series_with_and_without_item_labels() -> None:
    topics = (
        ("Revenue Recognition", "Revenue is recognized when control transfers."),
        (
            "Inventories",
            "Inventories are stated at the lower of cost or net realizable value.",
        ),
        ("Goodwill", "Goodwill is tested annually for impairment."),
        ("Leases", "Lease liabilities are measured at present value."),
        ("Income Taxes", "Deferred tax assets are assessed for realization."),
        (
            "Pension Obligations",
            "The defined benefit pension obligation is measured annually.",
        ),
    )

    def rows(*, item: str | None, include_item8_reference: bool) -> list[dict]:
        result: list[dict] = []
        if include_item8_reference:
            result.append(
                _topic_series_chunk(
                    evidence_id="item8-reference",
                    start=100,
                    heading="Financial Statements and Supplementary Data",
                    item="8",
                    text=(
                        "The financial statemen ts and no tes to consolidated "
                        "financial statements are incorporated below."
                    ),
                )
            )
        result.append(
            _topic_series_chunk(
                evidence_id="notes-reference",
                start=900,
                heading="Notes to Consolidated Financial Statements",
                item=item,
                text="Notes to Consolidated Financial Statements.",
            )
        )
        for index, (heading, text) in enumerate(topics, start=1):
            result.append(
                _topic_series_chunk(
                    evidence_id=f"topic-{index}",
                    start=1_000 + index * 1_000,
                    heading=heading,
                    item=item,
                    text=text,
                )
            )
        return result

    appended = _bind_item8_note_chunks_v24r2(
        rows(item="15", include_item8_reference=True)
    )
    appended_inventory = next(
        row for row in appended if row["chunk_evidence_id"] == "topic-2"
    )
    assert appended_inventory["item"] == "8"
    assert appended_inventory["source_item"] == "15"
    assert (
        appended_inventory["item_binding_method"]
        == "auditops.item8_topic_series_binding.v2.4r2"
    )

    missing = _bind_item8_note_chunks_v24r2(
        rows(item=None, include_item8_reference=False)
    )
    missing_inventory = next(
        row for row in missing if row["chunk_evidence_id"] == "topic-2"
    )
    assert missing_inventory["item"] == "8"
    assert missing_inventory["source_item"] is None
    assert (
        missing_inventory["item_binding_method"]
        == "auditops.missing_item_topic_series_binding.v2.4r2"
    )

    already_bound = rows(item="15", include_item8_reference=True)
    already_bound[1]["item"] = "8"
    already_bound[1]["heading"] = "Notes to Consolidated Financial Statements"
    already_bound[1]["heading_path"] = (
        "Item 8 > Notes to Consolidated Financial Statements"
    )
    assert not any(
        "item_binding_method" in row
        for row in _bind_item8_note_chunks_v24r2(already_bound)
    )


def test_v24r2_rejects_sparse_topic_series_binding() -> None:
    chunks = [
        _topic_series_chunk(
            evidence_id="notes-reference",
            start=900,
            heading="Notes to Consolidated Financial Statements",
            item=None,
            text="Notes to Consolidated Financial Statements.",
        )
    ]
    for index, (heading, text) in enumerate(
        (
            ("Revenue Recognition", "Revenue is recognized when control transfers."),
            ("Inventories", "Inventories are carried at cost."),
            ("Goodwill", "Goodwill is tested annually for impairment."),
            ("Leases", "Lease liabilities are measured at present value."),
            ("Income Taxes", "Deferred tax assets are assessed for realization."),
        ),
        start=1,
    ):
        chunks.append(
            _topic_series_chunk(
                evidence_id=f"topic-{index}",
                start=1_000 + index * 40_000,
                heading=heading,
                item=None,
                text=text,
            )
        )
    assert not any(
        "item_binding_method" in row for row in _bind_item8_note_chunks_v24r2(chunks)
    )


def test_v24r2_footnote_topic_handles_spacing_and_generic_note_container() -> None:
    inventory_sentence = (
        "Inventories are stated at the lower of cost or net realizable value."
    )
    assert (
        _footnote_topic_v24r2(
            {
                "item": "8",
                "heading": "FINANCIAL STATE MENTS AND SUPPLEMENTARY DATA",
                "subheading": "Inventories",
            },
            inventory_sentence,
        )
        == "inventory"
    )
    assert (
        _footnote_topic_v24r2(
            {
                "item": "8",
                "heading": "Financial Statements and Supplementary Data",
                "subheading": "Note 1—Summary of Significant Accounting Policies",
            },
            inventory_sentence,
        )
        == "inventory"
    )


def test_v24r2_opinion_uses_fixed_scope_not_stale_chunk_heading() -> None:
    sentence = (
        "In our opinion, the financial statements present fairly, in all material "
        "respects, the financial position of the Company in conformity with "
        "accounting principles generally accepted in the United States of America."
    )
    segment = {
        "evidence_id": "opinion-segment-1",
        "filing_id": "filing-1",
        "content": f"Content:\n{sentence}",
        "source_system": "SEC-EDGAR-CACHED",
        "period_key": "FY2025",
        "metadata": {
            "item": "8",
            "heading": "Financial Statements and Supplementary Data",
            "subheading": "Pension and Other Postretirement Benefit Expense",
            "source_file": "filing.htm",
            "char_start": 100,
            "char_end": 100 + len(sentence),
            "parent_evidence_id": "parent-1",
            "boundary_policy_version": "auditops.test",
            "segment_index": 1,
            "relative_char_start": 0,
            "relative_char_end": len(sentence),
        },
    }
    _, review = _build_positive_narrative(
        filing={
            "filing_id": "filing-1",
            "accession": "0000000001-26-000001",
            "ticker": "ACME",
            "source_entity_id": "0000000001",
            "form_type": "10-K",
            "fiscal_year_focus": 2025,
            "report_date": "2025-12-31",
        },
        subtype="auditor_report_opinion_language",
        ordinal=0,
        all_segments=[segment],
        seed=20260821,
        identity_namespace="auditops-agent-benchmark-candidate.v2.4r2",
        hardened_r2=True,
    )
    assert review["acceptable_answer_sets"][0]["extracts"][0]["evidence_id"] == (
        "opinion-segment-1"
    )
    assert review["scope_description"].endswith(
        "independent auditor's report, opinion on the financial statements"
    )


def test_v24r2_negative_screening_recognizes_reasonable_synonyms() -> None:
    retail = next(
        spec
        for spec in _NEGATIVE_CATALOG_R2["footnote_note"]
        if "retail inventory method" in spec["claim"]
    )
    assert _negative_spec_matches_scope_v24r2(
        retail,
        "inventories are valued under the retail method after applying markdowns",
    )
    assert not _negative_spec_matches_scope_v24r2(
        retail,
        "Inventory is held globally. The company has many unrelated disclosures. "
        "Retail stores use a sales method.",
    )
    pension = next(
        spec
        for spec in _NEGATIVE_CATALOG_R2["footnote_note"]
        if "pension plan" in spec["claim"]
    )
    assert _negative_spec_matches_scope_v24r2(
        pension,
        "the defined benefit retirement plan was settled and wound up in the year",
    )


def test_v24r2_negative_screening_rejects_shared_prefix_collisions() -> None:
    policies = {
        str(spec["claim"]): spec for spec in _NEGATIVE_CATALOG_R2["accounting_policy"]
    }
    assert not _negative_spec_matches_scope_v24r2(
        policies["all research and development costs are capitalized when incurred"],
        "The capital requirements resulting from stress testing apply next year.",
    )
    assert not _negative_spec_matches_scope_v24r2(
        policies["goodwill is amortized on a straight-line basis"],
        "The dividend rate was fixed before becoming a floating term SOFR rate.",
    )
    assert not _negative_spec_matches_scope_v24r2(
        policies["inventory is measured exclusively at replacement cost"],
        "Operational costs may increase when replacing disrupted services.",
    )
    assert not _negative_spec_matches_scope_v24r2(
        policies["revenue is recognized only when cash is collected"],
        "Brokerage commissions are collected and recognized when a trade occurs.",
    )


def test_v24p_review_is_explicitly_ai_and_cannot_enter_human_contract() -> None:
    review = _review()
    ids = [decision["task_id"] for decision in review["decisions"]]
    assert _validate_provisional_review(
        review,
        candidate_id="a" * 64,
        expected_kind="CRITIC",
        expected_ids=ids,
    ) == {task_id: "APPROVE" for task_id in ids}
    invalid = copy.deepcopy(review)
    invalid["reviewer"]["reviewer_kind"] = "HUMAN"
    with pytest.raises(ValueError, match="identity"):
        _validate_provisional_review(
            invalid,
            candidate_id="a" * 64,
            expected_kind="CRITIC",
            expected_ids=ids,
        )


def test_v24p_gate_and_exploratory_authorization_are_content_bound() -> None:
    decisions = [
        {"task_id": "released-1", "quote_answers_question": True, "comment": None}
    ]
    gate_material = {
        "gate_provisional_review_version": "auditops-gate-provisional-ai-review.v2.4p",
        "benchmark_id": "a" * 64,
        "run_manifest_sha256": "b" * 64,
        "release_review_packet_sha256": "c" * 64,
        "assurance": "PROVISIONAL_AI_REVIEW",
        "experiment_label": "EXPLORATORY",
        "reviewer": {
            "reviewer_kind": "AI",
            "model_identity": "Codex/GPT-5",
            "deployment_identifier": None,
            "deployment_identifier_status": "NOT_EXPOSED_TO_AGENT",
            "pass_kind": "GATE_RELEASES",
            "pass_id": "gate-pass",
            "independence_claim": False,
            "reviewed_at": "2026-08-26T12:00:00Z",
            "attestation": "Provisional AI review only.",
        },
        "benchmark_defect_status": "NONE_FOUND",
        "decisions": decisions,
    }
    gate = {**gate_material, "review_id": canonical_json_sha256(gate_material)}
    assert (
        validate_gate_provisional_review_v24p(
            gate,
            benchmark_id="a" * 64,
            run_manifest_sha256="b" * 64,
            release_task_ids={"released-1"},
        )
        == "CONFIRMED"
    )

    limitations = {
        "assurance": "PROVISIONAL_AI_REVIEW",
        "experiment_label": "EXPLORATORY",
        "generalization_label": "cached-20",
        "external_human_approval": False,
        "official_baseline": False,
    }
    auth_material = {
        "authorization_version": "auditops-exploratory-run-authorization.v2.4p.1",
        "benchmark_id": "a" * 64,
        "gate_metrics_sha256": "d" * 64,
        "decision": "GO_EXPLORATORY",
        "authorized_by": "benchmark-owner",
        "authorized_at": "2026-08-26T12:00:00Z",
        "assurance": "PROVISIONAL_AI_REVIEW",
        "experiment_label": "EXPLORATORY",
        "known_limitations": limitations,
        "known_limitations_sha256": canonical_json_sha256(limitations),
    }
    authorization = {
        **auth_material,
        "authorization_id": canonical_json_sha256(auth_material),
    }
    assert (
        validate_exploratory_authorization_v24p(
            authorization, benchmark_id="a" * 64, expected_authorizer="benchmark-owner"
        )
        == authorization
    )
    authorization["decision"] = "GO"
    with pytest.raises(ValueError, match="invalid"):
        validate_exploratory_authorization_v24p(
            authorization, benchmark_id="a" * 64, expected_authorizer="benchmark-owner"
        )


def test_v24p_public_schemas_validate_representative_objects() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    review = _review()
    review_schema = json.loads(
        (SPEC_ROOT / "provisional_ai_review.v2.4p.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(review_schema).validate(review)
    human_schema = json.loads(
        (SPEC_ROOT / "benchmark_review_decision.v2.4.schema.json").read_text()
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(human_schema).validate(review)
