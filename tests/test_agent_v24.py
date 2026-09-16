from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from auditops.agent_batch import (
    _execution_version,
    _runtime_contract_version,
    _validate_full_run_authorization_v24,
)
from auditops.agent_benchmark_v24 import (
    _anchors,
    _bm25_select,
    _footnote_topic,
    _gate_membership,
    _negative_question,
    _positive_candidate_score,
    _question_for_positive,
    _raw_cam_segments_v24,
    _secondary_sample,
    _segment_items,
    _sentence_spans,
    _source_manifest_errors,
    _substantive_sentence_text,
    _topic,
    _validate_approval_for_freeze,
    _validate_human_review,
    validate_narrative_gold_v24,
    validate_natural_question,
    validate_question_scope_v24,
)
from auditops.agent_context import validate_evidence_items
from auditops.agent_contracts import build_agent_task_input
from auditops.agent_evaluation_v24 import _matches_answer_set
from auditops.agent_operations import task_operation_spec
from auditops.canonical_json import canonical_json_bytes, canonical_json_sha256

SUBTYPES = (
    "footnote_note",
    "accounting_policy",
    "auditor_report_opinion_language",
    "critical_audit_matter",
)


def _filings() -> list[dict]:
    return [
        {
            "filing_id": f"filing-{index:02d}",
            "source_entity_id": f"{index:010d}",
            "ticker": f"T{index:02d}",
        }
        for index in range(20)
    ]


def _review_rows() -> list[dict]:
    rows = []
    for filing in _filings():
        for subtype in SUBTYPES:
            count = 2 if subtype == "footnote_note" else 1
            for answerability in ("ANSWERABLE", "UNANSWERABLE"):
                for ordinal in range(count):
                    rows.append(
                        {
                            "task_id": (
                                f"{filing['filing_id']}:{subtype}:"
                                f"{answerability}:{ordinal}"
                            ),
                            "subtype": subtype,
                            "answerability": answerability,
                            "source_entity_id": filing["source_entity_id"],
                            "filing_id": filing["filing_id"],
                        }
                    )
    return rows


def _case(subtype: str = "accounting_policy") -> dict:
    marker = {
        "footnote_note": "footnote",
        "accounting_policy": "accounting policy",
        "auditor_report_opinion_language": "auditor opinion",
        "critical_audit_matter": "auditor critical audit matter",
    }[subtype]
    return {
        "task_id": "task-1",
        "question": f"What exact {marker} language does ACME disclose for fiscal 2025 in this filing?",
        "narrative_subtype": subtype,
        "entity": {"name": "ACME", "ticker": "ACME"},
        "period": {"period_key": "FY2025"},
    }


def _answer_gold() -> dict:
    sentence = "Revenue is recognized when control of the promised service transfers to the customer."
    return {
        "gold_record_version": "v2.4",
        "task_id": "task-1",
        "target": {"status": "OK"},
        "acceptable_answer_sets": [
            {
                "set_id": "set-1",
                "extracts": [
                    {
                        "evidence_id": "e-1",
                        "support_window": sentence,
                        "required_semantic_anchors": [
                            "control of the promised service"
                        ],
                    }
                ],
            }
        ],
        "refusal_scope": None,
    }


def _raw_cam_fixture(tmp_path: Path) -> tuple[Path, dict]:
    raw_root = tmp_path / "raw"
    ticker_root = raw_root / "ACME"
    ticker_root.mkdir(parents=True)
    archive = ticker_root / "ACME_10-K_0000000001-26-000001.zip"
    filing_html = b"""<!doctype html><html><body>
    <h1>Critical Audit Matter</h1>
    <p>The critical audit matter communicated below arose from the current-period
    audit and was communicated to the audit committee.</p>
    <h2>Revenue recognition and contract estimates</h2>
    <h3>Description of the Matter</h3>
    <p>As described in Note 2 to the financial statements, management applies
    significant judgment when estimating variable consideration in long-term
    revenue contracts.</p>
    <h3>How the Critical Audit Matter Was Addressed in the Audit</h3>
    <p>Auditing the revenue estimates involved testing selected contract terms.</p>
    <p>/s/ Example Auditor LLP</p>
    </body></html>"""
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("acme-20251231.htm", filing_html)
        handle.writestr("acme-20251231.xsd", b"<schema/>")
    filing = {
        "accession": "0000000001-26-000001",
        "ticker": "ACME",
        "filing_id": "filing-1",
        "fiscal_year_focus": 2025,
        "source_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    return raw_root, filing


def test_v24_runtime_contract_is_independent_of_benchmark_identity():
    manifest = {
        "benchmark_version": "agent_benchmark.v2.4",
        "runtime_contract_version": "v2.3",
    }
    assert _runtime_contract_version(manifest) == "v2.3"
    assert _execution_version(manifest) == "v2.4"
    with pytest.raises(ValueError, match="must declare"):
        _runtime_contract_version(
            {
                "benchmark_version": "agent_benchmark.v2.4",
                "runtime_contract_version": "v2.4",
            }
        )


@pytest.mark.parametrize(
    "question",
    [
        "financial statements borrowings temporary differences carryforwards foo bar?",
        "What does the filing say about revenue",
        "What does ACME disclose? And why?",
        "Too short now?",
    ],
)
def test_v24_natural_question_validator_rejects_keyword_fragments(question):
    with pytest.raises(ValueError):
        validate_natural_question(question)


@pytest.mark.parametrize("subtype", SUBTYPES)
def test_v24_generated_positive_questions_are_single_purpose(subtype):
    question = _question_for_positive(
        subtype=subtype,
        ticker="ACME",
        fiscal_year=2025,
        topic="Revenue Recognition",
    )
    validate_natural_question(question)
    assert question.count("?") == 1
    assert question.endswith("?")


@pytest.mark.parametrize(
    ("subtype", "claim"),
    [
        ("footnote_note", "inventory is measured using the retail inventory method"),
        ("accounting_policy", "revenue is recognized only when cash is collected"),
        (
            "auditor_report_opinion_language",
            "the financial statements were prepared on the cash basis of accounting",
        ),
        (
            "critical_audit_matter",
            "cryptocurrency custody was identified as a critical audit matter",
        ),
    ],
)
def test_v24_generated_negative_questions_bind_subtype_scope(subtype, claim):
    question = _negative_question(
        subtype=subtype,
        ticker="ACME",
        fiscal_year=2025,
        claim=claim,
    )
    case = _case(subtype)
    case["question"] = question
    validate_question_scope_v24(case)


def test_v24_negative_inventory_question_is_a_complete_statement():
    question = _negative_question(
        subtype="footnote_note",
        ticker="ACME",
        fiscal_year=2025,
        claim="inventory is measured using the retail inventory method",
    )
    assert "state that inventory is measured" in question


@pytest.mark.parametrize("subtype", SUBTYPES)
def test_v24_question_scope_binds_issuer_period_and_subtype(subtype):
    validate_question_scope_v24(_case(subtype))
    wrong = _case(subtype)
    wrong["period"]["period_key"] = "FY2024"
    with pytest.raises(ValueError, match="fiscal year"):
        validate_question_scope_v24(wrong)
    wrong = _case(subtype)
    wrong["question"] = wrong["question"].replace("ACME", "OTHER")
    with pytest.raises(ValueError, match="issuer"):
        validate_question_scope_v24(wrong)


def test_v24_gold_requires_complete_bounded_support_and_anchors():
    gold = _answer_gold()
    validate_narrative_gold_v24(gold, _case())
    fragment = copy.deepcopy(gold)
    fragment["acceptable_answer_sets"][0]["extracts"][0]["support_window"] = (
        "Revenue is recognized when control transfers"
    )
    with pytest.raises(ValueError, match="punctuation"):
        validate_narrative_gold_v24(fragment, _case())
    missing_anchor = copy.deepcopy(gold)
    missing_anchor["acceptable_answer_sets"][0]["extracts"][0][
        "required_semantic_anchors"
    ] = ["cash collection"]
    with pytest.raises(ValueError, match="anchor"):
        validate_narrative_gold_v24(missing_anchor, _case())


def test_v24_semantic_anchor_is_an_exact_support_substring_across_punctuation():
    sentence = "Revenue recognition, which involves judgment, is disclosed in Note 2."
    anchors = _anchors(sentence)
    assert len(anchors) == 1
    assert anchors[0] in sentence


def test_v24_sentence_splitter_does_not_truncate_us_gaap_language():
    text = (
        "In our opinion, the statements present fairly in conformity with U.S. "
        "generally accepted accounting principles. The audit was completed."
    )
    sentences = [sentence for sentence, _, _ in _sentence_spans(text)]
    expected_opinion = (
        "In our opinion, the statements present fairly in conformity with U.S. "
        "generally accepted accounting principles."
    )
    assert sentences == [
        expected_opinion,
        "The audit was completed.",
    ]


def test_v24_raw_cam_supplement_is_checksum_bound_and_sentence_bounded(tmp_path):
    raw_root, filing = _raw_cam_fixture(tmp_path)
    segments, provenance = _raw_cam_segments_v24(raw_root, filing)
    assert provenance["archive_sha256"] == filing["source_sha256"]
    assert provenance["sentence_count"] == len(segments)
    assert any("variable consideration" in row["content"] for row in segments)
    assert all(row["content"].endswith((".", "!", "?")) for row in segments)
    assert all(len(row["content"]) <= 1_000 for row in segments)
    assert len({row["evidence_id"] for row in segments}) == len(segments)


def test_v24_raw_cam_supplement_rejects_checksum_mismatch(tmp_path):
    raw_root, filing = _raw_cam_fixture(tmp_path)
    filing["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum conflicts"):
        _raw_cam_segments_v24(raw_root, filing)


def test_v24_source_manifest_requires_cam_archive_to_bind_corpus_source():
    filing = {
        "filing_id": "filing-1",
        "ticker": "ACME",
        "source_entity_id": "0000000001",
        "source_sha256": "a" * 64,
        "source_size_bytes": 100,
        "raw_cam_source": {
            "archive_name": "ACME.zip",
            "archive_sha256": "b" * 64,
            "archive_size_bytes": 100,
            "html_member": "acme.htm",
            "html_sha256": "c" * 64,
            "html_size_bytes": 80,
            "supplement_version": "auditops.raw_cam_supplement.v2.4",
            "section_text_sha256": "d" * 64,
            "sentence_count": 3,
        },
    }
    manifest = {
        "source_manifest_version": "v2.4-candidate",
        "corpus_id": "cached-20",
        "sources": {
            "quant_jsonl": {"sha256": "e" * 64, "bytes": 1, "records": 1},
            "corpus_sqlite": {"sha256": "f" * 64, "bytes": 1},
            "filings": [copy.deepcopy(filing) for _ in range(20)],
        },
        "retrieval": {
            "policy_version": "auditops.frozen_bm25.v2.4",
            "top_k": 5,
            "gold_used_for_ranking": False,
            "answer_support_checked_after_ranking": True,
        },
    }
    errors = _source_manifest_errors(manifest)
    assert any("does not bind" in error for error in errors)


def test_v24_topic_rejects_generic_cam_heading():
    topic = _topic(
        {
            "heading": "Financial Statements and Supplementary Data",
            "subheading": "Critical Audit Matter Description",
        },
        "The Company recognizes revenue over time for long-term contracts.",
    )
    assert "critical audit matter" not in topic.casefold()
    assert "revenue" in topic.casefold()


def test_v24_footnote_topic_requires_specific_note_heading():
    assert (
        _footnote_topic(
            {
                "heading": "Financial Statements and Supplementary Data",
                "subheading": "Property and Equipment",
            }
        )
        == "property and equipment"
    )
    assert (
        _footnote_topic({"subheading": "Management's Discussion and Analysis"}) is None
    )
    assert (
        _footnote_topic(
            {
                "heading": "Report of Independent Registered Public Accounting Firm",
                "subheading": "Goodwill Impairment Critical Audit Matter",
            }
        )
        is None
    )


def test_v24_cam_candidate_excludes_audit_boilerplate():
    item = {
        "metadata": {
            "subheading": "Critical Audit Matter Description",
            "heading": "Financial Statements and Supplementary Data",
        }
    }
    assert (
        _positive_candidate_score(
            "critical_audit_matter",
            item,
            "Those standards require that we plan and perform the audit to obtain reasonable assurance.",
            0,
        )
        is None
    )
    assert (
        _positive_candidate_score(
            "critical_audit_matter",
            item,
            "The Company recognizes revenue over time for long-term contracts using estimated costs and margins.",
            0,
        )
        is not None
    )
    section_item = {
        "metadata": {
            "subheading": "REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM",
            "section_label": "Critical Audit Matters",
        }
    }
    assert (
        _positive_candidate_score(
            "critical_audit_matter",
            section_item,
            "Management determines the product warranty liability by applying estimated repair costs and failure rates.",
            1,
        )
        is not None
    )


def test_v24_accounting_policy_candidate_requires_policy_semantics():
    item = {"metadata": {"subheading": "Revenue Recognition", "item": "8"}}
    assert (
        _positive_candidate_score(
            "accounting_policy",
            item,
            "Revenue is recognized when control transfers to the customer.",
            0,
        )
        is not None
    )
    assert (
        _positive_candidate_score(
            "accounting_policy",
            item,
            "The table below presents revenue by geography.",
            0,
        )
        is None
    )


def test_v24_opinion_candidate_allows_masked_periods():
    sentence = (
        "In our opinion, the financial statements as of <NUM>, <NUM>, and <NUM> "
        "present fairly, in all material respects, the Company's financial position."
    )
    assert (
        _positive_candidate_score(
            "auditor_report_opinion_language",
            {"metadata": {"subheading": "Opinion on the Financial Statements"}},
            sentence,
            0,
        )
        is not None
    )


def test_v24_item_15_footnote_requires_explicit_note_heading():
    sentence = "The Firm reports leases longer than twelve months as lease liabilities."
    assert (
        _positive_candidate_score(
            "footnote_note",
            {"metadata": {"item": "15", "subheading": "Note 18 - Leases"}},
            sentence,
            0,
        )
        is not None
    )
    assert (
        _positive_candidate_score(
            "footnote_note",
            {"metadata": {"item": "15", "subheading": "AFS: Available-for-sale"}},
            sentence,
            0,
        )
        is None
    )
    assert (
        _positive_candidate_score(
            "footnote_note",
            {"metadata": {"item": None, "subheading": "Note 12. Long-Term Debt"}},
            sentence,
            0,
        )
        is not None
    )


def test_v24_item_8_footnote_uses_sentence_topic_when_chunk_heading_drifted():
    sentence = "We record property and equipment at cost and depreciate it over estimated useful lives."
    item = {
        "metadata": {
            "item": "8",
            "heading": "Financial Statements and Supplementary Data",
            "subheading": "Inventories",
        }
    }
    assert _positive_candidate_score("footnote_note", item, sentence, 0) is not None
    assert _topic(item["metadata"], sentence, subtype="footnote_note") == (
        "property and equipment"
    )


def test_v24_footnote_topic_comes_from_substantive_paragraph_not_stale_title():
    sentence = (
        "Note 4 - Cash Equivalents\n\n"
        "The Company measures its financial assets at fair value each reporting period."
    )
    item = {"metadata": {"item": "8", "subheading": "Cash Equivalents"}}
    assert _topic(item["metadata"], sentence, subtype="footnote_note") == (
        "fair value measurements"
    )
    assert _substantive_sentence_text(sentence) == (
        "The Company measures its financial assets at fair value each reporting period."
    )


def test_v24_footnote_candidate_rejects_cam_procedure_text():
    sentence = (
        "How the Critical Audit Matter Was Addressed in the Audit\n\n"
        "We tested management's measurement of deferred tax liabilities."
    )
    assert (
        _positive_candidate_score(
            "footnote_note",
            {"metadata": {"item": "8", "subheading": "Income Taxes"}},
            sentence,
            0,
        )
        is None
    )


def test_v24_refusal_requires_full_scope_proof():
    gold = {
        "gold_record_version": "v2.4",
        "task_id": "task-1",
        "target": {"status": "REFUSAL"},
        "acceptable_answer_sets": [],
        "refusal_scope": {
            "scope_description": "all filing chunks",
            "causal_negative_type": "human_verified_full_filing_absence",
            "absent_claim": "cash basis",
            "full_filing_chunk_count": 10,
            "full_filing_text_sha256": "a" * 64,
            "mechanical_absence_check": True,
        },
    }
    validate_narrative_gold_v24(gold, _case())
    gold["refusal_scope"]["mechanical_absence_check"] = False
    with pytest.raises(ValueError, match="mechanical"):
        validate_narrative_gold_v24(gold, _case())


def test_v24_secondary_and_gate_memberships_meet_exact_grid():
    rows = _review_rows()
    secondary = _secondary_sample(rows, _filings())
    assert len(secondary["task_ids"]) == 40
    selected = {task_id for task_id in secondary["task_ids"]}
    issuer_counts = {}
    for row in rows:
        if row["task_id"] in selected:
            issuer_counts[row["source_entity_id"]] = (
                issuer_counts.get(row["source_entity_id"], 0) + 1
            )
    assert set(issuer_counts.values()) == {2}
    gate = _gate_membership(rows, _filings())
    assert gate["case_count"] == 50
    assert gate["breakdown"]["footnote_note:ANSWERABLE"] == 10
    assert gate["breakdown"]["critical_audit_matter:UNANSWERABLE"] == 5
    assert min(gate["issuer_counts"].values()) >= 2


def test_v24_retrieval_ranking_does_not_accept_or_consult_gold():
    items = [
        {"evidence_id": "e-1", "content": "revenue recognition control transfer"},
        {"evidence_id": "e-2", "content": "inventory valuation lower cost"},
    ]
    first = _bm25_select("revenue control", items)
    mutated_evaluator_gold = {"target": "completely different"}
    assert mutated_evaluator_gold  # gold exists only outside the ranking call
    second = _bm25_select("revenue control", items)
    assert first == second
    assert first[0]["evidence_id"] == "e-1"


def test_v24_narrative_evidence_preserves_v23_runtime_metadata_contract():
    segments = _segment_items(
        [
            {
                "chunk_evidence_id": "parent-1",
                "filing_id": "filing-1",
                "period_key": "FY2025",
                "item": "8",
                "heading": "Revenue Recognition",
                "subheading": "Revenue Recognition",
                "heading_path": "Item 8 / Revenue Recognition",
                "source_file": "filing.htm",
                "char_start": 0,
                "char_end": 96,
                "retrieval_text": (
                    "Content:\nRevenue is recognized when control of the promised "
                    "service transfers to the customer."
                ),
                "text_masked": None,
                "text_sha1": "a" * 40,
            }
        ]
    )
    selected = _bm25_select("revenue control transfer", segments)
    spec = task_operation_spec("narrative_citation", version="v2.3")
    task = build_agent_task_input(
        task_id="v24-contract-probe",
        task_type="narrative_citation",
        jurisdiction="US",
        reporting_framework="US-GAAP",
        standards_profile={"name": "PCAOB filing verification", "version": "2026"},
        source_system="SEC-EDGAR-CACHED",
        question="What accounting policy does ACME disclose for revenue in fiscal 2025?",
        entity={"entity_id": "issuer-1", "name": "ACME", "ticker": "ACME"},
        filing={"filing_id": "filing-1", "form_type": "10-K"},
        period={"period_key": "FY2025"},
        retrieval_query="revenue control transfer",
        narrative_subtype="accounting_policy",
        allowed_tools=[spec.operation],
        evidence_ids=[item["evidence_id"] for item in selected],
        output_schema_id=spec.terminal_output_schema_id,
        refusal_codes=["NO_DIRECT_EVIDENCE"],
        contract_version="v2.3",
    )
    assert validate_evidence_items(task, selected) == selected
    assert "v23_bm25_score" in selected[0]["metadata"]
    assert "text_sha1" not in selected[0]["metadata"]


def test_v24_answer_set_matching_accepts_bounded_quote_with_all_anchors():
    proposal = {
        "claims": [
            {
                "evidence_ids": ["e-1"],
                "supporting_text": "control of the promised service transfers to the customer",
            }
        ]
    }
    matched, set_id = _matches_answer_set(
        proposal, _answer_gold()["acceptable_answer_sets"]
    )
    assert matched is True
    assert set_id == "set-1"
    proposal["claims"][0]["supporting_text"] = "Revenue is recognized."
    assert _matches_answer_set(proposal, _answer_gold()["acceptable_answer_sets"]) == (
        False,
        None,
    )


def test_v24_human_review_rejects_codex_and_incomplete_membership():
    base = {
        "review_decision_version": "auditops-benchmark-review-decision.v2.4",
        "candidate_id": "a" * 64,
        "rendered_packet_sha256": "b" * 64,
        "reviewer": {
            "identity": "Codex",
            "role": "PRIMARY",
            "reviewed_at": "2026-08-25T12:00:00Z",
            "attestation": "I independently reviewed the assigned AuditOps cases as a human reviewer.",
        },
        "decisions": [],
    }
    with pytest.raises(ValueError, match="external human"):
        _validate_human_review(
            base,
            candidate_id="a" * 64,
            packet_sha256="b" * 64,
            expected_role="PRIMARY",
            expected_ids={"task-1"},
        )


def test_v24_approval_tamper_is_rejected_before_freeze(tmp_path):
    candidate_manifest = {
        "candidate_id": "a" * 64,
    }
    (tmp_path / "candidate_manifest.json").write_bytes(
        canonical_json_bytes(candidate_manifest, newline=True)
    )
    (tmp_path / "review_packet.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "source_manifest.json").write_text("{}\n", encoding="utf-8")
    material = {
        "approval_version": "auditops-benchmark-approval.v2.4",
        "external_human_approval": True,
        "candidate_id": "a" * 64,
        "candidate_manifest_sha256": "0" * 64,
        "review_packet_json_sha256": "0" * 64,
        "source_manifest_sha256": "0" * 64,
        "approved_case_count": 200,
    }
    approval = {**material, "approval_id": canonical_json_sha256(material)}
    approval_path = tmp_path / "approval.json"
    approval_path.write_bytes(canonical_json_bytes(approval, newline=True))
    with pytest.raises(ValueError, match="manifest hash"):
        _validate_approval_for_freeze(tmp_path, approval_path, candidate_manifest)


def test_v24_public_schemas_accept_representative_objects():
    jsonschema = pytest.importorskip("jsonschema")
    spec_root = Path(__file__).parents[1] / "auditops" / "specs"
    schema = json.loads((spec_root / "narrative_gold.v2.4.schema.json").read_text())
    full = {
        **_answer_gold(),
        "schema_id": "agent_gold_record.v2.4",
        "task_type": "narrative_citation",
        "template_id": "v24:accounting_policy:answer",
        "task_family": "narrative_citation:accounting_policy:answerable",
        "strata": {},
        "jurisdiction": "US",
        "corpus_id": "cached-20",
        "source_task_sha256": "c" * 64,
        "evaluator_negative_cause": None,
        "legacy_refusal_code": None,
    }
    full["target"] = {
        "status": "OK",
        "answer_text": full["acceptable_answer_sets"][0]["extracts"][0][
            "support_window"
        ],
        "chunk_evidence_ids": ["e-1"],
        "refusal_code": None,
        "filing_id": "filing-1",
        "period_key": "FY2025",
    }
    jsonschema.Draft202012Validator(schema).validate(full)


def test_v24_full_run_authorization_is_strict_and_human():
    authorization = {
        "authorization_version": "auditops-full-run-authorization.v2.4",
        "benchmark_id": "a" * 64,
        "gate_metrics_sha256": "b" * 64,
        "decision": "GO",
        "authorized_by": "Independent Reviewer",
        "authorized_at": "2026-08-25T12:00:00Z",
    }
    assert (
        _validate_full_run_authorization_v24(authorization, benchmark_id="a" * 64)
        == authorization
    )
    invalid = copy.deepcopy(authorization)
    invalid["gate_metrics_sha256"] = "Z" * 64
    with pytest.raises(ValueError, match="identity"):
        _validate_full_run_authorization_v24(invalid, benchmark_id="a" * 64)
    invalid = copy.deepcopy(authorization)
    invalid["authorized_by"] = "Codex"
    with pytest.raises(ValueError, match="external human"):
        _validate_full_run_authorization_v24(invalid, benchmark_id="a" * 64)


def test_v24_full_run_authorization_schema_accepts_representative_object():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "auditops"
            / "specs"
            / "full_run_authorization.v2.4.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(
        {
            "authorization_version": "auditops-full-run-authorization.v2.4",
            "benchmark_id": "a" * 64,
            "gate_metrics_sha256": "b" * 64,
            "decision": "GO",
            "authorized_by": "Independent Reviewer",
            "authorized_at": "2026-08-25T12:00:00Z",
        }
    )
