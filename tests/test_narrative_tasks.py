from __future__ import annotations

import pytest

pytest.importorskip("haystack")

from auditops.narrative_tasks import (
    _build_unanswerable_narrative_tasks,
    _is_excluded_narrative_candidate,
    _narrative_label,
    answer_narrative,
    build_narrative_benchmark,
    build_narrative_routing_variants,
    evaluate_narrative_answers,
    evaluate_narrative_citations,
    evaluate_narrative_routing,
    read_narrative_task_specs,
    route_narrative_question,
    validate_narrative_answer,
    validate_narrative_task_spec,
    write_narrative_task_specs,
)


def test_build_narrative_benchmark_produces_answerable_note_tasks(
    populated_db, tmp_path
):
    task_specs = build_narrative_benchmark(
        str(populated_db), limit=10, per_filing_limit=1
    )

    assert task_specs
    assert all(task_spec["answerability"] == "ANSWERABLE" for task_spec in task_specs)
    assert all(
        task_spec["label"] in {"footnote_note", "accounting_policy"}
        for task_spec in task_specs
    )
    assert all(task_spec["scope_type"] == "note" for task_spec in task_specs)
    assert all(task_spec["extractive_answer"] for task_spec in task_specs)

    output_path = tmp_path / "narrative_tasks.jsonl"
    assert write_narrative_task_specs(output_path, task_specs) == len(task_specs)

    loaded = read_narrative_task_specs(output_path)
    assert loaded == task_specs
    for task_spec in loaded:
        validate_narrative_task_spec(task_spec)


def test_evaluate_narrative_citations_hits_fixture_chunks(populated_db):
    task_specs = build_narrative_benchmark(
        str(populated_db), limit=10, per_filing_limit=1
    )

    result = evaluate_narrative_citations(
        str(populated_db), task_specs, top_k=5, method="bm25_rerank", candidate_k=15
    )

    assert result["summary"]["task_count"] == len(task_specs)
    assert result["summary"]["answerable_task_count"] == len(task_specs)
    assert result["summary"]["unanswerable_task_count"] == 0
    assert result["summary"]["citation_coverage_at_k"] == 1.0
    assert result["summary"]["citation_precision_at_1"] == 1.0
    assert result["summary"]["citation_mrr_at_k"] == 1.0
    assert result["summary"]["answerability_accuracy"] is None
    assert result["summary"]["refusal_correctness"] is None


def test_narrative_label_excludes_audit_report_and_index_chunks():
    assert (
        _narrative_label(
            {
                "heading": "Financial Statements",
                "subheading": "Opinion on Internal Control Over Financial Reporting",
                "heading_path": "Item 8 > Financial Statements > Opinion on Internal Control Over Financial Reporting",
                "retrieval_text": "Definition and limitations of internal control over financial reporting.",
            }
        )
        is None
    )

    assert (
        _narrative_label(
            {
                "heading": "Financial Statements",
                "subheading": "Index to Consolidated Financial Statements",
                "heading_path": "Item 8 > Financial Statements > Index to Consolidated Financial Statements",
                "retrieval_text": "Index to consolidated financial statements",
            }
        )
        is None
    )


def test_narrative_candidate_excludes_internal_control_and_forward_looking_text():
    assert (
        _is_excluded_narrative_candidate(
            {
                "heading": "Financial Statements",
                "subheading": "Assessment of Internal Control Over Financial Reporting",
                "heading_path": "Item 8 > Financial Statements and Supplementary Data > Assessment of Internal Control Over Financial Reporting",
                "retrieval_text": "Assessment of Internal Control Over Financial Reporting",
            },
            "We have audited the accompanying consolidated balance sheets.",
        )
        is True
    )


def test_build_narrative_benchmark_can_include_unanswerable_tasks(populated_db):
    donor_pool = [
        {
            "narrative_task_spec_version": "v1",
            "narrative_task_schema_id": "narrative_task_spec.v1",
            "task_id": "task-a",
            "task_type": "narrative_citation",
            "filing_id": "filing-a",
            "ticker": "AAA",
            "form_type": "10-K",
            "period_key": "FY2025",
            "item": "8",
            "heading": "Financial Statements",
            "subheading": "Revenue Recognition",
            "heading_path": "Item 8 > Financial Statements > Revenue Recognition",
            "question": "financial statements Revenue Recognition contract assets deferred",
            "retrieval_query": "financial revenue recognition contract assets deferred",
            "label": "footnote_note",
            "scope_type": "note",
            "scope_key": "note:filing-a:item 8 > financial statements > revenue recognition",
            "answerability": "ANSWERABLE",
            "expected_chunk_ids": ["chunk-a"],
            "extractive_answer": "Revenue recognition answer.",
            "refusal_code": None,
            "negative_type": None,
            "donor_task_id": None,
            "citation_policy": {
                "require_chunk_evidence_ids": True,
                "retrieval_method": "bm25_rerank",
                "top_k": 5,
                "candidate_k": 15,
            },
        },
        {
            "narrative_task_spec_version": "v1",
            "narrative_task_schema_id": "narrative_task_spec.v1",
            "task_id": "task-a-alt",
            "task_type": "narrative_citation",
            "filing_id": "filing-a",
            "ticker": "AAA",
            "form_type": "10-K",
            "period_key": "FY2025",
            "item": "8",
            "heading": "Financial Statements",
            "subheading": "Inventory",
            "heading_path": "Item 8 > Financial Statements > Inventory",
            "question": "financial statements Inventory valuation raw materials obsolescence",
            "retrieval_query": "financial inventory valuation raw materials obsolescence",
            "label": "footnote_note",
            "scope_type": "note",
            "scope_key": "note:filing-a:item 8 > financial statements > inventory",
            "answerability": "ANSWERABLE",
            "expected_chunk_ids": ["chunk-a-alt"],
            "extractive_answer": "Inventory answer for filing a.",
            "refusal_code": None,
            "negative_type": None,
            "donor_task_id": None,
            "citation_policy": {
                "require_chunk_evidence_ids": True,
                "retrieval_method": "bm25_rerank",
                "top_k": 5,
                "candidate_k": 15,
            },
        },
        {
            "narrative_task_spec_version": "v1",
            "narrative_task_schema_id": "narrative_task_spec.v1",
            "task_id": "task-a-prior",
            "task_type": "narrative_citation",
            "filing_id": "filing-a-prior",
            "ticker": "AAA",
            "form_type": "10-Q",
            "period_key": "Q1_2025",
            "item": "1",
            "heading": "Financial Statements",
            "subheading": "Share-Based Compensation",
            "heading_path": "Item 1 > Financial Statements > Share-Based Compensation",
            "question": "financial statements Share-Based Compensation restricted stock vesting expense",
            "retrieval_query": "financial share-based compensation restricted stock vesting expense",
            "label": "footnote_note",
            "scope_type": "note",
            "scope_key": "note:filing-a-prior:item 1 > financial statements > share-based compensation",
            "answerability": "ANSWERABLE",
            "expected_chunk_ids": ["chunk-a-prior"],
            "extractive_answer": "Share-based compensation answer.",
            "refusal_code": None,
            "negative_type": None,
            "donor_task_id": None,
            "citation_policy": {
                "require_chunk_evidence_ids": True,
                "retrieval_method": "bm25_rerank",
                "top_k": 5,
                "candidate_k": 15,
            },
        },
        {
            "narrative_task_spec_version": "v1",
            "narrative_task_schema_id": "narrative_task_spec.v1",
            "task_id": "task-b",
            "task_type": "narrative_citation",
            "filing_id": "filing-b",
            "ticker": "BBB",
            "form_type": "10-K",
            "period_key": "FY2025",
            "item": "8",
            "heading": "Financial Statements",
            "subheading": "Inventory",
            "heading_path": "Item 8 > Financial Statements > Inventory",
            "question": "financial statements Inventory valuation raw materials obsolescence",
            "retrieval_query": "financial inventory valuation raw materials obsolescence",
            "label": "footnote_note",
            "scope_type": "note",
            "scope_key": "note:filing-b:item 8 > financial statements > inventory",
            "answerability": "ANSWERABLE",
            "expected_chunk_ids": ["chunk-b"],
            "extractive_answer": "Inventory answer.",
            "refusal_code": None,
            "negative_type": None,
            "donor_task_id": None,
            "citation_policy": {
                "require_chunk_evidence_ids": True,
                "retrieval_method": "bm25_rerank",
                "top_k": 5,
                "candidate_k": 15,
            },
        },
        {
            "narrative_task_spec_version": "v1",
            "narrative_task_schema_id": "narrative_task_spec.v1",
            "task_id": "task-c",
            "task_type": "narrative_citation",
            "filing_id": "filing-c",
            "ticker": "CCC",
            "form_type": "10-K",
            "period_key": "FY2025",
            "item": "8",
            "heading": "Financial Statements",
            "subheading": "Summary of Significant Accounting Policies",
            "heading_path": "Item 8 > Financial Statements > Summary of Significant Accounting Policies",
            "question": "financial statements Summary of Significant Accounting Policies estimates impairment goodwill",
            "retrieval_query": "financial summary significant accounting policies estimates impairment goodwill",
            "label": "accounting_policy",
            "scope_type": "note",
            "scope_key": "note:filing-c:item 8 > financial statements > summary of significant accounting policies",
            "answerability": "ANSWERABLE",
            "expected_chunk_ids": ["chunk-c"],
            "extractive_answer": "Accounting policy answer.",
            "refusal_code": None,
            "negative_type": None,
            "donor_task_id": None,
            "citation_policy": {
                "require_chunk_evidence_ids": True,
                "retrieval_method": "bm25_rerank",
                "top_k": 5,
                "candidate_k": 15,
            },
        },
    ]
    answerable_tasks = [
        donor_pool[0],
        donor_pool[2],
        donor_pool[3],
        donor_pool[4],
    ]
    task_specs = _build_unanswerable_narrative_tasks(
        answerable_tasks,
        {
            "filing-a": {
                "revenue",
                "recognition",
                "contract",
                "assets",
                "deferred",
                "inventory",
                "valuation",
                "raw",
                "materials",
                "obsolescence",
            },
            "filing-a-prior": {
                "share",
                "based",
                "compensation",
                "restricted",
                "stock",
                "vesting",
                "expense",
            },
            "filing-b": {"inventory", "valuation", "raw", "materials", "obsolescence"},
            "filing-c": {
                "summary",
                "significant",
                "accounting",
                "policies",
                "estimates",
                "impairment",
                "goodwill",
            },
        },
        limit=8,
        donor_task_specs=donor_pool,
    )

    assert task_specs
    negative_types = {task_spec["negative_type"] for task_spec in task_specs}
    assert "same_filing_wrong_note" in negative_types
    assert "same_issuer_wrong_period" in negative_types
    assert "unsupported_attribute" in negative_types
    assert "cross_label_query_transfer" in negative_types
    assert "cross_filing_query_transfer" in negative_types
    same_issuer_tasks = [
        task_spec
        for task_spec in task_specs
        if task_spec["negative_type"] == "same_issuer_wrong_period"
    ]
    unsupported_tasks = [
        task_spec
        for task_spec in task_specs
        if task_spec["negative_type"] == "unsupported_attribute"
    ]
    cross_filing_tasks = [
        task_spec
        for task_spec in task_specs
        if task_spec["negative_type"] == "cross_filing_query_transfer"
    ]
    assert same_issuer_tasks
    assert unsupported_tasks
    assert cross_filing_tasks
    assert all(
        task_spec["question"].startswith("For this period, does the ")
        for task_spec in same_issuer_tasks
    )
    assert all(
        task_spec["question"].startswith("Does the ") for task_spec in unsupported_tasks
    )
    assert all(
        task_spec["question"].startswith("Does the ")
        for task_spec in cross_filing_tasks
    )
    for unanswerable_task in task_specs:
        assert unanswerable_task["expected_chunk_ids"] == []
        assert unanswerable_task["extractive_answer"] is None
        assert unanswerable_task["refusal_code"] == "NARRATIVE_NOT_SUPPORTED"
        validate_narrative_task_spec(unanswerable_task)


def _mixed_fixture_tasks(populated_db):
    answerable_tasks = build_narrative_benchmark(
        str(populated_db), limit=10, per_filing_limit=1
    )
    answerable_task = answerable_tasks[0]
    unanswerable_task = {
        **answerable_task,
        "task_id": "manual-unanswerable",
        "question": "financial statements Inventory valuation raw materials obsolescence",
        "retrieval_query": "financial inventory valuation raw materials obsolescence",
        "scope_type": answerable_task["scope_type"],
        "scope_key": answerable_task["scope_key"],
        "answerability": "UNANSWERABLE",
        "expected_chunk_ids": [],
        "extractive_answer": None,
        "refusal_code": "NARRATIVE_NOT_SUPPORTED",
        "negative_type": "cross_filing_query_transfer",
        "donor_task_id": "synthetic-donor",
        "filing_id": answerable_task["filing_id"],
    }
    validate_narrative_task_spec(unanswerable_task)
    return answerable_tasks + [unanswerable_task]


def test_answer_narrative_returns_extract_and_refusal(populated_db):
    task_specs = _mixed_fixture_tasks(populated_db)
    answerable_task = next(
        task_spec
        for task_spec in task_specs
        if task_spec["answerability"] == "ANSWERABLE"
    )
    unanswerable_task = next(
        task_spec
        for task_spec in task_specs
        if task_spec["answerability"] == "UNANSWERABLE"
    )

    answerable_result = answer_narrative(
        str(populated_db),
        answerable_task["question"],
        answerable_task["filing_id"],
        task_specs=task_specs,
        top_k=5,
        method="bm25_rerank",
        candidate_k=15,
    )
    validate_narrative_answer(answerable_result)
    assert answerable_result["status"] == "OK"
    assert answerable_result["answer_text"] == answerable_task["extractive_answer"]
    assert (
        answerable_result["chunk_evidence_ids"] == answerable_task["expected_chunk_ids"]
    )

    unanswerable_result = answer_narrative(
        str(populated_db),
        unanswerable_task["question"],
        unanswerable_task["filing_id"],
        task_specs=task_specs,
        top_k=5,
        method="bm25_rerank",
        candidate_k=15,
    )
    validate_narrative_answer(unanswerable_result)
    assert unanswerable_result["status"] == "REFUSAL"
    assert unanswerable_result["refusal_code"] == unanswerable_task["refusal_code"]


def test_route_narrative_question_matches_loose_variant(populated_db):
    task_specs = _mixed_fixture_tasks(populated_db)
    answerable_task = next(
        task_spec
        for task_spec in task_specs
        if task_spec["answerability"] == "ANSWERABLE"
    )
    variant = next(
        variant
        for variant in build_narrative_routing_variants(
            [answerable_task], variants_per_task=1
        )
        if variant["source_task_id"] == answerable_task["task_id"]
    )

    matched = route_narrative_question(
        variant["question"], answerable_task["filing_id"], task_specs
    )

    assert matched["task_id"] == answerable_task["task_id"]


def test_build_narrative_routing_variants_strips_same_issuer_period_prefix(
    populated_db,
):
    task_specs = _mixed_fixture_tasks(populated_db)
    answerable_task = next(
        task_spec
        for task_spec in task_specs
        if task_spec["answerability"] == "ANSWERABLE"
    )
    same_issuer_task = {
        **answerable_task,
        "task_id": "manual-same-issuer-period",
        "question": "For this period, does the Revenue Recognition note discuss deferred revenue contract liabilities?",
        "retrieval_query": "revenue recognition deferred revenue contract liabilities",
        "answerability": "UNANSWERABLE",
        "expected_chunk_ids": [],
        "extractive_answer": None,
        "refusal_code": "NARRATIVE_NOT_SUPPORTED",
        "negative_type": "same_issuer_wrong_period",
        "donor_task_id": "synthetic-donor",
    }
    validate_narrative_task_spec(same_issuer_task)

    variant = next(
        variant
        for variant in build_narrative_routing_variants(
            [same_issuer_task], variants_per_task=1
        )
        if variant["source_task_id"] == same_issuer_task["task_id"]
    )

    assert variant["question"].startswith(
        "For this period, does the Revenue Recognition note discuss "
    )
    assert (
        "Does the Revenue Recognition note discuss For this period"
        not in variant["question"]
    )
    assert "deferred revenue contract liabilities" in variant["question"]


def test_evaluate_narrative_answers_scores_answerable_and_unanswerable(populated_db):
    task_specs = _mixed_fixture_tasks(populated_db)

    result = evaluate_narrative_answers(
        str(populated_db), task_specs, top_k=5, method="bm25_rerank", candidate_k=15
    )

    assert result["summary"]["task_count"] == len(task_specs)
    assert result["summary"]["answerability_accuracy"] == 1.0
    assert result["summary"]["citation_exactness"] == 1.0
    assert result["summary"]["answer_text_exactness"] == 1.0
    assert result["summary"]["refusal_correctness"] == 1.0
    assert result["summary"]["negative_type_counts"] == {
        "cross_filing_query_transfer": 1
    }


def test_evaluate_narrative_routing_scores_loose_variants(populated_db):
    task_specs = _mixed_fixture_tasks(populated_db)

    result = evaluate_narrative_routing(
        str(populated_db),
        task_specs,
        variants_per_task=1,
        top_k=5,
        method="bm25_rerank",
        candidate_k=15,
    )

    assert result["summary"]["variant_count"] == len(task_specs)
    assert result["summary"]["routing_accuracy"] == 1.0
    assert result["summary"]["answerability_accuracy"] == 1.0
    assert result["summary"]["citation_exactness"] == 1.0
    assert result["summary"]["answer_text_exactness"] == 1.0
    assert result["summary"]["refusal_correctness"] == 1.0
    assert result["summary"]["safe_refusal_accuracy"] == 1.0
    assert result["summary"]["negative_type_counts"] == {
        "cross_filing_query_transfer": 1
    }

    assert (
        _is_excluded_narrative_candidate(
            {
                "heading": "Financial Statements",
                "subheading": "Special Cautionary Notice Regarding Forward-Looking Statements",
                "heading_path": "Item 1 > Financial Statements > Special Cautionary Notice Regarding Forward-Looking Statements",
                "retrieval_text": "Forward-looking statements.",
            },
            "A material weakness in internal control related to ineffective information technology controls.",
        )
        is True
    )
