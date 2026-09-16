from __future__ import annotations

from auditops.runtime import (
    answer_quant,
    execute_task_plan,
    validate_structured_answer,
    validate_task_plan,
)
from auditops.tasks import (
    _build_question,
    _question_template_id,
    build_task_specs,
    read_jsonl,
    render_quant_datasets,
    validate_task_spec,
)


def test_quant_question_and_template_are_independent_of_gold_status(db_conn):
    source = build_task_specs(db_conn)[0]
    counterfactual = dict(source)
    counterfactual["target_status"] = (
        "REFUSAL" if source["target_status"] == "OK" else "OK"
    )

    assert _build_question(source) == _build_question(counterfactual)
    assert _question_template_id(source) == _question_template_id(counterfactual)


def _issuer_year_key(task_spec):
    fiscal_year = task_spec["period"].get("fiscal_year")
    if fiscal_year is None:
        fiscal_year = task_spec["filing_metadata"]["fiscal_year_focus"]
    return task_spec["ticker"], fiscal_year


def test_task_specs_capture_targets_and_negatives(db_conn):
    task_specs = build_task_specs(db_conn)

    assert task_specs
    for task_spec in task_specs[:10]:
        validate_task_spec(task_spec)

    gross_margin = next(
        task_spec
        for task_spec in task_specs
        if task_spec["metric_spec_id"] == "gross_margin"
        and task_spec["filing_id"] == "acme-20250630"
        and task_spec["period"]["period_key"] == "Q2_2025"
        and task_spec["target_status"] == "OK"
    )
    assert gross_margin["canonical_inputs"]
    assert gross_margin["distractors"]
    assert gross_margin["target_answer"]["evidence_ids"]

    refusal = next(
        task_spec
        for task_spec in task_specs
        if task_spec["metric_spec_id"] == "current_assets_rollup"
        and task_spec["filing_id"] == "acme-20250630"
        and task_spec["period"]["period_key"] == "ASOF_20250630"
        and task_spec["target_status"] == "REFUSAL"
    )
    assert refusal["negative_type"] == "missing_input"
    assert refusal["target_answer"]["refusal_code"] == "MISSING_INPUT"


def test_rendered_quant_datasets_reexecute_and_hold_out_by_issuer_year(
    tmp_path, db_conn
):
    summary = render_quant_datasets(db_conn, str(tmp_path))

    assert summary["counts"]["task_specs"] > 0
    assert summary["counts"]["train_quant_qa"] > 0
    assert summary["counts"]["train_quant_code"] > 0
    assert summary["counts"]["train_refusal"] > 0
    assert summary["counts"]["hard_negatives"] > 0
    assert summary["counts"]["eval_holdout"] > 0

    qa_rows = read_jsonl(tmp_path / "train_quant_qa.jsonl")
    refusal_rows = read_jsonl(tmp_path / "train_refusal.jsonl")
    task_specs = read_jsonl(tmp_path / "task_specs_quant.jsonl")
    hard_negatives = read_jsonl(tmp_path / "hard_negatives_quant.jsonl")

    gross_margin_row = qa_rows[0]
    validate_task_plan(gross_margin_row["task_plan"])
    validate_structured_answer(gross_margin_row["target_answer"])
    assert (
        execute_task_plan(db_conn, gross_margin_row["task_plan"])
        == gross_margin_row["target_answer"]
    )

    refusal_row = refusal_rows[0]
    validate_task_plan(refusal_row["task_plan"])
    validate_structured_answer(refusal_row["target_answer"])
    assert (
        execute_task_plan(db_conn, refusal_row["task_plan"])
        == refusal_row["target_answer"]
    )

    train_keys = {
        _issuer_year_key(task_spec)
        for task_spec in task_specs
        if task_spec["split"] == "train"
    }
    eval_keys = {
        _issuer_year_key(task_spec)
        for task_spec in task_specs
        if task_spec["split"] == "eval_holdout"
    }
    assert train_keys
    assert eval_keys
    assert train_keys.isdisjoint(eval_keys)

    negative_types = {row["negative_type"] for row in hard_negatives}
    assert {
        "missing_input",
        "distractor",
        "period_trap",
        "context_ambiguity",
        "unit_trap",
        "wrong_evidence_map",
    }.issubset(negative_types)


def test_answer_quant_routes_generated_questions(db_conn, tmp_path):
    render_quant_datasets(db_conn, str(tmp_path))
    qa_rows = read_jsonl(tmp_path / "train_quant_qa.jsonl")
    refusal_rows = read_jsonl(tmp_path / "train_refusal.jsonl")

    qa_row = qa_rows[0]
    assert (
        answer_quant(db_conn, qa_row["question"], qa_row["filing_id"])
        == qa_row["target_answer"]
    )

    refusal_row = refusal_rows[0]
    assert (
        answer_quant(db_conn, refusal_row["question"], refusal_row["filing_id"])
        == refusal_row["target_answer"]
    )


def test_answer_quant_prefers_exact_metric_and_longest_period_match(db_conn, tmp_path):
    render_quant_datasets(db_conn, str(tmp_path))
    qa_rows = read_jsonl(tmp_path / "train_quant_qa.jsonl") + read_jsonl(
        tmp_path / "eval_holdout.jsonl"
    )

    target_row = next(
        row
        for row in qa_rows
        if row["metric_spec_id"] == "free_cash_flow_margin"
        and row["filing_id"] == "acme-20250630"
        and row["period_key"] == "YTD_Q2_2025"
    )

    assert (
        answer_quant(db_conn, target_row["question"], target_row["filing_id"])
        == target_row["target_answer"]
    )


def test_answer_quant_ignores_duplicate_identical_task_specs(db_conn):
    task_specs = build_task_specs(db_conn)
    target_task = next(
        task_spec
        for task_spec in task_specs
        if task_spec["metric_spec_id"] == "gross_margin"
        and task_spec["filing_id"] == "acme-20250630"
        and task_spec["period"]["period_key"] == "Q2_2025"
        and task_spec["target_status"] == "OK"
    )

    duplicated_task_specs = list(task_specs) + [dict(target_task), dict(target_task)]

    question = f"{target_task['metric_label']} ({target_task['metric_spec_id']}) {target_task['period']['period_key']}"

    assert (
        answer_quant(db_conn, question, target_task["filing_id"], duplicated_task_specs)
        == target_task["target_answer"]
    )
