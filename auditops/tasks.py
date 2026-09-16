from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .metrics import GENERATOR_VERSION, generate_answer_objects, load_metric_specs_by_id
from .pipeline import _stable_digest

TASK_SPEC_VERSION = "v1"
TASK_TYPE = "quant_metric"
TASK_PLAN_SCHEMA_ID = "task_plan.v1"
STRUCTURED_ANSWER_SCHEMA_ID = "structured_answer.v1"
TASK_SPEC_SCHEMA_ID = "task_spec_quant.v1"
RENDER_VERSION = "template-v2"
CODE_TARGET_VERSION = "python-task-plan-v2"
HARD_NEGATIVE_VERSION = "v1"
HOLDOUT_SPLIT_VERSION = "issuer_year_v1"
EXECUTOR_OP = "evaluate_metric_spec"
OUTPUT_FIELDS = [
    "status",
    "value",
    "unit",
    "period_key",
    "evidence_ids",
    "refusal_code",
]

_NEGATIVE_TYPE_MAP = {
    "MISSING_INPUT": "missing_input",
    "ZERO_DENOMINATOR": "zero_denominator",
    "AMBIGUOUS_CONTEXT": "context_ambiguity",
    "INCOMPATIBLE_UNITS": "unit_trap",
    "PERIOD_NOT_SUPPORTED": "period_trap",
}

_QUANT_TEMPLATES = (
    "Using filing {filing_id} for {ticker}, calculate {metric_label} ({metric_spec_id}) for period {period_key}. Return status, value, unit, period_key, evidence_ids, and refusal_code.",
    "For {ticker} filing {filing_id}, determine {metric_label} ({metric_spec_id}) at {period_key}. Respond with the structured answer fields status, value, unit, period_key, evidence_ids, refusal_code.",
    "AuditOps quant task: compute {metric_label} ({metric_spec_id}) for filing {filing_id} and period {period_key}. The output must be a structured answer with evidence IDs.",
)


def _metric_label(metric_spec_id: str) -> str:
    return metric_spec_id.replace("_", " ")


def _filing_text_array(
    filing: Mapping[str, Any],
    field_name: str,
) -> List[str]:
    value = filing.get(field_name)
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid {field_name} JSON for filing {filing.get('filing_id')!r}"
            ) from error
    if not isinstance(value, list):
        raise ValueError(
            f"Invalid {field_name} for filing {filing.get('filing_id')!r}: "
            "expected a JSON array"
        )
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _read_facts(
    conn, filing_id: Optional[str] = None
) -> Dict[tuple[str, str], List[Dict[str, Any]]]:
    facts_by_period: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    if filing_id:
        rows = conn.execute(
            """
            SELECT filing_id, period_key, concept_norm, unit_canon, value_num_exact, value_text, fact_evidence_id
            FROM facts_canon
            WHERE filing_id=?
            ORDER BY filing_id, period_key, concept_norm, fact_evidence_id
            """,
            (filing_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT filing_id, period_key, concept_norm, unit_canon, value_num_exact, value_text, fact_evidence_id
            FROM facts_canon
            ORDER BY filing_id, period_key, concept_norm, fact_evidence_id
            """
        ).fetchall()
    for row in rows:
        payload = dict(row)
        if payload["value_num_exact"] is not None:
            payload["value_num_exact"] = str(payload["value_num_exact"])
        facts_by_period[(payload["filing_id"], payload["period_key"])].append(payload)
    return facts_by_period


def _read_validators(
    conn, filing_id: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    validators_by_filing: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if filing_id:
        rows = conn.execute(
            """
            SELECT filing_id, validator_code, entity_kind, entity_key, message, details_json
            FROM validators_v0
            WHERE filing_id=?
            ORDER BY filing_id, validator_code, entity_kind, entity_key
            """,
            (filing_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT filing_id, validator_code, entity_kind, entity_key, message, details_json
            FROM validators_v0
            ORDER BY filing_id, validator_code, entity_kind, entity_key
            """
        ).fetchall()
    for row in rows:
        payload = dict(row)
        details = payload.get("details_json")
        payload["details_json"] = json.loads(details) if details else None
        validators_by_filing[payload["filing_id"]].append(payload)
    return validators_by_filing


def _build_target_answer(answer: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    result = answer.get("result") or {}
    return {
        "structured_answer_version": TASK_SPEC_VERSION,
        "task_id": task_id,
        "metric_spec_id": answer["metric_spec_id"],
        "filing_id": answer["filing_id"],
        "status": answer["status"],
        "value": result.get("value"),
        "unit": result.get("unit"),
        "period_key": answer["period"]["period_key"],
        "evidence_ids": list(answer.get("evidence_ids", [])),
        "refusal_code": answer.get("refusal_code"),
    }


def _canonical_inputs(answer: Dict[str, Any]) -> List[Dict[str, Any]]:
    inputs = []
    for name, payload in sorted(answer.get("input_facts", {}).items()):
        inputs.append(
            {
                "input_name": name,
                "concept_norm": payload["concept_norm"],
                "period_key": payload["period_key"],
                "unit_canon": payload["unit_canon"],
                "value_num_exact": (
                    str(payload["numeric_value"])
                    if payload.get("numeric_value") is not None
                    else None
                ),
                "value_text": payload.get("text_value"),
                "fact_evidence_ids": list(payload["fact_evidence_ids"]),
                "derived": bool(payload["derived"]),
            }
        )
    return inputs


def _select_distractors(
    facts_by_period: Mapping[tuple[str, str], Sequence[Dict[str, Any]]],
    filing_id: str,
    period_key: str,
    used_concepts: Sequence[str],
    used_evidence_ids: Sequence[str],
    limit: int = 4,
) -> List[Dict[str, Any]]:
    distractors = []
    used_concepts_set = set(used_concepts)
    used_evidence_id_set = set(used_evidence_ids)
    for fact in facts_by_period.get((filing_id, period_key), []):
        if fact["concept_norm"] in used_concepts_set:
            continue
        if fact["fact_evidence_id"] in used_evidence_id_set:
            continue
        distractors.append(
            {
                "concept_norm": fact["concept_norm"],
                "period_key": fact["period_key"],
                "unit_canon": fact["unit_canon"],
                "fact_evidence_id": fact["fact_evidence_id"],
                "value_num_exact": fact["value_num_exact"],
                "value_text": fact["value_text"],
                "selection_reason": "same_filing_same_period",
            }
        )
        if len(distractors) >= limit:
            break
    return distractors


def _issuer_year_key(task_spec: Mapping[str, Any]) -> str:
    ticker = task_spec.get("ticker") or task_spec["filing_id"]
    fiscal_year = task_spec["period"].get("fiscal_year")
    if fiscal_year is None:
        fiscal_year = task_spec["filing_metadata"].get("fiscal_year_focus")
    return f"{ticker}:{fiscal_year}"


def _answer_negative_type(answer: Mapping[str, Any]) -> Optional[str]:
    if answer["status"] != "REFUSAL":
        return None
    refusal_code = answer.get("refusal_code")
    return _NEGATIVE_TYPE_MAP.get(refusal_code, "deterministic_refusal")


def dedupe_task_specs(task_specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    index_by_task_id: Dict[str, int] = {}
    for task_spec in task_specs:
        materialized = dict(task_spec)
        task_id = materialized["task_id"]
        existing_index = index_by_task_id.get(task_id)
        if existing_index is None:
            deduped.append(materialized)
            index_by_task_id[task_id] = len(deduped) - 1
            continue

        existing = deduped[existing_index]
        if "validator_codes" in materialized or "validator_codes" in existing:
            merged_codes = sorted(
                set(existing.get("validator_codes", []))
                | set(materialized.get("validator_codes", []))
            )
            existing["validator_codes"] = merged_codes
    return deduped


def validate_task_spec(task_spec: Mapping[str, Any]) -> None:
    required_fields = {
        "task_spec_version",
        "task_id",
        "task_type",
        "source_answer_id",
        "metric_spec_id",
        "metric_kind",
        "filing_id",
        "ticker",
        "filing_metadata",
        "period",
        "target_status",
        "target_answer",
        "required_inputs",
        "canonical_inputs",
        "distractors",
        "negative_type",
        "evidence_requirements",
        "output_schema",
        "refusal_policy",
    }
    missing = sorted(required_fields - set(task_spec))
    if missing:
        raise ValueError(f"TaskSpec is missing required fields: {', '.join(missing)}")
    if task_spec["task_spec_version"] != TASK_SPEC_VERSION:
        raise ValueError(
            f"Unsupported TaskSpec version: {task_spec['task_spec_version']}"
        )
    if task_spec["task_type"] != TASK_TYPE:
        raise ValueError(f"Unsupported task type: {task_spec['task_type']}")
    if task_spec["target_status"] not in {"OK", "REFUSAL"}:
        raise ValueError(f"Unsupported target status: {task_spec['target_status']}")
    if task_spec["output_schema"].get("schema_id") != STRUCTURED_ANSWER_SCHEMA_ID:
        raise ValueError("TaskSpec output schema must point at structured_answer.v1")
    target_answer = task_spec["target_answer"]
    for field in OUTPUT_FIELDS:
        if field not in target_answer:
            raise ValueError(f"TaskSpec target_answer is missing {field}")
    if target_answer["period_key"] != task_spec["period"]["period_key"]:
        raise ValueError("TaskSpec target_answer period_key must match task period")
    if task_spec["target_status"] == "OK" and not target_answer["evidence_ids"]:
        raise ValueError("OK TaskSpecs must include evidence IDs in the target answer")


def build_task_plan_target(task_spec: Mapping[str, Any]) -> Dict[str, Any]:
    validate_task_spec(task_spec)
    return {
        "task_plan_version": TASK_SPEC_VERSION,
        "task_id": task_spec["task_id"],
        "task_type": task_spec["task_type"],
        "metric_spec_id": task_spec["metric_spec_id"],
        "filing_id": task_spec["filing_id"],
        "period_key": task_spec["period"]["period_key"],
        "executor_op": EXECUTOR_OP,
        "required_output_schema": dict(task_spec["output_schema"]),
        "refusal_policy": dict(task_spec["refusal_policy"]),
    }


def build_task_specs_from_answers(
    conn, answers: Sequence[Dict[str, Any]], filing_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    specs_by_id = load_metric_specs_by_id()
    facts_by_period = _read_facts(conn, filing_id=filing_id)
    validators_by_filing = _read_validators(conn, filing_id=filing_id)
    if filing_id:
        filing_rows = conn.execute(
            "SELECT * FROM filings WHERE filing_id=? ORDER BY filing_id", (filing_id,)
        ).fetchall()
    else:
        filing_rows = conn.execute(
            "SELECT * FROM filings ORDER BY filing_id"
        ).fetchall()
    filings = {row["filing_id"]: dict(row) for row in filing_rows}

    task_specs: List[Dict[str, Any]] = []
    for answer in answers:
        spec = specs_by_id[answer["metric_spec_id"]]
        filing = filings[answer["filing_id"]]
        task_id = _stable_digest("task-spec", answer["answer_id"], TASK_SPEC_VERSION)
        canonical_inputs = _canonical_inputs(answer)
        used_concepts = [payload["concept_norm"] for payload in canonical_inputs]
        used_evidence_ids = [
            evidence_id
            for payload in canonical_inputs
            for evidence_id in payload["fact_evidence_ids"]
        ]
        task_spec = {
            "task_spec_version": TASK_SPEC_VERSION,
            "task_spec_schema_id": TASK_SPEC_SCHEMA_ID,
            "task_id": task_id,
            "task_type": TASK_TYPE,
            "source_answer_id": answer["answer_id"],
            "source_generator_version": GENERATOR_VERSION,
            "metric_spec_id": answer["metric_spec_id"],
            "metric_label": _metric_label(answer["metric_spec_id"]),
            "metric_kind": spec["kind"],
            "metric_version": spec["version"],
            "filing_id": answer["filing_id"],
            "ticker": answer["ticker"],
            "filing_metadata": {
                "form_type": filing.get("form_type"),
                "fiscal_year_focus": filing.get("fiscal_year_focus"),
                "fiscal_period_focus": filing.get("fiscal_period_focus"),
                "report_date": filing.get("report_date"),
                "cik": filing.get("cik"),
                "accession": filing.get("accession"),
                "source_zip_name": filing.get("zip_name"),
                "main_html": filing.get("main_html"),
                "source_sha256": filing.get("source_sha256"),
                "source_size_bytes": filing.get("source_size_bytes"),
                "taxonomy_refs": _filing_text_array(filing, "taxonomy_refs_json"),
                "taxonomy_identifiers": _filing_text_array(
                    filing, "taxonomy_identifiers_json"
                ),
            },
            "period": dict(answer["period"]),
            "target_status": answer["status"],
            "target_answer": _build_target_answer(answer, task_id),
            "required_inputs": list(spec["required_inputs"]),
            "canonical_inputs": canonical_inputs,
            "distractors": _select_distractors(
                facts_by_period,
                answer["filing_id"],
                answer["period"]["period_key"],
                used_concepts,
                used_evidence_ids,
            ),
            "negative_type": _answer_negative_type(answer),
            "evidence_requirements": {
                "policy": spec.get("evidence_policy"),
                "require_evidence_ids": True,
                "min_evidence_ids": 1 if answer["status"] == "OK" else 0,
            },
            "output_schema": {
                "schema_id": STRUCTURED_ANSWER_SCHEMA_ID,
                "required_fields": list(OUTPUT_FIELDS),
            },
            "refusal_policy": {
                "allowed_codes": list(spec["refusal_rules"]),
                "default_code": answer.get("refusal_code"),
            },
            "validator_codes": sorted(
                {
                    row["validator_code"]
                    for row in validators_by_filing.get(answer["filing_id"], [])
                }
            ),
        }
        task_spec["template_id"] = _question_template_id(task_spec)
        task_spec["task_family"] = (
            f"quant_metric:{task_spec['metric_spec_id']}:"
            f"{str(task_spec['target_status']).casefold()}"
        )
        task_spec["question"] = _build_question(task_spec)
        validate_task_spec(task_spec)
        task_specs.append(task_spec)
    return dedupe_task_specs(task_specs)


def build_task_specs(conn, filing_id: Optional[str] = None) -> List[Dict[str, Any]]:
    answers = generate_answer_objects(conn, filing_id=filing_id)
    return build_task_specs_from_answers(conn, answers, filing_id=filing_id)


def _assign_holdout_splits(task_specs: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    keys = sorted({_issuer_year_key(task_spec) for task_spec in task_specs})
    if len(keys) <= 1:
        eval_keys: set[str] = set()
    else:
        eval_keys = {
            key
            for key in keys
            if int(_stable_digest("holdout", HOLDOUT_SPLIT_VERSION, key)[:8], 16) % 5
            == 0
        }
        if not eval_keys:
            eval_keys = {keys[-1]}
        if len(eval_keys) == len(keys):
            eval_keys = {keys[-1]}

    split_by_task_id = {}
    for task_spec in task_specs:
        split_by_task_id[task_spec["task_id"]] = (
            "eval_holdout" if _issuer_year_key(task_spec) in eval_keys else "train"
        )
    return split_by_task_id


def _build_question(task_spec: Mapping[str, Any]) -> str:
    template_index = _question_template_index(task_spec)
    template = _QUANT_TEMPLATES[template_index]
    return template.format(
        filing_id=task_spec["filing_id"],
        ticker=task_spec["ticker"] or "UNKNOWN",
        metric_label=task_spec["metric_label"],
        metric_spec_id=task_spec["metric_spec_id"],
        period_key=task_spec["period"]["period_key"],
    )


def _question_template_index(task_spec: Mapping[str, Any]) -> int:
    return int(_stable_digest("question", task_spec["task_id"])[:8], 16) % len(
        _QUANT_TEMPLATES
    )


def _question_template_id(task_spec: Mapping[str, Any]) -> str:
    return f"quant:{_question_template_index(task_spec)}:{RENDER_VERSION}"


def _build_code_target(task_spec: Mapping[str, Any]) -> str:
    task_plan = json.dumps(build_task_plan_target(task_spec), indent=2, sort_keys=True)
    return "\n".join(
        [
            "import json",
            "from auditops.runtime import execute_task_plan",
            "",
            f"task_plan = json.loads({task_plan!r})",
            "structured_answer = execute_task_plan(conn, task_plan)",
        ]
    )


def _base_rendered_record(task_spec: Mapping[str, Any], split: str) -> Dict[str, Any]:
    return {
        "render_version": RENDER_VERSION,
        "split": split,
        "task_id": task_spec["task_id"],
        "source_answer_id": task_spec["source_answer_id"],
        "filing_id": task_spec["filing_id"],
        "ticker": task_spec["ticker"],
        "metric_spec_id": task_spec["metric_spec_id"],
        "period_key": task_spec["period"]["period_key"],
        "template_id": _question_template_id(task_spec),
        "task_family": f"quant_metric:{task_spec['metric_spec_id']}:{str(task_spec['target_status']).casefold()}",
        "question": _build_question(task_spec),
        "task_plan": build_task_plan_target(task_spec),
        "target_answer": dict(task_spec["target_answer"]),
        "evidence_requirements": dict(task_spec["evidence_requirements"]),
        "negative_type": task_spec["negative_type"],
    }


def _period_siblings(
    task_specs: Sequence[Mapping[str, Any]], source_task: Mapping[str, Any]
) -> List[Mapping[str, Any]]:
    siblings = []
    for candidate in task_specs:
        if candidate["task_id"] == source_task["task_id"]:
            continue
        if candidate["filing_id"] != source_task["filing_id"]:
            continue
        if candidate["metric_spec_id"] != source_task["metric_spec_id"]:
            continue
        if candidate["period"]["period_key"] == source_task["period"]["period_key"]:
            continue
        siblings.append(candidate)
    siblings.sort(key=lambda candidate: candidate["period"]["period_key"])
    return siblings


def build_hard_negatives(task_specs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    negatives: List[Dict[str, Any]] = []
    for task_spec in task_specs:
        if task_spec["target_status"] == "REFUSAL":
            negatives.append(
                {
                    "hard_negative_version": HARD_NEGATIVE_VERSION,
                    "negative_id": _stable_digest(
                        "hard-negative",
                        task_spec["task_id"],
                        task_spec["target_answer"]["refusal_code"] or "refusal",
                    ),
                    "source_task_id": task_spec["task_id"],
                    "filing_id": task_spec["filing_id"],
                    "metric_spec_id": task_spec["metric_spec_id"],
                    "period_key": task_spec["period"]["period_key"],
                    "negative_type": task_spec["negative_type"]
                    or "deterministic_refusal",
                    "target_answer": dict(task_spec["target_answer"]),
                    "challenge": {
                        "kind": "actual_refusal",
                        "refusal_code": task_spec["target_answer"]["refusal_code"],
                    },
                }
            )
            continue

        if task_spec["distractors"]:
            negatives.append(
                {
                    "hard_negative_version": HARD_NEGATIVE_VERSION,
                    "negative_id": _stable_digest(
                        "hard-negative", task_spec["task_id"], "distractor"
                    ),
                    "source_task_id": task_spec["task_id"],
                    "filing_id": task_spec["filing_id"],
                    "metric_spec_id": task_spec["metric_spec_id"],
                    "period_key": task_spec["period"]["period_key"],
                    "negative_type": "distractor",
                    "target_answer": dict(task_spec["target_answer"]),
                    "challenge": {
                        "kind": "distractor",
                        "distractor_facts": task_spec["distractors"],
                    },
                }
            )

        siblings = _period_siblings(task_specs, task_spec)
        if siblings:
            negatives.append(
                {
                    "hard_negative_version": HARD_NEGATIVE_VERSION,
                    "negative_id": _stable_digest(
                        "hard-negative", task_spec["task_id"], "period-trap"
                    ),
                    "source_task_id": task_spec["task_id"],
                    "filing_id": task_spec["filing_id"],
                    "metric_spec_id": task_spec["metric_spec_id"],
                    "period_key": task_spec["period"]["period_key"],
                    "negative_type": "period_trap",
                    "target_answer": dict(task_spec["target_answer"]),
                    "challenge": {
                        "kind": "period_trap",
                        "distractor_period_key": siblings[0]["period"]["period_key"],
                    },
                }
            )

        validator_codes = set(task_spec.get("validator_codes", []))
        if not validator_codes and task_spec.get("validator_context"):
            validator_codes = {
                row["validator_code"] for row in task_spec["validator_context"]
            }
        if "CONTEXT_SELECTION_AMBIGUOUS" in validator_codes:
            negatives.append(
                {
                    "hard_negative_version": HARD_NEGATIVE_VERSION,
                    "negative_id": _stable_digest(
                        "hard-negative", task_spec["task_id"], "context-ambiguity"
                    ),
                    "source_task_id": task_spec["task_id"],
                    "filing_id": task_spec["filing_id"],
                    "metric_spec_id": task_spec["metric_spec_id"],
                    "period_key": task_spec["period"]["period_key"],
                    "negative_type": "context_ambiguity",
                    "target_answer": dict(task_spec["target_answer"]),
                    "challenge": {
                        "kind": "validator_context",
                        "validator_code": "CONTEXT_SELECTION_AMBIGUOUS",
                    },
                }
            )

        if "UNIT_SCALE_MISMATCH" in validator_codes:
            negatives.append(
                {
                    "hard_negative_version": HARD_NEGATIVE_VERSION,
                    "negative_id": _stable_digest(
                        "hard-negative", task_spec["task_id"], "unit-trap"
                    ),
                    "source_task_id": task_spec["task_id"],
                    "filing_id": task_spec["filing_id"],
                    "metric_spec_id": task_spec["metric_spec_id"],
                    "period_key": task_spec["period"]["period_key"],
                    "negative_type": "unit_trap",
                    "target_answer": dict(task_spec["target_answer"]),
                    "challenge": {
                        "kind": "validator_context",
                        "validator_code": "UNIT_SCALE_MISMATCH",
                    },
                }
            )

        mutated_evidence = list(task_spec["target_answer"]["evidence_ids"][1:])
        if task_spec["distractors"]:
            mutated_evidence.append(task_spec["distractors"][0]["fact_evidence_id"])
        negatives.append(
            {
                "hard_negative_version": HARD_NEGATIVE_VERSION,
                "negative_id": _stable_digest(
                    "hard-negative", task_spec["task_id"], "wrong-evidence-map"
                ),
                "source_task_id": task_spec["task_id"],
                "filing_id": task_spec["filing_id"],
                "metric_spec_id": task_spec["metric_spec_id"],
                "period_key": task_spec["period"]["period_key"],
                "negative_type": "wrong_evidence_map",
                "target_answer": dict(task_spec["target_answer"]),
                "challenge": {
                    "kind": "wrong_evidence_map",
                    "mutated_evidence_ids": mutated_evidence,
                },
            }
        )

    return negatives


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_task_specs(conn, output_path: str, filing_id: Optional[str] = None) -> int:
    task_specs = build_task_specs(conn, filing_id=filing_id)
    return write_jsonl(output_path, task_specs)


def render_quant_datasets(
    conn, output_dir: str, filing_id: Optional[str] = None
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    task_specs = build_task_specs(conn, filing_id=filing_id)
    split_by_task_id = _assign_holdout_splits(task_specs)
    for task_spec in task_specs:
        task_spec["split"] = split_by_task_id[task_spec["task_id"]]
        task_spec["split_manifest_version"] = HOLDOUT_SPLIT_VERSION

    qa_rows: List[Dict[str, Any]] = []
    code_rows: List[Dict[str, Any]] = []
    refusal_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    for task_spec in task_specs:
        record = _base_rendered_record(task_spec, task_spec["split"])
        if task_spec["split"] == "eval_holdout":
            eval_rows.append(record)
            continue
        if task_spec["target_status"] == "OK":
            qa_rows.append(record)
            code_rows.append(
                {
                    **record,
                    "code_target": _build_code_target(task_spec),
                    "code_target_version": CODE_TARGET_VERSION,
                }
            )
        else:
            refusal_rows.append(record)

    hard_negatives = build_hard_negatives(task_specs)

    paths = {
        "task_specs": output_root / "task_specs_quant.jsonl",
        "train_quant_qa": output_root / "train_quant_qa.jsonl",
        "train_quant_code": output_root / "train_quant_code.jsonl",
        "train_refusal": output_root / "train_refusal.jsonl",
        "hard_negatives": output_root / "hard_negatives_quant.jsonl",
        "eval_holdout": output_root / "eval_holdout.jsonl",
    }

    counts = {
        "task_specs": write_jsonl(paths["task_specs"], task_specs),
        "train_quant_qa": write_jsonl(paths["train_quant_qa"], qa_rows),
        "train_quant_code": write_jsonl(paths["train_quant_code"], code_rows),
        "train_refusal": write_jsonl(paths["train_refusal"], refusal_rows),
        "hard_negatives": write_jsonl(paths["hard_negatives"], hard_negatives),
        "eval_holdout": write_jsonl(paths["eval_holdout"], eval_rows),
    }

    return {
        "render_version": RENDER_VERSION,
        "split_manifest_version": HOLDOUT_SPLIT_VERSION,
        "counts": counts,
        "paths": {name: str(path) for name, path in paths.items()},
    }
