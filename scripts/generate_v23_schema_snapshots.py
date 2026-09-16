"""Generate deterministic v2.3 JSON Schema snapshots from immutable v2.2 files."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "auditops" / "specs"
SOURCE_NAMES = (
    "agent_case_result.v2.2.schema.json",
    "agent_plan_response.v2.2.schema.json",
    "agent_proposal.v2.2.schema.json",
    "agent_run_record.v2.2.schema.json",
    "agent_task_input.v2.2.schema.json",
    "agent_terminal_response.v2.2.schema.json",
    "context_pack.v2.2.schema.json",
    "failure.v2.2.schema.json",
    "narrative_selection_or_refusal.v2.2.schema.json",
    "narrative_selection_response.v2.2.schema.json",
    "quantitative_answer_or_refusal.v2.2.schema.json",
    "tool_arguments.v2.2.schema.json",
    "tool_observation.v2.2.schema.json",
)


def _replace(value: Any) -> Any:
    if isinstance(value, str):
        materialized = value.replace("v2.2", "v2.3").replace("V2.2", "V2.3")
        return (
            "FROZEN_BM25_V23_SECTION_SPLIT"
            if materialized == "FROZEN_BM25"
            else materialized
        )
    if isinstance(value, list):
        result = [_replace(item) for item in value]
        if all(not isinstance(item, (dict, list)) for item in result):
            deduped: list[Any] = []
            for item in result:
                if item not in deduped:
                    deduped.append(item)
            return deduped
        return result
    if isinstance(value, dict):
        return {key: _replace(child) for key, child in value.items()}
    return copy.deepcopy(value)


def _selection_policy_schema() -> dict[str, Any]:
    return {
        "oneOf": [
            {"type": "null"},
            {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "narrative_selection_policy_version",
                    "subtype",
                    "min_extracts",
                    "max_extracts",
                    "selection_rule",
                    "policy_sha256",
                ],
                "properties": {
                    "narrative_selection_policy_version": {"const": "v2.3"},
                    "subtype": {
                        "enum": [
                            "footnote_note",
                            "accounting_policy",
                            "auditor_report_opinion_language",
                            "critical_audit_matter",
                        ]
                    },
                    "min_extracts": {"const": 1},
                    "max_extracts": {"type": "integer", "minimum": 1, "maximum": 3},
                    "selection_rule": {"type": "string", "minLength": 1},
                    "policy_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
            },
        ]
    }


def _task_schema(schema: dict[str, Any]) -> None:
    schema["required"].append("narrative_selection_policy")
    schema["properties"]["narrative_selection_policy"] = _selection_policy_schema()
    selection = schema["properties"]["evidence_scope"]["properties"]["selection_method"]
    selection["enum"] = [
        "TYPED_FACT_MATERIALIZATION",
        "FROZEN_BM25_V23_SECTION_SPLIT",
    ]
    branch = schema["allOf"][0]
    branch["then"]["properties"]["narrative_selection_policy"] = {"type": "null"}
    narrative = branch["else"]["properties"]
    narrative["evidence_scope"]["properties"]["selection_method"] = {
        "const": "FROZEN_BM25_V23_SECTION_SPLIT"
    }
    narrative["refusal_policy"]["properties"]["allowed_codes"]["items"]["enum"] = [
        "NO_DIRECT_EVIDENCE",
        "PROMPT_INJECTION_DETECTED",
    ]
    narrative["narrative_selection_policy"] = {"type": "object"}
    for subtype in (
        "footnote_note",
        "accounting_policy",
        "auditor_report_opinion_language",
        "critical_audit_matter",
    ):
        maximum = 3 if subtype == "critical_audit_matter" else 1
        rule = (
            "SELECT_SMALLEST_DIRECT_EXTRACT_SET;MULTIPLE_ONLY_FOR_DISTINCT_CAM_COMPONENTS"
            if subtype == "critical_audit_matter"
            else "SELECT_ONE_MINIMAL_DIRECT_EXTRACT"
        )
        schema["allOf"].append(
            {
                "if": {
                    "properties": {
                        "task_type": {"const": "narrative_citation"},
                        "narrative_subtype": {"const": subtype},
                    },
                    "required": ["task_type", "narrative_subtype"],
                },
                "then": {
                    "properties": {
                        "narrative_selection_policy": {
                            "properties": {
                                "subtype": {"const": subtype},
                                "min_extracts": {"const": 1},
                                "max_extracts": {"const": maximum},
                                "selection_rule": {"const": rule},
                            }
                        }
                    }
                },
            }
        )


def _run_schema(schema: dict[str, Any]) -> None:
    schema["required"].append("narrative_selection_policy_sha256")
    schema["properties"]["narrative_selection_policy_sha256"] = {
        "$ref": "#/$defs/nullableSha256"
    }
    schema.setdefault("allOf", []).append(
        {
            "if": {"properties": {"narrative_subtype": {"type": "string"}}},
            "then": {
                "properties": {
                    "narrative_selection_policy_sha256": {"$ref": "#/$defs/sha256"}
                }
            },
            "else": {
                "properties": {"narrative_selection_policy_sha256": {"type": "null"}}
            },
        }
    )


def _observation_schema(schema: dict[str, Any]) -> None:
    frozen = schema["$defs"]["frozenEvidence"]
    provenance = frozen["properties"]["selection_provenance"]
    provenance["properties"]["selection_method"] = {
        "const": "FROZEN_BM25_V23_SECTION_SPLIT"
    }


def _context_schema(schema: dict[str, Any]) -> None:
    metadata = schema["$defs"]["metadata"]["properties"]
    metadata.update(
        {
            "boundary_policy_version": {"type": "string", "minLength": 1},
            "parent_evidence_id": {"type": "string", "minLength": 1},
            "relative_char_start": {"type": "integer", "minimum": 0},
            "relative_char_end": {"type": "integer", "minimum": 0},
            "section_label": {"type": ["string", "null"], "minLength": 1},
            "segment_index": {"type": "integer", "minimum": 1},
            "v23_bm25_score": {"type": "number"},
        }
    )


def _rewrite_refusal_codes(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _rewrite_refusal_codes(child)
    elif isinstance(value, list):
        for index, child in enumerate(list(value)):
            if isinstance(child, str) and child in {
                "NARRATIVE_CITATION_MISS",
                "NARRATIVE_NOT_SUPPORTED",
            }:
                value[index] = "NO_DIRECT_EVIDENCE"
            else:
                _rewrite_refusal_codes(child)
        deduped: list[Any] = []
        for child in value:
            if child not in deduped:
                deduped.append(child)
        value[:] = deduped


def main() -> int:
    for source_name in SOURCE_NAMES:
        source = SPECS / source_name
        target = SPECS / source_name.replace("v2.2", "v2.3")
        schema = _replace(json.loads(source.read_text(encoding="utf-8")))
        _rewrite_refusal_codes(schema)
        if source_name == "agent_task_input.v2.2.schema.json":
            _task_schema(schema)
        elif source_name == "agent_run_record.v2.2.schema.json":
            _run_schema(schema)
        elif source_name == "tool_observation.v2.2.schema.json":
            _observation_schema(schema)
        elif source_name == "context_pack.v2.2.schema.json":
            _context_schema(schema)
        target.write_text(
            json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
