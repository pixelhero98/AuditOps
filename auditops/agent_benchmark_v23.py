"""Derive an immutable AuditOps v2.3 benchmark from a frozen v2.2 benchmark.

Evaluation membership is preserved exactly. Narrative evidence is split and
re-ranked without consulting evaluator gold; gold anchors are rebound only
after the inference-visible scope has been frozen. This keeps the retrieval
boundary auditable while preventing target-driven evidence selection.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from .agent_context import build_context_pack
from .agent_contracts import validate_agent_proposal, validate_agent_task_input
from .agent_operations import narrative_selection_contract, task_operation_spec
from .agent_tools import (
    evaluate_quant_evidence,
    load_frozen_evidence,
    metric_operation_contract,
)
from .canonical_json import canonical_json_bytes, canonical_json_sha256
from .provenance import assert_no_secrets
from .text_support import is_contiguous_text_supported

BENCHMARK_VERSION_V23 = "agent_benchmark.v2.3"
GOLD_RECORD_VERSION_V23 = "v2.3"
FEW_SHOT_EXAMPLE_VERSION_V23 = "v2.3"
CHUNK_BOUNDARY_POLICY_VERSION = "auditops.narrative_chunk_boundary.v2.3"
NARRATIVE_SELECTION_METHOD_V23 = "FROZEN_BM25_V23_SECTION_SPLIT"
MAX_EVIDENCE_ITEMS = 5
MAX_PARAGRAPH_GROUP_CHARS = 2_400

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")
_SECTION_RE = re.compile(
    r"(?im)^[ \t]*(?:"
    r"report of independent registered public accounting firm|"
    r"opinion on the financial statements|"
    r"opinion on internal control over financial reporting|"
    r"basis for opinion|"
    r"critical audit matters?|"
    r"definition and limitations of internal control over financial reporting"
    r")[ \t]*$"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path.name}:{line_number} must contain an object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def _sha256_path(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, records: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"sha256": _sha256_path(path), "bytes": path.stat().st_size}
    if records is not None:
        value["records"] = records
    return value


def _verify_parent(parent: Path) -> dict[str, Any]:
    manifest_path = parent / "benchmark_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("benchmark_version") != "agent_benchmark.v2.2":
        raise ValueError(
            "v2.3 derivation requires a frozen agent_benchmark.v2.2 parent"
        )
    material = {key: value for key, value in manifest.items() if key != "benchmark_id"}
    if manifest.get("benchmark_id") != canonical_json_sha256(material):
        raise ValueError("Parent benchmark_id does not bind its canonical manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TypeError("Parent benchmark artifact registry is missing")
    required = {
        "cases.jsonl",
        "evidence.jsonl",
        "gold.jsonl",
        "verifier_observations.jsonl",
        "few_shot.jsonl",
        "case_lineage.jsonl",
        "source_manifest.json",
        "parity_50.json",
    }
    for name in sorted(required):
        path = parent / name
        expected = artifacts.get(name)
        if not path.is_file() or not isinstance(expected, Mapping):
            raise ValueError(f"Parent benchmark is missing bound artifact {name}")
        if _sha256_path(path) != expected.get(
            "sha256"
        ) or path.stat().st_size != expected.get("bytes"):
            raise ValueError(f"Parent benchmark artifact mismatch: {name}")
    return manifest


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def _paragraph_groups(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split oversized sections only at paragraph boundaries."""

    if end - start <= MAX_PARAGRAPH_GROUP_CHARS:
        return [(start, end)]
    boundaries = [start]
    for match in re.finditer(r"\n\s*\n+", text[start:end]):
        boundaries.append(start + match.end())
    boundaries.append(end)
    paragraphs = [
        span
        for left, right in pairwise(boundaries)
        if (span := _trimmed_span(text, left, right)) is not None
    ]
    if len(paragraphs) <= 1:
        return [(start, end)]
    groups: list[tuple[int, int]] = []
    group_start, group_end = paragraphs[0]
    for paragraph_start, paragraph_end in paragraphs[1:]:
        if paragraph_end - group_start <= MAX_PARAGRAPH_GROUP_CHARS:
            group_end = paragraph_end
        else:
            groups.append((group_start, group_end))
            group_start, group_end = paragraph_start, paragraph_end
    groups.append((group_start, group_end))
    return groups


def split_narrative_evidence_item_v23(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Split one frozen parent chunk using filing-text structure only."""

    content = item.get("content")
    evidence_id = item.get("evidence_id")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Narrative evidence content must be non-empty text")
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise ValueError("Narrative evidence_id must be non-empty text")
    starts = sorted({0, *(match.start() for match in _SECTION_RE.finditer(content))})
    section_spans: list[tuple[int, int]] = []
    for start, end in zip(starts, [*starts[1:], len(content)]):
        span = _trimmed_span(content, start, end)
        if span is not None:
            section_spans.extend(_paragraph_groups(content, *span))
    segments: list[dict[str, Any]] = []
    parent_metadata = item.get("metadata")
    metadata_base = (
        dict(parent_metadata) if isinstance(parent_metadata, Mapping) else {}
    )
    parent_char_start = metadata_base.get("char_start")
    for index, (start, end) in enumerate(section_spans, start=1):
        segment_text = content[start:end]
        section_match = _SECTION_RE.match(segment_text)
        section_label = (
            section_match.group(0).strip() if section_match is not None else None
        )
        segment_material = {
            "boundary_policy_version": CHUNK_BOUNDARY_POLICY_VERSION,
            "parent_evidence_id": evidence_id,
            "segment_index": index,
            "relative_char_start": start,
            "relative_char_end": end,
            "content": segment_text,
        }
        metadata = {
            **metadata_base,
            "parent_evidence_id": evidence_id,
            "boundary_policy_version": CHUNK_BOUNDARY_POLICY_VERSION,
            "segment_index": index,
            "relative_char_start": start,
            "relative_char_end": end,
            "section_label": section_label,
        }
        if isinstance(parent_char_start, int) and not isinstance(
            parent_char_start, bool
        ):
            metadata["char_start"] = parent_char_start + start
            metadata["char_end"] = parent_char_start + end
        segment = copy.deepcopy(dict(item))
        segment.update(
            {
                "evidence_id": "v23e-" + canonical_json_sha256(segment_material)[:24],
                "content": segment_text,
                "metadata": metadata,
            }
        )
        segments.append(segment)
    return segments


def _tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


def _bm25_select(
    query: str, items: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    documents = [_tokenize(str(item.get("content") or "")) for item in items]
    average_length = sum(map(len, documents)) / len(documents) if documents else 1.0
    average_length = average_length or 1.0
    frequencies: Counter[str] = Counter()
    for document in documents:
        frequencies.update(set(document))
    query_terms = Counter(_tokenize(query))
    ranked: list[tuple[float, int, Mapping[str, Any]]] = []
    for index, (item, document) in enumerate(zip(items, documents, strict=True)):
        term_frequency = Counter(document)
        score = 0.0
        for term, query_frequency in query_terms.items():
            frequency = term_frequency.get(term, 0)
            if not frequency:
                continue
            document_frequency = frequencies[term]
            inverse = math.log(
                1.0
                + (len(documents) - document_frequency + 0.5)
                / (document_frequency + 0.5)
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
            metadata["v23_bm25_score"] = round(score, 12)
        selected.append(materialized)
    return selected


def _new_task_id(
    parent_benchmark_id: str, parent_task_id: str, task: Mapping[str, Any]
) -> str:
    return (
        "v23-"
        + canonical_json_sha256(
            {
                "benchmark_version": BENCHMARK_VERSION_V23,
                "parent_benchmark_id": parent_benchmark_id,
                "parent_task_id": parent_task_id,
                "parent_task_sha256": canonical_json_sha256(task),
                "chunk_boundary_policy_version": CHUNK_BOUNDARY_POLICY_VERSION,
            }
        )[:24]
    )


def _convert_task(
    task: Mapping[str, Any],
    *,
    task_id: str,
    evidence_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    converted = copy.deepcopy(dict(task))
    converted["agent_task_input_version"] = "v2.3"
    converted["task_id"] = task_id
    converted["evidence_scope"]["evidence_ids"] = [
        str(item["evidence_id"]) for item in evidence_items
    ]
    converted["evidence_scope"]["max_items"] = MAX_EVIDENCE_ITEMS
    if converted["task_type"] == "quant_metric":
        metric_id = str(converted["task_parameters"]["metric_spec_id"])
        converted["metric_operation_contract"] = metric_operation_contract(
            metric_id, contract_version="v2.3"
        )
        converted["output_schema_id"] = "quantitative_answer_or_refusal.v2.3"
        converted["narrative_selection_policy"] = None
    else:
        converted["evidence_scope"]["selection_method"] = NARRATIVE_SELECTION_METHOD_V23
        converted["output_schema_id"] = "narrative_selection_or_refusal.v2.3"
        converted["refusal_policy"] = {
            "allowed_codes": ["NO_DIRECT_EVIDENCE", "PROMPT_INJECTION_DETECTED"]
        }
        converted["narrative_selection_policy"] = narrative_selection_contract(
            str(converted["narrative_subtype"])
        )
    validate_agent_task_input(converted)
    build_context_pack(converted, evidence_items, token_counter=lambda _: 0)
    return converted


def _convert_evidence(
    task: Mapping[str, Any], items: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if task["task_type"] == "quant_metric":
        return [copy.deepcopy(dict(item)) for item in items]
    segments = [
        segment for item in items for segment in split_narrative_evidence_item_v23(item)
    ]
    query = str(task["task_parameters"]["retrieval_query"])
    return _bm25_select(query, segments)


def _supporting_ids(
    answer: str, evidence_items: Sequence[Mapping[str, Any]]
) -> list[str]:
    return [
        str(item["evidence_id"])
        for item in evidence_items
        if is_contiguous_text_supported(answer, str(item.get("content") or ""))
    ]


def _convert_gold(
    gold: Mapping[str, Any],
    *,
    task_id: str,
    task: Mapping[str, Any],
    evidence_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    converted = copy.deepcopy(dict(gold))
    converted["gold_record_version"] = GOLD_RECORD_VERSION_V23
    converted["task_id"] = task_id
    target = converted.get("target")
    if not isinstance(target, dict):
        raise TypeError("Gold record target must be an object")
    if task["task_type"] == "narrative_citation":
        status = str(target.get("status"))
        negative_type = converted.get("strata", {}).get("negative_type")
        converted["evaluator_negative_cause"] = (
            negative_type if status == "REFUSAL" else None
        )
        converted["legacy_refusal_code"] = (
            target.get("refusal_code") if status == "REFUSAL" else None
        )
        if status == "REFUSAL":
            target["refusal_code"] = "NO_DIRECT_EVIDENCE"
            target["chunk_evidence_ids"] = []
        else:
            answer = target.get("answer_text")
            if not isinstance(answer, str) or not answer:
                raise ValueError("Answerable narrative gold lacks exact answer text")
            supporting = _supporting_ids(answer, evidence_items)
            if not supporting:
                raise ValueError(
                    f"v2.3 inference-visible scope no longer contains gold anchor for {task_id}"
                )
            maximum = int(task["narrative_selection_policy"]["max_extracts"])
            target["chunk_evidence_ids"] = supporting[:maximum]
    converted["source_task_sha256"] = canonical_json_sha256(task)
    return converted


def _proposal_for_example(
    old: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    evidence_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    proposal = copy.deepcopy(dict(old))
    proposal["task_id"] = task["task_id"]
    if task["task_type"] == "narrative_citation":
        if proposal["action"] == "REFUSE":
            proposal["refusal_code"] = "NO_DIRECT_EVIDENCE"
        else:
            answer = str(proposal["answer_text"])
            supporting = _supporting_ids(answer, evidence_items)
            if not supporting:
                raise ValueError("v2.3 few-shot answer lacks selected exact support")
            proposal["evidence_ids"] = supporting[:1]
            proposal["claims"] = [
                {
                    "claim_id": "claim-1",
                    "text": answer,
                    "evidence_ids": supporting[:1],
                    "supporting_text": answer,
                }
            ]
    validate_agent_proposal(proposal)
    return proposal


def _convert_example(
    example: Mapping[str, Any], parent_benchmark_id: str
) -> dict[str, Any]:
    old_task = example["task"]
    old_evidence = example["evidence_items"]
    example_task_id = (
        "v23d-"
        + canonical_json_sha256(
            {
                "parent_benchmark_id": parent_benchmark_id,
                "example_id": example["example_id"],
                "source_task_sha256": example["source_task_sha256"],
            }
        )[:24]
    )
    evidence = _convert_evidence(old_task, old_evidence)
    task = _convert_task(old_task, task_id=example_task_id, evidence_items=evidence)
    spec = task_operation_spec(task["task_type"], version="v2.3")
    observation = (
        evaluate_quant_evidence(task, evidence, visibility="MODEL_VISIBLE")
        if task["task_type"] == "quant_metric"
        else load_frozen_evidence(task, evidence, visibility="MODEL_VISIBLE")
    )
    return {
        "few_shot_example_version": FEW_SHOT_EXAMPLE_VERSION_V23,
        "task_id": task["task_id"],
        "example_id": "v23d-" + canonical_json_sha256(example["example_id"])[:20],
        "review_status": "PENDING_HUMAN_REVIEW",
        "template_id": example["template_id"],
        "task_family": example["task_family"],
        "task": task,
        "evidence_items": evidence,
        "plan_response": {
            "task_id": task["task_id"],
            "action": "CALL_TOOL",
            "tool_name": spec.operation,
            "tool_arguments": spec.expected_arguments(task),
        },
        "tool_observation": observation,
        "assistant_response": _proposal_for_example(
            example["assistant_response"], task=task, evidence_items=evidence
        ),
        "source_task_sha256": example["source_task_sha256"],
    }


def derive_agent_benchmark_v23(
    parent_dir: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Publish a no-overwrite v2.3 benchmark with frozen parent membership."""

    parent = Path(parent_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite benchmark directory: {destination}"
        )
    parent_manifest = _verify_parent(parent)
    cases = _read_jsonl(parent / "cases.jsonl")
    evidence_rows = _read_jsonl(parent / "evidence.jsonl")
    gold_rows = _read_jsonl(parent / "gold.jsonl")
    old_observations = _read_jsonl(parent / "verifier_observations.jsonl")
    if not (
        len(cases) == len(evidence_rows) == len(gold_rows) == len(old_observations)
    ):
        raise ValueError("Parent benchmark case artifacts have inconsistent counts")
    evidence_by_id = {str(row["task_id"]): row["items"] for row in evidence_rows}
    gold_by_id = {str(row["task_id"]): row for row in gold_rows}
    parent_id = str(parent_manifest["benchmark_id"])

    new_cases: list[dict[str, Any]] = []
    new_evidence: list[dict[str, Any]] = []
    new_gold: list[dict[str, Any]] = []
    new_observations: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    task_id_map: dict[str, str] = {}
    for old_task in cases:
        old_task_id = str(old_task["task_id"])
        task_id = _new_task_id(parent_id, old_task_id, old_task)
        task_id_map[old_task_id] = task_id
        evidence = _convert_evidence(old_task, evidence_by_id[old_task_id])
        task = _convert_task(old_task, task_id=task_id, evidence_items=evidence)
        gold = _convert_gold(
            gold_by_id[old_task_id], task_id=task_id, task=task, evidence_items=evidence
        )
        spec = task_operation_spec(task["task_type"], version="v2.3")
        observation = (
            evaluate_quant_evidence(task, evidence)
            if task["task_type"] == "quant_metric"
            else load_frozen_evidence(task, evidence)
        )
        new_cases.append(task)
        new_evidence.append({"task_id": task_id, "items": evidence})
        new_gold.append(gold)
        new_observations.append(
            {
                "task_id": task_id,
                "tool_name": spec.operation,
                "observation": observation,
            }
        )
        lineage.append(
            {
                "task_id": task_id,
                "parent_task_id": old_task_id,
                "parent_task_sha256": canonical_json_sha256(old_task),
                "parent_benchmark_id": parent_id,
            }
        )

    few_shot = [
        _convert_example(example, parent_id)
        for example in _read_jsonl(parent / "few_shot.jsonl")
    ]
    old_parity = _read_json(parent / "parity_50.json")
    parity_ids = [task_id_map[str(task_id)] for task_id in old_parity["task_ids"]]
    parity = {
        **old_parity,
        "parity_slice_version": "auditops-agent-parity-slice.v2.3",
        "task_ids": parity_ids,
        "task_ids_sha256": canonical_json_sha256(parity_ids),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        counts = {
            "cases.jsonl": _write_jsonl(temporary / "cases.jsonl", new_cases),
            "evidence.jsonl": _write_jsonl(temporary / "evidence.jsonl", new_evidence),
            "gold.jsonl": _write_jsonl(temporary / "gold.jsonl", new_gold),
            "verifier_observations.jsonl": _write_jsonl(
                temporary / "verifier_observations.jsonl", new_observations
            ),
            "few_shot.jsonl": _write_jsonl(temporary / "few_shot.jsonl", few_shot),
            "case_lineage.jsonl": _write_jsonl(
                temporary / "case_lineage.jsonl", lineage
            ),
        }
        _write_json(temporary / "parity_50.json", parity)
        source_manifest = {
            "source_manifest_version": "v2.3",
            "parent_benchmark": {
                "benchmark_id": parent_id,
                "benchmark_version": parent_manifest["benchmark_version"],
                "benchmark_manifest_sha256": _sha256_path(
                    parent / "benchmark_manifest.json"
                ),
            },
            "derivation": {
                "benchmark_version": BENCHMARK_VERSION_V23,
                "chunk_boundary_policy_version": CHUNK_BOUNDARY_POLICY_VERSION,
                "chunk_boundary_policy_sha256": canonical_json_sha256(
                    {
                        "version": CHUNK_BOUNDARY_POLICY_VERSION,
                        "section_pattern": _SECTION_RE.pattern,
                        "max_paragraph_group_chars": MAX_PARAGRAPH_GROUP_CHARS,
                        "ranking": "stdlib_okapi_bm25_parent_top5_segments",
                        "top_k": MAX_EVIDENCE_ITEMS,
                    }
                ),
                "gold_used_for_selection": False,
            },
        }
        _write_json(temporary / "source_manifest.json", source_manifest)
        artifacts = {
            name: _artifact(temporary / name, counts.get(name))
            for name in (
                "cases.jsonl",
                "evidence.jsonl",
                "gold.jsonl",
                "verifier_observations.jsonl",
                "few_shot.jsonl",
                "case_lineage.jsonl",
                "source_manifest.json",
                "parity_50.json",
            )
        }
        manifest_material = {
            "benchmark_version": BENCHMARK_VERSION_V23,
            "benchmark_profile": parent_manifest["benchmark_profile"],
            "seed": parent_manifest["seed"],
            "corpus_id": str(parent_manifest["corpus_id"]) + ".v2.3-section-split",
            "jurisdiction": parent_manifest["jurisdiction"],
            "reporting_framework": parent_manifest["reporting_framework"],
            "standards_version": parent_manifest["standards_version"],
            "source_system": parent_manifest["source_system"],
            "counts": {
                **parent_manifest["counts"],
                "evidence_items": sum(len(row["items"]) for row in new_evidence),
            },
            "parent_benchmark_id": parent_id,
            "chunk_boundary_policy_version": CHUNK_BOUNDARY_POLICY_VERSION,
            "narrative_refusal_policy": {
                "model_facing_code": "NO_DIRECT_EVIDENCE",
                "evaluator_negative_cause_visible_to_model": False,
            },
            "narrative_selection_policies": {
                subtype: narrative_selection_contract(subtype)
                for subtype in sorted(
                    {
                        str(task["narrative_subtype"])
                        for task in new_cases
                        if task["task_type"] == "narrative_citation"
                    }
                )
            },
            "few_shot_policy": {
                **parent_manifest["few_shot_policy"],
                "requires_human_review": True,
                "approval_status": "PENDING_HUMAN_REVIEW_V2.3",
            },
            "stratification": parent_manifest["stratification"],
            "artifact_visibility": parent_manifest["artifact_visibility"],
            "artifacts": artifacts,
        }
        manifest = {
            **manifest_material,
            "benchmark_id": canonical_json_sha256(manifest_material),
        }
        _write_json(temporary / "benchmark_manifest.json", manifest)
        verification = verify_agent_benchmark_v23(temporary)
        if not verification["valid"]:
            raise ValueError(
                "Invalid v2.3 benchmark: " + "; ".join(verification["errors"])
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


def verify_agent_benchmark_v23(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    errors: list[str] = []
    try:
        manifest = _read_json(root / "benchmark_manifest.json")
        if manifest.get("benchmark_version") != BENCHMARK_VERSION_V23:
            errors.append("unsupported benchmark_version")
        material = {
            key: value for key, value in manifest.items() if key != "benchmark_id"
        }
        if manifest.get("benchmark_id") != canonical_json_sha256(material):
            errors.append("benchmark_id mismatch")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping):
            errors.append("artifact registry missing")
            artifacts = {}
        for name, expected in artifacts.items():
            path = root / str(name)
            if not path.is_file() or not isinstance(expected, Mapping):
                errors.append(f"artifact missing: {name}")
                continue
            if _sha256_path(path) != expected.get(
                "sha256"
            ) or path.stat().st_size != expected.get("bytes"):
                errors.append(f"artifact mismatch: {name}")
        cases = _read_jsonl(root / "cases.jsonl")
        evidence_rows = _read_jsonl(root / "evidence.jsonl")
        gold_rows = _read_jsonl(root / "gold.jsonl")
        observations = _read_jsonl(root / "verifier_observations.jsonl")
        evidence_by_id = {str(row["task_id"]): row["items"] for row in evidence_rows}
        gold_by_id = {str(row["task_id"]): row for row in gold_rows}
        observation_by_id = {str(row["task_id"]): row for row in observations}
        case_ids = [str(task["task_id"]) for task in cases]
        if case_ids != [str(row["task_id"]) for row in evidence_rows] or case_ids != [
            str(row["task_id"]) for row in gold_rows
        ]:
            errors.append("case/evidence/gold order mismatch")
        for task in cases:
            task_id = str(task["task_id"])
            items = evidence_by_id[task_id]
            try:
                validate_agent_task_input(task)
                build_context_pack(task, items, token_counter=lambda _: 0)
            except (TypeError, ValueError) as exc:
                errors.append(f"invalid task/context {task_id}: {exc}")
                continue
            gold = gold_by_id[task_id]
            observation = observation_by_id[task_id]["observation"]
            expected = (
                evaluate_quant_evidence(task, items)
                if task["task_type"] == "quant_metric"
                else load_frozen_evidence(task, items)
            )
            if observation != expected:
                errors.append(f"verifier observation mismatch: {task_id}")
            if task["task_type"] == "narrative_citation":
                if task["refusal_policy"]["allowed_codes"] != [
                    "NO_DIRECT_EVIDENCE",
                    "PROMPT_INJECTION_DETECTED",
                ]:
                    errors.append(f"narrative refusal policy mismatch: {task_id}")
                target = gold["target"]
                if target["status"] == "REFUSAL":
                    if target["refusal_code"] != "NO_DIRECT_EVIDENCE":
                        errors.append(f"narrative refusal target mismatch: {task_id}")
                else:
                    answer = str(target.get("answer_text") or "")
                    cited = set(target.get("chunk_evidence_ids") or [])
                    supported = any(
                        str(item["evidence_id"]) in cited
                        and is_contiguous_text_supported(answer, str(item["content"]))
                        for item in items
                    )
                    if not supported:
                        errors.append(f"narrative gold support mismatch: {task_id}")
        if any(
            key in json.dumps(cases).casefold()
            for key in ("gold_answer", "expected_answer", "target_answer")
        ):
            errors.append("gold-like field leaked into cases")
    except Exception as exc:  # noqa: BLE001 - verifier reports typed diagnostics
        errors.append(f"verification exception: {type(exc).__name__}: {exc}")
    return {"valid": not errors, "errors": errors}


__all__ = [
    "BENCHMARK_VERSION_V23",
    "CHUNK_BOUNDARY_POLICY_VERSION",
    "derive_agent_benchmark_v23",
    "split_narrative_evidence_item_v23",
    "verify_agent_benchmark_v23",
]
