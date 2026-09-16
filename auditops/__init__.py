"""AuditOps public API with dependency-light lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "answer_narrative": (".narrative_tasks", "answer_narrative"),
    "answer_quant": (".runtime", "answer_quant"),
    "build_agent_benchmark": (".agent_benchmark", "build_agent_benchmark"),
    "build_bm25_retriever": (".retrieval", "build_bm25_retriever"),
    "build_companies_house_corpus": (
        ".companies_house",
        "build_companies_house_corpus",
    ),
    "build_companies_house_vlm_diagnostic": (
        ".companies_house",
        "build_companies_house_vlm_diagnostic",
    ),
    "build_manifest": (".corpus", "build_manifest"),
    "build_narrative_benchmark": (".narrative_tasks", "build_narrative_benchmark"),
    "build_retrieval_benchmark_examples": (
        ".retrieval",
        "build_retrieval_benchmark_examples",
    ),
    "build_task_specs": (".tasks", "build_task_specs"),
    "build_uk_manifest": (".uk_corpus", "build_uk_manifest"),
    "build_us_audit_narrative_tasks": (
        ".audit_narrative",
        "build_us_audit_narrative_tasks",
    ),
    "download_filings": (".corpus", "download_filings"),
    "download_uk_filings": (".uk_corpus", "download_uk_filings"),
    "eval_corpus": (".corpus", "eval_corpus"),
    "evaluate_agent_run": (".agent_evaluation", "evaluate_agent_run"),
    "evaluate_companies_house_vlm_diagnostic": (
        ".vlm_evaluation",
        "evaluate_companies_house_vlm_diagnostic",
    ),
    "evaluate_bm25_retrieval": (".retrieval", "evaluate_bm25_retrieval"),
    "evaluate_narrative_answers": (
        ".narrative_tasks",
        "evaluate_narrative_answers",
    ),
    "evaluate_narrative_citations": (
        ".narrative_tasks",
        "evaluate_narrative_citations",
    ),
    "evaluate_narrative_routing": (
        ".narrative_tasks",
        "evaluate_narrative_routing",
    ),
    "execute_task_plan": (".runtime", "execute_task_plan"),
    "fetch_uk_constituents": (".bootstrap_inputs", "fetch_uk_constituents"),
    "fetch_us_constituents": (".bootstrap_inputs", "fetch_us_constituents"),
    "generate_corpus_datasets": (".corpus", "generate_corpus_datasets"),
    "ingest_corpus": (".corpus", "ingest_corpus"),
    "import_offline_us_cache": (".offline_us", "import_offline_us_cache"),
    "load_retrieval_examples": (".retrieval", "load_retrieval_examples"),
    "materialize_agent_source_evidence": (
        ".agent_evidence",
        "materialize_agent_source_evidence",
    ),
    "process_zip": (".pipeline", "process_zip"),
    "read_narrative_task_specs": (
        ".narrative_tasks",
        "read_narrative_task_specs",
    ),
    "rebuild_canonical_layers": (".pipeline", "rebuild_canonical_layers"),
    "render_quant_datasets": (".tasks", "render_quant_datasets"),
    "repair_corpus_filing": (".corpus", "repair_corpus_filing"),
    "run_agent_baseline": (".agent_batch", "run_agent_baseline"),
    "run_companies_house_vlm_diagnostic": (
        ".vlm_runtime",
        "run_companies_house_vlm_diagnostic",
    ),
    "verify_benchmark_artifacts": (
        ".agent_benchmark",
        "verify_benchmark_artifacts",
    ),
    "write_evaluation_artifacts": (
        ".agent_evaluation",
        "write_evaluation_artifacts",
    ),
    "write_few_shot_approval": (".agent_batch", "write_few_shot_approval"),
    "write_vlm_few_shot_approval": (
        ".vlm_runtime",
        "write_vlm_few_shot_approval",
    ),
    "write_narrative_task_specs": (
        ".narrative_tasks",
        "write_narrative_task_specs",
    ),
    "write_retrieval_examples": (".retrieval", "write_retrieval_examples"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
