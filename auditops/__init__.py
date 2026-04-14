"""AuditOps package."""

from .corpus import build_manifest, download_filings, eval_corpus, generate_corpus_datasets, ingest_corpus, repair_corpus_filing
from .pipeline import process_zip, rebuild_canonical_layers
from .narrative_tasks import (
    answer_narrative,
    build_narrative_benchmark,
    evaluate_narrative_answers,
    evaluate_narrative_citations,
    evaluate_narrative_routing,
    read_narrative_task_specs,
    write_narrative_task_specs,
)
from .retrieval import (
    build_bm25_retriever,
    build_retrieval_benchmark_examples,
    evaluate_bm25_retrieval,
    load_retrieval_examples,
    write_retrieval_examples,
)
from .runtime import answer_quant, execute_task_plan
from .tasks import build_task_specs, render_quant_datasets
from .uk_corpus import build_uk_manifest, download_uk_filings

__all__ = [
    "answer_quant",
    "build_manifest",
    "answer_narrative",
    "build_task_specs",
    "build_retrieval_benchmark_examples",
    "build_narrative_benchmark",
    "download_filings",
    "eval_corpus",
    "evaluate_narrative_answers",
    "evaluate_narrative_citations",
    "evaluate_narrative_routing",
    "evaluate_bm25_retrieval",
    "execute_task_plan",
    "generate_corpus_datasets",
    "ingest_corpus",
    "load_retrieval_examples",
    "read_narrative_task_specs",
    "repair_corpus_filing",
    "process_zip",
    "rebuild_canonical_layers",
    "render_quant_datasets",
    "build_bm25_retriever",
    "write_narrative_task_specs",
    "write_retrieval_examples",
    "build_uk_manifest",
    "download_uk_filings",
]
