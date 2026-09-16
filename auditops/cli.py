from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .agent_batch import (
    run_agent_baseline,
    write_few_shot_approval,
    write_few_shot_review_packet,
)
from .agent_benchmark import (
    BENCHMARK_PROFILES,
    build_agent_benchmark,
    verify_benchmark_artifacts,
)
from .agent_benchmark_v23 import (
    derive_agent_benchmark_v23,
    verify_agent_benchmark_v23,
)
from .agent_benchmark_v24 import (
    draft_agent_benchmark_v24,
    draft_agent_benchmark_v24r2,
    freeze_agent_benchmark_v24,
    render_agent_benchmark_review_v24,
    verify_agent_benchmark_candidate_v24,
    verify_agent_benchmark_v24,
    write_agent_benchmark_approval_v24,
)
from .agent_evaluation import write_evaluation_artifacts
from .agent_evaluation_v23 import write_evaluation_artifacts_v23
from .agent_evaluation_v24 import write_evaluation_artifacts_v24
from .agent_evidence import materialize_agent_source_evidence
from .agent_preflight_v24 import preflight_agent_requests_v24
from .agent_provisional_v24p import (
    freeze_agent_benchmark_v24p,
    verify_agent_benchmark_v24p,
    write_exploratory_authorization_v24p,
    write_gate_provisional_review_v24p,
    write_provisional_acceptance_v24p,
    write_provisional_ai_review_v24p,
)
from .agent_review_v24 import (
    prepare_agent_benchmark_review_bundle_v24,
    verify_agent_benchmark_review_bundle_v24,
)
from .audit_narrative import build_us_audit_narrative_tasks
from .bootstrap_inputs import fetch_uk_constituents, fetch_us_constituents
from .companies_house import (
    build_companies_house_corpus,
    build_companies_house_vlm_diagnostic,
)
from .corpus import (
    build_manifest,
    download_filings,
    eval_corpus,
    generate_corpus_datasets,
    ingest_corpus,
    repair_corpus_filing,
)
from .grammar_probe import run_structured_output_probes
from .metrics import generate_answer_objects_from_db
from .model_config import write_pinned_model_config
from .narrative_tasks import (
    answer_narrative,
    build_narrative_benchmark,
    evaluate_narrative_answers,
    evaluate_narrative_citations,
    evaluate_narrative_routing,
    read_narrative_task_specs,
    write_narrative_task_specs,
)
from .offline_us import import_offline_us_cache
from .pipeline import (
    connect_db,
    fetch_validators,
    inspect_chunk_canon,
    process_zip,
    rebuild_canonical_layers,
)
from .provenance import scan_artifacts_for_secrets, write_source_manifest
from .retrieval import (
    build_retrieval_benchmark_examples,
    evaluate_bm25_retrieval,
    load_retrieval_examples,
    write_retrieval_examples,
)
from .runtime import answer_quant
from .tasks import render_quant_datasets, write_task_specs
from .uk_corpus import build_uk_manifest, download_uk_filings
from .vlm_evaluation import evaluate_companies_house_vlm_diagnostic
from .vlm_runtime import (
    run_companies_house_vlm_diagnostic,
    write_vlm_few_shot_approval,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AuditOps CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser(
        "ingest",
        help="Ingest an SEC iXBRL ZIP into SQLite and materialize canonical layers.",
    )
    ingest.add_argument(
        "--zip", required=True, help="Path to the SEC iXBRL ZIP package."
    )
    ingest.add_argument("--db", required=True, help="SQLite DB path.")
    ingest.add_argument("--ticker", default=None, help="Optional ticker override.")
    ingest.add_argument(
        "--extract-narrative",
        action="store_true",
        help="Extract canonical narrative chunks.",
    )
    ingest.add_argument(
        "--reset-db", action="store_true", help="Drop and recreate the database schema."
    )
    ingest.add_argument(
        "--form-type", default=None, help="Optional form type override."
    )
    ingest.add_argument(
        "--fiscal-year-focus",
        type=int,
        default=None,
        help="Optional fiscal year focus override.",
    )
    ingest.add_argument(
        "--fiscal-period-focus",
        default=None,
        help="Optional fiscal period focus override.",
    )
    ingest.add_argument(
        "--report-date",
        default=None,
        help="Optional report date override (YYYY-MM-DD).",
    )

    rebuild = sub.add_parser(
        "rebuild-canon", help="Rebuild facts_canon, chunk_canon, and validators_v0."
    )
    rebuild.add_argument("--db", required=True, help="SQLite DB path.")
    rebuild.add_argument("--filing-id", default=None, help="Optional filing id filter.")

    validate = sub.add_parser("validate", help="Print validators_v0 rows as JSON.")
    validate.add_argument("--db", required=True, help="SQLite DB path.")
    validate.add_argument(
        "--filing-id", default=None, help="Optional filing id filter."
    )

    answers = sub.add_parser(
        "generate-answers", help="Write deterministic answer objects to JSONL."
    )
    answers.add_argument("--db", required=True, help="SQLite DB path.")
    answers.add_argument("--output", required=True, help="Output JSONL path.")
    answers.add_argument("--filing-id", default=None, help="Optional filing id filter.")

    task_specs = sub.add_parser(
        "generate-task-specs", help="Write deterministic quant TaskSpecs to JSONL."
    )
    task_specs.add_argument("--db", required=True, help="SQLite DB path.")
    task_specs.add_argument("--output", required=True, help="Output JSONL path.")
    task_specs.add_argument(
        "--filing-id", default=None, help="Optional filing id filter."
    )

    render = sub.add_parser(
        "render-quant-datasets",
        help="Render quant QA/code/refusal datasets and holdout manifests.",
    )
    render.add_argument("--db", required=True, help="SQLite DB path.")
    render.add_argument(
        "--output-dir", required=True, help="Output directory for rendered JSONL files."
    )
    render.add_argument("--filing-id", default=None, help="Optional filing id filter.")

    fetch_us_cmd = sub.add_parser(
        "fetch-us-constituents",
        help="Fetch a public S&P 500 constituents snapshot and write a CSV for build-manifest.",
    )
    fetch_us_cmd.add_argument("--output", required=True, help="Output CSV path.")
    fetch_us_cmd.add_argument(
        "--snapshot-date",
        default=None,
        help=(
            "Optional UTC as-of date in YYYY-MM-DD format. The source revision is "
            "resolved at or before 23:59:59Z; this is not merely a label."
        ),
    )
    fetch_us_cmd.add_argument(
        "--limit", type=int, default=None, help="Optional row limit for smoke testing."
    )
    fetch_us_cmd.add_argument(
        "--user-agent",
        default=None,
        help=(
            "Contact-bearing SEC identity; defaults to AUDITOPS_SEC_USER_AGENT and "
            "fails closed when absent."
        ),
    )

    import_offline_us_cmd = sub.add_parser(
        "import-offline-us-corpus",
        help=(
            "Import a pre-existing SEC 10-K iXBRL cache into a new, "
            "content-addressed corpus without network access."
        ),
    )
    import_offline_us_cmd.add_argument(
        "--source-root",
        required=True,
        help="Directory containing one validated cached filing directory per issuer.",
    )
    import_offline_us_cmd.add_argument(
        "--corpus-root",
        required=True,
        help="New immutable corpus root to create.",
    )
    import_offline_us_cmd.add_argument(
        "--snapshot-date",
        required=True,
        help="Nonfuture YYYY-MM-DD cutoff applied to every cached filing.",
    )
    import_offline_us_cmd.add_argument(
        "--corpus-id",
        required=True,
        help="Stable identifier that clearly labels the cached-source corpus.",
    )

    fetch_uk_cmd = sub.add_parser(
        "fetch-uk-constituents",
        help="Fetch a public FTSE 100 constituents snapshot and write a CSV for build-uk-manifest.",
    )
    fetch_uk_cmd.add_argument("--output", required=True, help="Output CSV path.")
    fetch_uk_cmd.add_argument(
        "--snapshot-date",
        default=None,
        help="Optional snapshot date label in YYYY-MM-DD format.",
    )
    fetch_uk_cmd.add_argument(
        "--limit", type=int, default=None, help="Optional row limit for smoke testing."
    )

    build_manifest_cmd = sub.add_parser(
        "build-manifest",
        help="Build a frozen current-universe manifest and latest filing targets.",
    )
    build_manifest_cmd.add_argument(
        "--constituents",
        required=True,
        help="Path to the dated constituents CSV snapshot.",
    )
    build_manifest_cmd.add_argument(
        "--corpus-root", required=True, help="Corpus root directory."
    )
    build_manifest_cmd.add_argument(
        "--snapshot-date", required=True, help="Snapshot date in YYYY-MM-DD format."
    )
    build_manifest_cmd.add_argument(
        "--trailing-fiscal-years",
        type=int,
        default=1,
        help="Number of trailing fiscal years of 10-K/10-Q filings to target per issuer. Default: latest-only (1).",
    )
    build_manifest_cmd.add_argument(
        "--constituent-source",
        default=None,
        help="Optional label for the constituents snapshot source.",
    )
    build_manifest_cmd.add_argument(
        "--user-agent", default=None, help="Optional SEC user agent override."
    )
    build_manifest_cmd.add_argument(
        "--allow-limited-constituents",
        action="store_true",
        help="Allow an explicitly limited constituent source manifest for smoke tests only.",
    )

    build_uk_manifest_cmd = sub.add_parser(
        "build-uk-manifest",
        help="Build a frozen FTSE-style UK manifest and latest FCA NSM report targets.",
    )
    build_uk_manifest_cmd.add_argument(
        "--constituents",
        required=True,
        help="Path to the dated UK constituents CSV snapshot.",
    )
    build_uk_manifest_cmd.add_argument(
        "--corpus-root", required=True, help="Corpus root directory."
    )
    build_uk_manifest_cmd.add_argument(
        "--snapshot-date", required=True, help="Snapshot date in YYYY-MM-DD format."
    )
    build_uk_manifest_cmd.add_argument(
        "--constituent-source",
        default=None,
        help="Optional label for the constituents snapshot source.",
    )
    build_uk_manifest_cmd.add_argument(
        "--user-agent", default=None, help="Optional FCA user agent override."
    )
    build_uk_manifest_cmd.add_argument(
        "--search-size",
        type=int,
        default=200,
        help="Maximum FCA NSM hits to scan per issuer/report query.",
    )

    download_cmd = sub.add_parser(
        "download-filings", help="Download filing ZIPs from a frozen filing manifest."
    )
    download_cmd.add_argument(
        "--corpus-root", required=True, help="Corpus root directory."
    )
    download_cmd.add_argument(
        "--user-agent", default=None, help="Optional SEC user agent override."
    )

    download_uk_cmd = sub.add_parser(
        "download-uk-filings",
        help="Archive FCA NSM details and attempt raw UK report downloads from a frozen UK manifest.",
    )
    download_uk_cmd.add_argument(
        "--corpus-root", required=True, help="Corpus root directory."
    )
    download_uk_cmd.add_argument(
        "--user-agent", default=None, help="Optional FCA user agent override."
    )

    ingest_corpus_cmd = sub.add_parser(
        "ingest-corpus", help="Ingest all downloaded filing ZIPs into one corpus DB."
    )
    ingest_corpus_cmd.add_argument(
        "--corpus-root", required=True, help="Corpus root directory."
    )
    ingest_corpus_cmd.add_argument(
        "--no-extract-narrative",
        action="store_true",
        help="Disable narrative extraction during corpus ingest.",
    )
    ingest_corpus_cmd.add_argument(
        "--replace-existing-db",
        action="store_true",
        help="Build a staged DB and atomically replace an existing corpus DB.",
    )

    repair_cmd = sub.add_parser(
        "repair-filing",
        help="Repair a single failed or missing corpus ingest and update the ingest ledger.",
    )
    repair_cmd.add_argument(
        "--corpus-root", required=True, help="Corpus root directory."
    )
    repair_cmd.add_argument("--ticker", required=True, help="Ticker to repair.")
    repair_cmd.add_argument(
        "--accession",
        required=True,
        help="SEC accession number for the filing to repair.",
    )
    repair_cmd.add_argument(
        "--no-extract-narrative",
        action="store_true",
        help="Disable narrative extraction during filing repair.",
    )

    datasets_cmd = sub.add_parser(
        "generate-corpus-datasets",
        help="Generate answer objects, task specs, datasets, and split manifests for a corpus.",
    )
    datasets_cmd.add_argument(
        "--corpus-root", required=True, help="Corpus root directory."
    )

    eval_cmd = sub.add_parser(
        "eval-corpus", help="Run data-quality and runtime evaluation for a corpus."
    )
    eval_cmd.add_argument("--corpus-root", required=True, help="Corpus root directory.")

    answer = sub.add_parser(
        "answer-quant",
        help="Route a constrained quant question and execute the deterministic runtime.",
    )
    answer.add_argument("--db", required=True, help="SQLite DB path.")
    answer.add_argument(
        "--filing-id", required=True, help="Filing id to route against."
    )
    answer.add_argument("--question", required=True, help="Quant question to answer.")

    inspect = sub.add_parser(
        "inspect-narrative", help="Preview canonical narrative chunks for a filing."
    )
    inspect.add_argument("--db", required=True, help="SQLite DB path.")
    inspect.add_argument("--filing-id", required=True, help="Filing id to inspect.")
    inspect.add_argument(
        "--limit", type=int, default=5, help="Number of chunks to print."
    )

    retrieval = sub.add_parser(
        "eval-retrieval", help="Evaluate BM25 retrieval against gold chunk ids."
    )
    retrieval.add_argument("--db", required=True, help="SQLite DB path.")
    retrieval.add_argument(
        "--examples", required=True, help="JSONL file of retrieval examples."
    )
    retrieval.add_argument(
        "--method",
        choices=("bm25", "bm25_rerank"),
        default="bm25_rerank",
        help="Retrieval method to evaluate.",
    )
    retrieval.add_argument(
        "--top-k", type=int, default=5, help="Top-k cutoff for retrieval metrics."
    )
    retrieval.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Candidate pool size for reranking.",
    )
    retrieval.add_argument("--output", default=None, help="Optional JSON output path.")

    build_retrieval = sub.add_parser(
        "build-retrieval-benchmark",
        help="Build deterministic retrieval examples from chunk_canon.",
    )
    build_retrieval.add_argument("--db", required=True, help="SQLite DB path.")
    build_retrieval.add_argument(
        "--output", required=True, help="Output JSONL path for retrieval examples."
    )
    build_retrieval.add_argument(
        "--limit", type=int, default=250, help="Maximum number of retrieval examples."
    )
    build_retrieval.add_argument(
        "--per-filing-limit",
        type=int,
        default=1,
        help="Maximum number of examples per filing.",
    )

    build_narrative = sub.add_parser(
        "build-narrative-benchmark",
        help="Build deterministic narrative citation tasks from chunk_canon.",
    )
    build_narrative.add_argument("--db", required=True, help="SQLite DB path.")
    build_narrative.add_argument(
        "--output", required=True, help="Output JSONL path for narrative task specs."
    )
    build_narrative.add_argument(
        "--limit", type=int, default=200, help="Maximum number of narrative task specs."
    )
    build_narrative.add_argument(
        "--per-filing-limit",
        type=int,
        default=1,
        help="Maximum number of narrative tasks per filing.",
    )
    build_narrative.add_argument(
        "--include-unanswerable",
        action="store_true",
        help="Include deterministic unanswerable/refusal tasks.",
    )
    build_narrative.add_argument(
        "--unanswerable-limit",
        type=int,
        default=None,
        help="Optional cap for unanswerable narrative tasks.",
    )

    eval_narrative = sub.add_parser(
        "eval-narrative-citations",
        help="Evaluate citation retrieval for narrative task specs.",
    )
    eval_narrative.add_argument("--db", required=True, help="SQLite DB path.")
    eval_narrative.add_argument(
        "--tasks", required=True, help="JSONL file of narrative task specs."
    )
    eval_narrative.add_argument(
        "--method",
        choices=("bm25", "bm25_rerank"),
        default="bm25_rerank",
        help="Retrieval method to evaluate.",
    )
    eval_narrative.add_argument(
        "--top-k", type=int, default=5, help="Top-k cutoff for citation coverage."
    )
    eval_narrative.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Candidate pool size for reranking.",
    )
    eval_narrative.add_argument(
        "--output", default=None, help="Optional JSON output path."
    )

    answer_narrative_cmd = sub.add_parser(
        "answer-narrative",
        help="Answer a constrained narrative benchmark question with chunk citations or a refusal.",
    )
    answer_narrative_cmd.add_argument("--db", required=True, help="SQLite DB path.")
    answer_narrative_cmd.add_argument(
        "--tasks", required=True, help="JSONL file of narrative task specs."
    )
    answer_narrative_cmd.add_argument(
        "--filing-id", required=True, help="Filing id to route against."
    )
    answer_narrative_cmd.add_argument(
        "--question", required=True, help="Narrative question to answer."
    )
    answer_narrative_cmd.add_argument(
        "--method",
        choices=("bm25", "bm25_rerank"),
        default="bm25_rerank",
        help="Retrieval method to use.",
    )
    answer_narrative_cmd.add_argument(
        "--top-k", type=int, default=5, help="Top-k cutoff for retrieval."
    )
    answer_narrative_cmd.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Candidate pool size for reranking.",
    )

    eval_narrative_answers_cmd = sub.add_parser(
        "eval-narrative-answers",
        help="Evaluate the deterministic narrative answer runtime against narrative task specs.",
    )
    eval_narrative_answers_cmd.add_argument(
        "--db", required=True, help="SQLite DB path."
    )
    eval_narrative_answers_cmd.add_argument(
        "--tasks", required=True, help="JSONL file of narrative task specs."
    )
    eval_narrative_answers_cmd.add_argument(
        "--method",
        choices=("bm25", "bm25_rerank"),
        default="bm25_rerank",
        help="Retrieval method to use.",
    )
    eval_narrative_answers_cmd.add_argument(
        "--top-k", type=int, default=5, help="Top-k cutoff for retrieval."
    )
    eval_narrative_answers_cmd.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Candidate pool size for reranking.",
    )
    eval_narrative_answers_cmd.add_argument(
        "--output", default=None, help="Optional JSON output path."
    )

    eval_narrative_routing_cmd = sub.add_parser(
        "eval-narrative-routing",
        help="Evaluate narrative routing on deterministic loose-question variants.",
    )
    eval_narrative_routing_cmd.add_argument(
        "--db", required=True, help="SQLite DB path."
    )
    eval_narrative_routing_cmd.add_argument(
        "--tasks", required=True, help="JSONL file of narrative task specs."
    )
    eval_narrative_routing_cmd.add_argument(
        "--variants-per-task",
        type=int,
        default=1,
        help="Number of deterministic loose-question variants to generate per task.",
    )
    eval_narrative_routing_cmd.add_argument(
        "--method",
        choices=("bm25", "bm25_rerank"),
        default="bm25_rerank",
        help="Retrieval method to use.",
    )
    eval_narrative_routing_cmd.add_argument(
        "--top-k", type=int, default=5, help="Top-k cutoff for retrieval."
    )
    eval_narrative_routing_cmd.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Candidate pool size for reranking.",
    )
    eval_narrative_routing_cmd.add_argument(
        "--output", default=None, help="Optional JSON output path."
    )

    build_ch_corpus_cmd = sub.add_parser(
        "build-companies-house-corpus",
        help="Parse a Companies House bulk XHTML archive into immutable filing and task manifests.",
    )
    build_ch_corpus_cmd.add_argument(
        "--bulk-zip", required=True, help="Companies House monthly bulk accounts ZIP."
    )
    build_ch_corpus_cmd.add_argument(
        "--output-dir", required=True, help="New, empty corpus output directory."
    )
    build_ch_corpus_cmd.add_argument(
        "--snapshot-date", default="2026-01-31", help="Frozen source snapshot date."
    )
    build_ch_corpus_cmd.add_argument(
        "--seed", type=int, default=20260821, help="Deterministic filing-order seed."
    )
    build_ch_corpus_cmd.add_argument(
        "--max-filings",
        type=int,
        default=None,
        help="Optional filing cap for smoke runs.",
    )

    build_ch_vlm_cmd = sub.add_parser(
        "build-companies-house-vlm-diagnostic",
        help="Separate inference cases from gold labels for the paired image-PDF diagnostic.",
    )
    build_ch_vlm_cmd.add_argument(
        "--pair-manifest",
        required=True,
        help="CSV describing paired PDF/XHTML filings.",
    )
    build_ch_vlm_cmd.add_argument(
        "--fact-audit", required=True, help="CSV of inspected XHTML/PDF facts."
    )
    build_ch_vlm_cmd.add_argument(
        "--output-dir", required=True, help="New, empty diagnostic output directory."
    )
    build_ch_vlm_cmd.add_argument("--expected-pair-count", type=int, default=20)
    build_ch_vlm_cmd.add_argument("--expected-fact-count", type=int, default=266)

    approve_ch_vlm = sub.add_parser(
        "approve-companies-house-vlm-few-shot",
        help="Bind four human-checked, entity-disjoint visual few-shot examples.",
    )
    approve_ch_vlm.add_argument("--few-shot-diagnostic-dir", required=True)
    approve_ch_vlm.add_argument("--evaluation-diagnostic-dir", required=True)
    approve_ch_vlm.add_argument("--visual-examples-jsonl", required=True)
    approve_ch_vlm.add_argument("--output", required=True)
    approve_ch_vlm.add_argument("--reviewer", required=True)
    approve_ch_vlm.add_argument("--reviewed-at", required=True)

    run_ch_vlm = sub.add_parser(
        "run-companies-house-vlm-diagnostic",
        help="Run the isolated image-PDF diagnostic through in-process offline vLLM.",
    )
    run_ch_vlm.add_argument("--diagnostic-dir", required=True)
    run_ch_vlm.add_argument("--model-config", required=True)
    run_ch_vlm.add_argument("--output-dir", required=True)
    run_ch_vlm.add_argument("--pdf-root", required=True)
    run_ch_vlm.add_argument(
        "--prompt-condition", required=True, choices=("zero_shot", "few_shot")
    )
    run_ch_vlm.add_argument("--few-shot-diagnostic-dir", default=None)
    run_ch_vlm.add_argument("--visual-examples-jsonl", default=None)
    run_ch_vlm.add_argument("--few-shot-approval", default=None)
    run_ch_vlm.add_argument("--few-shot-pdf-root", default=None)

    eval_ch_vlm = sub.add_parser(
        "eval-companies-house-vlm-diagnostic",
        help="Evaluate image-PDF predictions separately from the XHTML benchmark.",
    )
    eval_ch_vlm.add_argument("--diagnostic-dir", required=True)
    eval_ch_vlm.add_argument("--predictions-jsonl", required=True)
    eval_ch_vlm.add_argument("--output-dir", required=True)

    materialize_evidence = sub.add_parser(
        "materialize-agent-evidence",
        help="Resolve canonical fact evidence and rank narrative chunks with deterministic BM25.",
    )
    materialize_evidence.add_argument("--quant-jsonl", required=True)
    materialize_evidence.add_argument("--narrative-jsonl", required=True)
    materialize_evidence.add_argument(
        "--db", required=True, help="Canonical AuditOps SQLite database."
    )
    materialize_evidence.add_argument(
        "--output-dir", required=True, help="New immutable evidence directory."
    )
    materialize_evidence.add_argument(
        "--top-k", type=int, default=5, choices=range(1, 6)
    )
    materialize_evidence.add_argument("--candidate-k", type=int, default=15)
    materialize_evidence.add_argument(
        "--source-system",
        default="SEC-EDGAR",
        choices=("SEC-EDGAR", "SEC-EDGAR-CACHED"),
    )

    build_us_audit_tasks = sub.add_parser(
        "build-us-audit-narrative-tasks",
        help="Build evidence-grounded auditor-opinion and CAM extraction/refusal tasks.",
    )
    build_us_audit_tasks.add_argument(
        "--db", required=True, help="Canonical SEC corpus database."
    )
    build_us_audit_tasks.add_argument(
        "--output", required=True, help="New immutable narrative JSONL."
    )
    build_us_audit_tasks.add_argument(
        "--base-narrative-jsonl",
        default=None,
        help="Optional validated filing/accounting-policy tasks to merge.",
    )
    build_us_audit_tasks.add_argument("--max-answer-chars", type=int, default=320)
    build_us_audit_tasks.add_argument(
        "--source-system",
        default="SEC-EDGAR",
        choices=("SEC-EDGAR", "SEC-EDGAR-CACHED"),
    )

    build_agent = sub.add_parser(
        "build-agent-benchmark",
        help="Freeze v2.2 inference, evidence, lineage, gold, and subtype-specific few-shot artifacts.",
    )
    build_agent.add_argument(
        "--quant-jsonl", required=True, help="Evidence-materialized quantitative tasks."
    )
    build_agent.add_argument(
        "--narrative-jsonl",
        required=True,
        help="Evidence-materialized narrative tasks.",
    )
    build_agent.add_argument(
        "--output-dir", required=True, help="New benchmark output directory."
    )
    build_agent.add_argument(
        "--benchmark-profile",
        required=True,
        choices=BENCHMARK_PROFILES,
        help=(
            "Immutable benchmark contract. Use us_v0.1 for the real 500-case "
            "SEC benchmark or synthetic only for compatibility fixtures."
        ),
    )
    build_agent.add_argument("--jurisdiction", required=True, choices=("US", "UK"))
    build_agent.add_argument("--corpus-id", required=True)
    build_agent.add_argument("--reporting-framework", required=True)
    build_agent.add_argument("--standards-version", required=True)
    build_agent.add_argument("--source-system", required=True)
    build_agent.add_argument("--seed", type=int, default=20260821)
    build_agent.add_argument("--quant-count", type=int, default=300)
    build_agent.add_argument("--narrative-count", type=int, default=200)
    build_agent.add_argument(
        "--parent-benchmark-dir",
        default=None,
        help=(
            "Frozen agent_benchmark.v2 or v2.1 directory whose exact evaluation "
            "membership is preserved while v2.2 few-shot packs are rebuilt."
        ),
    )

    verify_agent = sub.add_parser(
        "verify-agent-benchmark",
        help="Verify v2.2 benchmark hashes, counts, schemas, and split separation.",
    )
    verify_agent.add_argument("--benchmark-dir", required=True)

    derive_agent_v23 = sub.add_parser(
        "derive-agent-benchmark-v23",
        help=(
            "Derive v2.3 tasks and section-bounded narrative evidence while "
            "preserving frozen v2.2 evaluation membership."
        ),
    )
    derive_agent_v23.add_argument("--parent-benchmark-dir", required=True)
    derive_agent_v23.add_argument("--output-dir", required=True)

    verify_agent_v23 = sub.add_parser(
        "verify-agent-benchmark-v23",
        help="Verify a derived v2.3 benchmark, hashes, scopes, and gold separation.",
    )
    verify_agent_v23.add_argument("--benchmark-dir", required=True)

    draft_agent_v24 = sub.add_parser(
        "draft-agent-benchmark-v24",
        help="Build an immutable cached-20 v2.4 candidate and evaluator-only review material.",
    )
    draft_agent_v24.add_argument("--quant-jsonl", required=True)
    draft_agent_v24.add_argument("--corpus-db", required=True)
    draft_agent_v24.add_argument("--raw-filing-root", required=True)
    draft_agent_v24.add_argument("--output-dir", required=True)
    draft_agent_v24.add_argument(
        "--corpus-id", default="us_sec_existing20_cached20_v2.4-candidate"
    )
    draft_agent_v24.add_argument("--seed", type=int, default=20260821)

    draft_agent_v24r2 = sub.add_parser(
        "draft-agent-benchmark-v24r2",
        help="Build the hardened cached-20 r2 candidate from frozen r1 AI feedback.",
    )
    draft_agent_v24r2.add_argument("--quant-jsonl", required=True)
    draft_agent_v24r2.add_argument("--corpus-db", required=True)
    draft_agent_v24r2.add_argument("--raw-filing-root", required=True)
    draft_agent_v24r2.add_argument("--r1-candidate-dir", required=True)
    draft_agent_v24r2.add_argument("--r1-rendered-packet", required=True)
    draft_agent_v24r2.add_argument("--r1-primary-review", required=True)
    draft_agent_v24r2.add_argument("--r1-secondary-review", required=True)
    draft_agent_v24r2.add_argument("--r1-validation-summary", required=True)
    draft_agent_v24r2.add_argument("--output-dir", required=True)
    draft_agent_v24r2.add_argument(
        "--corpus-id", default="us_sec_existing20_cached20_v2.4r2-candidate"
    )
    draft_agent_v24r2.add_argument("--seed", type=int, default=20260821)

    verify_candidate_v24 = sub.add_parser(
        "verify-agent-benchmark-candidate-v24",
        help="Verify a v2.4 candidate without treating it as a benchmark.",
    )
    verify_candidate_v24.add_argument("--candidate-dir", required=True)

    render_review_v24 = sub.add_parser(
        "render-agent-benchmark-review-v24",
        help="Render the self-contained 200-case v2.4 human review packet.",
    )
    render_review_v24.add_argument("--candidate-dir", required=True)
    render_review_v24.add_argument("--output-html", required=True)

    prepare_review_bundle_v24 = sub.add_parser(
        "prepare-agent-benchmark-review-bundle-v24",
        help=(
            "Package the exact v2.4 review packet, filing HTML, refusal scopes, "
            "and fail-closed human decision templates."
        ),
    )
    prepare_review_bundle_v24.add_argument("--candidate-dir", required=True)
    prepare_review_bundle_v24.add_argument("--rendered-packet", required=True)
    prepare_review_bundle_v24.add_argument("--corpus-db", required=True)
    prepare_review_bundle_v24.add_argument("--raw-filing-root", required=True)
    prepare_review_bundle_v24.add_argument("--output-dir", required=True)

    verify_review_bundle_v24 = sub.add_parser(
        "verify-agent-benchmark-review-bundle-v24",
        help="Verify every checksum, refusal scope, membership, and template in a v2.4 review bundle.",
    )
    verify_review_bundle_v24.add_argument("--bundle-dir", required=True)

    approve_benchmark_v24 = sub.add_parser(
        "approve-agent-benchmark-v24",
        help="Bind primary, secondary, and optional subtype-wide adjudication reviews.",
    )
    approve_benchmark_v24.add_argument("--candidate-dir", required=True)
    approve_benchmark_v24.add_argument("--rendered-packet", required=True)
    approve_benchmark_v24.add_argument("--primary-review", required=True)
    approve_benchmark_v24.add_argument("--secondary-review", required=True)
    approve_benchmark_v24.add_argument("--adjudication-review", default=None)
    approve_benchmark_v24.add_argument("--output", required=True)

    freeze_benchmark_v24 = sub.add_parser(
        "freeze-agent-benchmark-v24",
        help="Freeze an externally human-approved v2.4 candidate.",
    )
    freeze_benchmark_v24.add_argument("--candidate-dir", required=True)
    freeze_benchmark_v24.add_argument("--approval", required=True)
    freeze_benchmark_v24.add_argument("--output-dir", required=True)

    verify_benchmark_v24 = sub.add_parser(
        "verify-agent-benchmark-v24",
        help="Verify a frozen, human-approved AuditOps v2.4 benchmark.",
    )
    verify_benchmark_v24.add_argument("--benchmark-dir", required=True)

    provisional_review_v24p = sub.add_parser(
        "record-provisional-ai-review-v24p",
        help="Record an explicitly non-human Codex review pass for candidate v2.4r2.",
    )
    provisional_review_v24p.add_argument("--candidate-dir", required=True)
    provisional_review_v24p.add_argument("--review-bundle-dir", required=True)
    provisional_review_v24p.add_argument(
        "--pass-kind", required=True, choices=("PRIMARY", "CRITIC")
    )
    provisional_review_v24p.add_argument("--output", required=True)

    provisional_acceptance_v24p = sub.add_parser(
        "accept-agent-benchmark-provisional-v24p",
        help="Accept r2 only after complete agreeing primary and critic AI passes.",
    )
    provisional_acceptance_v24p.add_argument("--candidate-dir", required=True)
    provisional_acceptance_v24p.add_argument("--primary-review", required=True)
    provisional_acceptance_v24p.add_argument("--critic-review", required=True)
    provisional_acceptance_v24p.add_argument("--output", required=True)

    freeze_v24p = sub.add_parser(
        "freeze-agent-benchmark-provisional-v24p",
        help="Freeze an accepted r2 candidate as an exploratory v2.4p benchmark.",
    )
    freeze_v24p.add_argument("--candidate-dir", required=True)
    freeze_v24p.add_argument("--acceptance", required=True)
    freeze_v24p.add_argument("--output-dir", required=True)

    verify_v24p = sub.add_parser(
        "verify-agent-benchmark-v24p",
        help="Verify an exploratory, provisionally AI-reviewed v2.4p benchmark.",
    )
    verify_v24p.add_argument("--benchmark-dir", required=True)

    gate_review_v24p = sub.add_parser(
        "record-gate-provisional-review-v24p",
        help="Record the explicitly AI-only released-quote review for a v2.4p gate.",
    )
    gate_review_v24p.add_argument("--benchmark-dir", required=True)
    gate_review_v24p.add_argument("--run-dir", required=True)
    gate_review_v24p.add_argument("--release-review-packet", required=True)
    gate_review_v24p.add_argument("--output", required=True)

    authorize_v24p = sub.add_parser(
        "authorize-exploratory-full-run-v24p",
        help="Bind an explicit authorizer's GO_EXPLORATORY decision to passing v2.4p gate metrics.",
    )
    authorize_v24p.add_argument("--benchmark-dir", required=True)
    authorize_v24p.add_argument("--gate-metrics", required=True)
    authorize_v24p.add_argument("--authorized-by", required=True)
    authorize_v24p.add_argument("--output", required=True)

    preflight_v24 = sub.add_parser(
        "preflight-agent-requests-v24",
        help="Render and count every exact zero-shot Gemma narrative request on CPU.",
    )
    preflight_v24.add_argument("--benchmark-dir", required=True)
    preflight_v24.add_argument("--model-config", required=True)
    preflight_v24.add_argument("--output-dir", required=True)

    approve_examples = sub.add_parser(
        "approve-few-shot",
        help="Record v2.2 human approval of every frozen few-shot example.",
    )
    approve_examples.add_argument("--benchmark-dir", required=True)
    approve_examples.add_argument("--output", required=True)
    approve_examples.add_argument("--reviewer", required=True)
    approve_examples.add_argument(
        "--reviewed-at", required=True, help="Review timestamp, preferably RFC 3339."
    )
    approve_examples.add_argument("--example-id", action="append", required=True)
    approve_examples.add_argument(
        "--review-packet",
        required=True,
        help="Human-reviewed v2.2 rendered-message and tokenizer-count packet.",
    )

    review_examples = sub.add_parser(
        "render-few-shot-review",
        help="Render v2.2 demonstrations and count them with both pinned tokenizers.",
    )
    review_examples.add_argument("--benchmark-dir", required=True)
    review_examples.add_argument(
        "--model-config", action="append", required=True, dest="model_configs"
    )
    review_examples.add_argument("--output", required=True)

    model_config = sub.add_parser(
        "write-model-config",
        help="Create a deterministic offline-vLLM configuration for a pinned local snapshot.",
    )
    model_config.add_argument("--output", required=True)
    model_config.add_argument("--model-id", required=True)
    model_config.add_argument(
        "--revision", required=True, help="Exact 40-character Hugging Face commit hash."
    )
    model_config.add_argument(
        "--model-path", required=True, help="Resolved local snapshot directory."
    )
    model_config.add_argument(
        "--quantization",
        required=True,
        help="Quantization label, or 'none' for an unquantized BF16 snapshot.",
    )

    run_agent = sub.add_parser(
        "run-agent-baseline",
        help="Run the v2.2 safety-hybrid baseline or a direct/capability research ablation through offline vLLM.",
    )
    run_agent.add_argument("--benchmark-dir", required=True)
    run_agent.add_argument("--model-config", required=True)
    run_agent.add_argument(
        "--runtime-mode",
        default="safety_hybrid",
        choices=("direct", "capability_agent", "safety_hybrid"),
    )
    run_agent.add_argument(
        "--prompt-condition", required=True, choices=("zero_shot", "few_shot")
    )
    run_agent.add_argument(
        "--output-dir", required=True, help="New immutable run directory."
    )
    run_agent.add_argument("--few-shot-approval", default=None)
    run_agent.add_argument(
        "--selection-mode", choices=("full", "narrative_gate"), default=None
    )
    run_agent.add_argument("--full-run-authorization", default=None)
    run_agent.add_argument("--expected-authorizer", default=None)
    run_agent.add_argument(
        "--limit", type=int, default=None, help="Optional smoke/parity slice size."
    )

    run_gate_v24 = sub.add_parser(
        "run-agent-narrative-gate-v24",
        help="Run the frozen 50-case v2.4 narrative-only safety gate.",
    )
    run_gate_v24.add_argument("--benchmark-dir", required=True)
    run_gate_v24.add_argument("--model-config", required=True)
    run_gate_v24.add_argument("--output-dir", required=True)

    grammar_probe = sub.add_parser(
        "probe-agent-structured-output",
        help="Run synthetic v2.2 structured-generation probes without filing data.",
    )
    grammar_probe.add_argument("--model-config", required=True)
    grammar_probe.add_argument(
        "--output-dir", required=True, help="New immutable probe directory."
    )

    for command_name, command_help in (
        (
            "eval-agent-baseline",
            "Evaluate frozen run JSONL files and write metrics plus a report.",
        ),
        (
            "report-agent-baseline",
            "Generate the durable metrics, failures, and Markdown baseline report.",
        ),
    ):
        evaluation = sub.add_parser(command_name, help=command_help)
        evaluation.add_argument("--benchmark-dir", required=True)
        evaluation.add_argument(
            "--run",
            action="append",
            required=True,
            metavar="NAME=RESULTS_JSONL",
            help="Named run; repeat for comparisons.",
        )
        evaluation.add_argument(
            "--output-dir", required=True, help="New immutable evaluation directory."
        )
        evaluation.add_argument(
            "--quantization-pair",
            action="append",
            default=[],
            metavar="BF16,QUANTIZED[,NAME]",
            help="Optional matched parity comparison; repeat as needed.",
        )

    for command_name in (
        "eval-agent-baseline-v23",
        "report-agent-baseline-v23",
    ):
        evaluation_v23 = sub.add_parser(
            command_name,
            help=(
                "Evaluate v2.3 run directories with separate narrative support, "
                "citation, wording, and filing-clustered metrics."
            ),
        )
        evaluation_v23.add_argument("--benchmark-dir", required=True)
        evaluation_v23.add_argument(
            "--run-dir", action="append", required=True, help="Repeat for comparisons."
        )
        evaluation_v23.add_argument("--output-dir", required=True)

    for command_name in (
        "eval-agent-baseline-v24",
        "report-agent-baseline-v24",
    ):
        evaluation_v24 = sub.add_parser(
            command_name,
            help="Evaluate a v2.4 narrative gate or authorized full cached-20 run.",
        )
        evaluation_v24.add_argument("--benchmark-dir", required=True)
        evaluation_v24.add_argument("--run-dir", action="append", required=True)
        evaluation_v24.add_argument(
            "--human-release-review",
            action="append",
            default=[],
            help="Optional human review JSON; repeat one-for-one with --run-dir.",
        )
        evaluation_v24.add_argument(
            "--provisional-release-review",
            action="append",
            default=[],
            help=(
                "Optional v2.4p AI review JSON; repeat one-for-one with --run-dir. "
                "This never satisfies the v2.4 human-review interface."
            ),
        )
        evaluation_v24.add_argument("--output-dir", required=True)

    source_manifest = sub.add_parser(
        "write-source-manifest",
        help="Hash the Git base, diff, and complete deployable source tree.",
    )
    source_manifest.add_argument("--repo-root", default=".")
    source_manifest.add_argument("--output", required=True)

    secret_scan = sub.add_parser(
        "scan-artifacts",
        help="Fail when repository or result artifacts contain credential material.",
    )
    secret_scan.add_argument("--path", action="append", required=True)

    return parser


def _parse_named_runs(values: Sequence[str]) -> dict[str, str]:
    runs: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--run must use NAME=RESULTS_JSONL")
        name, path = value.split("=", 1)
        if not name.strip() or not path.strip():
            raise ValueError("--run requires a non-empty name and path")
        if name in runs:
            raise ValueError(f"Duplicate run name: {name}")
        runs[name] = path
    return runs


def _parse_quantization_pairs(values: Sequence[str]) -> list[tuple[str, ...]]:
    pairs: list[tuple[str, ...]] = []
    for value in values:
        fields = tuple(part.strip() for part in value.split(","))
        if len(fields) not in {2, 3} or any(not field for field in fields):
            raise ValueError("--quantization-pair must use BF16,QUANTIZED[,NAME]")
        pairs.append(fields)
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        overrides = {
            key: value
            for key, value in {
                "form_type": args.form_type,
                "fiscal_year_focus": args.fiscal_year_focus,
                "fiscal_period_focus": args.fiscal_period_focus,
                "report_date": args.report_date,
            }.items()
            if value is not None
        }
        summary = process_zip(
            zip_path=args.zip,
            out_db=args.db,
            ticker=args.ticker,
            extract_narrative=args.extract_narrative,
            reset_db=args.reset_db,
            metadata_overrides=overrides,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "rebuild-canon":
        conn = connect_db(args.db)
        try:
            rebuild_canonical_layers(conn, filing_id=args.filing_id)
        finally:
            conn.close()
        print("canonical layers rebuilt")
        return 0

    if args.command == "validate":
        conn = connect_db(args.db)
        try:
            for row in fetch_validators(conn, filing_id=args.filing_id):
                print(json.dumps(dict(row), sort_keys=True))
        finally:
            conn.close()
        return 0

    if args.command == "generate-answers":
        count = generate_answer_objects_from_db(
            args.db, args.output, filing_id=args.filing_id
        )
        print(f"wrote {count} answer objects")
        return 0

    if args.command == "generate-task-specs":
        conn = connect_db(args.db)
        try:
            count = write_task_specs(conn, args.output, filing_id=args.filing_id)
        finally:
            conn.close()
        print(f"wrote {count} task specs")
        return 0

    if args.command == "render-quant-datasets":
        conn = connect_db(args.db)
        try:
            summary = render_quant_datasets(
                conn, args.output_dir, filing_id=args.filing_id
            )
        finally:
            conn.close()
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "fetch-us-constituents":
        summary = fetch_us_constituents(
            args.output,
            snapshot_date=args.snapshot_date,
            limit=args.limit,
            user_agent=args.user_agent,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "import-offline-us-corpus":
        summary = import_offline_us_cache(
            args.source_root,
            args.corpus_root,
            snapshot_id=args.corpus_id,
            snapshot_date=args.snapshot_date,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "fetch-uk-constituents":
        summary = fetch_uk_constituents(
            args.output, snapshot_date=args.snapshot_date, limit=args.limit
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "build-manifest":
        summary = build_manifest(
            constituents_path=args.constituents,
            corpus_root=args.corpus_root,
            snapshot_date=args.snapshot_date,
            trailing_fiscal_years=args.trailing_fiscal_years,
            constituent_source=args.constituent_source,
            user_agent=args.user_agent,
            allow_limited_constituents=args.allow_limited_constituents,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "build-uk-manifest":
        summary = build_uk_manifest(
            constituents_path=args.constituents,
            corpus_root=args.corpus_root,
            snapshot_date=args.snapshot_date,
            constituent_source=args.constituent_source,
            user_agent=args.user_agent,
            search_size=args.search_size,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "download-filings":
        summary = download_filings(args.corpus_root, user_agent=args.user_agent)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1 if summary.get("error_count", 0) else 0

    if args.command == "download-uk-filings":
        summary = download_uk_filings(args.corpus_root, user_agent=args.user_agent)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "ingest-corpus":
        summary = ingest_corpus(
            args.corpus_root,
            extract_narrative=not args.no_extract_narrative,
            replace_existing=args.replace_existing_db,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return (
            1
            if summary.get("error_count", 0) or not summary.get("published", False)
            else 0
        )

    if args.command == "repair-filing":
        summary = repair_corpus_filing(
            args.corpus_root,
            ticker=args.ticker,
            accession=args.accession,
            extract_narrative=not args.no_extract_narrative,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "generate-corpus-datasets":
        summary = generate_corpus_datasets(args.corpus_root)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-corpus":
        summary = eval_corpus(args.corpus_root)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "answer-quant":
        conn = connect_db(args.db)
        try:
            response = answer_quant(conn, args.question, args.filing_id)
        finally:
            conn.close()
        print(json.dumps(response, indent=2, sort_keys=True))
        return 0

    if args.command == "inspect-narrative":
        conn = connect_db(args.db)
        try:
            rows = inspect_chunk_canon(conn, args.filing_id, limit=args.limit)
        finally:
            conn.close()
        for row in rows:
            print(json.dumps(dict(row), sort_keys=True))
        return 0

    if args.command == "eval-retrieval":
        examples = load_retrieval_examples(args.examples)
        result = evaluate_bm25_retrieval(
            args.db,
            examples,
            top_k=args.top_k,
            method=args.method,
            candidate_k=args.candidate_k,
        )
        if args.output:
            Path(args.output).write_text(
                json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
            )
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
        return 0

    if args.command == "build-retrieval-benchmark":
        examples = build_retrieval_benchmark_examples(
            args.db,
            limit=args.limit,
            per_filing_limit=args.per_filing_limit,
        )
        write_retrieval_examples(args.output, examples)
        summary = {
            "output": args.output,
            "query_count": len(examples),
            "limit": args.limit,
            "per_filing_limit": args.per_filing_limit,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "build-narrative-benchmark":
        task_specs = build_narrative_benchmark(
            args.db,
            limit=args.limit,
            per_filing_limit=args.per_filing_limit,
            include_unanswerable=args.include_unanswerable,
            unanswerable_limit=args.unanswerable_limit,
        )
        write_narrative_task_specs(args.output, task_specs)
        summary = {
            "output": args.output,
            "task_count": len(task_specs),
            "limit": args.limit,
            "per_filing_limit": args.per_filing_limit,
            "include_unanswerable": args.include_unanswerable,
            "unanswerable_limit": args.unanswerable_limit,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-narrative-citations":
        task_specs = read_narrative_task_specs(args.tasks)
        result = evaluate_narrative_citations(
            args.db,
            task_specs,
            top_k=args.top_k,
            method=args.method,
            candidate_k=args.candidate_k,
        )
        if args.output:
            Path(args.output).write_text(
                json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
            )
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
        return 0

    if args.command == "answer-narrative":
        task_specs = read_narrative_task_specs(args.tasks)
        result = answer_narrative(
            args.db,
            args.question,
            args.filing_id,
            task_specs=task_specs,
            top_k=args.top_k,
            method=args.method,
            candidate_k=args.candidate_k,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-narrative-answers":
        task_specs = read_narrative_task_specs(args.tasks)
        result = evaluate_narrative_answers(
            args.db,
            task_specs,
            top_k=args.top_k,
            method=args.method,
            candidate_k=args.candidate_k,
        )
        if args.output:
            Path(args.output).write_text(
                json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
            )
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
        return 0

    if args.command == "eval-narrative-routing":
        task_specs = read_narrative_task_specs(args.tasks)
        result = evaluate_narrative_routing(
            args.db,
            task_specs,
            variants_per_task=args.variants_per_task,
            top_k=args.top_k,
            method=args.method,
            candidate_k=args.candidate_k,
        )
        if args.output:
            Path(args.output).write_text(
                json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
            )
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
        return 0

    if args.command == "materialize-agent-evidence":
        result = materialize_agent_source_evidence(
            args.quant_jsonl,
            args.narrative_jsonl,
            args.db,
            args.output_dir,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            source_system=args.source_system,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "build-us-audit-narrative-tasks":
        result = build_us_audit_narrative_tasks(
            args.db,
            args.output,
            base_narrative_jsonl=args.base_narrative_jsonl,
            max_answer_chars=args.max_answer_chars,
            source_system=args.source_system,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "build-agent-benchmark":
        result = build_agent_benchmark(
            args.quant_jsonl,
            args.narrative_jsonl,
            args.output_dir,
            benchmark_profile=args.benchmark_profile,
            jurisdiction=args.jurisdiction,
            corpus_id=args.corpus_id,
            reporting_framework=args.reporting_framework,
            standards_version=args.standards_version,
            source_system=args.source_system,
            seed=args.seed,
            quant_count=args.quant_count,
            narrative_count=args.narrative_count,
            parent_benchmark_dir=args.parent_benchmark_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "verify-agent-benchmark":
        result = verify_benchmark_artifacts(args.benchmark_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1

    if args.command == "derive-agent-benchmark-v23":
        result = derive_agent_benchmark_v23(
            args.parent_benchmark_dir,
            args.output_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "verify-agent-benchmark-v23":
        result = verify_agent_benchmark_v23(args.benchmark_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1

    if args.command == "draft-agent-benchmark-v24":
        result = draft_agent_benchmark_v24(
            args.quant_jsonl,
            args.corpus_db,
            args.output_dir,
            raw_filing_root=args.raw_filing_root,
            corpus_id=args.corpus_id,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "draft-agent-benchmark-v24r2":
        result = draft_agent_benchmark_v24r2(
            args.quant_jsonl,
            args.corpus_db,
            args.output_dir,
            raw_filing_root=args.raw_filing_root,
            r1_candidate_dir=args.r1_candidate_dir,
            r1_rendered_packet=args.r1_rendered_packet,
            r1_primary_review=args.r1_primary_review,
            r1_secondary_review=args.r1_secondary_review,
            r1_validation_summary=args.r1_validation_summary,
            corpus_id=args.corpus_id,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "verify-agent-benchmark-candidate-v24":
        result = verify_agent_benchmark_candidate_v24(args.candidate_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1

    if args.command == "render-agent-benchmark-review-v24":
        result = render_agent_benchmark_review_v24(args.candidate_dir, args.output_html)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "prepare-agent-benchmark-review-bundle-v24":
        result = prepare_agent_benchmark_review_bundle_v24(
            args.candidate_dir,
            args.rendered_packet,
            args.corpus_db,
            args.raw_filing_root,
            args.output_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "verify-agent-benchmark-review-bundle-v24":
        result = verify_agent_benchmark_review_bundle_v24(args.bundle_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1

    if args.command == "approve-agent-benchmark-v24":
        result = write_agent_benchmark_approval_v24(
            args.candidate_dir,
            args.rendered_packet,
            args.primary_review,
            args.secondary_review,
            args.output,
            adjudication_review=args.adjudication_review,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "freeze-agent-benchmark-v24":
        result = freeze_agent_benchmark_v24(
            args.candidate_dir, args.approval, args.output_dir
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "verify-agent-benchmark-v24":
        result = verify_agent_benchmark_v24(args.benchmark_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1

    if args.command == "record-provisional-ai-review-v24p":
        result = write_provisional_ai_review_v24p(
            args.candidate_dir,
            args.review_bundle_dir,
            args.output,
            pass_kind=args.pass_kind,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["rejections"] == 0 else 1

    if args.command == "accept-agent-benchmark-provisional-v24p":
        result = write_provisional_acceptance_v24p(
            args.candidate_dir,
            args.primary_review,
            args.critic_review,
            args.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "freeze-agent-benchmark-provisional-v24p":
        result = freeze_agent_benchmark_v24p(
            args.candidate_dir, args.acceptance, args.output_dir
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "verify-agent-benchmark-v24p":
        result = verify_agent_benchmark_v24p(args.benchmark_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1

    if args.command == "record-gate-provisional-review-v24p":
        result = write_gate_provisional_review_v24p(
            args.benchmark_dir,
            args.run_dir,
            args.release_review_packet,
            args.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["confirmed"] else 1

    if args.command == "authorize-exploratory-full-run-v24p":
        result = write_exploratory_authorization_v24p(
            args.benchmark_dir,
            args.gate_metrics,
            args.output,
            authorized_by=args.authorized_by,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "preflight-agent-requests-v24":
        result = preflight_agent_requests_v24(
            args.benchmark_dir, args.model_config, args.output_dir
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1

    if args.command == "approve-few-shot":
        result = write_few_shot_approval(
            args.benchmark_dir,
            args.output,
            reviewer=args.reviewer,
            approved_example_ids=args.example_id,
            reviewed_at=args.reviewed_at,
            review_packet_path=args.review_packet,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "write-model-config":
        quantization = (
            None
            if args.quantization.strip().lower() in {"none", "null", "bf16"}
            else args.quantization
        )
        result = write_pinned_model_config(
            args.output,
            model_id=args.model_id,
            revision=args.revision,
            model_path=args.model_path,
            quantization=quantization,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "run-agent-baseline":
        result = run_agent_baseline(
            args.benchmark_dir,
            args.model_config,
            args.output_dir,
            runtime_mode=args.runtime_mode,
            prompt_condition=args.prompt_condition,
            few_shot_approval=args.few_shot_approval,
            limit=args.limit,
            selection_mode=args.selection_mode,
            full_run_authorization=args.full_run_authorization,
            expected_authorizer=args.expected_authorizer,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "run-agent-narrative-gate-v24":
        result = run_agent_baseline(
            args.benchmark_dir,
            args.model_config,
            args.output_dir,
            runtime_mode="safety_hybrid",
            prompt_condition="zero_shot",
            selection_mode="narrative_gate",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "render-few-shot-review":
        result = write_few_shot_review_packet(
            args.benchmark_dir, args.model_configs, args.output
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1

    if args.command == "probe-agent-structured-output":
        result = run_structured_output_probes(args.model_config, args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1

    if args.command in {"eval-agent-baseline", "report-agent-baseline"}:
        try:
            run_files = _parse_named_runs(args.run)
            quantization_pairs = _parse_quantization_pairs(args.quantization_pair)
        except ValueError as exc:
            parser.error(str(exc))
        result = write_evaluation_artifacts(
            args.benchmark_dir,
            run_files,
            args.output_dir,
            quantization_pairs=quantization_pairs,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command in {"eval-agent-baseline-v23", "report-agent-baseline-v23"}:
        result = write_evaluation_artifacts_v23(
            args.benchmark_dir,
            args.run_dir,
            args.output_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command in {"eval-agent-baseline-v24", "report-agent-baseline-v24"}:
        human_reviews: list[str | None]
        if args.human_release_review:
            if len(args.human_release_review) != len(args.run_dir):
                parser.error(
                    "--human-release-review must be omitted or repeated once per --run-dir"
                )
            human_reviews = list(args.human_release_review)
        else:
            human_reviews = [None] * len(args.run_dir)
        provisional_reviews: list[str | None]
        if args.provisional_release_review:
            if len(args.provisional_release_review) != len(args.run_dir):
                parser.error(
                    "--provisional-release-review must be omitted or repeated once per --run-dir"
                )
            provisional_reviews = list(args.provisional_release_review)
        else:
            provisional_reviews = [None] * len(args.run_dir)
        if args.human_release_review and args.provisional_release_review:
            parser.error(
                "human and provisional release-review interfaces are mutually exclusive"
            )
        result = write_evaluation_artifacts_v24(
            args.benchmark_dir,
            args.run_dir,
            args.output_dir,
            human_release_reviews=human_reviews,
            provisional_release_reviews=provisional_reviews,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "write-source-manifest":
        result = write_source_manifest(args.repo_root, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "scan-artifacts":
        findings = scan_artifacts_for_secrets(args.path)
        result = {
            "passed": not findings,
            "scanned_paths": [str(Path(path).resolve()) for path in args.path],
            "finding_count": len(findings),
            "findings": findings,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not findings else 1

    if args.command == "build-companies-house-corpus":
        result = build_companies_house_corpus(
            args.bulk_zip,
            args.output_dir,
            snapshot_date=args.snapshot_date,
            seed=args.seed,
            max_filings=args.max_filings,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "build-companies-house-vlm-diagnostic":
        result = build_companies_house_vlm_diagnostic(
            args.pair_manifest,
            args.fact_audit,
            args.output_dir,
            expected_pair_count=args.expected_pair_count,
            expected_fact_count=args.expected_fact_count,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "approve-companies-house-vlm-few-shot":
        result = write_vlm_few_shot_approval(
            args.few_shot_diagnostic_dir,
            args.evaluation_diagnostic_dir,
            args.visual_examples_jsonl,
            args.output,
            reviewer=args.reviewer,
            reviewed_at=args.reviewed_at,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "run-companies-house-vlm-diagnostic":
        result = run_companies_house_vlm_diagnostic(
            args.diagnostic_dir,
            args.model_config,
            args.output_dir,
            pdf_root=args.pdf_root,
            prompt_condition=args.prompt_condition,
            few_shot_diagnostic_dir=args.few_shot_diagnostic_dir,
            visual_examples_jsonl=args.visual_examples_jsonl,
            few_shot_approval=args.few_shot_approval,
            few_shot_pdf_root=args.few_shot_pdf_root,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-companies-house-vlm-diagnostic":
        result = evaluate_companies_house_vlm_diagnostic(
            args.diagnostic_dir,
            args.predictions_jsonl,
            args.output_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
