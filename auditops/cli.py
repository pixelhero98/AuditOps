from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .bootstrap_inputs import fetch_uk_constituents, fetch_us_constituents
from .corpus import build_manifest, download_filings, eval_corpus, generate_corpus_datasets, ingest_corpus, repair_corpus_filing
from .metrics import generate_answer_objects_from_db
from .narrative_tasks import (
    answer_narrative,
    build_narrative_benchmark,
    evaluate_narrative_answers,
    evaluate_narrative_citations,
    evaluate_narrative_routing,
    read_narrative_task_specs,
    write_narrative_task_specs,
)
from .pipeline import connect_db, fetch_validators, inspect_chunk_canon, process_zip, rebuild_canonical_layers
from .retrieval import (
    build_retrieval_benchmark_examples,
    evaluate_bm25_retrieval,
    load_retrieval_examples,
    write_retrieval_examples,
)
from .runtime import answer_quant
from .tasks import render_quant_datasets, write_task_specs
from .uk_corpus import build_uk_manifest, download_uk_filings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AuditOps CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest an SEC iXBRL ZIP into SQLite and materialize canonical layers.")
    ingest.add_argument("--zip", required=True, help="Path to the SEC iXBRL ZIP package.")
    ingest.add_argument("--db", required=True, help="SQLite DB path.")
    ingest.add_argument("--ticker", default=None, help="Optional ticker override.")
    ingest.add_argument("--extract-narrative", action="store_true", help="Extract canonical narrative chunks.")
    ingest.add_argument("--reset-db", action="store_true", help="Drop and recreate the database schema.")
    ingest.add_argument("--form-type", default=None, help="Optional form type override.")
    ingest.add_argument("--fiscal-year-focus", type=int, default=None, help="Optional fiscal year focus override.")
    ingest.add_argument("--fiscal-period-focus", default=None, help="Optional fiscal period focus override.")
    ingest.add_argument("--report-date", default=None, help="Optional report date override (YYYY-MM-DD).")

    rebuild = sub.add_parser("rebuild-canon", help="Rebuild facts_canon, chunk_canon, and validators_v0.")
    rebuild.add_argument("--db", required=True, help="SQLite DB path.")
    rebuild.add_argument("--filing-id", default=None, help="Optional filing id filter.")

    validate = sub.add_parser("validate", help="Print validators_v0 rows as JSON.")
    validate.add_argument("--db", required=True, help="SQLite DB path.")
    validate.add_argument("--filing-id", default=None, help="Optional filing id filter.")

    answers = sub.add_parser("generate-answers", help="Write deterministic answer objects to JSONL.")
    answers.add_argument("--db", required=True, help="SQLite DB path.")
    answers.add_argument("--output", required=True, help="Output JSONL path.")
    answers.add_argument("--filing-id", default=None, help="Optional filing id filter.")

    task_specs = sub.add_parser("generate-task-specs", help="Write deterministic quant TaskSpecs to JSONL.")
    task_specs.add_argument("--db", required=True, help="SQLite DB path.")
    task_specs.add_argument("--output", required=True, help="Output JSONL path.")
    task_specs.add_argument("--filing-id", default=None, help="Optional filing id filter.")

    render = sub.add_parser("render-quant-datasets", help="Render quant QA/code/refusal datasets and holdout manifests.")
    render.add_argument("--db", required=True, help="SQLite DB path.")
    render.add_argument("--output-dir", required=True, help="Output directory for rendered JSONL files.")
    render.add_argument("--filing-id", default=None, help="Optional filing id filter.")

    fetch_us_cmd = sub.add_parser("fetch-us-constituents", help="Fetch a public S&P 500 constituents snapshot and write a CSV for build-manifest.")
    fetch_us_cmd.add_argument("--output", required=True, help="Output CSV path.")
    fetch_us_cmd.add_argument("--snapshot-date", default=None, help="Optional snapshot date label in YYYY-MM-DD format.")
    fetch_us_cmd.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke testing.")

    fetch_uk_cmd = sub.add_parser("fetch-uk-constituents", help="Fetch a public FTSE 100 constituents snapshot and write a CSV for build-uk-manifest.")
    fetch_uk_cmd.add_argument("--output", required=True, help="Output CSV path.")
    fetch_uk_cmd.add_argument("--snapshot-date", default=None, help="Optional snapshot date label in YYYY-MM-DD format.")
    fetch_uk_cmd.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke testing.")

    build_manifest_cmd = sub.add_parser("build-manifest", help="Build a frozen current-universe manifest and latest filing targets.")
    build_manifest_cmd.add_argument("--constituents", required=True, help="Path to the dated constituents CSV snapshot.")
    build_manifest_cmd.add_argument("--corpus-root", required=True, help="Corpus root directory.")
    build_manifest_cmd.add_argument("--snapshot-date", required=True, help="Snapshot date in YYYY-MM-DD format.")
    build_manifest_cmd.add_argument(
        "--trailing-fiscal-years",
        type=int,
        default=1,
        help="Number of trailing fiscal years of 10-K/10-Q filings to target per issuer. Default: latest-only (1).",
    )
    build_manifest_cmd.add_argument("--constituent-source", default=None, help="Optional label for the constituents snapshot source.")
    build_manifest_cmd.add_argument("--user-agent", default=None, help="Optional SEC user agent override.")

    build_uk_manifest_cmd = sub.add_parser("build-uk-manifest", help="Build a frozen FTSE-style UK manifest and latest FCA NSM report targets.")
    build_uk_manifest_cmd.add_argument("--constituents", required=True, help="Path to the dated UK constituents CSV snapshot.")
    build_uk_manifest_cmd.add_argument("--corpus-root", required=True, help="Corpus root directory.")
    build_uk_manifest_cmd.add_argument("--snapshot-date", required=True, help="Snapshot date in YYYY-MM-DD format.")
    build_uk_manifest_cmd.add_argument("--constituent-source", default=None, help="Optional label for the constituents snapshot source.")
    build_uk_manifest_cmd.add_argument("--user-agent", default=None, help="Optional FCA user agent override.")
    build_uk_manifest_cmd.add_argument("--search-size", type=int, default=200, help="Maximum FCA NSM hits to scan per issuer/report query.")

    download_cmd = sub.add_parser("download-filings", help="Download filing ZIPs from a frozen filing manifest.")
    download_cmd.add_argument("--corpus-root", required=True, help="Corpus root directory.")
    download_cmd.add_argument("--user-agent", default=None, help="Optional SEC user agent override.")

    download_uk_cmd = sub.add_parser("download-uk-filings", help="Archive FCA NSM details and attempt raw UK report downloads from a frozen UK manifest.")
    download_uk_cmd.add_argument("--corpus-root", required=True, help="Corpus root directory.")
    download_uk_cmd.add_argument("--user-agent", default=None, help="Optional FCA user agent override.")

    ingest_corpus_cmd = sub.add_parser("ingest-corpus", help="Ingest all downloaded filing ZIPs into one corpus DB.")
    ingest_corpus_cmd.add_argument("--corpus-root", required=True, help="Corpus root directory.")
    ingest_corpus_cmd.add_argument("--no-extract-narrative", action="store_true", help="Disable narrative extraction during corpus ingest.")

    repair_cmd = sub.add_parser("repair-filing", help="Repair a single failed or missing corpus ingest and update the ingest ledger.")
    repair_cmd.add_argument("--corpus-root", required=True, help="Corpus root directory.")
    repair_cmd.add_argument("--ticker", required=True, help="Ticker to repair.")
    repair_cmd.add_argument("--accession", required=True, help="SEC accession number for the filing to repair.")
    repair_cmd.add_argument("--no-extract-narrative", action="store_true", help="Disable narrative extraction during filing repair.")

    datasets_cmd = sub.add_parser("generate-corpus-datasets", help="Generate answer objects, task specs, datasets, and split manifests for a corpus.")
    datasets_cmd.add_argument("--corpus-root", required=True, help="Corpus root directory.")

    eval_cmd = sub.add_parser("eval-corpus", help="Run data-quality and runtime evaluation for a corpus.")
    eval_cmd.add_argument("--corpus-root", required=True, help="Corpus root directory.")

    answer = sub.add_parser("answer-quant", help="Route a constrained quant question and execute the deterministic runtime.")
    answer.add_argument("--db", required=True, help="SQLite DB path.")
    answer.add_argument("--filing-id", required=True, help="Filing id to route against.")
    answer.add_argument("--question", required=True, help="Quant question to answer.")

    inspect = sub.add_parser("inspect-narrative", help="Preview canonical narrative chunks for a filing.")
    inspect.add_argument("--db", required=True, help="SQLite DB path.")
    inspect.add_argument("--filing-id", required=True, help="Filing id to inspect.")
    inspect.add_argument("--limit", type=int, default=5, help="Number of chunks to print.")

    retrieval = sub.add_parser("eval-retrieval", help="Evaluate BM25 retrieval against gold chunk ids.")
    retrieval.add_argument("--db", required=True, help="SQLite DB path.")
    retrieval.add_argument("--examples", required=True, help="JSONL file of retrieval examples.")
    retrieval.add_argument(
        "--method",
        choices=("bm25", "bm25_rerank"),
        default="bm25_rerank",
        help="Retrieval method to evaluate.",
    )
    retrieval.add_argument("--top-k", type=int, default=5, help="Top-k cutoff for retrieval metrics.")
    retrieval.add_argument("--candidate-k", type=int, default=None, help="Candidate pool size for reranking.")
    retrieval.add_argument("--output", default=None, help="Optional JSON output path.")

    build_retrieval = sub.add_parser("build-retrieval-benchmark", help="Build deterministic retrieval examples from chunk_canon.")
    build_retrieval.add_argument("--db", required=True, help="SQLite DB path.")
    build_retrieval.add_argument("--output", required=True, help="Output JSONL path for retrieval examples.")
    build_retrieval.add_argument("--limit", type=int, default=250, help="Maximum number of retrieval examples.")
    build_retrieval.add_argument("--per-filing-limit", type=int, default=1, help="Maximum number of examples per filing.")

    build_narrative = sub.add_parser("build-narrative-benchmark", help="Build deterministic narrative citation tasks from chunk_canon.")
    build_narrative.add_argument("--db", required=True, help="SQLite DB path.")
    build_narrative.add_argument("--output", required=True, help="Output JSONL path for narrative task specs.")
    build_narrative.add_argument("--limit", type=int, default=200, help="Maximum number of narrative task specs.")
    build_narrative.add_argument("--per-filing-limit", type=int, default=1, help="Maximum number of narrative tasks per filing.")
    build_narrative.add_argument("--include-unanswerable", action="store_true", help="Include deterministic unanswerable/refusal tasks.")
    build_narrative.add_argument("--unanswerable-limit", type=int, default=None, help="Optional cap for unanswerable narrative tasks.")

    eval_narrative = sub.add_parser("eval-narrative-citations", help="Evaluate citation retrieval for narrative task specs.")
    eval_narrative.add_argument("--db", required=True, help="SQLite DB path.")
    eval_narrative.add_argument("--tasks", required=True, help="JSONL file of narrative task specs.")
    eval_narrative.add_argument(
        "--method",
        choices=("bm25", "bm25_rerank"),
        default="bm25_rerank",
        help="Retrieval method to evaluate.",
    )
    eval_narrative.add_argument("--top-k", type=int, default=5, help="Top-k cutoff for citation coverage.")
    eval_narrative.add_argument("--candidate-k", type=int, default=None, help="Candidate pool size for reranking.")
    eval_narrative.add_argument("--output", default=None, help="Optional JSON output path.")

    answer_narrative_cmd = sub.add_parser("answer-narrative", help="Answer a constrained narrative benchmark question with chunk citations or a refusal.")
    answer_narrative_cmd.add_argument("--db", required=True, help="SQLite DB path.")
    answer_narrative_cmd.add_argument("--tasks", required=True, help="JSONL file of narrative task specs.")
    answer_narrative_cmd.add_argument("--filing-id", required=True, help="Filing id to route against.")
    answer_narrative_cmd.add_argument("--question", required=True, help="Narrative question to answer.")
    answer_narrative_cmd.add_argument(
        "--method",
        choices=("bm25", "bm25_rerank"),
        default="bm25_rerank",
        help="Retrieval method to use.",
    )
    answer_narrative_cmd.add_argument("--top-k", type=int, default=5, help="Top-k cutoff for retrieval.")
    answer_narrative_cmd.add_argument("--candidate-k", type=int, default=None, help="Candidate pool size for reranking.")

    eval_narrative_answers_cmd = sub.add_parser("eval-narrative-answers", help="Evaluate the deterministic narrative answer runtime against narrative task specs.")
    eval_narrative_answers_cmd.add_argument("--db", required=True, help="SQLite DB path.")
    eval_narrative_answers_cmd.add_argument("--tasks", required=True, help="JSONL file of narrative task specs.")
    eval_narrative_answers_cmd.add_argument(
        "--method",
        choices=("bm25", "bm25_rerank"),
        default="bm25_rerank",
        help="Retrieval method to use.",
    )
    eval_narrative_answers_cmd.add_argument("--top-k", type=int, default=5, help="Top-k cutoff for retrieval.")
    eval_narrative_answers_cmd.add_argument("--candidate-k", type=int, default=None, help="Candidate pool size for reranking.")
    eval_narrative_answers_cmd.add_argument("--output", default=None, help="Optional JSON output path.")

    eval_narrative_routing_cmd = sub.add_parser("eval-narrative-routing", help="Evaluate narrative routing on deterministic loose-question variants.")
    eval_narrative_routing_cmd.add_argument("--db", required=True, help="SQLite DB path.")
    eval_narrative_routing_cmd.add_argument("--tasks", required=True, help="JSONL file of narrative task specs.")
    eval_narrative_routing_cmd.add_argument("--variants-per-task", type=int, default=1, help="Number of deterministic loose-question variants to generate per task.")
    eval_narrative_routing_cmd.add_argument(
        "--method",
        choices=("bm25", "bm25_rerank"),
        default="bm25_rerank",
        help="Retrieval method to use.",
    )
    eval_narrative_routing_cmd.add_argument("--top-k", type=int, default=5, help="Top-k cutoff for retrieval.")
    eval_narrative_routing_cmd.add_argument("--candidate-k", type=int, default=None, help="Candidate pool size for reranking.")
    eval_narrative_routing_cmd.add_argument("--output", default=None, help="Optional JSON output path.")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
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
        count = generate_answer_objects_from_db(args.db, args.output, filing_id=args.filing_id)
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
            summary = render_quant_datasets(conn, args.output_dir, filing_id=args.filing_id)
        finally:
            conn.close()
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "fetch-us-constituents":
        summary = fetch_us_constituents(args.output, snapshot_date=args.snapshot_date, limit=args.limit)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "fetch-uk-constituents":
        summary = fetch_uk_constituents(args.output, snapshot_date=args.snapshot_date, limit=args.limit)
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
        return 0

    if args.command == "download-uk-filings":
        summary = download_uk_filings(args.corpus_root, user_agent=args.user_agent)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "ingest-corpus":
        summary = ingest_corpus(args.corpus_root, extract_narrative=not args.no_extract_narrative)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

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
            Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
        return 0

    if args.command == "build-retrieval-benchmark":
        examples = build_retrieval_benchmark_examples(
            args.db,
            limit=args.limit,
            per_filing_limit=args.per_filing_limit,
        )
        write_retrieval_examples(args.output, examples)
        summary = {"output": args.output, "query_count": len(examples), "limit": args.limit, "per_filing_limit": args.per_filing_limit}
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
            Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
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
            Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
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
            Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
