# AuditOps

`AuditOps` is an evidence-first, open-source multi-agent auditing framework. It ingests filing data into canonical fact and narrative layers, runs deterministic validators, emits traceable answer objects and task specs, and provides the benchmark and supervision foundation for planner, quant/code, text/RAG, and synthesizer/verifier agents running on local open-source models rather than hosted APIs.

Current implementation status:
- US SEC/EDGAR corpus: implemented
- UK FCA NSM corpus: manifest/download foundation implemented
- UK Companies House corpus: planned, not implemented
- multi-agent fine-tuning / RLHF: planned, not implemented

Today the implementation is strongest on deterministic corpus, canonical evidence, benchmark, and task-generation foundations. Broader multi-agent orchestration, local fine-tuning, and cross-corpus multimodal support remain planned work.

## Quick Start

```bash
python -m pip install -e .[dev]
pytest
auditops ingest --zip /path/to/sec-xbrl.zip --db auditops.sqlite --extract-narrative --reset-db
auditops generate-answers --db auditops.sqlite --output answer_objects_quant.jsonl
auditops generate-task-specs --db auditops.sqlite --output task_specs_quant.jsonl
auditops render-quant-datasets --db auditops.sqlite --output-dir rendered_data
```

Optional retrieval extra:

```bash
python -m pip install -e .[retrieval]
```

Corpus bootstrap:

```bash
auditops build-manifest --constituents sp500_snapshot.csv --corpus-root /path/to/corpora/sp500_latest_2026-03-20 --snapshot-date 2026-03-20
auditops download-filings --corpus-root /path/to/corpora/sp500_latest_2026-03-20
auditops ingest-corpus --corpus-root /path/to/corpora/sp500_latest_2026-03-20
auditops generate-corpus-datasets --corpus-root /path/to/corpora/sp500_latest_2026-03-20
auditops eval-corpus --corpus-root /path/to/corpora/sp500_latest_2026-03-20
```

UK corpus foundation:

```bash
auditops build-uk-manifest --constituents ftse100_snapshot.csv --corpus-root /path/to/corpora/uk_ftse100_nsm_latest_2026-03-21 --snapshot-date 2026-03-21
auditops download-uk-filings --corpus-root /path/to/corpora/uk_ftse100_nsm_latest_2026-03-21
```

## Environment

`AuditOps` works as a normal Python package. Create a virtual environment, install the editable package plus the extras you need, and choose corpus/output paths that fit your local or cluster setup.

## Open-Source Model Strategy

`AuditOps` assumes a local, open-source model stack. The default target architecture is mixed OSS per role rather than one model for everything:

- planner / synthesizer / verifier: instruct-style open-source model family
- quant / code agent: coder-specialized open-source model family
- PDF / image-heavy corpora: optional OCR and vision stack layered in later

In practice this usually means pairing an instruct-style model with a coder-style model, but the README stays family-agnostic because the best local stack may change over time. Deterministic validation remains primary, and any model-based judge stays secondary to evidence-backed checks.

## Corpus Families And Benchmark Design

`AuditOps` should treat each filing ecosystem as its own corpus family, with separate manifests, ingestion rules, evidence formats, and evaluation slices:

- `US SEC/EDGAR`
  - current implementation track
  - primarily HTML + iXBRL + tables
- `UK FCA NSM`
  - current UK foundation track
  - XHTML / HTML / PDF mix
- `UK Companies House`
  - planned separate UK corpus
  - annual statutory accounts with XHTML / PDF / image-heavy variants

These corpora do not share the same prompt-code-answer format, but they should share a common auditing task ontology:

- quant deterministic tasks
- citation-grounded narrative tasks
- refusal tasks
- hard negatives
- planner / judge trajectory tasks

Cross-corpus normalization should happen at the task layer, not at the raw-document layer. PDF and image-heavy corpora require corpus-specific parsing, OCR, and table extraction before they can feed the same task families used by SEC/EDGAR-style corpora.

## Training Data Mix

For later multi-agent tuning, use target ranges rather than fixed quotas. The current recommended starting mix is:

- `~65%` ground-truth-backed answerable tasks
- `~20%` refusal tasks
- `~10%` hard negatives
- `~5%` planner / judge / repair trajectories

The last `~5%` should cover artifacts such as:

- `TaskPlan` traces
- claim maps
- patch instructions
- verifier / judge notes
- repair trajectories

These are starting targets, not enforcement rules. The exact mix should be adjusted by corpus family, modality, and task family.

## Main Commands

- `auditops ingest`: load raw XBRL facts, narrative chunks, canonical layers, and validators into SQLite
- `auditops rebuild-canon`: rebuild `facts_canon`, `chunk_canon`, and `validators_v0`
- `auditops validate`: print structured validator rows
- `auditops generate-answers`: evaluate MetricSpecs and write `answer_objects_quant.jsonl`
- `auditops generate-task-specs`: materialize deterministic quant `TaskSpec` records from answer objects
- `auditops render-quant-datasets`: write `task_specs_quant.jsonl`, `train_quant_qa.jsonl`, `train_quant_code.jsonl`, `train_refusal.jsonl`, `hard_negatives_quant.jsonl`, and `eval_holdout.jsonl`
- `auditops build-manifest`: freeze a dated public constituents snapshot, resolve SEC CIKs, and write `issuer_manifest.jsonl` / `filing_manifest.jsonl`
- `auditops build-uk-manifest`: freeze a dated FTSE constituents snapshot, resolve latest FCA NSM annual + half-yearly reports, and write UK `issuer_manifest.jsonl` / `filing_manifest.jsonl`
- `auditops download-filings`: download latest filing ZIPs from `filing_manifest.jsonl` and write `download_ledger.jsonl`
- `auditops download-uk-filings`: archive FCA NSM disclosure details and attempt raw UK report downloads, writing a UK `download_ledger.jsonl`
- `auditops ingest-corpus`: ingest every successfully downloaded filing into a shared corpus SQLite DB with narrative extraction enabled by default
- `auditops repair-filing`: repair one failed or missing corpus filing ingest and update the ingest ledger
- `auditops generate-corpus-datasets`: generate answer objects, task specs, issuer-holdout split manifests, and rendered corpus datasets
- `auditops eval-corpus`: emit data-quality and runtime evaluation reports for a versioned corpus root
- `auditops answer-quant`: route a MetricSpec-backed question into a closed `TaskPlan` and execute it through the deterministic runtime
- `auditops inspect-narrative`: preview canonical narrative chunks for a filing
- `auditops build-retrieval-benchmark`: materialize deterministic retrieval examples from `chunk_canon`
- `auditops build-narrative-benchmark`: materialize narrative citation task specs from canonical footnote/accounting-note chunks
- `auditops eval-narrative-citations`: evaluate citation retrieval against narrative task specs
- `auditops answer-narrative`: answer a constrained narrative benchmark question with chunk citations or a refusal
- `auditops eval-narrative-answers`: evaluate the deterministic narrative answer runtime against narrative task specs
- `auditops eval-narrative-routing`: evaluate deterministic loose-question routing against narrative task specs
- `auditops eval-retrieval`: run a BM25 retrieval benchmark against gold `chunk_evidence_id` examples

## Later-Phase Contracts

Implemented now:
- `task_specs_quant.jsonl`: deterministic intermediate records carrying task identity, filing/period metadata, canonical inputs, distractors, evidence requirements, and target structured answers
- `train_quant_qa.jsonl`: rendered question -> `TaskPlan` -> structured answer rows for quant tasks in the train split
- `train_quant_code.jsonl`: aligned code-target rows that compile to the constrained executor contract, not arbitrary Python
- `train_refusal.jsonl`: explicit refusal tasks with deterministic refusal codes
- `hard_negatives_quant.jsonl`: typed adversarial manifests for distractor, period, unit, context, and evidence-map traps
- `eval_holdout.jsonl`: issuer holdout split with no train/eval leakage for the latest-filing corpus

Planned later for multi-agent supervision:
- planner / router trajectory records with `TaskPlan` targets
- claim-map and support-pack supervision for synthesis agents
- verifier / judge patch-note and repair trajectories
- corpus-tagged narrative datasets spanning SEC/EDGAR, FCA NSM, and Companies House slices

`TaskPlan` and structured-answer schemas ship in [auditops/specs/task_plan.schema.json](auditops/specs/task_plan.schema.json), [auditops/specs/structured_answer.schema.json](auditops/specs/structured_answer.schema.json), and [auditops/specs/task_spec_quant.schema.json](auditops/specs/task_spec_quant.schema.json).
The narrative citation benchmark schema ships in [auditops/specs/narrative_task_spec.schema.json](auditops/specs/narrative_task_spec.schema.json). The deterministic narrative answer schema ships in [auditops/specs/narrative_structured_answer.schema.json](auditops/specs/narrative_structured_answer.schema.json).

## Roadmap

### Phase 0: Canonical Evidence Layer

Status: implemented

- Keep raw ingest tables as immutable provenance and materialize canonical layers on top.
- `facts_canon` is the selected fact layer with deterministic `fact_evidence_id`, normalized period metadata, canonical unit families, exact numeric storage, and source anchors.
- `chunk_canon` is the canonical narrative layer with deterministic `chunk_evidence_id`, offsets, SHA1, retrieval text, item/heading/subheading metadata, and filing-level `period_key`.
- `validators_v0` currently enforce unit/scale, period alignment, context selection, and evidence-id existence checks.

Exit criteria:
- For a filing, core facts such as revenue, COGS, current assets, and current liabilities are retrievable with stable provenance anchors.

### Phase 1: MetricSpec Library And Deterministic Answer Objects

Status: implemented

- Quant v1 is strictly MetricSpec-backed.
- Metric evaluation is deterministic and emits full-trace `answer_objects_quant.jsonl`.
- The current runtime supports ratios, growth, rollups, differences, averages, and deterministic refusals.
- `Structured Answer` remains narrow: `status`, `value`, `unit`, `period_key`, `evidence_ids`, `refusal_code`.

Exit criteria:
- MetricSpecs generate reproducible `OK` and `REFUSAL` answer objects across filings without manual intervention.

### Phase 2A: Deterministic Task Construction

Status: implemented

- Build `task_specs_quant.jsonl` from answer objects.
- Each `TaskSpec` carries:
  - `task_id`, `source_answer_id`, `metric_spec_id`, `filing_id`, ticker, filing metadata, and period metadata
  - target status and target structured answer
  - canonical inputs, distractors, evidence requirements, refusal policy, and output schema
- Negative-task manifests are generated here, not inside rendering prompts.
- For single-database local rendering, holdout assignment is by issuer-year.

Deliverables:
- `task_specs_quant.jsonl`
- `hard_negatives_quant.jsonl`
- deterministic split manifest embedded in each task spec

### Phase 2B: Rendered Synthetic Data

Status: foundation implemented with deterministic templates

- Render quant tasks from `TaskSpec`, not directly from raw answer objects.
- Current renderer is deterministic `template-v1`; later LLM rendering should preserve the same interfaces.
- Produce:
  - `train_quant_qa.jsonl`
  - `train_quant_code.jsonl`
  - `train_refusal.jsonl`
  - `eval_holdout.jsonl`
- Code targets must stay aligned to the constrained runtime contract, not arbitrary Python.

Validation rules:
- Every rendered sample must re-execute against canonical truth.
- Deterministic checks are the hard gate: numeric match, refusal-code match, period/unit/context match, and evidence-id presence.
- Model-based judges, if added later, are sampling-only triage and not release blockers.

Open-source-only synthesis note:
- Keep `template-v1` as the deterministic baseline until local generation is ready.
- Later synthesis should assume a strong local generator model plus a separate local verifier model, not hosted APIs.
- Deterministic validators remain the hard gate for quant, refusal, period, unit, context, and evidence checks.
- Narrative and cross-corpus generation should carry corpus and modality tags so one renderer does not silently collapse different source families into the same format.

### Phase 3: Constrained Quant Runtime

Status: baseline implemented

- Runtime path is:
  - `Question -> TaskPlan -> deterministic executor -> structured answer`
- `TaskPlan` is closed and must include:
  - `task_id`
  - `task_type`
  - `metric_spec_id`
  - `filing_id`
  - `period_key`
  - `executor_op`
  - `required_output_schema`
  - `refusal_policy`
- Quant v1 does not support open-ended accounting reasoning, raw fact lookup, or broad comparison tasks.
- Generated code is auxiliary supervision or evaluation only, not the serving path.

Evaluation targets:
- numeric accuracy
- refusal correctness
- unit/period/context accuracy
- unsupported-claim rate
- evidence-id exactness

### Phase 3.5: S&P 500 Latest Corpus And Evaluation Foundation

Status: implemented

- The first real corpus is a frozen current-universe snapshot with latest `10-K` + latest `10-Q` per issuer.
- The corpus pipeline is manifest-driven and does not rely on the legacy hard-coded downloader.
- Raw lineage is persisted as:
  - `manifest/issuer_manifest.jsonl`
  - `manifest/filing_manifest.jsonl`
  - `manifest/download_ledger.jsonl`
  - `raw/submissions/*.json`
  - `raw/xbrl_zip/...`
- All filings ingest into one versioned shared DB at `db/corpus.sqlite`, with narrative extraction enabled during ingest.
- Downstream lineage is preserved:
  - `derived/answers/answer_objects_quant.jsonl`
  - `derived/tasks/task_specs_quant.jsonl`
  - `derived/datasets/train_quant_qa.jsonl`
  - `derived/datasets/train_quant_code.jsonl`
  - `derived/datasets/train_refusal.jsonl`
  - `derived/datasets/hard_negatives_quant.jsonl`
  - `derived/datasets/eval_holdout.jsonl`
- The primary holdout policy for this latest-only corpus is issuer holdout, not issuer-year.
- Evaluation emits:
  - `eval/data_quality_summary.json`
  - `eval/runtime_eval_summary.json`
  - `eval/coverage_by_metric.csv`
  - `eval/coverage_by_issuer.csv`
  - `eval/validator_distribution.csv`
  - `eval/runtime_failures.jsonl`
  - `eval/split_manifest.jsonl`

Current operating rule:
- Use the manifest-driven corpus commands for real S&P corpus work. The legacy hard-coded S&P downloader has been removed from the production path.

### Phase 3.6: UK FTSE 100 NSM Separate Corpus

Status: foundation implemented

- Build the FCA NSM expansion as a separate corpus family, not a mixed US/UK corpus.
- Target corpus name:
  - `uk_ftse100_nsm_latest_<snapshot_date>`
- Freeze a dated FTSE 100 constituents snapshot and treat that frozen manifest as the source of truth for the run.
- Source periodic reports from the FCA National Storage Mechanism.
- First UK filing scope:
  - latest `Annual Financial Report`
  - latest `Half-Yearly Financial Report`
- Keep Companies House annual accounts on a separate roadmap track with different ingestion and evaluation expectations.

Why this shape:
- FTSE 100 is the cleanest first UK listed-company universe for protocol stabilization.
- FCA NSM annual + half-yearly reports are the closest UK analogue to the US `10-K` + `10-Q` pair.
- Companies House remains useful later as a separate statutory-accounts corpus, not as a replacement for the FCA NSM track.

UK v1 schedule:
- `UK-0: Universe freeze`
  - freeze a dated FTSE 100 constituents snapshot
  - write `issuer_manifest.jsonl` with UK-specific source metadata
- `UK-1: Filing resolution`
  - resolve latest annual and half-yearly reports per issuer from FCA NSM
  - persist filing manifests, source metadata, and download ledger
- `UK-2: Raw archive`
  - store downloaded source files and raw metadata under a versioned corpus root
  - preserve source lineage exactly as done for the US corpus
- `UK-3: Ingest pilot`
  - adapt the ingest path for FCA NSM document shapes
  - prove canonical fact and chunk extraction on a pilot issuer set before full-universe ingest
- `UK-4: Full corpus build`
  - ingest the full FTSE 100 annual + half-yearly corpus into one shared DB
  - generate answer objects, task specs, rendered datasets, and eval outputs
- `UK-5: Freeze and review`
  - freeze the first UK corpus only after issuer-level coverage and runtime eval are stable

Current implementation status:
- Implemented now:
  - `auditops build-uk-manifest`
  - `auditops download-uk-filings`
  - FTSE snapshot ingest into a UK `issuer_manifest.jsonl`
  - FCA NSM resolution into a UK `filing_manifest.jsonl`
  - archive of NSM `details` payloads under the corpus `raw/` tree
  - UK `download_ledger.jsonl` with explicit `details_only` vs raw-download states
- Not implemented yet:
  - FCA document ingest into the shared SQLite corpus DB
  - UK answer-object, task-spec, and eval generation

Current operating caveat:
- The FCA NSM API resolution path is working, but direct raw document fetches are not yet reliably accessible from the current programmatic client flow.
- UK v1 is therefore a real separate corpus foundation with reproducible manifests and archived disclosure metadata, but it is not yet at US parity for ingestable filing packages.
- The next UK step is an ingest pilot on a small issuer subset once the raw-document retrieval path is stabilized.

Planned UK storage layout under `corpora/uk_ftse100_nsm_latest_<snapshot_date>/`:
- `manifest/`
- `raw/`
- `db/corpus.sqlite`
- `derived/`
- `eval/`
- `logs/`

UK-specific operating rules:
- Keep the UK corpus isolated from the US corpus at the storage, manifest, and eval levels.
- Add UK-specific metadata fields during manifest/ingest:
  - `source_system=FCA_NSM`
  - `report_type=annual|half_yearly`
  - `accounting_regime` when identifiable
  - `issuer_country`
- Reuse the same corpus lineage pattern:
  - manifest
  - raw source archive
  - shared DB
  - answers
  - task specs
  - rendered datasets
  - eval reports
- Hold out by issuer for the first latest-only UK corpus, matching the US latest-only corpus policy.

Exit criteria:
- A frozen FTSE 100 latest-report corpus with reproducible manifests, ingest lineage, answer/task generation, and runtime eval.

### Phase 3.7: UK Companies House Separate Corpus

Status: planned

- Build Companies House as a separate UK corpus family rather than folding it into FCA NSM.
- Treat it as an annual statutory-accounts corpus with its own manifests, ingest rules, and benchmark slices.
- Expected modalities are broader than FCA NSM:
  - XHTML
  - PDF
  - image-heavy annual accounts and attachments
- Initial scope should focus on annual accounts only, with no promise of interim-report parity.
- This corpus should share the common auditing task ontology, but it will require different prompt-code-answer renderers and different evidence extraction paths from SEC/EDGAR and FCA NSM.

Planned implementation priorities:
- freeze a dated Companies House target universe and issuer manifest
- build filing manifests with filing-type and modality metadata
- add OCR, PDF parsing, and table extraction before canonical fact and chunk generation
- keep benchmark and eval outputs separate from FCA NSM even when the task families overlap

Exit criteria:
- A reproducible annual-accounts corpus with modality-aware ingestion and benchmark slices that can plug into the shared task ontology without pretending to be SEC/EDGAR-like.

### Phase 3.8: Cross-Corpus Benchmark Normalization

Status: planned

- Normalize at the task layer, not at the raw-source layer.
- Keep shared task families across corpora:
  - quant deterministic tasks
  - citation-grounded narrative tasks
  - refusals
  - hard negatives
  - planner / judge trajectories
- Keep renderer and prompt shapes corpus-specific when needed.
- Tag future dataset rows with corpus and modality metadata so training and eval can be sliced safely across:
  - `SEC_EDGAR`
  - `FCA_NSM`
  - `COMPANIES_HOUSE`
  - `HTML`
  - `XHTML`
  - `PDF`
  - `IMAGE`

Exit criteria:
- Shared auditing tasks remain comparable across corpora without collapsing corpus-specific modality differences or evidence contracts.

### Phase 4: Open-Source Multi-Agent Hardening And Tuning

Status: hardening foundation implemented for US v1; open-source multi-agent tuning planned

- Do prompt/runtime hardening before any local fine-tuning work.
- Expand regression suites and bucket failures by routing, period selection, context choice, unit handling, refusal behavior, and evidence-map errors.
- Only tune when deterministic-runtime and prompt-only baselines plateau on fixed eval sets.
- Prioritize open-source multi-agent tuning in this order:
  - router / planner adapters
  - coder / quant adapters
  - synthesizer / verifier adapters
- Keep retrieval, SQL selection, and deterministic calculators outside the trainable core.
- Keep RLHF or preference-style optimization planned, not assumed in the current implementation.

Current US v1 hardening state on corpus `sp500_latest_2026-03-20`:
- duplicate quant `TaskSpec` rows are deduped before routing and dataset generation
- `eval-corpus` now emits `eval/runtime_failure_buckets.json`
- refreshed held-out runtime eval is currently:
  - `eval_task_count = 38,812`
  - `failure_count = 0`
  - `numeric_accuracy = 1.0000`
  - `refusal_correctness = 1.0000`
  - `period_accuracy = 1.0000`
  - `evidence_id_exactness = 1.0000`
  - `unsupported_claim_rate = 0.0`

Exit criteria:
- Hard negatives improve without increasing unsupported claims or hallucinated evidence maps.

### Phase 5A: Narrative Retrieval And Citation Benchmark

Status: baseline implemented and frozen for US v1

- Narrative expansion starts with footnotes and accounting-policy text, not broad MD&A.
- Add a narrative `TaskSpec` layer before any text-generation path.
- Require claim, answerability label, required `chunk_evidence_id`s, and citation rules.
- Evaluate retrieval before narrative QA.
- Future UK narrative benchmarks should stay corpus-specific until FCA NSM and Companies House modality handling is strong enough to share the same retrieval assumptions.

Current baseline:
- A retrieval benchmark entrypoint exists through `auditops eval-retrieval`.
- The current baseline is framework-light but Haystack-compatible and now defaults to a deterministic lexical pipeline:
  - `chunk_canon.retrieval_text` / cleaned narrative content is loaded into an in-memory BM25 retriever
  - a deterministic metadata-aware reranker (`bm25_rerank`) reorders the top lexical candidates within each filing
  - evaluation reports `hit_rate_at_k`, `top1_hit_rate`, and `mrr_at_k`
- This is for offline retrieval verification only; answer synthesis and citation-generation remain later work.

Frozen US v1 retrieval baseline:
- Corpus root:
  - `corpora/sp500_latest_2026-03-20`
- Benchmark artifacts under `eval/`:
  - `retrieval_benchmark_examples_v5.jsonl`
  - `retrieval_benchmark_summary_v5.json`
- Method:
  - `bm25_rerank`
  - `top_k = 5`
  - `candidate_k = 15`
- Verified metrics on the frozen v5 benchmark:
  - `hit_rate_at_k = 1.0000`
  - `top1_hit_rate = 0.9680`
  - `mrr_at_k = 0.9815`
- Label breakdown:
  - `footnote_note`: `query_count = 87`, `top1_hit_rate = 1.0000`, `mrr_at_k = 1.0000`
  - `subheading_chunk`: `query_count = 163`, `top1_hit_rate = 0.9509`, `mrr_at_k = 0.9716`

Operating rule for future retrieval work:
- Treat `retrieval_benchmark_examples_v5.jsonl` and `retrieval_benchmark_summary_v5.json` as the current frozen Phase 5A baseline for the US corpus.
- Compare later retrieval changes against this baseline explicitly; do not overwrite it.
- If a future benchmark query set changes materially, version it as a new benchmark generation rather than folding it into `v5`.

Evaluation targets:
- chunk-id precision
- chunk-id coverage
- refusal correctness when support is absent

### Phase 5B: Narrative QA

Status: foundation implemented for US latest and US trailing-2FY

- Build a narrative path only after retrieval quality is stable.
- Preserve evidence-first guarantees with explicit chunk citations.
- Keep partial-answer and refusal behavior explicit when text support is insufficient.
- Leave MD&A out until chunk-level period attribution is stronger.
- Start with answerable footnote/accounting-note tasks, then add deterministic unanswerable/refusal tasks before any generative narrative path.
- Keep the first path retrieval-first and extractive, not generative.
- Extend to UK corpora only after PDF / image / OCR pipelines can produce evidence objects comparable to the current SEC/EDGAR chunk layer.

Current narrative foundation:
- Narrative task specs are materialized with:
  - question / retrieval query
  - filing metadata and heading metadata
  - `scope_type` and `scope_key` for filing-vs-note retrieval scope
  - expected `chunk_evidence_id`s
  - extractive answer text
  - citation policy
- The benchmark now supports deterministic `UNANSWERABLE` tasks with explicit refusal codes and a typed negative taxonomy.
- Current unanswerable task families are:
  - `same_filing_wrong_note`
  - `same_issuer_wrong_period`
  - `unsupported_attribute`
  - `cross_label_query_transfer`
  - `cross_filing_query_transfer`
- A constrained narrative runtime now answers only benchmarked narrative questions:
  - the active trailing-2FY benchmark uses note-scoped retrieval for footnote and accounting-policy tasks, and includes `same_filing_wrong_note` refusals
  - `Question -> NarrativeTaskSpec -> retrieval -> extractive answer + chunk citation`
  - or `REFUSAL` for explicit unanswerable tasks / unsupported questions
- Current artifact paths for the latest-only US benchmark under `corpora/sp500_latest_2026-03-20/eval/`:
  - `narrative_benchmark_us_v1.jsonl`
  - `narrative_citation_summary_us_v1.json`
  - `narrative_answer_summary_us_v1.json`
- Current verified metrics on the cleaned US v1 benchmark:
  - citation benchmark:
    - `task_count = 200`
    - `answerable_task_count = 100`
    - `unanswerable_task_count = 100`
    - `citation_precision_at_1 = 0.9500`
    - `citation_coverage_at_k = 1.0000`
    - `citation_mrr_at_k = 0.9750`
  - deterministic narrative answer runtime:
    - `answerability_accuracy = 1.0000`
    - `citation_exactness = 1.0000`
    - `answer_text_exactness = 1.0000`
    - `refusal_correctness = 1.0000`
- Frozen trailing-2FY US narrative baseline under `corpora/sp500_trailing_2fy_2026-03-20/eval/`:
  - `narrative_benchmark_us_2fy_v7.jsonl`
  - `narrative_citation_summary_us_2fy_v7.json`
  - `narrative_answer_summary_us_2fy_v7.json`
  - `narrative_routing_summary_us_2fy_v8.json`
- Verified metrics on the frozen trailing-2FY US baseline:
  - citation benchmark:
    - `task_count = 200`
    - `answerable_task_count = 100`
    - `unanswerable_task_count = 100`
    - `citation_precision_at_1 = 1.0000`
    - `citation_coverage_at_k = 1.0000`
    - `citation_mrr_at_k = 1.0000`
    - negative-type mix:
      - `same_filing_wrong_note = 20`
      - `same_issuer_wrong_period = 20`
      - `unsupported_attribute = 20`
      - `cross_label_query_transfer = 20`
      - `cross_filing_query_transfer = 20`
  - deterministic narrative answer runtime:
    - `answerability_accuracy = 1.0000`
    - `citation_exactness = 1.0000`
    - `answer_text_exactness = 1.0000`
    - `refusal_correctness = 1.0000`
  - loose-question routing runtime:
    - `routing_accuracy = 1.0000`
    - `answerability_accuracy = 1.0000`
    - `citation_exactness = 1.0000`
    - `answer_text_exactness = 1.0000`
    - `refusal_correctness = 1.0000`
    - `safe_refusal_accuracy = 1.0000`
- Operating rule for future narrative work:
  - Treat `narrative_benchmark_us_2fy_v7.jsonl` as the frozen task set and `narrative_routing_summary_us_2fy_v8.json` as the current routing baseline.
  - Compare new narrative task families or routing changes against this baseline explicitly; do not overwrite it.

### Phase 6: Continuous Data Flywheel

Status: planned

- Version all moving parts:
  - canon build version
  - MetricSpec library version
  - TaskSpec schema version
  - rendering prompt or renderer version
  - split manifest version
- Track corpus family and modality versions separately for SEC/EDGAR, FCA NSM, and Companies House.
- Add dedupe and issuer-year leakage controls before retraining.
- For the latest-only corpus, issuer holdout is the active eval policy; issuer-year holdout returns when the corpus expands to multi-year history.
- Monitor:
  - refusal mix
  - validator distributions
  - unsupported-claim rate
  - evidence-id exactness
  - citation-noise rate

Operating rule:
- New filings should flow through canon build -> answer objects -> task specs -> rendered datasets -> validation -> evaluation, with local retraining optional and gated by drift.

