# Task catalogue

AuditOps answers questions about frozen filing evidence. Standards metadata identifies reporting context; it does not confer assurance or authorize audit opinions.

| Source | Implemented tasks | Verification and limits |
|---|---|---|
| SEC/EDGAR, US GAAP | MetricSpec ratios, rollups, differences, margins, growth and returns | Formula, period, context, unit and evidence checks; supported metrics are enumerated in the registry |
| SEC narrative | Footnotes, accounting policies, opinion language, critical audit matters | Frozen retrieval and exact extracts; question relevance requires separate semantic review |
| Companies House XHTML/iXBRL | Current ratio and working capital | Current asset/liability inputs with maturity, sign, entity, period, currency and missing-input guards |
| Companies House narrative | Audit/exemption status, opinion language, key/critical audit matter passages | Evidence-bound extraction or refusal; exemption is not an auditor opinion |
| FCA NSM | Manifest/download foundation | Not a completed UK benchmark |
| Paired image PDFs | Separate VLM extraction and page/bounding-box diagnostic | Rendering and mock-tested runtime; no box accuracy without ground-truth boxes |

The quantitative registry is [metric_specs.json](../auditops/specs/metric_specs.json).
Deterministic commands include `ingest`, `generate-answers`, `generate-task-specs`,
`render-quant-datasets`, and `answer-quant`. Retrieval uses `eval-retrieval`.
Agent commands include `build-agent-benchmark`, `run-agent-baseline`,
`eval-agent-baseline`, and `report-agent-baseline`, plus version-specific review workflows.
Use `python -m auditops.cli COMMAND --help` for exact arguments.

## Outcomes and boundaries

Missing inputs, incompatible units, ambiguous contexts, unsupported requests, and
unavailable evidence have explicit refusal/failure codes. Never fabricate a disclosure,
infer current maturity from a generic creditor name, or assume an image PDF has usable text.

The v2.3 text registry permits only `quant_metric` and `narrative_citation`.
VLM tasks have a separate contract and score. Quantitative safety-hybrid cases make
no model calls; direct controls receive the metric contract. Narrative selection
uses the smallest permitted contiguous extract set. Professional judgments, fraud
conclusions, and statutory opinions are outside this system.

## Evaluation status

Synthetic unit/integration tests cover these paths. Cached-source pilots informed the
design; the full 500-case official baseline was not completed. Few-shot execution
requires approved disjoint demonstrations. Tasks from one filing are correlated:
report filing-level as well as task-level metrics.
