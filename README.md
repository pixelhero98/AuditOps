# AuditOps

AuditOps verifies claims against public company filings. It combines US SEC and UK Companies House ingestion, canonical facts, BM25 retrieval, deterministic calculations, and bounded open-model extraction with citations.

The default agent profile is **v2.3 safety-hybrid**: quantitative answers are computed without a model; narrative answers are exact passages selected by Gemma or Qwen and checked before release. AuditOps performs filing verification, not statutory audits or professional audit opinions.

## Install and verify

Use Python 3.11 or newer in a virtual environment:

```bash
python -m venv .venv
# Activate .venv using your shell's standard activation command.
python -m pip install -e ".[dev,retrieval,models]"
python -m pytest -q
python -m auditops.cli --help
python scripts/smoke.py
```

The smoke uses bundled synthetic filings, makes no network requests, and requires no model weights. Linux GPU inference uses vLLM 0.26.0; install the `inference` extra in a compatible CUDA environment or use the pinned container workflow. The `pdf` extra supports the separate image-PDF diagnostic. CPU tests do not establish GPU compatibility or measured model accuracy.

## Framework

| Component | Responsibility |
|---|---|
| Router and registry | Bind tasks to allowed operations, parameters, and output schemas |
| Retriever | Select a stable BM25 evidence scope; load the frozen scope once |
| Constrained executor | Execute registered MetricSpec formulas and closed TaskPlan objects |
| Model adapter | Run revision-pinned Gemma or Qwen with stage-specific structured output |
| Context manager | Enforce evidence order, metadata completeness, and token budgets |
| Verifier | Check schema, evidence, exact support, arithmetic, period, unit, and observation binding |
| Batch runner | Write resumable checkpoints, typed failures, and immutable completed results |

The coder-related component produces constrained code-training targets for `execute_task_plan`; serving does not run arbitrary generated Python. There is no autonomous coder or multi-agent swarm.

`direct` and `capability_agent` remain available for controlled comparisons. The capability agent's registered tool call is bounded, not open-ended planning. The verifier permits at most one safe-format repair and never exposes target-sensitive repair hints.

## Models and execution

The [model registry](config/model_revisions.json) preserves exact revisions for:

- `Qwen/Qwen3.5-27B-FP8` and its BF16 parity checkpoint.
- `google/gemma-4-31B-it-qat-w4a16-ct` and its BF16 parity checkpoint.

Weights and real datasets are supplied separately. Credentials go only to acquisition processes through the environment. Inference uses prepared offline snapshots. See [portable execution](docs/execution.md) for setup, provenance, isolation, and generic scheduler helpers.

## Tasks and evidence

The [task catalogue](docs/tasks.md) distinguishes implemented US/UK quantitative, narrative, refusal, and PDF tasks. Corpora retain separate taxonomy adapters, source systems, and task scopes.

The [Companies House findings](docs/companies-house-findings.md) explain taxonomy variation, creditor/context ambiguity, missing disclosures, and image-PDF limitations. They include reproducible sanitized evidence and a proposed NLP-assisted canonicalization design. No NLP superiority result or completed 500-case official baseline is claimed.

## Contracts and research history

- Current [v2.3 runtime](docs/agent-runtime-protocol-v2.3.html) and [v2.3 schemas](docs/agent-tool-schema-v2.3.html).
- Earlier [runtime protocol](docs/agent-runtime-protocol.html) and [tool schema](docs/agent-tool-schema.html), retained for versioned contracts.
- [Benchmark review protocol](docs/agent-benchmark-v2.4.html), keeping external-human and provisional-AI review distinct.
- [Historical archive](archive/README.md), containing sanitized design history and bounded findings.
- [Release verification](docs/release-verification.md), with CPU test results and explicit validation limits.

Run `python -m auditops.agent_docs --check` to verify generated contract tables. Keep inference cases/evidence separate from evaluator gold and review decisions. Operational artifacts belong outside version control.
