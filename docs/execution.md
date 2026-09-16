# Portable execution

## CPU development

Install `.[dev,retrieval,models]` in a virtual environment.
`python scripts/smoke.py` ingests synthetic SEC fixtures, builds tasks, and replays
answers without credentials or network access. The test suite also covers UK parsing
and model-adapter mocks.

The historical hash-pinned Python 3.11 lock in `requirements/` records a research
environment. Portable development extras additionally declare schema/build tools.
GPU libraries remain separate from CPU command imports.

## Data preparation

SEC acquisition requires an explicit contact-bearing `AUDITOPS_SEC_USER_AGENT`.
Use `import-offline-us-corpus` for an existing SEC cache or manifest/download commands
for acquisition. Companies House ingestion takes a supplied electronic accounts
archive through `build-companies-house-corpus`; no dataset is shipped.

Builders separate `cases.jsonl` and `evidence.jsonl` from evaluator-only
`gold.jsonl` and review material. v2.4 freeze requires external-human approval;
v2.4p is explicitly exploratory and AI-reviewed. Setup starts no experiment.

## Pinned model inference

[Model revisions](../config/model_revisions.json) and
[container digests](../config/vllm_0.26.0.json) are immutable references.
Obtain access to gated checkpoints separately, prepare a snapshot with a file
manifest, and use `write-model-config` to bind it to its portable model identity.
Provide `HF_TOKEN` only during acquisition, never in a committed config or inference environment.

The dependency-light entry point is `python -m auditops.agent_cli --help`.
It uses offline vLLM 0.26.0 and strict provenance. Select the frozen benchmark,
verified model config, runtime mode, prompt condition, and output explicitly.
The recommended mode is `safety_hybrid`; few-shot requests require approval.

Optional Linux Slurm/Apptainer helpers live in `scripts/slurm/`. They assume no
site account, partition, module, or storage location. Set `AUDITOPS_REPO_ROOT`
when submitting spooled scripts; supply scheduler allocation options yourself.
`scripts/env.sh` defaults to ignored `.auditops/` storage.

Inference launchers require:

- A GPU allocation and `AUDITOPS_EXPECTED_GPU` matching the exact visible GPU name.
- `AUDITOPS_TARGET_ARCH` (`amd64` by default, or `arm64`) matching the container record.
- Benchmark, model-config, source-manifest, package-lock, container and output paths.
- External SHA-256 pins for source manifest, model config, model manifest and container.
- Read-only finalized inputs and a new output path under the configured run root.

Launchers stage only inference inputs, verify exact source/model file sets, scrub
inherited controls/secrets, mount inputs read-only, and require a separate network
namespace with IPv4/IPv6 denial probes. Unsupported isolation fails explicitly.
No model, precision, or hardware substitution occurs automatically.
GPU inference was not re-executed during this cleanup.

The separate image-PDF renderer also requires Poppler's `pdftoppm` executable;
install it in the diagnostic environment. The Python `pdf` extra does not supply it.

## Authorization and resumption

Portable exploratory authorization uses
`auditops-exploratory-run-authorization.v2.4p.1`. Creation requires
`--authorized-by`; inference requires an independently configured matching
`--expected-authorizer`. Scheduler helpers use `AUDITOPS_EXPLORATORY_AUTHORIZER`.
Identity, benchmark, gate metrics, limitations and timestamp are hash-bound.
This is an operator binding, not cryptographic authentication.
Earlier deployment-specific authorizations are rejected rather than reinterpreted.
External-human approval is unchanged.

Per-case checkpoints reside beside the intended output directory. Repeat identical
bindings to resume; changed bindings fail. Completed output directories are immutable.
Runtime logs/manifests can contain deployment paths: keep them outside Git and
sanitize deliberate exports.
