# Public release verification

Validation date: 2026-09-16. This is a CPU-only source release, not a new model-performance baseline.

## Initial cleanup

| Check | Result |
|---|---|
| Complete pre-cleanup suite | 639 passed, 4 platform skips |
| Complete publication suite | 623 passed, 4 platform skips |
| Installed-wheel synthetic smoke | 137 deterministic task replays passed |
| Constrained code targets | All synthetic plans compiled, executed and matched their deterministic answers |
| Gemma and Qwen adapters | Mocked structured-generation checks passed; no GPU inference |
| Packaging | Source distribution and wheel built; wheel installed into an isolated environment; CPU/inference CLI help passed |
| Static checks | Ruff lint/format, bytecode compilation, 37 shell-script syntax checks passed |
| Contracts | 71 Draft 2020-12 schema snapshots validated; 70 shared historical schemas semantically unchanged |
| Documentation | Generated tables current; relative links and status references checked |
| Companies House evidence | Source-bound count reconciliation and published summary checks passed |
| Publication | Working files, exact staged blobs, archives and distribution members scanned for secrets and private deployment identifiers |

Tests ran on CPython 3.13 on Windows. The four skips concern symlink privileges,
`O_NOFOLLOW`, and POSIX mode bits. The source-manifest test also runs in a Git
worktree rather than requiring a `.git` directory. A Linux/Windows Python
3.11/3.13 CI matrix is included; those remote runs are not claimed as completed here.

The suite-count change reflects removal of deployment-preset assertions and their
replacement with portable isolation, provenance, hardware/configuration binding,
authorization, evidence-summary and code-target regressions. Runtime-mode, repair,
refusal, context, arithmetic, evidence, review and resumable-batch tests remain.

## Release boundaries

- v2.3 safety-hybrid remains the documented default. Direct and capability-agent
  modes remain research controls; both open-model families and their revision pins remain.
- Portable exploratory authorization is explicitly versioned `v2.4p.1`, requires a
  configured matching authorizer, and rejects old authorizations. It does not replace
  external-human review or provide cryptographic authentication.
- No GPU job, fresh acquisition, 500-case experiment, or model download was run.
  Actual GPU inference and proposed NLP canonicalization remain untested in this cleanup.
- The privacy scan applies to this publication tree and its sanitized archive,
  not earlier Git history. History is retained; private working artifacts are not published.

See [portable execution](execution.md), the [task catalogue](tasks.md), and the
[Companies House findings](companies-house-findings.md) for operational and evidence limits.

## Follow-up review

The subsequent CI run exposed a Python 3.13 packaging failure: the non-isolated
build could not import setuptools, although its tests passed. CI now installs build
requirements in an isolated build environment and does not cancel other matrix
jobs when one job fails. Development dependencies enable JSON Schema format checks
instead of silently leaving date-time validation unavailable.

Reproduced and corrected boundary defects:

- Strict parsing now rejects exponent overflow, unpaired Unicode surrogates, and
  excessively nested data through sanitized typed failures. Duplicate-key error
  traces no longer retain the rejected key. Existing valid canonical hashes are unchanged.
- Attempt telemetry is type-checked before runtime use; string booleans and
  invalid counts cannot trigger truthy coercion or untyped numeric-conversion errors.
  A completion reporting disabled structured decoding fails closed.
  Request trace metadata is initialized before backend and demonstration preflight,
  so an unsupported adapter produces a typed failure rather than a trace-construction crash.
- Aborted generations cannot release a proposal or consume a repair, including
  aborts accompanied by malformed JSON or schema errors.
- Exploratory authorization validation enforces the published field set, all
  limitation labels, digest syntax, and RFC 3339 timestamps even after rehashing.
  This remains a configured operator binding, not a cryptographic signature.

The review changes neither model checkpoints nor historical contract schemas.
GPU inference, experiment execution, and NLP-assisted tagging remain outside this review.

Two local full-suite attempts encountered intermittent Windows `PermissionError`
failures during different temporary-directory renames. Both affected tests passed
in isolation. No production retry or test skip was added for these failures.
The fresh project-scoped full run passed 650 tests with four platform skips.
The final focused run passed 97 tests, including the additional unsupported-adapter
preflight regression. The isolated build, installed-wheel 137-task smoke, lint,
formatting, generated documentation, findings and publication scans passed.
The CI matrix tests the exact pushed revision on both supported operating systems
and Python versions; its results are recorded in the repository's Actions checks.
