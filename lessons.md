# Lessons

## Current Status

- v2.3 safety-hybrid is the default; quantitative execution is deterministic and narrative release is extractive.
- US SEC and UK Companies House tasks are implemented. PDF/VLM diagnostics are separate.
- The package contains no weights or production benchmark data; history is summarized in `archive/`.

## Known Issues

- Companies House concept selection uses bounded local-name allowlists and context guards, not a complete taxonomy registry.
- Exact support proves where text occurs, not whether it fully answers the question.
- CPU tests and adapter mocks cannot certify GPU compatibility.

## Decisions

- Keep contracts versioned and historical benchmark identities immutable.
- Refuse missing/ambiguous inputs; keep gold outside inference.
- NLP tagging is a proposed evidence-linked normalization layer, not a replacement for numeric validation.
- Portable exploratory authorization requires an explicitly configured authorizer and a distinct version.

## Resolved Failures

- Empty safe repair diagnostics now produce typed failure instead of a repair-construction crash.
- Narrative selection supplies exact quotes reconstructed deterministically, avoiding paraphrase-driven support failures.
- Portable model registration and hardware bindings replace deployment-specific assumptions.
- Constrained Python training targets decode an embedded JSON plan instead of treating JSON literals as Python; all synthetic targets are executed in regression tests.
- Symptom: a Python 3.13 CI build failed after tests passed. Cause: non-isolated builds assumed ambient setuptools. Fix: isolated builds and independent matrix jobs. Prevention: verify the installed wheel as well as source tests.
- Symptom: malformed completions could escape typed handling or retain a duplicate key in traceback locals. Cause: numeric overflow, Unicode and nested-JSON edge cases were incompletely checked. Fix: canonical UTF-8 validation and sanitized parse errors. Prevention: adversarial parser and adapter regressions.
- Symptom: aborted completions and malformed attempt telemetry could reach normal release handling. Fix: reject aborts without repair and validate telemetry types before runtime coercion. Prevention: valid-answer, malformed-JSON and schema-error abort tests.
- Symptom: an unsupported structured-decoding adapter triggered a trace-construction crash. Cause: request audit state was initialized after early checks. Fix: bind the trace before every backend/token preflight. Prevention: zero-generation typed-failure regression.
- Symptom: rehashed exploratory authorizations could contradict required limitation labels. Fix: exact field/label, digest and timestamp validation. Prevention: compare negative objects against Python and format-enabled JSON Schema validators.
