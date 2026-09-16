from __future__ import annotations

import json

from auditops.agent_contracts import build_model_config
from auditops.grammar_probe import GRAMMAR_PROBE_VERSION, run_structured_output_probes
from auditops.model_adapter import ModelResponse


def test_synthetic_grammar_probe_records_all_four_schema_shapes(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = build_model_config(
        model_id="google/gemma-4-31B-it-qat-w4a16-ct",
        revision="pinned-test-revision",
        model_path=str(model_dir),
        quantization="w4a16",
        backend="vllm_offline",
        chat_template_sha256="a" * 64,
    )
    config_path = tmp_path / "model-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    class FakeAdapter:
        constraint_backend = "xgrammar"

        def __init__(self, _config):
            self.calls = []

        def generate_json(self, request):
            self.calls.append(request)
            payload = {
                "probe:constant": {"status": "READY"},
                "probe:tool_action": {
                    "action": "CALL_TOOL",
                    "tool_name": "synthetic_noop",
                    "tool_arguments": {"fixture": 1},
                },
                "probe:terminal_union": {"action": "ANSWER", "value": "2"},
                "probe:narrative_selection": {
                    "action": "ANSWER",
                    "extracts": [
                        {
                            "evidence_id": "synthetic-1",
                            "exact_quote": "Synthetic exact quote.",
                        }
                    ],
                },
            }[request.request_id]
            return ModelResponse(
                request_id=request.request_id,
                payload=payload,
                response_sha256="b" * 64,
                input_tokens=10,
                output_tokens=5,
                duration_ms=1.0,
                finish_reason="stop",
                metadata={
                    "constraint_backend": "xgrammar",
                    "structured_output_applied": True,
                    "output_bytes": 16,
                    "cap_hit": False,
                },
            )

    monkeypatch.setattr("auditops.grammar_probe.OfflineVLLMAdapter", FakeAdapter)
    destination = tmp_path / "probe"
    report = run_structured_output_probes(config_path, destination)

    assert report["grammar_probe_version"] == GRAMMAR_PROBE_VERSION
    assert report["passed"] is True
    assert [row["probe"] for row in report["probes"]] == [
        "constant",
        "tool_action",
        "terminal_union",
        "narrative_selection",
    ]
    assert all(row["finish_reason"] == "stop" for row in report["probes"])
    assert (
        json.loads((destination / "probe_results.json").read_text(encoding="utf-8"))
        == report
    )
