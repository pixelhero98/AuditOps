from __future__ import annotations

import json

from auditops import few_shot_review_cli


def test_dependency_light_few_shot_review_cli(monkeypatch, capsys):
    observed = {}

    def fake_write(benchmark_dir, model_configs, output):
        observed.update(
            {
                "benchmark_dir": benchmark_dir,
                "model_configs": model_configs,
                "output": output,
            }
        )
        return {"passed": True, "max_demonstration_tokens": 123}

    monkeypatch.setattr(few_shot_review_cli, "write_few_shot_review_packet", fake_write)

    status = few_shot_review_cli.main(
        [
            "--benchmark-dir",
            "benchmark",
            "--model-config",
            "gemma.json",
            "--model-config",
            "qwen.json",
            "--output",
            "review.json",
        ]
    )

    assert status == 0
    assert observed == {
        "benchmark_dir": "benchmark",
        "model_configs": ["gemma.json", "qwen.json"],
        "output": "review.json",
    }
    assert json.loads(capsys.readouterr().out)["passed"] is True
