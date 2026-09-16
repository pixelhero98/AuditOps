from __future__ import annotations

import json
import subprocess
import sys

import pytest

import auditops.agent_cli as agent_cli_module
import auditops.cli as cli_module
from auditops.agent_cli import _build_parser as _build_agent_inference_parser
from auditops.cli import (
    _build_parser,
    _parse_named_runs,
    _parse_quantization_pairs,
    main,
)


def test_package_and_gpu_entrypoint_do_not_eagerly_import_corpus_dependencies() -> None:
    code = """
import sys
import auditops
assert 'auditops.companies_house' not in sys.modules
assert 'bs4' not in sys.modules
from auditops.agent_cli import _build_parser
_build_parser().parse_args(['--benchmark-dir','b','--model-config','m','--runtime-mode','direct','--prompt-condition','zero_shot','--output-dir','o'])
assert 'auditops.companies_house' not in sys.modules
assert 'bs4' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_main_cli_module_executes_instead_of_silently_returning() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "auditops.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "build-agent-benchmark" in completed.stdout
    assert "run-agent-baseline" in completed.stdout


def test_agent_baseline_commands_are_exposed() -> None:
    parser = _build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    expected = {
        "materialize-agent-evidence",
        "import-offline-us-corpus",
        "build-us-audit-narrative-tasks",
        "build-agent-benchmark",
        "verify-agent-benchmark",
        "approve-few-shot",
        "render-few-shot-review",
        "write-model-config",
        "run-agent-baseline",
        "probe-agent-structured-output",
        "eval-agent-baseline",
        "report-agent-baseline",
        "write-source-manifest",
        "scan-artifacts",
    }
    assert expected <= set(subparsers.choices)
    assert "v2.2" in parser.format_help()
    assert (
        "AuditOps offline text-agent inference"
        in _build_agent_inference_parser().format_help()
    )


def test_dependency_light_agent_cli_forwards_frozen_selection_controls(
    monkeypatch, capsys
) -> None:
    observed = {}

    def fake_run(benchmark_dir, model_config, output_dir, **kwargs):
        observed.update(
            {
                "benchmark_dir": benchmark_dir,
                "model_config": model_config,
                "output_dir": output_dir,
                **kwargs,
            }
        )
        return {"selection_mode": kwargs["selection_mode"]}

    monkeypatch.setattr(agent_cli_module, "run_agent_baseline", fake_run)
    assert (
        agent_cli_module.main(
            [
                "--benchmark-dir",
                "benchmark",
                "--model-config",
                "model.json",
                "--runtime-mode",
                "safety_hybrid",
                "--prompt-condition",
                "zero_shot",
                "--output-dir",
                "run",
                "--selection-mode",
                "narrative_gate",
            ]
        )
        == 0
    )
    assert observed == {
        "benchmark_dir": "benchmark",
        "model_config": "model.json",
        "output_dir": "run",
        "runtime_mode": "safety_hybrid",
        "prompt_condition": "zero_shot",
        "few_shot_approval": None,
        "limit": None,
        "selection_mode": "narrative_gate",
        "full_run_authorization": None,
        "expected_authorizer": None,
    }
    assert json.loads(capsys.readouterr().out) == {"selection_mode": "narrative_gate"}


def test_build_benchmark_cli_requires_an_explicit_profile() -> None:
    parser = _build_parser()
    arguments = [
        "build-agent-benchmark",
        "--quant-jsonl",
        "quant.jsonl",
        "--narrative-jsonl",
        "narrative.jsonl",
        "--output-dir",
        "benchmark",
        "--jurisdiction",
        "US",
        "--corpus-id",
        "corpus",
        "--reporting-framework",
        "US-GAAP",
        "--standards-version",
        "PCAOB-current",
        "--source-system",
        "SEC-EDGAR",
    ]

    with pytest.raises(SystemExit):
        parser.parse_args(arguments)

    parsed = parser.parse_args([*arguments, "--benchmark-profile", "synthetic"])
    assert parsed.benchmark_profile == "synthetic"


def test_offline_us_import_cli_forwards_the_explicit_cache_identity(
    monkeypatch, capsys
) -> None:
    observed = {}

    def fake_import(source_root, corpus_root, *, snapshot_id, snapshot_date):
        observed.update(
            {
                "source_root": source_root,
                "corpus_root": corpus_root,
                "snapshot_id": snapshot_id,
                "snapshot_date": snapshot_date,
            }
        )
        return {"filing_count": 20, "snapshot_id": snapshot_id}

    monkeypatch.setattr(cli_module, "import_offline_us_cache", fake_import)

    assert (
        main(
            [
                "import-offline-us-corpus",
                "--source-root",
                "cached-sec",
                "--corpus-root",
                "corpus",
                "--snapshot-date",
                "2026-08-21",
                "--corpus-id",
                "us_sec_existing20_realdata_v1",
            ]
        )
        == 0
    )
    assert observed == {
        "source_root": "cached-sec",
        "corpus_root": "corpus",
        "snapshot_id": "us_sec_existing20_realdata_v1",
        "snapshot_date": "2026-08-21",
    }
    assert json.loads(capsys.readouterr().out)["filing_count"] == 20


def test_cached_source_system_is_available_to_narrative_and_evidence_commands() -> None:
    parser = _build_parser()
    narrative = parser.parse_args(
        [
            "build-us-audit-narrative-tasks",
            "--db",
            "corpus.sqlite",
            "--output",
            "narrative.jsonl",
            "--source-system",
            "SEC-EDGAR-CACHED",
        ]
    )
    evidence = parser.parse_args(
        [
            "materialize-agent-evidence",
            "--quant-jsonl",
            "quant.jsonl",
            "--narrative-jsonl",
            "narrative.jsonl",
            "--db",
            "corpus.sqlite",
            "--output-dir",
            "evidence",
            "--source-system",
            "SEC-EDGAR-CACHED",
        ]
    )

    assert narrative.source_system == "SEC-EDGAR-CACHED"
    assert evidence.source_system == "SEC-EDGAR-CACHED"


def test_named_runs_and_quantization_pairs_are_strict() -> None:
    assert _parse_named_runs(["qwen=run-a.jsonl", "gemma=run-b.jsonl"]) == {
        "qwen": "run-a.jsonl",
        "gemma": "run-b.jsonl",
    }
    assert _parse_quantization_pairs(["qwen-bf16,qwen-fp8,qwen-parity"]) == [
        ("qwen-bf16", "qwen-fp8", "qwen-parity")
    ]


def test_secret_scan_cli_fails_without_echoing_secret(
    tmp_path, monkeypatch, capsys
) -> None:
    secret = "hf_fixture_token_that_must_not_be_printed"
    artifact = tmp_path / "result.json"
    artifact.write_text(f'{{"token":"{secret}"}}', encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", secret)

    assert main(["scan-artifacts", "--path", str(artifact)]) == 1
    output = capsys.readouterr().out
    assert secret not in output
    assert "ENV_SECRET_VALUE" in output
