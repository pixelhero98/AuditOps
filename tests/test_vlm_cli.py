from __future__ import annotations

from auditops.cli import _build_parser, main


def test_vlm_build_approval_run_and_evaluation_commands_are_exposed() -> None:
    parser = _build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    assert {
        "build-companies-house-vlm-diagnostic",
        "approve-companies-house-vlm-few-shot",
        "run-companies-house-vlm-diagnostic",
        "eval-companies-house-vlm-diagnostic",
    } <= set(subparsers.choices)


def test_vlm_run_cli_forwards_explicit_separate_inputs(monkeypatch, capsys) -> None:
    received = {}

    def fake_run(diagnostic, model, output, **kwargs):
        received.update(
            {"diagnostic": diagnostic, "model": model, "output": output, **kwargs}
        )
        return {"run_manifest": {"result_count": 20}}

    monkeypatch.setattr("auditops.cli.run_companies_house_vlm_diagnostic", fake_run)
    assert (
        main(
            [
                "run-companies-house-vlm-diagnostic",
                "--diagnostic-dir",
                "diagnostic",
                "--model-config",
                "model.json",
                "--output-dir",
                "run",
                "--pdf-root",
                "pdfs",
                "--prompt-condition",
                "few_shot",
                "--few-shot-diagnostic-dir",
                "examples",
                "--visual-examples-jsonl",
                "visual.jsonl",
                "--few-shot-approval",
                "approval.json",
                "--few-shot-pdf-root",
                "example-pdfs",
            ]
        )
        == 0
    )
    assert received == {
        "diagnostic": "diagnostic",
        "model": "model.json",
        "output": "run",
        "pdf_root": "pdfs",
        "prompt_condition": "few_shot",
        "few_shot_diagnostic_dir": "examples",
        "visual_examples_jsonl": "visual.jsonl",
        "few_shot_approval": "approval.json",
        "few_shot_pdf_root": "example-pdfs",
    }
    assert '"result_count": 20' in capsys.readouterr().out


def test_vlm_eval_cli_never_routes_through_agent_evaluator(monkeypatch) -> None:
    received = {}

    def fake_eval(diagnostic, predictions, output):
        received.update(
            {
                "diagnostic": diagnostic,
                "predictions": predictions,
                "output": output,
            }
        )
        return {"evaluation_id": "fixture"}

    monkeypatch.setattr(
        "auditops.cli.evaluate_companies_house_vlm_diagnostic", fake_eval
    )
    monkeypatch.setattr(
        "auditops.cli.write_evaluation_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("XHTML/agent evaluator must remain isolated")
        ),
    )
    assert (
        main(
            [
                "eval-companies-house-vlm-diagnostic",
                "--diagnostic-dir",
                "diagnostic",
                "--predictions-jsonl",
                "predictions.jsonl",
                "--output-dir",
                "evaluation",
            ]
        )
        == 0
    )
    assert received == {
        "diagnostic": "diagnostic",
        "predictions": "predictions.jsonl",
        "output": "evaluation",
    }
