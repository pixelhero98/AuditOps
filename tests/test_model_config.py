from __future__ import annotations

import json

import pytest

from auditops.agent_contracts import validate_model_config
from auditops.model_config import create_pinned_model_config, write_pinned_model_config


def test_create_pinned_model_config_hashes_chat_template(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    config = create_pinned_model_config(
        model_id="example/model",
        revision="a" * 40,
        model_path=model,
        quantization="fp8",
    )
    validate_model_config(config)
    assert config["model_path"] == str(model.resolve())
    assert len(config["chat_template_sha256"]) == 64
    assert config["thinking_enabled"] is False


def test_model_config_requires_commit_snapshot_and_no_overwrite(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "chat_template.jinja").write_text("template", encoding="utf-8")
    with pytest.raises(ValueError):
        create_pinned_model_config(
            model_id="example/model",
            revision="main",
            model_path=model,
            quantization=None,
        )
    output = tmp_path / "config.json"
    write_pinned_model_config(
        output,
        model_id="example/model",
        revision="b" * 40,
        model_path=model,
        quantization=None,
    )
    with pytest.raises(FileExistsError):
        write_pinned_model_config(
            output,
            model_id="example/model",
            revision="b" * 40,
            model_path=model,
            quantization=None,
        )


def test_model_config_secret_scan_prevents_publication(tmp_path, monkeypatch):
    synthetic_secret = "synthetic-model-config-secret-123456"
    monkeypatch.setenv("HF_TOKEN", synthetic_secret)
    model = tmp_path / "model"
    model.mkdir()
    (model / "chat_template.jinja").write_text("template", encoding="utf-8")
    output = tmp_path / "config.json"

    with pytest.raises(ValueError, match="Secret scan failed"):
        write_pinned_model_config(
            output,
            model_id=f"example/{synthetic_secret}",
            revision="c" * 40,
            model_path=model,
            quantization=None,
        )

    assert not output.exists()
