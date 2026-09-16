from __future__ import annotations

import hashlib
import json
import sys
import types

import pytest

from auditops.agent_contracts import build_model_config
from auditops.canonical_json import canonical_json_sha256
from auditops.model_adapter import (
    MockModelAdapter,
    ModelAdapterError,
    ModelBackendError,
    ModelDependencyError,
    ModelGenerationError,
    ModelRequest,
    ModelResponse,
    OfflineVLLMAdapter,
    StrictJSONError,
    parse_strict_json_object,
)

OBJECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


@pytest.mark.parametrize(
    "text",
    [
        '{"answer":1e999}',
        '{"answer":-1e999}',
        '{"answer":"\\ud800"}',
        '{"\\udfff":"answer"}',
        '{"answer":"' + chr(0xD800) + '"}',
        '{"answer":' + "[" * 1100 + "0" + "]" * 1100 + "}",
    ],
    ids=[
        "positive-overflow",
        "negative-overflow",
        "escaped-surrogate",
        "surrogate-key",
        "raw-surrogate",
        "deep-nesting",
    ],
)
def test_strict_json_rejects_unrepresentable_data_with_typed_failure(text):
    with pytest.raises(StrictJSONError) as caught:
        parse_strict_json_object(text)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    for frame in _tracebacks(caught.value):
        assert text not in str(frame.tb_frame.f_locals)


def test_duplicate_key_error_does_not_retain_untrusted_key():
    marker = "PRIVATE_DUPLICATE_KEY_MARKER"
    with pytest.raises(StrictJSONError) as caught:
        parse_strict_json_object(json.dumps({marker: 1})[:-1] + f',"{marker}":2}}')
    for frame in _tracebacks(caught.value):
        assert marker not in str(frame.tb_frame.f_locals)


@pytest.mark.parametrize(
    "metadata",
    [
        {"output_bytes": "not-a-count"},
        {"output_bytes": True},
        {"output_bytes": -1},
        {"structured_output_applied": "false"},
        {"constraint_backend": []},
        {"engine_initialization_ms": -1},
        {"engine_initialization_ms": 10**500},
        {"peak_vram_bytes": True},
        {"cap_hit": "false"},
    ],
)
def test_model_response_rejects_invalid_attempt_telemetry(metadata):
    with pytest.raises(ModelGenerationError):
        ModelResponse("id", {"answer": "ok"}, "a" * 64, metadata=metadata)


def _request(**overrides):
    fields = {
        "request_id": "request-1",
        "messages": (
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Answer the fixture task."},
        ),
        "json_schema": OBJECT_SCHEMA,
        "max_tokens": 32,
        "stage": "direct",
        "repair_attempt": False,
        "prompt_version": "auditops.agent_prompt.v2",
        "runtime_version": "auditops.agent_runtime.v2",
        "model_semantic_config": {
            "model_id": "auditops/mock",
            "revision": "test-v1",
        },
    }
    fields.update(overrides)
    return ModelRequest(**fields)


def test_strict_json_parser_rejects_prose_duplicates_nonfinite_and_nonobject():
    assert parse_strict_json_object('{"answer":"ok"}') == {"answer": "ok"}

    for invalid in (
        '```json\n{"answer":"ok"}\n```',
        '{"answer":"ok"} trailing',
        '{"answer":"first","answer":"second"}',
        '{"answer":NaN}',
        '[{"answer":"ok"}]',
    ):
        with pytest.raises(StrictJSONError) as exc_info:
            parse_strict_json_object(invalid)
        assert exc_info.value.response_sha256 is not None
        assert invalid not in str(exc_info.value)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        for traceback in _tracebacks(exc_info.value):
            assert invalid not in str(traceback.tb_frame.f_locals)


def test_strict_json_error_retains_only_canonical_safe_metadata():
    digest = "a" * 64
    error = StrictJSONError(
        response_sha256=digest,
        input_tokens=3,
        output_tokens=4,
        duration_ms=5.0,
        finish_reason="length",
    )
    assert error.args == ("Model output is not one strict JSON object",)
    assert error.response_sha256 == digest

    marker = "RAW_MODEL_OR_ENV_MARKER"
    sanitized = StrictJSONError(marker, response_sha256=digest)
    assert sanitized.args == ("Model output is not one strict JSON object",)
    assert marker not in str(sanitized)
    assert marker not in str(sanitized.__dict__)

    for invalid_digest in (None, "bad", "A" * 64):
        with pytest.raises(ModelAdapterError, match="response digest is invalid"):
            StrictJSONError(response_sha256=invalid_digest)

    for invalid_telemetry in (
        {"input_tokens": True},
        {"output_tokens": -1},
        {"duration_ms": float("nan")},
        {"duration_ms": 10**500},
        {"parse_category": [marker]},
        {"finish_reason": marker},
        {"finish_reason": {"environment": marker}},
        {"finish_reason": [marker]},
    ):
        with pytest.raises(ModelAdapterError) as telemetry_exc:
            StrictJSONError(response_sha256=digest, **invalid_telemetry)
        assert marker not in str(telemetry_exc.value)
        for traceback in _tracebacks(telemetry_exc.value):
            assert marker not in str(traceback.tb_frame.f_locals)

    with pytest.raises(ModelAdapterError) as exc_info:
        StrictJSONError(marker, response_sha256={"environment": marker})
    assert marker not in str(exc_info.value)
    for traceback in _tracebacks(exc_info.value):
        assert marker not in str(traceback.tb_frame.f_locals)


def _tracebacks(exc: BaseException):
    traceback = exc.__traceback__
    while traceback is not None:
        normalized = traceback.tb_frame.f_code.co_filename.replace("\\", "/")
        if normalized.endswith("/auditops/model_adapter.py"):
            yield traceback
        traceback = traceback.tb_next


def test_model_request_locks_sampling_and_message_shape():
    with pytest.raises(ValueError, match="temperature=0"):
        _request(temperature=0.1)
    with pytest.raises(ValueError, match="top_p=1"):
        _request(top_p=0.9)
    with pytest.raises(ValueError, match="only role and content"):
        _request(messages=({"role": "user", "content": "x", "name": "extra"},))
    with pytest.raises(ValueError, match="top-level object"):
        _request(json_schema={"type": "array"})
    with pytest.raises(ValueError, match="deployment-local model_path"):
        _request(model_semantic_config={"model_path": "C:/local/model"})


def test_model_request_hash_binds_complete_semantic_envelope():
    request = _request()

    assert request.request_envelope["response_schema_sha256"] == (
        canonical_json_sha256(OBJECT_SCHEMA)
    )
    assert request.request_sha256 == canonical_json_sha256(request.request_envelope)
    assert _request(request_id="deployment-local-id").request_sha256 == (
        request.request_sha256
    )
    assert (
        _request(
            json_schema={
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
                "type": "object",
            }
        ).request_sha256
        == request.request_sha256
    )

    mutations = (
        {"messages": ({"role": "user", "content": "different"},)},
        {
            "json_schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["answer"],
                "properties": {"answer": {"type": "number"}},
            }
        },
        {"max_tokens": 31},
        {"stage": "synthesis"},
        {"repair_attempt": True},
        {"prompt_version": "auditops.agent_prompt.v2.1"},
        {"runtime_version": "auditops.agent_runtime.v2.1"},
        {"model_semantic_config": {"model_id": "other", "revision": "test-v1"}},
        {"seed": 20260822},
    )
    for mutation in mutations:
        assert _request(**mutation).request_sha256 != request.request_sha256


def test_mock_adapter_is_deterministic_and_records_only_parsed_payloads():
    adapter = MockModelAdapter(({"answer": "first"}, '{"answer":"second"}'))

    first = adapter.generate_json(_request(request_id="one"))
    second = adapter.generate_json(_request(request_id="two"))

    assert first.payload == {"answer": "first"}
    assert second.payload == {"answer": "second"}
    assert [request.request_id for request in adapter.requests] == ["one", "two"]
    assert adapter.remaining_responses == 0
    assert "raw_text" not in first.__dict__
    with pytest.raises(ModelGenerationError, match="no scripted response"):
        adapter.generate_json(_request(request_id="three"))


def test_mock_adapter_propagates_typed_scripted_failure():
    failure = ModelGenerationError("fixture inference failed")
    adapter = MockModelAdapter((failure,))
    with pytest.raises(ModelGenerationError, match="fixture inference failed"):
        adapter.generate_json(_request())


@pytest.mark.parametrize(
    "model_id", ["Qwen/Qwen3.5-27B-FP8", "google/gemma-4-31B-it-qat-w4a16-ct"]
)
def test_offline_vllm_adapter_lazy_loads_and_uses_in_process_chat(
    monkeypatch, tmp_path, model_id
):
    calls = {"engine": [], "chat": [], "sampling": [], "structured": []}

    class FakeStructuredOutputsParams:
        def __init__(self, **kwargs):
            calls["structured"].append(kwargs)

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            calls["sampling"].append(kwargs)

    class FakeCompletion:
        text = '{"answer":"offline"}'
        token_ids = (10, 11)
        finish_reason = "stop"

    class FakeRequestOutput:
        prompt_token_ids = (1, 2, 3)
        outputs = (FakeCompletion(),)

    class FakeLLM:
        def __init__(self, **kwargs):
            calls["engine"].append(kwargs)

        def chat(self, messages, **kwargs):
            calls["chat"].append((messages, kwargs))
            return [FakeRequestOutput()]

        def get_tokenizer(self):
            return types.SimpleNamespace(
                encode=lambda text, add_special_tokens: list(range(len(text.split()))),
                apply_chat_template=lambda messages, **kwargs: {
                    "input_ids": list(
                        range(
                            sum(len(message["content"].split()) for message in messages)
                            + 3
                        )
                    ),
                    "attention_mask": [1]
                    * (
                        sum(len(message["content"].split()) for message in messages) + 3
                    ),
                },
            )

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    fake_sampling_module = types.ModuleType("vllm.sampling_params")
    fake_sampling_module.StructuredOutputsParams = FakeStructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_sampling_module)
    monkeypatch.setattr(
        "auditops.model_adapter.importlib_metadata.version", lambda _: "0.26.0"
    )

    model_path = tmp_path / "model-snapshot"
    model_path.mkdir()
    chat_template = "fixture {{ messages }}"
    (model_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": chat_template}), encoding="utf-8"
    )
    config = build_model_config(
        model_id=model_id,
        revision="abc123",
        model_path=str(model_path),
        quantization=None,
        backend="vllm_offline",
        chat_template_sha256=hashlib.sha256(chat_template.encode()).hexdigest(),
    )
    adapter = OfflineVLLMAdapter(config)
    assert adapter._engine is None

    assert adapter.count_tokens("one two three") == 3
    assert adapter.count_messages(_request().messages) == 10
    response = adapter.generate_json(_request())

    assert response.payload == {"answer": "offline"}
    assert response.input_tokens == 3
    assert response.output_tokens == 2
    assert calls["engine"] == [
        {
            "model": str(model_path.resolve()),
            "max_model_len": 16384,
            "max_num_seqs": 256,
            "seed": 20260821,
            "trust_remote_code": False,
            "structured_outputs_config": {"backend": "xgrammar"},
        }
    ]
    assert calls["structured"] == [{"json": OBJECT_SCHEMA}]
    assert calls["sampling"][0]["temperature"] == 0.0
    assert calls["sampling"][0]["top_p"] == 1.0
    assert calls["sampling"][0]["seed"] == 20260821
    assert calls["sampling"][0]["max_tokens"] == 32
    assert calls["chat"][0][1]["use_tqdm"] is False
    assert calls["chat"][0][1]["chat_template_kwargs"] == {"enable_thinking": False}

    unsupported_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string"},
            }
        },
    }
    with pytest.raises(ModelBackendError, match="unsupported by the pinned xgrammar"):
        adapter.generate_json(
            _request(request_id="unsupported", json_schema=unsupported_schema)
        )
    assert len(calls["chat"]) == 1

    FakeCompletion.text = '{"answer":'
    FakeCompletion.token_ids = (20, 21, 22, 23)
    FakeCompletion.finish_reason = "length"
    with pytest.raises(StrictJSONError) as exc_info:
        adapter.generate_json(_request(request_id="truncated"))
    assert (
        exc_info.value.response_sha256
        == hashlib.sha256(FakeCompletion.text.encode()).hexdigest()
    )
    assert exc_info.value.input_tokens == 3
    assert exc_info.value.output_tokens == 4
    assert exc_info.value.duration_ms is not None
    assert exc_info.value.finish_reason == "length"
    assert FakeCompletion.text not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    for traceback in _tracebacks(exc_info.value):
        assert FakeCompletion.text not in str(traceback.tb_frame.f_locals)

    for malformed in ('{"answer":"\\ud800"}', '{"answer":1e999}'):
        FakeCompletion.text = malformed
        FakeCompletion.finish_reason = "stop"
        with pytest.raises(StrictJSONError) as stopped_error:
            adapter.generate_json(_request(request_id="invalid-stopped"))
        assert stopped_error.value.finish_reason == "stop"
        assert stopped_error.value.structured_output_applied is True
        assert stopped_error.value.__context__ is None
        for traceback in _tracebacks(stopped_error.value):
            assert malformed not in str(traceback.tb_frame.f_locals)

    marker = "RAW_BACKEND_TELEMETRY_MARKER"
    malformed_text = FakeCompletion.text
    FakeCompletion.finish_reason = {"environment": marker}
    with pytest.raises(ModelAdapterError) as exc_info:
        adapter.generate_json(_request(request_id="invalid-telemetry"))
    assert marker not in str(exc_info.value)
    assert malformed_text not in str(exc_info.value)
    for traceback in _tracebacks(exc_info.value):
        frame_locals = str(traceback.tb_frame.f_locals)
        assert marker not in frame_locals
        assert malformed_text not in frame_locals

    wrong_template_config = {**config, "chat_template_sha256": "b" * 64}
    wrong_template = OfflineVLLMAdapter(wrong_template_config)
    with pytest.raises(ModelDependencyError, match="chat template hash"):
        wrong_template.count_tokens("template check")

    monkeypatch.setattr(
        "auditops.model_adapter.importlib_metadata.version", lambda _: "0.25.1"
    )
    mismatched = OfflineVLLMAdapter(config)
    with pytest.raises(ModelDependencyError, match="requires vLLM 0.26.0"):
        mismatched.count_tokens("version check")


def test_offline_vllm_adapter_refuses_missing_or_overridden_local_snapshot(tmp_path):
    missing_config = build_model_config(
        model_id="fixture/model",
        revision="abc123",
        model_path=str(tmp_path / "missing"),
        backend="vllm_offline",
        chat_template_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="does not exist"):
        OfflineVLLMAdapter(missing_config)

    model_path = tmp_path / "snapshot"
    model_path.mkdir()
    chat_template = "fixture template"
    (model_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": chat_template}), encoding="utf-8"
    )
    config = build_model_config(
        model_id="fixture/model",
        revision="abc123",
        model_path=str(model_path),
        backend="vllm_offline",
        chat_template_sha256=hashlib.sha256(chat_template.encode()).hexdigest(),
    )
    with pytest.raises(ValueError, match="cannot override"):
        OfflineVLLMAdapter(config, engine_options={"tokenizer": "remote/model"})
    with pytest.raises(ValueError, match="cannot override"):
        OfflineVLLMAdapter(config, engine_options={"max_num_seqs": 817})
