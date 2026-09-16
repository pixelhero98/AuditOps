"""Model-neutral, offline-only JSON generation adapters for AuditOps agents.

The runtime deliberately depends on this small interface instead of a model
server.  Production inference is performed with vLLM's in-process ``LLM`` API;
the vLLM package is imported only when the first request is made so CPU-only
corpus and evaluation commands do not acquire a GPU dependency.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from .canonical_json import (
    CANONICAL_JSON_VERSION,
    canonical_json_sha256,
)

DEFAULT_SEED = 20260821
REQUIRED_VLLM_VERSION = "0.26.0"
MODEL_REQUEST_ENVELOPE_VERSION = "auditops.model_request.v2.2"
# AuditOps executes one bounded case at a time.  Pinning a much smaller
# scheduler concurrency than vLLM's 1,024-sequence default avoids reserving an
# impossible number of recurrent-state cache blocks for hybrid GDN/Mamba
# checkpoints such as Qwen3.5, without changing any request semantics.
OFFLINE_MAX_NUM_SEQS = 256
_XGRAMMAR_SUPPORTED_STRING_FORMATS = {
    "email",
    "date",
    "time",
    "date-time",
    "duration",
    "ipv4",
    "ipv6",
    "hostname",
    "uuid",
    "uri",
    "uri-reference",
    "uri-template",
    "json-pointer",
    "relative-json-pointer",
}


class ModelAdapterError(RuntimeError):
    """Base class for typed model-adapter failures."""


class ModelDependencyError(ModelAdapterError):
    """Raised when the optional inference dependency is unavailable."""


class ModelGenerationError(ModelAdapterError):
    """Raised when the inference engine fails or returns no completion."""


class ModelTimeoutError(ModelGenerationError):
    """Raised when the offline backend exceeds its bounded execution time."""


class ModelOOMError(ModelGenerationError):
    """Raised when the pinned model cannot complete within device memory."""


class ModelBackendError(ModelGenerationError):
    """Raised for other offline backend failures."""


class StrictJSONError(ModelAdapterError):
    """Raised when a model completion is not exactly one JSON object.

    Only a digest and bounded primitive telemetry are retained.  In particular,
    the exception does not retain or echo a potentially sensitive/free-form
    completion into run artifacts or traceback state.
    """

    def __init__(
        self,
        _untrusted_message: object | None = None,
        *,
        response_sha256: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        duration_ms: float | None = None,
        finish_reason: str | None = None,
        output_bytes: int | None = None,
        parse_category: str | None = None,
        constraint_backend: str | None = None,
        structured_output_applied: bool | None = None,
    ) -> None:
        # BaseException.__new__ initially mirrors positional constructor values
        # into ``args``.  Replace them before any metadata validation can fail.
        super().__init__("Model output is not one strict JSON object")
        _untrusted_message = None
        digest_valid = (
            isinstance(response_sha256, str)
            and len(response_sha256) == 64
            and all(character in "0123456789abcdef" for character in response_sha256)
        )
        tokens_valid = all(
            value is None
            or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)
            for value in (input_tokens, output_tokens)
        )
        duration_valid = duration_ms is None or (
            isinstance(duration_ms, (int, float))
            and not isinstance(duration_ms, bool)
            and math.isfinite(float(duration_ms))
            and duration_ms >= 0
        )
        finish_reason_valid = finish_reason is None or (
            isinstance(finish_reason, str)
            and finish_reason in {"stop", "length", "abort"}
        )
        output_bytes_valid = output_bytes is None or (
            isinstance(output_bytes, int)
            and not isinstance(output_bytes, bool)
            and output_bytes >= 0
        )
        parse_category_valid = parse_category is None or parse_category in {
            "EMPTY",
            "NON_OBJECT",
            "SYNTAX",
            "DUPLICATE_KEY",
            "NON_JSON_NUMBER",
        }
        constraint_backend_valid = constraint_backend is None or (
            isinstance(constraint_backend, str) and bool(constraint_backend)
        )
        structured_output_valid = structured_output_applied is None or isinstance(
            structured_output_applied, bool
        )
        invalid_message = None
        if not digest_valid:
            invalid_message = "Strict JSON failure response digest is invalid"
        elif not tokens_valid:
            invalid_message = "Strict JSON failure token telemetry is invalid"
        elif not duration_valid:
            invalid_message = "Strict JSON failure duration telemetry is invalid"
        elif not finish_reason_valid:
            invalid_message = "Strict JSON failure finish reason is invalid"
        elif not output_bytes_valid:
            invalid_message = "Strict JSON failure byte telemetry is invalid"
        elif not parse_category_valid:
            invalid_message = "Strict JSON failure parse category is invalid"
        elif not constraint_backend_valid:
            invalid_message = "Strict JSON failure constraint backend is invalid"
        elif not structured_output_valid:
            invalid_message = (
                "Strict JSON failure structured-output telemetry is invalid"
            )
        if invalid_message is not None:
            # Untrusted adapters can pass arbitrary objects here.  Clear every
            # caller-controlled local before raising so the typed adapter error
            # and its traceback retain no raw output or environment material.
            _untrusted_message = None
            response_sha256 = None
            input_tokens = None
            output_tokens = None
            duration_ms = None
            finish_reason = None
            output_bytes = None
            parse_category = None
            constraint_backend = None
            structured_output_applied = None
            raise ModelAdapterError(invalid_message) from None

        _untrusted_message = None
        self.response_sha256 = str(response_sha256)
        self.input_tokens = int(input_tokens) if input_tokens is not None else None
        self.output_tokens = int(output_tokens) if output_tokens is not None else None
        self.duration_ms = float(duration_ms) if duration_ms is not None else None
        self.finish_reason = str(finish_reason) if finish_reason is not None else None
        self.output_bytes = int(output_bytes) if output_bytes is not None else None
        self.parse_category = (
            str(parse_category) if parse_category is not None else None
        )
        self.constraint_backend = (
            str(constraint_backend) if constraint_backend is not None else None
        )
        self.structured_output_applied = structured_output_applied


class ModelSchemaError(ModelAdapterError):
    """A parsed JSON object violated the exact request schema."""

    def __init__(
        self,
        *,
        response_sha256: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        output_bytes: int = 0,
        duration_ms: float | None = None,
        finish_reason: str | None = None,
        constraint_backend: str | None = None,
        structured_output_applied: bool | None = None,
    ) -> None:
        super().__init__("Model JSON object does not satisfy the response schema")
        self.response_sha256 = response_sha256
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.output_bytes = output_bytes
        self.duration_ms = duration_ms
        self.finish_reason = finish_reason
        self.constraint_backend = constraint_backend
        self.structured_output_applied = structured_output_applied


def _validate_schema_payload(
    payload: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    response_sha256: str,
    output_bytes: int,
) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - locked runtime dependency
        raise ModelDependencyError(
            "Structured response validation requires jsonschema"
        ) from exc
    validator = Draft202012Validator(schema)
    if next(validator.iter_errors(payload), None) is not None:
        raise ModelSchemaError(
            response_sha256=response_sha256,
            output_bytes=output_bytes,
        ) from None


def validate_schema_payload(
    payload: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    response_sha256: str,
    output_bytes: int,
) -> None:
    """Revalidate an adapter response at the runtime trust boundary."""

    _validate_schema_payload(
        payload,
        schema,
        response_sha256=response_sha256,
        output_bytes=output_bytes,
    )


def _reject_non_json_number(value: str) -> None:
    raise ValueError(f"Non-JSON numeric constant is not permitted: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key is not permitted: {key}")
        result[key] = value
    return result


def parse_strict_json_object(text: str) -> dict[str, Any]:
    """Parse one RFC-8259-style JSON object with no prose or code fences."""

    if not isinstance(text, str):
        raise TypeError("Model output must be text")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    output_bytes = len(text.encode("utf-8"))
    if not text.strip():
        text = ""
        raise StrictJSONError(
            response_sha256=digest,
            output_bytes=output_bytes,
            parse_category="EMPTY",
        ) from None
    parse_failed = False
    parse_category = "SYNTAX"
    try:
        value = json.loads(
            text,
            parse_constant=_reject_non_json_number,
            object_pairs_hook=_unique_object,
        )
    except ValueError as exc:
        parse_failed = True
        message = str(exc)
        if "Duplicate JSON object key" in message:
            parse_category = "DUPLICATE_KEY"
        elif "Non-JSON numeric constant" in message:
            parse_category = "NON_JSON_NUMBER"
        value = None
    if parse_failed:
        # Raise outside the parser exception handler and clear the source text
        # first.  Otherwise JSONDecodeError.doc, implicit exception context, or
        # traceback-frame locals can retain the untrusted completion.
        text = ""
        raise StrictJSONError(
            response_sha256=digest,
            output_bytes=output_bytes,
            parse_category=parse_category,
        ) from None
    if not isinstance(value, dict):
        # The parsed value can itself contain the full completion.  Do not leave
        # either representation reachable from a failure traceback.
        text = ""
        value = None
        raise StrictJSONError(
            response_sha256=digest,
            output_bytes=output_bytes,
            parse_category="NON_OBJECT",
        ) from None
    return value


def _validate_messages(
    messages: Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], ...]:
    if not messages:
        raise ValueError("At least one model message is required")
    normalized = []
    for index, message in enumerate(messages):
        if set(message) != {"role", "content"}:
            raise ValueError(f"Message {index} must contain only role and content")
        role = message["role"]
        content = message["content"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported message role at index {index}: {role!r}")
        if not isinstance(content, str) or not content:
            raise ValueError(f"Message content at index {index} must be non-empty text")
        normalized.append({"role": role, "content": content})
    return tuple(normalized)


def _single_input_token_ids(tokenized: Any) -> Sequence[int]:
    """Extract one flat token-ID sequence from a tokenizer result.

    Multimodal-family tokenizers such as Gemma 4 may return a BatchEncoding
    mapping even for a single text-only chat.  Counting that mapping reports
    its number of fields rather than its number of input tokens.
    """

    if isinstance(tokenized, Mapping):
        if "input_ids" not in tokenized:
            raise ModelGenerationError(
                "Rendered chat tokenizer did not return input_ids"
            )
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if isinstance(tokenized, tuple):
        tokenized = list(tokenized)
    if not isinstance(tokenized, list):
        raise ModelGenerationError(
            "Rendered chat tokenizer returned an unsupported token container"
        )
    if tokenized and isinstance(tokenized[0], (list, tuple)):
        if len(tokenized) != 1:
            raise ModelGenerationError("Rendered chat tokenizer returned a batch")
        tokenized = list(tokenized[0])
    if any(
        not isinstance(token_id, int) or isinstance(token_id, bool)
        for token_id in tokenized
    ):
        raise ModelGenerationError(
            "Rendered chat tokenizer returned non-integer token IDs"
        )
    return tokenized


def _classified_backend_error(exc: Exception, *, action: str) -> ModelAdapterError:
    """Map backend exceptions to stable classes without retaining raw details."""

    class_name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    if (
        isinstance(exc, TimeoutError)
        or "timeout" in class_name
        or "timed out" in message
    ):
        return ModelTimeoutError(f"Offline vLLM {action} timed out")
    oom_markers = ("outofmemory", "out of memory", "cuda oom", "memoryerror")
    if any(marker in class_name or marker in message for marker in oom_markers):
        return ModelOOMError(f"Offline vLLM {action} exhausted device memory")
    return ModelBackendError(f"Offline vLLM {action} failed")


def xgrammar_json_schema_supported(schema: Mapping[str, Any]) -> bool:
    """Mirror the pinned vLLM 0.26 xgrammar feature gate without importing vLLM."""

    def supported(value: Any) -> bool:
        if isinstance(value, Mapping):
            value_type = value.get("type")
            if value_type in {"integer", "number"} and "multipleOf" in value:
                return False
            if value_type == "array" and any(
                key in value
                for key in ("uniqueItems", "contains", "minContains", "maxContains")
            ):
                return False
            if (
                value_type == "string"
                and "format" in value
                and value["format"] not in _XGRAMMAR_SUPPORTED_STRING_FORMATS
            ):
                return False
            if value_type == "object" and any(
                key in value for key in ("patternProperties", "propertyNames")
            ):
                return False
            return all(supported(item) for item in value.values())
        if isinstance(value, list):
            return all(supported(item) for item in value)
        return True

    return supported(schema)


@dataclass(frozen=True)
class ModelRequest:
    """One schema-constrained, deterministic generation request."""

    request_id: str
    messages: Sequence[Mapping[str, str]]
    json_schema: Mapping[str, Any]
    max_tokens: int
    stage: str
    repair_attempt: bool
    prompt_version: str
    runtime_version: str
    model_semantic_config: Mapping[str, Any]
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be non-empty text")
        object.__setattr__(self, "messages", _validate_messages(self.messages))
        if self.stage not in {"direct", "plan", "synthesis"}:
            raise ValueError("stage must be direct, plan, or synthesis")
        if not isinstance(self.repair_attempt, bool):
            raise TypeError("repair_attempt must be a boolean")
        if not isinstance(self.prompt_version, str) or not self.prompt_version:
            raise ValueError("prompt_version must be non-empty text")
        if not isinstance(self.runtime_version, str) or not self.runtime_version:
            raise ValueError("runtime_version must be non-empty text")
        if not isinstance(self.model_semantic_config, Mapping):
            raise TypeError("model_semantic_config must be an object")
        semantic_config = copy.deepcopy(dict(self.model_semantic_config))
        if "model_path" in semantic_config:
            raise ValueError(
                "model_semantic_config cannot contain deployment-local model_path"
            )
        canonical_json_sha256(semantic_config)
        object.__setattr__(self, "model_semantic_config", semantic_config)
        if (
            not isinstance(self.json_schema, Mapping)
            or self.json_schema.get("type") != "object"
        ):
            raise ValueError("json_schema must describe a top-level object")
        try:
            json.dumps(self.json_schema, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("json_schema must be JSON serializable") from exc
        if (
            not isinstance(self.max_tokens, int)
            or isinstance(self.max_tokens, bool)
            or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(self.temperature, (int, float)) or not math.isfinite(
            float(self.temperature)
        ):
            raise ValueError("temperature must be finite")
        if float(self.temperature) != 0.0:
            raise ValueError("AuditOps baseline requires temperature=0")
        if not isinstance(self.top_p, (int, float)) or float(self.top_p) != 1.0:
            raise ValueError("AuditOps baseline requires top_p=1")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")

    @property
    def response_schema_sha256(self) -> str:
        """Digest the exact structured-output schema sent to the backend."""

        return canonical_json_sha256(self.json_schema)

    @property
    def request_envelope(self) -> dict[str, Any]:
        """Return the portable semantic request; deployment request IDs are excluded."""

        return {
            "request_envelope_version": MODEL_REQUEST_ENVELOPE_VERSION,
            "canonical_json_version": CANONICAL_JSON_VERSION,
            "stage": self.stage,
            "repair_attempt": self.repair_attempt,
            "messages": [dict(message) for message in self.messages],
            "response_schema": copy.deepcopy(dict(self.json_schema)),
            "response_schema_sha256": self.response_schema_sha256,
            "max_tokens": self.max_tokens,
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "seed": self.seed,
            "prompt_version": self.prompt_version,
            "runtime_version": self.runtime_version,
            "model_semantic_config": copy.deepcopy(dict(self.model_semantic_config)),
        }

    @property
    def request_sha256(self) -> str:
        """Hash the complete portable semantic request envelope."""

        return canonical_json_sha256(self.request_envelope)


@dataclass(frozen=True)
class ModelResponse:
    """Parsed model response; raw/free-form generations are not retained."""

    request_id: str
    payload: Mapping[str, Any]
    response_sha256: str
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    finish_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ModelGenerationError("Model response request_id is invalid")
        if not isinstance(self.payload, Mapping):
            raise ModelGenerationError("Model response payload is not a JSON object")
        materialized_payload = copy.deepcopy(dict(self.payload))
        try:
            canonical_json_sha256(materialized_payload)
        except (TypeError, ValueError) as exc:
            raise ModelGenerationError(
                "Model response payload is not canonical JSON data"
            ) from exc
        object.__setattr__(self, "payload", materialized_payload)
        if (
            not isinstance(self.response_sha256, str)
            or len(self.response_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.response_sha256
            )
        ):
            raise ModelGenerationError("Model response digest is invalid")
        for field_name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ModelGenerationError(
                    f"Model response {field_name} is not a nonnegative integer"
                )
        if (
            not isinstance(self.duration_ms, (int, float))
            or isinstance(self.duration_ms, bool)
            or not math.isfinite(float(self.duration_ms))
            or self.duration_ms < 0
        ):
            raise ModelGenerationError(
                "Model response duration_ms is not a finite nonnegative number"
            )
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise ModelGenerationError("Model response finish_reason is invalid")
        if not isinstance(self.metadata, Mapping):
            raise ModelGenerationError("Model response metadata is not an object")
        materialized_metadata = copy.deepcopy(dict(self.metadata))
        try:
            canonical_json_sha256(materialized_metadata)
        except (TypeError, ValueError) as exc:
            raise ModelGenerationError(
                "Model response metadata is not canonical JSON data"
            ) from exc
        object.__setattr__(self, "metadata", materialized_metadata)


class ModelAdapter(ABC):
    """Abstract in-process adapter used by the bounded AuditOps runtime."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the immutable model identifier used for this adapter."""

    @property
    @abstractmethod
    def model_revision(self) -> str:
        """Return the pinned model revision used for this adapter."""

    @property
    @abstractmethod
    def backend(self) -> str:
        """Return ``mock`` or ``vllm_offline``."""

    @property
    @abstractmethod
    def peak_vram_bytes(self) -> int | None:
        """Return the highest observed device-memory usage when available."""

    @property
    def constraint_backend(self) -> str:
        """Return the resolved structured-output implementation."""

        return "adapter_contract"

    @property
    def structured_output_applied(self) -> bool:
        """Whether every generate_json request is grammar constrained."""

        return True

    @property
    def engine_initialization_ms(self) -> float:
        """One-time engine/tokenizer initialization cost for this adapter."""

        return 0.0

    @abstractmethod
    def generate_json(self, request: ModelRequest) -> ModelResponse:
        """Generate exactly one schema-constrained JSON object."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count input tokens with the adapter's pinned tokenizer."""

    @abstractmethod
    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int:
        """Count a fully rendered chat, including template control tokens."""


class MockModelAdapter(ModelAdapter):
    """Deterministic scripted adapter for unit and CPU integration tests."""

    def __init__(
        self,
        responses: Sequence[Mapping[str, Any] | str | Exception],
        *,
        model_id: str = "auditops/mock",
        revision: str = "test-v1",
    ) -> None:
        if not responses:
            raise ValueError("MockModelAdapter requires at least one scripted response")
        self._responses: deque[Mapping[str, Any] | str | Exception] = deque(responses)
        self._model_id = model_id
        self._revision = revision
        self.requests: list[ModelRequest] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str:
        return self._revision

    @property
    def backend(self) -> str:
        return "mock"

    @property
    def peak_vram_bytes(self) -> int | None:
        return None

    @property
    def constraint_backend(self) -> str:
        return "mock_schema"

    @property
    def structured_output_applied(self) -> bool:
        return True

    @property
    def remaining_responses(self) -> int:
        return len(self._responses)

    def generate_json(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise ModelGenerationError(
                "MockModelAdapter has no scripted response remaining"
            )
        scripted = self._responses.popleft()
        if isinstance(scripted, Exception):
            raise scripted
        if isinstance(scripted, str):
            text = scripted
        else:
            text = json.dumps(
                scripted, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
        payload = parse_strict_json_object(text)
        response_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            _validate_schema_payload(
                payload,
                request.json_schema,
                response_sha256=response_sha256,
                output_bytes=len(text.encode("utf-8")),
            )
        except ModelSchemaError as exc:
            raise ModelSchemaError(
                response_sha256=exc.response_sha256,
                input_tokens=sum(
                    len(message["content"].split()) for message in request.messages
                ),
                output_tokens=len(text.split()),
                output_bytes=exc.output_bytes,
                duration_ms=0.0,
                finish_reason="stop",
                constraint_backend=self.constraint_backend,
                structured_output_applied=self.structured_output_applied,
            ) from None
        return ModelResponse(
            request_id=request.request_id,
            payload=payload,
            response_sha256=response_sha256,
            input_tokens=sum(
                len(message["content"].split()) for message in request.messages
            ),
            output_tokens=len(text.split()),
            duration_ms=0.0,
            finish_reason="stop",
            metadata={
                "backend": self.backend,
                "constraint_backend": self.constraint_backend,
                "structured_output_applied": self.structured_output_applied,
                "output_bytes": len(text.encode("utf-8")),
                "cap_hit": False,
            },
        )

    def count_tokens(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return len(text.split())

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int:
        normalized = _validate_messages(messages)
        return sum(len(message["content"].split()) + 1 for message in normalized) + 1


class OfflineVLLMAdapter(ModelAdapter):
    """Lazy, offline vLLM adapter with structured JSON decoding.

    ``LLM`` is instantiated in-process on first use.  This class does not open a
    socket or start an OpenAI-compatible server.  ``trust_remote_code`` is fixed
    to false; model snapshots must therefore work with the pinned vLLM release.
    """

    def __init__(
        self,
        model_config: Mapping[str, Any],
        *,
        engine_options: Mapping[str, Any] | None = None,
    ) -> None:
        from .agent_contracts import validate_model_config

        validate_model_config(model_config)
        if model_config["backend"] != "vllm_offline":
            raise ValueError("OfflineVLLMAdapter requires backend='vllm_offline'")
        if model_config["thinking_enabled"]:
            raise ValueError("AuditOps baseline does not permit thinking mode")
        self.model_config = dict(model_config)
        model_path = Path(str(model_config["model_path"])).expanduser()
        if not model_path.is_dir():
            raise ValueError(
                f"Offline model snapshot directory does not exist: {model_path}"
            )
        self._model_path = model_path.resolve()
        self._engine_options = dict(engine_options or {})
        forbidden_engine_options = {
            "model",
            "revision",
            "tokenizer",
            "tokenizer_revision",
            "max_model_len",
            "max_num_seqs",
            "seed",
            "trust_remote_code",
        }
        overlap = sorted(forbidden_engine_options & set(self._engine_options))
        if overlap:
            raise ValueError(
                f"Engine options cannot override locked model settings: {', '.join(overlap)}"
            )
        self._engine: Any = None
        self._sampling_params_type: Any = None
        self._structured_outputs_type: Any = None
        self._peak_vram_bytes: int | None = None
        self._engine_initialization_ms = 0.0

    @property
    def model_id(self) -> str:
        return str(self.model_config["model_id"])

    @property
    def model_revision(self) -> str:
        return str(self.model_config["revision"])

    @property
    def backend(self) -> str:
        return "vllm_offline"

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def peak_vram_bytes(self) -> int | None:
        return self._peak_vram_bytes

    @property
    def constraint_backend(self) -> str:
        return "xgrammar"

    @property
    def structured_output_applied(self) -> bool:
        return True

    @property
    def engine_initialization_ms(self) -> float:
        return self._engine_initialization_ms

    def _start_vram_sampler(
        self,
    ) -> tuple[threading.Event, threading.Thread, dict[str, int | None]]:
        stop = threading.Event()
        state: dict[str, int | None] = {"peak": None}

        def sample() -> None:
            try:
                import pynvml

                pynvml.nvmlInit()
                visible = [
                    part.strip()
                    for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                    if part.strip()
                ]
                if visible and all(part.isdigit() for part in visible):
                    handles = [
                        pynvml.nvmlDeviceGetHandleByIndex(int(part)) for part in visible
                    ]
                elif visible and all(part.startswith("GPU-") for part in visible):
                    handles = [
                        pynvml.nvmlDeviceGetHandleByUUID(part) for part in visible
                    ]
                else:
                    handles = [
                        pynvml.nvmlDeviceGetHandleByIndex(index)
                        for index in range(pynvml.nvmlDeviceGetCount())
                    ]
                while True:
                    used = max(
                        (
                            int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
                            for handle in handles
                        ),
                        default=0,
                    )
                    state["peak"] = max(state["peak"] or 0, used) or None
                    if stop.wait(0.05):
                        break
                used = max(
                    (
                        int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
                        for handle in handles
                    ),
                    default=0,
                )
                state["peak"] = max(state["peak"] or 0, used) or None
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001 - optional telemetry cannot fail inference
                return

        thread = threading.Thread(
            target=sample, name="auditops-vram-sampler", daemon=True
        )
        thread.start()
        return stop, thread, state

    def _load_engine(self) -> None:
        if self._engine is not None:
            return
        initialization_started = time.perf_counter()
        try:
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
        except ImportError as exc:
            raise ModelDependencyError(
                "Offline vLLM inference requires the pinned vllm package in the GPU environment"
            ) from exc
        try:
            installed_version = importlib_metadata.version("vllm")
        except importlib_metadata.PackageNotFoundError as exc:
            raise ModelDependencyError(
                "Cannot verify the installed vLLM version"
            ) from exc
        if installed_version.split("+", 1)[0] != REQUIRED_VLLM_VERSION:
            raise ModelDependencyError(
                f"AuditOps requires vLLM {REQUIRED_VLLM_VERSION}; found {installed_version}"
            )
        try:
            importlib_metadata.version("xgrammar")
        except importlib_metadata.PackageNotFoundError as exc:
            raise ModelDependencyError(
                "AuditOps v2.2 requires xgrammar for fail-closed structured output"
            ) from exc

        from .model_config import _chat_template_bytes

        expected_template_hash = self.model_config.get("chat_template_sha256")
        actual_template_hash = hashlib.sha256(
            _chat_template_bytes(self.model_path)
        ).hexdigest()
        if actual_template_hash != expected_template_hash:
            raise ModelDependencyError(
                "Pinned chat template hash does not match the model configuration"
            )

        engine_kwargs: dict[str, Any] = {
            # A resolved directory, never a Hub identifier. The revision stays
            # in ModelConfig as provenance for the pre-downloaded snapshot.
            "model": str(self.model_path),
            "max_model_len": int(self.model_config["context_window_tokens"]),
            "max_num_seqs": OFFLINE_MAX_NUM_SEQS,
            "seed": int(self.model_config["seed"]),
            "trust_remote_code": False,
            "structured_outputs_config": {"backend": "xgrammar"},
        }
        quantization = self.model_config.get("quantization")
        if quantization:
            engine_kwargs["quantization"] = quantization
        engine_kwargs.update(self._engine_options)
        try:
            self._engine = LLM(**engine_kwargs)
        except Exception as exc:
            raise _classified_backend_error(exc, action="initialization") from exc
        self._sampling_params_type = SamplingParams
        self._structured_outputs_type = StructuredOutputsParams
        self._engine_initialization_ms = (
            time.perf_counter() - initialization_started
        ) * 1000.0

    def generate_json(self, request: ModelRequest) -> ModelResponse:
        if not xgrammar_json_schema_supported(request.json_schema):
            raise ModelBackendError(
                "Response schema is unsupported by the pinned xgrammar backend"
            )
        if request.temperature != self.model_config["temperature"]:
            raise ValueError(
                "Request temperature does not match the locked model configuration"
            )
        if request.top_p != self.model_config["top_p"]:
            raise ValueError(
                "Request top_p does not match the locked model configuration"
            )
        if request.seed != self.model_config["seed"]:
            raise ValueError(
                "Request seed does not match the locked model configuration"
            )
        output_cap = max(
            int(self.model_config["max_plan_tokens"]),
            int(self.model_config["max_answer_tokens"]),
            int(self.model_config["max_repair_tokens"]),
        )
        if request.max_tokens > output_cap:
            raise ValueError(
                "Request output budget exceeds the locked model configuration"
            )

        stop_sampler, sampler_thread, sampler_state = self._start_vram_sampler()
        try:
            self._load_engine()
            structured = self._structured_outputs_type(json=dict(request.json_schema))
            sampling_params = self._sampling_params_type(
                temperature=float(request.temperature),
                top_p=float(request.top_p),
                seed=int(request.seed),
                max_tokens=request.max_tokens,
                structured_outputs=structured,
            )
            started = time.perf_counter()
            outputs = self._engine.chat(
                list(request.messages),
                sampling_params=sampling_params,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False},
            )
            duration_ms = (time.perf_counter() - started) * 1000.0
        except ModelAdapterError:
            raise
        except Exception as exc:
            raise _classified_backend_error(exc, action="generation") from exc
        finally:
            stop_sampler.set()
            sampler_thread.join(timeout=1.0)
            measured_peak = sampler_state["peak"]
            if measured_peak is not None:
                self._peak_vram_bytes = max(self._peak_vram_bytes or 0, measured_peak)
        if not outputs or not outputs[0].outputs:
            raise ModelGenerationError("Offline vLLM returned no completion")

        request_output = outputs[0]
        completion = request_output.outputs[0]
        prompt_token_ids = getattr(request_output, "prompt_token_ids", None) or ()
        output_token_ids = getattr(completion, "token_ids", None) or ()
        finish_reason = getattr(completion, "finish_reason", None)
        text = completion.text
        completion_metadata_valid = (
            isinstance(text, str)
            and isinstance(prompt_token_ids, (list, tuple))
            and all(
                isinstance(token_id, int) and not isinstance(token_id, bool)
                for token_id in prompt_token_ids
            )
            and isinstance(output_token_ids, (list, tuple))
            and all(
                isinstance(token_id, int) and not isinstance(token_id, bool)
                for token_id in output_token_ids
            )
            and (
                finish_reason is None
                or (
                    isinstance(finish_reason, str)
                    and finish_reason in {"stop", "length", "abort"}
                )
            )
        )
        if not completion_metadata_valid:
            # Backend response objects are untrusted adapter input.  Clear the
            # object graph before raising a fixed typed failure so traceback
            # locals cannot retain model text or arbitrary telemetry objects.
            text = ""
            finish_reason = None
            prompt_token_ids = ()
            output_token_ids = ()
            completion = None
            request_output = None
            outputs = None
            raise ModelAdapterError(
                "Offline model completion metadata is invalid"
            ) from None

        strict_response_sha256: str | None = None
        try:
            payload = parse_strict_json_object(text)
        except StrictJSONError as exc:
            strict_response_sha256 = exc.response_sha256
            strict_output_bytes = exc.output_bytes
            strict_parse_category = exc.parse_category
        else:
            strict_output_bytes = None
            strict_parse_category = None
        if strict_response_sha256 is not None:
            # Scrub every local reference to the backend completion before the
            # sanitized exception acquires this frame as its traceback.
            text = ""
            completion = None
            request_output = None
            outputs = None
            raise StrictJSONError(
                response_sha256=strict_response_sha256,
                input_tokens=len(prompt_token_ids),
                output_tokens=len(output_token_ids),
                duration_ms=duration_ms,
                finish_reason=finish_reason,
                output_bytes=strict_output_bytes,
                parse_category=strict_parse_category,
                constraint_backend=self.constraint_backend,
                structured_output_applied=self.structured_output_applied,
            ) from None
        response_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            _validate_schema_payload(
                payload,
                request.json_schema,
                response_sha256=response_sha256,
                output_bytes=len(text.encode("utf-8")),
            )
        except ModelSchemaError as exc:
            text = ""
            completion = None
            request_output = None
            outputs = None
            raise ModelSchemaError(
                response_sha256=exc.response_sha256,
                input_tokens=len(prompt_token_ids),
                output_tokens=len(output_token_ids),
                output_bytes=exc.output_bytes,
                duration_ms=duration_ms,
                finish_reason=finish_reason,
                constraint_backend=self.constraint_backend,
                structured_output_applied=self.structured_output_applied,
            ) from None
        return ModelResponse(
            request_id=request.request_id,
            payload=payload,
            response_sha256=response_sha256,
            input_tokens=len(prompt_token_ids),
            output_tokens=len(output_token_ids),
            duration_ms=duration_ms,
            finish_reason=finish_reason,
            metadata={
                "backend": self.backend,
                "model_id": self.model_id,
                "revision": self.model_revision,
                "payload_sha256": canonical_json_sha256(payload),
                "peak_vram_bytes": self.peak_vram_bytes,
                "constraint_backend": self.constraint_backend,
                "structured_output_applied": self.structured_output_applied,
                "output_bytes": len(text.encode("utf-8")),
                "cap_hit": finish_reason == "length",
                "engine_initialization_ms": self.engine_initialization_ms,
            },
        )

    def count_tokens(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        self._load_engine()
        try:
            tokenizer = self._engine.get_tokenizer()
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception as exc:
            raise ModelGenerationError(
                "Failed to count input tokens with the pinned tokenizer"
            ) from exc
        return len(token_ids)

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int:
        normalized = _validate_messages(messages)
        self._load_engine()
        try:
            tokenizer = self._engine.get_tokenizer()
            token_ids = tokenizer.apply_chat_template(
                list(normalized),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception as exc:
            raise ModelGenerationError(
                "Failed to count rendered chat tokens with the pinned tokenizer"
            ) from exc
        token_ids = _single_input_token_ids(token_ids)
        return len(token_ids)
