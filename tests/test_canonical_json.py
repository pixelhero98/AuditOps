from __future__ import annotations

import hashlib
import json
import math

import pytest

from auditops.canonical_json import (
    CANONICAL_JSON_VERSION,
    CanonicalJSONError,
    canonical_json_bytes,
    canonical_json_sha256,
    canonical_json_text,
)


@pytest.mark.parametrize(
    ("value", "expected_text", "expected_sha256"),
    [
        ({}, "{}", "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"),
        (
            {"b": 1, "a": ["café", False, None]},
            '{"a":["café",false,null],"b":1}',
            "3b89101eb97cd1d69930a9b6cff47d4b0210ddcffb9be01ab8e63c03e4d423c4",
        ),
        (
            {"nested": {"z": 1.0, "a": -2}, "text": "£"},
            '{"nested":{"a":-2,"z":1.0},"text":"£"}',
            "1c7f39adf53e4313adf740c1cf0ce96266ccef6cabad62b52ac21f095f046f10",
        ),
    ],
)
def test_golden_vectors(value, expected_text, expected_sha256):
    assert CANONICAL_JSON_VERSION == "auditops.canonical_json.v1"
    assert canonical_json_text(value) == expected_text
    assert canonical_json_bytes(value) == expected_text.encode("utf-8")
    assert canonical_json_sha256(value) == expected_sha256


def test_newline_is_explicit_and_is_not_part_of_canonical_hash():
    value = {"a": 1}
    assert canonical_json_text(value, newline=True) == '{"a":1}\n'
    assert canonical_json_bytes(value, newline=True) == b'{"a":1}\n'
    assert canonical_json_sha256(value) == hashlib.sha256(b'{"a":1}').hexdigest()


def test_matches_the_previous_compact_python_representation():
    value = {"unicode": "δ", "items": [3, 1.25, True, None], "object": {"b": 2, "a": 1}}
    previous = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert canonical_json_bytes(value) == previous
    assert canonical_json_sha256(value) == hashlib.sha256(previous).hexdigest()


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rejects_non_finite_numbers(value):
    with pytest.raises(CanonicalJSONError, match="not representable"):
        canonical_json_text({"value": value})


@pytest.mark.parametrize(
    "value",
    [
        {1: "integer key"},
        {"nested": [{"ok": True, 2: "integer key"}]},
        {"nested": {None: "null key"}},
    ],
)
def test_recursively_rejects_non_string_object_keys(value):
    with pytest.raises(CanonicalJSONError, match="object key.*must be a string"):
        canonical_json_bytes(value)


def test_rejects_circular_containers_with_a_typed_error():
    value: list[object] = []
    value.append(value)
    with pytest.raises(CanonicalJSONError, match="Circular reference"):
        canonical_json_bytes(value)


def test_rejects_non_json_values_with_a_typed_error():
    with pytest.raises(CanonicalJSONError, match="not representable"):
        canonical_json_bytes({"items": {1, 2}})


def test_rejects_excessive_nesting_with_a_typed_error():
    value: list[object] = []
    for _ in range(1_100):
        value = [value]
    with pytest.raises(CanonicalJSONError, match="not representable"):
        canonical_json_bytes(value)
