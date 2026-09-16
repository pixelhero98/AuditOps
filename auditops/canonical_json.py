"""AuditOps' versioned deterministic JSON representation for hashing.

This is a small, Python-specific canonicalizer.  It intentionally does not
claim conformance with RFC 8785 (JCS): finite JSON numbers retain Python's
standard JSON rendering rather than being normalized across implementations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

CANONICAL_JSON_VERSION = "auditops.canonical_json.v1"


class CanonicalJSONError(ValueError):
    """The value cannot be represented by AuditOps canonical JSON."""


def _validate_object_keys(
    value: Any,
    *,
    path: str = "$",
    active_containers: set[int] | None = None,
) -> None:
    """Reject non-string object keys and circular JSON containers."""

    active = active_containers if active_containers is not None else set()
    if isinstance(value, Mapping):
        container_id = id(value)
        if container_id in active:
            raise CanonicalJSONError(f"Circular reference at {path}")
        active.add(container_id)
        try:
            for index, (key, child) in enumerate(value.items()):
                if not isinstance(key, str):
                    raise CanonicalJSONError(
                        f"Canonical JSON object key at {path}[{index}] must be a string"
                    )
                _validate_object_keys(
                    child,
                    path=f"{path}[{index}]",
                    active_containers=active,
                )
        finally:
            active.remove(container_id)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        container_id = id(value)
        if container_id in active:
            raise CanonicalJSONError(f"Circular reference at {path}")
        active.add(container_id)
        try:
            for index, child in enumerate(value):
                _validate_object_keys(
                    child,
                    path=f"{path}[{index}]",
                    active_containers=active,
                )
        finally:
            active.remove(container_id)


def canonical_json_text(value: Any, *, newline: bool = False) -> str:
    """Return the deterministic AuditOps JSON text for a JSON-compatible value."""

    try:
        _validate_object_keys(value)
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        # Escaped lone surrogates can survive json.loads but have no UTF-8
        # representation. Reject them at the same typed boundary as NaN.
        serialized.encode("utf-8")
    except CanonicalJSONError:
        raise
    except (TypeError, ValueError, RecursionError) as exc:
        raise CanonicalJSONError(
            f"Value is not representable as canonical JSON: {exc}"
        ) from exc
    return serialized + ("\n" if newline else "")


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    """Return UTF-8 bytes for :func:`canonical_json_text`."""

    return canonical_json_text(value, newline=newline).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Hash canonical JSON bytes without a trailing newline."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "CANONICAL_JSON_VERSION",
    "CanonicalJSONError",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "canonical_json_text",
]
