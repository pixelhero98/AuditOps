from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .canonical_json import canonical_json_sha256

PROVENANCE_VERSION = "auditops-provenance.v1"
_EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".auditops",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
}
_SENSITIVE_PATTERNS = {
    "hugging_face_token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential_assignment": re.compile(
        rb"(?:COMPANIES_HOUSE_API_KEY|HF_TOKEN|AWS_SECRET_ACCESS_KEY)\s*=\s*['\"]?(?!\$\{|\$|<|REPLACE|CHANGE_ME)[^\s'\"]{8,}",
        re.IGNORECASE,
    ),
}
MODEL_FILE_MANIFEST_VERSION = "auditops-model-files.v1"
MODEL_FILES_MANIFEST_NAME = "AUDITOPS_FILES_SHA256"
MODEL_SNAPSHOT_DIGEST_NAME = "AUDITOPS_SNAPSHOT_SHA256"
MODEL_SNAPSHOT_COMPLETE_NAME = "AUDITOPS_SNAPSHOT_COMPLETE"
_MODEL_CONTROL_FILES = {
    MODEL_FILES_MANIFEST_NAME,
    MODEL_SNAPSHOT_DIGEST_NAME,
    MODEL_SNAPSHOT_COMPLETE_NAME,
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_HASH_CHUNK_BYTES = 1024 * 1024
_SECRET_SCAN_CHUNK_BYTES = 1024 * 1024
_SECRET_SCAN_PATTERN_OVERLAP = 512
_SECRET_ENVIRONMENT_NAMES = (
    "AWS_SECRET_ACCESS_KEY",
    "COMPANIES_HOUSE_API_KEY",
    "HF_TOKEN",
)
_SECRET_ENVIRONMENT_SUFFIXES = (
    "_API_KEY",
    "_PASSWORD",
    "_PRIVATE_KEY",
    "_SECRET",
    "_TOKEN",
)
_SECRET_ENVIRONMENT_MIN_BYTES = 8
_SECRET_ENVIRONMENT_MAX_BYTES = 64 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains a duplicate key: {key}")
        result[key] = value
    return result


def _canonical_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if "\\" in value or any(ord(character) < 32 for character in value):
        raise ValueError(f"{label} is not canonical: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError(f"{label} is not canonical: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} is unsafe: {value!r}")
    return value


def _stable_file_metadata(
    status: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _opened_path_metadata(status: os.stat_result) -> tuple[int, int, int, int, int]:
    # Windows reports st_ctime_ns differently for path and descriptor queries.
    # Keep ctime in each same-query stability check, but omit it only from the
    # cross-query identity comparison.
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
    )


def _stable_regular_file_record(path: Path, relative: str) -> dict[str, Any]:
    descriptor: int | None = None
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Snapshot entry is not a regular file: {relative}")
        flags = os.O_RDONLY
        flags |= getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        stable_before = _stable_file_metadata(before)
        if _opened_path_metadata(before) != _opened_path_metadata(
            opened
        ) or not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"Snapshot entry changed while being verified: {relative}")
        digest = hashlib.sha256()
        size = 0
        while block := os.read(descriptor, _MODEL_HASH_CHUNK_BYTES):
            digest.update(block)
            size += len(block)
        completed = os.fstat(descriptor)
        after = path.lstat()
    except OSError as exc:
        raise ValueError(f"Snapshot entry could not be read: {relative}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                raise ValueError(
                    f"Snapshot entry could not be read: {relative}"
                ) from exc
    if (
        _stable_file_metadata(opened) != _stable_file_metadata(completed)
        or stable_before != _stable_file_metadata(after)
        or not stat.S_ISREG(completed.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or size != before.st_size
    ):
        raise ValueError(f"Snapshot entry changed while being verified: {relative}")
    return {
        "path": relative,
        "size": size,
        "sha256": digest.hexdigest(),
    }


def _walk_snapshot(
    root: Path,
    *,
    require_read_only: bool,
) -> tuple[list[tuple[str, Path]], list[Path]]:
    try:
        root_status = root.lstat()
    except OSError as exc:
        raise ValueError(f"Model snapshot cannot be inspected: {root}") from exc
    if root.is_symlink() or not stat.S_ISDIR(root_status.st_mode):
        raise ValueError(f"Model snapshot root is not a real directory: {root}")
    if require_read_only and root_status.st_mode & 0o222:
        raise ValueError("Model snapshot root is writable")

    files: list[tuple[str, Path]] = []
    directories = [root]
    try:
        walker = os.walk(root, topdown=True, followlinks=False)
        for current, directory_names, file_names in walker:
            current_path = Path(current)
            retained_directories: list[str] = []
            for name in sorted(directory_names):
                candidate = current_path / name
                relative = candidate.relative_to(root).as_posix()
                status = candidate.lstat()
                if candidate.is_symlink() or not stat.S_ISDIR(status.st_mode):
                    raise ValueError(
                        f"Model snapshot contains a non-directory entry: {relative}"
                    )
                if require_read_only and status.st_mode & 0o222:
                    raise ValueError(
                        f"Model snapshot directory is writable: {relative}"
                    )
                retained_directories.append(name)
                directories.append(candidate)
            directory_names[:] = retained_directories
            for name in sorted(file_names):
                candidate = current_path / name
                relative = _canonical_relative_path(
                    candidate.relative_to(root).as_posix(),
                    label="Model snapshot path",
                )
                status = candidate.lstat()
                if candidate.is_symlink() or not stat.S_ISREG(status.st_mode):
                    raise ValueError(
                        f"Model snapshot contains a non-regular file: {relative}"
                    )
                if require_read_only and status.st_mode & 0o222:
                    raise ValueError(f"Model snapshot file is writable: {relative}")
                files.append((relative, candidate))
    except OSError as exc:
        raise ValueError(f"Model snapshot could not be traversed: {root}") from exc
    return sorted(files, key=lambda item: item[0]), directories


def _resolve_model_snapshot_root(snapshot_root: str | Path) -> Path:
    candidate = Path(snapshot_root)
    try:
        status = candidate.lstat()
    except OSError as exc:
        raise ValueError(f"Model snapshot cannot be inspected: {candidate}") from exc
    if candidate.is_symlink() or not stat.S_ISDIR(status.st_mode):
        raise ValueError(f"Model snapshot root is not a real directory: {candidate}")
    return candidate.resolve(strict=True)


def build_model_file_manifest(snapshot_root: str | Path) -> dict[str, Any]:
    """Build the exact regular-file manifest used to finalize a model snapshot."""

    root = _resolve_model_snapshot_root(snapshot_root)
    files, _ = _walk_snapshot(root, require_read_only=False)
    if any(relative in _MODEL_CONTROL_FILES for relative, _ in files):
        raise FileExistsError("Model snapshot control files already exist")
    if not files:
        raise ValueError("Model snapshot contains no data files")
    rows = [_stable_regular_file_record(path, relative) for relative, path in files]
    return {
        "manifest_version": MODEL_FILE_MANIFEST_VERSION,
        "file_count": len(rows),
        "files": rows,
    }


def write_model_file_manifest(snapshot_root: str | Path) -> tuple[dict[str, Any], str]:
    """Write a new canonical model manifest and its two completion records."""

    root = _resolve_model_snapshot_root(snapshot_root)
    manifest_path = root / MODEL_FILES_MANIFEST_NAME
    digest_path = root / MODEL_SNAPSHOT_DIGEST_NAME
    complete_path = root / MODEL_SNAPSHOT_COMPLETE_NAME
    for path in (manifest_path, digest_path, complete_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite model snapshot record: {path}"
            )
    manifest = build_model_file_manifest(root)
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    manifest_path.write_bytes(payload)
    record = f"{digest}\n"
    digest_path.write_text(record, encoding="ascii", newline="\n")
    complete_path.write_text(record, encoding="ascii", newline="\n")
    return manifest, digest


def verify_model_snapshot(
    snapshot_root: str | Path,
    expected_manifest_sha256: str,
    *,
    require_read_only: bool = True,
) -> dict[str, Any]:
    """Verify an exact, finalized model snapshot against an external digest pin."""

    if not isinstance(expected_manifest_sha256, str) or not _SHA256_RE.fullmatch(
        expected_manifest_sha256
    ):
        raise ValueError("Expected model-manifest digest is not canonical SHA-256")
    root = _resolve_model_snapshot_root(snapshot_root)
    files, _ = _walk_snapshot(root, require_read_only=require_read_only)
    actual_paths = [relative for relative, _ in files]
    for control_path in sorted(_MODEL_CONTROL_FILES):
        if control_path not in actual_paths:
            raise ValueError(f"Model snapshot is missing control file: {control_path}")

    manifest_path = root / MODEL_FILES_MANIFEST_NAME
    try:
        manifest_payload = manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError("Model snapshot manifest could not be read") from exc
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Model snapshot manifest differs from the external digest pin")
    expected_record = f"{expected_manifest_sha256}\n".encode("ascii")
    for record_name in (MODEL_SNAPSHOT_DIGEST_NAME, MODEL_SNAPSHOT_COMPLETE_NAME):
        try:
            record_payload = (root / record_name).read_bytes()
        except OSError as exc:
            raise ValueError(
                f"Model snapshot record could not be read: {record_name}"
            ) from exc
        if record_payload != expected_record:
            raise ValueError(
                f"Model snapshot record differs from the external digest pin: {record_name}"
            )

    try:
        manifest = json.loads(
            manifest_payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Model snapshot manifest is not UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != {
        "manifest_version",
        "file_count",
        "files",
    }:
        raise ValueError("Model snapshot manifest has an unexpected schema")
    if manifest["manifest_version"] != MODEL_FILE_MANIFEST_VERSION:
        raise ValueError("Model snapshot manifest version is unsupported")
    rows = manifest["files"]
    file_count = manifest["file_count"]
    if (
        not isinstance(rows, list)
        or not rows
        or not isinstance(file_count, int)
        or isinstance(file_count, bool)
        or file_count != len(rows)
    ):
        raise ValueError("Model snapshot manifest file_count does not match files")

    validated_rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
            raise ValueError("Model snapshot manifest contains a malformed file row")
        relative = _canonical_relative_path(row["path"], label="Model manifest path")
        if relative in _MODEL_CONTROL_FILES:
            raise ValueError(f"Model manifest includes a control file: {relative}")
        if relative in seen_paths:
            raise ValueError(f"Model manifest contains a duplicate path: {relative}")
        seen_paths.add(relative)
        size = row["size"]
        digest = row["sha256"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"Model manifest size is invalid: {relative}")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"Model manifest digest is invalid: {relative}")
        validated_rows.append({"path": relative, "size": size, "sha256": digest})
    if [row["path"] for row in validated_rows] != sorted(seen_paths):
        raise ValueError("Model manifest rows are not in canonical path order")

    expected_paths = sorted([*seen_paths, *_MODEL_CONTROL_FILES])
    if actual_paths != expected_paths:
        missing = sorted(set(expected_paths) - set(actual_paths))
        unexpected = sorted(set(actual_paths) - set(expected_paths))
        raise ValueError(
            f"Model snapshot file-set mismatch; missing={missing}; unexpected={unexpected}"
        )
    actual_by_path = dict(files)
    for expected in validated_rows:
        actual = _stable_regular_file_record(
            actual_by_path[expected["path"]], expected["path"]
        )
        if actual != expected:
            raise ValueError(
                f"Model snapshot metadata or checksum mismatch: {expected['path']}"
            )
    return {
        "manifest_sha256": manifest_sha256,
        "file_count": file_count,
        "files": validated_rows,
    }


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        if any(
            part in _EXCLUDED_DIRECTORY_NAMES for part in path.relative_to(root).parts
        ):
            continue
        yield path


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def build_source_manifest(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if not (root / ".git").exists():
        raise ValueError(f"Not a Git checkout: {root}")
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if any(part in _EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(
                f"Deployable source tree must not contain symlinks: {relative.as_posix()}"
            )
        if not path.is_file() and not path.is_dir():
            raise ValueError(
                f"Deployable source tree contains a special file: {relative.as_posix()}"
            )
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in _iter_files(root)
    ]
    diff = _git_output(root, "diff", "--binary", "--no-ext-diff")
    status = _git_output(root, "status", "--short", "--untracked-files=all")
    return {
        "provenance_version": PROVENANCE_VERSION,
        "base_commit": _git_output(root, "rev-parse", "HEAD"),
        "branch": _git_output(root, "branch", "--show-current"),
        "source_tree_sha256": canonical_json_sha256(files),
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "dirty": bool(status),
        "file_count": len(files),
        "files": files,
    }


def write_source_manifest(
    repo_root: str | Path, output_path: str | Path
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite source manifest: {output}")
    manifest = build_source_manifest(repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _iter_secret_scan_files(root: Path) -> list[Path]:
    files: list[Path] = []
    try:
        walker = os.walk(root, topdown=True, followlinks=False)
        for current, directory_names, file_names in walker:
            current_path = Path(current)
            retained_directories: list[str] = []
            for name in sorted(directory_names):
                candidate = current_path / name
                status = candidate.lstat()
                if candidate.is_symlink() or not stat.S_ISDIR(status.st_mode):
                    raise ValueError(
                        f"Secret-scan tree contains a non-directory entry: {candidate}"
                    )
                if name in _EXCLUDED_DIRECTORY_NAMES:
                    continue
                retained_directories.append(name)
            directory_names[:] = retained_directories
            for name in sorted(file_names):
                candidate = current_path / name
                status = candidate.lstat()
                if candidate.is_symlink() or not stat.S_ISREG(status.st_mode):
                    raise ValueError(
                        f"Secret-scan tree contains a non-regular file: {candidate}"
                    )
                files.append(candidate)
    except OSError as exc:
        raise ValueError(f"Secret-scan tree could not be traversed: {root}") from exc
    return sorted(files, key=lambda path: path.as_posix())


def _is_credential_environment_name(name: str) -> bool:
    normalized = name.upper()
    return normalized in _SECRET_ENVIRONMENT_NAMES or normalized.endswith(
        _SECRET_ENVIRONMENT_SUFFIXES
    )


def _secret_environment_values(
    environment_names: Sequence[str],
) -> dict[str, bytes]:
    requested: set[str] = set()
    for name in environment_names:
        if not isinstance(name, str) or not name:
            raise TypeError("Secret-scan environment names must be non-empty strings")
        requested.add(name)
    requested.update(
        name for name in os.environ if _is_credential_environment_name(name)
    )

    secret_values: dict[str, bytes] = {}
    for name in sorted(requested):
        value = os.environ.get(name)
        if not value:
            continue
        encoded = value.encode("utf-8")
        if len(encoded) < _SECRET_ENVIRONMENT_MIN_BYTES:
            continue
        if len(encoded) > _SECRET_ENVIRONMENT_MAX_BYTES:
            raise ValueError(
                f"Credential-like environment value is too large to scan: {name}"
            )
        secret_values[name] = encoded
    return secret_values


def scan_artifacts_for_secrets(
    paths: Sequence[str | Path],
    *,
    environment_names: Sequence[str] = _SECRET_ENVIRONMENT_NAMES,
) -> list[dict[str, str]]:
    secret_values = _secret_environment_values(environment_names)
    findings: list[dict[str, str]] = []
    visited: set[Path] = set()
    for raw_path in paths:
        unresolved = Path(raw_path)
        try:
            unresolved_status = unresolved.lstat()
        except OSError as exc:
            raise ValueError(
                f"Secret-scan input cannot be inspected: {unresolved}"
            ) from exc
        if unresolved.is_symlink():
            raise ValueError(f"Secret-scan input must not be a symlink: {unresolved}")
        candidate = unresolved.resolve(strict=True)
        if stat.S_ISDIR(unresolved_status.st_mode):
            files = _iter_secret_scan_files(candidate)
        elif stat.S_ISREG(unresolved_status.st_mode):
            files = [candidate]
        else:
            raise ValueError(
                f"Secret-scan input is not a regular file or directory: {candidate}"
            )
        for path in files:
            if path in visited:
                continue
            visited.add(path)
            matched_environment: set[str] = set()
            matched_patterns: set[str] = set()
            overlap_size = max(
                _SECRET_SCAN_PATTERN_OVERLAP,
                max((len(secret) - 1 for secret in secret_values.values()), default=0),
            )
            overlap = b""
            try:
                before = path.lstat()
                if path.is_symlink() or not stat.S_ISREG(before.st_mode):
                    raise ValueError(f"Secret-scan input is not a regular file: {path}")
                with path.open("rb") as handle:
                    while block := handle.read(_SECRET_SCAN_CHUNK_BYTES):
                        window = overlap + block
                        for name, secret in secret_values.items():
                            if name not in matched_environment and secret in window:
                                matched_environment.add(name)
                        for name, pattern in _SENSITIVE_PATTERNS.items():
                            if name not in matched_patterns and pattern.search(window):
                                matched_patterns.add(name)
                        overlap = window[-overlap_size:]
                after = path.lstat()
            except OSError as exc:
                raise ValueError(
                    f"Secret-scan input could not be read: {path}"
                ) from exc
            stable_before = (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            stable_after = (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if stable_before != stable_after or not stat.S_ISREG(after.st_mode):
                raise ValueError(f"Secret-scan input changed while being read: {path}")
            findings.extend(
                {
                    "path": str(path),
                    "code": "ENV_SECRET_VALUE",
                    "source": name,
                }
                for name in sorted(matched_environment)
            )
            findings.extend(
                {
                    "path": str(path),
                    "code": "SENSITIVE_PATTERN",
                    "source": name,
                }
                for name in sorted(matched_patterns)
            )
    return sorted(findings, key=lambda row: (row["path"], row["code"], row["source"]))


def assert_no_secrets(paths: Sequence[str | Path]) -> None:
    findings = scan_artifacts_for_secrets(paths)
    if findings:
        summary = ", ".join(
            f"{row['code']}:{row['source']} in {row['path']}" for row in findings
        )
        raise ValueError(f"Secret scan failed: {summary}")


def assert_no_secrets_in_value(value: Any, *, label: str) -> None:
    """Fail before publication when a JSON value contains credential material."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    matches: list[str] = []
    for name, secret in _secret_environment_values(_SECRET_ENVIRONMENT_NAMES).items():
        if secret in payload:
            matches.append(f"ENV_SECRET_VALUE:{name}")
    for name, pattern in _SENSITIVE_PATTERNS.items():
        if pattern.search(payload):
            matches.append(f"SENSITIVE_PATTERN:{name}")
    if matches:
        raise ValueError(
            f"Secret scan failed: {', '.join(sorted(set(matches)))} in {label}"
        )
