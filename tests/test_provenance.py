from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from auditops import provenance
from auditops.provenance import (
    assert_no_secrets,
    build_model_file_manifest,
    build_source_manifest,
    scan_artifacts_for_secrets,
    verify_model_snapshot,
    write_model_file_manifest,
    write_source_manifest,
)


def test_source_manifest_is_stable_and_refuses_overwrite(tmp_path):
    repo_root = Path(__file__).parents[1]
    if not (repo_root / ".git").exists():
        pytest.skip(
            "source-manifest construction requires the development Git checkout"
        )
    manifest = build_source_manifest(repo_root)
    assert len(manifest["base_commit"]) == 40
    assert len(manifest["source_tree_sha256"]) == 64
    output = tmp_path / "source_manifest.json"
    written = write_source_manifest(repo_root, output)
    assert written["base_commit"] == manifest["base_commit"]
    try:
        write_source_manifest(repo_root, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("write_source_manifest must refuse to overwrite")


def test_secret_scan_uses_environment_without_echoing_value(tmp_path, monkeypatch):
    secret = "hf_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
    monkeypatch.setenv("HF_TOKEN", secret)
    clean = tmp_path / "clean.txt"
    clean.write_text("HF_TOKEN=${HF_TOKEN}\n", encoding="utf-8")
    leaked = tmp_path / "leaked.txt"
    leaked.write_text(f"token={secret}\n", encoding="utf-8")
    findings = scan_artifacts_for_secrets([tmp_path])
    assert any(row["path"] == str(leaked.resolve()) for row in findings)
    assert all(secret not in str(row) for row in findings)


def test_secret_scan_excludes_generated_cache_directories(tmp_path, monkeypatch):
    secret = "hf_" + "CACHEONLYSYNTHETICVALUE1234567890"
    monkeypatch.setenv("HF_TOKEN", secret)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "compiled.pyc").write_bytes(secret.encode("ascii"))
    (tmp_path / "source.py").write_text("clean\n", encoding="utf-8")

    assert scan_artifacts_for_secrets([tmp_path]) == []


def test_secret_scan_discovers_aws_and_generic_credential_environment_values(
    tmp_path, monkeypatch
):
    credentials = {
        "AWS_SECRET_ACCESS_KEY": "synthetic-aws-secret-value-123456",
        "AUDITOPS_FIXTURE_SECRET": "synthetic-generic-secret-value-123456",
        "AUDITOPS_FIXTURE_TOKEN": "synthetic-generic-token-value-123456",
        "AUDITOPS_FIXTURE_API_KEY": "synthetic-generic-api-key-value-123456",
    }
    irrelevant_name = "AUDITOPS_FIXTURE_VALUE"
    irrelevant_value = "synthetic-irrelevant-value-123456"
    for name, value in credentials.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(irrelevant_name, irrelevant_value)
    artifact = tmp_path / "raw-values.txt"
    artifact.write_text(
        "\n".join([*credentials.values(), irrelevant_value]) + "\n",
        encoding="utf-8",
    )

    findings = scan_artifacts_for_secrets([artifact])

    environment_findings = {
        row["source"] for row in findings if row["code"] == "ENV_SECRET_VALUE"
    }
    assert environment_findings == set(credentials)
    assert irrelevant_name not in environment_findings
    rendered_findings = str(findings)
    assert all(value not in rendered_findings for value in credentials.values())
    assert irrelevant_value not in rendered_findings

    with pytest.raises(ValueError, match="Secret scan failed") as exc_info:
        assert_no_secrets([artifact])
    rendered_error = str(exc_info.value)
    assert all(value not in rendered_error for value in credentials.values())
    assert irrelevant_value not in rendered_error


def test_secret_scan_fails_closed_on_oversized_credential_environment_value(
    tmp_path, monkeypatch
):
    oversized_value = "x" * (64 * 1024 + 1)
    monkeypatch.setattr(
        "auditops.provenance.os.environ",
        {"AUDITOPS_OVERSIZED_TOKEN": oversized_value},
    )
    artifact = tmp_path / "clean.txt"
    artifact.write_text("clean\n", encoding="utf-8")

    with pytest.raises(ValueError, match="too large to scan") as exc_info:
        scan_artifacts_for_secrets([artifact])

    assert "AUDITOPS_OVERSIZED_TOKEN" in str(exc_info.value)
    assert oversized_value not in str(exc_info.value)


def test_secret_scan_stream_overlap_detects_boundary_spanning_value(
    tmp_path, monkeypatch
):
    secret = "hf_" + "B" * 40
    monkeypatch.setenv("HF_TOKEN", secret)
    split = len(secret) // 2
    artifact = tmp_path / "boundary.bin"
    artifact.write_bytes(
        b"x" * (1024 * 1024 - split - 1) + b" " + secret.encode("ascii") + b" "
    )

    findings = scan_artifacts_for_secrets([artifact])

    assert {(row["code"], row["source"]) for row in findings} >= {
        ("ENV_SECRET_VALUE", "HF_TOKEN"),
        ("SENSITIVE_PATTERN", "hugging_face_token"),
    }
    assert all(secret not in str(row) for row in findings)


def test_secret_scan_fails_closed_when_a_file_cannot_be_read(tmp_path, monkeypatch):
    artifact = (tmp_path / "unreadable.bin").resolve()
    artifact.write_bytes(b"not a secret")
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.resolve() == artifact:
            raise PermissionError("simulated")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="could not be read"):
        scan_artifacts_for_secrets([artifact])


def test_model_snapshot_manifest_rejects_tampering_and_extra_files(tmp_path):
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        '{"model_type":"fixture"}\n', encoding="utf-8"
    )
    weights = snapshot / "weights"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"weights")

    manifest, digest = write_model_file_manifest(snapshot)
    verified = verify_model_snapshot(snapshot, digest, require_read_only=False)
    assert verified["files"] == manifest["files"]

    (snapshot / "unexpected.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file-set mismatch"):
        verify_model_snapshot(snapshot, digest, require_read_only=False)
    (snapshot / "unexpected.txt").unlink()

    (weights / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="metadata or checksum mismatch"):
        verify_model_snapshot(snapshot, digest, require_read_only=False)


def test_model_snapshot_manifest_hashes_large_files_in_bounded_chunks(
    tmp_path, monkeypatch
):
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    payload = b"a" * (2 * provenance._MODEL_HASH_CHUNK_BYTES + 37)
    weights = snapshot / "model.safetensors"
    weights.write_bytes(payload)
    original_read = os.read
    read_sizes: list[int] = []

    def tracked_read(descriptor, size):
        block = original_read(descriptor, size)
        read_sizes.append(len(block))
        return block

    def reject_read_bytes(path):
        raise AssertionError(f"Path.read_bytes materialized {path}")

    monkeypatch.setattr(provenance.os, "read", tracked_read)
    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)

    manifest = build_model_file_manifest(snapshot)

    assert manifest["files"] == [
        {
            "path": "model.safetensors",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    assert read_sizes[-1] == 0
    assert sum(read_sizes) == len(payload)
    assert len(read_sizes[:-1]) >= 3
    assert all(
        0 < size <= provenance._MODEL_HASH_CHUNK_BYTES for size in read_sizes[:-1]
    )


def test_model_snapshot_manifest_fails_closed_on_midstream_mutation(
    tmp_path, monkeypatch
):
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"a" * (provenance._MODEL_HASH_CHUNK_BYTES + 1))
    original_read = os.read
    mutated = False

    def mutating_read(descriptor, size):
        nonlocal mutated
        block = original_read(descriptor, size)
        if block and not mutated:
            mutated = True
            with weights.open("ab") as handle:
                handle.write(b"changed")
        return block

    monkeypatch.setattr(provenance.os, "read", mutating_read)

    with pytest.raises(ValueError, match="changed while being verified"):
        build_model_file_manifest(snapshot)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW unavailable")
def test_model_snapshot_manifest_opens_files_without_following_symlinks(
    tmp_path, monkeypatch
):
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    (snapshot / "weights.bin").write_bytes(b"weights")
    original_open = os.open
    observed_flags: list[int] = []

    def tracked_open(path, flags, *args, **kwargs):
        observed_flags.append(flags)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(provenance.os, "open", tracked_open)

    build_model_file_manifest(snapshot)

    assert observed_flags
    assert all(flags & os.O_NOFOLLOW for flags in observed_flags)


def test_model_snapshot_manifest_rejects_duplicate_and_unsafe_paths(tmp_path):
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    (snapshot / "weights.bin").write_bytes(b"weights")
    manifest, _ = write_model_file_manifest(snapshot)

    def replace_manifest(payload):
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        (snapshot / "AUDITOPS_FILES_SHA256").write_bytes(encoded)
        for name in ("AUDITOPS_SNAPSHOT_SHA256", "AUDITOPS_SNAPSHOT_COMPLETE"):
            (snapshot / name).write_text(f"{digest}\n", encoding="ascii", newline="\n")
        return digest

    duplicate = json.loads(json.dumps(manifest))
    duplicate["files"].append(dict(duplicate["files"][0]))
    duplicate["file_count"] += 1
    duplicate_digest = replace_manifest(duplicate)
    with pytest.raises(ValueError, match="duplicate path"):
        verify_model_snapshot(snapshot, duplicate_digest, require_read_only=False)

    unsafe = json.loads(json.dumps(manifest))
    unsafe["files"][0]["path"] = "../weights.bin"
    unsafe_digest = replace_manifest(unsafe)
    with pytest.raises(ValueError, match="unsafe"):
        verify_model_snapshot(snapshot, unsafe_digest, require_read_only=False)


def test_model_snapshot_manifest_rejects_symlinks(tmp_path):
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    target = snapshot / "weights.bin"
    target.write_bytes(b"weights")
    try:
        (snapshot / "linked.bin").symlink_to(target.name)
    except OSError:
        pytest.skip("This platform does not permit test symlinks")
    with pytest.raises(ValueError, match="non-regular file"):
        write_model_file_manifest(snapshot)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode-bit invariant")
def test_model_snapshot_must_be_finalized_read_only(tmp_path):
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    (snapshot / "weights.bin").write_bytes(b"weights")
    _, digest = write_model_file_manifest(snapshot)
    with pytest.raises(ValueError, match="writable"):
        verify_model_snapshot(snapshot, digest, require_read_only=True)

    for path in snapshot.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    snapshot.chmod(0o555)
    verify_model_snapshot(snapshot, digest, require_read_only=True)
