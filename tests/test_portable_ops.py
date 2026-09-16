from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _extract_heredoc(content: str, marker: str) -> str:
    match = re.search(
        rf"<<'{re.escape(marker)}'\n(?P<body>.*?)\n{re.escape(marker)}",
        content,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group("body")


def _source_manifest(source_root: Path) -> dict[str, object]:
    rows = []
    for path in sorted(source_root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(source_root).as_posix(),
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    tree_payload = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        "provenance_version": "auditops-provenance.v1",
        "base_commit": "a" * 40,
        "branch": "feature/test",
        "source_tree_sha256": hashlib.sha256(tree_payload).hexdigest(),
        "git_diff_sha256": "b" * 64,
        "dirty": True,
        "file_count": len(rows),
        "files": rows,
    }


def _run_source_validator(
    script: str,
    source_root: Path,
    manifest: dict[str, object],
    manifest_path: Path,
    stage: Path,
) -> subprocess.CompletedProcess[str]:
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-",
            str(source_root),
            str(manifest_path),
            str(stage),
            digest,
        ],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )


def test_shell_and_slurm_job_files_contain_no_carriage_returns() -> None:
    paths = set((ROOT / "scripts").rglob("*.sh"))
    paths.update(ROOT.rglob("*.sbatch"))
    assert paths
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in paths
        if b"\r" in path.read_bytes()
    ]
    assert offenders == []


def test_inference_is_clean_offline_and_evaluator_artifacts_are_not_mounted() -> None:
    runner = _read("scripts/slurm/run_agent_baseline.sbatch")
    assert (
        "CONTAINER_ISOLATION_ARGS=(--cleanenv --containall --net --network none --nv)"
        in runner
    )
    assert 'env -i "${CONTAINER_LAUNCH_ENV[@]}" "$CONTAINER_RUNTIME_BIN" exec' in runner
    assert "CONTAINER_ISOLATION_ARGS+=(--no-mount bind-paths,hostfs,cwd)" in runner
    assert "export APPTAINERENV_" not in runner
    assert "export SINGULARITYENV_" not in runner
    assert "CONTAINER_ENV_ARGS=(" in runner
    assert "HF_HUB_OFFLINE=1" in runner
    assert "TRANSFORMERS_OFFLINE=1" in runner
    assert "VLLM_NO_USAGE_STATS=1" in runner
    assert "DO_NOT_TRACK=1" in runner
    assert "AUDITOPS_DISABLE_NETWORK=1" in runner
    assert 'cp -- "$AUDITOPS_BENCHMARK_DIR/gold.jsonl"' not in runner
    assert "verifier_observations.jsonl" not in runner
    assert 'Path("/auditops-benchmark/gold.jsonl").exists()' in runner
    assert (
        "for name in benchmark_manifest.json cases.jsonl evidence.jsonl parity_50.json"
    ) in runner
    assert 'cp -- "$AUDITOPS_BENCHMARK_DIR/$name" "$BENCHMARK_VIEW/$name"' in runner
    assert "runtime_observations.jsonl" not in runner
    assert '"HF_TOKEN",' in runner
    assert '"COMPANIES_HOUSE_API_KEY",' in runner
    assert "Forbidden secret/control environment is present" in runner


def test_inference_network_namespace_and_socket_preflight_fail_closed() -> None:
    runner = _read("scripts/slurm/run_agent_baseline.sbatch")
    preflight = _extract_heredoc(runner, "NETWORK_DENIAL_PREFLIGHT_PY")
    assert 'os.readlink("/proc/self/ns/net")' in preflight
    assert "socket.if_nameindex()" in preflight
    assert 'interfaces - {"lo"}' in preflight
    assert "probe.connect(address)" in preflight
    assert 'print("AUDITOPS_NETWORK_DENIED")' in preflight
    assert "if ! NETWORK_PREFLIGHT_OUTPUT=$(auditops_apptainer_exec" in runner
    assert 'if [[ "$NETWORK_PREFLIGHT_OUTPUT" != "AUDITOPS_NETWORK_DENIED" ]]' in runner
    assert '"preflight": "passed"' in runner


def test_inference_forwards_strict_provenance_and_verifies_inputs() -> None:
    runner = _read("scripts/slurm/run_agent_baseline.sbatch")
    required = (
        "AUDITOPS_GIT_COMMIT",
        "AUDITOPS_SOURCE_TREE_SHA256",
        "AUDITOPS_SOURCE_MANIFEST_SHA256",
        "AUDITOPS_CONTAINER_SHA256",
        "AUDITOPS_MODEL_SNAPSHOT_SHA256",
        "AUDITOPS_PACKAGE_LOCK_SHA256",
        "AUDITOPS_GPU_MODEL",
    )
    for name in required:
        assert f'--env "{name}=' in runner
    assert "verify_model_snapshot" in runner
    assert "slurm_launch_manifest.json" in runner
    assert "assert_no_secrets" in runner
    assert ': "${AUDITOPS_SOURCE_MANIFEST_SHA256:?' in runner
    assert (
        'ACTUAL_SOURCE_MANIFEST_SHA256=$(sha256sum "$AUDITOPS_SOURCE_MANIFEST"'
        in runner
    )
    assert '"$SOURCE_VIEW:/workspace/AuditOps:ro"' in runner
    assert '"$AUDITOPS_REPO_ROOT:/workspace/AuditOps:ro"' not in runner


def test_inference_requires_external_container_model_and_config_pins() -> None:
    runner = _read("scripts/slurm/run_agent_baseline.sbatch")
    assert 'config.get("model_config_version") != "v2"' in runner
    assert 'config.get("model_config_version") != "v1"' not in runner
    for name in (
        "AUDITOPS_VLLM_SIF_SHA256",
        "AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256",
        "AUDITOPS_MODEL_CONFIG_SHA256",
    ):
        assert f': "${{{name}:?' in runner
        assert f"auditops_require_sha256 {name}" in runner
    assert 'cmp -s -- "$AUDITOPS_VLLM_SIF_SHA256_FILE"' in runner
    assert 'cmp -s -- "$AUDITOPS_MODEL_CONFIG_SHA256_FILE"' in runner
    assert '"$ACTUAL_SIF_SHA256" != "$AUDITOPS_VLLM_SIF_SHA256"' in runner
    assert '"$ACTUAL_MODEL_CONFIG_SHA256" != "$AUDITOPS_MODEL_CONFIG_SHA256"' in runner
    assert '"$AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256"' in runner
    assert '"$SOURCE_VIEW/config/model_revisions.json"' in runner
    assert '"$AUDITOPS_MODEL_REGISTRY"' not in runner


def test_inference_scrubs_control_environment_and_checks_hidden_host_inputs() -> None:
    runner = _read("scripts/slurm/run_agent_baseline.sbatch")
    preflight = _extract_heredoc(runner, "NETWORK_DENIAL_PREFLIGHT_PY")
    assert 'APPTAINER_*|SINGULARITY_*) unset "$environment_name"' in runner
    assert 'env -i "${CONTAINER_LAUNCH_ENV[@]}"' in runner
    assert 'or name.startswith("APPTAINERENV_")' in preflight
    assert 'or name.startswith("SINGULARITYENV_")' in preflight
    assert "name.endswith(forbidden_secret_suffixes)" in preflight
    assert "original_benchmark = Path(sys.argv[1])" in preflight
    assert "original_gold = Path(sys.argv[2])" in preflight
    assert "Could not prove the {label} path is absent" in preflight
    assert "Inference-visible benchmark file set is not exact" in preflight
    assert "verify_model_snapshot(" in preflight


def test_source_manifest_validator_accepts_only_the_exact_canonical_snapshot(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "repo"
    source_root.mkdir()
    (source_root / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source_root / "pkg").mkdir()
    (source_root / "pkg" / "b.json").write_text('{"ok":true}\n', encoding="utf-8")
    manifest = _source_manifest(source_root)
    runner = _read("scripts/slurm/run_agent_baseline.sbatch")
    validator = _extract_heredoc(runner, "SOURCE_MANIFEST_VALIDATOR_PY")

    accepted = _run_source_validator(
        validator,
        source_root,
        manifest,
        tmp_path / "source-manifest.json",
        tmp_path / "accepted-stage",
    )
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.splitlines() == [
        "a" * 40,
        manifest["source_tree_sha256"],
        hashlib.sha256((tmp_path / "source-manifest.json").read_bytes()).hexdigest(),
    ]
    assert (tmp_path / "accepted-stage" / "a.py").read_bytes() == (
        source_root / "a.py"
    ).read_bytes()

    (source_root / "a.py").write_text("VALUE = 9\n", encoding="utf-8")
    changed_source = _run_source_validator(
        validator,
        source_root,
        manifest,
        tmp_path / "source-manifest-changed.json",
        tmp_path / "changed-stage",
    )
    assert changed_source.returncode != 0
    assert "metadata or checksum mismatch: a.py" in changed_source.stderr
    (source_root / "a.py").write_text("VALUE = 1\n", encoding="utf-8")

    (source_root / "sitecustomize.py").write_text(
        "raise RuntimeError('must never execute')\n", encoding="utf-8"
    )
    rejected = _run_source_validator(
        validator,
        source_root,
        manifest,
        tmp_path / "source-manifest-extra.json",
        tmp_path / "rejected-stage",
    )
    assert rejected.returncode != 0
    assert "unexpected=['sitecustomize.py']" in rejected.stderr


def test_source_manifest_validator_rejects_schema_count_size_and_tree_tampering(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "repo"
    source_root.mkdir()
    (source_root / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source_root / "b.py").write_text("VALUE = 2\n", encoding="utf-8")
    baseline = _source_manifest(source_root)
    validator = _extract_heredoc(
        _read("scripts/slurm/run_agent_baseline.sbatch"), "SOURCE_MANIFEST_VALIDATOR_PY"
    )

    mutations: list[tuple[str, dict[str, object], str]] = []
    extra_key = deepcopy(baseline)
    extra_key["untrusted"] = True
    mutations.append(("schema", extra_key, "unexpected top-level schema"))
    wrong_count = deepcopy(baseline)
    wrong_count["file_count"] = int(wrong_count["file_count"]) + 1
    mutations.append(("count", wrong_count, "file_count does not match"))
    wrong_size = deepcopy(baseline)
    wrong_size["files"][0]["size"] += 1  # type: ignore[index,operator]
    mutations.append(("size", wrong_size, "metadata or checksum mismatch"))
    wrong_tree = deepcopy(baseline)
    wrong_tree["source_tree_sha256"] = "0" * 64
    mutations.append(("tree", wrong_tree, "source_tree_sha256 does not match"))
    duplicate_path = deepcopy(baseline)
    duplicate_path["files"][1]["path"] = duplicate_path["files"][0]["path"]  # type: ignore[index]
    mutations.append(("duplicate", duplicate_path, "duplicate path"))
    unsorted = deepcopy(baseline)
    unsorted["files"].reverse()  # type: ignore[union-attr]
    mutations.append(("order", unsorted, "not in canonical path order"))

    for index, (name, manifest, expected_error) in enumerate(mutations):
        result = _run_source_validator(
            validator,
            source_root,
            manifest,
            tmp_path / f"manifest-{name}.json",
            tmp_path / f"stage-{index}",
        )
        assert result.returncode != 0, name
        assert expected_error in result.stderr, (name, result.stderr)


def test_registered_model_revisions_are_commit_pinned() -> None:
    registry = json.loads(_read("config/model_revisions.json"))
    assert len(registry["models"]) == 4
    for model in registry["models"]:
        assert re.fullmatch(r"[0-9a-f]{40}", model["revision"])


def test_model_config_stage_requires_finalized_snapshot() -> None:
    script = _read("scripts/slurm/write_model_config.sbatch")
    assert "write-model-config" in script
    assert "AUDITOPS_SNAPSHOT_COMPLETE" in script
    assert "auditops_assert_new_path" in script
    assert "AUDITOPS_MODEL_CONFIG_SHA256_FILE" in script
    assert "os.link(temporary_path, output_path)" in script
    assert "MODEL_CONFIG_SHA256" in script
