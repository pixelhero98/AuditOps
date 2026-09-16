from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _heredoc(source: str, marker: str) -> str:
    start_token = f"<<'{marker}'\n"
    start = source.index(start_token) + len(start_token)
    end = source.index(f"\n{marker}\n", start)
    return source[start:end]


def test_vlm_inference_is_hardware_bound_offline_and_gold_blind() -> None:
    generic = _text("scripts/slurm/run_companies_house_vlm_diagnostic.sbatch")
    assert "AUDITOPS_EXPECTED_GPU" in generic
    assert '"$AUDITOPS_GPU_NAME" != "$AUDITOPS_EXPECTED_GPU"' in generic
    assert "AUDITOPS_SOURCE_MANIFEST_SHA256" in generic
    assert ': "${AUDITOPS_VLLM_SIF_SHA256:?Externally pin' in generic
    assert ': "${AUDITOPS_MODEL_CONFIG_SHA256:?Externally pin' in generic
    assert "AUDITOPS_MODEL_SNAPSHOT_MANIFEST_SHA256" in generic
    assert "verify_model_snapshot" in generic
    assert "source_tree_sha256" in generic
    assert "Container checksum mismatch" in generic
    assert "Container checksum record differs from the external digest pin" in generic
    assert (
        "Model-config checksum record differs from the external digest pin" in generic
    )
    assert "Staged model config differs from its external pin" in generic
    assert "In-container model config differs from its external pin" in generic
    assert "cmp -s" in generic and "basename --" in generic
    assert "for name in diagnostic_manifest.json cases.jsonl" in generic
    assert (
        "Evaluator-only evaluation gold is intentionally neither copied nor mounted"
        in generic
    )
    assert "--net --network none" in generic
    assert "socket.AF_INET" in generic and "socket.AF_INET6" in generic
    assert "separate network namespace" in generic
    assert "non-loopback interface" in generic
    assert "COMPANIES_HOUSE_API_KEY" in generic and "HF_TOKEN" in generic
    assert "pdftoppm" in generic
    assert "vllm.__version__" in generic and "0.26.0" in generic
    assert "run-companies-house-vlm-diagnostic" in generic
    assert "run-agent-baseline" not in generic
    assert "shutil.copyfile" not in generic
    assert "\"$AUDITOPS_SOURCE_MANIFEST_SHA256\" <<'PY'" in generic
    assert 'sha256sum "$AUDITOPS_SOURCE_MANIFEST"' not in generic
    assert "read_regular_once(candidate" in generic
    assert '--bind "$AUDITOPS_CH_PDF_ROOT:$AUDITOPS_CH_PDF_ROOT:ro"' not in generic
    assert (
        '--bind "$AUDITOPS_CH_VLM_FEW_SHOT_PDF_ROOT:$AUDITOPS_CH_VLM_FEW_SHOT_PDF_ROOT:ro"'
        not in generic
    )
    assert '--bind "$EVALUATION_PDF_VIEW:$EVALUATION_PDF_SOURCE_ROOT:ro"' in generic
    assert '--bind "$FEW_SHOT_PDF_VIEW:$FEW_SHOT_PDF_SOURCE_ROOT:ro"' in generic
    assert '"$FEW_VIEW/cases.jsonl"' in generic
    assert "Frozen VLM cases contain duplicate PDF paths" in generic
    assert "VLM PDF path contains a symlink" in generic
    assert "Staged VLM PDF tree contains a non-PDF or special entry" in generic
    assert "PDF mount exposes adjacent XHTML/text/audit or special files" in generic
    assert "PDF mount exposes files outside the exact frozen PDF set" in generic
    assert '--bind "$RUNTIME_TMP:/auditops-tmp:rw"' not in generic
    assert "Installed Apptainer lacks required --no-mount isolation support" in generic
    assert "--no-mount bind-paths,hostfs,cwd" in generic
    assert "nested below a writable or separately exposed mount" in generic
    assert "python3 -I" in generic
    assert "exec python3 -m auditops.cli" in generic
    assert 'find "$resolved_tmp" -type d -exec chmod u+w -- {} +' in generic


def test_source_validator_stages_the_same_once_read_externally_pinned_bytes(
    tmp_path: Path,
) -> None:
    generic = _text("scripts/slurm/run_companies_house_vlm_diagnostic.sbatch")
    validator = _heredoc(generic, "PY")
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "a.txt").write_bytes(b"alpha\n")
    (source / "nested" / "b.py").write_bytes(b"print('beta')\n")
    rows = []
    for path in sorted((item for item in source.rglob("*") if item.is_file())):
        payload = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(source).as_posix(),
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    tree_payload = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    manifest = {
        "provenance_version": "auditops-provenance.v1",
        "base_commit": "a" * 40,
        "branch": "fixture",
        "source_tree_sha256": hashlib.sha256(tree_payload).hexdigest(),
        "git_diff_sha256": "b" * 64,
        "dirty": True,
        "file_count": len(rows),
        "files": rows,
    }
    manifest_path = tmp_path / "source-manifest.json"
    manifest_payload = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(manifest_payload)
    digest = hashlib.sha256(manifest_payload).hexdigest()
    stage = tmp_path / "stage"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-",
            str(source),
            str(manifest_path),
            str(stage),
            digest,
        ],
        input=validator,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "a" * 40,
        manifest["source_tree_sha256"],
        digest,
    ]
    assert (stage / "a.txt").read_bytes() == b"alpha\n"
    assert (stage / "nested" / "b.py").read_bytes() == b"print('beta')\n"

    rejected = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-",
            str(source),
            str(manifest_path),
            str(tmp_path / "wrong-stage"),
            "0" * 64,
        ],
        input=validator,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "external digest pin" in rejected.stderr
    assert not (tmp_path / "wrong-stage").exists()


def test_pdf_stager_exposes_only_case_selected_pdf_bytes(tmp_path: Path) -> None:
    generic = _text("scripts/slurm/run_companies_house_vlm_diagnostic.sbatch")
    stager = _heredoc(generic, "PDF_STAGE_PY")
    source = tmp_path / "pdf-root"
    filings = source / "filings"
    filings.mkdir(parents=True)
    selected = filings / "selected.pdf"
    selected.write_bytes(b"%PDF-1.4\nselected\n%%EOF\n")
    (filings / "adjacent.xhtml").write_text(
        "<html>not visible</html>", encoding="utf-8"
    )
    (source / "fact_audit.csv").write_text("not,visible\n", encoding="utf-8")
    (source.parent / "parent-audit.txt").write_text("not visible\n", encoding="utf-8")
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps({"task_id": "task-1", "pdf_path": "filings/selected.pdf"}) + "\n",
        encoding="utf-8",
    )
    stage = tmp_path / "pdf-stage"
    manifest = tmp_path / "pdf-stage-manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-",
            str(source),
            str(stage),
            str(manifest),
            str(cases),
        ],
        input=stager,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    staged_files = sorted(
        path.relative_to(stage).as_posix()
        for path in stage.rglob("*")
        if path.is_file()
    )
    assert staged_files == ["filings/selected.pdf"]
    assert (stage / "filings" / "selected.pdf").read_bytes() == selected.read_bytes()
    assert not (stage / "filings" / "adjacent.xhtml").exists()
    assert not (stage / "fact_audit.csv").exists()
    assert not (stage / "parent-audit.txt").exists()
    manifest_payload = manifest.read_bytes()
    assert completed.stdout.strip() == hashlib.sha256(manifest_payload).hexdigest()
    manifest_data = json.loads(manifest_payload)
    assert manifest_data["files"] == [
        {
            "task_id": "task-1",
            "path": "filings/selected.pdf",
            "size": selected.stat().st_size,
            "sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
        }
    ]

    duplicate_cases = tmp_path / "duplicate-cases.jsonl"
    duplicate_cases.write_text(
        "".join(
            json.dumps({"task_id": task_id, "pdf_path": "filings/selected.pdf"}) + "\n"
            for task_id in ("task-1", "task-2")
        ),
        encoding="utf-8",
    )
    rejected = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-",
            str(source),
            str(tmp_path / "duplicate-stage"),
            str(tmp_path / "duplicate-manifest.json"),
            str(duplicate_cases),
        ],
        input=stager,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "duplicate PDF paths" in rejected.stderr

    special_pdf = filings / "special.pdf"
    special_pdf.mkdir()
    special_cases = tmp_path / "special-cases.jsonl"
    special_cases.write_text(
        json.dumps({"task_id": "task-special", "pdf_path": "filings/special.pdf"})
        + "\n",
        encoding="utf-8",
    )
    special_rejected = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-",
            str(source),
            str(tmp_path / "special-stage"),
            str(tmp_path / "special-manifest.json"),
            str(special_cases),
        ],
        input=stager,
        text=True,
        capture_output=True,
        check=False,
    )
    assert special_rejected.returncode != 0
    assert "not a regular file" in special_rejected.stderr

    symlink_pdf = filings / "symlink.pdf"
    try:
        symlink_pdf.symlink_to(selected)
    except OSError:
        pass
    else:
        symlink_cases = tmp_path / "symlink-cases.jsonl"
        symlink_cases.write_text(
            json.dumps({"task_id": "task-symlink", "pdf_path": "filings/symlink.pdf"})
            + "\n",
            encoding="utf-8",
        )
        symlink_rejected = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-",
                str(source),
                str(tmp_path / "symlink-stage"),
                str(tmp_path / "symlink-manifest.json"),
                str(symlink_cases),
            ],
            input=stager,
            text=True,
            capture_output=True,
            check=False,
        )
        assert symlink_rejected.returncode != 0
        assert "contains a symlink" in symlink_rejected.stderr
