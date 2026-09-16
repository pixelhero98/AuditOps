from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

import auditops.agent_benchmark_v24 as benchmark_v24
import auditops.agent_review_v24 as review_v24
from auditops.agent_benchmark_v24 import (
    _validate_human_review,
    write_agent_benchmark_approval_v24,
)
from auditops.agent_review_v24 import (
    prepare_agent_benchmark_review_bundle_v24,
    verify_agent_benchmark_review_bundle_v24,
)
from auditops.canonical_json import canonical_json_bytes


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value, newline=True))


def _artifact(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _review(
    *,
    candidate_id: str,
    packet_sha256: str,
    role: str,
    identity: str,
    task_ids: list[str],
    disposition: str = "APPROVE",
) -> dict[str, object]:
    checks = {
        "question_quality": disposition == "APPROVE",
        "subtype": disposition == "APPROVE",
        "scope": disposition == "APPROVE",
        "answerability": disposition == "APPROVE",
        "evidence": disposition == "APPROVE",
        "semantic_anchors": disposition == "APPROVE",
        "refusal_validity": disposition == "APPROVE",
    }
    return {
        "review_decision_version": "auditops-benchmark-review-decision.v2.4",
        "candidate_id": candidate_id,
        "rendered_packet_sha256": packet_sha256,
        "reviewer": {
            "identity": identity,
            "role": role,
            "reviewed_at": "2026-08-26T12:00:00+01:00",
            "attestation": "I independently reviewed the assigned AuditOps cases as a human reviewer.",
        },
        "decisions": [
            {
                "task_id": task_id,
                "decision": disposition,
                "checks": checks,
                "comment": None if disposition == "APPROVE" else "Candidate defect",
            }
            for task_id in task_ids
        ],
    }


def _bundle_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path]:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    raw_root = tmp_path / "raw"
    database = tmp_path / "corpus.sqlite"
    database.write_bytes(b"offline-review-corpus")
    review_html = tmp_path / "review.html"
    review_html.write_bytes(b"<html><body>review</body></html>\n")
    candidate_id = "a" * 64

    filings: list[dict[str, object]] = []
    chunks: dict[str, list[dict[str, object]]] = {}
    source_filings: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    task_ids: list[str] = []
    raw_provenance: dict[str, dict[str, object]] = {}
    for issuer_index in range(20):
        ticker = f"T{issuer_index:02d}"
        filing_id = f"filing-{issuer_index:02d}"
        accession = f"0000000000-26-{issuer_index:06d}"
        source_entity_id = f"{issuer_index + 1:010d}"
        html_member = f"{ticker.casefold()}-20251231.htm"
        html_payload = f"<html><body>{ticker} frozen filing</body></html>".encode()
        archive_name = f"{ticker}_10-K_{accession}.zip"
        archive = raw_root / ticker / archive_name
        archive.parent.mkdir(parents=True)
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(html_member, html_payload)
        provenance = {
            "archive_name": archive_name,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "archive_size_bytes": archive.stat().st_size,
            "html_member": html_member,
            "html_sha256": hashlib.sha256(html_payload).hexdigest(),
            "html_size_bytes": len(html_payload),
            "supplement_version": "auditops.raw_cam_supplement.v2.4",
            "section_text_sha256": hashlib.sha256(f"cam-{ticker}".encode()).hexdigest(),
            "sentence_count": 1,
        }
        raw_provenance[filing_id] = provenance
        filing = {
            "filing_id": filing_id,
            "ticker": ticker,
            "form_type": "10-K",
            "fiscal_year_focus": 2025,
            "report_date": "2025-12-31",
            "cik": source_entity_id,
            "accession": accession,
            "source_sha256": provenance["archive_sha256"],
            "source_size_bytes": provenance["archive_size_bytes"],
            "source_entity_id": source_entity_id,
        }
        filings.append(filing)
        chunks[filing_id] = [{"content": f"canonical-{ticker}"}]
        source_filings.append(
            {
                "filing_id": filing_id,
                "ticker": ticker,
                "source_entity_id": source_entity_id,
                "source_sha256": provenance["archive_sha256"],
                "source_size_bytes": provenance["archive_size_bytes"],
                "raw_cam_source": provenance,
            }
        )
        canonical_text = f"canonical-{ticker}"
        cam_text = f"canonical-{ticker}\ncam-{ticker}"
        for case_index in range(10):
            task_id = f"task-{issuer_index:02d}-{case_index:02d}"
            task_ids.append(task_id)
            refusal_scope = None
            if case_index >= 5:
                use_cam = case_index == 9
                scope_text = cam_text if use_cam else canonical_text
                refusal_scope = {
                    "scope_description": "Complete frozen scope",
                    "causal_negative_type": "human_verified_full_filing_absence",
                    "absent_claim": f"absent-{task_id}",
                    "full_filing_chunk_count": 2 if use_cam else 1,
                    "full_filing_text_sha256": hashlib.sha256(
                        scope_text.encode()
                    ).hexdigest(),
                    "mechanical_absence_check": True,
                }
            records.append(
                {
                    "task_id": task_id,
                    "filing": {"filing_id": filing_id},
                    "refusal_scope": refusal_scope,
                }
            )

    _write_json(candidate / "candidate_manifest.json", {"candidate_id": candidate_id})
    _write_json(
        candidate / "review_packet.json",
        {
            "review_packet_version": "auditops-benchmark-review-packet.v2.4",
            "case_count": 200,
            "records": records,
        },
    )
    _write_json(
        candidate / "review_sample_40.json",
        {
            "secondary_sample_version": "auditops-agent-secondary-review-sample.v2.4",
            "task_ids": task_ids[:40],
        },
    )
    _write_json(
        candidate / "source_manifest.json",
        {
            "source_manifest_version": "v2.4-candidate",
            "corpus_id": "fixture",
            "sources": {
                "corpus_sqlite": _artifact(database),
                "filings": source_filings,
            },
        },
    )

    monkeypatch.setattr(
        review_v24,
        "verify_agent_benchmark_candidate_v24",
        lambda _: {"valid": True, "errors": [], "candidate_id": candidate_id},
    )
    monkeypatch.setattr(review_v24, "_load_corpus", lambda _: (filings, chunks))
    monkeypatch.setattr(
        review_v24,
        "_segment_items",
        lambda values: [dict(value) for value in values],
    )
    monkeypatch.setattr(
        review_v24,
        "_raw_cam_segments_v24",
        lambda _root, filing: (
            [{"content": f"cam-{filing['ticker']}"}],
            raw_provenance[str(filing["filing_id"])],
        ),
    )
    rendered_payload = review_html.read_bytes()

    def render(_candidate: Path, output: Path) -> dict[str, object]:
        output.write_bytes(rendered_payload)
        return {"output": str(output)}

    monkeypatch.setattr(benchmark_v24, "render_agent_benchmark_review_v24", render)
    return candidate, review_html, database, raw_root


def _approval_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, list[str], list[str]]:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    candidate_id = "a" * 64
    task_ids = [f"task-{index:03d}" for index in range(200)]
    sample_ids = task_ids[:40]
    _write_json(candidate / "candidate_manifest.json", {"candidate_id": candidate_id})
    _write_json(
        candidate / "review_packet.json",
        {
            "records": [
                {"task_id": task_id, "narrative_subtype": "footnote_note"}
                for task_id in task_ids
            ]
        },
    )
    _write_json(candidate / "review_sample_40.json", {"task_ids": sample_ids})
    _write_json(candidate / "source_manifest.json", {"sources": {}})
    rendered = tmp_path / "review.html"
    rendered.write_text("review", encoding="utf-8")
    monkeypatch.setattr(
        benchmark_v24,
        "verify_agent_benchmark_candidate_v24",
        lambda _: {"valid": True, "errors": []},
    )
    return candidate, rendered, task_ids, sample_ids


def test_v24_offline_review_bundle_is_bound_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, review_html, database, raw_root = _bundle_fixture(tmp_path, monkeypatch)
    output = tmp_path / "review-bundle"
    result = prepare_agent_benchmark_review_bundle_v24(
        candidate, review_html, database, raw_root, output
    )
    assert result["verified"] is True
    assert result["external_human_approval"] is False
    verification = verify_agent_benchmark_review_bundle_v24(output)
    assert verification["valid"] is True
    manifest = json.loads((output / "bundle_manifest.json").read_text())
    jsonschema = pytest.importorskip("jsonschema")
    bundle_schema = json.loads(
        (
            Path(__file__).parents[1]
            / "auditops"
            / "specs"
            / "benchmark_review_bundle.v2.4.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(bundle_schema).validate(manifest)
    assert manifest["counts"] == {
        "filings": 20,
        "narrative_cases": 200,
        "refusal_cases": 100,
        "secondary_cases": 40,
    }
    assert len(manifest["refusal_scope_bindings"]) == 100
    assert {item["scope_kind"] for item in manifest["refusal_scope_bindings"]} == {
        "CANONICAL",
        "CANONICAL_PLUS_RAW_CAM",
    }
    primary = json.loads((output / "primary_review.template.json").read_text())
    secondary = json.loads((output / "secondary_review.template.json").read_text())
    assert len(primary["decisions"]) == 200
    assert len(secondary["decisions"]) == 40
    assert all(item["decision"] == "REJECT" for item in primary["decisions"])
    with pytest.raises(ValueError, match="external human"):
        _validate_human_review(
            primary,
            candidate_id="a" * 64,
            packet_sha256=hashlib.sha256(review_html.read_bytes()).hexdigest(),
            expected_role="PRIMARY",
            expected_ids={item["task_id"] for item in primary["decisions"]},
        )


def test_v24_offline_review_bundle_detects_scope_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, review_html, database, raw_root = _bundle_fixture(tmp_path, monkeypatch)
    output = tmp_path / "review-bundle"
    prepare_agent_benchmark_review_bundle_v24(
        candidate, review_html, database, raw_root, output
    )
    scope = next((output / "filings").glob("*/canonical_scope.txt"))
    scope.write_text("tampered", encoding="utf-8")
    result = verify_agent_benchmark_review_bundle_v24(output)
    assert result["valid"] is False
    assert "artifact mismatch" in result["errors"][0]


def test_v24_offline_review_bundle_rejects_wrong_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, review_html, database, raw_root = _bundle_fixture(tmp_path, monkeypatch)
    review_html.write_text("not the candidate rendering", encoding="utf-8")
    with pytest.raises(ValueError, match="exact candidate rendering"):
        prepare_agent_benchmark_review_bundle_v24(
            candidate, review_html, database, raw_root, tmp_path / "bundle"
        )


def test_v24_review_rejection_requires_comment_and_boolean_checks() -> None:
    review = _review(
        candidate_id="a" * 64,
        packet_sha256="b" * 64,
        role="PRIMARY",
        identity="Human One",
        task_ids=["task-1"],
        disposition="REJECT",
    )
    review["decisions"][0]["comment"] = None  # type: ignore[index]
    with pytest.raises(ValueError, match="corrective comment"):
        _validate_human_review(
            review,
            candidate_id="a" * 64,
            packet_sha256="b" * 64,
            expected_role="PRIMARY",
            expected_ids={"task-1"},
        )
    review["decisions"][0]["comment"] = "defect"  # type: ignore[index]
    review["decisions"][0]["checks"]["scope"] = 1  # type: ignore[index]
    with pytest.raises(TypeError, match="booleans"):
        _validate_human_review(
            review,
            candidate_id="a" * 64,
            packet_sha256="b" * 64,
            expected_role="PRIMARY",
            expected_ids={"task-1"},
        )


def test_v24_approval_requires_distinct_human_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, rendered, task_ids, sample_ids = _approval_fixture(tmp_path, monkeypatch)
    candidate_id = "a" * 64
    packet_sha = hashlib.sha256(rendered.read_bytes()).hexdigest()
    primary_path = tmp_path / "primary.json"
    secondary_path = tmp_path / "secondary.json"
    _write_json(
        primary_path,
        _review(
            candidate_id=candidate_id,
            packet_sha256=packet_sha,
            role="PRIMARY",
            identity="Same Human",
            task_ids=task_ids,
        ),
    )
    _write_json(
        secondary_path,
        _review(
            candidate_id=candidate_id,
            packet_sha256=packet_sha,
            role="SECONDARY",
            identity=" same   human ",
            task_ids=sample_ids,
        ),
    )
    with pytest.raises(ValueError, match="different human identities"):
        write_agent_benchmark_approval_v24(
            candidate,
            rendered,
            primary_path,
            secondary_path,
            tmp_path / "approval.json",
        )


def test_v24_subtype_wide_adjudication_binds_both_reviewers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, rendered, task_ids, sample_ids = _approval_fixture(tmp_path, monkeypatch)
    candidate_id = "a" * 64
    packet_sha = hashlib.sha256(rendered.read_bytes()).hexdigest()
    primary = _review(
        candidate_id=candidate_id,
        packet_sha256=packet_sha,
        role="PRIMARY",
        identity="Human One",
        task_ids=task_ids,
    )
    primary["decisions"][0] = {  # type: ignore[index]
        "task_id": task_ids[0],
        "decision": "REJECT",
        "checks": {
            "question_quality": False,
            "subtype": True,
            "scope": True,
            "answerability": True,
            "evidence": True,
            "semantic_anchors": True,
            "refusal_validity": True,
        },
        "comment": "Wording required joint re-review.",
    }
    secondary = _review(
        candidate_id=candidate_id,
        packet_sha256=packet_sha,
        role="SECONDARY",
        identity="Human Two",
        task_ids=sample_ids,
    )
    primary_path = tmp_path / "primary.json"
    secondary_path = tmp_path / "secondary.json"
    _write_json(primary_path, primary)
    _write_json(secondary_path, secondary)
    with pytest.raises(ValueError, match="subtype-wide adjudication"):
        write_agent_benchmark_approval_v24(
            candidate,
            rendered,
            primary_path,
            secondary_path,
            tmp_path / "approval-without-adjudication.json",
        )

    adjudication = {
        "adjudication_version": "auditops-benchmark-adjudication.v2.4",
        "candidate_id": candidate_id,
        "rendered_packet_sha256": packet_sha,
        "affected_subtypes": ["footnote_note"],
        "reviewers": {
            "primary_identity": "Human One",
            "secondary_identity": "Human Two",
            "attestation": "Both named reviewers independently re-reviewed every case in each affected subtype.",
        },
        "decisions": [
            {
                "task_id": task_id,
                "primary_re_reviewed": True,
                "secondary_re_reviewed": True,
                "final_decision": "APPROVE",
                "comment": "Both reviewers confirmed the final candidate wording.",
            }
            for task_id in task_ids
        ],
    }
    adjudication_path = tmp_path / "adjudication.json"
    _write_json(adjudication_path, adjudication)
    approval_path = tmp_path / "approval.json"
    result = write_agent_benchmark_approval_v24(
        candidate,
        rendered,
        primary_path,
        secondary_path,
        approval_path,
        adjudication_review=adjudication_path,
    )
    assert result["approved_case_count"] == 200
    approval = json.loads(approval_path.read_text())
    assert approval["adjudication"]["affected_subtypes"] == ["footnote_note"]
    assert approval["adjudication"]["records"] == 200


def test_v24_review_and_adjudication_public_schemas() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    spec_root = Path(__file__).parents[1] / "auditops" / "specs"
    decision_schema = json.loads(
        (spec_root / "benchmark_review_decision.v2.4.schema.json").read_text()
    )
    rejected = _review(
        candidate_id="a" * 64,
        packet_sha256="b" * 64,
        role="PRIMARY",
        identity="Human One",
        task_ids=["task-1"],
        disposition="REJECT",
    )
    jsonschema.Draft202012Validator(decision_schema).validate(rejected)
    rejected["decisions"][0]["comment"] = None  # type: ignore[index]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(decision_schema).validate(rejected)

    adjudication_schema = json.loads(
        (spec_root / "benchmark_adjudication.v2.4.schema.json").read_text()
    )
    adjudication = {
        "adjudication_version": "auditops-benchmark-adjudication.v2.4",
        "candidate_id": "a" * 64,
        "rendered_packet_sha256": "b" * 64,
        "affected_subtypes": ["footnote_note"],
        "reviewers": {
            "primary_identity": "Human One",
            "secondary_identity": "Human Two",
            "attestation": "Both named reviewers independently re-reviewed every case in each affected subtype.",
        },
        "decisions": [
            {
                "task_id": "task-1",
                "primary_re_reviewed": True,
                "secondary_re_reviewed": True,
                "final_decision": "APPROVE",
                "comment": "Joint rationale",
            }
        ],
    }
    jsonschema.Draft202012Validator(adjudication_schema).validate(adjudication)
