"""Offline human-review packaging for the AuditOps v2.4 candidate.

The review bundle is evaluator-only.  It reconstructs the exact full-filing
text scopes used to create refusal gold, binds them to the immutable
candidate, and supplies fail-closed decision templates.  It never creates an
approval or a runnable benchmark.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .agent_benchmark_v24 import (
    REVIEW_DECISION_VERSION_V24,
    _load_corpus,
    _raw_cam_segments_v24,
    _read_json,
    _segment_items,
    _sha256_path,
    _write_json,
    verify_agent_benchmark_candidate_v24,
)
from .canonical_json import canonical_json_sha256
from .provenance import assert_no_secrets, assert_no_secrets_in_value

REVIEW_BUNDLE_VERSION_V24 = "auditops-benchmark-review-bundle.v2.4"
REVIEW_BUNDLE_MANIFEST_VERSION_V24 = "auditops-benchmark-review-bundle-manifest.v2.4"
HUMAN_ATTESTATION_V24 = (
    "I independently reviewed the assigned AuditOps cases as a human reviewer."
)

_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TEMPLATE_IDENTITY = "TEMPLATE_NOT_A_HUMAN_REVIEWER"
_TEMPLATE_COMMENT = "REVIEW_REQUIRED: record the independent human assessment."
_CHECK_NAMES = (
    "question_quality",
    "subtype",
    "scope",
    "answerability",
    "evidence",
    "semantic_anchors",
    "refusal_validity",
)


def _regular_file(path: Path, *, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _scope_text(items: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(str(item.get("content") or "") for item in items)


def _candidate_source_filings(
    source_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sources = source_manifest.get("sources")
    filings = sources.get("filings") if isinstance(sources, Mapping) else None
    if not isinstance(filings, list) or len(filings) != 20:
        raise ValueError("Candidate source manifest must bind exactly 20 filings")
    normalized: list[dict[str, Any]] = []
    for filing in filings:
        if not isinstance(filing, Mapping):
            raise TypeError("Candidate source filing records must be objects")
        normalized.append(dict(filing))
    return normalized


def _read_bound_html(
    raw_root: Path,
    filing: Mapping[str, Any],
    source_filing: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    ticker = str(filing["ticker"])
    accession = str(filing["accession"])
    if not _SAFE_COMPONENT_RE.fullmatch(ticker) or not _SAFE_COMPONENT_RE.fullmatch(
        accession
    ):
        raise ValueError("Filing ticker or accession is unsafe for offline packaging")
    provenance = source_filing.get("raw_cam_source")
    if not isinstance(provenance, Mapping):
        raise TypeError(f"Filing {ticker} lacks raw source provenance")
    archive_name = str(provenance.get("archive_name") or "")
    html_member = str(provenance.get("html_member") or "")
    if (
        not _SAFE_COMPONENT_RE.fullmatch(archive_name)
        or Path(archive_name).suffix.casefold() != ".zip"
        or Path(html_member).name != html_member
        or not html_member.casefold().endswith((".htm", ".html"))
    ):
        raise ValueError(f"Filing {ticker} has unsafe raw source names")
    archive = raw_root / ticker / archive_name
    _regular_file(archive, label=f"Filing {ticker} archive")
    if archive.stat().st_size != provenance.get("archive_size_bytes"):
        raise ValueError(
            f"Filing {ticker} archive size differs from candidate provenance"
        )
    if _sha256_path(archive) != provenance.get("archive_sha256"):
        raise ValueError(
            f"Filing {ticker} archive hash differs from candidate provenance"
        )
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        if any(
            info.is_dir()
            or (info.external_attr >> 16) & 0o170000 == 0o120000
            or Path(info.filename).name != info.filename
            for info in members
        ):
            raise ValueError(f"Filing {ticker} archive contains an unsafe member")
        html_members = [
            info
            for info in members
            if info.filename.casefold().endswith((".htm", ".html"))
        ]
        if len(html_members) != 1 or html_members[0].filename != html_member:
            raise ValueError(f"Filing {ticker} HTML membership differs from provenance")
        payload = handle.read(html_members[0])
    if len(payload) != provenance.get("html_size_bytes"):
        raise ValueError(f"Filing {ticker} HTML size differs from candidate provenance")
    if hashlib.sha256(payload).hexdigest() != provenance.get("html_sha256"):
        raise ValueError(f"Filing {ticker} HTML hash differs from candidate provenance")
    if str(source_filing.get("filing_id")) != str(filing["filing_id"]):
        raise ValueError(f"Filing {ticker} identity differs from candidate provenance")
    return payload, dict(provenance)


def _decision_template(
    *,
    candidate_id: str,
    rendered_packet_sha256: str,
    role: str,
    task_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "review_decision_version": REVIEW_DECISION_VERSION_V24,
        "candidate_id": candidate_id,
        "rendered_packet_sha256": rendered_packet_sha256,
        "reviewer": {
            "identity": _TEMPLATE_IDENTITY,
            "role": role,
            "reviewed_at": "1970-01-01T00:00:00Z",
            "attestation": HUMAN_ATTESTATION_V24,
        },
        "decisions": [
            {
                "task_id": task_id,
                "decision": "REJECT",
                "checks": {name: False for name in _CHECK_NAMES},
                "comment": _TEMPLATE_COMMENT,
            }
            for task_id in task_ids
        ],
    }


def _instructions() -> str:
    return """# AuditOps v2.4 human-review bundle

This directory is evaluator-only and must never be mounted into an inference
job or supplied to a model. Do not edit the bundle. Copy the appropriate JSON
template into a separate decisions directory before completing it.

## Assignments

- PRIMARY: review every task in `primary_review.template.json` (200 cases).
- SECONDARY: independently review only `secondary_review.template.json`
  (the frozen 40-case sample). The secondary reviewer must not see primary
  decisions before signing their own file.
- The reviewers must be different identifiable humans. Codex or another AI
  cannot review or approve the benchmark.

## Checks

- `question_quality`: lucid, single-purpose, and correctly worded.
- `subtype`: correct footnote, policy, opinion-language, or CAM classification.
- `scope`: correct issuer, filing, period, note or section, and attribute.
- `answerability`: directly supported answer or genuinely absent disclosure.
- `evidence`: evidence ID, exact quote, and support window match the filing.
- `semantic_anchors`: positive anchors capture the required information.
- `refusal_validity`: the absent claim and reasonable synonyms are absent from
  the complete bound scope file, not merely from the top-five evidence.

For a non-applicable check, set `true` only after verifying that its associated
field is correctly absent. Every `REJECT` decision requires a specific comment.
If a real defect is found, stop: rebuild a new candidate rather than modifying
this bundle or adjudicating the defect away.

Each filing directory contains exact source HTML plus `canonical_scope.txt`.
CAM refusals may bind `cam_augmented_scope.txt`, which adds the checksum-bound
raw CAM supplement used by candidate construction. `filing_manifest.json`
maps every refusal task to its exact scope file and hash.
"""


def _actual_bundle_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        status = path.lstat()
        if path.is_symlink():
            raise ValueError(f"Review bundle contains a symlink: {relative.as_posix()}")
        if stat.S_ISDIR(status.st_mode):
            continue
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(
                f"Review bundle contains a special file: {relative.as_posix()}"
            )
        files.append(path)
    return files


def prepare_agent_benchmark_review_bundle_v24(
    candidate_dir: str | Path,
    rendered_packet: str | Path,
    corpus_db: str | Path,
    raw_filing_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build an immutable-by-policy offline review bundle, never an approval."""

    candidate_root = Path(candidate_dir)
    verification = verify_agent_benchmark_candidate_v24(candidate_root)
    if not verification["valid"]:
        raise ValueError("Invalid v2.4 candidate: " + "; ".join(verification["errors"]))
    packet_html = _regular_file(Path(rendered_packet), label="Rendered review packet")
    database = _regular_file(Path(corpus_db), label="Canonical corpus database")
    raw_root = Path(raw_filing_root)
    if not raw_root.is_dir() or raw_root.is_symlink():
        raise ValueError("Raw filing root must be a non-symlink directory")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite review bundle: {destination}")

    candidate_manifest = _read_json(candidate_root / "candidate_manifest.json")
    review_packet = _read_json(candidate_root / "review_packet.json")
    sample = _read_json(candidate_root / "review_sample_40.json")
    source_manifest = _read_json(candidate_root / "source_manifest.json")
    candidate_id = str(candidate_manifest["candidate_id"])
    assert_no_secrets_in_value(
        {"candidate_id": candidate_id}, label="v2.4 review bundle configuration"
    )

    # The supplied HTML must be the deterministic renderer output for this
    # exact candidate, not merely an arbitrary file with a plausible heading.
    comparison_root = Path(tempfile.mkdtemp(prefix="auditops-v24-review-render-"))
    try:
        from .agent_benchmark_v24 import render_agent_benchmark_review_v24

        expected_html = comparison_root / "review_packet.html"
        render_agent_benchmark_review_v24(candidate_root, expected_html)
        if packet_html.read_bytes() != expected_html.read_bytes():
            raise ValueError(
                "Rendered review packet is not the exact candidate rendering"
            )
    finally:
        shutil.rmtree(comparison_root, ignore_errors=True)

    sources = source_manifest.get("sources")
    corpus_artifact = (
        sources.get("corpus_sqlite") if isinstance(sources, Mapping) else None
    )
    if (
        not isinstance(corpus_artifact, Mapping)
        or _artifact(database) != corpus_artifact
    ):
        raise ValueError("Canonical corpus database differs from candidate provenance")
    source_filings = _candidate_source_filings(source_manifest)
    source_by_id = {str(item["filing_id"]): item for item in source_filings}
    filings, chunks_by_filing = _load_corpus(database)
    records = review_packet.get("records")
    if not isinstance(records, list) or len(records) != 200:
        raise ValueError("Review packet must contain all 200 narrative cases")
    records_by_filing: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("Review packet records must be objects")
        filing = record.get("filing")
        filing_id = str(filing.get("filing_id")) if isinstance(filing, Mapping) else ""
        records_by_filing.setdefault(filing_id, []).append(dict(record))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        _write_bytes(temporary / "review_packet.html", packet_html.read_bytes())
        for name in (
            "candidate_manifest.json",
            "review_packet.json",
            "review_sample_40.json",
            "source_manifest.json",
        ):
            shutil.copyfile(candidate_root / name, temporary / name)
        _write_bytes(
            temporary / "REVIEWER_INSTRUCTIONS.md", _instructions().encode("utf-8")
        )

        packet_task_ids = [str(record["task_id"]) for record in records]
        sample_task_ids = sample.get("task_ids")
        if not isinstance(sample_task_ids, list) or len(sample_task_ids) != 40:
            raise ValueError("Secondary review membership is invalid")
        rendered_sha256 = _sha256_path(packet_html)
        _write_json(
            temporary / "primary_review.template.json",
            _decision_template(
                candidate_id=candidate_id,
                rendered_packet_sha256=rendered_sha256,
                role="PRIMARY",
                task_ids=packet_task_ids,
            ),
        )
        _write_json(
            temporary / "secondary_review.template.json",
            _decision_template(
                candidate_id=candidate_id,
                rendered_packet_sha256=rendered_sha256,
                role="SECONDARY",
                task_ids=[str(item) for item in sample_task_ids],
            ),
        )

        refusal_bindings: list[dict[str, Any]] = []
        for filing in filings:
            filing_id = str(filing["filing_id"])
            ticker = str(filing["ticker"])
            accession = str(filing["accession"])
            source_filing = source_by_id.get(filing_id)
            if source_filing is None:
                raise ValueError(
                    f"Filing {filing_id} is absent from candidate provenance"
                )
            html_payload, html_provenance = _read_bound_html(
                raw_root, filing, source_filing
            )
            filing_dir = temporary / "filings" / f"{ticker}_{accession}"
            _write_bytes(filing_dir / "source_filing.html", html_payload)

            canonical_segments = _segment_items(chunks_by_filing[filing_id])
            canonical_text = _scope_text(canonical_segments)
            canonical_payload = canonical_text.encode("utf-8")
            _write_bytes(filing_dir / "canonical_scope.txt", canonical_payload)
            canonical_scope = {
                "path": "canonical_scope.txt",
                "chunk_count": len(canonical_segments),
                "bytes": len(canonical_payload),
                "sha256": hashlib.sha256(canonical_payload).hexdigest(),
            }

            raw_cam_segments, raw_cam_provenance = _raw_cam_segments_v24(
                raw_root, filing
            )
            if raw_cam_provenance != html_provenance:
                raise ValueError(
                    f"Filing {ticker} raw CAM provenance is not reproducible"
                )
            cam_segments = [*canonical_segments, *raw_cam_segments]
            cam_text = _scope_text(cam_segments)
            cam_payload = cam_text.encode("utf-8")
            _write_bytes(filing_dir / "cam_augmented_scope.txt", cam_payload)
            cam_scope = {
                "path": "cam_augmented_scope.txt",
                "chunk_count": len(cam_segments),
                "bytes": len(cam_payload),
                "sha256": hashlib.sha256(cam_payload).hexdigest(),
            }

            filing_bindings: list[dict[str, Any]] = []
            for record in records_by_filing.get(filing_id, []):
                refusal = record.get("refusal_scope")
                if refusal is None:
                    continue
                if not isinstance(refusal, Mapping):
                    raise TypeError("Refusal review scope must be an object")
                expected = (
                    int(refusal.get("full_filing_chunk_count", -1)),
                    str(refusal.get("full_filing_text_sha256") or ""),
                )
                choices = [
                    ("CANONICAL", canonical_scope),
                    ("CANONICAL_PLUS_RAW_CAM", cam_scope),
                ]
                matches = [
                    (kind, scope)
                    for kind, scope in choices
                    if expected == (scope["chunk_count"], scope["sha256"])
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Refusal scope {record['task_id']} cannot be reproduced exactly"
                    )
                kind, scope = matches[0]
                binding = {
                    "task_id": str(record["task_id"]),
                    "scope_kind": kind,
                    "scope_path": f"filings/{ticker}_{accession}/{scope['path']}",
                    "chunk_count": scope["chunk_count"],
                    "sha256": scope["sha256"],
                }
                filing_bindings.append(binding)
                refusal_bindings.append(binding)

            filing_manifest = {
                "filing_id": filing_id,
                "ticker": ticker,
                "accession": accession,
                "form_type": str(filing["form_type"]),
                "fiscal_year_focus": int(filing["fiscal_year_focus"]),
                "source_entity_id": str(filing["source_entity_id"]),
                "source_filing_html": {
                    "path": "source_filing.html",
                    "bytes": len(html_payload),
                    "sha256": hashlib.sha256(html_payload).hexdigest(),
                    "archive_name": html_provenance["archive_name"],
                    "archive_sha256": html_provenance["archive_sha256"],
                    "html_member": html_provenance["html_member"],
                },
                "canonical_scope": canonical_scope,
                "cam_augmented_scope": cam_scope,
                "refusal_scope_bindings": filing_bindings,
            }
            _write_json(filing_dir / "filing_manifest.json", filing_manifest)

        if len(refusal_bindings) != 100:
            raise AssertionError(
                "Offline review bundle must bind all 100 refusal cases"
            )

        file_records = [
            {
                "path": path.relative_to(temporary).as_posix(),
                **_artifact(path),
            }
            for path in _actual_bundle_files(temporary)
        ]
        bundle_material = {
            "bundle_manifest_version": REVIEW_BUNDLE_MANIFEST_VERSION_V24,
            "bundle_version": REVIEW_BUNDLE_VERSION_V24,
            "candidate_id": candidate_id,
            "candidate_manifest_sha256": _sha256_path(
                candidate_root / "candidate_manifest.json"
            ),
            "rendered_packet_sha256": rendered_sha256,
            "review_packet_json_sha256": _sha256_path(
                candidate_root / "review_packet.json"
            ),
            "source_manifest_sha256": _sha256_path(
                candidate_root / "source_manifest.json"
            ),
            "primary_task_ids_sha256": canonical_json_sha256(packet_task_ids),
            "secondary_task_ids_sha256": canonical_json_sha256(
                [str(item) for item in sample_task_ids]
            ),
            "counts": {
                "filings": 20,
                "narrative_cases": 200,
                "refusal_cases": 100,
                "secondary_cases": 40,
            },
            "refusal_scope_bindings": sorted(
                refusal_bindings, key=lambda item: str(item["task_id"])
            ),
            "files": file_records,
            "inference_visible": False,
            "external_human_approval": False,
        }
        bundle_manifest = {
            **bundle_material,
            "bundle_id": canonical_json_sha256(bundle_material),
        }
        _write_json(temporary / "bundle_manifest.json", bundle_manifest)
        assert_no_secrets([temporary])
        result = verify_agent_benchmark_review_bundle_v24(temporary)
        if not result["valid"]:
            raise ValueError(
                "Invalid v2.4 review bundle: " + "; ".join(result["errors"])
            )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "bundle_id": bundle_manifest["bundle_id"],
        "candidate_id": candidate_id,
        "output_dir": str(destination),
        "review_status": "PENDING_EXTERNAL_HUMAN_REVIEW",
        "external_human_approval": False,
        "verified": True,
    }


def _validate_template(
    value: Any,
    *,
    role: str,
    candidate_id: str,
    rendered_sha256: str,
    task_ids: Sequence[str],
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{role} review template must be an object")
    reviewer = value.get("reviewer")
    if (
        value.get("review_decision_version") != REVIEW_DECISION_VERSION_V24
        or value.get("candidate_id") != candidate_id
        or value.get("rendered_packet_sha256") != rendered_sha256
        or not isinstance(reviewer, Mapping)
        or reviewer.get("identity") != _TEMPLATE_IDENTITY
        or reviewer.get("role") != role
        or reviewer.get("attestation") != HUMAN_ATTESTATION_V24
    ):
        raise ValueError(f"{role} review template identity is invalid")
    decisions = value.get("decisions")
    if not isinstance(decisions, list) or [
        item.get("task_id") for item in decisions
    ] != list(task_ids):
        raise ValueError(f"{role} review template membership/order is invalid")
    for decision in decisions:
        if (
            not isinstance(decision, Mapping)
            or decision.get("decision") != "REJECT"
            or decision.get("comment") != _TEMPLATE_COMMENT
            or decision.get("checks") != {name: False for name in _CHECK_NAMES}
        ):
            raise ValueError(f"{role} review template must fail closed")


def verify_agent_benchmark_review_bundle_v24(output_dir: str | Path) -> dict[str, Any]:
    """Verify a standalone offline review bundle and every bound scope file."""

    root = Path(output_dir)
    errors: list[str] = []
    try:
        if not root.is_dir() or root.is_symlink():
            raise ValueError("Review bundle root must be a non-symlink directory")
        manifest = _read_json(root / "bundle_manifest.json")
        expected_fields = {
            "bundle_manifest_version",
            "bundle_version",
            "candidate_id",
            "candidate_manifest_sha256",
            "rendered_packet_sha256",
            "review_packet_json_sha256",
            "source_manifest_sha256",
            "primary_task_ids_sha256",
            "secondary_task_ids_sha256",
            "counts",
            "refusal_scope_bindings",
            "files",
            "inference_visible",
            "external_human_approval",
            "bundle_id",
        }
        if set(manifest) != expected_fields:
            raise ValueError("Review bundle manifest fields are missing or unexpected")
        if (
            manifest.get("bundle_manifest_version")
            != REVIEW_BUNDLE_MANIFEST_VERSION_V24
            or manifest.get("bundle_version") != REVIEW_BUNDLE_VERSION_V24
            or manifest.get("inference_visible") is not False
            or manifest.get("external_human_approval") is not False
        ):
            raise ValueError("Review bundle policy identity is invalid")
        material = {key: value for key, value in manifest.items() if key != "bundle_id"}
        if manifest.get("bundle_id") != canonical_json_sha256(material):
            raise ValueError("Review bundle ID does not bind the complete manifest")
        if manifest.get("counts") != {
            "filings": 20,
            "narrative_cases": 200,
            "refusal_cases": 100,
            "secondary_cases": 40,
        }:
            raise ValueError("Review bundle counts are invalid")

        file_records = manifest.get("files")
        if not isinstance(file_records, list):
            raise TypeError("Review bundle file manifest must be an array")
        expected_files: dict[str, Mapping[str, Any]] = {}
        for record in file_records:
            if not isinstance(record, Mapping) or set(record) != {
                "path",
                "bytes",
                "sha256",
            }:
                raise ValueError("Review bundle file records are invalid")
            relative = str(record["path"])
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
                raise ValueError(f"Review bundle file path is unsafe: {relative}")
            if relative in expected_files or relative == "bundle_manifest.json":
                raise ValueError(f"Review bundle file path is duplicated: {relative}")
            expected_files[relative] = record
        actual_files = {
            path.relative_to(root).as_posix(): path
            for path in _actual_bundle_files(root)
            if path.name != "bundle_manifest.json"
        }
        if set(actual_files) != set(expected_files):
            raise ValueError("Review bundle file membership differs from its manifest")
        required_top_level = {
            "REVIEWER_INSTRUCTIONS.md",
            "candidate_manifest.json",
            "primary_review.template.json",
            "review_packet.html",
            "review_packet.json",
            "review_sample_40.json",
            "secondary_review.template.json",
            "source_manifest.json",
        }
        if not required_top_level.issubset(actual_files):
            raise ValueError("Review bundle top-level artifacts are incomplete")
        for relative, path in actual_files.items():
            if _artifact(path) != {
                "bytes": expected_files[relative]["bytes"],
                "sha256": expected_files[relative]["sha256"],
            }:
                raise ValueError(f"Review bundle artifact mismatch: {relative}")

        candidate_manifest = _read_json(root / "candidate_manifest.json")
        packet = _read_json(root / "review_packet.json")
        sample = _read_json(root / "review_sample_40.json")
        source_manifest = _read_json(root / "source_manifest.json")
        if manifest["candidate_id"] != candidate_manifest.get("candidate_id"):
            raise ValueError("Review bundle candidate binding is invalid")
        if manifest["candidate_manifest_sha256"] != _sha256_path(
            root / "candidate_manifest.json"
        ):
            raise ValueError("Review bundle candidate manifest hash is invalid")
        if manifest["rendered_packet_sha256"] != _sha256_path(
            root / "review_packet.html"
        ):
            raise ValueError("Review bundle rendered packet hash is invalid")
        if manifest["review_packet_json_sha256"] != _sha256_path(
            root / "review_packet.json"
        ):
            raise ValueError("Review bundle review packet hash is invalid")
        if manifest["source_manifest_sha256"] != _sha256_path(
            root / "source_manifest.json"
        ):
            raise ValueError("Review bundle source manifest hash is invalid")

        source_filings = _candidate_source_filings(source_manifest)
        source_by_id = {str(item["filing_id"]): item for item in source_filings}
        filing_manifests = sorted(root.glob("filings/*/filing_manifest.json"))
        if len(filing_manifests) != 20:
            raise ValueError("Review bundle must contain exactly 20 filing manifests")
        filing_binding_rows: list[Mapping[str, Any]] = []
        seen_filing_ids: set[str] = set()
        for filing_manifest_path in filing_manifests:
            filing_manifest = _read_json(filing_manifest_path)
            expected_filing_fields = {
                "filing_id",
                "ticker",
                "accession",
                "form_type",
                "fiscal_year_focus",
                "source_entity_id",
                "source_filing_html",
                "canonical_scope",
                "cam_augmented_scope",
                "refusal_scope_bindings",
            }
            if set(filing_manifest) != expected_filing_fields:
                raise ValueError(
                    "Offline filing manifest fields are missing or unexpected"
                )
            filing_id = str(filing_manifest["filing_id"])
            if filing_id in seen_filing_ids or filing_id not in source_by_id:
                raise ValueError(f"Offline filing identity is invalid: {filing_id}")
            seen_filing_ids.add(filing_id)
            source_filing = source_by_id[filing_id]
            if filing_manifest["ticker"] != source_filing.get(
                "ticker"
            ) or filing_manifest["source_entity_id"] != source_filing.get(
                "source_entity_id"
            ):
                raise ValueError(f"Offline filing provenance differs: {filing_id}")
            filing_dir = filing_manifest_path.parent
            html_record = filing_manifest["source_filing_html"]
            provenance = source_filing.get("raw_cam_source")
            if not isinstance(html_record, Mapping) or not isinstance(
                provenance, Mapping
            ):
                raise TypeError("Offline filing HTML provenance must be an object")
            html_path = filing_dir / str(html_record.get("path") or "")
            if (
                html_path.name != "source_filing.html"
                or _artifact(html_path)
                != {
                    "bytes": html_record.get("bytes"),
                    "sha256": html_record.get("sha256"),
                }
                or html_record.get("sha256") != provenance.get("html_sha256")
                or html_record.get("archive_name") != provenance.get("archive_name")
                or html_record.get("archive_sha256") != provenance.get("archive_sha256")
                or html_record.get("html_member") != provenance.get("html_member")
            ):
                raise ValueError(f"Offline filing HTML binding is invalid: {filing_id}")
            for scope_name in ("canonical_scope", "cam_augmented_scope"):
                scope_record = filing_manifest[scope_name]
                if not isinstance(scope_record, Mapping):
                    raise TypeError("Offline filing scope provenance must be an object")
                scope_path = filing_dir / str(scope_record.get("path") or "")
                if (
                    scope_path.parent != filing_dir
                    or _artifact(scope_path)
                    != {
                        "bytes": scope_record.get("bytes"),
                        "sha256": scope_record.get("sha256"),
                    }
                    or not isinstance(scope_record.get("chunk_count"), int)
                    or scope_record.get("chunk_count", 0) <= 0
                ):
                    raise ValueError(
                        f"Offline filing scope binding is invalid: {filing_id}"
                    )
            filing_bindings = filing_manifest["refusal_scope_bindings"]
            if not isinstance(filing_bindings, list) or len(filing_bindings) != 5:
                raise ValueError(f"Offline filing must bind five refusals: {filing_id}")
            filing_binding_rows.extend(filing_bindings)
        if seen_filing_ids != set(source_by_id):
            raise ValueError(
                "Offline filing membership differs from candidate provenance"
            )

        records = packet.get("records") if isinstance(packet, Mapping) else None
        sample_ids = sample.get("task_ids") if isinstance(sample, Mapping) else None
        if not isinstance(records, list) or len(records) != 200:
            raise ValueError("Review bundle must contain 200 review records")
        if not isinstance(sample_ids, list) or len(sample_ids) != 40:
            raise ValueError("Review bundle must contain the frozen 40-case sample")
        primary_ids = [str(record["task_id"]) for record in records]
        secondary_ids = [str(task_id) for task_id in sample_ids]
        if manifest["primary_task_ids_sha256"] != canonical_json_sha256(primary_ids):
            raise ValueError("Review bundle primary membership hash is invalid")
        if manifest["secondary_task_ids_sha256"] != canonical_json_sha256(
            secondary_ids
        ):
            raise ValueError("Review bundle secondary membership hash is invalid")
        _validate_template(
            _read_json(root / "primary_review.template.json"),
            role="PRIMARY",
            candidate_id=str(manifest["candidate_id"]),
            rendered_sha256=str(manifest["rendered_packet_sha256"]),
            task_ids=primary_ids,
        )
        _validate_template(
            _read_json(root / "secondary_review.template.json"),
            role="SECONDARY",
            candidate_id=str(manifest["candidate_id"]),
            rendered_sha256=str(manifest["rendered_packet_sha256"]),
            task_ids=secondary_ids,
        )

        refusal_records = {
            str(record["task_id"]): record["refusal_scope"]
            for record in records
            if record.get("refusal_scope") is not None
        }
        bindings = manifest.get("refusal_scope_bindings")
        if not isinstance(bindings, list) or len(bindings) != 100:
            raise ValueError("Review bundle must bind all 100 refusal scopes")
        if (
            sorted(
                (dict(item) for item in filing_binding_rows),
                key=lambda item: str(item["task_id"]),
            )
            != bindings
        ):
            raise ValueError(
                "Per-filing refusal bindings differ from the bundle manifest"
            )
        bound_ids: set[str] = set()
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise TypeError("Refusal scope bindings must be objects")
            task_id = str(binding.get("task_id") or "")
            if task_id in bound_ids or task_id not in refusal_records:
                raise ValueError(
                    f"Refusal scope binding identity is invalid: {task_id}"
                )
            relative = str(binding.get("scope_path") or "")
            if relative not in actual_files:
                raise ValueError(f"Refusal scope file is missing: {task_id}")
            scope_artifact = _artifact(actual_files[relative])
            refusal = refusal_records[task_id]
            if (
                binding.get("sha256") != scope_artifact["sha256"]
                or binding.get("sha256") != refusal.get("full_filing_text_sha256")
                or binding.get("chunk_count") != refusal.get("full_filing_chunk_count")
                or binding.get("scope_kind")
                not in {"CANONICAL", "CANONICAL_PLUS_RAW_CAM"}
            ):
                raise ValueError(f"Refusal scope binding is inconsistent: {task_id}")
            bound_ids.add(task_id)
        if bound_ids != set(refusal_records):
            raise ValueError("Review bundle refusal membership is incomplete")
        assert_no_secrets([root])
    except (
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        errors.append(str(exc))
    return {
        "valid": not errors,
        "errors": errors,
        "bundle_id": manifest.get("bundle_id") if "manifest" in locals() else None,
        "external_human_approval": False,
    }


__all__ = [
    "REVIEW_BUNDLE_MANIFEST_VERSION_V24",
    "REVIEW_BUNDLE_VERSION_V24",
    "prepare_agent_benchmark_review_bundle_v24",
    "verify_agent_benchmark_review_bundle_v24",
]
