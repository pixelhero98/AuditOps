from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import jsonschema
import pytest

from auditops.agent_provisional_v24p import validate_exploratory_authorization_v24p
from auditops.canonical_json import canonical_json_sha256
from scripts.check_findings import verify

ROOT = Path(__file__).resolve().parents[1]


def authorization():
    limitations = {
        "assurance": "PROVISIONAL_AI_REVIEW",
        "experiment_label": "EXPLORATORY",
        "generalization_label": "cached-20",
        "external_human_approval": False,
        "official_baseline": False,
    }
    material = {
        "authorization_version": "auditops-exploratory-run-authorization.v2.4p.1",
        "benchmark_id": "a" * 64,
        "gate_metrics_sha256": "b" * 64,
        "decision": "GO_EXPLORATORY",
        "authorized_by": "review-owner",
        "authorized_at": "2026-09-16T12:00:00Z",
        "assurance": "PROVISIONAL_AI_REVIEW",
        "experiment_label": "EXPLORATORY",
        "known_limitations": limitations,
        "known_limitations_sha256": canonical_json_sha256(limitations),
    }
    return {**material, "authorization_id": canonical_json_sha256(material)}


def test_exploratory_authorizer_is_explicit_bound_and_schema_valid():
    value = authorization()
    schema = json.loads(
        (
            ROOT / "auditops/specs/exploratory_run_authorization.v2.4p.1.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(value)
    assert (
        validate_exploratory_authorization_v24p(
            value, benchmark_id="a" * 64, expected_authorizer="review-owner"
        )
        == value
    )
    for owner in (None, "", "other-owner", "review\nowner", "review\rowner"):
        with pytest.raises(ValueError):
            validate_exploratory_authorization_v24p(
                value, benchmark_id="a" * 64, expected_authorizer=owner
            )
    for field, replacement in (
        ("authorized_by", "other-owner"),
        ("benchmark_id", "c" * 64),
        ("gate_metrics_sha256", "d" * 64),
    ):
        altered = copy.deepcopy(value)
        altered[field] = replacement
        with pytest.raises(ValueError):
            validate_exploratory_authorization_v24p(
                altered, benchmark_id="a" * 64, expected_authorizer="review-owner"
            )
    old = copy.deepcopy(value)
    old["authorization_version"] = "auditops-exploratory-run-authorization.v2.4p"
    old["authorization_id"] = canonical_json_sha256(
        {k: v for k, v in old.items() if k != "authorization_id"}
    )
    with pytest.raises(ValueError):
        validate_exploratory_authorization_v24p(
            old, benchmark_id="a" * 64, expected_authorizer="review-owner"
        )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(old)


def test_published_findings_are_consistent():
    verify(ROOT)


@pytest.mark.parametrize(
    "mutation", ["extra", "missing_label", "wrong_label", "timestamp", "non_digest"]
)
def test_rehashed_authorizations_still_enforce_the_public_contract(mutation):
    value = authorization()
    expected_benchmark = value["benchmark_id"]
    if mutation == "extra":
        value["unexpected"] = "not-permitted"
    elif mutation == "missing_label":
        del value["known_limitations"]["generalization_label"]
    elif mutation == "wrong_label":
        value["known_limitations"]["assurance"] = "HUMAN_APPROVED"
    elif mutation == "timestamp":
        value["authorized_at"] = "2026-09-16 12:00:00+00:00"
    else:
        value["benchmark_id"] = expected_benchmark = "not-a-digest"
    value["known_limitations_sha256"] = canonical_json_sha256(
        value["known_limitations"]
    )
    value["authorization_id"] = canonical_json_sha256(
        {key: item for key, item in value.items() if key != "authorization_id"}
    )
    schema = json.loads(
        (
            ROOT / "auditops/specs/exploratory_run_authorization.v2.4p.1.schema.json"
        ).read_text()
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        ).validate(value)
    with pytest.raises(ValueError):
        validate_exploratory_authorization_v24p(
            value, benchmark_id=expected_benchmark, expected_authorizer="review-owner"
        )


def test_model_and_container_pins_are_preserved():
    models = json.loads((ROOT / "config/model_revisions.json").read_text())["models"]
    assert len(models) == 4
    assert {model["model_id"].split("/")[0] for model in models} == {"Qwen", "google"}
    assert all(re.fullmatch(r"[0-9a-f]{40}", model["revision"]) for model in models)
    container = json.loads((ROOT / "config/vllm_0.26.0.json").read_text())
    assert container["version"] == "0.26.0"
    for architecture in ("amd64", "arm64"):
        assert re.fullmatch(
            r"sha256:[0-9a-f]{64}", container[f"{architecture}_manifest_digest"]
        )


def test_portable_launchers_preserve_isolation_and_hardware_binding():
    for name in ("run_agent_baseline", "run_companies_house_vlm_diagnostic"):
        text = (ROOT / f"scripts/slurm/{name}.sbatch").read_text()
        assert "AUDITOPS_EXPECTED_GPU" in text
        assert "--network none" in text
        assert "verify_model_snapshot" in text
        assert "AUDITOPS_SOURCE_MANIFEST_SHA256" in text
        assert "socket.if_nameindex()" in text
        assert "socket.AF_INET6" in text
        assert "config/model_revisions.json" in text


def test_code_targets_are_valid_constrained_python(db_conn):
    from auditops.tasks import _build_code_target, build_task_specs

    for task in build_task_specs(db_conn):
        code = _build_code_target(task)
        namespace = {"conn": db_conn}
        exec(compile(code, "<synthetic-code-target>", "exec"), namespace)
        assert namespace["structured_answer"] == task["target_answer"]
