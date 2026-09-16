from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

from auditops.corpus import ingest_corpus
from auditops.offline_us import (
    OFFLINE_US_IMPORT_VERSION,
    SEC_SOURCE_MANIFEST_VERSION,
    import_offline_us_cache,
)
from auditops.pipeline import connect_db
from auditops.tasks import read_jsonl

FIXTURE_HTML = Path(__file__).parent / "fixtures" / "filing_10k" / "acme-20241231.htm"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sec_paths(cik: int, accession: str, primary: str) -> tuple[str, str, str]:
    directory = f"/Archives/edgar/data/{cik}/{accession.replace('-', '')}"
    return (
        f"https://www.sec.gov{directory}",
        f"https://www.sec.gov{directory}/{primary}",
        directory,
    )


def _write_issuer(
    source_root: Path,
    *,
    ticker: str = "ACME",
    cik: int = 1,
    accession: str = "0000000001-25-000001",
    filing_date: str = "2025-02-14",
    period_end: str = "2024-12-31",
) -> Path:
    issuer = source_root / ticker
    issuer.mkdir(parents=True)
    primary = f"{ticker.lower().replace('.', '')}-20241231.htm"
    schema = f"{Path(primary).stem}.xsd"
    primary_bytes = FIXTURE_HTML.read_bytes()
    schema_bytes = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"/>\n'
    )
    (issuer / primary).write_bytes(primary_bytes)
    (issuer / schema).write_bytes(schema_bytes)

    directory_url, filing_url, directory_path = _sec_paths(cik, accession, primary)
    meta = {
        "accession": accession,
        "api_key": "hf_secret_that_must_not_be_copied",
        "cik": f"{cik:010d}",
        "company": f"{ticker} Corporation",
        "directory_url": directory_url,
        "filing_date": filing_date,
        "filing_url": filing_url,
        "is_inline_xbrl": "1",
        "local_path": "C:\\legacy\\private\\filing.htm",
        "period_end": period_end,
        "primary_document": primary,
        "schema_local_path": "C:\\legacy\\private\\filing.xsd",
        "sector": "Test sector",
        "ticker": ticker,
    }
    _write_json(issuer / "filing_meta.json", meta)
    index = {
        "directory": {
            "item": [
                {
                    "last-modified": f"{filing_date} 12:00:00",
                    "name": primary,
                    "size": str(len(primary_bytes)),
                    "type": "text.gif",
                },
                {
                    "last-modified": f"{filing_date} 12:00:00",
                    "name": schema,
                    "size": str(len(schema_bytes)),
                    "type": "text.gif",
                },
                {
                    "last-modified": f"{filing_date} 12:00:00",
                    "name": f"{accession}-xbrl.zip",
                    "size": "12345",
                    "type": "compressed.gif",
                },
            ],
            "name": directory_path,
            "parent-dir": f"/Archives/edgar/data/{cik}",
        },
        "ignored_credential": "another_secret_that_must_not_be_copied",
    }
    _write_json(issuer / "index.json", index)
    return issuer


def _update_json(path: Path, update) -> None:
    value = _read_json(path)
    update(value)
    _write_json(path, value)


def _artifact_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_import_builds_immutable_provenance_bound_ingestible_corpus(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_issuer(source)
    output = tmp_path / "corpus"

    summary = import_offline_us_cache(source, output)

    assert summary["snapshot_id"] == "us_sec_existing20_realdata_v1"
    assert summary["issuer_count"] == 1
    assert summary["filing_count"] == 1
    issuer_rows = read_jsonl(output / "manifest" / "issuer_manifest.jsonl")
    filing_rows = read_jsonl(output / "manifest" / "filing_manifest.jsonl")
    ledger_rows = read_jsonl(output / "manifest" / "download_ledger.jsonl")
    assert issuer_rows[0]["cik"] == 1
    assert filing_rows[0]["accession"] == "0000000001-25-000001"
    assert filing_rows[0]["form_type"] == "10-K"
    assert ledger_rows[0]["status"] == "existing_local"
    assert Path(ledger_rows[0]["local_path"]).is_file()
    assert "legacy" not in ledger_rows[0]["local_path"].casefold()
    assert ledger_rows[0]["sha256"] == _artifact_sha(Path(ledger_rows[0]["local_path"]))

    normalized_meta_path = (
        output / "raw" / "offline_source" / "ACME" / "filing_meta.json"
    )
    normalized_meta = _read_json(normalized_meta_path)
    assert normalized_meta["metadata_version"] == OFFLINE_US_IMPORT_VERSION
    assert normalized_meta["cik"] == 1
    assert normalized_meta["is_inline_xbrl"] is True
    assert "local_path" not in normalized_meta
    assert "schema_local_path" not in normalized_meta
    assert "api_key" not in normalized_meta

    manifest_path = output / "manifest" / "source_manifest.json"
    manifest = _read_json(manifest_path)
    assert manifest["source_manifest_version"] == SEC_SOURCE_MANIFEST_VERSION
    assert manifest["offline_import_version"] == OFFLINE_US_IMPORT_VERSION
    assert manifest["input_file_count"] == 4
    assert manifest["artifacts"]["issuer_manifest"]["rows"] == 1
    assert manifest["artifacts"]["filing_manifest"]["rows"] == 1
    assert manifest["artifacts"]["download_ledger"]["rows"] == 1
    for artifact in manifest["artifacts"].values():
        artifact_path = output / artifact["path"]
        assert artifact["bytes"] == artifact_path.stat().st_size
        assert artifact["sha256"] == _artifact_sha(artifact_path)

    persisted_text = "\n".join(
        path.read_text(encoding="utf-8") for path in output.rglob("*.json*")
    )
    assert "hf_secret_that_must_not_be_copied" not in persisted_text
    assert "another_secret_that_must_not_be_copied" not in persisted_text
    assert "C:\\legacy\\private" not in persisted_text

    ingest_summary = ingest_corpus(str(output), extract_narrative=True)
    assert ingest_summary["published"] is True
    assert ingest_summary["ingested_count"] == 1
    assert ingest_summary["error_count"] == 0
    assert (output / "logs" / "ingest_ledger.jsonl").is_file()
    connection = connect_db(str(output / "db" / "corpus.sqlite"))
    try:
        filing = connection.execute(
            "SELECT cik, accession, source_sha256, source_size_bytes FROM filings"
        ).fetchone()
    finally:
        connection.close()
    assert filing["cik"] == "1"
    assert filing["accession"] == "0000000001-25-000001"
    assert filing["source_sha256"] == ledger_rows[0]["sha256"]
    assert filing["source_size_bytes"] == ledger_rows[0]["bytes"]


def test_zip_and_path_independent_artifacts_are_deterministic(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_issuer(source)
    output_one = tmp_path / "corpus_one"
    output_two = tmp_path / "corpus_two"

    import_offline_us_cache(source, output_one)
    import_offline_us_cache(source, output_two)

    relative_paths = (
        Path("manifest/issuer_manifest.jsonl"),
        Path("manifest/filing_manifest.jsonl"),
        Path("raw/offline_source/ACME/filing_meta.json"),
        Path("raw/offline_source/ACME/index.json"),
        Path("raw/xbrl_zip/ACME/ACME_10-K_0000000001-25-000001.zip"),
    )
    for relative_path in relative_paths:
        assert (output_one / relative_path).read_bytes() == (
            output_two / relative_path
        ).read_bytes()

    zip_path = output_one / relative_paths[-1]
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["acme-20241231.htm", "acme-20241231.xsd"]
        assert archive.testzip() is None
        for member in archive.infolist():
            assert member.date_time == (1980, 1, 1, 0, 0, 0)
            assert member.compress_type == zipfile.ZIP_STORED


def test_import_accepts_valid_filing_agent_accession_prefix(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_issuer(
        source,
        cik=1,
        accession="0001628280-25-000001",
    )

    output = tmp_path / "corpus"
    import_offline_us_cache(source, output)

    filing = read_jsonl(output / "manifest" / "filing_manifest.jsonl")[0]
    assert filing["cik"] == 1
    assert filing["accession"] == "0001628280-25-000001"
    assert filing["zip_url"] == (
        "https://www.sec.gov/Archives/edgar/data/1/"
        "000162828025000001/0001628280-25-000001-xbrl.zip"
    )


def test_import_rejects_non_10k_inline_filing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    issuer = _write_issuer(source)
    primary = issuer / "acme-20241231.htm"
    payload = primary.read_bytes()
    assert b">10-K<" in payload
    primary.write_bytes(payload.replace(b">10-K<", b">10-Q<"))
    _update_json(
        issuer / "index.json",
        lambda index: index["directory"]["item"][0].update(
            {"size": str(primary.stat().st_size)}
        ),
    )

    with pytest.raises(ValueError, match="dei:DocumentType of 10-K"):
        import_offline_us_cache(source, tmp_path / "corpus")


def test_import_refuses_overwrite_without_mutating_existing_destination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_issuer(source)
    output = tmp_path / "corpus"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("owned by user", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        import_offline_us_cache(source, output)

    assert sentinel.read_text(encoding="utf-8") == "owned by user"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda meta: meta.update({"cik": "not-a-cik"}),
            "cik must be",
        ),
        (
            lambda meta: meta.update({"accession": "../bad"}),
            "Malformed SEC accession",
        ),
        (
            lambda meta: meta.update({"filing_url": "https://evil.example/filing.htm"}),
            "filing_url does not match",
        ),
        (
            lambda meta: meta.update({"primary_document": "../filing.htm"}),
            "primary_document must be a safe basename",
        ),
        (
            lambda meta: meta.update({"period_end": "2025-03-01"}),
            "period_end for ACME cannot be after filing_date",
        ),
    ],
)
def test_import_rejects_malformed_metadata_atomically(tmp_path, mutation, message):
    source = tmp_path / "source"
    source.mkdir()
    issuer = _write_issuer(source)
    _update_json(issuer / "filing_meta.json", mutation)
    output = tmp_path / "corpus"

    with pytest.raises(ValueError, match=message):
        import_offline_us_cache(source, output)

    assert not output.exists()
    assert not (tmp_path / ".corpus.building").exists()


def test_import_rejects_filing_after_snapshot_date(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_issuer(source)

    with pytest.raises(ValueError, match="after snapshot_date"):
        import_offline_us_cache(
            source,
            tmp_path / "corpus",
            snapshot_date="2025-01-01",
        )


@pytest.mark.parametrize("index_failure", ["directory", "primary_size", "missing_zip"])
def test_import_rejects_index_metadata_mismatch(tmp_path, index_failure):
    source = tmp_path / "source"
    source.mkdir()
    issuer = _write_issuer(source)

    def mutate(index):
        directory = index["directory"]
        if index_failure == "directory":
            directory["name"] = "/Archives/edgar/data/1/wrong"
        elif index_failure == "primary_size":
            directory["item"][0]["size"] = "1"
        else:
            directory["item"] = [
                item
                for item in directory["item"]
                if not item["name"].endswith("-xbrl.zip")
            ]

    _update_json(issuer / "index.json", mutate)

    with pytest.raises(ValueError, match="index.json"):
        import_offline_us_cache(source, tmp_path / "corpus")


def test_import_rejects_missing_schema_and_unexpected_nested_content(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    issuer = _write_issuer(source)
    (issuer / "acme-20241231.xsd").unlink()

    with pytest.raises(ValueError, match="exactly one XSD"):
        import_offline_us_cache(source, tmp_path / "missing_schema")

    (issuer / "acme-20241231.xsd").write_text(
        '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"/>',
        encoding="utf-8",
    )
    nested = issuer / "unexpected"
    nested.mkdir()
    with pytest.raises(ValueError, match="only regular, non-symlink files"):
        import_offline_us_cache(source, tmp_path / "nested_content")


def test_import_rejects_duplicate_cik_identity(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_issuer(source)
    _write_issuer(
        source,
        ticker="BETA",
        cik=1,
        accession="0000000002-25-000002",
    )

    with pytest.raises(ValueError, match="Duplicate CIK identity"):
        import_offline_us_cache(source, tmp_path / "corpus")


def test_import_rejects_symlinked_source_member(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    issuer = _write_issuer(source)
    schema = issuer / "acme-20241231.xsd"
    external = tmp_path / "external.xsd"
    external.write_bytes(schema.read_bytes())
    schema.unlink()
    try:
        os.symlink(external, schema)
    except OSError as error:  # pragma: no cover - host policy dependent
        pytest.skip(f"Host does not permit test symlink creation: {error}")

    with pytest.raises(ValueError, match="non-symlink"):
        import_offline_us_cache(source, tmp_path / "corpus")


def test_import_rejects_duplicate_json_keys(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    issuer = _write_issuer(source)
    meta_path = issuer / "filing_meta.json"
    raw = meta_path.read_text(encoding="utf-8").rstrip()
    assert raw.endswith("}")
    meta_path.write_text(raw[:-1] + ', "ticker": "ACME"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate key"):
        import_offline_us_cache(source, tmp_path / "corpus")
