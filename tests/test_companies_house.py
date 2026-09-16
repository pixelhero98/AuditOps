from __future__ import annotations

import csv
import hashlib
import json
import stat
import zipfile
from decimal import Decimal
from pathlib import Path

import pytest

from auditops.agent_benchmark import _build_case, _build_evidence_items
from auditops.agent_context import build_context_pack
from auditops.agent_tools import execute_deterministic_tool
from auditops.companies_house import (
    CompaniesHouseArchiveValidationError,
    CompaniesHouseZipLimits,
    build_companies_house_corpus,
    build_companies_house_narrative_tasks,
    build_companies_house_quant_tasks,
    build_companies_house_vlm_diagnostic,
    iter_companies_house_bulk,
    parse_companies_house_xhtml,
)

UK_XHTML = b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"
      xmlns:xbrli="http://www.xbrl.org/2003/instance"
      xmlns:uk-core="http://xbrl.frc.org.uk/fr/2022-01-01/core">
  <head><link rel="schemaRef" href="https://xbrl.frc.org.uk/fr/2022-01-01/core.xsd" /></head>
  <body>
    <xbrli:context id="c1"><xbrli:entity><xbrli:identifier scheme="CH">01234567</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:instant>2025-12-31</xbrli:instant></xbrli:period></xbrli:context>
    <xbrli:unit id="GBP"><xbrli:measure>iso4217:GBP</xbrli:measure></xbrli:unit>
    <p><ix:nonNumeric name="uk-core:UKCompaniesHouseRegisteredNumber" contextRef="c1">01234567</ix:nonNumeric></p>
    <p><ix:nonNumeric name="uk-core:EntityCurrentLegalOrRegisteredName" contextRef="c1">EXAMPLE LIMITED</ix:nonNumeric></p>
    <p><ix:nonNumeric name="uk-core:BalanceSheetDate" contextRef="c1">2025-12-31</ix:nonNumeric></p>
    <p>These micro-entity accounts use FRS 105. For the year the company was entitled to exemption from audit under section 477.</p>
    <p>Current assets <ix:nonFraction name="uk-core:CurrentAssets" contextRef="c1" unitRef="GBP">1,200</ix:nonFraction></p>
    <p>Creditors falling due within one year <ix:nonFraction name="uk-core:CreditorsAmountsFallingDueWithinOneYear" contextRef="c1" unitRef="GBP" sign="-">500</ix:nonFraction></p>
  </body>
</html>"""


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _flip_stored_member_byte(archive_path: Path, member_name: str) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member_name)
        assert info.compress_type == zipfile.ZIP_STORED
        header_offset = info.header_offset
    payload = bytearray(archive_path.read_bytes())
    name_length = int.from_bytes(
        payload[header_offset + 26 : header_offset + 28], "little"
    )
    extra_length = int.from_bytes(
        payload[header_offset + 28 : header_offset + 30], "little"
    )
    data_offset = header_offset + 30 + name_length + extra_length
    payload[data_offset + info.file_size // 2] ^= 0x01
    archive_path.write_bytes(payload)


def _mark_member_encrypted(archive_path: Path, member_name: str) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        header_offset = archive.getinfo(member_name).header_offset
    payload = bytearray(archive_path.read_bytes())
    local_flags = int.from_bytes(
        payload[header_offset + 6 : header_offset + 8], "little"
    )
    payload[header_offset + 6 : header_offset + 8] = (local_flags | 1).to_bytes(
        2, "little"
    )

    cursor = 0
    central_patched = False
    while True:
        cursor = payload.find(b"PK\x01\x02", cursor)
        if cursor < 0:
            break
        central_header_offset = int.from_bytes(
            payload[cursor + 42 : cursor + 46], "little"
        )
        if central_header_offset == header_offset:
            central_flags = int.from_bytes(payload[cursor + 8 : cursor + 10], "little")
            payload[cursor + 8 : cursor + 10] = (central_flags | 1).to_bytes(
                2, "little"
            )
            central_patched = True
            break
        cursor += 4
    assert central_patched
    archive_path.write_bytes(payload)


def _execute_benchmarked_quant_task(source_task: dict) -> dict:
    evidence = _build_evidence_items(source_task, "quant_metric", "UK_COMPANIES_HOUSE")
    task = _build_case(
        source_task,
        "quant_metric",
        jurisdiction="UK",
        corpus_id="companies_house_fixture",
        source_system="UK_COMPANIES_HOUSE",
        reporting_framework=source_task["reporting_framework"],
        standards_version=source_task["standards_version"],
        evidence_items=evidence,
    )
    build_context_pack(task, evidence, token_counter=lambda _text: 1)
    return execute_deterministic_tool(
        "evaluate_metric_spec",
        {
            "filing_id": task["filing"]["filing_id"],
            "metric_spec_id": task["task_parameters"]["metric_spec_id"],
            "period_key": task["period"]["period_key"],
        },
        task,
        evidence,
    )


def test_parse_companies_house_xhtml_and_signed_creditors():
    filing = parse_companies_house_xhtml(UK_XHTML, source_name="sample.xhtml")
    assert filing.company_number == "01234567"
    assert filing.company_name == "EXAMPLE LIMITED"
    assert filing.audit_status == "AUDIT_EXEMPT"
    assert filing.reporting_framework == "FRS_105"
    assert filing.account_type == "micro_entity"

    tasks = build_companies_house_quant_tasks(filing)
    current_ratio = next(
        task for task in tasks if task["metric_spec_id"] == "current_ratio"
    )
    working_capital = next(
        task for task in tasks if task["metric_spec_id"] == "working_capital"
    )
    assert current_ratio["target_answer"]["status"] == "OK"
    assert current_ratio["target_answer"]["value"] == "2.4"
    assert working_capital["target_answer"]["value"] == "700"


def test_dash_numeric_produces_refusal():
    filing = parse_companies_house_xhtml(
        UK_XHTML.replace(b">1,200<", b">-<"), source_name="dash.xhtml"
    )
    tasks = build_companies_house_quant_tasks(filing)
    assert all(task["target_answer"]["status"] == "REFUSAL" for task in tasks)
    assert {
        _execute_benchmarked_quant_task(task)["refusal_code"] for task in tasks
    } == {"MISSING_INPUT"}


def test_display_period_end_selects_exact_current_iso_context():
    payload = UK_XHTML.replace(
        b'<xbrli:context id="c1">',
        b'<xbrli:context id="prior"><xbrli:entity><xbrli:identifier scheme="CH">01234567</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:instant>2024-12-31</xbrli:instant></xbrli:period></xbrli:context>'
        b'<xbrli:context id="c1">',
    ).replace(
        b'<p><ix:nonNumeric name="uk-core:BalanceSheetDate" contextRef="c1">2025-12-31</ix:nonNumeric></p>',
        b'<p><ix:nonNumeric name="uk-core:BalanceSheetDate" contextRef="c1">31 December 2025</ix:nonNumeric></p>'
        b'<p><ix:nonFraction name="uk-core:CurrentAssets" contextRef="prior" unitRef="GBP">9,999</ix:nonFraction></p>'
        b'<p><ix:nonFraction name="uk-core:CreditorsAmountsFallingDueWithinOneYear" contextRef="prior" unitRef="GBP">1</ix:nonFraction></p>',
    )
    filing = parse_companies_house_xhtml(payload, source_name="display-date.xhtml")
    assert filing.period_end == "2025-12-31"
    tasks = build_companies_house_quant_tasks(filing)
    ratio = next(task for task in tasks if task["metric_spec_id"] == "current_ratio")
    assert ratio["period"]["period_end"] == "2025-12-31"
    assert ratio["target_answer"]["value"] == "2.4"


def test_short_uk_display_period_end_normalizes_to_iso():
    payload = UK_XHTML.replace(
        b">2025-12-31</ix:nonNumeric>", b">31.12.25</ix:nonNumeric>"
    )
    filing = parse_companies_house_xhtml(payload, source_name="short-date.xhtml")
    assert filing.period_end == "2025-12-31"
    assert all(
        task["target_answer"]["status"] == "OK"
        for task in build_companies_house_quant_tasks(filing)
    )


def test_transforms_continuations_and_actual_currency_are_preserved():
    payload = (
        UK_XHTML.replace(
            b'xmlns:uk-core="http://xbrl.frc.org.uk/fr/2022-01-01/core"',
            b'xmlns:uk-core="http://xbrl.frc.org.uk/fr/2022-01-01/core" xmlns:ixt="http://www.xbrl.org/inlineXBRL/transformation/2020-02-12"',
        )
        .replace(b'id="GBP"', b'id="EUR"')
        .replace(b"iso4217:GBP", b"iso4217:EUR")
        .replace(b'unitRef="GBP"', b'unitRef="EUR"')
        .replace(
            b">EXAMPLE LIMITED</ix:nonNumeric>",
            b' continuedAt="name-rest">EXAMPLE</ix:nonNumeric><ix:continuation id="name-rest"> LIMITED</ix:continuation>',
        )
        .replace(
            b'unitRef="EUR">1,200</ix:nonFraction>',
            b'unitRef="EUR" format="ixt:numcommadecimal">1.234,50</ix:nonFraction>',
        )
    )
    filing = parse_companies_house_xhtml(payload, source_name="transformed.xhtml")
    assert filing.company_name == "EXAMPLE LIMITED"
    tasks = build_companies_house_quant_tasks(filing)
    ratio = next(task for task in tasks if task["metric_spec_id"] == "current_ratio")
    working_capital = next(
        task for task in tasks if task["metric_spec_id"] == "working_capital"
    )
    assert ratio["target_answer"]["value"] == "2.469"
    assert working_capital["target_answer"]["value"] == "734.5"
    assert working_capital["target_answer"]["unit"] == "eur"
    assert {item["unit"] for item in working_capital["evidence_items"]} == {"eur"}


def test_eur_companies_house_task_survives_benchmark_and_verifier_layers():
    payload = (
        UK_XHTML.replace(
            b'xmlns:uk-core="http://xbrl.frc.org.uk/fr/2022-01-01/core"',
            b'xmlns:uk-core="http://xbrl.frc.org.uk/fr/2022-01-01/core" xmlns:ixt="http://www.xbrl.org/inlineXBRL/transformation/2020-02-12"',
        )
        .replace(b'id="GBP"', b'id="EUR"')
        .replace(b"iso4217:GBP", b"iso4217:EUR")
        .replace(b'unitRef="GBP"', b'unitRef="EUR"')
        .replace(
            b'unitRef="EUR">1,200</ix:nonFraction>',
            b'unitRef="EUR" format="ixt:numcommadecimal">1.234,50</ix:nonFraction>',
        )
    )
    filing = parse_companies_house_xhtml(payload, source_name="eur-integration.xhtml")
    source_task = next(
        task
        for task in build_companies_house_quant_tasks(filing)
        if task["metric_spec_id"] == "working_capital"
    )
    observation = _execute_benchmarked_quant_task(source_task)
    target = source_task["target_answer"]
    assert observation["status"] == target["status"] == "OK"
    assert observation["value"] == target["value"] == "734.5"
    assert observation["unit"] == target["unit"] == "eur"
    assert observation["evidence_ids"] == target["evidence_ids"]


def test_numeric_continuation_chain_is_joined_before_parsing():
    payload = UK_XHTML.replace(
        b'unitRef="GBP">1,200</ix:nonFraction>',
        b'unitRef="GBP" continuedAt="assets-rest">1,</ix:nonFraction><ix:continuation id="assets-rest">200</ix:continuation>',
    )
    tasks = build_companies_house_quant_tasks(
        parse_companies_house_xhtml(payload, source_name="continued-number.xhtml")
    )
    ratio = next(task for task in tasks if task["metric_spec_id"] == "current_ratio")
    assert ratio["target_answer"]["status"] == "OK"
    assert ratio["target_answer"]["value"] == "2.4"


@pytest.mark.parametrize(
    "replacement",
    [
        b'unitRef="GBP" continuedAt="missing-continuation">1,200</ix:nonFraction>',
        b'unitRef="GBP" format="ixt:unsupported-number">1,200</ix:nonFraction>',
    ],
)
def test_broken_continuation_and_unknown_transform_refuse(replacement):
    payload = UK_XHTML.replace(
        b'xmlns:uk-core="http://xbrl.frc.org.uk/fr/2022-01-01/core"',
        b'xmlns:uk-core="http://xbrl.frc.org.uk/fr/2022-01-01/core" xmlns:ixt="http://www.xbrl.org/inlineXBRL/transformation/2020-02-12"',
    ).replace(b'unitRef="GBP">1,200</ix:nonFraction>', replacement)
    tasks = build_companies_house_quant_tasks(
        parse_companies_house_xhtml(payload, source_name="invalid-number.xhtml")
    )
    assert {task["target_answer"]["refusal_code"] for task in tasks} == {
        "MISSING_INPUT"
    }


@pytest.mark.parametrize(
    ("mutation", "expected_refusal"),
    [
        (
            (
                b'name="uk-core:CurrentAssets" contextRef="c1" unitRef="GBP"',
                b'name="uk-core:CurrentAssets" contextRef="c1" unitRef="GBP" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:nil="true"',
            ),
            "MISSING_INPUT",
        ),
        (
            (
                b'<xbrli:unit id="GBP"><xbrli:measure>iso4217:GBP</xbrli:measure></xbrli:unit>',
                b'<xbrli:unit id="GBP"><xbrli:measure>iso4217:GBP</xbrli:measure></xbrli:unit><xbrli:unit id="USD"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>',
            ),
            "INCOMPATIBLE_UNITS",
        ),
    ],
)
def test_nil_and_incompatible_units_refuse_safely(mutation, expected_refusal):
    payload = UK_XHTML.replace(*mutation)
    if expected_refusal == "INCOMPATIBLE_UNITS":
        payload = payload.replace(
            b'name="uk-core:CurrentAssets" contextRef="c1" unitRef="GBP"',
            b'name="uk-core:CurrentAssets" contextRef="c1" unitRef="USD"',
        )
    tasks = build_companies_house_quant_tasks(
        parse_companies_house_xhtml(payload, source_name="unsafe-input.xhtml")
    )
    assert {task["target_answer"]["refusal_code"] for task in tasks} == {
        expected_refusal
    }
    assert {
        _execute_benchmarked_quant_task(task)["refusal_code"] for task in tasks
    } == {expected_refusal}


def test_dimensioned_current_fact_does_not_fall_back_to_prior_default_context():
    payload = (
        UK_XHTML.replace(
            b'xmlns:uk-core="http://xbrl.frc.org.uk/fr/2022-01-01/core"',
            b'xmlns:uk-core="http://xbrl.frc.org.uk/fr/2022-01-01/core" xmlns:xbrldi="http://xbrl.org/2006/xbrldi"',
        )
        .replace(
            b'<xbrli:context id="c1"><xbrli:entity><xbrli:identifier scheme="CH">01234567</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:instant>2025-12-31</xbrli:instant></xbrli:period></xbrli:context>',
            b'<xbrli:context id="prior"><xbrli:entity><xbrli:identifier scheme="CH">01234567</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:instant>2024-12-31</xbrli:instant></xbrli:period></xbrli:context>'
            b'<xbrli:context id="c1"><xbrli:entity><xbrli:identifier scheme="CH">01234567</xbrli:identifier><xbrli:segment><xbrldi:explicitMember dimension="uk-core:EntityOrGroupDimension">uk-core:Group</xbrldi:explicitMember></xbrli:segment></xbrli:entity><xbrli:period><xbrli:instant>2025-12-31</xbrli:instant></xbrli:period></xbrli:context>',
        )
        .replace(
            b"<p>Current assets ",
            b'<p><ix:nonFraction name="uk-core:CurrentAssets" contextRef="prior" unitRef="GBP">9,999</ix:nonFraction></p><p>Current assets ',
        )
    )
    filing = parse_companies_house_xhtml(payload, source_name="dimensioned.xhtml")
    current_assets = next(
        fact
        for fact in filing.facts
        if fact["concept_local"] == "currentassets" and fact["context_id"] == "c1"
    )
    assert current_assets["entity_identifier"] == "01234567"
    assert current_assets["dimensions"] == {
        "uk-core:EntityOrGroupDimension": {
            "kind": "explicit",
            "member": "uk-core:Group",
        }
    }
    tasks = build_companies_house_quant_tasks(filing)
    assert {task["target_answer"]["refusal_code"] for task in tasks} == {
        "AMBIGUOUS_CONTEXT"
    }
    assert {
        _execute_benchmarked_quant_task(task)["refusal_code"] for task in tasks
    } == {"AMBIGUOUS_CONTEXT"}


def test_conflicting_duplicate_current_facts_refuse_ambiguity():
    payload = UK_XHTML.replace(
        b"<p>Current assets ",
        b'<p><ix:nonFraction name="uk-core:CurrentAssets" contextRef="c1" unitRef="GBP">1,201</ix:nonFraction></p><p>Current assets ',
    )
    tasks = build_companies_house_quant_tasks(
        parse_companies_house_xhtml(payload, source_name="ambiguous.xhtml")
    )
    assert {task["target_answer"]["refusal_code"] for task in tasks} == {
        "AMBIGUOUS_CONTEXT"
    }
    assert {
        _execute_benchmarked_quant_task(task)["refusal_code"] for task in tasks
    } == {"AMBIGUOUS_CONTEXT"}


@pytest.mark.parametrize(
    (
        "dimension",
        "member",
        "expected_status",
        "expected_value",
        "expected_refusal",
    ),
    [
        (
            "uk-core:MaturitiesOrExpirationPeriodsDimension",
            "uk-core:WithinOneYear",
            "OK",
            "2.4",
            None,
        ),
        (
            "uk-core:FinancialInstrumentCurrentNon-currentDimension",
            "uk-core:CurrentFinancialInstruments",
            "OK",
            "2.4",
            None,
        ),
        (
            "uk-core:MaturitiesOrExpirationPeriodsDimension",
            "uk-core:AfterOneYear",
            "REFUSAL",
            None,
            "MISSING_INPUT",
        ),
        (
            "uk-core:EntityOrGroupDimension",
            "uk-core:Group",
            "REFUSAL",
            None,
            "AMBIGUOUS_CONTEXT",
        ),
    ],
)
def test_generic_creditors_require_allowlisted_current_dimension(
    dimension, member, expected_status, expected_value, expected_refusal
):
    context = (
        '<xbrli:context id="creditors-current"><xbrli:entity><xbrli:identifier scheme="CH">01234567</xbrli:identifier>'
        f'<xbrli:segment><xbrldi:explicitMember dimension="{dimension}">{member}</xbrldi:explicitMember></xbrli:segment>'
        "</xbrli:entity><xbrli:period><xbrli:instant>2025-12-31</xbrli:instant></xbrli:period></xbrli:context>"
    ).encode()
    payload = (
        UK_XHTML.replace(
            b'xmlns:uk-core="http://xbrl.frc.org.uk/fr/2022-01-01/core"',
            b'xmlns:uk-core="http://xbrl.frc.org.uk/fr/2022-01-01/core" xmlns:xbrldi="http://xbrl.org/2006/xbrldi"',
        )
        .replace(b'<xbrli:unit id="GBP">', context + b'<xbrli:unit id="GBP">')
        .replace(
            b'name="uk-core:CreditorsAmountsFallingDueWithinOneYear" contextRef="c1"',
            b'name="uk-core:Creditors" contextRef="creditors-current"',
        )
    )
    tasks = build_companies_house_quant_tasks(
        parse_companies_house_xhtml(payload, source_name="generic-creditors.xhtml")
    )
    ratio = next(task for task in tasks if task["metric_spec_id"] == "current_ratio")
    assert ratio["target_answer"]["status"] == expected_status
    assert ratio["target_answer"]["value"] == expected_value
    if expected_status == "OK":
        assets = next(
            item
            for item in ratio["evidence_items"]
            if item["metadata"]["input_name"] == "assets_current"
        )
        creditors = next(
            item
            for item in ratio["evidence_items"]
            if item["metadata"]["input_name"] == "liabilities_current"
        )
        assert "scenario" not in assets["metadata"]
        assert creditors["metadata"]["scenario"] == "CURRENT_CREDITOR_ALLOWLIST"
        assert creditors["metadata"]["entity_id"] == "01234567"
        assert creditors["metadata"]["dimensions"]
        observation = _execute_benchmarked_quant_task(ratio)
        assert observation["status"] == "OK"
        assert observation["value"] == expected_value
        assert observation["unit"] == "pure"
    else:
        assert ratio["target_answer"]["refusal_code"] == expected_refusal


@pytest.mark.parametrize(
    ("context", "expected_refusal"),
    [
        (
            b'<xbrli:context id="c1"><xbrli:entity><xbrli:identifier scheme="CH">99999999</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:instant>2025-12-31</xbrli:instant></xbrli:period></xbrli:context>',
            "AMBIGUOUS_CONTEXT",
        ),
        (
            b'<xbrli:context id="c1"><xbrli:entity><xbrli:identifier scheme="CH">01234567</xbrli:identifier></xbrli:entity><xbrli:scenario><uk-core:Restated>true</uk-core:Restated></xbrli:scenario><xbrli:period><xbrli:instant>2025-12-31</xbrli:instant></xbrli:period></xbrli:context>',
            "AMBIGUOUS_CONTEXT",
        ),
        (
            b'<xbrli:context id="c1"><xbrli:entity><xbrli:identifier scheme="CH">01234567</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate></xbrli:period></xbrli:context>',
            "PERIOD_NOT_SUPPORTED",
        ),
    ],
)
def test_wrong_entity_scenario_and_duration_contexts_refuse(context, expected_refusal):
    original = b'<xbrli:context id="c1"><xbrli:entity><xbrli:identifier scheme="CH">01234567</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:instant>2025-12-31</xbrli:instant></xbrli:period></xbrli:context>'
    filing = parse_companies_house_xhtml(
        UK_XHTML.replace(original, context), source_name="wrong-context.xhtml"
    )
    tasks = build_companies_house_quant_tasks(filing)
    assert {task["target_answer"]["refusal_code"] for task in tasks} == {
        expected_refusal
    }
    assert {
        _execute_benchmarked_quant_task(task)["refusal_code"] for task in tasks
    } == {expected_refusal}


def test_zero_dash_transform_is_zero_not_missing():
    payload = UK_XHTML.replace(
        b'xmlns:uk-core="http://xbrl.frc.org.uk/fr/2022-01-01/core"',
        b'xmlns:uk-core="http://xbrl.frc.org.uk/fr/2022-01-01/core" xmlns:ixt="http://www.xbrl.org/inlineXBRL/transformation/2020-02-12"',
    ).replace(
        b'unitRef="GBP" sign="-">500</ix:nonFraction>',
        'unitRef="GBP" sign="-" format="ixt:zerodash">—</ix:nonFraction>'.encode(),
    )
    tasks = build_companies_house_quant_tasks(
        parse_companies_house_xhtml(payload, source_name="zero-dash.xhtml")
    )
    ratio = next(task for task in tasks if task["metric_spec_id"] == "current_ratio")
    working_capital = next(
        task for task in tasks if task["metric_spec_id"] == "working_capital"
    )
    assert ratio["target_answer"]["refusal_code"] == "DIVISION_BY_ZERO"
    assert working_capital["target_answer"]["value"] == "1200"


def test_narrative_tasks_use_bm25_and_truthful_opinion_and_kam_evidence():
    payload = UK_XHTML.replace(
        b"<p>These micro-entity accounts use FRS 105. For the year the company was entitled to exemption from audit under section 477.</p>",
        "<p>Independent auditor’s report. In our opinion the financial statements give a true and fair view. The key audit matter was revenue recognition.</p>".encode(),
    )
    filing = parse_companies_house_xhtml(payload, source_name="audited.xhtml")
    assert filing.audit_status == "AUDITED"
    tasks = build_companies_house_narrative_tasks(filing)
    assert {task["label"] for task in tasks} == {
        "audit_status",
        "audit_opinion_language",
        "key_audit_matter",
    }
    assert all(task["answerability"] == "ANSWERABLE" for task in tasks)
    for task in tasks:
        assert task["citation_policy"] == {
            "require_chunk_evidence_ids": True,
            "retrieval_method": "bm25",
            "top_k": 5,
            "candidate_k": 15,
        }
        assert len(task["evidence_items"]) <= 5
        visible_ids = {item["evidence_id"] for item in task["evidence_items"]}
        assert set(task["expected_chunk_ids"]) <= visible_ids
        assert task["extractive_answer"] in {
            item["content"] for item in task["evidence_items"]
        }


def test_unsupported_narratives_use_releasable_refusal_code():
    payload = UK_XHTML.replace(
        b"</body>",
        b"<p>The directors prepared financial statements which give a true and fair view.</p></body>",
    )
    filing = parse_companies_house_xhtml(payload, source_name="unaudited.xhtml")
    tasks = build_companies_house_narrative_tasks(filing)
    opinion = next(task for task in tasks if task["label"] == "audit_opinion_language")
    assert opinion["answerability"] == "UNANSWERABLE"
    unsupported = [task for task in tasks if task["answerability"] == "UNANSWERABLE"]
    assert unsupported
    assert {task["refusal_code"] for task in unsupported} == {"NARRATIVE_NOT_SUPPORTED"}


def test_build_companies_house_corpus_is_immutable(tmp_path):
    archive = tmp_path / "bulk.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Accounts/one.xhtml", UK_XHTML)
    output = tmp_path / "corpus"
    manifest = build_companies_house_corpus(archive, output, max_filings=1)
    assert manifest["filing_count"] == 1
    assert manifest["quant_task_count"] == 2
    assert manifest["narrative_task_count"] == 3
    assert manifest["selected_member_count"] == 1
    assert manifest["source_manifest_count"] == 1
    assert manifest["parse_failure_count"] == 0
    assert len(_read_jsonl(output / "source_manifest.jsonl")) == 1
    assert (output / "parse_failures.jsonl").read_text(encoding="utf-8") == ""
    with pytest.raises(FileExistsError):
        build_companies_house_corpus(archive, output, max_filings=1)


def test_bulk_archive_member_count_cap_is_enforced_before_reads(tmp_path):
    archive = tmp_path / "too-many.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Accounts/one.xhtml", UK_XHTML)
        handle.writestr("Accounts/two.xhtml", UK_XHTML)

    with pytest.raises(CompaniesHouseArchiveValidationError) as exc_info:
        list(
            iter_companies_house_bulk(
                archive,
                zip_limits=CompaniesHouseZipLimits(max_archive_members=1),
            )
        )
    assert exc_info.value.code == "ARCHIVE_MEMBER_COUNT_LIMIT_EXCEEDED"


def test_bulk_archive_total_uncompressed_size_cap_is_enforced_before_reads(tmp_path):
    archive = tmp_path / "too-large-total.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Accounts/one.xhtml", UK_XHTML)

    with pytest.raises(CompaniesHouseArchiveValidationError) as exc_info:
        list(
            iter_companies_house_bulk(
                archive,
                zip_limits=CompaniesHouseZipLimits(
                    max_total_uncompressed_bytes=len(UK_XHTML) - 1
                ),
            )
        )
    assert exc_info.value.code == "ARCHIVE_UNCOMPRESSED_SIZE_LIMIT_EXCEEDED"


def test_oversized_member_is_ledgered_and_bound_to_source_manifest(tmp_path):
    archive = tmp_path / "oversized.zip"
    oversized_payload = UK_XHTML + b" " * len(UK_XHTML)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr("Accounts/good.xhtml", UK_XHTML)
        handle.writestr("Accounts/oversized.xhtml", oversized_payload)
    output = tmp_path / "oversized-corpus"
    manifest = build_companies_house_corpus(
        archive,
        output,
        zip_limits=CompaniesHouseZipLimits(
            max_member_uncompressed_bytes=len(UK_XHTML) + 1
        ),
    )

    failures = _read_jsonl(output / "parse_failures.jsonl")
    sources = _read_jsonl(output / "source_manifest.jsonl")
    assert manifest["selected_member_count"] == 2
    assert manifest["source_manifest_count"] == 2
    assert manifest["filing_count"] == 1
    assert manifest["parse_failure_count"] == 1
    assert manifest["parse_failure_counts"] == {"MEMBER_SIZE_LIMIT_EXCEEDED": 1}
    assert failures[0]["failure_code"] == "MEMBER_SIZE_LIMIT_EXCEEDED"
    assert failures[0]["failure_stage"] == "ZIP_INFO"
    assert failures[0]["source_sha256"] is None
    failed_source = next(row for row in sources if row["source_status"] == "FAILED")
    assert failed_source["parse_failure_id"] == failures[0]["failure_id"]
    assert failed_source["archive_member_id"] == failures[0]["archive_member_id"]
    assert {row["source_status"] for row in sources} == {"PARSED", "FAILED"}
    ledger_digest = hashlib.sha256(
        (output / "parse_failures.jsonl").read_bytes()
    ).hexdigest()
    assert manifest["files"]["parse_failures.jsonl"] == ledger_digest


def test_high_compression_ratio_is_rejected_before_parse(tmp_path):
    archive = tmp_path / "high-ratio.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "Accounts/good.xhtml", UK_XHTML, compress_type=zipfile.ZIP_STORED
        )
        handle.writestr(
            "Accounts/bomb.xhtml",
            b"A" * 20_000,
            compress_type=zipfile.ZIP_DEFLATED,
        )
    output = tmp_path / "high-ratio-corpus"
    manifest = build_companies_house_corpus(
        archive,
        output,
        zip_limits=CompaniesHouseZipLimits(
            max_member_uncompressed_bytes=32_000,
            max_compression_ratio=2,
        ),
    )

    failure = _read_jsonl(output / "parse_failures.jsonl")[0]
    assert manifest["parse_failure_counts"] == {"COMPRESSION_RATIO_LIMIT_EXCEEDED": 1}
    assert failure["failure_code"] == "COMPRESSION_RATIO_LIMIT_EXCEEDED"
    assert failure["failure_stage"] == "ZIP_INFO"
    assert Decimal(failure["compression_ratio"]) > 2


def test_corrupt_member_is_a_typed_read_failure(tmp_path):
    archive = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr("Accounts/good.xhtml", UK_XHTML)
        handle.writestr("Accounts/corrupt.xhtml", UK_XHTML)
    _flip_stored_member_byte(archive, "Accounts/corrupt.xhtml")
    output = tmp_path / "corrupt-corpus"
    manifest = build_companies_house_corpus(archive, output)

    failure = _read_jsonl(output / "parse_failures.jsonl")[0]
    assert manifest["filing_count"] == 1
    assert manifest["parse_failure_counts"] == {"ARCHIVE_MEMBER_READ_ERROR": 1}
    assert failure["failure_code"] == "ARCHIVE_MEMBER_READ_ERROR"
    assert failure["failure_stage"] == "READ"
    assert failure["error_type"] == "BadZipFile"


def test_invalid_xhtml_is_ledgered_without_replacement_and_is_reproducible(tmp_path):
    archive = tmp_path / "invalid-xhtml.zip"
    seed = 20260821
    names = [f"Accounts/member-{index}.xhtml" for index in range(4)]
    selected_names = sorted(
        names,
        key=lambda name: (
            hashlib.sha256(f"{seed}\x1f{name}".encode()).hexdigest(),
            name,
        ),
    )[:2]
    invalid_name = selected_names[0]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        for name in names:
            handle.writestr(name, b"not xml" if name == invalid_name else UK_XHTML)

    output_one = tmp_path / "invalid-corpus-one"
    output_two = tmp_path / "invalid-corpus-two"
    manifest_one = build_companies_house_corpus(
        archive, output_one, seed=seed, max_filings=2
    )
    manifest_two = build_companies_house_corpus(
        archive, output_two, seed=seed, max_filings=2
    )

    failure = _read_jsonl(output_one / "parse_failures.jsonl")[0]
    assert manifest_one["xhtml_candidate_member_count"] == 4
    assert manifest_one["selected_member_count"] == 2
    assert manifest_one["filing_count"] == 1
    assert manifest_one["parse_failure_count"] == 1
    assert failure["source_name"] == invalid_name
    assert failure["failure_code"] == "XHTML_PARSE_ERROR"
    assert failure["failure_stage"] == "PARSE"
    assert failure["source_sha256"] == hashlib.sha256(b"not xml").hexdigest()
    assert manifest_one == manifest_two
    for artifact_name in manifest_one["files"]:
        assert (output_one / artifact_name).read_bytes() == (
            output_two / artifact_name
        ).read_bytes()


def test_all_invalid_selected_members_publish_a_typed_failure_ledger(tmp_path):
    archive = tmp_path / "all-invalid.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr("Accounts/one.xhtml", b"not xml one")
        handle.writestr("Accounts/two.xhtml", b"not xml two")
    output = tmp_path / "all-invalid-corpus"

    with pytest.raises(ValueError, match="FAILED_NO_PARSEABLE_FILINGS|No parseable"):
        build_companies_house_corpus(archive, output)

    manifest = json.loads((output / "corpus_manifest.json").read_text(encoding="utf-8"))
    failures = _read_jsonl(output / "parse_failures.jsonl")
    sources = _read_jsonl(output / "source_manifest.jsonl")
    assert manifest["build_status"] == "FAILED_NO_PARSEABLE_FILINGS"
    assert manifest["selected_member_count"] == 2
    assert manifest["filing_count"] == 0
    assert manifest["parse_failure_count"] == 2
    assert len(failures) == len(sources) == 2
    assert {row["source_status"] for row in sources} == {"FAILED"}
    assert all(row["parse_failure_id"] for row in sources)


def test_corpus_secret_gate_rejects_before_atomic_publication(tmp_path, monkeypatch):
    archive = tmp_path / "secret.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr("Accounts/one.xhtml", UK_XHTML)
    synthetic_secret = "hf_" + "A" * 24
    monkeypatch.setattr(
        "auditops.companies_house.build_companies_house_narrative_tasks",
        lambda filing: [{"question": synthetic_secret}],
    )
    output = tmp_path / "secret-corpus"

    with pytest.raises(ValueError, match="Secret scan failed"):
        build_companies_house_corpus(archive, output)

    assert not output.exists()


def test_encrypted_and_pathological_members_are_ledgered(tmp_path):
    archive = tmp_path / "unsafe.zip"
    symlink = zipfile.ZipInfo("Accounts/link.xhtml")
    symlink.create_system = 3
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr("Accounts/good.xhtml", UK_XHTML)
        handle.writestr("../escape.xhtml", UK_XHTML)
        handle.writestr(symlink, b"Accounts/good.xhtml")
        handle.writestr("Accounts/encrypted.xhtml", UK_XHTML)
    _mark_member_encrypted(archive, "Accounts/encrypted.xhtml")
    output = tmp_path / "unsafe-corpus"
    manifest = build_companies_house_corpus(archive, output)

    failures = _read_jsonl(output / "parse_failures.jsonl")
    assert manifest["filing_count"] == 1
    assert manifest["parse_failure_count"] == 3
    assert manifest["parse_failure_counts"] == {
        "ENCRYPTED_MEMBER": 1,
        "NON_REGULAR_MEMBER": 1,
        "UNSAFE_MEMBER_NAME": 1,
    }
    assert {row["failure_code"] for row in failures} == {
        "ENCRYPTED_MEMBER",
        "NON_REGULAR_MEMBER",
        "UNSAFE_MEMBER_NAME",
    }
    assert {row["failure_stage"] for row in failures} == {"ZIP_INFO"}


def test_build_vlm_diagnostic_separates_cases_and_gold(tmp_path):
    pairs = tmp_path / "pairs.csv"
    with pairs.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "company_number",
                "transaction_id",
                "document_id",
                "pdf_path",
                "pdf_page_count",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "P001",
                "company_number": "01234567",
                "transaction_id": "tx1",
                "document_id": "doc1",
                "pdf_path": "/data/P001.pdf",
                "pdf_page_count": "3",
            }
        )
    facts = tmp_path / "facts.csv"
    with facts.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "fact_label",
                "ixbrl_tag_concept",
                "ixbrl_value",
                "unit",
                "period_context",
                "pdf_page",
                "pdf_evidence",
                "usable_as_benchmark",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "P001",
                "fact_label": "Current assets",
                "ixbrl_tag_concept": "uk-core:CurrentAssets",
                "ixbrl_value": "1200",
                "unit": "GBP",
                "period_context": "current year",
                "pdf_page": "page 2",
                "pdf_evidence": "Current assets 1,200",
                "usable_as_benchmark": "yes",
            }
        )
    output = tmp_path / "vlm"
    manifest = build_companies_house_vlm_diagnostic(pairs, facts, output)
    case = _read_jsonl(output / "cases.jsonl")[0]
    gold = _read_jsonl(output / "gold.jsonl")[0]
    assert manifest["case_count"] == 1
    assert "ixbrl_value" not in case
    assert gold["facts"][0]["page"] == 2
    assert manifest["bbox_gold_available"] is False
    with pytest.raises(ValueError, match="Expected 20 VLM filing cases"):
        build_companies_house_vlm_diagnostic(
            pairs,
            facts,
            tmp_path / "wrong-count",
            expected_pair_count=20,
            expected_fact_count=266,
        )
    assert not (tmp_path / "wrong-count").exists()
