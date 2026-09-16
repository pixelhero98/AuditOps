"""Verify the bounded public Companies House evidence without raw documents."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


def verify(root: Path) -> None:
    evidence = root / "docs" / "evidence"
    inspection = json.loads((evidence / "companies-house-inspection.json").read_text())
    filings, summary = inspection["filings"], inspection["summary"]
    assert len(filings) == summary["pair_count"] == 20
    assert len({row["pair_id"] for row in filings}) == 20
    assert (
        sum(row["inspected_fact_count"] for row in filings)
        == summary["inspected_fact_count"]
        == 266
    )
    assert (
        Counter(row["numeric_status_recorded"] for row in filings)
        == summary["numeric_status_recorded"]
        == {
            "computable": 9,
            "missing_input": 10,
            "non_numeric_input": 1,
        }
    )
    assert summary["match_status_recorded"] == {"exact": 266}
    assert summary["processing_mode_recorded"] == {"OCR-needed": 266}
    assert Counter(row["report_type"] for row in filings) == summary["report_types"]
    assert len({ref for row in filings for ref in row["schema_refs"]}) == 7
    for row in filings:
        assert row["pdf_text_layer_recorded"] == "none"
        assert row["same_filing_confirmed_recorded"] == "yes"
        assert row["document_metadata_url"].startswith(
            "https://document-api.company-information.service.gov.uk/document/"
        )
        for field in ("xhtml_sha256", "pdf_sha256"):
            assert re.fullmatch(r"[0-9a-f]{64}", row[field])
        assert (
            row["inline_fact_count"]
            >= row["distinct_expanded_qnames"]
            >= row["distinct_local_names"]
        )
    for row in inspection["source_tables"]:
        assert set(row) == {"role", "sha256"}
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
    p003 = next(row for row in filings if row["pair_id"] == "P003")
    assert (p003["inline_fact_count"], p003["distinct_expanded_qnames"]) == (105, 49)
    old = p003["historical_regex_count_diagnostic"]
    assert (old["fact_count"], old["distinct_lexical_qnames"]) == (103, 48)
    assert sum(old["missed_occurrences"].values()) == 2

    coverage = json.loads((evidence / "companies-house-coverage.json").read_text())
    for field in ("coverage_source_sha256", "listed_source_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", coverage[field])
    metrics = coverage["electronic_frame_metrics"]
    assert int(metrics["companies_processed"]) == 2000
    total = int(metrics["filing_denominator_all"])
    paired = int(metrics["filings_pdf_and_ixbrl_available"])
    assert (total, paired) == (2030, 2007)
    assert paired + int(metrics["filings_pdf_only"]) == total
    assert round(paired / total, 4) == float(metrics["filing_pair_coverage_ratio_all"])
    listed = coverage["listed_parent_observations"]
    assert len(listed) == 20
    assert all(row["structured_available"] == "no" for row in listed)
    assert all(row["resources"] == "application/pdf" for row in listed)


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[1])
    print("Companies House counts, source bindings and reconciliation verified")
