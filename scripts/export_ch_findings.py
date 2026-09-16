"""Export bounded, path-free inspection evidence from explicitly supplied sources.

No filing text, private notes, or reviewer identities are copied. Source hashes
identify the inspection tables; per-filing hashes bind documents counted here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from lxml import etree


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def counts(items: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row[field] for row in items).items()))


def build_summary(
    pair_manifest: Path, fact_audit: Path, task_results: Path, filing_root: Path
) -> dict:
    pairs, facts, tasks = rows(pair_manifest), rows(fact_audit), rows(task_results)
    pair_ids = [row["pair_id"] for row in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Duplicate pair identity")
    if set(row["pair_id"] for row in tasks) != set(pair_ids) or len(tasks) != len(
        pairs
    ):
        raise ValueError("Task result membership differs from pair membership")
    if set(row["pair_id"] for row in facts) != set(pair_ids):
        raise ValueError("Inspected fact membership differs from pair membership")
    by_task = {row["pair_id"]: row for row in tasks}
    filings = []
    for row in sorted(pairs, key=lambda row: row["pair_id"]):
        pair_id = row["pair_id"]
        if not pair_id.isalnum():
            raise ValueError("Non-canonical pair identity")
        xhtml = filing_root / pair_id / f"{pair_id}.xhtml"
        pdf = filing_root / pair_id / f"{pair_id}.pdf"
        document = etree.fromstring(
            xhtml.read_bytes(),
            etree.XMLParser(resolve_entities=False, no_network=True, recover=False),
        )
        if document is None:
            raise ValueError("Filing could not be parsed")
        fact_nodes = [
            node
            for node in document.iter()
            if isinstance(node.tag, str)
            and etree.QName(node).namespace
            in {
                "http://www.xbrl.org/2013/inlineXBRL",
                "http://www.xbrl.org/2008/inlineXBRL",
            }
            and etree.QName(node).localname in {"nonFraction", "nonNumeric", "fraction"}
            and node.get("name")
        ]
        expanded = set()
        for node in fact_nodes:
            prefix, separator, local = node.get("name").partition(":")
            namespace = node.nsmap.get(prefix if separator else None)
            if not namespace:
                raise ValueError("Unresolved fact QName")
            expanded.add((namespace, local if separator else prefix))
        schema_refs = sorted(
            {
                node.get("{http://www.w3.org/1999/xlink}href")
                for node in document.iter()
                if isinstance(node.tag, str)
                and etree.QName(node).localname == "schemaRef"
                and node.get("{http://www.w3.org/1999/xlink}href")
            }
        )
        source_url = row["document_metadata_url"]
        if not source_url.startswith(
            "https://document-api.company-information.service.gov.uk/document/"
        ):
            raise ValueError("Expected public Companies House document identifier")
        filings.append(
            {
                "pair_id": pair_id,
                "company_number": row["company_number"],
                "filing_date": row["filing_date"],
                "period_end": row["period_end_made_up_date"],
                "report_type": row["report_type"],
                "document_metadata_url": source_url,
                "same_filing_confirmed_recorded": row["same_filing_confirmed"],
                "pdf_text_layer_recorded": row["pdf_text_layer"],
                "xhtml_sha256": digest(xhtml),
                "pdf_sha256": digest(pdf),
                "schema_refs": schema_refs,
                "inline_fact_count": len(fact_nodes),
                "distinct_lexical_qnames": len(
                    {node.get("name") for node in fact_nodes}
                ),
                "distinct_expanded_qnames": len(expanded),
                "distinct_local_names": len({name for _, name in expanded}),
                "numeric_status_recorded": by_task[pair_id]["numeric_status"],
                "numeric_quality_recorded": by_task[pair_id]["numeric_quality"],
                "quantitative_evidence_recorded": {
                    key: by_task[pair_id][key]
                    for key in (
                        "current_assets_concept",
                        "current_assets_value",
                        "current_assets_evidence",
                        "current_assets_pdf_page",
                        "creditors_concept",
                        "creditors_ixbrl_value_used",
                        "creditors_abs_liability_value",
                        "creditors_evidence",
                        "creditors_pdf_page",
                    )
                },
                "inspected_fact_count": sum(
                    fact["pair_id"] == pair_id for fact in facts
                ),
            }
        )
        # Reproduce the old non-recursive counter solely to explain the published discrepancy.
        old_pattern = r"<ix:(?P<tag>nonNumeric|nonFraction)\b(?P<attrs>[^>]*?)(?<!/)>(?P<body>.*?)</ix:(?P=tag)>"
        old_names = []
        for match in re.finditer(
            old_pattern, xhtml.read_text(encoding="utf-8"), re.I | re.S
        ):
            name = re.search(r'\bname="([^"]+)"', match.group("attrs"))
            if name:
                old_names.append(name.group(1))
        for match in re.finditer(
            r"<ix:(?:nonNumeric|nonFraction)\b([^>]*)/>",
            xhtml.read_text(encoding="utf-8"),
            re.I | re.S,
        ):
            name = re.search(r'\bname="([^"]+)"', match.group(1))
            if name:
                old_names.append(name.group(1))
        filings[-1]["historical_regex_count_diagnostic"] = {
            "fact_count": len(old_names),
            "distinct_lexical_qnames": len(set(old_names)),
            "missed_occurrences": dict(
                sorted(
                    (
                        Counter(node.get("name") for node in fact_nodes)
                        - Counter(old_names)
                    ).items()
                )
            ),
        }
    return {
        "version": "auditops-ch-inspection-summary.v1",
        "assurance": "HISTORICAL_RECORDED_INSPECTION; NOT MODEL_ACCURACY",
        "concept_counting": "Distinct expanded QNames among inline nonFraction, nonNumeric and fraction elements with name; prefix aliases resolve to namespace URI. No text-word counts.",
        "source_tables": [
            {"role": role, "sha256": digest(path)}
            for role, path in (
                ("pair_manifest", pair_manifest),
                ("fact_audit", fact_audit),
                ("task_results", task_results),
            )
        ],
        "summary": {
            "pair_count": len(pairs),
            "inspected_fact_count": len(facts),
            "match_status_recorded": counts(facts, "match_status"),
            "processing_mode_recorded": counts(facts, "processing_mode"),
            "numeric_status_recorded": counts(tasks, "numeric_status"),
            "numeric_quality_recorded": counts(tasks, "numeric_quality"),
            "report_types": counts(pairs, "report_type"),
        },
        "filings": filings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "pair-manifest",
        "fact-audit",
        "task-results",
        "filing-root",
        "output",
    ):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    args = parser.parse_args()
    summary = build_summary(
        args.pair_manifest, args.fact_audit, args.task_results, args.filing_root
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    main()
