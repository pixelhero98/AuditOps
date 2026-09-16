# Companies House: taxonomy, iXBRL and PDF findings

AuditOps should normalize source facts into an evidence-linked audit vocabulary and use NLP to propose additional mappings from visible disclosures. A fixed list of local tag names is too narrow for the observed files. This is a design recommendation; no experiment here demonstrates that NLP tagging improves accuracy.

## What the tagging rules mean

There is no single built-in Companies House minimum list shared by the inspected filings. They reference external FRC or older UK-GAAP taxonomy entry points. HMRC's historical minimum lists specified which present items needed tags; its updated guidance distinguishes those lists from FRC taxonomies, which do not contain a separate minimum list. A sparse filing is not, by itself, evidence of a tagging violation. HMRC tax-return accounts and accounts made public through Companies House must not be treated as identical disclosure scopes. [HMRC tagging guidance, minimum-list and scope sections](https://assets.publishing.service.gov.uk/media/5e4e5144d3bf7f3944f0b305/XBRL_tagging.pdf).

Taxonomy version and reporting framework matter. The FRC updates taxonomies annually and directs preparers to versions matching the underlying standards; data collectors determine accepted versions. Interpret the actual `schemaRef`, namespace URI, and context rather than guessing from a preparer's XML prefix. [FRC current taxonomy guidance](https://www.frc.org.uk/library/standards-codes-policy/accounting-and-reporting/frc-taxonomies/current-uk-and-irish-digital-reporting-taxonomies/), [2026 tagging guide](https://media.frc.org.uk/documents/XBRL_Tagging_Guide_-_FRC_Taxonomies_2026.pdf).

Companies House's bulk accounts product covers electronically filed accounts and can contain inline HTML, XBRL XML, and zipped inline documents. It is not a census of every paper/PDF filing. AuditOps's current bulk adapter supports XHTML/iXBRL members; plain XBRL XML and nested ZIP ingestion must not be described as implemented. [Companies House accounts data product](https://www.gov.uk/guidance/companies-house-data-products#accounts-data-product).

## Recorded evidence and denominators

The [sanitized inspection table](evidence/companies-house-inspection.json) binds 20 paired filings to public document identifiers and SHA-256 hashes of the XHTML and PDF bytes. Its inspection classifications are historical records, not a new human validation or model-accuracy result. The sources are a convenience sample, not a representative sample of UK companies.

| Recorded measure | Result | Interpretation |
|---|---:|---|
| Paired filings | 20 | Same-filing pairing recorded for this cohort |
| Report types | 8 micro, 9 total-exemption full, 2 total-exemption small, 1 unaudited abridged | Mainly small/private-company accounts |
| Taxonomy generations | 7 | FRS-102 entry points from 2014, 2019, 2021, 2022, 2023 and 2025; UK-GAAP 2009 |
| Current-ratio/working-capital input status | 9 computable, 10 missing-input, 1 nonnumeric | Per filing, not a score over every possible audit task |
| Quantitative quality | 8 clean, 1 signed-creditor edge, 11 blocked | Historical calculation classification; current strict runtime may refuse more |
| Inspected facts | 266 recorded exact matches | Selected facts, not all facts in these accounts |
| Processing classification | 266 OCR-needed rows | Every sampled PDF was recorded as having no usable embedded text layer |

A separate January electronic-accounts metadata scan recorded 2,007 PDF+iXBRL pairs among 2,030 filing-history records for 2,000 sampled companies. This measures resource availability within an electronic-accounts sampling frame, not whole-register coverage, semantic equivalence, or extraction accuracy. Another 20-company listed-parent check found PDF but no XHTML resources through the inspected Companies House route. Those cohorts must not be pooled with the paired small-company sample or treated as proof that all listed companies lack structured accounts. The [coverage record](evidence/companies-house-coverage.json) preserves source-table hashes and bounded observations.

## Corrected concept-count discrepancy

Earlier reports gave P003 either 48 or 49 concepts. Recounting the same XHTML bytes using XML elements yields **105 inline facts and 49 distinct expanded QNames**. The old non-recursive regular expression counts **103 facts and 48 lexical names**: it misses nested occurrences of `business:EndDateForPeriodCoveredByReport` and `business:BalanceSheetDate`. One missed name is represented elsewhere, hence the one-concept difference. The exporter reproduces this discrepancy as a diagnostic; XML traversal is the counting method used in the published table.

P001 has 45 inline facts and 30 distinct expanded QNames. Counts include named `nonFraction`, `nonNumeric`, and `fraction` elements; repeated contexts count as separate facts, while concept counts resolve namespace URI plus local name. Neither count measures disclosure quality. A search for the phrase “minimum tagging list” is not a meaningful test of tagging compliance and is not used as evidence here.

## Failure modes and present safeguards

| Issue | Why a raw tag list fails | Required treatment |
|---|---|---|
| Taxonomy and prefix variation | Prefixes are arbitrary; local names can collide across namespaces and years | Preserve expanded QName and schemaRef; use versioned mappings |
| Generic `Creditors` | The same name can represent current or noncurrent maturities | Require supported maturity evidence; current dimension allowlists are deliberately narrow |
| Entity, period and dimensions | Prior-year, subsidiary or dimensioned values can resemble the desired fact | Match the task's complete context and reject ambiguity |
| Signs and scales | Display parentheses, explicit sign and iXBRL scale can differ from naïve text parsing | Preserve raw text and attributes; normalize once and reconcile where supported |
| Dash, nil, missing disclosure | A dash is not universally zero; unavailable turnover is not an extraction error | Honor the declared transformation; otherwise refuse unsupported numeric inputs |
| Continuations and duplicate facts | Regex extraction may miss nesting or joins; duplicates can conflict | Parse structure, validate continuation chains, retain identifiers, reject conflicts |
| Untagged narrative | Audit questions may concern labels, policies or statements not captured by a selected numeric tag | Retrieve visible evidence with exact spans and scope metadata |
| Image-only PDF | Text extraction can be empty even when information is visibly present | Use a separate OCR/VLM pipeline with page evidence and uncertainty |

The parser already tests creditor signs, maturity dimensions, units, continuations, nil/dash transformations, conflicts and refusal behavior. Its bounded concept selection still uses normalized local-name allowlists; it is **not** a complete namespace/version-aware semantic registry. The cleanup preserves that boundary instead of claiming the proposed solution already exists.

The PDF diagnostic can validate page citations and box syntax/bounds. The 266-fact inspection has no human-labelled bounding boxes, so box accuracy or IoU cannot be reported. Text- and image-based extraction must remain separate evaluation tracks.

## Proposed NLP-assisted canonicalization

1. Preserve source bytes and parse available iXBRL facts, labels, contexts, tables and narrative. For image PDFs, retain OCR text with page and spatial provenance.
2. Give a model the relevant evidence and a versioned audit vocabulary. Ask it to propose a canonical concept, exact supporting span, entity, period, unit, maturity and uncertainty; retain the original QName and attributes.
3. Validate source existence, exact span, context, numeric transformation and permitted mapping. An unresolved source/model conflict remains ambiguous; confidence alone cannot authorize a mapping.
4. Admit only verified canonical facts to registered calculations and extractive answers. Keep missing disclosures and failed mappings as typed refusals.
5. Compare raw-name allowlists, taxonomy-aware rules, and NLP-assisted mapping on entity-disjoint reviewed examples. Measure mapping precision/recall, period/unit/sign errors, false numeric substitutions, refusal behavior and evidence support separately for XHTML and PDFs.

This approach can cover linguistic variation and untagged passages while retaining structured facts as strong evidence. It cannot recover genuinely undisclosed information. No fine-tuning, new NLP runtime, or superiority claim is included in this release.

## Reproduction

Run `python scripts/export_ch_findings.py --help`. Supply the original pair manifest,
fact inspection table, task results, and a filing root containing `P001/P001.xhtml`
and `P001/P001.pdf` through `P020`. The output contains only allowlisted public fields
and hashes, never source paths or reviewer notes. The original files are not bundled.
`python scripts/check_findings.py` verifies published counts, reconciliation, coverage
arithmetic, and required source bindings without those private working files.
