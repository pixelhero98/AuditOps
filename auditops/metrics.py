from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from decimal import Decimal
from importlib import resources
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import sqlite3

from .pipeline import _stable_digest, connect_db, parse_date


GENERATOR_VERSION = "v0"


@dataclass(frozen=True)
class PeriodDescriptor:
    filing_id: str
    ticker: Optional[str]
    period_type: str
    period_key: str
    period_start: Optional[str]
    period_end: Optional[str]
    fiscal_year: Optional[int]
    fiscal_quarter: Optional[int]
    is_ytd: bool


@dataclass
class ResolvedInput:
    name: str
    concept_norm: str
    period_key: str
    period_type: str
    filing_id: str
    unit_canon: Optional[str]
    numeric_value: Optional[Decimal]
    text_value: Optional[str]
    fact_evidence_ids: List[str]
    trace: List[Dict[str, Any]]
    derived: bool = False


@dataclass
class EvalValue:
    value: Decimal
    unit_canon: Optional[str]
    evidence_ids: List[str]
    derivation_steps: List[Dict[str, Any]]


class Refusal(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class Catalog:
    def __init__(self, conn: sqlite3.Connection, filing_id: Optional[str] = None, ticker: Optional[str] = None):
        self.conn = conn
        filing_scope_ids: Optional[List[str]] = None
        if filing_id is not None and ticker is None:
            filing_row = conn.execute("SELECT * FROM filings WHERE filing_id=?", (filing_id,)).fetchone()
            if filing_row is None:
                raise KeyError(f"Unknown filing_id: {filing_id}")
            ticker = filing_row["ticker"]
            if ticker:
                filing_scope_ids = [
                    row["filing_id"]
                    for row in conn.execute("SELECT filing_id FROM filings WHERE ticker=? ORDER BY filing_id", (ticker,)).fetchall()
                ]
            else:
                filing_scope_ids = [filing_id]
        elif ticker is not None:
            filing_scope_ids = [
                row["filing_id"]
                for row in conn.execute("SELECT filing_id FROM filings WHERE ticker=? ORDER BY filing_id", (ticker,)).fetchall()
            ]

        filing_query = "SELECT * FROM filings"
        filing_params: Tuple[Any, ...] = ()
        fact_query = "SELECT * FROM facts_canon ORDER BY filing_id, period_key, concept_norm"
        fact_params: Tuple[Any, ...] = ()
        cal_query = "SELECT * FROM cal_edges ORDER BY filing_id, role, parent_concept_norm, ord"
        cal_params: Tuple[Any, ...] = ()

        if filing_scope_ids is not None:
            placeholders = ",".join("?" for _ in filing_scope_ids)
            filing_query = f"SELECT * FROM filings WHERE filing_id IN ({placeholders})"
            fact_query = f"SELECT * FROM facts_canon WHERE filing_id IN ({placeholders}) ORDER BY filing_id, period_key, concept_norm"
            cal_query = f"SELECT * FROM cal_edges WHERE filing_id IN ({placeholders}) ORDER BY filing_id, role, parent_concept_norm, ord"
            filing_params = tuple(filing_scope_ids)
            fact_params = tuple(filing_scope_ids)
            cal_params = tuple(filing_scope_ids)

        self.filings = {
            row["filing_id"]: dict(row)
            for row in conn.execute(filing_query, filing_params).fetchall()
        }
        self.facts = [dict(row) for row in conn.execute(fact_query, fact_params).fetchall()]
        self.cal_edges = [dict(row) for row in conn.execute(cal_query, cal_params).fetchall()]

        for fact in self.facts:
            if fact["value_num_exact"] is not None:
                fact["value_num_exact"] = Decimal(str(fact["value_num_exact"]))
            fact["is_ytd"] = bool(fact["is_ytd"])
            filing = self.filings[fact["filing_id"]]
            fact["ticker"] = filing["ticker"]
            fact["form_type"] = filing["form_type"]
            fact["report_date"] = filing["report_date"]

        self.facts_by_ticker: Dict[Optional[str], List[Dict[str, Any]]] = {}
        for fact in self.facts:
            self.facts_by_ticker.setdefault(fact["ticker"], []).append(fact)

        self.cal_edges_by_filing_parent: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for edge in self.cal_edges:
            self.cal_edges_by_filing_parent.setdefault((edge["filing_id"], edge["parent_concept_norm"]), []).append(edge)

    def periods_for_filing(self, filing_id: str) -> List[PeriodDescriptor]:
        """Return all period descriptors available for a filing.
        
        Parameters
        ----------
        filing_id : str
            Canonical filing identifier used by manifests and derived artifacts. e.g., '0000320193-2025-10K'
        
        Returns
        -------
        List[PeriodDescriptor]
            List of records for return all period descriptors available for a filing.
        """
        seen = set()
        periods: List[PeriodDescriptor] = []
        for fact in self.facts:
            if fact["filing_id"] != filing_id:
                continue
            key = (
                fact["period_key"],
                fact["period_type"],
                fact["period_start"],
                fact["period_end"],
                fact["fiscal_year"],
                fact["fiscal_quarter"],
                bool(fact["is_ytd"]),
            )
            if key in seen:
                continue
            seen.add(key)
            periods.append(
                PeriodDescriptor(
                    filing_id=filing_id,
                    ticker=fact["ticker"],
                    period_type=fact["period_type"],
                    period_key=fact["period_key"],
                    period_start=fact["period_start"],
                    period_end=fact["period_end"],
                    fiscal_year=fact["fiscal_year"],
                    fiscal_quarter=fact["fiscal_quarter"],
                    is_ytd=bool(fact["is_ytd"]),
                )
            )
        periods.sort(key=lambda period: (period.period_end or "", period.period_key))
        return periods

    def _matches_period(self, fact: Dict[str, Any], period: PeriodDescriptor) -> bool:
        return (
            fact["period_type"] == period.period_type
            and fact["period_key"] == period.period_key
            and fact["fiscal_year"] == period.fiscal_year
            and fact["fiscal_quarter"] == period.fiscal_quarter
            and bool(fact["is_ytd"]) == bool(period.is_ytd)
        )

    def _choose_match(self, matches: List[Dict[str, Any]], preferred_filing_id: str) -> Dict[str, Any]:
        if not matches:
            raise Refusal("MISSING_INPUT", "No canonical fact matched the requested input.")

        def score(row: Dict[str, Any]) -> Tuple[int, int]:
            return (0 if row["filing_id"] == preferred_filing_id else 1, 0 if row["source_anchor"] else 1)

        scored = {}
        for row in matches:
            scored.setdefault(score(row), []).append(row)
        top_score = sorted(scored)[0]
        top_rows = scored[top_score]
        if len(top_rows) > 1:
            values = {(row["value_num_exact"], row["value_text"]) for row in top_rows}
            if len(values) > 1:
                raise Refusal("AMBIGUOUS_CONTEXT", "Multiple canonical facts compete for the same input across filings.")
            top_rows.sort(key=lambda row: (row["filing_id"], row["fact_evidence_id"]))
        return top_rows[0]

    def find_alias_match(
        self,
        ticker: Optional[str],
        period: PeriodDescriptor,
        aliases: Sequence[str],
        unit_family: Optional[str],
        preferred_filing_id: str,
    ) -> Tuple[Dict[str, Any], str]:
        """Return the best matching fact for the aliases in a period.
        
        Parameters
        ----------
        ticker : Optional[str]
            Issuer ticker symbol used in manifests and derived outputs. e.g., 'AAPL'
        period : PeriodDescriptor
            Period descriptor with reporting boundaries and fiscal metadata.
        aliases : Sequence[str]
            Candidate concept aliases used to resolve canonical facts.
        unit_family : Optional[str]
            Canonical unit family used to filter comparable facts.
        preferred_filing_id : str
            Identifier for preferred filing.
        
        Returns
        -------
        Tuple[Dict[str, Any], str]
            Tuple with outputs produced while return the best matching fact for the aliases in a period.
        
        Raises
        ------
        Refusal
            MISSING_INPUT.
        """
        facts = self.facts_by_ticker.get(ticker, [])
        for alias in aliases:
            matches = [
                fact
                for fact in facts
                if fact["concept_norm"] == alias
                and self._matches_period(fact, period)
                and (unit_family is None or fact["unit_canon"] == unit_family)
            ]
            if matches:
                return self._choose_match(matches, preferred_filing_id), alias
        raise Refusal("MISSING_INPUT", f"No canonical fact found for aliases: {', '.join(aliases)}")

    def find_alias_match_near_asof(
        self,
        ticker: Optional[str],
        target_date,
        aliases: Sequence[str],
        unit_family: Optional[str],
        preferred_filing_id: str,
        max_day_delta: int = 7,
    ) -> Tuple[Dict[str, Any], str]:
        """Return the alias match nearest to the target as-of date.
        
        Parameters
        ----------
        ticker : Optional[str]
            Issuer ticker symbol used in manifests and derived outputs.
        target_date : Any
            Reference date used for nearest-period or as-of matching.
        aliases : Sequence[str]
            Candidate concept aliases used to resolve canonical facts.
        unit_family : Optional[str]
            Canonical unit family used to filter comparable facts.
        preferred_filing_id : str
            Identifier for preferred filing.
        max_day_delta : int, optional
            Maximum allowed day distance when matching near an as-of date.
        
        Returns
        -------
        Tuple[Dict[str, Any], str]
            Tuple with outputs produced while return the alias match nearest to the target as-of date.
        
        Raises
        ------
        Refusal
            AMBIGUOUS_CONTEXT.
        Refusal
            MISSING_INPUT.
        
        Examples
        --------
        >>> conn = connect_db('corpora/sp500_latest_2026-03-20/db/corpus.sqlite')
        >>> catalog = Catalog(conn)
        >>> result = catalog.find_alias_match_near_asof(ticker='AAPL', target_date=None, aliases=[])  # doctest: +SKIP
        >>> len(result)  # doctest: +SKIP
        """
        facts = self.facts_by_ticker.get(ticker, [])
        for alias in aliases:
            matches = []
            for fact in facts:
                if fact["concept_norm"] != alias or fact["period_type"] != "ASOF":
                    continue
                if unit_family is not None and fact["unit_canon"] != unit_family:
                    continue
                fact_end = parse_date(fact["period_end"]) if fact["period_end"] else None
                if fact_end is None:
                    continue
                day_delta = abs((fact_end - target_date).days)
                if day_delta <= max_day_delta:
                    matches.append((day_delta, fact))
            if not matches:
                continue

            matches.sort(
                key=lambda item: (
                    item[0],
                    0 if item[1]["filing_id"] == preferred_filing_id else 1,
                    0 if item[1]["source_anchor"] else 1,
                    item[1]["fact_evidence_id"],
                )
            )
            top_delta = matches[0][0]
            top_rows = [fact for delta, fact in matches if delta == top_delta]
            values = {(row["value_num_exact"], row["value_text"]) for row in top_rows}
            if len(values) > 1:
                raise Refusal("AMBIGUOUS_CONTEXT", "Multiple near-date ASOF facts compete for the same input.")
            top_rows.sort(key=lambda row: (0 if row["filing_id"] == preferred_filing_id else 1, row["fact_evidence_id"]))
            return top_rows[0], alias

        raise Refusal("MISSING_INPUT", f"No canonical ASOF fact found near {target_date.isoformat()} for aliases: {', '.join(aliases)}")

    def derive_quarter_value(
        self,
        ticker: Optional[str],
        aliases: Sequence[str],
        fiscal_year: int,
        fiscal_quarter: int,
        unit_family: Optional[str],
        preferred_filing_id: str,
    ) -> Tuple[ResolvedInput, str]:
        """Derive a quarter value from annual and cumulative period values.
        
        Parameters
        ----------
        ticker : Optional[str]
            Issuer ticker symbol used in manifests and derived outputs. e.g., 'AAPL'
        aliases : Sequence[str]
            Candidate concept aliases used to resolve canonical facts.
        fiscal_year : int
            Fiscal year used for period filtering or derivation.
        fiscal_quarter : int
            Fiscal quarter used for period filtering or derivation.
        unit_family : Optional[str]
            Canonical unit family used to filter comparable facts.
        preferred_filing_id : str
            Identifier for preferred filing.
        
        Returns
        -------
        Tuple[ResolvedInput, str]
            Tuple with outputs produced while derive a quarter value from annual and cumulative period values.
        
        Raises
        ------
        Refusal
            MISSING_INPUT.
        """
        direct_period = PeriodDescriptor(
            filing_id=preferred_filing_id,
            ticker=ticker,
            period_type="Q",
            period_key=f"Q{fiscal_quarter}_{fiscal_year}",
            period_start=None,
            period_end=None,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            is_ytd=False,
        )
        try:
            fact, alias = self.find_alias_match(ticker, direct_period, aliases, unit_family, preferred_filing_id)
            return (
                ResolvedInput(
                    name="derived-quarter",
                    concept_norm=fact["concept_norm"],
                    period_key=fact["period_key"],
                    period_type=fact["period_type"],
                    filing_id=fact["filing_id"],
                    unit_canon=fact["unit_canon"],
                    numeric_value=fact["value_num_exact"],
                    text_value=fact["value_text"],
                    fact_evidence_ids=[fact["fact_evidence_id"]],
                    trace=[
                        {
                            "kind": "fact",
                            "alias": alias,
                            "fact_evidence_id": fact["fact_evidence_id"],
                            "period_key": fact["period_key"],
                            "derived": False,
                        }
                    ],
                ),
                alias,
            )
        except Refusal:
            pass

        if fiscal_quarter == 1:
            ytd_period = PeriodDescriptor(
                filing_id=preferred_filing_id,
                ticker=ticker,
                period_type="YTD",
                period_key=f"YTD_Q1_{fiscal_year}",
                period_start=None,
                period_end=None,
                fiscal_year=fiscal_year,
                fiscal_quarter=1,
                is_ytd=True,
            )
            fact, alias = self.find_alias_match(ticker, ytd_period, aliases, unit_family, preferred_filing_id)
            return (
                ResolvedInput(
                    name="derived-quarter",
                    concept_norm=fact["concept_norm"],
                    period_key=f"Q1_{fiscal_year}",
                    period_type="Q",
                    filing_id=fact["filing_id"],
                    unit_canon=fact["unit_canon"],
                    numeric_value=fact["value_num_exact"],
                    text_value=fact["value_text"],
                    fact_evidence_ids=[fact["fact_evidence_id"]],
                    trace=[
                        {
                            "kind": "fact",
                            "alias": alias,
                            "fact_evidence_id": fact["fact_evidence_id"],
                            "period_key": fact["period_key"],
                            "derived": True,
                            "source": "ytd_q1",
                        }
                    ],
                    derived=True,
                ),
                alias,
            )

        current_ytd = PeriodDescriptor(
            filing_id=preferred_filing_id,
            ticker=ticker,
            period_type="YTD",
            period_key=f"YTD_Q{fiscal_quarter}_{fiscal_year}",
            period_start=None,
            period_end=None,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            is_ytd=True,
        )
        prev_selector = self.derive_quarter_value(ticker, aliases, fiscal_year, fiscal_quarter - 1, unit_family, preferred_filing_id)
        current_fact, alias = self.find_alias_match(ticker, current_ytd, aliases, unit_family, preferred_filing_id)
        previous_input, _ = prev_selector
        if current_fact["value_num_exact"] is None or previous_input.numeric_value is None:
            raise Refusal("MISSING_INPUT", "Derived quarter requires numeric YTD inputs.")
        value = current_fact["value_num_exact"] - previous_input.numeric_value
        return (
            ResolvedInput(
                name="derived-quarter",
                concept_norm=current_fact["concept_norm"],
                period_key=f"Q{fiscal_quarter}_{fiscal_year}",
                period_type="Q",
                filing_id=current_fact["filing_id"],
                unit_canon=current_fact["unit_canon"],
                numeric_value=value,
                text_value=str(value),
                fact_evidence_ids=[current_fact["fact_evidence_id"], *previous_input.fact_evidence_ids],
                trace=[
                    {
                        "kind": "derived_quarter",
                        "source": "ytd_delta",
                        "current_ytd_evidence_id": current_fact["fact_evidence_id"],
                        "previous_components": previous_input.fact_evidence_ids,
                        "alias": alias,
                    }
                ]
                + previous_input.trace,
                derived=True,
            ),
            alias,
        )

    def calc_edge_source(self, filing_id: str, target_aliases: Sequence[str], component_concepts: Sequence[str]) -> str:
        """Describe the source of a derived edge metric.
        
        Parameters
        ----------
        filing_id : str
            Canonical filing identifier used by manifests and derived artifacts. e.g., '0000320193-2025-10K'
        target_aliases : Sequence[str]
            Primary concept aliases that define the target metric edge.
        component_concepts : Sequence[str]
            Component concepts used to infer calculation-edge provenance.
        
        Returns
        -------
        str
            value for describe the source of a derived edge metric.
        """
        for alias in target_aliases:
            edges = self.cal_edges_by_filing_parent.get((filing_id, alias), [])
            children = {edge["child_concept_norm"] for edge in edges if float(edge["weight"]) > 0}
            if children and set(component_concepts).issubset(children):
                return "calculation_linkbase"
        return "explicit_override"


def load_metric_specs() -> List[Dict[str, Any]]:
    """Load metric specs from auditops/specs/resources.files.
    
    Returns
    -------
    List[Dict[str, Any]]
        List of records for load metric specs.
    """
    spec_path = resources.files("auditops.specs").joinpath("metric_specs.json")
    return json.loads(spec_path.read_text(encoding="utf-8"))


def load_metric_specs_by_id() -> Dict[str, Dict[str, Any]]:
    """Load metric specs dictionary indexed by spec["id"]
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary with spec["id"] as key str.
    """
    return {spec["id"]: spec for spec in load_metric_specs()}


def _shift_year_safe(value, years: int):
    target_year = value.year + years
    target_day = min(value.day, calendar.monthrange(target_year, value.month)[1])
    return value.replace(year=target_year, day=target_day)


def _period_end_asof(period: PeriodDescriptor, year_delta: int = 0) -> PeriodDescriptor:
    if not period.period_end:
        raise Refusal("PERIOD_NOT_SUPPORTED", "Period-end ASOF selector requires an end date.")

    end = parse_date(period.period_end)
    if end is None:
        raise Refusal("PERIOD_NOT_SUPPORTED", "Period-end ASOF selector requires an ISO end date.")

    target_end = _shift_year_safe(end, year_delta)
    fiscal_year = period.fiscal_year + year_delta if period.fiscal_year is not None else None
    return PeriodDescriptor(
        filing_id=period.filing_id,
        ticker=period.ticker,
        period_type="ASOF",
        period_key=f"ASOF_{target_end.strftime('%Y%m%d')}",
        period_start=target_end.isoformat(),
        period_end=target_end.isoformat(),
        fiscal_year=fiscal_year,
        fiscal_quarter=period.fiscal_quarter,
        is_ytd=False,
    )


def _resolve_asof_input(
    catalog: Catalog,
    base_period: PeriodDescriptor,
    name: str,
    aliases: Sequence[str],
    unit_family: Optional[str],
    selector: str,
    year_delta: int,
) -> ResolvedInput:
    target_period = _period_end_asof(base_period, year_delta=year_delta)
    target_date = parse_date(target_period.period_end) if target_period.period_end else None
    fallback_used = False
    try:
        fact, alias = catalog.find_alias_match(base_period.ticker, target_period, aliases, unit_family, base_period.filing_id)
    except Refusal as refusal:
        if refusal.code != "MISSING_INPUT" or target_date is None:
            raise
        fact, alias = catalog.find_alias_match_near_asof(
            base_period.ticker,
            target_date,
            aliases,
            unit_family,
            base_period.filing_id,
        )
        fallback_used = True

    trace = [
        {
            "kind": "fact",
            "input": name,
            "alias": alias,
            "fact_evidence_id": fact["fact_evidence_id"],
            "period_key": fact["period_key"],
            "derived": False,
        },
        {
            "kind": "selector",
            "selector": selector,
        },
    ]
    if fallback_used:
        trace.append(
            {
                "kind": "selector_fallback",
                "selector": selector,
                "target_period_key": target_period.period_key,
                "resolved_period_key": fact["period_key"],
            }
        )

    return ResolvedInput(
        name=name,
        concept_norm=fact["concept_norm"],
        period_key=fact["period_key"],
        period_type=fact["period_type"],
        filing_id=fact["filing_id"],
        unit_canon=fact["unit_canon"],
        numeric_value=fact["value_num_exact"],
        text_value=fact["value_text"],
        fact_evidence_ids=[fact["fact_evidence_id"]],
        trace=trace,
    )


def _shift_period(period: PeriodDescriptor, selector: str) -> PeriodDescriptor:
    if selector == "current":
        return period
    if selector == "prior_year_same_period":
        if period.fiscal_year is None:
            raise Refusal("PERIOD_NOT_SUPPORTED", "Prior-year comparison requires a fiscal year.")
        if period.period_type == "ASOF":
            return _period_end_asof(period, year_delta=-1)
        if period.period_type in {"Q", "YTD"} and period.fiscal_quarter is not None:
            prefix = "YTD_Q" if period.period_type == "YTD" else "Q"
            joiner = "" if period.period_type == "YTD" else ""
            key = f"{prefix}{period.fiscal_quarter}_{period.fiscal_year - 1}"
            return PeriodDescriptor(
                filing_id=period.filing_id,
                ticker=period.ticker,
                period_type=period.period_type,
                period_key=key,
                period_start=None,
                period_end=None,
                fiscal_year=period.fiscal_year - 1,
                fiscal_quarter=period.fiscal_quarter,
                is_ytd=period.is_ytd,
            )
        if period.period_type == "FY":
            return PeriodDescriptor(
                filing_id=period.filing_id,
                ticker=period.ticker,
                period_type="FY",
                period_key=f"FY{period.fiscal_year - 1}",
                period_start=None,
                period_end=None,
                fiscal_year=period.fiscal_year - 1,
                fiscal_quarter=4,
                is_ytd=False,
            )
        raise Refusal("PERIOD_NOT_SUPPORTED", "Prior-year selector is not implemented for this period type.")
    if selector == "previous_quarter":
        if period.fiscal_year is None or period.fiscal_quarter is None:
            raise Refusal("PERIOD_NOT_SUPPORTED", "Previous-quarter comparison requires fiscal year and fiscal quarter.")
        fiscal_year = period.fiscal_year
        fiscal_quarter = period.fiscal_quarter - 1
        if fiscal_quarter == 0:
            fiscal_year -= 1
            fiscal_quarter = 4
        return PeriodDescriptor(
            filing_id=period.filing_id,
            ticker=period.ticker,
            period_type="Q",
            period_key=f"Q{fiscal_quarter}_{fiscal_year}",
            period_start=None,
            period_end=None,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            is_ytd=False,
        )
    raise Refusal("PERIOD_NOT_SUPPORTED", f"Unsupported period selector: {selector}")


def resolve_input(catalog: Catalog, base_period: PeriodDescriptor, spec_input: Dict[str, Any]) -> ResolvedInput:
    """Resolve an input definition into a concrete metric input value.
    
    Parameters
    ----------
    catalog : Catalog
        Catalog object used to resolve canonical facts and period descriptors.
    base_period : PeriodDescriptor
        Reference period used to resolve shifted or comparative inputs.
    spec_input : Dict[str, Any]
        Input definition from a metric specification formula.
    
    Returns
    -------
    ResolvedInput
        Return value for resolve an input definition into a concrete metric input value.
    
    Raises
    ------
    Refusal
        PERIOD_NOT_SUPPORTED.
    """
    name = spec_input["name"]
    aliases = spec_input["aliases"]
    unit_family = spec_input.get("unit_family")
    selector = spec_input.get("period", "current")

    if selector in {"current", "prior_year_same_period"}:
        target_period = _shift_period(base_period, selector)
        fact, alias = catalog.find_alias_match(base_period.ticker, target_period, aliases, unit_family, base_period.filing_id)
        return ResolvedInput(
            name=name,
            concept_norm=fact["concept_norm"],
            period_key=fact["period_key"],
            period_type=fact["period_type"],
            filing_id=fact["filing_id"],
            unit_canon=fact["unit_canon"],
            numeric_value=fact["value_num_exact"],
            text_value=fact["value_text"],
            fact_evidence_ids=[fact["fact_evidence_id"]],
            trace=[
                {
                    "kind": "fact",
                    "input": name,
                    "alias": alias,
                    "fact_evidence_id": fact["fact_evidence_id"],
                    "period_key": fact["period_key"],
                    "derived": False,
                }
            ],
        )

    if selector == "current_period_end_asof":
        return _resolve_asof_input(catalog, base_period, name, aliases, unit_family, selector, year_delta=0)

    if selector == "prior_year_period_end_asof":
        return _resolve_asof_input(catalog, base_period, name, aliases, unit_family, selector, year_delta=-1)

    if selector == "current_quarter":
        if base_period.fiscal_year is None or base_period.fiscal_quarter is None:
            raise Refusal("PERIOD_NOT_SUPPORTED", "Current-quarter selector requires fiscal year and quarter.")
        resolved, alias = catalog.derive_quarter_value(
            base_period.ticker,
            aliases,
            base_period.fiscal_year,
            base_period.fiscal_quarter,
            unit_family,
            base_period.filing_id,
        )
        resolved.name = name
        resolved.trace.insert(0, {"kind": "selector", "selector": selector, "alias": alias})
        return resolved

    if selector == "previous_quarter":
        target_period = _shift_period(base_period, selector)
        resolved, alias = catalog.derive_quarter_value(
            base_period.ticker,
            aliases,
            target_period.fiscal_year,
            target_period.fiscal_quarter,
            unit_family,
            base_period.filing_id,
        )
        resolved.name = name
        resolved.trace.insert(0, {"kind": "selector", "selector": selector, "alias": alias})
        return resolved

    raise Refusal("PERIOD_NOT_SUPPORTED", f"Unsupported input selector: {selector}")


def _expect_numeric(resolved_inputs: Dict[str, ResolvedInput], name: str) -> ResolvedInput:
    resolved = resolved_inputs[name]
    if resolved.numeric_value is None:
        raise Refusal("MISSING_INPUT", f"Input {name} is not numeric.")
    return resolved


def evaluate_formula(
    formula: Dict[str, Any],
    resolved_inputs: Dict[str, ResolvedInput],
    catalog: Catalog,
    base_period: PeriodDescriptor,
) -> EvalValue:
    """Evaluate a metric formula from resolved inputs and operators.
    
    Parameters
    ----------
    formula : Dict[str, Any]
        Formula expression to evaluate against resolved metric inputs.
    resolved_inputs : Dict[str, ResolvedInput]
        Resolved metric inputs keyed by input name.
    catalog : Catalog
        Catalog object used to resolve canonical facts and period descriptors.
    base_period : PeriodDescriptor
        Reference period used to resolve shifted or comparative inputs.
    
    Returns
    -------
    EvalValue
        Value returned by this operation.
    
    Raises
    ------
    Refusal
        INCOMPATIBLE_UNITS.
    Refusal
        PERIOD_NOT_SUPPORTED.
    Refusal
        ZERO_DENOMINATOR.
    """
    op = formula["op"]

    if op == "input":
        resolved = _expect_numeric(resolved_inputs, formula["name"])
        return EvalValue(
            value=resolved.numeric_value,
            unit_canon=resolved.unit_canon,
            evidence_ids=list(resolved.fact_evidence_ids),
            derivation_steps=list(resolved.trace),
        )

    if op == "abs":
        inner = evaluate_formula(formula["value"], resolved_inputs, catalog, base_period)
        return EvalValue(
            value=abs(inner.value),
            unit_canon=inner.unit_canon,
            evidence_ids=inner.evidence_ids,
            derivation_steps=inner.derivation_steps + [{"kind": "abs"}],
        )

    if op in {"add", "subtract"}:
        left = evaluate_formula(formula["left"], resolved_inputs, catalog, base_period)
        right = evaluate_formula(formula["right"], resolved_inputs, catalog, base_period)
        if left.unit_canon != right.unit_canon:
            raise Refusal("INCOMPATIBLE_UNITS", "Addition/subtraction inputs must share the same unit family.")
        value = left.value + right.value if op == "add" else left.value - right.value
        return EvalValue(
            value=value,
            unit_canon=left.unit_canon,
            evidence_ids=left.evidence_ids + right.evidence_ids,
            derivation_steps=left.derivation_steps + right.derivation_steps + [{"kind": op}],
        )

    if op == "sum":
        args = [evaluate_formula(arg, resolved_inputs, catalog, base_period) for arg in formula["args"]]
        units = {arg.unit_canon for arg in args}
        if len(units) > 1:
            raise Refusal("INCOMPATIBLE_UNITS", "Summation inputs must share the same unit family.")
        total = sum((arg.value for arg in args), start=Decimal("0"))
        evidence_ids = [evidence_id for arg in args for evidence_id in arg.evidence_ids]
        steps = [step for arg in args for step in arg.derivation_steps] + [{"kind": "sum"}]
        return EvalValue(total, next(iter(units)), evidence_ids, steps)

    if op == "average":
        args = [evaluate_formula(arg, resolved_inputs, catalog, base_period) for arg in formula["args"]]
        if not args:
            raise Refusal("PERIOD_NOT_SUPPORTED", "Average requires at least one input.")
        units = {arg.unit_canon for arg in args}
        if len(units) > 1:
            raise Refusal("INCOMPATIBLE_UNITS", "Average inputs must share the same unit family.")
        total = sum((arg.value for arg in args), start=Decimal("0"))
        evidence_ids = [evidence_id for arg in args for evidence_id in arg.evidence_ids]
        steps = [step for arg in args for step in arg.derivation_steps] + [{"kind": "average", "count": len(args)}]
        return EvalValue(total / Decimal(len(args)), next(iter(units)), evidence_ids, steps)

    if op == "divide":
        left = evaluate_formula(formula["left"], resolved_inputs, catalog, base_period)
        right = evaluate_formula(formula["right"], resolved_inputs, catalog, base_period)
        if right.value == 0:
            raise Refusal("ZERO_DENOMINATOR", "Division by zero while evaluating metric.")
        return EvalValue(
            value=left.value / right.value,
            unit_canon=formula.get("output_unit", "pure"),
            evidence_ids=left.evidence_ids + right.evidence_ids,
            derivation_steps=left.derivation_steps + right.derivation_steps + [{"kind": "divide"}],
        )

    if op == "rollup_sum":
        component_names = formula["components"]
        component_results = [evaluate_formula({"op": "input", "name": name}, resolved_inputs, catalog, base_period) for name in component_names]
        units = {result.unit_canon for result in component_results}
        if len(units) > 1:
            raise Refusal("INCOMPATIBLE_UNITS", "Rollup inputs must share the same unit family.")
        component_concepts = [resolved_inputs[name].concept_norm for name in component_names]
        source = catalog.calc_edge_source(base_period.filing_id, formula.get("target_aliases", []), component_concepts)
        total = sum((result.value for result in component_results), start=Decimal("0"))
        evidence_ids = [evidence_id for result in component_results for evidence_id in result.evidence_ids]
        steps = [step for result in component_results for step in result.derivation_steps] + [
            {
                "kind": "rollup_sum",
                "source": source,
                "target_aliases": formula.get("target_aliases", []),
                "components": component_concepts,
            }
        ]
        return EvalValue(total, next(iter(units)), evidence_ids, steps)

    raise Refusal("PERIOD_NOT_SUPPORTED", f"Unsupported formula op: {op}")


def _build_answer_object(
    spec: Dict[str, Any],
    base_period: PeriodDescriptor,
    resolved_inputs: Dict[str, ResolvedInput],
    evaluation: EvalValue,
) -> Dict[str, Any]:
    answer_id = _stable_digest(spec["id"], base_period.filing_id, base_period.period_key, GENERATOR_VERSION)
    return {
        "answer_id": answer_id,
        "metric_spec_id": spec["id"],
        "generator_version": GENERATOR_VERSION,
        "filing_id": base_period.filing_id,
        "ticker": base_period.ticker,
        "period": {
            "period_key": base_period.period_key,
            "period_type": base_period.period_type,
            "fiscal_year": base_period.fiscal_year,
            "fiscal_quarter": base_period.fiscal_quarter,
            "is_ytd": base_period.is_ytd,
        },
        "status": "OK",
        "result": {
            "value": str(evaluation.value),
            "unit": spec.get("output_unit"),
        },
        "input_facts": {
            name: {
                "concept_norm": resolved.concept_norm,
                "period_key": resolved.period_key,
                "unit_canon": resolved.unit_canon,
                "fact_evidence_ids": resolved.fact_evidence_ids,
                "derived": resolved.derived,
            }
            for name, resolved in resolved_inputs.items()
        },
        "derivation_steps": evaluation.derivation_steps,
        "evidence_ids": sorted(dict.fromkeys(evaluation.evidence_ids)),
        "refusal_code": None,
    }


def _build_refusal_object(spec: Dict[str, Any], base_period: PeriodDescriptor, code: str, message: str) -> Dict[str, Any]:
    answer_id = _stable_digest(spec["id"], base_period.filing_id, base_period.period_key, code, GENERATOR_VERSION)
    return {
        "answer_id": answer_id,
        "metric_spec_id": spec["id"],
        "generator_version": GENERATOR_VERSION,
        "filing_id": base_period.filing_id,
        "ticker": base_period.ticker,
        "period": {
            "period_key": base_period.period_key,
            "period_type": base_period.period_type,
            "fiscal_year": base_period.fiscal_year,
            "fiscal_quarter": base_period.fiscal_quarter,
            "is_ytd": base_period.is_ytd,
        },
        "status": "REFUSAL",
        "result": None,
        "input_facts": {},
        "derivation_steps": [],
        "evidence_ids": [],
        "refusal_code": code,
        "refusal_message": message,
    }


def evaluate_metric_spec(catalog: Catalog, spec: Dict[str, Any], period: PeriodDescriptor) -> Dict[str, Any]:
    """Evaluate a metric spec for a filing period.
    
    Parameters
    ----------
    catalog : Catalog
        Catalog object used to resolve canonical facts and period descriptors.
    spec : Dict[str, Any]
        Metric specification payload.
    period : PeriodDescriptor
        Period descriptor with reporting boundaries and fiscal metadata.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with output fields for this evaluation.
    """
    if period.period_type not in spec["applicable_period_types"]:
        return _build_refusal_object(
            spec,
            period,
            "PERIOD_NOT_SUPPORTED",
            f"Metric {spec['id']} does not apply to period type {period.period_type}.",
        )

    try:
        resolved_inputs = {
            input_spec["name"]: resolve_input(catalog, period, input_spec)
            for input_spec in spec["required_inputs"]
        }
        evaluation = evaluate_formula(spec["formula"], resolved_inputs, catalog, period)
        return _build_answer_object(spec, period, resolved_inputs, evaluation)
    except Refusal as refusal:
        return _build_refusal_object(spec, period, refusal.code, refusal.message)


def get_period_descriptor(catalog: Catalog, filing_id: str, period_key: str) -> PeriodDescriptor:
    """Return the period descriptor for the filing and period key.
    
    Parameters
    ----------
    catalog : Catalog
        Catalog object used to resolve canonical facts and period descriptors.
    filing_id : str
        Canonical filing identifier used by manifests and derived artifacts. e.g., '0000320193-2025-10K'
    period_key : str
        Canonical period key such as 'FY2025' or 'Q1FY2026'.
    
    Returns
    -------
    PeriodDescriptor
        Period descriptor for the filing and period key.
    
    Raises
    ------
    KeyError
        No period {...} found for filing {...}.
    """
    for period in catalog.periods_for_filing(filing_id):
        if period.period_key == period_key:
            return period
    raise KeyError(f"No period {period_key!r} found for filing {filing_id!r}")


def answer_metric_spec(conn: sqlite3.Connection, filing_id: str, metric_spec_id: str, period_key: str) -> Dict[str, Any]:
    """Generate a structured answer for one metric spec and period.
    
    Parameters
    ----------
    conn : sqlite3.Connection
        SQLite connection for the corpus database.
    filing_id : str
        Canonical filing identifier used by manifests and derived artifacts. e.g., '0000320193-2025-10K'
    metric_spec_id : str
        Metric specification identifier from the metric catalog. e.g., 'revenue.total'
    period_key : str
        Canonical period key such as 'FY2025' or 'Q1FY2026'.
    
    Returns
    -------
    Dict[str, Any]
        Structured response payload for downstream execution or evaluation.
    
    Raises
    ------
    KeyError
        Unknown metric spec id: {...}.
    """
    catalog = Catalog(conn, filing_id=filing_id)
    specs = load_metric_specs_by_id()
    try:
        spec = specs[metric_spec_id]
    except KeyError as error:
        raise KeyError(f"Unknown metric spec id: {metric_spec_id}") from error

    period = get_period_descriptor(catalog, filing_id, period_key)
    return evaluate_metric_spec(catalog, spec, period)


def generate_answer_objects(conn: sqlite3.Connection, filing_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Generate metric answer objects from canonical filing facts.
    
    Parameters
    ----------
    conn : sqlite3.Connection
        SQLite connection for the corpus database.
    filing_id : Optional[str], optional
        Canonical filing identifier used by manifests and derived artifacts. e.g., '0000320193-2025-10K'
    
    Returns
    -------
    List[Dict[str, Any]]
        List of records for metric answer objects from canonical filing facts.
    """
    catalog = Catalog(conn, filing_id=filing_id) if filing_id else Catalog(conn)
    specs = load_metric_specs()
    filing_ids = [filing_id] if filing_id else sorted(catalog.filings)
    answers: List[Dict[str, Any]] = []

    for current_filing_id in filing_ids:
        periods = catalog.periods_for_filing(current_filing_id)
        for period in periods:
            for spec in specs:
                if period.period_type not in spec["applicable_period_types"]:
                    continue
                answers.append(evaluate_metric_spec(catalog, spec, period))

    return answers


def write_answer_objects(conn: sqlite3.Connection, output_path: str, filing_id: Optional[str] = None) -> int:
    """Write generated metric answer objects to a JSONL file.
    
    Parameters
    ----------
    conn : sqlite3.Connection
        SQLite connection for the corpus database.
    output_path : str
        Destination path for generated output artifacts. e.g., auditops-output.jsonl
    filing_id : Optional[str], optional
        Canonical filing identifier used by manifests and derived artifacts. e.g., '0000320193-2025-10K'
    
    Returns
    -------
    int
        Number of records written by this function.
    """
    answers = generate_answer_objects(conn, filing_id=filing_id)
    with open(output_path, "w", encoding="utf-8") as handle:
        for answer in answers:
            handle.write(json.dumps(answer, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return len(answers)


def generate_answer_objects_from_db(db_path: str, output_path: str, filing_id: Optional[str] = None) -> int:
    """Generate and persist answer objects from a database path.
    
    Parameters
    ----------
    db_path : str
        Filesystem path to the SQLite corpus database. e.g., auditops.sqlite
    output_path : str
        Destination path for generated output artifacts. e.g., auditops-output.jsonl
    filing_id : Optional[str], optional
        Canonical filing identifier used by manifests and derived artifacts. e.g., '0000320193-2025-10K'
    
    Returns
    -------
    int
        Number of generated objects.
    """
    conn = connect_db(db_path)
    try:
        return write_answer_objects(conn, output_path, filing_id=filing_id)
    finally:
        conn.close()
