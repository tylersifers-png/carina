"""Market-history acceptance, separate from financial mapping QA."""

from __future__ import annotations
from collections import Counter
from functools import lru_cache
from datetime import date
from pathlib import Path
import bz2
import hashlib
import json

import numpy as np
import pandas as pd
import exchange_calendars as xcals

from carina.registry import BENCHMARKS, BENCHMARK_STARTS, company
from carina import dataset
from carina import settings as decisions_input

VERSION = "company-market-quality-1.2"
VALUES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def policy():
    return decisions_input.thresholds("market")


def acknowledgements():
    """Recorded decisions about individual flags, each bound to its exact evidence."""
    return decisions_input.reviews("market_acknowledgements")


def documented_events(symbol):
    """Known issuer events remain reviewable even when vendor actions are absent."""
    result = []
    for event in decisions_input.reviews("market_events"):
        if set(event) != {
            "symbol",
            "session_dates",
            "event_type",
            "description",
            "evidence_refs",
        }:
            raise ValueError("Invalid documented market event fields")
        days = event["session_dates"]
        if (
            not isinstance(days, list)
            or not days
            or len(days) != len(set(days))
            or any(
                not isinstance(d, str) or date.fromisoformat(d).isoformat() != d
                for d in days
            )
            or not isinstance(event["event_type"], str)
            or not event["event_type"].strip()
            or not isinstance(event["description"], str)
            or not event["description"].strip()
            or not isinstance(event["evidence_refs"], list)
            or not event["evidence_refs"]
            or any(
                not isinstance(r, str) or not r.strip() for r in event["evidence_refs"]
            )
        ):
            raise ValueError("Invalid documented market event evidence")
        if event["symbol"] == symbol:
            company(symbol)
            result.append(event)
    return sorted(result, key=lambda e: (sorted(e["session_dates"]), e["event_type"]))


@lru_cache(maxsize=16)
def sessions(start, end):
    return (
        xcals.get_calendar("XNYS", start=start, end=end)
        .sessions.strftime("%Y-%m-%d")
        .tolist()
    )


def evaluate(
    daily,
    symbols,
    market_sources,
    market_updates=None,
    *,
    benchmarks=None,
    acknowledged=None,
    identity_tables=None,
):
    settings = policy()
    records = acknowledgements() if acknowledged is None else list(acknowledged)
    decisions = {d["flag_id"]: d for d in records}
    if len(decisions) != len(records):
        raise ValueError("Duplicate market acknowledgements")
    flags = []
    histories = {}

    def add(rule, symbol, evidence, severity="REVIEW"):
        flags.append(
            dict(
                id=digest([rule, symbol]),
                rule=rule,
                symbol=symbol,
                severity=severity,
                evidence=evidence,
                evidence_sha256=digest(evidence),
                status="OPEN",
            )
        )

    # A store can predate a newly registered benchmark.  Assess the benchmarks
    # actually declared by its manifest, rather than treating a registry
    # addition as a retroactive failure of an otherwise valid store.
    benchmarks = list(BENCHMARKS) if benchmarks is None else list(benchmarks)
    if len(benchmarks) != len(set(benchmarks)) or not set(benchmarks) <= set(
        BENCHMARKS
    ):
        raise ValueError("Invalid benchmark set for market assessment")
    empty_daily = daily.iloc[:0]
    daily_by_symbol = {
        symbol: frame.sort_values("session_date").reset_index(drop=True)
        for symbol, frame in daily.groupby("symbol", sort=False)
    }
    securities_by_symbol = (
        {
            symbol: frame
            for symbol, frame in identity_tables["securities"].groupby(
                "symbol", sort=False
            )
        }
        if identity_tables is not None
        else {}
    )
    companies_by_symbol = (
        {
            symbol: frame
            for symbol, frame in identity_tables["companies"].groupby(
                "symbol", sort=False
            )
        }
        if identity_tables is not None
        else {}
    )
    ends = {
        s: str(daily_by_symbol.get(s, empty_daily).session_date.max())
        for s in benchmarks
    }
    for symbol in [*symbols, *benchmarks]:
        q = daily_by_symbol.get(symbol, empty_daily)
        if q.empty:
            add("MISSING_MARKET_HISTORY", symbol, dict(rows=0), "BLOCK")
            continue
        expected_id = company(symbol)["issuer_id"] if symbol in symbols else ""
        expected_security = "yahoo:" + symbol
        wrong_ids = any(
            field in q and not q[field].fillna("").eq(value).all()
            for field, value in [
                ("issuer_id", expected_id),
                ("security_id", expected_security),
            ]
        )
        if identity_tables is not None:
            security = securities_by_symbol.get(
                symbol, identity_tables["securities"].iloc[:0]
            )
            wrong_ids = (
                wrong_ids
                or len(security) != 1
                or not security.security_id.eq(expected_security).all()
                or not security.issuer_id.fillna("").eq(expected_id).all()
            )
            if symbol in symbols:
                issuer = companies_by_symbol.get(
                    symbol, identity_tables["companies"].iloc[:0]
                )
                wrong_ids = (
                    wrong_ids
                    or len(issuer) != 1
                    or not issuer.issuer_id.eq(expected_id).all()
                    or not issuer.issuer_cik.eq(company(symbol)["issuer_cik"]).all()
                )
        if wrong_ids:
            add(
                "ISSUER_SECURITY_MAPPING",
                symbol,
                dict(
                    expected_issuer_id=expected_id,
                    expected_security_id=expected_security,
                ),
                "BLOCK",
            )
            continue
        numbers = q[VALUES].apply(pd.to_numeric, errors="coerce")
        bad = (
            ~np.isfinite(numbers.to_numpy()).all(axis=1)
            | (numbers[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any(axis=1)
            | (numbers[["Volume", "Dividends", "Stock Splits"]] < 0).any(axis=1)
            | (numbers.Low > numbers[["Open", "Close"]].min(axis=1))
            | (numbers.High < numbers[["Open", "Close"]].max(axis=1))
        )
        if bad.any() or q.session_date.duplicated().any():
            add(
                "INVALID_MARKET_HISTORY",
                symbol,
                dict(
                    invalid_rows=int(bad.sum()),
                    duplicate_sessions=int(q.session_date.duplicated().sum()),
                ),
                "BLOCK",
            )
            continue
        source = market_sources.get(symbol, {})
        receipt = source.get("receipt", {})
        updates = (market_updates or {}).get(symbol, [])
        end = (
            updates[-1]["receipt"]["last_session"]
            if updates
            else receipt.get("last_session")
        )
        start = (
            company(symbol)["requested_start"]
            if symbol in symbols
            else BENCHMARK_STARTS[symbol]
        )
        if (
            not end
            or receipt.get("metadata", {}).get("symbol") != symbol
            or receipt.get("metadata", {}).get("currency") != "USD"
        ):
            add(
                "MARKET_SOURCE_IDENTITY",
                symbol,
                dict(
                    source_symbol=receipt.get("metadata", {}).get("symbol"),
                    currency=receipt.get("metadata", {}).get("currency"),
                ),
                "BLOCK",
            )
            continue
        expected = set(sessions(start, end))
        observed = set(q.session_date)
        if expected != observed:
            add(
                "MARKET_SESSION_GAP",
                symbol,
                dict(
                    missing=sorted(expected - observed),
                    unexpected=sorted(observed - expected),
                ),
                "BLOCK",
            )
        r = numbers["Adj Close"].pct_change(fill_method=None)
        jumps = r.abs() > settings["large_daily_return"]
        if jumps.any():
            selected = q.loc[jumps, ["session_date", *VALUES]].copy()
            selected["adjusted_return"] = r[jumps].to_numpy()
            add(
                "LARGE_DAILY_RETURN",
                symbol,
                dict(observations=selected.to_dict("records")),
            )
        # A spin-off can be represented as both a split-like adjustment and a
        # distribution. Vendor dividend reconciliation can then pass while
        # double counting the same economic event, even below the jump threshold.
        # Genuine cash dividends coinciding with splits also require context;
        # this is a review trigger, never an automatic price/return correction.
        combined_action = numbers.Dividends.ne(0) & numbers["Stock Splits"].ne(0)
        if combined_action.any():
            add(
                "SIMULTANEOUS_SPLIT_AND_DIVIDEND",
                symbol,
                dict(
                    observations=q.loc[
                        combined_action, ["session_date", *VALUES]
                    ].to_dict("records")
                ),
            )
        # Issuer filings can establish a distribution that Yahoo omits entirely.
        # Registration creates a review flag, never a synthetic dividend/split or
        # an automatic correction/exclusion. Only the affected company's contract
        # binds this evidence, so later registry edits cannot reopen other issuers.
        events = (
            [
                e
                for e in documented_events(symbol)
                if any(start <= d <= end for d in e["session_dates"])
            ]
            if symbol in symbols
            else []
        )
        if events:
            days = sorted(
                {d for e in events for d in e["session_dates"] if start <= d <= end}
            )
            missing = sorted(set(days) - observed)
            positions = q.index[q.session_date.isin(days)]
            prior = sorted({i - 1 for i in positions if i > 0})
            add(
                "DOCUMENTED_CORPORATE_ACTION",
                symbol,
                dict(
                    events=events,
                    observations=q.loc[positions, ["session_date", *VALUES]].to_dict(
                        "records"
                    ),
                    prior_observations=q.loc[prior, ["session_date", *VALUES]].to_dict(
                        "records"
                    ),
                    missing_sessions=missing,
                ),
                "BLOCK" if missing else "REVIEW",
            )
        # Yahoo back-adjusts dividends: P(t)/P(t-1)/(1-D(t)/P(t-1)).
        # Split-adjusted Close should not be mechanically divided again by splits.
        expected_growth = (numbers.Close / numbers.Close.shift(1)) / (
            1 - numbers.Dividends / numbers.Close.shift(1)
        )
        mismatch = (r + 1 - expected_growth).abs() > settings["adjustment_tolerance"]
        if mismatch.any():
            events = q.loc[mismatch, ["session_date", *VALUES]].copy()
            events["adjusted_growth"] = (r + 1)[mismatch].to_numpy()
            events["expected_vendor_growth"] = expected_growth[mismatch].to_numpy()
            add(
                "ADJUSTMENT_RECONCILIATION",
                symbol,
                dict(observations=events.to_dict("records")),
            )
        for field, actual in [
            ("price_return", numbers.Close.pct_change(fill_method=None)),
            ("vendor_adjusted_return", r),
        ]:
            if field not in q or not np.allclose(
                pd.to_numeric(q[field], errors="coerce"),
                actual,
                rtol=1e-9,
                atol=1e-10,
                equal_nan=True,
            ):
                add("RETURN_COLUMN_MISMATCH", symbol, dict(field=field), "BLOCK")
        if symbol in symbols:
            first, last = str(q.session_date.min()), str(q.session_date.max())
            histories[symbol] = dict(
                issuer_id=company(symbol)["issuer_id"],
                first=first,
                last=last,
                rows=len(q),
                dividend_events=int(numbers.Dividends.ne(0).sum()),
                split_events=int(numbers["Stock Splits"].ne(0).sum()),
            )
            benchmark_end = min(ends.values())
            if last > benchmark_end:
                add(
                    "BENCHMARK_TAIL_UNMATCHED",
                    symbol,
                    dict(
                        company_last=last,
                        benchmark_last=benchmark_end,
                        scope="Exclude unmatched tail until explicit benchmark period update",
                    ),
                )
    # An acknowledgement covers one flag and its exact evidence; if the evidence
    # changes the flag reopens. Block errors are never waivable.
    for flag in flags:
        d = decisions.get(flag["id"], {})
        if (
            flag["severity"] != "BLOCK"
            and d.get("evidence_sha256") == flag["evidence_sha256"]
            and d.get("rationale")
            and d.get("evidence_refs")
            and d.get("disposition") in {"validated", "expected", "excluded"}
        ):
            flag.update(status="REVIEWED", decision=d)
    open_flags = [f for f in flags if f["status"] == "OPEN"]
    flagged = sorted({f["symbol"] for f in open_flags} & set(histories))
    clean = sorted(set(histories) - set(flagged))
    status = (
        "BLOCKED"
        if any(f["severity"] == "BLOCK" for f in open_flags)
        else "FLAGGED"
        if open_flags
        else "CLEAN"
    )
    issuers = {company(s)["issuer_id"] for s in clean}
    return dict(
        schema=VERSION,
        scope="MARKET_DATA",
        status=status,
        policy_sha256=digest(settings),
        acknowledgements_sha256=digest(records),
        clean_companies=clean,
        clean_unique_companies=len(issuers),
        flagged_companies=flagged,
        open_flags=len(open_flags),
        flag_counts=dict(Counter(f["rule"] for f in open_flags)),
        histories=histories,
        flags=flags,
        excluded_scopes=[
            dict(
                symbol=f["symbol"],
                rule=f["rule"],
                flag_id=f["id"],
                evidence=f["evidence"],
                rationale=f["decision"]["rationale"],
                **(
                    {"scope": f["decision"]["scope"]}
                    if isinstance(f["decision"].get("scope"), dict)
                    else {}
                ),
            )
            for f in flags
            if f.get("decision", {}).get("disposition") == "excluded"
        ],
        limitations=[
            "Vendor adjusted prices/dividends/splits, not independently certified total returns",
            "Current selected survivors, not a historical constituent/delisted universe",
            "Security/predecessor/share-class continuity remains vendor supplied",
            "Financial quality is tracked separately and does not approve financial model inputs",
        ],
    )


def assess(path, *, acknowledged=None):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_bytes())
    sources = json.loads(bz2.decompress((path / "sources.json.bz2").read_bytes()))
    return evaluate(
        dataset.read_table(path, "daily_bars"),
        manifest["companies"],
        sources.get("market", {}),
        sources.get("market_updates", {}),
        benchmarks=manifest["benchmarks"],
        acknowledged=acknowledged,
        identity_tables={
            name: dataset.read_table(path, name)
            for name in ["companies", "securities"]
        },
    )


def summary(report):
    return {
        k: report[k]
        for k in [
            "schema",
            "scope",
            "status",
            "clean_unique_companies",
            "clean_companies",
            "flagged_companies",
            "open_flags",
            "flag_counts",
        ]
    }


def require_expansion_ready(plan):
    base = plan["base"]
    new = set(plan["request"]["companies"]) - set(base["companies"])
    if (
        new
        and plan["request"]["add_companies"]
        and base["companies"]
        and base.get("market_quality_gate", {}).get("status") == "BLOCKED"
    ):
        raise ValueError(
            "EXPANSION_BLOCKED: resolve blocking market flags before adding companies"
        )

