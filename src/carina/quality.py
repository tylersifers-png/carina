"""Evidence-bound financial mapping flags and expansion acceptance; no source fetching."""

from __future__ import annotations
from collections import Counter
from datetime import date
from itertools import groupby
import bz2
import hashlib
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from carina.registry import metrics_for
from carina import dataset
from carina import settings as decisions

VERSION = "company-quality-1.1"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def settings():
    return decisions.thresholds("financial")


def acknowledgements():
    """Recorded decisions about individual flags, each bound to its exact evidence."""
    return decisions.reviews("financial_acknowledgements")


def dataset_digest(manifest):
    return digest(
        dict(
            files=manifest["files"],
            companies=sorted(manifest["companies"]),
            benchmarks=sorted(manifest["benchmarks"]),
        )
    )


def evaluate(
    facts,
    versions,
    daily,
    symbols,
    *,
    policy=None,
    previous=None,
    acknowledged=None,
    data_digest="",
    retained_flags=(),
    filings=None,
    normalization_issues=None,
):
    """Assess all tracked issuers, including those not refreshed in this run."""
    policy = settings() if policy is None else policy
    records = acknowledgements() if acknowledged is None else list(acknowledged)
    flags = []

    def add(rule, symbol, metric, basis, evidence, severity="REVIEW", period=""):
        identity = dict(
            rule=rule, symbol=symbol, metric=metric, basis=basis, period=period
        )
        flags.append(
            dict(
                id=digest(identity),
                **identity,
                severity=severity,
                evidence=evidence,
                evidence_sha256=digest(evidence),
                status="OPEN",
            )
        )

    empty_facts = facts.iloc[:0]
    empty_versions = versions.iloc[:0]
    empty_daily = daily.iloc[:0]
    facts_by_symbol = {
        symbol: frame for symbol, frame in facts.groupby("symbol", sort=False)
    }
    versions_by_symbol = {
        symbol: frame for symbol, frame in versions.groupby("symbol", sort=False)
    }
    daily_by_symbol = {
        symbol: frame for symbol, frame in daily.groupby("symbol", sort=False)
    }
    filings_by_symbol = (
        {symbol: frame for symbol, frame in filings.groupby("symbol", sort=False)}
        if filings is not None and len(filings)
        else {}
    )
    previous_by_symbol = (
        {symbol: frame for symbol, frame in previous.groupby("symbol", sort=False)}
        if previous is not None
        else {}
    )

    for symbol in symbols:
        selected = metrics_for(symbol)
        q = facts_by_symbol.get(symbol, empty_facts).copy()
        q["number"] = pd.to_numeric(q.value, errors="coerce")
        bad = (
            q.number.isna()
            | ~np.isfinite(q.number)
            | q.status.fillna("").ne("REPORTED")
        )
        if bad.any():
            add(
                "INVALID_FINANCIAL_FACT",
                symbol,
                "*",
                "*",
                dict(rows=int(bad.sum())),
                "BLOCK",
            )
        for metric, (unit, _, _) in selected.items():
            m = q[q.metric == metric]
            if len(m) and m.unit.fillna("").ne(unit).any():
                add(
                    "UNIT_MISMATCH",
                    symbol,
                    metric,
                    "*",
                    dict(
                        expected_unit=unit,
                        observed_units=sorted(
                            m.unit.fillna("<missing>").unique().tolist()
                        ),
                    ),
                    "BLOCK",
                )
        rejected = [
            x
            for x in (normalization_issues or {}).get(symbol, [])
            if x["reason"] != "UNSUPPORTED_FORM"
        ]
        for (metric, reason), items in groupby(
            sorted(rejected, key=lambda x: (x["metric"], x["reason"])),
            key=lambda x: (x["metric"], x["reason"]),
        ):
            items = list(items)
            add(
                "NORMALIZATION_REJECTION",
                symbol,
                metric,
                "*",
                dict(
                    reason=reason,
                    count=len(items),
                    observations_sha256=digest(
                        sorted(x.get("observation_sha256", x["fact_id"]) for x in items)
                    ),
                ),
                period=reason,
            )
        issuer_last = q.end.max() if len(q) else ""
        expected_end = ""
        statement = q[q.metric.ne("shares_outstanding")]
        actual_end = statement.end.max() if len(statement) else ""
        if symbol in filings_by_symbol:
            index = filings_by_symbol[symbol]
            index = index[index.form.isin(["10-K", "10-K/A", "10-Q", "10-Q/A"])]
            if len(index):
                expected_end = index.reportDate.max()
        issuer_behind = bool(
            expected_end and (not actual_end or actual_end < expected_end)
        )
        if issuer_behind:
            add(
                "FILING_FACT_LAG",
                symbol,
                "statement_coverage",
                "*",
                dict(
                    latest_index_report_end=expected_end,
                    latest_mapped_statement_end=actual_end,
                    note="API processing delay, stale source, or mapping changes require review",
                ),
                period=expected_end,
            )
        issuer_versions = versions_by_symbol.get(symbol, empty_versions)
        fiscal = set(
            issuer_versions.loc[
                issuer_versions.metric.eq("net_income")
                & issuer_versions.period_kind.eq("ANNUAL")
                & issuer_versions.form.isin(["10-K", "10-K/A"]),
                "end",
            ]
        )
        market = daily_by_symbol.get(symbol, empty_daily)
        prices = ["Open", "High", "Low", "Close", "Adj Close"]
        if len(market):
            numeric = market[prices].apply(pd.to_numeric, errors="coerce")
            invalid = (
                numeric.isna().any(axis=1)
                | ~np.isfinite(numeric).all(axis=1)
                | numeric.le(0).any(axis=1)
            )
            if invalid.any() or market.session_date.duplicated().any():
                add(
                    "INVALID_MARKET_DATA",
                    symbol,
                    "daily_prices",
                    "vendor",
                    dict(
                        invalid_rows=int(invalid.sum()),
                        duplicate_dates=int(market.session_date.duplicated().sum()),
                    ),
                    "BLOCK",
                )
        splits = market[market["Stock Splits"] != 0] if len(market) else market
        for metric in selected:
            m = q[q.metric == metric]
            if m.empty:
                add(
                    "MAPPED_FIELD_MISSING",
                    symbol,
                    metric,
                    "*",
                    dict(
                        selected=True,
                        observed_dividend_events=(
                            int((market["Dividends"] != 0).sum())
                            if metric == "dividends_paid" and "Dividends" in market
                            else None
                        ),
                        unit=selected[metric][0],
                        mapped_tags=[
                            f"{ns}:{tag}" for ns, tag, _ in selected[metric][2]
                        ],
                    ),
                )
                continue
            end = m.end.max()
            lag = (date.fromisoformat(issuer_last) - date.fromisoformat(end)).days
            if lag > policy["stale_after_days"]:
                add(
                    "MAPPED_FIELD_STALE",
                    symbol,
                    metric,
                    "*",
                    dict(
                        last_end=end,
                        issuer_latest_end=issuer_last,
                        lag_days=lag,
                        threshold_days=policy["stale_after_days"],
                    ),
                )
            # Catch newly missing quarterly mappings before the long-staleness
            # threshold, without imposing quarter cadence on sparse annual fields.
            if (
                expected_end
                and end < expected_end
                and not issuer_behind
                and metric != "shares_outstanding"
                and lag <= policy["stale_after_days"]
            ):
                dates = sorted(set(m.end))
                recent = [
                    date.fromisoformat(x)
                    for x in dates
                    if (date.fromisoformat(expected_end) - date.fromisoformat(x)).days
                    <= 730
                ]
                gaps = [(b - a).days for a, b in zip(recent, recent[1:])]
                if len(recent) >= 4 and np.median(gaps) <= 120:
                    add(
                        "FILING_PERIOD_MAPPING_GAP",
                        symbol,
                        metric,
                        "*",
                        dict(
                            latest_index_report_end=expected_end,
                            last_mapped_end=end,
                            observed_median_spacing_days=float(np.median(gaps)),
                            note="Historically quarterly mapping has not reached latest indexed period",
                        ),
                        period=expected_end,
                    )
            for basis, b in m.groupby("basis", sort=True):
                annual = b[b.period_kind.eq("ANNUAL") & b.end.isin(fiscal)]
                if len(annual):
                    first, last = annual.end.min(), annual.end.max()
                    missing = sorted(
                        x
                        for x in fiscal
                        if first < x < last and x not in set(annual.end)
                    )
                    if missing:
                        add(
                            "ANNUAL_FLOW_GAP",
                            symbol,
                            metric,
                            basis,
                            dict(
                                first_end=first,
                                last_end=last,
                                missing_fiscal_ends=missing,
                            ),
                        )
                if metric != "shares_outstanding":
                    comparable = (
                        (annual if selected[metric][1] == "duration" else b)
                        .sort_values("end")
                        .drop_duplicates("end")
                    )
                    rows = comparable.to_dict("records")
                    for before, after in zip(rows, rows[1:]):
                        if (
                            date.fromisoformat(after["end"])
                            - date.fromisoformat(before["end"])
                        ).days > 550:
                            continue  # Long coverage gaps are not one-period scale jumps.
                        x, y = float(before["number"]), float(after["number"])
                        if np.isfinite(x) and np.isfinite(y) and x * y > 0:
                            ratio = abs(y / x)
                            if (
                                ratio > policy["value_jump_factor"]
                                or ratio < 1 / policy["value_jump_factor"]
                            ):
                                add(
                                    "VALUE_SCALE_REVIEW",
                                    symbol,
                                    metric,
                                    basis,
                                    dict(
                                        previous_end=before["end"],
                                        previous_value=str(before["value"]),
                                        end=after["end"],
                                        value=str(after["value"]),
                                        ratio=ratio,
                                        note="Possible scale/basis change or genuine economic event; original-source review required",
                                    ),
                                    period=after["end"],
                                )
                if metric == "shares_outstanding":
                    points = b.sort_values("end").drop_duplicates("end")
                    for prior, current in zip(
                        points.to_dict("records"), points.to_dict("records")[1:]
                    ):
                        if prior["number"] > 0 and current["number"] > 0:
                            ratio = float(current["number"] / prior["number"])
                            if (
                                ratio > policy["share_jump_factor"]
                                or ratio < 1 / policy["share_jump_factor"]
                            ):
                                add(
                                    "SHARE_SCALE_ANOMALY",
                                    symbol,
                                    metric,
                                    basis,
                                    dict(
                                        previous_end=prior["end"],
                                        previous_value=str(prior["value"]),
                                        end=current["end"],
                                        value=str(current["value"]),
                                        ratio=ratio,
                                    ),
                                    period=current["end"],
                                )
            if metric in {"shares_outstanding", "diluted_eps"} and len(splits):
                first = pd.Timestamp(m.end.min()) - pd.Timedelta(days=370)
                relevant = splits[pd.to_datetime(splits.session_date) >= first]
                if len(relevant):
                    add(
                        "PER_SHARE_SPLIT_BASIS_UNVERIFIED",
                        symbol,
                        metric,
                        "*",
                        dict(
                            bases=sorted(m.basis.unique().tolist()),
                            splits=[
                                dict(
                                    date=r["session_date"],
                                    factor=float(r["Stock Splits"]),
                                )
                                for _, r in relevant.iterrows()
                            ],
                            normalization="NO_VERIFIED_CONSTANT_SHARE_BASIS",
                        ),
                    )
            if previous is not None:
                issuer_previous = previous_by_symbol.get(symbol, previous.iloc[:0])
                old = issuer_previous[issuer_previous.metric == metric]
                if len(old):
                    cols = ["basis", "unit", "start", "end", "period_kind"]
                    before = {
                        tuple(r) for r in old[cols].itertuples(index=False, name=None)
                    }
                    after = {
                        tuple(r) for r in m[cols].itertuples(index=False, name=None)
                    }
                    lost = sorted(before - after)
                    if lost:
                        add(
                            "HISTORICAL_COVERAGE_LOSS",
                            symbol,
                            metric,
                            "*",
                            dict(
                                lost_periods=len(lost),
                                lost_keys_sha256=digest(lost),
                                examples=[list(x) for x in lost[:5]],
                            ),
                        )
                    before_tags = set(
                        t for tags in old.source_tags for t in tags.split(";")
                    )
                    after_tags = set(
                        t for tags in m.source_tags for t in tags.split(";")
                    )
                    if after_tags - before_tags:
                        add(
                            "NEW_SOURCE_TAG",
                            symbol,
                            metric,
                            "*",
                            dict(
                                new_tags=sorted(after_tags - before_tags),
                                previous_tags=sorted(before_tags),
                            ),
                        )
        balance = q[
            q.metric.isin(["assets", "liabilities", "equity"])
            & q.period_kind.eq("INSTANT")
            & (
                q.basis.eq("reported")
                | (
                    q.metric.eq("equity")
                    & q.basis.eq("including_noncontrolling_interest")
                )
            )
        ]
        # Assets/liabilities cover the consolidated entity, so total equity is
        # the compatible comparator when reported. Keep parent equity separate.
        total_ends = set(
            balance.loc[
                balance.metric.eq("equity")
                & balance.basis.eq("including_noncontrolling_interest"),
                "end",
            ]
        )
        balance = balance[
            ~(
                balance.metric.eq("equity")
                & balance.basis.eq("reported")
                & balance.end.isin(total_ends)
            )
        ]
        if not balance.duplicated(["metric", "end"]).any():
            wide = balance.pivot(index="end", columns="metric", values="number")
            if {"assets", "liabilities", "equity"} <= set(wide):
                wide = wide.dropna()
                difference = (wide.assets - wide.liabilities - wide.equity).abs()
                bad = difference > np.maximum(
                    policy["balance_absolute_tolerance_usd"],
                    wide.assets.abs() * policy["balance_relative_tolerance"],
                )
                if bad.any():
                    add(
                        "BALANCE_IDENTITY_REVIEW",
                        symbol,
                        "assets_liabilities_equity",
                        "reported",
                        dict(
                            periods=int(bad.sum()),
                            ends=wide.index[bad].tolist(),
                            max_difference_usd=float(difference[bad].max()),
                            observations_sha256=digest(
                                balance.loc[balance.end.isin(wide.index[bad])]
                                .reindex(
                                    columns=[
                                        "metric",
                                        "basis",
                                        "end",
                                        "value",
                                        "accession",
                                    ],
                                    fill_value="",
                                )
                                .astype(str)
                                .sort_values(["end", "metric"])
                                .to_dict("records")
                            ),
                            note="Uses reported total equity where available; otherwise parent equity. Mixed reporting vintages may explain differences",
                        ),
                    )
    # Transition flags remain open after publication: the old baseline will no
    # longer be available on the next read. Never let a warning disappear merely
    # because the suspicious candidate became current.
    present = {x["id"] for x in flags}
    for prior in retained_flags:
        if prior["rule"] not in {"HISTORICAL_COVERAGE_LOSS", "NEW_SOURCE_TAG"}:
            continue
        identity = {
            k: prior[k] for k in ("rule", "symbol", "metric", "basis", "period")
        }
        if prior["id"] != digest(identity) or prior["evidence_sha256"] != digest(
            prior["evidence"]
        ):
            raise ValueError("Invalid retained mapping-transition evidence")
        if prior["id"] not in present:
            flag = deepcopy(prior)
            flag["status"] = "OPEN"
            flag.pop("decision", None)
            flags.append(flag)
            present.add(flag["id"])
    # An acknowledgement covers one flag and its exact evidence; if the evidence
    # changes the flag reopens. Block errors are never waivable.
    decisions = {d["flag_id"]: d for d in records}
    if len(decisions) != len(records):
        raise ValueError("Duplicate quality acknowledgements")
    limitations = []
    for flag in flags:
        decision = decisions.get(flag["id"])
        if (
            decision
            and flag["severity"] != "BLOCK"
            and decision.get("evidence_sha256") == flag["evidence_sha256"]
            and decision.get("disposition") in {"validated", "expected", "excluded"}
            and decision.get("rationale")
            and isinstance(decision.get("evidence_refs"), list)
            and decision["evidence_refs"]
            and all(isinstance(x, str) and x.strip() for x in decision["evidence_refs"])
        ):
            flag["status"] = "REVIEWED"
            flag["decision"] = decision
            if decision["disposition"] == "excluded":
                limitations.append(
                    dict(
                        symbol=flag["symbol"],
                        metric=flag["metric"],
                        flag_id=flag["id"],
                        rationale=decision["rationale"],
                    )
                )
    open_flags = [x for x in flags if x["status"] == "OPEN"]
    status = (
        "BLOCKED"
        if any(x["severity"] == "BLOCK" for x in open_flags)
        else "FLAGGED"
        if open_flags
        else "CLEAN"
    )
    return dict(
        schema=VERSION,
        status=status,
        dataset_sha256=data_digest,
        policy_sha256=digest(policy),
        acknowledgements_sha256=digest(records),
        open_flags=len(open_flags),
        flag_counts=dict(sorted(Counter(x["rule"] for x in open_flags).items())),
        limitations=limitations,
        flags=sorted(
            flags,
            key=lambda x: (
                x["symbol"],
                x["rule"],
                x["metric"],
                x["basis"],
                x["period"],
            ),
        ),
    )


def rejection_observations(symbol, issues, item):
    """Fingerprint rejected facts, not mutable response hashes/array positions.

    Preserve raw fact IDs for tracing. If a raw fact cannot be reconstructed,
    retain its capture-specific identity and fail conservatively into review.
    """
    result = deepcopy(issues)
    relevant = [x for x in result if x["reason"] != "UNSUPPORTED_FORM"]
    wanted = {f for x in relevant for f in x["fact_id"].split(";")}
    tags = {
        (ns, tag)
        for metric in {x["metric"] for x in relevant}
        for ns, tag, _ in metrics_for(symbol).get(metric, (None, None, []))[2]
    }
    found = {}
    sha = item.get("receipts", {}).get("facts", {}).get("sha256", "")
    for ns, tag in tags:
        units = (
            item.get("facts", {})
            .get("facts", {})
            .get(ns, {})
            .get(tag, {})
            .get("units", {})
        )
        for unit, values in units.items():
            for i, raw in enumerate(values):
                identity = digest([sha, ns, tag, unit, i])
                if identity in wanted:
                    found[identity] = dict(namespace=ns, tag=tag, unit=unit, raw=raw)
    for issue in relevant:
        ids = issue["fact_id"].split(";")
        if all(identity in found for identity in ids):
            issue["observation_sha256"] = digest(
                sorted(digest(found[identity]) for identity in ids)
            )
    return result


def assess(path, *, previous_path=None, acknowledged=None):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_bytes())
    prior_manifest = (
        json.loads((Path(previous_path) / "manifest.json").read_bytes())
        if previous_path is not None
        else manifest
    )
    retained = prior_manifest.get("quality", {}).get("flags", [])
    previous = (
        dataset.read_table(previous_path, "financial_latest")
        if previous_path is not None
        else None
    )
    coverage = json.loads((path / "coverage.json").read_bytes())
    sources = json.loads(bz2.decompress((path / "sources.json.bz2").read_bytes()))
    return evaluate(
        dataset.read_table(path, "financial_latest"),
        dataset.read_table(path, "fundamentals"),
        dataset.read_table(path, "daily_bars"),
        manifest["companies"],
        previous=previous,
        acknowledged=acknowledged,
        data_digest=dataset_digest(manifest),
        retained_flags=retained,
        filings=dataset.read_table(path, "filings"),
        normalization_issues={
            s: rejection_observations(
                s,
                coverage[s].get("financials", {}).get("excluded_facts", []),
                sources.get("sec_projection", {}).get("companies", {}).get(s, {}),
            )
            for s in manifest["companies"]
        },
    )


def summary(report):
    return {
        k: report[k]
        for k in (
            "schema",
            "status",
            "dataset_sha256",
            "policy_sha256",
            "acknowledgements_sha256",
            "open_flags",
            "flag_counts",
        )
    }


def require_expansion_ready(plan):
    # Financial QA stays visible, but financial mapping warnings do not block
    # adding companies; only blocking market flags do.
    from carina.market_quality import (
        require_expansion_ready as market_gate,
    )

    market_gate(plan)
