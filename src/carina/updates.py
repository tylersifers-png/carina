"""Selective company expansion and period extension for the shared current store."""

from __future__ import annotations

import bz2
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import time

from carina import quality
from carina import market_quality
from carina import store
from carina import common as sec, dataset
from carina import fundamentals as norm
from carina.registry import (
    COMPANIES,
    BENCHMARKS,
    company,
    metrics_for,
    applicability,
    quarantines_for,
)

SCHEMA = "company-update-v1.1"
KEY = [
    "issuer_id",
    "metric",
    "basis",
    "unit",
    "start",
    "end",
    "period_kind",
    "accession",
]
TABLES = [
    "companies",
    "securities",
    "daily_bars",
    "fundamentals",
    "financial_latest",
    "filings",
    "field_applicability",
]


class UpdateTables:
    """Disk-backed working tables: read/write one table, preserve untouched bytes."""

    def __init__(self, output, current=None):
        import pandas as pd

        self.output = output
        self.replacements = {}
        self.additions = {}
        for name in TABLES:
            target = output / (name + ".parquet")
            if current:
                source = current / target.name
                if source.is_file():
                    shutil.copyfile(source, target)
                else:  # a partitioned baseline table becomes one flat file here
                    dataset.read_table(current, name).to_parquet(
                        target, index=False, compression="zstd"
                    )
                continue
            frame = pd.DataFrame()
            self[name] = frame

    def __getitem__(self, name):
        import pandas as pd

        return pd.read_parquet(self.output / (name + ".parquet"))

    def __setitem__(self, name, frame):
        frame.to_parquet(
            self.output / (name + ".parquet"), index=False, compression="zstd"
        )

    def selected(self, name, symbol):
        import pandas as pd
        import pyarrow.parquet as pq

        path = self.output / (name + ".parquet")
        if "symbol" not in pq.read_schema(path).names:
            return pd.DataFrame()
        return pd.read_parquet(path, filters=[("symbol", "==", symbol)])

    def replace(self, name, symbol, incoming):
        self.replacements.setdefault(name, {})[symbol] = incoming

    def add(self, name, rows):
        import pandas as pd

        self.additions.setdefault(name, []).append(pd.DataFrame(rows))

    def flush(self):
        import pandas as pd

        for name in TABLES:
            replacements = self.replacements.pop(name, {})
            additions = self.additions.pop(name, [])
            if not replacements and not additions:
                continue
            frame = self[name]
            if replacements and "symbol" in frame:
                frame = frame.loc[~frame.symbol.isin(replacements)]
            self[name] = pd.concat(
                [frame, *replacements.values(), *additions], ignore_index=True
            )
            del frame, replacements, additions


def request(
    symbols,
    add_companies=True,
    add_periods=False,
    full_refresh=False,
    add_benchmarks=False,
    market_through=None,
    refresh_financials=None,
    market_refresh_symbols=None,
):
    result = dict(
        companies=list(dict.fromkeys(symbols)),
        add_companies=add_companies,
        add_periods=add_periods,
        full_refresh=full_refresh,
        add_benchmarks=add_benchmarks,
    )
    if not result["companies"] or set(result["companies"]) - set(COMPANIES):
        raise ValueError("Outside approved company panel")
    if any(
        type(result[k]) is not bool
        for k in ("add_companies", "add_periods", "add_benchmarks", "full_refresh")
    ):
        raise ValueError("Update controls must be booleans")
    if refresh_financials is not None:
        if type(refresh_financials) is not bool:
            raise ValueError("Financial refresh control must be boolean")
        result["refresh_financials"] = refresh_financials
    if market_refresh_symbols is not None:
        if (
            not isinstance(market_refresh_symbols, list)
            or set(market_refresh_symbols)
            - (set(result["companies"]) | set(BENCHMARKS))
            or add_companies
            or not add_periods
            or full_refresh
            or add_benchmarks
            or refresh_financials is not False
        ):
            raise ValueError("Market replacements require market-only period controls")
        result["market_refresh_symbols"] = sorted(set(market_refresh_symbols))
    if market_through is not None:
        if (
            not isinstance(market_through, str)
            or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", market_through)
            or not (
                add_companies
                and not add_periods
                or not add_companies
                and add_periods
                and refresh_financials is False
            )
            or add_benchmarks
            or full_refresh
        ):
            raise ValueError(
                "Market cutoff requires expansion-only controls and YYYY-MM-DD"
            )
        datetime.strptime(market_through, "%Y-%m-%d")
        result["market_through"] = market_through
    return result


def read_current(root, *, files_only=False):
    path = store.path_for(root, store.STAGED)
    if not path.exists():
        return None
    manifest = json.loads((path / "manifest.json").read_bytes())
    if manifest["status"] != "VERIFIED":
        raise ValueError("Current dataset is incomplete")
    if files_only:
        store.verify_local_files(path, manifest)
    else:
        store.verify_local(path, manifest)
    return path, manifest


def snapshot(root, req, planned_at=None):
    """Read the verified local dataset; never request SEC or Yahoo data."""

    req = request(
        **dict(
            symbols=req["companies"],
            **{k: v for k, v in req.items() if k != "companies"},
        )
    )
    current = read_current(root, files_only=True)
    base = dict(
        manifest_sha256=None,
        companies=[],
        benchmarks=[],
        last_sessions={},
        filed_through={},
        filings_through={},
    )
    if current:
        path, manifest = current
        saved_validation = store.validation_is_current(manifest)
        if not saved_validation:
            store.verify_local(path, manifest)
        daily = dataset.read_table(path, "daily_bars", columns=["symbol", "session_date"])
        facts = dataset.read_table(path, "fundamentals")
        filings = dataset.read_table(path, "filings")
        base.update(
            manifest_sha256=sec.digest((path / "manifest.json").read_bytes()),
            quality_gate=quality.summary(
                manifest["quality"] if saved_validation else quality.assess(path)
            ),
            market_quality_gate=market_quality.summary(
                manifest["market_quality"]
                if saved_validation
                else market_quality.assess(path)
            ),
            companies=manifest["companies"],
            benchmarks=manifest["benchmarks"],
            last_sessions=daily.groupby("symbol").session_date.max().to_dict(),
            filed_through=facts.groupby("symbol").filed.max().to_dict(),
            filings_through=(
                filings.groupby("symbol").filingDate.max().to_dict()
                if not filings.empty
                else {}
            ),
        )
    return dict(request=req, base=base, planned_at=planned_at or sec.now())


def operations(plan):
    req, base = plan["request"], plan["base"]
    latest = req.get("market_through") or store.last_completed(
        datetime.fromisoformat(plan["planned_at"])
    )
    known = set(base["companies"])
    new = [s for s in req["companies"] if s not in known and req["add_companies"]]
    existing = [
        s
        for s in req["companies"]
        if s in known and (req["add_periods"] or req["full_refresh"])
    ]
    markets = {s: dict(mode="full", start=company(s)["requested_start"]) for s in new}
    cutoff = req.get("market_through")
    if cutoff is not None:
        # Explicit request scope, never automatic truncation of a bad/latest row.
        if (
            cutoff > store.last_completed(datetime.fromisoformat(plan["planned_at"]))
            or store.last_completed(datetime.fromisoformat(cutoff + "T23:59:59+00:00"))
            != cutoff
        ):
            raise ValueError("Market cutoff must be a completed trading session")
        for task in markets.values():
            if task["start"] > cutoff:
                raise ValueError("Market cutoff precedes requested listing history")
            task["through"] = cutoff
    for s in existing + list(base["benchmarks"]):
        if s in BENCHMARKS and not (req["add_periods"] or req["full_refresh"]):
            continue
        if req["full_refresh"]:
            markets[s] = dict(
                mode="full",
                start=(
                    company(s)["requested_start"]
                    if s in COMPANIES
                    else store.BENCHMARK_STARTS[s]
                ),
            )
        elif base["last_sessions"][s] < latest:
            markets[s] = dict(mode="append", start=base["last_sessions"][s])
    # Bootstrap missing benchmarks only when adding companies; never re-fetch covered ones for expansion.
    if req["add_companies"] and new:
        for s in BENCHMARKS:
            if s not in base["benchmarks"]:
                markets[s] = dict(mode="full", start=store.BENCHMARK_STARTS[s])
                if cutoff is not None:
                    markets[s]["through"] = cutoff
    # New registered benchmarks are an explicit migration. They are fetched once
    # without also extending or replacing existing company/benchmark histories.
    if req["add_benchmarks"]:
        for s in BENCHMARKS:
            if s not in base["benchmarks"]:
                markets[s] = dict(mode="full", start=store.BENCHMARK_STARTS[s])
    financial_existing = existing if req.get("refresh_financials", True) else []
    for symbol in req.get("market_refresh_symbols", []):
        if symbol not in base["companies"] + base["benchmarks"]:
            raise ValueError("Market replacement requires an existing series")
        markets[symbol] = dict(
            mode="full",
            start=(
                company(symbol)["requested_start"]
                if symbol in COMPANIES
                else store.BENCHMARK_STARTS[symbol]
            ),
        )
    if cutoff is not None:
        for task in markets.values():
            task["through"] = cutoff
    return dict(
        sec_companies=new + financial_existing,
        new_companies=new,
        market=markets,
    )


def validate_capture(content):
    if (
        set(content) != {"schema", "collected_at", "plan", "companies"}
        or content["schema"] != SCHEMA
    ):
        raise ValueError("Invalid selective update capture")
    plan = content["plan"]
    if set(plan) != {"request", "base", "planned_at"}:
        raise ValueError("Invalid update plan")
    req = plan["request"]
    required_controls = {
        "companies",
        "add_companies",
        "add_periods",
        "add_benchmarks",
        "full_refresh",
    }
    if (
        not required_controls
        <= set(req)
        <= required_controls
        | {
            "market_through",
            "refresh_financials",
            "market_refresh_symbols",
        }
    ):
        raise ValueError("Invalid update controls")
    request(
        **dict(
            symbols=req["companies"],
            **{k: v for k, v in req.items() if k != "companies"},
        )
    )
    stamp = datetime.fromisoformat(plan["planned_at"])
    captured = datetime.fromisoformat(content["collected_at"])
    if (
        stamp.tzinfo is None
        or captured.tzinfo is None
        or not stamp <= captured <= datetime.now(timezone.utc)
    ):
        raise ValueError("Invalid update plan timestamp")
    base = plan["base"]
    if base["manifest_sha256"] is not None:
        sec.identity(base["manifest_sha256"])
    if set(base["companies"]) - set(COMPANIES) or set(base["benchmarks"]) - set(
        BENCHMARKS
    ):
        raise ValueError("Invalid baseline panel")
    if set(base["last_sessions"]) != set(base["companies"]) | set(base["benchmarks"]):
        raise ValueError("Incomplete baseline watermarks")
    for day in [
        *base["last_sessions"].values(),
        *base["filed_through"].values(),
        *base["filings_through"].values(),
    ]:
        datetime.strptime(day, "%Y-%m-%d")
    if set(content["companies"]) != set(operations(plan)["sec_companies"]):
        raise ValueError("SEC captures do not match selected controls")
    if content["companies"]:
        store.validate(
            dict(
                schema=store.SCHEMA,
                collected_at=content["collected_at"],
                companies=content["companies"],
            )
        )
    return content


def check_expansion(plan):
    quality.require_expansion_ready(plan)


def collect(
    root,
    symbols,
    run,
    *,
    add_companies=True,
    add_periods=False,
    full_refresh=False,
    add_benchmarks=False,
    market_through=None,
):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", run or ""):
        raise ValueError("Invalid run ID")
    req = request(
        symbols,
        add_companies,
        add_periods,
        full_refresh,
        add_benchmarks,
        market_through,
    )
    plan = snapshot(root, req)
    check_expansion(plan)
    path = store.path_for(root, "local/" + run)
    path.mkdir(parents=True, exist_ok=False)
    targets = operations(plan)
    content = dict(schema=SCHEMA, plan=plan, collected_at=sec.now(), companies={})
    capture_started = sec.now()
    for symbol in targets["sec_companies"]:
        content["companies"][symbol] = store.collect_company(path, symbol)
    sec.prepare_capture(path, content, validator=validate_capture, **store.BOUNDS)
    store.replace_generated(
        path / "collection_timing.json",
        sec.encoded(
            dict(
                bucket="import",
                activity="processing",
                step="SEC_capture",
                start=capture_started,
                end=sec.now(),
            )
        ),
    )
    print(
        json.dumps(
            dict(
                capture=str(path),
                controls=req,
                operations=targets,
                quality_gate=plan["base"].get("quality_gate"),
                status=(
                    "NO_CHANGES"
                    if not targets["sec_companies"] and not targets["market"]
                    else "CAPTURED"
                ),
            )
        ),
        flush=True,
    )
    return path


def assert_base(root, plan):
    current = read_current(root, files_only=True)
    actual = (
        sec.digest((current[0] / "manifest.json").read_bytes()) if current else None
    )
    if actual != plan["base"]["manifest_sha256"]:
        raise ValueError(
            "Current dataset changed since planning; obtain a fresh plan before fetching"
        )
    return current


def append_market(old, raw, receipt, start, *, through=None):
    """Keep stored rows unchanged and reject return-basis discontinuities."""
    import numpy as np
    import pandas as pd

    window, _ = store.validate_market(
        raw, receipt, expected_start=start, expected_end=through
    )
    before = old.sort_values("session_date").iloc[-1]
    boundary = window.loc[window.session_date == start]
    if len(boundary) != 1 or before.session_date != start:
        raise ValueError("Missing continuity boundary; use explicit full refresh")
    critical = ["Close", "Adj Close", "Stock Splits"]
    if not np.allclose(
        before[critical].to_numpy(dtype=float),
        boundary.iloc[0][critical].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(
            "Stored price/adjustment basis changed; explicit full refresh required"
        )
    # Yahoo sometimes finalizes non-basis fields after a recent row was first
    # captured. Preserve the accepted row, but retain exact differences with the
    # append receipt so revisions remain visible rather than silently replacing it.
    differences = {}
    incoming = boundary.iloc[0]
    for field in ["Open", "High", "Low", "Volume", "Dividends"]:
        retained_value = float(before[field])
        provider_value = float(incoming[field])
        if not np.isclose(retained_value, provider_value, rtol=1e-6, atol=1e-6):
            differences[field] = dict(
                retained=retained_value,
                provider=provider_value,
            )
    if differences:
        receipt["retained_boundary_revision"] = dict(
            session_date=start,
            fields=differences,
            disposition="STORED_ROW_RETAINED",
        )
    added = window.loc[window.session_date > start].copy()
    if (added["Stock Splits"] != 0).any():
        raise ValueError(
            "Split requires explicit full refresh; cannot splice adjusted histories"
        )
    return pd.concat([old, added], ignore_index=True)


def market_coverage(frame):
    return dict(
        rows=len(frame),
        first=frame.session_date.min(),
        last=frame.session_date.max(),
        missing_sessions=[],
        unexpected_sessions=[],
        invalid_rows=0,
        dividend_events=int((frame.Dividends != 0).sum()),
        split_events=int((frame["Stock Splits"] != 0).sum()),
    )


def normalize_company(symbol, item):
    _, canonical, _, issues = norm.normalize(
        item["facts"],
        item["filings"],
        item["receipts"]["facts"],
        metrics=metrics_for(symbol),
    )
    if not canonical or any(r["status"] != "REPORTED" for r in canonical):
        raise ValueError("Missing/conflicting normalized financials: " + symbol)
    canonical, issues = norm.quarantine_facts(
        canonical, issues, quarantines_for(symbol)
    )
    for rows in (canonical, item["filings"], issues):
        for row in rows:
            row.update(symbol=symbol, issuer_id=company(symbol)["issuer_id"])
    return canonical, issues


def financial_coverage(symbol, canonical, latest, issues):
    return dict(
        versions=len(canonical),
        latest_rows=len(latest),
        excluded_rows=len(issues),
        excluded_facts=issues,
        exclusion_scope="LATEST_SELECTED_SEC_RESPONSE",
        acceptance_timestamps=sum(bool(r["filing_accepted_at"]) for r in canonical),
        fields={
            name: dict(
                rows=sum(r["metric"] == name for r in latest),
                earliest_end=min(
                    (r["end"] for r in latest if r["metric"] == name), default=None
                ),
                latest_end=max(
                    (r["end"] for r in latest if r["metric"] == name), default=None
                ),
                status=(
                    "AVAILABLE"
                    if any(r["metric"] == name for r in latest)
                    else "NO_MAPPED_REPORTED_FACTS"
                ),
            )
            for name in metrics_for(symbol)
        },
    )


def merge_versions(old, incoming, watermark):
    """Append newly filed versions; keep every existing reporting-version row intact."""
    index = {tuple(r[k] for k in KEY): r for r in old}
    result = list(old)
    for row in incoming:
        if row["filed"] < watermark:
            continue
        key = tuple(row[k] for k in KEY)
        if key in index:
            if (norm.number(row["value"]), row["status"]) != (
                norm.number(index[key]["value"]),
                index[key]["status"],
            ):
                raise ValueError(
                    "Existing SEC fact revised; explicit full refresh required"
                )
        else:
            result.append(row)
            index[key] = row
    return result


def build_update(root, bundle, content, market_fetch, *, output=None):
    import pandas as pd

    total_started = time.monotonic()
    timings = {}
    validate_capture(content)
    plan = content["plan"]
    print("UPDATE_BUILD: verifying baseline", flush=True)
    phase_started = time.monotonic()
    current = assert_base(
        root, plan
    )  # Verify the hydrated baseline before *any* provider call.
    timings["baseline_guard_seconds"] = round(time.monotonic() - phase_started, 3)
    check_expansion(plan)
    targets = operations(plan)
    output = Path(output) if output else store.run_path(root, bundle) / "built"
    marker = output.parent / "update_result.json"
    if not targets["sec_companies"] and not targets["market"]:
        store.replace_generated(
            marker,
            sec.encoded(
                dict(
                    status="NO_CHANGES",
                    base_manifest_sha256=plan["base"]["manifest_sha256"],
                )
            ),
        )
        print(
            "NO_CHANGES: zero SEC/Yahoo requests and no current-store writes",
            flush=True,
        )
        return None
    if any((output / n).is_symlink() for n in store.FILES | {"manifest.json"}):
        raise ValueError("Symlink generated output")
    output.mkdir(parents=True, exist_ok=True)
    phase_started = time.monotonic()
    tables = UpdateTables(output, current[0] if current else None)
    print("UPDATE_BUILD: working tables prepared", flush=True)
    if current:
        current_path, _ = current
        coverage = json.loads((current_path / "coverage.json").read_bytes())
        sources = json.loads(
            bz2.decompress((current_path / "sources.json.bz2").read_bytes())
        )
    else:
        coverage = {}
        sources = dict(
            sec_projection=dict(
                schema=store.SCHEMA, collected_at=content["collected_at"], companies={}
            ),
            market={},
        )
    sources.setdefault("sec_receipts", {})
    sources.setdefault("market_updates", {})
    timings["working_tables_seconds"] = round(time.monotonic() - phase_started, 3)
    processing_started = time.monotonic()
    for symbol, task in targets["market"].items():
        print("UPDATE_BUILD: market " + symbol, flush=True)
        scope = dict(through=task["through"]) if "through" in task else {}
        raw, receipt = market_fetch(symbol, task["start"], root, **scope)
        old = tables.selected("daily_bars", symbol)
        if task["mode"] == "append":
            frame = append_market(
                old, raw, receipt, task["start"], through=task.get("through")
            )
            sources["market_updates"].setdefault(symbol, []).append(
                dict(receipt=receipt, wrapper_csv=raw.decode())
            )
        else:
            frame, _ = store.validate_market(
                raw, receipt, expected_end=task.get("through")
            )
            sources["market"][symbol] = dict(receipt=receipt, wrapper_csv=raw.decode())
            sources["market_updates"].pop(symbol, None)
        tables.replace("daily_bars", symbol, frame)
        coverage.setdefault(symbol, {})["market"] = market_coverage(frame)
        if tables.selected("securities", symbol).empty:
            spec = (
                company(symbol)
                if symbol in COMPANIES
                else dict(name=BENCHMARKS[symbol])
            )
            tables.add(
                "securities",
                [
                    dict(
                        symbol=symbol,
                        security_id="yahoo:" + symbol,
                        issuer_id=spec.get("issuer_id", ""),
                        name=spec["name"],
                        kind=(
                            "company_listing"
                            if symbol in COMPANIES
                            else "benchmark_index"
                        ),
                        identity_status="VENDOR_CONTINUITY_NOT_CERTIFIED",
                    )
                ],
            )
        del old, frame
    for symbol, item in content["companies"].items():
        canonical, issues = normalize_company(symbol, deepcopy(item))
        old = tables.selected("fundamentals", symbol).to_dict("records")
        if old and not plan["request"]["full_refresh"]:
            canonical = merge_versions(
                old, canonical, plan["base"]["filed_through"][symbol]
            )
        latest = norm.latest_view(canonical)
        if any(r["status"] != "REPORTED" for r in latest):
            raise ValueError("Conflicting latest financials: " + symbol)
        for name, rows in (("fundamentals", canonical), ("financial_latest", latest)):
            tables.replace(name, symbol, pd.DataFrame(rows))
        if tables.selected("companies", symbol).empty:
            tables.add("companies", [company(symbol)])
            tables.add("field_applicability", applicability(symbol))
        incoming = deepcopy(item["filings"])
        for row in incoming:
            row.update(symbol=symbol, issuer_id=company(symbol)["issuer_id"])
        old_filings = tables.selected("filings", symbol).to_dict("records")
        if old and not plan["request"]["full_refresh"]:
            accessions = {r["accessionNumber"] for r in old_filings}
            cutoff = plan["base"]["filings_through"].get(symbol, "")
            incoming = old_filings + [
                r
                for r in incoming
                if r["accessionNumber"] not in accessions and r["filingDate"] >= cutoff
            ]
        tables.replace("filings", symbol, pd.DataFrame(incoming))
        coverage.setdefault(symbol, {})["financials"] = financial_coverage(
            symbol, canonical, latest, issues
        )
        previous = sources["sec_projection"]["companies"].get(symbol)
        if previous:
            for r in previous["receipts"].values():
                sources["sec_receipts"][r["sha256"]] = r
        sources["sec_projection"]["companies"][symbol] = item
    store.replace_generated(output / "coverage.json", sec.encoded(coverage))
    store.replace_generated(
        output / "sources.json.bz2", bz2.compress(sec.encoded(sources))
    )
    del sources, coverage
    timings["selected_processing_seconds"] = round(
        time.monotonic() - processing_started, 3
    )
    print("UPDATE_BUILD: source evidence saved; merging changed tables", flush=True)
    phase_started = time.monotonic()
    tables.flush()
    timings["table_merge_seconds"] = round(time.monotonic() - phase_started, 3)
    phase_started = time.monotonic()
    file_manifest = {
        n: dict(
            sha256=sec.digest((output / n).read_bytes()),
            bytes=(output / n).stat().st_size,
        )
        for n in sorted(store.FILES)
    }
    timings["file_hashing_seconds"] = round(time.monotonic() - phase_started, 3)
    manifest = dict(
        schema=store.SCHEMA,
        status="BUILT_NOT_PUBLISHED",
        bundle=bundle,
        collected_at=content["collected_at"],
        built_at=sec.now(),
        code_commit=sec.code_version(),
        companies=tables["companies"].symbol.tolist(),
        benchmarks=(
            list(BENCHMARKS)
            if plan["request"]["add_benchmarks"] or not current
            else plan["base"]["benchmarks"]
        ),
        processing_libraries={
            p: version(p) for p in ("pandas", "pyarrow", "lxml", "exchange-calendars")
        },
        collection_mode=(
            "SELECTIVE_COMPANIES_AND_PERIODS"
        ),
        update_controls=plan["request"],
        operations=targets,
        base_manifest_sha256=plan["base"]["manifest_sha256"],
        build_seconds=round(time.monotonic() - total_started, 2),
        limits=store.LIMITS,
        files=file_manifest,
    )
    store.replace_generated(output / "manifest.json", sec.encoded(manifest))
    print("UPDATE_BUILD: validating candidate", flush=True)
    phase_started = time.monotonic()
    store.verify_local(
        output,
        manifest,
        require_all_benchmarks=plan["request"]["add_benchmarks"] or not current,
    )
    timings["structural_validation_seconds"] = round(
        time.monotonic() - phase_started, 3
    )
    phase_started = time.monotonic()
    manifest["quality"] = quality.assess(
        output, previous_path=current[0] if current else None
    )
    # Retain compact assessment evidence when a market-only invariant blocks the
    # candidate before market assessment and validation binding complete.
    store.replace_generated(output / "manifest.json", sec.encoded(manifest))
    if current and not content["companies"]:
        financial_tables = {
            "companies",
            "field_applicability",
            "filings",
            "financial_latest",
            "fundamentals",
        }
        changed = sorted(
            name
            for name in financial_tables
            if manifest["files"].get(name) != current[1]["files"].get(name)
        )
        if changed:
            raise ValueError("MARKET_ONLY_FINANCIAL_MUTATION: " + ", ".join(changed))
        prior_transition = {
            (flag["id"], flag["evidence_sha256"])
            for flag in current[1].get("quality", {}).get("flags", [])
            if flag["rule"] in {"HISTORICAL_COVERAGE_LOSS", "NEW_SOURCE_TAG"}
        }
        new_transition = [
            flag
            for flag in manifest["quality"]["flags"]
            if flag["rule"] in {"HISTORICAL_COVERAGE_LOSS", "NEW_SOURCE_TAG"}
            and (flag["id"], flag["evidence_sha256"]) not in prior_transition
        ]
        if new_transition:
            raise ValueError(
                "MARKET_ONLY_FINANCIAL_TRANSITION: "
                + ", ".join(sorted({flag["symbol"] for flag in new_transition}))
            )
    timings["financial_quality_seconds"] = round(time.monotonic() - phase_started, 3)
    phase_started = time.monotonic()
    manifest["market_quality"] = market_quality.assess(output)
    timings["market_quality_seconds"] = round(time.monotonic() - phase_started, 3)
    timings["total_seconds"] = round(time.monotonic() - total_started, 3)
    timings["candidate_bytes"] = sum(
        item["bytes"] for item in manifest["files"].values()
    )
    manifest["build_seconds"] = timings["total_seconds"]
    manifest["timing"] = timings
    store.bind_validation(manifest)
    store.replace_generated(output / "manifest.json", sec.encoded(manifest))
    print(
        "COMPANY_DATA_TIMING=" + json.dumps(dict(phase="build", **timings)),
        flush=True,
    )
    print(
        "COMPANY_DATA_QUALITY=" + json.dumps(quality.summary(manifest["quality"])),
        flush=True,
    )
    store.replace_generated(
        marker,
        sec.encoded(
            dict(status="BUILT", base_manifest_sha256=plan["base"]["manifest_sha256"])
        ),
    )
    return output

