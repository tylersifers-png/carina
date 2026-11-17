"""Collect, build and verify the company dataset (daily bars and selected SEC facts)."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import io
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import tempfile
import time

from carina import common as sec, dataset, settings
from carina.common import PARAMETERS, last_completed, receipt_valid
from carina import fundamentals as norm
from carina.registry import (
    COMPANIES,
    BENCHMARKS,
    BENCHMARK_STARTS,
    company,
    metrics_for,
    applicability,
)

SCHEMA = "company-current-v1.0"
# Paths inside the data directory (see ``carina.settings``).
STAGED = "staged/company"
RUNS = "runs"
RECEIPTS = "receipts"
BOUNDS = dict(max_parts=32, max_bundle=32 * 1024 * 1024)
FILES = {
    "daily_bars.parquet",
    "companies.parquet",
    "securities.parquet",
    "fundamentals.parquet",
    "financial_latest.parquet",
    "filings.parquet",
    "field_applicability.parquet",
    "coverage.json",
    "sources.json",
}
# Receipts for the latest capture of each series. Raw payloads stay in the run
# folder and are deleted with it once the run is promoted.
SOURCES = "sources.json"
FINANCIAL_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "10-K405", "8-K", "8-K/A"}
LIMITS = [
    "CURRENT_VINTAGE_NOT_HISTORICAL_AS_KNOWN",
    "Current representative companies are not a historical S&P 500 universe",
    "Ticker/share-class/predecessor continuity is vendor supplied, not certified",
    "Selected SEC facts only; entire companyfacts and older filing indexes not archived here",
    "Historical acceptance timestamps may be missing; filed date is not intraday availability",
    "No new original-filing certification; v1 coverage is distinct from prior pilot acceptance",
    "Yahoo OHLC is split adjusted; Adj Close is vendor adjusted, not independently certified total return",
    "No industrial FCF or debt inference for bank/insurance profiles",
    "No vintage/revision comparison system",
]

VALIDATION_SCHEMA = "company-candidate-validation-1.0"
# Saved assessments are reused only while the data files, the threshold and
# review files, and the explicit detector versions are unchanged.
VALIDATION_INPUTS = ("thresholds.json", "reviews.json")
VALIDATION_LIBRARIES = ("pandas", "pyarrow", "numpy", "exchange-calendars")
SEC_REPOSITORY = sec.REPOSITORY
SEC_CONTACT_PATTERN = re.compile(r"[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+")


def _contains_symlink_below(path: Path, boundary: Path) -> bool:
    """Check path components at or below a trusted containment boundary.

    macOS exposes its temporary directory through system aliases such as
    ``/var -> /private/var``.  Those aliases may be above the project or temp
    root and are not writable by this process.  Rejecting every symlink all the
    way to ``/`` therefore breaks otherwise safe atomic writes on macOS.
    """
    absolute = path.absolute()
    trusted = boundary.absolute()
    if not absolute.is_relative_to(trusted):
        return True
    current = trusted
    if current.is_symlink():
        return True
    for part in absolute.relative_to(trusted).parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _generated_write_boundary(path: Path) -> Path:
    """Return the narrowest trusted root containing a generated output."""
    absolute = path.absolute()
    candidates = [Path(tempfile.gettempdir()).absolute()]
    try:
        candidates.append(settings.data().absolute())
    except RuntimeError:
        pass
    containing = [root for root in candidates if absolute.is_relative_to(root)]
    return max(
        containing,
        key=lambda root: len(root.parts),
        default=Path(absolute.anchor),
    )


def path_for(root, relative):
    from carina.drive import normalize_relative_path

    root = Path(root)
    rel = normalize_relative_path(relative)
    path = root.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(root.resolve()) or _contains_symlink_below(
        path, root
    ):
        raise ValueError("Unsafe company-data path")
    return path


def replace_generated(path, raw):
    """Atomic local replacement for mutable current/generated outputs only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _contains_symlink_below(path, _generated_write_boundary(path)):
        raise ValueError("Symlink generated output")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        f.write(raw)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _validated_manifest_payload(manifest):
    """Return the candidate content covered by its reusable validation binding."""
    payload = deepcopy(manifest)
    for name in ("validation", "status", "verified_at", "readback"):
        payload.pop(name, None)
    return payload


def validation_inputs():
    from carina import market_quality
    from carina import quality

    inputs = {
        name: sec.digest(settings.decision(name).read_bytes()) for name in VALIDATION_INPUTS
    }
    inputs["detector_versions"] = [quality.VERSION, market_quality.VERSION]
    return inputs


def validation_libraries():
    return {name: version(name) for name in VALIDATION_LIBRARIES}


def validation_binding(manifest):
    """Bind one completed assessment to exact data, code, policy and review bytes."""
    if "quality" not in manifest or "market_quality" not in manifest:
        raise ValueError("Candidate validation requires both quality assessments")
    return dict(
        schema=VALIDATION_SCHEMA,
        bundle=manifest["bundle"],
        validated_at=sec.now(),
        manifest_payload_sha256=sec.digest(
            sec.encoded(_validated_manifest_payload(manifest))
        ),
        files=manifest["files"],
        inputs=validation_inputs(),
        libraries=validation_libraries(),
        quality_sha256=sec.digest(sec.encoded(manifest["quality"])),
        market_quality_sha256=sec.digest(sec.encoded(manifest["market_quality"])),
    )


def validation_is_current(manifest):
    """True only when saved assessments exactly cover current bytes and contracts."""
    saved = manifest.get("validation", {})
    try:
        return (
            saved.get("schema") == VALIDATION_SCHEMA
            and saved.get("bundle") == manifest["bundle"]
            and saved.get("files") == manifest["files"]
            and saved.get("inputs") == validation_inputs()
            and saved.get("libraries") == validation_libraries()
            and saved.get("manifest_payload_sha256")
            == sec.digest(sec.encoded(_validated_manifest_payload(manifest)))
            and saved.get("quality_sha256")
            == sec.digest(sec.encoded(manifest["quality"]))
            and saved.get("market_quality_sha256")
            == sec.digest(sec.encoded(manifest["market_quality"]))
        )
    except (KeyError, TypeError, ValueError):
        return False


def bind_validation(manifest):
    manifest["validation"] = validation_binding(manifest)
    return manifest


def validate(content):
    if content.get("schema") == "company-update-v1.1":
        from carina.updates import validate_capture

        return validate_capture(content)
    if (
        set(content) != {"schema", "collected_at", "companies"}
        or content["schema"] != SCHEMA
    ):
        raise ValueError("Invalid company-data envelope")
    if not content["companies"] or set(content["companies"]) - set(COMPANIES):
        raise ValueError("Outside approved company panel")
    captured = datetime.fromisoformat(content["collected_at"])
    if captured.tzinfo is None or captured > datetime.now(timezone.utc):
        raise ValueError("Invalid collection timestamp")
    for symbol, item in content["companies"].items():
        spec = company(symbol)
        if set(item) != {
            "facts",
            "filings",
            "receipts",
            "projection_sha256",
            "issuer_metadata",
        }:
            raise ValueError("Unexpected company projection fields")
        facts = item["facts"]
        identity = item["issuer_metadata"]
        if (
            int(identity["cik"]) != spec["issuer_cik"]
            or not identity["name"]
            or (identity["tickers"] and symbol not in identity["tickers"])
        ):
            raise ValueError("SEC submissions issuer/ticker mismatch")
        if int(facts["cik"]) != spec["issuer_cik"] or not facts.get("facts"):
            raise ValueError("SEC identity mismatch or empty facts")
        wanted = {
            (ns, tag)
            for _, _, tags in metrics_for(symbol).values()
            for ns, tag, _ in tags
        }
        if any(
            (ns, tag) not in wanted
            for ns, tags in facts["facts"].items()
            for tag in tags
        ):
            raise ValueError("Unexpected SEC fact outside selected profile")
        if item["projection_sha256"] != sec.digest(sec.encoded(facts)):
            raise ValueError("SEC projection checksum mismatch")
        for key, endpoint in (
            ("facts", "api/xbrl/companyfacts"),
            ("submissions", "submissions"),
        ):
            r = item["receipts"][key]
            url = f"https://data.sec.gov/{endpoint}/CIK{spec['issuer_cik']:010d}.json"
            if r["source_url"] != url or r["http_status"] != 200:
                raise ValueError("SEC receipt URL/status mismatch")
            sec.identity(r["sha256"])
            a, b = [
                datetime.fromisoformat(r[k]) for k in ("requested_at", "retrieved_at")
            ]
            if (
                a.tzinfo is None
                or b.tzinfo is None
                or not a <= b <= datetime.now(timezone.utc)
            ):
                raise ValueError("Invalid source receipt time")
        accessions = [r["accessionNumber"] for r in item["filings"]]
        if len(accessions) != len(set(accessions)) or any(
            r["form"] not in FINANCIAL_FORMS for r in item["filings"]
        ):
            raise ValueError("Invalid financial filing projection")
    return content


def sec_user_agent(contact=None, *, require_contact=False):
    """Build the declared SEC identity without retaining the contact in data."""
    if contact is None:
        contact = os.environ.get("SEC_CONTACT_EMAIL", "")
    contact = (contact or "").strip()
    if require_contact and not contact:
        raise ValueError(
            "SEC_CONTACT_EMAIL is required for hosted company maintenance"
        )
    if contact and not SEC_CONTACT_PATTERN.fullmatch(contact):
        raise ValueError("Invalid SEC contact email")
    if contact:
        return f"Carina Data Pipeline {contact}"
    return f"Carina Data Pipeline {SEC_REPOSITORY}"


def _source_failure(symbol, key, attempts):
    latest = attempts[-1] if attempts else {}
    if "http_status" in latest:
        reason = f"HTTP {latest['http_status']}"
    else:
        reason = latest.get("error_type", "no response receipt")
    return RuntimeError(f"SEC {key} unavailable for {symbol}: {reason}")


SEC_ENDPOINTS = {
    "facts": "api/xbrl/companyfacts",
    "submissions": "submissions",
}


def fetch_sec_json(
    symbol, key, *, user_agent=None, attempt_audit=None, fetch=None
):
    """Fetch and validate one SEC JSON response without writing its payload."""
    if key not in SEC_ENDPOINTS:
        raise ValueError("Unknown SEC endpoint")
    spec = company(symbol)
    url = (
        f"https://data.sec.gov/{SEC_ENDPOINTS[key]}/"
        f"CIK{spec['issuer_cik']:010d}.json"
    )
    raw, attempts = (fetch or sec.fetch)(
        url, user_agent or sec_user_agent(), max_body=32 * 1024 * 1024
    )
    if attempt_audit is not None:
        attempt_audit(symbol, key, url, attempts)
    if raw is None:
        raise _source_failure(symbol, key, attempts)
    receipt = attempts[-1]
    receipt_valid(receipt, raw, url)
    try:
        source = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid SEC {key} JSON for {symbol}") from exc
    if not isinstance(source, dict):
        raise ValueError(f"Invalid SEC {key} JSON for {symbol}")
    return source, receipt, raw


def project_submissions(symbol, source):
    """Retain issuer identity and the selected recent financial filing list."""
    spec = company(symbol)
    identity = {k: source[k] for k in ("cik", "name", "tickers")}
    if int(identity["cik"]) != spec["issuer_cik"] or (
        identity["tickers"] and symbol not in identity["tickers"]
    ):
        raise ValueError("SEC submissions issuer/ticker mismatch")
    filings = [
        row
        for row in norm.filing_rows(source["filings"]["recent"])
        if row["form"] in FINANCIAL_FORMS
    ]
    return identity, filings


def project_companyfacts(symbol, source):
    """Keep only the configured fact tags from a companyfacts response."""
    spec = company(symbol)
    if int(source.get("cik", -1)) != spec["issuer_cik"]:
        raise ValueError("SEC companyfacts issuer mismatch")
    wanted = {
        (ns, tag) for _, _, tags in metrics_for(symbol).values() for ns, tag, _ in tags
    }
    projection = dict(cik=source["cik"], entityName=source.get("entityName"), facts={})
    for namespace, tags in source.get("facts", {}).items():
        selected = {
            tag: value for tag, value in tags.items() if (namespace, tag) in wanted
        }
        if selected:
            projection["facts"][namespace] = selected
    if not projection["facts"]:
        raise ValueError("SEC identity mismatch or empty facts")
    return projection


def collect_company(path, symbol, *, user_agent=None, attempt_audit=None):
    agent = user_agent or sec_user_agent()
    item = dict(receipts={})
    source = {}
    for key in ("facts", "submissions"):
        value, receipt, raw = fetch_sec_json(
            symbol,
            key,
            user_agent=agent,
            attempt_audit=attempt_audit,
        )
        sec.write(path / (symbol + "_" + key + ".json"), raw)
        source[key] = value
        item["receipts"][key] = receipt
        time.sleep(1)
    item["issuer_metadata"], item["filings"] = project_submissions(
        symbol, source["submissions"]
    )
    item["facts"] = project_companyfacts(symbol, source["facts"])
    item["projection_sha256"] = sec.digest(sec.encoded(item["facts"]))
    return item


def collect(root, symbols, run, **controls):
    """Default to missing-company expansion; all provider work follows explicit controls."""
    from carina.updates import collect as collect_update

    return collect_update(root, symbols, run, **controls)


def load(path, bundle):
    return sec.load(path, bundle, validator=validate, **BOUNDS)


def rejection_observations(symbol, issues, item):
    from carina.quality import rejection_observations as observe

    return observe(symbol, issues, item)


def sec_receipts(item, collected_at):
    """What is kept of one company's SEC capture: its receipts, not its payload."""
    return dict(collected_at=collected_at, receipts=item["receipts"])


def run_path(root, bundle):
    return path_for(root, RUNS + "/" + sec.identity(bundle))


def market(symbol, start, root, *, through=None):
    import yfinance as yf

    yf.set_tz_cache_location(str(root / "yfinance_cache"))
    requested = sec.now()
    latest = last_completed(datetime.fromisoformat(requested))
    end = through if through is not None else latest
    if (
        end > latest
        or last_completed(datetime.fromisoformat(end + "T23:59:59+00:00")) != end
    ):
        raise ValueError("Market cutoff must be a completed trading session")
    exclusive = (datetime.fromisoformat(end).date() + timedelta(days=1)).isoformat()
    ticker = yf.Ticker(symbol)
    frame = ticker.history(start=start, end=exclusive, raise_errors=True, **PARAMETERS)
    if frame.empty:
        raise ValueError("Empty market history: " + symbol)
    metadata = ticker.get_history_metadata()
    if metadata.get("symbol") != symbol or metadata.get("currency") != "USD":
        raise ValueError("Market issuer/currency mismatch: " + symbol)
    raw = frame.to_csv().encode()
    receipt = dict(
        symbol=symbol,
        provider="yfinance_wrapper_not_raw_http",
        parameters=PARAMETERS,
        requested_start=start,
        last_session=end,
        exclusive_end=exclusive,
        requested_at=requested,
        retrieved_at=sec.now(),
        sha256=sec.digest(raw),
        bytes=len(raw),
        yfinance_version=yf.__version__,
        metadata={
            k: metadata.get(k)
            for k in [
                "symbol",
                "currency",
                "exchangeName",
                "instrumentType",
                "exchangeTimezoneName",
                "longName",
            ]
        },
    )
    if through is not None:
        receipt["market_through"] = through
    # Keep the exact wrapper CSV in the current source bundle, not a monthly report.
    return raw, receipt


def validate_market(raw, receipt, *, expected_start=None, expected_end=None):
    import numpy as np
    import pandas as pd
    import exchange_calendars as xcals

    symbol, start, end = [
        receipt[k] for k in ("symbol", "requested_start", "last_session")
    ]
    wanted_start = (
        company(symbol)["requested_start"]
        if symbol in COMPANIES
        else BENCHMARK_STARTS[symbol]
    )
    if symbol not in COMPANIES and symbol not in BENCHMARKS:
        raise ValueError("Unexpected listing")
    requested = datetime.fromisoformat(receipt["requested_at"])
    retrieved = datetime.fromisoformat(receipt["retrieved_at"])
    if (
        requested.tzinfo is None
        or retrieved.tzinfo is None
        or not requested <= retrieved <= datetime.now(timezone.utc)
        or start != (wanted_start if expected_start is None else expected_start)
        or end != (last_completed(requested) if expected_end is None else expected_end)
        or end > last_completed(requested)
        or last_completed(datetime.fromisoformat(end + "T23:59:59+00:00")) != end
        or receipt.get("market_through") != expected_end
        or receipt["exclusive_end"]
        != (datetime.fromisoformat(end).date() + timedelta(days=1)).isoformat()
        or receipt["metadata"].get("symbol") != symbol
        or receipt["metadata"].get("currency") != "USD"
    ):
        raise ValueError("Market identity/window mismatch")
    if receipt["parameters"] != PARAMETERS or sec.digest(raw) != receipt["sha256"]:
        raise ValueError("Market parameters/checksum mismatch")
    frame = pd.read_csv(io.BytesIO(raw))
    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    ]
    # Funds report capital gain distributions; other listings omit the column.
    optional = {"Capital Gains"}
    expected_columns = {"Date", *required}
    if set(frame.columns) not in (expected_columns, expected_columns | optional):
        raise ValueError("Unexpected market columns: " + symbol)
    frame.insert(0, "session_date", frame.Date.astype(str).str[:10])
    expected = xcals.get_calendar("XNYS", start=start, end=end).sessions.strftime(
        "%Y-%m-%d"
    )
    missing = sorted(set(expected) - set(frame.session_date))
    extra = sorted(set(frame.session_date) - set(expected))
    present_optional = sorted(optional & set(frame.columns))
    numeric = [*required, *present_optional]
    values = frame[numeric]
    invalid = (
        ~np.isfinite(values.to_numpy()).all(axis=1)
        | (frame[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any(axis=1)
        | (frame.Low > frame[["Open", "Close"]].min(axis=1))
        | (frame.High < frame[["Open", "Close"]].max(axis=1))
        | (frame[["Volume", "Dividends", "Stock Splits", *present_optional]] < 0).any(
            axis=1
        )
    )
    if (
        missing
        or extra
        or frame.session_date.duplicated().any()
        or invalid.any()
        or not frame.session_date.is_monotonic_increasing
    ):
        raise ValueError(
            f"Market coverage/price failure {symbol}: missing={len(missing)}, extra={len(extra)}, invalid={int(invalid.sum())}"
        )
    capital_gain_events = (
        int((frame["Capital Gains"] != 0).sum()) if "Capital Gains" in frame else 0
    )
    # The source wrapper retains ETF capital-gain distributions. Daily bars use
    # vendor-adjusted returns for the benchmark's total-return proxy and retain
    # the existing shared equity schema rather than reclassifying gains as dividends.
    frame = frame.drop(columns=present_optional)
    frame.insert(0, "security_id", "yahoo:" + symbol)
    frame.insert(0, "symbol", symbol)
    frame["issuer_id"] = company(symbol)["issuer_id"] if symbol in COMPANIES else ""
    frame["instrument_kind"] = (
        "company_listing" if symbol in COMPANIES else "benchmark_index"
    )
    frame["price_return"] = frame.Close.pct_change(fill_method=None)
    frame["vendor_adjusted_return"] = frame["Adj Close"].pct_change(fill_method=None)
    frame["source_sha256"] = receipt["sha256"]
    return frame, dict(
        rows=len(frame),
        first=frame.session_date.min(),
        last=frame.session_date.max(),
        missing_sessions=missing,
        unexpected_sessions=extra,
        invalid_rows=int(invalid.sum()),
        dividend_events=int((frame.Dividends != 0).sum()),
        split_events=int((frame["Stock Splits"] != 0).sum()),
        capital_gain_events=capital_gain_events,
    )


def build(root, bundle, market_fetch=market):
    import pandas as pd

    path = run_path(root, bundle)
    content = load(path, bundle)
    if content["schema"] == "company-update-v1.1":
        from carina.updates import build_update

        return build_update(root, bundle, content, market_fetch)
    # Legacy offline replay covers the whole approved panel only. Live expansion
    # always uses the selective-update envelope, not this path.
    if set(content["companies"]) != set(COMPANIES):
        raise ValueError("Legacy replay requires the full approved panel")
    output = path_for(root, RUNS + "/" + sec.identity(bundle) + "/built")
    if any((output / name).is_symlink() for name in FILES | {"manifest.json"}):
        raise ValueError("Symlink generated table")
    output.mkdir(exist_ok=True)
    started = time.monotonic()
    tables = dict(
        companies=[],
        securities=[],
        fundamentals=[],
        financial_latest=[],
        filings=[],
        field_applicability=[],
    )
    coverage = {}
    frames = []
    market_sources = {}
    for symbol in [*content["companies"], *BENCHMARKS]:
        spec = (
            company(symbol)
            if symbol in COMPANIES
            else dict(requested_start=BENCHMARK_STARTS[symbol], name=BENCHMARKS[symbol])
        )
        raw, receipt = market_fetch(symbol, spec["requested_start"], root)
        frame, market_coverage = validate_market(raw, receipt)
        frames.append(frame)
        market_sources[symbol] = receipt
        tables["securities"].append(
            dict(
                symbol=symbol,
                security_id="yahoo:" + symbol,
                issuer_id=spec.get("issuer_id", ""),
                name=spec["name"],
                kind="company_listing" if symbol in COMPANIES else "benchmark_index",
                identity_status="VENDOR_CONTINUITY_NOT_CERTIFIED",
            )
        )
        coverage[symbol] = dict(market=market_coverage)
        if symbol not in COMPANIES:
            continue
        tables["companies"].append(spec)
        item = content["companies"][symbol]
        _, canonical, latest, issues = norm.normalize(
            item["facts"],
            item["filings"],
            item["receipts"]["facts"],
            metrics=metrics_for(symbol),
        )
        if not canonical or any(r["status"] != "REPORTED" for r in canonical + latest):
            raise ValueError("Missing/conflicting normalized financials: " + symbol)
        for rows in [canonical, latest, item["filings"], issues]:
            for row in rows:
                row.update(symbol=symbol, issuer_id=spec["issuer_id"])
        tables["fundamentals"].extend(canonical)
        tables["financial_latest"].extend(latest)
        tables["filings"].extend(item["filings"])
        tables["field_applicability"].extend(applicability(symbol))
        coverage[symbol]["financials"] = dict(
            versions=len(canonical),
            latest_rows=len(latest),
            excluded_rows=len(issues),
            excluded_facts=rejection_observations(symbol, issues, item),
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
        print(
            f"Built {symbol}: {len(frame)} sessions, {len(canonical)} financial versions",
            flush=True,
        )
    pd.concat(frames, ignore_index=True).to_parquet(
        output / "daily_bars.parquet", index=False, compression="zstd"
    )
    for name, rows in tables.items():
        pd.DataFrame(rows).to_parquet(
            output / (name + ".parquet"), index=False, compression="zstd"
        )
    replace_generated(output / "coverage.json", sec.encoded(coverage))
    replace_generated(
        output / SOURCES,
        sec.encoded(
            dict(
                market=market_sources,
                sec={
                    s: sec_receipts(item, content["collected_at"])
                    for s, item in content["companies"].items()
                },
            )
        ),
    )
    manifest = dict(
        schema=SCHEMA,
        status="BUILT_NOT_PUBLISHED",
        bundle=bundle,
        collected_at=content["collected_at"],
        built_at=sec.now(),
        code_commit=sec.code_version(),
        companies=list(content["companies"]),
        benchmarks=list(BENCHMARKS),
        processing_libraries={
            k: version(k) for k in ["pandas", "pyarrow", "lxml", "exchange-calendars"]
        },
        collection_mode="FULL_SELECTED_SEC_FACTS_AND_FULL_DAILY_HISTORY",
        build_seconds=round(time.monotonic() - started, 2),
        limits=LIMITS,
        files={
            p.name: dict(sha256=sec.digest(p.read_bytes()), bytes=p.stat().st_size)
            for p in sorted(output.iterdir())
            if p.name != "manifest.json"
        },
    )
    replace_generated(output / "manifest.json", sec.encoded(manifest))
    return output


def verify_local_files(path, manifest, *, require_all_benchmarks=False):
    """Verify schema and every file hash without decoding the full tables."""
    if (
        manifest["schema"] not in (SCHEMA, dataset.SCHEMA)
        or not set(manifest["companies"]) <= set(COMPANIES)
        or not manifest["companies"]
        or not set(manifest["benchmarks"]) <= set(BENCHMARKS)
        or (require_all_benchmarks and set(manifest["benchmarks"]) != set(BENCHMARKS))
    ):
        raise ValueError("Unexpected central-store schema/panel")
    dataset.verify_files(path, manifest)


def verify_local(path, manifest, *, require_all_benchmarks=False):

    verify_local_files(path, manifest, require_all_benchmarks=require_all_benchmarks)

    tables = {
        name: dataset.read_table(path, name)
        for name in (
            "companies",
            "securities",
            "daily_bars",
            "fundamentals",
            "financial_latest",
        )
    }
    if (
        set(tables["companies"].symbol) != set(manifest["companies"])
        or tables["companies"].issuer_id.duplicated().any()
        or tables["securities"].security_id.duplicated().any()
        or set(tables["daily_bars"].symbol)
        != set(manifest["companies"]) | set(manifest["benchmarks"])
        or tables["daily_bars"].duplicated(["security_id", "session_date"]).any()
    ):
        raise ValueError("Invalid current-store primary keys or panel coverage")
    for name in ("fundamentals", "financial_latest"):
        frame = tables[name]
        key = ["issuer_id", "metric", "basis", "unit", "start", "end", "period_kind"]
        if name == "fundamentals":
            key.append("accession")
        if frame.duplicated(key).any() or not set(frame.issuer_id) <= set(
            tables["companies"].issuer_id
        ):
            raise ValueError("Invalid financial primary/foreign keys")


def main(argv=None):
    p = argparse.ArgumentParser(prog="carina", description=__doc__)
    p.add_argument("action", choices=["collect", "build", "verify"])
    p.add_argument("--run-id")
    p.add_argument(
        "--market-through",
        help="Explicit completed-session cutoff for NEW company histories only (YYYY-MM-DD)",
    )
    p.add_argument(
        "--add-companies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch histories only for missing selected companies (default: on)",
    )
    p.add_argument(
        "--add-periods",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Extend existing selected companies and benchmarks (default: off)",
    )
    p.add_argument(
        "--add-benchmarks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add newly registered benchmarks without refreshing covered histories (default: off)",
    )
    p.add_argument(
        "--full-refresh",
        action="store_true",
        help="Explicitly replace selected existing histories and benchmark history",
    )
    p.add_argument(
        "--companies",
        nargs="+",
        help="Approved symbols; defaults to the whole approved panel",
    )
    p.add_argument("--bundle")
    a = p.parse_args(argv)
    root = settings.data().resolve()
    a.companies = a.companies or list(COMPANIES)
    if set(a.companies) - set(COMPANIES):
        p.error("--companies must be approved symbols")
    if a.action == "collect":
        collect(
            root,
            a.companies,
            a.run_id,
            add_companies=a.add_companies,
            add_periods=a.add_periods,
            add_benchmarks=a.add_benchmarks,
            full_refresh=a.full_refresh,
            market_through=a.market_through,
        )
    elif a.action == "build":
        build(root, a.bundle)
    else:
        path = path_for(root, STAGED)
        manifest = json.loads((path / "manifest.json").read_bytes())
        if manifest["status"] != "VERIFIED":
            raise ValueError("Staged dataset is not verified")
        verify_local(path, manifest)
        print("Staged dataset checksums verified")

