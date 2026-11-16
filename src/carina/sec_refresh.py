"""Refresh selected SEC financial facts in the staged company dataset.

The command scans each selected issuer's recent submissions, fetches companyfacts
only when a new financial filing is present, builds and verifies a candidate, and
promotes it. Full provider payloads are never written to disk.

    carina refresh sec [--run-id ID] [--symbols A B ...]

Exit status: 0 promoted (or nothing to do), 1 source failure, 2 stopped for
review because an existing stored reporting version would change.
"""

from __future__ import annotations

import argparse
import bz2
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from carina import common as sec
from carina import dataset, refresh, settings, store, updates

SCHEMA = "company-sec-refresh-stage-1.0"
STOPPED = refresh.STOPPED
MIN_REQUEST_INTERVAL = 0.1


class RateLimiter:
    """Start no more than ten SEC endpoint requests per second."""

    def __init__(self, interval=MIN_REQUEST_INTERVAL, *, clock=None, sleep=None):
        self.interval = interval
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self.last = None

    def wait(self):
        now = self.clock()
        if self.last is not None:
            remaining = self.interval - (now - self.last)
            if remaining > 0:
                self.sleep(remaining)
                now = self.clock()
        self.last = now


def stage_file(root, run_id):
    return refresh.run_dir(root, run_id) / "sec-stage.json.bz2"


def save(root, run_id, state):
    store.replace_generated(stage_file(root, run_id), bz2.compress(sec.encoded(state)))


def load(root, run_id):
    state = json.loads(bz2.decompress(stage_file(root, run_id).read_bytes()))
    if state.get("schema") != SCHEMA or state.get("run_id") != run_id:
        raise ValueError("SEC refresh staging identity mismatch")
    return state


def identity(state):
    return sec.digest(
        sec.encoded(
            {
                key: state[key]
                for key in (
                    "schema",
                    "run_id",
                    "scan_plan",
                    "collected_at",
                    "scans",
                    "captures",
                )
            }
        )
    )


def _baseline(root, plan):
    return refresh.baseline(root, plan)


def initialize(root, run_id, symbols=None):
    if stage_file(root, run_id).exists():
        state = load(root, run_id)
        _baseline(root, state["scan_plan"])
        return state
    dataset.recover(store.path_for(root, store.STAGED))
    current = updates.read_current(root, files_only=True)
    if current is None:
        raise ValueError("An SEC refresh needs an existing verified staged dataset")
    selected = list(symbols) if symbols else list(current[1]["companies"])
    if not selected or set(selected) - set(current[1]["companies"]):
        raise ValueError("SEC refresh symbols must already be in the staged dataset")
    req = updates.request(
        selected,
        add_companies=False,
        add_periods=True,
        refresh_financials=True,
        refresh_market=False,
    )
    state = {
        "schema": SCHEMA,
        "run_id": run_id,
        "scan_plan": updates.snapshot(root, req),
        "collected_at": sec.now(),
        "scans": {},
        "captures": {},
        "errors": {},
    }
    save(root, run_id, state)
    return state


def _stored_filings(root):
    path = store.path_for(root, store.STAGED)
    frame = dataset.read_table(path, "filings")
    result = {}
    if frame.empty or "symbol" not in frame.columns:
        return result
    for symbol, rows in frame.groupby("symbol", sort=False):
        records = rows.to_dict("records")
        result[symbol] = {
            "accessions": {str(row["accessionNumber"]) for row in records},
            "latest_date": max((str(row["filingDate"]) for row in records), default=""),
        }
    return result


def _new_filings(filings, stored):
    """Include same-day new accessions while ignoring gaps older than the watermark."""
    accessions = stored.get("accessions", set())
    latest = stored.get("latest_date", "")
    return [
        row
        for row in filings
        if row["accessionNumber"] not in accessions and row["filingDate"] >= latest
    ]


def collect(root, run_id, *, fetch=None, limiter=None):
    """Checkpoint submissions scans and selected companyfacts projections."""
    state = load(root, run_id)
    _baseline(root, state["scan_plan"])
    provider = fetch or store.fetch_sec_json
    limiter = limiter or RateLimiter()
    stored = _stored_filings(root)
    selected = state["scan_plan"]["request"]["companies"]
    try:
        for symbol in selected:
            if symbol in state["scans"]:
                continue
            limiter.wait()
            source, receipt, _raw = provider(symbol, "submissions")
            issuer, filings = store.project_submissions(symbol, source)
            new = _new_filings(filings, stored.get(symbol, {}))
            state["scans"][symbol] = {
                "issuer_metadata": issuer,
                "filings": filings,
                "receipt": receipt,
                "new_accessions": [row["accessionNumber"] for row in new],
            }
            state["errors"].pop(symbol, None)
            state["collected_at"] = sec.now()
            save(root, run_id, state)
        for symbol in selected:
            scan = state["scans"][symbol]
            if not scan["new_accessions"] or symbol in state["captures"]:
                continue
            limiter.wait()
            source, receipt, _raw = provider(symbol, "facts")
            facts = store.project_companyfacts(symbol, source)
            state["captures"][symbol] = {
                "issuer_metadata": scan["issuer_metadata"],
                "filings": scan["filings"],
                "facts": facts,
                "projection_sha256": sec.digest(sec.encoded(facts)),
                "receipts": {
                    "submissions": scan["receipt"],
                    "facts": receipt,
                },
            }
            state["errors"].pop(symbol, None)
            state["collected_at"] = sec.now()
            save(root, run_id, state)
    except KeyboardInterrupt:
        save(root, run_id, state)
        raise SystemExit(130)
    except Exception as exc:
        symbol = locals().get("symbol", "unknown")
        state["errors"][symbol] = {"type": type(exc).__name__, "message": str(exc)}
        state["collected_at"] = sec.now()
        save(root, run_id, state)
    return state


def envelope(root, state):
    changed = sorted(state["captures"])
    if not changed:
        return None
    request = updates.request(
        changed,
        add_companies=False,
        add_periods=True,
        refresh_financials=True,
        refresh_market=False,
    )
    plan = updates.snapshot(root, request, planned_at=state["scan_plan"]["planned_at"])
    if plan["base"]["manifest_sha256"] != state["scan_plan"]["base"]["manifest_sha256"]:
        raise ValueError("The staged dataset changed since this run was planned")
    content = {
        "schema": updates.SCHEMA,
        "plan": plan,
        "collected_at": state["collected_at"],
        "companies": state["captures"],
    }
    updates.validate_capture(content)
    return content


def build(root, run_id):
    state = load(root, run_id)
    _baseline(root, state["scan_plan"])
    content = envelope(root, state)
    if content is None:
        return None
    expected = identity(state)
    output = refresh.run_dir(root, run_id) / "candidate"
    retained = output / "manifest.json"
    if retained.exists():
        manifest = json.loads(retained.read_bytes())
        if (
            manifest.get("maintenance_stage_sha256") == expected
            and store.validation_is_current(manifest)
        ):
            store.verify_local_files(output, manifest)
            return output
        shutil.rmtree(output)

    def no_market(*args, **kwargs):
        raise AssertionError("An SEC-only refresh must not request market data")

    bundle = sec.digest(sec.encoded(content))
    candidate = updates.build_update(root, bundle, content, no_market, output=output)
    if candidate is None:
        return None
    manifest = json.loads((candidate / "manifest.json").read_bytes())
    manifest["maintenance_stage_sha256"] = expected
    store.bind_validation(manifest)
    store.replace_generated(candidate / "manifest.json", sec.encoded(manifest))
    return candidate


def promote(root, run_id, candidate):
    manifest = refresh.verify_candidate(candidate)
    staged = store.path_for(root, store.STAGED)
    before_facts = len(dataset.read_table(staged, "fundamentals"))
    before_filings = len(dataset.read_table(staged, "filings"))
    bound = store.validation_is_current(manifest)

    def finalize(new):
        return store.bind_validation(new) if bound else new

    promoted = dataset.promote(candidate, staged, finalize=finalize)
    state = load(root, run_id)
    receipt = {
        "kind": "sec_refresh",
        "run_id": run_id,
        "stage_sha256": identity(state),
        "promoted_manifest_sha256": dataset.manifest_digest(staged),
        "lineage": promoted["lineage"],
        "companies_scanned": len(state["scans"]),
        "companies_refetched": sorted(state["captures"]),
        "new_accessions": {
            symbol: scan["new_accessions"]
            for symbol, scan in state["scans"].items()
            if scan["new_accessions"]
        },
        "fundamental_rows_added": len(dataset.read_table(staged, "fundamentals"))
        - before_facts,
        "filing_rows_added": len(dataset.read_table(staged, "filings")) - before_filings,
        "quality": refresh.quality_summary(manifest["quality"]),
        "market_quality": refresh.quality_summary(manifest["market_quality"]),
        "promoted_at": sec.now(),
    }
    store.replace_generated(
        store.path_for(root, store.RECEIPTS + "/" + run_id + ".json"),
        sec.encoded(receipt),
    )
    shutil.rmtree(refresh.run_dir(root, run_id))
    return receipt


def _finish_no_change(root, run_id, state):
    receipt = {
        "kind": "sec_refresh",
        "run_id": run_id,
        "status": "NO_CHANGES",
        "stage_sha256": identity(state),
        "base_manifest_sha256": state["scan_plan"]["base"]["manifest_sha256"],
        "companies_scanned": len(state["scans"]),
        "submissions_receipts": {
            symbol: scan["receipt"] for symbol, scan in state["scans"].items()
        },
        "completed_at": sec.now(),
    }
    store.replace_generated(
        store.path_for(root, store.RECEIPTS + "/" + run_id + ".json"),
        sec.encoded(receipt),
    )
    shutil.rmtree(refresh.run_dir(root, run_id))
    return receipt


def run(root, run_id=None, *, symbols=None, fetch=None, limiter=None):
    root = Path(root).resolve()
    run_id = run_id or "sec-" + datetime.now(timezone.utc).date().isoformat()
    receipt_path = store.path_for(root, store.RECEIPTS + "/" + run_id + ".json")
    if receipt_path.exists():
        print(f"Run {run_id} is already complete. Use --run-id to start another.")
        return 0
    state = initialize(root, run_id, symbols)
    state = collect(root, run_id, fetch=fetch, limiter=limiter)
    if state["errors"]:
        for symbol, error in state["errors"].items():
            print(f"{symbol}: {error['message']}")
        print("SEC source collection stopped; rerun the same command to resume.")
        return 1
    expected = {
        symbol
        for symbol, scan in state["scans"].items()
        if scan["new_accessions"]
    }
    missing = expected - set(state["captures"])
    if missing:
        print("Missing companyfacts captures: " + ", ".join(sorted(missing)))
        return 1
    if not expected:
        receipt = _finish_no_change(root, run_id, state)
        print(json.dumps({"status": receipt["status"], "companies_scanned": receipt["companies_scanned"]}))
        return 0
    try:
        candidate = build(root, run_id)
    except ValueError as exc:
        if "Existing SEC fact revised" not in str(exc):
            raise
        state = load(root, run_id)
        state["review"] = {
            "status": "NEEDS_REVIEW",
            "reason": str(exc),
            "stage_sha256": identity(state),
        }
        save(root, run_id, state)
        print("SEC_REVIEW_REQUIRED: " + str(exc))
        return STOPPED
    receipt = promote(root, run_id, candidate)
    print(
        json.dumps(
            {
                "promoted": receipt["promoted_manifest_sha256"],
                "companies_scanned": receipt["companies_scanned"],
                "companies_refetched": receipt["companies_refetched"],
                "fundamental_rows_added": receipt["fundamental_rows_added"],
                "filing_rows_added": receipt["filing_rows_added"],
                "quality": receipt["quality"]["status"],
            }
        )
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="carina refresh sec", description=__doc__)
    parser.add_argument("--run-id", help="Stable run identifier; rerunning resumes it")
    parser.add_argument("--symbols", nargs="+", help="Limit the scan to existing companies")
    args = parser.parse_args(argv)
    return run(settings.data(), args.run_id, symbols=refresh.names(args.symbols) or None)
