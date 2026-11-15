"""Refresh the company market data from the provider into the local staged dataset.

One command plans against the local dataset, collects with checkpoints, inspects
every series, builds a candidate offline, verifies it, and promotes it. It never
touches Drive. Rerun the same command to resume after an interruption.

    carina refresh market [--run-id ID] [--through DATE] [--symbols A B ...] [--retry A B ...]
        [--replace A B ...] [--yes]

The data directory comes from ``carina --data`` or ``CARINA_DATA``.

Exit status: 0 promoted (or nothing to do), 2 stopped for review, 130 interrupted.
"""

from __future__ import annotations

import argparse
import bz2
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re
import os
import shutil
import time

from carina import common as sec, store, updates, dataset, settings

RUNS = store.RUNS
RECEIPTS = store.RECEIPTS
SCHEMA = "company-refresh-stage-1.0"
STOPPED = 2
CHECKPOINT_EVERY = 25
FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits")
CRITICAL = {"Close", "Adj Close", "Dividends", "Stock Splits"}


def run_dir(root, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", run_id or ""):
        raise ValueError("Invalid refresh run ID")
    return store.path_for(root, RUNS + "/" + run_id)


def stage_file(root, run_id):
    return run_dir(root, run_id) / "stage.json.bz2"


def save(root, run_id, state):
    store.replace_generated(stage_file(root, run_id), bz2.compress(sec.encoded(state)))


def load(root, run_id):
    state = json.loads(bz2.decompress(stage_file(root, run_id).read_bytes()))
    if state["schema"] != SCHEMA or state["run_id"] != run_id:
        raise ValueError("Staging identity mismatch")
    updates.validate_capture(envelope(state))
    return state


def envelope(state):
    return dict(
        schema=updates.SCHEMA,
        plan=state["plan"],
        collected_at=state["collected_at"],
        companies={},
    )


def identity(state):
    """Digest of the exact inputs a review and a candidate are bound to."""
    return sec.digest(
        sec.encoded(
            {k: state[k] for k in ("schema", "run_id", "plan", "collected_at", "captures")}
        )
    )


def key(symbol, task):
    return sec.digest(sec.encoded(dict(symbol=symbol, **task)))


def baseline(root, plan):
    """The staged dataset must still be the one this run was planned against."""
    path = store.path_for(root, store.STAGED)
    raw = (path / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("status") != "VERIFIED" or sec.digest(raw) != plan["base"]["manifest_sha256"]:
        raise ValueError("The staged dataset changed since this run was planned")
    return path, manifest


def initialize(root, run_id, *, through=None, symbols=None):
    """Create the run's plan, or reload it and check its baseline is unchanged."""
    if stage_file(root, run_id).exists():
        state = load(root, run_id)
        baseline(root, state["plan"])
        return state
    dataset.recover(store.path_for(root, store.STAGED))
    current = updates.read_current(root, files_only=True)
    if current is None:
        raise ValueError("A refresh needs an existing verified staged dataset")
    cutoff = through or store.last_completed(datetime.fromisoformat(sec.now()))
    chosen = list(symbols) if symbols else current[1]["companies"]
    if set(chosen) - set(current[1]["companies"]):
        raise ValueError("Refresh symbols must already be in the staged dataset")
    req = updates.request(
        chosen,
        add_companies=False,
        add_periods=True,
        refresh_financials=False,
        market_through=cutoff,
    )
    state = dict(
        schema=SCHEMA,
        run_id=run_id,
        plan=updates.snapshot(root, req),
        collected_at=sec.now(),
        captures={},
        errors={},
    )
    save(root, run_id, state)
    return state


def names(value):
    """Accept spaces or a pasted comma-separated list; reject duplicates."""
    result = [s for s in re.split(r"[\s,]+", " ".join(value or [])) if s]
    if len(result) != len(set(result)):
        raise ValueError("Duplicate symbol")
    return result


def replace(root, run_id, symbols):
    """Schedule full-history refetches for the named symbols; resets the review."""
    state = load(root, run_id)
    req = state["plan"]["request"]
    planned = set(req["companies"]) | set(state["plan"]["base"]["benchmarks"])
    if set(symbols) - planned:
        raise ValueError("Replacement symbols outside this run: " + ", ".join(sorted(set(symbols) - planned)))
    wanted = sorted(set(req.get("market_refresh_symbols", [])) | set(symbols))
    if wanted == req.get("market_refresh_symbols"):
        return state
    state["plan"]["request"] = updates.request(
        **dict(
            symbols=req["companies"],
            **{k: v for k, v in req.items() if k not in {"companies", "market_refresh_symbols"}},
            market_refresh_symbols=wanted,
        )
    )
    state.pop("review", None)
    save(root, run_id, state)
    return state


def readiness(root, tasks, fetch):
    """Probe at most three exact planned responses before requesting the panel."""
    selected = [s for s in ("AAPL", "COST", "RSP") if s in tasks]
    selected += [s for s in tasks if s not in selected][: 3 - len(selected)]
    samples = []
    for symbol in selected:
        cutoff, start = tasks[symbol]["through"], tasks[symbol]["start"]
        sample = dict(symbol=symbol, session_date=cutoff, start=start, status="READY")
        try:
            raw, receipt = fetch(symbol, start, root, through=cutoff)
            sample.update(receipt=receipt, wrapper_csv=raw.decode())
            if receipt.get("symbol") != symbol:
                raise ValueError("Readiness source identity mismatch")
            store.validate_market(raw, receipt, expected_start=start, expected_end=cutoff)
        except Exception as exc:
            sample.update(status="WAITING_FOR_SOURCE", error=str(exc))
        samples.append(sample)
    ready = all(s["status"] == "READY" for s in samples)
    return dict(status="READY" if ready else "WAITING_FOR_SOURCE", checked_at=sec.now(), samples=samples)


def clock(seconds):
    return str(int(seconds) // 3600) + ":" + format(int(seconds) % 3600 // 60, "02d") + ":" + format(int(seconds) % 60, "02d")


def collect(root, run_id, fetch=None, *, retry=()):
    """Capture every planned series not yet captured. Interrupt-safe and resumable."""
    state = load(root, run_id)
    baseline(root, state["plan"])
    tasks = updates.operations(state["plan"])["market"]
    if set(retry) - set(tasks):
        raise ValueError("Retry symbols outside the planned market scope")
    provider = fetch or store.market
    pending = {
        symbol: task
        for symbol, task in tasks.items()
        if key(symbol, task) not in state["captures"] or symbol in retry
    }
    if not pending:
        return state
    state["readiness"] = readiness(root, pending, provider)
    save(root, run_id, state)
    if state["readiness"]["status"] != "READY":
        errors = "; ".join(s["error"] for s in state["readiness"]["samples"] if s["status"] != "READY")
        print("SOURCE_NOT_READY: " + errors + "\nNothing was lost; rerun later to resume.", flush=True)
        return state
    probed = {s["symbol"]: s for s in state["readiness"]["samples"]}
    started, done = time.monotonic(), 0
    try:
        for index, (symbol, task) in enumerate(tasks.items(), 1):
            if symbol not in pending:
                continue
            elapsed = time.monotonic() - started
            eta = elapsed / done * (len(pending) - done) if done else 0
            print(
                f"MARKET {index}/{len(tasks)} {symbol} elapsed {clock(elapsed)}"
                + (f" remaining ~{clock(eta)}" if done else ""),
                flush=True,
            )
            try:
                if symbol in probed:
                    raw, receipt = probed[symbol]["wrapper_csv"].encode(), probed[symbol]["receipt"]
                else:
                    raw, receipt = provider(symbol, task["start"], root, through=task["through"])
                # Keep even invalid responses so inspection can show them.
                state["captures"][key(symbol, task)] = dict(
                    symbol=symbol, task=task, receipt=receipt, wrapper_csv=raw.decode()
                )
                state["errors"].pop(symbol, None)
            except Exception as exc:  # a source failure stays local to this series
                state["errors"][symbol] = dict(type=type(exc).__name__, message=str(exc))
            state["collected_at"] = sec.now()
            done += 1
            if done % CHECKPOINT_EVERY == 0:
                save(root, run_id, state)
    except KeyboardInterrupt:
        state["collected_at"] = sec.now()
        save(root, run_id, state)
        print(f"\nInterrupted; saved {len(state['captures'])} captures. Rerun to resume.", flush=True)
        raise SystemExit(130)
    state.pop("review", None)
    save(root, run_id, state)
    return state


def inspect(root, run_id, *, persist_review=True):
    """Compare every capture with the retained data at its continuity boundary."""
    import numpy as np

    state = load(root, run_id)
    current_path, _ = baseline(root, state["plan"])
    old = dataset.read_table(current_path, "daily_bars")
    boundaries = old.sort_values("session_date").groupby("symbol").tail(1).set_index("symbol")
    cases = []
    for symbol, task in updates.operations(state["plan"])["market"].items():
        item = state["captures"].get(key(symbol, task))
        case = dict(symbol=symbol, status="READY", changes={}, boundary_date=task["start"])
        try:
            if symbol in state["errors"] or item is None:
                raise ValueError(state["errors"].get(symbol, {}).get("message", "Missing capture"))
            if item["symbol"] != symbol or item["task"] != task or item["receipt"].get("symbol") != symbol:
                raise ValueError("Retained market capture identity/scope mismatch")
            frame, _ = store.validate_market(
                item["wrapper_csv"].encode(),
                item["receipt"],
                expected_start=task["start"],
                expected_end=task["through"],
            )
            if task["mode"] == "append":
                boundary = frame.loc[frame.session_date == task["start"]]
                if len(boundary) != 1:
                    raise ValueError("Missing continuity boundary")
                for field in FIELDS:
                    a, b = float(boundaries.loc[symbol, field]), float(boundary.iloc[0][field])
                    if not np.isclose(a, b, rtol=1e-6, atol=1e-6):
                        case["changes"][field] = dict(retained=a, provider=b)
                later = frame.session_date > task["start"]
                splits = frame.loc[later & (frame["Stock Splits"] != 0), ["session_date", "Stock Splits"]]
                if not splits.empty:
                    case["changes"]["new_split_events"] = splits.to_dict("records")
                if set(case["changes"]) & CRITICAL or not splits.empty:
                    case["status"] = "NEEDS_REVIEW"
            else:
                case["replacement"] = True
                previous = old.loc[old.symbol == symbol].set_index("session_date")
                incoming = frame.set_index("session_date")
                missing = sorted(set(previous.index) - set(incoming.index))
                if missing:
                    raise ValueError(f"Replacement removes {len(missing)} retained sessions")
                shared = previous.index.intersection(incoming.index)
                historical = {}
                for field in FIELDS:
                    a, b = previous.loc[shared, field], incoming.loc[shared, field]
                    changed = ~np.isclose(a, b, rtol=1e-6, atol=1e-6)
                    if changed.any():
                        historical[field] = dict(
                            count=int(changed.sum()),
                            max_absolute_difference=float((a - b).abs().max()),
                            samples=[
                                dict(session_date=day, retained=float(a.loc[day]), provider=float(b.loc[day]))
                                for day in shared[changed][:5]
                            ],
                        )
                case["historical_changes"] = historical
        except (ValueError, KeyError, TypeError) as exc:
            case.update(status="BLOCKED", error=str(exc))
        case["fingerprint"] = sec.digest(sec.encoded(case))
        cases.append(case)
    report = dict(
        stage_sha256=identity(state),
        cases=cases,
        status="READY" if all(c["status"] == "READY" for c in cases) else "NEEDS_REVIEW",
    )
    if persist_review:
        state["review"] = report
        save(root, run_id, state)
    return report


def describe(case):
    if case["status"] == "BLOCKED":
        return case["error"]
    changes = case.get("historical_changes", case["changes"])
    if not changes:
        return "no differences"
    parts = []
    for field, value in changes.items():
        if field == "new_split_events":
            parts.append("new split " + ", ".join(str(e["session_date"]) for e in value))
        elif "count" in value:
            parts.append(f"{field}: {value['count']} rows, max diff {value['max_absolute_difference']:.6g}")
        else:
            parts.append(f"{field}: {value['retained']:.10g} -> {value['provider']:.10g}")
    return "; ".join(parts)


def exceptions(report):
    return [c for c in report["cases"] if c["status"] != "READY" or c.get("replacement")]


def stop_message(run_id, report, replaced, accepted):
    """The table of what needs a decision, and the exact commands that resolve it."""
    command = f"{os.environ.get('CARINA_PROG', 'carina')} refresh market --run-id {run_id}"
    lines = ["", "| Symbol | Status | What changed |", "|---|---|---|"]
    lines += [f"| {c['symbol']} | {c['status']}{' (replacement)' if c.get('replacement') else ''} | {describe(c)} |" for c in exceptions(report)]
    review = [c["symbol"] for c in report["cases"] if c["status"] == "NEEDS_REVIEW"]
    blocked = [c["symbol"] for c in report["cases"] if c["status"] == "BLOCKED"]
    lines.append("")
    if blocked:
        lines.append("Blocked series have no usable capture. Retry them, or investigate:")
        lines.append(f"  {command} --retry {' '.join(blocked)}")
    if review:
        lines.append("The provider revised retained history for these. To refetch their full history:")
        lines.append(f"  {command} --replace {' '.join(review)}")
    if replaced and not blocked and not review and not accepted:
        lines.append("Replaced histories change retained data. After reading the table, accept with:")
        lines.append(f"  {command} --yes")
    return "\n".join(lines)


def quality_summary(report):
    summary = {k: report.get(k) for k in ("status", "open_flags", "flag_counts")}
    summary["transition_flags"] = [
        f for f in report.get("flags", []) if f.get("rule") in {"HISTORICAL_COVERAGE_LOSS", "NEW_SOURCE_TAG"}
    ]
    return summary


def build(root, run_id):
    """Build the candidate offline from the retained captures; reuse a verified one."""
    state = load(root, run_id)
    baseline(root, state["plan"])
    expected = identity(state)
    review = state.get("review", {})
    if review.get("stage_sha256") != expected or review["status"] != "READY" or state["errors"]:
        raise ValueError("Unresolved market exceptions prevent a build")
    content = envelope(state)
    updates.validate_capture(content)
    bundle = sec.digest(sec.encoded(content))
    output = run_dir(root, run_id) / "candidate"
    retained = output / "manifest.json"
    if retained.exists():
        manifest = json.loads(retained.read_bytes())
        if manifest.get("maintenance_stage_sha256") == expected and store.validation_is_current(manifest):
            store.verify_local_files(output, manifest)
            return output
        shutil.rmtree(output)

    def offline(symbol, start, unused_root, **scope):
        task = updates.operations(state["plan"])["market"][symbol]
        item = state["captures"][key(symbol, task)]
        if start != task["start"] or scope != dict(through=task["through"]):
            raise ValueError("Cached capture scope mismatch")
        return item["wrapper_csv"].encode(), deepcopy(item["receipt"])

    try:
        built = updates.build_update(root, bundle, content, offline, output=output)
    except Exception:
        if retained.exists():  # keep the compact evidence of why it cannot promote
            partial = json.loads(retained.read_bytes())
            store.replace_generated(
                run_dir(root, run_id) / "candidate_review.json",
                sec.encoded(
                    dict(
                        run_id=run_id,
                        stage_sha256=expected,
                        quality=quality_summary(partial.get("quality", {})),
                        market_quality=quality_summary(partial.get("market_quality", {})),
                    )
                ),
            )
        raise
    if built is None:
        return None
    manifest = json.loads((built / "manifest.json").read_bytes())
    manifest["maintenance_stage_sha256"] = expected
    store.bind_validation(manifest)
    store.replace_generated(built / "manifest.json", sec.encoded(manifest))
    return built


def verify_candidate(candidate):
    """Check a candidate's files and assessments, recomputing any that are not current."""
    from carina import market_quality
    from carina import quality

    manifest = json.loads((candidate / "manifest.json").read_bytes())
    store.verify_local_files(candidate, manifest)
    if not store.validation_is_current(manifest):
        store.verify_local(candidate, manifest)
        manifest["quality"] = quality.assess(candidate)
        manifest["market_quality"] = market_quality.assess(candidate)
        store.bind_validation(manifest)
        store.replace_generated(candidate / "manifest.json", sec.encoded(manifest))
    if manifest["quality"]["status"] == "BLOCKED":
        raise ValueError("DATA_QUALITY_BLOCKED: candidate integrity flags prevent promotion")
    if manifest["market_quality"]["status"] == "BLOCKED":
        raise ValueError("MARKET_DATA_BLOCKED: invalid market data prevents promotion")
    return manifest


def promote(root, run_id, candidate):
    """Verify the candidate, make it the staged dataset, record a receipt, clean up."""
    manifest = verify_candidate(candidate)
    staged = store.path_for(root, store.STAGED)
    before = len(dataset.read_table(staged, "daily_bars", columns=["symbol"]))
    bound = store.validation_is_current(manifest)

    def finalize(new):
        return store.bind_validation(new) if bound else new

    promoted = dataset.promote(candidate, staged, finalize=finalize)
    after = len(dataset.read_table(staged, "daily_bars", columns=["symbol"]))
    state = load(root, run_id)
    receipt = dict(
        run_id=run_id,
        stage_sha256=identity(state),
        promoted_manifest_sha256=dataset.manifest_digest(staged),
        lineage=promoted["lineage"],
        market_rows_added=after - before,
        quality=quality_summary(manifest["quality"]),
        market_quality=quality_summary(manifest["market_quality"]),
        promoted_at=sec.now(),
    )
    store.replace_generated(store.path_for(root, RECEIPTS + "/" + run_id + ".json"), sec.encoded(receipt))
    shutil.rmtree(run_dir(root, run_id))
    return receipt


def run(root, run_id=None, *, through=None, symbols=None, replace_symbols=(), retry=(), yes=False, fetch=None):
    """The whole refresh. Returns 0 when promoted or unchanged, STOPPED when it needs you."""
    root = Path(root).resolve()
    run_id = run_id or through or store.last_completed(datetime.fromisoformat(sec.now()))
    receipt_path = store.path_for(root, RECEIPTS + "/" + run_id + ".json")
    if receipt_path.exists():
        print(f"Run {run_id} was already promoted. Use --run-id to start another.")
        return 0
    state = initialize(root, run_id, through=through, symbols=symbols)
    if replace_symbols:
        state = replace(root, run_id, replace_symbols)
    state = collect(root, run_id, fetch, retry=retry)
    if state["errors"] or any(key(s, t) not in state["captures"] for s, t in updates.operations(state["plan"])["market"].items()):
        missing = sorted(set(updates.operations(state["plan"])["market"]) - {c["symbol"] for c in state["captures"].values()})
        print(f"{len(state['captures'])} captured; {len(missing)} missing: {', '.join(missing[:20])}")
        for symbol, error in list(state["errors"].items())[:20]:
            print(f"  {symbol}: {error['message']}")
        print("Rerun the same command to retry the missing series.")
        return STOPPED
    report = inspect(root, run_id)
    replaced = [c for c in report["cases"] if c.get("replacement")]
    needs_decision = report["status"] != "READY" or (replaced and not yes)
    if needs_decision:
        print(f"Run {run_id}: {len(report['cases'])} series inspected, {len(exceptions(report))} need a decision.")
        print(stop_message(run_id, report, replaced, yes))
        return STOPPED
    candidate = build(root, run_id)
    if candidate is None:
        shutil.rmtree(run_dir(root, run_id))
        print("Nothing to update: the staged dataset already has the latest completed session.")
        return 0
    receipt = promote(root, run_id, candidate)
    print(
        json.dumps(
            dict(
                promoted=receipt["promoted_manifest_sha256"],
                series=len(report["cases"]),
                replaced=[c["symbol"] for c in replaced],
                market_rows_added=receipt["market_rows_added"],
                quality=receipt["quality"]["status"],
                open_quality_flags=receipt["quality"]["open_flags"],
                market_quality=receipt["market_quality"]["status"],
                open_market_flags=receipt["market_quality"]["open_flags"],
            )
        )
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="carina refresh", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scope", choices=["market"])
    parser.add_argument("--run-id", help="Defaults to the session cutoff date; rerunning resumes it")
    parser.add_argument("--through", help="Completed-session cutoff (YYYY-MM-DD); default: latest completed session")
    parser.add_argument("--symbols", nargs="+", help="Limit the refresh to these companies (benchmarks always included)")
    parser.add_argument("--retry", nargs="+", help="Refetch these series even if captured")
    parser.add_argument("--replace", nargs="+", help="Refetch the full history of these series")
    parser.add_argument("--yes", action="store_true", help="Accept reviewed full-history replacements")
    args = parser.parse_args(argv)
    return run(
        settings.data(),
        args.run_id,
        through=args.through,
        symbols=names(args.symbols) or None,
        replace_symbols=names(args.replace),
        retry=names(args.retry),
        yes=args.yes,
    )

