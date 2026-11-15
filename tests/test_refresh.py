from copy import deepcopy
from datetime import date, timedelta
import io
import json

import pandas as pd
import pytest

from tests.test_store import built, fake_market  # noqa: F401
from tests.test_updates import state  # noqa: F401
from carina import common as sec, store, dataset, refresh

FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
RUN = "test"
THROUGH = "1997-01-06"


def plan(root, run=RUN, **kwargs):
    return refresh.initialize(root, run, through=THROUGH, **kwargs)


def provider(root):
    """A fake provider whose first row matches the retained boundary row exactly."""
    before = dataset.read_table(root / store.STAGED, "daily_bars")

    def fetch(symbol, start, unused_root, *, through):
        raw, receipt = fake_market(symbol, start, root)
        frame = pd.read_csv(io.BytesIO(raw))
        boundary = before.loc[before.symbol == symbol].iloc[-1]
        for col in FIELDS:
            frame.loc[0, col] = boundary[col]
        for col in ["Open", "High", "Low", "Close", "Adj Close"]:
            frame.loc[1, col] = boundary[col] * 1.01
        frame.loc[1, "Dividends"] = 0
        frame = frame.loc[frame.Date.astype(str).str[:10] <= through]
        raw = frame.to_csv(index=False).encode()
        receipt.update(
            market_through=through,
            last_session=through,
            exclusive_end=(date.fromisoformat(through) + timedelta(days=1)).isoformat(),
            sha256=sec.digest(raw),
        )
        return raw, receipt

    return fetch


def tamper(state_value, symbol_index, edit):
    item = list(state_value["captures"].values())[symbol_index]
    frame = pd.read_csv(io.StringIO(item["wrapper_csv"]))
    edit(frame)
    raw = frame.to_csv(index=False).encode()
    item["wrapper_csv"] = raw.decode()
    item["receipt"]["sha256"] = sec.digest(raw)


def head(root):
    return dataset.manifest_digest(root / store.STAGED)


def test_collect_resumes_after_failure_and_pins_the_cutoff(state):
    value = plan(state)
    fetch = provider(state)
    targets = list(refresh.updates.operations(value["plan"])["market"])
    calls = []

    def failing(symbol, start, root, *, through):
        calls.append(symbol)
        assert through == THROUGH
        if symbol == targets[-1]:
            raise ValueError("provider unavailable")
        return fetch(symbol, start, root, through=through)

    refresh.collect(state, RUN, failing)
    assert calls == targets and refresh.inspect(state, RUN)["status"] == "NEEDS_REVIEW"
    calls.clear()

    def resumed(symbol, start, root, **scope):
        calls.append(symbol)
        return fetch(symbol, start, root, **scope)

    refresh.collect(state, RUN, resumed)
    assert calls == [targets[-1]]
    assert refresh.inspect(state, RUN)["status"] == "READY"


def test_interrupt_saves_the_checkpoint_and_rerun_resumes(state):
    value = plan(state)
    fetch = provider(state)
    targets = list(refresh.updates.operations(value["plan"])["market"])
    calls = []

    def interrupted(symbol, start, root, *, through):
        calls.append(symbol)
        if len(calls) == 4:
            raise KeyboardInterrupt
        return fetch(symbol, start, root, through=through)

    with pytest.raises(SystemExit) as stop:
        refresh.collect(state, RUN, interrupted)
    assert stop.value.code == 130
    assert len(refresh.load(state, RUN)["captures"]) == 3
    calls.clear()
    refresh.collect(state, RUN, lambda s, st, r, **k: (calls.append(s), fetch(s, st, r, **k))[1])
    assert calls == targets[3:]


def test_unready_source_defers_the_panel_and_keeps_captures(state):
    initial = plan(state)
    fetch = provider(state)
    calls = []

    def incomplete(symbol, start, root, *, through):
        calls.append(symbol)
        raw, receipt = fetch(symbol, start, root, through=through)
        frame = pd.read_csv(io.BytesIO(raw))
        frame.loc[frame.index[-1], ["Close", "Adj Close"]] = float("nan")
        raw = frame.to_csv(index=False).encode()
        receipt["sha256"] = sec.digest(raw)
        return raw, receipt

    deferred = refresh.collect(state, RUN, incomplete)
    assert deferred["readiness"]["status"] == "WAITING_FOR_SOURCE"
    assert len(calls) == 3 and deferred["captures"] == initial["captures"]


def test_retry_refetches_only_the_named_series(state):
    plan(state)
    fetch = provider(state)
    captured = refresh.collect(state, RUN, fetch)
    first = next(iter(captured["captures"]))
    symbol = captured["captures"][first]["symbol"]
    untouched = deepcopy({k: v for k, v in captured["captures"].items() if k != first})
    tamper(captured, 0, lambda f: f.loc.__setitem__((f.index[-1], ["Close", "Adj Close"]), float("nan")))
    refresh.save(state, RUN, captured)
    assert refresh.inspect(state, RUN)["status"] == "NEEDS_REVIEW"
    calls = []
    resumed = refresh.collect(
        state, RUN, lambda s, st, r, **k: (calls.append(s), fetch(s, st, r, **k))[1], retry=[symbol]
    )
    assert calls == [symbol]
    assert {k: resumed["captures"][k] for k in untouched} == untouched
    assert refresh.inspect(state, RUN)["status"] == "READY"


def test_provider_revisions_need_review_and_block_the_build(state):
    plan(state)
    result = refresh.collect(state, RUN, provider(state))
    tamper(result, 0, lambda f: f.loc.__setitem__((0, "Close"), f.loc[0, "Close"] + 0.001))
    tamper(result, 1, lambda f: f.loc.__setitem__((0, "Dividends"), f.loc[0, "Dividends"] + 0.001))
    tamper(result, 2, lambda f: f.loc.__setitem__((1, "Stock Splits"), 2.0))
    refresh.save(state, RUN, result)
    report = refresh.inspect(state, RUN)
    assert len([c for c in report["cases"] if c["status"] == "NEEDS_REVIEW"]) == 3
    split = next(c for c in report["cases"] if c["symbol"] == list(result["captures"].values())[2]["symbol"])
    assert split["boundary_date"] == "1997-01-03"
    assert split["changes"]["new_split_events"] == [{"session_date": "1997-01-06", "Stock Splits": 2.0}]
    with pytest.raises(ValueError, match="Unresolved"):
        refresh.build(state, RUN)


def test_run_promotes_with_lineage_receipt_and_cleanup(state, capsys):
    before = head(state)
    assert refresh.run(state, RUN, through=THROUGH, fetch=provider(state)) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    manifest = json.loads((state / store.STAGED / "manifest.json").read_bytes())
    assert manifest["schema"] == dataset.SCHEMA and manifest["lineage"][-1] == before
    assert store.validation_is_current(manifest)
    dataset.verify_files(state / store.STAGED, manifest)
    receipt = json.loads((state / refresh.RECEIPTS / (RUN + ".json")).read_bytes())
    assert receipt["promoted_manifest_sha256"] == head(state) == summary["promoted"]
    assert receipt["market_rows_added"] > 0
    assert not (state / refresh.RUNS / RUN).exists()
    # The promoted dataset is a valid baseline for the next plan, and a rerun is a no-op.
    assert refresh.updates.snapshot(state, refresh.updates.request(manifest["companies"]))["base"]["manifest_sha256"] == head(state)
    assert refresh.run(state, RUN, through=THROUGH, fetch=lambda *a, **k: pytest.fail("provider")) == 0


def full_history(root, revise=None):
    """A fake provider for full-history refetches: the retained sessions plus the new ones."""
    before = dataset.read_table(root / store.STAGED, "daily_bars")
    append = provider(root)

    def fetch(symbol, start, unused_root, *, through):
        old = before.loc[before.symbol == symbol]
        raw, receipt = append(symbol, old.session_date.max(), unused_root, through=through)
        tail = pd.read_csv(io.BytesIO(raw)).iloc[1:]
        head_ = old[["session_date", *FIELDS]].rename(columns={"session_date": "Date"})
        frame = pd.concat([head_, tail], ignore_index=True)
        frame["Date"] = frame["Date"].astype(str).str[:10]
        if revise:
            revise(symbol, frame)
        raw = frame.to_csv(index=False).encode()
        receipt.update(requested_start=start, sha256=sec.digest(raw))
        return raw, receipt

    return fetch


def test_a_review_stops_and_prints_the_rerun_then_replacement_needs_yes(state, capsys):
    fetch = provider(state)

    def revised(sym, start, root, **scope):
        raw, receipt = fetch(sym, start, root, **scope)
        if sym == "AAPL":
            frame = pd.read_csv(io.BytesIO(raw))
            frame.loc[0, "Close"] += 1.0
            raw = frame.to_csv(index=False).encode()
            receipt["sha256"] = sec.digest(raw)
        return raw, receipt

    before = head(state)
    assert refresh.run(state, RUN, through=THROUGH, fetch=revised) == refresh.STOPPED
    out = capsys.readouterr().out
    assert "--replace AAPL" in out and f"--run-id {RUN}" in out and head(state) == before

    def revise_history(symbol, frame):
        if symbol == "AAPL":
            frame.loc[0, "Close"] += 1.0

    full = full_history(state, revise_history)

    def mixed(sym, start, root, **scope):
        return (full if sym == "AAPL" else fetch)(sym, start, root, **scope)

    assert refresh.run(state, RUN, through=THROUGH, replace_symbols=["AAPL"], fetch=mixed) == refresh.STOPPED
    out = capsys.readouterr().out
    assert "(replacement)" in out and "--yes" in out and head(state) == before
    assert refresh.run(state, RUN, through=THROUGH, replace_symbols=["AAPL"], yes=True, fetch=mixed) == 0
    assert head(state) != before


def test_nothing_to_update_makes_no_provider_call_and_no_change(state, capsys):
    before = head(state)
    status = refresh.run(
        state, "quiet", through="1997-01-03", fetch=lambda *a, **k: pytest.fail("provider")
    )
    assert status == 0 and head(state) == before
    assert "Nothing to update" in capsys.readouterr().out
    assert not (state / refresh.RUNS / "quiet").exists()


def test_replacement_keeps_baseline_and_other_captures(state):
    plan(state)
    old = deepcopy(refresh.collect(state, RUN, provider(state)))
    refresh.replace(state, RUN, ["AAPL"])
    new = refresh.load(state, RUN)
    assert new["plan"]["base"] == old["plan"]["base"] and new["captures"] == old["captures"]
    assert refresh.updates.operations(new["plan"])["market"]["AAPL"]["mode"] == "full"
    with pytest.raises(ValueError, match="outside this run"):
        refresh.replace(state, RUN, ["NOTASYMBOL"])


def test_a_changed_capture_cannot_be_built(state):
    plan(state)
    refresh.collect(state, RUN, provider(state))
    refresh.inspect(state, RUN)
    value = refresh.load(state, RUN)
    next(iter(value["captures"].values()))["wrapper_csv"] += "\n"
    refresh.save(state, RUN, value)
    with pytest.raises(ValueError, match="Unresolved"):
        refresh.build(state, RUN)


def test_a_changed_baseline_stops_before_any_provider_request(state):
    plan(state)
    manifest_path = state / store.STAGED / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["note"] = "changed"
    store.replace_generated(manifest_path, sec.encoded(manifest))
    with pytest.raises(ValueError, match="changed since this run was planned"):
        refresh.collect(state, RUN, lambda *a, **k: pytest.fail("provider"))


def test_a_blocked_audit_keeps_the_old_dataset(state, monkeypatch):
    plan(state)
    refresh.collect(state, RUN, provider(state))
    refresh.inspect(state, RUN)
    candidate = refresh.build(state, RUN)
    before = head(state)
    manifest = json.loads((candidate / "manifest.json").read_bytes())
    manifest["market_quality"]["status"] = "BLOCKED"
    store.replace_generated(candidate / "manifest.json", sec.encoded(manifest))
    monkeypatch.setattr(store, "validation_is_current", lambda m: True)
    with pytest.raises(ValueError, match="MARKET_DATA_BLOCKED"):
        refresh.promote(state, RUN, candidate)
    assert head(state) == before and (state / refresh.RUNS / RUN).exists()


def test_a_failed_swap_leaves_the_previous_dataset_and_run_intact(state, monkeypatch):
    plan(state)
    refresh.collect(state, RUN, provider(state))
    refresh.inspect(state, RUN)
    candidate = refresh.build(state, RUN)
    before = head(state)

    def broken(new, staged):
        raise RuntimeError("disk full")

    monkeypatch.setattr(dataset, "swap_in", broken)
    with pytest.raises(RuntimeError, match="disk full"):
        refresh.promote(state, RUN, candidate)
    assert head(state) == before and (state / refresh.RUNS / RUN / "stage.json.bz2").exists()
    monkeypatch.undo()
    assert refresh.run(state, RUN, through=THROUGH, fetch=lambda *a, **k: pytest.fail("provider")) == 0
    assert head(state) != before


def test_names_accepts_commas_and_spaces_and_rejects_duplicates():
    assert refresh.names(["AAPL, MSFT", "NVDA"]) == ["AAPL", "MSFT", "NVDA"]
    assert refresh.names(None) == []
    with pytest.raises(ValueError, match="Duplicate"):
        refresh.names(["AAPL,AAPL"])


def test_run_ids_are_path_safe(state):
    with pytest.raises(ValueError, match="Invalid"):
        refresh.run_dir(state, "../escape")
