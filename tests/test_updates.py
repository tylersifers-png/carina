from copy import deepcopy
import io
import json
import shutil

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from carina import store, updates as update, common as sec
from tests.registry_fixture import (
    INITIAL_COMPANIES,
    INITIAL_ROWS,
    synthetic_rows,
)

COMPANIES = INITIAL_COMPANIES
from tests.test_store import built, envelope, fake_market


@pytest.fixture
def state(built, monkeypatch):
    monkeypatch.setattr(update, "check_expansion", lambda plan: None)
    root, _, output = built
    current = root / store.STAGED
    shutil.copytree(output, current)
    manifest = json.loads((current / "manifest.json").read_bytes())
    manifest.update(status="VERIFIED", verified_at=sec.now())
    store.replace_generated(current / "manifest.json", sec.encoded(manifest))
    return root


def subset(root, remove):
    current = root / store.STAGED
    for name in update.TABLES:
        frame = pd.read_parquet(current / (name + ".parquet"))
        if "symbol" in frame:
            frame = frame.loc[frame.symbol != remove]
        frame.to_parquet(current / (name + ".parquet"), index=False)
    coverage = json.loads((current / "coverage.json").read_bytes())
    coverage.pop(remove)
    store.replace_generated(current / "coverage.json", sec.encoded(coverage))
    manifest = json.loads((current / "manifest.json").read_bytes())
    manifest["companies"].remove(remove)
    for name in store.FILES:
        raw = (current / name).read_bytes()
        manifest["files"][name] = dict(sha256=sec.digest(raw), bytes=len(raw))
    store.replace_generated(current / "manifest.json", sec.encoded(manifest))


def remove_benchmark(root, symbol):
    current = root / store.STAGED
    for name in update.TABLES:
        frame = pd.read_parquet(current / (name + ".parquet"))
        if "symbol" in frame:
            frame = frame.loc[frame.symbol != symbol]
        frame.to_parquet(current / (name + ".parquet"), index=False)
    coverage = json.loads((current / "coverage.json").read_bytes())
    coverage.pop(symbol)
    store.replace_generated(current / "coverage.json", sec.encoded(coverage))
    sources = json.loads((current / store.SOURCES).read_bytes())
    sources["market"].pop(symbol)
    store.replace_generated(current / store.SOURCES, sec.encoded(sources))
    manifest = json.loads((current / "manifest.json").read_bytes())
    manifest["benchmarks"].remove(symbol)
    for name in store.FILES:
        raw = (current / name).read_bytes()
        manifest["files"][name] = dict(sha256=sec.digest(raw), bytes=len(raw))
    store.replace_generated(current / "manifest.json", sec.encoded(manifest))


def capture(root, req, *, plan=None, items=None):
    plan = plan or update.snapshot(root, req)
    targets = update.operations(plan)
    companies = (
        envelope(targets["sec_companies"])["companies"] if items is None else items
    )
    content = dict(
        schema=update.SCHEMA, plan=plan, collected_at=sec.now(), companies=companies
    )
    source = root / ("capture_" + sec.digest(sec.encoded(content))[:12])
    source.mkdir()
    sec.prepare_capture(
        source, content, validator=update.validate_capture, **store.BOUNDS
    )
    bundle = json.loads((source / "handoff.json").read_bytes())["bundle"]
    destination = store.run_path(root, bundle)
    destination.mkdir(parents=True)
    for part in source.glob("part_*.json"):
        shutil.copyfile(part, destination / part.name)
    return bundle, content


def extended_market(symbol, start, root):
    raw, receipt = fake_market(symbol, start, root)
    frame = pd.read_csv(io.BytesIO(raw))
    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        frame[col] += 1
    raw = frame.to_csv(index=False).encode()
    receipt["sha256"] = sec.digest(raw)
    return raw, receipt


def test_default_expansion_noop_has_zero_provider_calls_and_zero_writes(
    state, monkeypatch, capsys
):
    before = {p.name: p.read_bytes() for p in (state / store.STAGED).iterdir()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Provider must not be called")

    monkeypatch.setattr(store, "collect_company", forbidden)
    update.collect(state, list(COMPANIES), "default_noop")
    assert '"status": "NO_CHANGES"' in capsys.readouterr().out
    assert before == {p.name: p.read_bytes() for p in (state / store.STAGED).iterdir()}


@pytest.mark.parametrize(
    "add_companies,add_periods",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_independent_switch_matrix(state, add_companies, add_periods):
    subset(state, "MSFT")
    plan = update.snapshot(
        state, update.request(list(COMPANIES), add_companies, add_periods)
    )
    tasks = update.operations(plan)
    assert ("MSFT" in tasks["sec_companies"]) == add_companies
    assert ("MSFT" in tasks["market"]) == add_companies
    assert ("AAPL" in tasks["sec_companies"]) == add_periods
    assert ("AAPL" in tasks["market"]) == add_periods
    assert ("^GSPC" in tasks["market"]) == add_periods
    assert ("^SP500TR" in tasks["market"]) == add_periods


def test_period_update_can_preserve_existing_financials(state):
    req = update.request(
        ["AAPL"],
        add_companies=False,
        add_periods=True,
        refresh_financials=False,
    )
    tasks = update.operations(update.snapshot(state, req))
    assert tasks["sec_companies"] == []
    assert set(tasks["market"]) == {"AAPL", *store.BENCHMARKS}


def test_expansion_only_fetches_missing_company_preserves_all_existing_rows_and_sources(
    state,
):
    subset(state, "MSFT")
    current = state / store.STAGED
    old = {n: pd.read_parquet(current / (n + ".parquet")) for n in update.TABLES}
    original_sources = (current / store.SOURCES).read_bytes()
    bundle, _ = capture(state, update.request(["MSFT"]))
    calls = []

    def fetch(symbol, start, root):
        calls.append(symbol)
        return fake_market(symbol, start, root)

    output = store.build(state, bundle, fetch)
    assert calls == ["MSFT"]
    for n, frame in old.items():
        new = pd.read_parquet(output / (n + ".parquet"))
        if "symbol" in frame:
            new = new.loc[new.symbol != "MSFT"].reset_index(drop=True)
        assert_frame_equal(frame.reset_index(drop=True), new)
    old_sources = json.loads(original_sources)
    new_sources = json.loads((output / store.SOURCES).read_bytes())
    for symbol in store.BENCHMARKS:
        assert old_sources["market"][symbol] == new_sources["market"][symbol]


def test_period_updates_request_boundary_only_and_keep_all_old_daily_rows(state):
    current = state / store.STAGED
    old = pd.read_parquet(current / "daily_bars.parquet")
    req = update.request(["AAPL"], False, True)
    plan = update.snapshot(state, req, planned_at="1997-01-07T00:00:00+00:00")
    bundle, _ = capture(state, req, plan=plan)
    calls = []

    def fetch(symbol, start, root):
        calls.append((symbol, start))
        return extended_market(symbol, start, root)

    output = store.build(state, bundle, fetch)
    assert set(calls) == {
        ("AAPL", "1997-01-03"),
        ("^GSPC", "1997-01-03"),
        ("^SP500TR", "1997-01-03"),
    }
    new = pd.read_parquet(output / "daily_bars.parquet")
    for symbol in old.symbol.unique():
        original = old.loc[old.symbol == symbol].reset_index(drop=True)
        retained = new.loc[
            (new.symbol == symbol) & new.session_date.isin(original.session_date)
        ].reset_index(drop=True)
        assert_frame_equal(original, retained)
    added = new.loc[(new.symbol == "AAPL") & (new.session_date == "1997-01-06")].iloc[0]
    assert added.price_return == pytest.approx(13 / 12 - 1)
    assert added.vendor_adjusted_return == pytest.approx(12.5 / 11.5 - 1)
    assert len(new) == len(old) + 3


def test_no_new_completed_sessions_skips_every_yahoo_request(state):
    req = update.request(["AAPL"], False, True)
    plan = update.snapshot(state, req, planned_at="1997-01-04T00:00:00+00:00")
    bundle, _ = capture(state, req, plan=plan)
    output = store.build(
        state, bundle, lambda *args: pytest.fail("No new market session")
    )
    assert output is not None  # SEC was checked; market rows are unchanged.
    assert_frame_equal(
        pd.read_parquet(state / store.STAGED / "daily_bars.parquet"),
        pd.read_parquet(output / "daily_bars.parquet"),
    )


def test_both_switches_off_build_is_read_only(state):
    req = update.request(list(COMPANIES), False, False)
    bundle, _ = capture(state, req)
    assert (
        store.build(state, bundle, lambda *a: pytest.fail("No provider calls")) is None
    )
    marker = json.loads(
        (store.run_path(state, bundle) / "update_result.json").read_bytes()
    )
    assert marker["status"] == "NO_CHANGES"


@pytest.mark.parametrize("changed", ["adjustment", "split", "close"])
def test_append_rejects_incompatible_adjustments_or_split(state, changed):
    old = pd.read_parquet(state / store.STAGED / "daily_bars.parquet")
    old = old.loc[old.symbol == "AAPL"]
    raw, receipt = extended_market("AAPL", "1997-01-03", state)
    frame = pd.read_csv(io.BytesIO(raw))
    if changed == "adjustment":
        frame["Adj Close"] *= 0.99
    if changed == "close":
        frame["Close"] += 0.1
    if changed == "split":
        frame.loc[1, "Stock Splits"] = 2.0
    raw = frame.to_csv(index=False).encode()
    receipt["sha256"] = sec.digest(raw)
    with pytest.raises(ValueError, match="full refresh"):
        update.append_market(old, raw, receipt, "1997-01-03")


def test_append_audits_nonbasis_boundary_revision_and_retains_stored_row(state):
    old = pd.read_parquet(state / store.STAGED / "daily_bars.parquet")
    old = old.loc[old.symbol == "AAPL"]
    raw, receipt = extended_market("AAPL", "1997-01-03", state)
    frame = pd.read_csv(io.BytesIO(raw))
    frame.loc[0, "Open"] += 0.04
    frame.loc[0, "Low"] -= 0.002
    raw = frame.to_csv(index=False).encode()
    receipt["sha256"] = sec.digest(raw)

    result = update.append_market(old, raw, receipt, "1997-01-03")

    assert len(result) == len(old) + 1
    assert_frame_equal(
        result.loc[result.session_date == "1997-01-03"].reset_index(drop=True),
        old.loc[old.session_date == "1997-01-03"].reset_index(drop=True),
    )
    revision = receipt["retained_boundary_revision"]
    assert revision["session_date"] == "1997-01-03"
    assert revision["disposition"] == "STORED_ROW_RETAINED"
    assert set(revision["fields"]) == {"Open", "Low", "Volume", "Dividends"}


def test_explicit_full_refresh_is_separate_from_both_toggles(state):
    req = update.request(["AAPL"], False, False, True)
    tasks = update.operations(update.snapshot(state, req))
    assert tasks["sec_companies"] == ["AAPL"]
    assert set(tasks["market"]) == {"AAPL", *store.BENCHMARKS}
    assert tasks["market"] == {
        "AAPL": dict(mode="full", start="1997-01-01"),
        **{
            symbol: dict(mode="full", start=store.BENCHMARK_STARTS[symbol])
            for symbol in store.BENCHMARKS
        },
    }


def test_changed_baseline_stops_before_any_provider_request(state):
    req = update.request(["AAPL"], False, True)
    bundle, _ = capture(state, req)
    manifest = state / store.STAGED / "manifest.json"
    m = json.loads(manifest.read_bytes())
    m["verified_at"] = "different"
    store.replace_generated(manifest, sec.encoded(m))
    with pytest.raises(ValueError, match="changed since planning"):
        store.build(
            state, bundle, lambda *a: pytest.fail("Provider must not be called")
        )


def test_new_filings_append_versions_and_leave_existing_facts_intact(state):
    req = update.request(["AAPL"], False, True)
    plan = update.snapshot(state, req, planned_at="1997-01-04T00:00:00+00:00")
    items = envelope(["AAPL"])["companies"]
    item = items["AAPL"]
    points = item["facts"]["facts"]["us-gaap"]["Assets"]["units"]["USD"]
    points.append(
        dict(points[0], val=1100, filed="2026-02-14", accn="0000320193-26-000002")
    )
    item["projection_sha256"] = sec.digest(sec.encoded(item["facts"]))
    bundle, _ = capture(state, req, plan=plan, items=items)
    output = store.build(state, bundle, lambda *a: pytest.fail("No new daily session"))
    old = pd.read_parquet(state / store.STAGED / "fundamentals.parquet")
    new = pd.read_parquet(output / "fundamentals.parquet")
    assert len(new) == len(old) + 1
    retained = (
        new.loc[new.accession != "0000320193-26-000002"]
        .sort_values("symbol")
        .reset_index(drop=True)
    )
    assert_frame_equal(old.sort_values("symbol").reset_index(drop=True), retained)
    latest = pd.read_parquet(output / "financial_latest.parquet")
    assert latest.loc[latest.symbol == "AAPL", "value"].tolist() == ["1100"]


def test_existing_sec_revision_is_not_silently_overwritten(state):
    old = pd.read_parquet(state / store.STAGED / "fundamentals.parquet")
    rows = old.loc[old.symbol == "AAPL"].to_dict("records")
    incoming = deepcopy(rows)
    incoming[0]["value"] = "999"
    with pytest.raises(ValueError, match="full refresh"):
        update.merge_versions(rows, incoming, "2026-02-13")


def test_missing_company_is_not_added_when_disabled(state):
    subset(state, "MSFT")
    req = update.request(["MSFT"], False, False)
    assert update.operations(update.snapshot(state, req)) == dict(
        sec_companies=[], new_companies=[], market={}
    )


def test_explicit_benchmark_migration_fetches_only_missing_rsp_and_preserves_existing_rows(
    state,
):
    remove_benchmark(state, "RSP")
    before = pd.read_parquet(state / store.STAGED / "daily_bars.parquet")
    untouched = {
        name: (state / store.STAGED / (name + ".parquet")).read_bytes()
        for name in update.TABLES
        if name not in {"daily_bars", "securities"}
    }
    req = update.request(["AAPL"], add_companies=False, add_benchmarks=True)
    plan = update.snapshot(state, req)
    assert update.operations(plan) == dict(
        sec_companies=[],
        new_companies=[],
        market={"RSP": {"mode": "full", "start": "2003-05-01"}},
    )
    bundle, _ = capture(state, req, plan=plan)
    calls = []

    def fetch(symbol, start, root):
        calls.append((symbol, start))
        return fake_market(symbol, start, root)

    output = store.build(state, bundle, fetch)
    assert calls == [("RSP", "2003-05-01")]
    after = pd.read_parquet(output / "daily_bars.parquet")
    assert_frame_equal(
        before.sort_values(["symbol", "session_date"]).reset_index(drop=True),
        after.loc[after.symbol != "RSP"]
        .sort_values(["symbol", "session_date"])
        .reset_index(drop=True),
    )
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert set(manifest["benchmarks"]) == set(store.BENCHMARKS)
    assert {
        "baseline_guard_seconds",
        "working_tables_seconds",
        "selected_processing_seconds",
        "table_merge_seconds",
        "file_hashing_seconds",
        "structural_validation_seconds",
        "financial_quality_seconds",
        "market_quality_seconds",
        "total_seconds",
        "candidate_bytes",
    } == set(manifest["timing"])
    store.verify_local(output, manifest, require_all_benchmarks=True)
    for name, raw in untouched.items():
        assert (output / (name + ".parquet")).read_bytes() == raw


def test_working_tables_never_expand_whole_store_to_records(state, monkeypatch):
    output = state / "bounded_working_tables"
    output.mkdir()

    def forbidden(*args, **kwargs):
        pytest.fail("Whole-table dictionary expansion is forbidden")

    monkeypatch.setattr(pd.DataFrame, "to_dict", forbidden)
    tables = update.UpdateTables(output, state / store.STAGED)
    original = tables["daily_bars"]
    incoming = tables.selected("daily_bars", "AAPL").copy()
    incoming["Close"] += 1
    tables.replace("daily_bars", "AAPL", incoming)
    tables.flush()
    result = tables["daily_bars"]
    assert_frame_equal(
        result.loc[result.symbol != "AAPL"].reset_index(drop=True),
        original.loc[original.symbol != "AAPL"].reset_index(drop=True),
    )
    assert_frame_equal(
        result.loc[result.symbol == "AAPL"].reset_index(drop=True),
        incoming.reset_index(drop=True),
    )


def test_bootstrap_requires_add_companies_and_creates_benchmarks_once(tmp_path):
    plan = update.snapshot(tmp_path, update.request(["AAPL"]))
    assert set(update.operations(plan)["market"]) == {"AAPL", *store.BENCHMARKS}
    req = update.request(["AAPL"], False, True)
    assert update.operations(update.snapshot(tmp_path, req))["market"] == {}


def test_controls_cannot_be_changed_after_capture(state):
    bundle, content = capture(state, update.request(["AAPL"], False, True))
    content["plan"]["request"]["add_periods"] = False
    with pytest.raises(ValueError, match="controls"):
        update.validate_capture(content)


def test_collector_fetches_sec_only_for_missing_company(state, monkeypatch):
    subset(state, "MSFT")
    calls = []

    def fetch(path, symbol):
        calls.append(symbol)
        return envelope([symbol])["companies"][symbol]

    monkeypatch.setattr(store, "collect_company", fetch)
    path = update.collect(state, list(COMPANIES), "only_missing")
    assert calls == ["MSFT"]
    payload = json.loads((path / "capture.json").read_bytes())
    assert set(payload["companies"]) == {"MSFT"}
    assert "^GSPC" not in update.operations(payload["plan"])["market"]


def test_cli_defaults_and_both_disabled_controls(monkeypatch, tmp_path):
    import sys

    from carina import settings

    settings.configure(data=tmp_path)

    calls = []
    monkeypatch.setattr(update, "collect", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(sys, "argv", ["company_data.py", "collect", "--run-id", "test"])
    store.main()
    assert calls[-1]["add_companies"] is True
    assert calls[-1]["add_periods"] is False
    assert calls[-1]["full_refresh"] is False
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "company_data.py",
            "collect",
            "--run-id",
            "test",
            "--no-add-companies",
            "--no-add-periods",
        ],
    )
    store.main()
    assert calls[-1]["add_companies"] is False
    assert calls[-1]["add_periods"] is False


def test_second_expansion_fetches_only_new_histories_and_preserves_initial_panel(state):
    from carina import registry as live_registry

    extra = synthetic_rows("SYNA", "SYNB", "SYNC")
    live_registry.COMPANIES.use([*INITIAL_ROWS, *extra])
    registry = [*COMPANIES, *(row["symbol"] for row in extra)]

    path = state / store.STAGED
    before = pd.read_parquet(path / "daily_bars.parquet")
    bundle, content = capture(state, update.request(list(registry)))
    new = set(registry) - set(COMPANIES)
    assert set(content["companies"]) == new
    calls = []

    def fetch(symbol, start, root):
        calls.append(symbol)
        return fake_market(symbol, start, root)

    output = store.build(state, bundle, fetch)
    assert set(calls) == new and len(calls) == len(new)
    after = pd.read_parquet(output / "daily_bars.parquet")
    assert_frame_equal(
        before, after[after.symbol.isin(before.symbol.unique())].reset_index(drop=True)
    )
    assert len(pd.read_parquet(output / "companies.parquet")) == len(registry)


def test_explicit_cutoff_binds_new_history_request_and_preserves_current(state):
    subset(state, "MSFT")
    before = {p.name: p.read_bytes() for p in (state / store.STAGED).iterdir()}
    req = update.request(["AAPL", "MSFT"], market_through="1997-01-03")
    bundle, content = capture(state, req)
    assert update.operations(content["plan"])["market"] == {
        "MSFT": dict(mode="full", start="1997-01-01", through="1997-01-03")
    }
    calls = []

    def fetch(symbol, start, root, *, through):
        calls.append((symbol, start, through))
        raw, receipt = fake_market(symbol, start, root)
        receipt.update(
            requested_at=sec.now(), retrieved_at=sec.now(), market_through=through
        )
        return raw, receipt

    output = store.build(state, bundle, fetch)
    assert calls == [("MSFT", "1997-01-01", "1997-01-03")]
    assert before == {p.name: p.read_bytes() for p in (state / store.STAGED).iterdir()}
    prices = pd.read_parquet(output / "daily_bars.parquet")
    assert prices.loc[prices.symbol.eq("MSFT"), "session_date"].max() == "1997-01-03"
    sources = json.loads((output / store.SOURCES).read_bytes())
    receipt = sources["market"]["MSFT"]
    assert receipt["market_through"] == "1997-01-03"
    assert "wrapper_csv" not in json.dumps(sources)  # payloads are not kept
    raw, _ = fetch("MSFT", "1997-01-01", state, through="1997-01-03")
    # An older response is accepted only when the caller explicitly binds its scope.
    with pytest.raises(ValueError, match="Market identity/window mismatch"):
        store.validate_market(raw, receipt)
    # Explicit scope never waives an invalid close within that scope.
    frame = pd.read_csv(io.StringIO(raw.decode()))
    frame.loc[1, "Close"] = float("nan")
    raw = frame.to_csv(index=False).encode()
    receipt["sha256"] = sec.digest(raw)
    with pytest.raises(ValueError, match="invalid=1"):
        store.validate_market(raw, receipt, expected_end="1997-01-03")


@pytest.mark.parametrize(
    "controls",
    [dict(add_periods=True), dict(full_refresh=True), dict(add_companies=False)],
)
def test_explicit_cutoff_cannot_change_maintenance_scope(controls):
    with pytest.raises(ValueError, match="expansion-only"):
        update.request(["MSFT"], market_through="1997-01-03", **controls)


@pytest.mark.parametrize("day", ["2099-01-02", "1997-01-04", "1996-12-31"])
def test_explicit_cutoff_rejects_future_non_session_and_pre_listing(tmp_path, day):
    with pytest.raises(ValueError, match="Market cutoff"):
        update.operations(
            update.snapshot(tmp_path, update.request(["MSFT"], market_through=day))
        )


def test_cutoff_adapter_requests_exact_exclusive_end(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []
    raw, _ = fake_market("MSFT", "1997-01-01", tmp_path)
    frame = pd.read_csv(io.BytesIO(raw)).set_index("Date")

    class Ticker:
        def history(self, **kwargs):
            calls.append(kwargs)
            return frame

        def get_history_metadata(self):
            return dict(symbol="MSFT", currency="USD")

    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        SimpleNamespace(
            Ticker=lambda symbol: Ticker(),
            set_tz_cache_location=lambda path: None,
            __version__="fixture",
        ),
    )
    result, receipt = store.market("MSFT", "1997-01-01", tmp_path, through="1997-01-03")
    assert calls[0]["end"] == "1997-01-04"
    assert calls[0]["start"] == "1997-01-01"
    assert receipt["last_session"] == receipt["market_through"] == "1997-01-03"
    store.validate_market(result, receipt, expected_end="1997-01-03")

