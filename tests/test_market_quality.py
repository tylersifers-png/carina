from copy import deepcopy
import json
import pandas as pd
import pytest

from carina import market_quality as market, settings


@pytest.fixture
def series():
    dates = ["1997-01-02", "1997-01-03", "1997-01-06"]
    rows = []
    sources = {}
    for s in ["AAPL", "^GSPC", "^SP500TR"]:
        for day, close in zip(dates, [100.0, 101.0, 102.0]):
            rows.append(
                dict(
                    symbol=s,
                    session_date=day,
                    Open=close,
                    High=close + 1,
                    Low=close - 1,
                    Close=close,
                    **{
                        "Adj Close": close,
                        "Volume": 100.0,
                        "Dividends": 0.0,
                        "Stock Splits": 0.0,
                    }
                )
            )
        sources[s] = dict(
            receipt=dict(
                last_session=dates[-1], metadata=dict(symbol=s, currency="USD")
            )
        )
    frame = pd.DataFrame(rows)
    frame["price_return"] = frame.groupby("symbol").Close.pct_change(fill_method=None)
    frame["vendor_adjusted_return"] = frame.groupby("symbol")["Adj Close"].pct_change(
        fill_method=None
    )
    return frame, sources


def acks(*decisions):
    return dict(schema="company-market-acknowledgements-1.0", decisions=list(decisions))


def decide(flag, disposition="validated", **extra):
    return dict(
        flag_id=flag["id"],
        evidence_sha256=flag["evidence_sha256"],
        disposition=disposition,
        rationale="Independent fixture event reviewed",
        evidence_refs=["fixture://event"],
        **extra,
    )


def acknowledge_all(report):
    unique = {f["id"]: f for f in report["flags"]}
    return acks(*[decide(f) for f in unique.values()])


def assess(series, records=None):
    frame, sources = series
    return market.evaluate(frame, ["AAPL"], sources, acknowledged=records or acks())


@pytest.mark.parametrize("failure", ["price", "gap", "source", "return", "identity"])
def test_market_integrity_cannot_be_waived(series, failure):
    frame, sources = deepcopy(series)
    target = frame.symbol.eq("AAPL") & frame.session_date.eq("1997-01-03")
    if failure == "identity":
        frame["security_id"] = "yahoo:" + frame.symbol
        frame.loc[target, "security_id"] = "yahoo:WRONG"
    if failure == "price":
        frame.loc[target, "Close"] = -1
    if failure == "gap":
        frame = frame[~target]
    if failure == "source":
        sources["AAPL"]["receipt"]["metadata"]["symbol"] = "WRONG"
    if failure == "return":
        frame.loc[target, "price_return"] = 0.5
    damaged = assess((frame, sources))
    assert damaged["status"] == "BLOCKED"
    waived = assess((frame, sources), acknowledge_all(damaged))
    assert waived["status"] == "BLOCKED"


def test_large_move_review_is_evidence_bound(series):
    frame, sources = series
    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        frame.loc[
            frame.symbol.eq("AAPL") & frame.session_date.eq("1997-01-06"), col
        ] *= 2
    frame["price_return"] = frame.groupby("symbol").Close.pct_change(fill_method=None)
    frame["vendor_adjusted_return"] = frame.groupby("symbol")["Adj Close"].pct_change(
        fill_method=None
    )
    report = assess((frame, sources))
    flag = next(f for f in report["flags"] if f["rule"] == "LARGE_DAILY_RETURN")
    assert report["status"] == "FLAGGED"
    records = acks(decide(flag))
    assert assess((frame, sources), records)["status"] == "CLEAN"
    records["decisions"][0]["disposition"] = "excluded"
    excluded = assess((frame, sources), records)
    assert len(excluded["excluded_scopes"]) == 1
    assert excluded["excluded_scopes"][0]["evidence"] == flag["evidence"]
    records["decisions"][0]["evidence_sha256"] = "changed"
    assert assess((frame, sources), records)["open_flags"] == 1


@pytest.mark.parametrize("dividend", [10.0, 30.0])
def test_combined_action_review_catches_subthreshold_and_large_returns(
    series, dividend
):
    frame, sources = series
    target = frame.symbol.eq("AAPL") & frame.session_date.eq("1997-01-06")
    frame.loc[target, "Dividends"] = dividend
    frame.loc[target, "Stock Splits"] = 1.2
    # Vendor arithmetic reconciles, including a sub-35% apparent return. It does
    # not establish that a distribution and split describe different economics.
    frame.loc[target, "Adj Close"] = 102 / (1 - dividend / 101)
    frame["vendor_adjusted_return"] = frame.groupby("symbol")["Adj Close"].pct_change(
        fill_method=None
    )
    before = frame.copy(deep=True)
    report = assess(series)
    flag = next(
        f for f in report["flags"] if f["rule"] == "SIMULTANEOUS_SPLIT_AND_DIVIDEND"
    )
    assert flag["severity"] == "REVIEW"
    assert not any(f["rule"] == "ADJUSTMENT_RECONCILIATION" for f in report["flags"])
    assert any(f["rule"] == "LARGE_DAILY_RETURN" for f in report["flags"]) == (
        dividend == 30
    )
    pd.testing.assert_frame_equal(frame, before)
    decision = decide(
        flag,
        "excluded",
        scope=dict(
            session_dates=["1997-01-06"],
            excluded_fields=["vendor_adjusted_return"],
            excluded_uses=["return_paths_including_session"],
        ),
    )
    accepted = assess(series, acks(decision))
    assert any(
        x["rule"] == "SIMULTANEOUS_SPLIT_AND_DIVIDEND"
        for x in accepted["excluded_scopes"]
    )
    scope = next(
        x["scope"]
        for x in accepted["excluded_scopes"]
        if x["rule"] == "SIMULTANEOUS_SPLIT_AND_DIVIDEND"
    )
    assert scope == decision["scope"]
    # A changed action must not inherit an old event decision merely because
    # the date and rule identity remain the same.
    frame.loc[target, "Stock Splits"] = 1.4
    changed = assess(series, acks(decision))
    assert any(
        f["rule"] == "SIMULTANEOUS_SPLIT_AND_DIVIDEND" and f["status"] == "OPEN"
        for f in changed["flags"]
    )


@pytest.mark.parametrize("split,dividend", [(0.0, 10.0), (1.2, 0.0)])
def test_single_corporate_action_does_not_trigger_combination_review(
    series, split, dividend
):
    frame, _ = series
    target = frame.symbol.eq("AAPL") & frame.session_date.eq("1997-01-06")
    frame.loc[target, "Dividends"] = dividend
    frame.loc[target, "Stock Splits"] = split
    frame.loc[target, "Adj Close"] = 102 / (1 - dividend / 101)
    frame["vendor_adjusted_return"] = frame.groupby("symbol")["Adj Close"].pct_change(
        fill_method=None
    )
    assert not any(
        f["rule"] == "SIMULTANEOUS_SPLIT_AND_DIVIDEND" for f in assess(series)["flags"]
    )


def test_documented_action_requires_review_without_vendor_actions_and_reopens(
    series, monkeypatch
):
    event = dict(
        symbol="AAPL",
        session_dates=["1997-01-06"],
        event_type="noncash_stock_distribution",
        description="Fixture issuer reports an unrepresented stock distribution.",
        evidence_refs=["fixture://issuer"],
    )
    monkeypatch.setattr(
        market, "documented_events", lambda s: [event] if s == "AAPL" else []
    )
    report = assess(series)
    flag = next(
        f for f in report["flags"] if f["rule"] == "DOCUMENTED_CORPORATE_ACTION"
    )
    assert flag["status"] == "OPEN" and flag["severity"] == "REVIEW"
    assert flag["evidence"]["observations"][0]["Dividends"] == 0
    assert flag["evidence"]["observations"][0]["Stock Splits"] == 0
    records = acks(
        decide(
            flag,
            "excluded",
            scope=dict(
                session_dates=["1997-01-06"],
                excluded_fields=["vendor_adjusted_return"],
                excluded_uses=["return_window_including_session"],
            ),
        )
    )
    accepted = assess(series, records)
    assert accepted["open_flags"] == 0 and accepted["clean_unique_companies"] == 1
    assert accepted["excluded_scopes"][0]["scope"]["session_dates"] == ["1997-01-06"]
    frame, sources = series
    changed = frame.copy()
    changed.loc[
        changed.symbol.eq("AAPL") & changed.session_date.eq("1997-01-03"), "Volume"
    ] += 1
    reopened = assess((changed, sources), records)
    new_flag = next(f for f in reopened["flags"] if f["rule"] == flag["rule"])
    assert (
        new_flag["status"] == "OPEN"
        and new_flag["evidence_sha256"] != flag["evidence_sha256"]
    )


def test_documented_event_outside_window_ignored_but_missing_session_blocks(
    series, monkeypatch
):
    event = dict(
        symbol="AAPL",
        session_dates=["1998-01-02"],
        event_type="distribution",
        description="Fixture event",
        evidence_refs=["fixture://issuer"],
    )
    monkeypatch.setattr(
        market, "documented_events", lambda s: [event] if s == "AAPL" else []
    )
    assert not any(
        f["rule"] == "DOCUMENTED_CORPORATE_ACTION" for f in assess(series)["flags"]
    )
    event["session_dates"] = ["1997-01-04"]
    report = assess(series)
    flag = next(
        f for f in report["flags"] if f["rule"] == "DOCUMENTED_CORPORATE_ACTION"
    )
    assert flag["severity"] == "BLOCK" and flag["evidence"]["missing_sessions"] == [
        "1997-01-04"
    ]
    assert report["status"] == "BLOCKED"


def test_documented_event_registry_requires_source_evidence(tmp_path, monkeypatch):
    config = tmp_path / "decisions"
    config.mkdir(parents=True)
    event = dict(
        symbol="AAPL",
        session_dates=["1997-01-06"],
        event_type="distribution",
        description="Fixture event",
        evidence_refs=[],
    )
    path = config / "company_market_events.json"
    path.write_text(
        json.dumps(dict(schema="company-market-events-1.0", events=[event]))
    )
    settings.configure(decisions=config)
    with pytest.raises(ValueError, match="documented market event"):
        market.documented_events("AAPL")
    event["evidence_refs"] = ["fixture://issuer"]
    path.write_text(
        json.dumps(dict(schema="company-market-events-1.0", events=[event]))
    )
    assert market.documented_events("AAPL") == [event]


def test_clean_history_is_clean_and_counted(series):
    report = assess(series)
    assert report["status"] == "CLEAN" and report["open_flags"] == 0
    assert report["clean_companies"] == ["AAPL"]
    assert report["clean_unique_companies"] == 1 and report["flagged_companies"] == []


def test_flagged_company_is_not_clean_until_acknowledged(series):
    frame, sources = series
    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        frame.loc[
            frame.symbol.eq("AAPL") & frame.session_date.eq("1997-01-06"), col
        ] *= 2
    frame["price_return"] = frame.groupby("symbol").Close.pct_change(fill_method=None)
    frame["vendor_adjusted_return"] = frame.groupby("symbol")["Adj Close"].pct_change(
        fill_method=None
    )
    flagged = assess((frame, sources))
    assert flagged["flagged_companies"] == ["AAPL"] and flagged["clean_companies"] == []
    resolved = assess((frame, sources), acknowledge_all(flagged))
    assert resolved["clean_companies"] == ["AAPL"] and resolved["flagged_companies"] == []


def test_duplicate_or_unsupported_acknowledgements_are_rejected(series):
    flag = dict(id="x", evidence_sha256="y")
    with pytest.raises(ValueError, match="Duplicate"):
        assess(series, acks(decide(flag), decide(flag)))
    with pytest.raises(ValueError, match="schema"):
        assess(series, dict(schema="old", decisions=[]))


def test_only_blocking_market_flags_stop_expansion():
    plan = dict(
        request=dict(companies=["AAPL", "MSFT"], add_companies=True),
        base=dict(companies=["AAPL"], market_quality_gate=dict(status="FLAGGED")),
    )
    market.require_expansion_ready(plan)
    plan["base"]["market_quality_gate"]["status"] = "BLOCKED"
    with pytest.raises(ValueError, match="EXPANSION_BLOCKED"):
        market.require_expansion_ready(plan)
    plan["request"]["add_companies"] = False
    market.require_expansion_ready(plan)

