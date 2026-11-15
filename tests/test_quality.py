from copy import deepcopy
import pandas as pd
import pytest

from carina import quality
from carina.registry import metrics_for


@pytest.fixture
def sample():
    rows = []
    for year in (2023, 2024, 2025):
        for metric, (unit, kind, mappings) in metrics_for("AAPL").items():
            ns, tag, basis = mappings[0]
            value = {"assets": 100, "liabilities": 60, "equity": 40}.get(metric, 10)
            rows.append(
                dict(
                    symbol="AAPL",
                    metric=metric,
                    basis=basis,
                    unit=unit,
                    value=str(value),
                    status="REPORTED",
                    start=f"{year}-01-01" if kind == "duration" else "",
                    end=f"{year}-12-31",
                    period_kind="ANNUAL" if kind == "duration" else "INSTANT",
                    form="10-K",
                    source_tags=f"{ns}:{tag}",
                )
            )
    facts = pd.DataFrame(rows)
    market = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 2,
            "session_date": ["2024-01-02", "2024-01-03"],
            "Open": [10.0, 10.0],
            "High": [11.0, 11.0],
            "Low": [9.0, 9.0],
            "Close": [10.0, 10.0],
            "Adj Close": [10.0, 10.0],
            "Stock Splits": [0.0, 0.0],
        }
    )
    return facts, market


def audit(sample, **kwargs):
    f, m = sample
    return quality.evaluate(
        f,
        f,
        m,
        ["AAPL"],
        acknowledged=kwargs.pop(
            "acknowledged",
            dict(schema="company-quality-acknowledgements-1.0", decisions=[]),
        ),
        data_digest=kwargs.pop("data_digest", "a" * 64),
        **kwargs,
    )


def approval(report, disposition="validated"):
    return dict(
        schema="company-quality-acknowledgements-1.0",
        decisions=[
            dict(
                flag_id=f["id"],
                evidence_sha256=f["evidence_sha256"],
                disposition=disposition,
                rationale="Verified synthetic fixture",
                evidence_refs=["fixture://source"],
            )
            for f in report["flags"]
        ],
    )


def flagged_sample(sample):
    f, m = sample
    return f[f.metric != "cash"], m



@pytest.mark.parametrize(
    "change,rule",
    [
        ("missing", "MAPPED_FIELD_MISSING"),
        ("stale", "MAPPED_FIELD_STALE"),
        ("annual_gap", "ANNUAL_FLOW_GAP"),
        ("scale", "SHARE_SCALE_ANOMALY"),
        ("split", "PER_SHARE_SPLIT_BASIS_UNVERIFIED"),
        ("identity", "BALANCE_IDENTITY_REVIEW"),
    ],
)
def test_financial_mapping_flags(sample, change, rule):
    f, m = sample
    f = f.copy()
    m = m.copy()
    if change == "missing":
        f = f[f.metric != "cash"]
    if change == "stale":
        f = f[~(f.metric.eq("cash") & f.end.gt("2023-12-31"))]
    if change == "annual_gap":
        f = f[~(f.metric.eq("operating_cash_flow") & f.end.eq("2024-12-31"))]
    if change == "scale":
        f.loc[f.metric.eq("shares_outstanding") & f.end.eq("2024-12-31"), "value"] = (
            "10000"
        )
    if change == "split":
        m.loc[1, "Stock Splits"] = 4.0
    if change == "identity":
        f.loc[f.metric.eq("assets"), "value"] = "1000000000"
    report = audit((f, m))
    assert report["flag_counts"][rule] >= 1
    assert report["status"] == "FLAGGED"


@pytest.mark.parametrize("value", ["bad", None, "NaN", "Infinity"])
def test_invalid_financial_values_block_and_cannot_be_waived(sample, value):
    f, m = sample
    f = f.copy()
    f.loc[0, "value"] = value
    report = audit((f, m))
    assert report["status"] == "BLOCKED"
    waived = audit((f, m), acknowledged=approval(report, "expected"))
    assert waived["status"] == "BLOCKED"


def test_units_block_but_economic_losses_do_not(sample):
    f, m = sample
    f = f.copy()
    f.loc[f.metric.eq("net_income"), "value"] = "-10"
    assert audit((f, m))["open_flags"] == 0
    f.loc[f.metric.eq("cash"), "unit"] = "shares"
    assert audit((f, m))["flag_counts"]["UNIT_MISMATCH"] == 1


def test_invalid_market_values_block(sample):
    f, m = sample
    m = m.copy()
    m.loc[0, "Close"] = float("nan")
    assert audit((f, m))["status"] == "BLOCKED"


def test_full_refresh_coverage_loss_and_new_tags_are_flagged(sample):
    f, m = sample
    new = f[~(f.metric.eq("cash") & f.end.eq("2023-12-31"))].copy()
    new.loc[new.metric.eq("revenue"), "source_tags"] = "us-gaap:NewRevenueDefinition"
    report = audit((new, m), previous=f)
    assert report["flag_counts"]["HISTORICAL_COVERAGE_LOSS"] == 1
    assert report["flag_counts"]["NEW_SOURCE_TAG"] == 1


def test_transition_flags_cannot_disappear_after_candidate_becomes_current(sample):
    f, m = sample
    new = f.copy()
    new.loc[new.metric.eq("revenue"), "source_tags"] = "us-gaap:NewRevenueDefinition"
    initial = audit((new, m), previous=f)
    later = audit((new, m), retained_flags=initial["flags"])
    assert later["flag_counts"] == {"NEW_SOURCE_TAG": 1}
    assert initial["flags"][0]["id"] == later["flags"][0]["id"]
    bad = deepcopy(initial["flags"])
    bad[0]["evidence"]["new_tags"] = ["tampered"]
    with pytest.raises(ValueError, match="transition evidence"):
        audit((new, m), retained_flags=bad)


def test_review_decision_does_not_suppress_changed_evidence(sample):
    f, m = sample
    f = f.copy()
    f.loc[f.metric.eq("shares_outstanding") & f.end.eq("2024-12-31"), "value"] = "10000"
    report = audit((f, m))
    records = approval(report, "excluded")
    accepted = audit((f, m), acknowledged=records)
    assert accepted["status"] == "CLEAN"
    f.loc[f.metric.eq("shares_outstanding") & f.end.eq("2024-12-31"), "value"] = "12000"
    changed = audit(
        (f, m), acknowledged=records, data_digest="b" * 64
    )
    assert changed["status"] == "FLAGGED"
    assert changed["open_flags"] == 2


def test_large_monetary_scale_changes_are_reviewed_not_automatically_corrected(sample):
    f, m = sample
    f = f.copy()
    f.loc[f.metric.eq("operating_cash_flow") & f.end.eq("2024-12-31"), "value"] = (
        "10000"
    )
    report = audit((f, m))
    assert report["flag_counts"]["VALUE_SCALE_REVIEW"] == 2
    assert report["status"] == "FLAGGED"
    assert (
        f.loc[
            f.metric.eq("operating_cash_flow") & f.end.eq("2024-12-31"), "value"
        ].iloc[0]
        == "10000"
    )


def test_expected_missing_dividend_field_reopens_when_cash_events_appear(sample):
    f, m = sample
    f = f[f.metric != "dividends_paid"]
    m = m.copy()
    m["Dividends"] = 0.0
    report = audit((f, m))
    records = approval(report, "expected")
    accepted = audit((f, m), acknowledged=records)
    assert accepted["status"] == "CLEAN"
    m.loc[1, "Dividends"] = 0.5
    changed = audit(
        (f, m), acknowledged=records, data_digest="b" * 64
    )
    assert changed["status"] == "FLAGGED"
    assert changed["flag_counts"] == {"MAPPED_FIELD_MISSING": 1}


def test_latest_filing_ahead_of_all_mapped_statements_is_flagged(sample):
    f, m = sample
    index = pd.DataFrame([dict(symbol="AAPL", form="10-Q", reportDate="2026-03-31")])
    report = audit(sample, filings=index)
    assert report["flag_counts"]["FILING_FACT_LAG"] == 1


def test_new_quarter_mapping_disappearance_is_not_hidden_by_stale_threshold(sample):
    f, m = sample
    f = f.copy()
    rows = []
    for end in ["2025-03-31", "2025-06-30", "2025-09-30", "2026-03-31"]:
        row = f[f.metric.eq("cash")].iloc[-1].to_dict()
        row["end"] = end
        rows.append(row)
    f = pd.concat([f, pd.DataFrame(rows)], ignore_index=True)
    # Cash reaches the latest quarter, but a previously quarterly liabilities
    # mapping stops after Dec 2025. Not every field is globally behind.
    for end in ["2025-03-31", "2025-06-30", "2025-09-30"]:
        row = f[f.metric.eq("liabilities")].iloc[-1].to_dict()
        row["end"] = end
        f = pd.concat([f, pd.DataFrame([row])], ignore_index=True)
    index = pd.DataFrame([dict(symbol="AAPL", form="10-Q", reportDate="2026-03-31")])
    report = audit((f, m), filings=index)
    assert report["flag_counts"]["FILING_PERIOD_MAPPING_GAP"] == 1
    assert "FILING_FACT_LAG" not in report["flag_counts"]


def test_rejected_source_facts_are_not_silently_lost_during_normalization(sample):
    issues = {
        "AAPL": [
            dict(metric="cash", reason="UNIT_OR_PERIOD_MISMATCH", fact_id="case-one"),
            dict(metric="cash", reason="UNSUPPORTED_FORM", fact_id="expected-form"),
        ]
    }
    report = audit(sample, normalization_issues=issues)
    assert report["flag_counts"] == {"NORMALIZATION_REJECTION": 1}
    assert report["flags"][0]["evidence"]["count"] == 1


def test_consolidated_identity_uses_total_equity_without_replacing_parent(sample):
    f, m = sample
    f.loc[f.metric.eq("assets"), "value"] = "1000000000"
    f.loc[f.metric.eq("liabilities"), "value"] = "600000000"
    f.loc[f.metric.eq("equity"), "value"] = "300000000"
    assert any(x["rule"] == "BALANCE_IDENTITY_REVIEW" for x in audit((f, m))["flags"])
    total = f[f.metric.eq("equity")].copy()
    total["basis"] = "including_noncontrolling_interest"
    total["value"] = "400000000"
    total["source_tags"] = (
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
    )
    together = pd.concat([f, total], ignore_index=True)
    result = audit((together, m))
    assert not any(x["rule"] == "BALANCE_IDENTITY_REVIEW" for x in result["flags"])
    assert (
        together[together.metric.eq("equity") & together.basis.eq("reported")]
        .value.eq("300000000")
        .all()
    )


def test_balance_review_reopens_on_underlying_observation_change(sample):
    f, m = sample
    f.loc[f.metric.eq("assets"), "value"] = "1000000000"
    first = audit((f, m))
    before = next(x for x in first["flags"] if x["rule"] == "BALANCE_IDENTITY_REVIEW")
    # Same affected periods and same largest discrepancy: a smaller restatement
    # still needs a new evidence fingerprint.
    f.loc[f.metric.eq("equity") & f.end.eq("2023-12-31"), "value"] = "41"
    after = next(
        x for x in audit((f, m))["flags"] if x["rule"] == "BALANCE_IDENTITY_REVIEW"
    )
    assert (
        before["evidence"]["max_difference_usd"]
        == after["evidence"]["max_difference_usd"]
    )
    assert before["evidence_sha256"] != after["evidence_sha256"]


def test_rejection_review_survives_response_change_but_reopens_changed_fact(sample):
    raw = dict(
        start="2023-01-01",
        end="2023-12-31",
        val=2.5,
        accn="0000320193-24-000001",
        form="10-K",
        filed="2024-02-01",
    )

    def rejected(sha, values):
        item = dict(
            receipts=dict(facts=dict(sha256=sha)),
            facts=dict(
                facts={
                    "us-gaap": {"EarningsPerShareDiluted": {"units": {"pure": values}}}
                }
            ),
        )
        index = values.index(raw)
        issue = dict(
            metric="diluted_eps",
            reason="UNIT_OR_PERIOD_MISMATCH",
            fact_id=quality.digest(
                [sha, "us-gaap", "EarningsPerShareDiluted", "pure", index]
            ),
        )
        return quality.rejection_observations("AAPL", [issue], item)

    first = rejected("a" * 64, [raw])
    second = rejected("b" * 64, [dict(raw, val=5), raw])
    assert first[0]["fact_id"] != second[0]["fact_id"]
    assert first[0]["observation_sha256"] == second[0]["observation_sha256"]
    before = audit(sample, normalization_issues={"AAPL": first})
    records = approval(before)
    after = audit(
        sample,
        normalization_issues={"AAPL": second},
        acknowledged=records,
        data_digest="b" * 64,
    )
    assert after["status"] == "CLEAN"
    raw["val"] = 2.6
    changed = rejected("c" * 64, [raw])
    result = audit(
        sample,
        normalization_issues={"AAPL": changed},
        acknowledged=records,
        data_digest="b" * 64,
    )
    assert result["status"] == "FLAGGED"
    assert any(
        f["rule"] == "NORMALIZATION_REJECTION" and f["status"] == "OPEN"
        for f in result["flags"]
    )


def test_unreconstructable_rejection_keeps_capture_identity_for_review():
    issue = dict(
        metric="diluted_eps", reason="INVALID_VALUE_OR_DATES", fact_id="a" * 64
    )
    assert quality.rejection_observations("AAPL", [issue], {}) == [issue]


def test_clean_data_is_clean_and_missing_data_is_flagged(sample):
    clean = audit(sample)
    assert clean["status"] == "CLEAN" and clean["open_flags"] == 0
    flagged = audit(flagged_sample(sample))
    assert flagged["status"] == "FLAGGED"
    assert flagged["flag_counts"] == {"MAPPED_FIELD_MISSING": 1}


def test_acknowledging_every_flag_resolves_them(sample):
    flagged = audit(flagged_sample(sample))
    resolved = audit(flagged_sample(sample), acknowledged=approval(flagged))
    assert resolved["status"] == "CLEAN" and resolved["open_flags"] == 0
    assert all(f["status"] == "REVIEWED" for f in resolved["flags"])
    assert resolved["acknowledgements_sha256"] != flagged["acknowledgements_sha256"]


def test_excluded_acknowledgement_is_reported_as_a_limitation(sample):
    flagged = audit(flagged_sample(sample))
    resolved = audit(
        flagged_sample(sample), acknowledged=approval(flagged, "excluded")
    )
    assert resolved["status"] == "CLEAN"
    assert [x["metric"] for x in resolved["limitations"]] == ["cash"]


@pytest.mark.parametrize(
    "damage",
    [
        lambda d: d.update(rationale=""),
        lambda d: d.update(evidence_refs=[]),
        lambda d: d.update(evidence_refs=[" "]),
        lambda d: d.update(disposition="ignored"),
        lambda d: d.update(evidence_sha256="0" * 64),
    ],
)
def test_incomplete_or_mismatched_acknowledgement_does_not_resolve(sample, damage):
    flagged = audit(flagged_sample(sample))
    records = approval(flagged)
    damage(records["decisions"][0])
    assert audit(flagged_sample(sample), acknowledged=records)["status"] == "FLAGGED"


def test_new_issues_stay_open_when_others_are_acknowledged(sample):
    records = approval(audit(flagged_sample(sample)))
    f, m = sample
    other = f[~f.metric.isin(["cash", "assets"])]
    result = audit((other, m), acknowledged=records)
    assert result["status"] == "FLAGGED"
    assert result["flag_counts"] == {"MAPPED_FIELD_MISSING": 1}


def test_duplicate_acknowledgements_are_rejected(sample):
    records = approval(audit(flagged_sample(sample)))
    records["decisions"].append(dict(records["decisions"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        audit(flagged_sample(sample), acknowledged=records)


def test_unsupported_acknowledgement_schema_is_rejected(sample):
    with pytest.raises(ValueError, match="schema"):
        audit(sample, acknowledged=dict(schema="old", decisions=[]))
