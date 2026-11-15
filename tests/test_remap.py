"""Regression checks for reviewed remapping without price/provider refreshes."""

from copy import deepcopy
import bz2
import json
import shutil

import pytest

from carina import remap as cleanup, store, common as sec
from carina import fundamentals as norm
from carina.registry import metrics_for, quarantines_for
from tests.test_store import built
from tests.test_fundamentals import document


def original(value=1000000):
    raw = document(
        "2025-09-27",
        "2024-09-29",
        "10-K",
        [("us-gaap", "Assets", "USD", value, "instant")],
    ).replace(b'scale="6"', b'decimals="-6" scale="6"')
    filing = dict(
        accessionNumber="0000320193-25-000001",
        form="10-K",
        filingDate="2025-10-31",
        reportDate="2025-09-27",
        acceptanceDateTime="2025-10-31T10:00:00Z",
        primaryDocument="aapl.htm",
    )
    receipt = dict(
        sha256=sec.digest(raw),
        retrieved_at=sec.now(),
        source_url="https://www.sec.gov/Archives/edgar/data/320193/000032019325000001/aapl.htm",
    )
    projection = norm.original_projection(raw, cik=320193, metrics=metrics_for("AAPL"))
    return dict(filing=filing, receipt=receipt, projection=projection)


def test_joint_filing_scopes_dei_metadata_to_nondimensional_registrant():
    raw = document(
        "2025-09-27",
        "2024-09-29",
        "10-K",
        [("us-gaap", "Assets", "USD", 1000000, "instant")],
    ).decode()
    extra = '<xbrli:context id="other"><xbrli:entity><xbrli:identifier>320193</xbrli:identifier><xbrli:segment><member>OtherRegistrant</member></xbrli:segment></xbrli:entity><xbrli:period><xbrli:instant>2025-09-27</xbrli:instant></xbrli:period></xbrli:context><ix:nonNumeric name="dei:EntityCentralIndexKey" contextRef="other">2115436</ix:nonNumeric><ix:nonFraction name="us-gaap:Assets" contextRef="other" unitRef="usd">999999</ix:nonFraction>'
    values, meta = norm.inline_facts(raw.replace("</html>", extra + "</html>").encode())
    assert meta["EntityCentralIndexKey"] == ["0000320193"]
    assert {v["value"] for rows in values.values() for v in rows} == {
        norm.number(1000000)
    }
    with pytest.raises(ValueError, match="does not identify Apple"):
        norm.inline_facts(raw.replace("0000320193", "0000211543").encode())


def test_original_supplement_adds_absent_version_with_provenance():
    item = original()
    rows = norm.supplement_original([], item, cik=320193, metrics=metrics_for("AAPL"))
    assert len(rows) == 1 and rows[0]["value"] == "1000000"
    assert rows[0]["source_kind"] == "ORIGINAL_FILING_ENTITY_CONTEXT"
    assert rows[0]["source_sha256"] == item["receipt"]["sha256"]
    assert rows[0]["original_context_ids"] == "instant"
    assert (
        norm.supplement_original(rows, item, cik=320193, metrics=metrics_for("AAPL"))
        == rows
    )
    changed = deepcopy(rows)
    changed[0]["value"] = "1"
    with pytest.raises(ValueError, match="API/original disagreement"):
        norm.supplement_original(changed, item, cik=320193, metrics=metrics_for("AAPL"))


def test_original_precision_requires_exact_precise_value_and_consistent_rounding():
    item = original(9037000000)
    values = item["projection"]["facts"][0]["values"]
    values.append(
        dict(value="9000000000", decimals="-8", context_id="rounded", element_id="")
    )
    rows = norm.supplement_original([], item, cik=320193, metrics=metrics_for("AAPL"))
    assert rows[0]["value"] == "9037000000"
    values[-1]["value"] = "8000000000"
    with pytest.raises(ValueError, match="Inconsistent original"):
        norm.supplement_original([], item, cik=320193, metrics=metrics_for("AAPL"))
    values[-1].update(value="9038000000", decimals="-6")
    with pytest.raises(ValueError, match="equally precise"):
        norm.supplement_original([], item, cik=320193, metrics=metrics_for("AAPL"))


def test_quarantine_preserves_raw_and_reopens_if_evidence_changes():
    q = quarantines_for("AAPL")[1]
    row = {
        k: q[k] for k in ("metric", "basis", "unit", "start", "end", "accession")
    } | dict(value=q["expected_value"], fact_ids="a" * 64)
    raw = deepcopy(row)
    kept, issues = norm.quarantine_facts([row], [], [q])
    assert kept == [] and row == raw
    assert issues[0]["reason"] == "SOURCE_FACT_QUARANTINED"
    row["value"] = "861745000"
    kept, issues = norm.quarantine_facts([row], [], [q])
    assert kept == [row] and issues[0]["reason"] == "QUARANTINE_EVIDENCE_CHANGED"
    row["accession"] = "0000320193-25-000001"
    assert norm.quarantine_facts([row], [], [q]) == ([row], [])


def test_mapping_bases_keep_distinct_scopes_and_bank_profile():
    assert metrics_for("JNJ")["dividends_paid"][2][-1][1:] == (
        "PaymentsOfOrdinaryDividends",
        "ordinary_dividends",
    )
    assert (
        metrics_for("AAPL")["operating_cash_flow"][2][-1][2] == "continuing_operations"
    )
    assert "capex_productive_assets" in metrics_for("NVDA")
    assert (
        metrics_for("NVDA")["capex_productive_assets"][2][0][2]
        != metrics_for("NVDA")["capex_ppe"][2][0][2]
    )
    assert "operating_cash_flow" not in metrics_for("JPM")
    assert "cash_including_restricted" in metrics_for("BRK-B")


def current_and_content(built):
    root, bundle, old = built
    current = root / store.STAGED
    shutil.copytree(old, current)
    manifest = json.loads((current / "manifest.json").read_bytes())
    manifest["status"] = "VERIFIED"
    (current / "manifest.json").write_bytes(sec.encoded(manifest))
    sources = json.loads(bz2.decompress((current / "sources.json.bz2").read_bytes()))
    content = dict(
        schema=cleanup.SCHEMA,
        base_manifest_sha256=sec.digest((current / "manifest.json").read_bytes()),
        collected_at=manifest["collected_at"],
        companies=sources["sec_projection"]["companies"],
        review_evidence={},
    )
    return root, current, content


def test_remap_preserves_all_market_identity_and_filing_bytes(built):
    root, current, content = current_and_content(built)
    before = {n: (current / n).read_bytes() for n in store.FILES}
    output = cleanup.build(root, "f" * 64, content)
    for name in [
        "daily_bars.parquet",
        "companies.parquet",
        "securities.parquet",
        "filings.parquet",
    ]:
        assert (output / name).read_bytes() == before[name]
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["collection_mode"] == "OFFLINE_FINANCIAL_REMAP_NO_PROVIDER_REQUESTS"
    assert manifest["base_manifest_sha256"] == content["base_manifest_sha256"]
    store.verify_local(output, manifest)
    assert manifest["quality"]["status"] == "FLAGGED"


@pytest.mark.parametrize("change", ["baseline", "panel", "receipt", "raw_array"])
def test_remap_rejects_changed_baseline_panel_and_raw_inputs(built, change):
    root, current, content = current_and_content(built)
    if change == "baseline":
        content["base_manifest_sha256"] = "0" * 64
    if change == "panel":
        del content["companies"]["MSFT"]
    if change == "receipt":
        content["companies"]["AAPL"]["receipts"]["facts"]["sha256"] = "0" * 64
    if change == "raw_array":
        item = content["companies"]["AAPL"]
        item["facts"]["facts"]["us-gaap"]["Assets"]["units"]["USD"][0]["val"] = 20
        item["projection_sha256"] = sec.digest(sec.encoded(item["facts"]))
    with pytest.raises(ValueError):
        cleanup.build(root, "f" * 64, content)
    assert not (store.run_path(root, "f" * 64) / "built").exists()


@pytest.mark.parametrize(
    "symbol,end,filed,original_acc,amendment_acc,wrong_value,reported_value",
    [
        (
            "ROST",
            "2010-11-18",
            "2010-12-08",
            "0001206774-10-002571",
            "0001206774-10-002589",
            "19084540",
            "119084540",
        ),
        (
            "BSX",
            "2013-04-30",
            "2013-05-07",
            "0000885725-13-000013",
            "0000885725-13-000015",
            "1550351484",
            "1349092384",
        ),
    ],
)
def test_same_day_cover_error_quarantine_keeps_existing_amendment(
    symbol, end, filed, original_acc, amendment_acc, wrong_value, reported_value
):
    """A confirmed source error must not become an arbitrary amendment tie-break."""
    from copy import deepcopy
    from carina.registry import quarantines_for

    original = dict(
        metric="shares_outstanding",
        basis="cover_page",
        unit="shares",
        start="",
        end=end,
        period_kind="INSTANT",
        accession=original_acc,
        form="10-Q",
        filed=filed,
        filing_accepted_at="",
        value=wrong_value,
        status="REPORTED",
        fact_ids="original-fixture",
    )
    amendment = dict(
        original,
        accession=amendment_acc,
        form="10-Q/A",
        value=reported_value,
        fact_ids="amended-fixture",
    )
    rows = [original, amendment]
    snapshot = deepcopy(rows)
    assert norm.latest_view(rows)[0]["status"] == "LATEST_CONFLICT"
    kept, issues = norm.quarantine_facts(rows, [], quarantines_for(symbol))
    assert kept == [amendment] and rows == snapshot
    selected = norm.latest_view(kept)[0]
    assert selected["value"] == amendment["value"]
    assert selected["candidate_accessions"] == amendment["accession"]
    assert selected["status"] == "REPORTED"
    assert issues[0]["reason"] == "SOURCE_FACT_QUARANTINED"
    assert issues[0]["value"] == original["value"]
    changed = [dict(original, value=str(int(wrong_value) + 1)), amendment]
    kept, issues = norm.quarantine_facts(changed, [], quarantines_for(symbol))
    assert kept == changed
    assert issues[0]["reason"] == "QUARANTINE_EVIDENCE_CHANGED"
    assert norm.latest_view(kept)[0]["status"] == "LATEST_CONFLICT"


def test_hig_narrative_quarantine_keeps_detailed_annual_and_reopens_on_change():
    """Different precision must not become generic annual/amendment precedence."""
    from copy import deepcopy

    coarse = dict(
        metric="equity",
        basis="reported",
        unit="USD",
        start="",
        end="2010-12-31",
        period_kind="INSTANT",
        accession="0000874766-13-000020",
        form="10-Q/A",
        filed="2013-03-01",
        filing_accepted_at="",
        value="18700000000",
        status="REPORTED",
        fact_ids="coarse-narrative",
    )
    annual = dict(
        coarse,
        accession="0000874766-13-000021",
        form="10-K",
        value="18754000000",
        fact_ids="detailed-annual",
    )
    rows = [coarse, annual]
    original = deepcopy(rows)
    assert norm.latest_view(rows)[0]["status"] == "LATEST_CONFLICT"
    kept, issues = norm.quarantine_facts(rows, [], quarantines_for("HIG"))
    assert rows == original and kept == [annual]
    assert norm.latest_view(kept)[0]["value"] == "18754000000"
    assert issues[0]["reason"] == "SOURCE_FACT_QUARANTINED"
    changed = [dict(coarse, value="18754000000"), annual]
    kept, issues = norm.quarantine_facts(changed, [], quarantines_for("HIG"))
    assert kept == changed and issues[0]["reason"] == "QUARANTINE_EVIDENCE_CHANGED"
    other_accession = [dict(coarse, accession="other-unreviewed-filing"), annual]
    kept, issues = norm.quarantine_facts(other_accession, [], quarantines_for("HIG"))
    assert kept == other_accession and not issues
    assert norm.latest_view(kept)[0]["status"] == "LATEST_CONFLICT"


@pytest.mark.parametrize(
    "metric,old_value,new_value,old_accession,new_accession,start,end,period_kind,basis",
    [
        (
            "revenue",
            "3486000000",
            "14748000000",
            "0000079879-18-000040",
            "0000079879-18-000039",
            "2017-01-01",
            "2017-12-31",
            "ANNUAL",
            "reported_SalesRevenueNet",
        ),
        (
            "operating_income",
            "557000000",
            "561000000",
            "0000079879-18-000039",
            "0000079879-18-000040",
            "2017-01-01",
            "2017-03-31",
            "QUARTER",
            "reported",
        ),
    ],
)
def test_ppg_exact_source_quarantines_keep_valid_facts_and_reopen_changes(
    metric,
    old_value,
    new_value,
    old_accession,
    new_accession,
    start,
    end,
    period_kind,
    basis,
):
    old = dict(
        metric=metric,
        basis=basis,
        unit="USD",
        start=start,
        end=end,
        period_kind=period_kind,
        accession=old_accession,
        filed="2018-06-28",
        filing_accepted_at="",
        value=old_value,
        status="REPORTED",
        fact_ids="old",
    )
    correct = dict(old, value=new_value, accession=new_accession, fact_ids="correct")
    rows = [old, correct]
    raw = deepcopy(rows)
    assert norm.latest_view(rows)[0]["status"] == "LATEST_CONFLICT"
    kept, issues = norm.quarantine_facts(rows, [], quarantines_for("PPG"))
    assert rows == raw and kept == [correct]
    assert norm.latest_view(kept)[0]["value"] == new_value
    assert issues[0]["reason"] == "SOURCE_FACT_QUARANTINED"
    changed = [dict(old, value=str(int(old_value) + 1)), correct]
    kept, issues = norm.quarantine_facts(changed, [], quarantines_for("PPG"))
    assert kept == changed and issues[0]["reason"] == "QUARANTINE_EVIDENCE_CHANGED"
    assert norm.latest_view(kept)[0]["status"] == "LATEST_CONFLICT"
    unrelated = dict(old, accession="unreviewed-source")
    assert norm.quarantine_facts([unrelated], [], quarantines_for("PPG")) == (
        [unrelated],
        [],
    )
    if metric == "revenue":
        quarter = dict(old, end="2017-03-31", period_kind="QUARTER")
        assert norm.quarantine_facts([quarter], [], quarantines_for("PPG")) == (
            [quarter],
            [],
        )
