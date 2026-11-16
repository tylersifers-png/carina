from datetime import datetime, timedelta, timezone
import hashlib
import io
import json

import exchange_calendars as xcals
import pandas as pd
import pytest

from carina import settings, store
from carina import common as sec
from carina.drive import RemoteItem
from carina.registry import company, metrics_for
from tests.registry_fixture import (
    INITIAL_COMPANIES,
    INITIAL_ROWS,
    synthetic_rows,
)

# Archive/control fixtures use the fixed ten-company approved list.
COMPANIES = INITIAL_COMPANIES
from tests.drive_fakes import MemoryDrive


def test_sec_user_agent_uses_plain_declared_contact_format():
    assert store.sec_user_agent("operator@example.test", require_contact=True) == (
        "Carina Data Pipeline operator@example.test"
    )


class MutableDrive(MemoryDrive):
    def update_file(self, file_id, local_path, name):
        if name == self.fail_upload:
            raise RuntimeError("Simulated interruption")
        for items in self.children.values():
            for i, item in enumerate(items):
                if item.id == file_id:
                    items[i] = RemoteItem(
                        file_id,
                        name,
                        item.mime_type,
                        hashlib.md5(local_path.read_bytes()).hexdigest(),
                    )
        self.content[file_id] = local_path.read_bytes()
        self.updated_files.append((file_id, name))


def envelope(symbols=tuple(COMPANIES)):
    content = dict(schema=store.SCHEMA, collected_at=sec.now(), companies={})
    for symbol in symbols:
        cik = company(symbol)["issuer_cik"]
        fact = dict(
            cik=cik,
            facts={
                "us-gaap": {
                    "Assets": {
                        "units": {
                            "USD": [
                                dict(
                                    val=1000,
                                    end="2025-12-31",
                                    filed="2026-02-13",
                                    accn=f"{cik:010d}-26-000001",
                                    form="10-K",
                                )
                            ]
                        }
                    }
                }
            },
        )
        receipts = {}
        for key, endpoint in (
            ("facts", "api/xbrl/companyfacts"),
            ("submissions", "submissions"),
        ):
            receipts[key] = dict(
                source_url=f"https://data.sec.gov/{endpoint}/CIK{cik:010d}.json",
                http_status=200,
                requested_at=sec.now(),
                retrieved_at=sec.now(),
                sha256="a" * 64,
                bytes=100,
            )
        content["companies"][symbol] = dict(
            facts=fact,
            projection_sha256=sec.digest(sec.encoded(fact)),
            filings=[],
            receipts=receipts,
            issuer_metadata=dict(
                cik=cik, name=company(symbol)["name"], tickers=[symbol]
            ),
        )
    return content


def fake_market(symbol, start, root):
    dates = xcals.get_calendar(
        "XNYS",
        start=start,
        end=str(datetime.fromisoformat(start).date() + timedelta(days=10)),
    ).sessions[:2]
    frame = pd.DataFrame(
        dict(
            Open=[10.0, 11.0],
            High=[12.0, 13.0],
            Low=[9.0, 10.0],
            Close=[11.0, 12.0],
            **{
                "Adj Close": [10.5, 11.5],
                "Volume": [100, 200],
                "Dividends": [0.0, 0.1],
                "Stock Splits": [0.0, 0.0],
            },
        ),
        index=dates,
    )
    frame.index.name = "Date"
    raw = frame.to_csv().encode()
    return raw, dict(
        symbol=symbol,
        requested_start=start,
        last_session=str(dates[-1].date()),
        requested_at=(
            dates[-1].to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(days=1)
        ).isoformat(),
        retrieved_at=(
            dates[-1].to_pydatetime().replace(tzinfo=timezone.utc)
            + timedelta(days=1, seconds=1)
        ).isoformat(),
        exclusive_end=str(dates[-1].date() + timedelta(days=1)),
        metadata=dict(symbol=symbol, currency="USD"),
        parameters=store.PARAMETERS,
        sha256=sec.digest(raw),
    )


def test_rsp_market_accepts_retained_capital_gain_distribution(tmp_path):
    raw, receipt = fake_market("RSP", "2003-05-01", tmp_path)
    frame = pd.read_csv(io.BytesIO(raw))
    frame["Capital Gains"] = [0.0, 0.25]
    raw = frame.to_csv(index=False).encode()
    receipt["sha256"] = sec.digest(raw)
    normalized, coverage = store.validate_market(raw, receipt)
    assert "Capital Gains" not in normalized
    assert coverage["capital_gain_events"] == 1


@pytest.fixture
def built(tmp_path, monkeypatch):
    # An injected market source must work without the optional live Yahoo package.
    real_version = store.version

    def processing_version(package):
        if package == "yfinance":
            from importlib.metadata import PackageNotFoundError

            raise PackageNotFoundError(package)
        return real_version(package)

    monkeypatch.setattr(store, "version", processing_version)
    content = envelope()
    path = tmp_path / "input"
    path.mkdir()
    sec.prepare_capture(path, content, validator=store.validate, **store.BOUNDS)
    state = json.loads((path / "handoff.json").read_bytes())
    destination = store.run_path(tmp_path, state["bundle"])
    destination.mkdir(parents=True)
    for part in path.glob("part_*.json"):
        (destination / part.name).write_bytes(part.read_bytes())
    output = store.build(tmp_path, state["bundle"], market_fetch=fake_market)
    return tmp_path, state["bundle"], output


def test_build_has_registered_companies_two_distinct_benchmarks_and_no_report(built):
    root, bundle, output = built
    assert len(pd.read_parquet(output / "companies.parquet")) == len(COMPANIES)
    daily = pd.read_parquet(output / "daily_bars.parquet")
    assert len(daily) == 2 * (len(COMPANIES) + len(store.BENCHMARKS))
    assert (
        daily.groupby("symbol").price_return.apply(lambda x: x.isna().sum()).eq(1).all()
    )
    assert set(daily.symbol) == set(COMPANIES) | set(store.BENCHMARKS)
    assert not (root / "state/ops/reports").exists()
    assert not pd.read_parquet(output / "fundamentals.parquet").filing_accepted_at.any()
    coverage = json.loads((output / "coverage.json").read_bytes())
    assert (
        coverage["AAPL"]["financials"]["fields"]["revenue"]["status"]
        == "NO_MAPPED_REPORTED_FACTS"
    )
    sources = json.loads((output / store.SOURCES).read_bytes())
    assert set(sources) == {"market", "sec"}
    assert all(set(item) == {"collected_at", "receipts"} for item in sources["sec"].values())


def test_profiles_do_not_coerce_bank_and_insurance_into_industrial_fcf():
    assert "deposits" in metrics_for("JPM")
    assert "operating_cash_flow" not in metrics_for("JPM")
    assert "operating_cash_flow" not in metrics_for("BRK-B")
    assert "operating_cash_flow" in metrics_for("AAPL")
    from carina import registry

    bank = dict(synthetic_rows("SYNBANK")[0], accounting_profile="bank")
    registry.COMPANIES.use([*INITIAL_ROWS, bank])
    banks = [s for s in registry.COMPANIES if company(s)["accounting_profile"] == "bank"]
    assert {"JPM", "SYNBANK"} <= set(banks)
    for symbol in banks:
        assert "operating_cash_flow" not in metrics_for(symbol)
        assert "deposits" in metrics_for(symbol)
    assert company("AAPL")["issuer_cik"] == 320193
    with pytest.raises(ValueError):
        company("UNAPPROVED_FIXTURE")


@pytest.mark.parametrize("change", ["issuer", "hash", "tag", "url", "ticker"])
def test_reject_tampered_projection(change):
    data = envelope(["AAPL"])
    item = data["companies"]["AAPL"]
    if change == "issuer":
        item["facts"]["cik"] = 123
    if change == "hash":
        item["projection_sha256"] = "0" * 64
    if change == "tag":
        item["facts"]["facts"]["us-gaap"]["UnselectedMetric"] = {}
    if change == "url":
        item["receipts"]["facts"]["source_url"] = "https://example.com/"
    if change == "ticker":
        item["issuer_metadata"]["tickers"] = ["OTHER"]
    with pytest.raises(ValueError):
        store.validate(data)


def test_empty_sec_ticker_list_is_explicit_not_wrong_issuer():
    data = envelope(["XOM"])
    data["companies"]["XOM"]["issuer_metadata"]["tickers"] = []
    assert store.validate(data)["companies"]["XOM"]["issuer_metadata"]["tickers"] == []


@pytest.mark.parametrize(
    "change", ["gap", "duplicate", "negative", "nan", "order", "hash"]
)
def test_bad_market_data_fails_closed(change, tmp_path):
    raw, receipt = fake_market("AAPL", "1997-01-01", tmp_path)
    frame = pd.read_csv(__import__("io").BytesIO(raw))
    if change == "gap":
        frame = frame.iloc[1:]
    if change == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    if change == "negative":
        frame.loc[0, "Volume"] = -1
    if change == "nan":
        frame.loc[0, "Close"] = float("nan")
    if change == "order":
        frame = frame.iloc[::-1]
    raw = frame.to_csv(index=False).encode()
    if change != "hash":
        receipt["sha256"] = sec.digest(raw)
    else:
        receipt["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        store.validate_market(raw, receipt)


def test_corrupt_download_cannot_be_used_as_current(built):
    _, _, output = built
    manifest = json.loads((output / "manifest.json").read_bytes())
    (output / "daily_bars.parquet").write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        store.verify_local(output, manifest)


def test_partial_company_batch_cannot_replace_full_store(tmp_path):
    path = tmp_path / "capture"
    path.mkdir()
    sec.prepare_capture(
        path, envelope(["AAPL"]), validator=store.validate, **store.BOUNDS
    )
    bundle = json.loads((path / "handoff.json").read_bytes())["bundle"]
    destination = store.run_path(tmp_path, bundle)
    destination.mkdir(parents=True)
    for piece in path.glob("part_*.json"):
        (destination / piece.name).write_bytes(piece.read_bytes())
    with pytest.raises(ValueError, match="full approved panel"):
        store.build(tmp_path, bundle, fake_market)


def test_unsafe_path_rejected(tmp_path):
    with pytest.raises(Exception):
        store.path_for(tmp_path, "../escape")


def test_coexisting_revenue_bases_are_not_false_conflicts():
    from carina import fundamentals as norm

    data = envelope(["WMT"])["companies"]["WMT"]
    point = dict(
        val=100,
        start="2025-01-01",
        end="2025-12-31",
        filed="2026-02-13",
        accn="0000104169-26-000001",
        form="10-K",
    )
    data["facts"]["facts"]["us-gaap"] = {
        "SalesRevenueNet": {"units": {"USD": [point]}},
        "Revenues": {"units": {"USD": [dict(point, val=110)]}},
    }
    _, canonical, latest, _ = norm.normalize(
        data["facts"], [], data["receipts"]["facts"], metrics=metrics_for("WMT")
    )
    assert len(latest) == 2
    assert len({r["basis"] for r in latest}) == 2
    assert all(r["status"] == "REPORTED" for r in canonical + latest)


def test_generated_write_allows_symlink_only_above_trusted_root(tmp_path, monkeypatch):
    real = tmp_path / "real"
    trusted = real / "project"
    trusted.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    settings.configure(data=alias / "project")

    target = alias / "project" / "data" / "receipt.json"
    store.replace_generated(target, b"safe\n")

    assert target.read_bytes() == b"safe\n"


def test_generated_write_rejects_symlink_below_trusted_root(tmp_path, monkeypatch):
    trusted = tmp_path / "project"
    outside = tmp_path / "outside"
    trusted.mkdir()
    outside.mkdir()
    (trusted / "linked").symlink_to(outside, target_is_directory=True)
    settings.configure(data=trusted)

    with pytest.raises(ValueError, match="Symlink"):
        store.replace_generated(trusted / "linked" / "receipt.json", b"unsafe\n")
    assert not (outside / "receipt.json").exists()
