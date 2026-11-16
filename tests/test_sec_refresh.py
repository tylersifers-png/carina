import json
import shutil

import pytest

from carina import common as sec, sec_refresh, store, updates
from carina.registry import company
from tests.test_store import built


def _verified_store(built):
    root, _, output = built
    current = root / store.STAGED
    shutil.copytree(output, current)
    manifest = json.loads((current / "manifest.json").read_bytes())
    manifest.update(status="VERIFIED", verified_at=sec.now())
    store.replace_generated(current / "manifest.json", sec.encoded(manifest))
    return root


def _receipt(symbol, key):
    cik = company(symbol)["issuer_cik"]
    endpoint = "submissions" if key == "submissions" else "api/xbrl/companyfacts"
    return {
        "source_url": f"https://data.sec.gov/{endpoint}/CIK{cik:010d}.json",
        "http_status": 200,
        "requested_at": sec.now(),
        "retrieved_at": sec.now(),
        "sha256": "a" * 64,
        "bytes": 100,
        "etag": None,
        "last_modified": None,
    }


def _submissions(symbol, accession=None):
    cik = company(symbol)["issuer_cik"]
    recent = {
        "accessionNumber": [],
        "form": [],
        "filingDate": [],
        "reportDate": [],
        "acceptanceDateTime": [],
    }
    if accession:
        recent = {
            "accessionNumber": [accession],
            "form": ["10-Q"],
            "filingDate": ["2026-11-12"],
            "reportDate": ["2026-09-30"],
            "acceptanceDateTime": ["2026-11-12T21:00:00Z"],
        }
    return {
        "cik": str(cik),
        "name": company(symbol)["name"],
        "tickers": [symbol],
        "filings": {"recent": recent},
    }


def _facts(symbol, accession):
    cik = company(symbol)["issuer_cik"]
    return {
        "cik": cik,
        "entityName": company(symbol)["name"],
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "val": 424242,
                                "end": "2026-09-30",
                                "filed": "2026-11-12",
                                "accn": accession,
                                "form": "10-Q",
                                "fy": 2026,
                                "fp": "Q3",
                            }
                        ]
                    }
                }
            }
        },
    }


class NoWait:
    def __init__(self):
        self.calls = 0

    def wait(self):
        self.calls += 1


def test_sec_only_request_never_schedules_market_work(built, monkeypatch):
    root = _verified_store(built)
    monkeypatch.setattr(updates, "check_expansion", lambda plan: None)
    request = updates.request(
        ["AAPL"],
        add_companies=False,
        add_periods=True,
        refresh_financials=True,
        refresh_market=False,
    )
    operations = updates.operations(updates.snapshot(root, request))
    assert operations["sec_companies"] == ["AAPL"]
    assert operations["market"] == {}


def test_collect_scans_every_company_and_fetches_facts_only_for_new_filing(built):
    root = _verified_store(built)
    accession = "0000320193-26-000123"
    calls = []

    def provider(symbol, key):
        calls.append((symbol, key))
        source = (
            _submissions(symbol, accession if symbol == "AAPL" else None)
            if key == "submissions"
            else _facts(symbol, accession)
        )
        return source, _receipt(symbol, key), json.dumps(source).encode()

    sec_refresh.initialize(root, "sec-test", ["AAPL", "MSFT"])
    limiter = NoWait()
    state = sec_refresh.collect(root, "sec-test", fetch=provider, limiter=limiter)
    assert calls == [
        ("AAPL", "submissions"),
        ("MSFT", "submissions"),
        ("AAPL", "facts"),
    ]
    assert limiter.calls == 3
    assert set(state["captures"]) == {"AAPL"}
    assert state["scans"]["AAPL"]["new_accessions"] == [accession]
    assert not list((root / store.RUNS / "sec-test").glob("*.json"))
    assert not list((root / store.RUNS / "sec-test").glob("*AAPL*"))


def test_same_day_new_accession_is_detected_but_older_gap_is_not():
    stored = {"accessions": {"old"}, "latest_date": "2026-11-12"}
    rows = [
        {"accessionNumber": "old", "filingDate": "2026-11-12"},
        {"accessionNumber": "same-day", "filingDate": "2026-11-12"},
        {"accessionNumber": "historical-gap", "filingDate": "2026-11-11"},
    ]
    assert [row["accessionNumber"] for row in sec_refresh._new_filings(rows, stored)] == [
        "same-day"
    ]


def test_stage_identity_is_bound_to_compact_source_and_projection_hashes():
    state = {
        "schema": sec_refresh.SCHEMA,
        "run_id": "sec-test",
        "scan_plan": {"request": {"companies": ["AAPL"]}},
        "collected_at": "2026-11-16T12:00:00+00:00",
        "scans": {
            "AAPL": {
                "receipt": {"sha256": "a" * 64},
                "new_accessions": ["0000320193-26-000125"],
                "filings": [{"large": "projection need not be re-encoded"}],
            }
        },
        "captures": {
            "AAPL": {
                "projection_sha256": "b" * 64,
                "facts": {"large": "projection need not be re-encoded"},
                "receipts": {
                    "facts": {"sha256": "c" * 64},
                    "submissions": {"sha256": "a" * 64},
                },
            }
        },
    }
    original = sec_refresh.identity(state)
    state["captures"]["AAPL"]["facts"]["large"] = "ignored duplicate payload"
    assert sec_refresh.identity(state) == original
    state["captures"]["AAPL"]["projection_sha256"] = "d" * 64
    assert sec_refresh.identity(state) != original


def test_rate_limiter_enforces_tenth_second_between_request_starts():
    now = [0.0]
    sleeps = []

    def clock():
        return now[0]

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    limiter = sec_refresh.RateLimiter(clock=clock, sleep=sleep)
    limiter.wait()
    now[0] += 0.04
    limiter.wait()
    assert sleeps == [0.060000000000000005]


def test_production_sec_refresh_requires_contact_email(built, monkeypatch):
    root = _verified_store(built)
    sec_refresh.initialize(root, "sec-contact", ["AAPL"])
    monkeypatch.delenv("SEC_CONTACT_EMAIL", raising=False)
    with pytest.raises(ValueError, match="SEC_CONTACT_EMAIL is required"):
        sec_refresh.collect(root, "sec-contact", limiter=NoWait())


def test_run_returns_two_and_keeps_checkpoint_when_history_needs_review(
    built, monkeypatch
):
    root = _verified_store(built)
    accession = "0000320193-26-000124"

    def provider(symbol, key):
        source = _submissions(symbol, accession) if key == "submissions" else _facts(symbol, accession)
        return source, _receipt(symbol, key), json.dumps(source).encode()

    def blocked(*args, **kwargs):
        raise ValueError("Existing SEC fact revised; explicit full refresh required")

    monkeypatch.setattr(sec_refresh, "build", blocked)
    result = sec_refresh.run(
        root,
        "sec-review",
        symbols=["AAPL"],
        fetch=provider,
        limiter=NoWait(),
    )
    assert result == 2
    state = sec_refresh.load(root, "sec-review")
    assert state["review"]["status"] == "NEEDS_REVIEW"


def test_new_filing_builds_verifies_and_promotes_without_market_fetch(built):
    root = _verified_store(built)
    accession = "0000320193-26-000125"

    def provider(symbol, key):
        source = _submissions(symbol, accession) if key == "submissions" else _facts(symbol, accession)
        return source, _receipt(symbol, key), json.dumps(source).encode()

    result = sec_refresh.run(
        root,
        "sec-promote",
        symbols=["AAPL"],
        fetch=provider,
        limiter=NoWait(),
    )
    assert result == 0
    receipt = json.loads((root / store.RECEIPTS / "sec-promote.json").read_bytes())
    assert receipt["companies_refetched"] == ["AAPL"]
    assert receipt["new_accessions"] == {"AAPL": [accession]}
    assert receipt["fundamental_rows_added"] == 1
    assert receipt["filing_rows_added"] == 1
    assert not (root / store.RUNS / "sec-promote").exists()


def test_no_change_records_receipts_and_removes_run_folder(built):
    root = _verified_store(built)
    calls = []

    def provider(symbol, key):
        calls.append((symbol, key))
        source = _submissions(symbol)
        return source, _receipt(symbol, key), json.dumps(source).encode()

    result = sec_refresh.run(
        root,
        "sec-no-change",
        symbols=["AAPL"],
        fetch=provider,
        limiter=NoWait(),
    )
    assert result == 0
    assert calls == [("AAPL", "submissions")]
    assert not (root / store.RUNS / "sec-no-change").exists()
    receipt = json.loads((root / store.RECEIPTS / "sec-no-change.json").read_bytes())
    assert receipt["status"] == "NO_CHANGES"
    assert set(receipt["submissions_receipts"]) == {"AAPL"}
