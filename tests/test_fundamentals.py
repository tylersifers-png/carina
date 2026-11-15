from copy import deepcopy

import pytest

from carina import fundamentals as norm
from carina import common as sec

TARGETS = {
    "annual_2024": ("0000320193-24-000123", "10-K", "2024-09-28", "aapl-20240928.htm"),
    "annual_2025": ("0000320193-25-000079", "10-K", "2025-09-27", "aapl-20250927.htm"),
    "quarter_2026_2": ("0000320193-26-000013", "10-Q", "2026-03-28", "aapl-20260328.htm"),
    "quarter_2026_3": ("0000320193-26-000020", "10-Q", "2026-06-27", "aapl-20260627.htm"),
}
STARTS = {
    "2025-09-27": "2024-09-29",
    "2024-09-28": "2023-10-01",
    "2026-06-27": "2026-03-29",
    "2026-03-28": "2025-12-28",
}
FILED = {
    "2025-09-27": "2025-10-31",
    "2024-09-28": "2024-11-01",
    "2026-06-27": "2026-07-31",
    "2026-03-28": "2026-05-01",
}


def rows_to_columns(rows):
    return {key: [row[key] for row in rows] for key in rows[0]}


def document(end, start, form, entries):
    facts = []
    for namespace, tag, unit, value, kind in entries:
        unit_id, scale = {
            "USD": ("usd", 6),
            "shares": ("shares", 3),
            "USD/shares": ("eps", 0),
        }[unit]
        shown = norm.number(value) / (norm.Decimal(10) ** scale)
        facts.append(
            f'<ix:nonFraction name="{namespace}:{tag}" contextRef="{kind}" unitRef="{unit_id}" scale="{scale}" format="ixt:num-dot-decimal">{shown}</ix:nonFraction>'
        )
    return f"""<html xmlns="http://www.w3.org/1999/xhtml" xmlns:ix="http://www.xbrl.org/2013/inlineXBRL" xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:iso4217="http://www.xbrl.org/2003/iso4217" xmlns:us-gaap="http://fasb.org/us-gaap/2025" xmlns:dei="http://xbrl.sec.gov/dei/2025" xmlns:ixt="http://www.xbrl.org/inlineXBRL/transformation/2020-02-12">
    <ix:nonNumeric name="dei:EntityCentralIndexKey">0000320193</ix:nonNumeric>
    <ix:nonNumeric name="dei:DocumentType">{form}</ix:nonNumeric>
    <ix:nonNumeric name="dei:DocumentPeriodEndDate">{end}</ix:nonNumeric>
    <xbrli:context id="duration"><xbrli:entity><xbrli:identifier>320193</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:startDate>{start}</xbrli:startDate><xbrli:endDate>{end}</xbrli:endDate></xbrli:period></xbrli:context>
    <xbrli:context id="instant"><xbrli:entity><xbrli:identifier>320193</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:instant>{end}</xbrli:instant></xbrli:period></xbrli:context>
    <xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>
    <xbrli:unit id="shares"><xbrli:measure>xbrli:shares</xbrli:measure></xbrli:unit>
    <xbrli:unit id="eps"><xbrli:divide><xbrli:unitNumerator><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unitNumerator><xbrli:unitDenominator><xbrli:measure>xbrli:shares</xbrli:measure></xbrli:unitDenominator></xbrli:divide></xbrli:unit>
    {''.join(facts)}</html>""".encode()


@pytest.fixture
def sources():
    """Company facts, a filing index, and a source receipt for four Apple filings."""
    facts = {"cik": 320193, "facts": {"us-gaap": {}, "dei": {}}}
    recent = []
    for accn, form, end, filename in TARGETS.values():
        recent.append(
            {
                "accessionNumber": accn,
                "form": form,
                "filingDate": FILED[end],
                "reportDate": end,
                "acceptanceDateTime": FILED[end] + "T10:00:00.000Z",
                "primaryDocument": filename,
            }
        )
        for i, (metric, (unit, kind, mappings)) in enumerate(norm.METRICS.items()):
            namespace, tag, _ = mappings[0]
            value = (
                (i + 1) * (1000000 if unit == "USD" else 1000)
                if unit != "USD/shares"
                else 2.5
            )
            row = {
                "val": value,
                "end": end,
                "filed": FILED[end],
                "accn": accn,
                "form": form,
                "fy": int(end[:4]),
                "fp": "FY" if form == "10-K" else "Q3",
            }
            if kind == "duration":
                row["start"] = STARTS[end]
            facts["facts"][namespace].setdefault(tag, {"units": {unit: []}})["units"][
                unit
            ].append(row)
    body = sec.encoded(facts)
    receipt = {
        "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        "http_status": 200,
        "requested_at": sec.now(),
        "retrieved_at": sec.now(),
        "sha256": sec.digest(body),
        "bytes": len(body),
    }
    return facts, recent, receipt


def test_conflicting_aliases_are_not_added_or_arbitrarily_selected(sources):
    facts, recent, receipt = sources
    facts = deepcopy(facts)
    original = facts["facts"]["us-gaap"]["SalesRevenueNet"]["units"]["USD"][0]
    facts["facts"]["us-gaap"]["Revenues"] = {
        "units": {"USD": [{**original, "val": original["val"] + 1}]}
    }
    _, canonical, latest, _ = norm.normalize(
        facts, norm.filing_rows(rows_to_columns(recent)), receipt
    )
    conflict = next(r for r in canonical if r["status"] == "CONFLICT")
    assert conflict["metric"] == "revenue" and conflict["value"] == ""
    assert ";" in conflict["candidate_values"]
    assert any(r["status"] == "LATEST_CONFLICT" for r in latest)


def test_later_restatement_is_preserved_as_a_new_version(sources):
    facts, recent, receipt = sources
    facts = deepcopy(facts)
    values = facts["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"]
    old = deepcopy(next(r for r in values if r["end"] == "2024-09-28"))
    changed = {
        **old,
        "val": old["val"] + 123,
        "filed": "2025-10-31",
        "accn": "0000320193-25-000079",
        "fy": 2025,
    }
    values.append(changed)
    reported, canonical, latest, _ = norm.normalize(
        facts, norm.filing_rows(rows_to_columns(recent)), receipt
    )
    period_rows = [
        r for r in canonical if r["metric"] == "net_income" and r["end"] == old["end"]
    ]
    assert len(period_rows) == 2
    selected = next(
        r for r in latest if r["metric"] == "net_income" and r["end"] == old["end"]
    )
    assert norm.number(selected["value"]) == changed["val"]
    assert "NOT_HISTORICAL_AS_KNOWN" in selected["view"]
    assert any(
        r["reported_fiscal_year"] == 2025 and r["end"] == "2024-09-28" for r in reported
    )


def test_unsupported_unit_is_explicit_not_converted(sources):
    facts, recent, receipt = sources
    facts = deepcopy(facts)
    tag = facts["facts"]["us-gaap"]["Assets"]["units"]
    tag["EUR"] = tag.pop("USD")
    reported, _, _, issues = norm.normalize(
        facts, norm.filing_rows(rows_to_columns(recent)), receipt
    )
    assert not any(r["metric"] == "assets" for r in reported)
    assert any(r["reason"] == "UNIT_OR_PERIOD_MISMATCH" for r in issues)


def valid_index(sources):
    return rows_to_columns(sources[1])


def test_filing_rows_accepts_a_valid_index(sources):
    rows = norm.filing_rows(valid_index(sources))
    assert len(rows) == 4 and rows[0]["accessionNumber"].startswith("0000320193-")


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d.pop("acceptanceDateTime"), "Missing filing metadata"),
        (lambda d: d["form"].pop(), "Inconsistent filing arrays"),
        (lambda d: d["accessionNumber"].__setitem__(1, d["accessionNumber"][0]), "Duplicate filing accession"),
        (lambda d: d["accessionNumber"].__setitem__(0, "bad"), "Invalid filing accession"),
        (lambda d: d["acceptanceDateTime"].__setitem__(0, "2025-10-31T10:00:00"), "lacks timezone"),
    ],
)
def test_filing_rows_rejects_malformed_indexes(sources, change, message):
    data = deepcopy(valid_index(sources))
    change(data)
    with pytest.raises(ValueError, match=message):
        norm.filing_rows(data)


def test_annual_quarter_and_ytd_are_not_conflated():
    assert norm.period("2024-09-29", "2025-09-27", "duration") == "ANNUAL"
    assert norm.period("2024-09-29", "2025-06-28", "duration") == "YTD_9M"
    assert norm.period("2024-09-29", "2025-03-29", "duration") == "YTD_6M"
    assert norm.period("2025-03-30", "2025-06-28", "duration") == "QUARTER"
    assert (
        norm.period("2022-09-25", "2023-09-30", "duration") == "ANNUAL"
    )  # 53-week year.
    assert norm.period("", "2025-09-27", "instant") == "INSTANT"

def test_original_numeric_sign_scale_and_entity_context():
    raw = document(
        "2025-09-27",
        "2024-09-29",
        "10-K",
        [("us-gaap", "NetIncomeLoss", "USD", 2000000, "duration")],
    )
    raw = raw.replace(b'scale="6"', b'scale="6" sign="-"')
    values, _ = norm.inline_facts(raw)
    assert (
        values[("us-gaap:NetIncomeLoss", "USD", "2024-09-29", "2025-09-27")][0]["value"]
        == -2000000
    )
    with pytest.raises(ValueError, match="identify Apple"):
        norm.inline_facts(raw.replace(b"0000320193", b"0000000001"))

def test_dimensional_context_is_not_used_as_entity_total():
    raw = document(
        "2025-09-27",
        "2024-09-29",
        "10-K",
        [("us-gaap", "Assets", "USD", 2000000, "instant")],
    )
    raw = raw.replace(b"<xbrli:instant>", b"<xbrli:scenario/><xbrli:instant>")
    values, _ = norm.inline_facts(raw)
    assert not values

def test_wrong_taxonomy_cannot_impersonate_standard_facts():
    raw = document(
        "2025-09-27",
        "2024-09-29",
        "10-K",
        [("us-gaap", "Assets", "USD", 2000000, "instant")],
    )
    with pytest.raises(ValueError, match="taxonomy"):
        norm.inline_facts(
            raw.replace(b"http://fasb.org/us-gaap/2025", b"https://example.com/custom")
        )
