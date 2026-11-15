import json

import pandas as pd
import pytest

from carina import dataset, macro, settings

FRED = b"observation_date,M2NS\n2026-01-01,100.5\n2026-02-01,101.0\n"
FED = (
    b'"Series Description","M2; Not seasonally adjusted"\n'
    b'"Unit:","Currency"\n'
    b'"Unique Identifier: ","H6/H6_M2/M2_N.M"\n'
    b'"Time Period","M2_N.M"\n'
    b"2026-01,100.5\n2026-02,\"101.0\"\n"
)
FINRA = b"""<html><body><table>
<tr><th>Month/Year</th><th>Debit Balances in Customers' Securities Margin Accounts</th>
<th>Free Credit Balances in Customers' Cash Accounts</th></tr>
<tr><td>Feb-26</td><td>1,200</td><td>300</td></tr>
<tr><td>Jan-26</td><td>1,100</td><td>290</td></tr>
</table></body></html>"""
TREASURY = b"""<feed xmlns="http://www.w3.org/2005/Atom"
 xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
 xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices">
<entry><content><m:properties>
<d:NEW_DATE>2026-01-02T00:00:00</d:NEW_DATE><d:BC_1YEAR>3.47</d:BC_1YEAR>
<d:BC_10YEAR>4.10</d:BC_10YEAR><d:BC_30YEARDISPLAY>4.8</d:BC_30YEARDISPLAY>
</m:properties></content></entry></feed>"""
DEBIT = "debit_balances_in_customers_securities_margin_accounts"


class Client:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, timeout):
        self.calls.append(url)
        body = next((v for k, v in self.responses.items() if k in url), None)
        return Response(body)


class Response:
    def __init__(self, body):
        self.content = body

    def raise_for_status(self):
        if self.content is None:
            raise OSError("unavailable")


PLAN = dict(
    schema="carina-collection-1.0",
    companies=[],
    macro=dict(
        fred=[dict(series="M2NS", fallback=dict(url="https://fed.example/h6", column="H6/H6_M2/M2_N.M"))],
        finra_margin={},
        treasury_par_yields=dict(start_year=2026),
    ),
)


@pytest.fixture
def decisions(tmp_path):
    folder = tmp_path / "decisions"
    folder.mkdir()
    (folder / "collection.json").write_text(json.dumps(PLAN))
    settings.configure(data=tmp_path / "data", decisions=folder)
    return folder


def responses(**changes):
    found = {"fredgraph": FRED, "finra.org": FINRA, "treasury.gov": TREASURY}
    found.update(changes)
    return found


def test_parsers_return_long_rows():
    assert macro.parse_fred_csv(FRED, "M2NS").to_dict("records")[0] == dict(
        source="fred", series="M2NS", period="2026-01-01", value=100.5
    )
    assert macro.parse_fed_csv(FED, "H6/H6_M2/M2_N.M", "M2NS").equals(macro.parse_fred_csv(FRED, "M2NS"))
    finra = macro.parse_finra_html(FINRA)
    assert set(finra.series) == {DEBIT, "free_credit_balances_in_customers_cash_accounts"}
    assert finra[finra.series.eq(DEBIT)].value.tolist() == [1100.0, 1200.0]
    treasury = macro.parse_treasury_xml(TREASURY)
    assert sorted(treasury.series) == ["BC_10YEAR", "BC_1YEAR"]
    with pytest.raises(ValueError, match="missing"):
        macro.parse_fred_csv(FRED, "OTHER")


def test_revisions_close_old_values_and_keep_them(decisions, tmp_path):
    root = tmp_path / "data"
    first = macro.refresh(root, as_of="2026-03-01", client=Client(responses()))
    assert first["failures"] == {} and first["lineage"] == []
    head = dataset.manifest_digest(root / macro.STAGED)
    revised = FRED.replace(b"101.0", b"101.4") + b"2026-03-01,102.0\n"
    second = macro.refresh(root, as_of="2026-04-01", client=Client(responses(fredgraph=revised)))
    table = macro.read(root)
    m2 = table[table.series.eq("M2NS")]
    february = m2[m2.period.eq("2026-02-01")].sort_values("realtime_start")
    assert february.value.tolist() == [101.0, 101.4]
    assert february.realtime_end.tolist() == ["2026-04-01", ""]
    assert len(m2) == 4 and macro.current_values(table).duplicated(macro.KEY).sum() == 0
    assert second["lineage"] == [head]
    dataset.verify_files(root / macro.STAGED, second)


def test_same_day_rerun_replaces_instead_of_stacking(decisions, tmp_path):
    root = tmp_path / "data"
    macro.refresh(root, as_of="2026-03-01", client=Client(responses()))
    macro.refresh(root, as_of="2026-03-01", client=Client(responses(fredgraph=FRED.replace(b"101.0", b"99"))))
    m2 = macro.read(root).query("series == 'M2NS' and period == '2026-02-01'")
    assert m2.value.tolist() == [99.0] and m2.realtime_end.tolist() == [""]


def test_full_history_source_cannot_drop_periods(decisions, tmp_path):
    root = tmp_path / "data"
    macro.refresh(root, as_of="2026-03-01", client=Client(responses()))
    shorter = b"observation_date,M2NS\n2026-02-01,101.0\n"
    manifest = macro.refresh(root, as_of="2026-04-01", client=Client(responses(fredgraph=shorter)))
    assert "drops stored periods" in manifest["failures"]["fred:M2NS"]
    m2 = macro.read(root).query("series == 'M2NS'")
    assert m2.period.tolist() == ["2026-01-01", "2026-02-01"]


def test_finra_window_does_not_erase_older_months(decisions, tmp_path):
    root = tmp_path / "data"
    macro.refresh(root, as_of="2026-03-01", client=Client(responses()))
    later = FINRA.replace(b"Jan-26</td><td>1,100", b"Mar-26</td><td>1,300")
    macro.refresh(root, as_of="2026-04-01", client=Client(responses(**{"finra.org": later})))
    debit = macro.current_values(macro.read(root)).query("series == @DEBIT")
    assert debit.period.tolist() == ["2026-01-01", "2026-02-01", "2026-03-01"]


def test_fred_failure_uses_the_federal_reserve_fallback(decisions, tmp_path):
    root = tmp_path / "data"
    client = Client(responses(fredgraph=None, **{"fed.example": FED}))
    manifest = macro.refresh(root, as_of="2026-03-01", client=client)
    assert manifest["receipts"]["fred:M2NS"]["provider"] == "Federal Reserve Board"
    assert manifest["failures"] == {}


def test_one_failed_source_keeps_the_others(decisions, tmp_path):
    root = tmp_path / "data"
    manifest = macro.refresh(root, as_of="2026-03-01", client=Client(responses(**{"finra.org": None})))
    assert set(manifest["failures"]) == {"finra"}
    assert set(macro.read(root).source) == {"fred", "treasury"}
    assert macro.main(["verify"]) == 0


def test_treasury_fetches_from_the_latest_stored_year():
    current = macro.empty()
    assert macro.treasury_years(current, dict(start_year=2024), pd.Timestamp("2026-05-01").date()) == [
        2024,
        2025,
        2026,
    ]
    held = pd.DataFrame(
        [dict(source="treasury", series="BC_1YEAR", period="2025-12-31", value=1.0, realtime_start="x", realtime_end="")]
    )
    assert macro.treasury_years(held, dict(start_year=1990), pd.Timestamp("2026-05-01").date()) == [2025, 2026]
