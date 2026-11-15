"""Company identity, accounting rules, and benchmarks.

Listing identifiers are vendor-scoped, not point-in-time IDs. The approved company
list is data, read from the staged dataset's ``companies.parquet`` rather than kept
in code."""

from copy import deepcopy
import json
from carina import settings
from carina.profiles import metrics_for as pilot_metrics
from carina.fundamentals import METRICS

from collections.abc import Mapping

import pandas as pd

STORE_COMPANIES = "staged/companies.parquet"
IDENTITY_FIELDS = (
    "issuer_id",
    "issuer_cik",
    "security_id",
    "name",
    "sector",
    "accounting_profile",
    "requested_start",
    "identity_status",
    "selection",
)


class _Registry(Mapping):
    """Approved companies, read lazily from the current store's companies table.

    The data directory must hold a staged dataset. Tests install a small fixed set with ``use`` and clear it with ``reset``.
    """

    def __init__(self):
        self._rows = None

    def _load(self):
        if self._rows is None:
            path = settings.data() / STORE_COMPANIES
            if not path.is_file():
                raise FileNotFoundError(
                    f"Approved company list not found at {path}"
                )
            self.use(pd.read_parquet(path))
        return self._rows

    def use(self, rows):
        records = rows.to_dict("records") if hasattr(rows, "to_dict") else list(rows)
        self._rows = {
            r["symbol"]: {k: r.get(k, "") for k in ("symbol", *IDENTITY_FIELDS)}
            for r in records
        }

    def reset(self):
        self._rows = None

    def __getitem__(self, symbol):
        return self._load()[symbol]

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())


COMPANIES = _Registry()


BENCHMARKS = {
    "^GSPC": "S&P 500 price",
    "^SP500TR": "S&P 500 vendor gross total return",
    "RSP": "Invesco S&P 500 Equal Weight ETF (vendor-adjusted return proxy)",
}

# RSP began trading on 2003-05-01. It is an investable, net-of-fund-expenses
# equal-weight total-return proxy, not an official S&P equal-weight total-return
# index or a substitute for historic constituent membership.
BENCHMARK_STARTS = {
    "^GSPC": "1997-01-01",
    "^SP500TR": "1997-01-01",
    "RSP": "2003-05-01",
}


def quarantines_for(symbol):
    data = json.loads(settings.decision("company_fact_quarantines.json").read_bytes())
    if data["schema"] != "company-fact-quarantines-1.0":
        raise ValueError("Unsupported source-fact quarantine schema")
    return [r for r in data["records"] if r["symbol"] == symbol]


def company(symbol):
    if symbol not in COMPANIES:
        raise ValueError("Outside approved company registry")
    row = COMPANIES[symbol]
    return dict(
        symbol=symbol,
        issuer_id=row["issuer_id"],
        issuer_cik=int(row["issuer_cik"]),
        security_id=row["security_id"],
        name=row["name"],
        sector=row["sector"],
        accounting_profile=row["accounting_profile"],
        requested_start=row["requested_start"],
        identity_status=row["identity_status"],
        selection=row["selection"],
    )


def metrics_for(symbol):
    spec = company(symbol)
    if spec["accounting_profile"] == "bank":
        return pilot_metrics("JPM")
    result = deepcopy(METRICS)
    # Sales and broader revenue totals can coexist with different values. Preserve
    # their reported tag bases rather than treating every revenue tag as an alias.
    unit, kind, mappings = result["revenue"]
    result["revenue"] = (
        unit,
        kind,
        [(ns, tag, "reported_" + tag) for ns, tag, _ in mappings],
    )
    # Taxonomy transitions do not make economically different measures aliases.
    # Retain these reported bases separately; never splice, sum, or zero-fill.
    result["operating_cash_flow"][2].append(
        (
            "us-gaap",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            "continuing_operations",
        )
    )
    result["equity"][2].append(
        (
            "us-gaap",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "including_noncontrolling_interest",
        )
    )
    result["dividends_paid"][2].append(
        ("us-gaap", "PaymentsOfOrdinaryDividends", "ordinary_dividends")
    )
    if symbol in {"AMZN", "NVDA"}:
        result["capex_productive_assets"] = (
            "USD",
            "duration",
            [
                (
                    "us-gaap",
                    "PaymentsToAcquireProductiveAssets",
                    "ppe_software_and_intangibles",
                )
            ],
        )
    if symbol == "XOM":
        result["long_term_debt_including_leases"] = (
            "USD",
            "instant",
            [
                (
                    "us-gaap",
                    "LongTermDebtAndCapitalLeaseObligations",
                    "debt_and_leases_reported_scope",
                )
            ],
        )
    if symbol == "BRK-B":
        result["cash_including_restricted"] = (
            "USD",
            "instant",
            [
                (
                    "us-gaap",
                    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                    "cash_equivalents_and_restricted_cash",
                )
            ],
        )
    if symbol == "BRK-B":
        # Insurance/conglomerate earnings do not imply industrial FCF comparability.
        for key in (
            "operating_income",
            "operating_cash_flow",
            "capex_ppe",
            "current_long_term_debt",
            "noncurrent_long_term_debt",
        ):
            del result[key]
    return result


def applicability(symbol):
    selected = metrics_for(symbol)
    return [
        dict(
            symbol=symbol,
            metric=k,
            status="SELECTED" if k in selected else "NOT_SELECTED_PROFILE",
            reason=(
                "Reported basis retained; availability measured"
                if k in selected
                else "Accounting-profile comparability deferred; not a zero"
            ),
        )
        for k in sorted(set(METRICS) | set(selected))
    ]
