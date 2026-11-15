"""Company identity, accounting rules, and benchmarks.

Listing identifiers are vendor-scoped, not point-in-time IDs. Which companies and
benchmarks to collect is a decision, read from the caller's ``collection.json``
rather than kept in code."""

from copy import deepcopy
from carina import settings
from carina.profiles import metrics_for as pilot_metrics
from carina.fundamentals import METRICS

from collections.abc import Mapping

IDENTITY_FIELDS = (
    "issuer_id",
    "issuer_cik",
    "security_id",
    "name",
    "sector",
    "accounting_profile",
    "requested_start",
    "identity_status",
)
COMPANY_FIELDS = ("symbol", "issuer_cik", "name", "sector", "accounting_profile", "requested_start")
IDENTITY_STATUS = "VENDOR_LISTING_CONTINUITY_NOT_CERTIFIED"


def identity(row):
    """A company's full identity from its ``collection.json`` entry."""
    missing = set(COMPANY_FIELDS) - set(row)
    if missing:
        raise ValueError(f"Company {row.get('symbol')!r} is missing {sorted(missing)}")
    cik = int(row["issuer_cik"])
    return dict(
        symbol=row["symbol"],
        issuer_id=f"sec:{cik:010d}",
        issuer_cik=cik,
        security_id="yahoo:" + row["symbol"],
        name=row["name"],
        sector=row["sector"],
        accounting_profile=row["accounting_profile"],
        requested_start=row["requested_start"],
        identity_status=IDENTITY_STATUS,
    )


class _Lazy(Mapping):
    """A mapping read from ``collection.json`` on first use.

    Tests install a fixed set with ``use`` and clear it with ``reset``.
    """

    def __init__(self, build):
        self._build = build
        self._rows = None

    def _load(self):
        if self._rows is None:
            self._rows = self._build(settings.collection())
        return self._rows

    def use(self, rows):
        self._rows = self._build(None, rows)

    def reset(self):
        self._rows = None

    def __getitem__(self, key):
        return self._load()[key]

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())


def _unique(rows, what):
    symbols = [r["symbol"] for r in rows]
    if len(symbols) != len(set(symbols)):
        raise ValueError(f"Duplicate {what} in collection.json")
    return rows


def _companies(collection, rows=None):
    rows = collection["companies"] if rows is None else rows
    rows = rows.to_dict("records") if hasattr(rows, "to_dict") else list(rows)
    return {r["symbol"]: identity(r) for r in _unique(rows, "companies")}


def _benchmarks(collection, rows=None):
    rows = collection.get("benchmarks", []) if rows is None else list(rows)
    return {r["symbol"]: dict(r) for r in _unique(rows, "benchmarks")}


COMPANIES = _Lazy(_companies)
_BENCHMARK_ROWS = _Lazy(_benchmarks)


class _Field(Mapping):
    """One field of each benchmark, keyed by symbol."""

    def __init__(self, field):
        self._field = field

    def __getitem__(self, symbol):
        return _BENCHMARK_ROWS[symbol][self._field]

    def __iter__(self):
        return iter(_BENCHMARK_ROWS)

    def __len__(self):
        return len(_BENCHMARK_ROWS)


# Benchmark symbol -> description, and symbol -> first requested session.
BENCHMARKS = _Field("name")
BENCHMARK_STARTS = _Field("requested_start")


def use_benchmarks(rows):
    _BENCHMARK_ROWS.use(rows)


def reset():
    COMPANIES.reset()
    _BENCHMARK_ROWS.reset()


def quarantines_for(symbol):
    return [r for r in settings.reviews("fact_quarantines") if r["symbol"] == symbol]


def company(symbol):
    if symbol not in COMPANIES:
        raise ValueError("Not in collection.json: " + symbol)
    return dict(COMPANIES[symbol])


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
