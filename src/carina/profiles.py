"""Reference issuers whose accounting profiles define the metric sets."""

from copy import deepcopy
from carina.fundamentals import METRICS

COMPANIES = {
    "MSFT": {
        "cik": 789019,
        "name": "Microsoft",
        "profile": "operating_company",
        "rationale": "Operating-company reference; June fiscal year",
    },
    "JPM": {
        "cik": 19617,
        "name": "JPMorgan Chase",
        "profile": "bank",
        "rationale": "Bank financial-statement reference",
    },
}


def metrics_for(symbol):
    if symbol not in COMPANIES:
        raise ValueError("Only MSFT and JPM have reference profiles")
    if symbol == "MSFT":
        return deepcopy(METRICS)
    common = {
        k: deepcopy(METRICS[k])
        for k in (
            "net_income",
            "assets",
            "liabilities",
            "equity",
            "shares_outstanding",
            "diluted_eps",
            "dividends_paid",
            "repurchases",
        )
    }
    bank = {
        "revenue_net_of_interest_expense": (
            "USD",
            "duration",
            [
                (
                    "us-gaap",
                    "RevenuesNetOfInterestExpense",
                    "bank_net_revenue_not_industrial_sales",
                )
            ],
        ),
        "net_interest_income": (
            "USD",
            "duration",
            [("us-gaap", "InterestIncomeExpenseNet", "net_of_interest_expense")],
        ),
        "noninterest_income": (
            "USD",
            "duration",
            [("us-gaap", "NoninterestIncome", "reported")],
        ),
        "credit_loss_provision": (
            "USD",
            "duration",
            [
                (
                    "us-gaap",
                    "ProvisionForLoanLeaseAndOtherLosses",
                    "loans_leases_and_other_credit_losses",
                )
            ],
        ),
        "deposits": (
            "USD",
            "instant",
            [("us-gaap", "Deposits", "customer_deposits_not_industrial_debt")],
        ),
        "cash_due_from_banks": (
            "USD",
            "instant",
            [
                (
                    "us-gaap",
                    "CashAndDueFromBanks",
                    "cash_and_due_from_banks_not_total_liquidity",
                )
            ],
        ),
        "financing_receivables_net": (
            "USD",
            "instant",
            [
                (
                    "us-gaap",
                    "FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss",
                    "excludes_accrued_interest_after_allowance",
                )
            ],
        ),
    }
    return {**common, **bank}
