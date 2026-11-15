"""The ten-company approved list that company tests run against.

Production reads the list from the caller's collection.json; tests install this
fixed set so they never depend on live state."""

INITIAL_ROWS = [
    {
        'symbol': 'AAPL',
        'issuer_id': 'sec:0000320193',
        'issuer_cik': 320193,
        'security_id': 'yahoo:AAPL',
        'name': 'Apple',
        'sector': 'Information Technology',
        'accounting_profile': 'operating_company',
        'requested_start': '1997-01-01',
        'identity_status': 'VENDOR_LISTING_CONTINUITY_NOT_CERTIFIED',
    },
    {
        'symbol': 'MSFT',
        'issuer_id': 'sec:0000789019',
        'issuer_cik': 789019,
        'security_id': 'yahoo:MSFT',
        'name': 'Microsoft',
        'sector': 'Information Technology',
        'accounting_profile': 'operating_company',
        'requested_start': '1997-01-01',
        'identity_status': 'VENDOR_LISTING_CONTINUITY_NOT_CERTIFIED',
    },
    {
        'symbol': 'JPM',
        'issuer_id': 'sec:0000019617',
        'issuer_cik': 19617,
        'security_id': 'yahoo:JPM',
        'name': 'JPMorgan Chase',
        'sector': 'Financials',
        'accounting_profile': 'bank',
        'requested_start': '1997-01-01',
        'identity_status': 'VENDOR_LISTING_CONTINUITY_NOT_CERTIFIED',
    },
    {
        'symbol': 'NVDA',
        'issuer_id': 'sec:0001045810',
        'issuer_cik': 1045810,
        'security_id': 'yahoo:NVDA',
        'name': 'NVIDIA',
        'sector': 'Information Technology',
        'accounting_profile': 'operating_company',
        'requested_start': '1999-01-22',
        'identity_status': 'VENDOR_LISTING_CONTINUITY_NOT_CERTIFIED',
    },
    {
        'symbol': 'AMZN',
        'issuer_id': 'sec:0001018724',
        'issuer_cik': 1018724,
        'security_id': 'yahoo:AMZN',
        'name': 'Amazon',
        'sector': 'Consumer Discretionary',
        'accounting_profile': 'operating_company',
        'requested_start': '1997-05-15',
        'identity_status': 'VENDOR_LISTING_CONTINUITY_NOT_CERTIFIED',
    },
    {
        'symbol': 'GOOGL',
        'issuer_id': 'sec:0001652044',
        'issuer_cik': 1652044,
        'security_id': 'yahoo:GOOGL',
        'name': 'Alphabet',
        'sector': 'Communication Services',
        'accounting_profile': 'operating_company',
        'requested_start': '2004-08-19',
        'identity_status': 'VENDOR_LISTING_CONTINUITY_NOT_CERTIFIED',
    },
    {
        'symbol': 'WMT',
        'issuer_id': 'sec:0000104169',
        'issuer_cik': 104169,
        'security_id': 'yahoo:WMT',
        'name': 'Walmart',
        'sector': 'Consumer Staples',
        'accounting_profile': 'operating_company',
        'requested_start': '1997-01-01',
        'identity_status': 'VENDOR_LISTING_CONTINUITY_NOT_CERTIFIED',
    },
    {
        'symbol': 'JNJ',
        'issuer_id': 'sec:0000200406',
        'issuer_cik': 200406,
        'security_id': 'yahoo:JNJ',
        'name': 'Johnson & Johnson',
        'sector': 'Health Care',
        'accounting_profile': 'operating_company',
        'requested_start': '1997-01-01',
        'identity_status': 'VENDOR_LISTING_CONTINUITY_NOT_CERTIFIED',
    },
    {
        'symbol': 'XOM',
        'issuer_id': 'sec:0000034088',
        'issuer_cik': 34088,
        'security_id': 'yahoo:XOM',
        'name': 'Exxon Mobil',
        'sector': 'Energy',
        'accounting_profile': 'operating_company',
        'requested_start': '1997-01-01',
        'identity_status': 'VENDOR_LISTING_CONTINUITY_NOT_CERTIFIED',
    },
    {
        'symbol': 'BRK-B',
        'issuer_id': 'sec:0001067983',
        'issuer_cik': 1067983,
        'security_id': 'yahoo:BRK-B',
        'name': 'Berkshire Hathaway',
        'sector': 'Financials',
        'accounting_profile': 'insurance_conglomerate',
        'requested_start': '1997-01-01',
        'identity_status': 'VENDOR_LISTING_CONTINUITY_NOT_CERTIFIED',
    },
]

INITIAL_COMPANIES = tuple(row["symbol"] for row in INITIAL_ROWS)


def synthetic_rows(*symbols):
    """Approved companies that are not in a test's base store (for expansion tests)."""
    return [
        {
            "symbol": symbol,
            "issuer_id": f"sec:{900000 + index:010d}",
            "issuer_cik": 900000 + index,
            "security_id": f"yahoo:{symbol}",
            "name": f"Synthetic {symbol}",
            "sector": "Information Technology",
            "accounting_profile": "operating_company",
            "requested_start": "1997-01-01",
            "identity_status": "VENDOR_LISTING_CONTINUITY_NOT_CERTIFIED",
            "selection": "Test fixture: synthetic expansion company",
        }
        for index, symbol in enumerate(symbols, 1)
    ]
