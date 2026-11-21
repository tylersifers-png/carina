# Carina

Carina handles data ingestion up to staged parquet datasets. It covers source
capture, normalization, validation, quality flags, staging and promotion. It stores
no data and makes no decisions about data. The caller passes in both the data and
the decisions.

## Status

Version 0.3.0 maintains two datasets:

- **company**: daily market bars from Yahoo Finance for selected companies and
  benchmarks, and selected SEC companyfacts;
- **macro**: FRED series (with a Federal Reserve download fallback), FINRA margin
  statistics and the Treasury par yield curve, with every revision kept.

Version 0.3.0 adds a resumable SEC financial refresh intended for a machine
that can reach `data.sec.gov`. The workflow in `.github/workflows/refresh.yml` refreshes
the macro series every weekday evening and the market bars on Friday evening (04:00
UTC the next day), then pushes the datasets to Drive. Hosted workflows do not
refresh SEC financial facts.

Planned next:

- partitioning for intraday (minute) data;
- adapters for further public, proprietary and inbox sources.

## Use

```bash
make setup          # Python 3.12 virtual environment with uv
make test
carina --help
```

Every command reads two directories, set by flags or environment variables:

| Flag | Variable | Holds |
| --- | --- | --- |
| `--data` | `CARINA_DATA` | `staged/company/`, `staged/macro/`, `runs/` (work in progress, deleted on promotion), `receipts/` |
| `--decisions` | `CARINA_DECISIONS` | `collection.json`, `thresholds.json`, `reviews.json` |

The decision files:

- `collection.json`: what to collect. Companies (symbol, CIK, name, sector,
  accounting profile, first requested date), benchmarks, and the `macro` plan.
- `thresholds.json`: limits for the `financial` and `market` checks.
- `reviews.json`: `fact_quarantines`, `financial_acknowledgements`,
  `market_acknowledgements` and `market_events`.

```bash
carina --data DIR --decisions DIR refresh market   # stops for review with exit status 2
carina --data DIR --decisions DIR refresh sec      # scans submissions, fetches facts only when needed
carina --data DIR --decisions DIR macro refresh
carina --data DIR verify
carina sync push company --drive-root-id ID        # fast-forward only
carina export staged/macro --out DIR --limit 1000  # parquet to CSV for scratch analysis
carina export staged/company/daily_bars --out DIR --query "SELECT ... FROM daily_bars WHERE ..."
```

`carina export` writes each parquet file or dataset folder as `NAME.csv` plus
`NAME.schema.json` (column types, row counts, sizes and SHA-256 of each part). Dates
and timestamps are ISO 8601, decimals and integers are exact, nested values are JSON
text. Options: `--columns`, `--limit` (first N rows), `--max-mb` (split into parts)
and `--overwrite`. A folder must hold files of one schema; export mixed folders as
separate files or subfolders. The sources are never changed.

`--query SQL` (or `--query-file FILE`) exports the result of one SELECT instead. The
sources are DuckDB views named after them (`daily_bars`; a single source is also
`data`), and the result is written as `query.csv` unless `--name` says otherwise.
Rows come back in no fixed order unless the query says `ORDER BY`. The query runs in
memory and can read only the source files; it cannot write files, attach databases,
install extensions or change settings. The schema file records the query text.

Drive access uses `GOOGLE_SERVICE_ACCOUNT_JSON` when set, otherwise
`GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` and `GOOGLE_REFRESH_TOKEN`. A service
account cannot create files in a personal Google Drive, only update existing ones.
SEC refreshes require `SEC_CONTACT_EMAIL`; it is included in the declared
User-Agent for fair-access identification and is not retained in datasets or
receipts.

## Layout

| Module | Role |
| --- | --- |
| `dataset` | Company dataset format: Hive-partitioned parquet, a manifest with per-file hashes and lineage, verified promotion |
| `refresh`, `sec_refresh` | Resumable market and SEC refreshes: plan, collect, inspect, build, verify, promote |
| `store`, `updates` | Capture, build and verify; selective expansion |
| `fundamentals`, `registry`, `profiles` | SEC fact normalization, company identity, accounting profiles |
| `quality`, `market_quality`, `audit` | Evidence-bound quality flags and audits |
| `macro` | Macro series and their revisions |
| `sync`, `drive` | Fast-forward-only Drive sync by lineage; the narrow Drive client |
| `export` | Parquet files or datasets as exact CSV with a schema file |
| `settings` | The data and decisions directories, and the decision files |

There is no license, and no one else is expected to use this code.
