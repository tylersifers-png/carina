# Carina

Carina handles data ingestion up to staged parquet datasets. It covers source
capture, normalization, validation, quality flags, staging and promotion. It stores
no data and makes no decisions about data. The caller passes in both the data and
the decisions.

## Status

Version 0.2.0 maintains two datasets:

- **company**: daily market bars from Yahoo Finance for selected companies and
  benchmarks, and selected SEC companyfacts;
- **macro**: FRED series (with a Federal Reserve download fallback), FINRA margin
  statistics and the Treasury par yield curve, with every revision kept.

Released as `v0.2.0`. The workflow in `.github/workflows/refresh.yml` runs at
23:30 UTC on weekdays: it refreshes the macro series and market bars and pushes
both datasets to Drive. It does not refresh SEC financial facts.

Planned next:

- an add-company command and a refresh path for SEC financial facts (SEC rejects
  requests from GitHub-hosted runners, so these run elsewhere);
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
carina --data DIR --decisions DIR macro refresh
carina --data DIR verify
carina sync push company --drive-root-id ID        # fast-forward only
```

Drive access uses `GOOGLE_SERVICE_ACCOUNT_JSON` when set, otherwise
`GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` and `GOOGLE_REFRESH_TOKEN`. A service
account cannot create files in a personal Google Drive, only update existing ones.

## Layout

| Module | Role |
| --- | --- |
| `dataset` | Company dataset format: Hive-partitioned parquet, a manifest with per-file hashes and lineage, verified promotion |
| `refresh` | Resumable market refresh: plan, collect, inspect, build, verify, promote |
| `store`, `updates` | Capture, build and verify; selective expansion |
| `fundamentals`, `registry`, `profiles` | SEC fact normalization, company identity, accounting profiles |
| `quality`, `market_quality`, `audit` | Evidence-bound quality flags and audits |
| `macro` | Macro series and their revisions |
| `sync`, `drive` | Fast-forward-only Drive sync by lineage; the narrow Drive client |
| `settings` | The data and decisions directories, and the decision files |

There is no license, and no one else is expected to use this code.
