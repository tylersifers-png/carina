# Carina

Carina handles data ingestion up to a staged parquet dataset. It covers source
capture with receipts, normalization, validation, quality flags, staging and
promotion. It stores no data and makes no decisions about data. The caller passes in
both the data and the decisions.

## Status

Version 0.1.0 is the first extraction. It covers the company dataset: daily market
bars from Yahoo Finance and selected SEC companyfacts. The code works and its
tests pass. It has not been released, has no GitHub Actions workflows, and needs a
Drive credential before unattended runs.

Planned next:

- an add-company command and a refresh path for SEC financial facts;
- an Actions workflow for unattended market refreshes, with Drive sync of the staged
  dataset through a narrowly scoped service account;
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
| `--data` | `CARINA_DATA` | `staged/` (the dataset), `runs/` (work in progress), `receipts/` |
| `--decisions` | `CARINA_DECISIONS` | the decision files listed in `carina/settings.py` |

```bash
carina --data DIR --decisions DIR refresh market   # stops for review with exit status 2
carina --data DIR verify
carina audit market --dataset DIR/staged
```

## Layout

| Module | Role |
| --- | --- |
| `dataset` | Staged dataset format: Hive-partitioned parquet, a manifest with per-file hashes and lineage, verified promotion |
| `refresh` | Resumable market refresh: plan, collect, inspect, build, verify, promote |
| `store`, `updates`, `remap` | Capture, build and verify; selective expansion; offline SEC remapping |
| `fundamentals`, `registry`, `profiles` | SEC fact normalization, company identity, accounting profiles |
| `quality`, `market_quality`, `audit` | Evidence-bound quality flags and audits |
| `sync`, `drive` | Fast-forward-only Drive sync by lineage; the narrow Drive client |

There is no license, and no one else is expected to use this code.
