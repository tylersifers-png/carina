"""Macro series with their revision history: FRED, Federal Reserve data downloads,
FINRA margin statistics and the Treasury par yield curve.

Which series to collect is the caller's decision, in the ``macro`` section of
``collection.json``::

    "macro": {
      "fred": [{"series": "SERIES_ID",
                "fallback": {"url": "...", "column": "..."}}],
      "finra_margin": {"url": "..."},
      "treasury_par_yields": {"start_year": 1997}
    }

The dataset is ``staged/macro`` in the data directory: one table,
``observations.parquet``, and a manifest. Each row is one value of one series for
one period, valid from ``realtime_start`` until ``realtime_end`` (empty while it is
the current value). A refresh that finds a revised value closes the old row and
opens a new one, so every revision stays visible. Raw responses are not kept.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import io
import json
from pathlib import Path
import re
import shutil
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from carina import common as sec, dataset, settings

SCHEMA = "carina-macro-1.0"
STAGED = "staged/macro"
TABLE = "observations.parquet"
COLUMNS = ["source", "series", "period", "value", "realtime_start", "realtime_end"]
KEY = ["source", "series", "period"]
FINRA_URL = "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics"
TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
    "?data=daily_treasury_yield_curve&field_tdr_date_value={year}"
)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


@dataclass
class Capture:
    """One parsed response: long-form rows for one source, and its receipt.

    ``complete`` means the response holds a series' whole history, so a stored
    period missing from it is an error rather than a shorter publication window.
    """

    source: str
    rows: pd.DataFrame
    receipt: dict
    complete: bool = False
    series: list = field(default_factory=list)


# --- Parsers ---------------------------------------------------------------------


def _frame(source, records):
    frame = pd.DataFrame(records, columns=["series", "period", "value"])
    frame = frame.dropna()
    frame["value"] = frame["value"].astype("float64")
    if frame.empty:
        raise ValueError(f"{source} response produced no observations")
    if frame.duplicated(["series", "period"]).any():
        raise ValueError(f"{source} response repeats a period")
    if not np.isfinite(frame["value"]).all():
        raise ValueError(f"{source} response has non-finite values")
    frame.insert(0, "source", source)
    return frame.sort_values(["series", "period"]).reset_index(drop=True)


def _number(text):
    value = pd.to_numeric(str(text).replace(",", "").replace("$", "").strip(), errors="coerce")
    return None if pd.isna(value) else float(value)


def _day(text):
    stamp = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(stamp) else stamp.strftime("%Y-%m-%d")


def parse_fred_csv(payload: bytes, series: str) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload), dtype=str)
    if series not in frame.columns:
        raise ValueError(f"FRED response is missing {series}: {list(frame.columns)}")
    dates = frame.columns[0]
    return _frame(
        "fred",
        [(series, _day(d), _number(v)) for d, v in zip(frame[dates], frame[series])],
    )


def parse_fed_csv(payload: bytes, column: str, series: str) -> pd.DataFrame:
    """A Federal Reserve Data Download Program CSV (series in columns)."""
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
    header = next((i for i, r in enumerate(rows) if r and r[0].strip().rstrip(":") == "Unique Identifier"), None)
    if header is None:
        raise ValueError("Federal Reserve response has no Unique Identifier row")
    matches = [i for i, value in enumerate(rows[header]) if value.strip() == column]
    if len(matches) != 1:
        raise ValueError(f"Federal Reserve response has {len(matches)} {column} columns")
    start = next((i for i in range(header + 1, len(rows)) if rows[i] and rows[i][0].strip() == "Time Period"), None)
    if start is None:
        raise ValueError("Federal Reserve response has no Time Period row")
    target = matches[0]
    records = [
        (series, _day(r[0].strip()), _number(r[target]))
        for r in rows[start + 1 :]
        if r and len(r) > target and r[0].strip()
    ]
    return _frame("fred", records)


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower().replace("'", "")).strip("_")


def _finra_rows(table, month_column, month_format):
    records = []
    for column in table.columns:
        if column == month_column:
            continue
        for month, value in zip(table[month_column], table[column]):
            stamp = pd.to_datetime(str(month).strip(), format=month_format, errors="coerce")
            if pd.isna(stamp):
                continue
            records.append((_slug(column), stamp.strftime("%Y-%m-01"), _number(value)))
    return _frame("finra", records)


def parse_finra_html(payload: bytes) -> pd.DataFrame:
    """FINRA's margin statistics page: a rolling window of recent months."""
    for table in pd.read_html(io.StringIO(payload.decode("utf-8", errors="replace"))):
        columns = [str(c) for c in table.columns]
        if "Month/Year" in columns and any("Debit Balances" in c for c in columns):
            return _finra_rows(table, "Month/Year", "%b-%y")
    raise ValueError("FINRA page has no margin statistics table")


def parse_finra_excel(payload: bytes) -> pd.DataFrame:
    """FINRA's historical margin statistics workbook."""
    table = pd.read_excel(io.BytesIO(payload))
    if "Year-Month" not in table.columns:
        raise ValueError("FINRA workbook has no Year-Month column")
    return _finra_rows(table, "Year-Month", "%Y-%m")


def parse_treasury_xml(payload: bytes) -> pd.DataFrame:
    """One year of daily Treasury par yield curve rates, every maturity."""
    records = []
    for entry in ET.fromstring(payload).iter():
        if entry.tag.rsplit("}", 1)[-1] != "entry":
            continue
        fields = {c.tag.rsplit("}", 1)[-1]: c.text for c in entry.iter()}
        day = _day(fields.get("NEW_DATE"))
        if day is None:
            continue
        for name, text in fields.items():
            if name.startswith("BC_") and name != "BC_30YEARDISPLAY" and text not in (None, ""):
                records.append((name, day, _number(text)))
    return _frame("treasury", records)


# --- Fetching ----------------------------------------------------------------------


def session():
    import requests

    client = requests.Session()
    client.headers.update({"User-Agent": sec.UA, "Accept": "*/*"})
    return client


def _get(client, url, timeout=45):
    response = client.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def receipt(provider, url, payload, **extra):
    return dict(
        provider=provider,
        url=url,
        sha256=sec.digest(payload),
        bytes=len(payload),
        retrieved_at=sec.now(),
        **extra,
    )


def fetch_fred(client, entry):
    series = entry["series"]
    url = entry.get("url", FRED_URL.format(series=series))
    try:
        payload = _get(client, url, timeout=entry.get("timeout_seconds", 20))
        rows = parse_fred_csv(payload, series)
        return Capture("fred", rows, receipt("FRED", url, payload, series=series), complete=True)
    except Exception as primary:
        fallback = entry.get("fallback")
        if not fallback:
            raise
        payload = _get(client, fallback["url"])
        rows = parse_fed_csv(payload, fallback["column"], series)
        note = f"FRED failed ({type(primary).__name__}); used the Federal Reserve download"
        return Capture(
            "fred",
            rows,
            receipt("Federal Reserve Board", fallback["url"], payload, series=series, note=note),
            complete=True,
        )


def fetch_finra(client, entry):
    url = entry.get("url", FINRA_URL)
    payload = _get(client, url)
    return Capture("finra", parse_finra_html(payload), receipt("FINRA", url, payload))


def fetch_treasury(client, entry, year):
    url = TREASURY_URL.format(year=year)
    payload = _get(client, url)
    return Capture("treasury", parse_treasury_xml(payload), receipt("U.S. Treasury", url, payload, year=year))


def treasury_years(current, entry, as_of):
    held = current[current.source.eq("treasury")]
    first = int(held.period.max()[:4]) if not held.empty else int(entry.get("start_year", as_of.year))
    return list(range(first, as_of.year + 1))


def collect(current, plan, as_of, client=None):
    """Fetch every planned source. Returns captures and per-source failures."""
    client = client or session()
    jobs = [(f"fred:{e['series']}", lambda e=e: fetch_fred(client, e)) for e in plan.get("fred", [])]
    if "finra_margin" in plan:
        jobs.append(("finra", lambda: fetch_finra(client, plan["finra_margin"])))
    if "treasury_par_yields" in plan:
        entry = plan["treasury_par_yields"]
        jobs += [
            (f"treasury:{year}", lambda y=year: fetch_treasury(client, entry, y))
            for year in treasury_years(current, entry, as_of)
        ]
    captures, failures = [], {}
    for name, job in jobs:
        try:
            captures.append(job())
            print(f"MACRO: {name} ok", flush=True)
        except Exception as exc:  # one source failing does not discard the others
            failures[name] = f"{type(exc).__name__}: {exc}"
            print(f"MACRO: {name} FAILED {failures[name]}", flush=True)
    return captures, failures


# --- Revisions ---------------------------------------------------------------------


def empty():
    return pd.DataFrame({c: pd.Series(dtype="float64" if c == "value" else "object") for c in COLUMNS})


def revise(current: pd.DataFrame, capture: Capture, as_of: str):
    """Apply one capture as of a date. Returns the new table and a change count."""
    rows = capture.rows
    is_open = current.realtime_end.eq("")
    scope = current.source.eq(capture.source) & current.series.isin(set(rows.series))
    held = current[scope & is_open].set_index(KEY)["value"]
    incoming = rows.set_index(KEY)["value"]
    if capture.complete:
        dropped = held.index.difference(incoming.index)
        if len(dropped):
            sample = ", ".join("/".join(k[1:]) for k in dropped[:5])
            raise ValueError(f"{capture.source} response drops stored periods: {sample}")
    common = held.index.intersection(incoming.index)
    changed = common[held.loc[common].to_numpy() != incoming.loc[common].to_numpy()]
    added = incoming.index.difference(held.index)
    if not len(changed) and not len(added):
        return current, dict(added=0, revised=0)
    table = current.set_index(KEY)
    replace = table.index.isin(changed) & table.realtime_end.eq("").to_numpy()
    same_day = replace & table.realtime_start.eq(as_of).to_numpy()
    table.loc[replace & ~same_day, "realtime_end"] = as_of
    table = table[~same_day]
    new = incoming.loc[changed.append(added)].rename("value").reset_index()
    new["realtime_start"], new["realtime_end"] = as_of, ""
    result = pd.concat([table.reset_index(), new], ignore_index=True)
    result = result.sort_values([*KEY, "realtime_start"]).reset_index(drop=True)[COLUMNS]
    return result, dict(added=len(added), revised=len(changed))


def current_values(table: pd.DataFrame) -> pd.DataFrame:
    """The latest value of every series and period."""
    return table[table.realtime_end.eq("")].drop(columns=["realtime_end"]).reset_index(drop=True)


# --- Dataset -----------------------------------------------------------------------


def read(root=None):
    path = Path(root or settings.data()) / STAGED / TABLE
    return pd.read_parquet(path) if path.exists() else empty()


def write(root, table, *, receipts, as_of, failures=None, imported=None):
    """Write a candidate beside the staged dataset, then swap it in with lineage."""
    root = Path(root)
    staged = root / STAGED
    dataset.recover(staged)
    lineage, base = [], None
    if (staged / dataset.MANIFEST).exists():
        raw = (staged / dataset.MANIFEST).read_bytes()
        dataset.verify_files(staged, json.loads(raw))
        base = sec.digest(raw)
        lineage = [*json.loads(raw).get("lineage", []), base]
        receipts = {**json.loads(raw).get("receipts", {}), **receipts}
    if table.duplicated([*KEY, "realtime_start"]).any() or (
        table[table.realtime_end.eq("")].duplicated(KEY).any()
    ):
        raise ValueError("Macro table has overlapping values for one period")
    new = staged.with_name(staged.name + ".new")
    if new.exists():
        shutil.rmtree(new)
    new.mkdir(parents=True)
    dataset.write_parquet(new / TABLE, table.reset_index(drop=True)[COLUMNS])
    manifest = dict(
        schema=SCHEMA,
        status="VERIFIED",
        as_of=as_of,
        built_at=sec.now(),
        code_commit=sec.code_version(),
        base_manifest_sha256=base,
        lineage=lineage,
        series=sorted({f"{s}:{n}" for s, n in zip(table.source, table.series)}),
        rows=len(table),
        current_rows=int(table.realtime_end.eq("").sum()),
        receipts=receipts,
        failures=failures or {},
        files=dataset.inventory(new),
    )
    if imported:
        manifest["imported"] = imported
    (new / dataset.MANIFEST).write_bytes(sec.encoded(manifest))
    dataset.swap_in(new, staged)
    return manifest


def refresh(root=None, *, as_of=None, client=None):
    """Fetch every planned series, record revisions, and promote. Returns the manifest."""
    root = Path(root or settings.data()).resolve()
    as_of = as_of or datetime.now(timezone.utc).date().isoformat()
    plan = settings.collection().get("macro", {})
    if not plan:
        raise ValueError("collection.json has no macro section")
    table = read(root)
    captures, failures = collect(table, plan, date.fromisoformat(as_of), client)
    receipts, changes = {}, {}
    for capture in captures:
        try:
            table, count = revise(table, capture, as_of)
        except ValueError as exc:
            name = capture.source + ":" + ",".join(sorted(set(capture.rows.series)))
            failures[name] = str(exc)
            continue
        name = capture.source if capture.source != "fred" else "fred:" + capture.receipt["series"]
        if capture.source == "treasury":
            name += ":" + str(capture.receipt["year"])
        receipts[name] = dict(capture.receipt, latest_period=capture.rows.period.max())
        changes[name] = count
    manifest = write(root, table, receipts=receipts, as_of=as_of, failures=failures)
    print("MACRO_CHANGES=" + json.dumps(changes, sort_keys=True), flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="carina macro", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("refresh", help="fetch every planned series and record revisions")
    run.add_argument("--as-of", help="Date the values are recorded as of (default: today, UTC)")
    sub.add_parser("verify", help="check the staged macro dataset's file hashes")
    args = parser.parse_args(argv)
    root = settings.data().resolve()
    if args.action == "verify":
        staged = root / STAGED
        manifest = json.loads((staged / dataset.MANIFEST).read_bytes())
        dataset.verify_files(staged, manifest)
        print("Staged macro dataset checksums verified")
        return 0
    manifest = refresh(root, as_of=args.as_of)
    if manifest["failures"]:
        print("MACRO_FAILURES=" + json.dumps(manifest["failures"], sort_keys=True), flush=True)
        return 1
    return 0
