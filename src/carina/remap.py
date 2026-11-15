"""Explicit offline SEC remapping of an existing panel; never requests price data."""

from __future__ import annotations

import argparse
import bz2
from copy import deepcopy
import json
from datetime import datetime, timezone
from pathlib import Path
import re

from carina import store, updates, common as sec
from carina import quality, fundamentals as norm
from carina.registry import applicability, metrics_for
from carina.registry import company, quarantines_for

SCHEMA = "company-financial-remap-1.0"


def validate(content):
    required = {
        "schema",
        "base_manifest_sha256",
        "collected_at",
        "companies",
        "review_evidence",
    }
    if (
        set(content) not in (required, required | {"base_v2_manifest_sha256"})
        or content["schema"] != SCHEMA
    ):
        raise ValueError("Invalid financial-remap envelope")
    sec.identity(content["base_manifest_sha256"])
    if content.get("base_v2_manifest_sha256") is not None:
        sec.identity(content["base_v2_manifest_sha256"])
    store.validate(
        {k: content[k] for k in ("collected_at", "companies")}
        | {"schema": store.SCHEMA}
    )
    if not isinstance(content["review_evidence"], dict):
        raise ValueError("Invalid remap review evidence")
    for original in content["review_evidence"].get("originals", []):
        symbol = original["symbol"]
        if symbol not in content["companies"]:
            raise ValueError("Original outside remap panel")
        filing, receipt, projection = [
            original[k] for k in ("filing", "receipt", "projection")
        ]
        accession = filing["accessionNumber"]
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession) or not re.fullmatch(
            r"[A-Za-z0-9_-]+\.htm[l]?", filing["primaryDocument"]
        ):
            raise ValueError("Invalid original accession")
        prefix = f"https://www.sec.gov/Archives/edgar/data/{company(symbol)['issuer_cik']}/{accession.replace('-', '')}/"
        if (
            receipt["source_url"] != prefix + filing["primaryDocument"]
            or receipt["http_status"] != 200
        ):
            raise ValueError("Original source URL/status mismatch")
        sec.identity(receipt["sha256"])
        requested, retrieved = [
            datetime.fromisoformat(receipt[k]) for k in ("requested_at", "retrieved_at")
        ]
        if (
            requested.tzinfo is None
            or retrieved.tzinfo is None
            or not requested <= retrieved <= datetime.now(timezone.utc)
        ):
            raise ValueError("Invalid original receipt times")
        if sec.digest(sec.encoded(projection)) != original["projection_sha256"]:
            raise ValueError("Original projection checksum mismatch")
        meta = projection["metadata"]
        if {int(v) for v in meta.get("EntityCentralIndexKey", [])} != {
            company(symbol)["issuer_cik"]
        } or meta.get("DocumentType") != [filing["form"]]:
            raise ValueError("Original issuer/form mismatch")
        stamps = meta.get("DocumentPeriodEndDate", [])
        if len(stamps) != 1:
            raise ValueError("Missing original report end")
        try:
            end = datetime.fromisoformat(stamps[0]).date().isoformat()
        except ValueError:
            end = datetime.strptime(stamps[0], "%B %d, %Y").date().isoformat()
        if end != filing["reportDate"]:
            raise ValueError("Original report date mismatch")
        indexed = [
            r
            for r in content["companies"][symbol]["filings"]
            if r["accessionNumber"] == accession
        ]
        if len(indexed) != 1 or any(
            indexed[0][k] != filing[k]
            for k in ("form", "filingDate", "reportDate", "primaryDocument")
        ):
            raise ValueError("Original differs from retained filing index")
    return content


def prepare(dataset, raw_capture, output, evidence):
    """Reproject retained *hash-matching* endpoint bytes using the current mapping."""
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    store.verify_local(dataset, manifest)
    saved = json.loads(bz2.decompress((dataset / "sources.json.bz2").read_bytes()))
    companies = deepcopy(saved["sec_projection"]["companies"])
    if set(companies) != set(manifest["companies"]):
        raise ValueError("Remapping requires exactly the saved company panel")
    for symbol, item in companies.items():
        raw = (raw_capture / (symbol + "_facts.json")).read_bytes()
        if sec.digest(raw) != item["receipts"]["facts"]["sha256"]:
            raise ValueError(
                "Retained raw endpoint differs from saved source: " + symbol
            )
        full = json.loads(raw)
        wanted = {
            (ns, tag)
            for _, _, mappings in metrics_for(symbol).values()
            for ns, tag, _ in mappings
        }
        item["facts"] = {k: v for k, v in full.items() if k != "facts"}
        item["facts"]["facts"] = {
            ns: {tag: values for tag, values in tags.items() if (ns, tag) in wanted}
            for ns, tags in full["facts"].items()
            if any((ns, tag) in wanted for tag in tags)
        }
        item["projection_sha256"] = sec.digest(sec.encoded(item["facts"]))
    content = dict(
        schema=SCHEMA,
        base_manifest_sha256=sec.digest((dataset / "manifest.json").read_bytes()),
        collected_at=manifest["collected_at"],
        companies=companies,
        review_evidence=evidence,
    )
    v2_manifest = dataset.parent / "v2/current/manifest.json"
    if v2_manifest.exists():
        content["base_v2_manifest_sha256"] = sec.digest(v2_manifest.read_bytes())
    output.mkdir(parents=True, exist_ok=True)
    sec.prepare_capture(output, content, validator=validate, **store.BOUNDS)
    return json.loads((output / "handoff.json").read_bytes())


def build(root, bundle, content):
    """Rebuild financial tables only, with a checked baseline and retained evidence."""
    import pandas as pd

    validate(content)
    current = updates.read_current(root)
    if (
        current is None
        or sec.digest((current[0] / "manifest.json").read_bytes())
        != content["base_manifest_sha256"]
    ):
        raise ValueError(
            "Remap baseline changed; prepare again from verified current data"
        )
    base, old = current
    if set(content["companies"]) != set(old["companies"]):
        raise ValueError("Remapping cannot expand or truncate the saved company panel")
    saved_sources = json.loads(bz2.decompress((base / "sources.json.bz2").read_bytes()))
    for symbol, item in content["companies"].items():
        before = saved_sources["sec_projection"]["companies"][symbol]
        if any(
            item[k] != before[k] for k in ("receipts", "filings", "issuer_metadata")
        ):
            raise ValueError(
                "Offline remap cannot change retained endpoint/index identity"
            )
        for ns, tags in before["facts"]["facts"].items():
            for tag, values in tags.items():
                if item["facts"]["facts"].get(ns, {}).get(tag) != values:
                    raise ValueError(
                        "Offline remap cannot alter/remove retained raw tag arrays"
                    )
    output = store.run_path(root, bundle) / "built"
    output.mkdir(parents=True, exist_ok=True)
    # Byte-for-byte preservation also avoids rewriting financial-independent tables
    # with a different Parquet library version. No Yahoo/network adapter exists here.
    for name in store.FILES:
        store.replace_generated(output / name, (base / name).read_bytes())
    coverage = json.loads((base / "coverage.json").read_bytes())
    sources = json.loads(bz2.decompress((base / "sources.json.bz2").read_bytes()))
    tables = dict(fundamentals=[], financial_latest=[], field_applicability=[])
    for symbol, item in content["companies"].items():
        canonical, issues = updates.normalize_company(symbol, deepcopy(item))
        for original in content["review_evidence"].get("originals", []):
            if original["symbol"] == symbol:
                canonical = norm.supplement_original(
                    canonical,
                    original,
                    cik=company(symbol)["issuer_cik"],
                    metrics=metrics_for(symbol),
                )
        # Original supplements cannot reintroduce a known quarantined version.
        canonical, issues = norm.quarantine_facts(
            canonical, issues, quarantines_for(symbol)
        )
        for row in canonical:
            row.update(symbol=symbol, issuer_id=company(symbol)["issuer_id"])
        latest = norm.latest_view(canonical)
        tables["fundamentals"].extend(canonical)
        tables["financial_latest"].extend(latest)
        tables["field_applicability"].extend(applicability(symbol))
        coverage[symbol]["financials"] = updates.financial_coverage(
            symbol, canonical, latest, issues
        )
    for name, rows in tables.items():
        frame = pd.DataFrame(rows)
        if name != "field_applicability":
            keys = ["symbol", "metric", "basis", "unit", "start", "end", "period_kind"]
            if name == "fundamentals":
                keys.append("accession")
            frame = frame.sort_values(keys).reset_index(drop=True)
        frame.to_parquet(output / (name + ".parquet"), index=False, compression="zstd")
    sources["sec_projection"]["companies"] = content["companies"]
    sources["mapping_review"] = content["review_evidence"]
    store.replace_generated(output / "coverage.json", sec.encoded(coverage))
    store.replace_generated(
        output / "sources.json.bz2", bz2.compress(sec.encoded(sources))
    )
    manifest = dict(old)
    for key in ("quality", "readback", "verified_at"):
        manifest.pop(key, None)
    manifest.update(
        status="BUILT_NOT_PUBLISHED",
        bundle=bundle,
        built_at=sec.now(),
        code_commit=sec.code_version(),
        collection_mode="OFFLINE_FINANCIAL_REMAP_NO_PROVIDER_REQUESTS",
        limits=[
            x
            for x in old["limits"]
            if not x.startswith("No new original-filing certification")
        ]
        + [
            "Selected original annual/quarterly comparisons; not certification of every historical filing"
        ],
        base_manifest_sha256=content["base_manifest_sha256"],
        base_v2_manifest_sha256=content.get("base_v2_manifest_sha256"),
        processing_libraries={
            p: updates.version(p)
            for p in ("pandas", "pyarrow", "lxml", "exchange-calendars")
        },
        files={
            n: dict(
                sha256=sec.digest((output / n).read_bytes()),
                bytes=(output / n).stat().st_size,
            )
            for n in sorted(store.FILES)
        },
    )
    store.replace_generated(output / "manifest.json", sec.encoded(manifest))
    store.verify_local(output, manifest)
    manifest["quality"] = quality.assess(output, previous_path=base)
    store.replace_generated(output / "manifest.json", sec.encoded(manifest))
    store.replace_generated(
        output.parent / "update_result.json",
        sec.encoded(
            dict(
                status="BUILT",
                base_manifest_sha256=content["base_manifest_sha256"],
                base_v2_manifest_sha256=content.get("base_v2_manifest_sha256"),
            )
        ),
    )
    if manifest["quality"]["status"] == "BLOCKED":
        raise ValueError("Financial-remap candidate blocked by integrity flags")
    print(
        "COMPANY_DATA_QUALITY=" + json.dumps(quality.summary(manifest["quality"])),
        flush=True,
    )
    print(
        "Remap provider requests: 0; existing daily/identity/filing files preserved byte-for-byte",
        flush=True,
    )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(prog="carina remap", description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare(
                args.dataset,
                args.raw_capture,
                args.output,
                json.loads(args.evidence.read_bytes()),
            )
        )
    )

