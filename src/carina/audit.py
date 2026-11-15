"""Audit a verified dataset's financial facts or market histories; no network or Drive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from carina import market_quality, quality, store


def financial(argv=None):
    parser = argparse.ArgumentParser(prog="carina audit financial", description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in [args.dataset] + ([args.previous] if args.previous else []):
        manifest = json.loads((path / "manifest.json").read_bytes())
        store.verify_local(path, manifest)
    report = quality.assess(args.dataset, previous_path=args.previous)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "quality_flags.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Company-data quality audit",
        "",
        f"Status: **{report['status']}**; open flags: **{report['open_flags']}**.",
        "",
        "A flag is a review request, not a confirmed source error. No data was corrected, refreshed or accepted by this audit.",
        "",
        "| Company | Rule | Measure | Basis | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for flag in report["flags"]:
        if flag["status"] == "OPEN":
            cells = [
                flag["symbol"],
                flag["rule"],
                flag["metric"],
                flag["basis"],
                json.dumps(flag["evidence"], sort_keys=True).replace("|", "/"),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    (args.output / "quality_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(quality.summary(report), indent=2))
    return 0


def market(argv=None):
    parser = argparse.ArgumentParser(prog="carina audit market", description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads((args.dataset / "manifest.json").read_bytes())
    store.verify_local(args.dataset, manifest)
    report = market_quality.assess(args.dataset)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(market_quality.summary(report)), flush=True)
    return 0
