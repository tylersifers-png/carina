"""Conservative Apple financial-fact normalization with retained reporting vintages."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re

from lxml import etree

# Tags are alternatives, never amounts to sum. Economically different bases stay separate.
METRICS = {
    "revenue": (
        "USD",
        "duration",
        [
            ("us-gaap", t, "net_revenue")
            for t in (
                "SalesRevenueNet",
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
            )
        ],
    ),
    "operating_income": (
        "USD",
        "duration",
        [("us-gaap", "OperatingIncomeLoss", "reported")],
    ),
    "net_income": ("USD", "duration", [("us-gaap", "NetIncomeLoss", "reported")]),
    "operating_cash_flow": (
        "USD",
        "duration",
        [("us-gaap", "NetCashProvidedByUsedInOperatingActivities", "reported")],
    ),
    "capex_ppe": (
        "USD",
        "duration",
        [("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment", "ppe_only")],
    ),
    "assets": ("USD", "instant", [("us-gaap", "Assets", "reported")]),
    "liabilities": ("USD", "instant", [("us-gaap", "Liabilities", "reported")]),
    "equity": ("USD", "instant", [("us-gaap", "StockholdersEquity", "reported")]),
    "cash": (
        "USD",
        "instant",
        [
            (
                "us-gaap",
                "CashAndCashEquivalentsAtCarryingValue",
                "cash_equivalents_not_securities",
            )
        ],
    ),
    "current_long_term_debt": (
        "USD",
        "instant",
        [("us-gaap", "LongTermDebtCurrent", "current_portion_long_term_only")],
    ),
    "noncurrent_long_term_debt": (
        "USD",
        "instant",
        [("us-gaap", "LongTermDebtNoncurrent", "long_term_only")],
    ),
    "shares_outstanding": (
        "shares",
        "instant",
        [
            ("us-gaap", "CommonStockSharesOutstanding", "balance_sheet"),
            ("dei", "EntityCommonStockSharesOutstanding", "cover_page"),
        ],
    ),
    "diluted_eps": (
        "USD/shares",
        "duration",
        [("us-gaap", "EarningsPerShareDiluted", "reported_split_basis")],
    ),
    "dividends_paid": (
        "USD",
        "duration",
        [
            ("us-gaap", "PaymentsOfDividendsCommonStock", "common_stock"),
            ("us-gaap", "PaymentsOfDividends", "all_equity"),
        ],
    ),
    "repurchases": (
        "USD",
        "duration",
        [("us-gaap", "PaymentsForRepurchaseOfCommonStock", "common_stock")],
    ),
}


def number(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Non-finite financial value")
    return result


def numeric_text(value):
    return format(number(value), "f")


def filing_rows(data):
    required = {
        "accessionNumber",
        "form",
        "filingDate",
        "reportDate",
        "acceptanceDateTime",
    }
    if not required <= set(data) or not isinstance(data["accessionNumber"], list):
        raise ValueError("Missing filing metadata")
    n = len(data["accessionNumber"])
    if any(not isinstance(v, list) or len(v) != n for v in data.values()):
        raise ValueError("Inconsistent filing arrays")
    rows = [dict(zip(data, vals)) for vals in zip(*data.values())]
    if len({r["accessionNumber"] for r in rows}) != n:
        raise ValueError("Duplicate filing accession within source")
    for row in rows:
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", row["accessionNumber"]):
            raise ValueError("Invalid filing accession")
        date.fromisoformat(row["filingDate"])
        stamp = datetime.fromisoformat(row["acceptanceDateTime"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Filing acceptance timestamp lacks timezone")
    return rows


def period(start, end, kind):
    finish = date.fromisoformat(end)
    if not start:
        return "INSTANT" if kind == "instant" else "INVALID_MISSING_START"
    days = (finish - date.fromisoformat(start)).days + 1
    if kind == "instant" or days <= 0:
        return "INVALID_PERIOD"
    for label, lo, hi in (
        ("QUARTER", 70, 110),
        ("YTD_6M", 150, 210),
        ("YTD_9M", 230, 300),
        ("ANNUAL", 330, 400),
    ):
        if lo <= days <= hi:
            return label
    return "OTHER_DURATION"


def normalize(facts, filings, receipt, *, metrics=None):
    metrics = METRICS if metrics is None else metrics
    index = {row["accessionNumber"]: row for row in filings}
    reported, issues = [], []
    for metric, (expected_unit, kind, mappings) in metrics.items():
        for namespace, tag, basis in mappings:
            units = (
                facts.get("facts", {}).get(namespace, {}).get(tag, {}).get("units", {})
            )
            for unit, values in units.items():
                for i, raw in enumerate(values):
                    fact_id = hashlib.sha256(
                        json.dumps(
                            [receipt["sha256"], namespace, tag, unit, i],
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest()
                    try:
                        value = numeric_text(raw["val"])
                        classification = period(raw.get("start", ""), raw["end"], kind)
                        date.fromisoformat(raw["filed"])
                    except (KeyError, ValueError, InvalidOperation):
                        issues.append(
                            {
                                "metric": metric,
                                "fact_id": fact_id,
                                "reason": "INVALID_VALUE_OR_DATES",
                            }
                        )
                        continue
                    if raw.get("form") not in {"10-K", "10-K/A", "10-Q", "10-Q/A"}:
                        issues.append(
                            {
                                "metric": metric,
                                "fact_id": fact_id,
                                "reason": "UNSUPPORTED_FORM",
                            }
                        )
                        continue
                    if unit != expected_unit or classification.startswith("INVALID"):
                        issues.append(
                            {
                                "metric": metric,
                                "fact_id": fact_id,
                                "reason": "UNIT_OR_PERIOD_MISMATCH",
                            }
                        )
                        continue
                    filing = index.get(raw["accn"], {})
                    acceptance = filing.get("acceptanceDateTime", "")
                    if filing and (
                        filing["form"] != raw["form"]
                        or filing["filingDate"] != raw["filed"]
                    ):
                        issues.append(
                            {
                                "metric": metric,
                                "fact_id": fact_id,
                                "reason": "FILING_METADATA_MISMATCH",
                            }
                        )
                        continue
                    reported.append(
                        {
                            "fact_id": fact_id,
                            "source_fact_index": i,
                            "metric": metric,
                            "basis": basis,
                            "namespace": namespace,
                            "tag": tag,
                            "unit": unit,
                            "value": value,
                            "start": raw.get("start", ""),
                            "end": raw["end"],
                            "period_kind": classification,
                            "accession": raw["accn"],
                            "form": raw["form"],
                            "filed": raw["filed"],
                            "filing_accepted_at": acceptance,
                            "availability_status": (
                                "FILING_TIMESTAMP_LINKED_CURRENT_VINTAGE"
                                if acceptance
                                else "FILING_DATE_ONLY_CURRENT_VINTAGE"
                            ),
                            "captured_at": receipt["retrieved_at"],
                            "source_sha256": receipt["sha256"],
                            "reported_fiscal_year": raw.get("fy", ""),
                            "reported_fiscal_period": raw.get("fp", ""),
                            "frame": raw.get("frame", ""),
                        }
                    )
    groups = defaultdict(list)
    for row in reported:
        key = tuple(
            row[k]
            for k in (
                "metric",
                "basis",
                "unit",
                "start",
                "end",
                "period_kind",
                "accession",
            )
        )
        groups[key].append(row)
    canonical = []
    for key, rows in sorted(groups.items()):
        values = sorted({number(r["value"]) for r in rows})
        row = {
            k: rows[0][k]
            for k in (
                "metric",
                "basis",
                "unit",
                "start",
                "end",
                "period_kind",
                "accession",
                "form",
                "filed",
                "filing_accepted_at",
                "availability_status",
                "captured_at",
                "source_sha256",
            )
        }
        row.update(
            value=numeric_text(values[0]) if len(values) == 1 else "",
            status="REPORTED" if len(values) == 1 else "CONFLICT",
            fact_ids=";".join(sorted(r["fact_id"] for r in rows)),
            candidate_values=";".join(numeric_text(v) for v in values),
            source_tags=";".join(
                sorted({r["namespace"] + ":" + r["tag"] for r in rows})
            ),
        )
        canonical.append(row)
    return reported, canonical, latest_view(canonical), issues


def quarantine_facts(canonical, issues, records):
    """Exclude exact reviewed source errors; retain their raw facts and evidence.

    A changed value reopens review instead of inheriting an old exclusion.
    No corrected value or synthetic reporting version is manufactured.
    """
    keys = ("metric", "basis", "unit", "start", "end", "accession")
    index = {tuple(r[k] for k in keys): r for r in records}
    if len(index) != len(records):
        raise ValueError("Duplicate source-fact quarantine")
    kept = []
    for row in canonical:
        record = index.get(tuple(row[k] for k in keys))
        if record is None:
            kept.append(row)
            continue
        changed = number(row["value"]) != number(record["expected_value"])
        issues.append(
            dict(
                metric=row["metric"],
                fact_id=row["fact_ids"],
                reason=(
                    "QUARANTINE_EVIDENCE_CHANGED"
                    if changed
                    else "SOURCE_FACT_QUARANTINED"
                ),
                value=row["value"],
                end=row["end"],
                accession=row["accession"],
                evidence_url=record["evidence_url"],
                evidence_sha256=record["evidence_sha256"],
                rationale=record["reason"],
            )
        )
        if changed:
            kept.append(row)
    return kept, issues


def latest_view(canonical):
    """Select current reported versions within one issuer; preserve basis and conflicts."""
    latest_groups = defaultdict(list)
    for row in canonical:
        latest_groups[
            tuple(
                row[k]
                for k in ("metric", "basis", "unit", "start", "end", "period_kind")
            )
        ].append(row)
    latest = []
    for key, rows in sorted(latest_groups.items()):
        day = max(r["filed"] for r in rows)
        choices = [r for r in rows if r["filed"] == day]
        if all(r["filing_accepted_at"] for r in choices):
            last_stamp = max(
                datetime.fromisoformat(r["filing_accepted_at"].replace("Z", "+00:00"))
                for r in choices
            )
            choices = [
                r
                for r in choices
                if datetime.fromisoformat(
                    r["filing_accepted_at"].replace("Z", "+00:00")
                )
                == last_stamp
            ]
        row = dict(choices[0])
        if len({r["value"] for r in choices}) != 1 or any(
            r["status"] != "REPORTED" for r in choices
        ):
            row.update(value="", status="LATEST_CONFLICT")
        row["candidate_accessions"] = ";".join(sorted(r["accession"] for r in choices))
        row["view"] = "CURRENT_VINTAGE_LATEST_REPORT_NOT_HISTORICAL_AS_KNOWN"
        latest.append(row)
    return latest


def inline_facts(raw, *, cik=320193, metrics=None):
    metrics = METRICS if metrics is None else metrics
    parser = etree.XMLParser(
        resolve_entities=False, no_network=True, load_dtd=False, recover=False
    )
    root = etree.fromstring(raw, parser)
    if root.getroottree().docinfo.doctype:
        raise ValueError("Document types are not allowed in filing parser")
    contexts, units, metadata = {}, {}, {}
    for node in root.iter():
        local = etree.QName(node).localname if isinstance(node.tag, str) else ""
        if (
            local == "context"
            and etree.QName(node).namespace == "http://www.xbrl.org/2003/instance"
        ):
            if node.xpath(
                './/*[local-name()="explicitMember" or local-name()="typedMember" or local-name()="segment" or local-name()="scenario"]'
            ):
                continue
            entities = node.xpath('.//*[local-name()="identifier"]/text()')
            if not entities or int(entities[0]) != int(cik):
                continue
            start = node.xpath('.//*[local-name()="startDate"]/text()')
            end = node.xpath(
                './/*[local-name()="endDate" or local-name()="instant"]/text()'
            )
            if len(end) == 1:
                if node.get("id") in contexts:
                    raise ValueError("Duplicate context identifier")
                contexts[node.get("id")] = (start[0] if start else "", end[0])
        elif (
            local == "unit"
            and etree.QName(node).namespace == "http://www.xbrl.org/2003/instance"
        ):
            measures = node.xpath('.//*[local-name()="measure"]/text()')
            parts = []
            for measure in measures:
                prefix, name = measure.split(":")
                uri = node.nsmap.get(prefix)
                if name == "USD" and uri == "http://www.xbrl.org/2003/iso4217":
                    parts.append("USD")
                elif name == "shares" and uri == "http://www.xbrl.org/2003/instance":
                    parts.append("shares")
                else:
                    parts.append("UNSUPPORTED")
            divide = bool(node.xpath('.//*[local-name()="divide"]'))
            if node.get("id") in units:
                raise ValueError("Duplicate unit identifier")
            units[node.get("id")] = (
                "/".join(parts)
                if divide
                else (parts[0] if len(parts) == 1 else "UNSUPPORTED")
            )
        elif local == "nonNumeric" and (node.get("name") or "").startswith("dei:"):
            # Joint filings identify multiple registrants. Facts/DEI metadata
            # must belong to the requested entity's nondimensional contexts.
            # Metadata appears before contexts in many filings, so resolve the
            # context's entity directly rather than relying on traversal order.
            ref = node.get("contextRef")
            if ref:
                matches = root.xpath(
                    './/*[local-name()="context" and @id=$ref]', ref=ref
                )
                if len(matches) != 1:
                    raise ValueError("Missing/ambiguous DEI metadata context")
                if matches[0].xpath(
                    './/*[local-name()="explicitMember" or local-name()="typedMember" or local-name()="segment" or local-name()="scenario"]'
                ):
                    continue
                ids = matches[0].xpath('.//*[local-name()="identifier"]/text()')
                if len(ids) != 1 or int(ids[0]) != int(cik):
                    continue
            tag = node.get("name").split(":", 1)[1]
            if tag in {
                "EntityCentralIndexKey",
                "DocumentType",
                "DocumentPeriodEndDate",
            }:
                text = " ".join("".join(node.itertext()).split())
                metadata.setdefault(tag, set()).add(text)
    if {int(v) for v in metadata.get("EntityCentralIndexKey", [])} != {int(cik)}:
        raise ValueError(
            "Original filing does not identify Apple"
            if cik == 320193
            else "Original filing issuer mismatch"
        )
    observed = defaultdict(list)
    wanted = {
        ns + ":" + tag for _, _, mappings in metrics.values() for ns, tag, _ in mappings
    }
    for node in root.iter():
        if (
            not isinstance(node.tag, str)
            or etree.QName(node).localname != "nonFraction"
            or etree.QName(node).namespace != "http://www.xbrl.org/2013/inlineXBRL"
            or node.get("name") not in wanted
        ):
            continue
        if node.get("contextRef") not in contexts or node.get("unitRef") not in units:
            continue
        if node.get("{http://www.w3.org/2001/XMLSchema-instance}nil") in {"true", "1"}:
            continue
        prefix = node.get("name").split(":", 1)[0]
        uri = node.nsmap.get(prefix, "")
        if not (
            prefix == "us-gaap"
            and uri.startswith("http://fasb.org/us-gaap/")
            or prefix == "dei"
            and uri.startswith("http://xbrl.sec.gov/dei/")
        ):
            raise ValueError("Unexpected inline taxonomy namespace")
        text = "".join(node.itertext()).strip()
        transform = (node.get("format") or "").split(":")[-1]
        if transform in {"fixed-zero", "zerodash", "numdash"}:
            value = Decimal(0)
        elif transform in {"", "num-dot-decimal", "numdotdecimal"}:
            text = re.sub(r"[\s,]", "", text).replace("−", "-")
            if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
                continue
            value = number(text)
        else:
            continue
        scale = int(node.get("scale", "0"))
        if abs(scale) > 12 or node.get("sign", "") not in {"", "-"}:
            raise ValueError("Unsupported inline numeric scale/sign")
        value *= Decimal(10) ** scale
        if node.get("sign") == "-":
            value = -value
        start, end = contexts[node.get("contextRef")]
        key = (node.get("name"), units[node.get("unitRef")], start, end)
        observed[key].append(
            {
                "value": value,
                "context_id": node.get("contextRef"),
                "decimals": node.get("decimals", ""),
                "element_id": node.get("id", ""),
            }
        )
    return observed, {k: sorted(v) for k, v in metadata.items()}


def original_projection(raw, *, cik, metrics):
    """Retain selected nondimensional original facts with units and precision.

    This is an explicit projection, not a lossless copy of the filing bytes.
    """
    values, metadata = inline_facts(raw, cik=cik, metrics=metrics)
    return dict(
        metadata=metadata,
        facts=[
            dict(
                tag=tag,
                unit=unit,
                start=start,
                end=end,
                values=[dict(v, value=numeric_text(v["value"])) for v in rows],
            )
            for (tag, unit, start, end), rows in sorted(values.items())
        ],
    )


def supplement_original(canonical, original, *, cik, metrics):
    """Fill absent reporting-version keys from an explicitly retained original.

    Never overwrite an API value or mix dimensional/share-class contexts. Prefer
    the most precise original disclosure only when its value is unambiguous.
    """
    filing, receipt, projection = [
        original[k] for k in ("filing", "receipt", "projection")
    ]
    facts = dict(cik=cik, facts={})
    contexts = {}
    wanted = {
        (ns + ":" + tag, unit)
        for unit, _, mappings in metrics.values()
        for ns, tag, _ in mappings
    }
    for item in projection["facts"]:
        if (item["tag"], item["unit"]) not in wanted:
            raise ValueError("Original projection outside selected tags/units")
        values = item["values"]

        def precision(v):
            if v["decimals"].upper() == "INF":
                return Decimal("Infinity")
            value = Decimal(v["decimals"])
            if not value.is_finite() or value != int(value) or abs(value) > 18:
                raise ValueError("Unsupported original precision")
            return value

        best = max(precision(v) for v in values)
        candidates = {number(v["value"]) for v in values if precision(v) == best}
        if len(candidates) != 1:
            raise ValueError("Ambiguous equally precise original-filing values")
        value = next(iter(candidates))
        # Lower-precision duplicates must agree within their own rounding interval.
        if any(
            abs(number(v["value"]) - value)
            > (
                Decimal(0)
                if precision(v).is_infinite()
                else Decimal(10) ** (-int(precision(v))) / 2
            )
            for v in values
        ):
            raise ValueError("Inconsistent original-filing values")
        ns, tag = item["tag"].split(":", 1)
        row = dict(
            val=numeric_text(value),
            end=item["end"],
            accn=filing["accessionNumber"],
            filed=filing["filingDate"],
            form=filing["form"],
        )
        if item["start"]:
            row["start"] = item["start"]
        facts["facts"].setdefault(ns, {}).setdefault(tag, dict(units={}))[
            "units"
        ].setdefault(item["unit"], []).append(row)
        contexts[(item["tag"], item["unit"], item["start"], item["end"])] = ";".join(
            sorted({v["context_id"] for v in values if precision(v) == best})
        )
    _, incoming, _, issues = normalize(facts, [filing], receipt, metrics=metrics)
    if issues or any(r["status"] != "REPORTED" for r in incoming):
        raise ValueError("Invalid original-filing normalization")
    keys = ("metric", "basis", "unit", "start", "end", "period_kind", "accession")
    existing = {tuple(r[k] for k in keys): r for r in canonical}
    result = list(canonical)
    for row in incoming:
        key = tuple(row[k] for k in keys)
        if key in existing:
            if number(existing[key]["value"]) != number(row["value"]):
                raise ValueError("API/original disagreement; explicit review required")
            continue
        row.update(
            source_kind="ORIGINAL_FILING_ENTITY_CONTEXT",
            original_url=receipt["source_url"],
            original_context_ids=";".join(
                contexts[(tag, row["unit"], row["start"], row["end"])]
                for tag in row["source_tags"].split(";")
            ),
        )
        result.append(row)
    return result
