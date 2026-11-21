"""Export parquet files or datasets as plain CSV for human or scratch analysis.

    carina export SOURCE [SOURCE ...] --out DIR [--columns A,B] [--limit N]
                  [--max-mb MB] [--name NAME] [--overwrite]

``SOURCE`` is a parquet file, or a directory of parquet files (Hive-style
``key=value`` folders become columns). For each source the command writes
``NAME.csv`` and ``NAME.schema.json`` into ``DIR``; when ``--max-mb`` is given and
the output is larger, it writes ``NAME-001.csv``, ``NAME-002.csv`` and so on, each
with its own header and no larger than the limit.

The CSV keeps values exact:

- dates are ``YYYY-MM-DD``; timestamps are ISO 8601, in UTC with a trailing ``Z``
  when they carry a time zone;
- decimals and integers are written in full, and floats with the shortest text
  that reads back to the same value;
- nested values (lists, structs, maps) are JSON text and binary values are hex;
- a null is an empty field and an empty string is ``""``.

The schema file lists each column's parquet type and how it was written, the row
counts, and the size and SHA-256 of every CSV part. The export never changes its
source, and it refuses to replace earlier output unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SCHEMA = "carina-export-1.0"
BATCH_ROWS = 5_000
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ExportError(RuntimeError):
    """Raised when an export request is invalid or would overwrite earlier output."""


def _parquet_files(source: Path) -> tuple[list[Path], Path]:
    """The parquet files under ``source`` in a fixed order, and their base folder."""
    if source.is_symlink():
        raise ExportError(f"Source is a symlink: {source}")
    if source.is_file():
        return [source], source.parent
    if not source.is_dir():
        raise ExportError(f"Source not found: {source}")
    files = sorted(
        path
        for path in source.rglob("*.parquet")
        if path.is_file()
        and not path.is_symlink()
        and not any(part.startswith(".") for part in path.relative_to(source).parts)
    )
    if not files:
        raise ExportError(f"No parquet files under {source}")
    return files, source


def _require_one_schema(files: list[Path], base: Path) -> None:
    """Refuse a directory whose parquet files do not share one schema."""
    first = pq.read_schema(files[0])
    different = [
        str(path.relative_to(base))
        for path in files[1:]
        if not pq.read_schema(path).equals(first, check_metadata=False)
    ]
    if different:
        shown = ", ".join(different[:5]) + (" ..." if len(different) > 5 else "")
        raise ExportError(
            f"Parquet files under {base} do not share one schema ({len(different)} differ "
            f"from {files[0].relative_to(base)}: {shown}). "
            "Export the individual files or subfolders instead."
        )


def _open(source: Path) -> ds.Dataset:
    files, base = _parquet_files(source)
    _require_one_schema(files, base)
    return ds.dataset(
        [str(path) for path in files],
        format="parquet",
        partitioning=ds.partitioning(flavor="hive"),
        partition_base_dir=str(base),
    )


def _json_default(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return str(value)


def _plain(arrow_type: pa.DataType) -> pa.DataType:
    """The type with any dictionary encoding removed."""
    return arrow_type.value_type if pa.types.is_dictionary(arrow_type) else arrow_type


def _describe(arrow_type: pa.DataType) -> str:
    """How a column of this type is written in the CSV."""
    arrow_type = _plain(arrow_type)
    if pa.types.is_timestamp(arrow_type):
        return "iso8601-utc-timestamp" if arrow_type.tz else "iso8601-timestamp"
    if pa.types.is_date(arrow_type):
        return "iso8601-date"
    if pa.types.is_decimal(arrow_type):
        return "exact-decimal"
    if pa.types.is_nested(arrow_type):
        return "json"
    if (
        pa.types.is_binary(arrow_type)
        or pa.types.is_large_binary(arrow_type)
        or pa.types.is_fixed_size_binary(arrow_type)
    ):
        return "hex"
    return "as-is"


def _column(values: pa.ChunkedArray) -> pa.ChunkedArray:
    """Convert one column to a type the CSV writer renders exactly as described."""
    arrow_type = values.type
    if pa.types.is_dictionary(arrow_type):
        values = values.cast(arrow_type.value_type)
        arrow_type = values.type
    kind = _describe(arrow_type)
    if kind == "iso8601-utc-timestamp":
        utc = values.cast(pa.timestamp(arrow_type.unit, "UTC"))
        return pc.strftime(utc, format="%Y-%m-%dT%H:%M:%SZ")
    if kind == "iso8601-timestamp":
        return pc.strftime(values, format="%Y-%m-%dT%H:%M:%S")
    if kind == "json":
        return pa.chunked_array(
            [
                pa.array(
                    [
                        None if item is None else json.dumps(item, default=_json_default)
                        for item in chunk.to_pylist()
                    ],
                    pa.string(),
                )
                for chunk in values.chunks
            ]
            or [pa.array([], pa.string())]
        )
    if kind == "hex":
        return pa.chunked_array(
            [
                pa.array(
                    [None if item is None else bytes(item).hex() for item in chunk.to_pylist()],
                    pa.string(),
                )
                for chunk in values.chunks
            ]
            or [pa.array([], pa.string())]
        )
    if pa.types.is_null(arrow_type):
        return values.cast(pa.string())
    if pa.types.is_duration(arrow_type) or pa.types.is_interval(arrow_type):
        raise ExportError(f"Unsupported column type: {arrow_type}")
    return values


def _render(table: pa.Table, *, header: bool) -> bytes:
    prepared = pa.table(
        [_column(table.column(name)) for name in table.column_names],
        names=table.column_names,
    )
    buffer = io.BytesIO()
    pacsv.write_csv(prepared, buffer, pacsv.WriteOptions(include_header=header))
    return buffer.getvalue()


def _stem(source: Path, name: str | None) -> str:
    stem = name or (source.name if source.is_dir() else source.stem)
    if not _NAME.match(stem):
        raise ExportError(
            f"Unsafe output name {stem!r}: use letters, digits, '.', '_' and '-'"
        )
    return stem


def _prior_output(out: Path, stem: str) -> list[Path]:
    """Earlier files this export would produce or leave stale."""
    pattern = re.compile(rf"^{re.escape(stem)}(-\d+)?\.csv$|^{re.escape(stem)}\.schema\.json$")
    return sorted(
        path for path in out.iterdir() if pattern.match(path.name) and not path.is_dir()
    )


def export_one(
    source,
    out,
    *,
    name=None,
    columns=None,
    limit=None,
    max_bytes=None,
    overwrite=False,
    batch_rows=BATCH_ROWS,
) -> dict:
    """Export one parquet file or dataset; return the schema record written."""
    source, out = Path(source), Path(out)
    if limit is not None and limit < 0:
        raise ExportError("--limit must not be negative")
    if max_bytes is not None and max_bytes <= 0:
        raise ExportError("--max-mb must be positive")
    stem = _stem(source, name)
    dataset = _open(source)
    names = dataset.schema.names
    if columns:
        unknown = [column for column in columns if column not in names]
        if unknown:
            raise ExportError(f"Unknown columns: {', '.join(unknown)}")
        if len(set(columns)) != len(columns):
            raise ExportError("Duplicate columns requested")
        picked = list(columns)
    else:
        picked = list(names)
    out.mkdir(parents=True, exist_ok=True)
    if out.is_symlink():
        raise ExportError(f"Output directory is a symlink: {out}")
    existing = _prior_output(out, stem)
    if existing and not overwrite:
        raise ExportError(
            "Refusing to replace earlier output: "
            + ", ".join(path.name for path in existing)
            + ". Pass --overwrite to replace it."
        )

    scan_schema = pa.schema([dataset.schema.field(column) for column in picked])
    header = _render(scan_schema.empty_table(), header=True)
    temp = Path(tempfile.mkdtemp(prefix=f".export-{stem}-", dir=out))
    parts: list[dict] = []
    try:
        handle = None
        digest = None
        current: dict | None = None

        def start():
            nonlocal handle, digest, current
            path = temp / f"part-{len(parts) + 1:06d}.csv"
            handle = path.open("wb")
            handle.write(header)
            digest = hashlib.sha256(header)
            current = {"path": path, "rows": 0, "bytes": len(header)}
            parts.append(current)

        def finish():
            nonlocal handle
            if handle is not None:
                handle.close()
                current["sha256"] = digest.hexdigest()
                handle = None

        def pieces(batch):
            """Rendered row groups, halved until each fits in one part."""
            chunk = _render(pa.Table.from_batches([batch]), header=False)
            if (
                max_bytes is None
                or len(header) + len(chunk) <= max_bytes
                or batch.num_rows == 1
            ):
                yield batch.num_rows, chunk
                return
            half = batch.num_rows // 2
            yield from pieces(batch.slice(0, half))
            yield from pieces(batch.slice(half))

        written = 0
        start()
        for batch in dataset.scanner(columns=picked, batch_size=batch_rows).to_batches():
            if limit is not None:
                batch = batch.slice(0, limit - written)
            if batch.num_rows == 0:
                if limit is not None and written >= limit:
                    break
                continue
            for rows, chunk in pieces(batch):
                if (
                    max_bytes is not None
                    and current["rows"] > 0
                    and current["bytes"] + len(chunk) > max_bytes
                ):
                    finish()
                    start()
                handle.write(chunk)
                digest.update(chunk)
                current["rows"] += rows
                current["bytes"] += len(chunk)
                written += rows
            if limit is not None and written >= limit:
                break
        finish()

        total = dataset.count_rows()
        width = max(3, len(str(len(parts))))
        files = []
        for index, part in enumerate(parts, 1):
            final = f"{stem}.csv" if len(parts) == 1 else f"{stem}-{index:0{width}d}.csv"
            files.append(
                {
                    "name": final,
                    "rows": part["rows"],
                    "bytes": part["bytes"],
                    "sha256": part["sha256"],
                    "_path": part["path"],
                }
            )
        if max_bytes is not None:
            too_big = [f["name"] for f in files if f["bytes"] > max_bytes]
            if too_big:
                raise ExportError(
                    "A single row is larger than --max-mb: " + ", ".join(too_big)
                )
        record = {
            "schema": SCHEMA,
            "source": source.name,
            "rows": written,
            "source_rows": total,
            "complete": written == total,
            "columns": [
                {
                    "name": column,
                    "parquet_type": str(dataset.schema.field(column).type),
                    "csv_format": _describe(dataset.schema.field(column).type),
                    "nullable": dataset.schema.field(column).nullable,
                }
                for column in picked
            ],
            "null": "empty field (an empty string is written as \"\")",
            "files": [{k: v for k, v in f.items() if k != "_path"} for f in files],
        }
        schema_file = temp / f"{stem}.schema.json"
        schema_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

        for path in existing:
            path.unlink()
        for item in files:
            os.replace(item["_path"], out / item["name"])
        os.replace(schema_file, out / f"{stem}.schema.json")
        return record
    finally:
        if handle is not None:
            handle.close()
        shutil.rmtree(temp, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carina export",
        description="Export parquet files or datasets as CSV with a schema file.",
    )
    parser.add_argument("sources", nargs="+", help="Parquet files or dataset directories.")
    parser.add_argument("--out", required=True, help="Directory to write the CSV files to.")
    parser.add_argument("--columns", help="Comma-separated columns to export, in order.")
    parser.add_argument("--limit", type=int, help="Export only the first N rows.")
    parser.add_argument(
        "--max-mb",
        type=float,
        help="Split the output into parts no larger than this many megabytes.",
    )
    parser.add_argument("--name", help="Output name (one source only); default is the source name.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace earlier output of the same name.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.name and len(args.sources) > 1:
        raise SystemExit("--name works with a single source")
    columns = [c.strip() for c in args.columns.split(",") if c.strip()] if args.columns else None
    max_bytes = int(args.max_mb * 1024 * 1024) if args.max_mb is not None else None
    try:
        stems = [_stem(Path(source), args.name) for source in args.sources]
        if len(set(stems)) != len(stems):
            raise ExportError("Sources would write the same output name; export them separately")
        for source in args.sources:
            record = export_one(
                source,
                args.out,
                name=args.name,
                columns=columns,
                limit=args.limit,
                max_bytes=max_bytes,
                overwrite=args.overwrite,
            )
            print(
                json.dumps(
                    {
                        "source": record["source"],
                        "rows": record["rows"],
                        "source_rows": record["source_rows"],
                        "complete": record["complete"],
                        "files": [f["name"] for f in record["files"]],
                    }
                )
            )
    except ExportError as error:
        raise SystemExit(f"carina export: {error}")
    return 0
