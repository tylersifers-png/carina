"""The staged dataset: parquet tables in standard layout and one manifest.

A dataset directory holds the company tables and two auxiliary files. Each table
is either one parquet file (``companies.parquet``) or a directory of Hive-style
partitions (``daily_bars/year=2026/part.parquet``), so pandas, pyarrow, DuckDB
and Polars read it directly. ``manifest.json`` lists every file with its hash.
Candidate builds may still be flat (one file per table); ``read_table`` reads both.

The manifest also records the dataset's lineage: the digests of the manifests it
descends from. Two copies can then be compared by ancestry instead of by location.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil

import pandas as pd

from carina import common as sec

SCHEMA = "company-staged-1.0"
TABLES = (
    "companies",
    "securities",
    "daily_bars",
    "fundamentals",
    "financial_latest",
    "filings",
    "field_applicability",
)
AUX_FILES = ("coverage.json", "sources.json")
# Tables stored as partitions, and the partition column. Everything else is one file.
PARTITIONS = {"daily_bars": "year"}
TABLE_KEYS = {
    "companies": ["symbol"],
    "securities": ["symbol", "security_id"],
    "daily_bars": ["symbol", "session_date"],
    "fundamentals": [
        "symbol",
        "metric",
        "basis",
        "unit",
        "start",
        "end",
        "period_kind",
        "accession",
    ],
    "financial_latest": [
        "symbol",
        "metric",
        "basis",
        "unit",
        "start",
        "end",
        "period_kind",
    ],
    "filings": ["symbol", "filingDate", "accessionNumber"],
    "field_applicability": ["symbol", "metric"],
}
MANIFEST = "manifest.json"


def frame_digest(frame: pd.DataFrame, keys: list[str]) -> str:
    """Stable logical digest independent of Parquet encoding and row order."""
    present = [key for key in keys if key in frame.columns]
    ordered = (
        frame.sort_values(present, kind="stable", na_position="first")
        if present
        else frame
    ).reset_index(drop=True)
    rows = pd.util.hash_pandas_object(ordered, index=False, categorize=False)
    schema = sec.encoded(
        dict(
            columns=list(ordered.columns),
            dtypes=[str(dtype) for dtype in ordered.dtypes],
            rows=len(ordered),
        )
    )
    digest = hashlib.sha256(schema)
    digest.update(rows.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()


def member(relative: str) -> str:
    """The logical member (a table or auxiliary file) a file path belongs to."""
    first = PurePosixPath(relative).parts[0]
    return first[: -len(".parquet")] if first.endswith(".parquet") else first


def expected_members() -> set[str]:
    return set(TABLES) | set(AUX_FILES)


def safe_relative(relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or relative != path.as_posix()
        or relative == MANIFEST
    ):
        raise ValueError("Unsafe dataset path: " + relative)
    return path


def partition_slices(name: str, frame: pd.DataFrame):
    """Yield (relative path, rows) for each file a table is stored as."""
    if name not in PARTITIONS:
        yield f"{name}.parquet", frame
        return
    years = frame["session_date"].astype(str).str[:4]
    for year in sorted(years.unique()):
        yield f"{name}/{PARTITIONS[name]}={year}/part.parquet", frame.loc[years.eq(year)]


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def write_table(root: Path, name: str, frame: pd.DataFrame) -> list[str]:
    """Write one table in its stored form and return the relative paths written."""
    written = []
    for relative, part in partition_slices(name, frame):
        write_parquet(Path(root) / relative, part)
        written.append(relative)
    return written


def read_table(root, name, *, columns=None, filters=None, keep_partitions=False):
    """Read a table from a flat candidate or a partitioned dataset."""
    root = Path(root)
    directory, single = root / name, root / f"{name}.parquet"
    if directory.is_dir():
        frame = pd.read_parquet(directory, columns=columns, filters=filters)
        column = PARTITIONS.get(name)
        if column in frame.columns and not keep_partitions:
            frame = frame.drop(columns=[column])
        return frame
    return pd.read_parquet(single, columns=columns, filters=filters)


def inventory(root: Path) -> dict[str, dict]:
    """Every file under a dataset (except the manifest) with its hash and size."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Dataset root is a symlink")
    found = {}
    for current, directories, names in os.walk(root, followlinks=False):
        for entry in [*directories, *names]:
            if (Path(current) / entry).is_symlink():
                raise ValueError("Dataset contains a symlink: " + entry)
        for entry in names:
            path = Path(current) / entry
            relative = path.relative_to(root).as_posix()
            if relative == MANIFEST:
                continue
            raw = path.read_bytes()
            found[relative] = dict(sha256=sec.digest(raw), bytes=len(raw))
    return found


def verify_files(root: Path, manifest: dict) -> None:
    """Check the manifest's files against the directory: names, members, and hashes."""
    files = manifest["files"]
    for relative in files:
        safe_relative(relative)
    if manifest.get("schema") == SCHEMA and {
        member(relative) for relative in files
    } != expected_members():
        raise ValueError("Unexpected dataset members")
    found = inventory(root)
    if set(found) != set(files):
        raise ValueError("Unexpected/missing dataset files")
    for relative, item in files.items():
        if found[relative] != item:
            raise ValueError("Dataset checksum mismatch: " + relative)


def manifest_digest(root: Path) -> str:
    return sec.digest((Path(root) / MANIFEST).read_bytes())


def descends(manifest: dict, head: str, ancestor: str) -> bool:
    """True if ``ancestor`` is ``head`` itself or one of the manifest's ancestors."""
    return ancestor == head or ancestor in manifest.get("lineage", [])


def verify_equivalent(left: Path, right: Path) -> dict[str, dict]:
    """Compare two datasets table by table by logical digest; raise on any difference."""
    report = {}
    for name in TABLES:
        a, b = read_table(left, name), read_table(right, name)
        da, db = frame_digest(a, TABLE_KEYS[name]), frame_digest(b, TABLE_KEYS[name])
        report[name] = dict(rows=len(a), equal=da == db and len(a) == len(b))
    bad = [name for name, item in report.items() if not item["equal"]]
    if bad:
        raise ValueError("Datasets differ logically: " + ", ".join(bad))
    return report


def convert(
    source: Path, destination: Path, *, lineage: list[str] | None = None, finalize=None
) -> dict:
    """Convert a flat candidate or v1 current store into a staged dataset.

    Unpartitioned tables and auxiliary files are copied byte for byte, so their
    hashes are unchanged. The destination must be empty. The result is verified
    logically equal to the source before it is returned. ``lineage`` is the new
    dataset's list of ancestor digests; by default the source manifest is its parent.
    ``finalize`` may add to the manifest (such as a validation binding) before it is written.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Conversion destination must be empty")
    raw_manifest = (source / MANIFEST).read_bytes()
    legacy = json.loads(raw_manifest)
    verify_files(source, legacy)
    destination.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        if name in PARTITIONS:
            write_table(destination, name, read_table(source, name))
        else:
            shutil.copyfile(source / f"{name}.parquet", destination / f"{name}.parquet")
    for name in AUX_FILES:
        shutil.copyfile(source / name, destination / name)
    manifest = {
        key: value
        for key, value in legacy.items()
        if key not in {"validation", "readback", "files"}
    }
    manifest.update(
        schema=SCHEMA,
        files=inventory(destination),
        partitioning=dict(PARTITIONS),
        lineage=(
            [*legacy.get("lineage", []), sec.digest(raw_manifest)]
            if lineage is None
            else list(lineage)
        ),
        converted_from=dict(schema=legacy["schema"], manifest_sha256=sec.digest(raw_manifest)),
        verified_at=sec.now(),
        status="VERIFIED",
    )
    if finalize is not None:
        manifest = finalize(manifest)
    temporary = destination / (MANIFEST + ".tmp")
    temporary.write_bytes(sec.encoded(manifest))
    os.replace(temporary, destination / MANIFEST)
    verify_files(destination, manifest)
    verify_equivalent(source, destination)
    return manifest


def swap_in(new: Path, staged: Path) -> None:
    """Replace ``staged`` with ``new`` by renaming. The old copy is deleted only after
    the new one verifies; on any failure the old copy is put back."""
    new, staged = Path(new), Path(staged)
    previous = staged.with_name(staged.name + ".prev")
    if previous.exists():
        shutil.rmtree(previous)
    had_old = staged.exists()
    if had_old:
        os.rename(staged, previous)
    try:
        os.rename(new, staged)
        verify_files(staged, json.loads((staged / MANIFEST).read_bytes()))
    except Exception:
        if staged.exists():
            os.rename(staged, new)
        if had_old:
            os.rename(previous, staged)
        raise
    if had_old:
        shutil.rmtree(previous)


def recover(staged: Path) -> None:
    """Undo a swap interrupted between its two renames (the old copy is intact)."""
    staged = Path(staged)
    previous = staged.with_name(staged.name + ".prev")
    if not staged.exists() and previous.exists():
        os.rename(previous, staged)


def promote(candidate: Path, staged: Path, *, finalize=None) -> dict:
    """Make a verified candidate the staged dataset, recording its lineage.

    The candidate is converted beside the current dataset, verified logically equal,
    and swapped in. Refuses a damaged current dataset, and a candidate that was built
    from a different head than the one now staged.
    """
    candidate, staged = Path(candidate), Path(staged)
    recover(staged)
    lineage: list[str] = []
    if (staged / MANIFEST).exists():
        raw = (staged / MANIFEST).read_bytes()
        previous = json.loads(raw)
        verify_files(staged, previous)
        head = sec.digest(raw)
        lineage = [*previous.get("lineage", []), head]
        base = json.loads((candidate / MANIFEST).read_bytes()).get("base_manifest_sha256")
        if base and base != head:
            raise ValueError("The staged dataset changed since the candidate was built")
    new = staged.with_name(staged.name + ".new")
    if new.exists():
        shutil.rmtree(new)
    manifest = convert(candidate, new, lineage=lineage, finalize=finalize)
    swap_in(new, staged)
    return manifest
