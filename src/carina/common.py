"""Shared helpers for company collection and store code.

Generic digests, atomic writes, HTTP fetch with retry, the capture-bundle protocol,
and Drive upload checks. Capture-bundle callers pass their own validator and base."""

from __future__ import annotations

import base64
import bz2
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

import requests


MAX_BODY = 12 * 1024 * 1024
MAX_BUNDLE = 36 * 1024 * 1024
MAX_PARTS = 16
PARAMETERS = {
    "auto_adjust": False,
    "back_adjust": False,
    "repair": False,
    "actions": True,
}
REPOSITORY = "https://github.com/tylersifers-png/carina"
UA = f"Carina/0.1 ({REPOSITORY})"


def now():
    return datetime.now(timezone.utc).isoformat()


def code_version():
    """The code that ran: the git commit when known, else the installed release."""
    from importlib.metadata import PackageNotFoundError, distribution
    import subprocess

    try:
        dist = distribution("carina")
    except PackageNotFoundError:
        dist = None
    if dist is not None:
        direct = dist.read_text("direct_url.json")
        commit = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
        if commit:
            return commit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "carina-" + (dist.version if dist is not None else "unknown")


def encoded(obj):
    return (
        json.dumps(obj, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Symlink output")
    if path.exists():
        if path.read_bytes() == raw:
            return
        raise ValueError("Refusing to change existing SEC bytes")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(raw)
    os.replace(temporary, path)


def identity(value):
    if not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Expected SEC bundle SHA-256")
    return value


def folder(root, bundle, base):
    path = root / base / identity(bundle)
    if not path.resolve().is_relative_to(root.resolve()) or any(
        p.is_symlink() for p in [path, *path.parents]
    ):
        raise ValueError("Unsafe SEC folder")
    return path


def fetch(url, agent, *, max_body=MAX_BODY):
    attempts = []
    for attempt in range(3):
        started = now()
        try:
            with requests.get(
                url,
                headers={"User-Agent": agent, "Accept-Encoding": "gzip, deflate"},
                timeout=(10, 45),
                stream=True,
                allow_redirects=False,
            ) as response:
                raw = bytearray()
                for chunk in response.iter_content(65536):
                    raw.extend(chunk)
                    if len(raw) > max_body:
                        raise ValueError("Oversized SEC response")
                receipt = {
                    "source_url": url,
                    "requested_at": started,
                    "retrieved_at": now(),
                    "http_status": response.status_code,
                    "bytes": len(raw),
                    "sha256": digest(raw),
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                }
                attempts.append(receipt)
                if response.status_code == 200:
                    return bytes(raw), attempts
                if response.status_code not in {429, 500, 502, 503, 504}:
                    break  # In particular, do not retry a 403 or change identity/host.
                delay = response.headers.get("Retry-After", "")
                if delay and not delay.isdigit():
                    try:
                        delay = max(
                            1,
                            int(
                                (
                                    parsedate_to_datetime(delay)
                                    - datetime.now(timezone.utc)
                                ).total_seconds()
                            )
                            + 1,
                        )
                    except (ValueError, TypeError):
                        break
                else:
                    delay = int(delay) if delay else 2 ** (attempt + 1)
                if delay > 30:
                    break  # Defer to a later run rather than shorten SEC's requested wait.
        except requests.RequestException as exc:
            attempts.append(
                {
                    "requested_at": started,
                    "retrieved_at": now(),
                    "error_type": type(exc).__name__,
                }
            )
            delay = 2 ** (attempt + 1)
        if attempt < 2:
            time.sleep(max(1, delay))
    return None, attempts


def piece_valid(piece, *, max_parts=MAX_PARTS):
    if set(piece) != {"bundle_sha256", "index", "total", "data"}:
        raise ValueError("Unexpected SEC piece fields")
    identity(piece["bundle_sha256"])
    if (
        type(piece["index"]) is not int
        or type(piece["total"]) is not int
        or not 0 <= piece["index"] < piece["total"] <= max_parts
    ):
        raise ValueError("Invalid SEC piece numbering")
    if not isinstance(piece["data"], str) or not 0 < len(piece["data"]) <= 56000:
        raise ValueError("Oversized SEC piece")
    base64.b64decode(piece["data"], validate=True)
    return piece


def prepare_capture(
    path, content, *, max_bundle=MAX_BUNDLE, max_parts=MAX_PARTS, validator
):
    validator(content)
    write(path / "capture.json", encoded(content))
    if len(encoded(content)) > max_bundle:
        raise ValueError("Capture exceeds bounded raw size")
    compressed = bz2.compress(encoded(content))
    bundle = digest(compressed)
    data = base64.b64encode(compressed).decode()
    chunks = [data[i : i + 56000] for i in range(0, len(data), 56000)]
    if len(chunks) > max_parts:
        raise ValueError(
            "Capture exceeds bounded Actions transport; original local capture retained"
        )
    for index, chunk in enumerate(chunks):
        write(
            path / f"part_{index:02}.json",
            encoded(
                {
                    "bundle_sha256": bundle,
                    "index": index,
                    "total": len(chunks),
                    "data": chunk,
                }
            ),
        )
    write(path / "handoff.json", encoded({"bundle": bundle, "parts": len(chunks)}))


def load(
    path, bundle, *, max_parts=MAX_PARTS, max_bundle=MAX_BUNDLE, validator
):
    pieces = [
        piece_valid(json.loads(p.read_bytes()), max_parts=max_parts)
        for p in sorted(
            path.glob("part_*.json"), key=lambda p: int(p.stem.split("_")[-1])
        )
    ]
    if not pieces or len(pieces) != pieces[0]["total"]:
        raise ValueError("Incomplete SEC transfer")
    if any(
        p["bundle_sha256"] != identity(bundle)
        or p["index"] != i
        or p["total"] != len(pieces)
        for i, p in enumerate(pieces)
    ):
        raise ValueError("Conflicting SEC pieces")
    compressed = base64.b64decode("".join(p["data"] for p in pieces), validate=True)
    if digest(compressed) != bundle:
        raise ValueError("SEC transfer checksum mismatch")
    decoder = bz2.BZ2Decompressor()
    raw = decoder.decompress(compressed, max_length=max_bundle + 1)
    if len(raw) > max_bundle or not decoder.eof or decoder.unused_data:
        raise ValueError("Invalid/oversized compressed SEC capture")
    return validator(json.loads(raw))


def connect(root):
    from googleapiclient.discovery import build
    from carina.drive import DriveSync, DriveClient, build_credentials_from_env

    return DriveSync(
        DriveClient(
            build(
                "drive",
                "v3",
                credentials=build_credentials_from_env(),
                cache_discovery=False,
            )
        ),
        os.environ["DRIVE_PROJECT_FOLDER_ID"],
        root,
        {"state"},
    )


def upload_checked(sync, relative):
    from carina.drive import file_md5, DriveSyncError

    local = sync.local_root / relative
    before = sync._resolve_remote(sync._validated(relative), create_folders=False)
    if before and (before.is_folder or before.md5_checksum != file_md5(local)):
        raise DriveSyncError("Refusing to change existing SEC archive bytes")
    sync.upload(relative)
    remote = sync._resolve_remote(sync._validated(relative), create_folders=False)
    if remote.md5_checksum != file_md5(local):
        raise DriveSyncError("SEC upload MD5 mismatch")
    with tempfile.TemporaryDirectory() as temp:
        restored = Path(temp) / "readback"
        sync.client.download_file(remote.id, restored)
        if digest(restored.read_bytes()) != digest(local.read_bytes()):
            raise DriveSyncError("SEC upload SHA-256 readback mismatch")
    return {
        "name": local.name,
        "id": remote.id,
        "sha256": digest(local.read_bytes()),
        "bytes": local.stat().st_size,
    }


def stage(sync, piece, *, base, max_parts=MAX_PARTS):
    piece_valid(piece, max_parts=max_parts)
    path = (
        folder(sync.local_root, piece["bundle_sha256"], base)
        / f"part_{piece['index']:02}.json"
    )
    write(path, encoded(piece))
    upload_checked(sync, str(path.relative_to(sync.local_root)))
    print(f"Verified SEC piece {piece['index']+1}/{piece['total']}", flush=True)


def last_completed(cutoff):
    import pandas as pd
    import exchange_calendars as xcals

    cal = xcals.get_calendar(
        "XNYS", start=cutoff.date() - timedelta(days=10), end=cutoff.date()
    )
    completed = cal.schedule.loc[cal.schedule["close"] <= pd.Timestamp(cutoff)]
    return str(completed.index[-1].date())


def receipt_valid(receipt, raw, url=None):
    if url is not None and (
        receipt["source_url"] != url or receipt["http_status"] != 200
    ):
        raise ValueError("Source URL/status mismatch")
    requested = datetime.fromisoformat(receipt["requested_at"])
    retrieved = datetime.fromisoformat(receipt["retrieved_at"])
    if (
        requested.tzinfo is None
        or retrieved.tzinfo is None
        or requested > retrieved
        or retrieved > datetime.now(timezone.utc)
    ):
        raise ValueError("Invalid capture timestamp")
    if len(raw) != receipt["bytes"] or digest(raw) != receipt["sha256"]:
        raise ValueError("Source hash/size mismatch")
