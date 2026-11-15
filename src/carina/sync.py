"""Fast-forward-only sync of the staged dataset with Drive, decided by lineage.

Neither side is "the" copy. A dataset's head is the digest of its manifest, and each
manifest lists the digests it descends from. A push (local to Drive) is allowed only
when Drive's head is an ancestor of the local head; a pull only when the local head
is an ancestor of Drive's. Anything else is divergence: nothing is changed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from carina import common as sec, dataset
from carina.drive import DriveSync, DriveSyncError, file_md5

# Where the dataset sits under the sync root, locally and on Drive. Callers whose
# layout differs pass ``staged=``. ``legacy=`` names an older pointer manifest whose
# ``source_v1_manifest_sha256`` the local lineage may descend from (first push only).
STAGED = "staged/company"

PUSHABLE = {"local_ahead", "remote_empty", "local_ahead_of_legacy"}
PULLABLE = {"remote_ahead"}


def read_head(root: Path, staged: str = STAGED) -> tuple[dict, str] | None:
    path = Path(root) / staged / dataset.MANIFEST
    if not path.is_file():
        return None
    raw = path.read_bytes()
    return json.loads(raw), sec.digest(raw)


def read_remote_head(sync: DriveSync, relative: str) -> tuple[dict, str] | None:
    remote = sync._resolve_remote(sync._validated(relative), create_folders=False)
    if remote is None or remote.is_folder:
        return None
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "manifest.json"
        sync.client.download_file(remote.id, path)
        raw = path.read_bytes()
    return json.loads(raw), sec.digest(raw)


def classify(local, remote, legacy) -> str:
    if remote is None and legacy is not None:
        if local is None:
            return "legacy_only"
        source = legacy[0].get("source_v1_manifest_sha256")
        known = {local[1], *local[0].get("lineage", [])}
        return "local_ahead_of_legacy" if source in known else "diverged_from_legacy"
    if local is None and remote is None:
        return "both_empty"
    if remote is None:
        return "remote_empty"
    if local is None:
        return "remote_ahead"
    if local[1] == remote[1]:
        return "in_sync"
    if remote[1] in local[0].get("lineage", []):
        return "local_ahead"
    if local[1] in remote[0].get("lineage", []):
        return "remote_ahead"
    return "diverged"


def status(
    sync: DriveSync, root: Path | None = None, *, staged: str = STAGED, legacy: str | None = None
) -> dict:
    """Read-only: how the local and Drive heads relate."""
    local = read_head(root or sync.local_root, staged)
    remote = read_remote_head(sync, staged + "/" + dataset.MANIFEST)
    legacy = read_remote_head(sync, legacy) if remote is None and legacy else None
    return dict(
        relation=classify(local, remote, legacy),
        local=local[1] if local else None,
        remote=remote[1] if remote else None,
        legacy_source=legacy[0].get("source_v1_manifest_sha256") if legacy else None,
    )


def _upload_verified(sync: DriveSync, relative: str) -> None:
    sync.upload(relative)
    remote = sync._resolve_remote(sync._validated(relative), create_folders=False)
    if remote is None or remote.md5_checksum != file_md5(sync.local_root / relative):
        raise DriveSyncError("Upload MD5 mismatch: " + relative)


def push(sync: DriveSync, *, staged: str = STAGED, legacy: str | None = None) -> dict:
    """Upload the local dataset if Drive's head is its ancestor. The manifest goes last."""
    found = status(sync, staged=staged, legacy=legacy)
    if found["relation"] == "in_sync":
        return dict(found, pushed=0)
    if found["relation"] not in PUSHABLE:
        raise DriveSyncError("Refusing to push: " + found["relation"])
    local, head = read_head(sync.local_root, staged)
    dataset.verify_files(sync.local_root / staged, local)
    for relative in sorted(local["files"]):
        _upload_verified(sync, staged + "/" + relative)
    # Compare-and-swap: the manifest is the commit, so check the remote did not move.
    current = read_remote_head(sync, staged + "/" + dataset.MANIFEST)
    if (current[1] if current else None) != found["remote"]:
        raise DriveSyncError("Drive changed during the push; nothing was committed")
    _upload_verified(sync, staged + "/" + dataset.MANIFEST)
    committed = read_remote_head(sync, staged + "/" + dataset.MANIFEST)
    if committed is None or committed[1] != head:
        raise DriveSyncError("Drive manifest readback mismatch")
    return dict(found, relation="in_sync", remote=head, pushed=len(local["files"]) + 1)


def pull(sync: DriveSync, *, staged: str = STAGED) -> dict:
    """Download Drive's dataset if the local head is its ancestor (or local is empty)."""
    found = status(sync, staged=staged)
    if found["relation"] == "in_sync":
        return dict(found, pulled=0)
    if found["relation"] not in PULLABLE:
        raise DriveSyncError("Refusing to pull: " + found["relation"])
    scratch = sync.local_root / "scratch" / "pull"
    if scratch.exists():
        shutil.rmtree(scratch)
    try:
        side = DriveSync(sync.client, sync.drive_root_id, scratch, sync.allowed_roots)
        side.local_root.mkdir(parents=True, exist_ok=True)
        side.download(staged + "/" + dataset.MANIFEST)
        downloaded = scratch / staged
        manifest = json.loads((downloaded / dataset.MANIFEST).read_bytes())
        for relative in sorted(manifest["files"]):
            dataset.safe_relative(relative)
            side.download(staged + "/" + relative)
        dataset.verify_files(downloaded, manifest)
        target = sync.local_root / staged
        target.parent.mkdir(parents=True, exist_ok=True)
        dataset.recover(target)
        now = read_head(sync.local_root, staged)
        if (now[1] if now else None) != found["local"]:
            raise DriveSyncError("The local dataset changed during the pull")
        dataset.swap_in(downloaded, target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return dict(found, relation="in_sync", local=found["remote"], pulled=len(manifest["files"]) + 1)


DATASETS = {"company": "staged/company", "macro": "staged/macro"}


def main(argv=None) -> int:
    from carina import settings
    from carina.drive import connect

    parser = argparse.ArgumentParser(
        prog="carina sync",
        description="Fast-forward-only sync of one staged dataset between the data "
        "directory and the matching Drive folder.",
    )
    parser.add_argument("action", choices=["status", "pull", "push"])
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument(
        "--drive-root-id",
        default=os.environ.get("DRIVE_PROJECT_FOLDER_ID"),
        help="Drive folder that mirrors the data directory.",
    )
    parser.add_argument("--legacy", help="Older Drive pointer manifest (first push only).")
    args = parser.parse_args(argv)
    sync = connect(args.drive_root_id, settings.data(), ["staged"])
    staged = DATASETS[args.dataset]
    if args.action == "status":
        result = status(sync, staged=staged, legacy=args.legacy)
    elif args.action == "pull":
        result = pull(sync, staged=staged)
    else:
        result = push(sync, staged=staged, legacy=args.legacy)
    print(json.dumps(result, sort_keys=True))
    return 0
