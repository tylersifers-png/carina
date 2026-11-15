"""Narrow Google Drive sync between a Drive folder and a local directory.

This module intentionally supports only:
- recursive download of explicitly allowed Drive paths; and
- create/update upload of explicitly allowed local paths.

It does not implement deletion, mirroring, or rename semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import mimetypes
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_NATIVE_PREFIX = "application/vnd.google-apps."
DRIVE_API_RETRIES = 4


class DriveSyncError(RuntimeError):
    """Raised when a Drive sync invariant is violated."""


@dataclass(frozen=True)
class RemoteItem:
    id: str
    name: str
    mime_type: str
    md5_checksum: str | None = None
    size: int | None = None

    @property
    def is_folder(self) -> bool:
        return self.mime_type == GOOGLE_FOLDER_MIME


@dataclass
class SyncStats:
    downloaded: int = 0
    created: int = 0
    updated: int = 0
    skipped_unchanged: int = 0
    folders_created: int = 0
    bytes_downloaded: int = 0
    bytes_uploaded: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "downloaded": self.downloaded,
            "created": self.created,
            "updated": self.updated,
            "skipped_unchanged": self.skipped_unchanged,
            "folders_created": self.folders_created,
            "bytes_downloaded": self.bytes_downloaded,
            "bytes_uploaded": self.bytes_uploaded,
        }


def normalize_relative_path(value: str) -> PurePosixPath:
    """Return a safe project-relative POSIX path."""
    candidate = PurePosixPath(value.replace("\\", "/"))
    if candidate.is_absolute():
        raise DriveSyncError(f"Absolute paths are not allowed: {value}")
    if not candidate.parts or candidate == PurePosixPath("."):
        raise DriveSyncError("A non-empty relative path is required.")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise DriveSyncError(f"Unsafe relative path: {value}")
    return candidate


def assert_allowed_path(path: PurePosixPath, allowed_roots: set[str]) -> None:
    """Require the path to live under an explicitly approved top-level root."""
    if not allowed_roots:
        raise DriveSyncError("At least one allowed root is required.")
    if path.parts[0] not in allowed_roots:
        raise DriveSyncError(
            f"Path '{path}' is outside allowed roots: {sorted(allowed_roots)}"
        )


def unique_named_child(items: Iterable[RemoteItem], name: str) -> RemoteItem | None:
    """Return one exact-name child, failing closed on duplicate names."""
    matches = [item for item in items if item.name == name]
    if len(matches) > 1:
        raise DriveSyncError(
            f"Drive folder contains duplicate children named '{name}'. "
            "Resolve the ambiguity before syncing."
        )
    return matches[0] if matches else None


def validate_remote_name(name: str) -> None:
    """Require a Drive child name to be one safe local path component."""
    if not name or name in {".", ".."}:
        raise DriveSyncError(f"Unsafe Drive child name: {name!r}")
    if "/" in name or "\\" in name or "\x00" in name:
        raise DriveSyncError(f"Unsafe Drive child name: {name!r}")


def validate_remote_children(items: Iterable[RemoteItem]) -> list[RemoteItem]:
    """Validate sibling names and fail closed on duplicates."""
    materialized = list(items)
    seen: set[str] = set()
    for item in materialized:
        validate_remote_name(item.name)
        if item.name in seen:
            raise DriveSyncError(
                f"Drive folder contains duplicate children named '{item.name}'. "
                "Resolve the ambiguity before syncing."
            )
        seen.add(item.name)
    return materialized


def safe_local_child(parent: Path, name: str, containment_root: Path) -> Path:
    """Build a child destination and prove it stays within the download root."""
    validate_remote_name(name)
    if parent.is_symlink():
        raise DriveSyncError(f"Local download path contains symlink: {parent}")
    candidate = parent / name
    if candidate.is_symlink():
        raise DriveSyncError(f"Local download destination is a symlink: {candidate}")
    resolved = candidate.resolve(strict=False)
    root = containment_root.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise DriveSyncError(f"Drive child name escapes local download root: {name!r}")
    return candidate


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - used for change detection, not security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_credentials_from_env():
    """Build Drive credentials from environment variables.

    ``GOOGLE_SERVICE_ACCOUNT_JSON`` (a service account key) is used when set.
    Otherwise user OAuth: ``GOOGLE_CLIENT_ID``, ``GOOGLE_CLIENT_SECRET`` and
    ``GOOGLE_REFRESH_TOKEN``. A service account has no storage of its own, so in a
    personal (non-Workspace) Drive it can update existing files but not create new
    ones; user OAuth has no such limit.
    """
    key = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if key:
        import json

        from google.oauth2 import service_account

        return service_account.Credentials.from_service_account_info(
            json.loads(key), scopes=[DRIVE_SCOPE]
        )
    required = {
        "GOOGLE_CLIENT_ID": os.environ.get("GOOGLE_CLIENT_ID"),
        "GOOGLE_CLIENT_SECRET": os.environ.get("GOOGLE_CLIENT_SECRET"),
        "GOOGLE_REFRESH_TOKEN": os.environ.get("GOOGLE_REFRESH_TOKEN"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise DriveSyncError(
            "Missing required Google OAuth environment variables: " + ", ".join(missing)
        )

    credentials = Credentials(
        token=None,
        refresh_token=required["GOOGLE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=required["GOOGLE_CLIENT_ID"],
        client_secret=required["GOOGLE_CLIENT_SECRET"],
        scopes=[DRIVE_SCOPE],
    )
    credentials.refresh(GoogleAuthRequest())
    return credentials


class DriveClient:
    """Minimal Drive API wrapper with no delete operation."""

    def __init__(self, service) -> None:
        self.service = service

    def list_children(self, folder_id: str) -> list[RemoteItem]:
        items: list[RemoteItem] = []
        page_token: str | None = None
        while True:
            response = (
                self.service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    fields=(
                        "nextPageToken," "files(id,name,mimeType,md5Checksum,size)"
                    ),
                    pageSize=1000,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute(num_retries=DRIVE_API_RETRIES)
            )
            for raw in response.get("files", []):
                size = raw.get("size")
                items.append(
                    RemoteItem(
                        id=raw["id"],
                        name=raw["name"],
                        mime_type=raw["mimeType"],
                        md5_checksum=raw.get("md5Checksum"),
                        size=int(size) if size is not None else None,
                    )
                )
            page_token = response.get("nextPageToken")
            if not page_token:
                return items

    def find_child(self, folder_id: str, name: str) -> RemoteItem | None:
        return unique_named_child(self.list_children(folder_id), name)

    def create_folder(self, parent_id: str, name: str) -> RemoteItem:
        response = (
            self.service.files()
            .create(
                body={
                    "name": name,
                    "mimeType": GOOGLE_FOLDER_MIME,
                    "parents": [parent_id],
                },
                fields="id,name,mimeType",
                supportsAllDrives=True,
            )
            .execute()
        )
        return RemoteItem(
            id=response["id"],
            name=response["name"],
            mime_type=response["mimeType"],
        )

    def download_file(self, file_id: str, destination: Path) -> None:
        """Download atomically so a failed transfer never truncates the target."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = self.service.files().get_media(
            fileId=file_id,
            supportsAllDrives=True,
        )
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".part",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                downloader = MediaIoBaseDownload(handle, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk(num_retries=DRIVE_API_RETRIES)
            os.replace(temp_path, destination)
        except Exception:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise

    def create_file(self, parent_id: str, local_path: Path, name: str) -> None:
        mime_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        media = MediaFileUpload(
            str(local_path),
            mimetype=mime_type,
            resumable=False,
        )
        (
            self.service.files()
            .create(
                body={"name": name, "parents": [parent_id]},
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
            .execute()
        )

    def update_file(self, file_id: str, local_path: Path, name: str) -> None:
        mime_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        media = MediaFileUpload(
            str(local_path),
            mimetype=mime_type,
            resumable=False,
        )
        (
            self.service.files()
            .update(
                fileId=file_id,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
            .execute(num_retries=DRIVE_API_RETRIES)
        )


class DriveSync:
    def __init__(
        self,
        client: DriveClient,
        drive_root_id: str,
        local_root: Path,
        allowed_roots: set[str],
    ) -> None:
        self.client = client
        self.drive_root_id = drive_root_id
        self.local_root = local_root.resolve()
        self.allowed_roots = allowed_roots
        self.stats = SyncStats()

    def _validated(self, value: str) -> PurePosixPath:
        path = normalize_relative_path(value)
        assert_allowed_path(path, self.allowed_roots)
        return path

    def _resolve_remote(
        self,
        rel_path: PurePosixPath,
        *,
        create_folders: bool,
    ) -> RemoteItem | None:
        current_id = self.drive_root_id
        current_item: RemoteItem | None = None

        for index, part in enumerate(rel_path.parts):
            child = self.client.find_child(current_id, part)
            is_last = index == len(rel_path.parts) - 1

            if child is None:
                if not create_folders:
                    return None
                child = self.client.create_folder(current_id, part)
                self.stats.folders_created += 1

            if not child.is_folder and not is_last:
                raise DriveSyncError(
                    f"Remote path component '{part}' is a file, not a folder."
                )
            current_item = child
            if child.is_folder:
                current_id = child.id

        return current_item

    def _ensure_remote_folder(self, rel_path: PurePosixPath) -> RemoteItem:
        item = self._resolve_remote(rel_path, create_folders=True)
        if item is None or not item.is_folder:
            raise DriveSyncError(f"Expected Drive folder at '{rel_path}'.")
        return item

    def download(self, value: str) -> SyncStats:
        rel_path = self._validated(value)
        remote = self._resolve_remote(rel_path, create_folders=False)
        if remote is None:
            raise DriveSyncError(f"Drive path does not exist: {rel_path}")

        local_path = self.local_root.joinpath(*rel_path.parts)
        containment_root = local_path.resolve(strict=False)
        if not containment_root.is_relative_to(self.local_root):
            raise DriveSyncError(
                f"Local download path escapes project root: {local_path}"
            )
        if local_path.is_symlink():
            raise DriveSyncError(f"Local download root is a symlink: {local_path}")

        if remote.is_folder:
            self._download_folder(remote, local_path, containment_root)
        else:
            self._download_one(remote, local_path)
        return self.stats

    def _download_folder(
        self,
        remote_folder: RemoteItem,
        local_dir: Path,
        containment_root: Path,
    ) -> None:
        if local_dir.is_symlink():
            raise DriveSyncError(f"Local download path contains symlink: {local_dir}")
        local_dir.mkdir(parents=True, exist_ok=True)
        children = validate_remote_children(self.client.list_children(remote_folder.id))
        for child in sorted(children, key=lambda item: item.name):
            destination = safe_local_child(
                local_dir,
                child.name,
                containment_root,
            )
            if child.is_folder:
                self._download_folder(child, destination, containment_root)
            else:
                self._download_one(child, destination)

    def _download_one(self, remote: RemoteItem, local_path: Path) -> None:
        if remote.mime_type.startswith(GOOGLE_NATIVE_PREFIX):
            raise DriveSyncError(
                f"Native Google Workspace file '{remote.name}' is not supported "
                "inside operational sync paths."
            )
        self.client.download_file(remote.id, local_path)
        self.stats.downloaded += 1
        self.stats.bytes_downloaded += local_path.stat().st_size

    def upload(self, value: str) -> SyncStats:
        rel_path = self._validated(value)
        local_path = self.local_root.joinpath(*rel_path.parts)

        if not local_path.exists():
            raise DriveSyncError(f"Local upload path does not exist: {local_path}")
        if local_path.is_symlink():
            raise DriveSyncError(f"Symlinks are not allowed: {local_path}")

        if local_path.is_dir():
            self._ensure_remote_folder(rel_path)
            for candidate in sorted(local_path.rglob("*")):
                if candidate.is_symlink():
                    raise DriveSyncError(f"Symlinks are not allowed: {candidate}")
                if candidate.is_file():
                    child_rel = PurePosixPath(
                        candidate.relative_to(self.local_root).as_posix()
                    )
                    self._upload_one(candidate, child_rel)
        else:
            self._upload_one(local_path, rel_path)

        return self.stats

    def _upload_one(self, local_path: Path, rel_path: PurePosixPath) -> None:
        parent_rel = rel_path.parent
        if parent_rel == PurePosixPath("."):
            raise DriveSyncError(
                "Uploads must live under an allowed Drive-owned root folder."
            )
        parent = self._ensure_remote_folder(parent_rel)
        existing = self.client.find_child(parent.id, rel_path.name)

        if existing is not None and existing.is_folder:
            raise DriveSyncError(
                f"Cannot upload file '{rel_path}': a Drive folder has that name."
            )

        local_checksum = file_md5(local_path)
        if existing is not None and existing.md5_checksum == local_checksum:
            self.stats.skipped_unchanged += 1
            return

        if existing is None:
            self.client.create_file(parent.id, local_path, rel_path.name)
            self.stats.created += 1
            self.stats.bytes_uploaded += local_path.stat().st_size
        else:
            self.client.update_file(existing.id, local_path, rel_path.name)
            self.stats.updated += 1
            self.stats.bytes_uploaded += local_path.stat().st_size


def local_conflicts(sync: DriveSync, paths: Iterable[str]) -> list[str]:
    """Requested download targets that already hold local data."""
    found = []
    for value in paths:
        target = sync.local_root.joinpath(*sync._validated(value).parts)
        if (
            target.is_symlink()
            or target.is_file()
            or (target.is_dir() and any(target.iterdir()))
        ):
            found.append(value)
    return found


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carina drive",
        description="Non-destructive sync between a Google Drive folder and a local directory."
    )
    parser.add_argument("action", choices=["download", "upload"])
    parser.add_argument("paths", nargs="+", help="Paths relative to the local root to sync.")
    parser.add_argument(
        "--drive-root-id",
        default=os.environ.get("DRIVE_PROJECT_FOLDER_ID"),
        help="Google Drive folder ID of the sync root.",
    )
    parser.add_argument("--local-root", default=".")
    parser.add_argument(
        "--overwrite-local",
        action="store_true",
        help=(
            "Allow a download to replace existing local files. Without it, a "
            "download refuses to touch a path that already holds local data."
        ),
    )
    parser.add_argument(
        "--allowed-root",
        action="append",
        dest="allowed_roots",
        default=[],
        help="Approved top-level path. Repeat for multiple roots.",
    )
    return parser


def connect(drive_root_id, local_root, allowed_roots) -> DriveSync:
    """A DriveSync for one Drive folder and local directory, using the environment's credentials."""
    if not drive_root_id:
        raise DriveSyncError(
            "Drive root ID is required via --drive-root-id or DRIVE_PROJECT_FOLDER_ID."
        )
    service = build(
        "drive",
        "v3",
        credentials=build_credentials_from_env(),
        cache_discovery=False,
    )
    return DriveSync(
        DriveClient(service),
        drive_root_id=drive_root_id,
        local_root=Path(local_root),
        allowed_roots=set(allowed_roots),
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sync = connect(args.drive_root_id, args.local_root, args.allowed_roots or ["data"])

    if args.action == "download" and not args.overwrite_local:
        conflicts = local_conflicts(sync, args.paths)
        if conflicts:
            raise DriveSyncError(
                "Refusing to download over existing local data: "
                + ", ".join(conflicts)
                + ". Drive may hold an older copy. Back up first, or pass "
                "--overwrite-local after comparing."
            )

    for path in args.paths:
        if args.action == "download":
            sync.download(path)
        else:
            sync.upload(path)

    print(sync.stats.as_dict())
    return 0

