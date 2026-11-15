from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

import pytest

from carina.drive import (
    DriveClient,
    DriveSync,
    DriveSyncError,
    DRIVE_API_RETRIES,
    RemoteItem,
    assert_allowed_path,
    normalize_relative_path,
    unique_named_child,
)


class FakeDriveClient:
    def __init__(self) -> None:
        self.children: dict[str, list[RemoteItem]] = {"root": []}
        self.created_files: list[tuple[str, str]] = []
        self.updated_files: list[tuple[str, str]] = []
        self.downloaded_files: list[tuple[str, Path]] = []
        self._counter = 0

    def _id(self) -> str:
        self._counter += 1
        return f"id-{self._counter}"

    def list_children(self, folder_id: str) -> list[RemoteItem]:
        return list(self.children.get(folder_id, []))

    def find_child(self, folder_id: str, name: str) -> RemoteItem | None:
        return unique_named_child(self.list_children(folder_id), name)

    def create_folder(self, parent_id: str, name: str) -> RemoteItem:
        item = RemoteItem(
            id=self._id(),
            name=name,
            mime_type="application/vnd.google-apps.folder",
        )
        self.children.setdefault(parent_id, []).append(item)
        self.children[item.id] = []
        return item

    def create_file(self, parent_id: str, local_path: Path, name: str) -> None:
        item = RemoteItem(
            id=self._id(),
            name=name,
            mime_type="text/plain",
            md5_checksum=hashlib.md5(local_path.read_bytes()).hexdigest(),
        )
        self.children.setdefault(parent_id, []).append(item)
        self.created_files.append((parent_id, name))

    def update_file(self, file_id: str, local_path: Path, name: str) -> None:
        self.updated_files.append((file_id, name))

    def download_file(self, file_id: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"downloaded")
        self.downloaded_files.append((file_id, destination))


def test_normalize_relative_path_rejects_escape() -> None:
    with pytest.raises(DriveSyncError):
        normalize_relative_path("../state/raw")


def test_allowed_root_fails_closed() -> None:
    path = normalize_relative_path("scratch/monthly")
    with pytest.raises(DriveSyncError):
        assert_allowed_path(path, {"state"})


def test_duplicate_drive_names_fail_closed() -> None:
    items = [
        RemoteItem("1", "series.csv", "text/csv"),
        RemoteItem("2", "series.csv", "text/csv"),
    ]
    with pytest.raises(DriveSyncError, match="duplicate"):
        unique_named_child(items, "series.csv")


def test_recursive_download_rejects_duplicate_siblings(tmp_path: Path) -> None:
    client = FakeDriveClient()
    data = client.create_folder("root", "state")
    raw = client.create_folder(data.id, "raw")
    client.children[raw.id].extend(
        [
            RemoteItem("file-1", "same.csv", "text/csv"),
            RemoteItem("file-2", "same.csv", "text/csv"),
        ]
    )

    sync = DriveSync(client, "root", tmp_path, {"state"})
    with pytest.raises(DriveSyncError, match="duplicate"):
        sync.download("state/raw")

    assert not (tmp_path / "state" / "raw" / "same.csv").exists()


def test_recursive_download_rejects_escaping_remote_name(tmp_path: Path) -> None:
    client = FakeDriveClient()
    data = client.create_folder("root", "state")
    raw = client.create_folder(data.id, "raw")
    client.children[raw.id].append(
        RemoteItem("file-1", "../../outside/escape.csv", "text/csv")
    )

    sync = DriveSync(client, "root", tmp_path, {"state"})
    with pytest.raises(DriveSyncError, match="Unsafe Drive child name"):
        sync.download("state/raw")

    assert not (tmp_path.parent / "outside/escape.csv").exists()


def test_failed_download_preserves_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "state" / "raw" / "series" / "series.csv"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old-good-data")

    class FakeFilesApi:
        def get_media(self, **kwargs):
            return object()

    class FakeService:
        def files(self):
            return FakeFilesApi()

    retry_calls = []

    class FailingDownloader:
        def __init__(self, handle, request):
            self.handle = handle

        def next_chunk(self, **kwargs):
            retry_calls.append(kwargs)
            self.handle.write(b"partial")
            raise RuntimeError("network failure")

    monkeypatch.setattr("carina.drive.MediaIoBaseDownload", FailingDownloader)

    client = DriveClient(FakeService())
    with pytest.raises(RuntimeError, match="network failure"):
        client.download_file("file-1", destination)

    assert destination.read_bytes() == b"old-good-data"
    assert list(destination.parent.glob("*.part")) == []
    assert retry_calls == [{"num_retries": DRIVE_API_RETRIES}]


def test_upload_creates_only_missing_paths(tmp_path: Path) -> None:
    target = tmp_path / "state" / "raw" / "series" / "series.csv"
    target.parent.mkdir(parents=True)
    target.write_text("DATE,VALUE\n2026-08-01,22000\n", encoding="utf-8")

    client = FakeDriveClient()
    sync = DriveSync(client, "root", tmp_path, {"state"})
    stats = sync.upload("state/raw/series/series.csv")

    assert stats.folders_created == 3
    assert stats.created == 1
    assert stats.updated == 0
    assert stats.bytes_uploaded == target.stat().st_size
    assert client.created_files
    assert not hasattr(client, "delete_file")


def test_upload_skips_unchanged_file(tmp_path: Path) -> None:
    target = tmp_path / "state" / "raw" / "series" / "series.csv"
    target.parent.mkdir(parents=True)
    target.write_text("unchanged", encoding="utf-8")
    checksum = hashlib.md5(target.read_bytes()).hexdigest()

    client = FakeDriveClient()
    data = client.create_folder("root", "state")
    raw = client.create_folder(data.id, "raw")
    m2 = client.create_folder(raw.id, "series")
    client.children[m2.id].append(
        RemoteItem("file-1", "series.csv", "text/csv", md5_checksum=checksum)
    )

    sync = DriveSync(client, "root", tmp_path, {"state"})
    stats = sync.upload("state/raw/series/series.csv")

    assert stats.skipped_unchanged == 1
    assert stats.created == 0
    assert stats.updated == 0
    assert client.updated_files == []


def test_upload_updates_changed_file_without_replacing_id(tmp_path: Path) -> None:
    target = tmp_path / "state" / "raw" / "series" / "series.csv"
    target.parent.mkdir(parents=True)
    target.write_text("new value", encoding="utf-8")

    client = FakeDriveClient()
    data = client.create_folder("root", "state")
    raw = client.create_folder(data.id, "raw")
    m2 = client.create_folder(raw.id, "series")
    client.children[m2.id].append(
        RemoteItem("stable-file-id", "series.csv", "text/csv", md5_checksum="old")
    )

    sync = DriveSync(client, "root", tmp_path, {"state"})
    stats = sync.upload("state/raw/series/series.csv")

    assert stats.updated == 1
    assert stats.created == 0
    assert client.updated_files == [("stable-file-id", "series.csv")]


def test_download_hydrates_recursive_folder(tmp_path: Path) -> None:
    client = FakeDriveClient()
    data = client.create_folder("root", "state")
    raw = client.create_folder(data.id, "raw")
    m2 = client.create_folder(raw.id, "series")
    client.children[m2.id].append(RemoteItem("file-1", "series.csv", "text/csv"))

    sync = DriveSync(client, "root", tmp_path, {"state"})
    stats = sync.download("state/raw")

    assert stats.downloaded == 1
    assert stats.bytes_downloaded == len(b"downloaded")
    assert (tmp_path / "state" / "raw" / "series" / "series.csv").read_bytes() == b"downloaded"


def test_google_native_files_are_rejected(tmp_path: Path) -> None:
    client = FakeDriveClient()
    data = client.create_folder("root", "state")
    raw = client.create_folder(data.id, "raw")
    client.children[raw.id].append(
        RemoteItem(
            "native-1",
            "unexpected-sheet",
            "application/vnd.google-apps.spreadsheet",
        )
    )

    sync = DriveSync(client, "root", tmp_path, {"state"})
    with pytest.raises(DriveSyncError, match="Native Google Workspace"):
        sync.download("state/raw")


def _remote_state(drive_root_files=("state/ops/data/raw/a.csv",)) -> FakeDriveClient:
    client = FakeDriveClient()
    parent = "root"
    created: dict[str, str] = {}
    for rel in drive_root_files:
        parent = "root"
        for part in PurePosixPath(rel).parts[:-1]:
            key = f"{parent}/{part}"
            if key not in created:
                created[key] = client.create_folder(parent, part).id
            parent = created[key]
        client.children.setdefault(parent, []).append(
            RemoteItem(
                id=f"file-{rel}",
                name=PurePosixPath(rel).name,
                mime_type="text/csv",
                md5_checksum=hashlib.md5(b"drive").hexdigest(),
            )
        )
    return client


def test_download_guard_lists_only_paths_that_hold_local_data(tmp_path: Path) -> None:
    from carina.drive import local_conflicts

    sync = DriveSync(_remote_state(), "root", tmp_path, {"state"})
    assert local_conflicts(sync, ["state"]) == []
    (tmp_path / "state").mkdir()
    assert local_conflicts(sync, ["state"]) == []  # an empty directory holds nothing
    (tmp_path / "state/local.csv").write_text("local")
    assert local_conflicts(sync, ["state"]) == ["state"]
    assert local_conflicts(sync, ["state/ops/data/raw"]) == []
    with pytest.raises(DriveSyncError):
        local_conflicts(sync, ["../outside"])


@pytest.fixture
def cli(monkeypatch, tmp_path):
    import carina.drive as module

    client = _remote_state()
    monkeypatch.setattr(module, "build_credentials_from_env", lambda: None)
    monkeypatch.setattr(module, "build", lambda *a, **k: object())
    monkeypatch.setattr(module, "DriveClient", lambda service: client)

    def run(*extra):
        monkeypatch.setattr(
            "sys.argv",
            ["drive_sync.py", "--drive-root-id", "root", "--local-root", str(tmp_path),
             "--allowed-root", "state", *extra, "download", "state"],
        )
        return module.main()

    return run, client


def test_cli_download_refuses_to_overwrite_local_state(cli, tmp_path: Path) -> None:
    run, client = cli
    local = tmp_path / "state/ops/data/raw/a.csv"
    local.parent.mkdir(parents=True)
    local.write_text("local newer data")
    with pytest.raises(DriveSyncError, match="Refusing to download over existing local"):
        run()
    assert local.read_text() == "local newer data" and client.downloaded_files == []


def test_cli_download_into_an_empty_target_and_with_override(cli, tmp_path: Path) -> None:
    run, client = cli
    assert run() == 0 and len(client.downloaded_files) == 1
    (tmp_path / "state/ops/data/raw/a.csv").write_text("changed locally")
    with pytest.raises(DriveSyncError):
        run()
    assert run("--overwrite-local") == 0
