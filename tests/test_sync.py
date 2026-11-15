import json

import pytest

from tests.test_store import MutableDrive, built  # noqa: F401 (fixture)
from tests.test_dataset import second_candidate
from carina import dataset, sync as staged_sync
from carina.drive import DriveSync, DriveSyncError

STAGED = staged_sync.STAGED


def sync_for(root, drive):
    return DriveSync(drive, "root", root, {"staged"})


@pytest.fixture
def local(built, tmp_path):
    _, _, candidate = built
    root = tmp_path / "local"
    dataset.promote(candidate, root / STAGED)
    return root, candidate


def advance(root, candidate, tmp_path, delta, name):
    head = dataset.manifest_digest(root / STAGED)
    dataset.promote(second_candidate(candidate, head, tmp_path, delta, name), root / STAGED)


def find_id(drive, name):
    return next(i.id for items in drive.children.values() for i in items if i.name == name)


def test_empty_and_one_sided_states(local, tmp_path):
    root, _ = local
    assert staged_sync.status(sync_for(tmp_path / "nothing", MutableDrive()))["relation"] == "both_empty"
    assert staged_sync.status(sync_for(root, MutableDrive()))["relation"] == "remote_empty"


def test_first_push_uploads_the_manifest_last_and_is_then_in_sync(local):
    root, _ = local
    drive = MutableDrive()
    sync = sync_for(root, drive)
    result = staged_sync.push(sync)
    assert result["relation"] == "in_sync" and result["pushed"] > 1
    assert drive.created_files[-1][1] == "manifest.json"
    assert staged_sync.status(sync)["relation"] == "in_sync"
    uploads = len(drive.created_files) + len(drive.updated_files)
    assert staged_sync.push(sync)["pushed"] == 0
    assert len(drive.created_files) + len(drive.updated_files) == uploads


def test_fast_forward_push_updates_only_what_changed(local, tmp_path):
    root, candidate = local
    drive = MutableDrive()
    staged_sync.push(sync_for(root, drive))
    created = len(drive.created_files)
    advance(root, candidate, tmp_path, 1.0, "c2")
    sync = sync_for(root, drive)
    assert staged_sync.status(sync)["relation"] == "local_ahead"
    staged_sync.push(sync)
    assert len(drive.created_files) == created  # nothing new was created
    names = {name for _, name in drive.updated_files}
    assert "manifest.json" in names and len(names) <= 4
    assert staged_sync.status(sync)["relation"] == "in_sync"


def test_pull_into_an_empty_root_reproduces_the_dataset(local, tmp_path):
    root, _ = local
    drive = MutableDrive()
    staged_sync.push(sync_for(root, drive))
    other = tmp_path / "other"
    other.mkdir()
    sync = sync_for(other, drive)
    assert staged_sync.status(sync)["relation"] == "remote_ahead"
    staged_sync.pull(sync)
    assert dataset.manifest_digest(other / STAGED) == dataset.manifest_digest(root / STAGED)
    dataset.verify_files(other / STAGED, json.loads((other / STAGED / "manifest.json").read_bytes()))
    assert staged_sync.status(sync)["relation"] == "in_sync"
    assert not (other / "scratch" / "pull").exists()


def test_pull_fast_forwards_a_behind_local_copy(local, tmp_path):
    root, candidate = local
    drive = MutableDrive()
    staged_sync.push(sync_for(root, drive))
    behind = tmp_path / "behind"
    staged_sync.pull(sync_for(behind, drive))
    advance(root, candidate, tmp_path, 1.0, "c2")
    staged_sync.push(sync_for(root, drive))
    sync = sync_for(behind, drive)
    assert staged_sync.status(sync)["relation"] == "remote_ahead"
    staged_sync.pull(sync)
    assert dataset.manifest_digest(behind / STAGED) == dataset.manifest_digest(root / STAGED)


def test_a_stale_writer_cannot_push_or_pull_and_nothing_changes(local, tmp_path):
    root, candidate = local
    drive = MutableDrive()
    staged_sync.push(sync_for(root, drive))
    other = tmp_path / "other"
    staged_sync.pull(sync_for(other, drive))
    advance(root, candidate, tmp_path, 1.0, "c_a")
    staged_sync.push(sync_for(root, drive))
    advance(other, candidate, tmp_path, 5.0, "c_b")  # a different second head
    sync = sync_for(other, drive)
    assert staged_sync.status(sync)["relation"] == "diverged"
    before = (len(drive.created_files), len(drive.updated_files))
    local_head = dataset.manifest_digest(other / STAGED)
    for action in (staged_sync.push, staged_sync.pull):
        with pytest.raises(DriveSyncError, match="Refusing to .*: diverged"):
            action(sync)
    assert (len(drive.created_files), len(drive.updated_files)) == before
    assert dataset.manifest_digest(other / STAGED) == local_head


def test_direction_is_enforced(local, tmp_path):
    root, candidate = local
    drive = MutableDrive()
    staged_sync.push(sync_for(root, drive))
    behind = tmp_path / "behind"
    staged_sync.pull(sync_for(behind, drive))
    advance(root, candidate, tmp_path, 1.0, "c2")
    with pytest.raises(DriveSyncError, match="Refusing to pull: local_ahead"):
        staged_sync.pull(sync_for(root, drive))
    staged_sync.push(sync_for(root, drive))
    with pytest.raises(DriveSyncError, match="Refusing to push: remote_ahead"):
        staged_sync.push(sync_for(behind, drive))


def test_push_aborts_without_committing_if_drive_moves_meanwhile(local, monkeypatch):
    root, _ = local
    drive = MutableDrive()
    sync = sync_for(root, drive)
    real = staged_sync.read_remote_head
    calls = []

    def moving(sync_arg, relative):
        calls.append(relative)
        if len(calls) >= 2 and relative.endswith("manifest.json"):  # the commit check
            return ({"lineage": []}, "f" * 64)
        return real(sync_arg, relative)

    monkeypatch.setattr(staged_sync, "read_remote_head", moving)
    with pytest.raises(DriveSyncError, match="Drive changed during the push"):
        staged_sync.push(sync)
    assert not any(name == "manifest.json" for _, name in drive.created_files)


def test_an_interrupted_push_leaves_drive_without_a_new_head(local):
    root, _ = local
    drive = MutableDrive()
    drive.fail_upload = "manifest.json"
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        staged_sync.push(sync_for(root, drive))
    drive.fail_upload = None
    sync = sync_for(root, drive)
    assert staged_sync.status(sync)["relation"] == "remote_empty"  # still no head
    staged_sync.push(sync)  # the retry completes
    assert staged_sync.status(sync)["relation"] == "in_sync"


def test_a_tampered_download_is_rejected_and_local_is_untouched(local, tmp_path):
    root, _ = local
    drive = MutableDrive()
    staged_sync.push(sync_for(root, drive))
    drive.content[find_id(drive, "part.parquet")] = b"tampered"
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="checksum mismatch"):
        staged_sync.pull(sync_for(other, drive))
    assert not (other / STAGED).exists() and not (other / "scratch" / "pull").exists()
