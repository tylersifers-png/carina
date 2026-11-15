import json
import os
from pathlib import Path

import pandas as pd
import pytest

from tests.test_store import built
from carina import dataset


@pytest.fixture
def staged(built, tmp_path):
    _, _, candidate = built
    destination = tmp_path / "staged"
    manifest = dataset.convert(candidate, destination)
    return candidate, destination, manifest


def multi_year(rows_per_year=3):
    rows = []
    for year in (1998, 1999, 2000):
        for day in range(1, rows_per_year + 1):
            rows.append(dict(symbol="AAA", session_date=f"{year}-01-0{day}", Close=float(day)))
    return pd.DataFrame(rows)


def test_partitioned_table_round_trips_and_filters(tmp_path):
    frame = multi_year()
    written = dataset.write_table(tmp_path, "daily_bars", frame)
    assert written == [f"daily_bars/year={y}/part.parquet" for y in (1998, 1999, 2000)]
    again = dataset.read_table(tmp_path, "daily_bars")
    assert list(again.columns) == list(frame.columns)  # the partition column is hidden
    assert dataset.frame_digest(again, ["symbol", "session_date"]) == dataset.frame_digest(
        frame, ["symbol", "session_date"]
    )
    one = dataset.read_table(tmp_path, "daily_bars", filters=[("year", "=", 1999)])
    assert set(one.session_date.str[:4]) == {"1999"} and len(one) == 3
    assert "year" in dataset.read_table(tmp_path, "daily_bars", keep_partitions=True)


def test_unpartitioned_table_is_one_file(tmp_path):
    frame = pd.DataFrame(dict(symbol=["AAA"], name=["x"]))
    assert dataset.write_table(tmp_path, "companies", frame) == ["companies.parquet"]
    assert dataset.read_table(tmp_path, "companies").equals(frame)


def test_conversion_is_logically_equal_and_copies_unpartitioned_bytes(staged):
    candidate, destination, manifest = staged
    report = dataset.verify_equivalent(candidate, destination)
    assert all(item["equal"] for item in report.values())
    assert manifest["schema"] == dataset.SCHEMA and manifest["status"] == "VERIFIED"
    legacy = json.loads((candidate / "manifest.json").read_bytes())
    for name in ("companies.parquet", "fundamentals.parquet", "coverage.json", "sources.json"):
        assert manifest["files"][name] == legacy["files"][name]  # byte-identical copies
    assert "daily_bars.parquet" not in manifest["files"]
    assert any(path.startswith("daily_bars/year=") for path in manifest["files"])
    assert not (destination / "daily_bars.parquet").exists()
    assert "validation" not in manifest and "readback" not in manifest


def test_conversion_records_lineage_and_ancestry(staged):
    candidate, destination, manifest = staged
    parent = dataset.manifest_digest(candidate)
    assert manifest["lineage"] == [parent]
    assert manifest["converted_from"]["manifest_sha256"] == parent
    head = dataset.manifest_digest(destination)
    assert dataset.descends(manifest, head, parent)
    assert dataset.descends(manifest, head, head)
    assert not dataset.descends(manifest, head, "0" * 64)


def test_conversion_refuses_a_non_empty_destination(built, tmp_path):
    _, _, candidate = built
    (tmp_path / "occupied").mkdir()
    (tmp_path / "occupied/x").write_text("x")
    with pytest.raises(ValueError, match="must be empty"):
        dataset.convert(candidate, tmp_path / "occupied")


def reload(destination):
    return json.loads((destination / "manifest.json").read_bytes())


def test_verify_accepts_the_converted_dataset(staged):
    _, destination, _ = staged
    dataset.verify_files(destination, reload(destination))


def test_verify_detects_a_changed_partition(staged):
    _, destination, manifest = staged
    path = destination / next(p for p in manifest["files"] if p.startswith("daily_bars/"))
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        dataset.verify_files(destination, reload(destination))


def test_verify_detects_extra_and_missing_files(staged):
    _, destination, manifest = staged
    (destination / "stray.txt").write_text("stray")
    with pytest.raises(ValueError, match="Unexpected/missing"):
        dataset.verify_files(destination, manifest)
    (destination / "stray.txt").unlink()
    victim = destination / next(p for p in manifest["files"] if p.startswith("daily_bars/"))
    victim.unlink()
    with pytest.raises(ValueError, match="Unexpected/missing"):
        dataset.verify_files(destination, manifest)


def test_verify_rejects_a_symlink_anywhere(staged, tmp_path):
    _, destination, manifest = staged
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"x")
    partition = destination / next(p for p in manifest["files"] if p.startswith("daily_bars/"))
    partition.unlink()
    os.symlink(outside, partition)
    with pytest.raises(ValueError, match="symlink"):
        dataset.verify_files(destination, manifest)


@pytest.mark.parametrize("bad", ["../escape.parquet", "/abs.parquet", "a//b.parquet", "./x.parquet", "manifest.json", ""])
def test_unsafe_manifest_paths_are_rejected(staged, bad):
    _, destination, manifest = staged
    broken = dict(manifest, files={**manifest["files"], bad: dict(sha256="0" * 64, bytes=0)})
    with pytest.raises(ValueError, match="Unsafe dataset path"):
        dataset.verify_files(destination, broken)


def test_a_missing_member_is_rejected(staged):
    _, destination, manifest = staged
    files = {k: v for k, v in manifest["files"].items() if not k.startswith("filings")}
    with pytest.raises(ValueError, match="Unexpected dataset members"):
        dataset.verify_files(destination, dict(manifest, files=files))


def test_equivalence_check_catches_a_changed_row(staged):
    candidate, destination, manifest = staged
    partition = destination / next(p for p in manifest["files"] if p.startswith("daily_bars/"))
    frame = pd.read_parquet(partition)
    frame.loc[0, "Close"] = frame.loc[0, "Close"] + 1
    frame.to_parquet(partition, index=False)
    with pytest.raises(ValueError, match="differ logically: daily_bars"):
        dataset.verify_equivalent(candidate, destination)


# --- promotion -------------------------------------------------------------


def second_candidate(candidate, base_digest, tmp_path, delta=1.0, name="candidate2"):
    """A flat candidate with one changed Close value, built from ``base_digest``."""
    import shutil

    from carina import common as sec

    target = tmp_path / name
    shutil.copytree(candidate, target)
    frame = pd.read_parquet(target / "daily_bars.parquet")
    frame.loc[0, "Close"] = frame.loc[0, "Close"] + delta
    frame.to_parquet(target / "daily_bars.parquet", index=False)
    manifest = json.loads((target / "manifest.json").read_bytes())
    raw = (target / "daily_bars.parquet").read_bytes()
    manifest["files"]["daily_bars.parquet"] = dict(sha256=sec.digest(raw), bytes=len(raw))
    manifest["base_manifest_sha256"] = base_digest
    (target / "manifest.json").write_bytes(sec.encoded(manifest))
    return target


def test_first_promotion_has_no_parents_and_the_next_records_its_parent(built, tmp_path):
    _, _, candidate = built
    staged = tmp_path / "ds" / "staged"
    first = dataset.promote(candidate, staged)
    assert first["lineage"] == [] and (staged / "manifest.json").exists()
    head1 = dataset.manifest_digest(staged)
    second = dataset.promote(second_candidate(candidate, head1, tmp_path), staged)
    assert second["lineage"] == [head1]
    head2 = dataset.manifest_digest(staged)
    assert head2 != head1 and dataset.descends(second, head2, head1)
    assert not (tmp_path / "ds" / "staged.prev").exists()
    assert not (tmp_path / "ds" / "staged.new").exists()
    dataset.verify_files(staged, json.loads((staged / "manifest.json").read_bytes()))


def test_promotion_refuses_a_candidate_built_from_a_stale_head(built, tmp_path):
    _, _, candidate = built
    staged = tmp_path / "ds" / "staged"
    dataset.promote(candidate, staged)
    before = dataset.manifest_digest(staged)
    stale = second_candidate(candidate, "0" * 64, tmp_path)
    with pytest.raises(ValueError, match="changed since the candidate was built"):
        dataset.promote(stale, staged)
    assert dataset.manifest_digest(staged) == before


def test_promotion_refuses_to_build_on_a_damaged_dataset(built, tmp_path):
    _, _, candidate = built
    staged = tmp_path / "ds" / "staged"
    manifest = dataset.promote(candidate, staged)
    victim = staged / next(p for p in manifest["files"] if p.startswith("daily_bars/"))
    victim.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum mismatch"):
        dataset.promote(candidate, staged)


def test_failed_promotion_leaves_the_old_dataset_in_place(built, tmp_path, monkeypatch):
    _, _, candidate = built
    staged = tmp_path / "ds" / "staged"
    dataset.promote(candidate, staged)
    head1 = dataset.manifest_digest(staged)
    real = dataset.verify_files
    calls = []

    def flaky(root, manifest):
        calls.append(root)
        if Path(root).name == "staged" and len(calls) > 3:
            raise ValueError("simulated post-swap verification failure")
        return real(root, manifest)

    monkeypatch.setattr(dataset, "verify_files", flaky)
    with pytest.raises(ValueError, match="simulated"):
        dataset.promote(second_candidate(candidate, head1, tmp_path), staged)
    monkeypatch.setattr(dataset, "verify_files", real)
    assert dataset.manifest_digest(staged) == head1
    dataset.verify_files(staged, json.loads((staged / "manifest.json").read_bytes()))


def test_recover_restores_the_old_dataset_after_an_interrupted_swap(built, tmp_path):
    _, _, candidate = built
    staged = tmp_path / "ds" / "staged"
    dataset.promote(candidate, staged)
    head = dataset.manifest_digest(staged)
    os.rename(staged, tmp_path / "ds" / "staged.prev")  # crashed between the renames
    assert not staged.exists()
    dataset.recover(staged)
    assert dataset.manifest_digest(staged) == head
