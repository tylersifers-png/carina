from datetime import datetime, timedelta, timezone
import json
import secrets

import pytest

from carina import common

URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"


class Response:
    def __init__(self, status, headers=None, body=b"{}"):
        self.status_code = status
        self.headers = headers or {}
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, size):
        return [self._body]


def test_403_stops_without_retry_or_host_rotation(monkeypatch):
    seen = []

    def get(url, **kwargs):
        seen.append((url, kwargs))
        return Response(403, body=b"Denied")

    monkeypatch.setattr(common.requests, "get", get)
    raw, attempts = common.fetch(URL, "Project contact@example.com")
    assert raw is None and len(attempts) == 1
    assert len(seen) == 1
    assert seen[0][1]["allow_redirects"] is False


def test_large_retry_after_defers_instead_of_shortening_wait(monkeypatch):
    monkeypatch.setattr(
        common.requests,
        "get",
        lambda *a, **k: Response(429, {"Retry-After": "120"}, b"Rate limited"),
    )
    monkeypatch.setattr(
        common.time, "sleep", lambda x: pytest.fail("Must defer to a future run")
    )
    assert common.fetch(URL, "Project")[0] is None


def test_transient_response_retries_same_url_and_identity(monkeypatch):
    statuses = iter([429, 200])
    seen, waits = [], []

    def get(url, **kwargs):
        seen.append((url, kwargs["headers"]["User-Agent"]))
        return Response(next(statuses), {"Retry-After": "1"})

    monkeypatch.setattr(common.requests, "get", get)
    monkeypatch.setattr(common.time, "sleep", waits.append)
    raw, attempts = common.fetch(URL, "Project")
    assert raw == b"{}" and [a["http_status"] for a in attempts] == [429, 200]
    assert seen == [(URL, "Project")] * 2
    assert waits == [1]


def validate(content):
    if content.get("schema") != "test-capture":
        raise ValueError("Unexpected capture schema")
    return content


@pytest.fixture
def capture(tmp_path):
    # Random hex does not compress well, so the capture spans several pieces.
    content = {"schema": "test-capture", "payload": secrets.token_hex(120_000)}
    common.prepare_capture(tmp_path, content, validator=validate)
    bundle = json.loads((tmp_path / "handoff.json").read_bytes())["bundle"]
    return tmp_path, content, bundle


def test_capture_round_trips_through_pieces(capture):
    path, content, bundle = capture
    assert len(list(path.glob("part_*.json"))) >= 2
    assert common.load(path, bundle, validator=validate) == content


def test_missing_piece_fails_closed(capture):
    path, _, bundle = capture
    (path / "part_01.json").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        common.load(path, bundle, validator=validate)


def test_extra_piece_fails_closed(capture):
    path, _, bundle = capture
    (path / "part_09.json").write_bytes((path / "part_00.json").read_bytes())
    with pytest.raises(ValueError, match="Incomplete|Conflicting"):
        common.load(path, bundle, validator=validate)


def test_corrupt_piece_fails_checksum(capture):
    path, _, bundle = capture
    piece_path = path / "part_00.json"
    piece = json.loads(piece_path.read_bytes())
    first = "B" if piece["data"][0] == "A" else "A"
    piece["data"] = first + piece["data"][1:]
    piece_path.write_bytes(common.encoded(piece))
    with pytest.raises(ValueError):
        common.load(path, bundle, validator=validate)


def test_wrong_bundle_identity_is_rejected(capture):
    path, _, _ = capture
    with pytest.raises(ValueError, match="Conflicting"):
        common.load(path, "0" * 64, validator=validate)


def test_validator_failure_writes_nothing(tmp_path):
    with pytest.raises(ValueError, match="schema"):
        common.prepare_capture(tmp_path, {"schema": "other"}, validator=validate)
    assert not list(tmp_path.iterdir())


def test_oversized_capture_is_refused(tmp_path):
    content = {"schema": "test-capture", "payload": "x" * 100}
    with pytest.raises(ValueError, match="bounded raw size"):
        common.prepare_capture(tmp_path, content, max_bundle=10, validator=validate)


def test_identity_and_folder_reject_unsafe_values(tmp_path):
    with pytest.raises(ValueError):
        common.identity("not-a-digest")
    bundle = "a" * 64
    assert common.folder(tmp_path, bundle, "base") == tmp_path / "base" / bundle
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "base").symlink_to(outside)
    with pytest.raises(ValueError, match="Unsafe"):
        common.folder(root, bundle, "base")


def test_write_refuses_a_symlink_target(tmp_path):
    target = tmp_path / "real.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        common.write(link, b"{}")


def receipt_for(raw, **changes):
    now = datetime.now(timezone.utc)
    receipt = {
        "source_url": URL,
        "http_status": 200,
        "requested_at": (now - timedelta(seconds=2)).isoformat(),
        "retrieved_at": (now - timedelta(seconds=1)).isoformat(),
        "bytes": len(raw),
        "sha256": common.digest(raw),
    }
    return {**receipt, **changes}


def test_receipt_valid_accepts_a_matching_capture():
    raw = b'{"ok": true}'
    common.receipt_valid(receipt_for(raw), raw, URL)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"source_url": "https://example.com/other"}, "URL/status"),
        ({"http_status": 500}, "URL/status"),
        ({"sha256": "0" * 64}, "hash/size"),
        ({"bytes": 1}, "hash/size"),
        (
            {"retrieved_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()},
            "timestamp",
        ),
    ],
)
def test_receipt_valid_rejects_mismatches(changes, message):
    raw = b'{"ok": true}'
    with pytest.raises(ValueError, match=message):
        common.receipt_valid(receipt_for(raw, **changes), raw, URL)


def test_last_completed_session_respects_the_nyse_close():
    # Friday 2026-11-13 closes at 21:00 UTC (EST).
    after_close = datetime(2026, 11, 13, 22, 0, tzinfo=timezone.utc)
    before_close = datetime(2026, 11, 13, 15, 0, tzinfo=timezone.utc)
    assert common.last_completed(after_close) == "2026-11-13"
    assert common.last_completed(before_close) == "2026-11-12"
