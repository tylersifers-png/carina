import json
import pandas as pd
import pytest

from carina import store, updates as update, refresh
from tests.test_store import built  # noqa: F401 (fixture)
from tests.test_updates import state, subset, capture


def test_build_checks_expansion_before_any_market_fetch(state, monkeypatch):
    subset(state, "MSFT")

    def blocked(plan):
        raise ValueError("EXPANSION_BLOCKED: test")

    monkeypatch.setattr(update, "check_expansion", blocked)
    bundle, _ = capture(state, update.request(["MSFT"]))
    with pytest.raises(ValueError, match="EXPANSION_BLOCKED"):
        store.build(
            state,
            bundle,
            lambda *a: pytest.fail("No Yahoo request before the expansion check"),
        )


def test_promotion_recomputes_block_flags_before_anything_is_written(built):
    root, bundle, path = built
    frame = pd.read_parquet(path / "financial_latest.parquet")
    frame.loc[0, "value"] = "invalid numeric"
    frame.to_parquet(path / "financial_latest.parquet", index=False)
    manifest = json.loads((path / "manifest.json").read_bytes())
    raw = (path / "financial_latest.parquet").read_bytes()
    manifest["files"]["financial_latest.parquet"] = dict(
        sha256=store.sec.digest(raw), bytes=len(raw)
    )
    # Older snapshots lack quality metadata; the publisher must still assess them.
    manifest.pop("quality", None)
    store.replace_generated(path / "manifest.json", store.sec.encoded(manifest))
    with pytest.raises(ValueError, match="DATA_QUALITY_BLOCKED"):
        refresh.verify_candidate(path)
    assert not (root / store.STAGED).exists()


def test_noop_snapshot_exposes_quality_without_current_store_writes(state):
    path = state / store.STAGED
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    plan = update.snapshot(
        state,
        update.request(json.loads((path / "manifest.json").read_bytes())["companies"]),
    )
    assert plan["base"]["quality_gate"]["status"] == "FLAGGED"
    assert plan["base"]["quality_gate"]["open_flags"] > 0
    assert update.operations(plan) == dict(
        sec_companies=[], new_companies=[], market={}
    )
    assert before == {p.name: p.read_bytes() for p in path.iterdir()}
