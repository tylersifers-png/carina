"""Where Carina finds the dataset it maintains and the decisions it applies.

Carina stores no data and makes no decisions about it. The caller supplies two
directories:

- the data directory, which holds ``staged/`` (the datasets), ``runs/`` (work in
  progress, deleted once a run is promoted) and ``receipts/`` (one record per
  promotion); and
- the decisions directory, which holds three JSON files:

  ``collection.json``
      what to collect: companies, benchmarks and macro series.
  ``thresholds.json``
      the limits the financial and market checks use.
  ``reviews.json``
      reviewed outcomes: fact quarantines, acknowledged flags and known market
      events.

Set them with ``configure`` (the command line does this from ``--data`` and
``--decisions``) or with the ``CARINA_DATA`` and ``CARINA_DECISIONS`` variables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DECISION_FILES = {
    "collection.json": "carina-collection-1.0",
    "thresholds.json": "carina-thresholds-1.0",
    "reviews.json": "carina-reviews-1.0",
}
REVIEW_KINDS = (
    "fact_quarantines",
    "financial_acknowledgements",
    "market_acknowledgements",
    "market_events",
)

_paths: dict[str, Path] = {}


def configure(*, data=None, decisions=None) -> None:
    if data is not None:
        _paths["data"] = Path(data)
    if decisions is not None:
        _paths["decisions"] = Path(decisions)


def reset() -> None:
    _paths.clear()


def _get(name: str, variable: str) -> Path:
    if name in _paths:
        return _paths[name]
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(
            f"No {name} directory configured: pass --{name} or set {variable}"
        )
    return Path(value)


def data() -> Path:
    return _get("data", "CARINA_DATA")


def decisions() -> Path:
    return _get("decisions", "CARINA_DECISIONS")


def decision(name: str) -> Path:
    if name not in DECISION_FILES:
        raise ValueError("Unknown decision file: " + name)
    return decisions() / name


def load(name: str) -> dict:
    """Read one decision file, checking its schema."""
    content = json.loads(decision(name).read_bytes())
    if content.get("schema") != DECISION_FILES[name]:
        raise ValueError(f"{name} must have schema {DECISION_FILES[name]}")
    return content


def collection() -> dict:
    return load("collection.json")


def thresholds(kind: str) -> dict:
    """The ``financial`` or ``market`` check limits."""
    return dict(load("thresholds.json")[kind])


def reviews(kind: str) -> list:
    if kind not in REVIEW_KINDS:
        raise ValueError("Unknown review kind: " + kind)
    return list(load("reviews.json").get(kind, []))
