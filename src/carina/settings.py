"""Where Carina finds the dataset it maintains and the decisions it applies.

Carina stores no data and makes no decisions about it. The caller supplies two
directories:

- the data directory, which holds ``staged/`` (the dataset), ``runs/`` (work in
  progress) and ``receipts/`` (records of promotions); and
- the decisions directory, which holds the JSON files that accept, quarantine or
  acknowledge data, and the thresholds the quality checks use.

Set them with ``configure`` (the command line does this from ``--data`` and
``--decisions``) or with the ``CARINA_DATA`` and ``CARINA_DECISIONS`` variables.
"""

from __future__ import annotations

import os
from pathlib import Path

DECISION_FILES = (
    "company_data_quality.json",
    "company_quality_acknowledgements.json",
    "company_fact_quarantines.json",
    "company_market_quality.json",
    "company_market_acknowledgements.json",
    "company_market_events.json",
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
