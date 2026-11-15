"""Shared pytest setup: a fixed company list and fixture decision files."""

from pathlib import Path

import pytest

from carina import registry, settings
from tests.registry_fixture import INITIAL_ROWS

DECISIONS = Path(__file__).parent / "decisions"


@pytest.fixture(autouse=True)
def carina_inputs():
    settings.reset()
    settings.configure(decisions=DECISIONS)
    registry.reset()
    registry.COMPANIES.use(INITIAL_ROWS)
    yield
    registry.reset()
    settings.reset()
