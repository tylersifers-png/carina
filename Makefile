PYTHON := .venv/bin/python

.PHONY: setup test

setup:
	uv python install 3.12
	uv venv --managed-python --python 3.12 --seed .venv
	uv pip install --python $(PYTHON) -e ".[dev]"

test:
	$(PYTHON) -m pytest -q
