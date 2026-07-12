.PHONY: test lint format typecheck lockcheck verify

# Every target runs through `uv run`, so it uses the toolchain pinned in uv.lock.
# A bare `python -m ...` uses whatever venv is on PATH: that is how `make verify`
# and the documented `uv run mypy .` came to disagree about the same code (one
# venv had mypy 1.20, which reports a no-any-return that 2.x does not). A gate
# must not depend on which shell you happen to be in.

test:
	uv run python -m pytest tests/ -v

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

typecheck:
	uv run mypy .

lockcheck:
	uv lock --check

verify: lockcheck test lint format typecheck
