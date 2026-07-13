.PHONY: test lint format typecheck lockcheck verify

# uv is a hard prerequisite for developing (not for using) the plugin. Say so here rather
# than letting the first target die with a bare "uv: command not found".
ifeq (,$(shell command -v uv 2>/dev/null))
$(error uv is required to run the checks (every target goes through `uv run`). \
Install it: https://docs.astral.sh/uv/getting-started/installation/)
endif

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
