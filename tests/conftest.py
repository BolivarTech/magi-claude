# Author: Julian Bolivar
# Version: 2.0.0
# Date: 2026-09-20
"""Shared fixtures for the MAGI test suite: every test gets a real agents dir."""

import shutil
from pathlib import Path

import pytest

#: The three prompts the plugin actually ships.
_AGENTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "magi" / "agents"


@pytest.fixture(autouse=True)
def seeded_agents_dir(tmp_path):
    """Give every test a ``tmp_path`` that is a REAL agents dir (the shipped prompts).

    Almost every orchestrator test passes ``agents_dir=str(tmp_path)`` -- and until the
    prompt guard moved into ``run_orchestrator`` (MAGI gate, Balthasar), that directory was
    EMPTY: the tests exercised a path no user can ever take. Seeding the real prompts is not
    scaffolding to keep them green; it is what makes them exercise the contract they claim
    to, guard included. A test whose agents dir has no agents was never testing the run.

    Yields:
        The seeded ``tmp_path``.
    """
    for prompt in _AGENTS_DIR.glob("*.md"):
        shutil.copy(prompt, tmp_path / prompt.name)
    return tmp_path
