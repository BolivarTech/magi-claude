# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-09-18
"""Tests for the one-check project preflight of the magi-rs path (MS3 Task 2)."""

import pytest

from magi_rs_preflight import ProjectNotInitialisedError, ProjectPreflight


def test_an_initialised_project_passes_and_returns_the_config_path(tmp_path):
    (tmp_path / ".magi").mkdir()
    config = tmp_path / ".magi" / "magi.toml"
    config.write_text("provider = 'ollama'", encoding="utf-8")
    assert ProjectPreflight.require_initialised(str(tmp_path)) == str(config)


def test_a_project_without_magi_fails_naming_the_exact_command(tmp_path):
    # R5/BDD-7: a gate run that finds no .magi/ FAILS and says what to run.
    # It never initialises on the fly.
    with pytest.raises(ProjectNotInitialisedError) as exc:
        ProjectPreflight.require_initialised(str(tmp_path))
    assert "--ollama-init" in str(exc.value)


def test_the_preflight_writes_nothing_at_all(tmp_path):
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(ProjectNotInitialisedError):
        ProjectPreflight.require_initialised(str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_an_invalid_config_still_passes_the_preflight(tmp_path):
    # BDD-7d: the plugin checks PRESENCE, not validity. magi-rs owns that
    # file's contents and already fails fatally on a toml that does not parse.
    # Two validators of the same file diverge, and the day they do, ours lies.
    (tmp_path / ".magi").mkdir()
    (tmp_path / ".magi" / "magi.toml").write_text("this is not [valid toml", encoding="utf-8")
    assert ProjectPreflight.require_initialised(str(tmp_path)).endswith("magi.toml")


def test_the_preflight_error_is_not_a_validation_error():
    # Locked derogation: fail-closed and a retry does not create the file.
    from synthesize import ValidationError

    assert not issubclass(ProjectNotInitialisedError, ValidationError)
