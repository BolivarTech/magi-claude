# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-09-18
"""Tests for the one-check project preflight of the magi-rs path (MS3 Task 2)."""

import pytest

from magi_key import KeyMissingError
from magi_rs_embedder import EmbedderUnreachableError
from magi_rs_preflight import ProjectNotInitialisedError, ProjectPreflight
from magi_rs_trio import FAIL_CLOSED_ERRORS


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


class _Recorder:
    def __init__(self, order, name, error=None, value=None):
        self.order, self.name, self.error, self.value = order, name, error, value

    def load(self, workdir):
        self.order.append(self.name)
        if self.error:
            raise self.error
        return self.value

    def check(self, config):
        self.order.append(self.name)
        if self.error:
            raise self.error


def _init(tmp_path):
    (tmp_path / ".magi").mkdir()
    (tmp_path / ".magi" / "magi.toml").write_text("", encoding="utf-8")


def test_the_preflight_checks_the_config_then_returns_the_loaded_key(tmp_path):
    _init(tmp_path)
    order = []
    key = ProjectPreflight.require_ready(str(tmp_path), _Recorder(order, "key", value="K"))
    assert order == ["key"] and key == "K"


def test_a_key_error_propagates_from_the_preflight(tmp_path):
    _init(tmp_path)
    with pytest.raises(KeyMissingError):
        ProjectPreflight.require_ready(
            str(tmp_path), _Recorder([], "key", error=KeyMissingError("x"))
        )


def test_an_uninitialised_project_stops_before_the_key_is_read(tmp_path):
    order = []
    with pytest.raises(ProjectNotInitialisedError):
        ProjectPreflight.require_ready(str(tmp_path), _Recorder(order, "key"))
    assert order == []


def test_with_a_probe_the_embedder_is_checked_after_the_key(tmp_path):
    _init(tmp_path)
    order = []
    key = ProjectPreflight.require_ready(
        str(tmp_path), _Recorder(order, "key", value="K"), probe=_Recorder(order, "embedder")
    )
    assert order == ["key", "embedder"] and key == "K"


def test_a_missing_key_stops_before_the_embedder_is_probed(tmp_path):
    _init(tmp_path)
    order = []
    with pytest.raises(KeyMissingError):
        ProjectPreflight.require_ready(
            str(tmp_path),
            _Recorder(order, "key", error=KeyMissingError("x")),
            probe=_Recorder(order, "embedder"),
        )
    assert order == ["key"]


def test_an_unreachable_embedder_is_rendered_fatal():
    assert EmbedderUnreachableError in FAIL_CLOSED_ERRORS


def _names_the_whole_regeneration(message):
    """REQ-12's three steps: move .magi/ aside, --ollama-init, reload each API key with --ollama-vault-set."""
    return (
        "move the existing .magi/ aside" in message
        and "--ollama-init" in message
        and "--ollama-vault-set" in message
    )


def test_a_magi_dir_without_its_config_names_the_whole_regeneration(tmp_path):
    # [S3] --ollama-init refuses an existing .magi/: "run --ollama-init" alone would send the operator in a circle.
    (tmp_path / ".magi").mkdir()
    with pytest.raises(ProjectNotInitialisedError) as exc:
        ProjectPreflight.require_initialised(str(tmp_path))
    assert _names_the_whole_regeneration(str(exc.value))


def test_a_project_without_any_magi_dir_is_told_only_to_run_init(tmp_path):
    with pytest.raises(ProjectNotInitialisedError) as exc:
        ProjectPreflight.require_initialised(str(tmp_path))
    message = str(exc.value)
    assert "--ollama-init" in message and "move the existing" not in message
