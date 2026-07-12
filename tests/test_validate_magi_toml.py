# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-12
"""Tests for the v5 config validator -- the assisted migration path.

The validator is what every fail-closed v4 error message, the README, the skill and
``docs/ollama-backend.md`` tell the user to run. It is therefore **product**, not local
dev tooling: it must live in the shipped tree (``skills/magi/scripts/``), which is what
the import below asserts -- ``conftest.py`` puts only that directory on ``sys.path``.

It replaced the auto-shim that spec decision #14 rejected: converting a v4 string entry
would require *inferring* the lineage, and a wrong guess assigns a silently incorrect
lineage -- two mages of one lineage, a consensus that only LOOKS like three independent
perspectives. So the validator reports and never rewrites.
"""

import pytest

import validate_magi_toml
from ollama_init import render_template


def test_validator_accepts_a_v5_config(tmp_path, monkeypatch):
    """Exit 0 on the config that ``--ollama-init`` itself scaffolds."""
    path = tmp_path / "magi-ollama.toml"
    path.write_text(render_template(), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["validate_magi_toml.py", str(path)])

    assert validate_magi_toml.main() == 0


def test_validator_rejects_a_v4_config_and_says_what_to_write(tmp_path, monkeypatch, capsys):
    """Exit 1, and the message must show the NEW shape.

    This message IS the migration path that replaced the rejected auto-shim, so a
    message that does not say what to write defeats the whole argument for refusing
    to guess.
    """
    path = tmp_path / "magi-ollama.toml"
    path.write_text('[models]\nmelchior = "qwen3.5:397b-cloud"\n', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["validate_magi_toml.py", str(path)])

    assert validate_magi_toml.main() == 1

    err = capsys.readouterr().err
    assert "lineage" in err
    assert 'melchior = { model = "qwen3.5:397b-cloud", lineage = "alibaba" }' in err


def test_validator_rejects_a_trio_that_shares_a_lineage(tmp_path, monkeypatch, capsys):
    """Exit 1 when two mages declare one lineage -- invariant #1, checked before any run."""
    path = tmp_path / "magi-ollama.toml"
    path.write_text(
        'base_url = "http://localhost:11434/v1"\n'
        "[models]\n"
        'melchior  = { model = "qwen3.5:397b-cloud", lineage = "alibaba" }\n'
        'balthasar = { model = "qwen3-coder:480b-cloud", lineage = "alibaba" }\n'
        'caspar    = { model = "deepseek-v4-pro:cloud", lineage = "deepseek" }\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["validate_magi_toml.py", str(path)])

    assert validate_magi_toml.main() == 1
    assert "lineage" in capsys.readouterr().err


def test_validator_ignores_the_users_global_config(tmp_path, monkeypatch):
    """The verdict is about the file you passed -- nothing else.

    ``resolve_config(global_path=None)`` does NOT mean "no global": None is the
    sentinel for "use ~/.claude/magi-ollama.toml". So a broken file in the user's HOME
    made the validator report INVALID on a perfectly valid config, quoting an error
    about a file the user never passed -- the mirror image of the fail-open this tool
    exists to close, and a shipped test that reads the developer's home directory.
    """
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "magi-ollama.toml").write_text('base_url = "http\n', encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    path = tmp_path / "magi-ollama.toml"
    path.write_text(render_template(), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["validate_magi_toml.py", str(path)])

    assert validate_magi_toml.main() == 0


def test_validator_reports_cli_misuse_without_a_traceback(tmp_path, monkeypatch):
    """A missing file is a user error, not a crash: exit 1 with a message, no traceback."""
    missing = tmp_path / "does-not-exist.toml"
    monkeypatch.setattr("sys.argv", ["validate_magi_toml.py", str(missing)])

    with pytest.raises(SystemExit) as exc:
        validate_magi_toml.main()

    assert exc.value.code != 0
