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
from ollama_config import resolve_config
from ollama_init import render_template
from ollama_preflight import OllamaPreflightError, check_config_offline


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


def test_validator_rejects_every_config_the_preflight_rejects(tmp_path, monkeypatch):
    """A pre-run check that green-lights what the product refuses to run is worthless.

    Two ``[[fallback]]`` entries of one lineage are fail-closed at preflight (R11.3:
    only the first is ever reachable, so the second is a config error attacking the
    central invariant). The validator used to re-derive ONE of preflight's checks by
    hand and therefore said OK to exactly that config. Both now go through
    ``check_config_offline``, so they cannot drift apart again.
    """
    dup_fallback = (
        'base_url = "http://localhost:11434/v1"\n'
        "[models]\n"
        'melchior  = { model = "qwen3.5:397b-cloud", lineage = "alibaba" }\n'
        'balthasar = { model = "kimi-k2.6:cloud", lineage = "moonshot" }\n'
        'caspar    = { model = "deepseek-v4-pro:cloud", lineage = "deepseek" }\n'
        "[[fallback]]\n"
        'model = "glm-5.2:cloud"\n'
        'lineage = "zhipu"\n'
        "[[fallback]]\n"
        'model = "glm-5:cloud"\n'
        'lineage = "zhipu"\n'
    )
    path = tmp_path / "magi-ollama.toml"
    path.write_text(dup_fallback, encoding="utf-8")

    # The product's own verdict on this config, as the guard the validator must mirror.
    config = resolve_config(repo_path=str(path), global_path="", env={})
    with pytest.raises(OllamaPreflightError):
        check_config_offline(config)

    monkeypatch.setattr("sys.argv", ["validate_magi_toml.py", str(path)])
    assert validate_magi_toml.main() == 1


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


def test_validator_echoes_the_resolved_trio_so_you_can_see_it_was_read(
    tmp_path, monkeypatch, capsys
):
    """``OK`` must show WHAT was accepted, not just that something was.

    A bare "OK: valid v5 config" is what an EMPTY file printed too (MAGI falls back to
    the built-in defaults, which are valid) -- indistinguishable from a real endorsement
    of the user's own trio. Echoing the resolved models is how the tool proves it read
    the file you handed it.
    """
    path = tmp_path / "magi-ollama.toml"
    path.write_text(render_template(), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["validate_magi_toml.py", str(path)])

    assert validate_magi_toml.main() == 0

    out = capsys.readouterr().out
    assert "melchior" in out
    assert "qwen3.5:397b-cloud" in out
    assert "alibaba" in out


def test_validator_rejects_malformed_toml_without_the_lineage_lecture(
    tmp_path, monkeypatch, capsys
):
    """A syntax error is not a schema error: do not answer it with the lineage hint."""
    path = tmp_path / "magi-ollama.toml"
    path.write_text('base_url = "http\n', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["validate_magi_toml.py", str(path)])

    assert validate_magi_toml.main() == 1

    err = capsys.readouterr().err
    assert "TOML" in err
    assert "lineage is NOT inferred" not in err


def test_validator_says_a_directory_is_not_a_file(tmp_path, monkeypatch, capsys):
    """A directory EXISTS, so "no such config file" would be a lie about why it failed."""
    monkeypatch.setattr("sys.argv", ["validate_magi_toml.py", str(tmp_path)])

    with pytest.raises(SystemExit) as exc:
        validate_magi_toml.main()

    assert exc.value.code == 2
    assert "not a file" in capsys.readouterr().err


def test_validator_reports_an_unreadable_file_without_a_traceback(tmp_path, monkeypatch, capsys):
    """An OSError from open() must not reach the user as a stack trace."""
    path = tmp_path / "magi-ollama.toml"
    path.write_text(render_template(), encoding="utf-8")

    def _boom(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("builtins.open", _boom)
    monkeypatch.setattr("sys.argv", ["validate_magi_toml.py", str(path)])

    with pytest.raises(SystemExit) as exc:
        validate_magi_toml.main()

    assert exc.value.code == 2
    assert "Permission denied" in capsys.readouterr().err


def test_validator_reports_a_missing_path_as_cli_misuse(tmp_path, monkeypatch, capsys):
    """A missing file is CLI misuse (exit 2), pinned -- it must never be a silent OK.

    Resolving a path that does not exist falls through to the built-in defaults and
    validates THOSE, so the tool would report OK about a config it never read.
    """
    missing = tmp_path / "does-not-exist.toml"
    monkeypatch.setattr("sys.argv", ["validate_magi_toml.py", str(missing)])

    with pytest.raises(SystemExit) as exc:
        validate_magi_toml.main()

    assert exc.value.code == 2
    assert "no such config file" in capsys.readouterr().err
