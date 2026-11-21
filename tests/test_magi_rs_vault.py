# Author: Julian Bolivar
# Version: 7.1.0
# Date: 2026-11-19
"""Tests for the magi-rs vault client (MS1 Task 8)."""

import os

import pytest

from magi_key import MagiKey
from magi_rs_env import EnvironmentPolicy
from magi_rs_vault import InvalidVaultNameError, VaultClient, VaultCommandError

_KEY = MagiKey("vault-key-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC=")


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _client(seen, completed=None):
    def runner(argv, **kwargs):
        seen.append((argv, kwargs))
        return completed or _Completed()

    return VaultClient("/bin/magi-rs", "/repo", _KEY, runner=runner)


def test_a_name_outside_the_closed_list_is_refused_before_the_binary():
    seen = []
    with pytest.raises(InvalidVaultNameError):
        _client(seen).set("AWS_SECRET", "x")
    assert seen == []


def test_the_value_travels_on_stdin_and_the_key_by_environment():
    seen = []
    _client(seen).set("OPENAI_API_KEY", "sk-value")
    argv, kwargs = seen[0]
    assert argv == ["/bin/magi-rs", "vault", "set", "OPENAI_API_KEY", "-w", "/repo"]
    assert kwargs["input"] == "sk-value"
    assert kwargs["env"]["MAGI_PASSPHRASE"] == _KEY.reveal()
    assert "sk-value" not in " ".join(argv) and _KEY.reveal() not in " ".join(argv)


def test_ls_runs_vault_ls_with_the_key_by_environment_never_argv():
    # [CP2-M29] REQ-10 at its most sensitive point: the exposed key during a rotation.
    seen = []
    _client(seen, _Completed(stdout="")).ls()
    argv, kwargs = seen[0]
    assert argv == ["/bin/magi-rs", "vault", "ls", "-w", "/repo"]
    assert kwargs["env"]["MAGI_PASSPHRASE"] == _KEY.reveal() and _KEY.reveal() not in " ".join(argv)


def test_the_vault_child_environment_is_the_allowlist_plus_the_key(monkeypatch):
    # [CP2-M30] REQ-13c: parent-shell API keys and passphrase never reach the vault child.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-parent")
    monkeypatch.setenv("MAGI_PASSPHRASE", "hunter2")
    seen = []
    _client(seen).set("OPENAI_API_KEY", "sk-value")
    env = seen[0][1]["env"]
    assert "OPENAI_API_KEY" not in env and env["MAGI_PASSPHRASE"] == _KEY.reveal()
    assert set(env) - {"MAGI_PASSPHRASE"} <= set(EnvironmentPolicy.build(os.environ)) | {"PATH"}


def test_a_failed_ls_is_a_vault_command_error_never_an_empty_list():
    # [CP2-C34] an empty answer would read as "nothing to revoke" during a rotation (REQ-13g.4).
    seen = []
    with pytest.raises(VaultCommandError):
        _client(seen, _Completed(returncode=1, stderr="error: incorrect passphrase")).ls()


def test_an_unparseable_ls_is_a_vault_command_error():
    # [CP2-C34]
    seen = []
    with pytest.raises(VaultCommandError):
        _client(seen, _Completed(stdout="garbage without the M2 separators\n")).ls()


def test_a_failed_set_names_the_secret_but_never_its_value():
    seen = []
    with pytest.raises(VaultCommandError) as exc:
        _client(seen, _Completed(returncode=1, stderr="sk-value rejected " + _KEY.reveal())).set(
            "OPENAI_API_KEY", "sk-value"
        )
    assert "OPENAI_API_KEY" in str(exc.value) and "sk-value" not in str(exc.value)
    assert _KEY.reveal() not in str(exc.value)  # [CP2-C41] the vault child's stderr is redacted too


_LS_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "magi-rs-vault", "ls.txt"
)  # captured in M2
_LS_FIXTURE_NAMES = [
    "EMBEDDING_API_KEY",
    "OPENAI_API_KEY",
]  # [PM-MS1-C7] [RD-D18] the two names M2 loaded


def test_ls_returns_names_from_the_captured_output():
    with open(_LS_FIXTURE, encoding="utf-8") as handle:
        captured = handle.read()  # tests/fixtures/magi-rs-vault/ls.txt (M2)
    seen = []
    names = _client(seen, _Completed(stdout=captured)).ls()
    assert names == _LS_FIXTURE_NAMES  # declared next to the fixture, from M2
