# Author: Julian Bolivar
# Version: 2.0.0
# Date: 2026-09-20
"""``--ollama-init`` is one ``magi-rs init`` call (MS3 Task 4); nothing else survives here."""

import os
from types import SimpleNamespace

import pytest

from magi_rs_bootstrap import DigestMismatchError
from magi_rs_env import EnvironmentPolicy
from ollama_init import run_ollama_init


class _Ok:
    """A completed ``subprocess.run`` result for stubbing a child that exited 0."""

    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.fixture(autouse=True)
def _verified_binary(monkeypatch):
    """Stub the bootstrap: no test here may download into the real bin/."""
    monkeypatch.setattr(
        "ollama_init.BinaryBootstrap",
        lambda *a, **k: SimpleNamespace(
            ensure=lambda: "/bin/magi-rs", verify=lambda: "/bin/magi-rs"
        ),
    )


def _raising(exception):
    def _raise():
        raise exception

    return _raise


def test_ollama_init_invokes_magi_rs_init_with_the_workdir(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        "ollama_init.subprocess.run",
        lambda argv, **kw: seen.update(argv=argv, env=kw.get("env")) or _Ok(),
    )
    run_ollama_init(str(tmp_path))
    assert seen["argv"][1] == "init"
    assert seen["argv"][seen["argv"].index("-w") + 1] == str(tmp_path)
    assert "-p" not in seen["argv"]  # R5a: no passphrase, ever


def test_ollama_init_touches_nothing_under_dot_claude(monkeypatch, tmp_path):
    # R5/R5b/BDD-7c: a surviving v5 magi-ollama.toml is INVISIBLE to the code.
    # The run must be indistinguishable from one in a project where it never
    # existed -- not read, not checked, not mentioned.
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    stale = claude_dir / "magi-ollama.toml"
    stale.write_text("[models]\nmelchior = 'x'", encoding="utf-8")
    before = stale.read_bytes()
    monkeypatch.setattr("ollama_init.subprocess.run", lambda *a, **k: _Ok())
    output = run_ollama_init(str(tmp_path))
    assert stale.read_bytes() == before
    assert "magi-ollama" not in output
    assert sorted(p.name for p in claude_dir.iterdir()) == ["magi-ollama.toml"]


def test_ollama_init_reports_cleanly_on_an_already_initialised_project(monkeypatch, tmp_path):
    # BDD-7b: magi-rs may refuse, or may print the path and exit 0. The
    # requirement is on OUR side: finish well either way, without propagating
    # a cryptic error or forcing anything.
    monkeypatch.setattr(
        "ollama_init.subprocess.run",
        lambda *a, **k: _Ok(stdout="already initialised at /repo/.magi"),
    )
    assert "already initialised" in run_ollama_init(str(tmp_path))


def test_ollama_init_relays_a_refusal_without_a_traceback(monkeypatch, tmp_path):
    # BDD-7b, the other outcome: magi-rs refuses (non-zero). Relayed as text,
    # never forced, never a Python traceback.
    monkeypatch.setattr(
        "ollama_init.subprocess.run",
        lambda *a, **k: _Ok(stdout="", stderr="refusing to nest under /repo/.magi", returncode=1),
    )
    output = run_ollama_init(str(tmp_path))
    assert "refusing to nest" in output


def test_ollama_init_child_environment_carries_nothing_beyond_the_allowlist(monkeypatch, tmp_path):
    # The first draft CAPTURED this env and never asserted on it. R7 applies to
    # both subprocess sites, not just to the gate one.
    seen = {}
    monkeypatch.setenv("MAGI_PASSPHRASE", "hunter2")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    monkeypatch.setattr(
        "ollama_init.subprocess.run", lambda argv, **kw: seen.update(env=kw["env"]) or _Ok()
    )
    run_ollama_init(str(tmp_path))
    assert "MAGI_PASSPHRASE" not in seen["env"]
    assert "ANTHROPIC_API_KEY" not in seen["env"]
    assert set(seen["env"]) <= set(EnvironmentPolicy.build(os.environ)) | {"PATH"}


def test_ollama_init_does_not_run_an_unverified_binary(monkeypatch, tmp_path):
    # R4 has no carve-out for the init command: it executes the binary too.
    ran = []
    monkeypatch.setattr("ollama_init.subprocess.run", lambda *a, **k: ran.append(a))
    monkeypatch.setattr(
        "ollama_init.BinaryBootstrap",
        lambda *a, **k: SimpleNamespace(ensure=_raising(DigestMismatchError("digest differs"))),
    )
    with pytest.raises(DigestMismatchError):
        run_ollama_init(str(tmp_path))
    assert ran == []
