# Author: Julian Bolivar
# Version: 2.0.0
# Date: 2026-09-20
"""``--ollama-init`` is one ``magi-rs init`` call (MS3 Task 4); nothing else survives here."""

import os
from types import SimpleNamespace

import pytest

from magi_key import KeyHardeningError, KeyStore, MagiKey
from magi_rs_bootstrap import DigestMismatchError
from magi_rs_env import EnvironmentPolicy
from ollama_init import OllamaInitError, run_ollama_init


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
    run_ollama_init(str(tmp_path), key_store=_Store())
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
    output = run_ollama_init(str(tmp_path), key_store=_Store())
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
    assert "already initialised" in run_ollama_init(str(tmp_path), key_store=_Store())


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


class _Completed:
    def __init__(self, returncode=0, stdout="initialised\n", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _fake_run(seen, creates_magi=True, returncode=0):
    def run(argv, **kwargs):
        seen.append((argv, kwargs["env"]))
        if creates_magi and returncode == 0:
            os.makedirs(os.path.join(kwargs.get("cwd") or argv[argv.index("-w") + 1], ".magi"))
        return _Completed(returncode=returncode)

    return run


class _Store(KeyStore):
    def __init__(self, fail=False):
        self.persisted, self.fail = [], fail

    def new_key(self):
        return MagiKey("generated-key-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=")

    def persist(self, workdir, key):
        if self.fail:
            raise KeyHardeningError("cannot harden")
        self.persisted.append((workdir, key))
        return "path"


def test_init_uses_the_platform_key_store_by_default(monkeypatch, tmp_path):
    # [CP2-C36] the fourth production call site, the one every new user runs first.
    built = []
    store = _Store()
    monkeypatch.setattr("ollama_init.default_key_store", lambda: built.append("built") or store)
    monkeypatch.setattr("ollama_init.subprocess.run", _fake_run([]))
    run_ollama_init(str(tmp_path))
    assert built == ["built"] and store.persisted


def test_init_passes_the_new_key_by_environment_and_persists_it(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr("ollama_init.subprocess.run", _fake_run(seen))
    store = _Store()
    run_ollama_init(str(tmp_path), key_store=store)
    argv, env = seen[0]
    assert env["MAGI_PASSPHRASE"] == store.new_key().reveal()
    assert store.new_key().reveal() not in " ".join(argv)
    assert store.persisted and store.persisted[0][0] == str(tmp_path)


def test_an_existing_magi_dir_is_left_untouched_and_magi_rs_is_not_called(monkeypatch, tmp_path):
    (tmp_path / ".magi").mkdir()
    seen = []
    monkeypatch.setattr("ollama_init.subprocess.run", _fake_run(seen))
    store = _Store()
    message = run_ollama_init(str(tmp_path), key_store=store)
    assert seen == [] and store.persisted == []
    assert "already" in message and ".magi" in message


def test_a_failed_init_persists_no_key(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr("ollama_init.subprocess.run", _fake_run(seen, returncode=1))
    store = _Store()
    with pytest.raises(OllamaInitError):  # [CP2-C29]
        run_ollama_init(str(tmp_path), key_store=store)
    assert store.persisted == []


def test_a_failed_init_relays_its_output_and_removes_the_magi_dir_it_created(monkeypatch, tmp_path):
    # [CP2-M33]
    def run(argv, **kwargs):
        os.makedirs(os.path.join(argv[argv.index("-w") + 1], ".magi"))
        return _Completed(returncode=1, stdout="", stderr="error: init died")

    monkeypatch.setattr("ollama_init.subprocess.run", run)
    store = _Store()
    with pytest.raises(OllamaInitError) as exc:  # [CP2-C29]
        run_ollama_init(str(tmp_path), key_store=store)
    assert "init died" in str(exc.value) and store.persisted == []
    assert not (tmp_path / ".magi").exists()


def test_a_key_that_cannot_be_hardened_removes_the_magi_dir_this_call_created(
    monkeypatch, tmp_path
):
    seen = []
    monkeypatch.setattr("ollama_init.subprocess.run", _fake_run(seen))
    with pytest.raises(KeyHardeningError):
        run_ollama_init(str(tmp_path), key_store=_Store(fail=True))
    assert not (tmp_path / ".magi").exists()


def test_ollama_init_child_environment_carries_nothing_beyond_the_allowlist(monkeypatch, tmp_path):
    # [RD-D12] The injected key travels; the inherited passphrase and API keys never do.
    seen = {}
    monkeypatch.setenv("MAGI_PASSPHRASE", "hunter2")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    monkeypatch.setattr(
        "ollama_init.subprocess.run", lambda argv, **kw: seen.update(env=kw["env"]) or _Completed()
    )
    store = _Store()
    run_ollama_init(str(tmp_path), key_store=store)
    assert seen["env"]["MAGI_PASSPHRASE"] == store.new_key().reveal()
    assert "ANTHROPIC_API_KEY" not in seen["env"]
    assert set(seen["env"]) - {"MAGI_PASSPHRASE"} <= set(EnvironmentPolicy.build(os.environ)) | {
        "PATH"
    }
