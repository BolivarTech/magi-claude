# Author: Julian Bolivar
# Version: 2.0.0
# Date: 2026-09-20
"""``--ollama-init`` is one ``magi-rs init`` call (MS3 Task 4); nothing else survives here."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import run_magi
from magi_key import REDACTION_MARKER, KeyHardeningError, KeyStore, MagiKey
from magi_rs_bootstrap import DigestMismatchError
from magi_rs_env import EnvironmentPolicy
from ollama_init import MagiDirExistsError, OllamaInitError, run_ollama_init


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
    with pytest.raises(
        MagiDirExistsError
    ) as exc:  # [W4] an error: no caller can mistake it for success
        run_ollama_init(str(tmp_path), key_store=store)
    message = str(exc.value)
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


def test_an_interrupted_init_removes_the_magi_dir_it_created(monkeypatch, tmp_path):
    def run(argv, **kwargs):
        os.makedirs(os.path.join(argv[argv.index("-w") + 1], ".magi"))
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr("ollama_init.subprocess.run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        run_ollama_init(str(tmp_path), key_store=_Store())
    assert not (tmp_path / ".magi").exists()


def test_a_cleanup_that_fails_is_named_so_the_operator_deletes_it_by_hand(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise OSError("locked")

    monkeypatch.setattr("ollama_init.subprocess.run", _fake_run([], returncode=1))
    monkeypatch.setattr("ollama_init.shutil.rmtree", boom)
    with pytest.raises(OllamaInitError) as exc:
        run_ollama_init(str(tmp_path), key_store=_Store())
    assert ".magi" in str(exc.value) and "delete" in str(exc.value)


def _echoing_run(returncode, stream):
    """A `magi-rs init` double that echoes its MAGI_PASSPHRASE on `stream` (BDD-22's worst case)."""

    def run(argv, **kwargs):
        os.makedirs(os.path.join(argv[argv.index("-w") + 1], ".magi"))
        echoed = f"opening vault with {kwargs['env']['MAGI_PASSPHRASE']}\n"
        return _Completed(
            returncode=returncode,
            stdout=echoed if stream == "stdout" else "",
            stderr=echoed if stream == "stderr" else "",
        )

    return run


def test_an_init_that_echoes_the_new_key_on_stdout_returns_it_redacted(monkeypatch, tmp_path):
    # [Loop1 I-2] the init path relayed the child verbatim while the NEW key was in its environment.
    store = _Store()
    monkeypatch.setattr("ollama_init.subprocess.run", _echoing_run(0, "stdout"))
    output = run_ollama_init(str(tmp_path), key_store=store)
    assert store.new_key().reveal() not in output and REDACTION_MARKER in output


def test_an_init_that_echoes_the_new_key_on_stderr_and_fails_raises_it_redacted(
    monkeypatch, tmp_path
):
    store = _Store()
    monkeypatch.setattr("ollama_init.subprocess.run", _echoing_run(1, "stderr"))
    with pytest.raises(OllamaInitError) as exc:
        run_ollama_init(str(tmp_path), key_store=store)
    assert store.new_key().reveal() not in str(exc.value) and REDACTION_MARKER in str(exc.value)


@pytest.mark.parametrize("returncode, stream", [(0, "stdout"), (1, "stderr")])
def test_the_ollama_init_cli_never_prints_the_new_key(
    monkeypatch, tmp_path, capsys, returncode, stream
):
    store = _Store()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ollama_init.default_key_store", lambda: store)
    monkeypatch.setattr("ollama_init.subprocess.run", _echoing_run(returncode, stream))
    monkeypatch.setattr(sys, "argv", ["run_magi.py", "--ollama-init"])
    with pytest.raises(SystemExit) as exc:
        run_magi.main()
    captured = capsys.readouterr()
    assert exc.value.code == returncode
    assert store.new_key().reveal() not in captured.out + captured.err
    assert REDACTION_MARKER in captured.out + captured.err


def _raising_with_args(exception):
    def _raise(*args, **kwargs):
        raise exception

    return _raise


def test_the_ollama_init_cli_on_an_existing_magi_dir_prints_the_message_and_exits_zero(
    monkeypatch, capsys
):
    # [W4] REQ-12: "already exists, left untouched" stays a normal outcome at the CLI, never a [FATAL].
    message = (
        ".magi/ already exists in /w: left untouched (nothing was initialised, no key was created)."
    )
    monkeypatch.setattr(
        run_magi, "run_ollama_init", _raising_with_args(MagiDirExistsError(message))
    )
    monkeypatch.setattr(sys, "argv", ["run_magi.py", "--ollama-init"])
    with pytest.raises(SystemExit) as exc:
        run_magi.main()
    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "already exists" in captured.out and "untouched" in captured.out
    assert "[FATAL]" not in captured.err


OTHER_PROCESS_KEY = "other-process-key-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC="


def _write_other_process_key(workdir):
    """Write ``.magi/.secret/magi.key`` as a concurrent ``--ollama-init`` would."""
    secret = os.path.join(workdir, ".magi", ".secret")
    os.makedirs(secret, exist_ok=True)
    with open(os.path.join(secret, "magi.key"), "w", encoding="ascii") as handle:
        handle.write(OTHER_PROCESS_KEY)


class _NoopGuard:
    def harden(self, path, directory):
        pass

    def verify(self, path, directory):
        pass


def test_a_failed_init_never_removes_a_magi_dir_another_process_initialised(monkeypatch, tmp_path):
    # [W5] a key this call did not write means the .magi/ is someone else's: kept, and the error says so.
    def run(argv, **kwargs):
        _write_other_process_key(argv[argv.index("-w") + 1])
        return _Completed(returncode=1, stdout="", stderr="error: init died")

    monkeypatch.setattr("ollama_init.subprocess.run", run)
    with pytest.raises(OllamaInitError) as exc:
        run_ollama_init(str(tmp_path), key_store=_Store())
    key_file = tmp_path / ".magi" / ".secret" / "magi.key"
    assert key_file.read_text(encoding="ascii") == OTHER_PROCESS_KEY
    message = " ".join(str(exc.value).split())
    assert "another process" in message and str(tmp_path / ".magi") in message


def test_a_key_that_already_exists_at_persist_keeps_the_other_process_magi_dir(
    monkeypatch, tmp_path
):
    # [W5] persist's exclusive create fails (FileExistsError): the existing key and its .magi/ survive.
    def run(argv, **kwargs):
        _write_other_process_key(argv[argv.index("-w") + 1])
        return _Completed()

    monkeypatch.setattr("ollama_init.subprocess.run", run)
    # [R7] the note is folded into an OllamaInitError chained from the FileExistsError (no add_note).
    with pytest.raises(OllamaInitError) as exc:
        run_ollama_init(str(tmp_path), key_store=KeyStore(_NoopGuard()))
    key_file = tmp_path / ".magi" / ".secret" / "magi.key"
    assert key_file.read_text(encoding="ascii") == OTHER_PROCESS_KEY
    notes = " ".join(str(exc.value).split())
    assert "another process" in notes and str(tmp_path / ".magi") in notes
    assert isinstance(exc.value.__cause__, FileExistsError)


def test_a_key_this_call_wrote_and_could_not_remove_does_not_keep_the_magi_dir(
    monkeypatch, tmp_path
):
    # [W5] passes in Red by design: the guard must not over-protect this call's own leftover key.
    class _WritesThenFails(_Store):
        def persist(self, workdir, key):
            secret = os.path.join(workdir, ".magi", ".secret")
            os.makedirs(secret, exist_ok=True)
            with open(os.path.join(secret, "magi.key"), "w", encoding="ascii") as handle:
                handle.write(key.reveal())
            raise KeyHardeningError("cannot harden (and the key file could not be removed)")

    monkeypatch.setattr("ollama_init.subprocess.run", _fake_run([]))
    with pytest.raises(KeyHardeningError):
        run_ollama_init(str(tmp_path), key_store=_WritesThenFails())
    assert not (tmp_path / ".magi").exists()


def test_a_cleanup_note_is_part_of_the_message_never_an_exception_note(monkeypatch, tmp_path):
    # [R7] ``add_note`` needs Python 3.11; the note travels in ``str(error)`` of an OllamaInitError instead.
    def boom(*args, **kwargs):
        raise OSError("locked")

    monkeypatch.setattr("ollama_init.subprocess.run", _fake_run([]))
    monkeypatch.setattr("ollama_init.shutil.rmtree", boom)
    with pytest.raises(OllamaInitError) as exc:
        run_ollama_init(str(tmp_path), key_store=_Store(fail=True))
    assert "delete it by hand" in str(exc.value) and not hasattr(exc.value, "__notes__")
    assert isinstance(exc.value.__cause__, KeyHardeningError)


def test_an_empty_key_file_is_no_key_and_the_magi_dir_this_call_created_is_removed(
    monkeypatch, tmp_path
):
    # [R4] an empty key file is not another process's key: the cleanup removes the .magi/ this call created.
    def run(argv, **kwargs):
        secret = os.path.join(argv[argv.index("-w") + 1], ".magi", ".secret")
        os.makedirs(secret)
        open(os.path.join(secret, "magi.key"), "w").close()
        return _Completed(returncode=1, stdout="", stderr="error: init died")

    monkeypatch.setattr("ollama_init.subprocess.run", run)
    with pytest.raises(OllamaInitError) as exc:
        run_ollama_init(str(tmp_path), key_store=_Store())
    assert not (tmp_path / ".magi").exists() and "another process" not in str(exc.value)


def test_an_unreadable_key_file_is_treated_as_another_process_key_and_kept(monkeypatch, tmp_path):
    # [R4] passes in Red by design: a key that cannot be read is never deleted.
    def run(argv, **kwargs):
        os.makedirs(os.path.join(argv[argv.index("-w") + 1], ".magi", ".secret", "magi.key"))
        return _Completed(returncode=1, stdout="", stderr="error: init died")

    monkeypatch.setattr("ollama_init.subprocess.run", run)
    with pytest.raises(OllamaInitError) as exc:
        run_ollama_init(str(tmp_path), key_store=_Store())
    assert (tmp_path / ".magi" / ".secret" / "magi.key").exists() and "another process" in str(
        exc.value
    )


def test_an_interrupt_propagates_as_itself_even_when_a_cleanup_note_is_needed(
    monkeypatch, tmp_path, capsys
):
    # [n-1] Ctrl-C stays an interrupt; the note that the .magi/ could not be removed goes to stderr, one line.
    def run(argv, **kwargs):
        os.makedirs(os.path.join(argv[argv.index("-w") + 1], ".magi"))
        raise KeyboardInterrupt

    def boom(*args, **kwargs):
        raise OSError("locked")

    monkeypatch.setattr("ollama_init.subprocess.run", run)
    monkeypatch.setattr("ollama_init.shutil.rmtree", boom)
    with pytest.raises(KeyboardInterrupt):
        run_ollama_init(str(tmp_path), key_store=_Store())
    err = capsys.readouterr().err
    assert "delete it by hand" in err and err.count("\n") == 1


def test_a_hardening_failure_reaches_the_init_fatal_line_with_the_tool_and_its_exit_code(
    monkeypatch, tmp_path, capsys
):
    # [S1] the guard's own detail (tool, exit code, stderr headline) is part of the message, not only __cause__.
    from magi_key_permissions import WindowsPermissionGuard

    def _run(argv, **kwargs):
        tool = argv[0].lower()
        if tool.endswith("whoami.exe"):
            return SimpleNamespace(stdout='"host\\me","S-1-5-21-1111-2222-3333-1001"\r\n')
        if tool.endswith("icacls.exe"):
            raise subprocess.CalledProcessError(5, argv, output="", stderr="Access is denied.\r\n")
        os.makedirs(os.path.join(argv[argv.index("-w") + 1], ".magi"))
        return _Completed()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ollama_init.subprocess.run", _run)
    monkeypatch.setattr("ollama_init.default_key_store", lambda: KeyStore(WindowsPermissionGuard()))
    monkeypatch.setattr(sys, "argv", ["run_magi.py", "--ollama-init"])
    with pytest.raises(SystemExit) as exc:
        run_magi.main()
    err = capsys.readouterr().err
    assert exc.value.code == 1 and err.startswith("[FATAL]") and err.count("\n") == 1
    assert "icacls.exe" in err and "exit code 5" in err and "Access is denied." in err


def _names_the_whole_regeneration(message):
    """REQ-12's three steps: move .magi/ aside, --ollama-init, reload each API key with --ollama-vault-set."""
    return (
        "move the existing .magi/ aside" in message
        and "--ollama-init" in message
        and "--ollama-vault-set" in message
    )


def test_an_existing_magi_dir_names_the_whole_regeneration(tmp_path):
    # [S3] the message of MagiDirExistsError reuses the one recovery procedure.
    (tmp_path / ".magi").mkdir()
    with pytest.raises(MagiDirExistsError) as exc:
        run_ollama_init(str(tmp_path), key_store=_Store())
    message = str(exc.value)
    assert (
        "already exists" in message
        and "untouched" in message
        and _names_the_whole_regeneration(message)
    )
