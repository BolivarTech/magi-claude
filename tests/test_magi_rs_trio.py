# Author: Julian Bolivar
# Version: 1.5.0
# Date: 2026-09-29
"""Tests for MagiRsTrio, the second TrioExecutor (MS3 Task 3).

Every child process is faked: nothing here launches a binary, reaches the
network, or touches the plugin's real ``bin/``.
"""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

import magi_rs_trio
from magi_rs_bootstrap import DigestMismatchError
from magi_rs_preflight import ProjectNotInitialisedError
from magi_rs_trio import (
    _STDERR_ELISION_MARKER,
    MagiRsContractError,
    MagiRsProcessError,
    MagiRsTrio,
    build_argv,
    resolve_timeout,
)
from synthesize import ValidationError

_MS0_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "magi-rs-consult", "three-agents.json"
)


# --------------------------------------------------------------------------
# Helpers -- written before the tests so nothing is invented
# --------------------------------------------------------------------------


class _FakeProcess:
    """Stands in for the magi-rs child. Never launches anything."""

    def __init__(self, stdout=b"", stderr=b"", returncode=0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.stdin_seen = None
        self.killed = False

    async def communicate(self, input=None):
        self.stdin_seen = input
        return self._stdout, self._stderr

    async def wait(self):
        return self.returncode

    def kill(self):
        self.killed = True


def _exec_returning(process):
    async def _fake_exec(*argv, **kwargs):
        return process

    return _fake_exec


def _ok_payload():
    """A three-verdict consult envelope, derived from the MS0 fixture."""
    with open(_MS0_FIXTURE, encoding="utf-8") as handle:
        return handle.read().encode("utf-8")


def _raising(exception):
    """A zero-argument callable that raises, for stubbing a failing method."""

    def _raise():
        raise exception

    return _raise


def _payload_with_broken_verdict(agent):
    """The MS0 envelope with one mage's verdict missing a required key."""
    envelope = json.loads(open(_MS0_FIXTURE, encoding="utf-8").read())
    for verdict in envelope["consult"]["agents"]:
        if verdict["agent"] == agent:
            del verdict["recommendation"]  # one of the seven
    return json.dumps(envelope).encode("utf-8")


def _stub_pin():
    """A Pin whose contents no test in this file inspects.

    The bootstrap is stubbed out in every test here, so nothing reads it -- but
    the constructor requires it, and passing None would make a signature change
    fail as an AttributeError deep inside instead of at the call site.
    """
    return SimpleNamespace(version="0.19.1", assets={}, repository="BolivarTech/magi")


def _stub_bootstrap(monkeypatch):
    monkeypatch.setattr(
        "magi_rs_trio.BinaryBootstrap",
        lambda *a, **k: SimpleNamespace(
            ensure=lambda: "/bin/magi-rs", verify=lambda: "/bin/magi-rs"
        ),
    )


def _initialise(tmp_path):
    (tmp_path / ".magi").mkdir(exist_ok=True)
    (tmp_path / ".magi" / "magi.toml").write_text("provider='ollama'", encoding="utf-8")


_RUN_TIMEOUT = 1800  # the explicit timeout _invoke passes (arbitrary, PM-S-6)


async def _invoke(tmp_path, prompt="a payload", on_state=None):
    """Build a MagiRsTrio over tmp_path and run it. Shared by the runners."""
    trio = MagiRsTrio(
        workdir=str(tmp_path),
        mode="code-review",
        timeout=1800,
        bin_dir=str(tmp_path / "bin"),
        pin=_stub_pin(),
    )
    return await trio.run_trio(
        prompt=prompt,
        agents_dir="agents",
        output_dir=str(tmp_path),
        timeout=1800,
        models={},
        max_attempts=1,
        on_state=on_state,
    )


async def _run(monkeypatch, tmp_path, prompt="a payload", on_state=None):
    """Drive one MagiRsTrio.run_trio against a faked child process."""
    _initialise(tmp_path)
    _stub_bootstrap(monkeypatch)
    return await _invoke(tmp_path, prompt=prompt, on_state=on_state)


async def _run_without_bootstrap_stub(monkeypatch, tmp_path):
    """Like `_run`, but leaves the caller's BinaryBootstrap stub in place."""
    _initialise(tmp_path)
    return await _invoke(tmp_path)


async def _run_without_magi_dir(monkeypatch, tmp_path):
    """Like `_run`, but with NO .magi/ -- the preflight must stop the launch."""
    _stub_bootstrap(monkeypatch)
    return await _invoke(tmp_path)


def _argv(**overrides):
    base = dict(
        executable="/plugin/bin/magi-rs",
        workdir="/repo",
        mode="code-review",
        timeout=1800,
    )
    base.update(overrides)
    return build_argv(**base)


# --------------------------------------------------------------------------
# The argv
# --------------------------------------------------------------------------


def test_the_argv_uses_consult_and_not_query():
    assert _argv()[1] == "consult"
    assert "query" not in _argv()


def test_the_argv_carries_the_working_directory_explicitly():
    argv = _argv(workdir="/somewhere/else")
    assert argv[argv.index("-w") + 1] == "/somewhere/else"


def test_the_argv_requests_structured_verdicts():
    # Without it the verdict arrives only as markdown and R14 is impossible.
    assert "--structured-verdicts" in _argv()


def test_the_argv_always_declares_the_mode_explicitly():
    argv = _argv(mode="design")
    assert argv[argv.index("--mode") + 1] == "design"


def test_the_argv_always_declares_the_content_untrusted():
    assert "--untrusted-content" in _argv()


def test_the_argv_always_passes_no_memory():
    # BDD-22. Present because it is what allows running WITHOUT a passphrase --
    # not for memory isolation, which consult gives by construction.
    assert "--no-memory" in _argv()


def test_the_argv_never_carries_a_passphrase_flag():
    assert "-p" not in _argv()
    assert "--passphrase" not in _argv()


def test_the_argv_carries_the_wall_timeout():
    argv = _argv(timeout=900)
    assert argv[argv.index("--timeout") + 1] == "900"


def test_the_argv_requests_json_output():
    # Without it there is nothing to parse, and the failure surfaces as "the
    # output is not JSON" -- true, and three steps away from the cause.
    argv = _argv()
    assert argv[argv.index("--output-format") + 1] == "json"


def test_the_argv_never_carries_max_tool_calls():
    # BDD-32: this path uses no tool loop -- the material is in the payload --
    # so a tool-call bound would be a decorative flag, which the spec forbids
    # just as firmly as omitting a load-bearing one.
    assert "--max-tool-calls" not in _argv()


def test_an_unknown_mode_is_refused_before_the_process_starts():
    with pytest.raises(MagiRsContractError):
        _argv(mode="whatever")


# --------------------------------------------------------------------------
# The two error classes
# --------------------------------------------------------------------------


def test_a_deterministic_contract_failure_is_NOT_retryable():
    # D6: the orchestrator retries on ValidationError. A forgotten
    # --structured-verdicts fails identically every time, so retrying it burns
    # three real trio runs and then reports a transient failure. The guard is
    # STRUCTURAL: the class simply is not in the caught tuple.
    assert not issubclass(MagiRsContractError, ValidationError)


def test_a_process_failure_IS_retryable():
    # The mirror. A child that timed out or was OOM-killed may succeed on a
    # second attempt, which is exactly what the existing retry is for.
    assert issubclass(MagiRsProcessError, ValidationError)


# --------------------------------------------------------------------------
# The timeout
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "expected", "warns"),
    [
        (None, 16820, False),
        (16820, 16820, False),
        (16821, 16821, False),
        (16819, 16819, True),
        (1800, 1800, True),
        (60, 60, True),
    ],
)
def test_the_ollama_timeout_defaults_to_magi_rs_recommendation_and_warns_below_it(
    flag, expected, warns
):
    # v6.2.0 R2/R2a: 16820 is magi-rs 0.21.0's published gate value (replay B,
    # 2026-09-27) and derives its 2335 s default ceiling; below it the ceiling
    # falls under the only value measured with the 65 536-token cap.
    # Guard cases, green in Red on purpose: 16820, 16821 and 60.
    seconds, warning = resolve_timeout(flag)
    assert seconds == expected
    assert (warning is not None) == warns


def test_the_default_and_the_warning_threshold_are_one_public_constant():
    # D2 + DB-4: one number, public because tools/measure_marker_adherence.py consumes it.
    assert magi_rs_trio.DEFAULT_OLLAMA_TIMEOUT_SECONDS == 16820
    assert resolve_timeout(None) == (magi_rs_trio.DEFAULT_OLLAMA_TIMEOUT_SECONDS, None)
    below = magi_rs_trio.DEFAULT_OLLAMA_TIMEOUT_SECONDS - 1
    assert resolve_timeout(below)[1] is not None


def test_the_low_timeout_warning_says_the_value_the_recommendation_and_why():
    # Scenario 2: numbers alone would pass a warning that explains nothing.
    _, warning = resolve_timeout(16819)
    assert warning is not None
    assert "16819" in warning
    assert "16820" in warning
    assert "reasoning budget" in warning


@pytest.mark.parametrize("bad", [0, -1, -1800, True])
def test_a_non_positive_timeout_is_refused_deterministically(bad):
    # ``True`` is an int subclass that would otherwise slip through as 1 (gate, cycle 1).
    # The Raises block promised this and nothing executed it. Deterministic, so
    # MagiRsContractError -- a retry with the same value fails the same way.
    with pytest.raises(MagiRsContractError):
        resolve_timeout(bad)


# --------------------------------------------------------------------------
# The launch
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_prompt_travels_on_stdin_and_never_in_the_argv(monkeypatch, tmp_path):
    # The ~32K argv limit on Windows is still in force.
    seen = {}
    process = _FakeProcess(_ok_payload())

    async def _fake_exec(*argv, **kwargs):
        seen["argv"] = argv
        return process

    monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_exec)
    prompt = "x" * 100_000
    await _run(monkeypatch, tmp_path, prompt=prompt)
    assert prompt not in " ".join(seen["argv"])
    assert process.stdin_seen == prompt.encode("utf-8")


@pytest.mark.asyncio
async def test_the_child_environment_has_no_passphrase(monkeypatch, tmp_path):
    seen = {}

    async def _fake_exec(*argv, **kwargs):
        seen["env"] = kwargs["env"]
        return _FakeProcess(_ok_payload())

    monkeypatch.setenv("MAGI_PASSPHRASE", "hunter2")
    monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_exec)
    await _run(monkeypatch, tmp_path)
    assert "MAGI_PASSPHRASE" not in seen["env"]


@pytest.mark.asyncio
async def test_a_nonzero_exit_raises_relaying_the_code_and_stderr(monkeypatch, tmp_path):
    # BDD-14a: no report is produced, and NOTHING is recovered from the
    # partial output.
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec",
        _exec_returning(_FakeProcess(b"", stderr=b"boom", returncode=3)),
    )
    # PROCESS, not CONTRACT: the child ran and failed, which may be an OOM kill
    # or a transient CLI fault, so the orchestrator's retry is right to fire.
    with pytest.raises(MagiRsProcessError) as exc:
        await _run(monkeypatch, tmp_path)
    assert "3" in str(exc.value) and "boom" in str(exc.value)


@pytest.mark.asyncio
async def test_a_process_failure_names_the_allowlist_and_what_it_withheld(monkeypatch, tmp_path):
    # R7: the allowlist's failure mode is a certificate or host error that
    # looks nothing like "a variable was missing", so the message names the
    # allowlist as a suspect and lists the parent variables that did not travel
    # -- names only, never values.
    monkeypatch.setenv("CORP_PROXY_CA_HINT", "/etc/corp.pem")
    monkeypatch.setenv("MAGI_PASSPHRASE", "hunter2")
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec",
        _exec_returning(_FakeProcess(b"", stderr=b"certificate verify failed", returncode=1)),
    )
    with pytest.raises(MagiRsProcessError) as exc:
        await _run(monkeypatch, tmp_path)
    message = str(exc.value)
    assert "allowlist" in message
    assert "CORP_PROXY_CA_HINT" in message
    assert "hunter2" not in message and "/etc/corp.pem" not in message


@pytest.mark.asyncio
async def test_output_that_is_not_json_says_so_and_never_scrapes_the_markdown(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec",
        _exec_returning(_FakeProcess(b"## Report\n\nnot json at all")),
    )
    with pytest.raises(MagiRsContractError) as exc:
        await _run(monkeypatch, tmp_path)
    assert "json" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_json_without_the_agents_key_names_the_forgotten_flag(monkeypatch, tmp_path):
    # BDD-14b: the two failures are DIFFERENT and the message must say which
    # one happened. A missing `agents` means --structured-verdicts was omitted.
    payload = json.dumps({"consult": {"report": "# markdown"}, "stop_reason": "done"})
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec",
        _exec_returning(_FakeProcess(payload.encode())),
    )
    with pytest.raises(MagiRsContractError) as exc:
        await _run(monkeypatch, tmp_path)
    assert "--structured-verdicts" in str(exc.value)


@pytest.mark.asyncio
async def test_run_trio_returns_one_validated_verdict_per_mage(monkeypatch, tmp_path):
    # The happy path, absent from the first draft: every test asserted a
    # REJECTION, and a suite like that is compatible with a run_trio that can
    # never succeed. (The same shape of hole the MS2 gate found in verify().)
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", _exec_returning(_FakeProcess(_ok_payload()))
    )
    verdicts = await _run(monkeypatch, tmp_path)
    assert set(verdicts) == {"melchior", "balthasar", "caspar"}
    assert all(isinstance(v, dict) for v in verdicts.values())
    assert all(set(v) >= {"agent", "verdict", "confidence"} for v in verdicts.values())


@pytest.mark.asyncio
async def test_run_trio_persists_the_raw_envelope_as_its_own_artifact(monkeypatch, tmp_path):
    # R16: the raw magi-rs output is kept under a name of its own, so two
    # producers never share one file name.
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", _exec_returning(_FakeProcess(_ok_payload()))
    )
    await _run(monkeypatch, tmp_path)
    assert (tmp_path / "magi-rs-raw.json").read_bytes() == _ok_payload()


@pytest.mark.asyncio
async def test_a_verdict_breaking_the_seven_key_contract_is_rejected_per_mage(
    monkeypatch, tmp_path
):
    # BDD-14c: rejected EXACTLY as today, with no new validation code and no
    # relaxing of the contract -- and it does not take the other two down.
    payload = _payload_with_broken_verdict("caspar")
    monkeypatch.setattr("asyncio.create_subprocess_exec", _exec_returning(_FakeProcess(payload)))
    verdicts = await _run(monkeypatch, tmp_path)
    assert isinstance(verdicts["caspar"], ValidationError)
    # BOTH other seats survive: a _validate that dropped the rest on one
    # failure would still have passed a melchior-only assertion (gate, cycle 1).
    assert isinstance(verdicts["melchior"], dict) and isinstance(verdicts["balthasar"], dict)


@pytest.mark.asyncio
async def test_a_binary_that_does_not_verify_stops_the_launch(monkeypatch, tmp_path):
    # R4 wired, not merely asserted in prose: nothing is launched.
    launched: list[object] = []
    monkeypatch.setattr("asyncio.create_subprocess_exec", lambda *a, **k: launched.append(a))
    monkeypatch.setattr(
        "magi_rs_trio.BinaryBootstrap",
        lambda *a, **k: SimpleNamespace(
            ensure=lambda: "/bin/magi-rs",
            verify=_raising(DigestMismatchError("digest differs")),
        ),
    )
    with pytest.raises(DigestMismatchError):
        await _run_without_bootstrap_stub(monkeypatch, tmp_path)
    assert launched == []


@pytest.mark.asyncio
async def test_an_uninitialised_project_stops_the_launch(monkeypatch, tmp_path):
    # Fail before spending: the preflight costs nothing, the launch costs a run.
    launched: list[object] = []
    monkeypatch.setattr("asyncio.create_subprocess_exec", lambda *a, **k: launched.append(a))
    with pytest.raises(ProjectNotInitialisedError):
        await _run_without_magi_dir(monkeypatch, tmp_path)
    assert launched == []


@pytest.mark.asyncio
async def test_run_trio_emits_ONE_status_node_not_three(monkeypatch, tmp_path):
    """D5/BDD-27, executed instead of asserted.

    The plugin launches ONE child that runs the whole trio, so it cannot know
    when an individual mage starts or finishes. Three nodes would be a display
    that INVENTS progress -- worse than none, because it looks informed.
    """
    seen = []
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", _exec_returning(_FakeProcess(_ok_payload()))
    )
    await _run(monkeypatch, tmp_path, on_state=lambda node, state: seen.append((node, state)))

    nodes = {node for node, _ in seen}
    assert len(nodes) == 1, f"one child, one node -- got {nodes}"
    assert not (nodes & {"melchior", "balthasar", "caspar"}), "per-mage nodes are invented progress"


@pytest.mark.asyncio
async def test_a_status_callback_that_raises_does_not_take_the_run_down(monkeypatch, tmp_path):
    """The verdicts are the product; the tree is decoration.

    Same fail-soft contract `_safe_display_update` already has on the Claude
    path -- and the reason `_emit` swallows `Exception` rather than letting it
    through.
    """
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", _exec_returning(_FakeProcess(_ok_payload()))
    )

    def _explode(node, state):
        raise RuntimeError("the display is broken")

    verdicts = await _run(monkeypatch, tmp_path, on_state=_explode)
    assert set(verdicts) == {"melchior", "balthasar", "caspar"}


# --------------------------------------------------------------------------
# Task 5 -- the wiring: ``--ollama`` selects the magi-rs executor
# --------------------------------------------------------------------------


def _args(**overrides):
    """A parsed-args namespace shaped like run_magi's, for the wiring tests.

    Field names measured from ``parse_args`` (Task 5, Step 1): ``ollama``,
    ``model``, ``mode``, ``timeout``.
    """
    fields = dict(ollama=False, model=None, mode="design", timeout=None)
    fields.update(overrides)
    return SimpleNamespace(**fields)


class _StubTrio:
    """A TrioExecutor that returns three canned verdicts without launching."""

    def __init__(self):
        self.calls = 0

    async def run_trio(
        self, prompt, agents_dir, output_dir, timeout, models, max_attempts, on_state=None
    ):
        self.calls += 1
        envelope = json.loads(_ok_payload().decode("utf-8"))
        return {v["agent"]: v for v in envelope["consult"]["agents"]}


def test_the_ollama_flag_selects_the_magi_rs_executor(monkeypatch, tmp_path):
    import run_magi

    chosen = {}

    def _magi_rs(*args, **kwargs):
        chosen["executor"] = "magi-rs"
        return _StubTrio()

    monkeypatch.setattr(run_magi, "MagiRsTrio", _magi_rs)
    monkeypatch.chdir(tmp_path)
    executor = run_magi._select_executor(_args(ollama=True), 1800)
    assert chosen["executor"] == "magi-rs"
    assert isinstance(executor, _StubTrio)


def test_the_magi_rs_executor_is_bound_to_the_cwd_mode_timeout_and_plugin_paths(
    monkeypatch, tmp_path
):
    # R6 (-w is the project root, never inherited implicitly), R10 (the mode
    # is always explicit) and R3 (bin/ + pin live under the plugin root).
    import run_magi

    seen = {}

    def _record(**kwargs):
        seen.update(kwargs)
        return _StubTrio()

    monkeypatch.setattr(run_magi, "MagiRsTrio", _record)
    monkeypatch.chdir(tmp_path)
    run_magi._select_executor(_args(ollama=True, mode="code-review"), 1800)
    assert seen["workdir"] == os.getcwd()
    assert seen["mode"] == "code-review"
    assert seen["timeout"] == 1800
    assert os.path.basename(seen["bin_dir"]) == "bin"
    assert seen["pin"].version == "0.24.1"


def test_without_the_flag_no_executor_is_selected(monkeypatch):
    # The Claude path keeps ClaudeTrio bound inside run_orchestrator to its own
    # telemetry (R20): the selector must not construct anything for it.
    import run_magi

    monkeypatch.setattr(run_magi, "MagiRsTrio", _raising(AssertionError("constructed")))
    assert run_magi._select_executor(_args(ollama=False), 900) is None


@pytest.mark.asyncio
async def test_run_orchestrator_runs_the_given_executor_instead_of_claude(monkeypatch, tmp_path):
    # Everything below the raw verdicts does NOT fork: the same run_orchestrator
    # consumes what the magi-rs executor returns.
    import run_magi

    monkeypatch.setattr(run_magi, "ClaudeTrio", _raising(AssertionError("ClaudeTrio built")))
    stub = _StubTrio()
    report = await run_magi.run_orchestrator(
        agents_dir=str(tmp_path),
        prompt="payload",
        output_dir=str(tmp_path),
        timeout=1800,
        executor=stub,
        show_status=False,
    )
    assert stub.calls == 1
    assert {a["agent"] for a in report["agents"]} == {"melchior", "balthasar", "caspar"}
    assert "consensus" in report


def _main_with_ollama(monkeypatch, tmp_path, *extra_argv):
    """Drive run_magi.main() under --ollama up to the orchestrator, which is stubbed.

    Returns the SystemExit code main() ended with (0 when the stub orchestrator
    was reached).
    """
    import sys

    import run_magi

    async def _reached_orchestrator(*args, **kwargs):
        raise SystemExit(0)

    monkeypatch.setattr(run_magi, "run_orchestrator", _reached_orchestrator)
    monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
    monkeypatch.setattr(
        sys, "argv", ["run_magi.py", "design", "hello", "--ollama", "--no-status", *extra_argv]
    )
    with pytest.raises(SystemExit) as exc:
        run_magi.main()
    return exc.value.code


def _hermetic_project(monkeypatch, tmp_path):
    """An initialised project under tmp_path with the bootstrap stubbed.

    Loop 1 caught two tests reaching the REAL bin/ and .magi/ of this checkout
    (they passed only because SMOKE #1 had left both behind): every main()-level
    test that must get past the preflight goes through here instead.
    """
    _stub_bootstrap(monkeypatch)
    _initialise(tmp_path)
    monkeypatch.chdir(tmp_path)


def test_a_low_timeout_under_ollama_reaches_the_run_with_a_stderr_warning(
    monkeypatch, tmp_path, capsys
):
    # R12: honoured, never silently -- the warning names the measured-healthy floor.
    _hermetic_project(monkeypatch, tmp_path)
    assert _main_with_ollama(monkeypatch, tmp_path, "--timeout", "600") == 0
    err = capsys.readouterr().err
    assert "[!] WARNING:" in err and "--timeout 600" in err


def test_a_timeout_the_magi_rs_contract_rejects_is_an_ERROR_line_not_a_traceback(
    monkeypatch, tmp_path, capsys
):
    import run_magi

    def _reject(flag_value):
        raise MagiRsContractError("--timeout must be a positive integer, got 0")

    monkeypatch.setattr(run_magi, "resolve_magi_rs_timeout", _reject)
    assert _main_with_ollama(monkeypatch, tmp_path) == 1
    err = capsys.readouterr().err
    assert "ERROR: --timeout must be a positive integer" in err and "Traceback" not in err


def test_a_plugin_root_that_does_not_verify_is_FATAL_under_ollama(monkeypatch, tmp_path, capsys):
    # The same CLI-boundary shape as --ollama-init: a fail-closed sibling of
    # Exception becomes one [FATAL] line and exit 1, never a traceback.
    import run_magi
    from magi_rs_paths import PluginRootError

    def _unverified_root(args, timeout):
        raise PluginRootError("CLAUDE_PLUGIN_ROOT=/nope")

    monkeypatch.setattr(run_magi, "_select_executor", _unverified_root)
    assert _main_with_ollama(monkeypatch, tmp_path) == 1
    err = capsys.readouterr().err
    assert "[FATAL] CLAUDE_PLUGIN_ROOT=/nope" in err and "Traceback" not in err


# SMOKE #1 (Task 6, S3) found these: the preflight ran inside the orchestrator,
# after the banner and the temp dir, and escaped main() as a traceback.


def test_an_uninitialised_project_is_FATAL_before_the_payload_is_built(
    monkeypatch, tmp_path, capsys
):
    # R7/BDD-7 + the spec's own order (bootstrap and preflight BEFORE the
    # payload): fail-closed, names the command, and nothing downstream runs.
    import run_magi

    _stub_bootstrap(monkeypatch)
    monkeypatch.chdir(tmp_path)  # no .magi/ here

    def _payload_built(arg):
        raise AssertionError("the payload was built before the preflight")

    monkeypatch.setattr(run_magi, "_load_input_content", _payload_built)
    assert _main_with_ollama(monkeypatch, tmp_path) == 1
    err = capsys.readouterr().err
    assert "[FATAL]" in err and "/magi --ollama-init" in err and "Traceback" not in err
    assert not (tmp_path / ".magi").exists()


def test_the_binary_is_verified_before_the_payload_is_built(monkeypatch, tmp_path, capsys):
    # R4 at the CLI boundary: a digest that does not verify is one [FATAL] line.
    import run_magi

    monkeypatch.setattr(
        "magi_rs_trio.BinaryBootstrap",
        lambda *a, **k: SimpleNamespace(
            ensure=lambda: "/bin/magi-rs", verify=_raising(DigestMismatchError("digest differs"))
        ),
    )
    monkeypatch.chdir(tmp_path)

    def _payload_built(arg):
        raise AssertionError("the payload was built before the binary verified")

    monkeypatch.setattr(run_magi, "_load_input_content", _payload_built)
    assert _main_with_ollama(monkeypatch, tmp_path) == 1
    err = capsys.readouterr().err
    assert "[FATAL] digest differs" in err and "Traceback" not in err


@pytest.mark.parametrize(
    "failure",
    [MagiRsProcessError("magi-rs exited 2: boom"), MagiRsContractError("consult is null")],
    ids=["process", "contract"],
)
def test_a_failed_magi_rs_run_is_an_ERROR_line_not_a_traceback(
    monkeypatch, tmp_path, capsys, failure
):
    # BDD-14a/14b/14d: no report, the cause relayed, a non-zero exit -- and the
    # operator can tell "the trio never formed" from "the plugin broke".
    import run_magi

    _hermetic_project(monkeypatch, tmp_path)

    async def _dies(*args, **kwargs):
        raise failure

    monkeypatch.setattr(run_magi, "run_orchestrator", _dies)
    monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
    import sys

    monkeypatch.setattr(sys, "argv", ["run_magi.py", "design", "hello", "--ollama", "--no-status"])
    with pytest.raises(SystemExit) as exc:
        run_magi.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert f"ERROR: {failure}" in err and "Traceback" not in err
    assert not list(tmp_path.glob("**/magi-report.json"))


@pytest.mark.parametrize(
    "failure",
    [DigestMismatchError("digest differs"), ProjectNotInitialisedError("gone")],
    ids=["digest", "deinitialised"],
)
def test_a_preflight_failure_inside_the_run_is_FATAL_not_a_traceback(
    monkeypatch, tmp_path, capsys, failure
):
    # Loop 1: run_trio re-runs the preflight right before launching (R4). If
    # the binary or the project changed between the two checks, that failure
    # must reach the operator the same way the first one does.
    import sys

    import run_magi

    _hermetic_project(monkeypatch, tmp_path)

    async def _dies(*args, **kwargs):
        raise failure

    monkeypatch.setattr(run_magi, "run_orchestrator", _dies)
    monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["run_magi.py", "design", "hello", "--ollama", "--no-status"])
    with pytest.raises(SystemExit) as exc:
        run_magi.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert f"[FATAL] {failure}" in err and "Traceback" not in err


# Loop 2 cycle 1 (the gate itself) hit this: magi-rs reported Caspar in
# ``failed_agents`` and left it out of ``agents`` (degraded 2-of-3), and the
# executor returned a dict WITHOUT a caspar entry -- run_orchestrator died on
# ``verdicts["caspar"]`` (KeyError) instead of entering the degraded path.


def _payload_with_dead_mage(agent, cause):
    """The MS0 envelope as magi-rs emits it when one mage never delivered."""
    envelope = json.loads(open(_MS0_FIXTURE, encoding="utf-8").read())
    consult = envelope["consult"]
    consult["agents"] = [v for v in consult["agents"] if v["agent"] != agent]
    consult["failed_agents"] = {agent: cause}
    consult["degraded"] = True
    return json.dumps(envelope).encode("utf-8")


@pytest.mark.asyncio
async def test_a_mage_magi_rs_reports_as_failed_is_returned_as_its_failure(monkeypatch, tmp_path):
    # The contract: ONE entry per mage, the verdict or ITS exception. A mage
    # absent from ``agents`` is a dead mage, and its cause is what magi-rs said.
    cause = "no_fitting_candidate: timeout: request failed for http://localhost:11434/"
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec",
        _exec_returning(_FakeProcess(_payload_with_dead_mage("caspar", cause))),
    )
    verdicts = await _run(monkeypatch, tmp_path)
    assert set(verdicts) == {"melchior", "balthasar", "caspar"}
    assert isinstance(verdicts["caspar"], RuntimeError) and cause in str(verdicts["caspar"])
    assert isinstance(verdicts["melchior"], dict) and isinstance(verdicts["balthasar"], dict)


@pytest.mark.asyncio
async def test_a_two_of_three_envelope_degrades_the_run_instead_of_crashing_it(
    monkeypatch, tmp_path, capsys
):
    # D7: exactly two valid verdicts -> the EXISTING degraded machinery, flag
    # in the report and warning on stderr. Never a KeyError.
    import run_magi

    monkeypatch.setattr(
        "asyncio.create_subprocess_exec",
        _exec_returning(_FakeProcess(_payload_with_dead_mage("caspar", "timeout"))),
    )
    _initialise(tmp_path)
    _stub_bootstrap(monkeypatch)
    executor = MagiRsTrio(
        workdir=str(tmp_path),
        mode="design",
        timeout=1800,
        bin_dir=str(tmp_path / "bin"),
        pin=_stub_pin(),
    )
    report = await run_magi.run_orchestrator(
        agents_dir=str(tmp_path),
        prompt="payload",
        output_dir=str(tmp_path),
        timeout=1800,
        executor=executor,
        show_status=False,
    )
    assert report["degraded"] is True and report["failed_agents"] == ["caspar"]
    assert {a["agent"] for a in report["agents"]} == {"melchior", "balthasar"}
    assert "caspar" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_a_payload_that_cannot_be_utf8_encoded_is_refused_before_launch(
    monkeypatch, tmp_path
):
    # Gate (cycle 3): a lone surrogate -- possible when argv or a file arrives
    # via surrogateescape -- raised UnicodeEncodeError from inside communicate(),
    # a traceback no boundary rendered. It is a contract failure of the payload,
    # deterministic, and it is caught BEFORE any process starts.
    launched = []

    async def _never(*argv, **kwargs):
        launched.append(argv)
        raise AssertionError("the child must not start")

    monkeypatch.setattr("asyncio.create_subprocess_exec", _never)
    with pytest.raises(MagiRsContractError, match="UTF-8"):
        await _run(monkeypatch, tmp_path, prompt="payload with a lone surrogate \ud800")
    assert launched == []


# Gate, cycle 4 -- three defensive gaps in the result and timeout paths.


@pytest.mark.asyncio
async def test_a_verdict_carrying_a_lone_surrogate_fails_its_own_seat_only(monkeypatch, tmp_path):
    # json.loads happily produces a lone surrogate from a "\ud800" escape;
    # writing that seat's {mage}.json as UTF-8 raised UnicodeEncodeError past
    # every boundary. It is THAT seat's failure, and the other two survive.
    envelope = json.loads(_ok_payload().decode("utf-8"))
    for verdict in envelope["consult"]["agents"]:
        if verdict["agent"] == "caspar":
            verdict["summary"] = "lone \ud800 surrogate"
    payload = json.dumps(envelope).encode("utf-8")  # json escapes it; the file write cannot
    monkeypatch.setattr("asyncio.create_subprocess_exec", _exec_returning(_FakeProcess(payload)))
    verdicts = await _run(monkeypatch, tmp_path)
    assert isinstance(verdicts["caspar"], UnicodeEncodeError)
    assert isinstance(verdicts["melchior"], dict) and isinstance(verdicts["balthasar"], dict)


@pytest.mark.asyncio
async def test_a_timeout_names_the_allowlist_as_a_suspect_too(monkeypatch, tmp_path):
    # A proxy variable the allowlist withheld makes the child HANG on connect,
    # not exit non-zero: the diagnosis must reach the timeout path as well.
    class _Hanging(_FakeProcess):
        async def communicate(self, input=None):
            raise asyncio.TimeoutError()

    monkeypatch.setattr("asyncio.create_subprocess_exec", _exec_returning(_Hanging()))
    monkeypatch.setenv("ALL_PROXY", "http://proxy.corp:3128")
    with pytest.raises(MagiRsProcessError) as exc:
        await _run(monkeypatch, tmp_path)
    assert "ALL_PROXY" in str(exc.value) and "allowlist" in str(exc.value)


@pytest.mark.asyncio
async def test_run_trio_launches_with_the_constructor_timeout_not_its_argument(
    monkeypatch, tmp_path
):
    # Pinned, not only documented: the argv bound and the wall bound are one
    # value, the constructor's, so they cannot race.
    seen = {}

    async def _record(*argv, **kwargs):
        seen["argv"] = argv
        return _FakeProcess(_ok_payload())

    monkeypatch.setattr("asyncio.create_subprocess_exec", _record)
    _initialise(tmp_path)
    _stub_bootstrap(monkeypatch)
    trio = MagiRsTrio(
        workdir=str(tmp_path),
        mode="design",
        timeout=4242,
        bin_dir=str(tmp_path / "bin"),
        pin=_stub_pin(),
    )
    await trio.run_trio(
        prompt="p", agents_dir="a", output_dir=str(tmp_path), timeout=1, models={}, max_attempts=1
    )
    assert seen["argv"][-1] == "4242"


# MS4 Task 4 -- the wiring: the report gets the magi_rs block, the unpriced cost,
# and the warnings, from the executor's last result.


def test_main_writes_the_magi_rs_block_the_unpriced_cost_and_warns(monkeypatch, tmp_path, capsys):
    import sys

    import run_magi
    from magi_rs_result import ConsultResult

    envelope = json.loads(_ok_payload().decode("utf-8"))
    envelope["consult"]["ran_unmeasured"] = ["balthasar", "caspar", "melchior"]
    result = ConsultResult.from_json(json.dumps(envelope))

    class _Executor:
        last_result = result

        def preflight(self):
            return "magi-rs"

    monkeypatch.setattr(run_magi, "_select_executor", lambda args, timeout: _Executor())
    verdicts = {v["agent"]: v for v in result.agents}

    async def _orch(*args, **kwargs):
        from synthesize import determine_consensus

        agents = [verdicts[n] for n in ("caspar", "melchior", "balthasar")]
        return {"agents": agents, "consensus": determine_consensus(agents)}

    monkeypatch.setattr(run_magi, "run_orchestrator", _orch)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_magi.py", "design", "hello", "--ollama", "--no-status", "--output-dir", str(out)],
    )
    run_magi.main()
    report = json.loads((out / "magi-report.json").read_text(encoding="utf-8"))
    assert report["cost"] == {"total_usd": None, "per_agent": {}, "cost_source": "unpriced"}
    assert report["magi_rs"]["run_id"] == result.run_id
    assert report["magi_rs"]["gate_ran_on_unverified_context"] is True
    assert report["model"]["served"] is None  # D8, never the tag used
    err = capsys.readouterr().err
    assert "NOBODY MEASURED" in err


def test_main_shouts_invalid_when_a_verdict_fails_the_contract_locally(
    monkeypatch, tmp_path, capsys
):
    # Loop 1 (MS4): the D6/BDD-28 path -- magi-rs says degraded: false, one
    # verdict breaks the seven-key contract here -> 2 valid seats -> the run is
    # degraded on THIS side and the gate-INVALID warning must still fire.
    import sys

    import run_magi

    payload = _payload_with_broken_verdict("caspar")
    monkeypatch.setattr("asyncio.create_subprocess_exec", _exec_returning(_FakeProcess(payload)))
    _hermetic_project(monkeypatch, tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_magi.py", "design", "hello", "--ollama", "--no-status", "--output-dir", str(out)],
    )
    run_magi.main()
    report = json.loads((out / "magi-report.json").read_text(encoding="utf-8"))
    assert report["degraded"] is True and report["magi_rs"]["degraded"] is False
    err = capsys.readouterr().err.lower()
    assert "the gate is invalid" in err


@pytest.mark.asyncio
async def test_a_long_stderr_keeps_its_tail_where_the_real_cause_is(monkeypatch, tmp_path):
    # Scenario 15: 0.20+ writes WARN lines at startup; the cause comes last.
    stderr = b"WARN clock coverage\n" * 200 + b"FATAL: the real cause"
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec",
        _exec_returning(_FakeProcess(b"", stderr=stderr, returncode=2)),
    )
    with pytest.raises(MagiRsProcessError) as exc:
        await _run(monkeypatch, tmp_path)
    message = str(exc.value)
    assert "FATAL: the real cause" in message
    assert _STDERR_ELISION_MARKER in message


@pytest.mark.asyncio
async def test_a_short_stderr_goes_whole_and_unmarked(monkeypatch, tmp_path):
    # Scenario 17. Guard test: green in Red on purpose; protects the marker from
    # appearing when nothing was cut.
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec",
        _exec_returning(_FakeProcess(b"", stderr=b"boom", returncode=3)),
    )
    with pytest.raises(MagiRsProcessError) as exc:
        await _run(monkeypatch, tmp_path)
    assert "boom" in str(exc.value)
    assert _STDERR_ELISION_MARKER not in str(exc.value)


@pytest.mark.asyncio
async def test_a_non_utf8_stderr_still_shows_its_last_line(monkeypatch, tmp_path):
    # Review Focus 2.
    stderr = b"\xff\xfe garbage " * 400 + b"FATAL: tail survives"
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec",
        _exec_returning(_FakeProcess(b"", stderr=stderr, returncode=1)),
    )
    with pytest.raises(MagiRsProcessError) as exc:
        await _run(monkeypatch, tmp_path)
    assert "FATAL: tail survives" in str(exc.value)


@pytest.mark.asyncio
async def test_the_outer_clock_is_exactly_the_timeout_because_m4_found_no_overshoot(
    monkeypatch, tmp_path
):
    # R3a row 1 (planning/v0.21.0-verification/M4): no margin, measured, not assumed.
    # Guard test: green on purpose; fixes the measured decision.
    seen = {}
    real_wait_for = asyncio.wait_for

    async def _spy(awaitable, timeout):
        seen["timeout"] = timeout
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(magi_rs_trio.asyncio, "wait_for", _spy)
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", _exec_returning(_FakeProcess(_ok_payload()))
    )
    await _run(monkeypatch, tmp_path)
    assert seen["timeout"] == _RUN_TIMEOUT
