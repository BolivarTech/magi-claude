"""TrioExecutor is an abstract contract at the TRIO level, not per mage."""

import asyncio
import inspect
import json
import os

import pytest

from synthesize import ValidationError
import run_magi
from trio_executor import ClaudeTrio, TrioExecutor


def test_trio_executor_cannot_be_instantiated():
    with pytest.raises(TypeError):
        TrioExecutor()  # type: ignore[abstract]


def test_run_trio_is_abstract_coroutine():
    assert getattr(TrioExecutor.run_trio, "__isabstractmethod__", False) is True
    assert inspect.iscoroutinefunction(TrioExecutor.run_trio)


def test_subclass_without_run_trio_is_abstract():
    class Incomplete(TrioExecutor):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


_VALID = {
    "agent": "melchior",
    "verdict": "approve",
    "confidence": 0.9,
    "summary": "s",
    "reasoning": "r",
    "findings": [],
    "recommendation": "rec",
}


def _envelope(agent):
    body = dict(_VALID, agent=agent)
    return json.dumps(
        {"result": "<MAGI_VERDICT>\n" + json.dumps(body) + "\n</MAGI_VERDICT>"}
    ).encode("utf-8")


class _FakeProc:
    def __init__(self, stdout, stderr=b"", returncode=0):
        self._stdout, self._stderr, self.returncode = stdout, stderr, returncode

    async def communicate(self, input=None):
        return self._stdout, self._stderr


def _write_prompts(tmp_path):
    for mage in ("melchior", "balthasar", "caspar"):
        (tmp_path / f"{mage}.md").write_text("SYS", encoding="utf-8")


def test_run_trio_invokes_claude_with_resolved_model(monkeypatch, tmp_path):
    captured = []

    async def fake_exec(*args, **kwargs):
        agent = next(m for m in ("melchior", "balthasar", "caspar") if f"{m}.md" in " ".join(args))
        captured.append(args)
        return _FakeProc(_envelope(agent))

    monkeypatch.setattr("trio_executor.asyncio.create_subprocess_exec", fake_exec)
    _write_prompts(tmp_path)

    out = asyncio.run(
        ClaudeTrio().run_trio(
            "PROMPT",
            str(tmp_path),
            str(tmp_path),
            900,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            2,
        )
    )

    assert set(out) == {"melchior", "balthasar", "caspar"}
    assert out["melchior"]["verdict"] == "approve"
    assert all(args[0] == "claude" for args in captured)
    assert all("claude-opus-5" in args for args in captured)  # resolve_model("opus")


def test_run_trio_returns_the_runtimeerror_from_a_nonzero_exit(monkeypatch, tmp_path):
    async def fake_exec(*args, **kwargs):
        if "caspar.md" in " ".join(args):
            return _FakeProc(b"", stderr=b"boom", returncode=2)
        agent = next(m for m in ("melchior", "balthasar") if f"{m}.md" in " ".join(args))
        return _FakeProc(_envelope(agent))

    monkeypatch.setattr("trio_executor.asyncio.create_subprocess_exec", fake_exec)
    _write_prompts(tmp_path)

    out = asyncio.run(
        ClaudeTrio().run_trio(
            "P",
            str(tmp_path),
            str(tmp_path),
            900,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            2,
        )
    )

    # ONE ENTRY PER MAGE, ALWAYS: caspar's failure is returned, not omitted.
    assert set(out) == {"melchior", "balthasar", "caspar"}
    assert isinstance(out["caspar"], RuntimeError)
    assert "exited with code 2" in str(out["caspar"])
    assert isinstance(out["melchior"], dict)


def test_run_trio_retries_one_mage_with_corrective_feedback(monkeypatch, tmp_path):
    """The retry must re-prompt ONLY the failing mage, with the feedback appended."""
    attempts = {"caspar": 0}
    seen_prompts = []

    async def fake_exec(*args, **kwargs):
        joined = " ".join(args)
        agent = next(m for m in ("melchior", "balthasar", "caspar") if f"{m}.md" in joined)
        if agent == "caspar":
            attempts["caspar"] += 1
            if attempts["caspar"] == 1:
                return _FakeProc(b'{"result": "no markers here"}')
        return _FakeProc(_envelope(agent))

    class _CapturingProc(_FakeProc):
        pass

    monkeypatch.setattr("trio_executor.asyncio.create_subprocess_exec", fake_exec)
    _write_prompts(tmp_path)

    out = asyncio.run(
        ClaudeTrio(
            build_retry_prompt=lambda p, e: seen_prompts.append(p) or (p + "\nFIXIT")
        ).run_trio(
            "P",
            str(tmp_path),
            str(tmp_path),
            900,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            2,
        )
    )

    assert attempts["caspar"] == 2  # retried exactly once
    assert set(out) == {"melchior", "balthasar", "caspar"}
    assert len(seen_prompts) == 1  # ONLY caspar was re-prompted


killed = {"caspar": False}


class _FakeStderrStream:
    """Minimal stand-in for proc.stderr so reap_and_drain_stderr works unmocked.

    Returns NON-EMPTY bytes deliberately: `write_stderr_log` returns early on
    empty data, so an empty stream would write no file and every assertion
    about the log would pass vacuously.
    """

    async def read(self, n=-1):
        return b"hung"


class _HangingProc:
    """A child that never returns, shared by the timeout tests."""

    returncode = None
    pid = 4242  # windows_kill_tree names the pid in its warning

    def __init__(self):
        self.stderr = _FakeStderrStream()

    async def communicate(self, input=None):
        await asyncio.sleep(3600)  # never returns; wait_for cancels it

    def kill(self):
        killed["caspar"] = True

    async def wait(self):
        self.returncode = -9
        return -9


def test_run_trio_returns_the_timeout_error_and_persists_stderr(monkeypatch, tmp_path):
    """The timeout path kills, reaps and persists — it must not be left untested."""
    killed["caspar"] = False

    async def fake_exec(*args, **kwargs):
        joined = " ".join(args)
        if "caspar.md" in joined:
            return _HangingProc()
        agent = next(m for m in ("melchior", "balthasar") if f"{m}.md" in joined)
        return _FakeProc(_envelope(agent))

    monkeypatch.setattr("trio_executor.asyncio.create_subprocess_exec", fake_exec)
    _write_prompts(tmp_path)

    out = asyncio.run(
        ClaudeTrio().run_trio(
            "P",
            str(tmp_path),
            str(tmp_path),
            1,  # 1s timeout
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            2,
        )
    )

    assert isinstance(out["caspar"], TimeoutError)
    assert "timed out after 1s" in str(out["caspar"])
    assert killed["caspar"], "the hung subprocess must be killed, not leaked"
    # The test's NAME promised this and its body did not check it: a name that
    # over-claims is worse than a missing test, because the suite index reads
    # like coverage. `_FakeStderrStream` returns b"hung" on purpose --
    # `write_stderr_log` returns early on empty data, so an empty stream would
    # produce no file and the assertion would be vacuous.
    assert (tmp_path / "caspar.stderr.log").read_text(encoding="utf-8").strip() == "hung"


def test_a_timeout_is_NOT_retried(monkeypatch, tmp_path):
    """`max_attempts` governs schema failures only -- a timeout is terminal.

    The contract says so in run_trio's Raises block ("Not retried"), and until
    this test existed nothing executed it. Retrying a timeout would triple the
    wall clock of a hung mage and change what the orchestrator sees.
    """
    attempts = {"caspar": 0}

    async def fake_exec(*args, **kwargs):
        joined = " ".join(args)
        if "caspar.md" in joined:
            attempts["caspar"] += 1
            return _HangingProc()
        agent = next(m for m in ("melchior", "balthasar") if f"{m}.md" in joined)
        return _FakeProc(_envelope(agent))

    monkeypatch.setattr("trio_executor.asyncio.create_subprocess_exec", fake_exec)
    _write_prompts(tmp_path)
    out = asyncio.run(
        ClaudeTrio().run_trio(
            "P",
            str(tmp_path),
            str(tmp_path),
            1,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            3,  # THREE attempts allowed, and still only one is spent
        )
    )
    assert isinstance(out["caspar"], TimeoutError)
    assert attempts["caspar"] == 1, "a timeout must not be retried"


def test_a_nonzero_exit_is_NOT_retried(monkeypatch, tmp_path):
    """The mirror of the test above, for the other terminal failure.

    A non-zero exit means the CLI itself refused -- a bad flag, a missing
    system prompt. Retrying reproduces it exactly, which is the deterministic
    retry the token-economy rules forbid.
    """
    attempts = {"caspar": 0}

    async def fake_exec(*args, **kwargs):
        joined = " ".join(args)
        if "caspar.md" in joined:
            attempts["caspar"] += 1
            return _FakeProc(b"", stderr=b"bad flag", returncode=2)
        agent = next(m for m in ("melchior", "balthasar") if f"{m}.md" in joined)
        return _FakeProc(_envelope(agent))

    monkeypatch.setattr("trio_executor.asyncio.create_subprocess_exec", fake_exec)
    _write_prompts(tmp_path)
    out = asyncio.run(
        ClaudeTrio().run_trio(
            "P",
            str(tmp_path),
            str(tmp_path),
            30,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            3,
        )
    )
    assert isinstance(out["caspar"], RuntimeError)
    assert attempts["caspar"] == 1, "a non-zero exit must not be retried"


class _CustomFatal(BaseException):
    """A BaseException subclass that is NOT KeyboardInterrupt/SystemExit."""


def test_run_trio_returns_the_exact_exception_type_including_baseexception(monkeypatch, tmp_path):
    """The orchestrator branches on exception SUBTYPE; a wrapper would swallow it.

    MEASURED on CPython 3.14, not assumed: `asyncio.gather(return_exceptions=True)`
    DOES capture BaseException subclasses and CancelledError. This is CPython
    behaviour, not a documented language guarantee -- which is why it is pinned by
    a test rather than trusted: if a future runtime changes it, this fails loudly
    instead of silently dropping a mage's failure. The two it does NOT capture are
    `KeyboardInterrupt` and `SystemExit` — the event loop gives those special
    treatment, they escape `asyncio.run`, and one inside a test would abort the
    pytest session (verified: exit 130). That escape is the CORRECT behaviour —
    a fatal signal must tear the run down — so it is not something this test can
    or should exercise.
    """

    async def fake_exec(*args, **kwargs):
        joined = " ".join(args)
        if "caspar.md" in joined:
            raise _CustomFatal("fatal during caspar")
        agent = next(m for m in ("melchior", "balthasar") if f"{m}.md" in joined)
        return _FakeProc(_envelope(agent))

    monkeypatch.setattr("trio_executor.asyncio.create_subprocess_exec", fake_exec)
    _write_prompts(tmp_path)

    out = asyncio.run(
        ClaudeTrio().run_trio(
            "P",
            str(tmp_path),
            str(tmp_path),
            900,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            2,
        )
    )

    assert set(out) == {"melchior", "balthasar", "caspar"}  # one entry per mage, always
    assert isinstance(out["caspar"], _CustomFatal)  # the EXACT type, not a wrapper
    assert not isinstance(out["caspar"], Exception)  # and it stayed a BaseException
    assert isinstance(out["melchior"], dict)


def test_run_trio_returns_validation_error_after_the_last_attempt(monkeypatch, tmp_path):
    """A mage that never produces a valid verdict yields its ValidationError."""

    async def fake_exec(*args, **kwargs):
        joined = " ".join(args)
        if "caspar.md" in joined:
            return _FakeProc(b'{"result": "no markers, ever"}')
        agent = next(m for m in ("melchior", "balthasar") if f"{m}.md" in joined)
        return _FakeProc(_envelope(agent))

    monkeypatch.setattr("trio_executor.asyncio.create_subprocess_exec", fake_exec)
    _write_prompts(tmp_path)

    states = []
    out = asyncio.run(
        ClaudeTrio(build_retry_prompt=lambda p, e: p).run_trio(
            "P",
            str(tmp_path),
            str(tmp_path),
            900,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            2,
            on_state=lambda a, st: states.append((a, st)),
        )
    )

    assert isinstance(out["caspar"], (ValidationError, json.JSONDecodeError))
    # And its state sequence ends in `failed`, never `success`: the mage exhausted
    # its attempts, so the terminal state has to say so.
    assert [st for a, st in states if a == "caspar"] == ["running", "retrying", "failed"]


def test_run_trio_rejects_a_max_attempts_below_one(tmp_path):
    """A caller error fails FAST, not as three dead mages."""
    _write_prompts(tmp_path)
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        asyncio.run(
            ClaudeTrio().run_trio(
                "P",
                str(tmp_path),
                str(tmp_path),
                900,
                {"melchior": "opus"},
                0,
            )
        )


def test_the_default_retry_builder_resolves_without_a_circular_import():
    """The lazy import must actually work against the real run_magi module."""
    from synthesize import ValidationError as VE

    trio = ClaudeTrio()
    out = trio._retry_prompt("ORIGINAL", VE("missing key: verdict"))

    assert out.startswith("ORIGINAL")
    assert len(out) > len("ORIGINAL")  # feedback was appended


def test_run_trio_reports_the_full_state_sequence(monkeypatch, tmp_path):
    """running -> retrying -> success for a mage that needs one retry."""
    attempts = {"caspar": 0}

    async def fake_exec(*args, **kwargs):
        joined = " ".join(args)
        agent = next(m for m in ("melchior", "balthasar", "caspar") if f"{m}.md" in joined)
        if agent == "caspar":
            attempts["caspar"] += 1
            if attempts["caspar"] == 1:
                return _FakeProc(b'{"result": "no markers"}')
        return _FakeProc(_envelope(agent))

    monkeypatch.setattr("trio_executor.asyncio.create_subprocess_exec", fake_exec)
    _write_prompts(tmp_path)
    states = []

    asyncio.run(
        ClaudeTrio(build_retry_prompt=lambda p, e: p).run_trio(
            "P",
            str(tmp_path),
            str(tmp_path),
            900,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            2,
            on_state=lambda a, st: states.append((a, st)),
        )
    )

    caspar_states = [st for a, st in states if a == "caspar"]
    assert caspar_states == ["running", "retrying", "success"]
    assert [st for a, st in states if a == "melchior"] == ["running", "success"]


def test_run_trio_writes_the_four_artifacts_per_mage(monkeypatch, tmp_path):
    """The docstring promises four files per mage; verify them instead of asserting."""

    async def fake_exec(*args, **kwargs):
        agent = next(m for m in ("melchior", "balthasar", "caspar") if f"{m}.md" in " ".join(args))
        # stderr NON-EMPTY on purpose: MEASURED, `write_stderr_log` returns early
        # on empty data (`if not data: return`), so the .stderr.log file would not
        # exist and this test would assert a file the code never promises.
        return _FakeProc(_envelope(agent), stderr=b"warm-up noise")

    monkeypatch.setattr("trio_executor.asyncio.create_subprocess_exec", fake_exec)
    _write_prompts(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    asyncio.run(
        ClaudeTrio().run_trio(
            "PAYLOAD",
            str(tmp_path),
            str(out_dir),
            900,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            2,
        )
    )

    for mage in ("melchior", "balthasar", "caspar"):
        for suffix in ("prompt.txt", "raw.json", "json", "stderr.log"):
            path = out_dir / f"{mage}.{suffix}"
            assert path.exists(), f"missing {path.name}"
    assert (out_dir / "melchior.prompt.txt").read_text(encoding="utf-8") == "PAYLOAD"


def test_artifact_writes_are_fail_soft(monkeypatch, tmp_path):
    """A full disk must not cost a verdict — measured, not asserted in a docstring."""

    async def fake_exec(*args, **kwargs):
        agent = next(m for m in ("melchior", "balthasar", "caspar") if f"{m}.md" in " ".join(args))
        return _FakeProc(_envelope(agent))

    def failing_write(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("trio_executor.asyncio.create_subprocess_exec", fake_exec)
    # Patch the WRITER, not builtins.open: a global open() patch also breaks
    # pytest's own machinery and proves more than the failure being modelled.
    monkeypatch.setattr("trio_executor._write_text", failing_write)
    monkeypatch.setattr("trio_executor._write_stderr_log", failing_write)
    _write_prompts(tmp_path)

    out = asyncio.run(
        ClaudeTrio().run_trio(
            "P",
            str(tmp_path),
            str(tmp_path),
            900,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            2,
        )
    )

    assert all(isinstance(v, dict) for v in out.values())  # the verdicts stand


def test_run_trio_runs_the_three_mages_concurrently(monkeypatch, tmp_path):
    """A sequential implementation passes every other test; only this one catches it."""
    # A BARRIER, not a sleep: the three coroutines must all arrive before any is
    # released, so the test proves overlap instead of merely making it likely. A
    # sequential implementation does NOT deadlock -- each mage times out in turn,
    # the gate eventually opens, and the run completes -- but its peak stays at 1
    # and the assertion below fails. Deterministic either way: no flake on a slow
    # CI runner, and no false pass on a fast one.
    arrived = asyncio.Semaphore(0)
    release = asyncio.Event()
    in_flight = {"peak": 0, "now": 0}

    async def fake_exec(*args, **kwargs):
        joined = " ".join(args)
        agent = next(m for m in ("melchior", "balthasar", "caspar") if f"{m}.md" in joined)
        in_flight["now"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
        arrived.release()
        try:
            await asyncio.wait_for(release.wait(), timeout=5)
        finally:
            # try/finally, NO un decremento después del await: si el wait_for
            # expira, la excepción propaga hasta `_run_with_retry`, que la captura
            # POR MAGE -- y sin el finally el contador nunca baja, con lo que una
            # implementación SECUENCIAL acumula hasta 3 y PASA este test.
            # Reproducido: sin finally, secuencial da peak=3; con finally, peak=1.
            in_flight["now"] -= 1
        return _FakeProc(_envelope(agent))

    async def open_the_gate():
        for _ in range(3):
            await arrived.acquire()  # all three are in flight -> overlap proven
        release.set()

    monkeypatch.setattr("trio_executor.asyncio.create_subprocess_exec", fake_exec)
    _write_prompts(tmp_path)

    async def drive():
        gate = asyncio.create_task(open_the_gate())
        result = await ClaudeTrio().run_trio(
            "P",
            str(tmp_path),
            str(tmp_path),
            900,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            2,
        )
        await gate
        return result

    asyncio.run(drive())

    # A sequential gather never reaches three arrivals and dies on the wait_for.
    # Concurrency is the ONE guarantee of this refactor that no existing test catches.
    assert in_flight["peak"] == 3, in_flight


def test_retry_feedback_does_not_accumulate_across_attempts(monkeypatch, tmp_path):
    """With max_attempts=3, the third prompt carries ONE feedback block, not two."""
    seen = []

    async def fake_exec(*args, **kwargs):
        joined = " ".join(args)
        if "caspar.md" in joined:
            return _FakeProc(b'{"result": "no markers, ever"}')
        agent = next(m for m in ("melchior", "balthasar") if f"{m}.md" in joined)
        return _FakeProc(_envelope(agent))

    def builder(prompt, error):
        seen.append(prompt)
        return prompt + "|FB"

    monkeypatch.setattr("trio_executor.asyncio.create_subprocess_exec", fake_exec)
    _write_prompts(tmp_path)

    asyncio.run(
        ClaudeTrio(build_retry_prompt=builder).run_trio(
            "ORIG",
            str(tmp_path),
            str(tmp_path),
            900,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            3,
        )
    )

    # Two retries for caspar, and BOTH built from the ORIGINAL prompt.
    assert seen == ["ORIG", "ORIG"], seen
    # If feedback accumulated, the second call would have received "ORIG|FB".


def test_run_trio_emits_the_failed_state_on_a_nonzero_exit(monkeypatch, tmp_path):
    """A mage killed by a non-zero exit reports 'failed', never 'success'."""

    async def fake_exec(*args, **kwargs):
        joined = " ".join(args)
        if "caspar.md" in joined:
            return _FakeProc(b"", stderr=b"boom", returncode=2)
        agent = next(m for m in ("melchior", "balthasar") if f"{m}.md" in joined)
        return _FakeProc(_envelope(agent))

    monkeypatch.setattr("trio_executor.asyncio.create_subprocess_exec", fake_exec)
    _write_prompts(tmp_path)
    states = []

    asyncio.run(
        ClaudeTrio().run_trio(
            "P",
            str(tmp_path),
            str(tmp_path),
            900,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            2,
            on_state=lambda a, st: states.append((a, st)),
        )
    )

    assert [st for a, st in states if a == "caspar"] == ["running", "failed"]


def test_run_trio_reports_the_timeout_state_not_the_success_state(monkeypatch, tmp_path):
    """A mage that times out must emit running -> timeout, never success."""

    class _FakeStderrStream:
        async def read(self, n=-1):
            return b""

    class _HangingProc:
        pid = 4242  # windows_kill_tree names the pid in its warning
        returncode = None

        def __init__(self):
            self.stderr = _FakeStderrStream()

        async def communicate(self, input=None):
            await asyncio.sleep(3600)

        def kill(self):
            pass

        async def wait(self):
            self.returncode = -9
            return -9

    async def fake_exec(*args, **kwargs):
        joined = " ".join(args)
        if "caspar.md" in joined:
            return _HangingProc()
        agent = next(m for m in ("melchior", "balthasar") if f"{m}.md" in joined)
        return _FakeProc(_envelope(agent))

    monkeypatch.setattr("trio_executor.asyncio.create_subprocess_exec", fake_exec)
    _write_prompts(tmp_path)
    states = []

    asyncio.run(
        ClaudeTrio().run_trio(
            "P",
            str(tmp_path),
            str(tmp_path),
            1,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            2,
            on_state=lambda a, st: states.append((a, st)),
        )
    )

    assert [st for a, st in states if a == "caspar"] == ["running", "timeout"]
    assert "success" not in [st for a, st in states if a == "caspar"]


def test_run_trio_reports_states_without_letting_display_break_a_verdict(monkeypatch, tmp_path):
    async def fake_exec(*args, **kwargs):
        agent = next(m for m in ("melchior", "balthasar", "caspar") if f"{m}.md" in " ".join(args))
        return _FakeProc(_envelope(agent))

    monkeypatch.setattr("trio_executor.asyncio.create_subprocess_exec", fake_exec)
    _write_prompts(tmp_path)

    states = []

    def boom(agent, state):
        states.append((agent, state))
        raise RuntimeError("display exploded")

    out = asyncio.run(
        ClaudeTrio().run_trio(
            "P",
            str(tmp_path),
            str(tmp_path),
            900,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            2,
            on_state=boom,
        )
    )

    assert set(out) == {"melchior", "balthasar", "caspar"}  # the verdicts stand
    assert ("melchior", "running") in states


def _envelope_of(payload):
    """Wrap an ARBITRARY payload in verdict markers, the way a mage would.

    `_envelope` builds a valid verdict for a named mage; this one takes the
    payload as given, which is what the guard tests need: their whole point is
    a payload that is well-formed yet wrong.
    """
    inner = json.dumps(payload)
    return json.dumps({"result": "<MAGI_VERDICT>\n" + inner + "\n</MAGI_VERDICT>"}).encode("utf-8")


def _first_attempt_returns(bad_payload, attempts):
    """A fake exec where caspar answers `bad_payload` once, then behaves."""

    async def fake_exec(*args, **kwargs):
        joined = " ".join(args)
        agent = next(m for m in ("melchior", "balthasar", "caspar") if f"{m}.md" in joined)
        if agent == "caspar":
            attempts["caspar"] += 1
            if attempts["caspar"] == 1:
                return _FakeProc(_envelope_of(bad_payload))
        return _FakeProc(_envelope(agent))

    return fake_exec


def test_a_verdict_that_echoes_the_prompt_example_is_rejected(monkeypatch, tmp_path):
    """R6, the LAST belt against a fabricated verdict -- and it must survive MS1.

    `launch_agent` applied this guard AFTER `load_agent_output`, and neither
    `ClaudeBackend` nor the plan mentioned it. Wiring `run_orchestrator` to
    `ClaudeTrio` without it would drop the anti-echo canary from the live Claude
    path in silence: a mage that copies the worked example out of its own system
    prompt and wraps it in markers would enter the consensus with a fabricated
    verdict -- in the adversarial seat.

    The failing input is that exact payload: valid on all seven keys, and
    fingerprinted by ECHO_CANARY field for field.
    """
    attempts = {"caspar": 0}
    echoed = dict(
        _VALID,
        agent="caspar",
        summary="One-line verdict",
        recommendation="What you recommend",
    )

    monkeypatch.setattr(
        "trio_executor.asyncio.create_subprocess_exec",
        _first_attempt_returns(echoed, attempts),
    )
    _write_prompts(tmp_path)
    out = asyncio.run(
        ClaudeTrio(build_retry_prompt=lambda p, e: p).run_trio(
            "P",
            str(tmp_path),
            str(tmp_path),
            900,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            2,
        )
    )

    # It is a ValidationError subclass, so the retry loop caught it and the mage
    # got a second chance -- exactly as on the v5 path.
    assert attempts["caspar"] == 2
    assert isinstance(out["caspar"], dict)
    assert out["caspar"]["summary"] != "One-line verdict"


def test_a_verdict_claiming_another_mage_is_rejected(monkeypatch, tmp_path):
    """R10 -- identity, the other guard `launch_agent` applied and the plan omitted.

    `load_agent_output` checks that `agent` is in the enum; nobody checked it was
    the mage that was LAUNCHED. One mage's text in another's seat is counted by
    the consensus as an independent perspective THAT NEVER EXISTED.
    """
    attempts = {"caspar": 0}
    impostor = dict(_VALID, agent="melchior")  # caspar's slot, melchior's name

    monkeypatch.setattr(
        "trio_executor.asyncio.create_subprocess_exec",
        _first_attempt_returns(impostor, attempts),
    )
    _write_prompts(tmp_path)
    out = asyncio.run(
        ClaudeTrio(build_retry_prompt=lambda p, e: p).run_trio(
            "P",
            str(tmp_path),
            str(tmp_path),
            900,
            {"melchior": "opus", "balthasar": "opus", "caspar": "opus"},
            2,
        )
    )

    assert attempts["caspar"] == 2
    assert out["caspar"]["agent"] == "caspar"


def test_the_identity_guard_is_case_insensitive():
    """The guard's own contract, tested DIRECTLY -- and here is why.

    An earlier version of this test drove a "Caspar" verdict through the whole
    transport and asserted no retry was spent. It failed, and the reason is
    worth keeping: `load_agent_output` rejects the payload FIRST, because the
    agent enum is case-SENSITIVE ("Unknown agent 'Caspar'"). So the identity
    guard never sees a differently-cased name by that route.

    That makes `casefold()` defence in depth for a case the validator already
    catches -- not dead code, but not reachable through the transport either.
    Testing it through the flow would have pinned the VALIDATOR's behaviour
    while claiming to pin the guard's.
    """
    from trio_executor import _require_claimed_identity

    _require_claimed_identity(dict(_VALID, agent="Caspar"), "caspar")  # must not raise
    _require_claimed_identity(dict(_VALID, agent="  caspar  "), "caspar")  # stripped


def test_the_identity_guard_rejects_a_different_mage():
    """The failing input, at the unit level: a name that is valid but not ours."""
    from verdict_markers import AgentIdentityError
    from trio_executor import _require_claimed_identity

    with pytest.raises(AgentIdentityError) as exc:
        _require_claimed_identity(dict(_VALID, agent="melchior"), "caspar")
    assert "melchior" in str(exc.value) and "caspar" in str(exc.value)


def test_the_claude_path_goes_through_ClaudeTrio(monkeypatch, tmp_path):
    """The wiring, end to end: `rotation is None` must reach `ClaudeTrio.run_trio`.

    Without this, MS1 could ship a `ClaudeTrio` that 25 unit tests exercise and
    that production never calls -- green everywhere and dead in the one place
    that matters.

    The signature is the MEASURED one (`run_orchestrator` at run_magi.py:1598),
    read before writing this rather than assumed.
    """
    calls = {}

    class _StubTrio:
        async def run_trio(
            self, prompt, agents_dir, output_dir, timeout, models, max_attempts, on_state=None
        ):
            calls["models"] = dict(models)
            calls["on_state"] = on_state
            # A retry, so the bridge into `retried` is exercised too.
            if on_state is not None:
                on_state("caspar", "retrying")
            return {
                name: {
                    "agent": name,
                    "verdict": "approve",
                    "confidence": 0.9,
                    "summary": "s",
                    "reasoning": "r",
                    "findings": [],
                    "recommendation": "rec",
                }
                for name in run_magi.AGENTS
            }

    monkeypatch.setattr(run_magi, "ClaudeTrio", lambda *a, **k: _StubTrio())

    # The REAL agent prompts, not stubs: `run_orchestrator` runs `prompt_guard`
    # before the trio, and that guard checks the markers and the worked-example
    # text. A placeholder would exercise the guard's REJECTION path while
    # claiming to test the wiring.
    agents_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "skills",
        "magi",
        "agents",
    )

    report = asyncio.run(
        run_magi.run_orchestrator(
            agents_dir,
            "PAYLOAD",
            str(tmp_path),
            900,
            show_status=False,
        )
    )

    # 1. the executor was actually used, with one model per mage
    assert set(calls["models"]) == set(run_magi.AGENTS)
    # 2. the state bridge was handed over -- not None, or `retried_agents` dies
    assert calls["on_state"] is not None
    # 3. and the bridge fed the telemetry the report reads (run_magi.py:1683-1684)
    assert "caspar" in report["retried_agents"]
    # 4. the pipeline below the verdicts did not fork: a real report came out
    assert report["consensus"]
