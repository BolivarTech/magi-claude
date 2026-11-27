# Author: Julian Bolivar
# Version: 8.0.0
# Date: 2026-11-21
"""Tests for MagiRsCodeReview, the executor of one ``magi-rs code-review`` (MS2 Task 3).

Every child process is faked: nothing here launches a binary or reaches the network.
"""

import asyncio
import sys

import pytest

from _magi_rs_fakes import _StreamingProcess, _cr_payload, _stub_bootstrap, _stub_pin
from magi_key import KeyExposedError, MagiKey
from magi_rs_code_review import (
    HISTORY_WARNING_MARKER,
    HISTORY_WINDOW_SECONDS,
    TIMEOUT_REJECTION_MARKER,
    MagiRsCodeReview,
    build_review_argv,
)
from magi_rs_bootstrap import DigestMismatchError
from magi_rs_targets import ReviewTarget
from magi_rs_trio import MagiRsProcessError

_KEY = MagiKey("review-key-EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE=")
_TARGET = ReviewTarget("branch", ("--base", "main", "--head", "HEAD"), "branch main...HEAD")


def test_the_argv_maps_the_target_and_never_passes_no_memory_or_the_key():
    argv = build_review_argv("/bin/magi-rs", "/repo", _TARGET, 16820)
    assert argv == [
        "/bin/magi-rs",
        "code-review",
        "--base",
        "main",
        "--head",
        "HEAD",
        "--output-format",
        "json",
        "--structured-verdicts",
        "-w",
        "/repo",
        "--timeout",
        "16820",
    ]
    assert "--no-memory" not in argv and _KEY.reveal() not in " ".join(argv)


def _review(monkeypatch, tmp_path, process, timeout=1):
    monkeypatch.setattr(
        "magi_rs_code_review.ProjectPreflight.require_ready", staticmethod(lambda *a, **k: _KEY)
    )
    _stub_bootstrap(monkeypatch, module="magi_rs_code_review")
    seen = {}

    async def _exec(*argv, **kwargs):
        seen["argv"], seen["env"], seen["stdin"] = argv, kwargs["env"], kwargs.get("stdin")
        seen["calls"] = seen.get("calls", 0) + 1
        return process

    monkeypatch.setattr("asyncio.create_subprocess_exec", _exec)
    executor = MagiRsCodeReview(str(tmp_path), _TARGET, timeout, str(tmp_path / "bin"), _stub_pin())
    executor.preflight()  # stores the key, as main() does
    return executor, seen


def test_the_review_preflight_passes_an_embedder_probe(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        "magi_rs_code_review.ProjectPreflight.require_ready",
        staticmethod(lambda workdir, ks, probe=None: seen.append(probe) or _KEY),
    )
    _stub_bootstrap(monkeypatch, module="magi_rs_code_review")
    MagiRsCodeReview(str(tmp_path), _TARGET, 1, str(tmp_path / "bin"), _stub_pin()).preflight()
    assert len(seen) == 1 and seen[0] is not None


@pytest.mark.asyncio
async def test_a_key_loosened_after_the_preflight_is_refused_before_the_launch(
    monkeypatch, tmp_path
):
    # Loop 1 m3 (R4): run_trio re-verifies the key and the binary, without the embedder probe.
    calls = []

    def _require_ready(workdir, key_store, probe=None):
        calls.append(probe)
        if len(calls) > 1:
            raise KeyExposedError("key loosened")
        return _KEY

    monkeypatch.setattr(
        "magi_rs_code_review.ProjectPreflight.require_ready", staticmethod(_require_ready)
    )
    _stub_bootstrap(monkeypatch, module="magi_rs_code_review")
    launched = []

    async def _exec(*argv, **kwargs):
        launched.append(argv)
        return _StreamingProcess(_cr_payload())

    monkeypatch.setattr("asyncio.create_subprocess_exec", _exec)
    executor = MagiRsCodeReview(str(tmp_path), _TARGET, 1, str(tmp_path / "bin"), _stub_pin())
    executor.preflight()
    with pytest.raises(KeyExposedError):
        await executor.run_trio("", "agents", str(tmp_path), 1, {}, 1)
    assert launched == [] and calls[0] is not None and calls[1] is None


@pytest.mark.asyncio
async def test_a_binary_that_stops_verifying_is_refused_before_the_launch(monkeypatch, tmp_path):
    executor, seen = _review(monkeypatch, tmp_path, _StreamingProcess(_cr_payload()))

    def _fails(bootstrap):
        raise DigestMismatchError("digest differs")

    monkeypatch.setattr("magi_rs_code_review.verified_executable", _fails)
    with pytest.raises(DigestMismatchError):
        await executor.run_trio("", "agents", str(tmp_path), 1, {}, 1)
    assert "calls" not in seen


@pytest.mark.asyncio
async def test_the_key_travels_by_environment_and_stdin_is_closed(monkeypatch, tmp_path):
    executor, seen = _review(monkeypatch, tmp_path, _StreamingProcess(_cr_payload()))
    await executor.run_trio("", "agents", str(tmp_path), 1, {}, 1)
    assert seen["env"]["MAGI_PASSPHRASE"] == _KEY.reveal()
    assert seen["stdin"] == asyncio.subprocess.DEVNULL


@pytest.mark.asyncio
async def test_a_child_that_records_history_within_the_window_is_not_killed(monkeypatch, tmp_path):
    process = _StreamingProcess(_cr_payload(), exits_after=0.2)
    executor, _ = _review(monkeypatch, tmp_path, process, timeout=0)  # outer = 0 + window
    await executor.run_trio("", "agents", str(tmp_path), 0, {}, 1)
    assert process.killed is False and executor.history_recorded is True


@pytest.mark.asyncio
async def test_a_child_that_overruns_the_window_is_killed_but_its_report_stays_valid(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("magi_rs_code_review.HISTORY_WINDOW_SECONDS", 0.1)
    process = _StreamingProcess(_cr_payload(), exits_after=None, hold_eof=True)  # a wedged child
    executor, _ = _review(monkeypatch, tmp_path, process, timeout=0)
    verdicts = await asyncio.wait_for(executor.run_trio("", "agents", str(tmp_path), 0, {}, 1), 5)
    assert process.killed is True and executor.history_recorded is False
    assert len(verdicts) == 3


@pytest.mark.asyncio
async def test_a_killed_child_with_an_incomplete_envelope_is_a_process_error(monkeypatch, tmp_path):
    monkeypatch.setattr("magi_rs_code_review.HISTORY_WINDOW_SECONDS", 0.1)
    process = _StreamingProcess(_cr_payload()[:50], exits_after=None, hold_eof=True)
    executor, _ = _review(monkeypatch, tmp_path, process, timeout=0)
    with pytest.raises(MagiRsProcessError):
        await asyncio.wait_for(executor.run_trio("", "agents", str(tmp_path), 0, {}, 1), 5)
    assert process.killed is True


@pytest.mark.asyncio
async def test_the_outer_deadline_is_the_timeout_plus_the_window(monkeypatch, tmp_path):
    # A child exiting at 1.2 s survives only under timeout + window (1 + 0.5);
    # a timeout-only (1.0) or window-only (0.5) deadline kills it.
    monkeypatch.setattr("magi_rs_code_review.HISTORY_WINDOW_SECONDS", 0.5)
    process = _StreamingProcess(_cr_payload(), exits_after=1.2)
    executor, _ = _review(monkeypatch, tmp_path, process, timeout=1)
    await executor.run_trio("", "agents", str(tmp_path), 1, {}, 1)
    assert process.killed is False


@pytest.mark.asyncio
async def test_an_insufficient_timeout_is_one_error_naming_timeout(monkeypatch, tmp_path):
    process = _StreamingProcess(b"", TIMEOUT_REJECTION_MARKER.encode(), returncode=1)
    executor, seen = _review(monkeypatch, tmp_path, process)
    with pytest.raises(MagiRsProcessError) as exc:
        await executor.run_trio("", "agents", str(tmp_path), 1, {}, 1)
    assert "--timeout" in str(exc.value)
    assert seen["calls"] == 1  # REQ-14 "without retrying"


@pytest.mark.asyncio
async def test_no_failure_message_contains_the_key(monkeypatch, tmp_path):
    process = _StreamingProcess(b"", _KEY.reveal().encode(), returncode=1)
    executor, _ = _review(monkeypatch, tmp_path, process)
    with pytest.raises(MagiRsProcessError) as exc:
        await executor.run_trio("", "agents", str(tmp_path), 1, {}, 1)
    assert _KEY.reveal() not in str(exc.value)


def _review_error_line(code):
    # The shape magi-rs 0.26.0 prints (src/main.rs emit_review_error): the text, then the field.
    return (
        "2026-11-24T12:00:00Z ERROR run=1-a magi_rs::code_review: error: batch 1 with its "
        "containing units does not fit in one batch (69830 tokens, limit 65445) "
        f"review.error={code}"
    ).encode()


async def _failure(monkeypatch, tmp_path, stderr):
    monkeypatch.setenv("CORP_PROXY_CA_HINT", "x")
    executor, _ = _review(monkeypatch, tmp_path, _StreamingProcess(b"", stderr, returncode=2))
    with pytest.raises(MagiRsProcessError) as exc:
        await executor.run_trio("", "agents", str(tmp_path), 1, {}, 1)
    return str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code", ["too_large_for_batch", "nothing_to_review", "not_reviewable", "budget"]
)
async def test_a_review_error_about_the_input_carries_neither_hint(monkeypatch, tmp_path, code):
    message = await _failure(monkeypatch, tmp_path, _review_error_line(code))
    assert f"review.error={code}" in message  # the cause itself is still stated
    assert "--ollama-vault-set" not in message
    assert "allowlist" not in message and "CORP_PROXY_CA_HINT" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["git", "io"])
async def test_a_review_error_from_the_environment_keeps_only_the_allowlist_hint(
    monkeypatch, tmp_path, code
):
    message = await _failure(monkeypatch, tmp_path, _review_error_line(code))
    assert "allowlist" in message and "CORP_PROXY_CA_HINT" in message
    assert "--ollama-vault-set" not in message


@pytest.mark.asyncio
async def test_a_failure_without_a_review_error_code_keeps_both_hints(monkeypatch, tmp_path):
    message = await _failure(monkeypatch, tmp_path, b"error: provider answered 401")
    assert "--ollama-vault-set" in message
    assert "allowlist" in message and "CORP_PROXY_CA_HINT" in message


@pytest.mark.asyncio
async def test_a_failing_exit_still_persists_the_redacted_stdout(monkeypatch, tmp_path):
    # Loop 1 m5: a non-zero exit may follow paid per-batch output; keep it, key redacted.
    process = _StreamingProcess(b"partial " + _KEY.reveal().encode(), b"boom", returncode=1)
    executor, _ = _review(monkeypatch, tmp_path, process)
    with pytest.raises(MagiRsProcessError):
        await executor.run_trio("", "agents", str(tmp_path), 1, {}, 1)
    raw = (tmp_path / "magi-rs-raw.json").read_text(encoding="utf-8")
    assert raw.startswith("partial ") and _KEY.reveal() not in raw


@pytest.mark.asyncio
async def test_magi_rs_own_history_warning_means_history_was_not_recorded(monkeypatch, tmp_path):
    process = _StreamingProcess(
        _cr_payload(), stderr=HISTORY_WARNING_MARKER.encode(), exits_after=0.0
    )
    executor, _ = _review(monkeypatch, tmp_path, process)
    await executor.run_trio("", "agents", str(tmp_path), 1, {}, 1)
    assert process.killed is False and executor.history_recorded is False


def test_the_history_window_covers_magi_rs_thirty_seconds():
    assert HISTORY_WINDOW_SECONDS >= 30


@pytest.mark.asyncio
async def test_the_key_in_stdout_is_redacted_in_the_raw_artifact_and_the_result(
    monkeypatch, tmp_path
):
    process = _StreamingProcess(_cr_payload(note=_KEY.reveal()))
    executor, _ = _review(monkeypatch, tmp_path, process)
    await executor.run_trio("", "agents", str(tmp_path), 1, {}, 1)
    raw = (tmp_path / "magi-rs-raw.json").read_text(encoding="utf-8")
    assert _KEY.reveal() not in raw
    assert _KEY.reveal() not in str(executor.last_result.code_review)


class _UnreapableProcess(_StreamingProcess):
    """A wedged child that survives kill(): wait() never returns."""

    async def wait(self):
        await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_a_child_that_survives_the_kill_still_yields_its_complete_envelope(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("magi_rs_code_review.HISTORY_WINDOW_SECONDS", 0.1)
    monkeypatch.setattr("subprocess_utils.PROC_WAIT_REAP_TIMEOUT", 0.1)
    process = _UnreapableProcess(_cr_payload(), exits_after=None)
    executor, _ = _review(monkeypatch, tmp_path, process, timeout=0)
    verdicts = await asyncio.wait_for(executor.run_trio("", "agents", str(tmp_path), 0, {}, 1), 5)
    assert executor.history_recorded is False and len(verdicts) == 3


@pytest.mark.skipif(sys.platform != "win32", reason="the tree kill is Windows-only")
@pytest.mark.asyncio
async def test_the_overrun_kill_takes_down_the_process_tree(monkeypatch, tmp_path):
    killed_trees = []
    monkeypatch.setattr("subprocess_utils.windows_kill_tree", killed_trees.append)
    monkeypatch.setattr("magi_rs_code_review.HISTORY_WINDOW_SECONDS", 0.1)
    process = _StreamingProcess(_cr_payload(), exits_after=None, hold_eof=True)
    executor, _ = _review(monkeypatch, tmp_path, process, timeout=0)
    await asyncio.wait_for(executor.run_trio("", "agents", str(tmp_path), 0, {}, 1), 5)
    assert killed_trees == [process.pid]


class _LateStdoutProcess(_StreamingProcess):
    """A wedged child whose complete envelope reaches the pipe only once it is killed.

    Models an envelope written to the pipe but not yet read when the deadline fires: the
    reader sees nothing until the kill, after which the bytes are there and EOF follows.
    """

    def _streams(self):
        if self._readers is None:
            out, err = asyncio.StreamReader(), asyncio.StreamReader()
            err.feed_eof()
            self._readers = (out, err)
        return self._readers

    def kill(self):
        out = self._streams()[0]
        out.feed_data(self._out)
        out.feed_eof()
        super().kill()


@pytest.mark.asyncio
async def test_a_complete_envelope_still_unread_at_the_kill_is_kept(monkeypatch, tmp_path):
    # MAGI run 2, G4: the overrun kill must drain stdout before judging it incomplete.
    monkeypatch.setattr("magi_rs_code_review.HISTORY_WINDOW_SECONDS", 0.1)
    process = _LateStdoutProcess(_cr_payload(), exits_after=None)
    executor, _ = _review(monkeypatch, tmp_path, process, timeout=0)
    verdicts = await asyncio.wait_for(executor.run_trio("", "agents", str(tmp_path), 0, {}, 1), 5)
    assert process.killed is True and executor.history_recorded is False
    assert len(verdicts) == 3


def test_the_history_warning_marker_is_pinned_to_magi_rs_wording():
    """The prefix magi-rs 0.25.0 prints when the review history is not recorded.

    Source: magi-rs 0.25.0 ``src/tools/code_review.rs:1500``
    (``HISTORY_NOT_UPDATED``), followed by ``: <cause>``. A pin that rewords it must fail
    here before a run silently reports its history as recorded.
    """
    assert HISTORY_WARNING_MARKER == "warning: the review history was not updated"


def test_the_timeout_rejection_marker_is_pinned_to_magi_rs_wording():
    """Part of the budget rejection measured against magi-rs 0.25.0 (``pass --timeout``)."""
    assert TIMEOUT_REJECTION_MARKER == "pass --timeout"


class _TwoChunkStdout:
    """A stdout that is silent until released, then yields two chunks and never closes."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks, self.released = list(chunks), asyncio.Event()

    async def read(self, n=-1):
        await self.released.wait()
        if self._chunks:
            return self._chunks.pop(0)
        await asyncio.sleep(3600)  # the pipe never closes


class _TwoChunkProcess(_StreamingProcess):
    """After the kill the envelope arrives in two chunks and EOF never follows."""

    def __init__(self, envelope: bytes):
        super().__init__(envelope, exits_after=None)
        half = len(envelope) // 2
        self._two = _TwoChunkStdout([envelope[:half], envelope[half:]])

    @property
    def stdout(self):
        return self._two

    def kill(self):
        self._two.released.set()
        super().kill()


@pytest.mark.asyncio
async def test_chunks_read_before_the_drain_deadline_are_kept(monkeypatch, tmp_path):
    # G4 re-review: a single read() cancelled by the deadline loses what it already consumed.
    monkeypatch.setattr("magi_rs_code_review.HISTORY_WINDOW_SECONDS", 0.1)
    monkeypatch.setattr("magi_rs_code_review.PROC_STDERR_DRAIN_TIMEOUT", 0.3)
    process = _TwoChunkProcess(_cr_payload())
    executor, _ = _review(monkeypatch, tmp_path, process, timeout=0)
    verdicts = await asyncio.wait_for(executor.run_trio("", "agents", str(tmp_path), 0, {}, 1), 5)
    assert len(verdicts) == 3 and executor.history_recorded is False
