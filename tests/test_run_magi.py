# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-01
"""Tests for run_magi.py — async Python orchestrator."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.error
from email.message import Message
from typing import Any
from unittest.mock import patch

import pytest

from fallback_policy import AgentRotationState
from ollama_config import ModelSpec
from rotation_harness import FALLBACK, REQUIRED, TRIO, _rotation, _run, _valid
from validate import ValidationError


async def _preflight_ok(config, prompt=""):
    """Async PreflightResult stand-in for tests that patch out the network guard.

    T9: ``select_backend`` now reads ``result.fallback`` / ``.min_window_tokens`` /
    ``.capabilities``, so a bare ``None`` no longer satisfies it. Return a minimal
    real ``PreflightResult`` (the trio's lineages are distinct, windows big enough).
    """
    from rotation_harness import _preflight_result

    return _preflight_result()


class TestParseArgs:
    """Verify CLI argument parsing."""

    def test_minimal_args(self):
        from run_magi import parse_args

        args = parse_args(["code-review", "input.py"])
        assert args.mode == "code-review"
        assert args.input == "input.py"
        assert args.timeout == 900
        assert args.output_dir is None

    def test_custom_timeout(self):
        from run_magi import parse_args

        args = parse_args(["analysis", "file.txt", "--timeout", "60"])
        assert args.timeout == 60

    def test_custom_output_dir(self):
        from run_magi import parse_args

        args = parse_args(["design", "spec.md", "--output-dir", "/tmp/out"])
        assert args.output_dir == "/tmp/out"

    def test_invalid_mode_rejected(self):
        from run_magi import parse_args

        with pytest.raises(SystemExit):
            parse_args(["invalid-mode", "input.py"])

    def test_all_valid_modes(self):
        from run_magi import parse_args

        for mode in ("code-review", "design", "analysis"):
            args = parse_args([mode, "input.py"])
            assert args.mode == mode

    def test_default_model_for_code_review_is_opus(self):
        """code-review keeps opus as the default — dense technical reasoning
        warrants the cost. Backward-compatible with the 2.0.x-2.2.x default.
        """
        from run_magi import parse_args

        args = parse_args(["code-review", "input.py"])
        assert args.model == "opus"

    def test_default_model_for_design_is_opus(self):
        """design defaults to opus — multi-level abstraction (architecture,
        scaling, hidden coupling) where smaller models drop confidence sharply.
        """
        from run_magi import parse_args

        args = parse_args(["design", "spec.md"])
        assert args.model == "opus"

    def test_default_model_for_analysis_is_opus(self):
        """analysis defaults to opus.

        2.2.3 (released 2026-04-25) switched analysis to sonnet for cost
        relief. 2.2.5 (this release, 2026-04-26) reverts based on
        production evidence: Caspar (the most-output agent by design,
        consistently producing 4-7K output tokens vs Mel/Bal at 2-3K)
        failed in ≥33% of sbtdd Loop verifications under the sonnet
        default. That is an order of magnitude above the 3.3% design
        assumption documented in CLAUDE.md "Post-release hardening".

        The 2.2.4 retry could not recover Caspar consistently because
        the failure was structural (output-ceiling pressure on sonnet's
        ~8K max), not stochastic. The second attempt with the same
        model hit the same ceiling. Reverting analysis to opus restores
        the 32K max-output budget and gives Caspar headroom.

        The 2.2.4 retry path remains active for all three modes; only
        the per-mode default for analysis flips back to opus.
        ``code-review`` and ``design`` were never on sonnet.
        """
        from run_magi import parse_args

        args = parse_args(["analysis", "input.txt"])
        assert args.model == "opus"

    def test_explicit_model_overrides_mode_default(self):
        """``--model X`` always wins over any per-mode default. Without this,
        operators who want to force opus for analysis (or haiku for code-review)
        would have no way to do it.
        """
        from run_magi import parse_args

        # sonnet for analysis (override the opus default re-established in 2.2.5)
        args = parse_args(["analysis", "input.txt", "--model", "sonnet"])
        assert args.model == "sonnet"

        # haiku for code-review (override the opus default)
        args = parse_args(["code-review", "input.py", "--model", "haiku"])
        assert args.model == "haiku"

        # sonnet for design (override the opus default)
        args = parse_args(["design", "spec.md", "--model", "sonnet"])
        assert args.model == "sonnet"

    def test_custom_model(self):
        from run_magi import parse_args

        for model in ("opus", "sonnet", "haiku"):
            args = parse_args(["code-review", "input.py", "--model", model])
            assert args.model == model

    def test_invalid_model_rejected(self):
        from run_magi import parse_args

        with pytest.raises(SystemExit):
            parse_args(["code-review", "input.py", "--model", "gpt4"])

    def test_default_show_status_true(self):
        from run_magi import parse_args

        args = parse_args(["code-review", "input.py"])
        assert args.show_status is True

    def test_no_status_flag_sets_false(self):
        from run_magi import parse_args

        args = parse_args(["code-review", "input.py", "--no-status"])
        assert args.show_status is False

    def test_keep_runs_default(self):
        """Default --keep-runs value lines up with MAX_HISTORY_RUNS."""
        from run_magi import MAX_HISTORY_RUNS, parse_args

        args = parse_args(["code-review", "input.py"])
        assert args.keep_runs == MAX_HISTORY_RUNS

    def test_keep_runs_zero_rejected(self):
        """``--keep-runs 0`` is ambiguous and must be rejected at argparse.

        Regression for the v2.1.1 fix: previously, ``--keep-runs 0`` was
        silently interpreted as ``cleanup_old_runs(-1)`` ("disable
        cleanup"), producing unbounded accumulation — the opposite of
        what a user passing 0 would reasonably expect. The CLI now
        rejects 0 with an error that points to ``--keep-runs 1``
        (wipe-all) or ``--keep-runs -1`` (disable) as the disambiguating
        replacements.
        """
        from run_magi import parse_args

        with pytest.raises(SystemExit):
            parse_args(["code-review", "input.py", "--keep-runs", "0"])

    def test_keep_runs_negative_accepted(self):
        """``--keep-runs -1`` is the explicit "disable cleanup" value."""
        from run_magi import parse_args

        args = parse_args(["code-review", "input.py", "--keep-runs", "-1"])
        assert args.keep_runs == -1

    def test_keep_runs_one_accepted(self):
        """``--keep-runs 1`` is the explicit "wipe all prior" value."""
        from run_magi import parse_args

        args = parse_args(["code-review", "input.py", "--keep-runs", "1"])
        assert args.keep_runs == 1

    def test_warn_input_tokens_zero_or_negative_rejected(self):
        """``--warn-input-tokens`` must be a positive integer.

        A value <= 0 makes ``check_input_size`` flag every input as
        oversize (``chars > 0`` is always True), producing spurious
        warnings on trivial inputs. The CLI rejects non-positive values
        at argparse, mirroring the ``--keep-runs 0`` guard.
        """
        import run_magi

        for bad in ("0", "-1"):
            with pytest.raises(SystemExit):
                run_magi.parse_args(["code-review", "x", "--warn-input-tokens", bad])


class TestModeModelLockstepInvariant:
    """Pin the lockstep invariant claimed by the 2.2.3 docstrings.

    `MODE_DEFAULT_MODELS` (in models.py) and the inline comment in
    `run_magi.parse_args` both promise the test suite enforces:

      * Every key of MODE_DEFAULT_MODELS is a valid analysis mode.
      * Every value of MODE_DEFAULT_MODELS is a registered model.

    Without these tests, a future contributor adding a fourth mode to
    VALID_MODES (or removing one from MODE_DEFAULT_MODELS) would slip
    past CI and surface as a runtime KeyError on the
    `MODE_DEFAULT_MODELS[args.mode]` lookup. These tests convert the
    docstring promise into a regression-blocking guarantee.
    """

    def test_every_mode_has_a_default_model(self):
        from models import MODE_DEFAULT_MODELS
        from run_magi import VALID_MODES

        assert set(MODE_DEFAULT_MODELS.keys()) == set(VALID_MODES), (
            f"MODE_DEFAULT_MODELS keys {sorted(MODE_DEFAULT_MODELS.keys())} "
            f"must equal VALID_MODES {sorted(VALID_MODES)} — adding a mode "
            f"requires adding its default; removing a mode requires removing "
            f"its default. The post-parse resolution at run_magi.parse_args "
            f"depends on this set equality holding."
        )

    def test_every_mode_default_is_a_registered_model(self):
        from models import MODE_DEFAULT_MODELS, MODEL_IDS

        unknown = set(MODE_DEFAULT_MODELS.values()) - set(MODEL_IDS.keys())
        assert not unknown, (
            f"MODE_DEFAULT_MODELS contains short names not in MODEL_IDS: "
            f"{sorted(unknown)}. Every default must resolve through "
            f"resolve_model() at orchestrator startup, so the set of "
            f"values must be a subset of MODEL_IDS keys."
        )


class TestCreateOutputDir:
    """Verify cross-platform temp directory creation."""

    def test_uses_tempfile_mkdtemp(self):
        from run_magi import create_output_dir

        output_dir = create_output_dir(None)
        assert os.path.isdir(output_dir)
        assert "magi-run-" in os.path.basename(output_dir)
        os.rmdir(output_dir)

    def test_respects_explicit_output_dir(self, tmp_path):
        from run_magi import create_output_dir

        output_dir = create_output_dir(str(tmp_path / "custom"))
        assert output_dir == str(tmp_path / "custom")
        assert os.path.isdir(output_dir)

    def test_create_output_dir_uses_run_root(self, tmp_path):
        from temp_dirs import MAGI_DIR_PREFIX, create_output_dir

        out = create_output_dir(None, str(tmp_path))
        assert os.path.dirname(out) == str(tmp_path)
        assert os.path.basename(out).startswith(MAGI_DIR_PREFIX)


class TestRunOrchestrator:
    """Verify full orchestration with mocked agents."""

    @pytest.mark.asyncio
    async def test_all_three_agents_success(self, tmp_path):
        from run_magi import run_orchestrator

        agent_results = {}
        for name in ("melchior", "balthasar", "caspar"):
            agent_results[name] = {
                "agent": name,
                "verdict": "approve",
                "confidence": 0.9,
                "summary": f"{name} OK",
                "reasoning": "Fine",
                "findings": [],
                "recommendation": "Merge",
            }

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            return agent_results[agent_name]

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
            assert result["consensus"]["consensus"] == "STRONG GO"
            assert result.get("degraded") is not True
            assert len(result["agents"]) == 3

    @pytest.mark.asyncio
    async def test_agents_dispatched_and_reported_caspar_first(self, tmp_path):
        """Caspar leads the dispatch and the report (order Caspar→Melchior→Balthasar).

        The AGENTS order is deliberately Caspar-first so the adversarial critic
        leads the live status display and the report, mirroring the fallback's
        anti-anchoring 'Caspar first' ordering. Parallel execution is unchanged
        (all three still run concurrently via asyncio.gather); only the kickoff
        and stable output order change.
        """
        from run_magi import run_orchestrator

        launch_order: list[str] = []

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            launch_order.append(agent_name)
            return {
                "agent": agent_name,
                "verdict": "approve",
                "confidence": 0.9,
                "summary": f"{agent_name} OK",
                "reasoning": "Fine",
                "findings": [],
                "recommendation": "Merge",
            }

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )

        expected = ["caspar", "melchior", "balthasar"]
        assert launch_order == expected, f"dispatch order {launch_order} != {expected}"
        assert [a["agent"] for a in result["agents"]] == expected

    @pytest.mark.asyncio
    async def test_one_agent_fails_degraded_mode(self, tmp_path):
        from run_magi import run_orchestrator

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            if agent_name == "caspar":
                raise TimeoutError(f"Agent {agent_name} timed out")
            return {
                "agent": agent_name,
                "verdict": "approve",
                "confidence": 0.85,
                "summary": "OK",
                "reasoning": "Fine",
                "findings": [],
                "recommendation": "Merge",
            }

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
            assert result["degraded"] is True
            assert "caspar" in result["failed_agents"]
            assert len(result["agents"]) == 2

    @pytest.mark.asyncio
    async def test_all_agents_fail_raises(self, tmp_path):
        from run_magi import run_orchestrator

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            raise TimeoutError(f"Agent {agent_name} timed out")

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            with pytest.raises(RuntimeError, match="fewer than 2"):
                await run_orchestrator(
                    agents_dir=str(tmp_path),
                    prompt="test",
                    output_dir=str(tmp_path),
                    timeout=300,
                )

    @pytest.mark.asyncio
    async def test_model_passed_to_launch_agent(self, tmp_path):
        """Verify that the model propagates to launch_agent as a ModelSpec (T9)."""
        from run_magi import run_orchestrator

        captured: list[Any] = []

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            captured.append(spec)
            return {
                "agent": agent_name,
                "verdict": "approve",
                "confidence": 0.9,
                "summary": "OK",
                "reasoning": "Fine",
                "findings": [],
                "recommendation": "Merge",
            }

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
                model="sonnet",
            )
            assert all(isinstance(s, ModelSpec) and s.model == "sonnet" for s in captured)
            assert len(captured) == 3

    @pytest.mark.asyncio
    async def test_two_fail_one_succeeds_raises(self, tmp_path):
        from run_magi import run_orchestrator

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            if agent_name != "melchior":
                raise TimeoutError(f"Agent {agent_name} timed out")
            return {
                "agent": "melchior",
                "verdict": "approve",
                "confidence": 0.9,
                "summary": "OK",
                "reasoning": "Fine",
                "findings": [],
                "recommendation": "Merge",
            }

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            with pytest.raises(RuntimeError, match="fewer than 2"):
                await run_orchestrator(
                    agents_dir=str(tmp_path),
                    prompt="test",
                    output_dir=str(tmp_path),
                    timeout=300,
                )


class TestCleanupOldRuns:
    """Verify LRU cleanup of old MAGI temp directories."""

    def test_negative_keep_disables_cleanup(self, tmp_path):
        """keep < 0 should not scan or delete anything."""
        from run_magi import cleanup_old_runs

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            magi_dir = tmp_path / "magi-run-abc123"
            magi_dir.mkdir()
            cleanup_old_runs(-1)
            assert magi_dir.exists()

    def test_keep_zero_deletes_all_magi_dirs(self, tmp_path):
        """keep == 0 should remove every magi-run-* dir (reserves slot for new run)."""
        from run_magi import cleanup_old_runs

        magi_dirs = []
        for i in range(3):
            d = tmp_path / f"magi-run-{i:04d}"
            d.mkdir()
            magi_dirs.append(d)

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            cleanup_old_runs(0)

        for d in magi_dirs:
            assert not d.exists(), f"{d} should have been deleted"

    def test_keeps_most_recent(self, tmp_path):
        """Should keep the N most recent and remove the rest."""
        from run_magi import cleanup_old_runs

        dirs = []
        for i in range(4):
            d = tmp_path / f"magi-run-{i:04d}"
            d.mkdir()
            # Set different mtimes
            os.utime(d, (1000 + i, 1000 + i))
            dirs.append(d)

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            cleanup_old_runs(2)

        # Most recent (dirs[2], dirs[3]) should remain
        assert dirs[3].exists()
        assert dirs[2].exists()
        assert not dirs[0].exists()
        assert not dirs[1].exists()

    def test_mtime_tie_uses_path_ascending_tiebreaker(self, tmp_path):
        """B-2: on mtime ties, cleanup must keep the lex-smallest path.

        Two or more ``magi-run-*`` dirs with identical ``st_mtime`` must
        produce a deterministic survivor. The contract is: sort by mtime
        descending, then by path ascending. The lex-smallest path is
        treated as the canonical survivor — not whatever ``os.scandir``
        happened to yield first.
        """
        from run_magi import cleanup_old_runs

        names = ["magi-run-0003", "magi-run-0001", "magi-run-0002"]
        for name in names:
            d = tmp_path / name
            d.mkdir()
            os.utime(d, (1000, 1000))  # identical mtime across all three

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            cleanup_old_runs(1)

        survivors = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("magi-run-"))
        assert survivors == ["magi-run-0001"], (
            f"Under mtime ties, the lex-smallest path must be kept, got {survivors}"
        )

    def test_mtime_tie_tiebreaker_keeps_top_n(self, tmp_path):
        """B-2: with keep=2 and all mtimes tied, the two lex-smallest survive."""
        from run_magi import cleanup_old_runs

        for name in ("magi-run-b", "magi-run-d", "magi-run-a", "magi-run-c"):
            d = tmp_path / name
            d.mkdir()
            os.utime(d, (2000, 2000))

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            cleanup_old_runs(2)

        survivors = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("magi-run-"))
        assert survivors == ["magi-run-a", "magi-run-b"]

    def test_cleanup_noop_when_no_magi_dirs(self, tmp_path):
        """B-2: with no magi-run-* entries, cleanup is a no-op.

        Unrelated files and directories in the temp root must survive
        and no exception must escape.
        """
        from run_magi import cleanup_old_runs

        (tmp_path / "other-dir").mkdir()
        (tmp_path / "readme.txt").write_text("keep me")

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            cleanup_old_runs(1)

        assert (tmp_path / "other-dir").exists()
        assert (tmp_path / "readme.txt").exists()

    def test_cleanup_works_when_tmpdir_itself_is_symlink(self, tmp_path, monkeypatch):
        """D-1a: a symlinked TMPDIR must not disable cleanup entirely.

        On macOS ``/tmp`` is a symlink to ``/private/tmp``; ``gettempdir()``
        returns ``/tmp`` but ``os.path.realpath(entry.path)`` resolves
        through the root symlink, so every candidate appears to live
        outside the ``/tmp/`` prefix and the traversal guard skips
        everything. The fix is to resolve the temp root the same way
        before building the safe prefix.

        This test simulates the scenario by monkeypatching ``realpath``
        so it runs identically on platforms without symlink support
        (e.g. Windows under a non-admin pytest run).
        """
        import temp_dirs
        from run_magi import cleanup_old_runs

        older = tmp_path / "magi-run-0001"
        older.mkdir()
        os.utime(older, (1000, 1000))
        newer = tmp_path / "magi-run-0002"
        newer.mkdir()
        os.utime(newer, (2000, 2000))

        advertised_root = str(tmp_path).replace(os.sep + "tmp", os.sep + "resolved_tmp", 1)
        if advertised_root == str(tmp_path):
            # Fallback: prepend a fake segment so realpath differs from the advertised path.
            advertised_root = str(tmp_path) + "_advertised"
        real_root_str = str(tmp_path)

        real_realpath = os.path.realpath

        def fake_realpath(path: str) -> str:
            # Rewrite the advertised (symlinked) root to the real one so
            # both the candidate entries and — crucially — the temp
            # root itself resolve to the same physical directory.
            if path == advertised_root or path.startswith(advertised_root + os.sep):
                return real_realpath(real_root_str + path[len(advertised_root) :])
            return real_realpath(path)

        monkeypatch.setattr(temp_dirs.os.path, "realpath", fake_realpath)
        monkeypatch.setattr(temp_dirs.tempfile, "gettempdir", lambda: advertised_root)

        # Rewrite scandir so it iterates the real tmp_path when asked
        # for the advertised symlinked root. This mirrors the OS-level
        # behavior on macOS: scandir follows the symlink transparently.
        real_scandir = os.scandir

        def fake_scandir(path):
            if path == advertised_root:
                return real_scandir(real_root_str)
            return real_scandir(path)

        monkeypatch.setattr(temp_dirs.os, "scandir", fake_scandir)

        cleanup_old_runs(1)

        assert newer.exists(), "Newest magi-run dir must be retained"
        assert not older.exists(), (
            "Oldest magi-run dir must be deleted even when TMPDIR is a symlink to its realpath"
        )

    def test_symlink_outside_temp_root_skipped(self, tmp_path):
        """Symlinks resolving outside temp root should be skipped."""
        from run_magi import cleanup_old_runs

        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        symlink_path = tmp_path / "magi-run-evil"
        try:
            symlink_path.symlink_to(outside_dir, target_is_directory=True)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            cleanup_old_runs(0)
            # keep=0 disables, use keep=1 with 2 dirs to trigger cleanup
            real_dir = tmp_path / "magi-run-real"
            real_dir.mkdir()
            os.utime(real_dir, (2000, 2000))
            os.utime(symlink_path, (1000, 1000))
            cleanup_old_runs(1)

        # Outside dir should not be deleted
        assert outside_dir.exists()

    def test_skips_live_locked_dir_even_when_oldest(self, tmp_path):
        """BDD-2: a dir whose lock PID is alive is never pruned."""
        from run_lock import write_lock
        from temp_dirs import cleanup_old_runs

        live = tmp_path / "magi-run-0000"
        live.mkdir()
        os.utime(live, (1000, 1000))  # oldest by mtime
        write_lock(str(live))  # our own (alive) PID

        for i in (1, 2, 3):
            d = tmp_path / f"magi-run-{i:04d}"
            d.mkdir()
            os.utime(d, (2000 + i, 2000 + i))

        cleanup_old_runs(1, str(tmp_path))

        assert live.exists(), "Live-locked dir must survive even as the oldest"

    def test_deletes_dead_locked_dir(self, tmp_path, monkeypatch):
        """BDD-3/11: a lock with a dead PID stays eligible for LRU pruning."""
        import run_lock
        from run_lock import write_lock
        from temp_dirs import cleanup_old_runs

        dead = tmp_path / "magi-run-0000"
        dead.mkdir()
        write_lock(str(dead))
        # Set mtime AFTER write_lock so the atomic rename does not overwrite
        # the backdated timestamp with the current time.
        os.utime(dead, (1000, 1000))
        newer = tmp_path / "magi-run-0001"
        newer.mkdir()
        os.utime(newer, (2000, 2000))

        monkeypatch.setattr(run_lock, "is_pid_alive", lambda pid: False)
        cleanup_old_runs(1, str(tmp_path))

        assert not dead.exists()
        assert newer.exists()

    def test_run_root_param_overrides_gettempdir(self, tmp_path):
        """The explicit run_root is scanned; gettempdir is not consulted."""
        from temp_dirs import cleanup_old_runs

        for i in range(3):
            d = tmp_path / f"magi-run-{i:04d}"
            d.mkdir()
            os.utime(d, (1000 + i, 1000 + i))

        # No gettempdir patch: correctness depends on the run_root arg.
        cleanup_old_runs(1, str(tmp_path))

        survivors = sorted(p.name for p in tmp_path.iterdir())
        assert survivors == ["magi-run-0002"]

    def test_missing_run_root_is_noop(self, tmp_path):
        """BDD-15: a non-existent run_root degrades to no-op (no raise)."""
        from temp_dirs import cleanup_old_runs

        cleanup_old_runs(1, str(tmp_path / "does-not-exist"))  # must not raise

    def test_cleanup_total_on_out_of_range_pid_lock(self, tmp_path):
        """cleanup_old_runs must not raise when a lock contains an out-of-range PID.

        A corrupt lock whose first line is an astronomically large integer
        causes os.kill(huge, 0) to raise OverflowError (POSIX) or the ctypes
        call to raise ctypes.ArgumentError (Windows). Without the fix, that
        exception propagates through is_dir_live into the comprehension and
        out of cleanup_old_runs, breaking every subsequent launch.
        The dir must be treated as live (conservative) so it is NOT deleted.
        """
        from run_lock import LOCK_FILENAME
        from temp_dirs import cleanup_old_runs

        run_dir = tmp_path / "magi-run-poisoned"
        run_dir.mkdir()
        # Write a lock whose PID line is out of range for any OS call.
        (run_dir / LOCK_FILENAME).write_text("99999999999999999999\n", encoding="utf-8")

        # Must not raise; and the dir must survive (treated as live).
        cleanup_old_runs(0, str(tmp_path))
        assert run_dir.exists(), "Out-of-range-PID dir must be treated as live (not deleted)"


class TestStderrShimModule:
    """C-2: the stderr-buffering machinery lives in its own module.

    ``_StderrBufferShim``, ``_BinaryStderrBufferShim``, and the
    ``_buffered_stderr_while`` context manager were embedded in
    run_magi.py, bloating the orchestrator. Extracting them to
    stderr_shim.py keeps run_magi focused on orchestration and makes
    the shim machinery independently testable.
    """

    def test_stderr_shim_module_importable(self):
        """The stderr_shim module must be importable by its short name."""
        import importlib

        module = importlib.import_module("stderr_shim")
        assert module is not None

    def test_stderr_shim_exposes_expected_symbols(self):
        """stderr_shim must export the three shim primitives."""
        import stderr_shim

        assert hasattr(stderr_shim, "_StderrBufferShim")
        assert hasattr(stderr_shim, "_BinaryStderrBufferShim")
        assert hasattr(stderr_shim, "_buffered_stderr_while")

    def test_run_magi_does_not_reexport_private_shim_names(self):
        """Regression (v2.1.1): ``run_magi`` must not re-export the
        underscored shim names.

        The earlier pattern ``__all__ = [..., "_StderrBufferShim", ...]``
        was contradictory: an underscore says "private", yet ``__all__``
        says "part of the star-import contract". Tests that need the
        shims import them from ``stderr_shim`` directly — the single
        owner of that API.
        """
        import run_magi

        for private in ("_StderrBufferShim", "_BinaryStderrBufferShim"):
            assert not hasattr(run_magi, private), (
                f"run_magi must not re-export {private}; import from stderr_shim instead."
            )
        # ``_buffered_stderr_while`` is still imported for internal use,
        # so it is reachable as an attribute, but it must not appear in
        # ``__all__`` — asserted separately in
        # ``TestAllDoesNotExportPrivateShimNames``.


class TestModelsModule:
    """C-1: MODEL_IDS and VALID_MODELS live in a dedicated models module.

    Bumping a model ID must be a one-line change to a data module, not
    an edit to the orchestration code in run_magi.py.
    """

    def test_models_module_importable(self):
        """The models module must be importable by its short name."""
        import importlib

        module = importlib.import_module("models")
        assert module is not None

    def test_model_ids_contains_expected_keys(self):
        """MODEL_IDS must map the three short names to Anthropic model IDs."""
        from models import MODEL_IDS

        assert set(MODEL_IDS.keys()) == {"opus", "sonnet", "haiku"}
        assert all(isinstance(v, str) and v for v in MODEL_IDS.values())

    def test_valid_models_derived_from_model_ids(self):
        """VALID_MODELS must stay in lockstep with MODEL_IDS.keys()."""
        from models import MODEL_IDS, VALID_MODELS

        assert set(VALID_MODELS) == set(MODEL_IDS.keys())

    def test_run_magi_reexports_model_ids_from_models_module(self):
        """run_magi.MODEL_IDS must be the same object as models.MODEL_IDS.

        Reference identity (``is``) — not merely equality — rules out
        accidental shadowing where run_magi keeps its own local copy
        that could drift from the canonical source.
        """
        import models
        import run_magi

        assert run_magi.MODEL_IDS is models.MODEL_IDS

    def test_run_magi_reexports_valid_models_from_models_module(self):
        """Same identity guarantee for VALID_MODELS."""
        import models
        import run_magi

        assert run_magi.VALID_MODELS is models.VALID_MODELS


class TestLaunchAgentValidation:
    """Verify launch_agent input validation."""

    @pytest.mark.asyncio
    async def test_invalid_model_raises_value_error(self, tmp_path):
        from run_magi import launch_agent

        with pytest.raises(ValueError, match="Unknown model"):
            await launch_agent(
                agent_name="melchior",
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
                spec=ModelSpec("gpt4", "anthropic"),
            )


class _FakeDisplay:
    """Test double that records update() calls without writing to any stream."""

    def __init__(self, *args, **kwargs):
        self.calls: list[tuple[str, str]] = []

    def update(self, agent: str, state: str) -> None:
        self.calls.append((agent, state))

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def _ok_result(name: str) -> dict:
    return {
        "agent": name,
        "verdict": "approve",
        "confidence": 0.9,
        "summary": f"{name} OK",
        "reasoning": "Fine",
        "findings": [],
        "recommendation": "Merge",
    }


class TestTrackedLaunchStatusUpdates:
    """Verify tracked_launch wiring between run_orchestrator and StatusDisplay."""

    @pytest.mark.asyncio
    async def test_success_path_emits_running_then_success(self, tmp_path, monkeypatch):
        import run_magi

        instances: list[_FakeDisplay] = []

        def factory(*args, **kwargs):
            inst = _FakeDisplay()
            instances.append(inst)
            return inst

        monkeypatch.setattr(run_magi, "StatusDisplay", factory)

        async def mock_launch(agent_name, *args, **kwargs):
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
        )

        assert len(instances) == 1
        calls = instances[0].calls
        for name in ("melchior", "balthasar", "caspar"):
            assert (name, "running") in calls
            assert (name, "success") in calls
            assert (name, "failed") not in calls
            assert (name, "timeout") not in calls

    @pytest.mark.asyncio
    async def test_builtin_timeout_error_emits_timeout(self, tmp_path, monkeypatch):
        import run_magi

        instances: list[_FakeDisplay] = []
        monkeypatch.setattr(
            run_magi,
            "StatusDisplay",
            lambda *a, **kw: instances.append(_FakeDisplay()) or instances[-1],
        )

        async def mock_launch(agent_name, *args, **kwargs):
            if agent_name == "caspar":
                raise TimeoutError("builtin timeout")
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
        )

        assert ("caspar", "timeout") in instances[0].calls
        assert ("caspar", "failed") not in instances[0].calls

    @pytest.mark.asyncio
    async def test_asyncio_timeout_error_emits_timeout(self, tmp_path, monkeypatch):
        """Python 3.9/3.10: asyncio.TimeoutError must be treated as timeout too."""
        import run_magi

        instances: list[_FakeDisplay] = []
        monkeypatch.setattr(
            run_magi,
            "StatusDisplay",
            lambda *a, **kw: instances.append(_FakeDisplay()) or instances[-1],
        )

        async def mock_launch(agent_name, *args, **kwargs):
            if agent_name == "caspar":
                raise asyncio.TimeoutError("asyncio timeout")
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
        )

        assert ("caspar", "timeout") in instances[0].calls
        assert ("caspar", "failed") not in instances[0].calls

    @pytest.mark.asyncio
    async def test_generic_exception_emits_failed(self, tmp_path, monkeypatch):
        import run_magi

        instances: list[_FakeDisplay] = []
        monkeypatch.setattr(
            run_magi,
            "StatusDisplay",
            lambda *a, **kw: instances.append(_FakeDisplay()) or instances[-1],
        )

        async def mock_launch(agent_name, *args, **kwargs):
            if agent_name == "caspar":
                raise RuntimeError("boom")
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
        )

        assert ("caspar", "failed") in instances[0].calls
        assert ("caspar", "timeout") not in instances[0].calls

    @pytest.mark.asyncio
    async def test_show_status_false_skips_display(self, tmp_path, monkeypatch):
        import run_magi

        created: list[int] = []
        monkeypatch.setattr(
            run_magi,
            "StatusDisplay",
            lambda *a, **kw: created.append(1) or _FakeDisplay(),
        )

        async def mock_launch(agent_name, *args, **kwargs):
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
            show_status=False,
        )

        assert created == []

    @pytest.mark.asyncio
    async def test_cancelled_error_marks_display_failed(self, tmp_path, monkeypatch):
        """W4: CancelledError in an agent must mark its display row as failed."""
        import run_magi

        instances: list[_FakeDisplay] = []
        monkeypatch.setattr(
            run_magi,
            "StatusDisplay",
            lambda *a, **kw: instances.append(_FakeDisplay()) or instances[-1],
        )

        async def mock_launch(agent_name, *args, **kwargs):
            if agent_name == "caspar":
                raise asyncio.CancelledError()
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        result = await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
        )

        assert ("caspar", "running") in instances[0].calls
        assert ("caspar", "failed") in instances[0].calls
        # caspar's row must not be left in "running" state and must not
        # be marked as "success".
        assert ("caspar", "success") not in instances[0].calls
        assert result.get("degraded") is True

    @pytest.mark.asyncio
    async def test_display_start_failure_falls_through_gracefully(
        self, tmp_path, monkeypatch, capsys
    ):
        """A raised ``display.start()`` must not block the analysis."""
        import run_magi

        class _FailingStartDisplay:
            def __init__(self, *args, **kwargs):
                self.updates: list[tuple[str, str]] = []
                self.stop_called = False

            def update(self, agent: str, state: str) -> None:
                self.updates.append((agent, state))

            async def start(self) -> None:
                raise RuntimeError("simulated start failure")

            async def stop(self) -> None:
                self.stop_called = True

        instances: list[_FailingStartDisplay] = []

        def factory(*args, **kwargs):
            inst = _FailingStartDisplay()
            instances.append(inst)
            return inst

        monkeypatch.setattr(run_magi, "StatusDisplay", factory)

        async def mock_launch(agent_name, *args, **kwargs):
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        result = await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
        )

        assert result["consensus"]["consensus"] == "STRONG GO"
        assert len(instances) == 1
        # Display was dropped, so stop() is never called and no further
        # ``update()`` calls reach it after the start() failure — the
        # tracked_launch closure must see ``display is None``.
        assert instances[0].stop_called is False
        assert instances[0].updates == [], (
            f"No updates must reach a failed-start display, got {instances[0].updates}"
        )

        captured = capsys.readouterr()
        assert "status display failed to start" in captured.err

    @pytest.mark.asyncio
    async def test_display_update_errors_do_not_mask_original_exception(
        self, tmp_path, monkeypatch
    ):
        """If display.update() raises during shutdown, the real error must win."""
        import run_magi

        class _BrokenDisplay:
            def __init__(self, *args, **kwargs):
                self.stop_called = False

            def update(self, agent: str, state: str) -> None:
                raise RuntimeError("display is broken")

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                self.stop_called = True

        monkeypatch.setattr(run_magi, "StatusDisplay", _BrokenDisplay)

        async def mock_launch(agent_name, *args, **kwargs):
            if agent_name == "caspar":
                raise ValueError("original failure")
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        # The orchestrator must still return (degraded) — the BrokenDisplay
        # update call must not propagate and mask caspar's ValueError.
        result = await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
        )
        assert result.get("degraded") is True
        assert "caspar" in result.get("failed_agents", [])


class _FakeTimeoutProc:
    """Fake asyncio subprocess for timeout-path testing.

    ``communicate()`` hangs indefinitely so ``asyncio.wait_for`` fires a
    ``TimeoutError``. ``kill()`` and ``wait()`` record call order so tests
    can verify zombie reaping. ``proc.stderr`` is a prefilled
    :class:`asyncio.StreamReader` so the production code can drain buffered
    diagnostics after killing the process.
    """

    def __init__(
        self,
        stdout_bytes: bytes = b"",
        stderr_bytes: bytes = b"",
    ) -> None:
        self.returncode: int | None = None
        self.kill_called = False
        self.wait_called = False
        self.call_order: list[str] = []
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(stdout_bytes)
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_data(stderr_bytes)
        self.stderr.feed_eof()
        self.stdin = None
        # Fake pid so the Windows tree-kill path in ``_reap_and_drain_stderr``
        # has something to pass to ``taskkill``. Test fixtures monkeypatch
        # ``subprocess.run`` so the call is inert.
        self.pid = 999_000

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        # Hang so wait_for raises TimeoutError.
        await asyncio.sleep(3600)
        return b"", b""

    def kill(self) -> None:
        self.kill_called = True
        self.call_order.append("kill")
        self.returncode = -9

    async def wait(self) -> int | None:
        self.wait_called = True
        self.call_order.append("wait")
        return self.returncode


class TestLaunchAgentTimeoutReaping:
    """A-1: zombie reaping and stderr capture on agent timeout."""

    @pytest.fixture(autouse=True)
    def _stub_taskkill(self, monkeypatch):
        """Stub ``subprocess.run`` so the Windows tree-kill path in
        ``reap_and_drain_stderr`` does not invoke the real ``taskkill``
        against a fake pid and slow each test down by several seconds.
        """
        import subprocess_utils

        def _noop_run(*args, **kwargs):
            class _Completed:
                returncode = 0

            return _Completed()

        monkeypatch.setattr(subprocess_utils.subprocess, "run", _noop_run)

    @pytest.mark.asyncio
    async def test_wait_awaited_after_kill_on_timeout(self, tmp_path, monkeypatch):
        """``proc.kill()`` must be followed by ``await proc.wait()`` to reap."""
        import run_magi

        fake = _FakeTimeoutProc(stderr_bytes=b"")

        async def fake_create(*args, **kwargs):
            return fake

        monkeypatch.setattr(run_magi.asyncio, "create_subprocess_exec", fake_create)
        (tmp_path / "melchior.md").write_text("sys prompt", encoding="utf-8")

        with pytest.raises(TimeoutError):
            await run_magi.launch_agent(
                agent_name="melchior",
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=1,
            )

        assert fake.kill_called, "kill() must be called on timeout"
        assert fake.wait_called, "wait() must be awaited after kill() to reap zombie"
        assert fake.call_order == ["kill", "wait"], (
            f"Order must be kill→wait, got {fake.call_order}"
        )

    @pytest.mark.asyncio
    async def test_stderr_persisted_to_log_on_timeout(self, tmp_path, monkeypatch):
        """Buffered stderr must be written to ``{agent}.stderr.log`` on timeout."""
        import run_magi

        stderr_payload = b"agent started thinking\nmid-computation diag\n"
        fake = _FakeTimeoutProc(stderr_bytes=stderr_payload)

        async def fake_create(*args, **kwargs):
            return fake

        monkeypatch.setattr(run_magi.asyncio, "create_subprocess_exec", fake_create)
        (tmp_path / "melchior.md").write_text("sys prompt", encoding="utf-8")

        with pytest.raises(TimeoutError):
            await run_magi.launch_agent(
                agent_name="melchior",
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=1,
            )

        stderr_log = tmp_path / "melchior.stderr.log"
        assert stderr_log.exists(), (
            "Stderr log must be persisted on timeout for post-mortem diagnosis"
        )
        assert stderr_log.read_bytes() == stderr_payload

    @pytest.mark.asyncio
    async def test_timeout_error_surfaces_stderr_excerpt(self, tmp_path, monkeypatch):
        """TimeoutError message must include stderr excerpt so operators see why."""
        import run_magi

        fake = _FakeTimeoutProc(stderr_bytes=b"Connection refused to upstream API")

        async def fake_create(*args, **kwargs):
            return fake

        monkeypatch.setattr(run_magi.asyncio, "create_subprocess_exec", fake_create)
        (tmp_path / "melchior.md").write_text("sys prompt", encoding="utf-8")

        with pytest.raises(TimeoutError, match="Connection refused"):
            await run_magi.launch_agent(
                agent_name="melchior",
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=1,
            )

    @pytest.mark.asyncio
    async def test_write_stderr_log_oserror_does_not_mask_timeout(self, tmp_path, monkeypatch):
        """D-1b: OSError from the stderr-log write must not shadow TimeoutError.

        If the disk is full or read-only when we try to persist buffered
        diagnostics on the timeout path, the caller must still see the
        original ``TimeoutError`` — swallowing it behind an ``OSError``
        hides the real cause from the orchestrator's failure summary.
        """
        import run_magi

        fake = _FakeTimeoutProc(stderr_bytes=b"partial diagnostics before hang")

        async def fake_create(*args, **kwargs):
            return fake

        monkeypatch.setattr(run_magi.asyncio, "create_subprocess_exec", fake_create)
        (tmp_path / "melchior.md").write_text("sys prompt", encoding="utf-8")

        def failing_write(output_dir, agent_name, data):
            raise OSError(28, "No space left on device")

        import claude_backend as _claude_backend

        monkeypatch.setattr(_claude_backend, "_write_stderr_log", failing_write)

        with pytest.raises(TimeoutError, match="timed out after"):
            await run_magi.launch_agent(
                agent_name="melchior",
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=1,
            )

    @pytest.mark.asyncio
    async def test_empty_stderr_on_timeout_does_not_create_log(self, tmp_path, monkeypatch):
        """No stderr data ⇒ no empty .stderr.log file should be written."""
        import run_magi

        fake = _FakeTimeoutProc(stderr_bytes=b"")

        async def fake_create(*args, **kwargs):
            return fake

        monkeypatch.setattr(run_magi.asyncio, "create_subprocess_exec", fake_create)
        (tmp_path / "melchior.md").write_text("sys prompt", encoding="utf-8")

        with pytest.raises(TimeoutError):
            await run_magi.launch_agent(
                agent_name="melchior",
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=1,
            )

        assert not (tmp_path / "melchior.stderr.log").exists()


_FAKE_AGENT_JSON = (
    '{"agent": "melchior", "verdict": "approve", "confidence": 0.8, '
    '"summary": "ok", "reasoning": "looks fine", "findings": [], '
    '"recommendation": "merge"}'
)
# The ``claude -p --output-format json`` envelope wraps the agent JSON
# as a string under ``result`` — match that shape so the real
# ``parse_agent_output`` pipeline accepts the mock.
_FAKE_CLAUDE_ENVELOPE = (
    '{"result": "{\\"agent\\": \\"melchior\\", \\"verdict\\": \\"approve\\", '
    '\\"confidence\\": 0.8, \\"summary\\": \\"ok\\", \\"reasoning\\": '
    '\\"looks fine\\", \\"findings\\": [], \\"recommendation\\": \\"merge\\"}"}'
).encode("utf-8")


class _FakeSuccessProc:
    """Fake asyncio subprocess that simulates a successful agent run.

    Used by regression tests that need the full happy path through
    ``launch_agent`` without spawning the real ``claude`` CLI.
    """

    def __init__(
        self,
        stdout_bytes: bytes = _FAKE_CLAUDE_ENVELOPE,
        stderr_bytes: bytes = b"some stderr",
    ) -> None:
        self._stdout = stdout_bytes
        self._stderr = stderr_bytes
        self.returncode: int | None = None
        self.stdin = None

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        self.returncode = 0
        return self._stdout, self._stderr

    def kill(self) -> None:  # pragma: no cover — never called on success path
        pass

    async def wait(self) -> int | None:  # pragma: no cover
        return self.returncode


class TestLaunchAgentSuccessStderrLog:
    """Regression (v2.1.1): success-path stderr log write must not mask
    an otherwise-successful agent when disk/permission errors occur.
    """

    @pytest.mark.asyncio
    async def test_success_path_oserror_does_not_mask_result(self, tmp_path, monkeypatch, capsys):
        """D-1c: OSError from the stderr-log write on the success path
        must be caught and logged, not propagated — the agent's parsed
        JSON is already valid at that point.

        Pre-2.1.1, the success-path ``_write_stderr_log`` call was bare;
        a disk-full or antivirus-lock error on Windows would bubble up
        from ``launch_agent`` and be reported as an agent failure in
        ``tracked_launch`` even though the agent itself succeeded. The
        fix mirrors the timeout-path ``try/except OSError`` pattern and
        is covered by this test.
        """
        import run_magi

        fake = _FakeSuccessProc(stderr_bytes=b"diagnostic line")

        async def fake_create(*args, **kwargs):
            return fake

        def failing_write(output_dir, agent_name, data):
            raise OSError(13, "Permission denied")

        import claude_backend as _claude_backend

        monkeypatch.setattr(run_magi.asyncio, "create_subprocess_exec", fake_create)
        monkeypatch.setattr(_claude_backend, "_write_stderr_log", failing_write)
        (tmp_path / "melchior.md").write_text("sys prompt", encoding="utf-8")

        result = await run_magi.launch_agent(
            agent_name="melchior",
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=5,
        )
        assert result["agent"] == "melchior"
        assert result["verdict"] == "approve"
        captured = capsys.readouterr()
        assert "Failed to persist" in captured.err
        assert "melchior.stderr.log" in captured.err


class TestTaskkillTimeoutBudget:
    """Regression (v2.1.1): ``_TASKKILL_TIMEOUT`` must be independent of
    ``_PROC_WAIT_REAP_TIMEOUT`` so a slow ``taskkill`` does not consume
    the ``proc.wait()`` budget and fire a misleading orphan warning.
    """

    def test_taskkill_timeout_is_separate_constant(self):
        """The two timeouts are distinct module-level constants and
        operators can tune them without conflating the budgets.
        """
        from subprocess_utils import PROC_WAIT_REAP_TIMEOUT, TASKKILL_TIMEOUT

        # Both are floats > 0 — the exact values may change over time,
        # but they must live in separate constants so one slow call does
        # not poison the other's observability.
        assert isinstance(TASKKILL_TIMEOUT, float)
        assert isinstance(PROC_WAIT_REAP_TIMEOUT, float)
        assert TASKKILL_TIMEOUT > 0
        assert PROC_WAIT_REAP_TIMEOUT > 0

    def test_windows_kill_tree_uses_taskkill_timeout(self, monkeypatch):
        """``windows_kill_tree`` must pass ``TASKKILL_TIMEOUT`` to
        ``subprocess.run``, not ``PROC_WAIT_REAP_TIMEOUT`` — otherwise
        collapsing the two constants back into one would pass silently.
        """
        import sys as _sys

        if _sys.platform != "win32":
            pytest.skip("Windows-only path")

        import subprocess_utils

        captured: dict = {}

        def fake_run(argv, **kwargs):
            captured.update(kwargs)

            class _Completed:
                returncode = 0

            return _Completed()

        monkeypatch.setattr(subprocess_utils.subprocess, "run", fake_run)
        subprocess_utils.windows_kill_tree(54321)
        assert captured.get("timeout") == subprocess_utils.TASKKILL_TIMEOUT


class TestAllDoesNotExportPrivateShimNames:
    """Regression (v2.1.1): ``__all__`` must not expose underscore-prefixed
    names from ``stderr_shim`` — the shims are private to that module
    and tests should import them from ``stderr_shim`` directly.
    """

    def test_all_has_no_underscore_entries(self):
        from run_magi import __all__

        underscored = [name for name in __all__ if name.startswith("_")]
        assert not underscored, (
            f"__all__ must not expose private names: {underscored!r}. "
            "Tests needing the shims should import from stderr_shim."
        )

    def test_all_exposes_public_api(self):
        """The public API kept in __all__ must still be reachable."""
        from run_magi import __all__

        assert "MODEL_IDS" in __all__
        assert "VALID_MODELS" in __all__
        assert "resolve_model" in __all__


class TestSafeDisplayUpdate:
    """Verify ``_safe_display_update`` swallows display errors during shutdown."""

    def test_none_display_is_noop(self):
        from run_magi import _DisplayLogGate, _safe_display_update

        _safe_display_update(None, "melchior", "running", _DisplayLogGate())  # must not raise

    def test_exception_is_swallowed(self):
        from run_magi import _DisplayLogGate, _safe_display_update

        class _Broken:
            def update(self, agent: str, state: str) -> None:
                raise RuntimeError("broken")

        _safe_display_update(_Broken(), "melchior", "running", _DisplayLogGate())

    def test_first_exception_logged_subsequent_silent(self, capsys):
        """A broken display must surface its first error to stderr so the
        operator knows the live tree is blind, but subsequent errors stay
        silent to prevent the redraw path from flooding the log on every
        tick. The real shutdown signal from the caller is still preserved
        because ``_safe_display_update`` never re-raises."""
        from run_magi import _DisplayLogGate, _safe_display_update

        gate = _DisplayLogGate()

        class _Broken:
            def update(self, agent: str, state: str) -> None:
                raise RuntimeError("boom")

        broken = _Broken()
        _safe_display_update(broken, "melchior", "running", gate)
        _safe_display_update(broken, "balthasar", "running", gate)
        _safe_display_update(broken, "caspar", "running", gate)

        captured = capsys.readouterr()
        assert captured.err.count("status display") == 1, (
            "First failure must be logged exactly once; subsequent failures must stay silent."
        )
        assert "boom" in captured.err

    def test_fresh_gate_per_run_rearms_log(self, capsys):
        """Each run gets a new ``_DisplayLogGate``, so the first failure of
        every run surfaces to stderr. Without per-run isolation a long-lived
        host that reuses the module would never see display failures after
        the first run.
        """
        from run_magi import _DisplayLogGate, _safe_display_update

        class _Broken:
            def update(self, agent: str, state: str) -> None:
                raise RuntimeError("boom")

        broken = _Broken()
        # Run 1.
        _safe_display_update(broken, "melchior", "running", _DisplayLogGate())
        # Run 2 (separate gate).
        _safe_display_update(broken, "melchior", "running", _DisplayLogGate())

        captured = capsys.readouterr()
        assert captured.err.count("status display") == 2, (
            "A fresh gate per run must re-arm the first-failure log."
        )

    def test_successful_update_propagates(self):
        from run_magi import _DisplayLogGate, _safe_display_update

        class _Recorder:
            def __init__(self):
                self.calls: list[tuple[str, str]] = []

            def update(self, agent: str, state: str) -> None:
                self.calls.append((agent, state))

        rec = _Recorder()
        _safe_display_update(rec, "melchior", "running", _DisplayLogGate())
        assert rec.calls == [("melchior", "running")]

    def test_base_exception_is_swallowed(self):
        """The helper's contract explicitly names ``CancelledError`` and
        ``KeyboardInterrupt`` (both ``BaseException`` subclasses) as
        shutdown-path failures it must not propagate. ``tracked_launch``
        is wrapped in ``except BaseException`` and relies on this helper
        returning normally so the outer ``raise`` re-raises the *original*
        signal instead of whatever the display raised on the way down.
        """
        import asyncio

        from run_magi import _DisplayLogGate, _safe_display_update

        gate = _DisplayLogGate()

        class _CancelledRaiser:
            def update(self, agent: str, state: str) -> None:
                raise asyncio.CancelledError("display cancelled mid-shutdown")

        class _SystemExitRaiser:
            def update(self, agent: str, state: str) -> None:
                raise SystemExit(2)

        # Neither call may propagate — the documented contract says the
        # helper swallows shutdown-path failures so the caller's own
        # ``raise`` preserves the original exception.
        _safe_display_update(_CancelledRaiser(), "melchior", "failed", gate)
        _safe_display_update(_SystemExitRaiser(), "caspar", "failed", gate)


class TestReapAndDrainStderr:
    """Verify timeout warning when a killed subprocess fails to exit."""

    def test_warns_when_proc_wait_times_out(self, capsys, monkeypatch):
        """If ``proc.wait()`` still hasn't returned within
        ``_PROC_WAIT_REAP_TIMEOUT`` seconds after ``kill()``, the caller
        must emit a warning to stderr so an operator can notice an
        orphaned subprocess (Windows child-process-tree case). The
        function must still return the best-effort stderr buffer and
        must not raise."""
        import asyncio

        from subprocess_utils import PROC_WAIT_REAP_TIMEOUT, reap_and_drain_stderr

        class _FakeStderr:
            async def read(self) -> bytes:
                return b""

        class _FakeProc:
            pid = 9999
            stderr = _FakeStderr()
            kill_called = False

            def kill(self) -> None:
                type(self).kill_called = True

            async def wait(self) -> int:
                await asyncio.sleep(10)  # simulate hang
                return 0

        async def _fake_wait_for(awaitable, timeout):
            # Consume the coroutine so asyncio doesn't warn about it,
            # then raise to simulate the reap timeout on the wait() call.
            if timeout == PROC_WAIT_REAP_TIMEOUT:
                if asyncio.iscoroutine(awaitable):
                    awaitable.close()
                raise asyncio.TimeoutError
            return await awaitable

        monkeypatch.setattr("subprocess_utils.asyncio.wait_for", _fake_wait_for)

        proc = _FakeProc()
        result = asyncio.run(reap_and_drain_stderr(proc))  # type: ignore[arg-type]

        assert result == b""
        assert _FakeProc.kill_called is True
        captured = capsys.readouterr()
        assert "9999" in captured.err, (
            "Warning must name the unreaped subprocess so operators can identify the orphan."
        )
        assert "did not exit" in captured.err or "orphan" in captured.err.lower()

    def test_windows_invokes_taskkill_tree(self, monkeypatch):
        """On Windows, the reap path must also issue ``taskkill /F /T /PID``
        so orphan child processes (a real hazard when ``claude`` spawns
        its own helpers) do not survive a MAGI timeout.

        The existing ``proc.kill()`` is kept for signalling, and
        ``taskkill`` is invoked in addition to it — not as a replacement
        — because ``taskkill`` may fail if the binary is missing or a
        timeout cuts it off. Calling both makes the reap more robust
        without regressing the single-process case.
        """
        import asyncio
        import sys as _sys

        if _sys.platform != "win32":
            pytest.skip("Windows-only path")

        import subprocess_utils

        recorded_argv: list[list[str]] = []

        def fake_run(argv, **kwargs):
            recorded_argv.append(list(argv))

            class _Completed:
                returncode = 0

            return _Completed()

        monkeypatch.setattr(subprocess_utils.subprocess, "run", fake_run)

        class _FakeStderr:
            async def read(self) -> bytes:
                return b""

        class _FakeProc:
            pid = 12345
            stderr = _FakeStderr()

            def kill(self) -> None:
                pass

            async def wait(self) -> int:
                return 0

        asyncio.run(subprocess_utils.reap_and_drain_stderr(_FakeProc()))  # type: ignore[arg-type]

        assert any(
            argv[:4] == ["taskkill", "/F", "/T", "/PID"] and argv[4] == "12345"
            for argv in recorded_argv
        ), f"Expected taskkill invocation for pid 12345, recorded: {recorded_argv!r}"

    def test_windows_taskkill_runs_before_proc_kill(self, monkeypatch):
        """On Windows, ``taskkill /F /T /PID`` must be invoked BEFORE
        ``proc.kill()``. Calling ``proc.kill()`` first issues
        ``TerminateProcess`` against the parent, after which the
        kernel may have torn down the parent-child relationship that
        ``taskkill /T`` walks to enumerate descendants — leaving the
        orphan window the function exists to close still open.

        This is a regression guard: pre-2.1.2 the order was inverted
        and the tree-kill was effectively a no-op for child processes
        the ``claude`` CLI had spawned.
        """
        import sys as _sys

        if _sys.platform != "win32":
            pytest.skip("Windows-only path")

        import subprocess_utils

        call_order: list[str] = []

        def fake_run(argv, **kwargs):
            call_order.append("taskkill")

            class _Completed:
                returncode = 0

            return _Completed()

        monkeypatch.setattr(subprocess_utils.subprocess, "run", fake_run)

        class _FakeStderr:
            async def read(self) -> bytes:
                return b""

        class _FakeProc:
            pid = 99999
            stderr = _FakeStderr()

            def kill(self) -> None:
                call_order.append("proc_kill")

            async def wait(self) -> int:
                return 0

        asyncio.run(subprocess_utils.reap_and_drain_stderr(_FakeProc()))  # type: ignore[arg-type]

        assert call_order, "expected at least one of taskkill / proc_kill to fire"
        assert call_order[0] == "taskkill", (
            f"taskkill must run before proc.kill(); recorded order: {call_order!r}"
        )
        assert "proc_kill" in call_order, (
            "proc.kill() must still be invoked after the tree-kill so the "
            "asyncio.subprocess wrapper observes the exit cleanly."
        )


class TestBufferedStderrWhile:
    """Structural enforcement of the display-active stderr-quiet invariant (W3)."""

    def test_noop_when_inactive(self):
        """When active=False, sys.stderr is untouched and writes pass through."""
        from run_magi import _buffered_stderr_while

        original = sys.stderr
        with _buffered_stderr_while(active=False):
            assert sys.stderr is original

    def test_buffers_writes_when_active(self, capsys):
        """When active=True, writes are buffered and replayed on context exit."""
        from run_magi import _buffered_stderr_while

        with _buffered_stderr_while(active=True):
            print("line 1", file=sys.stderr)
            print("line 2", file=sys.stderr)
            # Nothing should have reached real stderr yet.
            captured_mid = capsys.readouterr()
            assert captured_mid.err == ""

        # After context exit, buffered content is replayed.
        captured_after = capsys.readouterr()
        assert "line 1" in captured_after.err
        assert "line 2" in captured_after.err

    def test_restores_original_stderr_on_exit(self):
        """The original sys.stderr reference must be restored after the context."""
        from run_magi import _buffered_stderr_while

        original = sys.stderr
        with _buffered_stderr_while(active=True):
            assert sys.stderr is not original
        assert sys.stderr is original

    def test_proxies_non_write_attributes(self):
        """The shim must proxy encoding/isatty/fileno to the real stderr."""
        from run_magi import _buffered_stderr_while

        real_encoding = getattr(sys.stderr, "encoding", None)
        with _buffered_stderr_while(active=True):
            # isatty() and encoding come from the real stderr via __getattr__.
            assert sys.stderr.encoding == real_encoding
            # The shim is not the real stream.
            assert sys.stderr is not sys.__stderr__

    def test_restores_stderr_even_on_exception(self):
        """Context manager must restore stderr when the body raises."""
        from run_magi import _buffered_stderr_while

        original = sys.stderr
        with pytest.raises(RuntimeError):
            with _buffered_stderr_while(active=True):
                raise RuntimeError("boom")
        assert sys.stderr is original

    def test_binary_buffer_writes_are_intercepted(self, capsys):
        """Writes through ``sys.stderr.buffer.write`` must also be buffered."""
        from run_magi import _buffered_stderr_while

        with _buffered_stderr_while(active=True):
            shim_buffer = getattr(sys.stderr, "buffer", None)
            if shim_buffer is None:
                pytest.skip("pytest capture stream has no .buffer attribute")
            shim_buffer.write(b"binary diag line\n")
            captured_mid = capsys.readouterr()
            assert captured_mid.err == ""

        captured_after = capsys.readouterr()
        assert "binary diag line" in captured_after.err

    def test_shim_buffer_attribute_exists_when_real_has_buffer(self):
        """The shim must expose a ``.buffer`` shim when the real stderr has one."""
        from stderr_shim import _BinaryStderrBufferShim, _StderrBufferShim

        class _FakeBinary:
            def write(self, data: bytes) -> int:
                return len(data)

            def flush(self) -> None:
                pass

        class _FakeStderr:
            def __init__(self):
                self.buffer = _FakeBinary()

            def write(self, data: str) -> int:
                return len(data)

            def flush(self) -> None:
                pass

        text_buffer: list[str] = []
        shim = _StderrBufferShim(_FakeStderr(), text_buffer)
        assert shim.buffer is not None
        assert isinstance(shim.buffer, _BinaryStderrBufferShim)

        shim.buffer.write(b"hello\n")
        assert text_buffer == ["hello\n"]

    def test_shim_buffer_none_when_real_has_no_buffer(self):
        """When the real stderr lacks ``.buffer``, the shim's ``.buffer`` is None."""
        import io

        from stderr_shim import _StderrBufferShim

        text_buffer: list[str] = []
        shim = _StderrBufferShim(io.StringIO(), text_buffer)
        assert shim.buffer is None

    @pytest.mark.asyncio
    async def test_orchestrator_buffers_stderr_during_gather(self, tmp_path, monkeypatch, capsys):
        """End-to-end: writes from tracked tasks are buffered, then flushed."""
        import run_magi

        monkeypatch.setattr(run_magi, "StatusDisplay", lambda *a, **kw: _FakeDisplay())

        async def mock_launch(agent_name, *args, **kwargs):
            # Simulate a task that writes to stderr mid-run.
            print(f"diag from {agent_name}", file=sys.stderr)
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
        )

        captured = capsys.readouterr()
        # Diagnostic writes must have been replayed after the display stopped.
        assert "diag from melchior" in captured.err
        assert "diag from balthasar" in captured.err
        assert "diag from caspar" in captured.err

    def test_replay_oserror_does_not_mask_body_exception(self):
        """If the buffered-stderr replay raises ``OSError`` (the real
        stderr is closed, the parent pipe is dead, the file descriptor
        is gone), the original exception in flight from the body must
        propagate — the write failure during cleanup must not shadow
        the root cause.

        Pre-2.1.2 the ``finally`` clause did ``saved.write(...);
        saved.flush()`` unguarded. A ``BrokenPipeError`` during replay
        would raise out of the context manager and overwrite the body's
        exception, hiding the real failure from the operator.
        """
        from stderr_shim import _buffered_stderr_while

        class _BrokenStderr:
            encoding = "utf-8"
            buffer = None

            def write(self, data: str) -> int:
                raise BrokenPipeError("pipe closed during replay")

            def flush(self) -> None:
                pass

            def isatty(self) -> bool:
                return False

        saved = sys.stderr
        sys.stderr = _BrokenStderr()  # type: ignore[assignment]
        try:
            with pytest.raises(RuntimeError, match="root cause"):
                with _buffered_stderr_while(active=True):
                    print("buffered diagnostic", file=sys.stderr)
                    raise RuntimeError("root cause")
        finally:
            sys.stderr = saved

    def test_replay_oserror_alone_is_swallowed(self):
        """When the body succeeds but the replay raises ``OSError``,
        the context manager must exit cleanly. Re-raising the write
        failure from a cleanup-only path would crash the orchestrator
        on the way out for what is purely a diagnostics-delivery
        problem.
        """
        from stderr_shim import _buffered_stderr_while

        class _BrokenStderr:
            encoding = "utf-8"
            buffer = None

            def write(self, data: str) -> int:
                raise OSError(32, "Broken pipe")

            def flush(self) -> None:
                pass

            def isatty(self) -> bool:
                return False

        saved = sys.stderr
        sys.stderr = _BrokenStderr()  # type: ignore[assignment]
        try:
            with _buffered_stderr_while(active=True):
                print("diag that will fail to replay", file=sys.stderr)
        finally:
            sys.stderr = saved


class TestSingleShotRetry:
    """2.2.0: single-shot retry when an agent fails schema validation.

    Contract driven by these tests:

    * When :func:`launch_agent` raises :class:`ValidationError`,
      :func:`run_orchestrator` retries that specific agent **once** with
      corrective feedback appended to the prompt.
    * Retry fires **only** on :class:`ValidationError`. ``TimeoutError``,
      ``RuntimeError``, ``ValueError``, ``asyncio.CancelledError``, and any
      other exception flow through the existing degraded-mode path
      unchanged.
    * Each attempt receives the full ``--timeout`` budget. The retry is
      not given a reduced ceiling, so operators never see a doubled wall
      clock but always see the full configured per-attempt budget.
    * A ``retrying`` display state is emitted between ``running`` and the
      terminal state (``success`` / ``failed``) for the retried agent.
    * If the retry succeeds, the run completes with full 3-agent
      consensus and ``degraded`` is **not** set.
    * If the retry also raises ``ValidationError`` (or any other
      exception), the agent is dropped and the run continues on the
      surviving agents under the pre-existing 2-agent minimum rule.
    """

    @staticmethod
    def _valid(agent: str) -> dict[str, Any]:
        """Helper: build a schema-valid agent output dict."""
        return {
            "agent": agent,
            "verdict": "approve",
            "confidence": 0.85,
            "summary": f"{agent} OK",
            "reasoning": "Fine",
            "findings": [],
            "recommendation": "Merge",
        }

    @pytest.mark.asyncio
    async def test_schema_failure_triggers_retry_success(self, tmp_path):
        """First call raises ValidationError, second call succeeds.

        The orchestrator must retry the single failing agent and emerge
        with a full 3-agent consensus, no ``degraded`` flag set.
        """
        from run_magi import run_orchestrator
        from validate import ValidationError

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            call_counts[agent_name] += 1
            if agent_name == "caspar" and call_counts[agent_name] == 1:
                raise ValidationError("missing keys: ['recommendation']")
            return TestSingleShotRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
            assert result.get("degraded") is not True, (
                "retry that succeeds must not leave degraded flag set"
            )
            assert len(result["agents"]) == 3
            assert call_counts["caspar"] == 2, "caspar must be retried exactly once"
            assert call_counts["melchior"] == 1, "melchior must not be retried"
            assert call_counts["balthasar"] == 1, "balthasar must not be retried"

    @pytest.mark.asyncio
    async def test_retry_also_fails_degraded_mode(self, tmp_path):
        """Both attempts raise ValidationError → agent dropped, degraded=True."""
        from run_magi import run_orchestrator
        from validate import ValidationError

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            call_counts[agent_name] += 1
            if agent_name == "caspar":
                raise ValidationError("missing keys: ['recommendation']")
            return TestSingleShotRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
            assert result["degraded"] is True
            assert "caspar" in result["failed_agents"]
            assert len(result["agents"]) == 2
            assert call_counts["caspar"] == 2, (
                "caspar must be attempted exactly twice (initial + one retry)"
            )

    @pytest.mark.asyncio
    async def test_two_agents_both_exhaust_retries_raises(self, tmp_path):
        """Two agents fail both attempts → only one survivor → RuntimeError.

        The 2-agent minimum is unchanged by retry. If two agents burn
        through their retry budget, the run must raise the same
        ``RuntimeError`` it raises today — retry does not lower the
        consensus floor.
        """
        from run_magi import run_orchestrator
        from validate import ValidationError

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            if agent_name in ("caspar", "melchior"):
                raise ValidationError(f"missing keys for {agent_name}")
            return TestSingleShotRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            with pytest.raises(RuntimeError, match="fewer than 2"):
                await run_orchestrator(
                    agents_dir=str(tmp_path),
                    prompt="test",
                    output_dir=str(tmp_path),
                    timeout=300,
                )

    @pytest.mark.asyncio
    async def test_timeout_does_not_trigger_retry(self, tmp_path):
        """``TimeoutError`` must not trigger retry (non-goal for 2.2.0)."""
        from run_magi import run_orchestrator

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            call_counts[agent_name] += 1
            if agent_name == "caspar":
                raise TimeoutError(f"agent {agent_name} timed out")
            return TestSingleShotRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
            assert result["degraded"] is True
            assert "caspar" in result["failed_agents"]
            assert call_counts["caspar"] == 1, (
                "timeout must not be retried — retry scope is schema only"
            )

    @pytest.mark.asyncio
    async def test_runtime_error_does_not_trigger_retry(self, tmp_path):
        """``RuntimeError`` (non-zero exit) must not trigger retry."""
        from run_magi import run_orchestrator

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            call_counts[agent_name] += 1
            if agent_name == "caspar":
                raise RuntimeError(f"agent {agent_name} exited non-zero")
            return TestSingleShotRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
            assert result["degraded"] is True
            assert call_counts["caspar"] == 1, "subprocess exit errors must not be retried"

    @pytest.mark.asyncio
    async def test_retry_uses_full_timeout_budget(self, tmp_path):
        """Each attempt receives the full ``timeout`` kwarg, not a reduced one.

        Operators configure ``--timeout`` as a per-attempt ceiling. The
        retry must honor the same ceiling; halving it (or consuming the
        first attempt's remaining budget) would introduce silent behavior
        the docs do not promise.
        """
        from run_magi import run_orchestrator
        from validate import ValidationError

        captured_timeouts: dict[str, list[int]] = {
            "melchior": [],
            "balthasar": [],
            "caspar": [],
        }
        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            captured_timeouts[agent_name].append(timeout)
            call_counts[agent_name] += 1
            if agent_name == "caspar" and call_counts[agent_name] == 1:
                raise ValidationError("schema fail, retry please")
            return TestSingleShotRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
        assert captured_timeouts["caspar"] == [300, 300], (
            "retry must be launched with the full per-agent timeout budget"
        )

    @pytest.mark.asyncio
    async def test_retry_injects_validation_error_feedback(self, tmp_path):
        """The retry prompt must carry corrective feedback from the error.

        Contract: the second call to ``launch_agent`` receives a prompt
        that (a) contains the original user prompt and (b) contains the
        ValidationError message so the model can self-correct. The
        feedback block format is implementation-defined but the error
        text must be substring-present.
        """
        from run_magi import run_orchestrator
        from validate import ValidationError

        error_msg = "Agent output missing keys: ['recommendation']"
        captured_prompts: dict[str, list[str]] = {
            "melchior": [],
            "balthasar": [],
            "caspar": [],
        }
        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            captured_prompts[agent_name].append(prompt)
            call_counts[agent_name] += 1
            if agent_name == "caspar" and call_counts[agent_name] == 1:
                raise ValidationError(error_msg)
            return TestSingleShotRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="ORIGINAL-USER-PROMPT-TOKEN",
                output_dir=str(tmp_path),
                timeout=300,
            )

        assert len(captured_prompts["caspar"]) == 2
        first_prompt, retry_prompt = captured_prompts["caspar"]
        assert first_prompt == "ORIGINAL-USER-PROMPT-TOKEN", (
            "first call must receive the untouched user prompt"
        )
        assert "ORIGINAL-USER-PROMPT-TOKEN" in retry_prompt, (
            "retry prompt must preserve the original user prompt"
        )
        assert "recommendation" in retry_prompt, (
            "retry prompt must surface the ValidationError message so the "
            "model can self-correct the specific missing field"
        )

    @pytest.mark.asyncio
    async def test_retry_emits_retrying_display_state(self, tmp_path):
        """A ``retrying`` display state must appear between running and the
        terminal state for the agent that hit ValidationError.

        Other agents must not see a ``retrying`` update.
        """
        from run_magi import run_orchestrator
        from validate import ValidationError

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}
        display_events: list[tuple[str, str]] = []

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            call_counts[agent_name] += 1
            if agent_name == "caspar" and call_counts[agent_name] == 1:
                raise ValidationError("schema fail")
            return TestSingleShotRetry._valid(agent_name)

        def capture_update(display, name, state, log_gate):
            display_events.append((name, state))

        with (
            patch("run_magi.launch_agent", side_effect=mock_launch),
            patch("run_magi._safe_display_update", side_effect=capture_update),
        ):
            await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
                show_status=True,
            )

        caspar_states = [state for name, state in display_events if name == "caspar"]
        assert "retrying" in caspar_states, (
            "caspar must transition through a 'retrying' state before success"
        )
        # Order: running → retrying → success
        running_idx = caspar_states.index("running")
        retrying_idx = caspar_states.index("retrying")
        success_idx = caspar_states.index("success")
        assert running_idx < retrying_idx < success_idx, (
            f"caspar state order violated: {caspar_states}"
        )

        for other in ("melchior", "balthasar"):
            other_states = [state for name, state in display_events if name == other]
            assert "retrying" not in other_states, (
                f"{other} must not emit 'retrying' — only the failing agent retries"
            )


class TestJsonDecodeRetry:
    """2.2.4: retry also fires on ``json.JSONDecodeError`` from parse_agent_output.

    Background: 2.2.0 scoped retry to :class:`ValidationError` only. A
    production ``iter 2 catastrophic failure`` (post-2.2.3) lost two of
    three agents to ``json.JSONDecodeError`` raised inside
    :func:`parse_agent_output.parse_agent_output` BEFORE
    :func:`validate.load_agent_output` could wrap the failure into
    ``ValidationError``. Without retry, both agents were dropped and
    synthesis aborted on the 2-agent minimum.

    2.2.4 widens the retry trigger to ``(ValidationError,
    json.JSONDecodeError)``. ``ValueError`` is **not** added to the
    trigger set: ``ValueError`` is also raised by ``resolve_model`` for
    invalid model short names (where retry is pointless — same input
    yields the same error) and by ``parse_agent_output._extract_text``
    for unrecognized Anthropic CLI output shapes (a structural change
    that needs a parser update, not a retry).

    Telemetry contract is unchanged: the `retried_agents` field
    introduced in 2.2.1 records the agent regardless of whether the
    triggering exception was ValidationError or JSONDecodeError.
    """

    @staticmethod
    def _valid(agent: str) -> dict[str, Any]:
        return {
            "agent": agent,
            "verdict": "approve",
            "confidence": 0.85,
            "summary": f"{agent} OK",
            "reasoning": "Fine",
            "findings": [],
            "recommendation": "Merge",
        }

    @pytest.mark.asyncio
    async def test_json_decode_error_triggers_retry_success(self, tmp_path):
        """First attempt raises json.JSONDecodeError, retry succeeds.

        The orchestrator must retry the agent and emerge with a full
        3-agent consensus, no `degraded` flag, and the agent listed in
        `retried_agents` so downstream telemetry sees the recovery.
        """
        import json as _json

        from run_magi import run_orchestrator

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            call_counts[agent_name] += 1
            if agent_name == "melchior" and call_counts[agent_name] == 1:
                # Simulate the exact failure mode reported in production:
                # parse_agent_output called json.loads on truncated text
                # and json raised JSONDecodeError.
                raise _json.JSONDecodeError("Expecting value", "truncated output...", 142)
            return TestJsonDecodeRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
        assert result.get("degraded") is not True, (
            "JSONDecodeError that recovered on retry must not leave "
            "degraded set — full 3-agent consensus is restored"
        )
        assert len(result["agents"]) == 3
        assert result.get("retried_agents") == ["melchior"], (
            "retried_agents must record the recovery regardless of "
            "whether the triggering exception was ValidationError or "
            "JSONDecodeError"
        )
        assert call_counts["melchior"] == 2

    @pytest.mark.asyncio
    async def test_json_decode_error_retry_also_fails_degrades(self, tmp_path):
        """Both attempts raise json.JSONDecodeError → agent dropped.

        Mirrors the ValidationError "retry-also-fails" path: the agent
        appears in BOTH `retried_agents` (it took the retry path) AND
        `failed_agents` (it ultimately failed). The intersection
        identifies the retry-also-failed cohort downstream tooling
        cares about.
        """
        import json as _json

        from run_magi import run_orchestrator

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            if agent_name == "balthasar":
                raise _json.JSONDecodeError("Unterminated string", "broken", 50)
            return TestJsonDecodeRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
        assert result["degraded"] is True
        assert "balthasar" in result["failed_agents"]
        assert result.get("retried_agents") == ["balthasar"], (
            "retried_agents records every agent that took the retry "
            "path, including those whose retry also failed"
        )

    @pytest.mark.asyncio
    async def test_value_error_from_parse_does_not_retry(self, tmp_path):
        """``ValueError`` is **out of scope** for retry — explicit boundary.

        ``ValueError`` is raised by both ``parse_agent_output`` (for
        unrecognized output shapes) and ``resolve_model`` (for invalid
        model short names). The latter is a configuration error where
        retry is pointless; the former is a structural change that
        needs a parser fix, not a retry. Catching ValueError would
        retry on both paths, masking the configuration bug and wasting
        a subprocess on a structural one.

        This test pins the boundary so a future ``except (ValidationError,
        JSONDecodeError, ValueError)`` change cannot slip past CI.
        """
        from run_magi import run_orchestrator

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            call_counts[agent_name] += 1
            if agent_name == "caspar":
                raise ValueError("Unexpected Claude CLI output type: int")
            return TestJsonDecodeRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
        assert result["degraded"] is True
        assert "caspar" in result["failed_agents"]
        assert call_counts["caspar"] == 1, (
            "ValueError must not trigger retry — that path is reserved "
            "for JSON parse failures, not parser-shape or config errors"
        )
        assert "retried_agents" not in result, (
            "no agent took the retry path, so the field must be omitted "
            "(conditional-presence convention)"
        )


class TestRetryTelemetry:
    """2.2.1: report exposes ``retried_agents`` for downstream telemetry.

    Until 2.2.0 the only auditable signal of retry activity was
    ``degraded=true`` + ``failed_agents`` — i.e. the worst-case where
    the retry also failed. A successful retry was indistinguishable
    from a clean first-attempt run, which is exactly the case
    operators need to size to evaluate the retry budget and count.

    Contract for the new ``retried_agents`` report field:

    * Lists every agent name that hit the retry path, regardless of
      whether the retry recovered or also failed.
    * Sorted alphabetically so the JSON serialisation is byte-stable
      across runs and platforms (cleanly diff-able in audit logs).
    * Conditionally present, mirroring the existing ``degraded`` and
      ``failed_agents`` keys: omitted entirely when no retry fired,
      so 2.2.0 consumers that ignore unknown keys are unaffected.
    * Composes with ``failed_agents`` to give two derived sets:
      ``set(retried_agents) - set(failed_agents)`` is "retry recovered",
      ``set(retried_agents) & set(failed_agents)`` is "retry also failed".
    """

    @staticmethod
    def _valid(agent: str) -> dict[str, Any]:
        return {
            "agent": agent,
            "verdict": "approve",
            "confidence": 0.85,
            "summary": f"{agent} OK",
            "reasoning": "Fine",
            "findings": [],
            "recommendation": "Merge",
        }

    @pytest.mark.asyncio
    async def test_report_lists_retried_agent_when_retry_succeeds(self, tmp_path):
        """Retry-recovered: agent in ``agents`` and in ``retried_agents``."""
        from run_magi import run_orchestrator
        from validate import ValidationError

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            call_counts[agent_name] += 1
            if agent_name == "caspar" and call_counts[agent_name] == 1:
                raise ValidationError("missing keys: ['recommendation']")
            return TestRetryTelemetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
        assert result.get("retried_agents") == ["caspar"], (
            "successful retry must still be recorded in retried_agents"
        )
        assert result.get("degraded") is not True
        assert "failed_agents" not in result, (
            "no failures => failed_agents must be omitted (2.1.x contract preserved)"
        )

    @pytest.mark.asyncio
    async def test_report_lists_retried_agent_when_retry_also_fails(self, tmp_path):
        """Retry-also-failed: agent in retried_agents AND failed_agents."""
        from run_magi import run_orchestrator
        from validate import ValidationError

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            if agent_name == "caspar":
                raise ValidationError("missing keys: ['recommendation']")
            return TestRetryTelemetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
        assert result.get("retried_agents") == ["caspar"]
        assert result.get("degraded") is True
        assert result.get("failed_agents") == ["caspar"]
        # Composability check: the intersection identifies retry-also-failed
        retried = set(result["retried_agents"])
        failed = set(result["failed_agents"])
        assert retried & failed == {"caspar"}

    @pytest.mark.asyncio
    async def test_report_omits_retried_agents_field_when_no_retry(self, tmp_path):
        """Field absent (not empty list) on a clean run.

        Mirrors the conditional-presence convention used by ``degraded``
        and ``failed_agents``: keys are introduced only when their value
        is informative, so 2.2.0 consumers reading these reports never
        see a meaningless ``"retried_agents": []`` they have to filter.
        """
        from run_magi import run_orchestrator

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            return TestRetryTelemetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
        assert "retried_agents" not in result, (
            "retried_agents must be omitted entirely when no agent retried "
            "(do not emit an empty list)"
        )

    @pytest.mark.asyncio
    async def test_report_lists_multiple_retried_agents_sorted(self, tmp_path):
        """Two retries (one recovers, one fails) → both listed, sorted."""
        from run_magi import run_orchestrator
        from validate import ValidationError

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            call_counts[agent_name] += 1
            # melchior recovers on retry; caspar fails twice.
            if agent_name == "melchior" and call_counts[agent_name] == 1:
                raise ValidationError("missing keys for melchior")
            if agent_name == "caspar":
                raise ValidationError("missing keys for caspar")
            return TestRetryTelemetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
        assert result.get("retried_agents") == ["caspar", "melchior"], (
            "retried_agents must list every agent that took the retry path, "
            "sorted alphabetically for byte-stable JSON"
        )
        # caspar failed both attempts; melchior recovered on retry.
        assert result.get("failed_agents") == ["caspar"]
        retried = set(result["retried_agents"])
        failed = set(result["failed_agents"])
        assert retried - failed == {"melchior"}, "retry-recovered set"
        assert retried & failed == {"caspar"}, "retry-also-failed set"


class TestCp1252Resilience:
    """2.2.6: orchestrator must not crash on Windows cp1252 environments.

    Two reproducible crash sites existed under cp1252:

    1. ``print(f"\\u26a0 WARNING: ...", file=sys.stderr)``: the warning sign
       ``\\u26a0`` (⚠) is not in cp1252's codepage (cp1252 covers
       U+0000-U+00FF plus a 0x80-0x9F extension; U+26A0 is outside).
       When MAGI runs as a subprocess (sbtdd's ``subprocess.run`` with
       ``capture_output=True``), ``sys.stderr`` is the locale-encoding
       text wrapper, which on Windows is cp1252. The print encodes
       with ``errors='strict'`` and raises ``UnicodeEncodeError``,
       crashing the orchestrator before the report is written.

    2. ``open(args.input, encoding='utf-8')``: any user input file
       written by Windows tooling with the default cp1252 encoding
       (Notepad, VS Code without explicit BOM, Python ``open()`` on
       Windows without ``encoding=``) raises ``UnicodeDecodeError`` on
       the first byte ≥0x80 that is not a valid UTF-8 start byte.

    These tests pin the contract that both sites stay non-crashing
    after 2.2.6 even when the environment locale is cp1252.
    """

    @pytest.mark.asyncio
    async def test_warning_print_does_not_crash_on_cp1252_stderr(self, tmp_path, monkeypatch):
        """Repro for crash site 1: WARNING about a failed agent must not
        encode-fail when ``sys.stderr`` is a cp1252-strict text stream.

        Pre-fix: the ``\\u26a0`` warning sign in run_orchestrator's
        WARNING messages crashes Python's ``print`` with
        ``UnicodeEncodeError`` on cp1252 locales.

        Post-fix: the message uses ASCII-only markers (``[!]`` instead
        of ``⚠``) so the print survives any encoding the parent process
        chooses for the captured stderr.
        """
        import codecs
        import io

        from run_magi import run_orchestrator

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, model="opus", backend=None
        ):
            if agent_name == "caspar":
                # Use a non-retryable error class so retry does not fire
                # and we go straight to the WARNING-then-degraded path.
                raise RuntimeError("subprocess died for the test")
            return {
                "agent": agent_name,
                "verdict": "approve",
                "confidence": 0.85,
                "summary": f"{agent_name} OK",
                "reasoning": "Fine",
                "findings": [],
                "recommendation": "Merge",
            }

        # Cp1252-strict stream: bytes that fall outside cp1252 will raise
        # UnicodeEncodeError on write. This mirrors what Python gives the
        # orchestrator when it runs as a subprocess on Windows.
        cp1252_buffer = io.BytesIO()
        cp1252_stderr = codecs.getwriter("cp1252")(cp1252_buffer, errors="strict")

        monkeypatch.setattr(sys, "stderr", cp1252_stderr)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            # Must not raise UnicodeEncodeError. Pre-fix: it does.
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
                show_status=False,  # keep stderr writes direct, no buffer shim
            )

        assert result["degraded"] is True
        assert "caspar" in result["failed_agents"]

        # Sanity: the WARNING actually made it through to the cp1252 stream.
        cp1252_stderr.flush()
        emitted = cp1252_buffer.getvalue().decode("cp1252", errors="replace")
        assert "WARNING" in emitted, (
            "the test must actually exercise the WARNING-emitting path; "
            "if this assert fails the mock arrangement is wrong"
        )
        assert "⚠" not in emitted, (
            "the warning sign \\u26a0 must not be emitted to stderr — "
            "it is the exact codepoint that breaks cp1252 environments"
        )

    def test_input_file_read_handles_cp1252_bytes(self, tmp_path):
        """Repro for crash site 2: a cp1252-encoded input file must not
        crash MAGI's input loader.

        Pre-fix: the ``open(args.input, encoding='utf-8')`` at the top of
        ``main()`` raises ``UnicodeDecodeError`` on any byte ≥0x80 that
        is not a valid UTF-8 start byte (e.g., the cp1252 em dash 0x97).

        Post-fix: the read uses ``errors='replace'`` so the file content
        is decoded with replacement characters in place of invalid bytes,
        and MAGI continues with whatever readable content remains.
        """
        from run_magi import _load_input_content

        cp1252_file = tmp_path / "cp1252-input.txt"
        # b'Hello \x97 world' — \x97 is the cp1252 em dash, NOT a valid
        # UTF-8 start byte. Reading this with strict UTF-8 raises.
        cp1252_file.write_bytes("Hello — world".encode("cp1252"))

        # Must not raise UnicodeDecodeError. Pre-fix: it does.
        content, label = _load_input_content(str(cp1252_file))

        assert "Hello" in content, "ASCII portions of the file must survive"
        assert "world" in content, "ASCII portions on either side of the bad byte"
        assert label == f"File: {cp1252_file}"

    def test_load_input_content_treats_inline_text_as_string(self):
        """``_load_input_content`` returns inline text untouched when the
        argument is not a file path. This pins the boundary between
        file-read (cp1252-tolerant) and inline-text (no decode) paths.
        """
        from run_magi import _load_input_content

        content, label = _load_input_content("inline analysis text — not a path")
        assert content == "inline analysis text — not a path"
        assert label == "Inline input"


class TestUtf8ConsoleReconfigure:
    """2.2.7: structural fix for the Windows encode-side cp1252 problem.

    The 2.2.6 hotfix removed the four ``\\u26a0`` warning signs that
    were the immediate crash trigger, but ``sys.stdout`` /
    ``sys.stderr`` themselves were left bound to the cp1252 locale
    wrapper Python gives child processes on Windows. Any **future**
    non-cp1252 codepoint emitted through ``print`` — a finding title
    that the LLM rolls with ``→``, ``≥``, curly quotes, or
    any character outside cp1252's 256-codepoint range — would
    re-introduce the same ``UnicodeEncodeError`` crash.

    The fix is a single helper, ``_enable_utf8_console_io()``, called
    at the top of ``main()``. On Windows it switches both standard
    streams to UTF-8 with ``errors="backslashreplace"``. On every
    other platform it is a no-op so POSIX shells (which already
    default to UTF-8) keep their existing byte contract.

    These tests pin:

    * the helper exists and is exported from ``run_magi``,
    * win32 reconfigures both streams to ``utf-8`` /
      ``backslashreplace``,
    * non-win32 platforms are untouched,
    * streams lacking ``reconfigure`` (e.g., a logger that wrapped
      stderr) are skipped silently rather than crashing,
    * after the helper runs, a print of a non-cp1252 codepoint
      survives without raising — the end-to-end guarantee the
      structural fix exists to provide.
    """

    def test_helper_is_exported(self):
        """The helper must be importable from run_magi — call sites
        live in main() and tests both depend on the public name.
        """
        from run_magi import _enable_utf8_console_io

        assert callable(_enable_utf8_console_io)

    def test_reconfigures_streams_on_win32(self, monkeypatch):
        """On win32, both stdout and stderr are reconfigured to utf-8
        with the backslashreplace error policy. ``backslashreplace``
        is non-negotiable: ``strict`` would re-introduce the crash,
        ``ignore`` would silently drop diagnostic content, and
        ``replace`` substitutes U+FFFD which is itself non-ASCII and
        thus pointless under cp1252. ``backslashreplace`` always
        produces ASCII output (``\\u26a0``) so the printed bytes are
        guaranteed encodable in any codepage.
        """
        import io

        from run_magi import _enable_utf8_console_io

        monkeypatch.setattr(sys, "platform", "win32")
        fake_stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
        fake_stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
        monkeypatch.setattr(sys, "stdout", fake_stdout)
        monkeypatch.setattr(sys, "stderr", fake_stderr)

        _enable_utf8_console_io()

        assert sys.stdout.encoding.lower() == "utf-8"
        assert sys.stdout.errors == "backslashreplace"
        assert sys.stderr.encoding.lower() == "utf-8"
        assert sys.stderr.errors == "backslashreplace"

    def test_noop_on_non_win32(self, monkeypatch):
        """On Linux / macOS the function is a no-op. POSIX shells
        already default to UTF-8 and changing the encoding could
        break downstream tooling that captured stdout assuming the
        locale-derived bytes contract.

        Compares via :func:`codecs.lookup` rather than raw string
        equality so the test is stable across Python versions
        (Python <=3.13 reports ``iso8859-1``, 3.14+ reports
        ``latin-1`` — both are aliases of the same codec).
        """
        import codecs
        import io

        from run_magi import _enable_utf8_console_io

        canonical_latin1 = codecs.lookup("latin-1").name

        monkeypatch.setattr(sys, "platform", "linux")
        fake_stdout = io.TextIOWrapper(io.BytesIO(), encoding="latin-1", errors="strict")
        fake_stderr = io.TextIOWrapper(io.BytesIO(), encoding="latin-1", errors="strict")
        monkeypatch.setattr(sys, "stdout", fake_stdout)
        monkeypatch.setattr(sys, "stderr", fake_stderr)

        _enable_utf8_console_io()

        # Untouched: encoding and errors policy still match the
        # pre-call values.
        assert codecs.lookup(sys.stdout.encoding).name == canonical_latin1
        assert sys.stdout.errors == "strict"
        assert codecs.lookup(sys.stderr.encoding).name == canonical_latin1
        assert sys.stderr.errors == "strict"

    def test_streams_without_reconfigure_method_are_skipped(self, monkeypatch):
        """If a parent process replaced ``sys.stderr`` with a custom
        object that lacks ``reconfigure`` — a logger sink, a buffer
        proxy, a pytest capture wrapper — the helper must not crash.

        The reconfigure method is a TextIOWrapper feature; nothing in
        Python's standard library guarantees every stdout-like object
        has it. Skipping silently is the right behavior because
        custom streams have already chosen their encoding contract;
        forcing UTF-8 would either fail or violate that contract.
        """

        class FakeStreamWithoutReconfigure:
            encoding = "ascii"
            errors = "strict"

            def write(self, _data):
                pass

            def flush(self):
                pass

        from run_magi import _enable_utf8_console_io

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(sys, "stdout", FakeStreamWithoutReconfigure())
        monkeypatch.setattr(sys, "stderr", FakeStreamWithoutReconfigure())

        # Must not raise AttributeError or any other exception.
        _enable_utf8_console_io()

    def test_print_of_non_cp1252_codepoint_survives_after_reconfigure(self, monkeypatch):
        """End-to-end guarantee: after the helper runs, ``print`` of a
        codepoint that is **not** in cp1252 (e.g., U+2192 right arrow,
        U+2265 greater-or-equal, U+2018 left single quote) does not
        raise UnicodeEncodeError, even though the underlying byte
        buffer was originally created as cp1252-strict.

        This is the test that would have caught the entire 2.2.6
        whack-a-mole pattern: it does not care which specific
        codepoint the LLM emits, only that the output path tolerates
        anything Unicode permits.
        """
        import io

        from run_magi import _enable_utf8_console_io

        monkeypatch.setattr(sys, "platform", "win32")
        stderr_buffer = io.BytesIO()
        fake_stderr = io.TextIOWrapper(stderr_buffer, encoding="cp1252", errors="strict")
        monkeypatch.setattr(sys, "stderr", fake_stderr)

        _enable_utf8_console_io()

        # All three are codepoints outside cp1252's range — pre-fix
        # any one of these would crash a strict cp1252 stream.
        for codepoint in ("→", "≥", "‘"):
            print(f"finding title: {codepoint}", file=sys.stderr)
        sys.stderr.flush()

        emitted = stderr_buffer.getvalue().decode("utf-8", errors="replace")
        assert "→" in emitted
        assert "≥" in emitted
        assert "‘" in emitted

    def test_main_invokes_reconfigure_before_any_print(self, monkeypatch):
        """``main()`` must call ``_enable_utf8_console_io`` *before*
        any ``print``, ``sys.exit``, or other output operation. If the
        call moved later (e.g., after the input-file load or after the
        ``claude`` PATH check), a crash on those code paths would
        re-introduce the original failure mode.

        Verified by stubbing the helper to raise a sentinel exception
        and a second function (``parse_args``) to raise a different
        sentinel. Whichever exception escapes ``main()`` was called
        first. We expect the helper's exception, proving ``main()``
        invoked the helper before any other output-bearing code.
        """

        class HelperCalledFirst(RuntimeError):
            pass

        class ParseArgsCalledFirst(RuntimeError):
            pass

        def fake_enable():
            raise HelperCalledFirst("helper ran first")

        def fake_parse_args():
            raise ParseArgsCalledFirst("parse_args ran first")

        monkeypatch.setattr("run_magi._enable_utf8_console_io", fake_enable)
        monkeypatch.setattr("run_magi.parse_args", fake_parse_args)

        from run_magi import main

        with pytest.raises(HelperCalledFirst):
            main()


class TestInputLabelBannerRegression:
    """Pin: the init banner in ``run_magi.main`` renders ``input_label``.

    Source-level grep — brittle to a banner refactor but cheap to update.
    Catches the regression where a future cleanup deletes the operator
    visibility into which file the user passed. Per Balthasar MAGI
    finding 2026-05-16, sanitize spec §7.
    """

    def test_main_banner_renders_input_label(self):
        from pathlib import Path

        src = Path(__file__).parent.parent / "skills" / "magi" / "scripts" / "run_magi.py"
        contents = src.read_text(encoding="utf-8")
        # The exact f-string used in main() to print the input source.
        # Changing the format is fine — update this assertion to match.
        assert 'f"|  Input: {input_label}"' in contents, (
            "Init banner must render input_label; see sanitize spec §7 and "
            "the Balthasar 2026-05-16 finding. If you intentionally refactored "
            "the banner, update this pin to match the new rendering."
        )


class TestEnrichIntegration:
    """Task 7: fail-safe code-review enrichment wired into run_magi."""

    def test_args_defaults(self):
        import run_magi

        a = run_magi.parse_args(["code-review", "in.txt"])
        assert a.base == "main" and a.enrich is True and a.enrich_max_chars == 512_000

    def test_no_enrich(self):
        import run_magi

        assert run_magi.parse_args(["code-review", "x", "--no-enrich"]).enrich is False

    def test_codereview_calls_lib(self, monkeypatch):
        import run_magi

        seen = {}

        def fake(c, **kw):
            seen.update(kw)
            return c + "\n[E]", "enriched: 1 file(s)"

        monkeypatch.setattr(run_magi, "enrich_code_review_context", fake)
        out, note = run_magi._maybe_enrich(
            "code-review", "D", base_ref="main", enrich=True, max_chars=99
        )
        assert "[E]" in out and seen["max_chars"] == 99 and note

    def test_passthrough_design(self, monkeypatch):
        import run_magi

        monkeypatch.setattr(run_magi, "enrich_code_review_context", lambda *a, **k: ("X", "x"))
        out, note = run_magi._maybe_enrich(
            "design", "D", base_ref="main", enrich=True, max_chars=99
        )
        assert out == "D" and note is None

    def test_boundary_failsafe(self, monkeypatch):
        import run_magi

        monkeypatch.setattr(
            run_magi,
            "enrich_code_review_context",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
        )
        out, note = run_magi._maybe_enrich(
            "code-review", "D", base_ref="main", enrich=True, max_chars=99
        )
        assert out == "D" and "error" in note.lower()


class TestProjectRootResolution:
    """BDD-13: git toplevel with cwd fallback."""

    def test_uses_git_toplevel_when_available(self, monkeypatch):
        import run_magi

        class FakeCompleted:
            returncode = 0
            stdout = "/repo/root\n"

        monkeypatch.setattr(run_magi.subprocess, "run", lambda *a, **k: FakeCompleted())
        assert run_magi._resolve_project_root() == "/repo/root"

    def test_falls_back_to_cwd_when_not_a_repo(self, monkeypatch):
        import os

        import run_magi

        class FakeCompleted:
            returncode = 128
            stdout = ""

        monkeypatch.setattr(run_magi.subprocess, "run", lambda *a, **k: FakeCompleted())
        assert run_magi._resolve_project_root() == os.path.realpath(os.getcwd())

    def test_falls_back_to_cwd_when_git_missing(self, monkeypatch):
        import os

        import run_magi

        def boom(*a, **k):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(run_magi.subprocess, "run", boom)
        assert run_magi._resolve_project_root() == os.path.realpath(os.getcwd())


class TestMainLockWiring:
    """BDD-8/9/14: lock written for temp runs, removed on success, bypassed
    for explicit --output-dir."""

    def _patch_run(self, monkeypatch, *, output_dir_arg=None):
        """Stub everything around main() except the temp/lock wiring."""
        import run_magi

        monkeypatch.setattr(run_magi, "_enable_utf8_console_io", lambda: None)
        monkeypatch.setattr(run_magi.shutil, "which", lambda name: "claude")
        monkeypatch.setattr(run_magi, "build_user_prompt", lambda mode, content: "PROMPT")
        monkeypatch.setattr(run_magi, "_load_input_content", lambda arg: ("BODY", "Inline input"))
        monkeypatch.setattr(run_magi, "_maybe_enrich", lambda *a, **k: ("BODY", None))
        monkeypatch.setattr(
            run_magi,
            "format_report",
            lambda agents, consensus, **kw: "REPORT",
        )

        async def fake_orch(*a, **k):
            return {"agents": [], "consensus": {}}

        monkeypatch.setattr(run_magi, "run_orchestrator", fake_orch)

    def test_lock_written_then_removed_on_success(self, tmp_path, monkeypatch):
        import run_lock
        import run_magi

        self._patch_run(monkeypatch)
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)

        cleanup_calls = {}

        def fake_cleanup(keep, run_root=None):
            cleanup_calls["keep"] = keep
            cleanup_calls["root"] = run_root

        monkeypatch.setattr(run_magi, "cleanup_old_runs", fake_cleanup)

        seen = {"lock_present_during_run": None}
        created = {}

        def fake_create(output_dir, run_root=None):
            d = tmp_path / "magi-run-xyz"
            d.mkdir()
            created["dir"] = str(d)
            return str(d)

        monkeypatch.setattr(run_magi, "create_output_dir", fake_create)

        async def fake_orch(*a, **k):
            # Lock must exist while the orchestrator runs.
            seen["lock_present_during_run"] = os.path.exists(
                os.path.join(created["dir"], run_lock.LOCK_FILENAME)
            )
            return {"agents": [], "consensus": {}}

        monkeypatch.setattr(run_magi, "run_orchestrator", fake_orch)
        monkeypatch.setattr(sys, "argv", ["run_magi.py", "design", "hello"])

        run_magi.main()

        assert seen["lock_present_during_run"] is True
        assert not os.path.exists(os.path.join(created["dir"], run_lock.LOCK_FILENAME)), (
            "Lock must be removed after a successful run"
        )
        # Off-by-one (Bal/Cas): default keep_runs=5 -> cleanup gets 4, namespaced root.
        assert cleanup_calls["keep"] == 4
        assert cleanup_calls["root"] == str(tmp_path)

    def test_failure_path_removes_dir_and_lock(self, tmp_path, monkeypatch):
        """BDD-10: when the orchestrator raises, the run dir AND its lock go."""
        import run_lock
        import run_magi

        self._patch_run(monkeypatch)
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)
        monkeypatch.setattr(run_magi, "cleanup_old_runs", lambda keep, run_root=None: None)

        created = {}

        def fake_create(output_dir, run_root=None):
            d = tmp_path / "magi-run-fail"
            d.mkdir()
            created["dir"] = str(d)
            return str(d)

        monkeypatch.setattr(run_magi, "create_output_dir", fake_create)

        async def boom_orch(*a, **k):
            raise RuntimeError("agents failed")

        monkeypatch.setattr(run_magi, "run_orchestrator", boom_orch)
        monkeypatch.setattr(sys, "argv", ["run_magi.py", "design", "hello"])

        with pytest.raises(RuntimeError):
            run_magi.main()

        assert not os.path.exists(created["dir"]), "Failed run's temp dir must be removed"
        assert not os.path.exists(os.path.join(created["dir"], run_lock.LOCK_FILENAME))

    def test_cleanup_receives_keep_runs_minus_one_for_keep_1(self, tmp_path, monkeypatch):
        """Boundary: --keep-runs 1 -> cleanup_old_runs(0) (wipe all non-live)."""
        import run_magi

        self._patch_run(monkeypatch)
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)

        cleanup_calls = {}
        monkeypatch.setattr(
            run_magi,
            "cleanup_old_runs",
            lambda keep, run_root=None: cleanup_calls.update(keep=keep),
        )
        monkeypatch.setattr(
            run_magi, "create_output_dir", lambda output_dir, run_root=None: str(tmp_path)
        )
        monkeypatch.setattr(run_magi, "write_lock", lambda d, max_age_seconds=None: None)
        monkeypatch.setattr(run_magi, "remove_lock", lambda d: None)
        monkeypatch.setattr(sys, "argv", ["run_magi.py", "design", "hello", "--keep-runs", "1"])

        run_magi.main()
        assert cleanup_calls["keep"] == 0

    def test_write_lock_receives_timeout_derived_bound(self, tmp_path, monkeypatch):
        """Cas iter-3: main() writes the staleness_bound_for_timeout(--timeout)."""
        import run_magi
        from run_lock import staleness_bound_for_timeout

        self._patch_run(monkeypatch)
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)
        monkeypatch.setattr(run_magi, "cleanup_old_runs", lambda keep, run_root=None: None)
        monkeypatch.setattr(
            run_magi, "create_output_dir", lambda output_dir, run_root=None: str(tmp_path)
        )
        monkeypatch.setattr(run_magi, "remove_lock", lambda d: None)

        captured = {}
        monkeypatch.setattr(
            run_magi,
            "write_lock",
            lambda d, max_age_seconds=None: captured.update(bound=max_age_seconds),
        )
        monkeypatch.setattr(sys, "argv", ["run_magi.py", "design", "hello", "--timeout", "14400"])

        run_magi.main()
        assert captured["bound"] == staleness_bound_for_timeout(14400)

    def test_explicit_output_dir_writes_no_lock(self, tmp_path, monkeypatch):
        import run_lock
        import run_magi

        self._patch_run(monkeypatch)
        out = tmp_path / "explicit"
        monkeypatch.setattr(
            sys, "argv", ["run_magi.py", "design", "hello", "--output-dir", str(out)]
        )

        run_magi.main()

        assert not (out / run_lock.LOCK_FILENAME).exists()


class TestNamespaceIntegration:
    """Cas finding: non-stubbed composition of project_run_root +
    create_output_dir + write_lock + cleanup_old_runs (real filesystem,
    no mocks of the units under test)."""

    def test_live_namespaced_dir_survives_peer_cleanup(self, tmp_path):
        from run_lock import LOCK_FILENAME, write_lock
        from temp_dirs import cleanup_old_runs, create_output_dir, project_run_root

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            root = project_run_root(str(tmp_path / "projA"))
            live = create_output_dir(None, root)
            write_lock(live)  # current (alive) PID
            # A peer session prunes as aggressively as possible.
            cleanup_old_runs(0, root)

        assert os.path.isdir(live), "Live namespaced dir must survive peer cleanup"
        assert os.path.exists(os.path.join(live, LOCK_FILENAME))

    def test_cross_project_cleanup_does_not_touch_other_project(self, tmp_path):
        """BDD-1 end-to-end: project B's cleanup never sees project A's dirs."""
        from temp_dirs import cleanup_old_runs, create_output_dir, project_run_root

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            root_a = project_run_root(str(tmp_path / "projA"))
            a_dir = create_output_dir(None, root_a)  # no lock -> eligible if scanned
            root_b = project_run_root(str(tmp_path / "projB"))
            cleanup_old_runs(0, root_b)  # wipe-all within B's namespace

        assert os.path.isdir(a_dir), "Project A's dir must be untouched by B's cleanup"


def _guard_agent(findings):
    """Build a minimal valid agent dict carrying the given findings."""
    return {
        "agent": "melchior",
        "verdict": "approve",
        "confidence": 0.9,
        "summary": "s",
        "reasoning": "r",
        "recommendation": "rec",
        "findings": findings,
    }


class TestFindingGuardWiring:
    """v3.0.0 Block A: code-review applies the diff guard per agent before
    consensus; design/analysis do not; cost is aggregated in all modes."""

    def test_resolve_diff_for_guard_returns_files_and_ranges(self):
        import run_magi

        diff = (
            "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
            "@@ -1,2 +1,3 @@\n ctx\n+added\n ctx2\n"
        )
        files, ranges = run_magi._diff_files_and_ranges(diff)
        assert files == {"x.py"} and 2 in ranges["x.py"]

    def test_diff_files_and_ranges_failsafe_returns_empty(self, monkeypatch):
        """A malformed-diff failure degrades to empty, never raises (R10)."""
        import run_magi

        monkeypatch.setattr(
            run_magi,
            "parse_diff_ranges",
            lambda diff: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        files, ranges = run_magi._diff_files_and_ranges("anything")
        assert files == set() and ranges == {}

    def test_guard_applied_only_in_code_review(self):
        import run_magi

        agents = [
            {
                "agent": "melchior",
                "verdict": "approve",
                "confidence": 0.9,
                "summary": "s",
                "reasoning": "r",
                "recommendation": "rec",
                "findings": [
                    {
                        "severity": "warning",
                        "title": "t",
                        "detail": "d",
                        "file": "ghost.py",
                        "line": 1,
                        "category": "other",
                    }
                ],
            }
        ]
        files, ranges = {"x.py"}, {"x.py": {2}}
        cr = run_magi._apply_finding_guard(agents, "code-review", files, ranges)
        assert cr[0]["findings"] == []
        dz = run_magi._apply_finding_guard(agents, "design", files, ranges)
        assert len(dz[0]["findings"]) == 1

    def test_guard_noop_when_no_diff(self):
        """Empty file-set (no diff resolved) -> guard is a no-op even in code-review."""
        import run_magi

        agents = [
            _guard_agent(
                [
                    {
                        "severity": "warning",
                        "title": "t",
                        "detail": "d",
                        "file": "ghost.py",
                        "line": 1,
                        "category": "other",
                    }
                ]
            )
        ]
        out = run_magi._apply_finding_guard(agents, "code-review", set(), {})
        assert len(out[0]["findings"]) == 1

    def test_guard_logs_dropped_finding_titles(self, capsys):
        """FIX 3a: when a finding is dropped, its title must appear in the
        [guard] stderr line so operators can identify false-drops."""
        import run_magi

        agents = [
            _guard_agent(
                [
                    {
                        "severity": "critical",
                        "title": "Null deref in parser",
                        "detail": "d",
                        "file": "ghost.py",
                        "line": 5,
                        "category": "null-deref",
                    },
                    {
                        "severity": "warning",
                        "title": "Real finding",
                        "detail": "d2",
                        "file": "x.py",
                        "line": 2,
                        "category": "other",
                    },
                ]
            )
        ]
        run_magi._apply_finding_guard(agents, "code-review", {"x.py"}, {"x.py": {2}})
        captured = capsys.readouterr()
        assert "Null deref in parser" in captured.err, (
            "dropped finding title must appear in [guard] stderr line"
        )
        assert "Real finding" not in captured.err, (
            "kept finding title must NOT appear in the dropped-titles list"
        )

    def test_guard_dropped_titles_excludes_annotated_findings(self, capsys):
        """BUG 1: annotated (soft-annotated, KEPT) findings must NOT appear in the
        [guard] dropped-titles list; only hard-dropped findings must be listed."""
        import run_magi

        agents = [
            _guard_agent(
                [
                    {
                        "severity": "critical",
                        "title": "Fabricated ghost finding",
                        "detail": "d",
                        "file": "ghost.py",
                        "line": 5,
                        "category": "null-deref",
                    },
                    {
                        "severity": "warning",
                        "title": "Line outside range",
                        "detail": "d2",
                        "file": "x.py",
                        "line": 999,
                        "category": "other",
                    },
                ]
            )
        ]
        # x.py is in the diff (ranges {2}), ghost.py is not.
        # ghost.py -> hard-dropped (dropped=1); x.py line 999 -> soft-annotated (annotated=1).
        run_magi._apply_finding_guard(agents, "code-review", {"x.py"}, {"x.py": {2}})
        captured = capsys.readouterr()
        # The hard-dropped finding's title must appear.
        assert "Fabricated ghost finding" in captured.err, (
            "hard-dropped finding title must appear in [guard] dropped_titles"
        )
        # The soft-annotated finding's title must NOT appear in dropped_titles.
        assert "Line outside range" not in captured.err, (
            "annotated (KEPT) finding title must NOT be listed as dropped"
        )
        # Counts must be: dropped 1, annotated 1.
        assert "dropped 1" in captured.err, "stderr must report dropped 1"
        assert "annotated 1" in captured.err, "stderr must report annotated 1"

    def test_guard_active_signal_with_diff(self, tmp_path, monkeypatch):
        """FIX 3b: code-review with a resolvable diff emits '[guard] active: N file(s)'."""
        import run_magi

        monkeypatch.setattr(run_magi, "_enable_utf8_console_io", lambda: None)
        monkeypatch.setattr(run_magi.shutil, "which", lambda name: "claude")
        monkeypatch.setattr(run_magi, "build_user_prompt", lambda mode, content: "PROMPT")
        monkeypatch.setattr(run_magi, "_load_input_content", lambda arg: ("BODY", "Inline input"))
        monkeypatch.setattr(run_magi, "_maybe_enrich", lambda *a, **k: ("BODY", None))
        monkeypatch.setattr(run_magi, "format_report", lambda agents, consensus, **kw: "REPORT")
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)
        monkeypatch.setattr(run_magi, "cleanup_old_runs", lambda keep, run_root=None: None)
        monkeypatch.setattr(run_magi, "write_lock", lambda d, max_age_seconds=None: None)
        monkeypatch.setattr(run_magi, "remove_lock", lambda d: None)
        monkeypatch.setattr(
            run_magi,
            "aggregate_cost",
            lambda output_dir, agents: {"per_agent": {}, "total_usd": 1.0},
        )
        # Return non-empty files/ranges so the guard reports "active".
        monkeypatch.setattr(
            run_magi,
            "_diff_files_and_ranges",
            lambda diff: ({"x.py"}, {"x.py": {1}}),
        )

        def fake_create(output_dir: object, run_root: object = None) -> str:
            d = tmp_path / "magi-run-active"
            d.mkdir(exist_ok=True)
            return str(d)

        monkeypatch.setattr(run_magi, "create_output_dir", fake_create)

        async def fake_orch(*a: object, **k: object) -> dict[str, Any]:
            return {"agents": [], "consensus": {}}

        monkeypatch.setattr(run_magi, "run_orchestrator", fake_orch)
        monkeypatch.setattr(sys, "argv", ["run_magi.py", "code-review", "hello"])

        import io

        buf = io.StringIO()
        with monkeypatch.context() as mp:
            mp.setattr(sys, "stderr", buf)
            run_magi.main()
        assert "[guard] active:" in buf.getvalue(), (
            "code-review with diff must emit '[guard] active: N file(s)' to stderr"
        )

    def test_guard_skipped_signal_when_no_diff(self, tmp_path, monkeypatch):
        """FIX 3b: code-review without a resolvable diff emits '[guard] skipped: no resolvable diff'."""
        import run_magi

        monkeypatch.setattr(run_magi, "_enable_utf8_console_io", lambda: None)
        monkeypatch.setattr(run_magi.shutil, "which", lambda name: "claude")
        monkeypatch.setattr(run_magi, "build_user_prompt", lambda mode, content: "PROMPT")
        monkeypatch.setattr(run_magi, "_load_input_content", lambda arg: ("BODY", "Inline input"))
        monkeypatch.setattr(run_magi, "_maybe_enrich", lambda *a, **k: ("BODY", None))
        monkeypatch.setattr(run_magi, "format_report", lambda agents, consensus, **kw: "REPORT")
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)
        monkeypatch.setattr(run_magi, "cleanup_old_runs", lambda keep, run_root=None: None)
        monkeypatch.setattr(run_magi, "write_lock", lambda d, max_age_seconds=None: None)
        monkeypatch.setattr(run_magi, "remove_lock", lambda d: None)
        monkeypatch.setattr(
            run_magi,
            "aggregate_cost",
            lambda output_dir, agents: {"per_agent": {}, "total_usd": 1.0},
        )
        # Empty files -> no diff resolved.
        monkeypatch.setattr(
            run_magi,
            "_diff_files_and_ranges",
            lambda diff: (set(), {}),
        )

        def fake_create(output_dir: object, run_root: object = None) -> str:
            d = tmp_path / "magi-run-skipped"
            d.mkdir(exist_ok=True)
            return str(d)

        monkeypatch.setattr(run_magi, "create_output_dir", fake_create)

        async def fake_orch(*a: object, **k: object) -> dict[str, Any]:
            return {"agents": [], "consensus": {}}

        monkeypatch.setattr(run_magi, "run_orchestrator", fake_orch)
        monkeypatch.setattr(sys, "argv", ["run_magi.py", "code-review", "hello"])

        import io

        buf = io.StringIO()
        with monkeypatch.context() as mp:
            mp.setattr(sys, "stderr", buf)
            run_magi.main()
        assert "[guard] skipped:" in buf.getvalue(), (
            "code-review without diff must emit '[guard] skipped: no resolvable diff' to stderr"
        )

    def test_cost_block_in_saved_report(self, tmp_path, monkeypatch):
        """BDD-13 wiring half: magi-report.json on disk carries a cost block."""
        import run_magi

        # Mirror TestMainLockWiring._patch_run: stub everything but the wiring
        # under test (cost aggregation into the saved report).
        monkeypatch.setattr(run_magi, "_enable_utf8_console_io", lambda: None)
        monkeypatch.setattr(run_magi.shutil, "which", lambda name: "claude")
        monkeypatch.setattr(run_magi, "build_user_prompt", lambda mode, content: "PROMPT")
        monkeypatch.setattr(run_magi, "_load_input_content", lambda arg: ("BODY", "Inline input"))
        monkeypatch.setattr(run_magi, "_maybe_enrich", lambda *a, **k: ("BODY", None))
        monkeypatch.setattr(run_magi, "format_report", lambda agents, consensus, **kw: "REPORT")
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)
        monkeypatch.setattr(run_magi, "cleanup_old_runs", lambda keep, run_root=None: None)
        monkeypatch.setattr(run_magi, "write_lock", lambda d, max_age_seconds=None: None)
        monkeypatch.setattr(run_magi, "remove_lock", lambda d: None)

        created = {}

        def fake_create(output_dir, run_root=None):
            d = tmp_path / "magi-run-cost"
            d.mkdir()
            created["dir"] = str(d)
            return str(d)

        monkeypatch.setattr(run_magi, "create_output_dir", fake_create)

        async def fake_orch(*a, **k):
            return {"agents": [_guard_agent([])], "consensus": {}}

        monkeypatch.setattr(run_magi, "run_orchestrator", fake_orch)
        # Stub cost aggregation so the test does not depend on raw-envelope files.
        monkeypatch.setattr(
            run_magi,
            "aggregate_cost",
            lambda output_dir, agents: {"per_agent": {"melchior": 0.25}, "total_usd": 0.25},
        )
        monkeypatch.setattr(sys, "argv", ["run_magi.py", "design", "hello"])

        run_magi.main()

        report_path = os.path.join(created["dir"], "magi-report.json")
        import json

        with open(report_path, encoding="utf-8") as fh:
            saved = json.load(fh)
        assert "cost" in saved
        assert saved["cost"]["total_usd"] == 0.25
        assert saved["cost"]["per_agent"] == {"melchior": 0.25}

    def test_score_invariant_under_fabricated_finding(self):
        """BDD-14: a finding the guard would drop does not change the consensus
        verdict/score/label. The guard filters findings, never the score."""
        from synthesize import determine_consensus

        clean = [
            _guard_agent([]),
            {
                "agent": "balthasar",
                "verdict": "reject",
                "confidence": 0.8,
                "summary": "s2",
                "reasoning": "r2",
                "recommendation": "rec2",
                "findings": [],
            },
        ]
        fabricated = [
            _guard_agent(
                [
                    {
                        "severity": "critical",
                        "title": "ghost",
                        "detail": "d",
                        "file": "ghost.py",
                        "line": 1,
                        "category": "other",
                    }
                ]
            ),
            {
                "agent": "balthasar",
                "verdict": "reject",
                "confidence": 0.8,
                "summary": "s2",
                "reasoning": "r2",
                "recommendation": "rec2",
                "findings": [],
            },
        ]
        c1 = determine_consensus(clean)
        c2 = determine_consensus(fabricated)
        assert c1["consensus"] == c2["consensus"]
        assert c1["consensus_verdict"] == c2["consensus_verdict"]
        assert c1["confidence"] == c2["confidence"]

    def test_shared_diff_source_feeds_enrichment_and_guard(self, tmp_path, monkeypatch):
        """A2: under code-review, main() resolves the diff ONCE and the SAME
        value flows to both the enrichment path and the finding guard.

        This drives the REAL ``_maybe_enrich`` + enrichment path (only the
        orchestrator/temp/lock are stubbed). ``resolve_diff`` is monkeypatched
        on BOTH the ``run_magi`` namespace (where ``main`` resolves it) and the
        ``review_context`` namespace (where ``_enrich`` would re-resolve it) so
        a single shared counter sees every resolution attempt. With A2 realized,
        ``main`` resolves once and threads that value into enrichment, so the
        counter is exactly 1; if enrichment re-resolves internally the counter
        would read 2 (the bug this test pins shut)."""
        import review_context
        import run_magi

        sentinel = "SENTINEL-DIFF-VALUE"
        resolve_calls = {"n": 0}

        def fake_resolve(input_content, repo_root, base_ref):
            resolve_calls["n"] += 1
            return sentinel

        # resolve_diff is referenced both in run_magi's namespace (main()
        # resolves it once) and in review_context's namespace (_enrich would
        # re-resolve it if it ignored the threaded value). Patch BOTH so the
        # single shared counter catches a double-resolution.
        monkeypatch.setattr(run_magi, "resolve_diff", fake_resolve)
        monkeypatch.setattr(review_context, "resolve_diff", fake_resolve)

        # Force the enrichment git gates open deterministically (independent of
        # the test runner's own working-tree state) so _enrich proceeds to the
        # point where the bug would re-resolve the diff.
        monkeypatch.setattr(review_context, "_git_toplevel", lambda start: str(tmp_path))
        monkeypatch.setattr(review_context, "_tree_is_clean", lambda root: True)

        seen = {"enrich_diff": None, "guard_diff": None}

        # Spy on enrich_code_review_context's diff kwarg by WRAPPING the real
        # function (not replacing it), so the REAL _maybe_enrich -> _enrich path
        # runs. With the bug, _enrich re-invokes review_context.resolve_diff
        # (the shared counter then reads 2); with A2 realized, the wrapper sees
        # the diff threaded from main() and _enrich consumes it (counter == 1).
        real_enrich = review_context.enrich_code_review_context

        def spy_enrich_lib(content, **kwargs):
            seen["enrich_diff"] = kwargs.get("diff")
            return real_enrich(content, **kwargs)

        monkeypatch.setattr(run_magi, "enrich_code_review_context", spy_enrich_lib)

        def fake_files_and_ranges(diff):
            seen["guard_diff"] = diff
            return {"x.py"}, {"x.py": {1}}

        monkeypatch.setattr(run_magi, "_diff_files_and_ranges", fake_files_and_ranges)

        # Stub the rest of main()'s wiring.
        monkeypatch.setattr(run_magi, "_enable_utf8_console_io", lambda: None)
        monkeypatch.setattr(run_magi.shutil, "which", lambda name: "claude")
        monkeypatch.setattr(run_magi, "build_user_prompt", lambda mode, content: "PROMPT")
        monkeypatch.setattr(run_magi, "_load_input_content", lambda arg: ("BODY", "Inline input"))
        monkeypatch.setattr(run_magi, "format_report", lambda agents, consensus, **kw: "REPORT")
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)
        monkeypatch.setattr(run_magi, "cleanup_old_runs", lambda keep, run_root=None: None)
        monkeypatch.setattr(run_magi, "write_lock", lambda d, max_age_seconds=None: None)
        monkeypatch.setattr(run_magi, "remove_lock", lambda d: None)
        monkeypatch.setattr(
            run_magi,
            "aggregate_cost",
            lambda output_dir, agents: {"per_agent": {}, "total_usd": 0.0},
        )

        def fake_create(output_dir, run_root=None):
            d = tmp_path / "magi-run-shared"
            d.mkdir()
            return str(d)

        monkeypatch.setattr(run_magi, "create_output_dir", fake_create)

        captured = {"guard_agents": None}

        def fake_guard(agents, mode, files, ranges, summary=None):
            captured["guard_agents"] = agents
            return agents

        monkeypatch.setattr(run_magi, "_apply_finding_guard", fake_guard)

        async def fake_orch(*a, **k):
            return {"agents": [_guard_agent([])], "consensus": {}}

        monkeypatch.setattr(run_magi, "run_orchestrator", fake_orch)
        monkeypatch.setattr(sys, "argv", ["run_magi.py", "code-review", "hello"])

        run_magi.main()

        assert resolve_calls["n"] == 1, (
            f"resolve_diff must be called exactly once (got {resolve_calls['n']}): "
            f"main() resolves it; enrichment must consume that value, not re-resolve."
        )
        assert seen["enrich_diff"] == sentinel, (
            "enrichment must receive the diff resolved once by main()"
        )
        assert seen["guard_diff"] == sentinel, (
            "the guard must receive the same diff resolved once by main()"
        )

    def test_cost_aggregates_all_launched_agents_in_degraded_mode(self, tmp_path, monkeypatch):
        """Finding #1 regression: cost must include all 3 launched agents even
        when the orchestrator returns only 2 (degraded mode).

        The failed/timed-out third agent may have already burned tokens and
        written its raw envelope to output_dir. Aggregating only over
        ``report["agents"]`` (the survivors) under-reports cost. This test
        verifies that the saved magi-report.json sums all 3 canonical agents
        (AGENTS constant) rather than just the 2 returned by the orchestrator.
        """
        import json as _json

        import run_magi

        monkeypatch.setattr(run_magi, "_enable_utf8_console_io", lambda: None)
        monkeypatch.setattr(run_magi.shutil, "which", lambda name: "claude")
        monkeypatch.setattr(run_magi, "build_user_prompt", lambda mode, content: "PROMPT")
        monkeypatch.setattr(run_magi, "_load_input_content", lambda arg: ("BODY", "Inline input"))
        monkeypatch.setattr(run_magi, "_maybe_enrich", lambda *a, **k: ("BODY", None))
        monkeypatch.setattr(run_magi, "format_report", lambda agents, consensus, **kw: "REPORT")
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)
        monkeypatch.setattr(run_magi, "cleanup_old_runs", lambda keep, run_root=None: None)
        monkeypatch.setattr(run_magi, "write_lock", lambda d, max_age_seconds=None: None)
        monkeypatch.setattr(run_magi, "remove_lock", lambda d: None)

        created: dict[str, str] = {}

        def fake_create(output_dir: object, run_root: object = None) -> str:
            d = tmp_path / "magi-run-degraded"
            d.mkdir(exist_ok=True)
            created["dir"] = str(d)
            return str(d)

        monkeypatch.setattr(run_magi, "create_output_dir", fake_create)

        # Orchestrator returns only melchior + balthasar (degraded: caspar failed).
        def _survivor(name: str) -> dict[str, Any]:
            return {
                "agent": name,
                "verdict": "approve",
                "confidence": 0.8,
                "summary": "s",
                "reasoning": "r",
                "recommendation": "rec",
                "findings": [],
            }

        async def fake_orch(*a: object, **k: object) -> dict[str, Any]:
            # Write raw envelopes for ALL 3 agents (caspar burned tokens too).
            out = created["dir"]
            for agent_name, cost in [
                ("melchior", 0.30),
                ("balthasar", 0.25),
                ("caspar", 0.20),
            ]:
                raw = {"total_cost_usd": cost, "result": "{}"}
                with open(os.path.join(out, f"{agent_name}.raw.json"), "w", encoding="utf-8") as fh:
                    _json.dump(raw, fh)
            # Only 2 survivors in the report (degraded).
            return {
                "agents": [_survivor("melchior"), _survivor("balthasar")],
                "consensus": {},
            }

        monkeypatch.setattr(run_magi, "run_orchestrator", fake_orch)
        monkeypatch.setattr(sys, "argv", ["run_magi.py", "design", "hello"])

        run_magi.main()

        report_path = os.path.join(created["dir"], "magi-report.json")
        with open(report_path, encoding="utf-8") as fh:
            saved = _json.load(fh)

        assert "cost" in saved
        # Must include caspar's 0.20 even though it is not in report["agents"].
        expected_total = round(0.30 + 0.25 + 0.20, 6)
        assert saved["cost"]["total_usd"] == expected_total, (
            f"Expected total_usd={expected_total} (all 3 agents), "
            f"got {saved['cost']['total_usd']} (only survivors counted)"
        )
        assert "caspar" in saved["cost"]["per_agent"], (
            "caspar must appear in per_agent cost even in degraded mode"
        )

    def test_a5_mode_strip_nulls_file_and_line_in_design_mode(self, tmp_path, monkeypatch):
        """Finding #2 coverage pin: A5 mode-strip zeroes file/line on every
        finding in non-code-review modes.

        The existing design-mode tests use empty findings, so the strip loop
        was never exercised on a populated finding. This test passes a finding
        with file and line set through a design-mode main() and asserts that
        the saved magi-report.json has both fields as None, confirming the
        existing strip code works on real data.
        """
        import json as _json

        import run_magi

        monkeypatch.setattr(run_magi, "_enable_utf8_console_io", lambda: None)
        monkeypatch.setattr(run_magi.shutil, "which", lambda name: "claude")
        monkeypatch.setattr(run_magi, "build_user_prompt", lambda mode, content: "PROMPT")
        monkeypatch.setattr(run_magi, "_load_input_content", lambda arg: ("BODY", "Inline input"))
        monkeypatch.setattr(run_magi, "_maybe_enrich", lambda *a, **k: ("BODY", None))
        monkeypatch.setattr(run_magi, "format_report", lambda agents, consensus, **kw: "REPORT")
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)
        monkeypatch.setattr(run_magi, "cleanup_old_runs", lambda keep, run_root=None: None)
        monkeypatch.setattr(run_magi, "write_lock", lambda d, max_age_seconds=None: None)
        monkeypatch.setattr(run_magi, "remove_lock", lambda d: None)
        monkeypatch.setattr(
            run_magi,
            "aggregate_cost",
            lambda output_dir, agents: {"per_agent": {}, "total_usd": 0.0},
        )

        created: dict[str, str] = {}

        def fake_create(output_dir: object, run_root: object = None) -> str:
            d = tmp_path / "magi-run-a5strip"
            d.mkdir(exist_ok=True)
            created["dir"] = str(d)
            return str(d)

        monkeypatch.setattr(run_magi, "create_output_dir", fake_create)

        # Return an agent whose finding has file and line set (non-null).
        agent_with_fields = {
            "agent": "melchior",
            "verdict": "approve",
            "confidence": 0.9,
            "summary": "s",
            "reasoning": "r",
            "recommendation": "rec",
            "findings": [
                {
                    "severity": "info",
                    "title": "T",
                    "detail": "d",
                    "file": "src/foo.py",
                    "line": 42,
                    "category": "style",
                }
            ],
        }
        second_agent = {
            "agent": "balthasar",
            "verdict": "approve",
            "confidence": 0.8,
            "summary": "s2",
            "reasoning": "r2",
            "recommendation": "rec2",
            "findings": [],
        }

        async def fake_orch(*a: object, **k: object) -> dict[str, Any]:
            return {"agents": [agent_with_fields, second_agent], "consensus": {}}

        monkeypatch.setattr(run_magi, "run_orchestrator", fake_orch)
        monkeypatch.setattr(sys, "argv", ["run_magi.py", "design", "hello"])

        run_magi.main()

        report_path = os.path.join(created["dir"], "magi-report.json")
        with open(report_path, encoding="utf-8") as fh:
            saved = _json.load(fh)

        # The consensus findings come from determine_consensus; verify by
        # checking that the consensus block exists and the finding's file/line
        # were stripped to None before determine_consensus ran.
        consensus_findings = saved.get("consensus", {}).get("findings", [])
        assert len(consensus_findings) == 1, "Expected 1 finding in consensus (from melchior)"
        fnd = consensus_findings[0]
        assert fnd.get("file") is None, (
            f"A5 strip must null file in design mode, got: {fnd.get('file')!r}"
        )
        assert fnd.get("line") is None, (
            f"A5 strip must null line in design mode, got: {fnd.get('line')!r}"
        )

    def _patch_main_for_cost_warn(self, tmp_path, monkeypatch, cost_total):
        """Shared setup for FIX 4 zero-cost warning tests.

        Returns a StringIO buffer capturing stderr from the main() call.
        """
        import io

        import run_magi

        monkeypatch.setattr(run_magi, "_enable_utf8_console_io", lambda: None)
        monkeypatch.setattr(run_magi.shutil, "which", lambda name: "claude")
        monkeypatch.setattr(run_magi, "build_user_prompt", lambda mode, content: "PROMPT")
        monkeypatch.setattr(run_magi, "_load_input_content", lambda arg: ("BODY", "Inline input"))
        monkeypatch.setattr(run_magi, "_maybe_enrich", lambda *a, **k: ("BODY", None))
        monkeypatch.setattr(run_magi, "format_report", lambda agents, consensus, **kw: "REPORT")
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)
        monkeypatch.setattr(run_magi, "cleanup_old_runs", lambda keep, run_root=None: None)
        monkeypatch.setattr(run_magi, "write_lock", lambda d, max_age_seconds=None: None)
        monkeypatch.setattr(run_magi, "remove_lock", lambda d: None)
        monkeypatch.setattr(
            run_magi,
            "aggregate_cost",
            lambda output_dir, agents: {
                "per_agent": {"melchior": cost_total},
                "total_usd": cost_total,
            },
        )

        def fake_create(output_dir: object, run_root: object = None) -> str:
            d = tmp_path / "magi-run-costwarn"
            d.mkdir(exist_ok=True)
            return str(d)

        monkeypatch.setattr(run_magi, "create_output_dir", fake_create)

        async def fake_orch(*a: object, **k: object) -> dict[str, Any]:
            return {
                "agents": [
                    {
                        "agent": "melchior",
                        "verdict": "approve",
                        "confidence": 0.9,
                        "summary": "s",
                        "reasoning": "r",
                        "recommendation": "rec",
                        "findings": [],
                    }
                ],
                "consensus": {},
            }

        monkeypatch.setattr(run_magi, "run_orchestrator", fake_orch)
        monkeypatch.setattr(sys, "argv", ["run_magi.py", "design", "hello"])

        buf: io.StringIO = io.StringIO()
        with monkeypatch.context() as mp:
            mp.setattr(sys, "stderr", buf)
            run_magi.main()
        return buf

    def test_zero_cost_warning_emitted_when_cost_is_zero(self, tmp_path, monkeypatch):
        """FIX 4: when aggregate_cost returns 0.0 and there is >= 1 agent,
        main() must emit a [!] WARNING to stderr so silent $0.00 mis-reporting
        is visible (the CLI may have renamed total_cost_usd)."""
        buf = self._patch_main_for_cost_warn(tmp_path, monkeypatch, cost_total=0.0)
        err = buf.getvalue()
        assert "[!] WARNING" in err and "$0.00" in err, (
            f"Expected zero-cost [!] WARNING in stderr, got:\n{err!r}"
        )

    def test_zero_cost_warning_not_emitted_when_cost_positive(self, tmp_path, monkeypatch):
        """FIX 4: when aggregate_cost returns > 0, no zero-cost warning is emitted."""
        buf = self._patch_main_for_cost_warn(tmp_path, monkeypatch, cost_total=0.75)
        err = buf.getvalue()
        assert not ("$0.00" in err and "[!] WARNING" in err), (
            f"Zero-cost warning must NOT appear when cost > 0; got:\n{err!r}"
        )

    def test_ollama_zero_cost_warning_not_emitted(self, tmp_path, monkeypatch):
        """BUG 2: on --ollama runs aggregate_cost always returns $0.00 (no
        total_cost_usd in Ollama responses). The spurious WARNING must be
        suppressed — $0 is correct for Ollama and the message is misleading."""
        import io

        import run_magi
        from ollama_backend import OllamaBackend
        from ollama_config import OllamaConfig

        monkeypatch.setattr(run_magi, "_enable_utf8_console_io", lambda: None)
        monkeypatch.setattr(run_magi, "build_user_prompt", lambda mode, content: "PROMPT")
        monkeypatch.setattr(run_magi, "_load_input_content", lambda arg: ("BODY", "Inline input"))
        monkeypatch.setattr(run_magi, "_maybe_enrich", lambda *a, **k: ("BODY", None))
        monkeypatch.setattr(run_magi, "format_report", lambda agents, consensus, **kw: "REPORT")
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)
        monkeypatch.setattr(run_magi, "cleanup_old_runs", lambda keep, run_root=None: None)
        monkeypatch.setattr(run_magi, "write_lock", lambda d, max_age_seconds=None: None)
        monkeypatch.setattr(run_magi, "remove_lock", lambda d: None)
        # Ollama runs always aggregate to $0.00 (no total_cost_usd field).
        monkeypatch.setattr(
            run_magi,
            "aggregate_cost",
            lambda output_dir, agents: {"per_agent": {}, "total_usd": 0.0},
        )

        cfg = OllamaConfig(
            base_url="http://h:11434/v1",
            api_key=None,
            models={
                "melchior": ModelSpec("m", "la"),
                "balthasar": ModelSpec("b", "lb"),
                "caspar": ModelSpec("c", "lc"),
            },
        )
        ollama_backend = OllamaBackend(cfg)

        # select_backend is async now (T9) and returns a 3-tuple; run_orchestrator is
        # faked below, so rotation=None is fine (the banner only reads .model).
        async def fake_select(args, prompt):
            return ollama_backend, dict(cfg.models), None

        monkeypatch.setattr(run_magi, "select_backend", fake_select)

        def fake_create(output_dir: object, run_root: object = None) -> str:
            d = tmp_path / "magi-run-ollama-cost"
            d.mkdir(exist_ok=True)
            return str(d)

        monkeypatch.setattr(run_magi, "create_output_dir", fake_create)

        async def fake_orch(*a: object, **k: object) -> dict[str, Any]:
            return {
                "agents": [
                    {
                        "agent": "melchior",
                        "verdict": "approve",
                        "confidence": 0.9,
                        "summary": "s",
                        "reasoning": "r",
                        "recommendation": "rec",
                        "findings": [],
                    }
                ],
                "consensus": {},
            }

        monkeypatch.setattr(run_magi, "run_orchestrator", fake_orch)
        monkeypatch.setattr(sys, "argv", ["run_magi.py", "design", "hello", "--ollama"])

        buf: io.StringIO = io.StringIO()
        with monkeypatch.context() as mp:
            mp.setattr(sys, "stderr", buf)
            run_magi.main()
        err = buf.getvalue()
        assert "$0.00" not in err, (
            f"Spurious zero-cost WARNING must not appear on --ollama runs; got:\n{err!r}"
        )

    def test_e2e_fabricated_finding_dropped_score_unchanged(self, tmp_path, monkeypatch):
        """FIX 5 (coverage pin): end-to-end BDD-14 invariant through main().

        Drives main() in code-review mode with a real diff (touching x.py only)
        and a stubbed orchestrator that returns two agents: melchior with a
        fabricated finding on ghost.py (not in the diff) plus a real finding on
        x.py, and balthasar with no findings.

        Asserts:
        (a) The saved magi-report.json Key Findings do NOT contain the fabricated
            finding (guard dropped it).
        (b) The consensus score/verdict/label equals the baseline run without the
            fabricated finding (the guard filters findings, not votes).

        This is a coverage pin — the behaviour already works via _apply_finding_guard.
        It closes the unit-only gap documented in the FIX 5 spec: no single test
        previously drove a real fabricated finding through main()'s actual guard +
        consensus-recompute path.
        """
        import io
        import json as _json

        import run_magi
        from synthesize import determine_consensus

        # Stub the boilerplate that is orthogonal to this test.
        monkeypatch.setattr(run_magi, "_enable_utf8_console_io", lambda: None)
        monkeypatch.setattr(run_magi.shutil, "which", lambda name: "claude")
        monkeypatch.setattr(run_magi, "build_user_prompt", lambda mode, content: "PROMPT")
        monkeypatch.setattr(run_magi, "_load_input_content", lambda arg: ("BODY", "Inline input"))
        monkeypatch.setattr(run_magi, "_maybe_enrich", lambda *a, **k: ("BODY", None))
        monkeypatch.setattr(run_magi, "format_report", lambda agents, consensus, **kw: "REPORT")
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)
        monkeypatch.setattr(run_magi, "cleanup_old_runs", lambda keep, run_root=None: None)
        monkeypatch.setattr(run_magi, "write_lock", lambda d, max_age_seconds=None: None)
        monkeypatch.setattr(run_magi, "remove_lock", lambda d: None)
        monkeypatch.setattr(
            run_magi,
            "aggregate_cost",
            lambda output_dir, agents: {"per_agent": {}, "total_usd": 0.50},
        )

        # Real diff touching only x.py (line 2 added).
        real_diff = (
            "diff --git a/x.py b/x.py\n"
            "--- a/x.py\n"
            "+++ b/x.py\n"
            "@@ -1,2 +1,3 @@\n"
            " ctx\n"
            "+added\n"
            " ctx2\n"
        )
        # Thread the real diff through _diff_files_and_ranges (real implementation).
        monkeypatch.setattr(run_magi, "resolve_diff", lambda *a, **k: real_diff)

        created: dict[str, str] = {}

        def fake_create(output_dir: object, run_root: object = None) -> str:
            d = tmp_path / "magi-run-e2e"
            d.mkdir(exist_ok=True)
            created["dir"] = str(d)
            return str(d)

        monkeypatch.setattr(run_magi, "create_output_dir", fake_create)

        # Real finding on x.py (in diff) and fabricated finding on ghost.py (not in diff).
        fabricated_finding: dict[str, Any] = {
            "severity": "critical",
            "title": "Fabricated hallucination",
            "detail": "fabricated",
            "file": "ghost.py",
            "line": 99,
            "category": "other",
        }
        real_finding: dict[str, Any] = {
            "severity": "warning",
            "title": "Real finding on x.py",
            "detail": "real",
            "file": "x.py",
            "line": 2,
            "category": "other",
        }

        def _agent_dict(name, findings, verdict="approve"):
            return {
                "agent": name,
                "verdict": verdict,
                "confidence": 0.8,
                "summary": "s",
                "reasoning": "r",
                "recommendation": "rec",
                "findings": findings,
            }

        # Orchestrator returns two agents (melchior has both findings; balthasar none).
        async def fake_orch(*a, **k):
            return {
                "agents": [
                    _agent_dict("melchior", [fabricated_finding, real_finding]),
                    _agent_dict("balthasar", []),
                ],
                "consensus": {},
            }

        monkeypatch.setattr(run_magi, "run_orchestrator", fake_orch)
        monkeypatch.setattr(sys, "argv", ["run_magi.py", "code-review", "hello"])

        buf: io.StringIO = io.StringIO()
        with monkeypatch.context() as mp:
            mp.setattr(sys, "stderr", buf)
            run_magi.main()

        report_path = created["dir"] + "/magi-report.json"
        with open(report_path, encoding="utf-8") as fh:
            saved = _json.load(fh)

        # (a) Fabricated finding must not appear in consensus findings.
        consensus_findings = saved.get("consensus", {}).get("findings", [])
        fabricated_titles = [f["title"] for f in consensus_findings if "ghost" in f.get("file", "")]
        assert fabricated_titles == [], (
            f"Fabricated finding on ghost.py must be dropped; still present: {fabricated_titles}"
        )
        all_titles = [f["title"] for f in consensus_findings]
        assert "Fabricated hallucination" not in all_titles, (
            f"Fabricated finding title must not appear in consensus findings: {all_titles}"
        )

        # (b) Score/verdict must match the baseline (same agents, same votes, no fabricated finding).
        baseline_agents = [
            _agent_dict("melchior", [real_finding]),
            _agent_dict("balthasar", []),
        ]
        baseline_consensus = determine_consensus(baseline_agents)
        saved_consensus = saved.get("consensus", {})
        assert saved_consensus["consensus"] == baseline_consensus["consensus"], (
            f"Guard must not change consensus label: "
            f"got {saved_consensus['consensus']!r}, expected {baseline_consensus['consensus']!r}"
        )
        assert saved_consensus["consensus_verdict"] == baseline_consensus["consensus_verdict"], (
            "Guard must not change consensus_verdict"
        )
        assert saved_consensus["confidence"] == baseline_consensus["confidence"], (
            "Guard must not change confidence"
        )


class TestInputSizeWiring:
    """Input-size telemetry + detect-and-warn wiring in run_magi.main()."""

    def _patch_main(self, tmp_path, monkeypatch, *, input_body="BODY", extra_argv=None):
        """Stub everything around main() except the input-size wiring under test.

        Returns ``created`` dict (keyed ``"dir"``) so callers can inspect the
        saved magi-report.json, and a StringIO capturing stderr.
        """
        import io

        import run_magi

        monkeypatch.setattr(run_magi, "_enable_utf8_console_io", lambda: None)
        monkeypatch.setattr(run_magi.shutil, "which", lambda name: "claude")
        monkeypatch.setattr(run_magi, "build_user_prompt", lambda mode, content: "PROMPT")
        monkeypatch.setattr(
            run_magi, "_load_input_content", lambda arg: (input_body, "Inline input")
        )
        monkeypatch.setattr(run_magi, "_maybe_enrich", lambda *a, **k: (input_body, None))
        monkeypatch.setattr(run_magi, "format_report", lambda agents, consensus, **kw: "REPORT")
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)
        monkeypatch.setattr(run_magi, "cleanup_old_runs", lambda keep, run_root=None: None)
        monkeypatch.setattr(run_magi, "write_lock", lambda d, max_age_seconds=None: None)
        monkeypatch.setattr(run_magi, "remove_lock", lambda d: None)
        monkeypatch.setattr(
            run_magi,
            "aggregate_cost",
            lambda output_dir, agents: {"per_agent": {}, "total_usd": 0.75},
        )

        created: dict[str, str] = {}

        def fake_create(output_dir: object, run_root: object = None) -> str:
            d = tmp_path / "magi-run-inputsize"
            d.mkdir(exist_ok=True)
            created["dir"] = str(d)
            return str(d)

        monkeypatch.setattr(run_magi, "create_output_dir", fake_create)

        async def fake_orch(*a: object, **k: object) -> dict[str, Any]:
            return {
                "agents": [
                    {
                        "agent": "melchior",
                        "verdict": "approve",
                        "confidence": 0.9,
                        "summary": "s",
                        "reasoning": "r",
                        "recommendation": "rec",
                        "findings": [],
                    },
                    {
                        "agent": "balthasar",
                        "verdict": "approve",
                        "confidence": 0.8,
                        "summary": "s2",
                        "reasoning": "r2",
                        "recommendation": "rec2",
                        "findings": [],
                    },
                ],
                "consensus": {},
            }

        monkeypatch.setattr(run_magi, "run_orchestrator", fake_orch)
        argv = ["run_magi.py", "design", "hello"] + (extra_argv or [])
        monkeypatch.setattr(sys, "argv", argv)

        buf: io.StringIO = io.StringIO()
        with monkeypatch.context() as mp:
            mp.setattr(sys, "stderr", buf)
            run_magi.main()

        return created, buf

    def test_input_size_block_in_saved_report(self, tmp_path, monkeypatch):
        """Telemetry: magi-report.json on disk carries an input_size block with
        chars and est_tokens fields."""
        import json

        body = "x" * 400  # 400 chars -> 100 est tokens
        created, _ = self._patch_main(tmp_path, monkeypatch, input_body=body)

        report_path = os.path.join(created["dir"], "magi-report.json")
        with open(report_path, encoding="utf-8") as fh:
            saved = json.load(fh)

        assert "input_size" in saved, "magi-report.json must carry an input_size block"
        assert saved["input_size"]["chars"] == 400
        assert saved["input_size"]["est_tokens"] == 100

    def test_input_size_block_is_self_describing(self, tmp_path, monkeypatch):
        """Telemetry: input_size block records oversize flag and warn threshold so
        the block is self-describing without external context.

        Uses a body of 400 chars (100 est tokens) with a low threshold of 50
        so oversize is deterministically True.
        """
        import json

        body = "x" * 400  # 400 chars -> 100 est tokens; 100 > 50 => oversize=True
        created, _ = self._patch_main(
            tmp_path,
            monkeypatch,
            input_body=body,
            extra_argv=["--warn-input-tokens", "50"],
        )

        report_path = os.path.join(created["dir"], "magi-report.json")
        with open(report_path, encoding="utf-8") as fh:
            saved = json.load(fh)

        block = saved["input_size"]
        assert "oversize" in block, "input_size must carry an 'oversize' key"
        assert "warn_threshold_tokens" in block, (
            "input_size must carry a 'warn_threshold_tokens' key"
        )
        assert block["oversize"] is True, (
            f"oversize must be True (100 est_tokens > 50 threshold); got {block['oversize']!r}"
        )
        assert block["warn_threshold_tokens"] == 50, (
            f"warn_threshold_tokens must equal the --warn-input-tokens value (50); "
            f"got {block['warn_threshold_tokens']!r}"
        )

    def test_oversize_warning_emitted_when_threshold_exceeded(self, tmp_path, monkeypatch):
        """Detect-and-warn: when estimated tokens exceed --warn-input-tokens,
        a [!] WARNING line is printed to stderr."""
        body = "x" * 4000  # 4000 chars -> 1000 est tokens > 5 threshold
        _, buf = self._patch_main(
            tmp_path, monkeypatch, input_body=body, extra_argv=["--warn-input-tokens", "5"]
        )
        err = buf.getvalue()
        assert "[!] WARNING" in err, (
            f"Expected oversize [!] WARNING in stderr when threshold exceeded; got:\n{err!r}"
        )

    def test_no_warning_when_threshold_not_exceeded(self, tmp_path, monkeypatch):
        """Detect-and-warn: when estimated tokens do NOT exceed --warn-input-tokens,
        no [!] WARNING is emitted to stderr."""
        body = "x" * 400  # 400 chars -> 100 est tokens, not > 200 threshold
        _, buf = self._patch_main(
            tmp_path,
            monkeypatch,
            input_body=body,
            extra_argv=["--warn-input-tokens", "200"],
        )
        err = buf.getvalue()
        assert "[!] WARNING" not in err, (
            f"[!] WARNING must NOT appear when threshold is not exceeded; got:\n{err!r}"
        )

    def test_input_size_chars_measures_raw_input_not_enriched(self, tmp_path, monkeypatch):
        """Regression: input_size.chars must reflect the RAW input length, not the
        post-enrichment string.  In code-review mode, _maybe_enrich reassigns
        input_content to a larger string; chars and est_tokens must both be
        derived from the original raw input so they are consistent
        (est_tokens == chars // 4).

        With the pre-fix code, chars == len(enriched_body) (wrong), while
        est_tokens is computed on raw_body (correct). This test fails on the
        buggy code and passes after the fix.
        """
        import json

        import run_magi

        RAW_BODY = "x" * 400  # 400 chars -> 100 est_tokens
        ENRICHED_SUFFIX = "y" * 5000
        ENRICHED_BODY = RAW_BODY + ENRICHED_SUFFIX  # 5400 chars, would give 1350 est tokens if raw

        # Monkeypatch _maybe_enrich to return the enriched body (simulating code-review enrichment).
        monkeypatch.setattr(
            run_magi, "_maybe_enrich", lambda *a, **k: (ENRICHED_BODY, "enriched context note")
        )
        # Use code-review mode so resolve_diff is called; stub it out.
        monkeypatch.setattr(run_magi, "resolve_diff", lambda content, cwd, base: "")

        # All stubs are set inline (not via _patch_main) so _maybe_enrich can be
        # set last to guarantee our enriched-body stub wins over any earlier setattr.
        monkeypatch.setattr(run_magi, "_enable_utf8_console_io", lambda: None)
        monkeypatch.setattr(run_magi.shutil, "which", lambda name: "claude")
        monkeypatch.setattr(run_magi, "build_user_prompt", lambda mode, content: "PROMPT")
        monkeypatch.setattr(run_magi, "_load_input_content", lambda arg: (RAW_BODY, "Inline input"))
        monkeypatch.setattr(run_magi, "format_report", lambda agents, consensus, **kw: "REPORT")
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)
        monkeypatch.setattr(run_magi, "cleanup_old_runs", lambda keep, run_root=None: None)
        monkeypatch.setattr(run_magi, "write_lock", lambda d, max_age_seconds=None: None)
        monkeypatch.setattr(run_magi, "remove_lock", lambda d: None)
        monkeypatch.setattr(
            run_magi,
            "aggregate_cost",
            lambda output_dir, agents: {"per_agent": {}, "total_usd": 0.75},
        )

        created: dict[str, str] = {}

        def fake_create(output_dir: object, run_root: object = None) -> str:

            d = tmp_path / "magi-run-raw-chars"
            d.mkdir(exist_ok=True)
            created["dir"] = str(d)
            return str(d)

        monkeypatch.setattr(run_magi, "create_output_dir", fake_create)

        async def fake_orch(*a: object, **k: object) -> dict[str, Any]:
            return {
                "agents": [
                    {
                        "agent": "melchior",
                        "verdict": "approve",
                        "confidence": 0.9,
                        "summary": "s",
                        "reasoning": "r",
                        "recommendation": "rec",
                        "findings": [],
                    },
                    {
                        "agent": "balthasar",
                        "verdict": "approve",
                        "confidence": 0.8,
                        "summary": "s2",
                        "reasoning": "r2",
                        "recommendation": "rec2",
                        "findings": [],
                    },
                ],
                "consensus": {},
            }

        monkeypatch.setattr(run_magi, "run_orchestrator", fake_orch)
        # Set _maybe_enrich AFTER all other stubs so this override wins.
        monkeypatch.setattr(
            run_magi, "_maybe_enrich", lambda *a, **k: (ENRICHED_BODY, "enriched context note")
        )
        monkeypatch.setattr(sys, "argv", ["run_magi.py", "code-review", "hello"])

        import io

        buf: io.StringIO = io.StringIO()
        with monkeypatch.context() as mp:
            mp.setattr(sys, "stderr", buf)
            run_magi.main()

        report_path = os.path.join(created["dir"], "magi-report.json")
        with open(report_path, encoding="utf-8") as fh:
            saved = json.load(fh)

        raw_len = len(RAW_BODY)  # 400
        assert saved["input_size"]["chars"] == raw_len, (
            f"input_size.chars must be the RAW input length ({raw_len}), "
            f"not the enriched length ({len(ENRICHED_BODY)}); "
            f"got {saved['input_size']['chars']!r}"
        )
        assert saved["input_size"]["est_tokens"] == raw_len // 4, (
            f"est_tokens must equal chars // 4 == {raw_len // 4}; "
            f"got {saved['input_size']['est_tokens']!r}"
        )


class TestF4GuardObservability:
    """F4: the finding guard drops/annotates findings but the consensus keeps the
    agent's vote. Without surfacing the drops, an agent can vote (e.g. reject)
    yet show no Key Findings, with no record of why. The guard must populate an
    optional ``summary`` out-param so ``main()`` can write a ``guard`` block to
    magi-report.json — the audit artifact then explains the empty findings."""

    def test_guard_summary_records_drops_and_annotations(self):
        """A populated summary carries per-agent dropped/annotated counts, the
        dropped titles (not the kept-but-annotated ones), and totals."""
        import run_magi

        agents = [
            _guard_agent(
                [
                    {
                        "severity": "critical",
                        "title": "Ghost",
                        "detail": "d",
                        "file": "ghost.py",
                        "line": 5,
                        "category": "null-deref",
                    },
                    {
                        "severity": "warning",
                        "title": "Outside",
                        "detail": "d2",
                        "file": "x.py",
                        "line": 999,
                        "category": "other",
                    },
                    {
                        "severity": "info",
                        "title": "Good",
                        "detail": "d3",
                        "file": "x.py",
                        "line": 2,
                        "category": "other",
                    },
                ]
            )
        ]
        summary: dict[str, Any] = {}
        run_magi._apply_finding_guard(
            agents, "code-review", {"x.py"}, {"x.py": {2}}, summary=summary
        )
        assert summary["active"] is True
        assert summary["files_in_diff"] == 1
        assert summary["total_dropped"] == 1
        assert summary["total_annotated"] == 1
        pa = summary["per_agent"]["melchior"]
        assert pa["dropped"] == 1 and pa["annotated"] == 1
        assert "Ghost" in pa["dropped_titles"]
        assert "Outside" not in pa["dropped_titles"], "annotated (kept) finding is not a drop"

    def test_guard_summary_inactive_in_non_code_review(self):
        """design/analysis -> guard is a no-op; summary records active=False only."""
        import run_magi

        agents = [
            _guard_agent(
                [
                    {
                        "severity": "warning",
                        "title": "t",
                        "detail": "d",
                        "file": "ghost.py",
                        "line": 1,
                    }
                ]
            )
        ]
        summary: dict[str, Any] = {}
        out = run_magi._apply_finding_guard(
            agents, "design", {"x.py"}, {"x.py": {2}}, summary=summary
        )
        assert summary == {"active": False}
        assert len(out[0]["findings"]) == 1

    def test_guard_summary_inactive_when_no_diff(self):
        """code-review with no resolvable diff (empty files) -> active=False."""
        import run_magi

        agents = [
            _guard_agent(
                [
                    {
                        "severity": "warning",
                        "title": "t",
                        "detail": "d",
                        "file": "ghost.py",
                        "line": 1,
                    }
                ]
            )
        ]
        summary: dict[str, Any] = {}
        run_magi._apply_finding_guard(agents, "code-review", set(), {}, summary=summary)
        assert summary == {"active": False}

    def test_guard_summary_omits_clean_agents(self):
        """An agent whose findings all survive contributes nothing to per_agent."""
        import run_magi

        agents = [
            _guard_agent(
                [{"severity": "info", "title": "ok", "detail": "d", "file": "x.py", "line": 2}]
            )
        ]
        summary: dict[str, Any] = {}
        run_magi._apply_finding_guard(
            agents, "code-review", {"x.py"}, {"x.py": {2}}, summary=summary
        )
        assert summary["active"] is True
        assert summary["total_dropped"] == 0 and summary["total_annotated"] == 0
        assert summary["per_agent"] == {}

    def test_guard_block_in_saved_report(self, tmp_path, monkeypatch):
        """F4 wiring: in code-review, magi-report.json carries a 'guard' block
        with the per-agent dropped titles, explaining an agent's empty findings."""
        import run_magi

        monkeypatch.setattr(run_magi, "_enable_utf8_console_io", lambda: None)
        monkeypatch.setattr(run_magi.shutil, "which", lambda name: "claude")
        monkeypatch.setattr(run_magi, "build_user_prompt", lambda mode, content: "PROMPT")
        monkeypatch.setattr(run_magi, "_load_input_content", lambda arg: ("BODY", "Inline input"))
        monkeypatch.setattr(run_magi, "_maybe_enrich", lambda *a, **k: ("BODY", None))
        monkeypatch.setattr(run_magi, "format_report", lambda agents, consensus, **kw: "REPORT")
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)
        monkeypatch.setattr(run_magi, "cleanup_old_runs", lambda keep, run_root=None: None)
        monkeypatch.setattr(run_magi, "write_lock", lambda d, max_age_seconds=None: None)
        monkeypatch.setattr(run_magi, "remove_lock", lambda d: None)
        # Control the diff so the guard runs with a known file-set/ranges.
        monkeypatch.setattr(run_magi, "resolve_diff", lambda content, cwd, base: "DIFF")
        monkeypatch.setattr(
            run_magi, "_diff_files_and_ranges", lambda diff: ({"x.py"}, {"x.py": {2}})
        )
        monkeypatch.setattr(
            run_magi,
            "aggregate_cost",
            lambda output_dir, agents: {"per_agent": {}, "total_usd": 0.0},
        )

        created = {}

        def fake_create(output_dir, run_root=None):
            d = tmp_path / "magi-run-guard"
            d.mkdir()
            created["dir"] = str(d)
            return str(d)

        monkeypatch.setattr(run_magi, "create_output_dir", fake_create)

        async def fake_orch(*a, **k):
            agent = _guard_agent(
                [
                    {
                        "severity": "critical",
                        "title": "Ghost",
                        "detail": "d",
                        "file": "ghost.py",
                        "line": 5,
                        "category": "null-deref",
                    }
                ]
            )
            agent["verdict"] = "reject"
            return {
                "agents": [agent],
                "consensus": {"consensus": "x", "consensus_verdict": "reject"},
            }

        monkeypatch.setattr(run_magi, "run_orchestrator", fake_orch)
        monkeypatch.setattr(sys, "argv", ["run_magi.py", "code-review", "hello"])

        run_magi.main()

        import json

        with open(os.path.join(created["dir"], "magi-report.json"), encoding="utf-8") as fh:
            saved = json.load(fh)
        assert "guard" in saved
        assert saved["guard"]["active"] is True
        assert saved["guard"]["total_dropped"] == 1
        assert "Ghost" in saved["guard"]["per_agent"]["melchior"]["dropped_titles"]

    def test_dropped_titles_handles_duplicate_titles(self):
        """F4 (loop-1): two findings sharing a title — one dropped (file not in
        diff), one kept (file in diff) — must still list the dropped title. A
        title-set reconstruction silently omits it (the kept one masks it)."""
        import run_magi

        agents = [
            _guard_agent(
                [
                    {
                        "severity": "critical",
                        "title": "Same title",
                        "detail": "d",
                        "file": "ghost.py",
                        "line": 5,
                        "category": "null-deref",
                    },
                    {
                        "severity": "warning",
                        "title": "Same title",
                        "detail": "d2",
                        "file": "x.py",
                        "line": 2,
                        "category": "other",
                    },
                ]
            )
        ]
        summary: dict[str, Any] = {}
        run_magi._apply_finding_guard(
            agents, "code-review", {"x.py"}, {"x.py": {2}}, summary=summary
        )
        pa = summary["per_agent"]["melchior"]
        assert pa["dropped"] == 1
        assert "Same title" in pa["dropped_titles"], (
            "the dropped finding's title must appear even when a kept finding shares it"
        )

    def test_guard_block_present_in_non_code_review_report(self, tmp_path, monkeypatch):
        """F4 (loop-1): the 'guard' block is ALWAYS present in the saved report;
        for design/analysis it is {'active': False}. Pins the always-present
        contract so a future mode-guard cannot silently drop the field."""
        import run_magi

        monkeypatch.setattr(run_magi, "_enable_utf8_console_io", lambda: None)
        monkeypatch.setattr(run_magi.shutil, "which", lambda name: "claude")
        monkeypatch.setattr(run_magi, "build_user_prompt", lambda mode, content: "PROMPT")
        monkeypatch.setattr(run_magi, "_load_input_content", lambda arg: ("BODY", "Inline input"))
        monkeypatch.setattr(run_magi, "_maybe_enrich", lambda *a, **k: ("BODY", None))
        monkeypatch.setattr(run_magi, "format_report", lambda agents, consensus, **kw: "REPORT")
        monkeypatch.setattr(run_magi, "_resolve_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(run_magi, "project_run_root", lambda root: str(tmp_path))
        monkeypatch.setattr(run_magi, "sweep_legacy_runs_once", lambda: None)
        monkeypatch.setattr(run_magi, "cleanup_old_runs", lambda keep, run_root=None: None)
        monkeypatch.setattr(run_magi, "write_lock", lambda d, max_age_seconds=None: None)
        monkeypatch.setattr(run_magi, "remove_lock", lambda d: None)
        monkeypatch.setattr(
            run_magi,
            "aggregate_cost",
            lambda output_dir, agents: {"per_agent": {}, "total_usd": 0.0},
        )

        created = {}

        def fake_create(output_dir, run_root=None):
            d = tmp_path / "magi-run-design-guard"
            d.mkdir()
            created["dir"] = str(d)
            return str(d)

        monkeypatch.setattr(run_magi, "create_output_dir", fake_create)

        async def fake_orch(*a, **k):
            return {"agents": [_guard_agent([])], "consensus": {}}

        monkeypatch.setattr(run_magi, "run_orchestrator", fake_orch)
        monkeypatch.setattr(sys, "argv", ["run_magi.py", "design", "hello"])

        run_magi.main()

        import json

        with open(os.path.join(created["dir"], "magi-report.json"), encoding="utf-8") as fh:
            saved = json.load(fh)
        assert saved["guard"] == {"active": False}


# ---------------------------------------------------------------------------
# Task 8: --ollama / --ollama-init flags + --model mutual exclusion (BDD-21)
# ---------------------------------------------------------------------------


def test_ollama_flag_defaults_false():
    from run_magi import parse_args

    args = parse_args(["code-review", "x"])
    assert args.ollama is False and args.ollama_init is False


def test_ollama_skips_claude_model_default():
    from run_magi import parse_args

    args = parse_args(["code-review", "x", "--ollama"])
    assert args.ollama is True
    assert args.model is None  # NOT filled with MODE_DEFAULT_MODELS


def test_ollama_with_explicit_model_errors():
    from run_magi import parse_args

    with pytest.raises(SystemExit):
        parse_args(["code-review", "x", "--ollama", "--model", "opus"])


def test_non_ollama_still_resolves_default_model():
    from run_magi import parse_args

    args = parse_args(["code-review", "x"])
    assert args.model == "opus"  # unchanged behavior


# ---------------------------------------------------------------------------
# Task 9: select_backend factory + back-compat run_orchestrator (BDD-1,2,3)
# ---------------------------------------------------------------------------


def test_select_backend_claude_default():
    from run_magi import parse_args, select_backend
    from claude_backend import ClaudeBackend

    args = parse_args(["design", "x"])
    backend, agent_models, rotation = asyncio.run(select_backend(args, "payload"))
    assert isinstance(backend, ClaudeBackend)
    assert {s.model for s in agent_models.values()} == {"opus"}
    assert rotation is None  # Claude path keeps v4 single-shot retry, no rotation


def test_select_backend_ollama_uses_trio(monkeypatch):
    from run_magi import parse_args, select_backend
    import run_magi
    from ollama_backend import OllamaBackend
    from ollama_config import ModelSpec, OllamaConfig
    from run_magi import RotationContext

    cfg = OllamaConfig(
        base_url="http://h/v1",
        api_key=None,
        models={
            "melchior": ModelSpec("m", "la"),
            "balthasar": ModelSpec("b", "lb"),
            "caspar": ModelSpec("c", "lc"),
        },
    )
    monkeypatch.setattr(run_magi, "resolve_config", lambda **k: cfg)
    monkeypatch.setattr(run_magi, "preflight", _preflight_ok)
    args = parse_args(["design", "x", "--ollama"])
    backend, agent_models, rotation = asyncio.run(select_backend(args, "payload"))
    assert isinstance(backend, OllamaBackend)
    assert agent_models == {
        "melchior": ModelSpec("m", "la"),
        "balthasar": ModelSpec("b", "lb"),
        "caspar": ModelSpec("c", "lc"),
    }
    assert isinstance(rotation, RotationContext)  # Ollama path carries the apparatus


def test_orchestrator_passes_per_agent_model(monkeypatch, tmp_path):
    """run_orchestrator threads per-agent models to launch_agent via backend."""
    import asyncio
    from run_magi import run_orchestrator

    seen: dict[str, str] = {}

    class FakeBackend:
        async def run(
            self,
            name: str,
            sp: str,
            prompt: str,
            model: str,
            timeout: int,
            out: str,
        ) -> bytes:
            seen[name] = model
            return (
                b'{"agent":"' + name.encode() + b'",'
                b'"verdict":"approve","confidence":0.5,'
                b'"summary":"s","reasoning":"r","findings":[],'
                b'"recommendation":"ok"}'
            )

    for a in ("melchior", "balthasar", "caspar"):
        (tmp_path / f"{a}.md").write_text("S", encoding="utf-8")

    asyncio.run(
        run_orchestrator(
            str(tmp_path),
            "P",
            str(tmp_path),
            900,
            agent_models={
                "melchior": ModelSpec("m", "la"),
                "balthasar": ModelSpec("b", "lb"),
                "caspar": ModelSpec("c", "lc"),
            },
            backend=FakeBackend(),
            show_status=False,
        )
    )
    assert seen == {"melchior": "m", "balthasar": "b", "caspar": "c"}


def test_resolve_config_called_once_in_select_backend(monkeypatch):
    """F-M invariant: resolve_config is called exactly once in select_backend."""
    from run_magi import parse_args, select_backend
    import run_magi
    from ollama_config import ModelSpec, OllamaConfig

    call_count = 0

    def counting_resolve(**k: object) -> OllamaConfig:
        nonlocal call_count
        call_count += 1
        return OllamaConfig(
            base_url="http://h/v1",
            api_key=None,
            models={
                "melchior": ModelSpec("m", "la"),
                "balthasar": ModelSpec("b", "lb"),
                "caspar": ModelSpec("c", "lc"),
            },
        )

    monkeypatch.setattr(run_magi, "resolve_config", counting_resolve)
    monkeypatch.setattr(run_magi, "preflight", _preflight_ok)
    args = parse_args(["design", "x", "--ollama"])
    asyncio.run(select_backend(args, "payload"))
    assert call_count == 1


# ---------------------------------------------------------------------------
# Task 10: main() wiring — --ollama-init short-circuit + skip claude gate
# ---------------------------------------------------------------------------


def _make_ollama_cfg():  # type: ignore[return]  # OllamaConfig imported lazily
    """Return a minimal OllamaConfig for T10 tests."""
    from ollama_config import ModelSpec, OllamaConfig

    return OllamaConfig(
        base_url="http://h/v1",
        api_key=None,
        models={
            "melchior": ModelSpec("m", "la"),
            "balthasar": ModelSpec("b", "lb"),
            "caspar": ModelSpec("c", "lc"),
        },
    )


def test_ollama_init_short_circuits(monkeypatch, tmp_path, capsys):
    """--ollama-init calls write_template() and exits 0 before mode/input checks."""
    import run_magi

    written: dict[str, str] = {}

    def _fake_write_template(**k: object) -> str:
        written["p"] = "X"
        return "X"

    monkeypatch.setattr(sys, "argv", ["run_magi.py", "--ollama-init"])
    monkeypatch.setattr(run_magi, "write_template", _fake_write_template)
    with pytest.raises(SystemExit) as ei:
        run_magi.main()
    assert ei.value.code == 0
    assert written.get("p") == "X"


def test_ollama_init_file_exists_exits_0(monkeypatch, capsys):
    """--ollama-init exits 0 (not 1) when config already exists (FileExistsError)."""
    import run_magi

    def _fake_write_template(**k: object) -> str:
        raise FileExistsError(".claude/magi-ollama.toml")

    monkeypatch.setattr(sys, "argv", ["run_magi.py", "--ollama-init"])
    monkeypatch.setattr(run_magi, "write_template", _fake_write_template)
    with pytest.raises(SystemExit) as ei:
        run_magi.main()
    assert ei.value.code == 0


def test_ollama_skips_claude_which_gate(monkeypatch, tmp_path):
    """--ollama must not abort when 'claude' is absent from PATH."""
    import run_magi

    monkeypatch.setattr(sys, "argv", ["run_magi.py", "design", "hello", "--ollama", "--no-status"])
    monkeypatch.setattr(run_magi.shutil, "which", lambda _: None)  # claude absent
    monkeypatch.setattr(run_magi, "resolve_config", lambda **k: _make_ollama_cfg())
    monkeypatch.setattr(run_magi, "preflight", _preflight_ok)

    captured: dict[str, object] = {}

    async def _fake_orch(*a: object, **k: object) -> dict[str, object]:
        captured["backend"] = k.get("backend")
        # Return a minimal-but-valid report shape so main() can run to completion
        # without crashing in format_report / consensus / cost paths.
        raise SystemExit(0)

    monkeypatch.setattr(run_magi, "run_orchestrator", _fake_orch)
    # main must NOT sys.exit(1) on missing claude when --ollama is set
    try:
        run_magi.main()
    except SystemExit as e:
        assert e.code != 1, f"Expected not exit(1), got exit({e.code})"
    from ollama_backend import OllamaBackend

    assert isinstance(captured["backend"], OllamaBackend)


def test_retry_feedback_truncates_a_huge_error():
    # BDD-45: the error is bounded so the retry prompt cannot grow without limit.
    from model_context import MAX_ERROR_CHARS
    from run_magi import _build_retry_prompt
    from validate import ValidationError

    err = ValidationError("x" * 10_000)
    out = _build_retry_prompt("PROMPT", err)
    assert len(out) < len("PROMPT") + MAX_ERROR_CHARS + 600
    assert "..." in out


def test_retry_feedback_bound_holds_for_NON_ASCII_errors():
    # Truncating CHARS does not bound TOKENS. Exercise the TRUE worst case: an emoji
    # is 4 UTF-8 bytes, and a byte-level BPE emits up to one token per byte (C2-3).
    from model_context import MAX_RETRY_FEEDBACK_TOKENS
    from run_magi import _build_retry_prompt
    from validate import ValidationError

    err = ValidationError("\U0001f525" * 10_000)
    block = _build_retry_prompt("", err)
    # UTF-8 byte count is a strict upper bound on tokens for any byte-level BPE.
    worst_case_tokens = len(block.encode("utf-8"))
    assert worst_case_tokens <= MAX_RETRY_FEEDBACK_TOKENS


@pytest.mark.parametrize(
    "exc,expected",
    [
        (ValidationError("missing keys"), "schema"),
        (json.JSONDecodeError("boom", "{", 0), "schema"),
        (TimeoutError(), "timeout"),
        (asyncio.TimeoutError(), "timeout"),
        # HTTPError IS a subclass of URLError -- check it FIRST.
        (urllib.error.HTTPError("u", 500, "err", Message(), None), "http"),
        (urllib.error.HTTPError("u", 429, "rate limited", Message(), None), "http"),
        # A socket timeout arrives WRAPPED: URLError(TimeoutError()).
        (urllib.error.URLError(TimeoutError()), "timeout"),
        (urllib.error.URLError(ConnectionRefusedError()), "connection"),
        (ConnectionRefusedError(), "connection"),
        (ConnectionResetError(), "connection"),
        # Backend-mapped transport RuntimeErrors: classified by MESSAGE signature.
        (RuntimeError("HTTP 503 Service Unavailable"), "http"),
        (RuntimeError("Ollama 404 at chat-time: model unavailable"), "http"),
        (
            RuntimeError("Cannot reach Ollama at http://h:11434: [Errno 111] Connection refused"),
            "connection",
        ),
        # A RuntimeError with NO transport signature is OUR bug -> "unexpected".
        (RuntimeError("assert self._pid is not None"), "unexpected"),
        (TypeError("a bug in OUR code"), "unexpected"),
    ],
)
def test_classify_pins_every_transport_variant(exc, expected):
    """_classify decides the SCOPE of a failure and whether the fast-fail fires."""
    from run_magi import _classify

    assert _classify(exc) == expected


def test_non_transport_runtimeerror_is_unexpected_not_transport():
    """A generic RuntimeError (our bug) must NOT masquerade as transport."""
    from run_magi import _FAIL_CONNECTION, _FAIL_HTTP, _FAIL_TIMEOUT, _FAIL_UNEXPECTED, _classify

    label = _classify(RuntimeError("some internal bug -- no HTTP status, no socket"))
    assert label == _FAIL_UNEXPECTED
    assert label not in (_FAIL_HTTP, _FAIL_CONNECTION, _FAIL_TIMEOUT)
    assert _classify(RuntimeError("Ollama HTTP 500: Internal Server Error")) == _FAIL_HTTP
    assert (
        _classify(
            RuntimeError("Cannot reach Ollama at http://h:11434: [Errno 111] Connection refused")
        )
        == _FAIL_CONNECTION
    )


def test_classify_matches_the_real_ollama_backend_messages():
    """CONTRACT: the markers must match the VERBATIM strings ollama_backend._call raises."""
    from run_magi import _FAIL_CONNECTION, _FAIL_HTTP, _FAIL_TIMEOUT, _classify

    http_error = RuntimeError("Ollama HTTP 503: Service Unavailable")
    chat_time_404 = RuntimeError(
        "Ollama 404 at chat-time: model unavailable (Not Found). "
        "Preflight passed -- possible ollama rm / auth expiry / TOCTOU."
    )
    unreachable = RuntimeError(
        "Cannot reach Ollama at http://nas:11434: [Errno 111] Connection refused"
    )
    timed_out = TimeoutError("Ollama request timed out: timed out")
    assert _classify(http_error) == _FAIL_HTTP
    assert _classify(chat_time_404) == _FAIL_HTTP
    assert _classify(unreachable) == _FAIL_CONNECTION
    assert _classify(timed_out) == _FAIL_TIMEOUT


# ---------------------------------------------------------------------------
# Task 9: two-level attempt loop (R1/R2/R12) -- attempts, retry scope, backoff
# ---------------------------------------------------------------------------


class TestAttemptsPerModel:
    """R1/R2/R12: attempts, retry scope and backoff for ONE active model."""

    @pytest.mark.asyncio
    async def test_transport_failure_is_now_retried(self, tmp_path):
        """v4 never retried transport -- a 503 killed the mage. R2 changes that."""
        calls = {"caspar": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name != "caspar":
                return _valid(agent_name)
            calls["caspar"] += 1
            if calls["caspar"] == 1:
                raise RuntimeError("HTTP 503 Service Unavailable")
            return _valid(agent_name)

        result = await _run(tmp_path, mock_launch, rotation=_rotation())

        assert calls["caspar"] == 2, "a transport failure must be retried, not fatal"
        assert result.get("degraded") is not True
        assert len(result["agents"]) == 3

    @pytest.mark.asyncio
    async def test_timeout_on_first_attempt_is_retried_not_terminal(self, tmp_path):
        """BDD-20 (integration): a timeout was TERMINAL in v4. R2 makes it a failed
        ATTEMPT, so the SAME model gets its retry."""
        calls = {"melchior": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name != "melchior":
                return _valid(agent_name)
            calls["melchior"] += 1
            if calls["melchior"] == 1:
                raise TimeoutError("Ollama request timed out")
            return _valid(agent_name)

        result = await _run(tmp_path, mock_launch, rotation=_rotation())

        assert calls["melchior"] == 2, (
            "a timeout must be RETRIED on the same model, not terminate the mage -- "
            "if this is 1, timeouts went terminal again (the v4 regression)"
        )
        assert result.get("degraded") is not True
        assert len(result["agents"]) == 3

    @pytest.mark.asyncio
    async def test_schema_retry_carries_feedback_but_transport_retry_does_not(self, tmp_path):
        """BDD-4: the model that ANSWERED needs correction; the one that never
        answered needs a pause, not a lecture."""
        prompts = {"caspar": [], "melchior": []}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name == "balthasar":
                return _valid(agent_name)
            prompts[agent_name].append(prompt)
            if len(prompts[agent_name]) == 1:
                if agent_name == "caspar":
                    raise ValidationError("missing keys: ['recommendation']")
                raise RuntimeError("HTTP 503 Service Unavailable")
            return _valid(agent_name)

        await _run(tmp_path, mock_launch, rotation=_rotation())

        assert "---RETRY-FEEDBACK---" in prompts["caspar"][1], "schema retry must correct"
        assert "recommendation" in prompts["caspar"][1], "it must cite the actual defect"
        assert "---RETRY-FEEDBACK---" not in prompts["melchior"][1], (
            "a transport retry must resend the ORIGINAL prompt: the model never "
            "answered, so there is nothing to correct"
        )

    @pytest.mark.asyncio
    async def test_backoff_waits_between_transport_attempts_only(self, tmp_path, monkeypatch):
        """BDD-35: waiting helps a rate-limited server; it does not help a model
        that produced malformed JSON -- there, the feedback is the fix."""
        import run_magi

        sleeps = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr(run_magi.asyncio, "sleep", fake_sleep)

        calls = {"caspar": 0, "melchior": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name == "balthasar":
                return _valid(agent_name)
            calls[agent_name] += 1
            if calls[agent_name] == 1:
                if agent_name == "caspar":
                    raise RuntimeError("HTTP 429 Too Many Requests")
                raise ValidationError("bad json")
            return _valid(agent_name)

        await _run(tmp_path, mock_launch, rotation=_rotation())

        assert sleeps == [2.0], (
            "exactly one backoff: the transport retry (caspar). The schema retry "
            f"(melchior) must not sleep. Got {sleeps}"
        )

    @pytest.mark.asyncio
    async def test_launch_agent_receives_a_ModelSpec_not_a_bare_tag(self, tmp_path):
        """The signature change, pinned (finding by Melchior, Checkpoint 2). The
        rotation path condemns a LINEAGE on failure, so it must reach the call site."""
        seen = []

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            seen.append(spec)
            return _valid(agent_name)

        await _run(tmp_path, mock_launch, rotation=_rotation())

        assert all(isinstance(s, ModelSpec) for s in seen), f"got {[type(s) for s in seen]}"
        assert {s.lineage for s in seen} == {"alibaba", "moonshot", "deepseek"}

    @pytest.mark.asyncio
    async def test_local_model_tags_get_no_special_treatment(self, tmp_path):
        """BDD-21: a local tag is just a tag. The declared lineage governs the skip;
        there is NO branch anywhere that asks "is this cloud or local?"."""
        local_trio = {
            "melchior": ModelSpec("qwen3:14b", "alibaba"),
            "balthasar": ModelSpec("gpt-oss:20b", "openai"),
            "caspar": ModelSpec("deepseek-v4-pro:cloud", "deepseek"),
        }

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            return _valid(agent_name)

        from run_magi import run_orchestrator

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="t",
                output_dir=str(tmp_path),
                timeout=300,
                agent_models=local_trio,
                rotation=_rotation(),
                show_status=False,
            )
        assert len(result["agents"]) == 3
        assert result.get("degraded") is not True

    @pytest.mark.asyncio
    async def test_claude_path_without_rotation_keeps_v4_behaviour(self, tmp_path):
        """BDD-19: rotation=None => single-shot schema retry, transport still fatal."""
        calls = {"caspar": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name != "caspar":
                return _valid(agent_name)
            calls["caspar"] += 1
            raise RuntimeError("HTTP 503 Service Unavailable")

        from run_magi import run_orchestrator

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="t",
                output_dir=str(tmp_path),
                timeout=300,
                model="opus",
                show_status=False,
            )

        assert calls["caspar"] == 1, "v4 contract: transport failures are NOT retried"
        assert result["degraded"] is True
        assert len(result["agents"]) == 2


# ---------------------------------------------------------------------------
# Task 10: rotation propose-verify-commit (R5/R8/R24) + failure routing (R13)
# ---------------------------------------------------------------------------


class TestRotationProposeVerifyCommit:
    """R24: propose under the lock, VERIFY with a probe outside it, then commit."""

    @pytest.mark.asyncio
    async def test_mage_rotates_after_exhausting_attempts_and_still_votes(self, tmp_path):
        """BDD-2/BDD-11: a dead model no longer costs a mage, and the run is VALID
        (not degraded), because the fallback was DECLARED."""
        seen: list[str] = []

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name != "caspar":
                return _valid(agent_name)
            seen.append(spec.model)
            if spec.model == "deepseek-v4-pro:cloud":
                raise RuntimeError("HTTP 503 Service Unavailable")
            return _valid(agent_name)

        result = await _run(tmp_path, mock_launch, rotation=_rotation())

        assert seen == ["deepseek-v4-pro:cloud", "deepseek-v4-pro:cloud", "glm-5.2:cloud"]
        assert result.get("degraded") is not True, (
            "a declared fallback keeps the run VALID -- that is the whole feature"
        )
        assert len(result["agents"]) == 3

    @pytest.mark.asyncio
    async def test_a_schema_failure_also_reaches_rotation(self, tmp_path):
        """BDD-3: rotation is failure-type AGNOSTIC. A mage that exhausts its attempts
        with SCHEMA failures (the model answered but never satisfied the contract)
        must rotate just the same and vote."""
        seen: list[str] = []

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name != "caspar":
                return _valid(agent_name)
            seen.append(spec.model)
            if spec.model == "deepseek-v4-pro:cloud":
                raise ValidationError("missing keys: ['findings']")  # SCHEMA, not transport
            return _valid(agent_name)

        result = await _run(tmp_path, mock_launch, rotation=_rotation())

        assert seen == ["deepseek-v4-pro:cloud", "deepseek-v4-pro:cloud", "glm-5.2:cloud"]
        assert result.get("degraded") is not True, "the schema path reaches rotation too"
        assert len(result["agents"]) == 3

    @pytest.mark.asyncio
    async def test_max_attempts_per_model_is_never_exceeded(self, tmp_path):
        """Two attempts per model, then the model is spent -- never a third."""
        per_model: dict[str, int] = {}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name != "caspar":
                return _valid(agent_name)
            per_model[spec.model] = per_model.get(spec.model, 0) + 1
            raise RuntimeError("HTTP 503 Service Unavailable")

        await _run(tmp_path, mock_launch, rotation=_rotation(max_attempts=2, max_rotations=1))

        assert per_model["deepseek-v4-pro:cloud"] == 2, "trio model: exactly max_attempts"
        assert per_model["glm-5.2:cloud"] == 2, "fallback: a FULL fresh budget"
        assert len(per_model) == 2, "1 + max_rotations models, no more"

    def test_fallback_reason_kind_is_always_an_R13_enum_value(self):
        """R13: telemetry carries {transport, schema, timeout} ONLY. The internal
        connection/http distinction must NEVER leak into fallback_reason.kind."""
        from run_magi import (
            _FAIL_CONNECTION,
            _FAIL_HTTP,
            _FAIL_SCHEMA,
            _FAIL_TIMEOUT,
            _AttemptsExhausted,
            _reason,
        )

        old = ModelSpec("deepseek-v4-pro:cloud", "deepseek")
        new = ModelSpec("glm-5.2:cloud", "zhipu")
        for fail_kind in (_FAIL_CONNECTION, _FAIL_HTTP, _FAIL_TIMEOUT, _FAIL_SCHEMA):
            exc = _AttemptsExhausted(fail_kind, "boom", http_status=None, attempts=2)
            reason = _reason(old, new, exc, AgentRotationState(rotations_done=1))
            assert reason["kind"] in ("transport", "schema", "timeout"), fail_kind

    @pytest.mark.asyncio
    async def test_probe_rejects_a_candidate_that_does_not_fit_and_we_repropose(self, tmp_path):
        """BDD-54: the pre-filter PROPOSED it; the probe MEASURED it; the probe wins."""
        probed: list[str] = []

        async def probe(model: str, prompt: str, timeout: int) -> int | None:
            probed.append(model)
            return 10_000_000 if model == "glm-5.2:cloud" else REQUIRED

        seen: list[str] = []

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name != "caspar":
                return _valid(agent_name)
            seen.append(spec.model)
            if spec.model == "deepseek-v4-pro:cloud":
                raise RuntimeError("HTTP 503 Service Unavailable")
            return _valid(agent_name)

        result = await _run(tmp_path, mock_launch, rotation=_rotation(probe=probe))

        assert probed == ["glm-5.2:cloud", "gpt-oss:120b-cloud"]
        assert "glm-5.2:cloud" not in seen, "a model that would truncate must NEVER run"
        assert seen[-1] == "gpt-oss:120b-cloud"
        assert result.get("degraded") is not True

    @pytest.mark.asyncio
    async def test_no_io_call_ever_holds_the_registry_lock(self, tmp_path):
        """Caspar's deadlock CRITICAL, made executable: no I/O under the registry lock."""
        ctx = _rotation()
        violations: list[str] = []

        async def probe(model: str, prompt: str, timeout: int) -> int | None:
            if ctx.registry._lock.locked():
                violations.append(f"probe({model})")
            return REQUIRED

        ctx = _rotation(probe=probe)

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name == "caspar" and spec.lineage == "deepseek":
                raise RuntimeError("HTTP 503 Service Unavailable")
            return _valid(agent_name)

        await _run(tmp_path, mock_launch, rotation=ctx)

        assert not violations, f"I/O executed while holding the registry lock: {violations}"

    @pytest.mark.asyncio
    async def test_probe_runs_outside_the_registry_lock(self, tmp_path):
        """BDD-55: the probe must never hold the registry lock."""
        ctx = _rotation()
        held_during_probe: list[bool] = []

        async def probe(model: str, prompt: str, timeout: int) -> int | None:
            held_during_probe.append(ctx.registry._lock.locked())
            return REQUIRED

        ctx = _rotation(probe=probe)

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name == "caspar" and spec.model == "deepseek-v4-pro:cloud":
                raise RuntimeError("HTTP 503 Service Unavailable")
            return _valid(agent_name)

        await _run(tmp_path, mock_launch, rotation=ctx)

        assert held_during_probe, "the probe must have run at least once"
        assert not any(held_during_probe), "the registry lock was held during a network call"

    @pytest.mark.asyncio
    async def test_attempt_counter_resets_on_rotation(self, tmp_path):
        """BDD-10: the new model gets a FULL budget, not the leftovers of the old."""
        per_model: dict[str, int] = {}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name != "caspar":
                return _valid(agent_name)
            per_model[spec.model] = per_model.get(spec.model, 0) + 1
            if spec.model == "deepseek-v4-pro:cloud":
                raise RuntimeError("HTTP 503 Service Unavailable")
            if per_model[spec.model] == 1:
                raise RuntimeError("HTTP 503 Service Unavailable")  # fallback fails ONCE
            return _valid(agent_name)

        result = await _run(tmp_path, mock_launch, rotation=_rotation())

        assert per_model == {"deepseek-v4-pro:cloud": 2, "glm-5.2:cloud": 2}
        assert result.get("degraded") is not True, "the fallback's 2nd attempt succeeded"

    @pytest.mark.asyncio
    async def test_max_probe_attempts_bounds_the_propose_verify_loop(self, tmp_path):
        """A stale window cache must not turn a correctness guard into a latency hole:
        the loop is bounded, and exhausting it kills the mage cleanly."""
        probed: list[str] = []

        async def probe(model: str, prompt: str, timeout: int) -> int | None:
            probed.append(model)
            return 10_000_000  # NOTHING fits

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name != "caspar":
                return _valid(agent_name)
            raise RuntimeError("HTTP 503 Service Unavailable")

        result = await _run(
            tmp_path,
            mock_launch,
            rotation=_rotation(probe=probe, max_probe_attempts=3, max_rotations=5),
        )

        assert len(probed) == 3, f"bounded by max_probe_attempts, got {len(probed)}"
        assert result["degraded"] is True, "no fitting candidate -> the mage dies"
        assert len(result["agents"]) == 2, "degraded mode still synthesizes with 2"


# ---------------------------------------------------------------------------
# Task 11: failure semantics (R5a) + fast-fail on a dead endpoint (R15)
# ---------------------------------------------------------------------------


class TestFailureSemanticsAndFastFail:
    """R5a: a failure is global or local by its NATURE, not by who suffered it."""

    @pytest.mark.asyncio
    async def test_transport_failure_condemns_the_lineage_for_every_mage(self, tmp_path):
        """BDD-30: if the glm-5.2 (zhipu) fallback is down for Melchior, it is down
        for Balthasar too -- never handed a lineage already condemned run-wide."""
        ctx = _rotation()
        used: dict[str, list[str]] = {"melchior": [], "balthasar": [], "caspar": []}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            used[agent_name].append(spec.model)
            if agent_name == "caspar":
                return _valid(agent_name)
            if spec.lineage in ("alibaba", "moonshot", "zhipu"):
                raise RuntimeError("HTTP 503 Service Unavailable")
            return _valid(agent_name)

        result = await _run(tmp_path, mock_launch, rotation=ctx)

        assert "zhipu" in ctx.registry.run_failed_lineages
        assert used["balthasar"].count("glm-5.2:cloud") <= 2, (
            "balthasar may have raced melchior into zhipu at most once; it must "
            "never be handed a lineage already condemned run-wide"
        )
        assert any(m == "gpt-oss:120b-cloud" for m in used["balthasar"])
        assert result.get("degraded") is not True

    @pytest.mark.asyncio
    async def test_schema_failure_stays_local_to_the_mage(self, tmp_path):
        """BDD-31: the model was ALIVE and answered -- condemning the lineage run-wide
        would throw away a model that works fine for the other two."""
        ctx = _rotation()

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name == "caspar" and spec.lineage == "deepseek":
                raise ValidationError("missing keys: ['findings']")
            return _valid(agent_name)

        result = await _run(tmp_path, mock_launch, rotation=ctx)

        assert ctx.registry.run_failed_lineages == set(), (
            "a schema failure must NEVER be globalized"
        )
        assert result.get("degraded") is not True

    @pytest.mark.asyncio
    async def test_two_connection_refused_lineages_abort_the_run_at_once(self, tmp_path):
        """BDD-40: rotating is pointless if what died is the SERVER."""
        attempts = {"n": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            attempts["n"] += 1
            raise ConnectionRefusedError("[Errno 111] Connection refused")

        with pytest.raises(RuntimeError, match="endpoint"):
            await _run(tmp_path, mock_launch, rotation=_rotation())

        assert attempts["n"] <= 6, (
            "the abort must be FAST: once 2 distinct lineages refuse the connection, "
            "the shared endpoint_down Event stops the siblings before they spend "
            f"their own budgets on the same dead server (got {attempts['n']} attempts; "
            "without the Event this would be 18)"
        )

    @pytest.mark.asyncio
    async def test_an_http_500_storm_does_NOT_trigger_the_fast_fail(self, tmp_path):
        """BDD-41: a 500 means SOMEONE answered -- the endpoint is alive and the next
        attempt may succeed. Aborting would kill healthy runs a backoff would save."""

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if spec.lineage in ("alibaba", "moonshot", "deepseek"):
                raise RuntimeError("HTTP 500 Internal Server Error")
            return _valid(agent_name)  # every fallback works

        result = await _run(tmp_path, mock_launch, rotation=_rotation())

        assert result["consensus"] is not None, "the run must continue and rotate"
        assert result.get("degraded") is not True
        assert len(result["agents"]) == 3


# ---------------------------------------------------------------------------
# Task 12: fault injection on the async paths (the paths a happy test never walks)
# ---------------------------------------------------------------------------


class TestRotationFaultInjection:
    """The paths a happy test never walks -- where all three gate bugs lived."""

    @pytest.mark.asyncio
    async def test_a_failed_status_display_after_the_verdict_does_not_cost_the_verdict(
        self, tmp_path
    ):
        """The 4th broad catch: the post-verdict status-display update is BEST-EFFORT.
        A UI glitch on the "success" update must be SWALLOWED: the run completes, the
        verdict stands, and the lineage stays claimed."""
        import run_magi

        ctx = _rotation()

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            return _valid(agent_name)

        def raise_only_on_success(display, name, state, log_gate):
            if state == "success":
                raise RuntimeError("status display died after the verdict")

        with patch.object(run_magi, "_safe_display_update", side_effect=raise_only_on_success):
            result = await _run(tmp_path, mock_launch, rotation=ctx)  # does NOT raise

        assert len(result["agents"]) == 3, "a display glitch must never drop a verdict"
        assert result.get("degraded") is not True
        assert "deepseek" in await ctx.registry.lineages_in_play(exclude=None), (
            "caspar emitted a valid verdict; its lineage stays claimed even though the "
            "display update failed afterwards"
        )

    @pytest.mark.asyncio
    async def test_a_late_exception_in_teardown_propagates_without_releasing_the_lineage(
        self, tmp_path
    ):
        """A GENUINE late exception reaching agent_slot.__aexit__ with succeeded=True must
        PROPAGATE yet still CONSERVE the lineage: succeeded is the sole determinant."""
        from fallback_policy import LineageRegistry

        reg = LineageRegistry(TRIO)
        state = AgentRotationState()

        with pytest.raises(RuntimeError, match="genuine teardown bug"):
            async with reg.agent_slot("caspar", state):
                state.succeeded = True  # a valid verdict exists
                raise RuntimeError("genuine teardown bug after the verdict")

        assert "deepseek" in await reg.lineages_in_play(exclude=None), (
            "succeeded=True conserves the lineage even though a real exception "
            "propagated from teardown"
        )

    @pytest.mark.asyncio
    async def test_concurrent_claims_are_serialised_even_with_injected_delays(self, tmp_path):
        """Caspar's TOCTOU objection, made falsifiable: force the interleaving and assert
        the invariant survives it."""
        from fallback_policy import LineageRegistry

        registry = LineageRegistry(TRIO)
        policy = _rotation().policy
        real_next = policy.next_model
        claims: list[str] = []

        def slow_next(*args, **kwargs):
            """Pure, but slow: widen the read-decide-commit window."""
            result = real_next(*args, **kwargs)
            if result:
                claims.append(result.lineage)
            return result

        policy.next_model = slow_next  # type: ignore[method-assign]

        s1, s2 = AgentRotationState(), AgentRotationState()
        got = await asyncio.gather(
            registry.claim_next("melchior", policy, s1),
            registry.claim_next("balthasar", policy, s2),
        )

        assert all(g is not None for g in got)
        assert got[0].lineage != got[1].lineage, (
            f"two mages reserved the same lineage under concurrency: {claims}"
        )

    @pytest.mark.asyncio
    async def test_two_mages_rotating_concurrently_get_distinct_lineages(self, tmp_path):
        """The cycle-1 TOCTOU: read-decide-commit must be atomic, or both mages pick the
        same lineage and the consensus only LOOKS like 3 perspectives."""
        ctx = _rotation()
        assigned: dict[str, str] = {}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name == "caspar":
                return _valid(agent_name)
            if spec.lineage in ("alibaba", "moonshot"):  # both trio models die
                await asyncio.sleep(0)  # force interleaving
                raise RuntimeError("HTTP 503 Service Unavailable")
            assigned[agent_name] = spec.lineage
            return _valid(agent_name)

        result = await _run(tmp_path, mock_launch, rotation=ctx)

        assert len(assigned) == 2
        assert len(set(assigned.values())) == 2, f"two mages landed on the same lineage: {assigned}"
        assert result.get("degraded") is not True

    @pytest.mark.asyncio
    async def test_cancellation_mid_run_releases_the_lineage(self, tmp_path):
        """CancelledError is a death, not a success: the slot must free the lineage."""
        ctx = _rotation()

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name == "caspar":
                raise asyncio.CancelledError()
            return _valid(agent_name)

        with pytest.raises((asyncio.CancelledError, RuntimeError)):
            await _run(tmp_path, mock_launch, rotation=ctx)

        assert "deepseek" not in await ctx.registry.lineages_in_play(exclude=None)

    @pytest.mark.asyncio
    async def test_a_probe_that_raises_reproposes_instead_of_killing_the_mage(self, tmp_path):
        """A probe is an accuracy optimisation. It must never be fatal: with the guard
        non-strict, an unmeasurable candidate is accepted (loudly)."""

        async def probe(model: str, prompt: str, timeout: int) -> int | None:
            raise urllib.error.HTTPError("u", 500, "boom", Message(), None)

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name == "caspar" and spec.lineage == "deepseek":
                raise RuntimeError("HTTP 503 Service Unavailable")
            return _valid(agent_name)

        result = await _run(tmp_path, mock_launch, rotation=_rotation(probe=probe))

        assert result.get("degraded") is not True, "a failed probe must not kill the mage"

    @pytest.mark.asyncio
    async def test_degraded_mode_survives_a_mage_that_exhausts_its_rotations(self, tmp_path):
        """BDD-7 / R7: rotation does not replace degraded mode -- it postpones it."""

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name != "caspar":
                return _valid(agent_name)
            raise RuntimeError("HTTP 503 Service Unavailable")  # every model fails

        result = await _run(tmp_path, mock_launch, rotation=_rotation(max_rotations=2))

        assert result["degraded"] is True
        assert len(result["agents"]) == 2
        assert result["consensus"] is not None, "2 agents still synthesize (v2.x minimum)"

    @pytest.mark.asyncio
    async def test_truncated_OUTPUT_is_loud_not_silent(self, tmp_path):
        """BDD-36: a truncated OUTPUT breaks the 7-key JSON -> JSONDecodeError -> a failed
        attempt -> retry -> rotation. It is LOUD, and lands in the fail-closed path."""
        calls = {"caspar": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name != "caspar":
                return _valid(agent_name)
            calls["caspar"] += 1
            if calls["caspar"] == 1:
                raise json.JSONDecodeError("Expecting ',' delimiter", '{"agent": "cas', 14)
            return _valid(agent_name)

        result = await _run(tmp_path, mock_launch, rotation=_rotation())

        assert calls["caspar"] == 2, "a truncated output is a schema failure -> retried"
        assert result.get("degraded") is not True, "and the retry recovered it"

    @pytest.mark.asyncio
    async def test_rotation_never_opens_a_fourth_concurrent_request(self, tmp_path):
        """BDD-18 / NR2: Ollama's Pro cap is 3 agents. Rotation is SEQUENTIAL within a
        mage: it must never add a slot."""
        in_flight = {"now": 0, "max": 0}

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            in_flight["now"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["now"])
            try:
                await asyncio.sleep(0)
                if spec.lineage == "deepseek":
                    raise RuntimeError("HTTP 503 Service Unavailable")
                return _valid(agent_name)
            finally:
                in_flight["now"] -= 1

        await _run(tmp_path, mock_launch, rotation=_rotation())

        assert in_flight["max"] <= 3, f"opened {in_flight['max']} concurrent requests"


# ---------------------------------------------------------------------------
# Task 13: noisy telemetry (R9/R13/R16/NR3b) -- no silent fallback ever
# ---------------------------------------------------------------------------


def _preflight(context_guard, *, deltas=(), warnings=()):
    """A PreflightResult carrying the telemetry the report reads from it (T8)."""
    from ollama_preflight import PreflightResult

    return PreflightResult(
        capabilities={},
        min_window_tokens=REQUIRED,
        required_tokens=REQUIRED,
        context_guard=context_guard,
        lineage_warnings=list(warnings),
        fallback=tuple(FALLBACK),
        token_estimate_delta=list(deltas),
    )


async def _rotated_report(tmp_path, *, preflight=None, api_key=None, secret_in_error=False):
    """Drive a REAL caspar rotation (deepseek-v4-pro -> glm-5.2) and return the report."""
    from dataclasses import replace

    async def mock_launch(
        agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
    ):
        if agent_name != "caspar":
            return _valid(agent_name)
        if spec.model == "deepseek-v4-pro:cloud":
            msg = "HTTP 503 Service Unavailable"
            if secret_in_error:
                msg += " (backend echoed auth token=sk-supersecret into the body)"
            raise RuntimeError(msg)
        return _valid(agent_name)

    rotation = _rotation()
    if preflight is not None:
        rotation = replace(rotation, preflight=preflight)
    if api_key is not None:
        rotation = replace(rotation, config=replace(rotation.config, api_key=api_key))
    return await _run(tmp_path, mock_launch, rotation=rotation)


def _caspar(report):
    """The rotated mage's agent entry -- located by NAME, never by list position."""
    return next(a for a in report["agents"] if a["agent"] == "caspar")


class TestTelemetrySurfaces:
    """R9/R13/R16: every rotation is visible on stderr, in the banner, and in the report."""

    @pytest.mark.asyncio
    async def test_rotation_is_announced_on_stderr_with_its_cause(self, tmp_path, capsys):
        """R9/BDD-12: a fallback is NEVER silent -- stderr names the new model."""
        await _rotated_report(tmp_path)
        assert "rotating to glm-5.2:cloud" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_report_carries_model_configured_used_rotations_and_reason(self, tmp_path):
        """R13/BDD-12: the rotated mage's telemetry is complete AND correctly scoped."""
        report = await _rotated_report(tmp_path)

        caspar = _caspar(report)
        assert caspar["model_configured"] == "deepseek-v4-pro:cloud"
        assert caspar["model_used"] == "glm-5.2:cloud"
        assert caspar["rotations"] == 1
        assert caspar["fallback_reason"]["kind"] == "transport"
        assert caspar["fallback_reason"]["from_model"] == "deepseek-v4-pro:cloud"
        assert caspar["fallback_reason"]["to_model"] == "glm-5.2:cloud"

        melchior = next(a for a in report["agents"] if a["agent"] == "melchior")
        assert melchior["model_configured"] == melchior["model_used"], "no rotation => equal"
        assert melchior["rotations"] == 0
        assert melchior["fallback_reason"] is None

    @pytest.mark.asyncio
    async def test_model_used_is_the_tag_string_not_a_serialized_ModelSpec(self, tmp_path):
        """R13 (a): the report stores the .model TAG, not the ModelSpec (not JSON-serialisable)."""
        caspar = _caspar(await _rotated_report(tmp_path))
        assert isinstance(caspar["model_used"], str)
        assert isinstance(caspar["model_configured"], str)
        assert not isinstance(caspar["model_used"], ModelSpec)
        assert caspar["model_used"] == "glm-5.2:cloud"

    @pytest.mark.asyncio
    async def test_fallback_agents_lists_exactly_the_rotated_mages(self, tmp_path):
        """R9 (b): the run-level roll-up names every mage that rotated -- and no other."""
        report = await _rotated_report(tmp_path)
        assert report["fallback_agents"] == ["caspar"]

    @pytest.mark.asyncio
    async def test_banner_marks_the_rotated_mage(self, tmp_path):
        """R9: the banner -- where the verdict's reader actually looks -- flags the swap."""
        from reporting import format_banner

        report = await _rotated_report(tmp_path)
        assert "[fallback: glm-5.2:cloud]" in format_banner(report)

    @pytest.mark.asyncio
    async def test_banner_renders_the_estimated_guard_and_lineage_warnings(self, tmp_path):
        """R16/R102: an ESTIMATED guard and any lineage warning must RENDER in the banner."""
        from reporting import format_banner

        warning = "deepseek-v4-pro declares 'deepseek' but no known pattern confirms it"
        report = await _rotated_report(
            tmp_path,
            preflight=_preflight("estimated", warnings=[warning]),
        )
        banner = format_banner(report)
        assert "estimated" in banner, "the reader must SEE the guard was not enforced (R16)"
        assert warning in banner, "the lineage warning must reach the banner (R102)"

    @pytest.mark.asyncio
    async def test_fallback_reason_is_the_structured_R13_dict_with_kind_in_the_enum(self, tmp_path):
        """R13 (d): fallback_reason is a STRUCTURED dict and its kind is an R13 enum member."""
        from run_magi import _KIND_SCHEMA, _KIND_TIMEOUT, _KIND_TRANSPORT

        reason = _caspar(await _rotated_report(tmp_path))["fallback_reason"]
        assert isinstance(reason, dict)
        assert {
            "kind",
            "from_model",
            "from_lineage",
            "to_model",
            "to_lineage",
            "detail",
            "http_status",
            "attempts",
        } <= set(reason)
        assert reason["kind"] in (_KIND_TRANSPORT, _KIND_SCHEMA, _KIND_TIMEOUT)
        assert reason["kind"] == _KIND_TRANSPORT, "a 503 is transport, not http/connection"

    @pytest.mark.asyncio
    async def test_context_guard_has_exactly_two_values(self, tmp_path):
        """R16/BDD-42: the field's domain is exactly {enforced, estimated}."""
        report = await _rotated_report(tmp_path)
        assert report["context_guard"] in ("enforced", "estimated")

    @pytest.mark.asyncio
    async def test_context_guard_is_enforced_when_measured_and_estimated_otherwise(self, tmp_path):
        """R16/R18 (c): the field's SEMANTICS -- enforced only when MEASURED, else estimated."""
        from ollama_preflight import CONTEXT_GUARD_ENFORCED, CONTEXT_GUARD_ESTIMATED

        enforced_report = await _rotated_report(
            tmp_path, preflight=_preflight(CONTEXT_GUARD_ENFORCED)
        )
        estimated_report = await _rotated_report(
            tmp_path, preflight=_preflight(CONTEXT_GUARD_ESTIMATED)
        )

        assert enforced_report["context_guard"] == "enforced"
        assert estimated_report["context_guard"] == "estimated"

    @pytest.mark.asyncio
    async def test_token_estimate_delta_is_reported(self, tmp_path):
        """R16/BDD-46: the estimate-vs-measured delta reaches the report per trio model."""
        from ollama_preflight import CONTEXT_GUARD_ENFORCED

        delta = {"agent": "melchior", "estimated": 13_827, "actual": 16_232, "error_pct": -14.8}
        report = await _rotated_report(
            tmp_path, preflight=_preflight(CONTEXT_GUARD_ENFORCED, deltas=[delta])
        )

        d = report["token_estimate_delta"][0]
        assert {"agent", "estimated", "actual", "error_pct"} <= set(d)
        assert d["actual"] == 16_232

    @pytest.mark.asyncio
    async def test_the_whole_report_round_trips_through_json_dumps(self, tmp_path):
        """R13 (e): the WHOLE report must be JSON-serialisable (no raw ModelSpec leaked)."""
        from ollama_preflight import CONTEXT_GUARD_ESTIMATED

        report = await _rotated_report(
            tmp_path,
            preflight=_preflight(
                CONTEXT_GUARD_ESTIMATED,
                deltas=[{"agent": "melchior", "estimated": 1, "actual": 2, "error_pct": 100.0}],
                warnings=["two mages look like the same lab"],
            ),
        )
        json.dumps(report)  # must NOT raise: no dataclass leaked into the report tree

    @pytest.mark.asyncio
    async def test_api_key_never_appears_in_any_error_surface(self, tmp_path, capsys):
        """NR3b: the api_key is redacted at the single boundary -- absent from BOTH the
        report JSON (fallback_reason detail) and stderr (the rotation notice)."""
        report = await _rotated_report(tmp_path, api_key="sk-supersecret", secret_in_error=True)

        blob = json.dumps(report) + capsys.readouterr().err
        assert "sk-supersecret" not in blob


def test_build_retry_prompt_redacts_the_api_key():
    """MAGI gate (Caspar, security): the retry prompt is written to {agent}.prompt.txt.
    If an error message ever carries the api_key, embedding it verbatim would leak it
    (NR3b: the key must appear on NO surface). Redact the error before embedding."""
    from run_magi import _build_retry_prompt
    from validate import ValidationError

    err = ValidationError("backend echoed auth token=sk-supersecret into the message")
    out = _build_retry_prompt("original prompt", err, api_key="sk-supersecret")
    assert "sk-supersecret" not in out


def test_classify_unwraps_socket_timeout_wrapped_in_urlerror():
    """MAGI gate (Caspar): a socket timeout arrives as URLError(socket.timeout()). It
    must classify as timeout, never connection, or two slow requests would trip the
    endpoint-down fast-fail on a reachable server (decisions #50/#98)."""
    import socket
    import urllib.error

    from run_magi import _FAIL_TIMEOUT, _classify

    assert _classify(urllib.error.URLError(socket.timeout())) == _FAIL_TIMEOUT


class TestContextGuardDowngradeOnRotation:
    """MAGI gate (Loop 1 pass 2): the run-level context_guard must not claim 'enforced'
    when a rotated mage ran on an estimated/unknown window -- R16 honesty on the
    highest-risk path."""

    @pytest.mark.asyncio
    async def test_guard_downgrades_when_a_rotated_mage_runs_unmeasured(self, tmp_path):
        async def probe(model, prompt, timeout):
            # glm's window is known (harness default), but its payload is UNMEASURABLE
            # (endpoint returned no usage) -> _rotate accepts it on the estimate.
            return None if model == "glm-5.2:cloud" else REQUIRED

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name != "caspar":
                return _valid(agent_name)
            if spec.model == "deepseek-v4-pro:cloud":
                raise RuntimeError("HTTP 503 Service Unavailable")
            return _valid(agent_name)

        result = await _run(tmp_path, mock_launch, rotation=_rotation(probe=probe))

        assert result.get("degraded") is not True, "the fallback still produced a verdict"
        assert result["context_guard"] == "estimated", (
            "caspar ran glm on an estimate -> the run was NOT fully enforced; the label "
            "must not keep claiming 'enforced' (preflight computed it from the trio only)"
        )

    @pytest.mark.asyncio
    async def test_guard_stays_enforced_when_the_rotated_mage_is_exactly_measured(self, tmp_path):
        """The downgrade is conditional: a rotation to an EXACTLY measured model keeps
        'enforced' -- the label must not over-report estimation either."""

        async def mock_launch(
            agent_name, agents_dir, prompt, output_dir, timeout, spec=None, backend=None
        ):
            if agent_name != "caspar":
                return _valid(agent_name)
            if spec.model == "deepseek-v4-pro:cloud":
                raise RuntimeError("HTTP 503 Service Unavailable")
            return _valid(agent_name)

        # default probe returns REQUIRED (an exact, fitting measurement) for glm too
        result = await _run(tmp_path, mock_launch, rotation=_rotation())

        assert result.get("degraded") is not True
        assert result["context_guard"] == "enforced", "glm was exactly measured and fits"
