# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-23
"""Tests for run_lock.py — process-liveness locking."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest


class TestIsPidAlive:
    """BDD-6 / BDD-7: liveness probe for a PID, cross-platform."""

    def test_current_process_is_alive(self):
        from run_lock import is_pid_alive

        assert is_pid_alive(os.getpid()) is True

    def test_finished_process_is_not_alive(self):
        """Best-effort native check. Deterministic dead-PID coverage lives in
        the mocked branch tests below; PID reuse can flip this on busy
        Windows CI, so a recycled PID skips rather than fails."""
        from run_lock import is_pid_alive

        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        result = is_pid_alive(proc.pid)
        if result:
            pytest.skip("PID recycled before probe; covered deterministically by mocked tests")
        assert result is False

    def test_nonpositive_pid_is_not_alive(self):
        from run_lock import is_pid_alive

        assert is_pid_alive(0) is False
        assert is_pid_alive(-1) is False

    def test_posix_branch_uses_os_kill(self, monkeypatch):
        """On a non-win32 platform the probe routes through os.kill."""
        import run_lock

        calls = {}

        def fake_kill(pid, sig):
            calls["pid"] = pid
            calls["sig"] = sig
            raise ProcessLookupError

        monkeypatch.setattr(run_lock.sys, "platform", "linux")
        monkeypatch.setattr(run_lock.os, "kill", fake_kill)
        assert run_lock.is_pid_alive(4242) is False
        assert calls == {"pid": 4242, "sig": 0}


class TestIsPidAliveWindowsBranch:
    """BDD-6/7 Windows branch, mocked so it runs on any platform.

    Each test injects a fake ``kernel32`` via ``ctypes.WinDLL`` (added with
    ``raising=False`` so it works on non-Windows CI where ``WinDLL`` is
    absent) and exercises ``_is_pid_alive_windows`` directly.
    """

    def _fake_kernel(self, monkeypatch, *, open_ret, wait_ret=0x00000102):
        import ctypes
        from unittest.mock import MagicMock

        fake = MagicMock()
        fake.OpenProcess.return_value = open_ret
        fake.WaitForSingleObject.return_value = wait_ret
        monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: fake, raising=False)
        return fake

    def test_alive_when_handle_open_and_wait_timeout(self, monkeypatch):
        import ctypes

        import run_lock

        fake = self._fake_kernel(monkeypatch, open_ret=0x1234, wait_ret=0x00000102)
        assert run_lock._is_pid_alive_windows(999) is True
        # Pin the HANDLE-truncation fix: restype/argtypes were declared.
        assert fake.OpenProcess.restype is ctypes.c_void_p
        assert fake.WaitForSingleObject.restype is ctypes.c_uint

    def test_dead_when_handle_open_and_wait_object_0(self, monkeypatch):
        import run_lock

        self._fake_kernel(monkeypatch, open_ret=0x1234, wait_ret=0x00000000)
        assert run_lock._is_pid_alive_windows(999) is False

    def test_alive_when_null_handle_access_denied(self, monkeypatch):
        import ctypes

        import run_lock

        self._fake_kernel(monkeypatch, open_ret=0)
        monkeypatch.setattr(ctypes, "get_last_error", lambda: 5)  # ERROR_ACCESS_DENIED
        assert run_lock._is_pid_alive_windows(999) is True

    def test_dead_when_null_handle_no_such_process(self, monkeypatch):
        import ctypes

        import run_lock

        self._fake_kernel(monkeypatch, open_ret=0)
        monkeypatch.setattr(ctypes, "get_last_error", lambda: 87)  # ERROR_INVALID_PARAMETER
        assert run_lock._is_pid_alive_windows(999) is False

    @pytest.mark.skipif(sys.platform != "win32", reason="real Win32 FFI probe")
    def test_real_windows_ffi_probe(self):
        """FFI-correctness pin (Mel iter-2): exercises the REAL ctypes path on
        Windows. The mocked tests above are shape/dispatch tripwires — they
        cannot catch a wrong restype/argtypes or a get_last_error mismatch.
        This one runs the actual OpenProcess/WaitForSingleObject FFI on the
        operator's win32 machine."""
        import run_lock

        assert run_lock._is_pid_alive_windows(os.getpid()) is True
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        result = run_lock._is_pid_alive_windows(proc.pid)
        if result:
            pytest.skip("PID recycled before probe")
        assert result is False

    def test_posix_permission_error_is_alive(self, monkeypatch):
        """POSIX: an existing-but-inaccessible process (PermissionError) is alive (R8)."""
        import run_lock

        def fake_kill(pid, sig):
            raise PermissionError

        monkeypatch.setattr(run_lock.sys, "platform", "linux")
        monkeypatch.setattr(run_lock.os, "kill", fake_kill)
        assert run_lock.is_pid_alive(4242) is True


class TestLockFile:
    """write/read/remove lifecycle of the .magi-lock file."""

    def test_write_then_read_returns_pid(self, tmp_path):
        from run_lock import LOCK_FILENAME, read_lock, write_lock

        write_lock(str(tmp_path))
        assert (tmp_path / LOCK_FILENAME).exists()
        assert read_lock(str(tmp_path)) == os.getpid()

    def test_read_missing_lock_returns_none(self, tmp_path):
        from run_lock import read_lock

        assert read_lock(str(tmp_path)) is None

    def test_read_corrupt_lock_returns_none(self, tmp_path):
        from run_lock import LOCK_FILENAME, read_lock

        (tmp_path / LOCK_FILENAME).write_text("not-a-pid\n", encoding="utf-8")
        assert read_lock(str(tmp_path)) is None

    def test_remove_lock_is_idempotent(self, tmp_path):
        from run_lock import LOCK_FILENAME, remove_lock, write_lock

        write_lock(str(tmp_path))
        remove_lock(str(tmp_path))
        assert not (tmp_path / LOCK_FILENAME).exists()
        remove_lock(str(tmp_path))  # second call must not raise


class TestLockTimestamp:
    """Pin the Python 3.9 fromisoformat invariant (Mel finding).

    write_lock emits ``datetime.now(timezone.utc).isoformat()`` (``+00:00``
    offset, never a ``Z`` suffix), which 3.9's restricted fromisoformat
    round-trips. Any unparseable timestamp must fail safe to ``age=None``
    (-> conservative live), not crash.
    """

    def test_isoformat_round_trips_to_finite_age(self, tmp_path):
        from run_lock import _parse_lock, write_lock

        write_lock(str(tmp_path))
        pid, age, _ = _parse_lock(str(tmp_path))
        assert pid == os.getpid()
        assert age is not None and age >= 0.0

    def test_unparseable_timestamp_degrades_to_none_age(self, tmp_path):
        from run_lock import LOCK_FILENAME, _parse_lock

        # A non-isoformat timestamp (version-independent: rejected on 3.9+).
        (tmp_path / LOCK_FILENAME).write_text(f"{os.getpid()}\nnot-a-timestamp\n", encoding="utf-8")
        pid, age, _ = _parse_lock(str(tmp_path))
        assert pid == os.getpid()
        assert age is None


class TestIsDirLive:
    """BDD-2/3/4/5/16: composed liveness decision."""

    def test_live_when_pid_alive_and_fresh(self, tmp_path):
        from run_lock import is_dir_live, write_lock

        write_lock(str(tmp_path))  # our own live PID, fresh timestamp
        assert is_dir_live(str(tmp_path)) is True

    def test_not_live_when_no_lock(self, tmp_path):
        from run_lock import is_dir_live

        assert is_dir_live(str(tmp_path)) is False

    def test_live_when_lock_corrupt(self, tmp_path):
        """BDD-5: an existing-but-unparseable lock is conservatively live."""
        from run_lock import LOCK_FILENAME, is_dir_live

        (tmp_path / LOCK_FILENAME).write_text("garbage\n", encoding="utf-8")
        assert is_dir_live(str(tmp_path)) is True

    def test_not_live_when_pid_dead(self, tmp_path, monkeypatch):
        """BDD-3/11: a lock whose PID is dead is eligible (not live)."""
        import run_lock
        from run_lock import write_lock

        write_lock(str(tmp_path))
        monkeypatch.setattr(run_lock, "is_pid_alive", lambda pid: False)
        assert run_lock.is_dir_live(str(tmp_path)) is False

    def test_not_live_when_pid_alive_but_stale(self, tmp_path):
        """BDD-16: PID alive but lock older than the threshold -> eligible."""
        from datetime import datetime, timedelta, timezone

        import run_lock
        from run_lock import LOCK_FILENAME, is_dir_live

        old = datetime.now(timezone.utc) - timedelta(seconds=run_lock.LOCK_STALE_AFTER_SECONDS + 60)
        # 2-line lock (no bound) -> floor LOCK_STALE_AFTER_SECONDS applies.
        (tmp_path / LOCK_FILENAME).write_text(
            f"{os.getpid()}\n{old.isoformat()}\n", encoding="utf-8"
        )
        assert is_dir_live(str(tmp_path)) is False

    def test_per_run_bound_protects_run_past_6h(self, tmp_path):
        """BDD-19 (F9 closed): a large persisted bound keeps a 7h-old live run."""
        from datetime import datetime, timedelta, timezone

        from run_lock import LOCK_FILENAME, is_dir_live

        seven_hours_ago = datetime.now(timezone.utc) - timedelta(hours=7)
        # bound 29400s (~8.17h) > 7h age -> still live (own/live PID).
        (tmp_path / LOCK_FILENAME).write_text(
            f"{os.getpid()}\n{seven_hours_ago.isoformat()}\n29400\n", encoding="utf-8"
        )
        assert is_dir_live(str(tmp_path)) is True

    def test_missing_bound_falls_back_to_floor(self, tmp_path):
        """BDD-20: a 2-line lock (no bound) uses LOCK_STALE_AFTER_SECONDS."""
        from datetime import datetime, timedelta, timezone

        from run_lock import LOCK_FILENAME, is_dir_live

        five_hours_ago = datetime.now(timezone.utc) - timedelta(hours=5)
        (tmp_path / LOCK_FILENAME).write_text(
            f"{os.getpid()}\n{five_hours_ago.isoformat()}\n", encoding="utf-8"
        )
        assert is_dir_live(str(tmp_path)) is True  # 5h < 6h floor

    def test_tiny_corrupt_bound_uses_floor(self, tmp_path):
        """Mel iter-3: a parseable but tiny/negative bound must not defeat the
        6h floor (R8 conservative bias)."""
        from datetime import datetime, timedelta, timezone

        from run_lock import LOCK_FILENAME, is_dir_live

        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        # bound=0 (corrupt-but-parseable); 1h age < 6h floor -> still live.
        (tmp_path / LOCK_FILENAME).write_text(
            f"{os.getpid()}\n{one_hour_ago.isoformat()}\n0\n", encoding="utf-8"
        )
        assert is_dir_live(str(tmp_path)) is True

    def test_not_live_when_age_exceeds_large_persisted_bound(self, tmp_path):
        """BDD-16 (non-None bound direction, Mel/Bal iter-4): alive but older
        than a large explicit bound -> not live (eligible)."""
        from datetime import datetime, timedelta, timezone

        from run_lock import LOCK_FILENAME, is_dir_live

        nine_hours_ago = datetime.now(timezone.utc) - timedelta(hours=9)
        # bound 29400s (~8.17h) < 9h age -> past bound -> not live.
        (tmp_path / LOCK_FILENAME).write_text(
            f"{os.getpid()}\n{nine_hours_ago.isoformat()}\n29400\n", encoding="utf-8"
        )
        assert is_dir_live(str(tmp_path)) is False


class TestStalenessBound:
    """BDD-21: staleness_bound_for_timeout = max(2*timeout+600, floor)."""

    def test_short_timeout_uses_floor(self):
        from run_lock import LOCK_STALE_AFTER_SECONDS, staleness_bound_for_timeout

        assert staleness_bound_for_timeout(900) == LOCK_STALE_AFTER_SECONDS

    def test_long_timeout_exceeds_floor(self):
        from run_lock import staleness_bound_for_timeout

        assert staleness_bound_for_timeout(14400) == 29400  # 2*14400 + 600

    def test_write_lock_round_trips_bound(self, tmp_path):
        """The bound passed to write_lock is persisted and parsed back (R2)."""
        from run_lock import _parse_lock, write_lock

        write_lock(str(tmp_path), 12345)
        _pid, _age, bound = _parse_lock(str(tmp_path))
        assert bound == 12345


class TestCorruptLockMtimeEscape:
    """Corrupt/empty lock dirs must age out via mtime escape, not leak forever."""

    def test_corrupt_lock_fresh_dir_is_live(self, tmp_path):
        """A garbage lock on a fresh dir is conservatively live (dir age < floor)."""
        from run_lock import LOCK_FILENAME, is_dir_live

        (tmp_path / LOCK_FILENAME).write_text("garbage\n", encoding="utf-8")
        # tmp_path mtime is now; dir_age is ~0 << LOCK_STALE_AFTER_SECONDS
        assert is_dir_live(str(tmp_path)) is True

    def test_corrupt_lock_old_dir_is_eligible(self, tmp_path):
        """A garbage lock on a stale dir is eligible (dir age >= floor)."""
        import run_lock
        from run_lock import LOCK_FILENAME, is_dir_live

        (tmp_path / LOCK_FILENAME).write_text("garbage\n", encoding="utf-8")
        # Back-date the dir mtime by more than LOCK_STALE_AFTER_SECONDS.
        old_mtime = time.time() - run_lock.LOCK_STALE_AFTER_SECONDS - 60
        os.utime(str(tmp_path), (old_mtime, old_mtime))
        assert is_dir_live(str(tmp_path)) is False

    def test_empty_lock_old_dir_is_eligible(self, tmp_path):
        """An empty lock file on a stale dir is also eligible."""
        import run_lock
        from run_lock import LOCK_FILENAME, is_dir_live

        (tmp_path / LOCK_FILENAME).write_text("", encoding="utf-8")
        old_mtime = time.time() - run_lock.LOCK_STALE_AFTER_SECONDS - 60
        os.utime(str(tmp_path), (old_mtime, old_mtime))
        assert is_dir_live(str(tmp_path)) is False

    def test_out_of_range_pid_corrupt_ts_old_dir_is_eligible(self, tmp_path):
        """Out-of-range PID + garbage timestamp + stale dir -> eligible (not live).

        is_pid_alive returns True (conservative) for huge PIDs, so without the
        age-None mtime-escape extension the lock keeps the dir live forever.
        The dir-mtime fallback must close this leak shape.

        Note: write_lock is called BEFORE os.utime so the dir mtime reflects
        the backdate, not the subsequent lock-file creation.
        """
        import run_lock
        from run_lock import LOCK_FILENAME, is_dir_live

        # Backdate the dir first, then write the lock (so dir mtime == old_mtime).
        old_mtime = time.time() - run_lock.LOCK_STALE_AFTER_SECONDS - 60
        os.utime(str(tmp_path), (old_mtime, old_mtime))
        (tmp_path / LOCK_FILENAME).write_text(
            "99999999999999999999\nnot-a-timestamp\n", encoding="utf-8"
        )
        # Backdate again after writing the lock file (write updated dir mtime).
        os.utime(str(tmp_path), (old_mtime, old_mtime))
        assert is_dir_live(str(tmp_path)) is False

    def test_out_of_range_pid_corrupt_ts_fresh_dir_is_live(self, tmp_path):
        """Out-of-range PID + garbage timestamp + fresh dir -> conservatively live.

        Same lock shape as above, but the dir mtime is recent, so the mtime
        escape keeps it live (conservative: we cannot compute a bounded age).
        """
        from run_lock import LOCK_FILENAME, is_dir_live

        (tmp_path / LOCK_FILENAME).write_text(
            "99999999999999999999\nnot-a-timestamp\n", encoding="utf-8"
        )
        # Dir mtime is ~now (tmp_path freshly created).
        assert is_dir_live(str(tmp_path)) is True


class TestWriteLockAtomic:
    """write_lock must produce exactly one .magi-lock file with no partial writes."""

    def test_write_lock_is_atomic_no_partial(self, tmp_path):
        """After write_lock, exactly .magi-lock exists and no .magi-lock.tmp remains."""
        from run_lock import LOCK_FILENAME, write_lock

        write_lock(str(tmp_path))
        lock = tmp_path / LOCK_FILENAME
        tmp = tmp_path / (LOCK_FILENAME + ".tmp")
        assert lock.exists(), ".magi-lock must exist after write_lock"
        assert not tmp.exists(), ".magi-lock.tmp must not remain after write_lock"
        # Verify the file has the expected 3 lines (pid, timestamp, bound).
        lines = lock.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}: {lines}"


class TestIsPidAliveTotality:
    """Out-of-range and unexpected-exception paths must return True (conservative-live)."""

    def test_out_of_range_pid_is_conservatively_alive(self):
        """An astronomically large PID must return True, never raise.

        On POSIX, os.kill(huge, 0) raises OverflowError. On Windows the
        ctypes call raises ctypes.ArgumentError. Neither is in the current
        except clause so the exception would propagate without the fix.
        """
        from run_lock import is_pid_alive

        # Must not raise; must return True (conservative-live).
        result = is_pid_alive(99999999999999999999)
        assert result is True


class TestProbeFailureWarning:
    """FIX E: unexpected probe failures emit one WARNING to stderr, stay conservative-live."""

    def test_is_pid_alive_unexpected_exception_returns_true_and_warns_once(
        self, monkeypatch, capsys
    ):
        """An unexpected non-OSError from the inner probe must:
        (a) return True (conservative), and
        (b) emit exactly one WARNING line to stderr on the FIRST call;
            subsequent calls stay silent (no per-call spam).
        """
        import run_lock

        # Reset the once-per-process flag so the test is self-contained.
        monkeypatch.setattr(run_lock, "_PROBE_FAILURE_WARNED", False)

        # Force the POSIX branch (platform-independent patch) and make os.kill
        # raise a non-OSError so the outer except Exception fires.
        monkeypatch.setattr(run_lock.sys, "platform", "linux")
        monkeypatch.setattr(run_lock.os, "kill", lambda pid, sig: (_ for _ in ()).throw(RuntimeError("boom")))

        # First call: returns True and emits warning.
        result1 = run_lock.is_pid_alive(42)
        assert result1 is True
        captured1 = capsys.readouterr()
        assert "WARNING" in captured1.err, f"Expected WARNING on first call, got: {captured1.err!r}"

        # Second call: returns True but no additional warning (flag already set).
        result2 = run_lock.is_pid_alive(42)
        assert result2 is True
        captured2 = capsys.readouterr()
        assert captured2.err == "", f"Expected no output on second call, got: {captured2.err!r}"

    def test_is_dir_live_unexpected_exception_returns_true_and_warns_once(
        self, tmp_path, monkeypatch, capsys
    ):
        """An unexpected Exception from _is_dir_live_inner must:
        (a) return True (conservative), and
        (b) emit exactly one WARNING to stderr on the first occurrence.
        """
        import run_lock

        # Reset the once-per-process flag.
        monkeypatch.setattr(run_lock, "_PROBE_FAILURE_WARNED", False)

        # Patch the inner function to raise an unexpected error.
        monkeypatch.setattr(
            run_lock, "_is_dir_live_inner", lambda run_dir: (_ for _ in ()).throw(RuntimeError("inner boom"))
        )

        result1 = run_lock.is_dir_live(str(tmp_path))
        assert result1 is True
        captured1 = capsys.readouterr()
        assert "WARNING" in captured1.err, f"Expected WARNING on first call, got: {captured1.err!r}"

        # Second call: still True, no second warning.
        result2 = run_lock.is_dir_live(str(tmp_path))
        assert result2 is True
        captured2 = capsys.readouterr()
        assert captured2.err == "", f"Expected no output on second call, got: {captured2.err!r}"
