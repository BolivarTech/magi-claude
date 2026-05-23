# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-23
"""Tests for run_lock.py — process-liveness locking."""

from __future__ import annotations

import os
import subprocess
import sys

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
