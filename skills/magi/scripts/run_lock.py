# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-23
"""Process-liveness locking for MAGI run directories.

Each run directory under the per-project temp namespace carries a
``.magi-lock`` file naming the PID and ISO start timestamp of the
orchestrator that owns it. ``temp_dirs.cleanup_old_runs`` consults
:func:`is_dir_live` so a concurrent MAGI session never prunes a run
directory whose owning process is still alive.

The lock is advisory and self-healing: a crashed process leaves a stale
lock behind, but :func:`is_pid_alive` reports the dead PID as not alive
on the next cleanup, and :data:`LOCK_STALE_AFTER_SECONDS` bounds the
window in which a reused PID could keep a dead run's directory alive.
"""

from __future__ import annotations

import os
import sys

LOCK_FILENAME = ".magi-lock"
# Floor / default for the per-run staleness guard (spec R13). Each lock
# persists its own bound derived from --timeout (see
# staleness_bound_for_timeout); this constant is used only when that bound
# is absent (2-line legacy lock) or corrupt, and as the lower clamp so a
# corrupt tiny bound can never drop the threshold below 6h.
LOCK_STALE_AFTER_SECONDS = 21_600  # 6 hours


def is_pid_alive(pid: int) -> bool:
    """Return True if a process with *pid* currently exists.

    POSIX: ``os.kill(pid, 0)`` — ``ProcessLookupError`` means dead; any
    other ``OSError`` (e.g. ``PermissionError``) means the process
    exists but is not ours, treated as alive. Windows: ``OpenProcess``
    + ``WaitForSingleObject(handle, 0)``; ``WAIT_TIMEOUT`` means still
    running, ``WAIT_OBJECT_0`` means exited. Any probe failure is
    treated conservatively as alive so cleanup never prunes a dir whose
    liveness it could not verify (spec R8).

    Args:
        pid: Process id to probe.

    Returns:
        True if the process appears to exist, False if definitively dead.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _is_pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _is_pid_alive_windows(pid: int) -> bool:
    """Windows liveness probe via ``OpenProcess`` + ``WaitForSingleObject``.

    ``restype``/``argtypes`` are declared so the pointer-sized ``HANDLE``
    is not truncated to a signed 32-bit ``c_int`` and sign-extended when
    reused (the latent bug the bare ``status_display`` pattern would carry
    for a *stored* handle). ``WinDLL(use_last_error=True)`` makes
    ``ctypes.get_last_error()`` reliable: a null handle with
    ``ERROR_ACCESS_DENIED`` means the process EXISTS but we lack rights to
    open it -> reported **alive** to mirror POSIX ``PermissionError ->
    alive`` and honor R8's conservative bias; any other null-handle error
    means the process is gone. An unexpected probe failure (including
    ``WinDLL`` being absent off-Windows) is conservatively reported alive.
    """
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]

        SYNCHRONIZE = 0x00100000
        WAIT_TIMEOUT = 0x00000102
        ERROR_ACCESS_DENIED = 5

        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not handle:
            # Existing-but-inaccessible (access denied) -> alive; gone -> dead.
            return ctypes.get_last_error() == ERROR_ACCESS_DENIED
        try:
            return bool(kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT)
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, AttributeError, ImportError):
        return True
