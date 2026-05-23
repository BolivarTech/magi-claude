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
Corrupt or empty locks are reclaimed once the run directory ages past the
staleness floor, bounding the leak to at most :data:`LOCK_STALE_AFTER_SECONDS`.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

LOCK_FILENAME = ".magi-lock"
# Floor / default for the per-run staleness guard. Each lock persists its own
# bound derived from --timeout (see staleness_bound_for_timeout); this
# constant is used only when that bound is absent (2-line legacy lock) or
# corrupt, and as the lower clamp so a corrupt tiny bound can never drop the
# threshold below 6h.
LOCK_STALE_AFTER_SECONDS = 21_600  # 6 hours

# Once-per-process gate: after the first unexpected probe failure emits a
# WARNING, subsequent failures stay silent to avoid per-directory spam.
# The flag is module-level so it survives across calls in the same process;
# tests reset it via the autouse fixture _reset_probe_warned in test_run_lock.py.
_PROBE_FAILURE_WARNED: bool = False


def _warn_probe_failure(probe: str, exc: Exception) -> None:
    """Emit one WARNING per process on an unexpected liveness-probe failure.

    After the first call the module-level :data:`_PROBE_FAILURE_WARNED` flag
    is set so subsequent failures stay silent (no per-directory spam).  The
    caller continues to return ``True`` (conservative) regardless.

    Args:
        probe: Name of the calling function (for context in the message).
        exc:   The unexpected exception that was caught.
    """
    global _PROBE_FAILURE_WARNED  # noqa: PLW0603
    if not _PROBE_FAILURE_WARNED:
        _PROBE_FAILURE_WARNED = True
        print(
            f"WARNING: run_lock.{probe} unexpected liveness-probe error"
            f" ({type(exc).__name__}: {exc}); treating as live",
            file=sys.stderr,
        )


def is_pid_alive(pid: int) -> bool:
    """Return True if a process with *pid* currently exists.

    POSIX: ``os.kill(pid, 0)`` — ``ProcessLookupError`` means dead; any
    other ``OSError`` (e.g. ``PermissionError``) means the process
    exists but is not ours, treated as alive. Windows: ``OpenProcess``
    + ``WaitForSingleObject(handle, 0)``; ``WAIT_TIMEOUT`` means still
    running, ``WAIT_OBJECT_0`` means exited. Any probe failure is
    treated conservatively as alive so cleanup never prunes a dir whose
    liveness it could not verify.

    Out-of-range PIDs (exceeding the platform's addressable integer
    range, e.g. from a corrupt lock file) are returned conservatively
    as alive rather than raising or silently misreporting as dead.
    The ``except Exception`` final catch ensures any unexpected probe
    failure — including ``OverflowError`` on POSIX or future ctypes
    changes — defaults to the safe side.  The first such failure emits
    one ``WARNING`` via :func:`_warn_probe_failure` so a silent regression
    to no-op cleanup is observable.  ``BaseException`` is NOT caught so
    ``KeyboardInterrupt`` and ``SystemExit`` propagate normally.

    Args:
        pid: Process id to probe.

    Returns:
        True if the process appears to exist, False if definitively dead.
    """
    if pid <= 0:
        return False
    # An out-of-range PID cannot belong to any real process; treat as
    # conservatively alive rather than letting os.kill raise OverflowError
    # (POSIX) or ctypes silently wrap the value to a different PID (Windows).
    if pid > 4_294_967_295:  # max uint32 — no OS supports PIDs above this
        return True
    try:
        if sys.platform == "win32":
            return _is_pid_alive_windows(pid)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True
        return True
    except Exception as exc:  # noqa: BLE001
        # Any unexpected failure (OverflowError, ctypes.ArgumentError, …)
        # defaults to conservatively alive.  Emit one WARNING per process so
        # the failure is observable without spamming on every cleanup call.
        _warn_probe_failure("is_pid_alive", exc)
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
    alive``; any other null-handle error means the process is gone. An
    unexpected probe failure (including ``WinDLL`` being absent off-Windows)
    is conservatively reported alive.
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


def _lock_path(run_dir: str) -> str:
    """Return the absolute path of the lock file inside *run_dir*."""
    return os.path.join(run_dir, LOCK_FILENAME)


def _dir_is_fresh(run_dir: str) -> bool:
    """Return True if *run_dir*'s mtime is younger than the staleness floor.

    Used as the liveness fallback whenever a bounded age cannot be computed
    (corrupt PID line, or alive PID with an unreadable timestamp), so an
    indeterminate lock cannot keep a directory live forever.  An unreadable
    mtime is treated as fresh (conservative).

    Args:
        run_dir: Absolute path of the MAGI run directory to check.

    Returns:
        True if the directory is younger than :data:`LOCK_STALE_AFTER_SECONDS`,
        False if it is stale.  Also True on any :exc:`OSError` (cannot stat).
    """
    try:
        return (time.time() - os.path.getmtime(run_dir)) < LOCK_STALE_AFTER_SECONDS
    except OSError:
        return True


def staleness_bound_for_timeout(timeout: int) -> int:
    """Return the per-run staleness bound (seconds) for a given ``--timeout``.

    The orchestrator kills each agent at ``timeout`` (x2 with the single
    retry), so a live run cannot exceed ~2x ``timeout``; ``+600`` adds
    orchestration margin. Floored at :data:`LOCK_STALE_AFTER_SECONDS` so
    short timeouts still get the generous 6h default. Persisting this in
    the lock ensures a long-``--timeout`` run is never pruned alive.
    """
    return max(2 * timeout + 600, LOCK_STALE_AFTER_SECONDS)


def write_lock(run_dir: str, max_age_seconds: int | None = None) -> None:
    """Write ``<run_dir>/.magi-lock`` with PID, start time, and staleness bound.

    Three lines: the integer PID, the ISO-8601 UTC start timestamp, and
    the per-run staleness bound in seconds. ``max_age_seconds=None``
    falls back to :data:`LOCK_STALE_AFTER_SECONDS`.

    The write is atomic: the payload is first written to a ``.magi-lock.tmp``
    sibling, then moved into place via :func:`os.replace`. A crash between
    the two steps leaves a zero-byte or partial ``.magi-lock`` absent (the
    original is gone only after ``os.replace`` succeeds), so readers never
    see a partial payload in the canonical filename.

    Best-effort — an I/O error is reported to stderr and swallowed; a
    missing lock merely degrades this one dir to pre-2.6.0 behavior, it
    does not break the run.
    """
    bound = LOCK_STALE_AFTER_SECONDS if max_age_seconds is None else int(max_age_seconds)
    payload = f"{os.getpid()}\n{datetime.now(timezone.utc).isoformat()}\n{bound}\n"
    final = _lock_path(run_dir)
    tmp = final + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, final)
    except OSError as exc:
        print(f"WARNING: could not write run lock in {run_dir}: {exc}", file=sys.stderr)
        try:
            os.remove(tmp)
        except OSError:
            pass


def _parse_lock(run_dir: str) -> tuple[int | None, float | None, int | None]:
    """Return ``(pid, age_seconds, max_age_seconds)`` parsed from the lock.

    Any element is ``None`` when missing or unparseable. ``age_seconds`` is
    wall-clock seconds since the recorded ISO start timestamp;
    ``max_age_seconds`` is the persisted per-run staleness bound.
    """
    try:
        with open(_lock_path(run_dir), encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None, None, None
    pid: int | None = None
    age: float | None = None
    bound: int | None = None
    if lines:
        try:
            pid = int(lines[0].strip())
        except ValueError:
            pid = None
    if len(lines) > 1:
        try:
            started = datetime.fromisoformat(lines[1].strip())
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - started).total_seconds()
        except ValueError:
            age = None
    if len(lines) > 2:
        try:
            bound = int(lines[2].strip())
        except ValueError:
            bound = None
    return pid, age, bound


def read_lock(run_dir: str) -> int | None:
    """Return the PID recorded in the lock, or None if absent/corrupt."""
    return _parse_lock(run_dir)[0]


def remove_lock(run_dir: str) -> None:
    """Remove the lock file if present. Best-effort, never raises."""
    try:
        os.remove(_lock_path(run_dir))
    except OSError:
        pass


def is_dir_live(run_dir: str) -> bool:
    """Return True if *run_dir* belongs to a still-running MAGI process.

    Decision table:

    * No lock file at all -> not live (a completed/legacy run).
    * Lock present but PID unparseable and dir is fresh -> conservatively live.
    * Lock present but PID unparseable and dir is stale -> not live (mtime escape).
    * PID present and dead -> not live.
    * PID present and alive but lock age >= the **persisted per-run bound**
      (falling back to ``LOCK_STALE_AFTER_SECONDS`` when the bound line is
      absent/corrupt) -> not live, mitigating PID reuse.
    * PID present, alive, and within the bound -> live.

    Structurally total: any unexpected ``Exception`` from a probe returns
    ``True`` (conservative) so ``cleanup_old_runs``'s comprehension can
    never raise even if a future probe change regresses.  The first such
    failure emits one ``WARNING`` to stderr via :func:`_warn_probe_failure`
    to prevent a regression silently becoming a no-op cleanup.
    ``BaseException`` is NOT caught so ``KeyboardInterrupt`` and
    ``SystemExit`` propagate.
    """
    try:
        return _is_dir_live_inner(run_dir)
    except Exception as exc:  # noqa: BLE001
        _warn_probe_failure("is_dir_live", exc)
        return True


def _is_dir_live_inner(run_dir: str) -> bool:
    """Inner (non-total) implementation of :func:`is_dir_live`."""
    pid, age, bound = _parse_lock(run_dir)
    if pid is None:
        lock = _lock_path(run_dir)
        if not os.path.exists(lock):
            # No lock file at all -> eligible (not live).
            return False
        # Corrupt/empty lock: conservatively live when the dir is fresh,
        # eligible once the dir ages past the staleness floor.
        return _dir_is_fresh(run_dir)
    if not is_pid_alive(pid):
        return False
    # PID is alive.  If the timestamp is unreadable (age is None) OR negative
    # (future-dated timestamp from clock skew or a corrupt lock), we cannot
    # compute a meaningful bounded age, so fall back to the dir-mtime escape.
    # A negative age must NOT be compared against the threshold: it would
    # always satisfy age < threshold (unbounded live leak).
    if age is None or age < 0:
        return _dir_is_fresh(run_dir)
    # Floor the threshold so a corrupt-but-parseable tiny/negative bound
    # cannot defeat the conservative bias; a legitimate bound is always
    # >= the floor by construction (staleness_bound_for_timeout).
    threshold = LOCK_STALE_AFTER_SECONDS if bound is None else max(bound, LOCK_STALE_AFTER_SECONDS)
    if age >= threshold:
        return False
    return True
