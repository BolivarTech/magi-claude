#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.1.0
# Date: 2026-05-23
"""Temp-directory housekeeping for the MAGI orchestrator.

Extracted from ``run_magi.py`` so the orchestrator file no longer has to
hold the LRU + symlink-traversal + mtime-tie-break rules in the same
mental model as subprocess and display concerns. The helpers here are
pure filesystem manipulation — no asyncio, no subprocesses, no CLI —
and can be unit-tested independently of the orchestrator wiring.

Public contract:

* :data:`MAGI_DIR_PREFIX` is the single source of truth for the
  ``magi-run-*`` directory naming convention; both cleanup and creation
  must agree on it.
* :func:`cleanup_old_runs` is the LRU entry point. It is deliberately
  total (never raises on scan/stat errors, only on programmer errors)
  so the orchestrator can call it unconditionally.
* :func:`create_output_dir` is the counterpart that honors the same
  prefix when generating a temp dir.

``_scan_magi_dirs``, ``_safe_temp_prefix`` and ``_safe_rmtree_under``
are internal helpers exposed with leading underscores — they are only
module-public for the regression suite that drills into specific TOCTOU
and mtime tie-break behaviors.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import time

from run_lock import LOCK_STALE_AFTER_SECONDS, is_dir_live

MAGI_DIR_PREFIX = "magi-run-"
MAGI_RUNS_CONTAINER = "magi-runs"
LEGACY_SWEEP_MARKER = ".legacy-swept"


def _scan_magi_dirs(tmp_root: str) -> list[tuple[float, str]]:
    """Return ``(mtime, path)`` tuples for every ``magi-run-*`` dir under *tmp_root*.

    Entries that disappear between scan and stat are silently skipped.
    """
    results: list[tuple[float, str]] = []
    for entry in os.scandir(tmp_root):
        if not (entry.is_dir() and entry.name.startswith(MAGI_DIR_PREFIX)):
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        results.append((mtime, entry.path))
    return results


def _safe_temp_prefix(tmp_root: str) -> str:
    """Return the normalized temp-root prefix used for traversal checks.

    Resolves symlinks in *tmp_root* before building the prefix so that
    ``os.path.realpath(entry.path).startswith(prefix)`` stays consistent
    when the temp root itself is a symlink (e.g. ``/tmp`` ->
    ``/private/tmp`` on macOS). Without this, every scanned entry
    resolves outside the advertised prefix and cleanup becomes a
    silent no-op.
    """
    prefix = os.path.normcase(os.path.realpath(tmp_root))
    if not prefix.endswith(os.sep):
        prefix += os.sep
    return prefix


def _safe_rmtree_under(path: str, safe_prefix: str) -> None:
    """Remove *path* only if it resolves strictly inside *safe_prefix*.

    The realpath check prevents symlink traversal attacks on shared
    systems. Failures are logged to stderr - cleanup must never raise.
    """
    resolved = os.path.normcase(os.path.realpath(path))
    if not resolved.startswith(safe_prefix):
        print(
            f"WARNING: Skipping cleanup of {path} (resolves outside temp root: {resolved})",
            file=sys.stderr,
        )
        return
    try:
        shutil.rmtree(resolved)
    except OSError as exc:
        print(
            f"WARNING: Failed to remove old run {resolved}: {exc}",
            file=sys.stderr,
        )


def cleanup_old_runs(keep: int, run_root: str | None = None) -> None:
    """Remove oldest MAGI temp directories under *run_root*, keeping recent ones.

    Scans *run_root* (defaulting to the system temp dir for backward
    compatibility) for ``magi-run-*`` directories. Directories whose
    ``.magi-lock`` shows a still-running owner are **excluded entirely**
    — they are neither counted against ``keep`` nor deleted — so a
    concurrent session's in-progress run is never pruned. Among the
    remaining (non-live) dirs, the oldest beyond ``keep`` are removed,
    sorted by ``st_mtime`` descending then path ascending for a
    deterministic LRU under mtime ties.

    Live (locked) dirs are excluded from the count, so the on-disk total
    can exceed ``keep`` when concurrent or stale-locked runs are present.

    Total: a missing/unscannable *run_root* and per-entry stat/rmtree
    errors degrade to no-op/warning, never raising into the orchestrator.

    Args:
        keep: Maximum number of non-live runs to retain. ``keep < 0``
            disables cleanup; ``keep == 0`` removes every non-live dir.
        run_root: Directory to scan. ``None`` -> ``tempfile.gettempdir()``.
    """
    if keep < 0:
        return

    if run_root is None:
        run_root = tempfile.gettempdir()

    try:
        magi_dirs = _scan_magi_dirs(run_root)
    except OSError:
        return

    # Protect live dirs first: exclude from both the survivor budget and
    # the deletion set.
    candidates = [(mtime, path) for (mtime, path) in magi_dirs if not is_dir_live(path)]

    if len(candidates) <= keep:
        return

    candidates.sort(key=lambda entry: (-entry[0], entry[1]))

    safe_prefix = _safe_temp_prefix(run_root)
    for _, path in candidates[keep:]:
        _safe_rmtree_under(path, safe_prefix)


def create_output_dir(output_dir: str | None, run_root: str | None = None) -> str:
    """Create and return the output directory.

    Args:
        output_dir: Explicit path, or None to create a temp dir.
        run_root: Parent directory for the temp dir when *output_dir* is
            None. Defaults to ``tempfile.gettempdir()`` for backward
            compatibility; the namespaced flow passes the per-project
            container from :func:`project_run_root`.

    Returns:
        Path to the created output directory.
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    if run_root is None:
        run_root = tempfile.gettempdir()
    return tempfile.mkdtemp(prefix=MAGI_DIR_PREFIX, dir=run_root)


def project_run_root(project_root: str) -> str:
    """Return (creating if needed) the per-project run container.

    *project_root* is normalized (``normcase`` + ``realpath``) and hashed
    to a 16-hex-char key so the same project always maps to the same
    container regardless of casing or symlinks. The container is
    ``<gettempdir>/magi-runs/<key>/``. Runs from different projects live
    under different containers, so one project's cleanup can never see or
    prune another's.

    If the container cannot be created (permissions, read-only temp), it
    degrades to ``tempfile.gettempdir()`` with a warning rather than
    raising into ``main()`` — the namespace is best-effort, consistent
    with the total-cleanup contract.

    Args:
        project_root: The resolved project root path (git toplevel or cwd).

    Returns:
        Absolute path to the created per-project run container, or the
        system temp dir if the container could not be created.
    """
    norm = os.path.normcase(os.path.realpath(project_root))
    key = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
    tmp_root = tempfile.gettempdir()
    root = os.path.join(tmp_root, MAGI_RUNS_CONTAINER, key)
    try:
        os.makedirs(root, exist_ok=True)
    except OSError as exc:
        print(
            f"WARNING: could not create per-project run root {root}: {exc}; "
            f"falling back to {tmp_root}",
            file=sys.stderr,
        )
        return tmp_root
    return root


def sweep_legacy_runs_once() -> None:
    """One-shot removal of pre-2.6.0 ``magi-run-*`` dirs directly under temp.

    Older MAGI versions created run dirs directly under
    ``tempfile.gettempdir()`` without the per-project namespace or a lock
    file, so they are now orphaned. This removes them once — guarded by a
    marker file under the ``magi-runs`` container so the (potentially
    slow) global temp scan does not recur on every run — deleting only
    dirs older than :data:`run_lock.LOCK_STALE_AFTER_SECONDS` so a
    concurrently-running old version's in-progress dir is not removed.
    The ``magi-runs`` container itself does not match ``MAGI_DIR_PREFIX``
    and is never a candidate.

    Total: every failure path degrades to no-op/warning; never raises
    into the orchestrator.
    """
    tmp_root = tempfile.gettempdir()
    container = os.path.join(tmp_root, MAGI_RUNS_CONTAINER)
    marker = os.path.join(container, LEGACY_SWEEP_MARKER)

    try:
        os.makedirs(container, exist_ok=True)
        if os.path.exists(marker):
            return
    except OSError:
        return

    now = time.time()
    safe_prefix = _safe_temp_prefix(tmp_root)
    try:
        entries = _scan_magi_dirs(tmp_root)
    except OSError:
        entries = []

    for mtime, path in entries:
        if now - mtime >= LOCK_STALE_AFTER_SECONDS:
            _safe_rmtree_under(path, safe_prefix)

    try:
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write("swept\n")
    except OSError:
        pass
