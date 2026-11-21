#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 3.0.0
# Date: 2026-09-20
"""``--ollama-init``: create the passphrase key, then one ``magi-rs init`` call.

:func:`run_ollama_init` bootstraps and verifies the pinned binary (R4 has no
carve-out for the init command, which executes the binary too), generates the
key, hands it to ``magi-rs init -w <root>`` through the child environment and
only then persists it. A ``.magi/`` that already exists is left untouched:
nothing is called, nothing is written, and :class:`MagiDirExistsError` says so
(REQ-12), so no caller can mistake it for a new ``.magi/``. The key never
travels in argv.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from enum import Enum

from magi_key import (
    KEY_DIR_PARTS,
    REGENERATION_GUIDANCE,
    KeyHardeningError,
    KeyMissingError,
    KeyStore,
    KeyStoreError,
    MagiKey,
)
from magi_key_permissions import default_key_store
from magi_rs_bootstrap import verified_plugin_executable
from magi_rs_env import EnvironmentPolicy
from one_line import one_line

_INIT_SUBCOMMAND = "init"
_WORKDIR_FLAG = "-w"
_INIT_TIMEOUT_SECONDS = 120
"""``magi-rs init`` scaffolds a directory and a small database; two minutes is
far beyond any healthy run and only bounds a hung child."""


_MAGI_DIR = KEY_DIR_PARTS[0]
_ALREADY_INITIALISED = (
    f"{_MAGI_DIR}/ already exists in {{workdir}}: left untouched (nothing was initialised, "
    f"no key was created). To start over: {REGENERATION_GUIDANCE}."
)


class InitLeftover(Enum):
    """What a failed ``run_ollama_init`` left at ``<workdir>/.magi``, decided by its own cleanup."""

    REMOVED = "removed"
    """Nothing: the ``.magi/`` this call created was removed."""
    KEY_WRITTEN_KEPT = "key-written-kept"
    """This call's key is on disk and the ``.magi/`` could not be removed."""
    HARDENING_FAILED_KEPT = "hardening-failed-kept"
    """As ``KEY_WRITTEN_KEPT``, and the key could not be hardened: it may be readable by others."""
    FOREIGN_KEPT = "foreign-kept"
    """The ``.magi/`` holds a key this call did not write: another process initialised it; kept."""
    HALF_MADE_KEPT = "half-made-kept"
    """A ``.magi/`` without a key that this call created and could not remove."""


class OllamaInitError(Exception):
    """``magi-rs init`` failed; the message carries what it printed (a sibling of ``ValidationError``).

    Attributes:
        leftover: What the failure left at ``.magi/`` (:class:`InitLeftover`),
            or ``None`` when not reported (an error raised outside
            ``run_ollama_init``'s cleanup).
    """

    def __init__(self, message: str, leftover: InitLeftover | None = None) -> None:
        """Build the error.

        Args:
            message: The one-line (or multi-line) description.
            leftover: What the failure left at ``.magi/``; ``None`` when not reported.
        """
        super().__init__(message)
        self._leftover = leftover

    @property
    def leftover(self) -> InitLeftover | None:
        """What the failure left at ``.magi/``, as decided by ``run_ollama_init``'s cleanup."""
        return self._leftover


class MagiDirExistsError(OllamaInitError):
    """``.magi/`` already exists: nothing was initialised and no key was created.

    Raised before anything runs. The ``--ollama-init`` CLI prints its message
    and exits 0 (an already initialised project is not a failure there); the
    key rotation treats it as a failure, since the ``.magi/`` it finds is not
    the one it was creating.
    """


def _remove_created_magi_dir(magi_dir: str) -> str:
    """Remove a ``.magi/`` this call created, so a retry is not told it is initialised.

    Args:
        magi_dir: The directory to remove; absent is fine.

    Returns:
        An empty string when it was removed; otherwise a sentence telling the
        operator to delete the directory by hand before retrying.
    """
    try:
        shutil.rmtree(magi_dir)
    except FileNotFoundError:
        return ""
    except OSError:
        return f"{magi_dir} could not be removed: delete it by hand before retrying."
    return ""


_OTHER_PROCESS_NOTE = (
    "{magi_dir} now holds a key this run did not write: another process initialised it, "
    "so it was left in place."
)


def _holds_other_key(store: KeyStore, workdir: str, key: MagiKey) -> bool:
    """Tell whether the key file under ``workdir`` holds a key other than ``key``.

    Read through ``store.read_unverified``, the key file's only reader. An
    absent or empty key file holds no key. A key file that exists but cannot
    be read counts as someone else's (``read_unverified`` chains the read
    failure as the cause of its ``KeyMissingError``), so a cleanup never
    deletes a key it cannot prove is its own.

    Args:
        store: The key store of this call.
        workdir: The project root.
        key: The key this call generated.

    Returns:
        True when another key, or an unreadable one, is there; False when
        there is none, it is empty, or it is ``key``.
    """
    key_file = store.key_path(workdir)
    if not os.path.lexists(key_file):
        return False
    try:
        return store.read_unverified(key_file) != key
    except KeyMissingError as error:
        return error.__cause__ is not None
    except (KeyStoreError, OSError):
        return True


def run_ollama_init(workdir: str, key_store: KeyStore | None = None) -> str:
    """Initialise ``workdir`` for the magi-rs backend with a freshly generated key.

    Args:
        workdir: The project root, passed to ``magi-rs`` as ``-w``.
        key_store: Where the key is persisted; ``None`` selects the platform
            store (:func:`magi_key_permissions.default_key_store`).

    Returns:
        What ``magi-rs init`` printed (stdout, then stderr), relayed verbatim
        except that the new key is replaced by ``REDACTION_MARKER``.

    Raises:
        MagiDirExistsError: ``<workdir>/.magi/`` already exists; nothing was
            called and nothing was written.
        Any exception from the child launch or the key store: the ``.magi/`` this call
            created is removed first and the exception propagates unchanged. When something
            needs saying instead (the removal failed, so the message says to delete it by hand;
            or the ``.magi/`` now holds a key this call did not write, e.g. ``persist`` raised
            ``FileExistsError`` because another process initialised it, so it is never removed),
            an ``OllamaInitError`` carrying that sentence and its ``leftover``
            (:class:`InitLeftover`) is raised from the original. A failed child
            always ends as an ``OllamaInitError`` with its ``leftover``. An
            interrupt (``KeyboardInterrupt``, ``SystemExit``) is re-raised as itself; its
            sentence is written to stderr as one ``[!] WARNING:`` line first.
        OllamaInitError: ``magi-rs init`` exited non-zero; the message is its
            output with the new key redacted. No key is persisted and the
            ``.magi/`` this call saw appear is removed.
        KeyHardeningError: The key could not be restricted to its owner. The
            ``.magi/`` this call created is removed.
        UnsafeBinDirectoryError, DigestMismatchError, VersionMismatchError,
        DownloadError, UnsupportedPlatformError, PinFormatError: The binary
            could not be obtained or did not verify. It is NOT executed (R4).
        PluginRootError: ``CLAUDE_PLUGIN_ROOT`` is set but wrong.
        subprocess.TimeoutExpired: ``magi-rs init`` hung past the bound.

    Side effects:
        May download and install the binary; launches ``magi-rs init`` with
        ``MAGI_PASSPHRASE`` in its environment, which creates
        ``<workdir>/.magi/``; writes ``.magi/.secret/magi.key`` on success.
    """
    magi_dir = os.path.join(workdir, _MAGI_DIR)
    if os.path.exists(magi_dir):
        raise MagiDirExistsError(_ALREADY_INITIALISED.format(workdir=workdir))
    store = key_store if key_store is not None else default_key_store()
    executable = verified_plugin_executable()
    key = store.new_key()
    try:
        completed = subprocess.run(
            [executable, _INIT_SUBCOMMAND, _WORKDIR_FLAG, workdir],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=EnvironmentPolicy.build(os.environ, passphrase=key),
            timeout=_INIT_TIMEOUT_SECONDS,
        )
        # The new key is in the child's environment: whatever it printed is redacted
        # before it is returned or raised (BDD-22).
        output = key.redact("".join(part for part in (completed.stdout, completed.stderr) if part))
        if completed.returncode != 0:
            raise OllamaInitError(output)
        store.persist(workdir, key)
    except BaseException as error:
        leftover, note = _clean_up(store, workdir, key, error)
        if not isinstance(error, Exception):
            # An interrupt stays an interrupt; its cleanup sentence goes to stderr, one line.
            if note:
                print(f"[!] WARNING: {one_line(note)}", file=sys.stderr)
            raise
        if leftover is InitLeftover.REMOVED and not isinstance(error, OllamaInitError):
            raise
        detail = one_line(key.redact(str(error))) or type(error).__name__
        raise OllamaInitError(f"{detail}\n{note}" if note else detail, leftover=leftover) from error
    return output


def _clean_up(
    store: KeyStore, workdir: str, key: MagiKey, error: BaseException
) -> tuple[InitLeftover, str]:
    """Remove what a failed initialisation created, or keep what is not this call's; say which.

    The one place that decides whether the key at ``workdir`` is this call's.

    Args:
        store: The key store of this call.
        workdir: The project root.
        key: The key this call generated.
        error: What stopped the initialisation.

    Returns:
        ``(leftover, note)``: what is left at ``.magi/`` and the sentence the
        operator needs about it (empty when it was removed).
    """
    magi_dir = os.path.join(workdir, _MAGI_DIR)
    if _holds_other_key(store, workdir, key):
        return InitLeftover.FOREIGN_KEPT, _OTHER_PROCESS_NOTE.format(magi_dir=magi_dir)
    note = _remove_created_magi_dir(magi_dir)
    if not note:
        return InitLeftover.REMOVED, ""
    if not os.path.lexists(store.key_path(workdir)):
        return InitLeftover.HALF_MADE_KEPT, note
    if isinstance(error, KeyHardeningError):
        return InitLeftover.HARDENING_FAILED_KEPT, note
    return InitLeftover.KEY_WRITTEN_KEPT, note
