# Author: Julian Bolivar
# Version: 7.1.0
# Date: 2026-11-19
"""Full rotation of an exposed magi-rs memory key (REQ-13e-h).

A key whose permissions were loosened may have been copied, and with it the
memory database and every secret in the vault. Re-wrapping the database key
(``vault passwd``) does not protect a copy that already left, so the rotation
replaces everything:

1. it is authorised explicitly (``--yes``, or ``y`` at a terminal prompt);
2. the old key is read (unverified: it is exposed by definition) before
   anything moves;
3. ``<backup root>/<id>``, outside every working tree, is created and
   restricted to its owner; the old ``.magi/`` is moved into it and every
   directory and file of it is restricted again and verified (a move across
   volumes does not carry the ACL, a rename keeps each child's own);
4. a new ``.magi/`` is created by the injected ``init`` (new key, new database),
   and its key is loaded back through the store (present, owner-only and
   different from the old one), so a ``.magi/`` that some other run recreated,
   or the old tree moved back by hand, is never reported as rotated; an
   ``init`` that finds a ``.magi/`` already there fails the rotation and that
   ``.magi/`` is left as it is;
5. the old key is used once more, only for ``vault ls`` on the backup, so the
   operator learns which API keys to revoke (names only, never values).

Every failure leaves a described state in one line and never prints either
key. A hard stop (killed process, power loss) between steps 3 and 4 prints
nothing: the root then has no ``.magi/`` and the old one is intact in the
backup, from where it can be moved back.

Side effects of :meth:`KeyRotation.rotate`: creates ``<backup root>/<id>/``,
moves ``<workdir>/.magi`` into it, restricts the backup's permissions, and
creates a new ``<workdir>/.magi`` through ``init``.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import shutil
import stat
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from magi_key import DIR_MODE, KEY_DIR_PARTS, KeyStore, KeyStoreError, MagiKey
from magi_rs_vault import VAULT_KEY_NAMES
from ollama_init import MagiDirExistsError

MAGI_DIR = KEY_DIR_PARTS[0]
"""The project directory that a rotation moves aside and recreates."""

_BACKUP_PARTS = ("magi", "rotated")
"""The backup root, relative to the platform's per-user state directory."""

_WINDOWS_STATE_VARIABLE = "LOCALAPPDATA"
_WINDOWS_STATE_FALLBACK_PARTS = ("AppData", "Local")
"""``%LOCALAPPDATA%`` relative to the profile, used only when the variable is unset."""

_POSIX_STATE_VARIABLE = "XDG_STATE_HOME"
_POSIX_STATE_FALLBACK_PARTS = (".local", "state")
"""The XDG default of ``$XDG_STATE_HOME``, relative to ``$HOME``."""

_FINGERPRINT_HEX_CHARS = 16
"""Length of the project fingerprint in a backup id."""

_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
"""UTC timestamp of a backup id, e.g. ``20261120T120000Z``."""

_FIRST_COLLISION_SUFFIX = 2
"""The suffix of the second backup created within the same second (``-2``)."""

_AUTHORISING_ANSWERS = frozenset({"y", "Y"})
"""The only answers to the confirmation prompt that authorise a rotation."""

_CONFIRM_PROMPT = (
    "Rotate the magi-rs key of {workdir}? The current .magi/ is moved to a backup and its memory "
    "and history are lost. [y/N] "
)


class RotationNotAuthorisedError(Exception):
    """The rotation was not authorised; nothing changed (a sibling of ``ValidationError``)."""


class _LinkInBackupError(Exception):
    """A link or reparse point was found in the moved ``.magi/``; it is never followed or hardened."""


class RotationFailedError(Exception):
    """The rotation stopped; the message describes, in one line, the state left behind.

    A sibling of ``ValidationError``: a stopped rotation is never something a
    retry handler may consume.
    """


@dataclass
class RotationReport:
    """What a completed rotation did.

    Attributes:
        backup_dir: ``<backup root>/<id>``, the parent of the old ``.magi/``.
        names_to_revoke: The API key names to revoke at their providers.
        names_from_vault: True when the names were read from the old vault;
            False when it could not be read and the names are the whole closed
            list (``VAULT_KEY_NAMES``).
    """

    backup_dir: str
    names_to_revoke: list[str]
    names_from_vault: bool


def rotation_backup_root() -> str:
    """Return the per-user directory that holds rotation backups, outside every working tree.

    Returns:
        On Windows ``%LOCALAPPDATA%\\magi\\rotated`` (``~\\AppData\\Local`` when the
        variable is unset); elsewhere ``$XDG_STATE_HOME/magi/rotated``, or
        ``~/.local/state/magi/rotated`` when the variable is unset or empty.
        Never fails for a missing variable.
    """
    if sys.platform == "win32":
        variable, fallback = _WINDOWS_STATE_VARIABLE, _WINDOWS_STATE_FALLBACK_PARTS
    else:
        variable, fallback = _POSIX_STATE_VARIABLE, _POSIX_STATE_FALLBACK_PARTS
    state = os.environ.get(variable) or os.path.join(os.path.expanduser("~"), *fallback)
    return os.path.join(state, *_BACKUP_PARTS)


def _utc_now() -> datetime:
    """Return the current time, timezone-aware, in UTC.

    Returns:
        ``datetime.now(timezone.utc)``.
    """
    return datetime.now(timezone.utc)


def _canonical(path: str) -> str:
    """Return ``path`` resolved and case-normalised, for containment checks.

    Args:
        path: Any path; it need not exist.

    Returns:
        The real, normalised path.
    """
    return os.path.normcase(os.path.realpath(path))


def _is_within(path: str, parent: str) -> bool:
    """Tell whether ``path`` is ``parent`` or lies under it.

    Args:
        path: The candidate path.
        parent: The directory it may lie in.

    Returns:
        True when ``path`` resolves inside ``parent``; False otherwise,
        including when the two are on different drives.
    """
    candidate, root = _canonical(path), _canonical(parent)
    try:
        return os.path.commonpath([candidate, root]) == root
    except ValueError:
        return False


def _is_link(path: str) -> bool:
    """Tell whether ``path`` itself is a symbolic link or a Windows reparse point (a junction included).

    Decided from ``os.lstat``, so the link is never followed.

    Args:
        path: The entry to inspect.

    Returns:
        True for a link or reparse point.

    Raises:
        OSError: The entry cannot be inspected.
    """
    status = os.lstat(path)
    attributes = getattr(status, "st_file_attributes", 0)
    return stat.S_ISLNK(status.st_mode) or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _raise_walk_error(error: OSError) -> None:
    """Re-raise a listing failure of ``os.walk``, which would otherwise skip the directory silently.

    Args:
        error: The failure ``os.walk`` reports.

    Raises:
        OSError: Always, ``error`` itself.
    """
    raise error


def _one_line(text: str) -> str:
    """Collapse every run of whitespace, newlines included, into one space.

    Args:
        text: Any message.

    Returns:
        ``text`` on a single line.
    """
    return " ".join(text.split())


def _reason(error: BaseException) -> str:
    """Return the one-line, key-free reason of a guard or filesystem failure.

    Guard and ``OSError`` messages carry paths and SIDs, never a key.

    Args:
        error: The failure.

    Returns:
        Its message on one line, or its class name when the message is empty.
    """
    return _one_line(str(error)) or type(error).__name__


class KeyRotation:
    """Moves an exposed ``.magi/`` to a backup and creates a new one.

    Example:
        >>> def vault_names(workdir, key):
        ...     return VaultClient(executable, workdir, key).ls()
        >>> rotation = KeyRotation(default_key_store(), run_ollama_init, vault_names)
        >>> report = rotation.rotate(os.getcwd(), assume_yes=True, is_tty=False)
        >>> report.names_to_revoke
        ['OPENAI_API_KEY']
    """

    def __init__(
        self,
        key_store: KeyStore,
        init: Callable[[str, KeyStore], str],
        vault_ls: Callable[[str, MagiKey], list[str]],
        backup_root: str | None = None,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Bind the rotation to its collaborators.

        Args:
            key_store: Reads the old key; its ``guard`` re-hardens the backup.
            init: Creates a new ``.magi/`` (with a new key) in a working
                directory; ``run_ollama_init`` in production.
            vault_ls: Lists the vault names of the ``.magi/`` under a directory,
                opened with a key; it receives the backup directory and the old key.
            backup_root: Where backups go; ``None`` selects
                :func:`rotation_backup_root`.
            clock: Returns the current UTC time; it names the backup.
        """
        self._store = key_store
        self._init = init
        self._vault_ls = vault_ls
        self._backup_root = backup_root if backup_root is not None else rotation_backup_root()
        self._clock = clock

    def rotate(
        self,
        workdir: str,
        assume_yes: bool,
        is_tty: bool,
        confirm: Callable[[str], str] = input,
    ) -> RotationReport:
        """Rotate the key of ``workdir``: backup, new ``.magi/``, names to revoke.

        Args:
            workdir: The project root whose ``.magi/`` is rotated.
            assume_yes: True when ``--yes`` was given; it authorises without asking.
            is_tty: True when ``confirm`` can ask an operator.
            confirm: Asks the confirmation question and returns the answer.

        Returns:
            The report of the completed rotation.

        Raises:
            RotationNotAuthorisedError: Neither ``--yes`` nor a ``y``/``Y``
                answer; nothing changed.
            RotationFailedError: The rotation stopped; the one-line message
                describes the state left behind and names the backup when one
                exists. Neither key ever appears in it.
        """
        self._authorise(workdir, assume_yes, is_tty, confirm)
        magi_dir = os.path.join(workdir, MAGI_DIR)
        if not os.path.isdir(magi_dir):
            raise RotationFailedError(f"no {MAGI_DIR}/ in {workdir}: there is nothing to rotate")
        if _is_within(self._backup_root, workdir):
            raise RotationFailedError(
                f"the rotation backup root {self._backup_root} is inside the working tree {workdir}: "
                "nothing was moved; point the state directory outside the project"
            )
        old_key = self._read_old_key(workdir)
        backup_dir = self._create_backup_dir(workdir)
        self._secure_backup_dir(backup_dir)
        self._move(magi_dir, backup_dir)
        self._reharden(backup_dir, workdir)
        self._create_new(workdir, backup_dir, old_key)
        self._verify_new(workdir, backup_dir, old_key)
        names, from_vault = self._names_to_revoke(backup_dir, old_key)
        return RotationReport(backup_dir, names, from_vault)

    @staticmethod
    def _authorise(
        workdir: str, assume_yes: bool, is_tty: bool, confirm: Callable[[str], str]
    ) -> None:
        """Require ``--yes`` or, at a terminal, a ``y``/``Y`` answer.

        Args:
            workdir: The project root, named in the question.
            assume_yes: ``--yes`` was given.
            is_tty: An operator can be asked.
            confirm: Asks the question once.

        Raises:
            RotationNotAuthorisedError: The rotation is not authorised.
        """
        if assume_yes:
            return
        if not is_tty:
            raise RotationNotAuthorisedError(
                "the key rotation needs authorisation: without a terminal, rerun with --yes "
                "(nothing was changed)"
            )
        try:
            answer = confirm(_CONFIRM_PROMPT.format(workdir=workdir))
        except EOFError:
            answer = ""
        if answer.strip() not in _AUTHORISING_ANSWERS:
            raise RotationNotAuthorisedError(
                "the key rotation was not confirmed (nothing was changed)"
            )

    def _read_old_key(self, workdir: str) -> MagiKey:
        """Read the exposed key, without verifying it, before anything moves.

        Args:
            workdir: The project root.

        Returns:
            The old key.

        Raises:
            RotationFailedError: The key cannot be read; nothing was moved.
        """
        key_file = self._store.key_path(workdir)
        try:
            return self._store.read_unverified(key_file)
        except (KeyStoreError, OSError) as error:
            raise RotationFailedError(
                f"cannot read the current key at {key_file} ({_one_line(str(error))}): nothing was moved"
            ) from error

    def _create_backup_dir(self, workdir: str) -> str:
        """Create ``<backup root>/<id>`` with owner-only mode, never reusing an existing one.

        ``<id>`` is ``<16 hex of the project path's sha256>-<UTC timestamp>``;
        a collision within the same second gets ``-2``, ``-3``, ... appended.

        Args:
            workdir: The project root, fingerprinted into the id.

        Returns:
            The created directory.

        Raises:
            RotationFailedError: The directory could not be created; nothing was moved.
        """
        fingerprint = hashlib.sha256(_canonical(workdir).encode("utf-8")).hexdigest()
        stamp = self._clock().astimezone(timezone.utc).strftime(_TIMESTAMP_FORMAT)
        base = os.path.join(self._backup_root, f"{fingerprint[:_FINGERPRINT_HEX_CHARS]}-{stamp}")
        suffixed = (f"{base}-{n}" for n in itertools.count(_FIRST_COLLISION_SUFFIX))
        candidate = next(
            path for path in itertools.chain([base], suffixed) if not os.path.lexists(path)
        )
        try:
            os.makedirs(candidate, mode=DIR_MODE)
        except OSError as error:
            raise RotationFailedError(
                f"cannot create the rotation backup {candidate} ({_one_line(str(error))}): nothing was moved"
            ) from error
        return candidate

    def _secure_backup_dir(self, backup_dir: str) -> None:
        """Restrict ``<backup root>/<id>`` to its owner and verify it, before anything goes into it.

        Args:
            backup_dir: The directory just created by :meth:`_create_backup_dir`.

        Raises:
            RotationFailedError: Hardening or verification failed; the empty
                directory is removed (or the message says it was left) and
                nothing was moved.
        """
        guard = self._store.guard
        try:
            guard.harden(backup_dir, True)
            guard.verify(backup_dir, True)
        except (KeyStoreError, OSError) as error:
            leftover = ""
            try:
                shutil.rmtree(backup_dir)
            except OSError:
                leftover = f" (the empty {backup_dir} could not be removed: delete it)"
            raise RotationFailedError(
                f"cannot restrict the rotation backup {backup_dir} to its owner "
                f"({_reason(error)}){leftover}: nothing was moved"
            ) from error

    @staticmethod
    def _source_still_complete(magi_dir: str, target: str) -> bool:
        """Tell whether every path copied to ``target`` still exists under ``magi_dir``.

        One walk of the copied tree: O(n) in the number of copied entries.

        Args:
            magi_dir: The source ``.magi/``.
            target: The (possibly partial) copy.

        Returns:
            True when nothing that reached ``target`` is missing from the source.
        """
        if not os.path.isdir(magi_dir):
            return False
        for current, dirs, files in os.walk(target):
            relative = os.path.relpath(current, target)
            for name in (*dirs, *files):
                if not os.path.lexists(os.path.normpath(os.path.join(magi_dir, relative, name))):
                    return False
        return True

    @classmethod
    def _move(cls, magi_dir: str, backup_dir: str) -> None:
        """Move ``.magi/`` into ``backup_dir``; on failure, keep whichever copy is complete.

        Across volumes ``shutil.move`` copies and then deletes the source, so a
        failure may leave the source partly deleted. The partial backup is
        removed only when the source is still complete (every path that reached
        the copy still exists in the source); otherwise nothing is deleted and
        both paths are named.

        Args:
            magi_dir: ``<workdir>/.magi``.
            backup_dir: ``<backup root>/<id>``.

        Raises:
            RotationFailedError: The move failed; the message says where the
                old ``.magi/`` is.
        """
        target = os.path.join(backup_dir, MAGI_DIR)
        try:
            shutil.move(magi_dir, target)
        except OSError as error:
            reason = _one_line(str(error))
            if not cls._source_still_complete(magi_dir, target):
                raise RotationFailedError(
                    f"moving {magi_dir} to {target} failed ({reason}) after files were removed from the source: "
                    f"both copies were kept and nothing was deleted; {target} holds what was copied, compare it "
                    f"with {magi_dir} and move the complete one back to {magi_dir} to retry"
                ) from error
            try:
                shutil.rmtree(backup_dir)
            except OSError:
                raise RotationFailedError(
                    f"moving {magi_dir} to {target} failed ({reason}): {magi_dir} is unchanged; a partial "
                    f"copy was left in {backup_dir}, delete it"
                ) from error
            raise RotationFailedError(
                f"moving {magi_dir} to {target} failed ({reason}): {magi_dir} is unchanged"
            ) from error

    def _reharden(self, backup_dir: str, workdir: str) -> None:
        """Restrict ``<id>`` and every directory and file of the moved ``.magi/`` to the owner, then verify each.

        No path relies on inheritance: a move across volumes does not carry the
        ACL, and a same-volume rename keeps each child's own, possibly loosened
        one. The tree is listed first, without following links, and any link or
        reparse point stops the rotation before anything is hardened: hardening
        it would change the permissions of its target, outside the backup. One
        walk of the moved tree, O(n) in its entries (a handful).

        Args:
            backup_dir: ``<backup root>/<id>``.
            workdir: The project root, named in the recovery instruction.

        Raises:
            RotationFailedError: Hardening or verification failed; the backup
                is kept and no new ``.magi/`` is created.
        """
        guard = self._store.guard
        try:
            entries = self._backup_entries(os.path.join(backup_dir, MAGI_DIR))
            guard.harden(backup_dir, True)
            for path, directory in entries:
                guard.harden(path, directory)
                guard.verify(path, directory)
        except (KeyStoreError, OSError, _LinkInBackupError) as error:
            raise RotationFailedError(
                f"the old {MAGI_DIR}/ was moved to {os.path.join(backup_dir, MAGI_DIR)} but could not be "
                f"restricted to its owner ({_reason(error)}), so it may be exposed; no new "
                f"{MAGI_DIR}/ was created: move it back to {os.path.join(workdir, MAGI_DIR)} to retry"
            ) from error

    @staticmethod
    def _backup_entries(magi_dir: str) -> list[tuple[str, bool]]:
        """List every directory and file of the moved ``.magi/``, refusing any link or reparse point.

        Walks with ``followlinks=False`` and inspects every entry with
        ``os.lstat`` before it is listed, so nothing outside the tree is ever
        reached. O(n) in the number of entries.

        Args:
            magi_dir: ``<backup root>/<id>/.magi``.

        Returns:
            ``(path, is_directory)`` pairs, ``magi_dir`` first.

        Raises:
            _LinkInBackupError: An entry (``magi_dir`` included) is a link or a
                reparse point; the message names its path relative to the backup.
            OSError: An entry could not be inspected or a directory listed (a
                directory the walk cannot list is never skipped silently).
        """
        backup_dir = os.path.dirname(magi_dir)

        def _refuse_link(path: str) -> None:
            if _is_link(path):
                raise _LinkInBackupError(
                    f"{os.path.relpath(path, backup_dir)} is a link or reparse point and was not followed"
                )

        _refuse_link(magi_dir)
        entries = [(magi_dir, True)]
        for current, dirs, files in os.walk(magi_dir, onerror=_raise_walk_error, followlinks=False):
            for name in dirs:
                path = os.path.join(current, name)
                _refuse_link(path)
                entries.append((path, True))
            for name in files:
                path = os.path.join(current, name)
                _refuse_link(path)
                entries.append((path, False))
        return entries

    def _create_new(self, workdir: str, backup_dir: str, old_key: MagiKey) -> None:
        """Create the new ``.magi/``; on any failure, leave none of this rotation's at the root.

        A ``.magi/`` that ``init`` found already there (``MagiDirExistsError``)
        is not this rotation's: it is left as it is and named.

        Args:
            workdir: The project root.
            backup_dir: ``<backup root>/<id>``, named in the recovery instruction.
            old_key: Redacted from the failure detail.

        Raises:
            RotationFailedError: ``init`` failed with any exception.
        """
        magi_dir = os.path.join(workdir, MAGI_DIR)
        try:
            self._init(workdir, self._store)
        except MagiDirExistsError as error:
            raise RotationFailedError(
                f"a {MAGI_DIR}/ appeared at {magi_dir} before the new one could be created; it was left "
                f"as it is because this rotation did not create it, and the old one is intact at "
                f"{os.path.join(backup_dir, MAGI_DIR)}; inspect {magi_dir} and move that aside, then move "
                f"the old one back to {magi_dir} to retry"
            ) from error
        except Exception as error:  # noqa: BLE001 - any init failure must end in the described state
            raw = " ".join([str(error), *getattr(error, "__notes__", [])])
            detail = _one_line(old_key.redact(raw)) or type(error).__name__
            leftover = ""
            if os.path.lexists(magi_dir):
                try:
                    shutil.rmtree(magi_dir)
                except OSError:
                    leftover = (
                        f" (and the half-made {magi_dir} could not be removed: delete it first)"
                    )
            raise RotationFailedError(
                f"creating the new {MAGI_DIR}/ failed ({detail}){leftover}: the old one is intact at "
                f"{os.path.join(backup_dir, MAGI_DIR)}; move it back to {magi_dir} to retry"
            ) from error

    def _verify_new(self, workdir: str, backup_dir: str, old_key: MagiKey) -> None:
        """Require the new ``.magi/`` to hold a present, owner-only key other than the old one (REQ-13g.2).

        An ``init`` double, or a ``.magi/`` restored by hand, can leave a tree
        that holds no key of this rotation; the exposed key back in service is
        never reported as rotated.

        Args:
            workdir: The project root.
            backup_dir: ``<backup root>/<id>``, named in the recovery instruction.
            old_key: The exposed key; the new one must differ from it.

        Raises:
            RotationFailedError: The key is absent, unreadable, not owner-only,
                or the old key. The ``.magi/`` at the root is left as it is (this
                rotation may not have created it); the message names it and the
                backup, never either key.
        """
        magi_dir = os.path.join(workdir, MAGI_DIR)

        def _failure(reason: str) -> RotationFailedError:
            """Build the one-line failure naming ``magi_dir``, the backup and ``reason``."""
            return RotationFailedError(
                f"the {MAGI_DIR}/ now at {magi_dir} does not hold a verified key of this rotation "
                f"({reason}); it was left as it is because this rotation may not have created it, "
                f"and the old one is intact at {os.path.join(backup_dir, MAGI_DIR)}; inspect {magi_dir} and "
                f"move that aside, then move it back to {magi_dir} to retry"
            )

        try:
            new_key = self._store.load(workdir)
        except (KeyStoreError, OSError) as error:
            raise _failure(_reason(error)) from error
        if new_key == old_key:
            raise _failure("it holds the old key")

    def _names_to_revoke(self, backup_dir: str, old_key: MagiKey) -> tuple[list[str], bool]:
        """List the old vault's names with the old key: the key's only use after the move.

        Args:
            backup_dir: ``<backup root>/<id>``, the backup's working directory.
            old_key: The exposed key.

        Returns:
            ``(names, True)`` from the vault, or ``(list(VAULT_KEY_NAMES), False)``
            when the vault cannot be read for any reason.
        """
        try:
            return list(self._vault_ls(backup_dir, old_key)), True
        except Exception:  # noqa: BLE001 - an unreadable old vault falls back to the closed list, reported
            return list(VAULT_KEY_NAMES), False
