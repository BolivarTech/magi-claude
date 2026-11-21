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
key. Every step after the move handles any exception, ``KeyboardInterrupt``
and ``SystemExit`` included: it reaches its described state first, then raises
:class:`RotationFailedError` naming the backup, chained to the cause. Only a
hard stop (killed process, power loss) between steps 3 and 4 prints nothing:
the root then has no ``.magi/`` and the old one is intact in the backup, from
where it can be moved back.

Side effects of :meth:`KeyRotation.rotate`: creates ``<backup root>/<id>/``,
moves ``<workdir>/.magi`` into it, restricts the backup's permissions, and
creates a new ``<workdir>/.magi`` through ``init``.

Error paths (state left on disk, the clause of the one-line message, what the
operator is told to do). Each row is pinned by tests in ``tests/test_magi_key_rotation.py``:
the ``init`` rows by the parametrized
``test_the_rotation_words_the_failure_by_the_leftover_init_reported``, the others by the
test named for their stage (move, re-harden, verify, names), so this table is a map of
pinned behaviour and not a second, unchecked source::

    stage        state left on disk                       message says           operator verb
    -----------  ---------------------------------------  ---------------------  --------------
    pre-move     nothing changed (empty backup dirs this   "nothing was moved"    fix / retry
                 call created are removed, or named)
    pre-move     a link in .magi/                          "remove the link"      remove, retry
    pre-move     an entry that cannot be listed            "fix the cause"        fix, retry
    move         source unchanged (partial copy removed)   "is unchanged"         retry
    move         source gone, complete copy in the backup  "holds the complete    move back
                                                           copy"
    move         source partly gone, partial copy          "both copies were      compare, move
                 kept as well                              kept"                  back
    re-harden    old .magi/ in the backup, maybe exposed,  "may be exposed"       move back
                 no new .magi/
    init failed  InitLeftover.REMOVED: no new .magi/       "failed (...)"         move back
    init failed  KEY_WRITTEN_KEPT / HARDENING_FAILED_KEPT: "was kept"             remove first
                 the new .magi/ stays (key unprotected
                 in the second case)
    init failed  FOREIGN_KEPT: another process made the    "another process"      move aside
                 .magi/ now there, left in place
    init failed  HALF_MADE_KEPT: a half-made .magi/ that   "could not be          delete it
                 could not be removed                      removed"
    verify new   a .magi/ without a verified key of this   "does not hold a       inspect, move
                 rotation, left as it is                   verified key"          aside, move back
    verify new   this rotation's own key could not be      "created by this       fix the cause
                 verified                                  rotation"              (never move back
                                                                                  the old one)
    names        new .magi/ in place, listing the old      "listing ... was       revoke every
                 vault was interrupted                     interrupted"           key, store new
"""

from __future__ import annotations

import hashlib
import itertools
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from cli_commands import YES_FLAG
from magi_key import (
    DIR_MODE,
    KEY_DIR_PARTS,
    KeyGuard,
    KeyStore,
    KeyStoreError,
    MagiKey,
)
from magi_key_permissions import link_refusal
from magi_rs_vault import VAULT_KEY_NAMES
from ollama_init import InitLeftover, MagiDirExistsError, OllamaInitError
from one_line import one_line

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

_MAX_BACKUP_CANDIDATES = 100
"""How many backup names (the plain one, then ``-2``, ``-3``, ...) are tried before giving up."""

_AUTHORISING_ANSWERS = frozenset({"y", "Y"})
"""The only answers to the confirmation prompt that authorise a rotation."""

_CONFIRM_PROMPT = (
    "Rotate the magi-rs key of {workdir}? The current .magi/ is moved to a backup and its memory "
    "and history are lost. [y/N] "
)


class RotationNotAuthorisedError(Exception):
    """The rotation was not authorised; nothing changed (a sibling of ``ValidationError``)."""


_LEFTOVER_CLAUSES = {
    InitLeftover.REMOVED: "",
    InitLeftover.KEY_WRITTEN_KEPT: (
        " (the new {magi_dir} was kept: initialisation failed after its key was written; remove it first)"
    ),
    InitLeftover.HARDENING_FAILED_KEPT: (
        " (the new {magi_dir} was kept: its key could not be hardened and may be readable by others; "
        "remove it first)"
    ),
    InitLeftover.FOREIGN_KEPT: (
        " (and the {magi_dir} now there holds a key this rotation did not write: another process "
        "initialised it, so it was left in place; move it aside first)"
    ),
    InitLeftover.HALF_MADE_KEPT: " (and the half-made {magi_dir} could not be removed: delete it first)",
}
"""The clause of a failed ``init``'s message, by what ``run_ollama_init`` reported it left."""


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
            False when it could not be listed exactly and the names are the whole
            closed list (``VAULT_KEY_NAMES``).
        listing_detail: Why the listing fell back, on one line, key-redacted (it names
            an unrecognised entry, never a value); empty when the names came from the vault.
    """

    backup_dir: str
    names_to_revoke: list[str]
    names_from_vault: bool
    listing_detail: str = ""


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


def _raise_walk_error(error: OSError) -> None:
    """Re-raise a listing failure of ``os.walk``, which would otherwise skip the directory silently.

    Args:
        error: The failure ``os.walk`` reports.

    Raises:
        OSError: Always, ``error`` itself.
    """
    raise error


def _restore_instruction(destination: str, *, inspect_first: bool = False, item: str = "it") -> str:
    """Return the recovery instruction that ends a rotation message: put the old ``.magi/`` back.

    Args:
        destination: The ``.magi/`` path the old tree goes back to.
        inspect_first: True when something else now sits at ``destination``, so the
            operator must look at it and move it aside before restoring.
        item: How the instruction names what is moved back (``"it"``, ``"the old one"``).

    Returns:
        ``"move <item> back to <destination> to retry"``, preceded by the inspect-and-move-aside
        step when ``inspect_first`` is true.
    """
    restore = f"move {item} back to {destination} to retry"
    if inspect_first:
        return f"inspect {destination} and move that aside, then {restore}"
    return restore


class _RecordingKeyStore(KeyStore):
    """The store handed to ``init``: a ``KeyStore`` that remembers the keys it was asked to persist.

    Lets the rotation tell its own just-written key from one another process
    wrote, after ``init`` failed or was interrupted.
    """

    def __init__(self, guard: KeyGuard) -> None:
        """Bind the store to the rotation's permission guard.

        Args:
            guard: The guard of the rotation's store.
        """
        super().__init__(guard)
        self._written: list[MagiKey] = []

    def persist(self, workdir: str, key: MagiKey) -> str:
        """Record ``key`` (before the write, so a write that fails halfway still counts), then persist it.

        Args:
            workdir: The project working directory.
            key: The key to write.

        Returns:
            The path of the key file written.

        Raises:
            Whatever ``KeyStore.persist`` raises.
        """
        self._written.append(key)
        return super().persist(workdir, key)

    def holds_written_key(self, workdir: str) -> bool:
        """Tell whether the key now at ``workdir`` is one this store was asked to persist.

        Args:
            workdir: The project working directory.

        Returns:
            True only when the key file reads back as a recorded key; an
            absent, empty or unreadable file is not this store's.
        """
        try:
            current = self.read_unverified(self.key_path(workdir))
        except (KeyStoreError, OSError):
            return False
        return any(current == key for key in self._written)

    def redact(self, text: str) -> str:
        """Replace every key this store was asked to persist in ``text`` with ``REDACTION_MARKER``.

        Args:
            text: Text that may echo a new key, e.g. an ``init`` failure.

        Returns:
            ``text`` with every recorded key redacted.
        """
        for key in self._written:
            text = key.redact(text)
        return text


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
        self._redactors: list[Callable[[str], str]] = []

    def _reason(self, error: BaseException) -> str:
        """Return the one-line, key-free reason of a guard or filesystem failure.

        Every key this rotation has handled so far (the old one once read, each
        one the new store persisted) is redacted first, so the discipline does
        not depend on each message being key-free.

        Args:
            error: The failure.

        Returns:
            Its redacted message on one line, or its class name when the message is empty.
        """
        text = str(error)
        for redact in self._redactors:
            text = redact(text)
        return one_line(text) or type(error).__name__

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
        if not os.path.isabs(self._backup_root):
            raise RotationFailedError(
                f"the rotation backup root {self._backup_root} is not an absolute path: nothing was done; "
                "point the state directory at an absolute path"
            )
        self._authorise(workdir, assume_yes, is_tty, confirm)
        magi_dir = os.path.join(workdir, MAGI_DIR)
        if not os.path.isdir(magi_dir):
            raise RotationFailedError(f"no {MAGI_DIR}/ in {workdir}: there is nothing to rotate")
        if _is_within(self._backup_root, workdir):
            raise RotationFailedError(
                f"the rotation backup root {self._backup_root} is inside the working tree {workdir}: "
                "nothing was moved; point the state directory outside the project"
            )
        self._redactors = []
        self._refuse_links_in_source(magi_dir)
        old_key = self._read_old_key(workdir)
        self._redactors.append(old_key.redact)
        backup_dir, created_parents = self._create_backup_dir(workdir)
        self._secure_backup_dir(backup_dir, created_parents)
        self._move(magi_dir, backup_dir, created_parents)
        self._reharden(backup_dir, workdir)
        new_store = _RecordingKeyStore(self._store.guard)
        self._redactors.append(new_store.redact)
        self._create_new(workdir, backup_dir, new_store)
        self._verify_new(workdir, backup_dir, old_key, new_store)
        names, from_vault, detail = self._names_to_revoke(workdir, backup_dir, old_key)
        return RotationReport(backup_dir, names, from_vault, detail)

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
                f"the key rotation needs authorisation: without a terminal, rerun with {YES_FLAG} "
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

    def _refuse_links_in_source(self, magi_dir: str) -> None:
        """Refuse a source ``.magi/`` that holds a link or reparse point, before anything moves.

        Across volumes ``shutil.move`` copies with ``symlinks=False`` and would
        follow such a link into whatever it points at; the walk itself never
        follows one (``_backup_entries``). O(n) in the entries of ``.magi/``.

        Residual, declared: a link planted between this scan and a
        cross-volume move is still followed by ``copytree``. That needs a local
        process racing the rotation inside the owner's own ``.magi/``, which is
        outside the threat model (DIR-4); the post-move walk then refuses to
        harden it.

        Args:
            magi_dir: ``<workdir>/.magi``.

        Raises:
            RotationFailedError: A link was found or an entry could not be
                inspected; nothing was moved.
        """
        try:
            self._backup_entries(magi_dir)
        except (_LinkInBackupError, OSError) as error:
            advice = (
                "remove the link and retry"
                if isinstance(error, _LinkInBackupError)
                else "fix the cause and retry"
            )
            raise RotationFailedError(
                f"the {MAGI_DIR}/ at {magi_dir} cannot be rotated safely ({self._reason(error)}): nothing was "
                f"moved; {advice}"
            ) from error

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
                f"cannot read the current key at {key_file} ({self._reason(error)}): nothing was moved"
            ) from error

    @staticmethod
    def _remove_created(created: list[str]) -> str:
        """Remove the empty directories a failed backup creation made, innermost first, best effort.

        Args:
            created: The directories the call created, outermost first; one that
                does not exist is skipped.

        Returns:
            An empty string when all were removed; otherwise a clause naming
            those that could not be (non-empty or in use) so the operator can delete them.
        """
        stuck = []
        for directory in reversed(created):
            try:
                os.rmdir(directory)
            except FileNotFoundError:
                continue
            except OSError:
                stuck.append(directory)
        if not stuck:
            return ""
        return f" (these empty directories could not be removed, delete them: {', '.join(stuck)})"

    def _feature_root(self) -> str:
        """Return the topmost directory this feature owns on the way to the backup root.

        With the default root that is ``<state base>/magi`` (the ``_BACKUP_PARTS``
        components); a custom root is owned only from itself down. What lies above it
        (the platform's state directory) is created when missing but never hardened,
        recorded or cleaned up.

        Returns:
            The path of the topmost feature-owned directory.
        """
        root = self._backup_root
        if os.path.normcase(root).endswith(os.path.normcase(os.path.join(*_BACKUP_PARTS))):
            for _ in range(len(_BACKUP_PARTS) - 1):
                root = os.path.dirname(root)
        return root

    def _create_backup_dir(self, workdir: str) -> tuple[str, list[str]]:
        """Create ``<backup root>/<id>`` with owner-only mode, never reusing an existing one.

        ``<id>`` is ``<16 hex of the project path's sha256>-<UTC timestamp>``;
        a collision within the same second gets ``-2``, ``-3``, ... appended.
        ``os.makedirs`` applies the mode to the leaf only, so the feature-owned
        parents this call creates (:meth:`_feature_root` and below) are returned for the
        caller to restrict; parents that already existed, and any missing directory
        above the feature-owned root, are never restricted or recorded.

        Args:
            workdir: The project root, fingerprinted into the id.

        Returns:
            ``(the created directory, the parents this call created, outermost first)``.

        Raises:
            RotationFailedError: No free backup name was found within
                ``_MAX_BACKUP_CANDIDATES`` tries, or the directory could not be created, or creation was
                interrupted; the empty directories this call made are removed (any
                that cannot be are named) and nothing was moved.
        """
        fingerprint = hashlib.sha256(_canonical(workdir).encode("utf-8")).hexdigest()
        stamp = self._clock().astimezone(timezone.utc).strftime(_TIMESTAMP_FORMAT)
        base = os.path.join(self._backup_root, f"{fingerprint[:_FINGERPRINT_HEX_CHARS]}-{stamp}")
        suffixed = (f"{base}-{n}" for n in itertools.count(_FIRST_COLLISION_SUFFIX))
        candidate = next(
            (
                path
                for path in itertools.islice(
                    itertools.chain([base], suffixed), _MAX_BACKUP_CANDIDATES
                )
                if not os.path.lexists(path)
            ),
            None,
        )
        if candidate is None:
            raise RotationFailedError(
                f"cannot find a free backup name in {self._backup_root} after "
                f"{_MAX_BACKUP_CANDIDATES} tries: nothing was moved; clear old backups there"
            )
        created_parents: list[str] = []
        parent = os.path.dirname(candidate)
        platform_base = os.path.dirname(self._feature_root())
        while (
            parent != platform_base
            and not os.path.lexists(parent)
            and os.path.dirname(parent) != parent
        ):
            created_parents.insert(0, parent)
            parent = os.path.dirname(parent)
        try:
            os.makedirs(candidate, mode=DIR_MODE)
        except BaseException as error:  # noqa: BLE001 - before the move, even an interrupt must end described
            # A FileExistsError means another process made the leaf (and so its parents): nothing here is ours.
            ours = [] if isinstance(error, FileExistsError) else [*created_parents, candidate]
            leftover = self._remove_created(ours)
            raise RotationFailedError(
                f"cannot create the rotation backup {candidate} ({self._reason(error)}){leftover}: "
                "nothing was moved"
            ) from error
        return candidate, created_parents

    def _secure_backup_dir(self, backup_dir: str, created_parents: list[str]) -> None:
        """Restrict the parents the rotation created and ``<backup root>/<id>`` to the owner, then verify each.

        Done before anything goes into the backup.

        Args:
            backup_dir: The directory just created by :meth:`_create_backup_dir`.
            created_parents: The parents that call created, outermost first.

        Raises:
            RotationFailedError: Hardening or verification failed, or was
                interrupted (``KeyboardInterrupt``, ``SystemExit``); the empty
                directories this rotation created are removed (any that cannot be
                are named) and nothing was moved.
        """
        guard = self._store.guard
        directory = backup_dir
        try:
            for directory in (*created_parents, backup_dir):
                guard.harden(directory, True)
                guard.verify(directory)
        except BaseException as error:  # noqa: BLE001 - before the move, even an interrupt must end described
            leftover = self._remove_created([*created_parents, backup_dir])
            raise RotationFailedError(
                f"cannot restrict the rotation backup directory {directory} to its owner "
                f"({self._reason(error)}){leftover}: nothing was moved"
            ) from error

    @staticmethod
    def _source_still_complete(magi_dir: str, target: str) -> bool:
        """Tell whether every path copied to ``target`` still exists under ``magi_dir``.

        One walk of the copied tree: O(n) in the number of copied entries. A
        directory the walk cannot list is never skipped: completeness that
        cannot be proven is reported as not complete, so the caller keeps both
        copies (the partial one may hold the only copy of what the source lost).

        Args:
            magi_dir: The source ``.magi/``.
            target: The (possibly partial) copy.

        Returns:
            True only when every path that reached ``target`` provably still
            exists in the source.
        """
        if not os.path.isdir(magi_dir):
            return False
        try:
            os.lstat(target)
        except FileNotFoundError:
            return True  # nothing was copied: the source is trivially complete
        except OSError:
            return False  # a copy that cannot be inspected cannot be proven absent
        try:
            for current, dirs, files in os.walk(target, onerror=_raise_walk_error):
                relative = os.path.relpath(current, target)
                for name in (*dirs, *files):
                    if not os.path.lexists(
                        os.path.normpath(os.path.join(magi_dir, relative, name))
                    ):
                        return False
        except OSError:
            return False
        return True

    def _move(self, magi_dir: str, backup_dir: str, created_parents: list[str]) -> None:
        """Move ``.magi/`` into ``backup_dir``; on failure, keep whichever copy is complete.

        Across volumes ``shutil.move`` copies and then deletes the source, so a
        failure may leave the source partly deleted, or gone (the copy is then
        complete, since it finishes before the delete starts). The partial backup is
        removed only when the source is still complete (every path that reached
        the copy still exists in the source); otherwise nothing is deleted and
        both paths are named. An interrupt (``KeyboardInterrupt``,
        ``SystemExit``) during a long cross-volume copy follows the same
        contract as a failed move.

        Args:
            magi_dir: ``<workdir>/.magi``.
            backup_dir: ``<backup root>/<id>``.
            created_parents: The parents of ``backup_dir`` this rotation created,
                outermost first; removed when the source is unchanged.

        Raises:
            RotationFailedError: The move failed or was interrupted; the message
                says where the old ``.magi/`` is.
        """
        target = os.path.join(backup_dir, MAGI_DIR)
        try:
            shutil.move(magi_dir, target)
        except BaseException as error:  # noqa: BLE001 - an interrupted copy must end in the described state
            reason = self._reason(error)
            if not os.path.lexists(magi_dir):
                raise RotationFailedError(
                    f"moving {magi_dir} to {target} failed ({reason}) after the copy finished: the source is "
                    f"gone and {target} holds the complete copy; {_restore_instruction(magi_dir)}"
                ) from error
            if not self._source_still_complete(magi_dir, target):
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
            leftover = self._remove_created(created_parents)
            raise RotationFailedError(
                f"moving {magi_dir} to {target} failed ({reason}): {magi_dir} is unchanged{leftover}"
            ) from error

    def _reharden(self, backup_dir: str, workdir: str) -> None:
        """Restrict ``<id>`` again, then restrict and verify every directory and file of the moved ``.magi/``.

        ``<id>`` itself was verified before the move (:meth:`_secure_backup_dir`)
        and is only hardened again here; every entry of the moved ``.magi/``
        (the directory itself included) is hardened and then verified. No path
        relies on inheritance: a move across volumes does not carry the
        ACL, and a same-volume rename keeps each child's own, possibly loosened
        one. The tree is listed first, without following links, and any link or
        reparse point stops the rotation before anything is hardened: hardening
        it would change the permissions of its target, outside the backup. One
        walk of the moved tree, O(n) in its entries (a handful).

        Args:
            backup_dir: ``<backup root>/<id>``.
            workdir: The project root, named in the recovery instruction.

        Raises:
            RotationFailedError: Hardening or verification failed, or was
                interrupted; the backup is kept and no new ``.magi/`` is created.
        """
        guard = self._store.guard
        try:
            entries = self._backup_entries(os.path.join(backup_dir, MAGI_DIR))
            guard.harden(backup_dir, True)
            for path, directory in entries:
                guard.harden(path, directory)
                guard.verify(path)
        except BaseException as error:  # noqa: BLE001 - after the move, even an interrupt must name the backup
            raise RotationFailedError(
                f"the old {MAGI_DIR}/ was moved to {os.path.join(backup_dir, MAGI_DIR)} but could not be "
                f"restricted to its owner ({self._reason(error)}), so it may be exposed; no new "
                f"{MAGI_DIR}/ was created: {_restore_instruction(os.path.join(workdir, MAGI_DIR))}"
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
            refusal = link_refusal(path, os.path.relpath(path, backup_dir))
            if refusal is not None:
                raise _LinkInBackupError(refusal)

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

    def _create_new(self, workdir: str, backup_dir: str, store: _RecordingKeyStore) -> None:
        """Create the new ``.magi/``; on any failure, leave none of this rotation's at the root.

        What a failed ``init`` left is taken from its typed report
        (``OllamaInitError.leftover``, decided by ``run_ollama_init``'s own
        cleanup); the rotation never decides who owns a key. A ``.magi/`` that
        ``init`` found already there (``MagiDirExistsError``) is left as it is
        and named. An interrupt after this rotation's store persisted the key
        keeps the new ``.magi/`` with the interrupt wording. Any other failure
        gets the generic wording: a key-less ``.magi/`` is removed, one holding a
        key is left as it is.

        Args:
            workdir: The project root.
            backup_dir: ``<backup root>/<id>``, named in the recovery instruction.
            store: The store handed to ``init``; it records the keys persisted.
                The failure detail is redacted through ``self._redactors``, which
                :meth:`rotate` filled with the old key and this store.

        Raises:
            RotationFailedError: ``init`` failed with any exception,
                ``KeyboardInterrupt`` and ``SystemExit`` included.
        """
        magi_dir = os.path.join(workdir, MAGI_DIR)
        try:
            self._init(workdir, store)
        except MagiDirExistsError as error:
            raise RotationFailedError(
                f"a {MAGI_DIR}/ appeared at {magi_dir} before the new one could be created; it was left "
                f"as it is because this rotation did not create it, and the old one is intact at "
                f"{os.path.join(backup_dir, MAGI_DIR)}; "
                f"{_restore_instruction(magi_dir, inspect_first=True, item='the old one')}"
            ) from error
        except BaseException as error:  # noqa: BLE001 - any init failure, an interrupt included, ends in the described state
            detail = self._reason(error)
            if isinstance(error, (KeyboardInterrupt, SystemExit)) and store.holds_written_key(
                workdir
            ):
                leftover = (
                    f" (the new {magi_dir} was kept: initialisation was interrupted after its key was written; "
                    "move it aside first)"
                )
            # An unmapped leftover value gets the generic wording below, never a KeyError.
            elif isinstance(error, OllamaInitError) and error.leftover in _LEFTOVER_CLAUSES:
                leftover = _LEFTOVER_CLAUSES[error.leftover].format(magi_dir=magi_dir)
            else:
                leftover = self._unreported_leftover(workdir, magi_dir)
            raise RotationFailedError(
                f"creating the new {MAGI_DIR}/ failed ({detail}){leftover}: the old one is intact at "
                f"{os.path.join(backup_dir, MAGI_DIR)}; {_restore_instruction(magi_dir)}"
            ) from error

    def _unreported_leftover(self, workdir: str, magi_dir: str) -> str:
        """Handle a ``.magi/`` that an init failure left without a typed report, guessing nothing.

        A ``.magi/`` without a key holds nothing secret and is removed; one
        holding a key is left as it is, whoever wrote it.

        Args:
            workdir: The project root.
            magi_dir: ``<workdir>/.magi``.

        Returns:
            The parenthetical clause of the failure message; empty when nothing is left.
        """
        if os.path.lexists(self._store.key_path(workdir)):
            return f" (a {magi_dir} holding a key was left as it is: move it aside first)"
        if os.path.lexists(magi_dir):
            try:
                shutil.rmtree(magi_dir)
            except OSError:
                return _LEFTOVER_CLAUSES[InitLeftover.HALF_MADE_KEPT].format(magi_dir=magi_dir)
        return ""

    def _verify_new(
        self, workdir: str, backup_dir: str, old_key: MagiKey, store: _RecordingKeyStore
    ) -> None:
        """Require the new ``.magi/`` to hold a present, owner-only key other than the old one (REQ-13g.2).

        An ``init`` double, or a ``.magi/`` restored by hand, can leave a tree
        that holds no key of this rotation; the exposed key back in service is
        never reported as rotated.

        Args:
            workdir: The project root.
            backup_dir: ``<backup root>/<id>``, named in the recovery instruction.
            old_key: The exposed key; the new one must differ from it.
            store: The store ``init`` persisted through; it tells whether the key
                on disk is this rotation's.

        Raises:
            RotationFailedError: The key is absent, unreadable, not owner-only,
                or the old key, or the check was interrupted. The ``.magi/`` at
                the root is left as it is; the message names it and the backup,
                never either key. When the key on disk is this rotation's, the
                message owns the new ``.magi/`` and says not to move the exposed
                backup back; otherwise it says this rotation may not have created it.
        """
        magi_dir = os.path.join(workdir, MAGI_DIR)

        def _failure(reason: str) -> RotationFailedError:
            """Build the one-line failure naming ``magi_dir``, the backup and ``reason``."""
            return RotationFailedError(
                f"the {MAGI_DIR}/ now at {magi_dir} does not hold a verified key of this rotation "
                f"({reason}); it was left as it is because this rotation may not have created it, "
                f"and the old one is intact at {os.path.join(backup_dir, MAGI_DIR)}; "
                f"{_restore_instruction(magi_dir, inspect_first=True)}"
            )

        try:
            new_key = self._store.load(workdir)
        except BaseException as error:  # noqa: BLE001 - after the move, even an interrupt must name the backup
            if store.holds_written_key(workdir):
                raise RotationFailedError(
                    f"the new {MAGI_DIR}/ at {magi_dir} was created by this rotation but its key could not be "
                    f"verified ({self._reason(error)}); fix the cause, then run any --ollama command (its preflight "
                    f"verifies the key); the old one is at "
                    f"{os.path.join(backup_dir, MAGI_DIR)}: do not move the exposed backup back"
                ) from error
            raise _failure(self._reason(error)) from error
        if new_key == old_key:
            raise _failure("it holds the old key")

    def _names_to_revoke(
        self, workdir: str, backup_dir: str, old_key: MagiKey
    ) -> tuple[list[str], bool, str]:
        """List the old vault's names with the old key: the key's only use after the move.

        Args:
            workdir: The project root, whose new ``.magi/`` is named if the listing is interrupted.
            backup_dir: ``<backup root>/<id>``, the backup's working directory.
            old_key: The exposed key.

        Returns:
            ``(names, True, "")`` from the vault, or ``(list(VAULT_KEY_NAMES), False,
            <one-line reason>)`` when the vault cannot be listed exactly for any reason.

        Raises:
            RotationFailedError: The listing was interrupted (``KeyboardInterrupt``,
                ``SystemExit``). The new ``.magi/`` is already in place; the
                message names it and the backup and lists every name to revoke.
        """
        try:
            return list(self._vault_ls(backup_dir, old_key)), True, ""
        except Exception as error:  # noqa: BLE001 - an unlistable old vault falls back to the closed list, reported
            return list(VAULT_KEY_NAMES), False, self._reason(error)
        except BaseException as error:
            raise RotationFailedError(
                f"the new {MAGI_DIR}/ is in place at {os.path.join(workdir, MAGI_DIR)} and the old one is at "
                f"{os.path.join(backup_dir, MAGI_DIR)}, but listing the old vault was interrupted "
                f"({self._reason(error)}): treat every API key it may have held as compromised, revoke each one "
                f"at its provider ({', '.join(VAULT_KEY_NAMES)}) and store the new values"
            ) from error
