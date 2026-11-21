# Author: Julian Bolivar
# Version: 7.1.0
# Date: 2026-11-19
"""Custody of the passphrase that opens the ``magi-rs`` memory database.

One key opens both the memory database and the vault. It lives in
``<workdir>/.magi/.secret/magi.key`` as base64 text (ASCII, no trailing
newline), restricted to its owner from the moment the file appears. The key
travels to ``magi-rs`` only through the child environment; this module never
puts its value in a message, a ``repr`` or a log.

Every failure here is fail-closed, and every exception is a SIBLING of the
project's ``ValidationError`` (it subclasses ``Exception`` directly): the
orchestrator's retry handler catches ``ValidationError``, and a missing or
exposed key must never be consumed as something a retry can fix.
"""

from __future__ import annotations

import base64
import hmac
import os
import secrets
from typing import Protocol

from link_guard import link_refusal
from cli_commands import INIT_FLAG, ROTATE_KEY_FLAG, VAULT_SET_FLAG, command
from one_line import one_line

KEY_DIR_PARTS = (".magi", ".secret")
"""The key's directory, relative to the working directory."""

KEY_FILENAME = "magi.key"
"""The key file's name inside ``KEY_DIR_PARTS``."""

KEY_BYTES = 32
"""Random bytes behind each key, before base64 encoding: 32 bytes encode to 44 base64
characters, the last one ``=`` padding."""

DIR_MODE = 0o700
"""POSIX mode of the key directory: owner only."""

FILE_MODE = 0o600
"""POSIX mode of the key file: owner read/write only."""

KEY_ENCODING = "ascii"
"""The key file holds base64 text, which is pure ASCII."""

REDACTION_MARKER = "<redacted>"
"""Replaces the key wherever text captured from a child echoes it (BDD-22)."""

_EXCLUSIVE_CREATE_FLAGS = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
"""Create-or-fail, write-only; binary on Windows so no newline translation happens."""

_ROTATE_COMMAND = command(ROTATE_KEY_FLAG)

REGENERATION_GUIDANCE = (
    f"move the existing .magi/ aside to a backup, run {command(INIT_FLAG)} (it leaves an existing .magi/ "
    f"untouched), then reload each API key with {command(VAULT_SET_FLAG, '<NAME>')}"
)
"""How to regenerate a ``.magi/`` that cannot be used (REQ-12): the one procedure, stated once.

Used by a missing or empty key, by a ``.magi/`` without its config, and by an
``--ollama-init`` that finds a ``.magi/`` already there."""


class KeyStoreError(Exception):
    """Base class of every key-custody failure (a sibling of ``ValidationError``)."""


class KeyMissingError(KeyStoreError):
    """The key is absent, empty or unreadable."""


class EmptyKeyError(KeyMissingError):
    """The key file exists but holds no key (an unreadable file is a plain ``KeyMissingError``)."""


class KeyExposedError(KeyStoreError):
    """The key or its directory is readable by someone other than its owner."""


class KeyUnverifiableError(KeyExposedError):
    """The key's permissions could not be read, so they cannot be shown to be owner-only."""


class KeyHardeningError(KeyStoreError):
    """The key's permissions could not be restricted, or the restriction could not be verified."""


class KeyGuard(Protocol):
    """Restricts and verifies owner-only permissions on a path.

    Satisfied structurally by the platform permission guards; nothing needs to
    inherit from it.
    """

    def harden(self, path: str, directory: bool) -> None:
        """Restrict ``path`` to its owner.

        Args:
            path: The file or directory to restrict.
            directory: True when ``path`` is a directory.

        Raises:
            KeyHardeningError: The restriction could not be applied.
        """

    def verify(self, path: str) -> None:
        """Check that ``path`` is restricted to its owner.

        Args:
            path: The file or directory to check.

        Raises:
            KeyUnverifiableError: The permissions of ``path`` could not be read.
            KeyExposedError: Someone other than the owner can access ``path``.
        """


class MagiKey:
    """The passphrase, redacted in every textual representation.

    The text is stored as given and not validated: its shape is guaranteed by
    ``KeyStore.new_key``.

    Example:
        >>> key = MagiKey("c2VjcmV0")
        >>> repr(key)
        'MagiKey(<redacted>)'
        >>> key.reveal()
        'c2VjcmV0'
    """

    _REDACTED = f"MagiKey({REDACTION_MARKER})"

    def __init__(self, value: str) -> None:
        """Wrap the passphrase text.

        Args:
            value: The base64 text of the key.
        """
        self._value = value

    def reveal(self) -> str:
        """Return the passphrase, for injection into the child environment only.

        Returns:
            The key text.
        """
        return self._value

    def __repr__(self) -> str:
        """Return a redacted representation.

        Returns:
            ``"MagiKey(<redacted>)"``, never the value.
        """
        return self._REDACTED

    __str__ = __repr__

    def redact(self, text: str) -> str:
        """Replace every occurrence of the key in ``text`` with ``REDACTION_MARKER``.

        Scope: only the key's exact text is replaced. A transformed copy (split
        across lines, re-encoded, with whitespace collapsed inside it) is not
        recognised, so redact BEFORE any such transformation.

        Args:
            text: Text that may echo the key, e.g. a child's output.

        Returns:
            ``text`` with the key replaced; unchanged when it does not occur
            or the key is empty (an empty pattern would match everywhere).

        Complexity:
            O(n) in the length of ``text``.

        Example:
            >>> MagiKey("c2VjcmV0").redact("got c2VjcmV0")
            'got <redacted>'
        """
        if not self._value:
            return text
        return text.replace(self._value, REDACTION_MARKER)

    def redact_bytes(self, data: bytes) -> bytes:
        """Replace every occurrence of the key's ASCII bytes in ``data`` with ``REDACTION_MARKER``.

        Same exact-bytes scope as :meth:`redact`.

        Args:
            data: Bytes that may echo the key, e.g. a child's captured stream.

        Returns:
            ``data`` with the key replaced; unchanged when it does not occur,
            the key is empty or its text cannot be encoded with ``KEY_ENCODING``.

        Complexity:
            O(n) in the length of ``data``.
        """
        if not self._value:
            return data
        try:
            needle = self._value.encode(KEY_ENCODING)
        except UnicodeEncodeError:
            return data  # a key that cannot be encoded cannot appear in the bytes either
        return data.replace(needle, REDACTION_MARKER.encode(KEY_ENCODING))

    def __eq__(self, other: object) -> bool:
        """Compare two keys by value, in constant time.

        Args:
            other: The object to compare with.

        Returns:
            True when ``other`` is a ``MagiKey`` holding the same text; False when
            either text cannot be encoded with ``KEY_ENCODING`` (never raises);
            ``NotImplemented`` for any other type.
        """
        if not isinstance(other, MagiKey):
            return NotImplemented
        try:
            mine, theirs = self._value.encode(KEY_ENCODING), other._value.encode(KEY_ENCODING)
        except UnicodeEncodeError:
            return False  # a key that cannot be written to the key file never equals anything
        return hmac.compare_digest(mine, theirs)

    def __hash__(self) -> int:
        """Refuse to hash a key: a secret must not become a dict key or a set member by accident.

        Stated explicitly (``__eq__`` alone already disables hashing) so it
        stays so if ``__eq__`` is ever removed.

        Raises:
            TypeError: Always.
        """
        raise TypeError("a MagiKey is not hashable")


class KeyStore:
    """Generates, persists and loads the key under a working directory.

    Example:
        >>> store = KeyStore(guard)  # guard: any KeyGuard
        >>> path = store.persist(workdir, store.new_key())
        >>> key = store.load(workdir)

    Attributes:
        guard: The permission guard used to harden and verify the key; set once.
    """

    def __init__(self, guard: KeyGuard) -> None:
        """Bind the store to a permission guard.

        Args:
            guard: Hardens and verifies owner-only permissions.
        """
        self.guard = guard

    def key_path(self, workdir: str) -> str:
        """Return the key file path under ``workdir``.

        Args:
            workdir: The project working directory.

        Returns:
            ``<workdir>/.magi/.secret/magi.key``.
        """
        return os.path.join(workdir, *KEY_DIR_PARTS, KEY_FILENAME)

    def new_key(self) -> MagiKey:
        """Return a new random key.

        Returns:
            A key holding the standard base64 encoding of ``KEY_BYTES`` bytes
            from the operating system's CSPRNG.
        """
        return MagiKey(base64.b64encode(secrets.token_bytes(KEY_BYTES)).decode(KEY_ENCODING))

    def persist(self, workdir: str, key: MagiKey) -> str:
        """Write ``key`` under ``workdir``, restricted from creation, then harden and verify it.

        The directory is created with ``DIR_MODE`` and hardened by the guard
        BEFORE the file exists, so on every platform and Python build the file
        is created inside an owner-only directory (on Windows ``os.makedirs``
        honours ``mode`` only on CPython builds with the CVE-2024-4030 fix; the
        guard's inheritable owner-only grant covers the others). The file is
        then created with ``FILE_MODE`` through an exclusive create, so there is
        no check-then-write race: an existing key makes the create itself fail.
        Directory and file are hardened and verified again after the write.

        Args:
            workdir: The project working directory.
            key: The key to write.

        Returns:
            The path of the key file written.

        Raises:
            FileExistsError: A key already exists; it is left untouched.
            KeyHardeningError: ``.magi``, the key directory or the key path is a link
                or reparse point (nothing is written); hardening or verification failed; the file this
                call created is removed (none exists when the directory could
                not be hardened first).
            OSError: The directory or the file could not be created.
        """
        key_file = self.key_path(workdir)
        secret_dir = os.path.dirname(key_file)
        self._refuse_links(workdir, secret_dir, key_file)
        os.makedirs(secret_dir, mode=DIR_MODE, exist_ok=True)
        try:
            self.guard.harden(secret_dir, True)
        except KeyStoreError as error:
            raise KeyHardeningError(
                f"could not restrict the key directory {secret_dir} to its owner "
                f"({one_line(str(error))}): no key was written"
            ) from error
        descriptor = os.open(key_file, _EXCLUSIVE_CREATE_FLAGS, FILE_MODE)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(key.reveal().encode(KEY_ENCODING))
            self.guard.harden(secret_dir, True)
            self.guard.harden(key_file, False)
            self.guard.verify(secret_dir)
            self.guard.verify(key_file)
        except KeyStoreError as error:
            cleanup_note = self._discard(key_file)
            raise KeyHardeningError(
                f"could not restrict the key at {key_file} to its owner ({one_line(str(error))}){cleanup_note}"
            ) from error
        except BaseException:
            self._discard(key_file)
            raise
        return key_file

    @staticmethod
    def _refuse_links(workdir: str, secret_dir: str, key_file: str) -> None:
        """Refuse to write through a link: the key would land wherever it points.

        Checked before anything is created or opened, on ``<workdir>/.magi``, the key
        directory and the key path. An absent entry is fine (it is about to be made).

        Args:
            workdir: The project working directory.
            secret_dir: ``<workdir>/.magi/.secret``.
            key_file: The key file path.

        Raises:
            KeyHardeningError: An entry is a link or reparse point, or cannot be inspected;
                nothing was written.
        """
        for entry in (os.path.join(workdir, KEY_DIR_PARTS[0]), secret_dir, key_file):
            try:
                refusal = link_refusal(entry)
            except FileNotFoundError:
                continue
            except OSError as error:
                raise KeyHardeningError(
                    f"cannot inspect {entry} before writing the key ({one_line(str(error))}): "
                    "no key was written"
                ) from error
            if refusal is not None:
                raise KeyHardeningError(f"{refusal}: no key was written")

    @staticmethod
    def _discard(key_file: str) -> str:
        """Remove a key file this store just created, never masking the failure being handled.

        Args:
            key_file: The key file to remove.

        Returns:
            An empty string when the file was removed; otherwise a message
            suffix naming the file that could not be removed.
        """
        try:
            os.remove(key_file)
        except OSError:
            return f" (and the key file {key_file} could not be removed: delete it by hand)"
        return ""

    def load(self, workdir: str) -> MagiKey:
        """Verify the key's permissions, then read it.

        Args:
            workdir: The project working directory.

        Returns:
            The key.

        The message of a permission failure ends with the guard's own one-line
        detail (paths and SIDs only, never the key), so the operator sees why.

        Raises:
            KeyMissingError: The key is absent, empty or unreadable.
            KeyUnverifiableError: The permissions could not be read (a tool or
                read failure); the message says to fix the cause, never that the
                key is compromised. The file is not read.
            KeyExposedError: The key or its directory is readable by others;
                the file is not read.
        """
        key_file = self.key_path(workdir)
        if not os.path.exists(key_file):
            raise KeyMissingError(f"no magi-rs memory key at {key_file}: {REGENERATION_GUIDANCE}")
        try:
            self.guard.verify(os.path.dirname(key_file))
            self.guard.verify(key_file)
        except KeyUnverifiableError as error:
            raise KeyUnverifiableError(
                f"the permissions of {key_file} could not be verified ({one_line(str(error))}); "
                f"fix the cause before considering {_ROTATE_COMMAND}"
            ) from error
        except KeyExposedError as error:
            raise KeyExposedError(
                f"the key at {key_file} or its directory is readable by others: it may have been "
                f"compromised, so treat the memory database and the vault secrets as compromised "
                f"and rotate the key with {_ROTATE_COMMAND} ({one_line(str(error))})"
            ) from error
        return self.read_unverified(key_file)

    def read_unverified(self, key_file: str) -> MagiKey:
        """Read ``key_file`` without verifying its permissions.

        The only read of the key file. Outside ``load`` it is used solely by
        the key rotation, on the backup of an exposed key.

        Args:
            key_file: The key file path.

        Returns:
            The key, with surrounding whitespace removed.

        Raises:
            EmptyKeyError: The file exists and is empty (a ``KeyMissingError``).
            KeyMissingError: The file cannot be read or decoded.
        """
        try:
            with open(key_file, encoding=KEY_ENCODING) as handle:
                value = handle.read().strip()
        except (OSError, UnicodeDecodeError) as error:
            raise KeyMissingError(f"cannot read the magi-rs memory key at {key_file}") from error
        if not value:
            raise EmptyKeyError(
                f"the magi-rs memory key at {key_file} is empty: {REGENERATION_GUIDANCE}"
            )
        return MagiKey(value)
