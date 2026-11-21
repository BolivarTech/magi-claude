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

KEY_DIR_PARTS = (".magi", ".secret")
"""The key's directory, relative to the working directory."""

KEY_FILENAME = "magi.key"
"""The key file's name inside ``KEY_DIR_PARTS``."""

KEY_BYTES = 32
"""Random bytes behind each key, before base64 encoding."""

DIR_MODE = 0o700
"""POSIX mode of the key directory: owner only."""

FILE_MODE = 0o600
"""POSIX mode of the key file: owner read/write only."""

KEY_ENCODING = "ascii"
"""The key file holds base64 text, which is pure ASCII."""

_EXCLUSIVE_CREATE_FLAGS = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
"""Create-or-fail, write-only; binary on Windows so no newline translation happens."""

_REGENERATE_COMMAND = "--ollama-init"
_ROTATE_COMMAND = "--ollama-rotate-key"


class KeyStoreError(Exception):
    """Base class of every key-custody failure (a sibling of ``ValidationError``)."""


class KeyMissingError(KeyStoreError):
    """The key is absent, empty or unreadable."""


class KeyExposedError(KeyStoreError):
    """The key or its directory is readable by someone other than its owner."""


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

    def verify(self, path: str, directory: bool) -> None:
        """Check that ``path`` is restricted to its owner.

        Args:
            path: The file or directory to check.
            directory: True when ``path`` is a directory.

        Raises:
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

    _REDACTED = "MagiKey(<redacted>)"

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

    def __eq__(self, other: object) -> bool:
        """Compare two keys by value, in constant time.

        Args:
            other: The object to compare with.

        Returns:
            True when ``other`` is a ``MagiKey`` holding the same text;
            ``NotImplemented`` for any other type.
        """
        if not isinstance(other, MagiKey):
            return NotImplemented
        return hmac.compare_digest(self._value.encode(), other._value.encode())


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

        The directory is created with ``DIR_MODE`` and the file with
        ``FILE_MODE`` through an exclusive create, so there is no window in
        which either is readable by others and no check-then-write race: an
        existing key makes the create itself fail.

        Args:
            workdir: The project working directory.
            key: The key to write.

        Returns:
            The path of the key file written.

        Raises:
            FileExistsError: A key already exists; it is left untouched.
            KeyHardeningError: Hardening or verification failed; the file this
                call created is removed.
            OSError: The directory or the file could not be created.
        """
        key_file = self.key_path(workdir)
        secret_dir = os.path.dirname(key_file)
        os.makedirs(secret_dir, mode=DIR_MODE, exist_ok=True)
        descriptor = os.open(key_file, _EXCLUSIVE_CREATE_FLAGS, FILE_MODE)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(key.reveal().encode(KEY_ENCODING))
            self.guard.harden(secret_dir, True)
            self.guard.harden(key_file, False)
            self.guard.verify(secret_dir, True)
            self.guard.verify(key_file, False)
        except KeyStoreError as error:
            os.remove(key_file)
            raise KeyHardeningError(
                f"could not restrict the key at {key_file} to its owner"
            ) from error
        except BaseException:
            os.remove(key_file)
            raise
        return key_file

    def load(self, workdir: str) -> MagiKey:
        """Verify the key's permissions, then read it.

        Args:
            workdir: The project working directory.

        Returns:
            The key.

        Raises:
            KeyMissingError: The key is absent, empty or unreadable.
            KeyExposedError: The key or its directory is readable by others;
                the file is not read.
        """
        key_file = self.key_path(workdir)
        if not os.path.exists(key_file):
            raise KeyMissingError(
                f"no magi-rs memory key at {key_file}: run {_REGENERATE_COMMAND} to create a new .magi/ "
                f"(a 7.x .magi/ without a key must first be moved aside to a backup)"
            )
        try:
            self.guard.verify(os.path.dirname(key_file), True)
            self.guard.verify(key_file, False)
        except KeyExposedError as error:
            raise KeyExposedError(
                f"the key at {key_file} or its directory is readable by others: it may have been "
                f"compromised, so treat the memory database and the vault secrets as compromised "
                f"and rotate the key with {_ROTATE_COMMAND}"
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
            KeyMissingError: The file cannot be read or decoded, or is empty.
        """
        try:
            with open(key_file, encoding=KEY_ENCODING) as handle:
                value = handle.read().strip()
        except (OSError, UnicodeDecodeError) as error:
            raise KeyMissingError(f"cannot read the magi-rs memory key at {key_file}") from error
        if not value:
            raise KeyMissingError(
                f"the magi-rs memory key at {key_file} is empty: run {_REGENERATE_COMMAND}"
            )
        return MagiKey(value)
