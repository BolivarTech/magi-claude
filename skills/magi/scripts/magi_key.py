# Author: Julian Bolivar
# Version: 7.1.0
# Date: 2026-11-19
"""Custody of the passphrase that opens the ``magi-rs`` memory database."""

from __future__ import annotations

from typing import Protocol

KEY_DIR_PARTS = (".magi", ".secret")
KEY_FILENAME = "magi.key"
KEY_BYTES = 32
DIR_MODE = 0o700
FILE_MODE = 0o600


class KeyStoreError(Exception):
    """Base class of every key-custody failure."""


class KeyMissingError(KeyStoreError):
    """The key is absent, empty or unreadable."""


class KeyExposedError(KeyStoreError):
    """The key or its directory is readable by someone other than its owner."""


class KeyHardeningError(KeyStoreError):
    """The key's permissions could not be restricted or verified."""


class KeyGuard(Protocol):
    """Restricts and verifies owner-only permissions on a path."""

    def harden(self, path: str, directory: bool) -> None:
        """Restrict ``path`` to its owner."""

    def verify(self, path: str, directory: bool) -> None:
        """Check that ``path`` is restricted to its owner."""


class MagiKey:
    """The passphrase, redacted in every textual representation."""

    def __init__(self, value: str) -> None:
        raise NotImplementedError

    def reveal(self) -> str:
        """Return the passphrase."""
        raise NotImplementedError


class KeyStore:
    """Generates, persists and loads the key."""

    def __init__(self, guard: KeyGuard) -> None:
        raise NotImplementedError

    def key_path(self, workdir: str) -> str:
        """Return the key file path under ``workdir``."""
        raise NotImplementedError

    def new_key(self) -> MagiKey:
        """Return a new random key."""
        raise NotImplementedError

    def persist(self, workdir: str, key: MagiKey) -> str:
        """Write ``key`` under ``workdir``."""
        raise NotImplementedError

    def load(self, workdir: str) -> MagiKey:
        """Verify and read the key under ``workdir``."""
        raise NotImplementedError

    def read_unverified(self, key_file: str) -> MagiKey:
        """Read ``key_file`` without verifying its permissions."""
        raise NotImplementedError
