# Author: Julian Bolivar
# Version: 7.1.0
# Date: 2026-11-19
"""Owner-only permission guards for the magi-rs memory key (signatures only)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from magi_key import DIR_MODE, FILE_MODE, KeyStore

__all__ = [
    "DIR_MODE",
    "FILE_MODE",
    "PermissionGuard",
    "PosixPermissionGuard",
    "WindowsPermissionGuard",
    "default_key_store",
    "parse_sddl_dacl",
]


def _run(argv: list[str]) -> str:
    """Run ``argv`` and return its standard output."""
    raise NotImplementedError


def parse_sddl_dacl(sddl: str) -> tuple[bool, list[str]]:
    """Return whether the DACL is protected and the SID of every ACE."""
    raise NotImplementedError


class PermissionGuard(ABC):
    """Restricts and verifies owner-only permissions."""

    @abstractmethod
    def harden(self, path: str, directory: bool) -> None:
        """Restrict ``path`` to its owner."""

    @abstractmethod
    def verify(self, path: str, directory: bool) -> None:
        """Check that ``path`` is restricted to its owner."""

    @staticmethod
    def for_platform() -> PermissionGuard:
        """Return the guard for the running platform."""
        raise NotImplementedError


class PosixPermissionGuard(PermissionGuard):
    """POSIX mode-bit guard."""

    def harden(self, path: str, directory: bool) -> None:
        """Restrict ``path`` to its owner."""
        raise NotImplementedError

    def verify(self, path: str, directory: bool) -> None:
        """Check that ``path`` is restricted to its owner."""
        raise NotImplementedError


class WindowsPermissionGuard(PermissionGuard):
    """Windows DACL guard, by SID."""

    def __init__(self, runner: Callable[[list[str]], str] = _run) -> None:
        """Bind the guard to a command runner."""
        raise NotImplementedError

    def current_sid(self) -> str:
        """Return the current user's SID."""
        raise NotImplementedError

    def read_sddl(self, path: str) -> str:
        """Return the SDDL of ``path``."""
        raise NotImplementedError

    def harden(self, path: str, directory: bool) -> None:
        """Restrict ``path`` to its owner."""
        raise NotImplementedError

    def verify(self, path: str, directory: bool) -> None:
        """Check that ``path`` is restricted to its owner."""
        raise NotImplementedError


def default_key_store() -> KeyStore:
    """Return the production key store."""
    raise NotImplementedError
