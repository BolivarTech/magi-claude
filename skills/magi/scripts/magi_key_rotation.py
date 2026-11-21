# Author: Julian Bolivar
# Version: 7.1.0
# Date: 2026-11-19
"""Full rotation of an exposed magi-rs memory key (REQ-13e-h)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from magi_key import KeyStore, MagiKey


class RotationNotAuthorisedError(Exception):
    """The rotation was not authorised; nothing changed."""


class RotationFailedError(Exception):
    """The rotation stopped; the message describes the state left behind in one line."""


@dataclass
class RotationReport:
    """What a completed rotation did."""

    backup_dir: str
    names_to_revoke: list[str]
    names_from_vault: bool


def rotation_backup_root() -> str:
    """Return the platform directory that holds rotation backups."""
    raise NotImplementedError


def _utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


class KeyRotation:
    """Moves an exposed ``.magi/`` aside and creates a new one."""

    def __init__(
        self,
        key_store: KeyStore,
        init: Callable[[str, KeyStore], str],
        vault_ls: Callable[[str, MagiKey], list[str]],
        backup_root: str | None = None,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Bind the rotation to its collaborators."""
        raise NotImplementedError

    def rotate(
        self,
        workdir: str,
        assume_yes: bool,
        is_tty: bool,
        confirm: Callable[[str], str] = input,
    ) -> RotationReport:
        """Rotate the key of ``workdir``."""
        raise NotImplementedError
