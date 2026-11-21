# Author: Julian Bolivar
# Version: 7.1.0
# Date: 2026-11-19
"""Client for ``magi-rs vault set|ls`` (stub)."""

from __future__ import annotations

import subprocess
from typing import Callable

from magi_key import MagiKey
from validate import ValidationError

VAULT_KEY_NAMES: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "MAGI_API_KEY",
    "CODE_REVIEW_API_KEY",
    "EMBEDDING_API_KEY",
    "DECISION_API_KEY",
)


class InvalidVaultNameError(ValidationError):
    """A vault name outside the closed list."""


class VaultCommandError(ValidationError):
    """A ``magi-rs vault`` command failed."""


class VaultClient:
    """Wraps ``magi-rs vault``."""

    def __init__(
        self,
        executable: str,
        workdir: str,
        key: MagiKey,
        runner: Callable[..., object] = subprocess.run,
    ) -> None:
        raise NotImplementedError

    def set(self, name: str, value: str) -> None:
        raise NotImplementedError

    def ls(self) -> list[str]:
        raise NotImplementedError
