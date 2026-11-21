# Author: Julian Bolivar
# Version: 7.1.0
# Date: 2026-11-19
"""Client for ``magi-rs vault set|ls``.

The vault is the only source of API keys (REQ-13c): a value reaches the
binary on its standard input, and the passphrase that opens the vault reaches
it only through the child environment. Neither ever appears in argv, in a
message or in an exception.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable

from magi_key import REDACTION_MARKER, MagiKey
from magi_rs_env import EnvironmentPolicy
from one_line import one_line
from validate import ValidationError

VAULT_KEY_NAMES: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "MAGI_API_KEY",
    "CODE_REVIEW_API_KEY",
    "EMBEDDING_API_KEY",
    "DECISION_API_KEY",
)
"""The closed list of secrets ``magi-rs`` reads from the vault (REQ-13a)."""

_VAULT_SUBCOMMAND = "vault"
_SET_ACTION = "set"
_LS_ACTION = "ls"
_WORKDIR_FLAG = "-w"
_VAULT_TIMEOUT_SECONDS = 60
"""A vault operation touches one small database; a minute only bounds a hung child."""

_LS_SEPARATOR = " · "
"""Separator of the ``vault ls`` columns (U+00B7), as captured from magi-rs 0.25.0."""
_LS_MIN_COLUMNS = 3
"""``NAME``, created, updated."""
_LS_NAME_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_LS_EMPTY_VAULT = "(vault empty)"
"""What magi-rs 0.25.0 prints, with exit 0, when the vault holds nothing (measurement M1)."""


class InvalidVaultNameError(ValidationError):
    """The vault name is not in :data:`VAULT_KEY_NAMES`."""


class VaultCommandError(ValidationError):
    """A ``magi-rs vault`` command failed, or answered in an unknown shape."""


class VaultClient:
    """Runs ``magi-rs vault set`` and ``vault ls`` for one project.

    Example:
        >>> client = VaultClient("/bin/magi-rs", "/repo", key)  # doctest: +SKIP
        >>> client.set("OPENAI_API_KEY", "sk-...")  # doctest: +SKIP
        >>> client.ls()  # doctest: +SKIP
        ['OPENAI_API_KEY']
    """

    def __init__(
        self,
        executable: str,
        workdir: str,
        key: MagiKey,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        """Bind the client to a verified binary, a project and its key.

        Args:
            executable: Path of the verified ``magi-rs`` executable.
            workdir: Project root, passed as ``-w``.
            key: The passphrase that opens the vault.
            runner: Process launcher; replaced only by tests.
        """
        self._executable = executable
        self._workdir = workdir
        self._key = key
        self._runner = runner

    def set(self, name: str, value: str) -> None:
        """Store one secret in the vault.

        Args:
            name: One of :data:`VAULT_KEY_NAMES`.
            value: The secret; sent on stdin, never in argv.

        Raises:
            InvalidVaultNameError: ``name`` is outside the closed list; the
                binary is not launched.
            VaultCommandError: The command failed. The message names the secret
                and never carries the value or the key.

        Side effects:
            Launches ``magi-rs vault set`` with the key in its environment.
        """
        if name not in VAULT_KEY_NAMES:
            raise InvalidVaultNameError(
                f"{name!r} is not a vault name; expected one of {', '.join(VAULT_KEY_NAMES)}"
            )
        completed = self._run([_SET_ACTION, name], f"vault set {name}", value, secrets=(value,))
        if completed.returncode != 0:
            raise VaultCommandError(self._failure(f"vault set {name}", completed, (value,)))

    def ls(self) -> list[str]:
        """List the names held by the vault (never the values).

        Returns:
            The names, in the order ``magi-rs`` printed them. An empty vault
            answers ``[]``; a failure never does (an empty answer would read
            as "nothing to revoke" during a rotation).

        Raises:
            VaultCommandError: The command failed, or a line does not have the
                shape ``NAME · created · updated``.

        Side effects:
            Launches ``magi-rs vault ls`` with the key in its environment.
        """
        completed = self._run([_LS_ACTION], "vault ls", None, secrets=())
        if completed.returncode != 0:
            raise VaultCommandError(self._failure("vault ls", completed, ()))
        if completed.stdout.strip() == _LS_EMPTY_VAULT:
            return []
        return [self._parse_ls_line(line) for line in completed.stdout.splitlines() if line.strip()]

    def _run(
        self, action: list[str], label: str, stdin: str | None, secrets: tuple[str, ...]
    ) -> subprocess.CompletedProcess[str]:
        """Launch ``magi-rs vault <action> -w <workdir>`` with the key in the environment.

        Args:
            action: The vault subcommand and its arguments.
            label: Human label of the command, for error messages.
            stdin: Text for the child's stdin, or ``None``.
            secrets: Values to scrub from a launch failure message.

        Returns:
            The completed process.

        Raises:
            VaultCommandError: The launch failed or timed out.
        """
        argv = [self._executable, _VAULT_SUBCOMMAND, *action, _WORKDIR_FLAG, self._workdir]
        try:
            return self._runner(
                argv,
                input=stdin,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=EnvironmentPolicy.build(os.environ, passphrase=self._key),
                timeout=_VAULT_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise VaultCommandError(
                self._redact(f"{label} could not run: {error}", secrets)
            ) from None

    def _failure(
        self, label: str, completed: subprocess.CompletedProcess[str], secrets: tuple[str, ...]
    ) -> str:
        """Build a failure message from a child's output with every secret scrubbed.

        Args:
            label: Human label of the command.
            completed: The failed process.
            secrets: Values to scrub besides the key.

        Returns:
            A one-line message.
        """
        # Redact BEFORE collapsing whitespace: collapsing can alter a secret that contains
        # runs of whitespace so that it no longer matches; the second pass covers the rest.
        detail = one_line(self._redact(completed.stderr or "", secrets))
        return self._redact(f"{label} failed (exit {completed.returncode}): {detail}", secrets)

    def _redact(self, text: str, secrets: tuple[str, ...]) -> str:
        """Replace the key and any given secret in ``text``.

        Scope: only exact occurrences are replaced; a secret whose whitespace was
        collapsed or that was split or re-encoded is not recognised, so this runs
        before any such transformation.

        Args:
            text: Text that may echo a secret.
            secrets: Values to scrub besides the key.

        Returns:
            ``text`` with the key and each non-empty secret replaced by
            ``REDACTION_MARKER``.
        """
        text = self._key.redact(text)
        for secret in secrets:
            if secret:
                text = text.replace(secret, REDACTION_MARKER)
        return text

    @staticmethod
    def _parse_ls_line(line: str) -> str:
        """Extract the name from one ``vault ls`` line.

        Args:
            line: ``NAME · created · updated`` (the name may be padded).

        Returns:
            The name.

        Raises:
            VaultCommandError: The line does not have that shape.
        """
        columns = line.split(_LS_SEPARATOR)
        name = columns[0].strip()
        if len(columns) < _LS_MIN_COLUMNS or not _LS_NAME_PATTERN.fullmatch(name):
            raise VaultCommandError("vault ls answered in an unrecognised format")
        return name
