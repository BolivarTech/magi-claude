# Author: Julian Bolivar
# Version: 7.1.0
# Date: 2026-11-19
"""Owner-only permission guards for the magi-rs memory key.

``PermissionGuard.for_platform()`` picks the guard for the running platform and
``default_key_store()`` builds the production ``KeyStore`` around it. Both
guards harden a path to its owner and verify that nobody else can reach it;
``verify`` never repairs anything, it only reports.

* POSIX: the mode bits. Hardening applies ``DIR_MODE``/``FILE_MODE``;
  verification requires the effective user as owner and no group or other
  bits (a stricter mode such as ``0400`` is accepted).
* Windows: the DACL, read as SDDL and compared by SID, never by an account
  name (names are localised; SIDs are not). Hardening cuts inheritance and
  grants full control to the current user's SID only; verification requires a
  protected DACL whose every ACE belongs to that SID.

Any failure to read what verification needs (the ``stat``, the current SID,
the DACL) is reported as ``KeyExposedError``: permissions that cannot be read
are never treated as safe.

Declared limits: the permissions of ``.magi/`` itself are not verified, POSIX
extended ACLs are not read, and on Windows the file owner (the ``O:`` field of
the SDDL) is not checked, only the DACL.
"""

from __future__ import annotations

import csv
import io
import ntpath
import os
import re
import stat
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

from magi_key import (
    DIR_MODE,
    FILE_MODE,
    KeyExposedError,
    KeyHardeningError,
    KeyStore,
    KeyUnverifiableError,
)
from magi_rs_env import EnvironmentPolicy

__all__ = [
    "DIR_MODE",
    "FILE_MODE",
    "PermissionGuard",
    "PosixPermissionGuard",
    "WindowsPermissionGuard",
    "default_key_store",
    "parse_sddl_dacl",
]

_GROUP_AND_OTHER_BITS = stat.S_IRWXG | stat.S_IRWXO
"""POSIX permission bits that grant access to anyone but the owner."""

_COMMAND_TIMEOUT_SECONDS = 30
"""Upper bound on each permission tool run (whoami, icacls, PowerShell)."""

_DEFAULT_SYSTEM_ROOT = r"C:\Windows"
"""Fallback when ``SystemRoot`` is absent from the environment."""

_SYSTEM32_PARTS = ("System32",)
_WHOAMI_PARTS = (*_SYSTEM32_PARTS, "whoami.exe")
_ICACLS_PARTS = (*_SYSTEM32_PARTS, "icacls.exe")
_POWERSHELL_PARTS = (*_SYSTEM32_PARTS, "WindowsPowerShell", "v1.0", "powershell.exe")

_WHOAMI_ARGS = ("/user", "/fo", "csv", "/nh")
"""One CSV line, ``"<host>\\<user>","<SID>"``, without a header."""

_ACL_PATH_VARIABLE = "MAGI_ACL_PATH"
"""Environment variable that carries the path to PowerShell; never interpolated into the command."""

_STDERR_EXCERPT_CHARS = 300
"""How much of a failed tool's stderr an error message carries, from its start (the headline)."""

_YES = "yes"
_NO = "no"

_SDDL_COMMAND = f"(Get-Acl -LiteralPath $env:{_ACL_PATH_VARIABLE}).Sddl"

_SID_ALIAS_VARIABLE = "MAGI_SID_ALIAS"
"""Environment variable that carries an SDDL alias to PowerShell; never interpolated into the command."""

_RESOLVE_ALIAS_COMMAND = (
    f"([System.Security.Principal.SecurityIdentifier]$env:{_SID_ALIAS_VARIABLE}).Value"
)

_POWERSHELL_OPTIONS = ("-NoProfile", "-NonInteractive", "-Command")

_REMOVE_INHERITANCE = "/inheritance:r"
_REPLACE_GRANT = "/grant:r"
_FILE_GRANT = "F"
"""Full control on the key file."""
_DIRECTORY_GRANT = "(OI)(CI)F"
"""Full control on the directory, inherited by the files and folders inside it."""
_REMOVE_GRANTS = "/remove:g"

_MKDIR_EXPLICIT_SIDS = ("S-1-3-4", "S-1-5-18", "S-1-5-32-544")
"""OWNER RIGHTS, SYSTEM and Administrators: the explicit ACEs ``os.mkdir(mode=0o700)`` writes on Windows.

``/inheritance:r`` drops only inherited ACEs, so these are removed by name;
any other explicit ACE survives hardening and is caught by ``verify``.
"""

_SID_PATTERN = re.compile(r"S-1-\d+(?:-\d+)+")
"""A literal SID string; whoami must return one."""

_SID_ALIAS_PATTERN = re.compile(r"[A-Z]{2}")
"""An SDDL SID alias (``LA``, ``SY``, ``BA``, ...). Windows writes the current user's own SID
this way when it is a well-known account: the built-in Administrator reads back as ``LA``."""

_DACL_PATTERN = re.compile(r"D:(?P<flags>[A-Z_]*)(?P<aces>(?:\([^()]*\))*)")
"""The DACL section: its flags, then a run of parenthesised ACEs."""

_ACE_PATTERN = re.compile(r"\(([^()]*)\)")
"""One parenthesised ACE; the group is its semicolon-separated body."""

_DACL_FLAGS_PATTERN = re.compile(r"(?:P|AI|AR)*")
"""The DACL control flags SDDL allows (protected, auto-inherited, auto-inherit required)."""

_PROTECTED_FLAG = "P"
_SACL_PREFIX = "S:"
_ACE_FIELD_COUNT = 6
"""type;flags;rights;object_guid;inherit_object_guid;sid."""
_ACE_SID_INDEX = 5


def _stderr_excerpt(error: subprocess.SubprocessError) -> str:
    """Return the start of a failed tool's stderr as one bounded line.

    The start, because PowerShell writes its headline first and the trace after.

    Args:
        error: The failure raised by ``subprocess.run``.

    Returns:
        At most ``_STDERR_EXCERPT_CHARS`` characters, whitespace collapsed; empty
        when the tool wrote nothing.
    """
    stderr = getattr(error, "stderr", None) or ""
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    return " ".join(stderr.split())[:_STDERR_EXCERPT_CHARS]


def _tool_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the environment of a permission tool: the magi-rs allowlist plus ``extra``.

    The same allowlist the ``magi-rs`` child gets (``EnvironmentPolicy``), so no
    API key and no parent ``MAGI_PASSPHRASE`` reach whoami, icacls or
    PowerShell. ``PSModulePath`` is not allowlisted: a PowerShell 7 module path
    inherited by Windows PowerShell 5.1 keeps ``Get-Acl`` from loading, and
    unset, 5.1 rebuilds its default.

    Args:
        extra: The variables that carry the tool's input, added on top.

    Returns:
        A new environment mapping.
    """
    env = EnvironmentPolicy.build(os.environ)
    env.update(extra or {})
    return env


def _run(argv: list[str], env: Mapping[str, str] | None = None) -> str:
    """Run a permission tool and return its standard output.

    Args:
        argv: The command, as an argument list (never through a shell).
        env: The variables that carry the tool's input; they are added to
            the allowlisted environment (``_tool_environment``), never to the
            whole parent environment.

    Returns:
        The decoded standard output.

    Raises:
        OSError: The tool could not start, timed out or exited non-zero; the
            message names the tool, the exit code or timeout, and the start of
            its stderr.
    """
    tool = ntpath.basename(argv[0])
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_COMMAND_TIMEOUT_SECONDS,
            check=True,
            env=_tool_environment(env),
        )
    except subprocess.CalledProcessError as error:
        raise OSError(
            f"{tool} failed with exit code {error.returncode}: {_stderr_excerpt(error)}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise OSError(
            f"{tool} timed out after {_COMMAND_TIMEOUT_SECONDS} s: {_stderr_excerpt(error)}"
        ) from error
    except subprocess.SubprocessError as error:
        raise OSError(f"{tool} failed: {_stderr_excerpt(error)}") from error
    return completed.stdout


def _run_powershell(command: str, variable: str, value: str) -> str:
    """Run one Windows PowerShell command that reads its input from one variable.

    The value travels only in the child environment, never in the command
    text; that environment is the allowlist plus ``variable``
    (``_tool_environment``).

    Args:
        command: The PowerShell command text.
        variable: The environment variable that carries ``value``.
        value: The input, e.g. a path or an SDDL alias.

    Returns:
        The output, stripped.

    Raises:
        OSError: PowerShell could not run or exited non-zero.
    """
    return _run(
        [_system_tool(_POWERSHELL_PARTS), *_POWERSHELL_OPTIONS, command], {variable: value}
    ).strip()


def _system_tool(parts: tuple[str, ...]) -> str:
    """Return the absolute path of a Windows system tool.

    Absolute on purpose: on ``PATH``, git-bash resolves ``whoami`` to the
    coreutils one.

    Args:
        parts: Path components under ``%SystemRoot%``.

    Returns:
        ``%SystemRoot%\\<parts>``.
    """
    return ntpath.join(os.environ.get("SystemRoot", _DEFAULT_SYSTEM_ROOT), *parts)


def parse_sddl_dacl(sddl: str) -> tuple[bool, list[str]]:
    """Return whether the DACL is protected and the SID of every ACE.

    Fail-closed: every ACE counts whatever its type (``A``, ``D``, ...) or its
    flags (``OICI``, ``ID``); a SID alias (``AU``, ``WD``, ``BA``, ...) is
    returned as written. Runs in O(n) over the SDDL length.

    Args:
        sddl: A security descriptor in SDDL form.

    Returns:
        ``(protected, sids)``: ``protected`` is True iff the DACL flags include
        ``P``; ``sids`` lists each ACE's SID field in order.

    Raises:
        ValueError: The DACL is absent, is ``NO_ACCESS_CONTROL`` (a NULL DACL
            grants everyone access), has unknown flags, or holds an ACE without
            its six fields.

    Example:
        >>> parse_sddl_dacl("D:PAI(A;;FA;;;S-1-5-18)")
        (True, ['S-1-5-18'])
    """
    descriptor = sddl.strip()
    # Assumes the section order O:, G:, D:, then S: as Get-Acl emits it.
    match = _DACL_PATTERN.search(descriptor)
    if match is None:
        raise ValueError("the security descriptor has no DACL")
    rest = descriptor[match.end() :]
    if rest and not rest.startswith(_SACL_PREFIX):
        raise ValueError("the DACL holds a malformed ACE")
    flags = match.group("flags")
    if not _DACL_FLAGS_PATTERN.fullmatch(flags):
        raise ValueError(f"the DACL has unknown flags {flags!r}")
    sids = []
    for ace in _ACE_PATTERN.findall(match.group("aces")):
        fields = ace.split(";")
        if len(fields) != _ACE_FIELD_COUNT or not fields[_ACE_SID_INDEX]:
            raise ValueError("the DACL holds a malformed ACE")
        sids.append(fields[_ACE_SID_INDEX])
    return _PROTECTED_FLAG in flags, sids


class PermissionGuard(ABC):
    """Restricts and verifies owner-only permissions on the key and its directory.

    Example:
        >>> guard = PermissionGuard.for_platform()
        >>> guard.harden(path, directory=False)
        >>> guard.verify(path, directory=False)
    """

    @abstractmethod
    def harden(self, path: str, directory: bool) -> None:
        """Restrict ``path`` to its owner.

        Args:
            path: The file or directory to restrict.
            directory: True when ``path`` is a directory.

        Raises:
            KeyHardeningError: The restriction could not be applied.
        """

    @abstractmethod
    def verify(self, path: str, directory: bool) -> None:
        """Check that ``path`` is restricted to its owner, without repairing it.

        Args:
            path: The file or directory to check.
            directory: True when ``path`` is a directory.

        Raises:
            KeyUnverifiableError: The permissions could not be read.
            KeyExposedError: Someone else can access ``path``.
        """

    @staticmethod
    def for_platform() -> PermissionGuard:
        """Return the guard for the running platform.

        Returns:
            A ``WindowsPermissionGuard`` on Windows, else a ``PosixPermissionGuard``.
        """
        if sys.platform == "win32":
            return WindowsPermissionGuard()
        return PosixPermissionGuard()


class PosixPermissionGuard(PermissionGuard):
    """Owner-only permissions through POSIX mode bits."""

    def harden(self, path: str, directory: bool) -> None:
        """Apply ``DIR_MODE`` or ``FILE_MODE`` to ``path``.

        Args:
            path: The file or directory to restrict.
            directory: True when ``path`` is a directory.

        Raises:
            KeyHardeningError: ``chmod`` failed.
        """
        try:
            os.chmod(path, DIR_MODE if directory else FILE_MODE)
        except OSError as error:
            raise KeyHardeningError(f"cannot restrict {path} to its owner") from error

    def verify(self, path: str, directory: bool) -> None:
        """Require the effective user as owner and no group or other bits.

        Args:
            path: The file or directory to check.
            directory: True when ``path`` is a directory (same rule for both).

        Raises:
            KeyUnverifiableError: ``path`` cannot be inspected, or this is Windows.
            KeyExposedError: Another user owns ``path``, or a group or other bit
                is set.
        """
        if sys.platform == "win32":
            raise KeyUnverifiableError("POSIX permissions cannot be verified on Windows")
        try:
            status = os.stat(path)
        except OSError as error:
            raise KeyUnverifiableError(f"cannot read the permissions of {path}: {error}") from error
        if status.st_uid != os.geteuid():
            raise KeyExposedError(f"{path} is owned by another user")
        if status.st_mode & _GROUP_AND_OTHER_BITS:
            raise KeyExposedError(f"{path} is accessible to its group or to others")


class WindowsPermissionGuard(PermissionGuard):
    """Owner-only permissions through the Windows DACL, compared by SID.

    Example:
        >>> guard = WindowsPermissionGuard()
        >>> guard.harden(r"C:\\work\\.magi\\.secret", directory=True)
        >>> guard.verify(r"C:\\work\\.magi\\.secret", directory=True)
    """

    def __init__(self, runner: Callable[[list[str]], str] = _run) -> None:
        """Bind the guard to the runner of ``whoami`` and ``icacls``.

        Args:
            runner: Runs an argument list and returns its standard output,
                raising ``OSError`` on failure.
        """
        self._runner = runner

    def current_sid(self) -> str:
        """Return the SID of the user running this process.

        Returns:
            The SID string from ``whoami.exe /user /fo csv /nh``.

        Raises:
            OSError: ``whoami.exe`` could not run.
            ValueError: Its output is not one ``"<name>","<SID>"`` line.
        """
        output = self._runner([_system_tool(_WHOAMI_PARTS), *_WHOAMI_ARGS])
        rows = list(csv.reader(io.StringIO(output.strip())))
        if len(rows) != 1 or len(rows[0]) != 2 or not _SID_PATTERN.fullmatch(rows[0][1]):
            raise ValueError("whoami did not report the current user's SID")
        return rows[0][1]

    def read_sddl(self, path: str) -> str:
        """Return the SDDL of ``path``, read by PowerShell's ``Get-Acl``.

        The path reaches PowerShell only through ``MAGI_ACL_PATH``, never in
        the command text, so no character in it can be interpreted. The
        child gets the allowlisted environment only (``_tool_environment``).

        Args:
            path: The file or directory to read.

        Returns:
            The SDDL text; empty when the ACL could not be read.

        Raises:
            OSError: PowerShell could not run.
        """
        return _run_powershell(_SDDL_COMMAND, _ACL_PATH_VARIABLE, path)

    def resolve_alias(self, alias: str) -> str:
        """Return the numeric SID behind an SDDL alias.

        The alias reaches PowerShell only through ``MAGI_SID_ALIAS``, with the
        same clean environment as ``read_sddl``.

        Args:
            alias: An SDDL SID alias such as ``LA`` or ``SY``.

        Returns:
            The SID string, e.g. ``S-1-5-18``.

        Raises:
            OSError: PowerShell failed (an invalid alias exits 1) or answered
                with something that is not a SID.
        """
        sid = _run_powershell(_RESOLVE_ALIAS_COMMAND, _SID_ALIAS_VARIABLE, alias)
        if not _SID_PATTERN.fullmatch(sid):
            raise OSError(f"the SDDL alias {alias} did not resolve to a SID")
        return sid

    def _is_foreign(self, ace_sid: str, sid: str, resolved: dict[str, str]) -> bool:
        """Tell whether an ACE's SID is someone other than the current user.

        A two-letter alias is resolved to its numeric SID first, once per alias
        (memoised in ``resolved``); a numeric SID is compared as is, so the
        normal path starts no extra process.

        Args:
            ace_sid: The SID field of the ACE, numeric or an alias.
            sid: The current user's SID.
            resolved: The aliases resolved so far during this check.

        Returns:
            True when the ACE belongs to someone else.

        Raises:
            OSError: An alias could not be resolved.
        """
        if ace_sid == sid:
            return False
        if not _SID_ALIAS_PATTERN.fullmatch(ace_sid):
            return True
        if ace_sid not in resolved:
            resolved[ace_sid] = self.resolve_alias(ace_sid)
        return resolved[ace_sid] != sid

    def harden(self, path: str, directory: bool) -> None:
        """Cut inheritance and grant full control to the current user's SID only.

        The same ``icacls`` call also removes the explicit ACEs
        ``os.mkdir(mode=0o700)`` leaves on Windows (``_MKDIR_EXPLICIT_SIDS``),
        so the DACL only ever narrows.

        Args:
            path: The file or directory to restrict.
            directory: True when ``path`` is a directory; its grant is then
                inherited by what it contains.

        Raises:
            KeyHardeningError: The SID could not be read or ``icacls`` failed.
        """
        try:
            grant = _DIRECTORY_GRANT if directory else _FILE_GRANT
            sid = self.current_sid()
            removals = [f"*{other}" for other in _MKDIR_EXPLICIT_SIDS if other != sid]
            self._runner(
                [
                    _system_tool(_ICACLS_PARTS),
                    path,
                    _REMOVE_INHERITANCE,
                    _REPLACE_GRANT,
                    f"*{sid}:{grant}",
                    _REMOVE_GRANTS,
                    *removals,
                ]
            )
        except (OSError, ValueError) as error:
            raise KeyHardeningError(f"cannot restrict {path} to its owner") from error

    def verify(self, path: str, directory: bool) -> None:
        """Require a protected DACL whose every ACE is the current user's SID.

        Args:
            path: The file or directory to check.
            directory: True when ``path`` is a directory (same rule for both).

        Raises:
            KeyUnverifiableError: The DACL, the current SID or an alias could
                not be read or parsed; the message names the cause.
            KeyExposedError: The DACL inherits or grants another SID. The
                message is one line naming the foreign SIDs (never the current
                user's) and whether the DACL is protected.
        """
        try:
            sid = self.current_sid()
            protected, sids = parse_sddl_dacl(self.read_sddl(path))
            resolved: dict[str, str] = {}
            foreign = [
                ace_sid
                for ace_sid in dict.fromkeys(sids)
                if self._is_foreign(ace_sid, sid, resolved)
            ]
        except (OSError, ValueError) as error:
            raise KeyUnverifiableError(f"cannot read the permissions of {path}: {error}") from error
        protection = f"(DACL protected: {_YES if protected else _NO})"
        if foreign:
            raise KeyExposedError(f"{path} grants access to {', '.join(foreign)} {protection}")
        if not protected:
            raise KeyExposedError(f"{path} still inherits permissions from its parent {protection}")


def default_key_store() -> KeyStore:
    """Return the production key store, guarded for the running platform.

    The only production constructor of the store.

    Returns:
        ``KeyStore(PermissionGuard.for_platform())``.
    """
    return KeyStore(PermissionGuard.for_platform())
