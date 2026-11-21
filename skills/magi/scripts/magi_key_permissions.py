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
  grants full control to the current user's SID only (``icacls``);
  verification requires a protected DACL whose every ACE belongs to that SID.
  The SDDL and SDDL aliases are read natively through ``advapi32`` (ctypes),
  without starting a process: under the allowlisted environment every Windows
  PowerShell start on the CI runner took tens of seconds.

Any failure to read what verification needs (the ``stat``, the current SID,
the DACL, an SDDL alias) is reported as ``KeyUnverifiableError``, a subclass of
``KeyExposedError``: permissions that cannot be read are never treated as safe,
and the message says they could not be verified, never that they are exposed.

Declared limits: the permissions of ``.magi/`` itself are not verified, POSIX
extended ACLs are not read, and on Windows the file owner (the ``O:`` field of
the SDDL) is not checked, only the DACL.
"""

from __future__ import annotations

import csv
import ctypes
import io
import ntpath
import os
import re
import stat
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable

from magi_key import (
    DIR_MODE,
    FILE_MODE,
    KeyExposedError,
    KeyHardeningError,
    KeyStore,
    KeyUnverifiableError,
)
from magi_rs_env import EnvironmentPolicy
from one_line import one_line

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
"""Upper bound on each permission tool run (whoami, icacls)."""

_DEFAULT_SYSTEM_ROOT = r"C:\Windows"
"""Fallback when ``SystemRoot`` is absent from the environment."""

_SYSTEM32_PARTS = ("System32",)
_WHOAMI_PARTS = (*_SYSTEM32_PARTS, "whoami.exe")
_ICACLS_PARTS = (*_SYSTEM32_PARTS, "icacls.exe")

_WHOAMI_ARGS = ("/user", "/fo", "csv", "/nh")
"""One CSV line, ``"<host>\\<user>","<SID>"``, without a header."""

_STDERR_EXCERPT_CHARS = 300
"""How much of a failed tool's stderr an error message carries, from its start (the headline)."""

_YES = "yes"
_NO = "no"

_SE_FILE_OBJECT = 1
"""``SE_OBJECT_TYPE`` of a file or directory for ``GetNamedSecurityInfoW``."""

_OWNER_SECURITY_INFORMATION = 0x1
_GROUP_SECURITY_INFORMATION = 0x2
_DACL_SECURITY_INFORMATION = 0x4
_SDDL_SECTIONS = (
    _OWNER_SECURITY_INFORMATION | _GROUP_SECURITY_INFORMATION | _DACL_SECURITY_INFORMATION
)
"""Owner, group and DACL: the sections ``Get-Acl``'s ``.Sddl`` renders (the M9 fixtures' shape)."""

_SDDL_REVISION_1 = 1
_ERROR_SUCCESS = 0

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

    The start, because a tool writes its headline first and any trace after.

    Args:
        error: The failure raised by ``subprocess.run``.

    Returns:
        At most ``_STDERR_EXCERPT_CHARS`` characters, whitespace collapsed; empty
        when the tool wrote nothing.
    """
    stderr = getattr(error, "stderr", None) or ""
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    return one_line(stderr)[:_STDERR_EXCERPT_CHARS]


def _run(argv: list[str]) -> str:
    """Run a permission tool, with the allowlisted environment, and return its standard output.

    The child gets exactly ``EnvironmentPolicy.build(os.environ)``, the
    allowlist the ``magi-rs`` child gets: no API key, no parent
    ``MAGI_PASSPHRASE``, no ``PSModulePath``.

    Args:
        argv: The command, as an argument list (never through a shell).

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
            env=EnvironmentPolicy.build(os.environ),
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


class _NativeSecurity:
    """The ``advapi32``/``kernel32`` calls the Windows guard reads with, bound through ctypes.

    Built lazily by the guard on its first read, so importing this module on
    POSIX never loads a Windows DLL. Every prototype is declared explicitly;
    every buffer the API allocates is released with ``LocalFree``.

    Example:
        >>> native = _NativeSecurity()
        >>> native.resolve_alias("BA")
        'S-1-5-32-544'
    """

    _get_named_security_info: Callable[..., int]
    _descriptor_to_string: Callable[..., int]
    _string_to_sid: Callable[..., int]
    _sid_to_string: Callable[..., int]
    _local_free: Callable[..., object]

    def __init__(self) -> None:
        """Load the two DLLs and declare the prototypes.

        Raises:
            OSError: Not Windows, or a DLL could not be loaded.
        """
        if sys.platform != "win32":
            raise OSError("native ACL reads exist only on Windows")
        from ctypes import wintypes

        pointer = ctypes.POINTER(ctypes.c_void_p)
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._get_named_security_info = advapi32.GetNamedSecurityInfoW
        self._get_named_security_info.argtypes = [
            wintypes.LPCWSTR,
            ctypes.c_int,
            wintypes.DWORD,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
        ]
        self._get_named_security_info.restype = wintypes.DWORD
        self._descriptor_to_string = advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW
        self._descriptor_to_string.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            pointer,
            ctypes.POINTER(wintypes.ULONG),
        ]
        self._descriptor_to_string.restype = wintypes.BOOL
        self._string_to_sid = advapi32.ConvertStringSidToSidW
        self._string_to_sid.argtypes = [wintypes.LPCWSTR, pointer]
        self._string_to_sid.restype = wintypes.BOOL
        self._sid_to_string = advapi32.ConvertSidToStringSidW
        self._sid_to_string.argtypes = [ctypes.c_void_p, pointer]
        self._sid_to_string.restype = wintypes.BOOL
        self._local_free = kernel32.LocalFree
        self._local_free.argtypes = [ctypes.c_void_p]
        self._local_free.restype = ctypes.c_void_p

    @staticmethod
    def _failure(call: str, subject: str, code: int) -> OSError:
        """Build the one-line error of a failed call, naming the Win32 error.

        Args:
            call: The API function that failed.
            subject: The path or alias it was called for.
            code: The Win32 error code.

        Returns:
            ``OSError("<call> failed for <subject>: Win32 error <code>: <text>")``.
        """
        if sys.platform != "win32":
            return OSError(f"{call} failed for {subject}: Win32 error {code}")
        return OSError(
            f"{call} failed for {subject}: Win32 error {code}: {one_line(ctypes.FormatError(code))}"
        )

    @staticmethod
    def _null_result(call: str, subject: str) -> OSError:
        """Build the one-line error of a call that reported success but returned a NULL pointer.

        Args:
            call: The API function.
            subject: The path or alias it was called for.

        Returns:
            ``OSError("<call> failed for <subject>: it reported success but returned NULL")``.
        """
        return OSError(f"{call} failed for {subject}: it reported success but returned NULL")

    @staticmethod
    def _last_error() -> int:
        """Return the calling thread's last Win32 error, as ctypes saved it.

        Returns:
            The error code.
        """
        if sys.platform != "win32":
            return 0
        return ctypes.get_last_error()

    def read_sddl(self, path: str) -> str:
        """Return the owner, group and DACL of ``path`` as SDDL.

        Args:
            path: The file or directory to read.

        Returns:
            The SDDL text.

        Raises:
            OSError: Either call failed, or reported success with a NULL
                result (never dereferenced); the message names the call.
        """
        descriptor = ctypes.c_void_p()
        code = self._get_named_security_info(
            path, _SE_FILE_OBJECT, _SDDL_SECTIONS, None, None, None, None, ctypes.byref(descriptor)
        )
        if code != _ERROR_SUCCESS:
            raise self._failure("GetNamedSecurityInfoW", path, code)
        if not descriptor.value:
            raise self._null_result("GetNamedSecurityInfoW", path)
        try:
            text = ctypes.c_void_p()
            call = "ConvertSecurityDescriptorToStringSecurityDescriptorW"
            if not self._descriptor_to_string(
                descriptor, _SDDL_REVISION_1, _SDDL_SECTIONS, ctypes.byref(text), None
            ):
                raise self._failure(call, path, self._last_error())
            if not text.value:
                raise self._null_result(call, path)
            try:
                return ctypes.wstring_at(text)
            finally:
                self._local_free(text)
        finally:
            self._local_free(descriptor)

    def resolve_alias(self, alias: str) -> str:
        """Return the numeric SID behind an SDDL alias (or a SID string).

        Args:
            alias: An SDDL SID alias such as ``LA`` or ``SY``.

        Returns:
            The SID string, e.g. ``S-1-5-18``.

        Raises:
            OSError: The alias is unknown, a call failed, or a call reported
                success with a NULL result (never dereferenced); the message
                names the call.
        """
        sid = ctypes.c_void_p()
        if not self._string_to_sid(alias, ctypes.byref(sid)):
            raise self._failure("ConvertStringSidToSidW", alias, self._last_error())
        if not sid.value:
            raise self._null_result("ConvertStringSidToSidW", alias)
        try:
            text = ctypes.c_void_p()
            if not self._sid_to_string(sid, ctypes.byref(text)):
                raise self._failure("ConvertSidToStringSidW", alias, self._last_error())
            if not text.value:
                raise self._null_result("ConvertSidToStringSidW", alias)
            try:
                return ctypes.wstring_at(text)
            finally:
                self._local_free(text)
        finally:
            self._local_free(sid)


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
        self._sid: str | None = None
        self._native: _NativeSecurity | None = None

    def _security(self) -> _NativeSecurity:
        """Return the native API binding, built on first use.

        Returns:
            The binding.

        Raises:
            OSError: The DLLs could not be loaded (or this is not Windows).
        """
        if self._native is None:
            self._native = _NativeSecurity()
        return self._native

    def current_sid(self) -> str:
        """Return the SID of the user running this process, asking ``whoami`` once per guard.

        The SID of a process does not change, so the first successful answer is
        kept; a failure is not, and the next call asks again.

        Returns:
            The SID string from ``whoami.exe /user /fo csv /nh``.

        Raises:
            OSError: ``whoami.exe`` could not run.
            ValueError: Its output is not one ``"<name>","<SID>"`` line.
        """
        if self._sid is None:
            output = self._runner([_system_tool(_WHOAMI_PARTS), *_WHOAMI_ARGS])
            rows = list(csv.reader(io.StringIO(output.strip())))
            if len(rows) != 1 or len(rows[0]) != 2 or not _SID_PATTERN.fullmatch(rows[0][1]):
                raise ValueError("whoami did not report the current user's SID")
            self._sid = rows[0][1]
        return self._sid

    def read_sddl(self, path: str) -> str:
        """Return the SDDL of ``path`` (owner, group, DACL), read natively; no process is started.

        Args:
            path: The file or directory to read.

        Returns:
            The SDDL text, the same rendering as ``Get-Acl``'s ``.Sddl``.

        Raises:
            OSError: The ACL could not be read; the message names the Win32 error.
        """
        return self._security().read_sddl(path)

    def resolve_alias(self, alias: str) -> str:
        """Return the numeric SID behind an SDDL alias.

        Resolved natively; no process is started.

        Args:
            alias: An SDDL SID alias such as ``LA`` or ``SY``.

        Returns:
            The SID string, e.g. ``S-1-5-18``.

        Raises:
            OSError: The alias is unknown (fail closed) or the answer is not a SID.
        """
        sid = self._security().resolve_alias(alias)
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
