import ntpath
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from magi_key import (
    DIR_MODE,
    FILE_MODE,
    KeyExposedError,
    KeyHardeningError,
    KeyStore,
    KeyUnverifiableError,
)
from magi_key_permissions import (
    PermissionGuard,
    PosixPermissionGuard,
    WindowsPermissionGuard,
    default_key_store,
    _parse_dacl_aces,
)
from magi_key_permissions import _NativeSecurity
from magi_rs_env import EnvironmentPolicy

ME = "S-1-5-21-1111-2222-3333-1001"
OTHER = "S-1-5-11"  # Authenticated Users
_ACL = os.path.join(
    os.path.dirname(__file__), "fixtures", "windows-acl"
)  # [CP2-M45] the M9 capture


def _protection_and_sids(sddl):
    """``(protected, sids)`` through the one parser, for the SID-only assertions below [U3]."""
    protected, aces = _parse_dacl_aces(sddl)
    return protected, [sid for _, sid in aces]


def _fixture(name):
    return open(os.path.join(_ACL, name), encoding="utf-8").read()


def test_sddl_parser_reads_protection_and_every_ace_sid():
    protected, sids = _protection_and_sids(f"O:{ME}G:{ME}D:PAI(A;;FA;;;{ME})")
    assert protected is True
    assert sids == [ME]


def test_sddl_parser_sees_an_inherited_extra_ace():
    protected, sids = _protection_and_sids(f"D:AI(A;;FA;;;{ME})(A;ID;0x1301bf;;;{OTHER})")
    assert protected is False
    assert sids == [ME, OTHER]


def test_sddl_parser_returns_aliases_and_container_aces_as_sids():
    # [CP2-M3] fail-closed: an alias and an OICI-flagged ACE are ACEs too.
    protected, sids = _protection_and_sids(f"D:PAI(A;OICI;FA;;;{ME})(A;;FR;;;AU)")
    assert protected is True
    assert sids == [ME, "AU"]


def test_sddl_parser_reads_a_flagless_dacl_as_unprotected():
    # [CP2-M24]
    protected, sids = _protection_and_sids(f"D:(A;;FA;;;{ME})")
    assert protected is False
    assert sids == [ME]


@pytest.mark.parametrize("sddl", [f"O:{ME}G:{ME}", "D:NO_ACCESS_CONTROL"])
def test_sddl_parser_rejects_an_absent_or_null_dacl(sddl):
    # [CP2-C38] a NULL or missing DACL grants everyone full access: never "protected, no ACEs".
    with pytest.raises(ValueError):
        _protection_and_sids(sddl)


def test_sddl_parser_rejects_a_malformed_ace():
    with pytest.raises(ValueError):
        _protection_and_sids(f"D:PAI(A;;FA;;;{ME})(A;;FA)")


_ALIASES = {"SY": "S-1-5-18", "BA": "S-1-5-32-544", "AU": "S-1-5-11", "WD": "S-1-1-0"}
"""M10: what resolve_alias answers for the aliases the fixtures carry; never a process in a unit test."""


def _windows_guard(sddl, aliases=None):
    def runner(argv):
        if argv[0].lower().endswith("whoami.exe"):
            return _fixture("whoami-nh.txt")  # [CP2-M45] the real /nh shape
        if argv[0].lower().endswith("icacls.exe") or argv[0] == "icacls":
            return ""
        raise AssertionError(argv)

    guard = WindowsPermissionGuard(runner=runner)
    guard.read_sddl = lambda path: sddl
    guard.resolve_alias = {**_ALIASES, **(aliases or {})}.__getitem__
    return guard


def test_windows_verify_accepts_only_the_current_user_with_inheritance_cut():
    _windows_guard(f"D:PAI(A;;FA;;;{ME})").verify("k", directory=False)


def test_windows_verify_rejects_an_inherited_authenticated_users_ace():
    with pytest.raises(KeyExposedError):
        _windows_guard(f"D:PAI(A;;FA;;;{ME})(A;;0x1200a9;;;{OTHER})").verify("k", directory=False)


def test_windows_verify_rejects_a_dacl_that_still_inherits():
    with pytest.raises(KeyExposedError):
        _windows_guard(f"D:AI(A;;FA;;;{ME})").verify("k", directory=False)


def test_windows_current_sid_parses_the_captured_whoami_output():
    # [CP2-M45]
    assert _windows_guard("").current_sid() == ME


@pytest.mark.parametrize("name", ["sddl-file-hardened.txt", "sddl-dir-hardened.txt"])
def test_windows_verify_accepts_the_captured_hardened_sddl(name):
    # [CP2-M45] the real shapes icacls leaves behind for the key file and for .secret/.
    _windows_guard(_fixture(name).strip()).verify("k", directory=name.startswith("sddl-dir"))


def test_windows_verify_rejects_the_captured_inherited_sddl():
    # [CP2-M45] SYSTEM and Administrators inherited from the volume: exposed (PM-S-1).
    with pytest.raises(KeyExposedError):
        _windows_guard(_fixture("sddl-inherited.txt").strip()).verify("k", directory=False)


def test_windows_verify_treats_an_empty_sddl_as_exposed():
    # [CP2-M45] M9: Get-Acl on a missing path exits 0 with no output.
    with pytest.raises(KeyExposedError):
        _windows_guard("").verify("k", directory=False)


def test_windows_verify_rejects_an_alias_ace():
    with pytest.raises(KeyExposedError):
        _windows_guard(f"D:PAI(A;;FA;;;{ME})(A;;FR;;;WD)").verify("k", directory=False)


def test_windows_verify_treats_an_unparseable_dacl_as_exposed():
    with pytest.raises(KeyExposedError):
        _windows_guard("D:PAI(garbage").verify("k", directory=False)


def test_windows_verify_that_cannot_read_the_acl_or_the_sid_is_exposed():
    # [CP2-M16] unknown permissions are never safe.
    guard = _windows_guard("")

    def _fail(path):
        raise OSError("access denied")

    guard.read_sddl = _fail
    with pytest.raises(KeyExposedError):
        guard.verify("k", directory=False)

    def runner(argv):
        raise OSError("whoami missing")

    with pytest.raises(KeyExposedError):
        WindowsPermissionGuard(runner=runner).verify("k", directory=False)


def test_windows_harden_uses_the_sid_never_a_localised_name():
    seen = []

    def runner(argv):
        seen.append(argv)
        return _fixture("whoami-nh.txt") if argv[0].lower().endswith("whoami.exe") else ""

    guard = WindowsPermissionGuard(runner=runner)
    guard.read_sddl = lambda path: f"D:PAI(A;;FA;;;{ME})"  # [S2] harden reads the DACL first
    guard.harden("k", directory=False)
    # [CP2-M45] whoami by absolute path: on PATH, git-bash would answer with coreutils' whoami (M9).
    assert ntpath.isabs(seen[0][0]) and seen[0][0].lower().endswith("system32\\whoami.exe")
    icacls = next(a for a in seen if "icacls" in a[0].lower())
    assert "/inheritance:r" in icacls
    assert any(part.startswith(f"*{ME}:") for part in icacls)


def test_a_failing_permission_tool_reports_its_stderr_and_exit_code(monkeypatch):
    def _fake_run(argv, **kwargs):
        raise subprocess.CalledProcessError(
            1, argv, output="", stderr="Get-Acl : module could not be loaded\r\n"
        )

    monkeypatch.setattr("magi_key_permissions.subprocess.run", _fake_run)
    with pytest.raises(OSError) as raised:
        WindowsPermissionGuard().current_sid()
    message = str(raised.value)
    assert "Get-Acl : module could not be loaded" in message
    assert "exit code 1" in message
    assert "\n" not in message


def test_windows_harden_failure_is_a_hardening_error():
    def runner(argv):
        raise OSError("icacls missing")

    with pytest.raises(KeyHardeningError):
        WindowsPermissionGuard(runner=runner).harden("k", directory=False)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_posix_harden_then_verify_round_trips(tmp_path):
    path = tmp_path / "k"
    path.write_text("x")
    guard = PosixPermissionGuard()
    guard.harden(str(path), directory=False)
    assert (os.stat(path).st_mode & 0o777) == FILE_MODE
    guard.verify(str(path), directory=False)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_posix_verify_rejects_a_group_readable_key(tmp_path):
    path = tmp_path / "k"
    path.write_text("x")
    os.chmod(path, 0o640)
    with pytest.raises(KeyExposedError):
        PosixPermissionGuard().verify(str(path), directory=False)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
@pytest.mark.parametrize("mode, directory", [(0o604, False), (0o701, True)])
def test_posix_verify_rejects_other_bits_alone(tmp_path, mode, directory):
    # [CP2-C39] the other-bits isolated: a group-only mask (st_mode & 0o070) would pass a world-readable key.
    path = tmp_path / ("d" if directory else "k")
    if directory:
        path.mkdir()
    else:
        path.write_text("x")
    os.chmod(path, mode)
    with pytest.raises(KeyExposedError):
        PosixPermissionGuard().verify(str(path), directory=directory)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_posix_verify_accepts_a_stricter_mode(tmp_path):
    # [CP2-M40] 0400 is readable by no one else: not exposed (a false "exposed" forces a rotation, PM-S-6).
    path = tmp_path / "k"
    path.write_text("x")
    os.chmod(path, 0o400)
    PosixPermissionGuard().verify(str(path), directory=False)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_posix_verify_rejects_a_world_listable_secret_dir(tmp_path):
    os.chmod(tmp_path, 0o755)
    with pytest.raises(KeyExposedError):
        PosixPermissionGuard().verify(str(tmp_path), directory=True)
    assert DIR_MODE == 0o700


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX stat")
def test_posix_verify_that_cannot_stat_is_exposed(tmp_path, monkeypatch):
    # [CP2-M16]
    def _boom(*args, **kwargs):
        raise OSError("stale handle")

    monkeypatch.setattr(os, "stat", _boom)
    with pytest.raises(KeyExposedError):
        PosixPermissionGuard().verify(str(tmp_path / "k"), directory=False)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX owner")
def test_posix_verify_rejects_a_key_owned_by_another_user(tmp_path, monkeypatch):
    # [CP2-M2] a 0600 key owned by someone else is readable by that someone.
    path = tmp_path / "k"
    path.write_text("x")
    os.chmod(path, FILE_MODE)
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(path).st_uid + 1)
    with pytest.raises(KeyExposedError):
        PosixPermissionGuard().verify(str(path), directory=False)


@pytest.mark.skipif(sys.platform != "win32", reason="real ACLs")
def test_windows_real_secret_directory_hardens_and_verifies(tmp_path):
    # [CP2-M3] the .secret directory is the production artifact; round-trip it against real ACLs.
    secret = tmp_path / ".secret"
    secret.mkdir()
    guard = WindowsPermissionGuard()
    guard.harden(str(secret), directory=True)
    guard.verify(str(secret), directory=True)


@pytest.mark.skipif(sys.platform != "win32", reason="real ACLs")
def test_windows_real_file_hardened_then_loosened_is_detected(tmp_path):
    import subprocess

    path = tmp_path / "k"
    path.write_text("x")
    guard = WindowsPermissionGuard()
    guard.harden(str(path), directory=False)
    guard.verify(str(path), directory=False)
    subprocess.run(
        ["icacls", str(path), "/grant", "*S-1-5-32-545:R"], check=True, capture_output=True
    )
    with pytest.raises(KeyExposedError):
        guard.verify(str(path), directory=False)


def test_the_default_key_store_refuses_a_loosened_key_on_the_real_filesystem(tmp_path):
    # [CP2-C40] the composed negative of C32: loosen the real key, and the real store refuses it.
    import subprocess

    store = default_key_store()
    store.persist(str(tmp_path), store.new_key())
    path = store.key_path(str(tmp_path))
    if sys.platform == "win32":
        subprocess.run(
            ["icacls", path, "/grant", "*S-1-5-32-545:R"], check=True, capture_output=True
        )
    else:
        os.chmod(path, 0o644)
    with pytest.raises(KeyExposedError):
        store.load(str(tmp_path))


def test_the_default_key_store_round_trips_on_the_real_filesystem(tmp_path):
    # [CP2-C32] the two halves of key custody composed: persist then load with the platform guard.
    store = default_key_store()
    key = store.new_key()
    store.persist(str(tmp_path), key)
    assert store.load(str(tmp_path)) == key


def test_the_default_key_store_carries_the_platform_guard():
    # [CP2-C26] REQ-13: the production store verifies with the real guard, never a lenient one.
    expected = WindowsPermissionGuard if sys.platform == "win32" else PosixPermissionGuard
    assert isinstance(default_key_store().guard, expected)


def test_the_platform_factory_picks_the_matching_guard():
    expected = WindowsPermissionGuard if sys.platform == "win32" else PosixPermissionGuard
    assert isinstance(PermissionGuard.for_platform(), expected)


def test_a_failing_permission_tool_keeps_the_headline_of_a_long_stderr(monkeypatch):
    # A tool writes its headline first; a long trace must not push it out of the message.
    headline = "Get-Acl : The term 'Get-Acl' is not recognized"
    trace = " ".join(f"At line:{n} char:1" for n in range(200))

    def _fake_run(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, output="", stderr=f"{headline}\r\n{trace}\r\n")

    monkeypatch.setattr("magi_key_permissions.subprocess.run", _fake_run)
    with pytest.raises(OSError) as raised:
        WindowsPermissionGuard().current_sid()
    message = str(raised.value)
    assert headline in message
    assert "\n" not in message
    assert len(message) < len(headline) + len(trace)


def test_windows_verify_names_the_foreign_sids_and_the_protection(monkeypatch):
    # Operator diagnostics: SIDs and DACL flags are not secrets; the current user's SID is not listed.
    sddl = f"O:{ME}G:{ME}D:PAI(A;;FA;;;{ME})(A;OICI;FA;;;S-1-5-32-544)(A;;FA;;;BA)"
    with pytest.raises(KeyExposedError) as raised:
        _windows_guard(sddl).verify("k", directory=False)
    message = str(raised.value)
    assert "S-1-5-32-544" in message and "BA" in message
    assert "DACL protected: yes" in message
    assert ME not in message
    assert "\n" not in message


def test_windows_verify_that_cannot_read_names_the_cause_in_the_message():
    guard = _windows_guard("")

    def _fail(path):
        raise OSError("powershell.exe failed with exit code 1: Get-Acl : access denied")

    guard.read_sddl = _fail
    with pytest.raises(KeyExposedError) as raised:
        guard.verify("k", directory=False)
    assert "Get-Acl : access denied" in str(raised.value)


def test_windows_verify_accepts_the_current_user_rendered_as_an_alias():
    # M10: on windows-latest runneradmin is the built-in Administrator and its own SID reads back as LA.
    _windows_guard("D:PAI(A;;FA;;;LA)", aliases={"LA": ME}).verify("k", directory=False)


def test_windows_verify_names_an_alias_that_resolves_to_someone_else():
    with pytest.raises(KeyExposedError) as raised:
        _windows_guard(f"D:PAI(A;;FA;;;{ME})(A;;FA;;;LA)", aliases={"LA": OTHER}).verify(
            "k", directory=False
        )
    assert "LA" in str(raised.value)


def test_windows_verify_that_cannot_resolve_an_alias_is_exposed():
    guard = _windows_guard("D:PAI(A;;FA;;;LA)")

    def _fail(alias):
        raise OSError("powershell.exe failed with exit code 1: invalid SID")

    guard.resolve_alias = _fail
    with pytest.raises(KeyExposedError):
        guard.verify("k", directory=False)


def test_windows_verify_of_numeric_sids_never_resolves_an_alias():
    guard = _windows_guard(_fixture("sddl-file-hardened.txt").strip())

    def _never(alias):
        raise AssertionError(alias)

    guard.resolve_alias = _never
    guard.verify("k", directory=False)


def test_windows_verify_that_cannot_read_the_acl_is_unverifiable_not_exposed():
    # [Loop1 I-1] a read failure says "could not be verified", never "compromised".
    guard = _windows_guard("")

    def _fail(path):
        raise OSError("powershell.exe failed with exit code 1: Get-Acl : access denied")

    guard.read_sddl = _fail
    with pytest.raises(KeyUnverifiableError):
        guard.verify("k", directory=False)


def test_windows_verify_that_cannot_resolve_an_alias_is_unverifiable():
    guard = _windows_guard("D:PAI(A;;FA;;;LA)")
    guard.resolve_alias = _raise_os_error
    with pytest.raises(KeyUnverifiableError):
        guard.verify("k", directory=False)


def test_windows_verify_with_a_foreign_ace_is_exposed_not_unverifiable():
    with pytest.raises(KeyExposedError) as raised:
        _windows_guard(f"D:PAI(A;;FA;;;{ME})(A;;0x1200a9;;;{OTHER})").verify("k", directory=False)
    assert not isinstance(raised.value, KeyUnverifiableError)


def test_windows_verify_of_an_inheriting_dacl_is_exposed_not_unverifiable():
    with pytest.raises(KeyExposedError) as raised:
        _windows_guard(f"D:AI(A;;FA;;;{ME})").verify("k", directory=False)
    assert not isinstance(raised.value, KeyUnverifiableError)


def test_posix_verify_on_windows_is_unverifiable(monkeypatch):
    monkeypatch.setattr("magi_key_permissions.sys.platform", "win32")
    with pytest.raises(KeyUnverifiableError):
        PosixPermissionGuard().verify("k", directory=False)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX stat")
def test_posix_verify_that_cannot_stat_is_unverifiable(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "stat", _raise_os_error)
    with pytest.raises(KeyUnverifiableError):
        PosixPermissionGuard().verify(str(tmp_path / "k"), directory=False)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
def test_posix_verify_of_a_group_readable_key_is_exposed_not_unverifiable(tmp_path):
    path = tmp_path / "k"
    path.write_text("x")
    os.chmod(path, 0o640)
    with pytest.raises(KeyExposedError) as raised:
        PosixPermissionGuard().verify(str(path), directory=False)
    assert not isinstance(raised.value, KeyUnverifiableError)


def _raise_os_error(*args, **kwargs):
    raise OSError("tool failed")


_PARENT_SECRETS = {"OPENAI_API_KEY": "sk-parent-secret", "MAGI_PASSPHRASE": "parent-passphrase"}


def _capture_tool_envs(monkeypatch):
    """Fake ``subprocess.run`` for the permission tools, recording each child's environment."""
    seen = []

    def _fake_run(argv, **kwargs):
        seen.append((ntpath.basename(argv[0]).lower(), kwargs["env"]))
        if argv[0].lower().endswith("whoami.exe"):
            return SimpleNamespace(stdout=_fixture("whoami-nh.txt"))
        return SimpleNamespace(stdout=f"D:PAI(A;;FA;;;{ME})")

    for name, value in _PARENT_SECRETS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("magi_key_permissions.subprocess.run", _fake_run)
    # [S2] harden reads the DACL natively before icacls; the doubles answer an owner-only one.
    monkeypatch.setattr(
        "magi_key_permissions.WindowsPermissionGuard.read_sddl",
        lambda self, path: f"D:PAI(A;;FA;;;{ME})",
    )
    return seen


def test_no_permission_tool_inherits_an_api_key_or_a_passphrase(monkeypatch):
    # [W6] REQ-13c: whoami and icacls get the allowlist, never the operator's whole shell.
    seen = _capture_tool_envs(monkeypatch)
    guard = WindowsPermissionGuard()
    guard.harden("k", directory=False)
    tools = {tool for tool, _ in seen}
    assert {"whoami.exe", "icacls.exe"} <= tools
    for tool, env in seen:
        assert env is not None, tool
        assert not set(_PARENT_SECRETS) & {name.upper() for name in env}, tool
        assert not any(value in env.values() for value in _PARENT_SECRETS.values()), tool


def test_icacls_gets_exactly_the_allowlist(monkeypatch):
    # [n-3] whoami and icacls need only the allowlist (at a132856 they ran with it alone; only PowerShell, now
    # gone, failed): an exact equality, so a Windows system variable or any other addition fails it.
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\u\AppData\Local")
    monkeypatch.setenv("WINDIR", r"C:\Windows")
    monkeypatch.setenv("PATHEXT", ".COM;.EXE")
    seen = _capture_tool_envs(monkeypatch)
    WindowsPermissionGuard().harden("k", directory=False)
    ((_, env),) = [entry for entry in seen if entry[0] == "icacls.exe"]
    assert env == EnvironmentPolicy.build(os.environ)


def test_a_guard_asks_whoami_once_however_many_paths_it_checks():
    # [I-b] the SID does not change within a process: one whoami per guard, not per harden/verify.
    calls = []

    def runner(argv):
        calls.append(argv[0])
        return _fixture("whoami-nh.txt") if argv[0].lower().endswith("whoami.exe") else ""

    guard = WindowsPermissionGuard(runner=runner)
    guard.read_sddl = lambda path: f"D:PAI(A;;FA;;;{ME})"
    guard.harden("k", directory=False)
    guard.verify("k", directory=False)
    guard.verify("d", directory=True)
    assert sum(1 for tool in calls if tool.lower().endswith("whoami.exe")) == 1


def test_the_permission_tools_never_get_an_api_key_a_passphrase_or_psmodulepath(monkeypatch):
    # [W6b/n-3] secrets and PSModulePath never reach whoami or icacls.
    monkeypatch.setenv("PSModulePath", r"C:\Program Files\PowerShell\7\Modules")
    seen = _capture_tool_envs(monkeypatch)
    guard = WindowsPermissionGuard()
    guard.harden("k", directory=False)
    assert {"whoami.exe", "icacls.exe"} <= {tool for tool, _ in seen}
    for tool, env in seen:
        names = {name.upper() for name in env}
        assert not names & {"OPENAI_API_KEY", "MAGI_PASSPHRASE", "PSMODULEPATH"}, tool


def _no_process(*args, **kwargs):
    raise AssertionError("a process was started")


@pytest.mark.skipif(sys.platform != "win32", reason="real ACLs")
def test_reading_a_hardened_acl_starts_no_process_and_returns_its_sddl(tmp_path, monkeypatch):
    # [W6d] every PowerShell start on the CI runner took tens of seconds: the read is native (advapi32).
    path = tmp_path / "k"
    path.write_text("x")
    guard = WindowsPermissionGuard()
    guard.harden(str(path), directory=False)
    sid = guard.current_sid()
    monkeypatch.setattr("magi_key_permissions.subprocess.run", _no_process)
    sddl = guard.read_sddl(str(path))
    dacl = sddl[sddl.index("D:") :]
    assert dacl in (f"D:PAI(A;;FA;;;{sid})", "D:PAI(A;;FA;;;LA)")
    assert sddl.startswith("O:")


@pytest.mark.skipif(sys.platform != "win32", reason="real SIDs")
def test_resolving_an_alias_starts_no_process(monkeypatch):
    monkeypatch.setattr("magi_key_permissions.subprocess.run", _no_process)
    assert WindowsPermissionGuard().resolve_alias("BA") == "S-1-5-32-544"


@pytest.mark.skipif(sys.platform != "win32", reason="real SIDs")
def test_resolving_an_unknown_alias_is_an_os_error(monkeypatch):
    monkeypatch.setattr("magi_key_permissions.subprocess.run", _no_process)
    with pytest.raises(OSError):
        WindowsPermissionGuard().resolve_alias("ZZ")


@pytest.mark.skipif(sys.platform != "win32", reason="real ACLs")
def test_reading_the_acl_of_a_missing_path_names_the_win32_error(tmp_path, monkeypatch):
    # [W6d] a failed native read is an OSError naming the Win32 error; verify turns it into "unverifiable".
    monkeypatch.setattr("magi_key_permissions.subprocess.run", _no_process)
    missing = str(tmp_path / "absent")
    with pytest.raises(OSError) as raised:
        WindowsPermissionGuard().read_sddl(missing)
    message = str(raised.value)
    assert "Win32 error 2" in message and missing in message and "\n" not in message


@pytest.mark.skipif(sys.platform != "win32", reason="real ACLs")
def test_verifying_a_path_whose_acl_cannot_be_read_is_unverifiable(tmp_path):
    with pytest.raises(KeyUnverifiableError) as raised:
        WindowsPermissionGuard().verify(str(tmp_path / "absent"), directory=False)
    assert "Win32 error" in str(raised.value)


_FAKE_ADDRESS = 0x1000
"""A non-NULL pointer value a native double writes into an out-parameter; never dereferenced."""


def _native_double(**calls):
    """A ``_NativeSecurity`` whose API functions are the given doubles (no DLL is loaded).

    Every function not given fails the test if called; ``LocalFree`` records what it frees.
    """
    native = object.__new__(_NativeSecurity)
    freed = []

    def _unexpected(*args):
        raise AssertionError("unexpected native call")

    for name in (
        "_get_named_security_info",
        "_descriptor_to_string",
        "_string_to_sid",
        "_sid_to_string",
    ):
        setattr(native, name, calls.get(name, _unexpected))
    native._local_free = lambda pointer: freed.append(pointer)
    return native, freed


def _set_out(argument, value=_FAKE_ADDRESS):
    """Write ``value`` through a ``ctypes.byref`` out-parameter, as the API does."""
    argument._obj.value = value


def test_a_null_descriptor_after_a_reported_success_is_an_os_error():
    # [n-1] fail closed: a success with a NULL out-pointer is never dereferenced.
    native, _ = _native_double(_get_named_security_info=lambda *args: 0)
    with pytest.raises(OSError) as raised:
        native.read_sddl("k")
    assert "GetNamedSecurityInfoW" in str(raised.value) and "NULL" in str(raised.value)


def test_a_null_sddl_string_after_a_reported_success_is_an_os_error_and_frees_the_descriptor():
    # [n-1] the descriptor the first call returned is still released.
    native, freed = _native_double(
        _get_named_security_info=lambda *args: _set_out(args[-1]) or 0,
        _descriptor_to_string=lambda *args: 1,
    )
    with pytest.raises(OSError) as raised:
        native.read_sddl("k")
    message = str(raised.value)
    assert "ConvertSecurityDescriptorToStringSecurityDescriptorW" in message and "NULL" in message
    assert [pointer.value for pointer in freed] == [_FAKE_ADDRESS]


def test_a_null_sid_after_a_reported_success_is_an_os_error():
    native, _ = _native_double(_string_to_sid=lambda *args: 1)
    with pytest.raises(OSError) as raised:
        native.resolve_alias("BA")
    assert "ConvertStringSidToSidW" in str(raised.value) and "NULL" in str(raised.value)


def test_a_null_sid_string_after_a_reported_success_is_an_os_error_and_frees_the_sid():
    native, freed = _native_double(
        _string_to_sid=lambda *args: _set_out(args[-1]) or 1,
        _sid_to_string=lambda *args: 1,
    )
    with pytest.raises(OSError) as raised:
        native.resolve_alias("BA")
    assert "ConvertSidToStringSidW" in str(raised.value) and "NULL" in str(raised.value)
    assert [pointer.value for pointer in freed] == [_FAKE_ADDRESS]


class _NativeAnswering:
    """A native binding double whose alias resolution answers ``answer``."""

    def __init__(self, answer):
        self.answer = answer

    def resolve_alias(self, alias):
        return self.answer


def test_an_alias_that_resolves_to_something_other_than_a_sid_is_an_os_error():
    # [n-2] passes in Red by design: pins the SID-shape check the guard keeps on the native answer.
    guard = WindowsPermissionGuard()
    guard._native = _NativeAnswering("Administrators")
    with pytest.raises(OSError):
        guard.resolve_alias("BA")


def test_a_dacl_whose_alias_resolves_to_a_non_sid_is_unverifiable():
    # [n-2] the same check reached through verify: fail closed as "could not be verified".
    guard = _windows_guard("D:PAI(A;;FA;;;LA)")
    del guard.resolve_alias  # back to the real method, over the native double
    guard._native = _NativeAnswering("not a sid")
    with pytest.raises(KeyUnverifiableError):
        guard.verify("k", directory=False)


_REAL_STAT = os.stat


def _owned_by(monkeypatch, owner_uid, current_uid, paths):
    """Make ``paths`` look owner-only (0600/0700) but owned by ``owner_uid``, as a POSIX host would report."""
    targets = {os.path.normcase(str(path)) for path in paths}

    def _stat(path, *args, **kwargs):
        if os.path.normcase(str(path)) in targets:
            return SimpleNamespace(st_uid=owner_uid, st_mode=0o600)
        return _REAL_STAT(path, *args, **kwargs)

    monkeypatch.setattr("magi_key_permissions.sys.platform", "linux")
    monkeypatch.setattr("magi_key_permissions.os.stat", _stat)
    monkeypatch.setattr("magi_key_permissions.os.geteuid", lambda: current_uid, raising=False)


def test_a_key_owned_by_another_uid_is_unverifiable_never_called_compromised(tmp_path, monkeypatch):
    # [R1] a UID mismatch (container mount, sudo-created file) is not proof of exposure: never route it to a
    # destructive rotation as "compromised".
    store = KeyStore(PosixPermissionGuard())
    key_file = tmp_path / ".magi" / ".secret" / "magi.key"
    key_file.parent.mkdir(parents=True)
    key_file.write_text("k" * 44, encoding="ascii")
    _owned_by(monkeypatch, owner_uid=0, current_uid=1000, paths=[key_file.parent, key_file])
    with pytest.raises(KeyUnverifiableError) as raised:
        store.load(str(tmp_path))
    message = str(raised.value)
    assert "uid 0" in message and "1000" in message and "ownership" in message
    assert "compromised" not in message and "--ollama-rotate-key" in message


def test_group_or_other_bits_on_a_posix_key_stay_exposed(tmp_path, monkeypatch):
    # [R1] only the owner check moves to "unverifiable"; readable bits remain an exposure.
    path = tmp_path / "k"
    path.write_text("x")

    def _stat(target, *args, **kwargs):
        if os.path.normcase(str(target)) == os.path.normcase(str(path)):
            return SimpleNamespace(st_uid=1000, st_mode=0o640)
        return _REAL_STAT(target, *args, **kwargs)

    monkeypatch.setattr("magi_key_permissions.sys.platform", "linux")
    monkeypatch.setattr("magi_key_permissions.os.stat", _stat)
    monkeypatch.setattr("magi_key_permissions.os.geteuid", lambda: 1000, raising=False)
    with pytest.raises(KeyExposedError) as raised:
        PosixPermissionGuard().verify(str(path), directory=False)
    assert not isinstance(raised.value, KeyUnverifiableError)


EVERYONE = "S-1-1-0"


def test_a_foreign_deny_entry_alone_is_unverifiable_and_never_called_compromised(tmp_path):
    # [S2] a deny entry takes access away: it is not an exposure, so it never routes to a destructive rotation.
    key_file = tmp_path / ".magi" / ".secret" / "magi.key"
    key_file.parent.mkdir(parents=True)
    key_file.write_text("k" * 44, encoding="ascii")
    store = KeyStore(_windows_guard(f"D:PAI(A;;FA;;;{ME})(D;;FR;;;{EVERYONE})"))
    with pytest.raises(KeyUnverifiableError) as raised:
        store.load(str(tmp_path))
    message = str(raised.value)
    assert "deny entry" in message and EVERYONE in message and "compromised" not in message


def test_a_foreign_allow_entry_is_exposed_and_named_an_access_entry():
    with pytest.raises(KeyExposedError) as raised:
        _windows_guard(f"D:PAI(A;;FA;;;{ME})(A;;FR;;;{OTHER})").verify("k", directory=False)
    assert not isinstance(raised.value, KeyUnverifiableError)
    assert f"has an access entry for {OTHER}" in str(raised.value)


def test_a_foreign_sid_with_both_a_deny_and_an_allow_entry_is_exposed():
    # [S2] the ACE type only classifies the message: any non-deny entry of a foreign SID is an exposure.
    sddl = f"D:PAI(A;;FA;;;{ME})(D;;FW;;;{OTHER})(A;;FR;;;{OTHER})"
    with pytest.raises(KeyExposedError) as raised:
        _windows_guard(sddl).verify("k", directory=False)
    assert not isinstance(raised.value, KeyUnverifiableError)


def test_a_deny_only_sid_beside_another_sid_with_access_is_exposed():
    sddl = f"D:PAI(A;;FA;;;{ME})(D;;FR;;;{EVERYONE})(A;;FR;;;{OTHER})"
    with pytest.raises(KeyExposedError) as raised:
        _windows_guard(sddl).verify("k", directory=False)
    assert not isinstance(raised.value, KeyUnverifiableError) and OTHER in str(raised.value)


def test_a_deny_entry_still_fails_verification_whatever_its_flags():
    # [S2] fail-closed: a deny entry never makes verify pass, inherited or not.
    for sddl in (
        f"D:PAI(A;;FA;;;{ME})(D;ID;FR;;;{EVERYONE})",
        f"D:PAI(A;;FA;;;{ME})(D;OICI;FR;;;WD)",
    ):
        with pytest.raises(KeyExposedError):
            _windows_guard(sddl).verify("k", directory=False)


def test_windows_harden_removes_the_deny_entries_it_finds():
    # [S2] /remove:g alone leaves an explicit deny ACE behind; harden removes the SID entirely.
    # [S2 correction] adapted: removing only a deny could widen access, so the SID goes with a plain /remove.
    seen = []

    def runner(argv):
        seen.append(argv)
        return _fixture("whoami-nh.txt") if argv[0].lower().endswith("whoami.exe") else ""

    guard = WindowsPermissionGuard(runner=runner)
    guard.read_sddl = lambda path: f"D:PAI(A;;FA;;;{ME})(D;;FR;;;WD)"
    guard.resolve_alias = _ALIASES.__getitem__
    guard.harden("k", directory=False)
    icacls = next(argv for argv in seen if "icacls" in argv[0].lower())
    assert "/remove:d" not in icacls
    assert f"*{EVERYONE}" in icacls[icacls.index("/remove") + 1 :]


@pytest.mark.skipif(sys.platform != "win32", reason="real ACLs")
def test_windows_real_directory_with_a_deny_entry_hardens_and_verifies(tmp_path):
    # [S2] real icacls: an explicit deny for Everyone survives /remove:g; harden must remove it. (W), not (R):
    # tmp_path is a 0o700 directory whose inherited OWNER RIGHTS entry cancels the owner's implicit
    # READ_CONTROL, so a deny of (R) makes the DACL unreadable and harden fails closed instead.
    secret = tmp_path / ".secret"
    secret.mkdir()
    subprocess.run(
        ["icacls", str(secret), "/deny", f"*{EVERYONE}:(W)"], check=True, capture_output=True
    )
    guard = WindowsPermissionGuard()
    guard.harden(str(secret), directory=True)
    guard.verify(str(secret), directory=True)


def test_windows_harden_removes_a_foreign_sid_with_an_allow_and_a_deny_entirely_in_one_call():
    # [S2 correction] removing only the deny would leave the allow it blocked: harden must never widen access.
    seen = []

    def runner(argv):
        seen.append(argv)
        return _fixture("whoami-nh.txt") if argv[0].lower().endswith("whoami.exe") else ""

    guard = WindowsPermissionGuard(runner=runner)
    guard.read_sddl = lambda path: f"D:PAI(A;;FA;;;{ME})(D;;FW;;;{OTHER})(A;;FR;;;{OTHER})"
    guard.resolve_alias = _ALIASES.__getitem__
    guard.harden("k", directory=False)
    icacls_calls = [argv for argv in seen if "icacls" in argv[0].lower()]
    assert len(icacls_calls) == 1
    (icacls,) = icacls_calls
    assert "/remove:d" not in icacls and "/remove:g" not in icacls
    assert f"*{OTHER}" in icacls[icacls.index("/remove") + 1 :]
    assert "/inheritance:r" in icacls and any(part.startswith(f"*{ME}:") for part in icacls)


def test_windows_harden_removes_a_foreign_allow_only_sid_too():
    # [S2 correction] every foreign SID in the DACL goes, allow-only ones included.
    seen = []

    def runner(argv):
        seen.append(argv)
        return _fixture("whoami-nh.txt") if argv[0].lower().endswith("whoami.exe") else ""

    guard = WindowsPermissionGuard(runner=runner)
    guard.read_sddl = lambda path: f"D:PAI(A;;FA;;;{ME})(A;;FR;;;{OTHER})"
    guard.harden("k", directory=False)
    (icacls,) = [argv for argv in seen if "icacls" in argv[0].lower()]
    assert f"*{OTHER}" in icacls[icacls.index("/remove") + 1 :]


@pytest.mark.skipif(sys.platform != "win32", reason="real ACLs")
def test_windows_real_file_with_an_allow_and_a_deny_for_users_hardens_to_owner_only(tmp_path):
    # [S2 correction] grant Users:R and deny Users:W, harden, verify: the DACL is the owner's alone.
    path = tmp_path / "k"
    path.write_text("x")
    users = "S-1-5-32-545"
    subprocess.run(
        ["icacls", str(path), "/grant", f"*{users}:(R)"], check=True, capture_output=True
    )
    subprocess.run(["icacls", str(path), "/deny", f"*{users}:(W)"], check=True, capture_output=True)
    guard = WindowsPermissionGuard()
    guard.harden(str(path), directory=False)
    guard.verify(str(path), directory=False)
    _, sids = _protection_and_sids(guard.read_sddl(str(path)))
    assert sids and all(sid in (guard.current_sid(), "LA") for sid in sids)


def test_a_foreign_object_deny_entry_alone_is_unverifiable():
    # [n-1] an object deny ACE (OD) takes access away like a plain deny.
    sddl = f"D:PAI(A;;FA;;;{ME})(OD;;FR;;;{EVERYONE})"
    with pytest.raises(KeyUnverifiableError) as raised:
        _windows_guard(sddl).verify("k", directory=False)
    assert "deny entry" in str(raised.value)


def test_a_foreign_object_deny_beside_an_allow_for_the_same_sid_is_exposed():
    sddl = f"D:PAI(A;;FA;;;{ME})(OD;;FR;;;{OTHER})(A;;FR;;;{OTHER})"
    with pytest.raises(KeyExposedError) as raised:
        _windows_guard(sddl).verify("k", directory=False)
    assert not isinstance(raised.value, KeyUnverifiableError)
