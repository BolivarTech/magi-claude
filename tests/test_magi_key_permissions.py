import ntpath
import os
import sys

import pytest

from magi_key import KeyExposedError, KeyHardeningError
from magi_key_permissions import (
    DIR_MODE,
    FILE_MODE,
    PermissionGuard,
    PosixPermissionGuard,
    WindowsPermissionGuard,
    default_key_store,
    parse_sddl_dacl,
)

ME = "S-1-5-21-1111-2222-3333-1001"
OTHER = "S-1-5-11"  # Authenticated Users
_ACL = os.path.join(
    os.path.dirname(__file__), "fixtures", "windows-acl"
)  # [CP2-M45] the M9 capture


def _fixture(name):
    return open(os.path.join(_ACL, name), encoding="utf-8").read()


def test_sddl_parser_reads_protection_and_every_ace_sid():
    protected, sids = parse_sddl_dacl(f"O:{ME}G:{ME}D:PAI(A;;FA;;;{ME})")
    assert protected is True
    assert sids == [ME]


def test_sddl_parser_sees_an_inherited_extra_ace():
    protected, sids = parse_sddl_dacl(f"D:AI(A;;FA;;;{ME})(A;ID;0x1301bf;;;{OTHER})")
    assert protected is False
    assert sids == [ME, OTHER]


def test_sddl_parser_returns_aliases_and_container_aces_as_sids():
    # [CP2-M3] fail-closed: an alias and an OICI-flagged ACE are ACEs too.
    protected, sids = parse_sddl_dacl(f"D:PAI(A;OICI;FA;;;{ME})(A;;FR;;;AU)")
    assert protected is True
    assert sids == [ME, "AU"]


def test_sddl_parser_reads_a_flagless_dacl_as_unprotected():
    # [CP2-M24]
    protected, sids = parse_sddl_dacl(f"D:(A;;FA;;;{ME})")
    assert protected is False
    assert sids == [ME]


@pytest.mark.parametrize("sddl", [f"O:{ME}G:{ME}", "D:NO_ACCESS_CONTROL"])
def test_sddl_parser_rejects_an_absent_or_null_dacl(sddl):
    # [CP2-C38] a NULL or missing DACL grants everyone full access: never "protected, no ACEs".
    with pytest.raises(ValueError):
        parse_sddl_dacl(sddl)


def test_sddl_parser_rejects_a_malformed_ace():
    with pytest.raises(ValueError):
        parse_sddl_dacl(f"D:PAI(A;;FA;;;{ME})(A;;FA)")


def _windows_guard(sddl):
    def runner(argv):
        if argv[0].lower().endswith("whoami.exe"):
            return _fixture("whoami-nh.txt")  # [CP2-M45] the real /nh shape
        if argv[0].lower().endswith("icacls.exe") or argv[0] == "icacls":
            return ""
        raise AssertionError(argv)

    guard = WindowsPermissionGuard(runner=runner)
    guard.read_sddl = lambda path: sddl
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

    WindowsPermissionGuard(runner=runner).harden("k", directory=False)
    # [CP2-M45] whoami by absolute path: on PATH, git-bash would answer with coreutils' whoami (M9).
    assert ntpath.isabs(seen[0][0]) and seen[0][0].lower().endswith("system32\\whoami.exe")
    icacls = next(a for a in seen if "icacls" in a[0].lower())
    assert "/inheritance:r" in icacls
    assert any(part.startswith(f"*{ME}:") for part in icacls)


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
