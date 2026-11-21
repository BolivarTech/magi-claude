# Author: Julian Bolivar
# Version: 7.1.0
# Date: 2026-11-19
"""Tests for the key rotation (MS1 Task 9)."""

import os
import shutil
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from magi_key import KeyHardeningError, KeyStore, MagiKey
from ollama_init import OllamaInitError, run_ollama_init
from magi_key_rotation import (
    KeyRotation,
    RotationFailedError,
    RotationNotAuthorisedError,
    rotation_backup_root,
)
from magi_rs_vault import VAULT_KEY_NAMES, VaultCommandError

OLD = MagiKey("old-key-DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD=")


def _project(tmp_path):
    work = tmp_path / "repo"
    secret = work / ".magi" / ".secret"
    secret.mkdir(parents=True)
    (secret / "magi.key").write_text(OLD.reveal(), encoding="ascii")
    (work / ".magi" / "magi.toml").write_text("old", encoding="utf-8")
    return work


class _Store(KeyStore):
    """Test double: only `read_unverified` and `guard` are used by the rotation ([CP2-B14])."""

    def __init__(self, guard=None):
        self.guard = guard or _NoopGuard()

    def read_unverified(self, path):
        return MagiKey(open(path, encoding="ascii").read())


NEW = MagiKey(
    "new-key-HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH="
)  # [CP2-M49] written by the init double


def _rotation(tmp_path, init=None, vault_ls=None, guard=None, clock=None):
    used = {"ls": []}

    def _init(workdir, store):
        os.makedirs(os.path.join(workdir, ".magi", ".secret"))
        open(os.path.join(workdir, ".magi", "magi.toml"), "w").write("new")
        open(os.path.join(workdir, ".magi", ".secret", "magi.key"), "w", encoding="ascii").write(
            NEW.reveal()
        )
        return "initialised"

    def _ls(backup_workdir, key):
        used["ls"].append((backup_workdir, key))
        return ["OPENAI_API_KEY"]

    extra = {"clock": clock} if clock is not None else {}
    rotation = KeyRotation(
        _Store(guard),
        init or _init,
        vault_ls or _ls,
        backup_root=str(tmp_path / "rotated"),
        **extra,
    )
    return rotation, used


class _NoopGuard:
    def harden(self, path, directory):
        pass

    def verify(self, path, directory):
        pass


class _RecordingGuard(_NoopGuard):
    """[CP2-M1] Records every harden/verify call."""

    def __init__(self):
        self.calls = []

    def harden(self, path, directory):
        self.calls.append(("harden", path, directory))

    def verify(self, path, directory):
        self.calls.append(("verify", path, directory))


class _FailingOnBackupGuard(_NoopGuard):
    """[CP2-M18] Fails to harden anything inside the moved backup, ``<backup>/<id>/.magi``."""

    def harden(self, path, directory):
        parts = os.path.normpath(path).split(os.sep)
        if "rotated" in parts and ".magi" in parts[parts.index("rotated") :]:
            raise KeyHardeningError(f"cannot harden {path}")


class _FailingOnBackupIdGuard(_NoopGuard):
    """[CP2-M32] Fails to harden the backup directory ``<backup>/<id>`` itself."""

    def harden(self, path, directory):
        parts = os.path.normpath(path).split(os.sep)
        if len(parts) >= 2 and parts[-2] == "rotated":
            raise KeyHardeningError(f"cannot harden {path}")


def test_without_tty_and_without_yes_nothing_changes(tmp_path):
    work = _project(tmp_path)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationNotAuthorisedError):
        rotation.rotate(str(work), assume_yes=False, is_tty=False)
    assert (work / ".magi" / "magi.toml").read_text() == "old"


def test_yes_on_a_tty_authorises_without_asking(tmp_path):
    # [CP2-M50] decided by the user: --yes is the authorisation; confirm is never called.
    work = _project(tmp_path)
    rotation, _ = _rotation(tmp_path)
    prompts = []
    rotation.rotate(
        str(work), assume_yes=True, is_tty=True, confirm=lambda p: prompts.append(p) or ""
    )
    assert prompts == [] and (work / ".magi" / "magi.toml").read_text() == "new"


def test_a_rotation_without_an_explicit_root_uses_the_platform_state_dir(monkeypatch, tmp_path):
    # [CP2-M51] the default backup_root is constructed for real, not only asserted to be None.
    variable = "LOCALAPPDATA" if sys.platform == "win32" else "XDG_STATE_HOME"
    monkeypatch.setenv(variable, str(tmp_path / "state"))
    work = _project(tmp_path)

    def _init(workdir, store):
        os.makedirs(os.path.join(workdir, ".magi", ".secret"))
        return "initialised"

    report = KeyRotation(_Store(), _init, lambda w, k: []).rotate(
        str(work), assume_yes=True, is_tty=False
    )
    assert report.backup_dir.startswith(os.path.join(str(tmp_path / "state"), "magi", "rotated"))


def test_a_tty_answer_y_authorises_the_rotation(tmp_path):
    # [CP2-M43] REQ-13f: the interactive path is exercised, not only its refusal.
    work = _project(tmp_path)
    rotation, _ = _rotation(tmp_path)
    prompts = []
    rotation.rotate(
        str(work), assume_yes=False, is_tty=True, confirm=lambda p: prompts.append(p) or "y"
    )
    assert len(prompts) == 1 and (work / ".magi" / "magi.toml").read_text() == "new"


def test_a_backup_root_inside_the_working_tree_is_refused_before_anything_moves(tmp_path):
    # [CP2-C28] BDD-18 / PM-C-1 must not rest on an environment variable.
    work = _project(tmp_path)
    rotation = KeyRotation(
        _Store(), lambda w, s: "x", lambda w, k: [], backup_root=str(work / "state" / "rotated")
    )
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert (work / ".magi" / "magi.toml").read_text() == "old" and not (work / "state").exists()
    assert "\n" not in str(exc.value).strip()


def test_a_tty_answer_other_than_y_changes_nothing(tmp_path):
    work = _project(tmp_path)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationNotAuthorisedError):
        rotation.rotate(str(work), assume_yes=False, is_tty=True, confirm=lambda prompt: "")
    assert (work / ".magi" / "magi.toml").read_text() == "old"


def test_yes_moves_the_old_magi_outside_the_tree_and_creates_a_new_one(tmp_path):
    work = _project(tmp_path)
    rotation, _ = _rotation(tmp_path)
    report = rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert (work / ".magi" / "magi.toml").read_text() == "new"
    assert open(os.path.join(report.backup_dir, ".magi", "magi.toml")).read() == "old"
    assert not os.path.commonpath([report.backup_dir, str(work)]) == str(work)


def test_nothing_of_the_backup_is_left_inside_the_working_tree(tmp_path):
    work = _project(tmp_path)
    rotation, _ = _rotation(tmp_path)
    rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert sorted(os.listdir(work)) == [".magi"]


def test_the_old_key_is_used_only_for_vault_ls_on_the_backup(tmp_path):
    work = _project(tmp_path)
    rotation, used = _rotation(tmp_path)
    report = rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert used["ls"] == [(report.backup_dir, OLD)]
    assert report.names_to_revoke == ["OPENAI_API_KEY"] and report.names_from_vault is True


def test_a_rotation_without_a_magi_dir_changes_nothing_and_says_so(tmp_path):
    # [PM-MS1-C9]
    work = tmp_path / "repo"
    work.mkdir()
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert ".magi" in str(exc.value) and not (tmp_path / "rotated").exists()


@pytest.mark.parametrize(
    "error", [RuntimeError, VaultCommandError, OSError]
)  # [CP2-C45] what production raises
def test_a_failing_vault_ls_lists_the_whole_closed_list(tmp_path, error):
    work = _project(tmp_path)

    def _ls(*a):
        raise error("cannot open")

    rotation, _ = _rotation(tmp_path, vault_ls=_ls)
    report = rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert report.names_to_revoke == list(VAULT_KEY_NAMES) and report.names_from_vault is False


@pytest.mark.parametrize(
    "error", [RuntimeError, KeyHardeningError, OSError, OllamaInitError]
)  # [CP2-C29]
def test_a_failed_init_leaves_the_backup_intact_and_no_half_made_magi(tmp_path, error):
    work = _project(tmp_path)

    def _init(workdir, store):
        os.makedirs(os.path.join(workdir, ".magi"))
        raise error("init died")

    rotation, _ = _rotation(tmp_path, init=_init)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert not (work / ".magi").exists()
    backups = os.listdir(tmp_path / "rotated")
    assert len(backups) == 1
    assert (tmp_path / "rotated" / backups[0] / ".magi" / "magi.toml").read_text() == "old"
    assert "\n" not in str(exc.value).strip()
    assert OLD.reveal() not in str(exc.value)  # [CP2-M7]
    assert backups[0] in str(exc.value)  # [CP2-M47] the operator learns where the old .magi/ went


def test_a_real_init_that_fails_makes_the_rotation_fail(tmp_path, monkeypatch):
    # [CP2-C29] the composition: the real run_ollama_init over a non-zero child never yields a "successful" rotation.
    work = _project(tmp_path)
    monkeypatch.setattr(
        "ollama_init.subprocess.run",
        lambda argv, **kw: SimpleNamespace(returncode=1, stdout="", stderr="error: init died"),
    )
    monkeypatch.setattr(
        "ollama_init.BinaryBootstrap",
        lambda *a, **k: SimpleNamespace(
            ensure=lambda: "/bin/magi-rs", verify=lambda: "/bin/magi-rs"
        ),
    )
    rotation, _ = _rotation(tmp_path, init=run_ollama_init)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert not (work / ".magi").exists()
    backups = os.listdir(tmp_path / "rotated")
    assert len(backups) == 1 and backups[0] in str(exc.value)  # [CP2-M47]


def test_a_move_that_copied_and_then_failed_mid_delete_keeps_both_copies_and_names_them(
    tmp_path, monkeypatch
):
    # [CP2-M44] cross-volume move = copytree + rmtree: a half-deleted source must not cost the full copy.
    work = _project(tmp_path)

    def _move(src, dst):
        shutil.copytree(src, dst)
        os.remove(os.path.join(src, "magi.toml"))
        raise OSError("file in use")

    monkeypatch.setattr("magi_key_rotation.shutil.move", _move)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    backups = os.listdir(tmp_path / "rotated")
    assert len(backups) == 1
    full = tmp_path / "rotated" / backups[0] / ".magi"
    assert (full / "magi.toml").read_text() == "old" and (full / ".secret" / "magi.key").exists()
    message = str(exc.value)
    assert backups[0] in message and "\n" not in message.strip() and OLD.reveal() not in message


def test_the_backup_key_and_its_directory_are_hardened_and_verified(tmp_path):
    # [CP2-M1] REQ-13g.1: a cross-volume move does not carry the ACL; the backup is re-hardened.
    work = _project(tmp_path)
    guard = _RecordingGuard()
    rotation, _ = _rotation(tmp_path, guard=guard)
    report = rotation.rotate(str(work), assume_yes=True, is_tty=False)
    secret = os.path.join(report.backup_dir, ".magi", ".secret")
    key = os.path.join(secret, "magi.key")
    for call in (
        ("harden", secret, True),
        ("harden", key, False),
        ("verify", secret, True),
        ("verify", key, False),
    ):
        assert call in guard.calls
    assert (
        "harden",
        report.backup_dir,
        True,
    ) in guard.calls  # [CP2-C44] the backup directory itself
    assert ("verify", report.backup_dir, True) in guard.calls  # [fix 1] verified, not only hardened


def test_a_failed_move_leaves_the_magi_dir_in_place_and_no_partial_backup(tmp_path, monkeypatch):
    # [CP2-M5] REQ-13h at the most plausible failure point (cross-volume copy, locked or full target).
    work = _project(tmp_path)

    def _move(src, dst):
        os.makedirs(os.path.join(dst, ".secret"))
        raise OSError("disk full")

    monkeypatch.setattr("magi_key_rotation.shutil.move", _move)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert (work / ".magi" / "magi.toml").read_text() == "old"
    assert list((tmp_path / "rotated").glob("*")) == []
    assert "\n" not in str(exc.value).strip() and OLD.reveal() not in str(exc.value)


def test_an_unwritable_backup_root_is_one_line_and_moves_nothing(tmp_path, monkeypatch):
    # [CP2-M32] REQ-13h: creating <backup>/<id> fails before the move.
    work = _project(tmp_path)
    real_makedirs = os.makedirs

    def _makedirs(path, *args, **kwargs):
        if "rotated" in os.path.normpath(str(path)).split(os.sep):
            raise PermissionError("read-only backup root")
        return real_makedirs(path, *args, **kwargs)

    monkeypatch.setattr("magi_key_rotation.os.makedirs", _makedirs)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert (work / ".magi" / "magi.toml").read_text() == "old"
    assert "\n" not in str(exc.value).strip() and OLD.reveal() not in str(exc.value)


def test_an_unreadable_old_key_stops_the_rotation_before_anything_moves(tmp_path):
    # [CP2-M5] the old key is read before the move.
    work = _project(tmp_path)

    class _Broken(_Store):
        def read_unverified(self, path):
            raise OSError("unreadable")

    rotation = KeyRotation(
        _Broken(), lambda w, s: "x", lambda w, k: [], backup_root=str(tmp_path / "rotated")
    )
    with pytest.raises(RotationFailedError):
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert (work / ".magi" / "magi.toml").read_text() == "old"
    assert list((tmp_path / "rotated").glob("*")) == []


def test_two_rotations_in_the_same_second_get_distinct_backups(tmp_path):
    # [CP2-M6] consumes `clock`: an immediate retry must not collide with the intact backup.
    fixed = datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc)
    work = _project(tmp_path)
    rotation, _ = _rotation(tmp_path, clock=lambda: fixed)
    first = rotation.rotate(str(work), assume_yes=True, is_tty=False)
    (work / ".magi" / ".secret" / "magi.key").write_text(OLD.reveal(), encoding="ascii")
    second = rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert first.backup_dir.endswith("-20261120T120000Z")
    assert second.backup_dir.endswith("-20261120T120000Z-2")


def test_a_backup_that_cannot_be_rehardened_stops_the_rotation_and_names_the_backup(tmp_path):
    # [CP2-M18] REQ-13h: the backup is kept (never deleted), no new .magi/, one line naming the backup.
    work = _project(tmp_path)
    rotation, _ = _rotation(tmp_path, guard=_FailingOnBackupGuard())
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    backups = os.listdir(tmp_path / "rotated")
    assert len(backups) == 1
    assert (tmp_path / "rotated" / backups[0] / ".magi" / "magi.toml").read_text() == "old"
    assert not (work / ".magi").exists()
    message = str(exc.value)
    assert backups[0] in message and "\n" not in message.strip() and OLD.reveal() not in message


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fallback")
def test_the_backup_root_falls_back_to_local_state_without_xdg(monkeypatch, tmp_path):
    # [CP2-M25]
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert rotation_backup_root() == os.path.join(
        str(tmp_path), ".local", "state", "magi", "rotated"
    )


def test_the_default_backup_root_is_the_platform_state_dir(monkeypatch, tmp_path):
    # [CP2-M4] the outside-the-tree guarantee of BDD-18 rests on this default.
    variable = "LOCALAPPDATA" if sys.platform == "win32" else "XDG_STATE_HOME"
    monkeypatch.setenv(variable, str(tmp_path / "state"))
    assert rotation_backup_root() == os.path.join(str(tmp_path / "state"), "magi", "rotated")


def test_no_output_of_a_rotation_contains_either_key(tmp_path, capsys):
    # [CP2-M49] the init double writes NEW, so the new-key half can redden too; stdout/stderr included.
    work = _project(tmp_path)
    rotation, _ = _rotation(tmp_path)
    report = rotation.rotate(str(work), assume_yes=True, is_tty=False)
    out = capsys.readouterr()
    text = repr(report) + out.out + out.err
    assert OLD.reveal() not in text and NEW.reveal() not in text


def test_a_backup_dir_that_cannot_be_hardened_stops_before_anything_moves(tmp_path):
    # [CP2-M32] REQ-13g.1: <backup>/<id> is restricted before the old .magi/ goes into it.
    work = _project(tmp_path)
    rotation, _ = _rotation(tmp_path, guard=_FailingOnBackupIdGuard())
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert (work / ".magi" / "magi.toml").read_text() == "old"
    assert (work / ".magi" / ".secret" / "magi.key").read_text() == OLD.reveal()
    assert list((tmp_path / "rotated").glob("*")) == []
    message = str(exc.value)
    assert (
        "nothing was moved" in message
        and "\n" not in message.strip()
        and OLD.reveal() not in message
    )


def test_a_move_that_lost_any_source_file_keeps_both_copies_and_names_them(tmp_path, monkeypatch):
    # [fix 2] REQ-13h: "complete" compares contents, not only magi.toml and the key.
    work = _project(tmp_path)
    (work / ".magi" / ".magi-rs-memory.db").write_bytes(b"memory")

    def _move(src, dst):
        shutil.copytree(src, dst)
        os.remove(os.path.join(src, ".magi-rs-memory.db"))
        raise OSError("file in use")

    monkeypatch.setattr("magi_key_rotation.shutil.move", _move)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    backups = os.listdir(tmp_path / "rotated")
    assert len(backups) == 1
    full = tmp_path / "rotated" / backups[0] / ".magi"
    assert (full / ".magi-rs-memory.db").read_bytes() == b"memory"
    assert (work / ".magi" / "magi.toml").read_text() == "old"
    message = str(exc.value)
    assert backups[0] in message and str(work / ".magi") in message
    assert (
        "unchanged" not in message and "\n" not in message.strip() and OLD.reveal() not in message
    )


def test_an_eof_at_the_confirmation_prompt_is_not_an_authorisation(tmp_path):
    # [fix 3] a closed stdin at the prompt changes nothing.
    work = _project(tmp_path)
    rotation, _ = _rotation(tmp_path)

    def _eof(prompt):
        raise EOFError

    with pytest.raises(RotationNotAuthorisedError):
        rotation.rotate(str(work), assume_yes=False, is_tty=True, confirm=_eof)
    assert (work / ".magi" / "magi.toml").read_text() == "old"


def test_an_init_error_quoting_the_old_key_is_redacted(tmp_path):
    # [fix 3] passes in Red by design (the redaction exists); pins it across the reorder.
    work = _project(tmp_path)

    def _init(workdir, store):
        raise RuntimeError(f"bad passphrase {OLD.reveal()}\nsecond line")

    rotation, _ = _rotation(tmp_path, init=_init)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert OLD.reveal() not in str(exc.value) and "<redacted>" in str(exc.value)
