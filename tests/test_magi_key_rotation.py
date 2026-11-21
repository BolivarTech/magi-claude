# Author: Julian Bolivar
# Version: 7.1.0
# Date: 2026-11-19
"""Tests for the key rotation (MS1 Task 9)."""

import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from magi_key import KeyExposedError, KeyHardeningError, KeyMissingError, KeyStore, MagiKey
from ollama_init import InitLeftover, OllamaInitError, run_ollama_init
from magi_key_rotation import (
    KeyRotation,
    RotationFailedError,
    RotationNotAuthorisedError,
    rotation_backup_root,
)
from magi_rs_vault import VAULT_KEY_NAMES, VaultClient, VaultCommandError

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
        # [Loop1 M-6] the rotation now loads the new key back, so the double writes one.
        os.makedirs(os.path.join(workdir, ".magi", ".secret"))
        with open(os.path.join(workdir, ".magi", ".secret", "magi.key"), "w") as handle:
            handle.write(NEW.reveal())
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
        "magi_rs_bootstrap.BinaryBootstrap",
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
    # [fix 2] a same-volume rename does not re-propagate inheritance: <id> is hardened again after the move,
    # so its owner-only inheritable grant reaches the moved DB and magi.toml.
    pre_move_verify = guard.calls.index(("verify", report.backup_dir, True))
    assert ("harden", report.backup_dir, True) in guard.calls[pre_move_verify + 1 :]


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


def test_a_backup_dir_that_cannot_be_hardened_names_the_guards_reason(tmp_path):
    # [Loop1 M-5] the key-free one-line guard message reaches the operator, not only the class name.
    work = _project(tmp_path)
    rotation, _ = _rotation(tmp_path, guard=_FailingOnBackupIdGuard())
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert "cannot harden" in str(exc.value) and OLD.reveal() not in str(exc.value)


def test_a_backup_that_cannot_be_rehardened_names_the_guards_reason(tmp_path):
    work = _project(tmp_path)
    rotation, _ = _rotation(tmp_path, guard=_FailingOnBackupGuard())
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert "cannot harden" in str(exc.value) and OLD.reveal() not in str(exc.value)


def test_a_magi_dir_the_rotation_did_not_create_is_never_reported_as_rotated(tmp_path):
    # [Loop1 M-6] a concurrent run recreated .magi/ between the move and init: init answers "already
    # exists" and returns normally, but the new .magi/ has no key of this rotation (REQ-13g.2).
    work = _project(tmp_path)

    def _init(workdir, store):
        os.makedirs(os.path.join(workdir, ".magi"))
        open(os.path.join(workdir, ".magi", "magi.toml"), "w").write("someone else's")
        return ".magi/ already exists: left untouched"

    rotation, _ = _rotation(tmp_path, init=_init)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    backups = os.listdir(tmp_path / "rotated")
    assert len(backups) == 1
    assert (tmp_path / "rotated" / backups[0] / ".magi" / "magi.toml").read_text() == "old"
    assert (work / ".magi" / "magi.toml").read_text() == "someone else's"  # not ours to delete
    message = str(exc.value)
    assert backups[0] in message and "move it back" in message
    assert "\n" not in message.strip() and OLD.reveal() not in message


def test_a_new_key_that_does_not_verify_stops_the_rotation(tmp_path):
    # [Loop1 M-6] presence AND owner-only: a new key readable by others is not a completed rotation.
    work = _project(tmp_path)

    class _NewKeyExposedGuard(_NoopGuard):
        def verify(self, path, directory):
            if str(work) in path and not directory:
                raise KeyExposedError(f"{path} grants access to S-1-5-11 (DACL protected: no)")

    rotation, _ = _rotation(tmp_path, guard=_NewKeyExposedGuard())
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert os.listdir(tmp_path / "rotated")[0] in message and "S-1-5-11" in message
    assert OLD.reveal() not in message and NEW.reveal() not in message


def _project_with_db(tmp_path):
    """The rotation's project plus the memory database and a nested directory."""
    work = _project(tmp_path)
    (work / ".magi" / ".magi-rs-memory.db").write_bytes(b"db")
    (work / ".magi" / ".magi-rs-memory.db-wal").write_bytes(b"wal")
    (work / ".magi" / "index").mkdir()
    (work / ".magi" / "index" / "chunks.bin").write_bytes(b"chunks")
    return work


def test_every_path_of_the_moved_backup_is_hardened_and_verified(tmp_path):
    # [W3] REQ-13g.1: a same-volume rename keeps each child's own DACL, so no path relies on inheritance.
    work = _project_with_db(tmp_path)
    expected = []
    for current, dirs, files in os.walk(work / ".magi"):
        relative = os.path.relpath(current, work)
        expected.append((relative, True))
        expected.extend((os.path.join(relative, name), False) for name in files)
    guard = _RecordingGuard()
    rotation, _ = _rotation(tmp_path, guard=guard)
    report = rotation.rotate(str(work), assume_yes=True, is_tty=False)
    for relative, directory in expected:
        path = os.path.normpath(os.path.join(report.backup_dir, relative))
        assert ("harden", path, directory) in guard.calls, path
        assert ("verify", path, directory) in guard.calls, path


def test_a_backup_database_that_does_not_verify_stops_the_rotation_and_names_the_backup(tmp_path):
    # [W3] a guard that cannot restrict the moved DB: "may be exposed", no new .magi/, the backup kept.
    work = _project_with_db(tmp_path)

    class _DbExposedGuard(_NoopGuard):
        def verify(self, path, directory):
            if os.path.basename(path) == ".magi-rs-memory.db" and "rotated" in path:
                raise KeyExposedError(f"{path} grants access to S-1-5-11 (DACL protected: no)")

    rotation, _ = _rotation(tmp_path, guard=_DbExposedGuard())
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    backups = os.listdir(tmp_path / "rotated")
    assert len(backups) == 1
    assert (
        tmp_path / "rotated" / backups[0] / ".magi" / ".magi-rs-memory.db"
    ).read_bytes() == b"db"
    assert not (work / ".magi").exists()
    message = str(exc.value)
    assert backups[0] in message and "may be exposed" in message and "S-1-5-11" in message
    assert "\n" not in message.strip() and OLD.reveal() not in message


def _link_victim_into_magi(tmp_path, work, kind):
    """Plant a link to a path outside the project inside ``.magi/``; skip where it cannot be made."""
    victim = tmp_path / "victim"
    if kind == "file-symlink":
        victim.write_text("outside", encoding="utf-8")
        try:
            os.symlink(victim, work / ".magi" / "planted")
        except (OSError, NotImplementedError) as error:
            pytest.skip(f"symlinks are not permitted here: {error}")
    else:  # a directory junction: a Windows reparse point that needs no privilege
        if sys.platform != "win32":
            pytest.skip("junctions are Windows-only")
        import _winapi

        victim.mkdir()
        _winapi.CreateJunction(str(victim), str(work / ".magi" / "index"))
    return victim


def _require_links(tmp_path, kind):
    """Skip where a link of ``kind`` cannot be made (an unprivileged Windows account cannot symlink)."""
    if kind == "junction":
        if sys.platform != "win32":
            pytest.skip("junctions are Windows-only")
        return
    probe = tmp_path / "probe-link"
    try:
        os.symlink(tmp_path, probe)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks are not permitted here: {error}")
    probe.unlink()


@pytest.mark.parametrize("kind", ["file-symlink", "junction"])
def test_a_link_inside_the_backup_stops_the_rotation_and_its_target_is_never_touched(
    tmp_path, kind, monkeypatch
):
    # [W3 addition] hardening a link would chmod/icacls its TARGET, outside the backup: refuse, never follow.
    # [R3] a link in the SOURCE is now refused before the move, so the link appears during the move here,
    # which keeps the post-move walk under test.
    work = _project(tmp_path)
    _require_links(tmp_path, kind)
    real_move = shutil.move
    planted_victims = []

    def _move_then_plant(src, dst):
        real_move(src, dst)
        planted_victims.append(_link_victim_into_magi(tmp_path, Path(dst).parent, kind))

    monkeypatch.setattr("magi_key_rotation.shutil.move", _move_then_plant)
    guard = _RecordingGuard()
    rotation, _ = _rotation(tmp_path, guard=guard)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    (victim,) = planted_victims
    backups = os.listdir(tmp_path / "rotated")
    assert len(backups) == 1 and not (work / ".magi").exists()
    planted = "planted" if kind == "file-symlink" else "index"
    touched = {os.path.normcase(os.path.realpath(path)) for _, path, _ in guard.calls}
    assert os.path.normcase(os.path.realpath(victim)) not in touched
    assert not any(os.path.basename(path) == planted for _, path, _ in guard.calls)
    message = str(exc.value)
    assert backups[0] in message and planted in message and "may be exposed" in message
    assert "\n" not in message.strip() and OLD.reveal() not in message


FOREIGN = MagiKey("foreign-key-FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF=")


def _recreate_magi(workdir, key):
    """Put a complete ``.magi/`` holding ``key`` at ``workdir``, as another process would."""
    secret = os.path.join(workdir, ".magi", ".secret")
    os.makedirs(secret)
    open(os.path.join(workdir, ".magi", "magi.toml"), "w").write("someone else's")
    open(os.path.join(secret, "magi.key"), "w", encoding="ascii").write(key.reveal())


@pytest.mark.parametrize("key", [OLD, FOREIGN], ids=["restored-old-tree", "concurrent-init"])
def test_a_magi_dir_that_appears_before_init_is_never_reported_as_rotated(tmp_path, key):
    # [W4] the real init refuses an existing .magi/ with an error; the rotation stops and deletes nothing.
    work = _project(tmp_path)

    def _init(workdir, store):
        _recreate_magi(workdir, key)
        return run_ollama_init(workdir, store)

    rotation, _ = _rotation(tmp_path, init=_init)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    backups = os.listdir(tmp_path / "rotated")
    assert len(backups) == 1
    assert (work / ".magi" / ".secret" / "magi.key").read_text(encoding="ascii") == key.reveal()
    message = str(exc.value)
    assert backups[0] in message and "\n" not in message.strip()
    assert OLD.reveal() not in message and FOREIGN.reveal() not in message


def test_a_new_magi_dir_holding_the_old_key_is_never_reported_as_rotated(tmp_path):
    # [W4] REQ-13g.2: a "new" .magi/ that holds the exposed key puts that key back in service.
    work = _project(tmp_path)

    def _init(workdir, store):
        _recreate_magi(workdir, OLD)
        return "initialised"

    rotation, _ = _rotation(tmp_path, init=_init)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert os.listdir(tmp_path / "rotated")[0] in message and "old key" in message
    assert (work / ".magi" / ".secret" / "magi.key").exists()  # left as it is, never deleted
    assert OLD.reveal() not in message and "\n" not in message.strip()


def test_a_failed_init_never_removes_a_magi_dir_holding_another_process_key(tmp_path):
    # [W5] the rotation's own cleanup honours the same rule as run_ollama_init's.
    work = _project(tmp_path)

    def _init(workdir, store):
        _recreate_magi(workdir, FOREIGN)
        raise OllamaInitError(
            "init died", leftover=InitLeftover.FOREIGN_KEPT
        )  # [S4] as the real init reports

    rotation, _ = _rotation(tmp_path, init=_init)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert (work / ".magi" / ".secret" / "magi.key").read_text(encoding="ascii") == FOREIGN.reveal()
    message = str(exc.value)
    assert os.listdir(tmp_path / "rotated")[0] in message and "another process" in message
    assert "\n" not in message.strip() and FOREIGN.reveal() not in message


class _InterruptingGuard(_NoopGuard):
    """Raises KeyboardInterrupt on the first harden of a path inside the moved backup."""

    def harden(self, path, directory):
        parts = os.path.normpath(path).split(os.sep)
        if "rotated" in parts and ".magi" in parts[parts.index("rotated") :]:
            raise KeyboardInterrupt


class _InterruptingVerifyGuard(_NoopGuard):
    """Raises KeyboardInterrupt when the new key at the project root is verified."""

    def __init__(self, work):
        self.work = str(work)

    def verify(self, path, directory):
        if path.startswith(self.work) and not directory:
            raise KeyboardInterrupt


def _interrupt(*args, **kwargs):
    raise KeyboardInterrupt


@pytest.mark.parametrize("step", ["reharden", "init", "verify", "vault-ls"])
def test_an_interrupt_after_the_move_ends_as_one_line_naming_the_backup(tmp_path, step):
    # [W1] REQ-13h: a Ctrl-C after the move never leaves the operator without the backup's location.
    work = _project(tmp_path)
    extra = {
        "reharden": {"guard": _InterruptingGuard()},
        "init": {"init": _interrupt},
        "verify": {"guard": _InterruptingVerifyGuard(work)},
        "vault-ls": {"vault_ls": _interrupt},
    }[step]
    rotation, _ = _rotation(tmp_path, **extra)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    backups = os.listdir(tmp_path / "rotated")
    assert len(backups) == 1
    assert (tmp_path / "rotated" / backups[0] / ".magi" / "magi.toml").read_text() == "old"
    message = str(exc.value)
    assert os.path.join(backups[0], ".magi") in message
    assert isinstance(exc.value.__cause__, KeyboardInterrupt)
    assert "\n" not in message.strip() and OLD.reveal() not in message


def test_an_interrupted_init_leaves_no_half_made_magi_dir(tmp_path):
    # [W1] the described safe state: the old .magi/ in the backup, none at the root, "move it back".
    work = _project(tmp_path)

    def _init(workdir, store):
        os.makedirs(os.path.join(workdir, ".magi"))
        raise KeyboardInterrupt

    rotation, _ = _rotation(tmp_path, init=_init)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert not (work / ".magi").exists() and "move it back" in str(exc.value)


def test_an_interrupted_move_with_the_source_complete_removes_the_partial_copy(
    tmp_path, monkeypatch
):
    # [m-1] a Ctrl-C during a cross-volume copytree: same contract as a failed move, one line, chained.
    work = _project(tmp_path)

    def _move(src, dst):
        os.makedirs(os.path.join(dst, ".secret"))
        shutil.copy2(
            os.path.join(src, ".secret", "magi.key"), os.path.join(dst, ".secret", "magi.key")
        )
        raise KeyboardInterrupt

    monkeypatch.setattr("magi_key_rotation.shutil.move", _move)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert (work / ".magi" / "magi.toml").read_text() == "old"
    assert not (tmp_path / "rotated").exists()  # the partial copy and the created root are gone
    message = str(exc.value)
    assert "is unchanged" in message and "KeyboardInterrupt" in message
    assert isinstance(exc.value.__cause__, KeyboardInterrupt)
    assert "\n" not in message.strip() and OLD.reveal() not in message


def test_an_interrupted_move_after_source_files_went_keeps_both_copies_and_names_them(
    tmp_path, monkeypatch
):
    # [m-1] the interrupt lands in the delete half of a cross-volume move: nothing is deleted, both are named.
    work = _project(tmp_path)

    def _move(src, dst):
        shutil.copytree(src, dst)
        os.remove(os.path.join(src, "magi.toml"))
        raise KeyboardInterrupt

    monkeypatch.setattr("magi_key_rotation.shutil.move", _move)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    (backup,) = os.listdir(tmp_path / "rotated")
    full = tmp_path / "rotated" / backup / ".magi"
    assert (full / "magi.toml").read_text() == "old" and (full / ".secret" / "magi.key").exists()
    message = str(exc.value)
    assert (
        backup in message and str(work / ".magi") in message and "both copies were kept" in message
    )
    assert isinstance(exc.value.__cause__, KeyboardInterrupt)
    assert "\n" not in message.strip() and OLD.reveal() not in message


def test_an_init_interrupted_after_writing_its_key_is_never_blamed_on_another_process(tmp_path):
    # [m-3] the key at the root is the one this rotation's init just wrote: kept, and said so truthfully.
    work = _project(tmp_path)

    def _init(workdir, store):
        os.makedirs(os.path.join(workdir, ".magi"))
        store.persist(workdir, store.new_key())
        raise KeyboardInterrupt

    rotation, _ = _rotation(tmp_path, init=_init)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert (work / ".magi" / ".secret" / "magi.key").exists()
    message = str(exc.value)
    assert "another process" not in message
    assert "interrupted after" in message and "key was written" in message
    assert os.listdir(tmp_path / "rotated")[0] in message
    assert "\n" not in message.strip() and OLD.reveal() not in message


def test_a_new_key_that_could_not_be_hardened_is_never_reported_as_an_interruption(
    tmp_path, monkeypatch
):
    # [R2] persist wrote the key, hardening failed and its discard failed too: the key may be readable by
    # others, which is a different state (and a different remedy) from an interrupt.
    work = _project(tmp_path)
    new_key_file = os.path.join(str(work), ".magi", ".secret", "magi.key")

    class _NewKeyUnhardenableGuard(_NoopGuard):
        def harden(self, path, directory):
            if os.path.normcase(path) == os.path.normcase(new_key_file):
                raise KeyHardeningError(f"cannot harden {path}")

    real_remove = os.remove

    def _remove(path, *args, **kwargs):
        if os.path.normcase(str(path)) == os.path.normcase(new_key_file):
            raise OSError("file in use")
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr("magi_key.os.remove", _remove)

    def _init(workdir, store):
        try:
            store.persist(workdir, store.new_key())
        except KeyHardeningError as error:  # [S4] as run_ollama_init reports a kept, unhardened key
            raise OllamaInitError(
                str(error), leftover=InitLeftover.HARDENING_FAILED_KEPT
            ) from error
        return "initialised"

    rotation, _ = _rotation(tmp_path, init=_init, guard=_NewKeyUnhardenableGuard())
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert "interrupted" not in message
    assert "could not be hardened" in message and "may be readable by others" in message
    assert os.listdir(tmp_path / "rotated")[0] in message and "\n" not in message.strip()


def test_an_init_error_echoing_the_new_key_is_redacted(tmp_path):
    # [I-1] the keys this rotation persisted are redacted from the failure detail, not only the old one.
    work = _project(tmp_path)
    written = []

    def _init(workdir, store):
        key = store.new_key()
        written.append(key)
        store.persist(workdir, key)
        raise RuntimeError(f"init echoed {key.reveal()}")

    rotation, _ = _rotation(tmp_path, init=_init)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert written[0].reveal() not in message and "<redacted>" in message


class _NewKeyFileExposedGuard(_NoopGuard):
    """Reports the key file at the project root as exposed while ``armed``; everything else verifies."""

    def __init__(self, work):
        self.key_file = os.path.normcase(os.path.join(str(work), ".magi", ".secret", "magi.key"))
        self.armed = True

    def verify(self, path, directory):
        if self.armed and os.path.normcase(path) == self.key_file:
            raise KeyExposedError(f"{path} grants access to S-1-5-11 (DACL protected: no)")


def test_an_unverifiable_key_this_rotation_wrote_is_owned_and_the_backup_stays_put(tmp_path):
    # [R5] the recorded store proves the key on disk is this rotation's: never deny authorship, and never send
    # the operator back to the exposed backup.
    work = _project(tmp_path)

    guard = _NewKeyFileExposedGuard(work)

    def _init(workdir, store):
        guard.armed = (
            False  # persist's own verification passes; the rotation's later check does not
        )
        store.persist(workdir, store.new_key())
        guard.armed = True
        return "initialised"

    rotation, _ = _rotation(tmp_path, init=_init, guard=guard)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert "created by this rotation" in message and "could not be verified" in message
    assert "S-1-5-11" in message and "do not move the exposed backup back" in message
    # [n-4] the action that verifies the key again is named: any --ollama command runs that preflight.
    assert "run any --ollama command" in message and "preflight verifies the key" in message
    assert "may not have created it" not in message and "\n" not in message.strip()


def test_an_unverifiable_key_this_rotation_did_not_write_keeps_the_may_not_have_created_wording(
    tmp_path,
):
    # [R5] the other branch: a key the recording store never persisted is not claimed.
    work = _project(tmp_path)

    def _init(workdir, store):
        _recreate_magi(workdir, FOREIGN)
        return "initialised"

    rotation, _ = _rotation(tmp_path, init=_init, guard=_NewKeyFileExposedGuard(work))
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert "may not have created it" in message and "created by this rotation" not in message


def test_an_old_key_read_failure_without_a_message_still_names_its_cause(tmp_path):
    # [I-4] the pre-move steps render errors like every other step: one line, the class name when empty.
    work = _project(tmp_path)

    class _SilentlyUnreadable(_Store):
        def read_unverified(self, path):
            raise KeyMissingError()

    rotation = KeyRotation(
        _SilentlyUnreadable(), lambda *a: "", lambda *a: [], str(tmp_path / "rotated")
    )
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert "(KeyMissingError)" in str(exc.value) and "nothing was moved" in str(exc.value)


def test_a_backup_dir_creation_failure_without_a_message_still_names_its_cause(
    tmp_path, monkeypatch
):
    work = _project(tmp_path)

    def _makedirs(*args, **kwargs):
        raise OSError()

    monkeypatch.setattr("magi_key_rotation.os.makedirs", _makedirs)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert "(OSError)" in str(exc.value) and "nothing was moved" in str(exc.value)


@pytest.mark.parametrize("kind", ["file-symlink", "junction"])
def test_a_link_inside_the_source_magi_dir_stops_the_rotation_before_anything_moves(
    tmp_path, kind, monkeypatch
):
    # [R3] across volumes shutil.move copies with symlinks=False and would FOLLOW the link before any guard ran.
    work = _project(tmp_path)
    _require_links(tmp_path, kind)
    victim = _link_victim_into_magi(tmp_path, work, kind)
    moved = []
    monkeypatch.setattr("magi_key_rotation.shutil.move", lambda *a: moved.append(a))
    guard = _RecordingGuard()
    rotation, _ = _rotation(tmp_path, guard=guard)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert moved == [] and (work / ".magi" / "magi.toml").read_text() == "old"
    assert not (tmp_path / "rotated").exists() or os.listdir(tmp_path / "rotated") == []
    assert not any(
        os.path.normcase(os.path.realpath(p)) == os.path.normcase(os.path.realpath(victim))
        for _, p, _ in guard.calls
    )
    message = str(exc.value)
    planted = "planted" if kind == "file-symlink" else "index"
    assert planted in message and "nothing was moved" in message and "\n" not in message.strip()


def test_every_backup_directory_the_rotation_creates_is_restricted_and_pre_existing_ones_are_not(
    tmp_path,
):
    # [R3] os.makedirs(mode=...) restricts only the leaf: every parent the rotation creates is hardened and
    # verified too, and a parent that already existed is never touched.
    work = _project(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    root = state / "magi" / "rotated"
    guard = _RecordingGuard()
    rotation = KeyRotation(_Store(guard), lambda *a: "", lambda *a: [], str(root))
    with pytest.raises(
        RotationFailedError
    ):  # the no-op init leaves no new key; the backup is what is asserted
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    for created in (state / "magi", root):
        assert ("harden", str(created), True) in guard.calls, created
        assert ("verify", str(created), True) in guard.calls, created
    assert not any(
        os.path.normcase(path) == os.path.normcase(str(state)) for _, path, _ in guard.calls
    )


def test_a_relative_backup_root_is_refused_before_anything_happens(tmp_path):
    # [R3] a relative root would resolve against whatever the cwd is: refused before the prompt or any move.
    work = _project(tmp_path)
    asked = []
    rotation = KeyRotation(
        _Store(), lambda *a: "", lambda *a: [], os.path.join("relative", "rotated")
    )
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(
            str(work), assume_yes=False, is_tty=True, confirm=lambda p: asked.append(p) or "y"
        )
    assert asked == [] and (work / ".magi" / "magi.toml").read_text() == "old"
    assert "absolute" in str(exc.value)


def _real_init_whose_cleanup_fails(tmp_path, monkeypatch, work, failure):
    """The REAL run_ollama_init: the child succeeds, hardening the new key raises ``failure``, and neither
    the key's discard nor the removal of the new .magi/ succeeds (so the key stays on disk)."""
    new_key_file = os.path.normcase(os.path.join(str(work), ".magi", ".secret", "magi.key"))
    new_magi_dir = os.path.normcase(os.path.join(str(work), ".magi"))

    def _run(argv, **kwargs):
        os.makedirs(os.path.join(argv[argv.index("-w") + 1], ".magi"))
        return SimpleNamespace(returncode=0, stdout="initialised", stderr="")

    monkeypatch.setattr("ollama_init.subprocess.run", _run)
    monkeypatch.setattr(
        "magi_rs_bootstrap.BinaryBootstrap",
        lambda *a, **k: SimpleNamespace(
            ensure=lambda: "/bin/magi-rs", verify=lambda: "/bin/magi-rs"
        ),
    )
    real_remove, real_rmtree = os.remove, shutil.rmtree

    def _remove(path, *args, **kwargs):
        if os.path.normcase(str(path)) == new_key_file:
            raise OSError("file in use")
        return real_remove(path, *args, **kwargs)

    def _rmtree(path, *args, **kwargs):
        if os.path.normcase(str(path)) == new_magi_dir:
            raise OSError("directory in use")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("magi_key.os.remove", _remove)
    monkeypatch.setattr("ollama_init.shutil.rmtree", _rmtree)

    class _FailingNewKeyGuard(_NoopGuard):
        def harden(self, path, directory):
            if os.path.normcase(path) == new_key_file:
                raise failure

    return _FailingNewKeyGuard()


def test_a_hardening_failure_inside_the_real_init_keeps_the_hardening_wording(
    tmp_path, monkeypatch
):
    # [n-1] run_ollama_init folds its cleanup note into an OllamaInitError chained from the KeyHardeningError:
    # the rotation still says the key may be readable by others.
    work = _project(tmp_path)
    guard = _real_init_whose_cleanup_fails(
        tmp_path, monkeypatch, work, KeyHardeningError("cannot harden")
    )
    rotation, _ = _rotation(tmp_path, init=run_ollama_init, guard=guard)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert "may be readable by others" in message and "interrupted" not in message


def test_an_interrupt_inside_the_real_init_reaches_the_rotation_as_an_interrupt(
    tmp_path, monkeypatch
):
    # [n-1] a KeyboardInterrupt is never turned into OllamaInitError on the way out of run_ollama_init.
    work = _project(tmp_path)
    guard = _real_init_whose_cleanup_fails(tmp_path, monkeypatch, work, KeyboardInterrupt())
    rotation, _ = _rotation(tmp_path, init=run_ollama_init, guard=guard)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert isinstance(exc.value.__cause__, KeyboardInterrupt)
    assert "interrupted after its key was written" in str(exc.value)


_LEFTOVER_WORDING = {
    InitLeftover.KEY_WRITTEN_KEPT: "initialisation failed after its key was written",
    InitLeftover.HARDENING_FAILED_KEPT: "may be readable by others",
    InitLeftover.FOREIGN_KEPT: "another process",
    InitLeftover.HALF_MADE_KEPT: "could not be removed",
}


@pytest.mark.parametrize("leftover", list(InitLeftover))
def test_the_rotation_words_the_failure_by_the_leftover_init_reported(tmp_path, leftover):
    # [S4] the rotation trusts init's typed report; it never inspects the disk to guess ownership.
    work = _project(tmp_path)

    def _init(workdir, store):
        raise OllamaInitError("init died", leftover=leftover)

    rotation, _ = _rotation(tmp_path, init=_init)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    for other, wording in _LEFTOVER_WORDING.items():
        assert (wording in message) is (other is leftover), (leftover, other)
    assert os.listdir(tmp_path / "rotated")[0] in message and "\n" not in message.strip()


def test_a_non_init_error_from_the_injected_init_gets_the_generic_wording(tmp_path):
    # [S4] no guessing: an error that is not init's typed report never claims who wrote what.
    work = _project(tmp_path)

    def _init(workdir, store):
        _recreate_magi(workdir, FOREIGN)
        raise RuntimeError("init died")

    rotation, _ = _rotation(tmp_path, init=_init)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    for wording in _LEFTOVER_WORDING.values():
        assert wording not in message
    assert "left as it is" in message and (work / ".magi" / ".secret" / "magi.key").exists()


def test_a_move_that_lost_the_whole_source_says_the_backup_holds_the_complete_copy(
    tmp_path, monkeypatch
):
    # [T1] the copy finishes before the delete starts: with no source left, the backup IS the complete copy.
    work = _project(tmp_path)

    def _move(src, dst):
        shutil.copytree(src, dst)
        shutil.rmtree(src)
        raise OSError("interrupted after the delete")

    monkeypatch.setattr("magi_key_rotation.shutil.move", _move)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert "the source is gone" in message and "holds the complete copy" in message
    assert "compare" not in message and os.listdir(tmp_path / "rotated")[0] in message


def test_a_move_that_lost_part_of_the_source_asks_to_compare_both_copies(tmp_path, monkeypatch):
    work = _project(tmp_path)

    def _move(src, dst):
        shutil.copytree(src, dst)
        os.remove(os.path.join(src, "magi.toml"))
        raise OSError("file in use")

    monkeypatch.setattr("magi_key_rotation.shutil.move", _move)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert (
        "both copies were kept" in message
        and "compare" in message
        and "the source is gone" not in message
    )


def test_a_backup_hardening_failure_names_the_directory_that_failed(tmp_path):
    # [T5] with created parents in the loop, the message names the one that could not be restricted.
    work = _project(tmp_path)
    root = tmp_path / "state" / "rotated"

    class _FailsOnRoot(_NoopGuard):
        def harden(self, path, directory):
            if os.path.normcase(path) == os.path.normcase(str(root)):
                raise KeyHardeningError(f"cannot harden {path}")

    rotation = KeyRotation(_Store(_FailsOnRoot()), lambda *a: "", lambda *a: [], str(root))
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert f"cannot restrict the rotation backup directory {root} to its owner" in message
    assert "nothing was moved" in message and (work / ".magi" / "magi.toml").read_text() == "old"


def test_a_copy_whose_directory_cannot_be_listed_is_never_proven_complete(tmp_path, monkeypatch):
    # [U1] a directory the completeness walk cannot list must never be skipped: "cannot prove complete" keeps
    # both copies, because the partial backup may hold the only copy of what the source lost.
    work = _project(tmp_path)
    (work / ".magi" / "index").mkdir()
    (work / ".magi" / "index" / "chunks.bin").write_bytes(b"chunks")

    def _move(src, dst):
        shutil.copytree(src, dst)
        os.remove(os.path.join(src, "index", "chunks.bin"))  # the only copy now lives in the backup
        raise OSError("file in use")

    real_walk = os.walk

    def _walk(top, topdown=True, onerror=None, followlinks=False):
        for current, dirs, files in real_walk(top, topdown, onerror, followlinks):
            in_backup = "rotated" in os.path.normpath(current).split(os.sep)
            if in_backup and "index" in dirs:
                dirs.remove("index")  # what os.walk does with a directory it cannot list
                if onerror is not None:
                    onerror(
                        PermissionError(13, "Permission denied", os.path.join(current, "index"))
                    )
            yield current, dirs, files

    monkeypatch.setattr("magi_key_rotation.shutil.move", _move)
    monkeypatch.setattr("magi_key_rotation.os.walk", _walk)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    (backup,) = os.listdir(tmp_path / "rotated")
    assert (
        tmp_path / "rotated" / backup / ".magi" / "index" / "chunks.bin"
    ).read_bytes() == b"chunks"
    assert "both copies were kept" in str(exc.value)


def test_a_rotation_whose_vault_ls_prints_nothing_lists_the_whole_closed_list(tmp_path):
    # [U2] the real client over a child that exits 0 and prints nothing: never "no API keys to revoke".
    work = _project(tmp_path)

    def _vault_ls(backup_workdir, key):
        def runner(argv, **kwargs):
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        return VaultClient("/bin/magi-rs", backup_workdir, key, runner=runner).ls()

    rotation, _ = _rotation(tmp_path, vault_ls=_vault_ls)
    report = rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert report.names_to_revoke == list(VAULT_KEY_NAMES) and report.names_from_vault is False


class _KeyEchoingBackupGuard(_NoopGuard):
    """[V8] Fails to harden the moved backup with a message that echoes the old key."""

    def harden(self, path, directory):
        parts = os.path.normpath(path).split(os.sep)
        if "rotated" in parts and ".magi" in parts[parts.index("rotated") :]:
            raise KeyHardeningError(f"cannot harden {path} (key {OLD.reveal()})")


class _ArmedKeyEchoingGuard(_NoopGuard):
    """[V8] Once armed, every verify fails with a message that echoes the new key."""

    armed = False

    def verify(self, path, directory):
        if self.armed:
            raise KeyHardeningError(f"cannot verify {path} (key {NEW.reveal()})")


def test_a_guard_reason_that_echoes_the_old_key_is_redacted_in_the_message(tmp_path):
    # [V8] the redaction is structural: every step's reason goes through it, not only init's.
    work = _project(tmp_path)
    rotation, _ = _rotation(tmp_path, guard=_KeyEchoingBackupGuard())
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert OLD.reveal() not in message and "<redacted>" in message


def test_a_guard_reason_that_echoes_the_new_key_is_redacted_in_the_message(tmp_path):
    # [V8] a key this rotation persisted is redacted from the reasons of the later steps too.
    work = _project(tmp_path)
    guard = _ArmedKeyEchoingGuard()

    def _init(workdir, store):
        store.persist(workdir, NEW)
        guard.armed = True
        return "initialised"

    rotation, _ = _rotation(tmp_path, init=_init, guard=guard)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert NEW.reveal() not in message and "<redacted>" in message


class _InterruptedOnBackupIdGuard(_NoopGuard):
    """[V3] Raises KeyboardInterrupt while hardening the backup directory ``<backup>/<id>``."""

    def harden(self, path, directory):
        parts = os.path.normpath(path).split(os.sep)
        if len(parts) >= 2 and parts[-2] == "rotated":
            raise KeyboardInterrupt


def test_an_interrupt_while_restricting_the_backup_dir_ends_as_the_one_line_failure(tmp_path):
    # [V3] before the move an interrupt is handled like any failure: the empty dir goes, nothing moved.
    work = _project(tmp_path)
    rotation, _ = _rotation(tmp_path, guard=_InterruptedOnBackupIdGuard())
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert isinstance(exc.value.__cause__, KeyboardInterrupt)
    assert "nothing was moved" in message and "\n" not in message.strip()
    assert (work / ".magi" / "magi.toml").read_text() == "old"
    assert not (tmp_path / "rotated").exists()  # [m-3] the parents the call created go too


def test_a_move_that_failed_before_creating_the_copy_reports_the_source_unchanged(
    tmp_path, monkeypatch
):
    # [m-1] nothing was copied, so the source is trivially complete: say "unchanged", drop the empty dir.
    work = _project(tmp_path)

    def _move(src, dst):
        raise OSError("quota exceeded")

    monkeypatch.setattr("magi_key_rotation.shutil.move", _move)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert "is unchanged" in message and "both copies" not in message
    assert not (tmp_path / "rotated").exists()  # [X6] the created root goes with the empty backup
    assert (work / ".magi" / "magi.toml").read_text() == "old"


def test_the_rotation_refuses_any_reparse_point_and_names_what_it_is(tmp_path, monkeypatch):
    # [m-2] a cloud-sync placeholder is refused too (data safety), without being called a link.
    from magi_key_rotation import _LinkInBackupError

    magi = tmp_path / "id" / ".magi"
    magi.mkdir(parents=True)
    status = SimpleNamespace(st_mode=0o40700, st_file_attributes=0x400, st_reparse_tag=0x9000601A)
    monkeypatch.setattr("magi_key_permissions.os.lstat", lambda path: status)
    with pytest.raises(_LinkInBackupError) as exc:
        KeyRotation._backup_entries(str(magi))
    message = str(exc.value)
    assert ".magi is a reparse point" in message and "is a link" not in message
    assert "\n" not in message


class _InterruptedOnParentGuard(_NoopGuard):
    """[m-3] Raises KeyboardInterrupt while hardening the backup root the rotation created."""

    def harden(self, path, directory):
        if os.path.basename(os.path.normpath(path)) == "rotated":
            raise KeyboardInterrupt


def test_an_interrupt_hardening_a_created_parent_removes_the_dirs_it_created(tmp_path):
    # [m-3] one-line error chained from the interrupt; nothing the call created is left behind.
    work = _project(tmp_path)
    rotation, _ = _rotation(tmp_path, guard=_InterruptedOnParentGuard())
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert isinstance(exc.value.__cause__, KeyboardInterrupt)
    assert "nothing was moved" in str(exc.value) and "\n" not in str(exc.value).strip()
    assert not (tmp_path / "rotated").exists()
    assert (work / ".magi" / "magi.toml").read_text() == "old"


def test_an_interrupt_in_makedirs_ends_as_the_one_line_failure_and_cleans_up(tmp_path, monkeypatch):
    # [m-3] makedirs may have created the leaf and its parents before the interrupt arrives.
    work = _project(tmp_path)
    real_makedirs = os.makedirs

    def _makedirs(path, *args, **kwargs):
        real_makedirs(path, *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr("magi_key_rotation.os.makedirs", _makedirs)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert isinstance(exc.value.__cause__, KeyboardInterrupt)
    assert "nothing was moved" in str(exc.value) and "\n" not in str(exc.value).strip()
    assert not (tmp_path / "rotated").exists()
    assert (work / ".magi" / "magi.toml").read_text() == "old"


def test_a_created_dir_that_cannot_be_removed_is_named(tmp_path, monkeypatch):
    # [m-3] best effort: whatever stays is named so the operator can delete it.
    work = _project(tmp_path)

    def _rmdir(path):
        raise OSError("in use")

    monkeypatch.setattr("magi_key_rotation.os.rmdir", _rmdir)
    rotation, _ = _rotation(tmp_path, guard=_InterruptedOnParentGuard())
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert str(tmp_path / "rotated") in message and "delete" in message
    assert "\n" not in message.strip()


def test_a_target_that_cannot_be_inspected_is_never_proven_absent(tmp_path, monkeypatch):
    # [tighten-1] "cannot stat the copy" is not "nothing was copied": both copies are kept, nothing deleted.
    work = _project(tmp_path)
    holder = {}

    def _move(src, dst):
        shutil.copytree(src, dst)
        holder["target"] = dst
        raise OSError("file in use")

    real_lstat, real_lexists = os.lstat, os.path.lexists

    def _lstat(path, *args, **kwargs):
        if "target" in holder and os.fspath(path) == os.fspath(holder["target"]):
            raise PermissionError("denied")
        return real_lstat(path, *args, **kwargs)

    def _lexists(path):
        if "target" in holder and os.fspath(path) == os.fspath(holder["target"]):
            return False  # lexists swallows every OSError
        return real_lexists(path)

    monkeypatch.setattr("magi_key_rotation.shutil.move", _move)
    monkeypatch.setattr("magi_key_rotation.os.lstat", _lstat)
    monkeypatch.setattr("magi_key_rotation.os.path.lexists", _lexists)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert "both copies were kept" in str(exc.value)
    assert (Path(holder["target"]) / "magi.toml").exists() and (work / ".magi").exists()


def test_a_backup_leaf_that_already_existed_is_never_removed(tmp_path, monkeypatch):
    # [tighten-2] FileExistsError: another rotation made the leaf; this call did not create it.
    work = _project(tmp_path)
    leaf = {}
    real_makedirs = os.makedirs

    def _makedirs(path, *args, **kwargs):
        real_makedirs(path)
        leaf["path"] = path
        raise FileExistsError(path)

    monkeypatch.setattr("magi_key_rotation.os.makedirs", _makedirs)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert "nothing was moved" in str(exc.value)
    assert os.path.isdir(leaf["path"])


def test_a_backup_name_search_that_finds_every_candidate_taken_gives_up(tmp_path, monkeypatch):
    # [X4] the search is bounded; exhausted -> one line naming the root, nothing moved.
    work = _project(tmp_path)
    calls = []

    def _lexists(path):
        calls.append(path)
        assert len(calls) < 10_000, "the backup name search is unbounded"
        return True

    monkeypatch.setattr("magi_key_rotation.os.path.lexists", _lexists)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert str(tmp_path / "rotated") in message and "nothing was moved" in message
    assert "\n" not in message.strip()
    assert (work / ".magi" / "magi.toml").read_text() == "old"


def test_a_failed_move_with_an_unchanged_source_also_removes_the_parents_it_created(
    tmp_path, monkeypatch
):
    # [X6] the same cleanup as every other pre-move failure: the created backup root goes too.
    work = _project(tmp_path)

    def _move(src, dst):
        raise OSError("rename refused")

    monkeypatch.setattr("magi_key_rotation.shutil.move", _move)
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert "is unchanged" in str(exc.value) and "\n" not in str(exc.value).strip()
    assert not (tmp_path / "rotated").exists()
    assert (work / ".magi" / "magi.toml").read_text() == "old"


def test_an_unlistable_source_entry_is_not_told_to_remove_a_link(tmp_path, monkeypatch):
    # [X7] an OSError is not a link: the advice is to fix its cause.
    work = _project(tmp_path)

    def _entries(magi_dir):
        raise PermissionError("denied")

    monkeypatch.setattr(KeyRotation, "_backup_entries", staticmethod(_entries))
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    message = str(exc.value)
    assert "(denied): nothing was moved; fix the cause and retry" in message
    assert message.count("denied") == 1 and "remove the link" not in message
    assert "nothing was moved" in message and "\n" not in message.strip()


def test_a_link_in_the_source_is_told_to_remove_the_link(tmp_path, monkeypatch):
    from magi_key_rotation import _LinkInBackupError

    work = _project(tmp_path)

    def _entries(magi_dir):
        raise _LinkInBackupError("x is a link: replace it with a real file/directory")

    monkeypatch.setattr(KeyRotation, "_backup_entries", staticmethod(_entries))
    rotation, _ = _rotation(tmp_path)
    with pytest.raises(RotationFailedError) as exc:
        rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert "remove the link and retry" in str(exc.value)


def test_only_the_feature_owned_backup_dirs_are_hardened_never_the_platform_base(tmp_path):
    # [Z2] `<base>/magi/rotated/<id>`: the base is the platform's, so it is created but never hardened.
    work = _project(tmp_path)
    base = tmp_path / "state-base"
    guard = _RecordingGuard()
    store = _Store(guard)
    rotation = KeyRotation(
        store,
        lambda workdir, new_store: _write_new_magi(workdir),
        lambda workdir, key: [],
        backup_root=str(base / "magi" / "rotated"),
    )
    rotation.rotate(str(work), assume_yes=True, is_tty=False)
    hardened = {os.path.normcase(path) for call, path, _ in guard.calls if call == "harden"}
    assert os.path.normcase(str(base)) not in hardened
    assert os.path.normcase(str(base / "magi")) in hardened
    assert os.path.normcase(str(base / "magi" / "rotated")) in hardened
    assert base.is_dir()


def _write_new_magi(workdir):
    os.makedirs(os.path.join(workdir, ".magi", ".secret"))
    with open(os.path.join(workdir, ".magi", ".secret", "magi.key"), "w", encoding="ascii") as f:
        f.write(NEW.reveal())
    return "initialised"


def test_a_vault_listing_that_fails_reports_why_in_one_line(tmp_path):
    # [Z3] the operator learns which entry was not recognised (a name, never a value).
    work = _project(tmp_path)

    def _ls(*a):
        raise VaultCommandError(
            "vault ls listed 'STRAY_NAME', which is not a vault name this plugin uses"
        )

    rotation, _ = _rotation(tmp_path, vault_ls=_ls)
    report = rotation.rotate(str(work), assume_yes=True, is_tty=False)
    assert report.names_from_vault is False and "STRAY_NAME" in report.listing_detail
    assert "\n" not in report.listing_detail
