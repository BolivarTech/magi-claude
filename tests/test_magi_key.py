import base64
import os
import sys

import pytest

from magi_key import (
    KEY_BYTES,
    KeyExposedError,
    KeyHardeningError,
    KeyMissingError,
    KeyStore,
    KeyStoreError,
    REDACTION_MARKER,
    KeyUnverifiableError,
    MagiKey,
)
from synthesize import ValidationError


class _Guard:
    """Records calls; `exposed` paths fail verification."""

    def __init__(self, exposed=(), harden_fails=False):
        self.exposed = set(exposed)
        self.harden_fails = harden_fails
        self.hardened = []
        self.verified = []  # [CP2-M22]

    def harden(self, path, directory):
        if self.harden_fails:
            raise KeyHardeningError(f"cannot harden {path}")
        self.hardened.append((path, directory))

    def verify(self, path):
        self.verified.append(path)
        if path in self.exposed:
            raise KeyExposedError(f"{path} is readable by others")


def _store(**kw):
    return KeyStore(_Guard(**kw))


def test_a_new_key_is_base64_of_at_least_32_random_bytes():
    key = _store().new_key()
    assert len(base64.b64decode(key.reveal(), validate=True)) >= KEY_BYTES


def test_the_key_comes_from_the_csprng(monkeypatch):
    # [CP2-C31] REQ-09: secrets.token_bytes(KEY_BYTES), never a predictable source (a weak key looks identical).
    calls = []
    monkeypatch.setattr("magi_key.secrets.token_bytes", lambda n: calls.append(n) or b"\x01" * n)
    key = _store().new_key()
    assert calls == [KEY_BYTES] and base64.b64decode(key.reveal()) == b"\x01" * KEY_BYTES


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX umask")
def test_the_key_is_restricted_from_creation_not_after(tmp_path):
    # [CP2-C30] the guard here hardens nothing: the modes must already be right when the file appears (umask 022).
    old = os.umask(0o022)
    try:
        store = KeyStore(_Guard())
        path = store.persist(str(tmp_path), store.new_key())
    finally:
        os.umask(old)
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert os.stat(os.path.dirname(path)).st_mode & 0o777 == 0o700


def test_two_new_keys_differ():
    store = _store()
    assert store.new_key() != store.new_key()


def test_the_key_never_appears_in_repr_or_str():
    key = _store().new_key()
    assert key.reveal() not in repr(key)
    assert key.reveal() not in str(key)
    assert "redacted" in repr(key)


def test_persist_writes_the_key_under_magi_secret_and_hardens_dir_and_file(tmp_path):
    guard = _Guard()
    store = KeyStore(guard)
    key = store.new_key()
    path = store.persist(str(tmp_path), key)
    assert path == os.path.join(str(tmp_path), ".magi", ".secret", "magi.key")
    assert open(path, encoding="ascii").read() == key.reveal()
    assert (os.path.dirname(path), True) in guard.hardened
    assert (path, False) in guard.hardened
    # [CP2-M22] REQ-13 "verified after creating them"
    assert os.path.dirname(path) in guard.verified
    assert path in guard.verified


def test_a_verification_failure_at_persist_is_fatal_and_leaves_no_key_behind(tmp_path):
    # [CP2-M22] a hardened-but-unverified key never stays on disk.
    key_file = os.path.join(str(tmp_path), ".magi", ".secret", "magi.key")
    store = KeyStore(_Guard(exposed={key_file}))
    with pytest.raises(KeyHardeningError):
        store.persist(str(tmp_path), store.new_key())
    assert not os.path.exists(key_file)


def test_load_returns_exactly_what_read_unverified_reads(tmp_path, monkeypatch):
    # [CP2-M23] the only read goes through read_unverified (makes the never-reads-an-exposed-key test non-vacuous).
    store = _store()
    store.persist(str(tmp_path), store.new_key())
    sentinel = MagiKey("sentinel-value-GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG=")
    monkeypatch.setattr(KeyStore, "read_unverified", lambda self, p: sentinel)
    assert store.load(str(tmp_path)) is sentinel


def test_persist_never_overwrites_an_existing_key(tmp_path):
    store = _store()
    first = store.new_key()
    store.persist(str(tmp_path), first)
    with pytest.raises(FileExistsError):
        store.persist(str(tmp_path), store.new_key())
    assert store.load(str(tmp_path)) == first


def test_persist_refuses_an_existing_key_even_if_the_existence_check_is_fooled(
    tmp_path, monkeypatch
):
    # [CP2-C1] no check-then-write window: the refusal comes from the exclusive create.
    store = _store()
    first = store.new_key()
    store.persist(str(tmp_path), first)
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    with pytest.raises(FileExistsError):
        store.persist(str(tmp_path), store.new_key())
    monkeypatch.undo()
    assert store.load(str(tmp_path)) == first


def test_an_unreadable_key_is_missing_in_one_line_naming_the_path(tmp_path):
    # [CP2-C2] REQ-11 "unreadable": an OSError on read is a KeyMissingError, never a traceback.
    (tmp_path / ".magi" / ".secret" / "magi.key").mkdir(parents=True)
    with pytest.raises(KeyMissingError) as exc:
        _store().load(str(tmp_path))
    assert "magi.key" in str(exc.value) and "\n" not in str(exc.value)


def test_a_hardening_failure_is_fatal_and_leaves_no_key_behind(tmp_path):
    store = _store(harden_fails=True)
    with pytest.raises(KeyHardeningError):
        store.persist(str(tmp_path), store.new_key())
    assert not os.path.exists(os.path.join(str(tmp_path), ".magi", ".secret", "magi.key"))


def test_load_of_a_missing_key_names_the_path_and_the_regeneration(tmp_path):
    with pytest.raises(KeyMissingError) as exc:
        _store().load(str(tmp_path))
    message = str(exc.value)
    assert os.path.join(".magi", ".secret", "magi.key") in message
    assert "--ollama-init" in message
    # [PM-MS1-C4] a 7.x .magi/ exists and --ollama-init refuses it: the message must say to move it aside.
    assert "move" in message and "backup" in message


def test_load_of_an_empty_key_is_missing(tmp_path):
    secret = tmp_path / ".magi" / ".secret"
    secret.mkdir(parents=True)
    (secret / "magi.key").write_text("", encoding="ascii")
    with pytest.raises(KeyMissingError) as exc:
        _store().load(str(tmp_path))
    # --ollama-init leaves an existing .magi/ untouched: the message must say to move it aside first.
    message = str(exc.value)
    assert "--ollama-init" in message and "move" in message and "backup" in message


class _FileHardenFailingGuard(_Guard):
    """Hardens directories; fails on the key file, i.e. after the file was written."""

    def harden(self, path, directory):
        if not directory:
            raise KeyHardeningError(f"cannot harden {path}")
        super().harden(path, directory)


def test_a_failed_cleanup_never_masks_the_hardening_failure(tmp_path, monkeypatch):
    def _remove_fails(path):
        raise OSError("sharing violation")

    # [Loop1 M-1] the failure is put on the FILE harden: the directory is now hardened before the
    # file exists, and this test is about cleaning up a file that was written.
    store = KeyStore(_FileHardenFailingGuard())
    key_file = store.key_path(str(tmp_path))
    monkeypatch.setattr("magi_key.os.remove", _remove_fails)
    with pytest.raises(KeyHardeningError) as exc:
        store.persist(str(tmp_path), store.new_key())
    assert key_file in str(exc.value)
    assert "could not be removed" in str(exc.value)


def test_load_never_reads_an_exposed_key(tmp_path, monkeypatch):
    # [PM-MS1-C3] "before reading" is measured: the raw read must not run at all.
    store = _store()
    store.persist(str(tmp_path), store.new_key())
    path = store.key_path(str(tmp_path))
    reads = []
    monkeypatch.setattr(KeyStore, "read_unverified", lambda self, p: reads.append(p))
    with pytest.raises(KeyExposedError) as exc:
        KeyStore(_Guard(exposed={path})).load(str(tmp_path))
    assert reads == []
    assert "--ollama-rotate-key" in str(exc.value)


def test_an_exposed_secret_directory_is_also_exposed(tmp_path):
    store = _store()
    store.persist(str(tmp_path), store.new_key())
    secret_dir = os.path.dirname(store.key_path(str(tmp_path)))
    with pytest.raises(KeyExposedError):
        KeyStore(_Guard(exposed={secret_dir})).load(str(tmp_path))


def test_the_exposed_message_says_db_and_vault_must_be_treated_as_compromised(tmp_path):
    store = _store()
    store.persist(str(tmp_path), store.new_key())
    path = store.key_path(str(tmp_path))
    with pytest.raises(KeyExposedError) as exc:
        KeyStore(_Guard(exposed={path})).load(str(tmp_path))
    assert "compromised" in str(exc.value)
    assert path in str(exc.value)  # [CP2-M39] the operator must see which file is flagged
    # [CP2-M48] and is never told to "fix" the permissions and carry on with a possibly compromised key.
    assert not any(word in str(exc.value).lower() for word in ("chmod", "icacls", "restore"))


def test_no_key_error_message_ever_contains_the_key(tmp_path):
    store = _store()
    key = store.new_key()
    store.persist(str(tmp_path), key)
    path = store.key_path(str(tmp_path))
    with pytest.raises(KeyStoreError) as exc:
        KeyStore(_Guard(exposed={path})).load(str(tmp_path))
    assert key.reveal() not in str(exc.value)


def test_key_errors_are_siblings_of_validation_error():
    for error in (KeyMissingError, KeyExposedError, KeyHardeningError):
        assert not issubclass(error, ValidationError)
        assert issubclass(error, KeyStoreError)


class _RaisingGuard(_Guard):
    """Verification of the key file raises `error` (the guard's own one-line message)."""

    def __init__(self, error):
        super().__init__()
        self.error = error

    def verify(self, path):
        if path.endswith("magi.key"):
            raise self.error


def _load_with(tmp_path, error):
    store = _store()
    store.persist(str(tmp_path), store.new_key())
    with pytest.raises(KeyExposedError) as exc:
        KeyStore(_RaisingGuard(error)).load(str(tmp_path))
    return store.key_path(str(tmp_path)), exc.value


def test_an_exposed_key_message_carries_the_guards_detail(tmp_path):
    # [Loop1 I-1] the foreign SID and the protection reach the operator, not only __cause__.
    path, error = _load_with(
        tmp_path, KeyExposedError("/w/magi.key grants access to S-1-5-11 (DACL protected: no)")
    )
    message = str(error)
    assert not isinstance(error, KeyUnverifiableError)
    assert "S-1-5-11" in message and "DACL protected: no" in message
    assert "compromised" in message and "--ollama-rotate-key" in message and path in message
    assert "\n" not in message


def test_an_unverifiable_key_is_never_called_compromised(tmp_path):
    # [Loop1 I-1] a tooling failure is not an exposure: rotating would destroy the memory for nothing.
    cause = "cannot read the permissions of /w/magi.key: powershell.exe failed with exit code 1: Get-Acl"
    path, error = _load_with(tmp_path, KeyUnverifiableError(cause))
    message = str(error)
    assert isinstance(error, KeyUnverifiableError)
    assert "could not be verified" in message and "powershell.exe failed" in message
    assert path in message and "fix the cause" in message and "--ollama-rotate-key" in message
    assert "compromised" not in message and "\n" not in message


def test_an_unverifiable_key_is_a_fail_closed_sibling_of_validation_error():
    assert issubclass(KeyUnverifiableError, KeyExposedError)
    assert not issubclass(KeyUnverifiableError, ValidationError)


def test_redact_replaces_every_occurrence_of_the_key_with_the_one_marker():
    # [Loop1 I-2] the one redaction helper every child-output path shares.
    key = MagiKey("c2VjcmV0LWtleQ==")
    text = f"echo {key.reveal()} and again {key.reveal()}\n"
    assert key.redact(text) == f"echo {REDACTION_MARKER} and again {REDACTION_MARKER}\n"
    assert key.redact("nothing to hide") == "nothing to hide" and key.redact("") == ""


def test_redact_bytes_replaces_every_occurrence_of_the_key():
    key = MagiKey("c2VjcmV0LWtleQ==")
    data = b"x" + key.reveal().encode() + b"y" + key.reveal().encode()
    marker = REDACTION_MARKER.encode()
    assert key.redact_bytes(data) == b"x" + marker + b"y" + marker
    assert key.redact_bytes(b"") == b""


class _ExistenceRecordingGuard(_Guard):
    """Records, at each harden call, whether the key file already exists."""

    def __init__(self, key_file):
        super().__init__()
        self.key_file = key_file
        self.seen = []

    def harden(self, path, directory):
        self.seen.append((path, directory, os.path.exists(self.key_file)))


def test_the_secret_directory_is_hardened_before_the_key_file_exists(tmp_path):
    # [Loop1 M-1] os.makedirs(mode=0o700) is owner-only on Windows only on CPython with the CVE-2024-4030
    # fix; hardening the directory first makes the file inherit the owner-only grant on every build.
    key_file = KeyStore(_Guard()).key_path(str(tmp_path))
    guard = _ExistenceRecordingGuard(key_file)
    KeyStore(guard).persist(str(tmp_path), MagiKey("c2VjcmV0LWtleQ=="))
    assert guard.seen[0] == (os.path.dirname(key_file), True, False)
    # The later harden of directory and file stays.
    assert (os.path.dirname(key_file), True, True) in guard.seen
    assert (key_file, False, True) in guard.seen


def test_a_key_is_never_usable_as_a_dict_key_or_set_member():
    # [I-2] passes in Red by design (``__eq__`` without ``__hash__`` already disables hashing): pins that a
    # secret is never hashed, so it cannot become a dict key or a set member by accident.
    with pytest.raises(TypeError):
        hash(MagiKey("c2VjcmV0"))


def _names_the_whole_regeneration(message):
    """REQ-12's three steps: move .magi/ aside, --ollama-init, reload each API key with --ollama-vault-set."""
    return (
        "move the existing .magi/ aside" in message
        and "--ollama-init" in message
        and "--ollama-vault-set" in message
    )


def test_a_missing_key_names_the_whole_regeneration(tmp_path):
    # [S3] REQ-12's third step (reload each API key) is part of the recovery.
    class _Guard:
        def harden(self, path, directory):
            pass

        def verify(self, path):
            pass

    (tmp_path / ".magi" / ".secret").mkdir(parents=True)
    with pytest.raises(KeyMissingError) as exc:
        KeyStore(_Guard()).load(str(tmp_path))
    assert _names_the_whole_regeneration(str(exc.value))


def test_read_unverified_of_an_empty_file_raises_the_empty_key_error(tmp_path):
    # [X3] "exists but empty" has its own type, so callers never infer it from __cause__.
    from magi_key import EmptyKeyError

    key_file = tmp_path / "magi.key"
    key_file.write_text("  \n", encoding="ascii")
    with pytest.raises(EmptyKeyError) as exc:
        _store().read_unverified(str(key_file))
    assert isinstance(exc.value, KeyMissingError)


def test_read_unverified_of_an_unreadable_file_is_missing_but_not_empty(tmp_path):
    from magi_key import EmptyKeyError

    unreadable = tmp_path / "magi.key"
    unreadable.mkdir()
    with pytest.raises(KeyMissingError) as exc:
        _store().read_unverified(str(unreadable))
    assert not isinstance(exc.value, EmptyKeyError)


def test_comparing_a_key_that_cannot_be_encoded_never_raises_and_is_unequal():
    # [m-3] equality is part of every redaction/rotation path: it must not raise on odd text.
    odd = MagiKey("\u00f1" * 44)
    ascii_key = MagiKey("a" * 44)
    assert (odd == ascii_key) is False and (ascii_key == odd) is False
    assert (odd == MagiKey("\u00f1" * 44)) is False  # unencodable values never compare equal
    assert (ascii_key == MagiKey("a" * 44)) is True


def _junction_or_skip(link, target):
    """Make ``link`` a directory link to ``target`` (a junction on Windows); skip where impossible."""
    try:
        if sys.platform == "win32":
            import _winapi

            _winapi.CreateJunction(str(target), str(link))
        else:
            os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"cannot create a link here: {error}")


@pytest.mark.parametrize("linked", [".magi", ".magi/.secret"])
def test_persist_refuses_a_linked_key_directory_before_writing_anything(tmp_path, linked):
    # [L1] the write path never follows a link: the key would land wherever the link points.
    work = tmp_path / "work"
    outside = tmp_path / "outside"
    outside.mkdir()
    if linked == ".magi/.secret":
        (work / ".magi").mkdir(parents=True)
    else:
        work.mkdir()
    _junction_or_skip(work / linked, outside)
    store = _store()
    with pytest.raises(KeyHardeningError) as exc:
        store.persist(str(work), store.new_key())
    assert "is a link" in str(exc.value) and "\n" not in str(exc.value)
    assert os.listdir(outside) == []


def test_redact_bytes_with_an_unencodable_key_returns_the_input_unchanged():
    # [M7] like __eq__, redaction never raises on odd key text: nothing can echo what cannot be encoded.
    assert MagiKey("\u00f1" * 44).redact_bytes(b"some child output") == b"some child output"


_SLASHY_KEY = MagiKey("aB/cD+eF/gH+iJ=" + "k" * 29)


@pytest.mark.parametrize(
    "escape",
    [
        lambda v: v.replace("/", "\\/"),
        lambda v: v.replace("/", "\\u002f").replace("+", "\\u002b"),
        lambda v: v.replace("/", "\\u002F").replace("+", "\\u002B"),
    ],
    ids=["slash-escaped", "u-escaped-lower", "u-escaped-upper"],
)
def test_redaction_also_scrubs_the_json_escaped_forms_of_the_key(escape):
    # [N1] a child that emits the key inside JSON may escape "/" and "+": no key fragment may survive.
    value = _SLASHY_KEY.reveal()
    leaked = '{"detail": "' + escape(value) + '"}'
    scrubbed = _SLASHY_KEY.redact(leaked)
    assert "aB" not in scrubbed and "cD" not in scrubbed and REDACTION_MARKER in scrubbed
    as_bytes = _SLASHY_KEY.redact_bytes(leaked.encode("ascii")).decode("ascii")
    assert "aB" not in as_bytes and "cD" not in as_bytes and REDACTION_MARKER in as_bytes
    assert _SLASHY_KEY.redact("plain " + value) == "plain " + REDACTION_MARKER
