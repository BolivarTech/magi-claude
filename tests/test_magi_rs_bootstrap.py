# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-09-18
"""Tests for the magi-rs binary bootstrap (MS2 Tasks 2-5).

No test touches the network or the plugin's real ``bin/``: downloads are served
by an ``http.server`` on an ephemeral loopback port and every directory is a
``tmp_path``.
"""

import contextlib
import hashlib
import json
import http.server
import io
import os
import platform
import subprocess
import tarfile
import threading
import time
import zipfile
from types import SimpleNamespace
from unittest import mock

import pytest

from magi_rs_bootstrap import (
    _STALE_TEMPORARY_SECONDS,  # the cutoff the age tests backdate against
    BinaryBootstrap,
    DigestMismatchError,
    DownloadError,
    UnsafeArchiveError,
    UnsafeBinDirectoryError,
    VersionMismatchError,
    download_to,
    extract_executable,
    require_real_directory,
    sha256_of,
    sweep,
)
from magi_rs_pin import Asset, Pin

# --------------------------------------------------------------------------
# Block 1: download and digest verification
# --------------------------------------------------------------------------


class _Server:
    """Serves a fixed payload once, on an ephemeral loopback port."""

    def __init__(self, payload: bytes, status: int = 200, truncate: bool = False):
        body = payload[: len(payload) // 2] if truncate else payload

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server's contract
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if status == 200:
                    self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        self._httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}/asset"

    def __enter__(self) -> "_Server":
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def test_sha256_of_a_file_larger_than_one_chunk_matches_hashlib(tmp_path):
    payload = bytes(range(256)) * 1024  # 256 KiB: four chunks, not one
    target = tmp_path / "asset.bin"
    target.write_bytes(payload)
    assert sha256_of(str(target)) == hashlib.sha256(payload).hexdigest()


def test_a_download_whose_digest_matches_lands_at_the_destination(tmp_path):
    payload = b"pretend this is a release asset" * 100
    digest = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "asset.bin"
    with _Server(payload) as server:
        download_to(server.url, str(destination), digest)
    assert destination.read_bytes() == payload


def test_a_download_whose_digest_differs_is_deleted_and_names_both_digests(tmp_path):
    payload = b"tampered"
    wrong = "0" * 64
    destination = tmp_path / "asset.bin"
    with _Server(payload) as server:
        with pytest.raises(DigestMismatchError) as exc:
            download_to(server.url, str(destination), wrong)
    assert not destination.exists()  # BDD-6b: nothing partial survives
    message = str(exc.value)
    assert wrong in message
    assert hashlib.sha256(payload).hexdigest() in message


def test_an_http_error_aborts_naming_the_url_and_leaves_nothing_behind(tmp_path):
    destination = tmp_path / "asset.bin"
    with _Server(b"", status=404) as server:
        with pytest.raises(DownloadError) as exc:
            download_to(server.url, str(destination), "0" * 64)
        assert server.url in str(exc.value)
    assert not destination.exists()


def test_a_stream_cut_in_half_aborts_and_leaves_nothing_behind(tmp_path):
    payload = b"x" * 8192
    destination = tmp_path / "asset.bin"
    with _Server(payload, truncate=True) as server:
        with pytest.raises((DownloadError, DigestMismatchError)):
            download_to(server.url, str(destination), hashlib.sha256(payload).hexdigest())
    assert not destination.exists()


def test_an_unreachable_host_aborts_with_a_download_error(tmp_path):
    destination = tmp_path / "asset.bin"
    # Port 1 on loopback: refused immediately, no DNS, no wait.
    with pytest.raises(DownloadError):
        download_to("http://127.0.0.1:1/asset", str(destination), "0" * 64)
    assert not destination.exists()


def test_a_download_never_overwrites_a_file_already_at_its_destination(tmp_path):
    # The destination name is unique per process, so anything already there is
    # foreign -- and if it were a symlink, writing through it would land the
    # archive bytes outside bin/. Refusing is the only cross-platform answer:
    # O_NOFOLLOW does not exist on Windows.
    payload = b"asset"
    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"planted before the download")
    with _Server(payload) as server:
        with pytest.raises(DownloadError):
            download_to(server.url, str(destination), hashlib.sha256(payload).hexdigest())
    assert destination.read_bytes() == b"planted before the download"


# --------------------------------------------------------------------------
# Block 2: validated extraction -- traversal, bomb, layout, execute bit
# --------------------------------------------------------------------------


def _make_tar(path, members, *, mode=0o755):
    """Build a .tar.gz from (name, bytes) pairs, preserving mode."""
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = mode
            archive.addfile(info, io.BytesIO(payload))


def _make_tar_with_symlink(path, link_name, target):
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(link_name)
        info.type = tarfile.SYMTYPE
        info.linkname = target
        archive.addfile(info)


def _make_zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members:
            archive.writestr(name, payload)


@pytest.fixture
def low_ceilings(monkeypatch):
    """Shrink the bomb ceilings so the suite does not allocate 130 MiB a run.

    The ceilings are module constants, not parameters: production must not be
    able to lower them, and a test must not have to build a real 64 MiB bomb to
    prove the guard fires. Patching them keeps both properties.
    """
    size, ratio = 256 * 1024, 20
    monkeypatch.setattr("magi_rs_bootstrap._MAX_EXTRACTED_BYTES", size)
    monkeypatch.setattr("magi_rs_bootstrap._MAX_COMPRESSION_RATIO", ratio)
    return SimpleNamespace(size=size, ratio=ratio)


def test_a_flat_zip_yields_the_executable_at_the_destination(tmp_path):
    archive = tmp_path / "asset.zip"
    _make_zip(archive, [("magi-rs.exe", b"MZ binary"), ("README.md", b"docs")])
    destination = tmp_path / "out"
    destination.mkdir()
    installed = extract_executable(str(archive), "zip", str(destination), "magi-rs.exe")
    assert os.path.basename(installed) == "magi-rs.exe"
    assert open(installed, "rb").read() == b"MZ binary"


def test_a_nested_tarball_finds_the_executable_below_the_version_directory(tmp_path):
    # MS0 measured this: the four .tar.gz assets nest under
    # magi-rs-v<X.Y.Z>-<platform>/ -- the container name must NOT be hardcoded.
    archive = tmp_path / "asset.tar.gz"
    _make_tar(
        archive,
        [
            ("magi-rs-v0.19.1-linux-x86_64/magi-rs", b"ELF binary"),
            ("magi-rs-v0.19.1-linux-x86_64/LICENSE-MIT", b"license"),
        ],
    )
    destination = tmp_path / "out"
    destination.mkdir()
    installed = extract_executable(str(archive), "tar.gz", str(destination), "magi-rs")
    assert os.path.basename(installed) == "magi-rs"
    assert os.path.dirname(installed) == str(destination)  # moved UP, not left nested
    assert open(installed, "rb").read() == b"ELF binary"


def test_the_installed_executable_carries_the_execute_bit(tmp_path):
    archive = tmp_path / "asset.tar.gz"
    _make_tar(archive, [("magi-rs-v0.19.1-linux-x86_64/magi-rs", b"ELF")], mode=0o644)
    destination = tmp_path / "out"
    destination.mkdir()
    installed = extract_executable(str(archive), "tar.gz", str(destination), "magi-rs")
    if os.name != "nt":
        assert os.stat(installed).st_mode & 0o111  # set even though the member was 0o644


def test_a_member_escaping_upward_is_refused_and_nothing_is_written(tmp_path):
    archive = tmp_path / "asset.tar.gz"
    _make_tar(archive, [("../escaped", b"pwned"), ("magi-rs", b"ELF")])
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(UnsafeArchiveError):
        extract_executable(str(archive), "tar.gz", str(destination), "magi-rs")
    assert not (tmp_path / "escaped").exists()
    assert list(destination.iterdir()) == []


def test_a_member_with_an_absolute_path_is_refused(tmp_path):
    archive = tmp_path / "asset.tar.gz"
    _make_tar(archive, [("/tmp/escaped", b"pwned")])
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(UnsafeArchiveError):
        extract_executable(str(archive), "tar.gz", str(destination), "magi-rs")


def test_a_symlink_member_is_refused_even_when_it_points_inside(tmp_path):
    # A symlink pointing inside is still refused: a LATER member could then be
    # written THROUGH it to somewhere outside. The guard is on the member kind,
    # not on where this one happens to point.
    archive = tmp_path / "asset.tar.gz"
    _make_tar_with_symlink(archive, "magi-rs", "./harmless")
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(UnsafeArchiveError):
        extract_executable(str(archive), "tar.gz", str(destination), "magi-rs")


def test_an_archive_expanding_past_the_SIZE_ceiling_is_aborted(tmp_path, low_ceilings):
    # INCOMPRESSIBLE payload on purpose: random bytes barely compress, so the
    # ratio bound is never crossed and this test exercises _MAX_EXTRACTED_BYTES
    # and nothing else. A run of zeros would trip the ratio bound first and
    # this test would pass while the size bound was broken.
    archive = tmp_path / "bomb.tar.gz"
    _make_tar(archive, [("magi-rs", os.urandom(low_ceilings.size + 4096))])
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(UnsafeArchiveError) as exc:
        extract_executable(str(archive), "tar.gz", str(destination), "magi-rs")
    assert "size" in str(exc.value).lower()
    assert list(destination.iterdir()) == []  # the partial write is cleaned up


def test_an_archive_expanding_past_the_RATIO_ceiling_is_aborted(tmp_path, low_ceilings):
    # Highly compressible payload: it stays UNDER the size ceiling yet expands
    # far past the ratio bound. This is the attack the size ceiling alone
    # cannot see, which is why there are two bounds and not one.
    archive = tmp_path / "ratio.tar.gz"
    _make_tar(archive, [("magi-rs", b"\0" * (low_ceilings.size - 1024))])
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(UnsafeArchiveError) as exc:
        extract_executable(str(archive), "tar.gz", str(destination), "magi-rs")
    assert "ratio" in str(exc.value).lower()


def test_the_guard_never_reads_a_members_declared_size(tmp_path):
    # THE THIRD ATTEMPT AT THIS TEST, and the first honest one. The two before
    # it tried to build an archive whose header LIES and assert that the guard
    # stops it. Both were impossible, each for its own reason, and BOTH WERE
    # EXECUTED before being written down:
    #
    #   tar: a size of 1 over megabytes of payload makes `tarfile` fail while
    #        PARSING -- ReadError, not our guard.
    #   zip: `zipfile` caps reads at `file_size` and then raises
    #        `BadZipFile: Bad CRC-32` -- verified by running it.
    #
    # The lesson is the one this repo already paid for three times over the
    # verdict sentinel: a guarantee you have not executed is a guess wearing a
    # suit. And the honest reading is that the RATIO ceiling already stops the
    # attack -- an archive that lies about its size and expands is caught by
    # expansion, whatever its header says.
    #
    # What is left to establish is a property of the IMPLEMENTATION, not of
    # observable behaviour: that no ceiling is ever computed from attacker
    # data. That is checkable by reading the source, and this test reads it.
    # KNOWN LIMITATION, stated rather than hidden: this reads the source of ONE
    # function, so it is coupled to that name and blind to a ceiling computed
    # from header data somewhere else. It fails LOUDLY if the function is
    # renamed (AttributeError, not a silent pass), which is the property that
    # makes the coupling acceptable. It is a floor, not a proof -- the real
    # guarantee is the ratio ceiling, which the two tests above execute.
    import inspect

    import magi_rs_bootstrap

    source = inspect.getsource(magi_rs_bootstrap._guarded_copy)
    assert "member.size" not in source
    assert "file_size" not in source
    assert "len(chunk)" in source  # the count is over bytes ACTUALLY written


def test_an_archive_without_the_executable_is_refused_naming_it(tmp_path):
    archive = tmp_path / "asset.tar.gz"
    _make_tar(archive, [("magi-rs-v0.19.1-linux-x86_64/README.md", b"docs")])
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(UnsafeArchiveError) as exc:
        extract_executable(str(archive), "tar.gz", str(destination), "magi-rs")
    assert "magi-rs" in str(exc.value)


@pytest.mark.parametrize("archive_kind", ["zip", "tar.gz"])
def test_a_verified_file_that_is_not_an_archive_is_refused_as_unsafe(tmp_path, archive_kind):
    # A pin whose archive_sha256 matches bytes that are not an archive of the
    # declared kind must fail with the documented error, not with a raw
    # tarfile.ReadError / zipfile.BadZipFile that no caller is told to expect.
    archive = tmp_path / "asset.bin"
    archive.write_bytes(b"not an archive at all")
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(UnsafeArchiveError):
        extract_executable(str(archive), archive_kind, str(destination), "magi-rs")
    assert list(destination.iterdir()) == []


def test_an_archive_holding_the_executable_twice_is_refused(tmp_path):
    # Two candidates would mean choosing between them, and choosing is the
    # failure shape this project already paid for in the verdict sentinel.
    archive = tmp_path / "asset.zip"
    _make_zip(archive, [("magi-rs.exe", b"one"), ("nested/magi-rs.exe", b"two")])
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(UnsafeArchiveError):
        extract_executable(str(archive), "zip", str(destination), "magi-rs.exe")
    assert list(destination.iterdir()) == []


# --------------------------------------------------------------------------
# Block 3: the bin/ precondition and the sweep -- the only destructive operation
# --------------------------------------------------------------------------


def test_a_missing_bin_directory_is_created(tmp_path):
    target = tmp_path / "bin"
    require_real_directory(str(target))
    assert target.is_dir()


def test_a_real_bin_directory_is_accepted(tmp_path):
    target = tmp_path / "bin"
    target.mkdir()
    require_real_directory(str(target))  # does not raise


def test_a_bin_that_is_a_regular_file_is_refused_naming_it(tmp_path):
    # R3c: the condition is "not a real directory" -- ONE condition, not a
    # list of forbidden kinds. A list gets enumerated short and the missing
    # case falls through to the happy path.
    target = tmp_path / "bin"
    target.write_text("not a directory", encoding="utf-8")
    with pytest.raises(UnsafeBinDirectoryError) as exc:
        require_real_directory(str(target))
    assert str(target) in str(exc.value)


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
def test_a_bin_that_is_a_symlink_is_refused_naming_its_target(tmp_path):
    victim = tmp_path / "cargo-bin"
    victim.mkdir()
    (victim / "magi-rs").write_bytes(b"the user's own install")
    target = tmp_path / "bin"
    target.symlink_to(victim, target_is_directory=True)
    with pytest.raises(UnsafeBinDirectoryError) as exc:
        require_real_directory(str(target))
    assert str(victim) in str(exc.value)
    # Nothing was touched through the link:
    assert (victim / "magi-rs").read_bytes() == b"the user's own install"


def test_the_sweep_leaves_the_kept_executable_and_removes_the_rest(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "magi-rs").write_bytes(b"pinned")
    (bin_dir / "magi-rs.exe").write_bytes(b"previous version")
    (bin_dir / f".magi-rs-tmp-{os.getpid()}-asset").write_bytes(b"our own leftover")
    (bin_dir / f".magi-rs-old-{os.getpid()}-old").write_bytes(b"our own discard")
    sweep(str(bin_dir), keep="magi-rs")
    assert sorted(p.name for p in bin_dir.iterdir()) == ["magi-rs"]


def test_the_sweep_SPARES_another_process_discard(tmp_path):
    # A foreign discard is not inert rubbish. `_replace` renames the old binary
    # aside and renames it BACK if the second rename fails; deleting it inside
    # that window destroys the rollback and leaves bin/ with no executable.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "magi-rs").write_bytes(b"pinned")
    foreign = bin_dir / ".magi-rs-old-999999-old"
    foreign.write_bytes(b"another process may still roll back to this")
    sweep(str(bin_dir), keep="magi-rs")
    assert foreign.exists()


def test_the_sweep_SPARES_another_process_fresh_temporary(tmp_path):
    # BDD-3l. This is the round-1 CRITICAL, pinned: deleting a foreign
    # temporary that is still in flight breaks the very concurrency guarantee
    # the module claims. A test that only checked "bin/ ends up clean" would
    # BLESS that bug -- and the first draft of the test above did exactly that.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "magi-rs").write_bytes(b"pinned")
    foreign = bin_dir / ".magi-rs-tmp-999999-asset"
    foreign.write_bytes(b"another bootstrap is downloading into this RIGHT NOW")
    sweep(str(bin_dir), keep="magi-rs")
    assert foreign.exists()


def test_the_sweep_COLLECTS_another_process_abandoned_temporary(tmp_path):
    # BDD-3m. The mirror: past the cutoff it cannot still be in flight, and
    # leaving it forever would break "exactly one executable" the other way.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "magi-rs").write_bytes(b"pinned")
    abandoned = bin_dir / ".magi-rs-tmp-999999-asset"
    abandoned.write_bytes(b"a bootstrap that died last week")
    old = time.time() - (_STALE_TEMPORARY_SECONDS + 60)
    os.utime(abandoned, (old, old))
    sweep(str(bin_dir), keep="magi-rs")
    assert not abandoned.exists()


def test_a_short_pid_does_not_claim_a_longer_pids_temporary(tmp_path, monkeypatch):
    # BDD-3n. Without the delimiter, pid 42 matches ".magi-rs-tmp-421-..." by
    # prefix and deletes pid 421's LIVE download. The bug is invisible until
    # two pids happen to nest, which is exactly when nobody is looking.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "magi-rs").write_bytes(b"pinned")
    victim = bin_dir / ".magi-rs-tmp-421-asset"
    victim.write_bytes(b"pid 421 is using this")
    monkeypatch.setattr(os, "getpid", lambda: 42)
    sweep(str(bin_dir), keep="magi-rs")
    assert victim.exists()


def test_the_sweep_never_touches_a_file_it_does_not_recognise(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "magi-rs").write_bytes(b"pinned")
    (bin_dir / "README.md").write_bytes(b"someone put this here")
    (bin_dir / "ripgrep.exe").write_bytes(b"another tool")
    sweep(str(bin_dir), keep="magi-rs")
    assert (bin_dir / "README.md").exists()
    assert (bin_dir / "ripgrep.exe").exists()


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
def test_the_sweep_removes_a_link_and_never_its_target(tmp_path):
    # R3b limit 3. Deleting the target would leave the sweep's blast radius
    # OUTSIDE bin/ -- the exact damage limit 1 exists to prevent, entering by
    # the back door.
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    victim = outside / "magi-rs"
    victim.write_bytes(b"the user's own install")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "magi-rs.exe").symlink_to(victim)
    (bin_dir / "magi-rs").write_bytes(b"pinned")
    sweep(str(bin_dir), keep="magi-rs")
    assert not (bin_dir / "magi-rs.exe").exists()  # the LINK is gone
    assert victim.read_bytes() == b"the user's own install"  # the TARGET is not


@pytest.mark.skipif(os.name != "nt", reason="junctions only exist on Windows")
def test_a_bin_that_is_a_windows_junction_is_refused(tmp_path):
    # R3c's Windows half, and it DOES run on the development platform: a
    # junction needs no privilege, unlike a symlink. Without this,
    # `_is_reparse_point` -- the guard that stops the bootstrap writing into
    # the user's own binary directory -- had zero automated coverage anywhere.
    import subprocess

    victim = tmp_path / "cargo-bin"
    victim.mkdir()
    (victim / "magi-rs.exe").write_bytes(b"the user's own install")
    target = tmp_path / "bin"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(target), str(victim)],
        check=True,
        capture_output=True,
    )

    with pytest.raises(UnsafeBinDirectoryError) as exc:
        require_real_directory(str(target))
    assert str(victim) in str(exc.value)
    assert (victim / "magi-rs.exe").read_bytes() == b"the user's own install"


def test_a_delete_that_fails_is_reported_and_does_not_raise(tmp_path, monkeypatch):
    # R3b limit 4: on Windows an executable in use cannot be deleted. If the
    # pinned one is present and verified the gate is still correct, so the
    # sweep warns and the run proceeds; it is idempotent and retries next boot.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "magi-rs").write_bytes(b"pinned")
    (bin_dir / "magi-rs.exe").write_bytes(b"held open by another run")

    real_unlink = os.unlink

    def _refuse(path, *args, **kwargs):
        if path.endswith("magi-rs.exe"):
            raise PermissionError("file in use")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", _refuse)
    failed = sweep(str(bin_dir), keep="magi-rs")
    assert [os.path.basename(p) for p in failed] == ["magi-rs.exe"]


# --------------------------------------------------------------------------
# Block 4: BinaryBootstrap -- the order IS the correctness
# --------------------------------------------------------------------------


def _pin_with(executable_sha256, archive_sha256="a" * 64, version="0.19.1", url=None):
    """Build a Pin whose single asset matches whatever THIS platform selects.

    When ``url`` is given the pin serves it from ``download_url`` -- the seam
    the end-to-end tests need to point the bootstrap at the loopback server
    without reaching into a private attribute.
    """
    asset = Asset(
        name="magi-rs-vX-test.tar.gz",
        archive_sha256=archive_sha256,
        executable_sha256=executable_sha256,
        archive_kind="tar.gz",
    )
    key = _platform_key()
    pin = Pin(version=version, repository="BolivarTech/magi", assets={key: asset})
    if url is None:
        return pin
    return _PinnedAt(pin, url)


def _platform_key():
    """The pin key this platform selects, so tests need not hardcode one."""
    from magi_rs_pin import _PLATFORM_KEYS

    return _PLATFORM_KEYS[(platform.system(), platform.machine())]


class _PinnedAt:
    """A Pin that serves one fixed download URL. Delegates everything else."""

    def __init__(self, pin, url):
        self._pin, self._url = pin, url

    def __getattr__(self, name):
        return getattr(self._pin, name)

    def download_url(self, asset):
        return self._url


@contextlib.contextmanager
def _version_reporting(text):
    """Make the bootstrap's `--version` probe report `text` without launching."""
    with mock.patch(
        "magi_rs_bootstrap.subprocess.run",
        return_value=SimpleNamespace(stdout=text, stderr="", returncode=0),
    ):
        yield


def _installed_name():
    """The file name the bootstrap installs on THIS platform."""
    return "magi-rs.exe" if os.name == "nt" else "magi-rs"


def test_a_binary_whose_digest_does_not_match_the_pin_is_never_executed(tmp_path, monkeypatch):
    # R4/BDD-4: the strongest guarantee in the milestone. The failing input is
    # a present, plausible, WRONG binary -- and the proof is that no execution
    # was ATTEMPTED, not merely that a check reported false. A verify() that
    # checked the digest, raised, and had ALREADY run --version to learn the
    # version passes a test that only asserts the exception, and violates R4
    # just the same.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / _installed_name()).write_bytes(b"an impostor")

    executed: list[object] = []
    monkeypatch.setattr(
        "magi_rs_bootstrap.subprocess.run",
        lambda *a, **k: executed.append(a),
    )
    with pytest.raises(DigestMismatchError):
        BinaryBootstrap(_pin_with("0" * 64), str(bin_dir)).verify()
    assert executed == []


def test_a_binary_matching_the_pin_is_accepted_end_to_end(tmp_path, monkeypatch):
    # The mirror of the test above, and the one that would have caught the
    # archive-vs-executable digest confusion: the suite proving only REJECTION
    # is compatible with a verify() that can never accept anything.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    installed = bin_dir / _installed_name()
    installed.write_bytes(b"a plausible binary")
    monkeypatch.setattr(
        "magi_rs_bootstrap.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout="magi-rs 0.19.1", stderr="", returncode=0),
    )
    pin = _pin_with(hashlib.sha256(b"a plausible binary").hexdigest())
    assert BinaryBootstrap(pin, str(bin_dir)).verify() == str(installed)


def test_a_matching_digest_with_the_wrong_reported_version_is_refused(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / _installed_name()).write_bytes(b"binary")
    monkeypatch.setattr(
        "magi_rs_bootstrap.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout="magi-rs 0.9.9", stderr="", returncode=0),
    )
    pin = _pin_with(hashlib.sha256(b"binary").hexdigest())
    with pytest.raises(VersionMismatchError):
        BinaryBootstrap(pin, str(bin_dir)).verify()


def test_ensure_does_not_download_when_the_installed_digest_matches(tmp_path, monkeypatch):
    # BDD-2, at the ensure() level rather than the helper level.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / _installed_name()).write_bytes(b"already correct")
    downloads: list[str] = []
    monkeypatch.setattr(
        "magi_rs_bootstrap.download_to",
        lambda url, *a, **k: downloads.append(url),
    )
    pin = _pin_with(hashlib.sha256(b"already correct").hexdigest())
    BinaryBootstrap(pin, str(bin_dir)).ensure()
    assert downloads == []


def test_ensure_sweeps_even_when_it_does_not_download(tmp_path, monkeypatch):
    # The steady state IS every run but the first. A sweep placed after the
    # digest short circuit would never run there, and "exactly one executable"
    # would hold only on install day. The leftover carries THIS process's
    # token: a foreign, fresh discard is deliberately spared by the sweep
    # (R3b limit 2), so seeding one would test the wrong rule.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / _installed_name()).write_bytes(b"already correct")
    stale = bin_dir / f".magi-rs-old-{os.getpid()}-4242"
    stale.write_bytes(b"a leftover discard")
    pin = _pin_with(hashlib.sha256(b"already correct").hexdigest())
    BinaryBootstrap(pin, str(bin_dir)).ensure()
    assert not stale.exists()


def test_ensure_installs_from_a_served_asset_and_returns_a_binary_that_verifies(tmp_path):
    # The whole download-extract-replace-verify path, over the loopback server
    # already used above. Round 1's archive-vs-executable digest confusion
    # lived exactly here and NO automated test touched it -- the path rested on
    # a SMOKE, which a tired operator can report as passed.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # The member name must be the one THIS platform installs: on Windows the
    # bootstrap looks for magi-rs.exe, and a hardcoded "magi-rs" would make
    # both end-to-end tests fail on the development platform.
    executable = b"#!/bin/sh\necho magi-rs 0.19.1\n"
    archive_path = tmp_path / "asset.tar.gz"
    _make_tar(archive_path, [(f"magi-rs-v0.19.1-x/{_installed_name()}", executable)])
    payload = archive_path.read_bytes()

    with _Server(payload) as server:
        pin = _pin_with(
            hashlib.sha256(executable).hexdigest(),
            archive_sha256=hashlib.sha256(payload).hexdigest(),
            url=server.url,
        )
        with _version_reporting("magi-rs 0.19.1"):
            installed = BinaryBootstrap(pin, str(bin_dir)).ensure()

    assert os.path.basename(installed) == _installed_name()
    assert sha256_of(installed) == hashlib.sha256(executable).hexdigest()
    # R3b: nothing else survives -- no archive, no staging directory.
    assert sorted(p.name for p in bin_dir.iterdir()) == [_installed_name()]


def test_ensure_fails_closed_when_the_pinned_executable_digest_is_wrong(tmp_path):
    # Without the post-install verify(), a wrong executable_sha256 degrades
    # into a SILENT download loop: mismatch, re-download, install, mismatch,
    # forever, with no error anyone sees.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = b"#!/bin/sh\necho magi-rs 0.19.1\n"
    archive_path = tmp_path / "asset.tar.gz"
    _make_tar(archive_path, [(f"magi-rs-v0.19.1-x/{_installed_name()}", executable)])
    payload = archive_path.read_bytes()

    with _Server(payload) as server:
        pin = _pin_with(
            "f" * 64,  # <-- the wrong executable digest
            archive_sha256=hashlib.sha256(payload).hexdigest(),
            url=server.url,
        )
        with pytest.raises(DigestMismatchError):
            BinaryBootstrap(pin, str(bin_dir)).ensure()

    # NOT merely "it raised": an unverified binary must not be sitting in bin/.
    # A version that installs first and checks afterwards raises identically
    # and leaves the impostor in place for the next run to execute.
    assert not (bin_dir / _installed_name()).exists()


def test_ensure_refuses_before_touching_anything_when_bin_is_not_a_directory(tmp_path, monkeypatch):
    # R3c is a precondition of the WHOLE bootstrap, so it must fire before the
    # download -- not somewhere in the middle.
    not_a_dir = tmp_path / "bin"
    not_a_dir.write_text("regular file", encoding="utf-8")
    downloads: list[str] = []
    monkeypatch.setattr(
        "magi_rs_bootstrap.download_to",
        lambda url, *a, **k: downloads.append(url),
    )
    with pytest.raises(UnsafeBinDirectoryError):
        BinaryBootstrap(_pin_with("0" * 64), str(not_a_dir)).ensure()
    assert downloads == []


def test_ensure_clears_a_leftover_of_its_own_pid_before_downloading(tmp_path):
    # Pids are reused across runs: a crashed earlier bootstrap can leave a
    # temporary carrying THIS pid's token at the exact archive path the next
    # run will use. download_to refuses to overwrite, so the install path must
    # sweep its own leftovers first or the retry would fail forever.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = b"#!/bin/sh\necho magi-rs 0.19.1\n"
    archive_path = tmp_path / "asset.tar.gz"
    _make_tar(archive_path, [(f"magi-rs-v0.19.1-x/{_installed_name()}", executable)])
    payload = archive_path.read_bytes()
    leftover = bin_dir / f".magi-rs-tmp-{os.getpid()}-magi-rs-vX-test.tar.gz"
    leftover.write_bytes(b"half of a download that died")

    with _Server(payload) as server:
        pin = _pin_with(
            hashlib.sha256(executable).hexdigest(),
            archive_sha256=hashlib.sha256(payload).hexdigest(),
            url=server.url,
        )
        with _version_reporting("magi-rs 0.19.1"):
            installed = BinaryBootstrap(pin, str(bin_dir)).ensure()

    assert sha256_of(installed) == hashlib.sha256(executable).hexdigest()
    assert sorted(p.name for p in bin_dir.iterdir()) == [_installed_name()]


def test_the_bootstrap_errors_do_not_inherit_from_validationerror():
    # Locked derogation, rule 3 of the milestone: a fail-closed event must not
    # be swallowed by the orchestrator's
    # `except (ValidationError, json.JSONDecodeError)`. Same guard as the pin
    # module's; the five errors of this module were the ones nothing checked.
    from synthesize import ValidationError

    for error in (
        DownloadError,
        DigestMismatchError,
        UnsafeArchiveError,
        UnsafeBinDirectoryError,
        VersionMismatchError,
    ):
        assert not issubclass(error, ValidationError), error.__name__
        assert not issubclass(error, json.JSONDecodeError), error.__name__


def test_a_hanging_version_probe_is_a_version_failure_not_a_stray_timeout(tmp_path, monkeypatch):
    # _VERSION_PROBE_TIMEOUT_SECONDS bounds the hang; this pins that the
    # RESULT of the bound is part of the module's own error taxonomy, not a
    # subprocess.TimeoutExpired no caller is told to expect.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / _installed_name()).write_bytes(b"binary")

    def _hang(*a, **k):
        raise subprocess.TimeoutExpired(cmd=a[0], timeout=k.get("timeout", 0))

    monkeypatch.setattr("magi_rs_bootstrap.subprocess.run", _hang)
    pin = _pin_with(hashlib.sha256(b"binary").hexdigest())
    with pytest.raises(VersionMismatchError):
        BinaryBootstrap(pin, str(bin_dir)).verify()


def test_a_permission_error_on_a_first_install_is_not_masked_by_the_rename_dance(
    tmp_path, monkeypatch
):
    # The rename-aside dance answers ONE cause of PermissionError: the final
    # path is a running executable. On a first install there is no final path
    # to rename aside, so the original error must surface -- not a
    # FileNotFoundError from trying to move a file that never existed.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = b"#!/bin/sh\necho magi-rs 0.19.1\n"
    archive_path = tmp_path / "asset.tar.gz"
    _make_tar(archive_path, [(f"magi-rs-v0.19.1-x/{_installed_name()}", executable)])
    payload = archive_path.read_bytes()
    real_replace = os.replace

    def _refuse_final(src, dst, *a, **k):
        if os.path.basename(dst) == _installed_name():
            raise PermissionError("read-only bin/")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", _refuse_final)
    with _Server(payload) as server:
        pin = _pin_with(
            hashlib.sha256(executable).hexdigest(),
            archive_sha256=hashlib.sha256(payload).hexdigest(),
            url=server.url,
        )
        with pytest.raises(PermissionError):
            BinaryBootstrap(pin, str(bin_dir)).ensure()
    assert not (bin_dir / _installed_name()).exists()


# --------------------------------------------------------------------------
# Block 5: MAGI gate cycle 1 -- the four warnings, pinned
# --------------------------------------------------------------------------


def test_a_download_larger_than_the_ceiling_is_aborted_before_the_digest(tmp_path, monkeypatch):
    # The digest is compared only after the whole body arrived, so without a
    # byte ceiling a hostile server fills the disk before the mismatch is ever
    # seen. Content-Length is attacker data: the count is over bytes received.
    monkeypatch.setattr("magi_rs_bootstrap._MAX_DOWNLOAD_BYTES", 4096)
    payload = b"x" * (4096 + 1024)
    destination = tmp_path / "asset.bin"
    with _Server(payload) as server:
        with pytest.raises(DownloadError) as exc:
            download_to(server.url, str(destination), hashlib.sha256(payload).hexdigest())
    assert "ceiling" in str(exc.value)
    assert not destination.exists()


def test_a_zip_symlink_member_is_refused(tmp_path):
    # The tar guard has its own test; this pins the zip half, where a symlink
    # is only a mode in external_attr and nothing in the name gives it away.
    archive = tmp_path / "asset.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        info = zipfile.ZipInfo("magi-rs.exe")
        info.external_attr = 0o120777 << 16  # S_IFLNK | 0777
        bundle.writestr(info, "./harmless")
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(UnsafeArchiveError):
        extract_executable(str(archive), "zip", str(destination), "magi-rs.exe")
    assert list(destination.iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="junctions only exist on Windows")
def test_the_sweep_removes_a_junction_and_never_its_target(tmp_path):
    # R3b limit 3 on Windows: a junction is a directory to isdir() and not a
    # link to islink(), so a naive "rmtree the directory" branch either follows
    # it or refuses forever. The entry must go; what it points at must not.
    import subprocess

    victim = tmp_path / "elsewhere"
    victim.mkdir()
    (victim / "keep.txt").write_bytes(b"the user's own file")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "magi-rs").write_bytes(b"pinned")
    junction = bin_dir / "magi-rs.exe"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(victim)],
        check=True,
        capture_output=True,
    )
    failed = sweep(str(bin_dir), keep="magi-rs")
    assert failed == []
    assert not junction.exists()
    assert (victim / "keep.txt").read_bytes() == b"the user's own file"


@pytest.mark.parametrize(
    ("reported", "accepted"),
    [
        ("magi-rs 0.19.1", True),
        ("magi-rs 0.19.1 (release build abc123)", True),
        ("magi-rs version 0.19.1", True),
        ("magi-rs 0.19.10", False),
        ("magi-rs 10.19.1", False),
        ("magi-rs 0.19.1-rc1", False),
    ],
)
def test_the_version_probe_matches_the_pinned_version_as_a_whole_token(
    tmp_path, monkeypatch, reported, accepted
):
    # The contract is "the output carries the pinned version as a standalone
    # token", not "the last word is the version": a build suffix must not turn
    # into a false mismatch, and a longer version must not match by prefix.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / _installed_name()).write_bytes(b"binary")
    monkeypatch.setattr(
        "magi_rs_bootstrap.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout=reported, stderr="", returncode=0),
    )
    pin = _pin_with(hashlib.sha256(b"binary").hexdigest())
    bootstrap = BinaryBootstrap(pin, str(bin_dir))
    if accepted:
        bootstrap.verify()
    else:
        with pytest.raises(VersionMismatchError):
            bootstrap.verify()


def test_a_zip_member_that_is_neither_a_file_nor_a_directory_is_refused(tmp_path):
    # Symmetry with the tar guard: a zip carries the member kind only in the
    # mode bits of external_attr, and a FIFO or device there is as unwanted as
    # in a tar. A zero mode (Windows-created zips) still passes as a file.
    archive = tmp_path / "asset.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        info = zipfile.ZipInfo("magi-rs.exe")
        info.external_attr = 0o010644 << 16  # S_IFIFO
        bundle.writestr(info, "fifo?")
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(UnsafeArchiveError):
        extract_executable(str(archive), "zip", str(destination), "magi-rs.exe")
    assert list(destination.iterdir()) == []
