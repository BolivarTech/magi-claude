# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-09-18
"""Tests for the magi-rs binary bootstrap (MS2 Tasks 2-5).

No test touches the network or the plugin's real ``bin/``: downloads are served
by an ``http.server`` on an ephemeral loopback port and every directory is a
``tmp_path``.
"""

import hashlib
import http.server
import io
import os
import tarfile
import threading
import zipfile
from types import SimpleNamespace

import pytest

from magi_rs_bootstrap import (
    DigestMismatchError,
    DownloadError,
    UnsafeArchiveError,
    download_to,
    extract_executable,
    sha256_of,
)

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
