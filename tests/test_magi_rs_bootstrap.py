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
import threading

import pytest

from magi_rs_bootstrap import (
    DigestMismatchError,
    DownloadError,
    download_to,
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
