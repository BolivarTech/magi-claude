# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-09-18
"""Obtain, verify and install the pinned ``magi-rs`` executable.

Side effects -- this module has five, and every public function documents its
own:

* network (downloads the release asset),
* file writes (staging files and the final executable),
* file deletion (the sweep, and the downloaded archive),
* archive extraction (validated member by member),
* mode changes (the execute bit on POSIX).

Nothing here ever runs the executable it installs; verification is the caller's
gate and lives in :meth:`BinaryBootstrap.verify`.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import urllib.error
import urllib.request

_CHUNK_BYTES = 64 * 1024
"""Streaming unit for download and hashing. Keeps memory constant regardless
of asset size -- the asset is never held whole in memory."""

_HTTP_TIMEOUT_SECONDS = 60
"""Per-``read()`` timeout on the release download."""


class DownloadError(Exception):
    """The release asset could not be retrieved.

    Deliberately NOT a subclass of ``ValidationError``: fail-closed, and the
    orchestrator's ``except (ValidationError, json.JSONDecodeError)`` must not
    consume it.
    """


class DigestMismatchError(Exception):
    """A file's sha256 does not match the pin.

    Deliberately NOT a subclass of ``ValidationError`` -- see
    :class:`DownloadError`.
    """


def sha256_of(path: str) -> str:
    """Hash a file in constant memory.

    Args:
        path: File to hash.

    Returns:
        The lowercase hex sha256 digest.

    Raises:
        OSError: The file cannot be read.

    Complexity:
        O(n) time in file size, O(1) space -- the file is read in
        ``_CHUNK_BYTES`` chunks and never held whole.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_to(url: str, destination: str, expected_sha256: str) -> None:
    """Download one asset and verify it, leaving nothing behind on failure.

    The bytes are streamed to ``destination`` and hashed in the same pass, so
    the asset is never held whole in memory. On ANY failure -- transport error,
    truncated stream, digest mismatch -- ``destination`` is removed before the
    exception propagates: a partial or unverified file must never survive where
    a later step could mistake it for an installed asset.

    Args:
        url: Absolute URL of the release asset.
        destination: Path to write to. Overwritten if present.
        expected_sha256: The digest from the pin, 64 lowercase hex characters.

    Returns:
        None.

    Raises:
        DownloadError: The asset could not be retrieved. The message names the
            URL and what went wrong.
        DigestMismatchError: The bytes arrived but hash to something else. The
            message names both digests.

    Side effects:
        Network read; writes ``destination``; deletes it again on failure.
    """
    digest = hashlib.sha256()
    try:
        with (
            urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_SECONDS) as response,
            open(destination, "wb") as handle,
        ):
            for chunk in iter(lambda: response.read(_CHUNK_BYTES), b""):
                handle.write(chunk)
                digest.update(chunk)
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        # URLError covers HTTPError (a status) and the connection failures;
        # HTTPException covers IncompleteRead (a stream cut short of its
        # Content-Length); OSError covers the write side and socket timeouts.
        _discard(destination)
        raise DownloadError(f"could not download {url}: {exc}") from exc
    actual = digest.hexdigest()
    if actual != expected_sha256:
        _discard(destination)
        raise DigestMismatchError(
            f"downloaded {url} hashes to {actual}, the pin expects {expected_sha256}"
        )


def _discard(path: str) -> None:
    """Remove a file if it exists, ignoring the failure to do so.

    Used only on paths this module itself just created, on an error path where
    the caller is already raising: a failure to clean up must not mask the
    error that caused it.

    Args:
        path: File to remove.

    Returns:
        None.

    Side effects:
        Deletes ``path`` if present.
    """
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    except OSError:
        # Justified silent failure (see the docstring): an exception is already
        # propagating and this path is ours; masking it would replace "the
        # digest does not match" with "could not delete a temporary".
        pass
