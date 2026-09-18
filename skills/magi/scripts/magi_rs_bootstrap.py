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
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from typing import IO

_CHUNK_BYTES = 64 * 1024
"""Streaming unit for download and hashing. Keeps memory constant regardless
of asset size -- the asset is never held whole in memory."""

_HTTP_TIMEOUT_SECONDS = 60
"""Per-``read()`` timeout on the release download."""

_TEMP_PREFIX = ".magi-rs-tmp-"
"""Prefix of this module's own staging files and directories. Every name built
from it carries this process's pid and a trailing ``-`` (see the sweep)."""

_MAX_EXTRACTED_BYTES = 64 * 1024 * 1024
"""Decompression-bomb ceiling: total bytes written across all members. Task 0
measured the largest real executable at 24.7 MB (macOS), so 64 MiB is 2.7x the
largest asset and far below a full disk."""

_MAX_COMPRESSION_RATIO = 200
"""Decompression-bomb ceiling: written bytes per compressed byte. Catches the
archive that expands hugely from very little and still fits under the absolute
ceiling above -- the two bounds cover different attacks. The real assets expand
by about 2x."""

_EXECUTABLE_MODE = 0o755


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


class UnsafeArchiveError(Exception):
    """The archive would escape its destination, expand past the ceilings, or
    does not contain the expected executable.

    Deliberately NOT a subclass of ``ValidationError``: fail-closed, and a
    retry does not fix a malicious or malformed archive.
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


def extract_executable(
    archive: str,
    archive_kind: str,
    destination_dir: str,
    member_name: str,
    install_as: str | None = None,
) -> str:
    """Extract exactly the executable from a verified archive.

    Every member is validated BEFORE it is written, and the running total of
    written bytes is checked DURING the write -- the size a member declares in
    its header is attacker-controlled data and is never trusted. The archive's
    internal layout is not assumed: the executable is located by name wherever
    it sits, because MS0 measured that the ``.zip`` is flat while the four
    ``.tar.gz`` assets nest under a version-named directory.

    Args:
        archive: Path to the already digest-verified archive.
        archive_kind: ``"zip"`` or ``"tar.gz"``.
        destination_dir: Directory the executable is placed in. Must exist and
            must already have passed the ``bin/`` precondition (R3c).
        member_name: File name to look for INSIDE the archive, e.g.
            ``"magi-rs.exe"``. Matched on the base name, at any depth.
        install_as: File name to write it out as. Defaults to ``member_name``,
            which is what the TESTS use -- they extract and assert in one step.
            PRODUCTION always passes it explicitly: ``ensure()`` needs a
            per-process-unique staging name so the rename into the final path
            is a separate, atomic step. The default is a test convenience, not
            a production path, and that is why it exists rather than the
            parameter being required.

    Returns:
        Absolute path of the extracted executable.

    Raises:
        UnsafeArchiveError: A member escapes the destination, is a link, the
            ceilings are crossed, or the executable is absent.
        OSError: The archive cannot be read or the destination written.

    Side effects:
        Writes into a temporary directory under ``destination_dir``, moves the
        executable into place, sets its execute bit on POSIX, and removes the
        temporary directory -- including on every failure path.

    Complexity:
        O(n) time and O(1) space in the extracted size: members are streamed in
        ``_CHUNK_BYTES`` chunks and never held whole in memory.
    """
    staging = tempfile.mkdtemp(prefix=f"{_TEMP_PREFIX}{os.getpid()}-", dir=destination_dir)
    try:
        if archive_kind == "zip":
            extracted = _extract_zip(archive, staging, member_name)
        elif archive_kind == "tar.gz":
            extracted = _extract_tar(archive, staging, member_name)
        else:
            raise UnsafeArchiveError(f"unknown archive kind {archive_kind!r} for {archive}")
        if extracted is None:
            raise UnsafeArchiveError(
                f"{archive} does not contain an executable named {member_name!r}"
            )
        final = os.path.join(destination_dir, install_as or member_name)
        os.replace(extracted, final)
        if os.name != "nt":
            # tarfile preserves the member's mode and zipfile does not; the
            # execute bit is guaranteed here rather than trusted to the format.
            os.chmod(final, _EXECUTABLE_MODE)
        return final
    finally:
        # On every path, including the bomb aborting mid-write: bin/ must be
        # left as it was. ignore_errors is bounded to a directory this call
        # created, on a path where an exception may already be propagating.
        shutil.rmtree(staging, ignore_errors=True)


def _safe_member_path(name: str, staging: str) -> str:
    """Resolve one member name inside the staging directory, or refuse it.

    Args:
        name: The member's declared name.
        staging: The directory it must stay inside.

    Returns:
        The absolute path to write the member to.

    Raises:
        UnsafeArchiveError: The name is absolute or escapes ``staging``.
    """
    if not name or os.path.isabs(name) or os.path.splitdrive(name)[0] or name[0] in "/\\":
        raise UnsafeArchiveError(f"archive member {name!r} has an absolute path")
    root = os.path.normpath(staging)
    candidate = os.path.normpath(os.path.join(root, name))
    if os.path.commonpath([root, candidate]) != root:
        raise UnsafeArchiveError(f"archive member {name!r} escapes the extraction directory")
    return candidate


def _guarded_copy(source: IO[bytes], target: str, written: int, compressed: int) -> int:
    """Copy one member's bytes, aborting the moment a ceiling is crossed.

    Args:
        source: A readable file object over the member.
        target: Path to write to.
        written: Bytes written so far across all members.
        compressed: Size of the archive on disk, for the ratio bound.

    Returns:
        The new running total of written bytes.

    Raises:
        UnsafeArchiveError: Either ceiling is crossed. The check is over the
            bytes ACTUALLY written, never over the declared member size.
    """
    with open(target, "wb") as handle:
        for chunk in iter(lambda: source.read(_CHUNK_BYTES), b""):
            written += len(chunk)
            if written > _MAX_EXTRACTED_BYTES:
                raise UnsafeArchiveError(
                    f"extraction aborted: more than {_MAX_EXTRACTED_BYTES} bytes written, "
                    "past the size ceiling"
                )
            if written > _MAX_COMPRESSION_RATIO * compressed:
                raise UnsafeArchiveError(
                    f"extraction aborted: {written} bytes written from {compressed} "
                    f"compressed, past the compression ratio ceiling of "
                    f"{_MAX_COMPRESSION_RATIO}"
                )
            handle.write(chunk)
    return written


def _extract_tar(archive: str, staging: str, executable_name: str) -> str | None:
    """Extract a .tar.gz, refusing links and escapes. See extract_executable."""
    compressed = os.path.getsize(archive)
    written = 0
    found: str | None = None
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            target = _safe_member_path(member.name, staging)
            if member.issym() or member.islnk():
                raise UnsafeArchiveError(f"archive member {member.name!r} is a link; refused")
            if member.isdir():
                continue
            if not member.isfile():
                raise UnsafeArchiveError(f"archive member {member.name!r} is not a regular file")
            if os.path.basename(member.name) != executable_name:
                continue
            if found is not None:
                raise UnsafeArchiveError(f"archive holds more than one {executable_name!r}")
            source = tar.extractfile(member)
            if source is None:  # pragma: no cover - isfile() above excludes this
                raise UnsafeArchiveError(f"archive member {member.name!r} cannot be read")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with source:
                written = _guarded_copy(source, target, written, compressed)
            found = target
    return found


def _extract_zip(archive: str, staging: str, executable_name: str) -> str | None:
    """Extract a .zip, refusing escapes. See extract_executable."""
    compressed = os.path.getsize(archive)
    written = 0
    found: str | None = None
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            target = _safe_member_path(info.filename, staging)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise UnsafeArchiveError(f"archive member {info.filename!r} is a link; refused")
            if info.is_dir():
                continue
            if os.path.basename(info.filename) != executable_name:
                continue
            if found is not None:
                raise UnsafeArchiveError(f"archive holds more than one {executable_name!r}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with bundle.open(info) as source:
                written = _guarded_copy(source, target, written, compressed)
            found = target
    return found


_EXECUTABLE_NAMES = ("magi-rs", "magi-rs.exe")
"""The ONLY names the sweep matches. Never a generic directory wipe: a user may
legitimately keep other files here."""

_DISCARD_PREFIX = ".magi-rs-old-"
"""Prefix of a previous binary renamed aside because it could not be replaced
in place (Windows holds a running executable). Carries the pid and a trailing
``-`` exactly like ``_TEMP_PREFIX``."""

_STALE_TEMPORARY_SECONDS = 3600
"""How old ANOTHER process's temporary must be before the sweep may remove it.

**Chosen, not derived, and the distinction is the point.** An earlier draft
summed the download and extraction timeouts and called the result a bound --
but ``_HTTP_TIMEOUT_SECONDS`` is a per-``read()`` timeout, not a total, and
nothing enforced a deadline over the extraction at all. A docstring proving a
property from premises the code does not enforce is worse than an unproven
number: it reads as a guarantee. *(Caught by the Checkpoint 2 gate, round 2, in
all three seats.)*

One hour is justified on its own terms: it is far longer than any plausible
bootstrap of a ~10 MB asset on any link that completes at all, so a temporary
older than this is abandoned with practical certainty -- while a bootstrap that
IS still running keeps its file. The cost of erring long is one leftover file
until the next run; the cost of erring short is deleting a live download."""


class UnsafeBinDirectoryError(Exception):
    """``bin/`` exists and is not a real directory.

    Deliberately NOT a subclass of ``ValidationError``: fail-closed. If the
    container is not trustworthy, not even READING from it to verify a digest
    is meaningful -- there is no way to know which directory is being read.
    """


def require_real_directory(path: str) -> None:
    """Assert that ``path`` is a real directory, creating it if absent.

    This is the precondition of the WHOLE bootstrap (R3c) -- download,
    extraction, chmod, rename and sweep -- evaluated exactly once, before
    anything is touched. It is not a guard repeated at each step: a guard you
    must remember to repeat in five places is forgotten in the sixth, and the
    dangerous half is the WRITE, not the delete. A ``bin/`` linked to
    ``~/.cargo/bin`` would make every write land in the user's own binary
    directory, overwriting their installation.

    Args:
        path: The plugin's ``bin/`` directory.

    Returns:
        None.

    Raises:
        UnsafeBinDirectoryError: ``path`` exists and is not a real directory --
            a symlink, a Windows reparse point (junction, mount point), or a
            regular file. The message names the path, what it is, and its
            target if it is a link.
        OSError: The directory cannot be created.

    Side effects:
        Creates ``path`` when it does not exist.
    """
    if not os.path.lexists(path):
        os.makedirs(path)
        return
    if os.path.islink(path) or _is_reparse_point(path):
        try:
            target = os.readlink(path)
        except OSError:
            target = "<unresolvable>"
        raise UnsafeBinDirectoryError(
            f"{path} is a link to {target}, not a real directory; the bootstrap refuses to "
            "write through or sweep through a link. Replace it with a real directory."
        )
    if not os.path.isdir(path):
        raise UnsafeBinDirectoryError(
            f"{path} exists and is not a directory; the bootstrap refuses to touch it. "
            "Remove it so a real bin/ directory can be created."
        )


def _is_reparse_point(path: str) -> bool:
    """Report whether a path is a Windows junction or mount point.

    ``os.path.islink`` does not report junctions, which are exactly as
    dangerous as a symlink for this purpose.

    Args:
        path: Directory to inspect.

    Returns:
        True if the entry carries the reparse-point attribute. Always False on
        non-Windows platforms.

    Raises:
        UnsafeBinDirectoryError: On Windows, the attribute cannot be read. The
            probe fails CLOSED: answering "not a reparse point" when unsure
            would let the bootstrap write through a junction.
    """
    if os.name != "nt":
        return False
    try:
        attributes = os.lstat(path).st_file_attributes
    except (OSError, AttributeError) as exc:
        raise UnsafeBinDirectoryError(
            f"{path}: cannot determine whether it is a reparse point ({exc}); refusing to "
            "proceed on an unknown directory kind"
        ) from exc
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def sweep(bin_dir: str, keep: str) -> list[str]:
    """Delete every magi-rs executable in ``bin_dir`` except ``keep``.

    This is the only destructive operation in the whole feature, and it obeys
    four limits (R3b):

    1. **Only inside the plugin's own bin/.** Never a ``magi-rs`` on PATH, in
       ``~/.cargo/bin``, or anywhere else -- the user may have their own
       install and destroying it would be real damage caused by a plugin that
       had no business touching it.
    2. **The directory has already passed** :func:`require_real_directory`.
       The sweep does NOT resolve ``bin_dir``.
    3. **Only files matching a known name** (``_EXECUTABLE_NAMES`` and this
       module's own temporaries). Never a generic wipe.
    4. **Links are unlinked, never followed.** The directory entry is removed;
       whatever it pointed at survives. Following one would put the blast
       radius outside limit 1 by the back door. (Hard links need no special
       case: removing one entry leaves the file alive through its others.)

    A delete that fails does NOT abort: on Windows an executable held by a
    concurrent run cannot be deleted, and if the pinned one is present and
    verified the gate is still correct. The sweep is idempotent and retries on
    the next start.

    Args:
        bin_dir: The plugin's ``bin/``, already validated.
        keep: File name of the pinned executable to preserve.

    Returns:
        Paths that could not be deleted, for the caller to warn about. Empty
        when everything was removed.

    Side effects:
        Deletes files inside ``bin_dir``.

    Complexity:
        O(n) in the number of entries in ``bin_dir``, once per run.
    """
    pid = os.getpid()
    own_token = f"{_TEMP_PREFIX}{pid}-"
    own_discard_token = f"{_DISCARD_PREFIX}{pid}-"
    now = time.time()
    failed: list[str] = []
    for entry in os.listdir(bin_dir):
        if entry == keep:
            continue
        path = os.path.join(bin_dir, entry)
        if not _is_sweepable(entry, path, own_token, own_discard_token, now):
            continue
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                # An abandoned staging directory. rmtree never follows links
                # out of it; a link ENTRY is handled by unlink below.
                shutil.rmtree(path)
            else:
                os.unlink(path)
        except OSError:
            failed.append(path)
    return failed


def _is_sweepable(
    entry: str, path: str, own_token: str, own_discard_token: str, now: float
) -> bool:
    """Decide whether the sweep may remove one directory entry.

    Args:
        entry: The entry's base name.
        path: Its full path.
        own_token: This process's temporary prefix, pid and delimiter included.
        own_discard_token: This process's discard prefix, likewise.
        now: Current wall-clock time, seconds since the epoch.

    Returns:
        True if the entry is a recognised magi-rs artifact this process is
        allowed to delete.

    Note:
        A FOREIGN temporary younger than ``_STALE_TEMPORARY_SECONDS`` is left
        alone: it may belong to a bootstrap running right now, and deleting it
        would break the concurrency guarantee this module claims. Our own
        temporaries are ours to remove at any age. The same rule covers a
        foreign DISCARD, which a concurrent ``_replace`` may still roll back
        to. A name carrying a prefix but no parseable pid is treated as
        foreign: this module never writes one, so it is someone else's or
        rubbish, and the age cutoff collects it either way.
    """
    if entry in _EXECUTABLE_NAMES:
        return True
    if entry.startswith((own_token, own_discard_token)):
        return True
    if not entry.startswith((_TEMP_PREFIX, _DISCARD_PREFIX)):
        return False
    try:
        age = now - os.lstat(path).st_mtime
    except OSError:
        return False
    return age > _STALE_TEMPORARY_SECONDS
