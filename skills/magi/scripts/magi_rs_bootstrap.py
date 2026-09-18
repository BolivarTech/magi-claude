# Author: Julian Bolivar
# Version: 1.1.1
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
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from typing import IO

from magi_rs_pin import Pin

_CHUNK_BYTES = 64 * 1024
"""Streaming unit for download and hashing. Keeps memory constant regardless
of asset size -- the asset is never held whole in memory."""

_HTTP_TIMEOUT_SECONDS = 60
"""Per-``read()`` timeout on the release download."""

_MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
"""Ceiling on bytes RECEIVED for one asset. The digest is only compared once
the whole body has arrived, so without this a hostile server fills the disk
before the mismatch is seen. Counted over bytes received, never over
Content-Length (attacker data). Task 0 measured the largest real archive at
11.6 MB; 64 MiB matches the extraction ceiling."""

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


class _DownloadCeilingReached(Exception):
    """Internal signal: the download loop crossed ``_MAX_DOWNLOAD_BYTES``.

    Raised inside the ``with`` block so the response and the file close before
    the destination is discarded; translated to :class:`DownloadError` at the
    boundary. Never leaves the module.
    """

    def __init__(self, received: int) -> None:
        super().__init__(received)
        self.received = received


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
        destination: Path to write to. Must NOT already exist: the name is
            unique per process, so a pre-existing entry is foreign and is
            refused rather than written through.
        expected_sha256: The digest from the pin, 64 lowercase hex characters.

    Returns:
        None.

    Raises:
        DownloadError: The asset could not be retrieved, ``destination``
            already exists, or more than ``_MAX_DOWNLOAD_BYTES`` arrived. The
            message names the URL and what went wrong.
        DigestMismatchError: The bytes arrived but hash to something else. The
            message names both digests.

    Side effects:
        Network read; writes ``destination``; deletes it again on failure.
    """
    if os.path.lexists(destination):
        # The name is unique to this process, so anything already there is
        # foreign. Opening it would follow a planted symlink and write the
        # archive bytes outside bin/; refusing narrows that to the window
        # between this check and open() (O_NOFOLLOW does not exist on
        # Windows), which the spec's threat model accepts: the planter already
        # has write access to the plugin directory.
        raise DownloadError(
            f"refusing to download {url}: {destination} already exists and is not ours"
        )
    digest = hashlib.sha256()
    try:
        with (
            urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_SECONDS) as response,
            open(destination, "wb") as handle,
        ):
            received = 0
            for chunk in iter(lambda: response.read(_CHUNK_BYTES), b""):
                received += len(chunk)
                if received > _MAX_DOWNLOAD_BYTES:
                    raise _DownloadCeilingReached(received)
                handle.write(chunk)
                digest.update(chunk)
    except _DownloadCeilingReached as exc:
        _discard(destination)
        raise DownloadError(
            f"aborted downloading {url}: more than {_MAX_DOWNLOAD_BYTES} bytes received, "
            f"past the download ceiling (got {exc.received} and counting)"
        ) from None
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
            ceilings are crossed, the executable is absent, or the file is not
            an archive of ``archive_kind`` at all.
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
        try:
            if archive_kind == "zip":
                extracted = _extract_zip(archive, staging, member_name)
            elif archive_kind == "tar.gz":
                extracted = _extract_tar(archive, staging, member_name)
            else:
                raise UnsafeArchiveError(f"unknown archive kind {archive_kind!r} for {archive}")
        except (tarfile.TarError, zipfile.BadZipFile) as exc:
            # Verified bytes that are not an archive of the declared kind: the
            # pin is inconsistent, and the caller is told UnsafeArchiveError,
            # not a stdlib type it has no reason to expect.
            raise UnsafeArchiveError(
                f"{archive} is not a valid {archive_kind} archive: {exc}"
            ) from exc
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
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise UnsafeArchiveError(f"archive member {info.filename!r} is a link; refused")
            if info.is_dir():
                continue
            # Only the file-TYPE bits decide: a zip written without Unix
            # attributes (Windows, or permission bits alone as zipfile's own
            # writestr emits) carries no type and passes; a type that is set
            # must say regular file, as the tar guard demands.
            kind = stat.S_IFMT(mode)
            if kind and kind != stat.S_IFREG:
                raise UnsafeArchiveError(f"archive member {info.filename!r} is not a regular file")
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
            if os.path.islink(path):
                os.unlink(path)  # the link entry, never what it points at
            elif _is_reparse_point(path):
                # A Windows junction: isdir() says directory, islink() says
                # no. rmdir removes the junction entry and leaves its target
                # untouched; rmtree would refuse it and leave it forever.
                os.rmdir(path)
            elif os.path.isdir(path):
                # An abandoned staging directory of our own. rmtree does not
                # follow links out of it.
                shutil.rmtree(path)
            else:
                os.unlink(path)
        except (OSError, UnsafeBinDirectoryError):
            # UnsafeBinDirectoryError: the reparse probe failed closed on an
            # entry it could not classify; leave it and report it.
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


_INSTALLED_NAME = "magi-rs.exe" if os.name == "nt" else "magi-rs"
"""File name the executable is installed under on THIS platform, and the name
it is looked up by inside the archive."""

_VERSION_FLAG = "--version"
_VERSION_PROBE_TIMEOUT_SECONDS = 30
"""Ceiling on the ``--version`` probe: a verified binary that hangs on its
cheapest command must not hang the gate."""


class VersionMismatchError(Exception):
    """The installed binary reports a version other than the pinned one.

    Deliberately NOT a subclass of ``ValidationError`` -- see
    :class:`DownloadError`.
    """


class BinaryBootstrap:
    """Obtain, verify and install the pinned ``magi-rs`` executable.

    The order of operations IS the correctness of this class, and each step
    exists because a specific failure was measured or specified:

    1. :func:`require_real_directory` -- ONCE, before anything (R3c).
    2. If the installed binary already matches the pin, stop: no download
       (BDD-2). The trigger is the PIN, not the file's absence, so a perfectly
       healthy previous version is replaced when the pin moves (R3a/BDD-3a).
    3. Sweep our own leftovers, then download to a per-process-unique
       temporary and verify it THERE. Nothing moves before it verifies.
    4. Extract to staging inside the same ``bin/`` -- same volume, so the
       rename is atomic and not a copy.
    5. Rename into place. If the destination is held (Windows cannot overwrite
       a running executable), rename the OLD one aside first: an executable in
       use cannot be DELETED but CAN be renamed.
    6. Sweep, then warn about what could not be deleted.

    Two instances bootstrapping the same ``bin/`` cannot corrupt each other: the
    temporaries are unique per process and the CONTENT is fixed by digest, so
    both move identical bytes and "last one wins" means nothing. No lock is
    needed -- and adding one would introduce a state file in the very release
    that removes the only one there was.

    The window between renaming the old aside and moving the new in is real and
    accepted: a run starting exactly then finds no binary and SELF-HEALS by
    reinstalling (R3/BDD-3k). Overwriting in place does not exist on Windows,
    so there is no window-free design to choose instead.

    Example:
        >>> pin = Pin.load("skills/magi/magi-rs.pin.json")
        >>> path = BinaryBootstrap(pin, "bin").ensure()

    Side effects:
        Network, file writes, file deletion, archive extraction, chmod.
    """

    def __init__(self, pin: Pin, bin_dir: str) -> None:
        """Bind the bootstrap to one pin and one directory.

        Args:
            pin: The parsed pin.
            bin_dir: The plugin's ``bin/``. Validated on the first ``ensure()``
                or ``verify()``, never in the constructor -- construction must
                not have side effects.
        """
        self._pin = pin
        self._bin_dir = bin_dir

    @property
    def executable_path(self) -> str:
        """Path the pinned executable occupies, whether or not it exists yet."""
        return os.path.join(self._bin_dir, _INSTALLED_NAME)

    def verify(self) -> str:
        """Check the installed binary against the pin WITHOUT executing it.

        The digest is checked first and the version second, and that order is
        load-bearing: reading the version means RUNNING the binary, and R4
        forbids running one that has not verified.

        Returns:
            Path of the verified executable.

        Raises:
            DigestMismatchError: The file is present but hashes to something
                else. The binary is not executed.
            DownloadError: The file is absent.
            VersionMismatchError: The digest matches but ``--version`` reports
                another version -- which means the pin itself is inconsistent.
            UnsafeBinDirectoryError: ``bin/`` is not a real directory.
            UnsupportedPlatformError: No asset for this platform.
            OSError: The digest matched but the binary could not be launched.

        Side effects:
            Runs the executable ONLY after its digest has matched.
        """
        require_real_directory(self._bin_dir)
        asset = self._pin.asset_for_current_platform()
        path = self.executable_path
        if not os.path.isfile(path):
            raise DownloadError(
                f"{path} is absent; run the bootstrap (ensure) or place the pinned "
                f"magi-rs {self._pin.version} executable there by hand"
            )
        self._require_digest(path, asset.executable_sha256)
        self._require_version(path)
        return path

    def ensure(self) -> str:
        """Install the pinned executable if it is not already installed.

        Idempotent and self-healing: an absent binary (a plugin update wiped
        the ephemeral plugin root) or one whose digest no longer matches the
        pin is downloaded again.

        Returns:
            Path of the installed, verified executable.

        Raises:
            UnsafeBinDirectoryError: ``bin/`` is not a real directory. Nothing
                is downloaded, extracted, written or deleted.
            UnsupportedPlatformError: No asset for this platform.
            DownloadError: The release could not be retrieved. The escape is
                explicit: place the pinned executable in ``bin/`` by hand --
                ``verify()`` checks its digest just the same.
            DigestMismatchError: What arrived is not what the pin expects.
            UnsafeArchiveError: The archive would escape or expand past the
                ceilings.
            VersionMismatchError: The fresh install reports another version.
            OSError: The replacement could not be completed (see ``_replace``).

        Side effects:
            All five.
        """
        require_real_directory(self._bin_dir)
        asset = self._pin.asset_for_current_platform()
        final = self.executable_path
        if os.path.isfile(final) and sha256_of(final) == asset.executable_sha256:
            # Steady state: the sweep still runs, or "exactly one executable"
            # would hold only on install day.
            self._sweep_and_warn(_INSTALLED_NAME)
            return final
        # Install path: sweep FIRST. Pids are reused, so a crashed earlier run
        # may have left a temporary carrying this pid's token; download_to
        # refuses a pre-existing destination, and the sweep is what clears
        # our own leftovers (and stale foreign ones) before we write.
        self._sweep_and_warn(_INSTALLED_NAME)
        token = f"{_TEMP_PREFIX}{os.getpid()}-"
        archive = os.path.join(self._bin_dir, f"{token}{asset.name}")
        download_to(self._pin.download_url(asset), archive, asset.archive_sha256)
        try:
            staged = extract_executable(
                archive,
                asset.archive_kind,
                self._bin_dir,
                _INSTALLED_NAME,
                install_as=f"{token}{_INSTALLED_NAME}",
            )
        finally:
            _discard(archive)
        try:
            # Verified while still in STAGING: the final path is what R4 gates
            # on, so an unverified file must not occupy it for an instant.
            self._require_digest(staged, asset.executable_sha256)
            self._replace(staged, final)
        except Exception:
            # Broad on purpose and never silent: whatever failed, the staged
            # file must not survive where a later run could take it for an
            # install; the original exception propagates unchanged.
            _discard(staged)
            raise
        self._sweep_and_warn(_INSTALLED_NAME)
        return self.verify()

    def _require_digest(self, path: str, expected: str) -> None:
        """Compare a file's digest with the pin's, raising on any difference.

        Args:
            path: File to hash.
            expected: The pin's ``executable_sha256``.

        Returns:
            None.

        Raises:
            DigestMismatchError: The digests differ; the message names both.
        """
        actual = sha256_of(path)
        if actual != expected:
            raise DigestMismatchError(
                f"{path} hashes to {actual}, the pin expects {expected}; the file is not "
                "executed. Delete it and re-run the bootstrap, or place the pinned "
                "executable there by hand."
            )

    def _require_version(self, path: str) -> None:
        """Run ``--version`` on an already digest-verified binary and compare.

        Args:
            path: The verified executable.

        Returns:
            None.

        Raises:
            VersionMismatchError: The reported version is not the pinned one,
                the probe exited non-zero, or it did not answer within
                ``_VERSION_PROBE_TIMEOUT_SECONDS``.
            OSError: The binary could not be launched.

        Side effects:
            Executes ``path`` once, with a timeout.
        """
        try:
            probe = subprocess.run(
                [path, _VERSION_FLAG],
                capture_output=True,
                text=True,
                timeout=_VERSION_PROBE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise VersionMismatchError(
                f"{path} {_VERSION_FLAG} did not answer within "
                f"{_VERSION_PROBE_TIMEOUT_SECONDS}s; no usable version"
            ) from exc
        reported = probe.stdout.strip()
        if probe.returncode != 0 or not reported:
            raise VersionMismatchError(
                f"{path} {_VERSION_FLAG} exited {probe.returncode} with no usable version "
                f"(stderr: {probe.stderr.strip()!r})"
            )
        # The contract: the output carries the pinned version as a standalone
        # token. A build suffix must not read as a mismatch, and a longer
        # version must not match by prefix.
        token = rf"(?<![\w.]){re.escape(self._pin.version)}(?![\w.-])"
        if re.search(token, reported) is None:
            raise VersionMismatchError(
                f"{path} reports {reported!r}, the pin expects version {self._pin.version}; "
                "the pin is inconsistent with the asset it names"
            )

    def _sweep_and_warn(self, keep: str) -> None:
        """Sweep ``bin/`` and report on stderr what could not be removed.

        Args:
            keep: File name of the pinned executable to preserve.

        Returns:
            None.

        Side effects:
            Deletes files inside ``bin/``; writes warnings to stderr.
        """
        for leftover in sweep(self._bin_dir, keep):
            print(
                f"[WARNING] could not remove {leftover} from bin/ (held by another run?); "
                "the next bootstrap will retry",
                file=sys.stderr,
            )

    def _replace(self, staged: str, final: str) -> None:
        """Move the staged executable into place, working around a held file.

        Args:
            staged: The verified executable, in ``bin/`` under a temporary name.
            final: Where it must end up.

        Returns:
            None.

        Raises:
            OSError: The replacement could not be completed even after renaming
                the old binary aside. Aborting is correct: an error naming the
                held file beats a half-updated ``bin/``. If the roll-back of
                the old binary fails too, ``bin/`` is left with no executable
                at the final path -- the next run finds it absent and
                reinstalls (self-healing, BDD-3k); it never runs an unverified
                one.

        Side effects:
            Renames within ``bin/``; may leave a discard file behind for the
            sweep to collect and report.
        """
        try:
            os.replace(staged, final)
            return
        except PermissionError:
            # Not swallowed -- a change of strategy. PermissionError, not
            # OSError: only "the destination is a running executable" (Windows)
            # is what the rename dance below addresses. A full disk or a
            # read-only directory must propagate, or bin/ ends up with NO
            # executable after the old one was renamed aside. And with no
            # previous binary there is nothing to rename aside, so the error
            # is the original one, not a stray FileNotFoundError.
            if not os.path.lexists(final):
                raise
        discard = os.path.join(self._bin_dir, f"{_DISCARD_PREFIX}{os.getpid()}-{int(time.time())}")
        os.replace(final, discard)
        try:
            os.replace(staged, final)
        except OSError:
            # Roll the old binary back so bin/ keeps an executable; the error
            # that follows names the held file (BDD-3h).
            os.replace(discard, final)
            raise
        # Best effort: a discard still held by a running old binary cannot be
        # deleted on Windows. The sweep that follows in ensure() retries it and
        # is the single place that warns, so nothing is reported twice.
        _discard(discard)
