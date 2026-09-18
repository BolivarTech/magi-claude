# Author: Julian Bolivar
# Version: 1.0.3
# Date: 2026-09-18
"""Read-only view over the pinned ``magi-rs`` release.

This module performs no I/O beyond reading the pin file itself: it neither
downloads nor writes anything. Selecting an asset is a pure lookup keyed on the
running platform.

Example:
    >>> pin = Pin.load("skills/magi/magi-rs.pin.json")
    >>> pin.asset_for("Windows", "AMD64").name
    'magi-rs-v0.19.1-windows-x86_64.zip'

Known limitation:
    The ``linux-x86_64-compat`` asset (built for older glibc) is published by
    the release but is deliberately absent from the pin: choosing between it
    and the regular build would require probing the host libc, which no
    requirement asks for and which would fail silently in the rare case.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_PLATFORM_KEYS: Mapping[tuple[str, str], str] = {
    ("Windows", "AMD64"): "windows-x86_64",
    ("Windows", "x86_64"): "windows-x86_64",
    ("Linux", "x86_64"): "linux-x86_64",
    ("Linux", "aarch64"): "linux-aarch64",
    ("Darwin", "arm64"): "macos-universal2",
    ("Darwin", "x86_64"): "macos-universal2",
}

_REQUIRED_PIN_FIELDS = ("version", "repository", "assets")
_REQUIRED_ASSET_FIELDS = ("name", "archive_sha256", "executable_sha256", "archive_kind")
_DIGEST_FIELDS = ("archive_sha256", "executable_sha256")
_SHA256_HEX_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")
_NAME_SEPARATORS = ("/", "\\", "\x00")
_ARCHIVE_KINDS = ("zip", "tar.gz")
_RELEASE_URL_TEMPLATE = "https://github.com/{repository}/releases/download/v{version}/{name}"


class PinFormatError(Exception):
    """The pin file is absent, unparsable, or missing a required field.

    Deliberately NOT a subclass of ``ValidationError``: this is a fail-closed
    event that a retry does not fix, so the orchestrator's
    ``except (ValidationError, json.JSONDecodeError)`` must not consume it.
    """


class UnsupportedPlatformError(Exception):
    """No asset is published for the running platform.

    Deliberately NOT a subclass of ``ValidationError`` -- see
    :class:`PinFormatError`.
    """


@dataclass(frozen=True)
class Asset:
    """One published release artifact.

    TWO digests, and conflating them makes the happy path impossible: the
    archive and the executable inside it are different files with different
    hashes. ``archive_sha256`` gates the download; ``executable_sha256`` is what
    R4 re-checks on EVERY run, long after the archive is gone.

    Attributes:
        name: File name of the asset within the release.
        archive_sha256: Digest of the downloaded archive, from the release's
            ``SHA256SUMS.txt``. 64 lowercase hex characters.
        executable_sha256: Digest of the executable extracted from it, measured
            in Task 0. This is the one ``verify()`` compares against.
        archive_kind: Either ``"zip"`` or ``"tar.gz"``.
    """

    name: str
    archive_sha256: str
    executable_sha256: str
    archive_kind: str


@dataclass(frozen=True)
class Pin:
    """The pinned release: one version, one asset per supported platform."""

    version: str
    repository: str
    assets: Mapping[str, Asset]

    @classmethod
    def load(cls, path: str) -> Pin:
        """Read and validate the pin file.

        Args:
            path: Path to ``magi-rs.pin.json``.

        Returns:
            The parsed pin.

        Raises:
            PinFormatError: The file is missing, unparsable, a required field
                is absent, a digest is not 64 lowercase hex characters, an
                asset name is not a plain file name, or an archive kind is
                neither ``"zip"`` nor ``"tar.gz"``.
        """
        try:
            with open(path, encoding="utf-8") as handle:
                raw: Any = json.load(handle)
        except OSError as exc:
            raise PinFormatError(f"pin file {path!r} cannot be read: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise PinFormatError(f"pin file {path!r} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise PinFormatError(f"pin file {path!r} must hold a JSON object")
        for field in _REQUIRED_PIN_FIELDS:
            if field not in raw:
                raise PinFormatError(f"pin file {path!r} lacks the required field {field!r}")
        if not isinstance(raw["assets"], dict):
            raise PinFormatError(f"pin file {path!r}: 'assets' must be a JSON object")
        assets: dict[str, Asset] = {}
        for key, entry in raw["assets"].items():
            if not isinstance(entry, dict):
                raise PinFormatError(f"pin file {path!r}: asset {key!r} must be a JSON object")
            for field in _REQUIRED_ASSET_FIELDS:
                if field not in entry:
                    raise PinFormatError(
                        f"pin file {path!r}: asset {key!r} lacks the required field {field!r}"
                    )
            for field in _DIGEST_FIELDS:
                digest = str(entry[field])
                # A digest one character short passes any presence check and
                # fails only at download time, as a message about digests
                # rather than about a pin mistyped while moving the version.
                if len(digest) != _SHA256_HEX_LENGTH or not set(digest) <= _HEX_DIGITS:
                    raise PinFormatError(
                        f"pin file {path!r}: asset {key!r} field {field!r} is not a "
                        f"{_SHA256_HEX_LENGTH}-character lowercase hex sha256"
                    )
            name = str(entry["name"])
            # The name becomes a path component inside bin/: a separator in
            # it would escape bin/ through the trust channel itself.
            if name in ("", ".", "..") or any(sep in name for sep in _NAME_SEPARATORS):
                raise PinFormatError(
                    f"pin file {path!r}: asset {key!r} name {name!r} is not a plain file name"
                )
            archive_kind = str(entry["archive_kind"])
            if archive_kind not in _ARCHIVE_KINDS:
                raise PinFormatError(
                    f"pin file {path!r}: asset {key!r} archive_kind {archive_kind!r} is not "
                    f"one of {_ARCHIVE_KINDS}"
                )
            assets[key] = Asset(
                name=name,
                archive_sha256=str(entry["archive_sha256"]),
                executable_sha256=str(entry["executable_sha256"]),
                archive_kind=archive_kind,
            )
        return cls(version=str(raw["version"]), repository=str(raw["repository"]), assets=assets)

    def asset_for(self, system: str, machine: str) -> Asset:
        """Select the asset for one platform.

        Args:
            system: A ``platform.system()`` value, e.g. ``"Linux"``.
            machine: A ``platform.machine()`` value, e.g. ``"x86_64"``.

        Returns:
            The asset published for that platform.

        Raises:
            UnsupportedPlatformError: No asset exists for it. The message names
                the platform; no other architecture is ever substituted.
        """
        key = _PLATFORM_KEYS.get((system, machine))
        if key is None or key not in self.assets:
            raise UnsupportedPlatformError(
                f"no magi-rs {self.version} asset is published for {system}/{machine}; "
                "build it from source or place the pinned executable in bin/ by hand"
            )
        return self.assets[key]

    def asset_for_current_platform(self) -> Asset:
        """Select the asset for the platform this process runs on.

        Returns:
            The asset for ``platform.system()``/``platform.machine()``.

        Raises:
            UnsupportedPlatformError: No asset exists for it.
        """
        return self.asset_for(platform.system(), platform.machine())

    def download_url(self, asset: Asset) -> str:
        """Build the release download URL for one asset.

        Args:
            asset: The asset to download.

        Returns:
            The absolute HTTPS URL of that asset in the pinned release.
        """
        return _RELEASE_URL_TEMPLATE.format(
            repository=self.repository, version=self.version, name=asset.name
        )
