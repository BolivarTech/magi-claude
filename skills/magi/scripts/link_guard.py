# Author: Julian Bolivar
# Version: 8.0.0
# Date: 2026-11-21
"""Refusal of a path that is a link or reparse point, shared by every place that touches the key.

The key store (before it writes), the permission guards (before they read a DACL or mode), the
rotation and ``--ollama-init`` all ask the same question with the same wording.
"""

from __future__ import annotations

import os
import stat

_IO_REPARSE_TAG_SYMLINK = 0xA000000C
_IO_REPARSE_TAG_MOUNT_POINT = 0xA0000003
"""Windows reparse tags of a symbolic link and of a junction (the ``stat`` module defines them
only on Windows builds, and this module is imported everywhere)."""

_LINK_REPARSE_TAGS = frozenset({_IO_REPARSE_TAG_SYMLINK, _IO_REPARSE_TAG_MOUNT_POINT})
"""The reparse tags that are links."""

_LINK_REFUSAL = "{path} is a link: replace it with a real file/directory"
_REPARSE_REFUSAL = (
    "{path} is a reparse point (for example a cloud-sync placeholder): keep the project's .magi/ "
    "on a local, non-synced folder"
)


def link_refusal(path: str, display: str | None = None) -> str | None:
    """Return why ``path`` must not be trusted as a real file or directory, or ``None`` when it may.

    Fails closed for every Windows reparse point, but names what it is: a symlink or junction is
    "a link", any other reparse point (a cloud-sync placeholder, deduplication) is not. Decided
    from ``os.lstat``, so nothing is followed.

    Args:
        path: The entry to inspect.
        display: How the message names ``path``; defaults to ``path`` itself.

    Returns:
        A one-line refusal, or ``None`` for an ordinary entry.

    Raises:
        OSError: The entry cannot be inspected.
    """
    status = os.lstat(path)
    shown = path if display is None else display
    if stat.S_ISLNK(status.st_mode):
        return _LINK_REFUSAL.format(path=shown)
    if getattr(status, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        if getattr(status, "st_reparse_tag", 0) in _LINK_REPARSE_TAGS:
            return _LINK_REFUSAL.format(path=shown)
        return _REPARSE_REFUSAL.format(path=shown)
    return None
