# Author: Julian Bolivar
# Version: 8.0.0
# Date: 2026-11-21
"""Path canonicalisation and containment, shared by the review-target guard and the rotation.

Both callers must agree on what "inside" means, or a path could be refused by one and
accepted by the other.
"""

from __future__ import annotations

import os


def canonical_path(path: str) -> str:
    """Return ``path`` resolved and case-normalised, for containment checks.

    Args:
        path: Any path; it need not exist.

    Returns:
        The real path (links and ``..`` resolved), folded to the filesystem's case rules.
    """
    return os.path.normcase(os.path.realpath(path))


def is_within(path: str, root: str) -> bool:
    """Tell whether ``path`` is ``root`` or lies under it, after canonicalising both.

    Args:
        path: The candidate path.
        root: The directory it may lie in.

    Returns:
        True when ``path`` resolves inside ``root``; False otherwise, including when the
        two are on different drives.
    """
    candidate, canonical_root = canonical_path(path), canonical_path(root)
    try:
        return os.path.commonpath([candidate, canonical_root]) == canonical_root
    except ValueError:  # different drives
        return False
