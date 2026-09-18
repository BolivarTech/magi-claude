# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-09-18
"""Locate the plugin's own tree: its ``bin/`` and its pin.

``${CLAUDE_PLUGIN_ROOT}`` is what Claude Code sets when the plugin runs
installed. It is ephemeral by their documentation (R3) and the bootstrap
absorbs that; what this module decides is only WHERE to look.
"""

from __future__ import annotations

import os

PLUGIN_ROOT_VARIABLE = "CLAUDE_PLUGIN_ROOT"
_BIN_DIRNAME = "bin"
_PIN_RELATIVE_PATH = os.path.join("skills", "magi", "magi-rs.pin.json")
_THIS_TREE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class PluginRootError(Exception):
    """``CLAUDE_PLUGIN_ROOT`` is set but does not name a directory.

    Deliberately NOT a subclass of ``ValidationError``: fail-closed, and a
    retry does not create the directory.
    """


def plugin_root(environ: os._Environ[str] | dict[str, str] | None = None) -> str:
    """Resolve the plugin's root directory.

    ``CLAUDE_PLUGIN_ROOT`` wins when set. When it is ABSENT the root of the
    tree this module runs from is used -- the same tree the pin ships in --
    which is deterministic and is what the project's own gate relies on when
    it runs ``run_magi.py`` from a plain shell. Never the cwd, never ``PATH``.

    Args:
        environ: The environment to read; defaults to ``os.environ``.

    Returns:
        An absolute directory path.

    Raises:
        PluginRootError: The variable is set but empty or not a directory. It
            is named in the message; nothing is guessed in its place.
    """
    env = os.environ if environ is None else environ
    if PLUGIN_ROOT_VARIABLE in env:
        configured = env[PLUGIN_ROOT_VARIABLE]
        if not configured or not os.path.isdir(configured):
            raise PluginRootError(
                f"{PLUGIN_ROOT_VARIABLE}={configured!r} is not an existing directory; the "
                "bootstrap refuses to guess another location"
            )
        return os.path.abspath(configured)
    return os.path.abspath(_THIS_TREE_ROOT)


def bin_dir(root: str) -> str:
    """The plugin's ``bin/`` under ``root``.

    Args:
        root: The plugin root.

    Returns:
        ``<root>/bin``; not created here (the bootstrap's precondition does).
    """
    return os.path.join(root, _BIN_DIRNAME)


def pin_path(root: str) -> str:
    """The shipped pin under ``root``.

    Args:
        root: The plugin root.

    Returns:
        ``<root>/skills/magi/magi-rs.pin.json``.
    """
    return os.path.join(root, _PIN_RELATIVE_PATH)
