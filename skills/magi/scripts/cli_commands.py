# Author: Julian Bolivar
# Version: 7.1.0
# Date: 2026-11-21
"""The one home of the key subcommands' flag literals and of how a message names them.

``run_magi``'s argument parser and every operator-facing message (key custody,
preflight, vault hints) build from these, so a flag and the text that tells an
operator to run it cannot drift apart.
"""

from __future__ import annotations

INIT_FLAG = "--ollama-init"
VAULT_SET_FLAG = "--ollama-vault-set"
ROTATE_KEY_FLAG = "--ollama-rotate-key"
YES_FLAG = "--yes"
"""The key subcommands and the flag that authorises a rotation."""

_COMMAND_PREFIX = "run_magi.py"
"""How every message names the script (the ``/magi`` skill form is documented separately)."""


def command(flag: str, argument: str = "") -> str:
    """Return how a message names a command: ``run_magi.py <flag> [<argument>]``.

    Args:
        flag: One of the flag constants of this module.
        argument: A placeholder or value that follows the flag, such as ``<NAME>``.

    Returns:
        The command, on one line.

    Example:
        >>> command(VAULT_SET_FLAG, "<NAME>")
        'run_magi.py --ollama-vault-set <NAME>'
    """
    parts = [_COMMAND_PREFIX, flag]
    if argument:
        parts.append(argument)
    return " ".join(parts)
