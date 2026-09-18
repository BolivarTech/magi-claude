# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-09-18
"""Check that a project has been initialised for the magi-rs backend."""

from __future__ import annotations

import os

_CONFIG_RELATIVE_PATH = os.path.join(".magi", "magi.toml")
_INIT_COMMAND = "/magi --ollama-init"


class ProjectNotInitialisedError(Exception):
    """The project has no ``.magi/magi.toml``.

    Deliberately NOT a subclass of ``ValidationError``: fail-closed, and a
    retry does not create the file.
    """


class ProjectPreflight:
    """Verifies the ONE precondition the plugin still owns.

    The check is PRESENCE, not validity: ``magi-rs`` owns that file's contents
    and already fails fatally on a toml that does not parse. Validating it here
    would reimplement the parser this release deletes -- and worse, two
    validators of one file diverge, and the day they do, ours is the one lying.

    Example:
        >>> config = ProjectPreflight.require_initialised("/repo")
    """

    @staticmethod
    def require_initialised(workdir: str) -> str:
        """Assert the project is initialised.

        Args:
            workdir: Project root, the same path passed to ``magi-rs`` as ``-w``.

        Returns:
            Absolute path of ``.magi/magi.toml``.

        Raises:
            ProjectNotInitialisedError: The file is absent. The message names
                the exact command that creates it.

        Side effects:
            Reads the filesystem. Writes NOTHING -- a gate run never
            initialises a project on the fly.
        """
        config = os.path.join(workdir, _CONFIG_RELATIVE_PATH)
        if not os.path.isfile(config):
            raise ProjectNotInitialisedError(
                f"{workdir} is not initialised for magi-rs ({config} is absent); "
                f"run `{_INIT_COMMAND}` in that project first. A gate run never "
                "initialises a project on its own."
            )
        return config
