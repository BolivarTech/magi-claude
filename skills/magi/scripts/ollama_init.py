#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 3.0.0
# Date: 2026-09-20
"""``--ollama-init``: one call to ``magi-rs init`` and nothing else (v6.0.0).

:func:`run_ollama_init` is the whole command. It bootstraps and verifies the
pinned binary first -- R4 has no carve-out for the init command, which
executes the binary too -- then invokes ``magi-rs init -w <root>`` with the
allowlisted environment and relays whatever it said. It writes no file of its
own, reads nothing under ``.claude/``, and passes no ``-p``: there is no
passphrase, no key and no plugin config any more (R5, R5a, R5b).
"""

from __future__ import annotations

import os
import subprocess

from magi_rs_bootstrap import BinaryBootstrap
from magi_rs_env import EnvironmentPolicy
from magi_rs_paths import bin_dir, pin_path, plugin_root
from magi_rs_pin import Pin

_INIT_SUBCOMMAND = "init"
_WORKDIR_FLAG = "-w"
_INIT_TIMEOUT_SECONDS = 120
"""``magi-rs init`` scaffolds a directory and a small database; two minutes is
far beyond any healthy run and only bounds a hung child."""


def run_ollama_init(workdir: str) -> str:
    """Initialise ``workdir`` for the magi-rs backend, by delegating entirely.

    Args:
        workdir: The project root, passed to ``magi-rs`` as ``-w``.

    Returns:
        What ``magi-rs init`` printed (stdout, then stderr), relayed verbatim.
        A refusal (non-zero exit) is relayed the same way: the requirement is
        on this side -- finish cleanly whatever magi-rs decided -- and never
        force anything (BDD-7b).

    Raises:
        UnsafeBinDirectoryError, DigestMismatchError, VersionMismatchError,
        DownloadError, UnsupportedPlatformError, PinFormatError: The binary
            could not be obtained or did not verify. It is NOT executed (R4).
        PluginRootError: ``CLAUDE_PLUGIN_ROOT`` is set but wrong.
        subprocess.TimeoutExpired: ``magi-rs init`` hung past the bound.

    Side effects:
        May download and install the binary; launches ``magi-rs init``, which
        creates ``<workdir>/.magi/``. Writes nothing itself and never touches
        ``.claude/``.
    """
    root = plugin_root()
    bootstrap = BinaryBootstrap(Pin.load(pin_path(root)), bin_dir(root))
    bootstrap.ensure()
    executable = bootstrap.verify()
    completed = subprocess.run(
        [executable, _INIT_SUBCOMMAND, _WORKDIR_FLAG, workdir],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=EnvironmentPolicy.build(os.environ),
        timeout=_INIT_TIMEOUT_SECONDS,
    )
    return "".join(part for part in (completed.stdout, completed.stderr) if part)
