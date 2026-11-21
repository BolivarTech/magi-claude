#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 3.0.0
# Date: 2026-09-20
"""``--ollama-init``: create the passphrase key, then one ``magi-rs init`` call.

:func:`run_ollama_init` bootstraps and verifies the pinned binary (R4 has no
carve-out for the init command, which executes the binary too), generates the
key, hands it to ``magi-rs init -w <root>`` through the child environment and
only then persists it. A ``.magi/`` that already exists is left untouched:
nothing is called, nothing is written (REQ-12). The key never travels in argv.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from magi_key import KeyStore
from magi_key_permissions import default_key_store
from magi_rs_bootstrap import BinaryBootstrap
from magi_rs_env import EnvironmentPolicy
from magi_rs_paths import bin_dir, pin_path, plugin_root
from magi_rs_pin import Pin

_INIT_SUBCOMMAND = "init"
_WORKDIR_FLAG = "-w"
_INIT_TIMEOUT_SECONDS = 120
"""``magi-rs init`` scaffolds a directory and a small database; two minutes is
far beyond any healthy run and only bounds a hung child."""


_MAGI_DIR = ".magi"
_ALREADY_INITIALISED = (
    f"{_MAGI_DIR}/ already exists in {{workdir}}: left untouched (nothing was initialised, "
    "no key was created)."
)


class OllamaInitError(Exception):
    """``magi-rs init`` failed; the message carries what it printed (a sibling of ``ValidationError``)."""


def _remove_created_magi_dir(magi_dir: str) -> None:
    """Remove a ``.magi/`` this call created, so a retry is not told it is initialised.

    Args:
        magi_dir: The directory to remove; absent is fine.
    """
    shutil.rmtree(magi_dir, ignore_errors=True)


def run_ollama_init(workdir: str, key_store: KeyStore | None = None) -> str:
    """Initialise ``workdir`` for the magi-rs backend with a freshly generated key.

    Args:
        workdir: The project root, passed to ``magi-rs`` as ``-w``.
        key_store: Where the key is persisted; ``None`` selects the platform
            store (:func:`magi_key_permissions.default_key_store`).

    Returns:
        A message saying ``.magi/`` already exists (nothing was done), or what
        ``magi-rs init`` printed (stdout, then stderr), relayed verbatim.

    Raises:
        OllamaInitError: ``magi-rs init`` exited non-zero. No key is persisted
            and the ``.magi/`` this call saw appear is removed.
        KeyHardeningError: The key could not be restricted to its owner. The
            ``.magi/`` this call created is removed.
        UnsafeBinDirectoryError, DigestMismatchError, VersionMismatchError,
        DownloadError, UnsupportedPlatformError, PinFormatError: The binary
            could not be obtained or did not verify. It is NOT executed (R4).
        PluginRootError: ``CLAUDE_PLUGIN_ROOT`` is set but wrong.
        subprocess.TimeoutExpired: ``magi-rs init`` hung past the bound.

    Side effects:
        May download and install the binary; launches ``magi-rs init`` with
        ``MAGI_PASSPHRASE`` in its environment, which creates
        ``<workdir>/.magi/``; writes ``.magi/.secret/magi.key`` on success.
    """
    magi_dir = os.path.join(workdir, _MAGI_DIR)
    if os.path.exists(magi_dir):
        return _ALREADY_INITIALISED.format(workdir=workdir)
    store = key_store if key_store is not None else default_key_store()
    root = plugin_root()
    bootstrap = BinaryBootstrap(Pin.load(pin_path(root)), bin_dir(root))
    bootstrap.ensure()
    executable = bootstrap.verify()
    key = store.new_key()
    completed = subprocess.run(
        [executable, _INIT_SUBCOMMAND, _WORKDIR_FLAG, workdir],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=EnvironmentPolicy.build(os.environ, passphrase=key),
        timeout=_INIT_TIMEOUT_SECONDS,
    )
    output = "".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        _remove_created_magi_dir(magi_dir)
        raise OllamaInitError(output)
    try:
        store.persist(workdir, key)
    except BaseException:
        _remove_created_magi_dir(magi_dir)
        raise
    return output
