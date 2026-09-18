# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-09-18
"""Build the environment handed to the ``magi-rs`` child process.

The environment is built by ALLOWLIST: only what is declared here is inherited,
and everything else -- credentials included -- simply does not arrive.

A denylist would mean tracking, forever, every secret variable ``magi-rs`` (or
any future version of it) decides to read: a point-in-time copy of another
project's behaviour that, the day it goes stale, does not error -- it yields a
run that behaves differently depending on who launched it. An allowlist fails
closed by construction.
"""

from __future__ import annotations

from collections.abc import Mapping

_PLATFORM_VARIABLES = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "TEMP",
    "TMP",
    "TMPDIR",
    "SYSTEMROOT",
    "COMSPEC",
    "LANG",
)
"""Without these the process does not start, or starts and finds nowhere to
write. The locale family is enumerated separately below -- by name, never by
prefix."""

_PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)
"""The operator's network. Without them a remote endpoint is unreachable."""

_CA_VARIABLES = (
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
)
"""An allowlist's own risk is coming up short, and this is the case that bites:
behind a corporate proxy with a private CA, omitting these breaks ALL HTTPS
with a certificate error that does not look like a missing variable."""

_OLLAMA_VARIABLES = ("OLLAMA_HOST",)
"""The daemon endpoint. ENUMERATED, never ``OLLAMA_*``: a prefix match would
re-open the door every time upstream adds a variable."""

_LOCALE_VARIABLES = (
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
    "LC_NUMERIC",
    "LC_TIME",
    "LC_COLLATE",
)
"""ENUMERATED, like everything else. A ``LC_`` prefix match would be the very
wildcard this module refuses for ``OLLAMA_``, and the argument does not change
because the family looks harmless: an allowlist with one prefix rule in it is an
allowlist with one hole in it.

These six are the categories this project needs. The POSIX/glibc family is
LARGER -- ``LC_PAPER``, ``LC_ADDRESS``, ``LC_TELEPHONE`` and others exist -- and
they are left OUT deliberately: none of them changes how ``magi-rs`` formats
anything this plugin reads, and an allowlist grows by evidence of a run that
failed, never by completeness for its own sake."""

ALLOWED_VARIABLES = frozenset(
    _PLATFORM_VARIABLES + _PROXY_VARIABLES + _CA_VARIABLES + _OLLAMA_VARIABLES + _LOCALE_VARIABLES
)
"""The whole allowlist, as one set: what the child may inherit, and nothing else."""


class EnvironmentPolicy:
    """Decides what the ``magi-rs`` child process inherits.

    Deliberately EXCLUDED, and this is a security decision rather than an
    oversight: the dynamic-linker variables (``LD_LIBRARY_PATH``,
    ``DYLD_LIBRARY_PATH``, ``LD_PRELOAD``). They are a code-injection vector --
    they load arbitrary shared objects into a binary that was just verified by
    digest, voiding the guarantee of R4 -- and the ``magi-rs`` releases are Rust
    binaries that do not need them (the ``-compat`` Linux variant exists for
    that). If some platform ever demonstrably needs them, the allowlist grows by
    EVIDENCE and the security decision is re-taken explicitly, never by drip.

    Example:
        >>> import os
        >>> child_env = EnvironmentPolicy.build(os.environ)
    """

    @staticmethod
    def build(parent_env: Mapping[str, str]) -> dict[str, str]:
        """Project the parent environment onto the allowlist.

        Args:
            parent_env: Usually ``os.environ``.

        Returns:
            A new dict holding only the declared variables that were present.
            ``MAGI_PASSPHRASE`` is absent by construction -- it is not declared,
            so it is never copied.

        Side effects:
            None. The parent environment is not modified.

        Complexity:
            O(n) in the number of variables in ``parent_env``, once per launch.
        """
        return {name: value for name, value in parent_env.items() if name in ALLOWED_VARIABLES}

    @staticmethod
    def withheld(parent_env: Mapping[str, str]) -> list[str]:
        """Name the parent variables that did NOT travel.

        Used to make a child failure diagnosable: an allowlist's failure mode
        is a certificate or host error that looks nothing like "a variable was
        missing", so the error message names the allowlist as a suspect and
        lists exactly what it withheld. Names only -- values never leave this
        process.

        Args:
            parent_env: Usually ``os.environ``.

        Returns:
            The sorted names present in ``parent_env`` and absent from the
            allowlist.

        Side effects:
            None.
        """
        return sorted(name for name in parent_env if name not in ALLOWED_VARIABLES)
