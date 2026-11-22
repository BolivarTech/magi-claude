# Author: Julian Bolivar
# Version: 8.0.0
# Date: 2026-11-21
"""What the two magi-rs executors share: raw-artifact persistence, verdict validation, status
forwarding and the failure-message helpers.

``MagiRsTrio`` (``consult``) and ``MagiRsCodeReview`` (``code-review``) both launch one
magi-rs child and turn its envelope into three validated verdicts. Everything they have in
common after the launch lives here under public names, so neither executor reaches into the
other's private members.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import Any

from cli_commands import VAULT_SET_FLAG, command
from magi_rs_env import PASSPHRASE_VARIABLE, EnvironmentPolicy
from magi_rs_result import ConsultResult
from synthesize import ValidationError, load_agent_output

__all__ = [
    "MAGES",
    "STDERR_ELISION_MARKER",
    "VAULT_HINT",
    "allowlist_hint",
    "emit_state",
    "persist_raw",
    "stderr_tail",
    "validate_verdicts",
    "write_raw_artifact",
]

RAW_ARTIFACT = "magi-rs-raw.json"
#: The three seats. A mage magi-rs leaves out of ``agents`` (it is in
#: ``failed_agents`` instead) still gets an entry: its failure.
MAGES = ("melchior", "balthasar", "caspar")
NO_VERDICT_CAUSE = "absent from agents and from failed_agents"
VAULT_HINT = (
    "If this is an authentication failure, load the API keys the config needs with "
    f"`{command(VAULT_SET_FLAG, '<NAME>')}`."
)
"""Every non-zero exit carries it: magi-rs offers no free check for a missing key (REQ-13d)."""
STDERR_EXCERPT_CHARS = 2000
STDERR_ELISION_MARKER = "[... earlier stderr elided ...] "
"""Prefixed to a failure's stderr excerpt when its head was cut. Since 0.20 magi-rs
writes WARN lines at startup and the real cause of a failure comes last, so the
excerpt keeps the tail (v6.2.0 R14); the marker says the head is missing."""


def write_raw_artifact(output_dir: str, raw: str | bytes) -> str:
    """Persist magi-rs's stdout VERBATIM, under its own name (R16).

    Two producers under one name is ambiguity nobody disambiguates later, so
    this writes ``magi-rs-raw.json`` and never ``magi-report.json``.

    Args:
        output_dir: The run's output directory.
        raw: The child's stdout, exactly as received. NOT re-serialised -- a
            round-trip through json would silently normalise key order and
            float formatting, and this file is the audit artifact. ``str`` is
            written as UTF-8.

    Returns:
        The path written.

    Raises:
        OSError: The file cannot be written. NOT swallowed here: the strict
            primitive. :func:`persist_raw` is the fail-soft caller
            that turns it into a warning, because an audit artifact must not
            be able to destroy the verdicts it records.

    Side effects:
        Writes one file.
    """
    path = os.path.join(output_dir, RAW_ARTIFACT)
    payload = raw.encode("utf-8") if isinstance(raw, str) else raw
    with open(path, "wb") as handle:
        handle.write(payload)
    return path


def emit_state(on_state: Callable[[str, str], None] | None, node: str, state: str) -> None:
    """Forward one status transition, tolerating a callback that raises.

    Args:
        on_state: The caller's callback, or None when status is off.
        node: The node's name. ONE node in this executor -- see run_trio.
        state: ``running``, ``success``, ``failed`` or ``timeout``.

    Returns:
        None.

    Note:
        A status display that raises must not take the run down with it: the
        verdicts are the product, the tree is decoration. Same fail-soft
        contract ``_safe_display_update`` already has on the Claude path.
    """
    if on_state is None:
        return
    try:
        on_state(node, state)
    except Exception as exc:  # noqa: BLE001 - fail-soft by contract (see Note)
        print(f"WARNING: status display failed for {node!r}: {exc}", file=sys.stderr)


def stderr_tail(stderr: bytes) -> str:
    """Return the last ``STDERR_EXCERPT_CHARS`` characters of a child's stderr.

    Args:
        stderr: The raw bytes the child wrote; need not be valid UTF-8.

    Returns:
        The decoded text (undecodable bytes replaced), whole when it fits, else
        its tail prefixed with ``STDERR_ELISION_MARKER``.

    Example:
        >>> stderr_tail(b"boom")
        'boom'

    Note:
        The tail, not the head: magi-rs's startup WARN lines come first and the
        cause of a failure comes last (v6.2.0 R14). O(n) in the stderr size.
    """
    text = stderr.decode("utf-8", errors="replace")
    if len(text) <= STDERR_EXCERPT_CHARS:
        return text
    return STDERR_ELISION_MARKER + text[-STDERR_EXCERPT_CHARS:]


def allowlist_hint() -> str:
    """The diagnosis every process failure carries: what the allowlist withheld.

    Returns:
        One sentence naming the allowlist as a suspect and listing the
        parent variables (names only, never values) that did not travel;
        ``MAGI_PASSPHRASE`` is excluded because the plugin injects it.
    """
    # The plugin injects MAGI_PASSPHRASE itself, so it is never "withheld" from the child.
    withheld = [
        name for name in EnvironmentPolicy.withheld(os.environ) if name != PASSPHRASE_VARIABLE
    ]
    return (
        "The child ran with an allowlisted environment (see magi_rs_env.py); if the "
        "failure looks like a missing certificate, proxy or host, the allowlist is "
        f"a suspect -- it withheld: {withheld}"
    )


def persist_raw(output_dir: str, stdout: bytes) -> None:
    """Write the raw envelope under its own name (R16); fail-soft.

    Args:
        output_dir: Destination directory.
        stdout: The child's stdout, verbatim.

    Returns:
        None.

    Side effects:
        Writes ``magi-rs-raw.json``; a failed write only warns, because an
        audit artifact must not be able to destroy the verdicts it records.
    """
    try:
        write_raw_artifact(output_dir, stdout)
    except OSError as exc:
        print(f"WARNING: failed to persist {RAW_ARTIFACT}: {exc}", file=sys.stderr)


def validate_verdicts(
    result: ConsultResult, output_dir: str
) -> dict[str, dict[str, Any] | BaseException]:
    """Push each verdict through the EXISTING validator, unchanged.

    Each verdict is written to ``{mage}.json`` -- the same per-mage artifact
    the Claude path leaves -- and read back by ``load_agent_output``, whose
    contract is a file path.

    Args:
        result: The parsed consult result.
        output_dir: Where the per-mage files are written.

    Returns:
        One entry per mage of the trio: the validated dict, the
        ``ValidationError`` it failed with, or -- for a mage magi-rs did not
        deliver at all -- a ``RuntimeError`` carrying magi-rs's own cause
        from ``failed_agents`` (the shape the orchestrator's degraded path
        already handles, BDD-15/D7). A verdict that breaks the seven-key
        contract is rejected EXACTLY as today (BDD-14c) -- no new validation
        code, and no relaxing of the contract.

    Side effects:
        Writes one ``{mage}.json`` per verdict.
    """
    failed = result.telemetry.get("failed_agents") or {}
    verdicts: dict[str, dict[str, Any] | BaseException] = {
        mage: RuntimeError(
            f"magi-rs delivered no verdict for {mage}: "
            f"{failed.get(mage, NO_VERDICT_CAUSE) if isinstance(failed, dict) else failed}"
        )
        for mage in MAGES
    }
    for index, verdict in enumerate(result.agents):
        name = verdict.get("agent") if isinstance(verdict, dict) else None
        # The name is data from the child: only a seat of the trio may name
        # the artifact, so a crafted name cannot write outside output_dir.
        mage = name if name in MAGES else f"agent-{index}"
        path = os.path.join(output_dir, f"{mage}.json")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(verdict, handle, ensure_ascii=False, indent=2)
            verdicts[mage] = load_agent_output(path)
        except (ValidationError, json.JSONDecodeError, OSError, UnicodeEncodeError) as exc:
            # UnicodeEncodeError: json.loads yields a lone surrogate from a
            # JSON escape that the UTF-8 file write cannot take -- that
            # seat's failure, never the run's.
            verdicts[mage] = exc
    return verdicts
