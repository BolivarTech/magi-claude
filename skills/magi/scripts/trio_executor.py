#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-29
"""Trio-level execution contract: run the three mages, return validated verdicts.

The abstraction sits at the TRIO level, not at the individual mage, because the
two transports have different shapes: `claude -p` is three concurrent
subprocesses, while `magi-rs consult` is one process that answers for all three.
A per-mage contract can only express the first honestly.

The contract covers VALIDATED verdicts rather than raw bytes, and the reason is
the retry. The Claude transport re-prompts a single mage with the sentinel's
cause-specific feedback when its verdict fails the schema, so it must know what
a valid verdict is; the magi-rs transport has already spent its own retries
before it answers. Either way the caller receives three verdicts that passed
`load_agent_output`, and never has to know which transport retried what.

Everything downstream -- consensus, reporting, artifacts -- is transport-agnostic
and lives in run_magi.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any


class TrioExecutor(ABC):
    """Runs the three mages for one payload and returns their validated verdicts."""

    @abstractmethod
    async def run_trio(
        self,
        prompt: str,
        agents_dir: str,
        output_dir: str,
        timeout: int,
        models: Mapping[str, str],
        max_attempts: int,
        on_state: Callable[[str, str], None] | None = None,
    ) -> dict[str, dict[str, Any] | BaseException]:
        """Run all three mages and return their validated verdicts.

        Args:
            prompt: The user prompt payload, identical for the three mages.
            agents_dir: Directory holding the ``{mage}.md`` system prompts.
            output_dir: Directory for raw and per-mage debug artifacts.
            timeout: Per-mage timeout in seconds.
            models: Mapping of agent name to model short name (e.g. ``"opus"``),
                already extracted from any richer spec by the caller.
            max_attempts: Attempts per mage before giving up. A transport that
                retries internally ignores this.
            on_state: Optional callback ``(agent_name, state)`` for a live status
                display, where state is one of ``running``, ``retrying``,
                ``success``, ``timeout``, ``failed``. It must not raise; a
                display glitch must never cost a verdict.

        Returns:
            Mapping of agent name to either the validated verdict dict (the
            seven-key contract) or the exception that killed that mage. There is
            always one entry per mage in *models*.

            Failures are returned rather than raised, and the exact exception
            object is preserved rather than wrapped: the caller branches on its
            subtype -- re-raising ``_EndpointDown``, re-raising ``CancelledError``
            on the rotation path, and letting ``KeyboardInterrupt`` and
            ``SystemExit`` propagate while treating every other exception as a
            degraded agent. Collapsing them into one type would swallow a fatal
            signal, and the caller's warning prints the exception text, so a
            wrapper would also cost the diagnosis.

        Raises:
            Nothing per-mage. A failure in the transport's own setup -- before any
            mage runs -- still propagates.
        """
        ...
