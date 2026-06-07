#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-06-06
"""Backend strategy contract for launching a single MAGI agent.

Implementations return the agent's RAW verdict bytes (the JSON text, already
unwrapped from any transport envelope). All downstream processing
(parse -> validate -> consensus) is backend-agnostic and lives in run_magi.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class AgentBackend(ABC):
    """Launches one agent and returns its raw verdict bytes."""

    @abstractmethod
    async def run(
        self,
        agent_name: str,
        system_prompt_path: str,
        prompt: str,
        model: str,
        timeout: int,
        output_dir: str,
    ) -> bytes:
        """Run *agent_name* and return its raw output bytes.

        Args:
            agent_name: One of 'melchior', 'balthasar', 'caspar'.
            system_prompt_path: Path to the agent's system-prompt .md file.
            prompt: The user prompt payload.
            model: Backend-specific model identifier for this agent.
            timeout: Per-agent timeout in seconds.
            output_dir: Directory for backend-specific debug artifacts.

        Returns:
            Raw response bytes to be parsed by ``parse_agent_output``.
        """
        raise NotImplementedError
