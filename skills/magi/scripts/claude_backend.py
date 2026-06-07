#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-06-06
"""Claude CLI backend: shells out to `claude -p` (default MAGI behavior)."""

from __future__ import annotations

import asyncio
import sys

from backend import AgentBackend
from models import resolve_model
from subprocess_utils import (
    format_stderr_excerpt as _format_stderr_excerpt,
    reap_and_drain_stderr as _reap_and_drain_stderr,
    write_stderr_log as _write_stderr_log,
)


class ClaudeBackend(AgentBackend):
    """Runs an agent via the `claude -p` subprocess (unchanged 3.x behavior)."""

    async def run(
        self,
        agent_name: str,
        system_prompt_path: str,
        prompt: str,
        model: str,
        timeout: int,
        output_dir: str,
    ) -> bytes:
        model_id = resolve_model(model)
        proc = await asyncio.create_subprocess_exec(
            "claude",
            "-p",
            "--output-format",
            "json",
            "--model",
            model_id,
            "--system-prompt-file",
            system_prompt_path,
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")), timeout=timeout
            )
        except asyncio.TimeoutError:
            stderr_buffered = await _reap_and_drain_stderr(proc)
            try:
                _write_stderr_log(output_dir, agent_name, stderr_buffered)
            except OSError as log_exc:
                print(
                    f"WARNING: Failed to persist {agent_name}.stderr.log on timeout: {log_exc}",
                    file=sys.stderr,
                )
            raise TimeoutError(
                f"Agent '{agent_name}' timed out after {timeout}s"
                f"{_format_stderr_excerpt(stderr_buffered)}"
            ) from None

        try:
            _write_stderr_log(output_dir, agent_name, stderr)
        except OSError as log_exc:
            print(
                f"WARNING: Failed to persist {agent_name}.stderr.log: {log_exc}",
                file=sys.stderr,
            )

        if proc.returncode != 0:
            stderr_text = (
                stderr.decode("utf-8", errors="replace").strip() if stderr else "no stderr"
            )
            raise RuntimeError(
                f"Agent '{agent_name}' exited with code {proc.returncode}: {stderr_text}"
            )
        return stdout
