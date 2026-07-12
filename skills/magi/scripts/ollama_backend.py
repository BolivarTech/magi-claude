#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-06-06
"""OpenAI-compatible (Ollama) backend over stdlib urllib."""

from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.request
from typing import Any, cast  # mypy strict: used by dict[str, Any] annotations below

from agent_schema import AGENT_OUTPUT_JSON_SCHEMA
from backend import AgentBackend
from ollama_config import OllamaConfig

_REDACTED = "***"


class _ResponseFormatRejected(Exception):
    """Internal signal: server returned 400 rejecting response_format -> R15 downgrade."""


class OllamaBackend(AgentBackend):
    """Runs an agent via POST {base_url}/chat/completions (no new deps)."""

    def __init__(self, config: OllamaConfig) -> None:
        self._config = config

    def _response_format(self) -> dict[str, Any] | None:
        # R16: structured ∈ {"schema","object","off"}.
        if self._config.structured == "off":
            return None
        if self._config.structured == "object":
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "magi_agent_output",
                "strict": False,  # R7/F-A: portability over strict enforcement
                "schema": AGENT_OUTPUT_JSON_SCHEMA,
            },
        }

    def _build_request(
        self, system_prompt: str, prompt: str, model: str, *, with_format: bool = True
    ) -> urllib.request.Request:
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        rf = self._response_format() if with_format else None
        if rf is not None:
            body["response_format"] = rf
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        url = f"{self._config.base_url}/chat/completions"
        return urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )

    def _redact(self, text: str) -> str:
        key = self._config.api_key
        return text.replace(key, _REDACTED) if key else text

    def _call(self, req: urllib.request.Request, timeout: int) -> bytes:
        # CONTRACT (MAGI gate, Balthasar): the RuntimeError message FORMATS below --
        # "Ollama HTTP {code}", "Ollama 404 at chat-time", "Cannot reach Ollama at ..." --
        # are matched by ``run_magi._classify`` (via ``_HTTP_MESSAGE_RE`` /
        # ``_CONNECTION_MESSAGE_MARKERS``) to tell a transport failure from a coding bug.
        # Reword a message here and you MUST update those markers + the pinning test
        # ``test_classify_matches_the_real_ollama_backend_messages``, or a real transport
        # error silently becomes "unexpected" and the mage dies on its first attempt.
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return cast(bytes, resp.read())
        except urllib.error.HTTPError as exc:
            # exc.fp is single-consumption (Caspar): read the body ONCE up front.
            detail = exc.read().decode("utf-8", "replace") if exc.fp else ""
            low = detail.lower()
            # R15 downgrade trigger: 400 rejecting response_format/json_schema.
            # Trigger only on the specific field tokens — NOT bare "schema"
            # (Caspar ciclo 3: bare "schema" over-matches unrelated 400s).
            if exc.code == 400 and ("response_format" in low or "json_schema" in low):
                raise _ResponseFormatRejected() from None
            if exc.code == 404:
                raise RuntimeError(
                    self._redact(
                        f"Ollama 404 at chat-time: model unavailable ({exc.reason}). "
                        f"Preflight passed — possible ollama rm / auth expiry / TOCTOU. {detail}".strip()
                    )
                ) from None
            raise RuntimeError(
                self._redact(f"Ollama HTTP {exc.code}: {exc.reason} {detail}".strip())
            ) from None
        except (socket.timeout, TimeoutError) as exc:
            raise TimeoutError(self._redact(f"Ollama request timed out: {exc}")) from None
        except urllib.error.URLError as exc:
            raise RuntimeError(
                self._redact(f"Cannot reach Ollama at {self._config.base_url}: {exc.reason}")
            ) from None

    async def run(
        self,
        agent_name: str,
        system_prompt_path: str,
        prompt: str,
        model: str,
        timeout: int,
        output_dir: str,
    ) -> bytes:
        """Run *agent_name* against the Ollama-compatible endpoint and return verdict bytes.

        Args:
            agent_name: One of 'melchior', 'balthasar', 'caspar'.
            system_prompt_path: Path to the agent's system-prompt .md file.
            prompt: The user prompt payload.
            model: Model identifier for this agent (passed verbatim to the API).
            timeout: Per-agent HTTP timeout in seconds.
            output_dir: Directory for debug artifacts (unused; present for ABC compat).

        Returns:
            Raw UTF-8 bytes of ``choices[0].message.content`` — the JSON verdict
            string produced by the model, ready for ``parse_agent_output``.

        Raises:
            TimeoutError: When the HTTP request exceeds *timeout* seconds.
            RuntimeError: On HTTP errors (4xx/5xx) or connection failures.
            ValueError: When the response envelope lacks the expected shape.
        """
        with open(system_prompt_path, encoding="utf-8") as f:
            system_prompt = f.read()
        req = self._build_request(system_prompt, prompt, model)
        try:
            body = await asyncio.to_thread(self._call, req, timeout)
        except _ResponseFormatRejected:
            # R15 single-shot downgrade: retry once WITHOUT response_format; the
            # backstop parser/retry enforces the 7-key contract from the content.
            req2 = self._build_request(system_prompt, prompt, model, with_format=False)
            body = await asyncio.to_thread(self._call, req2, timeout)
        try:
            envelope = json.loads(body)
            content = envelope["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected OpenAI-compatible response shape: {exc}") from exc
        # R-B: some OpenAI-compatible servers decode message.content into a
        # dict before serializing the response.  str(dict) produces a Python
        # repr (single-quoted), which is not valid JSON.  Serialize dicts with
        # json.dumps; leave strings as-is.
        text = json.dumps(content) if isinstance(content, dict) else str(content)
        return text.encode("utf-8")
