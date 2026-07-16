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
from datetime import datetime, timezone
from typing import Any, Callable, NoReturn, cast  # mypy strict: used by annotations below

from backend import AgentBackend
from ollama_config import OllamaConfig

_REDACTED = "***"

#: SINGLE SOURCE OF TRUTH for the transport-failure message contract (MAGI gate,
#: Balthasar). ``_call`` below raises transport failures as RuntimeError -- which has no
#: type to inspect -- so ``run_magi._classify`` recovers the failure CLASS by matching
#: these against the message. They live HERE, next to the messages they must match, so
#: rewording a message and its marker cannot drift apart. ``test_classify_matches_the_real_
#: ollama_backend_messages`` pins the coupling; change a message below AND its marker here.
#: ``TRANSPORT_HTTP_PATTERN`` uses ``HTTP \d`` (a digit MUST follow) so a coding-bug
#: message like "no HTTP status" is NOT misread as transport.
TRANSPORT_HTTP_PATTERN = r"HTTP \d|at chat-time"
TRANSPORT_CONNECTION_MARKERS = ("Cannot reach Ollama",)


class OllamaHTTPError(RuntimeError):
    """A transport HTTP failure, carrying the fields the retry loop's backoff needs.

    Subclasses ``RuntimeError`` so ``run_magi._classify`` still routes it via the
    ``HTTP \\d`` message marker (the message keeps the ``Ollama HTTP <code>``
    shape); the added fields let the retry loop (MS3) pick exponential-vs-flat
    backoff and honor ``Retry-After`` without re-parsing the message.

    Attributes:
        status: HTTP status code from the response.
        retry_after: Raw ``Retry-After`` header value, or None when absent.
        receipt: UTC-aware timestamp captured atomically with reading the
            response, so a later backoff decision measures delay from the
            moment the server actually answered.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None,
        retry_after: str | None,
        receipt: datetime,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.receipt = receipt


def _raise_http_error(exc: urllib.error.HTTPError, redact: Callable[[str], str]) -> NoReturn:
    """Turn a non-404 ``HTTPError`` into an :class:`OllamaHTTPError` and raise it.

    Args:
        exc: The HTTPError raised by ``urllib.request.urlopen``.
        redact: Callable that strips secrets (the API key) from the message
            before it is raised, so a rendered error never leaks ``api_key``.

    Raises:
        OllamaHTTPError: Always -- this function never returns normally.
    """
    # Captured atomically with reading the response: this IS "the point of
    # receipt" the backoff's Retry-After math measures delay from (R2).
    receipt = datetime.now(timezone.utc)
    # An empty headers mapping is falsy, so this also correctly yields None
    # when the server sent no headers at all.
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    # exc.fp is single-consumption (Caspar): read the body ONCE up front.
    detail = exc.read().decode("utf-8", "replace") if exc.fp else ""
    # `from None`: matches the original branch's suppression of Python's implicit
    # exception-chaining note -- the caller sees only the redacted OllamaHTTPError,
    # not "During handling of the above exception...".
    raise OllamaHTTPError(
        redact(f"Ollama HTTP {exc.code}: {exc.reason} {detail}".strip()),
        status=exc.code,
        retry_after=retry_after,
        receipt=receipt,
    ) from None


class OllamaBackend(AgentBackend):
    """Runs an agent via POST {base_url}/chat/completions (no new deps)."""

    def __init__(self, config: OllamaConfig) -> None:
        self._config = config

    def _build_request(self, system_prompt: str, prompt: str, model: str) -> urllib.request.Request:
        # MS2 (R7): no ``response_format`` on the wire. A model constrained to a raw
        # JSON object cannot ALSO emit the ``<MAGI_VERDICT>`` sentinel markers -- the
        # two are mutually exclusive. json_schema also never guaranteed prose
        # suppression in the first place (glm-5.2 fenced its output while
        # response_format was active), so it bought nothing the sentinel needs.
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
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
            if exc.code == 404:
                # exc.fp is single-consumption (Caspar): read the body ONCE up front.
                detail = exc.read().decode("utf-8", "replace") if exc.fp else ""
                raise RuntimeError(
                    self._redact(
                        f"Ollama 404 at chat-time: model unavailable ({exc.reason}). "
                        f"Preflight passed — possible ollama rm / auth expiry / TOCTOU. {detail}".strip()
                    )
                ) from None
            # MS3: every OTHER HTTP failure (transitory or not) carries status /
            # Retry-After / receipt via OllamaHTTPError, so the retry loop can pick
            # exponential-vs-flat backoff and honor Retry-After (R2, R5). 404 stays a
            # plain RuntimeError above: it is neither reintentable nor transitory.
            _raise_http_error(exc, self._redact)
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
            Raw UTF-8 bytes of ``choices[0].message.content`` verbatim (may carry
            the ``<MAGI_VERDICT>``/``</MAGI_VERDICT>`` sentinel markers, prose, or
            a bare JSON verdict), ready for ``parse_agent_output``.

        Raises:
            TimeoutError: When the HTTP request exceeds *timeout* seconds.
            RuntimeError: On HTTP errors (4xx/5xx) or connection failures.
            ValueError: When the response envelope lacks the expected shape.
        """
        with open(system_prompt_path, encoding="utf-8") as f:
            system_prompt = f.read()
        req = self._build_request(system_prompt, prompt, model)
        body = await asyncio.to_thread(self._call, req, timeout)
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
