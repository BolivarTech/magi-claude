# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-11-21
"""Reachability probe for the embedder that ``magi-rs code-review`` depends on."""

from __future__ import annotations

import json
import tomllib
import urllib.error
import urllib.request
from typing import Any, Callable

from one_line import one_line

EMBEDDER_PROBE_TIMEOUT_SECONDS = 10
EMBEDDINGS_PATH = "/embeddings"
_EMBEDDING_TABLE = "embedding"
_BASE_URL_KEY = "base_url"
_MODEL_KEY = "model"
_PROBE_INPUT = "ping"
_JSON_CONTENT_TYPE = {"Content-Type": "application/json"}

Opener = Callable[..., Any]


class EmbedderUnreachableError(Exception):
    """The embedder endpoint cannot be reached, or its config cannot be resolved.

    Deliberately NOT a subclass of ``ValidationError``: fail-closed, and a retry
    does not bring an endpoint back.
    """


class EmbedderProbe:
    """Checks that the ``[embedding]`` endpoint of ``magi.toml`` answers.

    Any HTTP answer (401, 404, 500 included) counts as reachable: only a
    connection failure or a timeout is fatal. The probe carries no credential
    and never reads the vault; an authentication error is magi-rs's to report.

    Example:
        >>> EmbedderProbe().check("/repo/.magi/magi.toml")
    """

    def __init__(
        self,
        opener: Opener = urllib.request.urlopen,
        timeout: float = EMBEDDER_PROBE_TIMEOUT_SECONDS,
    ) -> None:
        """Create a probe.

        Args:
            opener: Callable ``(request, timeout)`` performing the request.
            timeout: Seconds before the request counts as a failure.
        """
        self._opener = opener
        self._timeout = timeout

    def endpoint(self, config_path: str) -> tuple[str, str]:
        """Resolve the embeddings URL and model from the config.

        ``[embedding].base_url`` wins; otherwise the root ``base_url`` is inherited.

        Args:
            config_path: Path of ``.magi/magi.toml``.

        Returns:
            ``(url, model)``.

        Raises:
            EmbedderUnreachableError: The file cannot be parsed or declares no
                base URL or no embedding model.
        """
        try:
            with open(config_path, "rb") as handle:
                config = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise EmbedderUnreachableError(
                f"cannot read the embedder config {config_path}: {one_line(str(error))}"
            ) from error
        table = config.get(_EMBEDDING_TABLE, {})
        base_url = table.get(_BASE_URL_KEY) or config.get(_BASE_URL_KEY)
        model = table.get(_MODEL_KEY)
        if not base_url or not model:
            raise EmbedderUnreachableError(
                f"{config_path} declares no embedder base_url or [{_EMBEDDING_TABLE}] model"
            )
        return str(base_url).rstrip("/") + EMBEDDINGS_PATH, str(model)

    def check(self, config_path: str) -> None:
        """Send one minimal embedding request.

        Args:
            config_path: Path of ``.magi/magi.toml``.

        Raises:
            EmbedderUnreachableError: Connection failure, timeout, or
                unresolvable config; names the endpoint and the model.
        """
        url, model = self.endpoint(config_path)
        body = json.dumps({"model": model, "input": _PROBE_INPUT}).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=_JSON_CONTENT_TYPE, method="POST")
        try:
            with self._opener(request, timeout=self._timeout) as response:
                response.read()
        except urllib.error.HTTPError:
            return
        except (urllib.error.URLError, OSError) as error:
            raise EmbedderUnreachableError(
                f"embedder {model} is unreachable at {url}: {one_line(str(error))}"
            ) from error
