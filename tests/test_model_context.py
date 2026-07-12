# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-11
"""Tests for model window/capability pre-loading (Task 6, R18/R19/R20)."""

import pytest

from fallback_policy import ModelCapability
from model_context import fetch_capabilities
from ollama_config import ModelSpec, OllamaConfig


@pytest.fixture
def ollama_config():
    """A minimal resolved config. ``retry_backoff_seconds=0`` keeps retry tests fast."""
    return OllamaConfig(
        base_url="http://localhost:11434/v1",
        api_key=None,
        models={"melchior": ModelSpec("m", "l")},
        retry_backoff_seconds=0.0,
    )


class FakeShow:
    """Stand-in for POST /api/show, keyed by model, with scripted failures."""

    def __init__(self, table, fail=()):
        self.table = table
        self.fail = set(fail)
        self.calls: list[str] = []

    async def __call__(self, model, timeout):
        self.calls.append(model)
        if model in self.fail:
            raise OSError("boom")
        return self.table[model]


async def test_reads_window_and_completion_capability(ollama_config):
    show = FakeShow({"a": {"capabilities": ["completion", "thinking"], "context_length": 128_000}})
    caps = await fetch_capabilities(ollama_config, ["a"], _show=show)
    assert caps["a"] == ModelCapability(window=128_000, supports_completion=True)


async def test_embedding_model_is_flagged_as_not_completion(ollama_config):
    # BDD-47: nomic-embed appears in /models like any other model.
    show = FakeShow({"e": {"capabilities": ["embedding"], "context_length": 8_192}})
    caps = await fetch_capabilities(ollama_config, ["e"], _show=show)
    assert caps["e"].supports_completion is False


async def test_missing_capabilities_field_yields_unknown_not_false(ollama_config):
    show = FakeShow({"a": {"context_length": 128_000}})  # old Ollama
    caps = await fetch_capabilities(ollama_config, ["a"], _show=show)
    assert caps["a"].supports_completion is True  # warn-and-proceed, not a hard no
    assert caps["a"].window == 128_000


async def test_show_failure_yields_unknown_window_not_an_abort(ollama_config):
    # BDD-28
    show = FakeShow({"a": {"capabilities": ["completion"], "context_length": 1}}, fail={"a"})
    caps = await fetch_capabilities(ollama_config, ["a"], _show=show)
    assert caps["a"].window is None


async def test_transient_failure_is_retried_before_being_believed(ollama_config):
    calls = {"n": 0}

    async def flaky(model, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient blip")
        return {"capabilities": ["completion"], "context_length": 64_000}

    caps = await fetch_capabilities(ollama_config, ["a"], _show=flaky)
    assert caps["a"].window == 64_000  # strict must be strict about EVIDENCE,
    assert calls["n"] == 2  # not about bad luck


async def test_one_model_failing_does_not_discard_the_others(ollama_config):
    # gather(return_exceptions=True), NOT TaskGroup: the calls are independent.
    show = FakeShow(
        {
            "a": {"capabilities": ["completion"], "context_length": 1000},
            "b": {"capabilities": ["completion"], "context_length": 2000},
        },
        fail={"a"},
    )
    caps = await fetch_capabilities(ollama_config, ["a", "b"], _show=show)
    assert caps["a"].window is None
    assert caps["b"].window == 2000  # useful result NOT thrown away
