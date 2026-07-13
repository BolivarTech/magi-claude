# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-11
"""Shared fixtures for the MAGI test suite (config builders + preflight I/O patches)."""

import shutil
from pathlib import Path

import pytest

from fallback_policy import ModelCapability
from ollama_config import (
    DEFAULT_FALLBACK,
    DEFAULT_MODELS,
    ModelSpec,
    OllamaConfig,
    _normalise_lineage,
)

#: The three prompts the plugin actually ships.
_AGENTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "magi" / "agents"


@pytest.fixture(autouse=True)
def seeded_agents_dir(tmp_path):
    """Give every test a ``tmp_path`` that is a REAL agents dir (the shipped prompts).

    Almost every orchestrator test passes ``agents_dir=str(tmp_path)`` -- and until the
    prompt guard moved into ``run_orchestrator`` (MAGI gate, Balthasar), that directory was
    EMPTY: the tests exercised a path no user can ever take. Seeding the real prompts is not
    scaffolding to keep them green; it is what makes them exercise the contract they claim
    to, guard included. A test whose agents dir has no agents was never testing the run.

    Yields:
        The seeded ``tmp_path``.
    """
    for prompt in _AGENTS_DIR.glob("*.md"):
        shutil.copy(prompt, tmp_path / prompt.name)
    return tmp_path


def _normalised(specs):
    """Rebuild each ModelSpec with a normalised lineage, mirroring resolve_config.

    Production configs pass through ``_parse_model_spec`` which normalises the
    lineage; a config built directly in a test must do the same, or a mixed-case
    duplicate ("alibaba" vs "ALIBABA") would slip past the R22 clash guard.
    """
    return {name: ModelSpec(s.model, _normalise_lineage(s.lineage)) for name, s in specs.items()}


@pytest.fixture
def config_factory():
    """Return a builder for :class:`OllamaConfig`, overriding only real config fields."""

    def _make(**overrides):
        models = overrides.pop("models", dict(DEFAULT_MODELS))
        fallback = overrides.pop("fallback", tuple(DEFAULT_FALLBACK))
        base = dict(
            base_url="http://localhost:11434/v1",
            api_key=None,
            models=_normalised(models),
            fallback=tuple(ModelSpec(f.model, _normalise_lineage(f.lineage)) for f in fallback),
        )
        base.update(overrides)
        return OllamaConfig(**base)

    return _make


@pytest.fixture
def ollama_config(config_factory):
    """A default resolved config; ``retry_backoff_seconds=0`` keeps retry tests fast."""
    return config_factory(retry_backoff_seconds=0.0)


@pytest.fixture
def preflight_env(monkeypatch):
    """Patch the preflight's three I/O boundaries. Returns a knob to set each one."""
    state = {
        "available": {s.model for s in DEFAULT_MODELS.values()}
        | {f.model for f in DEFAULT_FALLBACK},
        "caps": {},  # model -> ModelCapability; a default is filled in below
        "probe": 16_232,  # exact tokens (int), None ("unmeasurable"), or {model: count}
    }

    async def fake_list(config):
        return state["available"]

    async def fake_caps(config, models, **kw):
        default = ModelCapability(window=200_000, supports_completion=True)
        return {m: state["caps"].get(m, default) for m in models}

    async def fake_probe(config, model, prompt, **kw):
        p = state["probe"]
        return p.get(model) if isinstance(p, dict) else p

    monkeypatch.setattr("ollama_preflight._list_models", fake_list)
    monkeypatch.setattr("ollama_preflight.fetch_capabilities", fake_caps)
    monkeypatch.setattr("ollama_preflight.probe_prompt_tokens", fake_probe)
    return state
