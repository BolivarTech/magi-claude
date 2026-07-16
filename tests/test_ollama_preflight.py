# Author: Julian Bolivar
# Version: 2.0.0
# Date: 2026-07-11
"""Preflight tests: /models listing (_list_models) + the v5 guard enforcement."""

import io
import json
import urllib.error

import pytest

from fallback_policy import ModelCapability
from ollama_config import ModelSpec
from ollama_preflight import (
    CONTEXT_GUARD_ENFORCED,
    CONTEXT_GUARD_ESTIMATED,
    ContextWindowUnmeasurableError,
    DigestCollisionError,
    FamilyContradictionError,
    MissingDigestError,
    OllamaPreflightError,
    PreflightResult,
    _is_cloud_tag,
    _list_models,
    preflight,
)
from validate import ValidationError


@pytest.mark.parametrize(
    "exc",
    [
        DigestCollisionError,
        FamilyContradictionError,
        ContextWindowUnmeasurableError,
        MissingDigestError,
    ],
)
def test_ms4_exceptions_are_preflight_errors(exc):
    """Every MS4 preflight exception is an OllamaPreflightError (and thus a
    ValidationError), so the orchestrator's fail-closed handling catches them (R9)."""
    assert issubclass(exc, OllamaPreflightError)
    assert issubclass(exc, ValidationError)


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _models_body(ids):
    return json.dumps({"data": [{"id": i} for i in ids]}).encode()


def _patch(monkeypatch, *, body=None, exc=None):
    cap = {}

    def fake_urlopen(req, timeout=None):
        cap["url"], cap["timeout"] = req.full_url, timeout
        if exc is not None:
            raise exc
        return _Resp(body)

    monkeypatch.setattr("ollama_preflight.urllib.request.urlopen", fake_urlopen)
    return cap


# --------------------------------------------------------------------------
# _list_models: reachability / auth / 404 (the /models half, now its own fn).
# --------------------------------------------------------------------------


async def test_list_models_returns_available_ids(config_factory, monkeypatch):
    cap = _patch(monkeypatch, body=_models_body(["m", "b", "c", "x"]))
    available = await _list_models(config_factory())
    assert available == {"m", "b", "c", "x"}
    assert cap["url"].endswith("/models")


async def test_list_models_honors_the_configured_preflight_timeout(config_factory, monkeypatch):
    """MAGI gate (Balthasar): the /models call must use config.preflight_timeout_seconds,
    not a hardcoded value -- a slow NAS configured for 45s was silently cut at 10s."""
    cap = _patch(monkeypatch, body=_models_body(["m"]))
    await _list_models(config_factory(preflight_timeout_seconds=45))
    assert cap["timeout"] == 45


async def test_list_models_auth_error_redacts_key(config_factory, monkeypatch):
    _patch(monkeypatch, exc=urllib.error.HTTPError("u", 401, "Unauthorized", {}, None))
    with pytest.raises(OllamaPreflightError) as ei:
        await _list_models(config_factory(api_key="sk-secret"))
    assert "sk-secret" not in str(ei.value)


async def test_list_models_unreachable_aborts(config_factory, monkeypatch):
    _patch(monkeypatch, exc=urllib.error.URLError("refused"))
    with pytest.raises(OllamaPreflightError):
        await _list_models(config_factory())


async def test_list_models_404_warns_and_treats_all_as_present(config_factory, monkeypatch, capsys):
    # Listing unsupported: warn and return every configured tag, so nothing is
    # flagged missing (preserves the pre-v5 warn-and-proceed behaviour).
    _patch(monkeypatch, exc=urllib.error.HTTPError("u", 404, "NF", {}, None))
    cfg = config_factory()
    available = await _list_models(cfg)
    assert "models" in capsys.readouterr().err.lower()
    assert {s.model for s in cfg.models.values()} <= available


def test_is_cloud_tag_rejects_non_suffix_cloud():
    assert _is_cloud_tag("glm-5:cloud") is True
    assert _is_cloud_tag("gpt-oss:120b-cloud") is True
    assert _is_cloud_tag("foo:precloud") is False
    assert _is_cloud_tag("llama3.1:8b") is False


# --------------------------------------------------------------------------
# preflight: the v5 guards (R11/R19/R21/R22/R5b/R18).
# --------------------------------------------------------------------------


async def test_a_lineage_clash_survives_a_change_of_case(config_factory, preflight_env):
    """Normalisation is what makes R22 real: 'ALIBABA' and 'alibaba' must clash."""
    cfg = config_factory(
        models={
            "melchior": ModelSpec("qwen3.5:397b-cloud", "alibaba"),
            "balthasar": ModelSpec("qwen3-coder:480b-cloud", "ALIBABA"),  # normalised -> clash
            "caspar": ModelSpec("deepseek-v4-pro:cloud", "deepseek"),
        }
    )
    with pytest.raises(OllamaPreflightError, match="alibaba"):
        await preflight(cfg, "payload")


async def test_trio_sharing_a_lineage_aborts(config_factory, preflight_env):
    # BDD-52 -- the CRITICAL of cycle 9.
    cfg = config_factory(
        models={
            "melchior": ModelSpec("qwen3.5:397b-cloud", "alibaba"),
            "balthasar": ModelSpec("qwen3-coder:480b-cloud", "alibaba"),  # same lab!
            "caspar": ModelSpec("deepseek-v4-pro:cloud", "deepseek"),
        }
    )
    with pytest.raises(OllamaPreflightError) as exc:
        await preflight(cfg, "payload")
    assert "melchior" in str(exc.value) and "balthasar" in str(exc.value)
    assert "alibaba" in str(exc.value)


async def test_missing_trio_model_aborts_but_missing_fallback_only_warns(
    ollama_config, preflight_env, capsys
):
    """R11.1: the trio is a REQUIREMENT; the fallbacks are insurance."""
    preflight_env["available"] = {s.model for s in ollama_config.models.values()}
    result = await preflight(ollama_config, "payload")  # every fallback is absent
    assert result.fallback == ()  # pruned, not fatal
    assert "fallback" in capsys.readouterr().err.lower()


async def test_a_wholly_absent_trio_model_aborts(config_factory, preflight_env):
    cfg = config_factory()
    preflight_env["available"] = {s.model for s in cfg.models.values()} - {
        cfg.models["caspar"].model
    }
    with pytest.raises(OllamaPreflightError, match="caspar|not available|Missing"):
        await preflight(cfg, "payload")


async def test_cloud_trio_with_no_cloud_available_hints_at_signin(config_factory, preflight_env):
    # BDD-27: preserved from v4 -- the signin diagnostic beats a generic "missing".
    preflight_env["available"] = {"llama3:8b", "qwen3:8b"}  # none :cloud
    with pytest.raises(OllamaPreflightError, match="signin"):
        await preflight(config_factory(), "payload")


async def test_embedding_model_in_fallback_aborts(config_factory, preflight_env):
    """BDD-47 / R19: an embeddings model would fail 100% of the time and burn a rotation."""
    embed = ModelSpec("nomic-embed-text-v2-moe:latest", "nomic")
    cfg = config_factory(fallback=(embed,))
    preflight_env["available"] |= {embed.model}
    preflight_env["caps"][embed.model] = ModelCapability(window=8192, supports_completion=False)
    with pytest.raises(OllamaPreflightError, match="completion"):
        await preflight(cfg, "payload")


async def test_duplicate_lineage_among_fallbacks_aborts(config_factory, preflight_env):
    """BDD-29 -- fail-closed: a duplicate lineage attacks the central invariant."""
    cfg = config_factory(
        fallback=(
            ModelSpec("glm-5.2:cloud", "zhipu"),
            ModelSpec("glm-4.7:cloud", "zhipu"),
        )
    )
    with pytest.raises(OllamaPreflightError, match="zhipu"):
        await preflight(cfg, "payload")


async def test_trio_model_too_small_for_the_payload_aborts(ollama_config, preflight_env):
    """BDD-26: a model that would truncate never runs."""
    small = ModelCapability(window=1_000, supports_completion=True)
    preflight_env["caps"] = {s.model: small for s in ollama_config.models.values()}
    preflight_env["probe"] = 100_000
    with pytest.raises(OllamaPreflightError, match="window|too small"):
        await preflight(ollama_config, "payload")


async def test_a_trio_model_is_checked_against_its_OWN_count_not_a_global_max(
    ollama_config, preflight_env
):
    """R5c is PER MODEL: a model whose OWN exact count fits must not be aborted over
    a worse-tokenising sibling's larger count."""
    a, b, c = (s.model for s in ollama_config.models.values())
    preflight_env["probe"] = {a: 80_000, b: 150_000, c: 80_000}
    fits = ModelCapability(window=1_000_000, supports_completion=True)
    tight = ModelCapability(window=120_000, supports_completion=True)
    preflight_env["caps"] = {a: tight, b: fits, c: fits}
    result = await preflight(ollama_config, "payload")  # no abort: each on its own count
    assert result.context_guard == CONTEXT_GUARD_ENFORCED


async def test_unmeasurable_payload_aborts_by_default_and_estimates_when_opted_out(
    ollama_config, config_factory, preflight_env
):
    """BDD-32 / BDD-53, updated for MS4: strict_context_guard now DEFAULTS to True, so
    an unmeasurable payload aborts unless the user opts out with
    strict_context_guard=false -- in which case it is REPORTED (never hidden), not
    silently accepted."""
    preflight_env["probe"] = None  # the endpoint reports no usage
    with pytest.raises(ContextWindowUnmeasurableError):
        await preflight(ollama_config, "payload")  # ollama_config keeps the built-in default

    result = await preflight(config_factory(strict_context_guard=False), "payload")
    assert result.context_guard == CONTEXT_GUARD_ESTIMATED


async def test_measured_payload_reports_enforced(ollama_config, preflight_env):
    preflight_env["probe"] = 16_232
    result = await preflight(ollama_config, "payload")
    assert result.context_guard == CONTEXT_GUARD_ENFORCED
    assert result.min_window_tokens == 16_232  # RAW: the pre-filter threshold
    assert result.required_tokens > 16_232  # PADDED: + retry feedback + headroom
    assert result.token_estimate_delta[0]["actual"] == 16_232


async def test_a_fallback_that_reuses_a_trio_lineage_warns_as_a_dead_entry(
    config_factory, preflight_env, capsys
):
    """R11.4: it can never be eligible, so it is dead weight. A warning, not an abort."""
    dead = ModelSpec("deepseek-v4-flash:cloud", "deepseek")  # the lineage caspar holds
    cfg = config_factory(fallback=(dead,))
    preflight_env["available"] |= {dead.model}
    await preflight(cfg, "payload")
    err = capsys.readouterr().err
    assert "deepseek" in err and "never be eligible" in err.lower()


async def test_suspicious_lineage_label_warns_without_overriding_the_user(
    config_factory, preflight_env
):
    """BDD-49 / R21: the table DETECTS TYPOS. It is never the authority."""
    cfg = config_factory(fallback=(ModelSpec("deepseek-v4-flash:cloud", "acme"),))
    preflight_env["available"] |= {"deepseek-v4-flash:cloud"}  # available so it survives pruning
    result = await preflight(cfg, "payload")
    assert any("acme" in w for w in result.lineage_warnings)
    assert result.fallback[0].lineage == "acme"  # the declaration STANDS


async def test_no_preflight_error_path_ever_leaks_the_api_key(
    config_factory, preflight_env, capsys
):
    """NR3b: a scattered redaction is a forgotten redaction. Prove there is none.

    strict_context_guard=False is explicit here: MS4 flipped the default to True,
    which would abort before ever reaching the warning/estimated path this targets.
    """
    cfg = config_factory(api_key="sk-supersecret-do-not-leak", strict_context_guard=False)
    preflight_env["probe"] = None  # force the estimated/warning path
    result = await preflight(cfg, "payload")
    captured = capsys.readouterr()
    blob = captured.out + captured.err + json.dumps(result.lineage_warnings)
    assert "sk-supersecret-do-not-leak" not in blob


async def test_the_strict_abort_path_does_not_leak_the_api_key(config_factory, preflight_env):
    """NR4: the fail-closed MS4 abort must not echo the api_key either."""
    cfg = config_factory(api_key="sk-supersecret-do-not-leak")  # strict is the default now
    preflight_env["probe"] = None
    with pytest.raises(ContextWindowUnmeasurableError) as ei:
        await preflight(cfg, "payload")
    assert "sk-supersecret-do-not-leak" not in str(ei.value)


async def test_the_list_models_abort_path_redacts_the_api_key(config_factory, monkeypatch):
    """NR3b: the _list_models 401/unreachable path echoes the URL + Authorization
    header; the preflight call boundary must redact it."""
    cfg = config_factory(api_key="sk-supersecret-do-not-leak")

    async def failing_list(config):
        raise OllamaPreflightError(
            f"GET {config.base_url}/models -> 401 (Authorization: Bearer {config.api_key})"
        )

    monkeypatch.setattr("ollama_preflight._list_models", failing_list)
    with pytest.raises(OllamaPreflightError) as ei:
        await preflight(cfg, "payload")
    assert "sk-supersecret-do-not-leak" not in str(ei.value)


async def test_measured_payload_with_unknown_windows_aborts_by_default_and_estimates_when_opted_out(
    ollama_config, config_factory, preflight_env
):
    """The guard is 'enforced' only when the payload was measured AND every window is
    known. Measured payload + unknown windows cannot prove invariant #3 -> unmeasurable,
    and MS4's new default (strict_context_guard=True) aborts; opting out with false still
    reports it, as 'estimated', never silently."""
    a, b, c = (s.model for s in ollama_config.models.values())
    preflight_env["probe"] = 16_232  # payload measured for all three
    unknown = ModelCapability(window=None, supports_completion=True)
    preflight_env["caps"] = {a: unknown, b: unknown, c: unknown}

    with pytest.raises(ContextWindowUnmeasurableError):
        await preflight(ollama_config, "payload")  # default is strict (True)

    result = await preflight(config_factory(strict_context_guard=False), "payload")
    assert result.context_guard == CONTEXT_GUARD_ESTIMATED, (
        "measured payload but unknown windows cannot be reported 'enforced'"
    )


# --------------------------------------------------------------------------
# MS4: strict_context_guard defaults to True -- fail-closed on an unmeasurable window.
# --------------------------------------------------------------------------


async def test_unmeasurable_window_aborts_by_default(ollama_config, preflight_env):
    """BDD-3: a payload that cannot be measured, on the (now fail-closed) default
    config, aborts the run instead of silently falling back to an estimate."""
    preflight_env["probe"] = None  # the endpoint reports no usage
    with pytest.raises(ContextWindowUnmeasurableError):
        await preflight(ollama_config, "payload")


async def test_abort_message_names_the_optout(ollama_config, preflight_env):
    """The abort must NAME the opt-out -- a guard that flips its default and gives no
    escape hatch in its own error message is a trap, not a fail-closed design."""
    preflight_env["probe"] = None
    with pytest.raises(ContextWindowUnmeasurableError) as ei:
        await preflight(ollama_config, "payload")
    message = str(ei.value)
    assert "strict_context_guard" in message
    assert "false" in message


async def test_window_present_but_invalid_is_unmeasurable(ollama_config, preflight_env):
    """BDD-3c: /api/show responds, but the window is absent/zero/non-numeric -- already
    collapsed to None by ``_read_window`` (model_context.py) -- and the preflight treats
    that exactly like never hearing back: unmeasurable, and strict aborts."""
    a, b, c = (s.model for s in ollama_config.models.values())
    invalid_window = ModelCapability(window=None, supports_completion=True)
    preflight_env["caps"] = {a: invalid_window, b: invalid_window, c: invalid_window}
    with pytest.raises(ContextWindowUnmeasurableError):
        await preflight(ollama_config, "payload")


async def test_explicit_false_proceeds_with_estimated_guard(config_factory, preflight_env):
    """BDD-3b: the opt-out remains -- strict_context_guard=false on an unmeasurable
    payload still proceeds, downgraded to 'estimated', never silently 'enforced'."""
    preflight_env["probe"] = None
    result = await preflight(config_factory(strict_context_guard=False), "payload")
    assert result.context_guard == CONTEXT_GUARD_ESTIMATED


async def test_payload_probes_run_concurrently_not_sequentially(ollama_config, monkeypatch):
    """NR6b (MAGI gate, Balthasar): the trio's payload probes must run CONCURRENTLY --
    three serialized round-trips triple the preflight latency for no reason."""
    import asyncio as _asyncio

    from ollama_preflight import _measure_payload

    inflight = {"now": 0, "max": 0}

    async def fake_probe(config, model, prompt, **kw):
        inflight["now"] += 1
        inflight["max"] = max(inflight["max"], inflight["now"])
        await _asyncio.sleep(0)  # yield so siblings can start before this one finishes
        inflight["now"] -= 1
        return 100

    monkeypatch.setattr("ollama_preflight.probe_prompt_tokens", fake_probe)
    measured, _deltas, _est = await _measure_payload(ollama_config, "payload")
    assert inflight["max"] == 3, "all three probes must be in flight at once, not serialized"
    assert len(measured) == 3


# --------------------------------------------------------------------------
# Task 5a (Grieta 2): digest collision, missing digest (R5b), family check (R6),
# R6b unmapped-architecture info note, and R21/family mutual exclusivity.
# --------------------------------------------------------------------------


async def test_digest_collision_between_two_trio_mages_aborts(ollama_config, preflight_env):
    """R5/R5a: two trio mages resolving to the SAME digest is ensemble collapse --
    checked regardless of strict_lineage."""
    a, b, c = (s.model for s in ollama_config.models.values())
    same = ModelCapability(window=200_000, supports_completion=True, digest="sha256:same")
    other = ModelCapability(window=200_000, supports_completion=True, digest="sha256:other")
    preflight_env["caps"] = {a: same, b: same, c: other}
    with pytest.raises(DigestCollisionError) as exc:
        await preflight(ollama_config, "payload")
    assert "melchior" in str(exc.value) and "balthasar" in str(exc.value)


async def test_missing_digest_on_a_non_cloud_trio_model_aborts(config_factory, preflight_env):
    """R5b: a NON-cloud trio model whose /api/show gave no digest cannot have its
    uniqueness verified -- fail closed."""
    cfg = config_factory(
        models={
            "melchior": ModelSpec("qwen3.5:32b-local", "alibaba"),
            "balthasar": ModelSpec("kimi-k2.6:cloud", "moonshot"),
            "caspar": ModelSpec("deepseek-v4-pro:cloud", "deepseek"),
        }
    )
    preflight_env["available"] |= {"qwen3.5:32b-local"}
    preflight_env["caps"] = {
        "qwen3.5:32b-local": ModelCapability(window=200_000, supports_completion=True, digest=None)
    }
    with pytest.raises(MissingDigestError, match="melchior"):
        await preflight(cfg, "payload")


async def test_missing_digest_on_a_cloud_trio_model_does_not_abort(ollama_config, preflight_env):
    """_CLOUD_HAS_DIGEST: a :cloud tag's absent digest is EXPECTED, not a failure --
    the default preflight_env caps (digest=None) model exactly this."""
    result = await preflight(ollama_config, "payload")
    assert result.digest_by_model == {}


async def test_family_contradiction_aborts_when_strict_lineage_is_true(config_factory, preflight_env):
    """R6: strict_lineage=True upgrades a probed-architecture/declared-lineage
    contradiction to a fail-closed abort."""
    cfg = config_factory(
        strict_lineage=True,
        models={
            "melchior": ModelSpec("qwen3.5:397b-cloud", "wrongvendor"),
            "balthasar": ModelSpec("kimi-k2.6:cloud", "moonshot"),
            "caspar": ModelSpec("deepseek-v4-pro:cloud", "deepseek"),
        },
    )
    a = cfg.models["melchior"].model
    preflight_env["caps"] = {
        a: ModelCapability(window=200_000, supports_completion=True, architecture="qwen3.5")
    }
    with pytest.raises(FamilyContradictionError, match="melchior"):
        await preflight(cfg, "payload")


async def test_family_contradiction_warns_and_proceeds_by_default(config_factory, preflight_env):
    """Default strict_lineage=False: a contradiction only WARNS -- in the SAME
    lineage_warnings collection R21 uses -- and the run proceeds."""
    cfg = config_factory(
        models={
            "melchior": ModelSpec("qwen3.5:397b-cloud", "wrongvendor"),
            "balthasar": ModelSpec("kimi-k2.6:cloud", "moonshot"),
            "caspar": ModelSpec("deepseek-v4-pro:cloud", "deepseek"),
        }
    )
    a = cfg.models["melchior"].model
    preflight_env["caps"] = {
        a: ModelCapability(window=200_000, supports_completion=True, architecture="qwen3.5")
    }
    result = await preflight(cfg, "payload")
    assert any("melchior" in w and "wrongvendor" in w for w in result.lineage_warnings)


async def test_family_unknown_architecture_never_blocks(ollama_config, preflight_env):
    """family_verdict fails OPEN for an unmapped architecture -- it must never
    block a run, even under strict_lineage."""
    a, _b, _c = (s.model for s in ollama_config.models.values())
    unmapped = ModelCapability(
        window=200_000, supports_completion=True, architecture="some-selfhosted-arch"
    )
    preflight_env["caps"] = {a: unmapped}
    result = await preflight(ollama_config, "payload")  # must not raise
    assert isinstance(result, PreflightResult)


async def test_unmapped_architecture_with_real_lineage_adds_an_info_note(
    ollama_config, preflight_env
):
    """R6b: architecture WAS probed but is not in the known vendor map, and the
    declared lineage is real (non-trivial) -- an INFO note joins lineage_warnings."""
    a = ollama_config.models["melchior"].model
    preflight_env["caps"] = {
        a: ModelCapability(window=200_000, supports_completion=True, architecture="llama")
    }
    result = await preflight(ollama_config, "payload")
    assert any(w.startswith("INFO:") and "melchior" in w for w in result.lineage_warnings)


async def test_unmapped_architecture_with_trivial_lineage_adds_no_note(
    config_factory, preflight_env
):
    """R6b: a TRIVIAL declared lineage has nothing for the note to cross-check --
    no INFO note fires."""
    cfg = config_factory(
        models={
            "melchior": ModelSpec("qwen3.5:397b-cloud", "unknown"),
            "balthasar": ModelSpec("kimi-k2.6:cloud", "moonshot"),
            "caspar": ModelSpec("deepseek-v4-pro:cloud", "deepseek"),
        }
    )
    a = cfg.models["melchior"].model
    preflight_env["caps"] = {
        a: ModelCapability(window=200_000, supports_completion=True, architecture="llama")
    }
    result = await preflight(cfg, "payload")
    assert not any("melchior" in w for w in result.lineage_warnings)


async def test_tag_prefix_check_fires_for_trio_when_architecture_is_absent(
    config_factory, preflight_env
):
    """R21: with no /api/show architecture data (the preflight_env default), the
    trio still falls back to the tag-prefix typo detector (unchanged behaviour)."""
    cfg = config_factory(
        models={
            "melchior": ModelSpec("qwen3.5:397b-cloud", "acme"),  # tag prefix says alibaba
            "balthasar": ModelSpec("kimi-k2.6:cloud", "moonshot"),
            "caspar": ModelSpec("deepseek-v4-pro:cloud", "deepseek"),
        }
    )
    result = await preflight(cfg, "payload")
    assert any("qwen3.5:397b-cloud" in w and "acme" in w for w in result.lineage_warnings)


async def test_tag_prefix_check_is_superseded_once_architecture_is_known(
    config_factory, preflight_env
):
    """R21 vs R6 mutual exclusivity: once /api/show reports a real architecture,
    the family check supersedes the tag-prefix guess for that model -- no
    redundant tag-based warning, only the (more informative) family warning."""
    cfg = config_factory(
        models={
            "melchior": ModelSpec("qwen3.5:397b-cloud", "acme"),
            "balthasar": ModelSpec("kimi-k2.6:cloud", "moonshot"),
            "caspar": ModelSpec("deepseek-v4-pro:cloud", "deepseek"),
        }
    )
    a = cfg.models["melchior"].model
    preflight_env["caps"] = {
        a: ModelCapability(window=200_000, supports_completion=True, architecture="qwen3.5")
    }
    result = await preflight(cfg, "payload")
    assert not any("tag suggests" in w for w in result.lineage_warnings)
    assert any("melchior" in w and "acme" in w for w in result.lineage_warnings)


async def test_preflight_result_carries_digest_by_model_as_a_seed_field():
    """PreflightResult must expose digest_by_model (Task 5b's rotation lookup
    seed), defaulting to empty so every existing call site keeps working."""
    from dataclasses import fields

    names = {f.name for f in fields(PreflightResult)}
    assert "digest_by_model" in names


async def test_digest_by_model_is_seeded_only_for_trio_models_that_have_one(
    config_factory, preflight_env
):
    """Populated with trio model -> digest ONLY for models that HAVE one -- never
    pre-populated for a fallback (no I/O the preflight would not otherwise do)."""
    cfg = config_factory(
        models={
            "melchior": ModelSpec("qwen3.5:32b-local", "alibaba"),
            "balthasar": ModelSpec("kimi-k2.6:cloud", "moonshot"),
            "caspar": ModelSpec("deepseek-v4-pro:cloud", "deepseek"),
        }
    )
    preflight_env["available"] |= {"qwen3.5:32b-local"}
    preflight_env["caps"] = {
        "qwen3.5:32b-local": ModelCapability(
            window=200_000, supports_completion=True, digest="sha256:local1"
        )
    }
    result = await preflight(cfg, "payload")
    assert result.digest_by_model == {"qwen3.5:32b-local": "sha256:local1"}


async def test_digest_by_model_is_not_read_anywhere_in_the_report_builder():
    """digest_by_model is internal preflight seed data (Task 5b): it must never
    leak into the 7-key agent JSON or magi-report.json. Proven by inspecting the
    module that actually builds the report, not merely asserted."""
    import inspect

    import run_magi

    source = inspect.getsource(run_magi)
    assert "digest_by_model" not in source
