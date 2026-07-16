# tests/test_ollama_config.py
"""Layered, per-key resolution of OllamaConfig (BDD-4,5,6)."""

import dataclasses  # for FrozenInstanceError

import pytest
from validate import ValidationError
from ollama_config import (
    DEFAULT_MODELS,
    DEFAULT_BASE_URL,
    ModelSpec,
    OllamaConfig,
    OllamaConfigError,
    resolve_config,
)


def _write(p, text):
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_defaults_when_no_files_no_env(monkeypatch):
    for var in (
        "MAGI_OLLAMA_HOST",
        "OLLAMA_HOST",
        "MAGI_OLLAMA_API_KEY",
        "OLLAMA_API_KEY",
        "MAGI_OLLAMA_MODEL_MELCHIOR",
        "MAGI_OLLAMA_MODEL_BALTHASAR",
        "MAGI_OLLAMA_MODEL_CASPAR",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = resolve_config(global_path="/nope/g.toml", repo_path="/nope/r.toml", env={})
    assert cfg.base_url == DEFAULT_BASE_URL
    assert cfg.api_key is None
    assert cfg.models == dict(DEFAULT_MODELS)


def test_repo_overrides_global_per_key(tmp_path):
    g = _write(
        tmp_path / "g.toml",
        'base_url="http://g:11434/v1"\n[models]\n'
        'melchior  = { model = "gm", lineage = "la" }\n'
        'balthasar = { model = "gb", lineage = "lb" }\n'
        'caspar    = { model = "gc", lineage = "lc" }\n',
    )
    r = _write(tmp_path / "r.toml", 'base_url="http://r:11434/v1"\n')
    cfg = resolve_config(global_path=g, repo_path=r, env={})
    assert cfg.base_url == "http://r:11434/v1"  # repo wins
    assert cfg.models == {  # from global
        "melchior": ModelSpec("gm", "la"),
        "balthasar": ModelSpec("gb", "lb"),
        "caspar": ModelSpec("gc", "lc"),
    }


def test_env_overrides_files(tmp_path):
    r = _write(tmp_path / "r.toml", '[models]\ncaspar = { model = "rc", lineage = "zhipu" }\n')
    cfg = resolve_config(
        global_path="/nope.toml", repo_path=r, env={"MAGI_OLLAMA_MODEL_CASPAR": "ec"}
    )
    # env forces the tag; the DECLARED lineage of the resolved spec is preserved.
    assert cfg.models["caspar"] == ModelSpec("ec", "zhipu")


def test_ollama_host_is_fallback_below_files(tmp_path):
    cfg = resolve_config(
        global_path="/nope.toml", repo_path="/nope.toml", env={"OLLAMA_HOST": "1.2.3.4:11434"}
    )
    assert cfg.base_url == "http://1.2.3.4:11434/v1"


def test_magi_host_beats_ollama_host_and_files(tmp_path):
    r = _write(tmp_path / "r.toml", 'base_url="http://r:11434/v1"\n')
    cfg = resolve_config(
        global_path="/nope.toml",
        repo_path=r,
        env={"MAGI_OLLAMA_HOST": "http://m:9/v1", "OLLAMA_HOST": "x:1"},
    )
    assert cfg.base_url == "http://m:9/v1"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("http://h:11434/v1", "http://h:11434/v1"),
        ("https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1"),
        ("1.2.3.4:11434", "http://1.2.3.4:11434/v1"),
        ("http://h:11434/v1/", "http://h:11434/v1"),
        ("http://gw/proxy", "http://gw/proxy"),  # F-F: any path preserved verbatim, no /v1 appended
        ("https://openrouter.ai/api/v1/", "https://openrouter.ai/api/v1"),
    ],
)
def test_base_url_normalization(raw, expected):
    cfg = resolve_config(global_path="/n.toml", repo_path="/n.toml", env={"MAGI_OLLAMA_HOST": raw})
    assert cfg.base_url == expected


def test_api_key_precedence(tmp_path):
    g = _write(tmp_path / "g.toml", 'api_key="gk"\n')
    cfg = resolve_config(global_path=g, repo_path="/n.toml", env={"OLLAMA_API_KEY": "ok"})
    assert cfg.api_key == "gk"  # file beats generic OLLAMA_API_KEY
    cfg2 = resolve_config(global_path=g, repo_path="/n.toml", env={"MAGI_OLLAMA_API_KEY": "mk"})
    assert cfg2.api_key == "mk"  # MAGI-specific env wins


def test_empty_magi_api_key_env_is_none_not_inherited(tmp_path):  # BDD-26 / F-C
    g = _write(tmp_path / "g.toml", 'api_key="gk"\n')
    cfg = resolve_config(global_path=g, repo_path="/n.toml", env={"MAGI_OLLAMA_API_KEY": ""})
    assert cfg.api_key is None  # explicit empty => no auth, NOT the file's key (CI leak guard)


def test_malformed_toml_raises_named_error(tmp_path):
    bad = _write(tmp_path / "bad.toml", "this is = = not toml")
    with pytest.raises(OllamaConfigError) as ei:
        resolve_config(global_path=bad, repo_path="/n.toml", env={})
    assert "bad.toml" in str(ei.value)


def test_config_error_is_validation_error():
    assert issubclass(OllamaConfigError, ValidationError)


def test_unknown_keys_warn_and_proceed(tmp_path, capsys):
    g = _write(tmp_path / "g.toml", 'wat=1\nbase_url="http://h:1/v1"\n')
    cfg = resolve_config(global_path=g, repo_path="/n.toml", env={})
    assert cfg.base_url == "http://h:1/v1"
    assert "wat" in capsys.readouterr().err


def test_the_inert_structured_key_is_GONE_and_the_toml_says_so(tmp_path, capsys):
    """MS2/R7: ``structured`` governed ``response_format``, which is no longer sent.

    Keeping the field would have been **a config key that LIES**: the user sets it, MAGI
    accepts it... and ignores it. This project forbids *silent failures*, and a user's
    intent discarded in silence is one.

    Removing it breaks NOBODY: the resolver **already warns** on unknown keys, so a TOML
    carrying ``structured`` gets an **actionable** warning instead of a mute no-op.
    """
    from ollama_config import _load_toml

    toml = tmp_path / "magi-ollama.toml"
    toml.write_text('structured = "schema"\n', encoding="utf-8")

    assert not hasattr(OllamaConfig, "structured")
    _load_toml(str(toml))
    assert "unknown key 'structured'" in capsys.readouterr().err


def test_empty_base_url_in_toml_falls_through_to_default(tmp_path):
    """An empty base_url in a TOML file must be treated as unset (fall through).

    BDD: Given a repo TOML with base_url="", When resolve_config is called,
    Then cfg.base_url equals DEFAULT_BASE_URL, NOT the malformed "http:///v1".
    """
    r = _write(tmp_path / "r.toml", 'base_url=""\n')
    cfg = resolve_config(global_path="/nope.toml", repo_path=r, env={})
    assert cfg.base_url == DEFAULT_BASE_URL


# --- Task 1: ModelSpec + table-shaped [models] with mandatory lineage (BREAKING) ---

NEW_TOML = """
base_url = "http://localhost:11434/v1"
[models]
melchior  = { model = "qwen3.5:397b-cloud",   lineage = "alibaba" }
balthasar = { model = "kimi-k2.6:cloud",      lineage = "moonshot" }
caspar    = { model = "deepseek-v4-pro:cloud", lineage = "deepseek" }
"""

OLD_TOML = """
[models]
melchior  = "qwen3.5:397b-cloud"
balthasar = "kimi-k2.6:cloud"
caspar    = "deepseek-v4-pro:cloud"
"""


def _write_toml(tmp_path, text):
    p = tmp_path / "magi-ollama.toml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_models_resolve_to_model_spec_with_lineage(tmp_path):
    cfg = resolve_config(repo_path=_write_toml(tmp_path, NEW_TOML), global_path=None, env={})
    assert cfg.models["melchior"] == ModelSpec(model="qwen3.5:397b-cloud", lineage="alibaba")
    assert cfg.models["caspar"].lineage == "deepseek"


def test_old_string_schema_raises_actionable_migration_error(tmp_path):
    path = _write_toml(tmp_path, OLD_TOML)
    with pytest.raises(OllamaConfigError) as exc:
        resolve_config(repo_path=path, global_path=None, env={})
    msg = str(exc.value)
    assert path in msg  # names the offending file
    assert "lineage" in msg  # shows the new shape
    assert "melchior" in msg


def test_lineage_is_normalised_so_case_cannot_defeat_the_invariant(tmp_path):
    """Every guard compares lineages with == / in. A capital letter must not be able
    to smuggle two mages of the same lab past R22."""
    toml = NEW_TOML.replace('lineage = "alibaba"', 'lineage = "  AliBaba  "')
    cfg = resolve_config(repo_path=_write_toml(tmp_path, toml), global_path=None, env={})
    assert cfg.models["melchior"].lineage == "alibaba"


def test_empty_lineage_is_rejected(tmp_path):
    toml = NEW_TOML.replace('lineage = "alibaba"', 'lineage = ""')
    with pytest.raises(OllamaConfigError):
        resolve_config(repo_path=_write_toml(tmp_path, toml), global_path=None, env={})


def test_model_spec_is_frozen():
    # Specific exception, not a blanket Exception (§Error Handling): a frozen
    # dataclass raises FrozenInstanceError, and asserting anything looser would
    # also pass on an AttributeError from a typo in the test itself.
    with pytest.raises(dataclasses.FrozenInstanceError):
        ModelSpec(model="a", lineage="b").model = "c"  # type: ignore[misc]


# ----------------------------------------------------------------------------
# Task 2: [[fallback]] array + rotation config scalars (R4/R6/R12/R14/NR3)
# ----------------------------------------------------------------------------

# Scalars MUST precede [models]: TOML binds bare keys to the current table, so a
# scalar written after [models] would nest under [models] (top-level resolution
# would then never see it). The plan's fixture had this bug; corrected here.
FALLBACK_TOML = """
base_url = "http://localhost:11434/v1"
max_attempts_per_model = 3
max_rotations = 1
strict_context_guard = true

[models]
melchior  = { model = "qwen3.5:397b-cloud",   lineage = "alibaba" }
balthasar = { model = "kimi-k2.6:cloud",      lineage = "moonshot" }
caspar    = { model = "deepseek-v4-pro:cloud", lineage = "deepseek" }

[[fallback]]
model = "glm-5.2:cloud"
lineage = "zhipu"

[[fallback]]
model = "gpt-oss:120b-cloud"
lineage = "openai"
"""


def test_fallback_list_is_ordered_sequence_of_model_specs(tmp_path):
    cfg = resolve_config(repo_path=_write_toml(tmp_path, FALLBACK_TOML), global_path=None, env={})
    assert [f.model for f in cfg.fallback] == ["glm-5.2:cloud", "gpt-oss:120b-cloud"]
    assert cfg.fallback[0].lineage == "zhipu"


def test_rotation_scalars_are_read_and_defaulted(tmp_path):
    cfg = resolve_config(repo_path=_write_toml(tmp_path, FALLBACK_TOML), global_path=None, env={})
    assert cfg.max_attempts_per_model == 3  # from file
    assert cfg.max_rotations == 1  # from file
    assert cfg.strict_context_guard is True  # from file
    assert cfg.output_headroom_tokens == 8192  # built-in default
    assert cfg.input_margin_pct == 40  # built-in default
    assert cfg.max_probe_attempts == 3  # built-in default
    assert cfg.retry_backoff_seconds == 2.0  # built-in default
    assert cfg.preflight_timeout_seconds == 30  # built-in default
    assert cfg.probe_timeout_seconds == 120  # built-in default


def test_missing_fallback_section_disables_rotation(tmp_path):
    """R4: absent or empty [[fallback]] => the feature is INACTIVE (v4 behaviour).

    NOT a fall-through to the built-in list: a hand-written v5 config that omits
    [[fallback]] omitted it on purpose; silently rotating anyway would be MAGI
    substituting a judge the operator never declared.
    """
    cfg = resolve_config(repo_path=_write_toml(tmp_path, NEW_TOML), global_path=None, env={})
    assert cfg.fallback == ()  # empty => no rotation, whatever max_rotations says


def test_ollama_init_template_ships_the_default_fallback_list(tmp_path):
    """The built-in list reaches the user through the TEMPLATE (decision #65).

    DEVIATION from plan line 726: the plan hardcoded the PRE-swap lineages
    ["deepseek", ...]; after the deepseek->Caspar swap, deepseek is a TRIO lineage
    and cannot be a fallback (R11.4 dead entry). The authoritative DEFAULT_FALLBACK
    and spec §10 ship glm-5.2 (zhipu) as #1. Corrected to match.
    """
    from ollama_init import render_template

    path = tmp_path / "magi-ollama.toml"
    path.write_text(render_template(), encoding="utf-8")
    cfg = resolve_config(repo_path=str(path), global_path=None, env={})
    assert [f.lineage for f in cfg.fallback] == [
        "zhipu",
        "openai",
        "minimax",
        "nvidia",
        "google",
    ]


@pytest.mark.parametrize(
    "key,bad",
    [
        ("max_attempts_per_model", "0"),
        ("max_rotations", "-1"),
        ("max_probe_attempts", "0"),
        ("input_margin_pct", "-5"),
    ],
)
def test_invalid_scalar_raises_never_silently_defaults(tmp_path, key, bad):
    # Prepend (not append): the scalar must be top-level, i.e. BEFORE [models].
    toml = f"{key} = {bad}\n" + NEW_TOML
    with pytest.raises(OllamaConfigError):
        resolve_config(repo_path=_write_toml(tmp_path, toml), global_path=None, env={})


@pytest.mark.parametrize("bad", ["2.7", "true", '"three"'])
def test_non_integer_scalars_are_rejected_never_coerced(tmp_path, bad):
    # int(2.7) == 2 and isinstance(True, int) is True: both would SILENTLY
    # accept a config the user got wrong. Prepend so the scalar is top-level.
    toml = f"max_rotations = {bad}\n" + NEW_TOML
    with pytest.raises(OllamaConfigError):
        resolve_config(repo_path=_write_toml(tmp_path, toml), global_path=None, env={})


def test_env_overrides_file_for_kill_switch(tmp_path):
    cfg = resolve_config(
        repo_path=_write_toml(tmp_path, FALLBACK_TOML),
        global_path=None,
        env={"MAGI_OLLAMA_MAX_ROTATIONS": "0"},
    )
    assert cfg.max_rotations == 0  # kill-switch (R17)


def test_require_float_rejects_non_finite_values():
    """MAGI gate (Caspar): inf/nan pass the ``>= minimum`` check (nan comparisons are
    always False, inf is >= anything), so ``retry_backoff_seconds = inf`` would reach
    ``asyncio.sleep(inf)`` and hang the orchestrator forever. Reject non-finite."""
    from ollama_config import OllamaConfigError, _require_float

    for bad in ("inf", "-inf", "nan"):
        with pytest.raises(OllamaConfigError):
            _require_float(bad, key="retry_backoff_seconds", minimum=0.0, path="t.toml")


def test_require_bool_accepts_integer_1_and_0_but_not_other_ints():
    """MAGI gate (Balthasar): the error text promises '1/0', so a TOML integer literal
    (strict_context_guard = 1) must be accepted; other integers still fail closed."""
    from ollama_config import OllamaConfigError, _require_bool

    assert _require_bool(1, key="strict_context_guard", path="t.toml") is True
    assert _require_bool(0, key="strict_context_guard", path="t.toml") is False
    for bad in (2, -1, 42):
        with pytest.raises(OllamaConfigError):
            _require_bool(bad, key="strict_context_guard", path="t.toml")


def test_strict_context_guard_defaults_to_true():
    """MS4: the context guard is fail-closed by default -- an unmeasurable window now
    aborts a run unless the user opts out explicitly with strict_context_guard=false."""
    from ollama_config import DEFAULT_STRICT_CONTEXT_GUARD, ModelSpec, OllamaConfig

    assert DEFAULT_STRICT_CONTEXT_GUARD is True

    cfg = OllamaConfig(
        base_url=DEFAULT_BASE_URL,
        api_key=None,
        models={"melchior": ModelSpec("m", "la")},
    )
    assert cfg.strict_context_guard is True


def test_strict_context_guard_resolves_to_true_when_absent_from_every_source(monkeypatch):
    """The default flip must reach resolve_config too, not just the dataclass field."""
    monkeypatch.delenv("MAGI_OLLAMA_STRICT_CONTEXT", raising=False)
    cfg = resolve_config(global_path="/nope/g.toml", repo_path="/nope/r.toml", env={})
    assert cfg.strict_context_guard is True


def test_no_shipped_default_carries_a_provisional_tag():
    """A default we ship must not be a tag the vendor has told us is provisional.

    ``gemini-3-flash-preview:latest`` was accepted into the fallback list as a known risk
    (the Google slot takes one model, and Gemini 3 beat gemma4). Ollama then announced its
    retirement, which would have left every user's scaffolded config carrying a dead entry
    -- harmless by R11.1 (a missing fallback warns, never aborts) but noisy on every run,
    and a default nobody chose.

    Scope, stated honestly (finding: Caspar, MAGI gate 2026-07-12): this catches only the
    labels that ANNOUNCE impermanence. Any tag can be retired without one --
    ``nemotron-3-ultra`` had no manifest at all -- and the guard for that is
    ``scripts/verify_fallback_tags.py``, which hits registry.ollama.ai before every
    release. This test is the cheap, offline half: it cannot see the registry, but it can
    refuse a name that says "temporary" out loud, and it runs on every commit.

    Matching is on TOKENS, not substrings, and allows a trailing number: ``arcee`` contains
    "rc" and is not a release candidate, while ``rc1`` and ``beta2`` are.
    """
    import re

    from ollama_config import DEFAULT_FALLBACK, DEFAULT_MODELS

    provisional = re.compile(r"^(preview|alpha|beta|rc|experimental|nightly|test)\d*$")

    shipped = [spec.model for spec in DEFAULT_MODELS.values()]
    shipped += [spec.model for spec in DEFAULT_FALLBACK]

    offenders = [
        tag
        for tag in shipped
        if any(provisional.match(token) for token in re.split(r"[^a-z0-9]+", tag.lower()))
    ]
    assert offenders == [], f"provisional tags shipped as defaults: {offenders}"


def test_the_TOML_attempt_budget_obeys_the_SAME_cap_as_the_CLI(tmp_path):
    """MAGI gate (Balthasar, cycle 10): two doors, one lock -- and this one had none.

    ``--max-attempts`` is capped at ``MAX_ATTEMPTS_CAP`` precisely because a zero too many
    turns one obstinate mage into a thousand calls: expensive on Ollama, where the ``:cloud``
    tags are a paid tier, and hundreds of dollars on Claude. The TOML sets the very same budget
    and accepted **1000** -- measured, not supposed. A bound that only guards one of the two
    ways in is not a bound.
    """
    from run_magi import MAX_ATTEMPTS_CAP

    toml = tmp_path / "magi-ollama.toml"
    toml.write_text(
        'base_url = "http://localhost:11434/v1"\n'
        f"max_attempts_per_model = {MAX_ATTEMPTS_CAP + 1}\n"
        "\n[models]\n"
        'melchior  = { model = "a:cloud", lineage = "alibaba" }\n'
        'balthasar = { model = "b:cloud", lineage = "moonshot" }\n'
        'caspar    = { model = "c:cloud", lineage = "deepseek" }\n'
        "\n[[fallback]]\n"
        'model = "d:cloud"\n'
        'lineage = "zhipu"\n',
        encoding="utf-8",
    )

    with pytest.raises(OllamaConfigError, match="max_attempts_per_model"):
        resolve_config(repo_path=toml, global_path=tmp_path / "absent.toml", env={})


# ----------------------------------------------------------------------------
# Task 5a: strict_lineage config key (Grieta 2 -- family contradiction gate)
# ----------------------------------------------------------------------------


def test_strict_lineage_defaults_to_false_when_key_absent(tmp_path):
    """A TOML that never sets strict_lineage keeps the warn-only path (default
    False): a family contradiction warns instead of aborting the run."""
    from ollama_config import DEFAULT_STRICT_LINEAGE

    assert DEFAULT_STRICT_LINEAGE is False

    cfg = resolve_config(repo_path=_write_toml(tmp_path, NEW_TOML), global_path=None, env={})
    assert cfg.strict_lineage is False


def test_strict_lineage_reads_from_file_and_env(tmp_path):
    toml = "strict_lineage = true\n" + NEW_TOML
    cfg = resolve_config(repo_path=_write_toml(tmp_path, toml), global_path=None, env={})
    assert cfg.strict_lineage is True

    cfg_env = resolve_config(
        repo_path=_write_toml(tmp_path, NEW_TOML),
        global_path=None,
        env={"MAGI_OLLAMA_STRICT_LINEAGE": "true"},
    )
    assert cfg_env.strict_lineage is True


def test_ollama_init_template_scaffolds_strict_lineage_at_its_default(tmp_path):
    """The scaffold must emit strict_lineage (visible + editable), not omit it and
    rely on a hidden default (decision mirrored from strict_context_guard)."""
    from ollama_init import render_template

    path = tmp_path / "magi-ollama.toml"
    path.write_text(render_template(), encoding="utf-8")
    cfg = resolve_config(repo_path=str(path), global_path=None, env={})
    assert cfg.strict_lineage is False
    assert "strict_lineage" in path.read_text(encoding="utf-8")
