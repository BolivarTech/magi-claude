import os
import pytest
from ollama_config import DEFAULT_BASE_URL, DEFAULT_FALLBACK, DEFAULT_MODELS, resolve_config
from ollama_init import REPO_CONFIG_RELPATH, render_template, write_template


def test_creates_template_with_active_base_url_and_trio(tmp_path):
    path = write_template(repo_root=str(tmp_path))
    assert os.path.isfile(path)
    text = open(path, encoding="utf-8").read()
    assert f'base_url = "{DEFAULT_BASE_URL}"' in text  # active, not commented
    assert "# api_key" in text  # commented
    assert "signin" in text.lower()  # 2-mode header (F-B)
    for spec in DEFAULT_MODELS.values():
        assert spec.model in text  # tag rendered
        assert spec.lineage in text  # lineage declared (v5.0.0 schema)


def test_refuses_to_overwrite(tmp_path):
    path = write_template(repo_root=str(tmp_path))
    before = open(path, encoding="utf-8").read()
    with pytest.raises(FileExistsError):
        write_template(repo_root=str(tmp_path))
    assert open(path, encoding="utf-8").read() == before


def test_roundtrip_equals_defaults(tmp_path):
    write_template(repo_root=str(tmp_path))
    repo_toml = os.path.join(str(tmp_path), REPO_CONFIG_RELPATH)
    cfg = resolve_config(global_path="/nope.toml", repo_path=repo_toml, env={})
    assert cfg.base_url == DEFAULT_BASE_URL
    assert cfg.models == dict(DEFAULT_MODELS)


def test_template_round_trips_to_the_builtin_defaults(tmp_path):
    # BDD-17: the template IS the defaults -- one source of truth (invariant #4).
    path = tmp_path / "magi-ollama.toml"
    path.write_text(render_template(), encoding="utf-8")
    cfg = resolve_config(repo_path=str(path), global_path=None, env={})
    assert cfg.models == dict(DEFAULT_MODELS)
    assert tuple(cfg.fallback) == tuple(DEFAULT_FALLBACK)
    assert cfg.max_rotations == 2
    assert cfg.max_attempts_per_model == 2


def test_template_comments_carry_no_internal_spec_ids():
    # The comments are for whoever edits the file, not for the spec authors:
    # internal requirement IDs like (R24)/(R5b)/(R4) mean nothing to a user.
    import re

    text = render_template()
    hits = re.findall(r"\(R\d+[a-z]?\)", text)
    assert hits == [], f"internal spec IDs leaked into template comments: {hits}"


def test_template_documents_the_kill_switch_and_the_pull_requirement():
    text = render_template()
    assert "MAGI_OLLAMA_MAX_ROTATIONS=0" in text  # R17 must be discoverable
    assert "ollama pull" in text  # fallbacks need manifests
    assert "lineage" in text


def test_template_emits_rotation_tunables_as_keys_before_models():
    # R12: the scaffold must SHOW max_attempts_per_model / max_rotations (not just
    # rely on resolver defaults) so the knobs -- and the kill-switch -- are editable.
    # They are top-level scalars: TOML requires them BEFORE any [table] header, and
    # they apply to all mages, so they belong ahead of [models].
    import re

    text = render_template()
    models_at = text.index("\n[models]\n")  # the table header, not the comment mention
    for key in (
        "max_attempts_per_model",
        "max_rotations",
        "max_probe_attempts",
        "output_headroom_tokens",
        "input_margin_pct",
        "strict_context_guard",
        "retry_backoff_seconds",
        "preflight_timeout_seconds",
        "probe_timeout_seconds",
    ):
        # An ACTIVE top-level key: start of line, key, optional alignment spaces, '='.
        match = re.search(rf"(?m)^{key} *=", text)
        assert match is not None, f"{key} missing as an active key"
        assert match.start() < models_at, f"{key} must precede [models]"


def test_template_rotation_tunables_reparse_to_the_defaults(tmp_path):
    # The emitted values ARE the built-in defaults: editing them is opt-in, and an
    # untouched scaffold still round-trips to DEFAULT_* (invariant #4 preserved).
    from ollama_config import (
        DEFAULT_MAX_ATTEMPTS_PER_MODEL,
        DEFAULT_MAX_PROBE_ATTEMPTS,
        DEFAULT_MAX_ROTATIONS,
        DEFAULT_OUTPUT_HEADROOM_TOKENS,
        DEFAULT_INPUT_MARGIN_PCT,
        DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
        DEFAULT_PROBE_TIMEOUT_SECONDS,
        DEFAULT_RETRY_BACKOFF_SECONDS,
        DEFAULT_STRICT_CONTEXT_GUARD,
    )

    path = tmp_path / "magi-ollama.toml"
    path.write_text(render_template(), encoding="utf-8")
    cfg = resolve_config(repo_path=str(path), global_path=None, env={})
    assert cfg.max_attempts_per_model == DEFAULT_MAX_ATTEMPTS_PER_MODEL
    assert cfg.max_rotations == DEFAULT_MAX_ROTATIONS
    assert cfg.max_probe_attempts == DEFAULT_MAX_PROBE_ATTEMPTS
    assert cfg.output_headroom_tokens == DEFAULT_OUTPUT_HEADROOM_TOKENS
    assert cfg.input_margin_pct == DEFAULT_INPUT_MARGIN_PCT
    assert cfg.strict_context_guard == DEFAULT_STRICT_CONTEXT_GUARD
    assert cfg.retry_backoff_seconds == DEFAULT_RETRY_BACKOFF_SECONDS
    assert cfg.preflight_timeout_seconds == DEFAULT_PREFLIGHT_TIMEOUT_SECONDS
    assert cfg.probe_timeout_seconds == DEFAULT_PROBE_TIMEOUT_SECONDS
