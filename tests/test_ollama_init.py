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


def test_template_documents_the_kill_switch_and_the_pull_requirement():
    text = render_template()
    assert "MAGI_OLLAMA_MAX_ROTATIONS=0" in text  # R17 must be discoverable
    assert "ollama pull" in text  # fallbacks need manifests
    assert "lineage" in text
