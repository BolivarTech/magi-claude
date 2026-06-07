import os
import pytest
from ollama_config import DEFAULT_BASE_URL, DEFAULT_MODELS, resolve_config
from ollama_init import write_template, REPO_CONFIG_RELPATH


def test_creates_template_with_active_base_url_and_trio(tmp_path):
    path = write_template(repo_root=str(tmp_path))
    assert os.path.isfile(path)
    text = open(path, encoding="utf-8").read()
    assert f'base_url = "{DEFAULT_BASE_URL}"' in text     # active, not commented
    assert "# api_key" in text                             # commented
    assert "signin" in text.lower()                        # 2-mode header (F-B)
    for tag in DEFAULT_MODELS.values():
        assert tag in text


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
