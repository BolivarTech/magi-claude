# tests/test_ollama_config.py
"""Layered, per-key resolution of OllamaConfig (BDD-4,5,6)."""

import pytest
from validate import ValidationError
from ollama_config import (
    DEFAULT_MODELS,
    DEFAULT_BASE_URL,
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
        'base_url="http://g:11434/v1"\n[models]\nmelchior="gm"\nbalthasar="gb"\ncaspar="gc"\n',
    )
    r = _write(tmp_path / "r.toml", 'base_url="http://r:11434/v1"\n')
    cfg = resolve_config(global_path=g, repo_path=r, env={})
    assert cfg.base_url == "http://r:11434/v1"  # repo wins
    assert cfg.models == {"melchior": "gm", "balthasar": "gb", "caspar": "gc"}  # from global


def test_env_overrides_files(tmp_path):
    r = _write(tmp_path / "r.toml", '[models]\ncaspar="rc"\n')
    cfg = resolve_config(
        global_path="/nope.toml", repo_path=r, env={"MAGI_OLLAMA_MODEL_CASPAR": "ec"}
    )
    assert cfg.models["caspar"] == "ec"


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


def test_ollama_config_structured_default_is_schema():
    cfg = OllamaConfig(
        base_url=DEFAULT_BASE_URL,
        api_key=None,
        models=dict(DEFAULT_MODELS),
    )
    assert cfg.structured == "schema"


def test_empty_base_url_in_toml_falls_through_to_default(tmp_path):
    """An empty base_url in a TOML file must be treated as unset (fall through).

    BDD: Given a repo TOML with base_url="", When resolve_config is called,
    Then cfg.base_url equals DEFAULT_BASE_URL, NOT the malformed "http:///v1".
    """
    r = _write(tmp_path / "r.toml", 'base_url=""\n')
    cfg = resolve_config(global_path="/nope.toml", repo_path=r, env={})
    assert cfg.base_url == DEFAULT_BASE_URL
