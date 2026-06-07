# tests/test_ollama_preflight.py
import io
import json
import urllib.error
import pytest
from ollama_config import OllamaConfig
from ollama_preflight import preflight, OllamaPreflightError, PREFLIGHT_TIMEOUT


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def _cfg(api_key=None):
    return OllamaConfig(base_url="http://h:11434/v1", api_key=api_key,
                        models={"melchior": "m", "balthasar": "b", "caspar": "c"})


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


def test_passes_when_all_models_present(monkeypatch):
    cap = _patch(monkeypatch, body=_models_body(["m", "b", "c", "x"]))
    preflight(_cfg())  # no raise
    assert cap["url"] == "http://h:11434/v1/models"
    assert cap["timeout"] == PREFLIGHT_TIMEOUT


def test_missing_model_aborts_with_name(monkeypatch):
    _patch(monkeypatch, body=_models_body(["m", "b"]))
    with pytest.raises(OllamaPreflightError) as ei:
        preflight(_cfg())
    assert "c" in str(ei.value)


def test_cloud_tags_no_signin_emits_signin_hint(monkeypatch):  # BDD-27
    cfg = OllamaConfig(base_url="http://h:11434/v1", api_key=None,
                       models={"melchior": "glm-5:cloud", "balthasar": "gpt-oss:120b-cloud",
                               "caspar": "deepseek-v4-pro:cloud"})
    _patch(monkeypatch, body=_models_body(["llama3:8b", "qwen3:8b"]))  # none :cloud
    with pytest.raises(OllamaPreflightError) as ei:
        preflight(cfg)
    assert "signin" in str(ei.value).lower()


def test_unreachable_aborts(monkeypatch):
    _patch(monkeypatch, exc=urllib.error.URLError("refused"))
    with pytest.raises(OllamaPreflightError):
        preflight(_cfg())


def test_auth_error_redacts_key(monkeypatch):
    _patch(monkeypatch, exc=urllib.error.HTTPError("u", 401, "Unauthorized", {}, None))
    with pytest.raises(OllamaPreflightError) as ei:
        preflight(_cfg(api_key="sk-secret"))
    assert "sk-secret" not in str(ei.value)


def test_models_endpoint_404_warns_and_proceeds(monkeypatch, capsys):
    _patch(monkeypatch, exc=urllib.error.HTTPError("u", 404, "NF", {}, None))
    preflight(_cfg())  # no raise
    assert "models" in capsys.readouterr().err.lower()
