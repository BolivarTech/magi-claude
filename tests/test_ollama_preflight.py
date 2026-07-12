# tests/test_ollama_preflight.py
import io
import json
import urllib.error
import pytest
from ollama_config import ModelSpec, OllamaConfig
from ollama_preflight import preflight, OllamaPreflightError, PREFLIGHT_TIMEOUT, _is_cloud_tag


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _cfg(api_key=None):
    return OllamaConfig(
        base_url="http://h:11434/v1",
        api_key=api_key,
        models={
            "melchior": ModelSpec("m", "la"),
            "balthasar": ModelSpec("b", "lb"),
            "caspar": ModelSpec("c", "lc"),
        },
    )


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
    """The signin-specific branch fires when ALL trio tags are cloud-tagged
    (including -cloud variants like gpt-oss:120b-cloud) and NO cloud model
    is available. Asserts text UNIQUE to that branch, not shared with the
    generic message."""
    cfg = OllamaConfig(
        base_url="http://h:11434/v1",
        api_key=None,
        models={
            "melchior": ModelSpec("glm-5:cloud", "zhipu"),
            "balthasar": ModelSpec("gpt-oss:120b-cloud", "openai"),
            "caspar": ModelSpec("deepseek-v4-pro:cloud", "deepseek"),
        },
    )
    _patch(monkeypatch, body=_models_body(["llama3:8b", "qwen3:8b"]))  # none :cloud
    with pytest.raises(OllamaPreflightError) as ei:
        preflight(cfg)
    # "No :cloud models available" is ONLY in the signin branch, NOT in the
    # generic "Missing models" message — this distinguishes the two branches.
    assert "No :cloud models available" in str(ei.value)


def test_generic_missing_model_fires_when_some_cloud_available(monkeypatch):
    """Regression: the GENERIC message (not signin) fires when the daemon DOES
    expose some cloud model but a specific trio tag is still absent.

    Scenario: balthsar's tag 'new-model:cloud' is missing from the daemon, but
    the daemon lists 'glm-5:cloud' and 'deepseek-v4-pro:cloud'. Since not ALL
    cloud models are absent (none_cloud_available=False), the cloud-no-signin
    diagnostic must NOT fire; the generic "Missing models" branch must fire.
    """
    cfg = OllamaConfig(
        base_url="http://h:11434/v1",
        api_key=None,
        models={
            "melchior": ModelSpec("glm-5:cloud", "zhipu"),
            "balthasar": ModelSpec("new-model:cloud", "openai"),
            "caspar": ModelSpec("deepseek-v4-pro:cloud", "deepseek"),
        },
    )
    # Daemon lists two of the three — new-model:cloud is missing, but cloud IS available.
    _patch(monkeypatch, body=_models_body(["glm-5:cloud", "deepseek-v4-pro:cloud"]))
    with pytest.raises(OllamaPreflightError) as ei:
        preflight(cfg)
    msg = str(ei.value)
    # Generic branch fires: "Missing models" present, signin-specific branch text absent.
    assert "Missing models" in msg
    assert "No :cloud models available" not in msg


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


def test_is_cloud_tag_rejects_non_suffix_cloud():
    """_is_cloud_tag must match ONLY exact ':cloud' or '-cloud' suffix.

    True cases: tags whose variant is exactly 'cloud' or ends in '-cloud'.
    False cases: 'precloud' (substring, not exact suffix), untagged names,
    and bare strings without a ':' separator.
    """
    # Must return True — exact :cloud or -cloud suffix
    assert _is_cloud_tag("glm-5:cloud") is True
    assert _is_cloud_tag("gpt-oss:120b-cloud") is True
    assert _is_cloud_tag("deepseek-v4-pro:cloud") is True
    # Must return False — 'cloud' only as substring, not exact suffix
    assert _is_cloud_tag("foo:precloud") is False
    assert _is_cloud_tag("llama3.1:8b") is False
    assert _is_cloud_tag("mycloud") is False
