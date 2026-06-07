#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-06-06
"""OllamaBackend talks /v1/chat/completions via urllib (mocked, no network)."""
import asyncio
import io
import json
import socket
import urllib.error
import pytest
from ollama_config import OllamaConfig
from ollama_backend import OllamaBackend

_OK_BODY = json.dumps({
    "choices": [{"message": {"content": '{"agent":"melchior","verdict":"approve",'
                 '"confidence":0.8,"summary":"s","reasoning":"r","findings":[],'
                 '"recommendation":"go"}'}}]
}).encode()


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def _cfg(api_key=None):
    return OllamaConfig(base_url="http://h:11434/v1", api_key=api_key,
                        models={"melchior": "m", "balthasar": "b", "caspar": "c"})


def _backend_with(monkeypatch, *, body=_OK_BODY, exc=None):
    captured = {}
    def fake_urlopen(req, timeout=None):
        captured["req"], captured["timeout"] = req, timeout
        if exc is not None:
            raise exc
        return _Resp(body)
    monkeypatch.setattr("ollama_backend.urllib.request.urlopen", fake_urlopen)
    return captured


def _run(cfg, tmp_path, model="m"):
    sp = tmp_path / "melchior.md"; sp.write_text("SYS", encoding="utf-8")
    return asyncio.run(OllamaBackend(cfg).run("melchior", str(sp), "P", model, 900, str(tmp_path)))


def test_builds_chat_completions_request_with_schema(monkeypatch, tmp_path):
    cap = _backend_with(monkeypatch)
    _run(_cfg(), tmp_path)
    req = cap["req"]
    assert req.full_url == "http://h:11434/v1/chat/completions"
    body = json.loads(req.data)
    assert body["model"] == "m" and body["stream"] is False
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["content"] == "P"
    assert body["response_format"]["type"] == "json_schema"


def test_auth_header_present_only_with_key(monkeypatch, tmp_path):
    cap = _backend_with(monkeypatch)
    _run(_cfg(api_key="sk-xyz"), tmp_path)
    assert cap["req"].get_header("Authorization") == "Bearer sk-xyz"

    cap2 = _backend_with(monkeypatch)
    _run(_cfg(api_key=None), tmp_path)
    assert cap2["req"].get_header("Authorization") is None


def test_extracts_message_content(monkeypatch, tmp_path):
    _backend_with(monkeypatch)
    raw = _run(_cfg(), tmp_path)
    assert json.loads(raw)["verdict"] == "approve"


def test_http_error_maps_to_runtimeerror_redacted(monkeypatch, tmp_path):
    err = urllib.error.HTTPError("u", 500, "ServerErr", {}, io.BytesIO(b"oops"))
    _backend_with(monkeypatch, exc=err)
    with pytest.raises(RuntimeError) as ei:
        _run(_cfg(api_key="sk-secret"), tmp_path)
    assert "500" in str(ei.value) and "sk-secret" not in str(ei.value)


def test_urlerror_maps_to_runtimeerror(monkeypatch, tmp_path):
    _backend_with(monkeypatch, exc=urllib.error.URLError("down"))
    with pytest.raises(RuntimeError):
        _run(_cfg(), tmp_path)


def test_timeout_maps_to_timeouterror(monkeypatch, tmp_path):
    _backend_with(monkeypatch, exc=socket.timeout("slow"))
    with pytest.raises(TimeoutError):
        _run(_cfg(), tmp_path)


def test_missing_choices_maps_to_valueerror(monkeypatch, tmp_path):
    _backend_with(monkeypatch, body=json.dumps({"nope": 1}).encode())
    with pytest.raises(ValueError):
        _run(_cfg(), tmp_path)


def test_downgrade_on_400_response_format(monkeypatch, tmp_path):  # BDD-25
    calls = {"n": 0}
    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError("u", 400, "Bad", {},
                                         io.BytesIO(b"unsupported response_format"))
        return _Resp(_OK_BODY)
    monkeypatch.setattr("ollama_backend.urllib.request.urlopen", fake_urlopen)
    raw = _run(_cfg(), tmp_path)
    assert calls["n"] == 2  # downgraded then retried without response_format
    assert json.loads(raw)["verdict"] == "approve"


def test_structured_off_omits_response_format(monkeypatch, tmp_path):  # BDD-28
    cap = _backend_with(monkeypatch)
    cfg = OllamaConfig(base_url="http://h:11434/v1", api_key=None,
                       models={"melchior": "m", "balthasar": "b", "caspar": "c"},
                       structured="off")
    sp = tmp_path / "melchior.md"; sp.write_text("S", encoding="utf-8")
    asyncio.run(OllamaBackend(cfg).run("melchior", str(sp), "P", "m", 900, str(tmp_path)))
    assert "response_format" not in json.loads(cap["req"].data)


def test_bdd7_localhost_cloud_tag_no_key_no_auth(monkeypatch, tmp_path):  # BDD-7
    cap = _backend_with(monkeypatch)
    cfg = OllamaConfig(base_url="http://localhost:11434/v1", api_key=None,
                       models={"melchior": "deepseek-v4-pro:cloud", "balthasar": "x", "caspar": "y"})
    sp = tmp_path / "melchior.md"; sp.write_text("S", encoding="utf-8")
    asyncio.run(OllamaBackend(cfg).run("melchior", str(sp), "P",
                                       "deepseek-v4-pro:cloud", 900, str(tmp_path)))
    assert cap["req"].get_header("Authorization") is None  # mode A: daemon attaches creds
    assert cap["req"].full_url == "http://localhost:11434/v1/chat/completions"


def test_roundtrip_bare_content_parses(monkeypatch, tmp_path):  # BDD-29
    _backend_with(monkeypatch)
    raw = _run(_cfg(), tmp_path)
    raw_file = tmp_path / "melchior.raw.json"; raw_file.write_bytes(raw)
    parsed_file = tmp_path / "melchior.json"
    from parse_agent_output import parse_agent_output as parse_raw_output
    from synthesize import load_agent_output
    parse_raw_output(str(raw_file), str(parsed_file))
    data = load_agent_output(str(parsed_file))
    assert data["agent"] == "melchior" and data["verdict"] == "approve"
