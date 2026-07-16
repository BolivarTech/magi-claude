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
from datetime import timezone

import pytest
from ollama_config import OllamaConfig
from ollama_backend import OllamaBackend, OllamaHTTPError, _raise_http_error

# MS2: the model's content carries the delimited block. With no markers, the sentinel
# rejects -- which is exactly the point: a bare verdict is no longer accepted (R15).
_OK_CONTENT = (
    "<MAGI_VERDICT>\n"
    '{"agent":"melchior","verdict":"approve","confidence":0.8,"summary":"s",'
    '"reasoning":"r","findings":[],"recommendation":"go"}\n'
    "</MAGI_VERDICT>"
)

_OK_BODY = json.dumps({"choices": [{"message": {"content": _OK_CONTENT}}]}).encode()


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _cfg(api_key=None):
    return OllamaConfig(
        base_url="http://h:11434/v1",
        api_key=api_key,
        models={"melchior": "m", "balthasar": "b", "caspar": "c"},
    )


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
    sp = tmp_path / "melchior.md"
    sp.write_text("SYS", encoding="utf-8")
    return asyncio.run(OllamaBackend(cfg).run("melchior", str(sp), "P", model, 900, str(tmp_path)))


def test_builds_chat_completions_request(monkeypatch, tmp_path):
    cap = _backend_with(monkeypatch)
    _run(_cfg(), tmp_path)
    req = cap["req"]
    assert req.full_url == "http://h:11434/v1/chat/completions"
    body = json.loads(req.data)
    assert body["model"] == "m" and body["stream"] is False
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["content"] == "P"


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
    assert raw.decode() == _OK_CONTENT  # MS2: the content arrives VERBATIM, markers and all


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


def test_the_request_body_NEVER_carries_response_format(monkeypatch, tmp_path):
    """R7: ``response_format`` and the markers are MUTUALLY EXCLUSIVE.

    A model that **honours** ``json_schema`` is constrained to emit a JSON object and so
    **cannot** emit ``<MAGI_VERDICT>``. And what ``response_format`` bought -- suppressing
    the prose -- is exactly what the sentinel makes irrelevant: **everything outside the
    markers is ignored**. (And it never guaranteed it anyway: glm-5.2 fenced its output
    **with** ``json_schema`` active.)
    """
    cap = _backend_with(monkeypatch)
    _run(_cfg(), tmp_path)
    assert "response_format" not in json.loads(cap["req"].data)


def test_bdd7_localhost_cloud_tag_no_key_no_auth(monkeypatch, tmp_path):  # BDD-7
    cap = _backend_with(monkeypatch)
    cfg = OllamaConfig(
        base_url="http://localhost:11434/v1",
        api_key=None,
        models={"melchior": "deepseek-v4-pro:cloud", "balthasar": "x", "caspar": "y"},
    )
    sp = tmp_path / "melchior.md"
    sp.write_text("S", encoding="utf-8")
    asyncio.run(
        OllamaBackend(cfg).run(
            "melchior", str(sp), "P", "deepseek-v4-pro:cloud", 900, str(tmp_path)
        )
    )
    assert cap["req"].get_header("Authorization") is None  # mode A: daemon attaches creds
    assert cap["req"].full_url == "http://localhost:11434/v1/chat/completions"


def test_dict_content_serialized_as_valid_json(monkeypatch, tmp_path):  # Finding B
    """When message.content is already a decoded dict, the backend must return
    valid UTF-8 JSON bytes — not a Python repr string."""
    dict_content = {
        "agent": "melchior",
        "verdict": "approve",
        "confidence": 0.8,
        "summary": "s",
        "reasoning": "r",
        "findings": [],
        "recommendation": "go",
    }
    body = json.dumps({"choices": [{"message": {"content": dict_content}}]}).encode()
    _backend_with(monkeypatch, body=body)
    raw = _run(_cfg(), tmp_path)
    # Must round-trip to a dict identical to dict_content.
    parsed = json.loads(raw)
    assert parsed["agent"] == dict_content["agent"]
    assert parsed["verdict"] == dict_content["verdict"]


def test_roundtrip_bare_content_parses(monkeypatch, tmp_path):  # BDD-29
    _backend_with(monkeypatch)
    raw = _run(_cfg(), tmp_path)
    raw_file = tmp_path / "melchior.raw.json"
    raw_file.write_bytes(raw)
    parsed_file = tmp_path / "melchior.json"
    from parse_agent_output import parse_agent_output as parse_raw_output
    from synthesize import load_agent_output

    parse_raw_output(str(raw_file), str(parsed_file))
    data = load_agent_output(str(parsed_file))
    assert data["agent"] == "melchior" and data["verdict"] == "approve"


# ----------------------------------------------------------------------------
# Task 4 (MS3): OllamaHTTPError exposes status / Retry-After / receipt so the
# retry loop's backoff can pick exponential-vs-flat and honor Retry-After (R2, R5).
# ----------------------------------------------------------------------------


def test_http_error_carries_status_retry_after_and_receipt():
    hdrs = {"Retry-After": "42"}
    exc = urllib.error.HTTPError("u", 429, "Too Many Requests", hdrs, None)
    with pytest.raises(OllamaHTTPError) as ei:
        _raise_http_error(exc, redact=lambda s: s)
    err = ei.value
    assert err.status == 429
    assert err.retry_after == "42"
    assert err.receipt.tzinfo is timezone.utc
    assert isinstance(err, RuntimeError)  # _classify still sees a RuntimeError


def test_http_error_without_retry_after_header_is_none():
    exc = urllib.error.HTTPError("u", 500, "ServerErr", {}, None)
    with pytest.raises(OllamaHTTPError) as ei:
        _raise_http_error(exc, redact=lambda s: s)
    assert ei.value.retry_after is None


def test_raise_http_error_redacts_the_message():
    exc = urllib.error.HTTPError("u", 500, "sk-secret leaked", {}, None)
    with pytest.raises(OllamaHTTPError) as ei:
        _raise_http_error(exc, redact=lambda s: s.replace("sk-secret", "***"))
    assert "sk-secret" not in str(ei.value)


def test_ollama_http_error_is_classified_as_http_directly():
    # gate CP2 loop 6 (Balthasar): a DIRECT guard on the message->classify coupling, in
    # T4's own task, so a message-format drift breaks HERE (not only via the backend
    # message-contract test). Imported locally to avoid a run_magi import cycle at top.
    from run_magi import _classify, _FAIL_HTTP

    exc = urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None)
    with pytest.raises(OllamaHTTPError) as ei:
        _raise_http_error(exc, redact=lambda s: s)
    assert _classify(ei.value) == _FAIL_HTTP


def test_run_raises_ollama_http_error_with_retry_after_end_to_end(monkeypatch, tmp_path):
    """The real _call() path (not just the helper in isolation) surfaces the new
    fields -- guards against _raise_http_error being wired up but never called."""
    hdrs = {"Retry-After": "7"}
    err = urllib.error.HTTPError("u", 503, "Service Unavailable", hdrs, None)
    _backend_with(monkeypatch, exc=err)
    with pytest.raises(OllamaHTTPError) as ei:
        _run(_cfg(), tmp_path)
    assert ei.value.status == 503
    assert ei.value.retry_after == "7"


def test_404_path_is_unaffected_still_plain_runtimeerror(monkeypatch, tmp_path):
    """The 404 branch is untouched by T4 -- it stays a plain RuntimeError, not
    OllamaHTTPError (it is not reintentable/transitory, R5)."""
    err = urllib.error.HTTPError("u", 404, "NotFound", {}, io.BytesIO(b"gone"))
    _backend_with(monkeypatch, exc=err)
    with pytest.raises(RuntimeError) as ei:
        _run(_cfg(), tmp_path)
    assert not isinstance(ei.value, OllamaHTTPError)
