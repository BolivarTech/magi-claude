# Author: Julian Bolivar
# Version: 8.0.2
# Date: 2026-11-26
"""Tests for the embedder reachability probe of the code-review preflight (REQ-11a)."""

import http.client
import urllib.error

import pytest

from magi_rs_embedder import (
    EMBEDDER_PROBE_TIMEOUT_SECONDS,
    EMBEDDINGS_PATH,
    EmbedderProbe,
    EmbedderUnreachableError,
)
from synthesize import ValidationError

_TOML = 'base_url = "http://localhost:11434/v1"\n[embedding]\nmodel = "nomic-embed-text-v2-moe:latest"\n'


def _config(tmp_path, text=_TOML):
    path = tmp_path / "magi.toml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_the_endpoint_inherits_the_root_base_url(tmp_path):
    url, model = EmbedderProbe().endpoint(_config(tmp_path))
    assert url == "http://localhost:11434/v1" + EMBEDDINGS_PATH
    assert model == "nomic-embed-text-v2-moe:latest"


def test_an_embedding_base_url_wins_over_the_root(tmp_path):
    text = _TOML.replace("[embedding]\n", '[embedding]\nbase_url = "http://gpu:11434/v1"\n')
    url, _ = EmbedderProbe().endpoint(_config(tmp_path, text))
    assert url.startswith("http://gpu:11434/v1")


def test_any_http_answer_counts_as_reachable(tmp_path):
    def opener(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    EmbedderProbe(opener=opener).check(_config(tmp_path))


def test_a_refused_connection_is_fatal_naming_endpoint_and_model(tmp_path):
    def opener(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    with pytest.raises(EmbedderUnreachableError) as exc:
        EmbedderProbe(opener=opener).check(_config(tmp_path))
    assert "localhost:11434" in str(exc.value)
    assert "nomic-embed-text-v2-moe" in str(exc.value)


def test_a_timeout_is_fatal(tmp_path):
    def opener(request, timeout):
        raise TimeoutError("slow")

    with pytest.raises(EmbedderUnreachableError):
        EmbedderProbe(opener=opener).check(_config(tmp_path))


@pytest.mark.parametrize(
    "failure",
    [TimeoutError("timed out"), urllib.error.URLError(TimeoutError("timed out"))],
    ids=["read-timeout", "connect-timeout"],
)
def test_a_timeout_names_the_seconds_and_a_cold_load_not_unreachable(tmp_path, failure):
    def opener(request, timeout):
        raise failure

    with pytest.raises(EmbedderUnreachableError) as exc:
        EmbedderProbe(opener=opener, timeout=7).check(_config(tmp_path))
    message = str(exc.value)
    assert "did not answer within 7 s" in message
    assert "cold model may still be loading" in message
    assert "unreachable" not in message
    assert "nomic-embed-text-v2-moe" in message
    assert "localhost:11434" in message


def test_a_refused_connection_still_says_unreachable_not_a_cold_load(tmp_path):
    def opener(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    with pytest.raises(EmbedderUnreachableError) as exc:
        EmbedderProbe(opener=opener).check(_config(tmp_path))
    assert "is unreachable at" in str(exc.value)
    assert "did not answer within" not in str(exc.value)


def test_the_default_budget_covers_the_measured_cold_load():
    # Measured 2026-11-24: a cold nomic-embed-text-v2-moe took 20 s to answer on a 4 GB GPU.
    assert EMBEDDER_PROBE_TIMEOUT_SECONDS > 20


def test_the_probe_sends_one_minimal_request_with_the_named_timeout(tmp_path):
    calls = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        return _Response()

    EmbedderProbe(opener=opener, timeout=3).check(_config(tmp_path))
    assert len(calls) == 1 and calls[0][1] == 3


def test_the_unreachable_error_is_not_a_validation_error():
    assert not issubclass(EmbedderUnreachableError, ValidationError)


def _never_called(request, timeout):
    raise AssertionError("no request may be sent for an unresolvable config")


def test_an_unreadable_config_is_one_error_naming_the_path_and_sends_nothing(tmp_path):
    path = str(tmp_path / "absent.toml")
    with pytest.raises(EmbedderUnreachableError) as exc:
        EmbedderProbe(opener=_never_called).check(path)
    assert path in str(exc.value)


@pytest.mark.parametrize(
    "text",
    [
        "this is not [valid toml",
        'base_url = "http://localhost:11434/v1"\n[embedding]\nbackend = "x"\n',
        'base_url = "http://localhost:11434/v1"\nembedding = "nomic"\n',
        '[embedding]\nbase_url = "localhost:11434/v1"\nmodel = "m"\n',
        '[embedding]\nbase_url = "http://"\nmodel = "m"\n',
    ],
)
def test_a_malformed_config_is_one_error_naming_the_path_and_sends_nothing(tmp_path, text):
    path = _config(tmp_path, text)
    with pytest.raises(EmbedderUnreachableError) as exc:
        EmbedderProbe(opener=_never_called).check(path)
    assert path in str(exc.value)


def test_userinfo_never_appears_in_a_message(tmp_path):
    def opener(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    text = _TOML.replace("http://localhost", "http://user:pass@localhost")
    with pytest.raises(EmbedderUnreachableError) as exc:
        EmbedderProbe(opener=opener).check(_config(tmp_path, text))
    assert "user:pass" not in str(exc.value) and "localhost:11434" in str(exc.value)


def test_a_reply_that_is_not_http_counts_as_reachable(tmp_path):
    def opener(request, timeout):
        raise http.client.BadStatusLine("garbage")

    EmbedderProbe(opener=opener).check(_config(tmp_path))


def test_the_http_error_response_is_closed(tmp_path):
    closed = []

    class _Body:
        def read(self):
            return b""

        def close(self):
            closed.append(True)

    def opener(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 500, "boom", {}, _Body())

    EmbedderProbe(opener=opener).check(_config(tmp_path))
    assert closed


def test_a_client_side_invalid_url_is_fatal_not_an_answer(tmp_path):
    # InvalidURL subclasses HTTPException; it means the request never left, not that
    # someone answered, so it must not count as reachable.
    def opener(request, timeout):
        raise http.client.InvalidURL("nonnumeric port: 'x'")

    config = _config(tmp_path)
    with pytest.raises(EmbedderUnreachableError) as exc:
        EmbedderProbe(opener=opener).check(config)
    assert config in str(exc.value)
