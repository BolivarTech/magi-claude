# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-11
"""Tests for the single api-key redaction point (NR3b)."""

from redaction import redact_secrets


def test_configured_key_is_stripped_from_an_error_message():
    key = "sk-secret-abc123"
    text = f"Auth failed for 'Bearer {key}' at the endpoint"
    out = redact_secrets(text, key)
    assert key not in out
    assert "REDACTED" in out


def test_none_key_leaves_the_text_untouched():
    text = "Cannot reach Ollama at http://host:11434"
    assert redact_secrets(text, None) == text


def test_empty_key_leaves_the_text_untouched():
    # "" is falsy: an unauthenticated endpoint must not turn every string into
    # a redaction of the empty string (which str.replace would splice everywhere).
    text = "some error text"
    assert redact_secrets(text, "") == text


def test_the_key_never_survives_even_across_multiple_occurrences():
    key = "sk-xyz-9"
    text = f"{key} leaked here, and again {key}, and trailing {key}"
    out = redact_secrets(text, key)
    assert key not in out
