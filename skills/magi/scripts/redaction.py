#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 5.0.0
# Date: 2026-07-11
"""The single point where secrets are stripped from anything that becomes text."""

from __future__ import annotations

_REDACTED = "***REDACTED***"


def redact_secrets(text: str, api_key: str | None) -> str:
    """Strip *api_key* from *text*.

    Applied at EVERY boundary where an exception becomes a string: rotation
    notices, ``fallback_reason.detail``, preflight/probe/fast-fail errors. A single
    call site is the point -- a scattered redaction is a forgotten redaction.

    Args:
        text: Text that may embed the key (an error message, a URL, a header).
        api_key: The configured key, or None when the endpoint needs no auth.

    Returns:
        *text* with every occurrence of *api_key* replaced. Returned unchanged
        when there is no key to hide (``None`` or empty -- an empty key is falsy,
        and replacing "" would splice the placeholder between every character).
    """
    if not api_key:
        return text
    return text.replace(api_key, _REDACTED)
