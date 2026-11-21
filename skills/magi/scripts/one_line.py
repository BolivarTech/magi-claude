# Author: Julian Bolivar
# Version: 8.0.0
# Date: 2026-11-21
"""Collapse a message onto one line, for every one-line error the CLI prints."""

from __future__ import annotations


def one_line(text: str) -> str:
    """Collapse every run of whitespace, newlines included, into one space.

    Args:
        text: Any message.

    Returns:
        ``text`` on a single line, without leading or trailing whitespace.
    """
    return " ".join(text.split())
