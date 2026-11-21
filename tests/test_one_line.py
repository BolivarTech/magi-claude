# Author: Julian Bolivar
# Version: 7.1.0
# Date: 2026-11-19
"""Tests for one_line.py — the single whitespace collapse behind every one-line error."""

from __future__ import annotations

import pytest

from one_line import one_line


@pytest.mark.parametrize(
    "text, expected",
    [
        ("cannot read\nthe key", "cannot read the key"),
        ("a\r\n  b\t\tc", "a b c"),
        ("  padded  ", "padded"),
        ("already one line", "already one line"),
        ("", ""),
        (" \n\t ", ""),
    ],
)
def test_one_line_collapses_every_whitespace_run_into_one_space(text, expected):
    assert one_line(text) == expected


def test_one_line_also_collapses_unicode_line_separators():
    # A message quoting a child's output may carry U+2028/U+2029, which a terminal breaks on too.
    assert one_line("first second third") == "first second third"
