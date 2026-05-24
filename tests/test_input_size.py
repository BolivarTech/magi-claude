# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-24
"""Tests for input_size.py — input-size estimation + oversize detection."""

from __future__ import annotations


class TestInputSize:
    def test_estimate_tokens_is_chars_over_four(self):
        from input_size import estimate_tokens

        assert estimate_tokens("a" * 400) == 100
        assert estimate_tokens("") == 0

    def test_check_input_size_flags_oversize(self):
        from input_size import check_input_size

        est, exceeds = check_input_size("x" * 4000, threshold=100)  # ~1000 tokens > 100
        assert est == 1000 and exceeds is True

    def test_check_input_size_not_oversize_at_or_below_threshold(self):
        from input_size import check_input_size

        est, exceeds = check_input_size("x" * 400, threshold=100)  # ~100 tokens, not > 100
        assert est == 100 and exceeds is False

    def test_estimate_tokens_counts_unicode_code_points_not_bytes(self):
        """estimate_tokens uses len() which counts Unicode code points, not bytes.

        A CJK character (e.g. U+4E2D, len() == 1) encodes as 3 bytes in UTF-8
        but counts as a single code point for the heuristic.  400 CJK chars ->
        400 // 4 == 100 estimated tokens (same as 400 ASCII chars).
        """
        from input_size import estimate_tokens

        assert estimate_tokens("中" * 400) == 100

    def test_warn_input_tokens_default_is_150000(self):
        """Pin the WARN_INPUT_TOKENS constant so accidental changes are caught."""
        from input_size import WARN_INPUT_TOKENS

        assert WARN_INPUT_TOKENS == 150_000
