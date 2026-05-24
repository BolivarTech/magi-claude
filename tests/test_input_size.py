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
