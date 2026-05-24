# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-23
"""Tests for finding_id.py — stable structured finding identity."""

from __future__ import annotations


class TestNormalizePath:
    def test_backslash_dotprefix_and_double_slash(self):
        from finding_id import normalize_path

        assert normalize_path("src\\a.py") == "src/a.py"
        assert normalize_path("./src/a.py") == "src/a.py"
        assert normalize_path("src//a.py") == "src/a.py"
        assert normalize_path(".\\src\\a.py") == "src/a.py"


class TestNormalizeCategory:
    def test_known_passthrough_and_unknown_to_other(self):
        from finding_id import normalize_category

        assert normalize_category("injection") == "injection"
        assert normalize_category("memory-leak") == "other"
        assert normalize_category(None) == "other"
        assert normalize_category("Error_Handling") == "error-handling"


class TestGenerateFindingId:
    def test_stable_and_title_independent(self):
        from finding_id import generate_finding_id

        a = generate_finding_id("src/a.py", 10, "logic-error")
        b = generate_finding_id("src/a.py", 10, "logic-error")
        assert a == b and len(a) == 16

    def test_path_normalized_into_id(self):
        from finding_id import generate_finding_id

        assert generate_finding_id("src\\a.py", 10, "logic-error") == generate_finding_id(
            "src/a.py", 10, "logic-error"
        )

    def test_different_category_different_id(self):
        from finding_id import generate_finding_id

        assert generate_finding_id("src/a.py", 10, "logic-error") != generate_finding_id(
            "src/a.py", 10, "injection"
        )
