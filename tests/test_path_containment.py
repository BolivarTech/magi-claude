# Author: Julian Bolivar
# Version: 8.0.0
# Date: 2026-11-21
"""Tests for the shared path containment helper."""

import os

from path_containment import canonical_path, is_within


def test_a_path_below_the_root_is_within(tmp_path):
    assert is_within(str(tmp_path / "a" / "b"), str(tmp_path))


def test_the_root_itself_is_within(tmp_path):
    assert is_within(str(tmp_path), str(tmp_path))


def test_a_sibling_with_the_same_prefix_is_not_within(tmp_path):
    assert not is_within(str(tmp_path / "magi-other"), str(tmp_path / "magi"))


def test_dot_dot_cannot_escape_the_check(tmp_path):
    assert not is_within(str(tmp_path / "root" / ".." / "elsewhere"), str(tmp_path / "root"))


def test_the_canonical_form_is_absolute_and_stable(tmp_path):
    once = canonical_path(str(tmp_path / "x" / ".." / "y"))
    assert os.path.isabs(once) and canonical_path(once) == once
