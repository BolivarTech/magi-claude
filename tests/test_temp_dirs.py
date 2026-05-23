# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-23
"""Tests for temp_dirs.py — per-project namespace and legacy sweep."""

from __future__ import annotations

import os

from unittest.mock import patch


class TestProjectRunRoot:
    """BDD-1/12: per-project run container under the temp namespace."""

    def test_root_is_under_magi_runs_container(self, tmp_path):
        import temp_dirs

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            root = temp_dirs.project_run_root(str(tmp_path / "projA"))

        assert os.path.isdir(root)
        assert os.path.dirname(root) == str(tmp_path / "magi-runs")

    def test_same_project_same_key(self, tmp_path):
        import temp_dirs

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            a = temp_dirs.project_run_root(str(tmp_path / "projA"))
            b = temp_dirs.project_run_root(str(tmp_path / "projA"))
        assert a == b

    def test_different_projects_different_roots(self, tmp_path):
        """BDD-1: distinct projects map to distinct, isolated roots."""
        import temp_dirs

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            a = temp_dirs.project_run_root(str(tmp_path / "projA"))
            b = temp_dirs.project_run_root(str(tmp_path / "projB"))
        assert a != b

    def test_falls_back_to_tempdir_on_makedirs_error(self, tmp_path, monkeypatch):
        """makedirs failure degrades to gettempdir() instead of raising."""
        import temp_dirs

        def boom(*a, **k):
            raise OSError("denied")

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            monkeypatch.setattr(temp_dirs.os, "makedirs", boom)
            root = temp_dirs.project_run_root(str(tmp_path / "projA"))
        assert root == str(tmp_path)
