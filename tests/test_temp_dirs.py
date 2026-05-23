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


class TestLegacySweep:
    """BDD-17/18: one-shot removal of pre-2.6.0 magi-run-* dirs under temp."""

    def test_removes_old_keeps_recent_and_marks(self, tmp_path):
        import time

        import temp_dirs

        old = tmp_path / "magi-run-old"
        old.mkdir()
        os.utime(old, (1000, 1000))  # ancient mtime -> older than threshold
        recent = tmp_path / "magi-run-recent"
        recent.mkdir()
        now = time.time()
        os.utime(recent, (now, now))

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            temp_dirs.sweep_legacy_runs_once()

        assert not old.exists(), "Old legacy dir must be swept"
        assert recent.exists(), "Recent legacy dir must be preserved"
        marker = tmp_path / "magi-runs" / temp_dirs.LEGACY_SWEEP_MARKER
        assert marker.exists(), "Sweep marker must be written"

    def test_second_call_is_noop_after_marker(self, tmp_path):
        import temp_dirs

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            temp_dirs.sweep_legacy_runs_once()  # creates marker
            # A legacy dir created AFTER the marker must NOT be swept.
            late = tmp_path / "magi-run-late"
            late.mkdir()
            os.utime(late, (1000, 1000))
            temp_dirs.sweep_legacy_runs_once()

        assert late.exists(), "Once marked, the global temp is not re-scanned"

    def test_container_dir_is_not_a_candidate(self, tmp_path):
        """BDD-18: the magi-runs container never matches the run prefix."""
        import temp_dirs

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            container = temp_dirs.project_run_root(str(tmp_path / "projA"))
            temp_dirs.sweep_legacy_runs_once()

        assert os.path.isdir(container), "Namespaced container must survive the sweep"
        assert os.path.isdir(tmp_path / "magi-runs")
