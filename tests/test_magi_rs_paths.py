# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-09-18
"""Tests for locating the plugin's own bin/ and pin (MS3)."""

import os

import pytest

from magi_rs_paths import PluginRootError, bin_dir, pin_path, plugin_root


def test_a_configured_plugin_root_wins(tmp_path):
    assert plugin_root({"CLAUDE_PLUGIN_ROOT": str(tmp_path)}) == str(tmp_path)


@pytest.mark.parametrize("value", ["", "/does/not/exist"])
def test_a_configured_root_that_is_not_a_directory_aborts_naming_it(value):
    # BDD-3i: never the cwd, never PATH, never a guess.
    with pytest.raises(PluginRootError) as exc:
        plugin_root({"CLAUDE_PLUGIN_ROOT": value})
    assert "CLAUDE_PLUGIN_ROOT" in str(exc.value)


def test_an_absent_variable_resolves_to_the_tree_the_code_ships_in():
    # The project's own gate runs run_magi.py from a plain shell with no
    # CLAUDE_PLUGIN_ROOT: the tree the pin ships in is the only deterministic
    # answer, and it is where the pin actually is.
    root = plugin_root({})
    assert os.path.isfile(pin_path(root))
    assert bin_dir(root) == os.path.join(root, "bin")
