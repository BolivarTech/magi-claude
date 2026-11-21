# Author: Julian Bolivar
# Version: 8.0.0
# Date: 2026-11-21
"""The preflight refuses a linked ``.magi/`` like every other entry point (MS1 gate, O4)."""

import os
import sys

import pytest

from magi_rs_preflight import ProjectNotInitialisedError, ProjectPreflight


def test_a_linked_magi_dir_is_refused_even_when_the_target_holds_a_config(tmp_path):
    # [O4] the fourth entry point: isfile() follows the link, so the target's magi.toml used to pass.
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "magi.toml").write_text("x", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    try:
        if sys.platform == "win32":
            import _winapi

            _winapi.CreateJunction(str(target), str(work / ".magi"))
        else:
            os.symlink(target, work / ".magi", target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"cannot create a link here: {error}")
    with pytest.raises(ProjectNotInitialisedError) as exc:
        ProjectPreflight.require_initialised(str(work))
    assert "is a link" in str(exc.value) and "\n" not in str(exc.value)
