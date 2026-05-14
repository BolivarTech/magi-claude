# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-14
"""Regression tests for entry-point invocation forms (2.2.8 bootstrap).

The 2.2.8 release added a 4-line ``sys.path`` bootstrap to each entry-point
script in ``skills/magi/scripts/`` so that ``python -m
skills.magi.scripts.<name>`` no longer fails with ImportError on sibling
modules. Direct invocation (``python skills/magi/scripts/<name>.py``) was
already supported via Python's script-directory auto-injection. Both forms
are pinned here so that any future change which inadvertently breaks
either invocation surfaces as a CI failure rather than a runtime crash.

See CLAUDE.md "Open technical debt / synthesize import gap [LOCKED]" for
the architectural rationale and the locked fix snippet that applies to
future outside-callers.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent

# (module-form, direct-script-form, args, output fragment that must appear)
# The fragment is checked against stdout+stderr combined so it works for
# scripts that print usage to either stream.
_ENTRY_POINTS = [
    (
        "skills.magi.scripts.run_magi",
        "skills/magi/scripts/run_magi.py",
        ["--help"],
        "code-review",  # appears in the {code-review,design,analysis} positional choice
    ),
    (
        "skills.magi.scripts.synthesize",
        "skills/magi/scripts/synthesize.py",
        ["--help"],
        "MAGI Synthesis Engine",  # argparse description string
    ),
    (
        "skills.magi.scripts.parse_agent_output",
        "skills/magi/scripts/parse_agent_output.py",
        [],  # no args — prints custom usage and exits 0
        "Usage: parse_agent_output.py",
    ),
]


@pytest.mark.parametrize(
    "module_form,_script_form,args,output_fragment",
    _ENTRY_POINTS,
    ids=[ep[0] for ep in _ENTRY_POINTS],
)
def test_entry_point_python_dash_m(
    module_form: str,
    _script_form: str,
    args: list[str],
    output_fragment: str,
) -> None:
    """``python -m <module>`` must not raise ImportError on sibling modules."""
    result = subprocess.run(
        [sys.executable, "-m", module_form, *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        shell=False,
        timeout=30,
    )
    # The bootstrap's job is to prevent ImportError under ``-m``.
    assert "ImportError" not in result.stderr, (
        f"`python -m {module_form}` raised ImportError:\n{result.stderr}"
    )
    assert "No module named" not in result.stderr, (
        f"`python -m {module_form}` failed to find a module:\n{result.stderr}"
    )
    # Sanity: the script ran far enough to print its usage/help banner.
    combined = (result.stdout or "") + (result.stderr or "")
    assert output_fragment in combined, (
        f"Expected fragment {output_fragment!r} not found in output of "
        f"`-m {module_form}`:\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )


@pytest.mark.parametrize(
    "_module_form,script_form,args,output_fragment",
    _ENTRY_POINTS,
    ids=[ep[1] for ep in _ENTRY_POINTS],
)
def test_entry_point_direct_script(
    _module_form: str,
    script_form: str,
    args: list[str],
    output_fragment: str,
) -> None:
    """``python <script_path>`` (direct) must continue to work unchanged."""
    result = subprocess.run(
        [sys.executable, script_form, *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        shell=False,
        timeout=30,
    )
    assert "ImportError" not in result.stderr, (
        f"`python {script_form}` raised ImportError:\n{result.stderr}"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert output_fragment in combined, (
        f"Expected fragment {output_fragment!r} not found in output of "
        f"`{script_form}`:\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
