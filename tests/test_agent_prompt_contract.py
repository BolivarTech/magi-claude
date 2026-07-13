# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-13
"""Anchors the cross-file contract between the code and the three system prompts.

The markers and the canary's fingerprint live in ``verdict_markers.py``, but **the model
only sees the ``.md`` files**. If they drift apart, two things happen and neither is good:

* if a **marker** breaks, the mage **dies** (noisy);
* if the **canary's fingerprint** breaks, the guard **disarms itself in SILENCE** -- it is
  still there, still running, and it no longer protects anything. That is the worst
  possible failure mode.

These tests make the drift **break the build**, not production.
"""

import json
from pathlib import Path

import pytest

from verdict_markers import ECHO_CANARY, VERDICT_CLOSE, VERDICT_OPEN, VerdictSentinel

AGENTS_DIR = Path(__file__).parent.parent / "skills" / "magi" / "agents"
PROMPTS = ["melchior.md", "balthasar.md", "caspar.md"]


@pytest.mark.parametrize("name", PROMPTS)
class TestPromptContract:
    @staticmethod
    def _text(name: str) -> str:
        # utf-8-sig, not utf-8: it eats the BOM BEFORE anything is compared.
        return (AGENTS_DIR / name).read_text(encoding="utf-8-sig")

    def _example_line(self, name: str) -> str:
        return next(
            line for line in self._text(name).splitlines() if line.lstrip().startswith('{"agent"')
        )

    def test_declares_exactly_one_marker_pair(self, name):
        """Exactly ONE pair: the same rule as R3a, applied to OUR OWN files.

        If the prompt teaches the format **twice**, the model sees **two examples** and the
        guard does not even know **which** pair to inspect.
        """
        sentinel = VerdictSentinel()
        lines = self._text(name).splitlines()
        assert sum(sentinel.is_exact_marker_line(ln, VERDICT_OPEN) for ln in lines) == 1
        assert sum(sentinel.is_exact_marker_line(ln, VERDICT_CLOSE) for ln in lines) == 1

    def test_nothing_between_the_markers_is_a_valid_json_object(self, name):
        """The LONE echo produces ONE block -- the ambiguity guard **does not see it**.

        If there were a valid verdict between the markers, a model that copied that block
        would fabricate an ``approve`` **sentinel and all**. Between the markers goes a SLOT.
        """
        with pytest.raises(json.JSONDecodeError):
            json.loads(VerdictSentinel().extract(self._text(name)))

    def test_the_canary_fingerprint_is_still_in_the_example_line(self, name):
        """If someone edits the example without updating ``ECHO_CANARY``, the canary compares
        against a text **nobody emits any more**: a green test over a disarmed guard."""
        for value in ECHO_CANARY.values():
            assert value in self._example_line(name)

    def test_the_example_names_ITS_OWN_agent(self, name):
        """An example carrying the wrong mage's name **kills that mage on EVERY run**.

        If ``melchior.md`` ships an example that says ``"agent": "caspar"``, the model
        imitates it and the identity check (R10) discards it **every single time**.
        """
        agent = name.removesuffix(".md")
        assert f'"agent": "{agent}"' in self._example_line(name)

    def test_the_worked_example_is_NOT_an_approve(self, name):
        """A belt over the canary: if something got copied despite everything, let it **not**
        be a fabricated approve in the adversarial seat."""
        assert '"verdict": "approve"' not in self._example_line(name)
