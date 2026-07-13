# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-13

"""Installation-time guard for agent prompt files.

This guard covers what the anchoring test cannot see: the user's installation.
The anchoring test runs in the developer's repo; the stale-copy bug
(``mklink /D`` silently degrading to a copy on Windows) produces old prompts
with a new parser on the user's machine, where no test of ours ever reaches.
"""

from __future__ import annotations

import json
from pathlib import Path

from validate import REQUIRED_KEYS
from verdict_markers import (
    VERDICT_CLOSE,
    VERDICT_OPEN,
    VerdictExtractionError,
    VerdictSentinel,
)

AGENT_NAMES = ("melchior", "balthasar", "caspar")
AGENT_FILE_SUFFIX = ".md"
EXPECTED_MARKER_COUNT = 1


class PromptContractError(Exception):
    """Raised when a shipped agent prompt violates the runtime contract.

    This error intentionally inherits from ``Exception`` rather than
    ``ValidationError``: a stale or corrupted prompt cannot be fixed by retrying
    the model, so the run must abort immediately.
    """


class AgentPromptGuard:
    """Validates the three shipped agent prompts before they are used.

    Detects stale or corrupted installations that repository tests cannot
    reach.
    """

    def __init__(self, agents_dir: Path, sentinel: VerdictSentinel) -> None:
        """Initialize the guard.

        Args:
            agents_dir: Directory containing the agent prompt markdown files.
            sentinel: Marker sentinel used to locate and extract verdict blocks.
        """
        self._dir = Path(agents_dir)
        self._sentinel = sentinel

    def check(self) -> None:
        """Validates all three prompts. Raises PromptContractError on the first failure."""
        for name in AGENT_NAMES:
            self._check_one(self._dir / f"{name}{AGENT_FILE_SUFFIX}")

    def _check_one(self, path: Path) -> None:
        """Validate a single agent prompt file.

        Args:
            path: Path to the agent prompt markdown file.

        Raises:
            PromptContractError: If the file is unreadable, has malformed
                markers, or contains a fabricable verdict example.
        """
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise PromptContractError(
                f"{path}: cannot read prompt file. Reinstall the plugin."
            ) from exc
        except UnicodeDecodeError as exc:
            raise PromptContractError(f"{path}: prompt file is not valid UTF-8.") from exc

        lines = text.splitlines()
        opens = sum(self._sentinel.is_exact_marker_line(ln, VERDICT_OPEN) for ln in lines)
        closes = sum(self._sentinel.is_exact_marker_line(ln, VERDICT_CLOSE) for ln in lines)

        if opens != EXPECTED_MARKER_COUNT or closes != EXPECTED_MARKER_COUNT:
            raise PromptContractError(
                f"{path}: found {opens} open and {closes} close marker lines. "
                "Marker lines are ONLY for the verdict block; repeating them, "
                "even inside a code block, makes the model emit two examples. "
                "Reinstall the plugin (likely cause: v5.0.x prompts)."
            )

        try:
            between = self._sentinel.extract(text)
        except VerdictExtractionError as exc:
            raise PromptContractError(f"{path}: {exc}") from exc

        if self._is_fabricable_verdict(between):
            raise PromptContractError(
                f"{path}: content between markers is a valid verdict. "
                "The model can COPY it and the copy would be accepted as its verdict; "
                "a PLACEHOLDER goes between the markers, not an example; "
                "the worked example goes OUTSIDE the markers."
            )

    @staticmethod
    def _is_fabricable_verdict(text: str) -> bool:
        """Return True if the text is a JSON object with all required verdict keys.

        Args:
            text: Text extracted between the verdict markers.

        Returns:
            True if the text is a JSON object containing all required verdict
            keys, otherwise False.
        """
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return False
        return isinstance(obj, dict) and REQUIRED_KEYS.issubset(obj)
