# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-24
"""Input-size estimation and oversize detection for MAGI.

MAGI reviews each input whole (no map-reduce — opus context holds realistic
diffs). This module estimates the input's token footprint with a stdlib-only
heuristic (chars / 4, the common English approximation) and flags inputs large
enough to warrant an operator warning. Pure/total: never raises.
"""

from __future__ import annotations

#: Default warn threshold (estimated tokens). Well inside opus context; a value
#: this large signals an unusually big review input worth splitting.
WARN_INPUT_TOKENS: int = 150_000

#: Divisor for the chars->tokens heuristic (English avg ~4 chars/token).
_CHARS_PER_TOKEN: int = 4


def estimate_tokens(text: str) -> int:
    """Estimate the token count of *text* with the stdlib chars/4 heuristic.

    Args:
        text: Input text to estimate.

    Returns:
        Estimated token count (non-negative integer).
    """
    return len(text) // _CHARS_PER_TOKEN


def check_input_size(text: str, threshold: int) -> tuple[int, bool]:
    """Return ``(estimated_tokens, exceeds)`` where *exceeds* is True iff the
    estimate is strictly greater than *threshold*.

    Args:
        text: Input text to check.
        threshold: Token count above which the input is considered oversized.

    Returns:
        A tuple of ``(estimated_tokens, exceeds)`` where *exceeds* is True
        when the estimate strictly exceeds *threshold*.
    """
    est = estimate_tokens(text)
    return est, est > threshold
