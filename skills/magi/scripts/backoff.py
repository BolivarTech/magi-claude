# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-16
"""Exponential backoff for transient transport retries (MS3).

Pure, I/O-free helpers: :func:`next_backoff` computes the wait before the next
retry, and :func:`parse_retry_after` (Task 2) turns an HTTP ``Retry-After``
header into capped seconds. Both are side-effect-free so the retry loop's timing
is testable without a network or an event loop.
"""

from __future__ import annotations

#: Ceiling for the exponential FORMULA (not for an explicit server Retry-After).
DEFAULT_RETRY_BACKOFF_MAX_SECONDS = 60.0
#: Ceiling for a server-sent Retry-After (defense against a hostile/buggy server).
DEFAULT_RETRY_AFTER_MAX_SECONDS = 300.0
#: Growth factor of the exponential (2 -> 2, 4, 8, ...).
_BACKOFF_FACTOR = 2
# NOTE (gate CP2 loop 1, Caspar — cohesion): DEFAULT_TIMEOUT_SECONDS lives in
# ollama_config.py, NOT here. A request timeout is a config/orchestration concern
# with no role in the backoff math; keeping it out of backoff.py avoids config and
# run_magi depending on this module for an unrelated constant.


def next_backoff(attempt: int, base: float, ceiling: float, retry_after: float | None) -> float:
    """Return the seconds to wait before the next retry.

    When *retry_after* is given (already parsed and capped by
    :func:`parse_retry_after`), it wins verbatim: an explicit server instruction
    overrides the formula and is NOT re-capped by *ceiling* (the ceiling bounds
    only our own computed backoff). Otherwise the wait grows exponentially,
    ``base * 2**(attempt - 1)``, bounded above by *ceiling*.

    Args:
        attempt: 1-based attempt number within THIS model's budget (resets to 1
            on rotation). ``attempt == 1`` yields ``base``.
        base: Base backoff in seconds (MS1 ``retry_backoff_seconds``). ``0``
            disables the wait (R12), and the formula preserves that: 0 in, 0 out.
        ceiling: Upper bound for the FORMULA only (``retry_backoff_max_seconds``).
        retry_after: Already-parsed, already-capped server wait in seconds, or
            ``None`` when the server sent no usable ``Retry-After``.

    Returns:
        Seconds to sleep (``>= 0``).

    Example:
        >>> next_backoff(1, 2.0, 60.0, None)
        2.0
        >>> next_backoff(3, 2.0, 60.0, None)
        8.0
        >>> next_backoff(1, 2.0, 60.0, 10.0)  # server Retry-After wins
        10.0
    """
    if retry_after is not None:
        return retry_after
    # float(...) around the power: typeshed types a non-literal int exponent of
    # `int ** int` as returning Any (it can be int OR float at runtime for a
    # negative exponent), which mypy --strict rejects as an implicit Any return.
    # attempt >= 1 makes the exponent >= 0, so this is always an int value;
    # float(...) just makes that explicit for the type checker.
    return min(base * float(_BACKOFF_FACTOR ** (attempt - 1)), ceiling)
