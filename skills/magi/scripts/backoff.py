# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-16
"""Exponential backoff for transient transport retries (MS3).

Pure, I/O-free helpers: :func:`next_backoff` computes the wait before the next
retry, and :func:`parse_retry_after` turns an HTTP ``Retry-After`` header into
capped seconds. Both are side-effect-free so the retry loop's timing is testable
without a network or an event loop.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

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


def parse_retry_after(header_value: str | None, receipt: datetime, cap: float) -> float | None:
    """Parse an HTTP ``Retry-After`` into capped seconds, or ``None`` if unusable.

    Accepts delta-seconds (``"120"``) and HTTP-date
    (``"Wed, 21 Oct 2025 07:28:00 GMT"``). All datetimes are normalized to UTC
    before comparison (a naive result assumes UTC; an aware result is converted),
    and an HTTP-date delay is measured against *receipt* -- the response reception
    time -- not the caller's later sleep decision. Anything unusable (absent,
    unparseable, ``<= 0``, or in the past) returns ``None`` so the caller falls
    back to the formula. Never raises on malformed input.

    Args:
        header_value: The raw ``Retry-After`` header, or ``None`` if absent.
        receipt: UTC-aware timestamp captured when the response was received.
        cap: Upper bound (``retry_after_max_seconds``) so a hostile
            ``Retry-After: 999999`` cannot hang the run.

    Returns:
        Capped seconds to wait (``> 0`` and ``<= cap``), or ``None`` if unusable.
    """
    # Defensive: a naive receipt would make the HTTP-date subtraction raise
    # TypeError (naive vs aware) and break the "never raises" promise. T4 passes
    # an aware receipt, but the function must not trust its caller (gate CP2 loop 1,
    # Caspar). Normalize once, here.
    if receipt.tzinfo is None:
        receipt = receipt.replace(tzinfo=timezone.utc)
    if header_value is None:
        return None
    raw = header_value.strip()
    if not raw:
        return None
    seconds = _delta_or_http_date_seconds(raw, receipt)
    if seconds is None or seconds <= 0:
        return None
    capped = min(seconds, cap)
    # Re-check AFTER capping: a degenerate cap=0 (retry_after_max_seconds=0, allowed)
    # would otherwise return 0.0 -> sleep(0), when the spec (§6) says it must fall to
    # the formula. cap=0 -> capped=0 -> None -> formula (gate CP2 plan loop 6, Caspar).
    return capped if capped > 0 else None


def _delta_or_http_date_seconds(raw: str, receipt: datetime) -> float | None:
    """Seconds from a delta-seconds or HTTP-date value, or ``None`` if neither.

    Captures ONLY ``(ValueError, TypeError)`` -- the exact exceptions ``int()`` and
    ``parsedate_to_datetime`` raise on malformed input -- and treats a ``None``
    return from ``parsedate_to_datetime`` (some versions return it instead of
    raising) as unusable, so a bad header never propagates to the subtraction.
    """
    try:
        return float(int(raw))
    except (ValueError, TypeError):
        pass
    try:
        parsed = parsedate_to_datetime(raw)
    except (ValueError, TypeError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return (parsed - receipt).total_seconds()
