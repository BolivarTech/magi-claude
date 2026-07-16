# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-16
"""Tests for the pure backoff helpers (MS3)."""

import pytest
from hypothesis import given, strategies as st

from backoff import (
    DEFAULT_RETRY_BACKOFF_MAX_SECONDS,
    next_backoff,
)


def test_next_backoff_first_attempt_returns_base():
    assert next_backoff(1, 2.0, 60.0, None) == 2.0


def test_next_backoff_grows_exponentially_across_attempts():
    assert [next_backoff(a, 2.0, 60.0, None) for a in (1, 2, 3)] == [2.0, 4.0, 8.0]


def test_next_backoff_is_bounded_by_the_ceiling():
    # 2 * 2**6 = 128, capped to 60
    assert next_backoff(7, 2.0, 60.0, None) == 60.0


def test_next_backoff_zero_base_stays_zero_preserving_r12_disable():
    assert next_backoff(5, 0.0, 60.0, None) == 0.0


def test_next_backoff_retry_after_wins_verbatim_over_formula():
    assert next_backoff(1, 2.0, 60.0, 10.0) == 10.0


def test_next_backoff_retry_after_greater_than_ceiling_is_respected():
    # server instruction is NOT re-capped by the formula ceiling
    assert next_backoff(1, 2.0, 60.0, 120.0) == 120.0


@given(
    attempt=st.integers(min_value=1, max_value=10),
    base=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    ceiling=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_next_backoff_never_exceeds_ceiling_without_retry_after(attempt, base, ceiling):
    assert next_backoff(attempt, base, ceiling, None) <= ceiling


from datetime import datetime, timedelta, timezone

from backoff import DEFAULT_RETRY_AFTER_MAX_SECONDS, parse_retry_after

_RECEIPT = datetime(2025, 10, 21, 7, 28, 0, tzinfo=timezone.utc)


def test_parse_retry_after_delta_seconds():
    assert parse_retry_after("120", _RECEIPT, 300.0) == 120.0


def test_parse_retry_after_absent_returns_none():
    assert parse_retry_after(None, _RECEIPT, 300.0) is None
    assert parse_retry_after("   ", _RECEIPT, 300.0) is None


def test_parse_retry_after_non_numeric_returns_none():
    assert parse_retry_after("soon", _RECEIPT, 300.0) is None


def test_parse_retry_after_zero_and_negative_fall_to_formula():
    assert parse_retry_after("0", _RECEIPT, 300.0) is None
    assert parse_retry_after("-5", _RECEIPT, 300.0) is None


def test_parse_retry_after_giant_value_is_capped():
    assert parse_retry_after("999999", _RECEIPT, DEFAULT_RETRY_AFTER_MAX_SECONDS) == 300.0


def test_parse_retry_after_http_date_future_returns_delta():
    future = _RECEIPT + timedelta(seconds=30)
    header = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(header, _RECEIPT, 300.0) == pytest.approx(30.0, abs=1.0)


def test_parse_retry_after_http_date_past_falls_to_formula():
    past = _RECEIPT - timedelta(seconds=30)
    header = past.strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(header, _RECEIPT, 300.0) is None


def test_parse_retry_after_aware_offset_normalized_to_utc():
    # 07:28:00 -0500 == 12:28:00 UTC == receipt + 5h; capped to 300
    header = "Tue, 21 Oct 2025 07:28:00 -0500"
    assert parse_retry_after(header, _RECEIPT, 300.0) == 300.0


def test_parse_retry_after_naive_receipt_does_not_raise():
    # gate CP2 loop 1 (Caspar): a naive receipt must NOT raise TypeError on the
    # HTTP-date subtraction -- it is treated as UTC.
    naive = datetime(2025, 10, 21, 7, 28, 0)  # no tzinfo
    header = "Wed, 21 Oct 2025 07:28:30 GMT"
    assert parse_retry_after(header, naive, 300.0) == pytest.approx(30.0, abs=1.0)


def test_parse_retry_after_cap_zero_falls_to_formula_not_zero_sleep():
    # gate CP2 loop 6 (Caspar): a degenerate cap=0 must return None (-> formula),
    # NOT 0.0 (which would sleep(0)). The post-cap re-check enforces this.
    assert parse_retry_after("120", _RECEIPT, 0.0) is None
