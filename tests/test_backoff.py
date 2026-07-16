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
