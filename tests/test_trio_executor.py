"""TrioExecutor is an abstract contract at the TRIO level, not per mage."""

import inspect

import pytest

from trio_executor import TrioExecutor


def test_trio_executor_cannot_be_instantiated():
    with pytest.raises(TypeError):
        TrioExecutor()  # type: ignore[abstract]


def test_run_trio_is_abstract_coroutine():
    assert getattr(TrioExecutor.run_trio, "__isabstractmethod__", False) is True
    assert inspect.iscoroutinefunction(TrioExecutor.run_trio)


def test_subclass_without_run_trio_is_abstract():
    class Incomplete(TrioExecutor):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]
