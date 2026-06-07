"""AgentBackend is an abstract contract; concretes must implement run()."""

import inspect
import pytest
from backend import AgentBackend


def test_agent_backend_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AgentBackend()  # type: ignore[abstract]


def test_run_is_abstract_coroutine():
    assert getattr(AgentBackend.run, "__isabstractmethod__", False) is True
    assert inspect.iscoroutinefunction(AgentBackend.run)


def test_subclass_without_run_is_abstract():
    class Incomplete(AgentBackend):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]
