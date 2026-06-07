"""ClaudeBackend.run shells out to `claude -p` exactly as launch_agent did."""

import asyncio
import pytest
from claude_backend import ClaudeBackend


class _FakeProc:
    def __init__(self, stdout=b'{"result": "{}"}', stderr=b"", returncode=0):
        self._stdout, self._stderr, self.returncode = stdout, stderr, returncode

    async def communicate(self, input=None):
        return self._stdout, self._stderr


def test_run_invokes_claude_with_resolved_model(monkeypatch, tmp_path):
    captured = {}

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        captured["stdin"] = kwargs.get("stdin")
        return _FakeProc()

    monkeypatch.setattr("claude_backend.asyncio.create_subprocess_exec", fake_exec)
    sys_prompt = tmp_path / "melchior.md"
    sys_prompt.write_text("SYS", encoding="utf-8")

    raw = asyncio.run(
        ClaudeBackend().run("melchior", str(sys_prompt), "PROMPT", "opus", 900, str(tmp_path))
    )

    assert b'"result"' in raw
    assert captured["args"][0] == "claude"
    assert "claude-opus-4-7" in captured["args"]  # resolve_model("opus")
    assert str(sys_prompt) in captured["args"]


def test_run_nonzero_exit_raises_runtimeerror(monkeypatch, tmp_path):
    async def fake_exec(*args, **kwargs):
        return _FakeProc(stdout=b"", stderr=b"boom", returncode=2)

    monkeypatch.setattr("claude_backend.asyncio.create_subprocess_exec", fake_exec)
    sp = tmp_path / "caspar.md"
    sp.write_text("S", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exited with code 2"):
        asyncio.run(ClaudeBackend().run("caspar", str(sp), "P", "opus", 900, str(tmp_path)))
