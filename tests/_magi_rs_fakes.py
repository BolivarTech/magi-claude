# Author: Julian Bolivar
# Version: 7.1.0
# Date: 2026-11-21
"""Shared fakes for the ``magi-rs code-review`` tests.

Nothing here launches a binary, reaches the network or touches the plugin's
real ``bin/``.
"""

import asyncio
import json
import os
from types import SimpleNamespace

_CR_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "magi-rs-code-review", "one-commit.json"
)


def _cr_payload(**review_overrides) -> bytes:
    """The captured code-review envelope (M6), optionally with `code_review` fields overridden."""
    with open(_CR_FIXTURE, encoding="utf-8") as handle:
        envelope = json.load(handle)
    envelope["code_review"].update(review_overrides)
    return json.dumps(envelope).encode("utf-8")


def _stub_pin():
    """A Pin no test inspects (the bootstrap is stubbed)."""
    return SimpleNamespace(version="0.25.0", assets={}, repository="BolivarTech/magi")


def _stub_bootstrap(monkeypatch, module):
    """Replace `<module>.BinaryBootstrap` so nothing downloads or verifies a binary."""
    monkeypatch.setattr(
        f"{module}.BinaryBootstrap",
        lambda *a, **k: SimpleNamespace(
            ensure=lambda: "/bin/magi-rs", verify=lambda: "/bin/magi-rs"
        ),
    )


class _StreamingProcess:
    """A magi-rs child whose output is already written and whose exit can be delayed or never come.

    ``kill()`` sets the exit event, so a ``wait()`` that was blocked returns -- a fake whose
    wait() hangs after kill() would hang the executor under test, not prove anything.
    The readers are built on first use, inside the running loop: Python 3.14 refuses an
    ``asyncio.StreamReader()`` with no loop, and ``main()``-level tests build this fake
    synchronously.
    """

    def __init__(
        self, stdout: bytes, stderr: bytes = b"", returncode=0, exits_after=0.0, hold_eof=False
    ):
        # hold_eof: stdout reaches EOF only when the child exits or is killed (a wedged child).
        self._out, self._err, self._readers, self._hold_eof = stdout, stderr, None, hold_eof
        self._rc, self._exits_after = returncode, exits_after
        self._exited = asyncio.Event()
        self.killed, self.returncode = False, None
        self.pid = -1  # never a real process: a tree kill on it is a harmless no-op

    def _streams(self):
        if self._readers is None:
            out, err = asyncio.StreamReader(), asyncio.StreamReader()
            out.feed_data(self._out)
            if not self._hold_eof:
                out.feed_eof()
            err.feed_data(self._err)
            err.feed_eof()
            self._readers = (out, err)
        return self._readers

    @property
    def stdout(self):
        return self._streams()[0]

    @property
    def stderr(self):
        return self._streams()[1]

    async def wait(self):
        if self._exits_after is not None and not self._exited.is_set():
            await asyncio.sleep(self._exits_after)
            self._exited.set()
        await self._exited.wait()
        self._release()
        if self.returncode is None:
            self.returncode = self._rc
        return self.returncode

    def _release(self):
        if self._hold_eof and self._readers is not None:
            self._readers[0].feed_eof()

    def kill(self):
        self.killed, self.returncode = True, -9
        self._release()
        self._exited.set()
