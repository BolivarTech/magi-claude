"""Anti-drift guard over the frozen v5 transport copy.

`ClaudeBackend` is dead code from MS1 until MS5: nothing in production calls it,
so nothing else would notice an edit -- and an unnoticed edit is exactly how the
two copies of the `claude -p` transport diverge. This is the only witness.
"""

import pytest


def test_the_frozen_transport_copy_has_not_drifted():
    """ClaudeBackend is dead code until MS5; a change to it must be deliberate.

    Two live copies of the `claude -p` transport exist between MS1 and MS5. The
    old one is unreachable in production, so nothing else would notice an edit --
    and an unnoticed edit is exactly how the two diverge. The AST is hashed rather
    than the source text: reformatting or a comment fix must not trip this, only a
    change in logic.
    """
    import ast
    import hashlib
    import inspect

    from claude_backend import ClaudeBackend

    import sys

    tree = ast.parse(inspect.getsource(ClaudeBackend.run).lstrip())
    normalised = ast.dump(tree, annotate_fields=False, include_attributes=False)
    digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    # The digest is keyed by interpreter minor version: `ast.dump` output can
    # change across Python releases, and a red guard after an upgrade would be
    # noise, not drift. An unknown version SKIPS rather than fails -- a guard that
    # cries wolf on a toolchain bump is a guard someone deletes.
    # MEDIDO el 2026-08-29 sobre ClaudeBackend.run en CPython 3.14.
    expected = {(3, 14): "ca7c7fbff2767f35a83eebedd14f0790008574253e41b875b1f102bcb0172ecc"}
    key = sys.version_info[:2]
    if key not in expected:
        pytest.skip(f"no frozen-copy digest recorded for Python {key[0]}.{key[1]}")

    assert digest == expected[key], (
        "ClaudeBackend.run changed. It is frozen dead code until MS5. If the edit "
        "is deliberate, mirror it in ClaudeTrio._run_one and update this digest; "
        "if it is not, revert it."
    )
