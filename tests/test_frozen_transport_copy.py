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


def _digest_of_source(source: str) -> str:
    """Hash the NORMALISED AST of a function's source.

    Args:
        source: The function's source text.

    Returns:
        The sha256 of `ast.dump` with fields and attributes stripped, so a
        reformat or a comment edit does not trip the guard -- only a change in
        logic does.
    """
    import ast
    import hashlib

    tree = ast.parse(source.lstrip())
    normalised = ast.dump(tree, annotate_fields=False, include_attributes=False)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _closure_digest(module_source: str, name: str) -> str:
    """Hash a nested function that `inspect.getsource` cannot reach by name.

    `_legacy_tracked_launch` is a closure inside `run_orchestrator`, so it has
    no module attribute to introspect. Walking the module's AST finds it.

    Args:
        module_source: The whole module's source.
        name: The nested function's name.

    Returns:
        Its normalised-AST digest.

    Raises:
        AssertionError: The function is not in the module any more -- which is
            itself drift worth failing on.
    """
    import ast
    import hashlib

    for node in ast.walk(ast.parse(module_source)):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            normalised = ast.dump(node, annotate_fields=False, include_attributes=False)
            return hashlib.sha256(normalised.encode("utf-8")).hexdigest()
    raise AssertionError(f"{name} is gone from the module")


def test_the_frozen_retry_loop_and_its_guards_have_not_drifted():
    """The OTHER two frozen copies -- the transport is not the only one.

    `test_the_frozen_transport_copy_has_not_drifted` protects
    `ClaudeBackend.run` against silent divergence from `ClaudeTrio._run_one`.
    But MS1 left THREE things frozen, not one, and the review of the milestone's
    own diff pointed out that two of them had no net:

    * `_legacy_tracked_launch` -- the retry loop, whose live twin is
      `ClaudeTrio._run_with_retry`. This is where the milestone's one real
      behaviour change lives (the prompt artifact's fail-soft write), which is
      precisely why an unnoticed edit here would be expensive.
    * `launch_agent` -- which still carries the echo-canary and identity guards,
      now duplicated as module functions in `trio_executor`.

    Both are dead on the Claude path and live on the rotation path, so a fix
    applied to one copy and not the other would pass every other test.

    Same normalisation and the same version-keyed skip as the transport guard: a
    guard that cries wolf on a toolchain bump is a guard someone deletes.
    """
    import inspect
    import io
    import os
    import sys

    import run_magi

    module_source = io.open(
        os.path.join(os.path.dirname(inspect.getfile(run_magi)), "run_magi.py"),
        encoding="utf-8",
    ).read()

    # MEDIDO el 2026-08-29 en CPython 3.14.
    expected = {
        (3, 14): {
            "_legacy_tracked_launch": (
                "56c100b3e3c0b13e9a66f8f3ff8cf47b1400428e1a68e4a0acb22fae2856915d"
            ),
            "launch_agent": ("af0e2e3c48d825a968c0ffa9efd9bf47c071a6b6d41b2e1eca6d75ee186a4b78"),
        }
    }
    key = sys.version_info[:2]
    if key not in expected:
        pytest.skip(f"no frozen-copy digests recorded for Python {key[0]}.{key[1]}")

    actual = {
        "_legacy_tracked_launch": _closure_digest(module_source, "_legacy_tracked_launch"),
        "launch_agent": _digest_of_source(inspect.getsource(run_magi.launch_agent)),
    }

    for name, digest in actual.items():
        assert digest == expected[key][name], (
            f"{name} changed. It is frozen dead code on the Claude path until MS5. "
            f"If the edit is deliberate, mirror it in its ClaudeTrio twin and update "
            f"this digest; if it is not, revert it."
        )
