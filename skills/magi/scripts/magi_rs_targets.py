# Author: Julian Bolivar
# Version: 8.0.0
# Date: 2026-11-21
"""What ``code-review --ollama`` reviews: a path or a git revision (REQ-01..03).

``magi-rs code-review`` takes exactly one target: a file or directory, a
commit, a ``--from/--to`` range, a ``--base/--head`` branch or the uncommitted
changes. :class:`ReviewTarget` validates the operator's choice and turns it into
the argv fragment magi-rs expects, before any binary or model is touched.

A path is never read as a prompt: free text that is not an existing path is
refused. A path that resolves inside ``<workdir>/.magi`` or inside a rotation
backup is refused too (PM-C-1): magi-rs only excludes ``.magi/`` from a
directory review, not from an explicit path, and the key lives there.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from validate import ValidationError

DEFAULT_BASE = "main"
DEFAULT_HEAD = "HEAD"

KIND_PATH = "path"
KIND_COMMIT = "commit"
KIND_RANGE = "range"
KIND_BRANCH = "branch"
KIND_UNCOMMITTED = "uncommitted"

_MAGI_DIRNAME = ".magi"
_OPTION_PREFIX = "-"
_GIT_COUNT_ARGV = ["rev-list", "--count"]
_EMPTY_RANGE_COUNT = 0
TARGETS_HELP = "a path (file or directory), --commit <C>, --from <A> --to <B>, --base <B> --head <H>, or --uncommitted"


class InvalidTargetError(ValidationError):
    """The requested review target is not one magi-rs can review."""


@dataclass(frozen=True)
class ReviewTarget:
    """A validated review target.

    Attributes:
        kind: One of ``path``, ``commit``, ``range``, ``branch``, ``uncommitted``.
        argv: The fragment appended to ``magi-rs code-review``.
        label: The banner text (``"path <p>"``, ``"commit <c>"``, ...).
    """

    kind: str
    argv: tuple[str, ...]
    label: str

    @classmethod
    def resolve(
        cls,
        *,
        path: str | None,
        commit: str | None,
        from_: str | None,
        to: str | None,
        base: str | None,
        head: str | None,
        uncommitted: bool,
        workdir: str,
        forbidden_roots: Sequence[str],
        git: Callable[[list[str]], str],
    ) -> ReviewTarget:
        """Validate the operator's target flags and build the target.

        Exactly one target kind is allowed; none selects the default branch range
        (``--base main --head HEAD``), which must contain at least one commit.

        Args:
            path: Positional file or directory, or None.
            commit: ``--commit`` value, or None.
            from_: ``--from`` value, or None.
            to: ``--to`` value, or None.
            base: ``--base`` value, or None.
            head: ``--head`` value, or None.
            uncommitted: ``--uncommitted`` given.
            workdir: Project root; a relative path resolves against it, and
                ``<workdir>/.magi`` is always forbidden.
            forbidden_roots: Further directories no path may lie inside.
            git: Runs ``git <argv>`` and returns its stdout (asked only for the
                default range check).

        Returns:
            The validated target.

        Raises:
            InvalidTargetError: Mixed kinds, an empty value, an incomplete pair,
                free text, a forbidden path, an empty default range, or a git
                failure.
        """
        given = {
            KIND_PATH: [path],
            KIND_COMMIT: [commit],
            KIND_RANGE: [from_, to],
            KIND_BRANCH: [base, head],
        }
        kinds = [k for k, values in given.items() if any(v is not None for v in values)]
        if uncommitted:
            kinds.append(KIND_UNCOMMITTED)
        if any(v == "" for values in given.values() for v in values):
            raise InvalidTargetError(f"an empty target value is not allowed; use {TARGETS_HELP}")
        for flag, value in (
            ("--commit", commit),
            ("--from", from_),
            ("--to", to),
            ("--base", base),
            ("--head", head),
        ):
            if value is not None and value.startswith(_OPTION_PREFIX):
                raise InvalidTargetError(f"the value of {flag} must not start with '-'")
        if len(kinds) > 1:
            raise InvalidTargetError(
                f"choose one review target, not {' and '.join(kinds)}; use {TARGETS_HELP}"
            )
        if not kinds:
            return cls._default_branch(git)
        kind = kinds[0]
        if kind == KIND_PATH:
            return cls._path(str(path), workdir, forbidden_roots)
        if kind == KIND_COMMIT:
            return cls(KIND_COMMIT, ("--commit", str(commit)), f"commit {commit}")
        if kind == KIND_RANGE:
            if from_ is None or to is None:
                raise InvalidTargetError("--from and --to must be given together")
            return cls(KIND_RANGE, ("--from", from_, "--to", to), f"range {from_}..{to}")
        if kind == KIND_BRANCH:
            if base is None or head is None:
                raise InvalidTargetError("--base and --head must be given together")
            return cls(KIND_BRANCH, ("--base", base, "--head", head), f"branch {base}...{head}")
        return cls(KIND_UNCOMMITTED, ("--uncommitted",), KIND_UNCOMMITTED)

    @classmethod
    def _default_branch(cls, git: Callable[[list[str]], str]) -> ReviewTarget:
        """Everything the branch added since it left ``DEFAULT_BASE``; empty is an error."""
        try:
            count = int(git([*_GIT_COUNT_ARGV, f"{DEFAULT_BASE}..{DEFAULT_HEAD}"]).strip())
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise InvalidTargetError(
                f"cannot tell whether {DEFAULT_HEAD} has commits beyond {DEFAULT_BASE} (git: {exc}); "
                f"name a target: {TARGETS_HELP}"
            ) from exc
        if count == _EMPTY_RANGE_COUNT:
            raise InvalidTargetError(
                f"no commit of {DEFAULT_HEAD} is absent from {DEFAULT_BASE}, so there is nothing to review by "
                f"default; name a target: {TARGETS_HELP}"
            )
        return cls(
            KIND_BRANCH,
            ("--base", DEFAULT_BASE, "--head", DEFAULT_HEAD),
            f"branch {DEFAULT_BASE}...{DEFAULT_HEAD}",
        )

    @classmethod
    def _path(cls, path: str, workdir: str, forbidden_roots: Sequence[str]) -> ReviewTarget:
        """A file or directory that exists and lies outside every forbidden root."""
        candidate = path if os.path.isabs(path) else os.path.join(workdir, path)
        if not os.path.exists(candidate):
            raise InvalidTargetError(f"{path!r} is not an existing path; use {TARGETS_HELP}")
        real = _canonical(candidate)
        for root in (os.path.join(workdir, _MAGI_DIRNAME), *forbidden_roots):
            if _is_within(real, _canonical(root)):
                raise InvalidTargetError(f"{path!r} lies inside {root!r}, which is never reviewed")
        return cls(KIND_PATH, (candidate,), f"path {path}")


def _canonical(path: str) -> str:
    """Resolve links and ``..`` and fold case where the filesystem does."""
    return os.path.normcase(os.path.realpath(path))


def _is_within(path: str, root: str) -> bool:
    """True when ``path`` is ``root`` or lies below it (both canonical)."""
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives
        return False
