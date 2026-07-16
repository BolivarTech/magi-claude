# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-16
"""Pure, I/O-free architecture-to-vendor lineage identity checks (MS4).

Groups everything about model identity -- comparing a probed architecture
family against a declared lineage, and comparing model digests for accidental
collisions -- in one place (SRP). Both operations are pure: no network calls,
no ``await``, no state beyond the class constant. Later tasks (preflight,
rotation) delegate to :class:`LineageIdentityGuard` instead of re-deriving
this logic inline.

Not to be confused with ``ollama_preflight.LINEAGE_PATTERNS`` (a tag-prefix
typo detector). :data:`LineageIdentityGuard.ARCHITECTURE_VENDOR` is a
different concept: architecture FAMILY (from ``/api/show``) -> vendor.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal, Mapping, Sequence

#: Verdict returned by :meth:`LineageIdentityGuard.family_verdict`.
FamilyVerdict = Literal["ok", "contradiction", "unknown"]


class LineageIdentityGuard:
    """Pure checks that compare probed model identity against declared config.

    Two independent, stateless checks live here: matching a probed
    architecture family to a declared lineage (:meth:`family_verdict`), and
    detecting accidental digest collisions across mages
    (:meth:`digest_collision`). Neither performs I/O; both are safe to call
    from an ``async`` context without an event-loop concern.

    Example:
        >>> guard = LineageIdentityGuard()
        >>> guard.family_verdict("deepseek4", "deepseek")
        'ok'
        >>> guard.family_verdict("deepseek4", "acme")
        'contradiction'
        >>> guard.family_verdict("llama", "acme")
        'unknown'
        >>> guard.digest_collision(["sha256:a", "sha256:b", "sha256:a"])
        (0, 2)
    """

    #: Architecture family (as reported by Ollama's ``/api/show``) -> vendor.
    #:
    #: NON-EXHAUSTIVE and a maintenance point: only architecture families that
    #: are unambiguously tied to a single vendor belong here. Ambiguous base
    #: architectures used across many vendors/finetunes (e.g. "llama",
    #: "mistral", generic "qwen") are deliberately EXCLUDED -- mapping them
    #: would produce false "contradiction" verdicts for legitimate models.
    #: Vendor values use the EXACT SAME vocabulary as the TOML ``lineage``
    #: strings (e.g. "alibaba", "deepseek") -- see
    #: ``test_map_vendors_match_declared_lineage_vocabulary``.
    ARCHITECTURE_VENDOR: Mapping[str, str] = MappingProxyType(
        {
            "qwen3.5": "alibaba",
            "kimi-k2": "moonshot",
            "deepseek4": "deepseek",
            "glm": "zhipu",
            "gpt-oss": "openai",
            "minimax": "minimax",
            "gemma": "google",
            "nemotron": "nvidia",
        }
    )

    def family_verdict(self, architecture: str | None, declared_lineage: str) -> FamilyVerdict:
        """Compare a probed architecture family against a declared lineage.

        Args:
            architecture: The architecture family reported by the backend
                (e.g. ``"deepseek4"``), or ``None`` if unavailable.
            declared_lineage: The lineage declared in the TOML config for the
                same mage (e.g. ``"deepseek"``).

        Returns:
            ``"ok"`` if the mapped vendor matches ``declared_lineage``;
            ``"contradiction"`` if it differs; ``"unknown"`` if
            ``architecture`` is ``None`` or not in :data:`ARCHITECTURE_VENDOR`.

        Example:
            >>> LineageIdentityGuard().family_verdict("deepseek4", "deepseek")
            'ok'
        """
        if architecture is None or architecture not in self.ARCHITECTURE_VENDOR:
            # Fail OPEN, deliberately: this map is a typo detector, never an
            # authority. The TOML lineage declaration always wins (decisions
            # #5/#7) -- an architecture we cannot classify must never block a
            # run. Do NOT "tighten" this branch into a contradiction; that
            # turns a best-effort hint into a false authority.
            return "unknown"
        vendor = self.ARCHITECTURE_VENDOR[architecture]
        return "ok" if vendor == declared_lineage else "contradiction"

    def digest_collision(self, digests: Sequence[str | None]) -> tuple[int, int] | None:
        """Find the first pair of indices sharing an equal, non-``None`` digest.

        Precondition: a ``None`` entry is treated as NON-comparable and never
        collides with anything, including another ``None`` -- a missing
        digest is handled fail-closed earlier, by R5b in a later task, not
        here.

        Args:
            digests: Per-mage digest strings (or ``None`` when unavailable),
                in mage order.

        Returns:
            The first colliding pair ``(i, j)`` with ``i < j``, or ``None`` if
            no two non-``None`` digests are equal.

        Example:
            >>> LineageIdentityGuard().digest_collision(["a", "b", "a"])
            (0, 2)
        """
        for i in range(len(digests)):
            first = digests[i]
            if first is None:
                continue
            for j in range(i + 1, len(digests)):
                if digests[j] == first:
                    return (i, j)
        return None
