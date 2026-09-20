#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.3.1
# Date: 2026-09-20
"""MAGI model registry.

Single source of truth for the Claude model short names accepted on the
MAGI command line and the Anthropic model IDs they resolve to. Bumping a
model (e.g. from ``claude-opus-4-7`` to ``claude-opus-5``) is a one-line
edit here; the orchestrator in :mod:`run_magi` imports the mapping
without needing to change.

Every ID is an **alias without a date suffix** — the supported, stable
form. ``haiku`` was pinned to the dated ``claude-haiku-4-5-20251001``
until 5.4.0; a dated snapshot buys nothing here and silently ages out.

The module also owns :class:`ModelSpec`, the (tag, lineage) pair every
seat is launched with. It lived in the open-weight config module until
6.1.0; the Claude path is its only remaining consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

_MODEL_IDS_MUTABLE: dict[str, str] = {
    "opus": "claude-opus-5",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5",
}

#: Read-only view of the short-name → Anthropic model-ID mapping.
#: Exposed as a :class:`~types.MappingProxyType` so downstream code cannot
#: accidentally mutate the canonical registry at runtime.
MODEL_IDS: Mapping[str, str] = MappingProxyType(_MODEL_IDS_MUTABLE)

#: Tuple of accepted short names, kept in lockstep with :data:`MODEL_IDS`.
VALID_MODELS: tuple[str, ...] = tuple(MODEL_IDS.keys())

#: Per-mode default short name. When ``--model`` is not given on the
#: CLI, :func:`run_magi.parse_args` looks the analysis mode up here
#: to pick the default. Explicit ``--model X`` always wins.
#:
#: **History:**
#:
#: * 2.2.2 and earlier: uniform ``opus`` default for every mode.
#: * 2.2.3 (2026-04-25): switched ``analysis`` to ``sonnet`` for cost
#:   relief, on the assumption that sonnet would match opus quality on
#:   exploratory questions at ~4× lower cost.
#: * 2.2.5 (2026-04-26): reverted ``analysis`` to ``opus``. Production
#:   evidence: Caspar (the most-output agent by design, consistently
#:   producing 4-7K output tokens) failed in ≥33% of sbtdd Loop
#:   verifications under sonnet — an order of magnitude above the
#:   3.3% design assumption. The 2.2.4 retry could not recover Caspar
#:   because the failure was structural (sonnet's ~8K output ceiling
#:   pressure on Caspar's adversarial-by-design verbosity), not
#:   stochastic. Reverting restores opus's 32K output budget for the
#:   mode where Caspar runs into the ceiling. The tripwire policy in
#:   memory/routine_telemetry_post_2.2.1.md fired by sustained
#:   evidence rather than by the literal "n=2 iter-2-style" letter.
#:
#: Every key MUST be in :data:`run_magi.VALID_MODES` and every value
#: MUST be in :data:`MODEL_IDS`; the test suite enforces both invariants.
_MODE_DEFAULTS_MUTABLE: dict[str, str] = {
    "code-review": "opus",
    "design": "opus",
    "analysis": "opus",
}
MODE_DEFAULT_MODELS: Mapping[str, str] = MappingProxyType(_MODE_DEFAULTS_MUTABLE)


@dataclass(frozen=True)
class ModelSpec:
    """A model tag plus the lineage (originating lab) it belongs to.

    The lineage is DECLARED by the caller, never inferred from the tag. On the
    Claude path, the only one that builds these since 6.1.0, every seat carries
    ``"anthropic"`` and nothing consults the field: the "one lineage, one mage"
    invariant it once fed is enforced on the magi-rs side now. It is carried,
    not read.

    Attributes:
        model: Model identifier, e.g. a short name such as ``"opus"``.
        lineage: Originating lab, e.g. ``"anthropic"``. Non-empty.

    Example:
        >>> ModelSpec("opus", "anthropic").model
        'opus'
    """

    model: str
    lineage: str


def resolve_model(short_name: str) -> str:
    """Return the Anthropic model ID for *short_name*.

    Args:
        short_name: A MAGI model short name (e.g. ``"opus"``).

    Returns:
        The corresponding Anthropic model identifier.

    Raises:
        ValueError: If *short_name* is not a registered model.
    """
    try:
        return MODEL_IDS[short_name]
    except KeyError:
        raise ValueError(
            f"Unknown model '{short_name}'. Must be one of {sorted(MODEL_IDS.keys())}."
        ) from None
