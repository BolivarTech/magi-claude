#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-30
"""Read the model that ACTUALLY served each agent out of its raw envelope.

Why this exists: every Claude model ID in :mod:`models` is a floating alias
(``claude-opus-5``), and an alias is not a fixed point — the same string can be
served by a revised model later. A report that records only what was
*requested* therefore stops attributing anything the moment the alias moves,
and these reports are kept as audit artifacts and read back long after the run.

The ``claude -p --output-format json`` envelope carries the served identity in
``modelUsage``: a mapping keyed by model ID whose entries carry
``canonicalModel``. This module reads exactly that and nothing else.

**It never guesses.** When an envelope reports more than one model (a
server-side fallback mid-turn bills both), every one is recorded, sorted.
Collapsing to "the real one" would mean choosing among candidates on a
heuristic — the failure shape this project already paid for in the verdict
parser, where every attempt to pick the right candidate turned out to be a
fail-open (see the sentinel history in ``CLAUDE.md``).

**Fail-soft.** Attribution is telemetry and telemetry never kills a verdict: an
unreadable, missing, or Ollama-shaped envelope simply omits that agent. An
omitted agent says *not known*; a fabricated one would say something false
about which model judged the input.

Deliberately separate from :mod:`cost`, which reads the same envelopes for a
different reason. Money and identity have different reasons to change; the
duplicated ``json.load`` is cheaper than the coupling.

Example::

    >>> read_served_models(run_dir, ["melchior", "balthasar", "caspar"])
    {'melchior': ['claude-opus-5'], 'balthasar': ['claude-opus-5'],
     'caspar': ['claude-opus-5']}
"""

from __future__ import annotations

import json
import os
from typing import Iterable

#: Envelope field holding per-model usage, keyed by the model ID the CLI billed.
_MODEL_USAGE_FIELD = "modelUsage"

#: Per-entry field naming the resolved model. Preferred over the mapping key:
#: the key is how the request was labelled, this is what answered it.
_CANONICAL_FIELD = "canonicalModel"

#: Suffix of the per-agent raw envelope written by the orchestrator.
_RAW_SUFFIX = ".raw.json"


def read_served_models(output_dir: str, agents: Iterable[str]) -> dict[str, list[str]]:
    """Map each agent to the model IDs its raw envelope reports as served.

    Args:
        output_dir: Run directory holding ``{agent}.raw.json`` envelopes.
        agents: Agent names to look for. Order does not matter.

    Returns:
        Agent name → sorted list of served model IDs, for every agent whose
        envelope could be read and reported at least one model. Agents whose
        envelope is missing, unreadable, malformed, or carries no
        ``modelUsage`` are **omitted** rather than mapped to a placeholder.

    Raises:
        Nothing. Every failure mode degrades to omitting that agent.
    """
    served: dict[str, list[str]] = {}
    for agent in agents:
        models = _read_one(os.path.join(output_dir, f"{agent}{_RAW_SUFFIX}"))
        if models:
            served[agent] = models
    return served


def _read_one(path: str) -> list[str]:
    """Return the sorted served model IDs in one envelope, or ``[]``.

    Args:
        path: Path to a single ``{agent}.raw.json`` envelope.

    Returns:
        Sorted, de-duplicated model IDs; empty when the envelope cannot be
        read or reports none.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []

    usage = data.get(_MODEL_USAGE_FIELD) if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return []

    identities: set[str] = set()
    for key, entry in usage.items():
        canonical = entry.get(_CANONICAL_FIELD) if isinstance(entry, dict) else None
        identity = canonical if isinstance(canonical, str) and canonical else key
        if isinstance(identity, str) and identity:
            identities.add(identity)
    return sorted(identities)
