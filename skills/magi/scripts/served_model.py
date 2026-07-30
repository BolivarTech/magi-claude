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

**It never guesses — and that includes not guessing from the key.** Only
``canonicalModel`` is read. When it is absent the agent is omitted, because the
mapping key is the label the request was *billed under*, not a confirmation of
what answered; recording it here would put an unconfirmed value in a field whose
whole purpose is confirmation. When an envelope reports more than one model (a
server-side fallback mid-turn bills both), every one is recorded, sorted:
collapsing to "the real one" would mean choosing among candidates on a heuristic
— the failure shape this project already paid for in the verdict parser, where
every attempt to pick the right candidate turned out to be a fail-open (see the
sentinel history in ``CLAUDE.md``). A reader seeing two IDs should conclude
exactly that: two models served this agent's calls, and the envelope does not
say which produced which token.

**Fail-soft, but not silent.** Attribution is telemetry and telemetry never
kills a verdict: an unreadable, missing, or Ollama-shaped envelope simply omits
that agent. An omitted agent says *not known*; a fabricated one would say
something false about which model judged the input. Silence about a *total*
failure would be its own lie, so the caller lists the unattributed agents and
warns when a Claude-path run attributes none of them — a field that quietly
stopped working must not look like a field with nothing to say.

**Scope of the claim.** These are the models that served an agent's calls in
this run directory, which is not identical to "produced the verdict": an agent
whose output failed validation still wrote an envelope, and a reused
``--output-dir`` can still hold a previous run's envelope for an agent that died
before overwriting it. Cost aggregation reads the same files with the same
exposure; on the default per-run temp namespace neither applies.

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
        envelope could be read and reported at least one **confirmed**
        (``canonicalModel``) identity. Agents whose envelope is missing,
        unreadable, malformed, carries no ``modelUsage``, or reports usage
        without ``canonicalModel`` are **omitted** rather than mapped to a
        placeholder or to the mapping key.

    Raises:
        Nothing — enforced, not merely intended: the per-agent read catches
        ``Exception``. This call sits immediately before the report is
        serialized, so anything escaping here would cost the audit artifact and
        leave the run directory live-locked. A telemetry read must not be able
        to destroy the verdict it annotates.
    """
    served: dict[str, list[str]] = {}
    for agent in agents:
        models = _read_one(os.path.join(output_dir, f"{agent}{_RAW_SUFFIX}"))
        if models:
            served[agent] = models
    return served


def _read_one(path: str) -> list[str]:
    """Return the sorted CONFIRMED served model IDs in one envelope, or ``[]``.

    Args:
        path: Path to a single ``{agent}.raw.json`` envelope.

    Returns:
        Sorted, de-duplicated ``canonicalModel`` values; empty when the envelope
        cannot be read, reports no usage, or reports usage without a confirmed
        identity. The mapping key is deliberately **not** used as a fallback —
        see the module docstring.

    Raises:
        Nothing. The broad ``except`` is deliberate and is the enforcement of
        the no-raise contract this module's caller depends on: the enumerated
        errors are the ones foreseen, and an unforeseen one here would destroy
        the report this data only annotates. Failure is total for this agent
        and never partial, so nothing half-read reaches the artifact.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)

        usage = data.get(_MODEL_USAGE_FIELD) if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            return []

        identities: set[str] = set()
        for entry in usage.values():
            canonical = entry.get(_CANONICAL_FIELD) if isinstance(entry, dict) else None
            if isinstance(canonical, str) and canonical:
                identities.add(canonical)
        return sorted(identities)
    except Exception:  # noqa: BLE001 -- see Raises above: telemetry must not kill the verdict
        return []
