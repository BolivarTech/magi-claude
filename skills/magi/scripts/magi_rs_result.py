# Author: Julian Bolivar
# Version: 1.3.0
# Date: 2026-09-29
"""Read the result of one ``magi-rs consult`` invocation.

The JSON is NESTED: the telemetry fields plus ``agents``, ``consensus`` and
``report`` live INSIDE the ``consult`` object, and a total failure is signalled
by ``stop_reason``/``error`` at the top level with ``consult`` set to null.
Measured in MS0 against a real captured envelope, not inferred.

Nothing here translates a field. ``magi-rs`` emits exactly this project's
seven-key contract, so the verdicts go to ``load_agent_output`` as they arrive;
an adapter would be a second place the contract lives.

:class:`MagiRsContractError` is defined here rather than in the trio module
because the trio imports this module: defining it on the other side would be a
circular import. The trio module re-exports it, so callers import it from
either.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_STRUCTURED_FLAG = "--structured-verdicts"

#: Key of the clock-typo flag inside the envelope's top-level ``applied_caps``
#: (captured from magi-rs v0.21.0, tests/fixtures/magi-rs-consult/three-agents.json).
_CEILING_FLAG = "ceiling_above_sanity"

_TELEMETRY_FIELDS = (
    "degraded",
    "mode",
    "mode_source",
    "extraction_failures",
    "input_size",
    "report_truncated",
    "endpoint_divergence",
    "timeout_below_formula",
    "failed_agents",
    "rotations",
    "ran_unmeasured",
    "completions",
    "pool_eligibility",
)
"""Always present by contract. An EMPTY value is meaningful -- an empty
``rotations`` is the positive certificate that nobody rotated, not silence
about whether anybody did -- so nothing here is dropped for being falsy."""


#: The one finish reason that means "the model chose to stop". Everything
#: else -- ``length``, a future variant -- is a cap or an interruption (D2).
_FINISH_CLEAN = "stop"


class MagiRsContractError(Exception):
    """``magi-rs`` ran, but its output does not satisfy the contract.

    An unknown ``--mode``, output that is not JSON, a null ``consult``, an
    absent ``agents`` key: every one of them fails IDENTICALLY the second time.
    So this is a SISTER of ``Exception``, never a ``ValidationError`` -- if the
    orchestrator's ``except (ValidationError, json.JSONDecodeError)`` could
    consume it, a forgotten ``--structured-verdicts`` would be retried three
    times and then reported as a transient failure.

    The rule is not "bootstrap errors are sisters". It is: **when the retry IS
    the fix, inherit; when the failure is deterministic, do not.**
    """


@dataclass(frozen=True)
class ConsultResult:
    """One completed ``consult`` invocation, read but not interpreted.

    Attributes:
        agents: The per-mage verdicts, exactly as emitted (seven-key contract).
        telemetry: The ``consult`` telemetry fields, carried verbatim.
        report: The rendered markdown report. Never scraped for findings.
        degraded: ``magi-rs``'s own degraded flag.
        run_id: Top-level ``run_id`` of the envelope (magi-rs >= 0.18.0): the
            pointer into ``.magi/logs/magi-<date>.log``, where the per-mage
            ceiling and the probe notices live since 0.18.1 (stderr no longer
            carries INFO). ``None`` when absent -- telemetry is fail-soft, never
            a contract error.
        ceiling_above_sanity: ``applied_caps.ceiling_above_sanity`` (top level
            of the envelope, magi-rs >= 0.21.0): True when the ``--timeout``
            handed to magi-rs derives a per-mage ceiling above its sanity
            bound. ``None`` unless the value is a JSON boolean -- an absent
            ``applied_caps``, a non-object one, a missing field and a
            non-boolean value (``"true"``, ``1``, ``null``) all read as unknown.
    """

    agents: list[dict[str, Any]]
    telemetry: dict[str, Any]
    report: str
    degraded: bool
    run_id: str | None
    ceiling_above_sanity: bool | None = None

    @property
    def capped_agents(self) -> list[str]:
        """Agents whose LAST attempt did not finish cleanly.

        Per D2: a verdict whose accepted attempt carries ``finish: length`` is
        counted as VALID -- the cut fell after the closing marker, so the
        verdict is whole and only trailing prose was lost -- but it is never
        counted silently. The damaging case (a cut that breaks extraction) is
        already handled upstream by magi-rs, which retries.

        The test is ``finish != "stop"``, not ``== "length"``: magi-core's
        finish reasons are non-exhaustive, and an unknown one is exactly where
        silence costs most. A ``null`` finish is different from an unknown one:
        it is an attempt that produced no output (a timeout, measured in MS0c)
        and is skipped, never counted as capped.

        Returns:
            Sorted agent names, empty when every agent finished cleanly or when
            no ``completions`` telemetry was emitted at all. Never raises --
            telemetry must not be able to destroy the artifact it annotates.
            ``completions`` is a mapping of mage -> attempts (measured shape);
            the LAST attempt of each mage is the one that decides.

        Complexity:
            O(m) in the number of seats in ``completions`` -- only each seat's
            LAST attempt is examined -- once per run.
        """
        completions = self.telemetry.get("completions")
        if not isinstance(completions, dict):
            return []
        capped: list[str] = []
        for mage, attempts in completions.items():
            if not isinstance(attempts, list) or not attempts:
                continue
            last = attempts[-1]
            finish = last.get("finish") if isinstance(last, dict) else None
            if finish is not None and finish != _FINISH_CLEAN:
                capped.append(str(mage))
        return sorted(capped)

    @classmethod
    def from_json(cls, raw: str) -> ConsultResult:
        """Parse the invocation's stdout.

        Args:
            raw: The process's stdout, expected to be one JSON object.

        Returns:
            The parsed result.

        Raises:
            MagiRsContractError: The output is not JSON; or ``consult`` is
                null, meaning fewer than two mages completed and there is
                nothing to report but the failure; or ``agents`` is absent,
                which means ``--structured-verdicts`` was omitted. The message
                says WHICH of the three happened -- and the markdown in
                ``report`` is never scraped for findings.
        """
        try:
            envelope: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MagiRsContractError(
                f"magi-rs output is not JSON ({exc}); the report markdown is never scraped"
            ) from exc
        if not isinstance(envelope, dict):
            raise MagiRsContractError("magi-rs output is JSON but not an object")
        consult = envelope.get("consult")
        if consult is None:
            raise MagiRsContractError(
                "magi-rs returned consult: null -- fewer than two mages completed, so there "
                f"is no verdict to report (stop_reason={envelope.get('stop_reason')!r}, "
                f"error={envelope.get('error')!r})"
            )
        if not isinstance(consult, dict):
            raise MagiRsContractError("magi-rs returned a consult object of the wrong shape")
        if "agents" not in consult:
            raise MagiRsContractError(
                f"magi-rs output carries no 'agents': {_STRUCTURED_FLAG} was not passed, so the "
                "verdicts arrived only as markdown"
            )
        agents = consult["agents"]
        if not isinstance(agents, list):
            raise MagiRsContractError("magi-rs 'agents' is not a list")
        run_id = envelope.get("run_id")
        applied_caps = envelope.get("applied_caps")
        ceiling = applied_caps.get(_CEILING_FLAG) if isinstance(applied_caps, dict) else None
        return cls(
            agents=list(agents),
            telemetry={field: consult.get(field) for field in _TELEMETRY_FIELDS},
            report=str(consult.get("report") or ""),  # a null report is empty, not "None"
            degraded=bool(consult.get("degraded", False)),
            run_id=str(run_id) if run_id is not None else None,
            ceiling_above_sanity=ceiling if isinstance(ceiling, bool) else None,
        )


_PRINTABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{0,63}")
_UNPRINTABLE = "unprintable"
_NOT_REPORTED = "not reported"
_MEASURED = "Measured"
_UNSUPPORTED = "Unsupported"
_NOT_MEASURED = "NotMeasured"
_UNKNOWN = "unknown"
_CHARS = " chars"
_MISSING: object = object()


def _printable_name(value: Any) -> str:
    """Return *value* if it is a printable identifier, else ``_UNPRINTABLE``."""
    if isinstance(value, str) and _PRINTABLE_NAME.fullmatch(value):
        return value
    return _UNPRINTABLE


def _int_chars_part(value: Any) -> str | None:
    """Return ``"{value} chars"`` if *value* is an int but not a bool, else ``None``."""
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value}{_CHARS}"
    return None


def describe_reasoning(attempt: Mapping[str, Any]) -> str:
    """Describe one attempt's reasoning state for a stderr warning.

    Never prints model text or any value of an unknown form: only a variant or
    type name, and only when it is a printable identifier. Never raises.

    Args:
        attempt: One ``completions[<seat>][k]`` record.

    Returns:
        ``Measured N chars`` · ``NotMeasured`` · ``Unsupported (<backend>, N chars)``
        (parts omitted when absent or malformed) · ``not reported`` (key absent) ·
        ``unknown (<variant or type>)`` · with ``unprintable`` for a non-identifier name.

    Complexity: O(k) in the number of keys of the reasoning object.
    """
    if not isinstance(attempt, Mapping):
        return _NOT_REPORTED

    reasoning = attempt.get("reasoning", _MISSING)
    if reasoning is _MISSING:
        return _NOT_REPORTED

    if isinstance(reasoning, str):
        return _NOT_MEASURED if reasoning == _NOT_MEASURED else f"{_UNKNOWN} (str)"

    if isinstance(reasoning, Mapping):
        keys = list(reasoning.keys())
        if len(keys) == 1:
            key = keys[0]
            if key == _MEASURED:
                inner = reasoning[_MEASURED]
                chars = inner.get("chars", _MISSING) if isinstance(inner, Mapping) else _MISSING
                chars_part = _int_chars_part(chars)
                return _MEASURED if chars_part is None else f"{_MEASURED} {chars_part}"
            if key == _UNSUPPORTED:
                inner = reasoning[_UNSUPPORTED]
                parts: list[str] = []
                if isinstance(inner, Mapping):
                    backend = inner.get("backend", _MISSING)
                    if isinstance(backend, str):
                        parts.append(_printable_name(backend))
                    chars_part = _int_chars_part(inner.get("chars", _MISSING))
                    if chars_part is not None:
                        parts.append(chars_part)
                if not parts:
                    return _UNSUPPORTED
                return f"{_UNSUPPORTED} ({', '.join(parts)})"

        if keys:
            return f"{_UNKNOWN} ({_printable_name(keys[0])})"
        return f"{_UNKNOWN} ({type(reasoning).__name__})"

    return f"{_UNKNOWN} ({type(reasoning).__name__})"
