# Author: Julian Bolivar
# Version: 1.2.0
# Date: 2026-09-19
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
from dataclasses import dataclass
from typing import Any

_STRUCTURED_FLAG = "--structured-verdicts"

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
    """

    agents: list[dict[str, Any]]
    telemetry: dict[str, Any]
    report: str
    degraded: bool
    run_id: str | None

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
            O(n) in the number of completion records, once per run.
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
        return cls(
            agents=list(agents),
            telemetry={field: consult.get(field) for field in _TELEMETRY_FIELDS},
            report=str(consult.get("report") or ""),  # a null report is empty, not "None"
            degraded=bool(consult.get("degraded", False)),
            run_id=str(run_id) if run_id is not None else None,
        )
