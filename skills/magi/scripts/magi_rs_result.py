# Author: Julian Bolivar
# Version: 8.0.0
# Date: 2026-11-21
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
        degraded: ``magi-rs``'s own degraded flag. False when the field is
            absent; True for any value that is not a JSON boolean, because an
            off-contract flag cannot vouch for a healthy run.
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
        flag = consult.get("degraded", False)
        return cls(
            agents=list(agents),
            telemetry={field: consult.get(field) for field in _TELEMETRY_FIELDS},
            report=str(consult.get("report") or ""),  # a null report is empty, not "None"
            degraded=flag if isinstance(flag, bool) else True,
            run_id=str(run_id) if run_id is not None else None,
            ceiling_above_sanity=ceiling if isinstance(ceiling, bool) else None,
        )


class MemoryContractError(MagiRsContractError):
    """The envelope reports ``without_memory`` although memory was required.

    This plugin never passes ``--no-memory``, so the flag means a bug here or a
    behaviour change in magi-rs. Deterministic, hence a sister of
    ``ValidationError`` through :class:`MagiRsContractError`.
    """


#: ``code_review`` flags that must be JSON booleans when present.
_REVIEW_FLAGS = ("without_memory", "without_embedder")
#: ``code_review`` keys copied into :meth:`CodeReviewResult.effort`.
_EFFORT_FROM_REVIEW = ("batches_planned", "batches_completed", "gate_questions", "judged_fraction")
#: Per-batch keys copied into ``tokens_by_batch`` (``index`` is the batch number, which a failed batch leaves a gap in).
_BATCH_FIELDS = ("index", "input_tokens", "output_tokens", "ms")


@dataclass(frozen=True)
class CodeReviewResult:
    """One completed ``magi-rs code-review`` invocation.

    The envelope is the ``consult`` envelope plus a ``code_review`` object, so
    the verdict parsing is delegated to :class:`ConsultResult`. Telemetry that a
    code-review envelope does not carry reads as absent there, never as an
    invention: an absent ``consult.degraded`` reads as ``False``, the absent
    consult telemetry fields (``extraction_failures``...) as ``None``, and an
    absent ``completions`` yields an empty ``capped_agents``.

    Attributes:
        consult: The parsed verdicts and consult-level telemetry (``run_id``
            included).
        code_review: The ``code_review`` object, verbatim.
        total_ms: ``timings.total_ms`` of the envelope, or ``None``.
    """

    consult: ConsultResult
    code_review: dict[str, Any]
    total_ms: int | None = None

    @property
    def without_memory(self) -> bool:
        """True when the review reports it ran without memory."""
        return self.code_review.get("without_memory") is True

    @property
    def without_embedder(self) -> bool:
        """True when the review reports it ran without the embedder."""
        return self.code_review.get("without_embedder") is True

    @property
    def degrading_causes(self) -> list[str]:
        """Return the causes that make this review approve no gate, in a stable order.

        Lost context is truncation under the integrity rule: ``without_embedder``; a batch
        that failed (``failed_batches`` non-empty, or ``batches_completed`` short of
        ``batches_planned``) although magi-rs still consolidated the rest; and
        ``undelivered_units`` (listed, or only counted as unlisted).

        Returns:
            The matching names among ``without_embedder``, ``failed_batches``,
            ``undelivered_units``; empty for a complete review.
        """
        review = self.code_review
        causes: list[str] = []
        if self.without_embedder:
            causes.append("without_embedder")
        planned, completed = review.get("batches_planned"), review.get("batches_completed")
        if review.get("failed_batches") or (
            isinstance(planned, int) and isinstance(completed, int) and completed != planned
        ):
            causes.append("failed_batches")
        if review.get("undelivered_units") or review.get("undelivered_units_unlisted"):
            causes.append("undelivered_units")
        return causes

    def effort(self) -> dict[str, Any]:
        """Return the review-effort telemetry; an absent field is ``None``.

        ``duration_ms`` is the envelope's ``timings.total_ms`` (the capture has
        no ``code_review.duration_ms``); ``tokens_by_batch`` is copied from
        ``batches[]``, never estimated.

        Returns:
            A dict with exactly ``batches_planned``, ``batches_completed``,
            ``duration_ms``, ``gate_questions``, ``judged_fraction`` and
            ``tokens_by_batch``.

        Complexity: O(b) in the number of batches.
        """
        effort: dict[str, Any] = {key: self.code_review.get(key) for key in _EFFORT_FROM_REVIEW}
        effort["duration_ms"] = self.total_ms
        batches = self.code_review.get("batches")
        effort["tokens_by_batch"] = (
            [
                {field: batch.get(field) for field in _BATCH_FIELDS}
                for batch in batches
                if isinstance(batch, dict)
            ]
            if isinstance(batches, list)
            else None
        )
        return effort

    @classmethod
    def from_json(cls, raw: str) -> CodeReviewResult:
        """Parse the invocation's stdout.

        Args:
            raw: The process's stdout, one JSON object.

        Returns:
            The parsed result.

        Raises:
            MemoryContractError: ``code_review.without_memory`` is true.
            MagiRsContractError: Any :meth:`ConsultResult.from_json` failure, or
                ``code_review`` absent or not an object.
        """
        consult = ConsultResult.from_json(raw)
        envelope: dict[str, Any] = json.loads(raw)
        review = envelope.get("code_review")
        if not isinstance(review, dict):
            raise MagiRsContractError("magi-rs output carries no 'code_review' object")
        for flag in _REVIEW_FLAGS:
            if flag in review and not isinstance(review[flag], bool):
                raise MagiRsContractError(
                    f"magi-rs 'code_review.{flag}' is not a JSON boolean: an off-contract envelope"
                )
        if review.get("without_memory") is True:
            raise MemoryContractError(
                "magi-rs reports without_memory although --no-memory was never passed; "
                "this is a plugin bug or a magi-rs behaviour change"
            )
        timings = envelope.get("timings")
        total = timings.get("total_ms") if isinstance(timings, dict) else None
        return cls(
            consult=consult,
            code_review=review,
            total_ms=total if isinstance(total, int) and not isinstance(total, bool) else None,
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
    """Return a JSON-supplied name only when it is safe to print.

    Args:
        value: A name read from untrusted JSON (a variant key or a backend).

    Returns:
        ``value`` when it is a str that fully matches ``_PRINTABLE_NAME``, else
        ``_UNPRINTABLE`` -- a partial match would let an escape sequence through.
    """
    if isinstance(value, str) and _PRINTABLE_NAME.fullmatch(value):
        return value
    return _UNPRINTABLE


def _int_chars_part(value: Any) -> str | None:
    """Format a reasoning length for the warning.

    Args:
        value: The ``chars`` field of a reasoning object, of any type.

    Returns:
        ``"<value> chars"`` when ``value`` is an int (``bool`` excluded), else None.
    """
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

    Raises:
        Nothing: every input shape is handled explicitly, because this runs in
        telemetry right before a report is written.

    Example:
        >>> describe_reasoning({"reasoning": {"Measured": {"chars": 4005, "text": None}}})
        'Measured 4005 chars'

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
