#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-13
"""Release gate: measure MAGI's adherence to the verdict-sentinel markers (R17a).

MS2 replaced heuristic verdict recovery with the ``<MAGI_VERDICT>`` / ``</MAGI_VERDICT>``
sentinel (``skills/magi/scripts/verdict_markers.py``): the parser now EXTRACTS the verdict
from between two marker lines instead of scanning the whole response for whatever object
"looks like" one. The residual risk this milestone leaves is that a model simply does not
emit the markers -- and that has to be MEASURED before it ships, not discovered in
production.

This tool drives *N* real MAGI runs over real code bundles and reads back **the run's own
adherence tally** (``extraction_failures`` in ``magi-report.json``, R18) -- the count each
run keeps, per ATTEMPT, at the point where it classifies the failure.

That indirection is the whole design, and the obvious alternative is a trap: re-parsing
each mage's ``{agent}.raw.json`` cannot see a marker omission that the retry then
recovered, because ``launch_agent`` rewrites that file on every attempt, so only the LAST
one survives on disk. An instrument blind to the recovered omissions reports a rate that
is systematically optimistic -- and it would have read GREEN on exactly the drift this
gate exists to catch.

The raw-file path survives as a FALLBACK for a run that died before writing a report, and
it still INSTRUMENTS THE REAL PARSER (it wraps :meth:`verdict_markers.VerdictSentinel.extract`
rather than reimplementing the sentinel's matching rules -- reimplementing them would
measure a COPY of the parser, not the code that ships).

Why this lives in ``tools/`` and not ``skills/magi/scripts/`` (the plugin that ships to
users) or ``scripts/`` (this project's gitignored, per-developer tooling): ``make
release-check`` lives in the tracked ``Makefile`` and calls this script by name. A
tracked target that calls a gitignored script is exactly the bug this project already
paid for once (v5.0.3, ``validate_magi_toml.py`` lived in a gitignored directory for a
whole major release while four published surfaces told the user to run it). The rule this
project learned from that: if a published surface tells the user to run something, that
something ships. ``tools/`` is tracked for exactly that reason.

Usage::

    # Drive N real MAGI runs over real bundles and (re-)write the release artifact.
    # THIS COSTS MONEY / QUOTA -- see the cost note below.
    python tools/measure_marker_adherence.py measure \\
        --ollama-bundle diff1.diff --ollama-bundle diff2.diff \\
        --claude-bundle diff3.diff \\
        --out marker-adherence-report.json

    # Verify the artifact is green AND fresh (what `make release-check` runs).
    python tools/measure_marker_adherence.py check --report marker-adherence-report.json

Cost note: every ``measure`` invocation performs REAL MAGI runs. Ollama's ``:cloud`` tags
are a paid tier, and Claude runs consume the account's usage quota. This is why the tool
is deliberately NOT part of ``make verify`` (which runs before every TDD-phase commit):
it belongs to the release gate alone, run once, on purpose, with the operator's
authorization -- never automatically.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

# Bootstrap: this tool lives outside the ``skills/magi/scripts`` package and is invoked
# directly (``python tools/measure_marker_adherence.py``), so the sibling modules it
# instruments are not importable without this. Mirrors the bootstrap already used
# throughout ``skills/magi/scripts`` (see CLAUDE.md "Open technical debt / synthesize
# import gap [LOCKED]").
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = str(_REPO_ROOT / "skills" / "magi" / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from parse_agent_output import parse_agent_output as _parse_agent_output  # noqa: E402
from retry_feedback import (  # noqa: E402
    CAUSE_INVALID_JSON,
    FEEDBACK_TEMPLATES,
    retry_feedback_cause,
)
from run_magi import AGENTS  # noqa: E402
from verdict_markers import VerdictExtractionError, VerdictSentinel  # noqa: E402

#: The three mage seats, in a fixed order. Single source of truth for BOTH the
#: measurement tally and the ``prompts_sha256`` concatenation below (NR6b DRY):
#: re-declaring this list here would risk it drifting from the real dispatch order.
AGENT_NAMES: tuple[str, ...] = AGENTS

_AGENTS_DIR = _REPO_ROOT / "skills" / "magi" / "agents"
_RUN_MAGI_PATH = _REPO_ROOT / "skills" / "magi" / "scripts" / "run_magi.py"

#: Defaults for the CLI. Named, not magic (NR6).
DEFAULT_TIMEOUT_SECONDS = 900

#: How much longer than ONE mage's timeout the parent waits before calling a run hung. A
#: rotation serialises what would otherwise be concurrent, so the child's own worst case is
#: several times its per-mage bound -- this leaves room for that and for shutdown.
PARENT_TIMEOUT_FACTOR = 6
DEFAULT_REPORT_FILENAME = "marker-adherence-report.json"
DEFAULT_MODE = "code-review"

#: The report a real ``run_magi.py`` invocation writes into its ``--output-dir``. It
#: carries ``extraction_failures`` -- the run's OWN per-attempt tally (R18), which is the
#: only place a marker omission that a retry later recovered is still visible.
RUN_REPORT_FILENAME = "magi-report.json"

#: The one tally key that is not a failure cause.
OK_TALLY_KEY = "ok"

#: The failure vocabulary, taken from the retry contract (R12) rather than re-declared
#: here. Enumerating causes locally would mean a cause added to the contract has no
#: column in the artifact -- and a real failure would be reported as a zero, which is the
#: exact blindness this instrument exists to prevent.
CAUSES: tuple[str, ...] = tuple(FEEDBACK_TEMPLATES)

#: How many runs had to be measured through the BLIND fallback (no ``magi-report.json``,
#: so only each mage's last attempt is on disk). It is carried INTO the artifact and it
#: forbids a green verdict: ``make release-check`` reads the artifact and nothing else, so
#: a warning printed during ``measure`` governs nothing by the time the gate runs.
fallback_measured: int = 0

#: Per-agent tally. Keys are :data:`OK_TALLY_KEY` or one of :data:`CAUSES` -- the same
#: vocabulary ``magi-report.json``'s ``extraction_failures`` uses, so the two sources
#: fold together without translation.
tally: "collections.defaultdict[str, collections.Counter[str]]" = collections.defaultdict(
    collections.Counter
)

#: Set by :func:`agent_context` immediately before parsing ONE mage's raw output, and
#: cleared right after. NEVER a parameter of ``_spy``: the real caller
#: (``parse_agent_output._extract_verdict``) invokes ``sentinel.extract(text)`` with
#: exactly that signature, and a required keyword here would raise ``TypeError`` on
#: every real call -- the instrumentation would break the parser it exists to measure.
_current_agent: str = ""

#: The REAL, unpatched method -- captured once, at import time, before :func:`install_spy`
#: ever runs, so ``_spy`` always calls the genuine sentinel logic and never itself once
#: patched.
_real_extract = VerdictSentinel.extract


def _spy(self, text: str) -> str:  # type: ignore[no-untyped-def]
    """Drop-in replacement for :meth:`VerdictSentinel.extract` that TALLIES, never DECIDES.

    Signature IDENTICAL to the real method on purpose (pinned by
    ``test_the_spy_preserves_the_real_signature``, which compares
    ``inspect.signature`` of the two functions): ``self`` is deliberately LEFT
    UNANNOTATED because the real ``VerdictSentinel.extract`` leaves it unannotated too
    (it is typed implicitly, as a method inside the class body) -- annotating it here
    would make the two signatures compare unequal and silently defeat the very test
    that exists to catch a signature drift. The ``type: ignore[no-untyped-def]`` is the
    deliberate cost of that exact match: a free function (unlike a real method) is not
    exempt from mypy strict's "self must be annotated" rule, so this single ignore
    buys back the fidelity the test depends on.

    The agent whose output is being parsed travels through :data:`_current_agent`, a
    module-level variable set by :func:`agent_context`, never through an extra
    parameter.

    Args:
        self: The :class:`VerdictSentinel` instance the real call was made on.
        text: The raw model output being parsed -- untrusted input.

    Returns:
        Whatever the real :meth:`VerdictSentinel.extract` returns: the delimited block,
        with any wrapping fence stripped.

    Raises:
        RuntimeError: If :data:`_current_agent` is unset. A silent fall-through to the
            ``""`` bucket would produce a report that is PLAUSIBLE and FALSE -- the worst
            outcome for an instrument that exists to be trusted. Fails closed instead.
        VerdictExtractionError: Whatever the real ``extract`` raised, propagated
            unchanged after being tallied -- this spy must never change the parser's
            observable behaviour, only observe it.
        json.JSONDecodeError: If the delimited block decodes to nothing (content-level
            drift, R7) -- tallied and re-raised the same way.
    """
    if not _current_agent:
        raise RuntimeError(
            "measurement bug: _current_agent was never set before this parse -- "
            "refusing to tally into the '' bucket, which would produce a report that "
            "is plausible and false"
        )

    try:
        block = _real_extract(self, text)
    except VerdictExtractionError as exc:
        tally[_current_agent][retry_feedback_cause(exc)] += 1
        raise

    # TALLY the content, do NOT decide on it. ``extract`` RETURNS the block -- the JSON inside is
    # decoded later, by the parser -- so an earlier version of this spy that decoded here and
    # RAISED was sending the parser down a path it never takes in production: an instrument
    # perturbing its subject, while its docstring promised the opposite (MAGI gate, Balthasar).
    # The parser now goes on to fail exactly the way it always would have, and this only counts.
    try:
        json.loads(block)
    except (ValueError, RecursionError):
        # Every way ``json.loads`` refuses a payload: a syntax error, a number over
        # ``int_max_str_digits`` (a PLAIN ValueError), or nesting too deep (RecursionError).
        tally[_current_agent][CAUSE_INVALID_JSON] += 1
    else:
        tally[_current_agent][OK_TALLY_KEY] += 1

    return block


def install_spy() -> None:
    """Patch :class:`VerdictSentinel` so every ``extract`` call in this process tallies.

    Calling it twice WITHOUT an uninstall in between is a no-op (MAGI gate, Balthasar): the
    second install would otherwise leave ``VerdictSentinel.extract`` already pointing at the spy,
    and ``uninstall_spy`` would then restore... the spy. The instrument would stay attached,
    silently, for the rest of the process -- and an instrument you cannot remove is one you can
    no longer trust. Install -> uninstall -> install DOES re-patch, which is the intended cycle.

    Patches the CLASS attribute, not an instance: ``parse_agent_output.py`` already
    holds a constructed ``VerdictSentinel`` at import time, and Python resolves a
    bound-method call through the class at call time, not at construction -- so the
    existing instance is instrumented too, with no changes to that module.
    """
    if VerdictSentinel.extract is _spy:
        return
    VerdictSentinel.extract = _spy  # type: ignore[method-assign]


def uninstall_spy() -> None:
    """Restore the real, unpatched :meth:`VerdictSentinel.extract`."""
    VerdictSentinel.extract = _real_extract  # type: ignore[method-assign]


def reset_tally() -> None:
    """Clear all accumulated counts. Call between independent measurement runs."""
    global fallback_measured
    tally.clear()
    fallback_measured = 0


@contextmanager
def agent_context(agent: str) -> Iterator[None]:
    """Bind *agent* as the implicit target of the next :func:`_spy` call.

    Args:
        agent: The mage whose raw output is about to be parsed ('melchior',
            'balthasar', or 'caspar').

    Yields:
        None.
    """
    global _current_agent
    _current_agent = agent
    try:
        yield
    finally:
        _current_agent = ""


def measure_raw_file(raw_path: Path, agent: str) -> None:
    """Feed one already-captured raw completion through the REAL parser, tallying it.

    Calls the production :func:`parse_agent_output.parse_agent_output` entry point
    directly -- the exact function ``launch_agent`` calls on every real run -- so the
    envelope-unwrap and sentinel-extraction code paths measured here are the ones that
    actually ship, not a reconstruction of them.

    Args:
        raw_path: Path to a ``{agent}.raw.json`` file, exactly as ``launch_agent``
            writes it (the backend's stdout bytes, unmodified).
        agent: The mage that produced *raw_path*.

    Raises:
        FileNotFoundError: If *raw_path* does not exist.
        ValueError: If the raw file exceeds ``validate.MAX_INPUT_FILE_SIZE`` or has an
            unrecognised shape unrelated to marker adherence -- an infrastructure
            problem with the sample, not a data point for this measurement, so it is
            NOT swallowed.
    """
    with tempfile.TemporaryDirectory(prefix="magi-marker-adherence-parse-") as scratch:
        parsed_path = Path(scratch) / f"{agent}.json"
        with agent_context(agent):
            try:
                _parse_agent_output(str(raw_path), str(parsed_path))
            except (VerdictExtractionError, ValueError, RecursionError):
                # ONE clause, because ``json.JSONDecodeError`` IS a ``ValueError`` and listing
                # both invites the reader to think the second adds something. What each is:
                # ``VerdictExtractionError`` -- the markers were missing, doubled or unterminated;
                # ``ValueError`` -- the content between them did not decode (a syntax error, or a
                # number over ``int_max_str_digits``, which raises a PLAIN ValueError);
                # ``RecursionError`` -- it was nested too deep to decode at all.
                #
                # All of them were already TALLIED by ``_spy``, which re-raises; catching them
                # here is what keeps the measurement going. An instrument that dies on one
                # pathological completion leaves the release with NO artifact -- the single
                # outcome it must never produce (MAGI gate, Balthasar and Caspar, two cycles).
                return


def measure_run_report(report_path: Path) -> None:
    """Fold ONE run's own adherence tally (``extraction_failures``, R18) into :data:`tally`.

    **This is the source of truth, and the raw files are not.** ``launch_agent`` writes
    ``{agent}.raw.json`` in ``"wb"`` on EVERY attempt and every rotation, so on disk only
    the LAST attempt survives. A mage that omitted the markers on attempt 1 and got it
    right on the retry therefore leaves a spotless raw behind: re-parsing it would tally
    ``ok`` and the artifact would read green with a real omission inside it. With
    ``max_attempts = 2`` and a true 5% per-attempt omission rate, such an artifact would
    report about 0.25% -- systematically optimistic about the single number the milestone's
    success criterion hangs on.

    The run's own tally does not have that blind spot: it is written where the failure is
    classified, once per ATTEMPT, and it also carries the two causes the spy structurally
    cannot see (``echoed_example`` and ``agent_identity`` are raised by ``launch_agent``
    AFTER ``parse_agent_output`` has already returned).

    ``ok`` is counted once per mage that delivered a verdict in this run: a mage stops
    attempting as soon as it succeeds. So ``ok + failures`` is that seat's attempts **that
    reached the parser** -- NOT its attempt count: a transport retry (5xx, timeout) never
    reaches the sentinel and is deliberately not tallied here. A rate computed from these
    numbers is therefore the marker-omission rate per PARSED attempt, which is the quantity
    R17 is about; counting a timed-out attempt as a clean one would understate it.

    Args:
        report_path: Path to a run's ``magi-report.json``.

    Raises:
        json.JSONDecodeError: If the report is not valid JSON -- an infrastructure problem
            with the sample, never silently swallowed into a zero.
    """
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    failures: Mapping[str, Mapping[str, int]] = report.get("extraction_failures", {})
    delivered = {agent.get("agent") for agent in report.get("agents", [])}

    for agent in AGENT_NAMES:
        for cause, count in failures.get(agent, {}).items():
            tally[agent][cause] += count
        if agent in delivered:
            tally[agent][OK_TALLY_KEY] += 1


def measure_output_dir(output_dir: Path) -> None:
    """Measure one completed run: its own tally when it wrote a report, its raws otherwise.

    A run that dies below the two-agent floor never reaches the report-writing step. The
    raw files its mages did produce are still real samples, so they are measured through
    the production parser -- what is never done is fabricating a zero for a seat that
    produced no data at all (:func:`build_artifact` raises on that).

    Args:
        output_dir: The ``--output-dir`` a real ``run_magi.py`` invocation was given.
    """
    global fallback_measured

    report_path = output_dir / RUN_REPORT_FILENAME
    if report_path.exists():
        measure_run_report(report_path)
        return

    # The fallback is BLIND in two ways, so it is recorded and it forbids a green verdict
    # (:func:`build_artifact`): the raws hold only each mage's LAST attempt, and an ``ok``
    # here is signed by the sentinel plus ``json.loads`` alone -- ``load_agent_output``, the
    # echo canary and the identity check never run, so it would count as clean a verdict the
    # real run REJECTED. A stderr warning cannot carry that: ``check`` (what
    # ``make release-check`` runs) reads the artifact and nothing else.
    fallback_measured += 1
    print(
        f"WARNING: no {RUN_REPORT_FILENAME} in {output_dir} (the run died before writing "
        "one) -- falling back to the raw files, which only hold each mage's LAST attempt. "
        "This run CANNOT certify the release; the artifact will not be green.",
        file=sys.stderr,
    )
    for agent in AGENT_NAMES:
        raw_path = output_dir / f"{agent}.raw.json"
        if raw_path.exists():
            measure_raw_file(raw_path, agent)


def run_real_magi(
    mode: str,
    bundle: Path,
    *,
    ollama: bool,
    timeout: int,
    output_dir: Path,
) -> None:
    """Invoke the real ``run_magi.py`` CLI as a subprocess -- an actual MAGI run.

    This is a genuine end-to-end invocation: real subprocess/HTTP calls to the
    configured backend, real prompt enrichment, real tool use by the agents. It is NOT
    a simulation of MAGI -- reproducing its orchestration logic here would defeat the
    entire point of measuring the code that ships.

    Args:
        mode: One of ``run_magi``'s ``VALID_MODES`` (default usage: ``"code-review"``).
        bundle: Path to the input file (a real ``git diff``, not synthetic text).
        ollama: If ``True``, pass ``--ollama``; otherwise the default Claude backend.
        timeout: Per-agent timeout in seconds, forwarded as ``--timeout``.
        output_dir: Directory the run writes its per-agent artifacts into. Created if
            absent.

    Note:
        The subprocess's own exit code is deliberately not checked: a degraded run (or
        one that dies below the 2-agent floor) still leaves behind whatever raw files
        each mage managed to produce before failing, and those are still valid
        measurement samples.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(_RUN_MAGI_PATH),
        mode,
        str(bundle),
        "--output-dir",
        str(output_dir),
        "--timeout",
        str(timeout),
        "--no-status",
    ]
    if ollama:
        command.append("--ollama")
    # The parent gets a bound of its own (MAGI gate, Balthasar). ``--timeout`` bounds each MAGE,
    # which reads like the whole thing is bounded -- and it is not: a child that deadlocks before
    # its own timeouts fire, or while shutting down, would hang the release measurement FOREVER,
    # with no artifact and no error. A bound only the child enforces is not a bound on the parent.
    #
    # Generous on purpose: three mages can run concurrently but a rotation makes them serial, so
    # the child's own worst case is several times its per-mage timeout. Killing a healthy run
    # would be worse than waiting on a sick one; this is a deadlock guard, not a schedule.
    try:
        subprocess.run(
            command, check=False, cwd=_REPO_ROOT, timeout=timeout * PARENT_TIMEOUT_FACTOR
        )
    except subprocess.TimeoutExpired:
        print(
            f"WARNING: the run over {bundle} did not finish within "
            f"{timeout * PARENT_TIMEOUT_FACTOR}s and was killed. Its mages contribute no data "
            "point; the measurement carries on.",
            file=sys.stderr,
        )


def _git_head_sha(repo_root: Path) -> str:
    """Return the current ``HEAD`` commit SHA of *repo_root*.

    Args:
        repo_root: Path to the git working tree.

    Returns:
        The full 40-character commit hash.

    Raises:
        subprocess.CalledProcessError: If ``git rev-parse`` fails (not a git repo, or
            no commits yet).
    """
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _prompts_sha256(agents_dir: Path) -> str:
    """Hash the three shipped agent prompts together, in a fixed order.

    This is the field that actually matters for the release gate (more than
    ``git_sha``): what is measured is the models' adherence to THESE markers and THIS
    text. If any ``agents/*.md`` changes -- even a single comma -- the measurement no
    longer applies, and the hash changing is exactly what makes :func:`check_release_gate`
    catch that.

    Args:
        agents_dir: Directory containing ``melchior.md``, ``balthasar.md``,
            ``caspar.md``.

    Returns:
        The hex-encoded SHA-256 digest of the three files' bytes, concatenated in
        :data:`AGENT_NAMES` order.
    """
    hasher = hashlib.sha256()
    for name in AGENT_NAMES:
        hasher.update((agents_dir / f"{name}.md").read_bytes())
    return hasher.hexdigest()


def build_artifact(
    repo_root: Path,
    agents_dir: Path,
    runs: Mapping[str, int],
) -> dict[str, Any]:
    """Build the release-gate artifact from the accumulated module-level :data:`tally`.

    Args:
        repo_root: Repository root, used to read the current ``HEAD`` SHA.
        agents_dir: Directory of the shipped agent prompts, used for the provenance
            hash.
        runs: Number of real runs performed per backend, e.g.
            ``{"ollama": 5, "claude": 2}``.

    Returns:
        A JSON-serialisable dict matching the artifact schema: ``git_sha``,
        ``prompts_sha256``, ``measured_at``, ``runs``, ``per_seat`` (per agent: ``ok``
        plus each cause in :data:`CAUSES`), and ``verdict`` -- ``"green"`` only
        if every seat has zero extraction failures, ``"red"`` otherwise.

    Raises:
        RuntimeError: If a seat has NO data at all (zero ``ok`` and zero failures) --
            an artifact built from zero samples for a seat is not a measurement, it is
            a guess dressed as one.
    """
    per_seat: dict[str, dict[str, int]] = {}
    verdict = "green"
    for agent in AGENT_NAMES:
        counts = tally.get(agent, collections.Counter())
        seat: dict[str, int] = {OK_TALLY_KEY: counts.get(OK_TALLY_KEY, 0)}
        for cause in CAUSES:
            count = counts.get(cause, 0)
            seat[cause] = count
            if count:
                verdict = "red"
        per_seat[agent] = seat

        if seat[OK_TALLY_KEY] == 0 and all(seat[cause] == 0 for cause in CAUSES):
            raise RuntimeError(
                f"no raw output was measured for agent {agent!r} -- an artifact built "
                "from zero samples for a seat cannot certify anything"
            )

    if fallback_measured:
        # A blind measurement cannot certify anything -- not even when it saw no failure.
        # This is the one place the instrument used to hand out a green it had not earned.
        verdict = "red"

    return {
        "git_sha": _git_head_sha(repo_root),
        "prompts_sha256": _prompts_sha256(agents_dir),
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": dict(runs),
        "fallback_measured": fallback_measured,
        "per_seat": per_seat,
        "verdict": verdict,
    }


def check_release_gate(
    report_path: Path,
    repo_root: Path,
    agents_dir: Path,
) -> tuple[bool, str]:
    """Verify the artifact at *report_path* exists, is GREEN, and is FRESH.

    "Fresh" means its ``git_sha`` matches ``HEAD`` and its ``prompts_sha256`` matches
    the CURRENT ``agents/*.md`` files. A report that is green but stale would certify a
    parser or a set of prompts that is no longer what ships -- the same failure mode as
    an expired ``STRONG GO``.

    Args:
        report_path: Path to the artifact written by ``measure``.
        repo_root: Repository root, used to read the current ``HEAD`` SHA.
        agents_dir: Directory of the shipped agent prompts.

    Returns:
        A ``(passed, message)`` tuple. *message* explains WHY on failure, and is a
        short confirmation on success.
    """
    if not report_path.exists():
        return False, (
            f"no marker-adherence report at {report_path} -- run "
            "`python tools/measure_marker_adherence.py measure ...` before releasing"
        )

    try:
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"marker-adherence report at {report_path} is not valid JSON: {exc}"

    current_sha = _git_head_sha(repo_root)
    if report.get("git_sha") != current_sha:
        return False, (
            f"stale marker-adherence report: measured at commit {report.get('git_sha')!r}, "
            f"HEAD is now {current_sha!r} -- re-run the measurement"
        )

    current_prompts_sha = _prompts_sha256(agents_dir)
    if report.get("prompts_sha256") != current_prompts_sha:
        return False, (
            "stale marker-adherence report: agents/*.md changed since it was measured "
            f"(measured {report.get('prompts_sha256')!r}, current {current_prompts_sha!r}) "
            "-- re-run the measurement"
        )

    if "fallback_measured" not in report:
        # A fail-closed gate cannot read an ABSENT field as "there was no blindness": that is
        # inferring a guarantee from a silence. Today it is unreachable (the freshness check
        # rejects any old artifact first), and precisely for that reason the safe default is
        # explicit: the day freshness changes, this does not turn into a fail-open.
        return False, (
            f"marker-adherence report at {report_path} predates the blind-measurement check "
            "(no 'fallback_measured' field) -- re-run the measurement"
        )

    blind = report["fallback_measured"]
    if blind:
        return False, (
            f"{blind} of the measured runs died before writing {RUN_REPORT_FILENAME}, so they "
            "were measured through the BLIND fallback (each mage's LAST attempt only, with no "
            "canary or identity check) -- that cannot certify a release. Fix the runs that "
            "died and re-measure"
        )

    if report.get("verdict") != "green":
        return False, (
            f"marker-adherence verdict is {report.get('verdict')!r}, not green -- "
            f"per_seat: {report.get('per_seat')}"
        )

    return (
        True,
        f"marker-adherence gate OK ({report.get('measured_at')}, runs={report.get('runs')})",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the two-subcommand CLI (``measure`` and ``check``).

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    measure = subparsers.add_parser(
        "measure",
        help="Drive real MAGI runs over real bundles and (re-)write the artifact.",
    )
    measure.add_argument(
        "--ollama-bundle",
        action="append",
        default=[],
        metavar="PATH",
        help="A real code-review bundle (e.g. a git diff) to run via --ollama. Repeatable.",
    )
    measure.add_argument(
        "--claude-bundle",
        action="append",
        default=[],
        metavar="PATH",
        help="A real code-review bundle to run via the default Claude backend. Repeatable.",
    )
    measure.add_argument(
        "--mode", default=DEFAULT_MODE, help=f"Analysis mode (default: {DEFAULT_MODE})"
    )
    measure.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-agent timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    measure.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_REPORT_FILENAME),
        help=f"Artifact output path (default: {DEFAULT_REPORT_FILENAME})",
    )

    check = subparsers.add_parser(
        "check",
        help="Verify the artifact is green and fresh (what `make release-check` runs).",
    )
    check.add_argument(
        "--report",
        type=Path,
        default=Path(DEFAULT_REPORT_FILENAME),
        help=f"Artifact to verify (default: {DEFAULT_REPORT_FILENAME})",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success/green, 1 otherwise.
    """
    args = _build_arg_parser().parse_args(argv)

    if args.command == "check":
        passed, message = check_release_gate(args.report, _REPO_ROOT, _AGENTS_DIR)
        print(message, file=sys.stdout if passed else sys.stderr)
        return 0 if passed else 1

    if not args.ollama_bundle and not args.claude_bundle:
        print(
            "measure: pass at least one --ollama-bundle or --claude-bundle",
            file=sys.stderr,
        )
        return 1

    reset_tally()
    install_spy()
    try:
        with tempfile.TemporaryDirectory(prefix="magi-marker-adherence-run-") as scratch_root:
            scratch = Path(scratch_root)
            for index, bundle in enumerate(args.ollama_bundle):
                run_dir = scratch / f"ollama-{index}"
                run_real_magi(
                    args.mode, Path(bundle), ollama=True, timeout=args.timeout, output_dir=run_dir
                )
                measure_output_dir(run_dir)
            for index, bundle in enumerate(args.claude_bundle):
                run_dir = scratch / f"claude-{index}"
                run_real_magi(
                    args.mode, Path(bundle), ollama=False, timeout=args.timeout, output_dir=run_dir
                )
                measure_output_dir(run_dir)

            artifact = build_artifact(
                _REPO_ROOT,
                _AGENTS_DIR,
                {"ollama": len(args.ollama_bundle), "claude": len(args.claude_bundle)},
            )
    finally:
        uninstall_spy()

    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} -- verdict: {artifact['verdict']}")
    return 0 if artifact["verdict"] == "green" else 1


if __name__ == "__main__":
    sys.exit(main())
