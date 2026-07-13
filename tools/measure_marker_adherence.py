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

This tool drives *N* real MAGI runs over real code bundles and INSTRUMENTS THE REAL
PARSER: it wraps :meth:`verdict_markers.VerdictSentinel.extract` and tallies the
exception TYPE it raises. It never reimplements the sentinel's matching rules --
reimplementing them would measure a COPY of the parser, not the code that actually ships
and runs in production.

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
from run_magi import AGENTS  # noqa: E402
from verdict_markers import (  # noqa: E402
    AmbiguousVerdictMarkers,
    MissingVerdictMarkers,
    UnterminatedVerdictBlock,
    VerdictExtractionError,
    VerdictSentinel,
)

#: The three mage seats, in a fixed order. Single source of truth for BOTH the
#: measurement tally and the ``prompts_sha256`` concatenation below (NR6b DRY):
#: re-declaring this list here would risk it drifting from the real dispatch order.
AGENT_NAMES: tuple[str, ...] = AGENTS

_AGENTS_DIR = _REPO_ROOT / "skills" / "magi" / "agents"
_RUN_MAGI_PATH = _REPO_ROOT / "skills" / "magi" / "scripts" / "run_magi.py"

#: Defaults for the CLI. Named, not magic (NR6).
DEFAULT_TIMEOUT_SECONDS = 900
DEFAULT_REPORT_FILENAME = "marker-adherence-report.json"
DEFAULT_MODE = "code-review"

#: The two tally keys ``_spy`` uses that are NOT a :class:`VerdictExtractionError`
#: subclass name.
OK_TALLY_KEY = "ok"
INVALID_JSON_TALLY_KEY = "InvalidJSON"

#: Maps a tally key (an exception class name, or :data:`INVALID_JSON_TALLY_KEY`) to the
#: snake_case cause label used in the artifact's ``per_seat`` section -- the same
#: vocabulary ``magi-report.json``'s ``extraction_failures`` field already uses
#: (``retry_feedback._retry_feedback_cause``), so a human reading both never has to
#: mentally translate between two names for the same failure.
_CAUSE_LABELS: dict[str, str] = {
    MissingVerdictMarkers.__name__: "missing_markers",
    UnterminatedVerdictBlock.__name__: "unterminated_block",
    AmbiguousVerdictMarkers.__name__: "ambiguous_markers",
    INVALID_JSON_TALLY_KEY: "invalid_json",
}

#: Per-agent tally of the sentinel's REAL behaviour, filled in by :func:`_spy`. Keys are
#: ``OK_TALLY_KEY``, ``INVALID_JSON_TALLY_KEY``, or a :class:`VerdictExtractionError`
#: subclass's ``__name__``.
tally: "collections.defaultdict[str, collections.Counter[str]]" = collections.defaultdict(
    collections.Counter
)

#: Set by :func:`agent_context` immediately before parsing ONE mage's raw output, and
#: cleared right after. NEVER a parameter of ``_spy``: the real caller
#: (``parse_agent_output._loads_lenient``) invokes ``sentinel.extract(text)`` with
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
        tally[_current_agent][type(exc).__name__] += 1
        raise

    try:
        json.loads(block)
    except json.JSONDecodeError:
        tally[_current_agent][INVALID_JSON_TALLY_KEY] += 1
        raise

    tally[_current_agent][OK_TALLY_KEY] += 1
    return block


def install_spy() -> None:
    """Patch :class:`VerdictSentinel` so every ``extract`` call in this process tallies.

    Patches the CLASS attribute, not an instance: ``parse_agent_output.py`` already
    holds a constructed ``VerdictSentinel`` at import time, and Python resolves a
    bound-method call through the class at call time, not at construction -- so the
    existing instance is instrumented too, with no changes to that module.
    """
    VerdictSentinel.extract = _spy  # type: ignore[method-assign]


def uninstall_spy() -> None:
    """Restore the real, unpatched :meth:`VerdictSentinel.extract`."""
    VerdictSentinel.extract = _real_extract  # type: ignore[method-assign]


def reset_tally() -> None:
    """Clear all accumulated counts. Call between independent measurement runs."""
    tally.clear()


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
            except (VerdictExtractionError, json.JSONDecodeError):
                # Already tallied by ``_spy`` (VerdictExtractionError: marker omission
                # / ambiguity; JSONDecodeError: invalid JSON inside the markers, or an
                # unrecognised envelope shape that ``parse_agent_output`` remaps to
                # JSONDecodeError). Either way this measurement run continues.
                return


def measure_output_dir(output_dir: Path) -> None:
    """Measure every mage's raw completion found in one completed run's output dir.

    A mage that died before ever producing a raw file (e.g. the preflight rejected the
    whole run) simply contributes no data point for that seat in that run -- silently
    skipped, not fabricated as a zero.

    Args:
        output_dir: The ``--output-dir`` a real ``run_magi.py`` invocation was given.
    """
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
    subprocess.run(command, check=False, cwd=_REPO_ROOT)


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
        plus each cause in :data:`_CAUSE_LABELS`), and ``verdict`` -- ``"green"`` only
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
        for tally_key, label in _CAUSE_LABELS.items():
            count = counts.get(tally_key, 0)
            seat[label] = count
            if count:
                verdict = "red"
        per_seat[agent] = seat

        if seat[OK_TALLY_KEY] == 0 and all(seat[label] == 0 for label in _CAUSE_LABELS.values()):
            raise RuntimeError(
                f"no raw output was measured for agent {agent!r} -- an artifact built "
                "from zero samples for a seat cannot certify anything"
            )

    return {
        "git_sha": _git_head_sha(repo_root),
        "prompts_sha256": _prompts_sha256(agents_dir),
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": dict(runs),
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
