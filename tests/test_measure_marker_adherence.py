# Author: Julian Bolivar
# Version: 8.0.0
# Date: 2026-11-21
"""Tests for the R17a release gate: ``tools/measure_marker_adherence.py``.

A broken measurement instrument does not give a bad number -- it gives a FALSE one, which
is worse than giving none. These tests exist to pin exactly that: the spy's signature must
never diverge from the real ``VerdictSentinel.extract`` it wraps, a missing agent context
must fail closed rather than silently mis-tally, and the release-check gate must reject an
artifact that no longer describes the code it would certify.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

# Bootstrap: ``tools/`` is not a package on sys.path by default (mirrors the bootstrap
# pattern used throughout skills/magi/scripts).
_TOOLS_DIR = str(Path(__file__).resolve().parent.parent / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import measure_marker_adherence as mma  # noqa: E402
from verdict_markers import VerdictSentinel  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_instrument_state():
    """Guarantee every test starts and ends with the spy uninstalled and tally empty.

    Without this, a test that installs the spy and then fails would leave
    ``VerdictSentinel.extract`` patched for every subsequent test in the process --
    exactly the kind of cross-test bleed the project's TDD rules forbid.
    """
    mma.reset_tally()
    mma.uninstall_spy()
    yield
    mma.uninstall_spy()
    mma.reset_tally()


def _write_raw(tmp_path: Path, agent: str, content: str) -> Path:
    """Write *content* as the raw completion file ``launch_agent`` would produce."""
    raw_path = tmp_path / f"{agent}.raw.json"
    raw_path.write_bytes(content.encode("utf-8"))
    return raw_path


# --- The spy itself ----------------------------------------------------------------


def test_the_spy_preserves_the_real_signature():
    """If the spy required a kwarg, it would raise TypeError on every real call."""
    import inspect

    assert inspect.signature(mma._spy) == inspect.signature(mma._real_extract)


def test_spy_fails_closed_when_agent_not_set():
    """A parse attempted with no agent bound must raise, not tally into ''."""
    mma.install_spy()
    sentinel = VerdictSentinel()

    with pytest.raises(RuntimeError, match="_current_agent"):
        sentinel.extract("<MAGI_VERDICT>\n{}\n</MAGI_VERDICT>")

    assert not mma.tally


def test_spy_tallies_missing_markers_and_reraises():
    """A response with no markers at all increments MissingVerdictMarkers for caspar."""
    from verdict_markers import MissingVerdictMarkers

    mma.install_spy()
    sentinel = VerdictSentinel()

    with mma.agent_context("caspar"):
        with pytest.raises(MissingVerdictMarkers):
            sentinel.extract("just some prose, no markers here")

    assert mma.tally["caspar"]["missing_markers"] == 1


def test_spy_tallies_unterminated_block():
    """An opening marker with no closing marker tallies as a truncation signature."""
    from verdict_markers import UnterminatedVerdictBlock

    mma.install_spy()
    sentinel = VerdictSentinel()

    with mma.agent_context("melchior"):
        with pytest.raises(UnterminatedVerdictBlock):
            sentinel.extract("<MAGI_VERDICT>\n{'truncated':")

    assert mma.tally["melchior"]["unterminated_block"] == 1


def test_spy_tallies_invalid_json_inside_markers_WITHOUT_raising():
    """Markers present, body not valid JSON (R7 content drift): counted, not decided on.

    The spy used to RAISE here, and ``extract`` does not: it returns the block and lets the
    parser decode it. Raising sent the parser down a path it never takes in production -- an
    instrument perturbing its subject (MAGI gate, Balthasar). The count is the same; the
    behaviour is now the real one.
    """
    mma.install_spy()
    sentinel = VerdictSentinel()

    with mma.agent_context("balthasar"):
        block = sentinel.extract("<MAGI_VERDICT>\nnot valid json at all\n</MAGI_VERDICT>")

    assert block == "not valid json at all", "the block is RETURNED, as the real method does"
    assert mma.tally["balthasar"]["invalid_json"] == 1
    # The marker layer itself succeeded -- only the content tally fires.
    assert "missing_markers" not in mma.tally["balthasar"]


def test_spy_tallies_a_payload_too_deep_to_decode_WITHOUT_raising():
    """CPython raises ``RecursionError`` -- not ``JSONDecodeError`` -- on deeply nested JSON.

    Counted like any other undecodable content, and NOT re-raised: the block is RETURNED, and the
    parser then fails the way it always would (which is what ``measure_raw_file`` absorbs). The
    spy used to raise here, which crashed the instrument in cycle 3 and, once that was patched,
    still left it perturbing the very code it measures (cycle 19). It observes now.
    """
    mma.install_spy()
    sentinel = VerdictSentinel()
    too_deep = "[" * 20_000 + "]" * 20_000

    with mma.agent_context("caspar"):
        block = sentinel.extract(f"<MAGI_VERDICT>\n{too_deep}\n</MAGI_VERDICT>")

    assert block == too_deep

    assert mma.tally["caspar"]["invalid_json"] == 1


def test_measuring_an_undecodable_payload_does_not_ABORT_the_measurement(tmp_path):
    """MAGI gate (Balthasar, cycle 8) -- and he caught a defect I introduced fixing his last one.

    The spy tallies the ``RecursionError`` and re-raises it, and its own comment promised *"the
    measurement carries on"*. It did not: ``measure_raw_file`` caught only
    ``VerdictExtractionError`` and ``JSONDecodeError``, so the re-raise escaped and took the
    whole release measurement down -- leaving no artifact at all, which is the one outcome this
    instrument must never produce. A comment asserting more than the code does is the exact
    defect this project keeps paying for.
    """
    raw = _write_raw(
        tmp_path, "caspar", "<MAGI_VERDICT>\n" + "[" * 20_000 + "]" * 20_000 + "\n</MAGI_VERDICT>"
    )
    mma.install_spy()

    mma.measure_raw_file(raw, "caspar")  # must NOT raise

    assert mma.tally["caspar"]["invalid_json"] == 1


def test_spy_tallies_ok_on_a_clean_verdict():
    """A well-formed delimited block increments the 'ok' bucket and returns the block."""
    mma.install_spy()
    sentinel = VerdictSentinel()

    with mma.agent_context("caspar"):
        block = sentinel.extract('<MAGI_VERDICT>\n{"agent": "caspar"}\n</MAGI_VERDICT>')

    assert block == '{"agent": "caspar"}'
    assert mma.tally["caspar"][mma.OK_TALLY_KEY] == 1


def test_install_spy_instruments_the_module_level_sentinel_instance():
    """Patching the CLASS attribute must affect an ALREADY-CONSTRUCTED instance too.

    This is exactly the situation in production: ``parse_agent_output.py`` builds its
    ``_SENTINEL`` at import time, long before this tool ever runs.
    """
    from verdict_markers import MissingVerdictMarkers

    pre_existing_instance = VerdictSentinel()
    mma.install_spy()

    with mma.agent_context("caspar"):
        with pytest.raises(MissingVerdictMarkers):
            pre_existing_instance.extract("no markers")

    assert mma.tally["caspar"]["missing_markers"] == 1


# --- measure_raw_file / measure_output_dir -----------------------------------------


def test_measure_raw_file_tallies_through_the_real_parse_agent_output(tmp_path):
    """End-to-end through the production ``parse_agent_output`` entry point."""
    raw_path = _write_raw(
        tmp_path, "caspar", 'some prose\n<MAGI_VERDICT>\n{"agent": "caspar"}\n</MAGI_VERDICT>\n'
    )
    mma.install_spy()

    mma.measure_raw_file(raw_path, "caspar")

    assert mma.tally["caspar"][mma.OK_TALLY_KEY] == 1


def test_measure_raw_file_tallies_missing_markers(tmp_path):
    raw_path = _write_raw(tmp_path, "melchior", "the model forgot the markers entirely")
    mma.install_spy()

    mma.measure_raw_file(raw_path, "melchior")

    assert mma.tally["melchior"]["missing_markers"] == 1


def test_measure_output_dir_skips_agents_with_no_raw_file(tmp_path):
    """A mage that never produced a raw file contributes no data point, silently."""
    _write_raw(tmp_path, "caspar", '<MAGI_VERDICT>\n{"agent": "caspar"}\n</MAGI_VERDICT>')
    mma.install_spy()

    mma.measure_output_dir(tmp_path)

    assert mma.tally["caspar"][mma.OK_TALLY_KEY] == 1
    assert "melchior" not in mma.tally
    assert "balthasar" not in mma.tally


# --- The run's OWN tally is the source of truth (R17/R18) -----------------------------


def _write_run_report(tmp_path: Path, *, agents, extraction_failures=None) -> Path:
    """Write a ``magi-report.json`` exactly as ``run_magi.main`` writes it."""
    report: dict = {"agents": [{"agent": name} for name in agents], "consensus": {}}
    if extraction_failures:
        report["extraction_failures"] = extraction_failures
    path = tmp_path / mma.RUN_REPORT_FILENAME
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_a_retry_recovered_omission_is_NOT_measured_as_clean(tmp_path):
    """The failure the instrument exists to count, and which it could NOT see.

    ``launch_agent`` rewrites ``{agent}.raw.json`` on EVERY attempt, so the file only keeps
    the LAST one. A mage that omits the markers and gets it right on the retry left behind a
    spotless raw -> the artifact came out **green** with a real omission inside it. With
    ``max_attempts=2`` and a true rate of 5 %, the artifact would have reported ~0.25 %:
    systematically optimistic, in exactly the number the success criterion hangs from.
    """
    for agent in mma.AGENT_NAMES:  # the LAST attempt's raw: clean, from the retry that hit
        _write_raw(tmp_path, agent, f'<MAGI_VERDICT>\n{{"agent": "{agent}"}}\n</MAGI_VERDICT>')
    _write_run_report(
        tmp_path,
        agents=mma.AGENT_NAMES,
        extraction_failures={"caspar": {"missing_markers": 1}},
    )
    mma.install_spy()

    mma.measure_output_dir(tmp_path)

    assert mma.tally["caspar"]["missing_markers"] == 1, "the first attempt's omission IS VISIBLE"
    assert mma.tally["caspar"][mma.OK_TALLY_KEY] == 1, "and so is the retry that got it right"
    artifact = mma.build_artifact(
        mma._REPO_ROOT, mma._AGENTS_DIR, {"ollama": 1, "claude": 0}, mode="analysis"
    )
    assert artifact["verdict"] == "red", "a run with an omission is NOT a green run"


def test_the_report_carries_the_causes_the_spy_STRUCTURALLY_cannot_see(tmp_path):
    """``EchoedExampleRejected`` and ``AgentIdentityError`` are raised AFTER the parser.

    The spy wraps ``VerdictSentinel.extract``, so those two never go through it: an artifact
    built from the spy alone certified 4 of the 6 causes. The run's own tally has them all --
    because it classifies them where they are decided.
    """
    for agent in mma.AGENT_NAMES:
        _write_raw(tmp_path, agent, f'<MAGI_VERDICT>\n{{"agent": "{agent}"}}\n</MAGI_VERDICT>')
    _write_run_report(
        tmp_path,
        agents=mma.AGENT_NAMES,
        extraction_failures={
            "caspar": {"echoed_example": 1},
            "melchior": {"agent_identity": 2},
        },
    )
    mma.install_spy()

    mma.measure_output_dir(tmp_path)

    artifact = mma.build_artifact(
        mma._REPO_ROOT, mma._AGENTS_DIR, {"ollama": 1, "claude": 0}, mode="analysis"
    )
    assert artifact["per_seat"]["caspar"]["echoed_example"] == 1
    assert artifact["per_seat"]["melchior"]["agent_identity"] == 2
    assert artifact["verdict"] == "red"


def test_a_run_that_died_without_a_report_falls_back_to_the_raw_files(tmp_path):
    """A run that dies below the floor of 2 mages writes no report.

    The raws it did manage to produce are still valid samples: they are measured with the
    real parser, as before. What is NEVER done is to fabricate a zero.
    """
    _write_raw(tmp_path, "caspar", "the model forgot the markers")
    mma.install_spy()

    mma.measure_output_dir(tmp_path)

    assert mma.tally["caspar"]["missing_markers"] == 1
    assert "melchior" not in mma.tally


def test_a_run_measured_by_the_BLIND_fallback_can_NEVER_certify_green(tmp_path):
    """The blind path cannot sign off the release -- not even if it sees zero failures.

    The fallback re-parses the raws, which only keep each mage's LAST attempt: it is
    **exactly** the optimistic instrument this gate has just eliminated. Worse: the
    fallback's ``ok`` is signed by the sentinel + ``json.loads``, with no canary and no
    identity check, so it counts as good a verdict the real run would have **rejected**.

    The WARNING on stderr is not enough: ``make release-check`` runs ``check``, which reads
    **only the artifact** -- and by then the warning no longer exists. What is not written
    into the artifact governs nothing.
    """
    for agent in mma.AGENT_NAMES:  # three spotless raws, no report at all
        _write_raw(tmp_path, agent, f'<MAGI_VERDICT>\n{{"agent": "{agent}"}}\n</MAGI_VERDICT>')
    mma.install_spy()

    mma.measure_output_dir(tmp_path)
    artifact = mma.build_artifact(
        mma._REPO_ROOT, mma._AGENTS_DIR, {"ollama": 1, "claude": 0}, mode="analysis"
    )

    assert artifact["fallback_measured"] == 1, "the artifact SAYS it measured blind"
    assert artifact["verdict"] != "green", "a blind measurement certifies nothing"


def test_check_rejects_an_artifact_that_was_measured_blind(tmp_path):
    """And the gate rejects it by reading the artifact, which is all it ever sees."""
    _write_report(tmp_path / "r.json", fallback_measured=1)

    passed, message = mma.check_release_gate(tmp_path / "r.json", mma._REPO_ROOT, mma._AGENTS_DIR)

    assert not passed
    assert "blind" in message.lower() or "fallback" in message.lower()


def test_every_cause_of_the_retry_contract_has_a_column_in_the_artifact():
    """There is ONE vocabulary of causes: that of ``FEEDBACK_TEMPLATES`` (R12).

    If the artifact enumerated its own causes, a new cause in the retry contract would have
    no column -- and a real failure would be counted as zero. Which is exactly the class of
    blindness this cycle is fixing.
    """
    from retry_feedback import FEEDBACK_TEMPLATES

    for agent in mma.AGENT_NAMES:
        mma.tally[agent][mma.OK_TALLY_KEY] = 1

    artifact = mma.build_artifact(
        mma._REPO_ROOT, mma._AGENTS_DIR, {"ollama": 1, "claude": 0}, mode="analysis"
    )

    for agent in mma.AGENT_NAMES:
        assert set(artifact["per_seat"][agent]) == {mma.OK_TALLY_KEY, *FEEDBACK_TEMPLATES}


# --- build_artifact ------------------------------------------------------------------


def _fill_tally_all_ok() -> None:
    for agent in mma.AGENT_NAMES:
        mma.tally[agent][mma.OK_TALLY_KEY] = 3


def test_build_artifact_includes_git_sha_and_prompts_sha256():
    _fill_tally_all_ok()

    artifact = mma.build_artifact(
        mma._REPO_ROOT, mma._AGENTS_DIR, {"ollama": 1, "claude": 1}, mode="analysis"
    )

    assert artifact["git_sha"] == mma._git_head_sha(mma._REPO_ROOT)
    assert artifact["prompts_sha256"] == mma._prompts_sha256(mma._AGENTS_DIR)
    assert len(artifact["prompts_sha256"]) == 64  # hex sha256
    assert artifact["verdict"] == "green"
    assert artifact["runs"] == {"ollama": 1, "claude": 1}


def test_build_artifact_verdict_red_when_any_failure_present():
    _fill_tally_all_ok()
    mma.tally["caspar"]["missing_markers"] = 1

    artifact = mma.build_artifact(
        mma._REPO_ROOT, mma._AGENTS_DIR, {"ollama": 1, "claude": 0}, mode="analysis"
    )

    assert artifact["verdict"] == "red"
    assert artifact["per_seat"]["caspar"]["missing_markers"] == 1


def test_build_artifact_raises_when_a_seat_has_zero_data():
    mma.tally["caspar"][mma.OK_TALLY_KEY] = 3
    mma.tally["melchior"][mma.OK_TALLY_KEY] = 3
    # balthasar never appears at all.

    with pytest.raises(RuntimeError, match="balthasar"):
        mma.build_artifact(
            mma._REPO_ROOT, mma._AGENTS_DIR, {"ollama": 1, "claude": 0}, mode="analysis"
        )


# --- check_release_gate --------------------------------------------------------------


def _write_report(path: Path, **overrides) -> None:
    report = {
        "git_sha": mma._git_head_sha(mma._REPO_ROOT),
        "prompts_sha256": mma._prompts_sha256(mma._AGENTS_DIR),
        "measured_at": "2026-07-13T00:00:00Z",
        "schema": "v8",
        "mode": "analysis",
        "runs": {"ollama": 5, "claude": 2},
        "fallback_measured": 0,
        "per_seat": {a: {"ok": 5, "missing_markers": 0} for a in mma.AGENT_NAMES},
        "verdict": "green",
    }
    report.update(overrides)
    path.write_text(json.dumps(report), encoding="utf-8")


def test_check_rejects_an_artifact_with_no_blindness_field_at_all(tmp_path):
    """A fail-closed gate does not infer a guarantee from a SILENCE.

    Reading an absent field as *"there was no blindness"* is exactly the form of fail-open
    this project has already paid for three times: absence of evidence turned into evidence.
    """
    report_path = tmp_path / "legacy.json"
    _write_report(report_path)
    stale = json.loads(report_path.read_text(encoding="utf-8"))
    del stale["fallback_measured"]
    report_path.write_text(json.dumps(stale), encoding="utf-8")

    passed, message = mma.check_release_gate(report_path, mma._REPO_ROOT, mma._AGENTS_DIR)

    assert not passed
    assert "fallback_measured" in message


def test_check_release_gate_accepts_a_fresh_green_report(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(report_path)

    passed, message = mma.check_release_gate(report_path, mma._REPO_ROOT, mma._AGENTS_DIR)

    assert passed is True
    assert "OK" in message


def test_check_release_gate_rejects_a_missing_report(tmp_path):
    passed, message = mma.check_release_gate(
        tmp_path / "does-not-exist.json", mma._REPO_ROOT, mma._AGENTS_DIR
    )

    assert passed is False
    assert "no marker-adherence report" in message


def test_check_release_gate_rejects_a_stale_git_sha(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(report_path, git_sha="0" * 40)

    passed, message = mma.check_release_gate(report_path, mma._REPO_ROOT, mma._AGENTS_DIR)

    assert passed is False
    assert "stale" in message
    assert "0" * 40 in message


def test_check_release_gate_rejects_a_stale_prompts_hash(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(report_path, prompts_sha256=hashlib.sha256(b"not the real prompts").hexdigest())

    passed, message = mma.check_release_gate(report_path, mma._REPO_ROOT, mma._AGENTS_DIR)

    assert passed is False
    assert "agents/*.md changed" in message


def test_check_release_gate_rejects_a_red_verdict(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(report_path, verdict="red")

    passed, message = mma.check_release_gate(report_path, mma._REPO_ROOT, mma._AGENTS_DIR)

    assert passed is False
    assert "not green" in message


def test_check_release_gate_rejects_malformed_json(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text("{not valid json", encoding="utf-8")

    passed, message = mma.check_release_gate(report_path, mma._REPO_ROOT, mma._AGENTS_DIR)

    assert passed is False
    assert "not valid JSON" in message


def test_installing_the_spy_twice_does_not_make_it_permanent():
    """MAGI gate (Balthasar, cycle 11): an instrument you cannot remove is one you cannot trust.

    A second ``install_spy()`` would leave ``extract`` already pointing at the spy, so
    ``uninstall_spy()`` would restore the spy over itself and the patch would survive the
    uninstall -- silently, for the rest of the process.
    """
    mma.install_spy()
    mma.install_spy()
    mma.uninstall_spy()

    assert VerdictSentinel.extract is mma._real_extract


def test_measuring_a_payload_with_a_HUGE_INTEGER_does_not_abort_the_measurement(tmp_path):
    """MAGI gate (Caspar, cycle 15): the same hole as RecursionError, wearing another name.

    ``json.loads`` raises a PLAIN ``ValueError`` -- not ``JSONDecodeError`` -- when a number
    exceeds ``int_max_str_digits`` (4300 by default). The production parser has always mapped
    that to ``JSONDecodeError`` so the mage keeps its retry; the instrument did not, so one
    pathological completion would take the whole release measurement down and leave no artifact
    at all. Catching only the exceptions you thought of is how an instrument you trust dies on
    the input you did not.
    """
    huge = "1" * 5000
    raw = _write_raw(tmp_path, "caspar", '<MAGI_VERDICT>\n{"n": ' + huge + "}\n</MAGI_VERDICT>")
    mma.install_spy()

    mma.measure_raw_file(raw, "caspar")  # must NOT raise

    assert mma.tally["caspar"]["invalid_json"] == 1


def test_a_hung_run_cannot_hang_the_measurement(monkeypatch, tmp_path):
    """MAGI gate (Balthasar, cycle 18): the parent had no guard against a hung child.

    ``--timeout`` bounds each MAGE, and the operator reads that and reasonably concludes the
    measurement is bounded. It was not: the parent ``subprocess.run`` had no timeout of its own,
    so a child that deadlocked -- before its own timeouts could fire, or while shutting down --
    would hang the release measurement forever, with no artifact and no error. A bound that only
    the child enforces is not a bound on the parent.

    The parent's bound is the child's own, plus a margin for its three mages and its shutdown --
    generous, because killing a healthy run would be worse than waiting for a sick one.
    """
    captured: dict = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)

        class _Done:
            returncode = 0

        return _Done()

    monkeypatch.setattr(mma.subprocess, "run", fake_run)

    mma.run_real_magi(
        "code-review", tmp_path / "bundle.md", ollama=True, timeout=900, output_dir=tmp_path
    )

    assert captured.get("timeout") is not None, "the parent must not wait forever on a child"
    assert captured["timeout"] > 900, "and its bound must exceed the child's own"


@pytest.mark.parametrize(
    "raw",
    [
        "just prose, no markers",
        "<MAGI_VERDICT>\n{",
        "<MAGI_VERDICT>\n{}\n</MAGI_VERDICT>\n<MAGI_VERDICT>\n{}\n</MAGI_VERDICT>",
        "<MAGI_VERDICT>\nnot json at all\n</MAGI_VERDICT>",
        '<MAGI_VERDICT>\n{"n": ' + "1" * 5000 + "}\n</MAGI_VERDICT>",
        '<MAGI_VERDICT>\n{"agent": "caspar"}\n</MAGI_VERDICT>',
    ],
    ids=["no-markers", "unterminated", "ambiguous", "bad-json", "huge-int", "clean"],
)
def test_the_spy_OBSERVES_and_never_decides(raw):
    """MAGI gate (Balthasar, cycle 19): the instrument was changing what it measured.

    ``extract`` RETURNS the block; the JSON inside it is decoded later, by the parser. The spy
    decoded it too -- and RAISED where the real method returns. So under measurement the parser
    took a different path than it takes in production: an instrument that perturbs its subject.
    Its own docstring promised *"this spy must never change the parser's observable behaviour,
    only observe it"*, which is the third time in this branch that a comment has asserted more
    than the code delivered.

    It tallies the content now instead of raising on it, and the parser goes on to fail the way
    it always would have.
    """
    real = VerdictSentinel.extract

    def outcome(fn):
        try:
            return ("returned", fn(VerdictSentinel(), raw))
        except BaseException as exc:  # noqa: BLE001 -- the TYPE is the thing under test
            return ("raised", type(exc).__name__)

    before = outcome(real)
    mma.install_spy()
    with mma.agent_context("caspar"):
        during = outcome(VerdictSentinel.extract)
    mma.uninstall_spy()

    assert before == during, "the spy must observe the parser, not change it"


def test_an_ollama_bundle_without_a_timeout_uses_the_plugins_default():
    # Scenario 14 / R13: 900 would derive ~125 s per attempt under 0.21.0.
    import magi_rs_trio

    assert mma.resolve_tool_timeout(None, ollama=True) == (
        magi_rs_trio.DEFAULT_OLLAMA_TIMEOUT_SECONDS
    )


def test_a_claude_bundle_without_a_timeout_keeps_900():
    assert mma.resolve_tool_timeout(None, ollama=False) == 900


@pytest.mark.parametrize("ollama", [True, False])
def test_an_explicit_timeout_wins_on_both_backends(ollama):
    assert mma.resolve_tool_timeout(4321, ollama=ollama) == 4321


def test_the_tool_reads_the_constant_instead_of_copying_it(monkeypatch):
    # Mutation check (MAGI r3): a copied 16820 in the tool fails here.
    import magi_rs_trio

    monkeypatch.setattr(magi_rs_trio, "DEFAULT_OLLAMA_TIMEOUT_SECONDS", 12345)
    assert mma.resolve_tool_timeout(None, ollama=True) == 12345


def test_main_passes_the_backend_timeout_to_each_run(monkeypatch, tmp_path):
    # The wiring (CP2 r3, Balthasar): resolve_tool_timeout alone is not enough if main()
    # still forwards args.timeout. main's loops call run_real_magi per bundle (line ~718).
    import magi_rs_trio

    seen = []
    monkeypatch.setattr(mma, "prepare_workspace", lambda scratch: tmp_path)
    monkeypatch.setattr(
        mma,
        "run_real_magi",
        lambda mode, bundle, *, ollama, timeout, output_dir, cwd=mma._REPO_ROOT: seen.append(
            (ollama, timeout)
        ),
    )
    monkeypatch.setattr(mma, "measure_output_dir", lambda run_dir: None)
    monkeypatch.setattr(mma, "build_artifact", lambda *a, **k: {"verdict": "green"})
    bundle = tmp_path / "b.md"
    bundle.write_text("x", encoding="utf-8")
    mma.main(
        [
            "measure",
            "--ollama-bundle",
            str(bundle),
            "--claude-bundle",
            str(bundle),
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert seen == [(True, magi_rs_trio.DEFAULT_OLLAMA_TIMEOUT_SECONDS), (False, 900)]


def test_the_parent_bound_exceeds_the_childs_worst_case_at_the_new_default(monkeypatch, tmp_path):
    # R13's fixed criterion: under --ollama the child is one magi-rs launch bounded by
    # the plugin's outer clock (T; M4 found no overshoot, so there is no margin).
    # Guard test: green before the change on purpose.
    import magi_rs_trio

    captured: dict = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)

        class _Done:
            returncode = 0

        return _Done()

    monkeypatch.setattr(mma.subprocess, "run", fake_run)
    timeout = magi_rs_trio.DEFAULT_OLLAMA_TIMEOUT_SECONDS
    bundle = tmp_path / "bundle.md"
    bundle.write_text("x", encoding="utf-8")
    mma.run_real_magi("code-review", bundle, ollama=True, timeout=timeout, output_dir=tmp_path)
    assert captured["timeout"] > timeout


# --- v8: analysis lens, isolated workspace, schema (REQ-15, BDD-21) -------------------


def test_the_default_mode_is_analysis_because_code_review_is_native_in_v8():
    assert mma.DEFAULT_MODE == "analysis"


def _record_runs(monkeypatch, tmp_path):
    workspace = tmp_path / "scratch-ws"
    workspace.mkdir()
    seen = []
    monkeypatch.setattr(mma, "prepare_workspace", lambda scratch: workspace)
    monkeypatch.setattr(
        mma,
        "run_real_magi",
        lambda mode, bundle, *, ollama, timeout, output_dir, cwd=mma._REPO_ROOT: seen.append(
            (mode, ollama, cwd)
        ),
    )
    monkeypatch.setattr(mma, "measure_output_dir", lambda run_dir: None)
    monkeypatch.setattr(mma, "build_artifact", lambda *a, **k: {"verdict": "green"})
    bundle = tmp_path / "b.md"
    bundle.write_text("x", encoding="utf-8")
    return workspace, seen, bundle


def test_an_ollama_run_executes_in_the_prepared_workspace_never_the_repo_root(
    monkeypatch, tmp_path
):
    workspace, seen, bundle = _record_runs(monkeypatch, tmp_path)
    mma.main(["measure", "--ollama-bundle", str(bundle), "--out", str(tmp_path / "out.json")])
    assert seen == [("analysis", True, workspace)]
    assert workspace.resolve() != mma._REPO_ROOT.resolve()


def test_a_claude_run_keeps_the_repo_root(monkeypatch, tmp_path):
    _, seen, bundle = _record_runs(monkeypatch, tmp_path)
    mma.main(["measure", "--claude-bundle", str(bundle), "--out", str(tmp_path / "out.json")])
    assert seen == [("analysis", False, mma._REPO_ROOT)]


def test_a_claude_only_measurement_never_prepares_a_workspace(monkeypatch, tmp_path):
    _, _, bundle = _record_runs(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(mma, "prepare_workspace", lambda scratch: calls.append(scratch))
    mma.main(["measure", "--claude-bundle", str(bundle), "--out", str(tmp_path / "out.json")])
    assert calls == []


class _FakePopen:
    """A ``subprocess.Popen`` stand-in for ``--ollama-init``.

    ``hangs``: the first ``communicate`` raises TimeoutExpired. ``immortal``: it never dies,
    so a bounded ``wait`` expires. Otherwise a hung child dies on ``kill()`` only.
    ``interrupt``: the first ``communicate`` raises KeyboardInterrupt.
    """

    instances: list = []

    def __init__(self, command, returncode=0, hangs=False, immortal=False, interrupt=False, **kw):
        self.command, self.kwargs, self.pid = command, kw, 4242
        self.returncode, self._hangs, self._immortal = returncode, hangs, immortal
        self._interrupt, self.killed = interrupt, False
        self.timeouts, self.wait_timeouts = [], []
        _FakePopen.instances.append(self)

    def communicate(self, timeout=None):
        self.timeouts.append(timeout)
        if self._interrupt and len(self.timeouts) == 1:
            raise KeyboardInterrupt
        if self._hangs and len(self.timeouts) == 1:
            raise mma.subprocess.TimeoutExpired(self.command, timeout)
        return None, None

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self._immortal or (self._hangs and not self.killed):
            raise mma.subprocess.TimeoutExpired(self.command, timeout)
        return self.returncode

    def kill(self):
        self.killed = True


def _fake_popen(monkeypatch, returncode=0, **flags):
    _FakePopen.instances = []
    monkeypatch.setattr(
        mma.subprocess,
        "Popen",
        lambda command, **kwargs: _FakePopen(command, returncode, **flags, **kwargs),
    )
    return _FakePopen.instances


def test_prepare_workspace_runs_ollama_init_inside_a_fresh_directory(monkeypatch, tmp_path):
    started = _fake_popen(monkeypatch)
    workspace = mma.prepare_workspace(tmp_path)
    assert workspace.is_dir() and workspace.parent == tmp_path
    assert "--ollama-init" in started[0].command
    assert Path(started[0].kwargs["cwd"]) == workspace


def test_prepare_workspace_fails_loudly_when_init_fails(monkeypatch, tmp_path):
    _fake_popen(monkeypatch, returncode=3)
    with pytest.raises(RuntimeError, match="3"):
        mma.prepare_workspace(tmp_path)


def test_the_workspace_is_removed_even_when_a_run_fails(monkeypatch, tmp_path):
    made = []

    def fake_prepare(scratch):
        workspace = scratch / "ws"
        workspace.mkdir()
        made.append(workspace)
        return workspace

    def boom(*a, **k):
        raise RuntimeError("died")

    monkeypatch.setattr(mma, "prepare_workspace", fake_prepare)
    monkeypatch.setattr(mma, "run_real_magi", boom)
    bundle = tmp_path / "b.md"
    bundle.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError):
        mma.main(["measure", "--ollama-bundle", str(bundle), "--out", str(tmp_path / "out.json")])
    assert made and not made[0].exists()


def test_run_real_magi_runs_the_child_in_the_given_cwd(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)

        class _Done:
            returncode = 0

        return _Done()

    monkeypatch.setattr(mma.subprocess, "run", fake_run)
    mma.run_real_magi(
        "analysis", tmp_path / "b.md", ollama=True, timeout=900, output_dir=tmp_path, cwd=tmp_path
    )
    assert Path(captured["cwd"]) == tmp_path


def test_run_real_magi_passes_a_relative_bundle_as_an_absolute_path(monkeypatch, tmp_path):
    # [CP2-M15] the child runs in the workspace; a relative bundle would not resolve there.
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command

        class _Done:
            returncode = 0

        return _Done()

    monkeypatch.setattr(mma.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    mma.run_real_magi(
        "analysis",
        Path("b.md"),
        ollama=True,
        timeout=900,
        output_dir=tmp_path,
        cwd=tmp_path / "ws",
    )
    assert any(Path(str(part)) == tmp_path / "b.md" for part in captured["command"])


def test_the_artifact_declares_the_v8_baseline_and_its_mode():
    _fill_tally_all_ok()
    artifact = mma.build_artifact(
        mma._REPO_ROOT, mma._AGENTS_DIR, {"ollama": 1, "claude": 0}, mode="analysis"
    )
    assert artifact["schema"] == "v8" and artifact["mode"] == "analysis"


def test_check_rejects_an_artifact_of_the_7x_series(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(report_path)
    legacy = json.loads(report_path.read_text(encoding="utf-8"))
    del legacy["schema"]
    report_path.write_text(json.dumps(legacy), encoding="utf-8")
    passed, message = mma.check_release_gate(report_path, mma._REPO_ROOT, mma._AGENTS_DIR)
    assert not passed and "schema" in message


# --- magi-rs per-attempt tally (Loop 1, C1) ------------------------------------------

_MAGI_RS_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "magi-rs-run-reports"
    / "extraction-failures.json"
)


def _copy_magi_rs_report(tmp_path: Path, mutate=None) -> Path:
    """Copy the real-derived magi-rs report fixture into *tmp_path* (never edit it on disk)."""
    report = json.loads(_MAGI_RS_FIXTURE.read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(report)
    path = tmp_path / mma.RUN_REPORT_FILENAME
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_magi_rs_marker_failures_are_folded_into_the_tally(tmp_path):
    """A seat that failed on magi-rs attempt 1 must NOT be scored clean (real corpus shape)."""
    path = _copy_magi_rs_report(tmp_path)

    mma.measure_run_report(path)

    assert mma.tally["melchior"]["missing_markers"] == 1
    assert mma.tally["balthasar"]["invalid_json"] == 1
    assert mma.tally["caspar"]["missing_markers"] == 0
    assert mma.tally["melchior"][mma.OK_TALLY_KEY] == 1


@pytest.mark.parametrize(
    ("magi_rs_cause", "tool_cause"),
    [
        ("missing-markers", "missing_markers"),
        ("unterminated", "unterminated_block"),
        ("ambiguous", "ambiguous_markers"),
        ("invalid-json", "invalid_json"),
        ("malformed-object", "schema"),
        ("schema", "schema"),
        ("echoed-example", "echoed_example"),
        ("agent-identity", "agent_identity"),
    ],
)
def test_every_magi_rs_cause_maps_into_the_tool_vocabulary(tmp_path, magi_rs_cause, tool_cause):
    """The closed mapping covers every label magi-core 4.2.0 serialises."""

    def mutate(report):
        report["magi_rs"]["extraction_failures"]["caspar"] = [
            {"attempt": 1, "cause": magi_rs_cause, "model": "m"}
        ]

    mma.measure_run_report(_copy_magi_rs_report(tmp_path, mutate))

    assert mma.tally["caspar"][tool_cause] == 1
    assert tool_cause in mma.CAUSES


def test_an_unknown_magi_rs_cause_is_rejected_not_dropped(tmp_path):
    """An unmapped cause raises one line naming it: dropping it would score the seat clean."""

    def mutate(report):
        report["magi_rs"]["extraction_failures"]["caspar"] = [
            {"attempt": 1, "cause": "brand-new-cause", "model": "m"}
        ]

    with pytest.raises(ValueError, match="brand-new-cause"):
        mma.measure_run_report(_copy_magi_rs_report(tmp_path, mutate))


def test_an_ollama_measurement_refuses_a_mode_other_than_analysis(monkeypatch, tmp_path, capsys):
    # Loop 1 m4: the v8 baseline is `analysis` (DIR-6); another mode measures another lens.
    _, seen, bundle = _record_runs(monkeypatch, tmp_path)
    code = mma.main(
        [
            "measure",
            "--ollama-bundle",
            str(bundle),
            "--mode",
            "design",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert code == 1 and seen == []
    err = capsys.readouterr().err
    assert "analysis" in err and len(err.strip().splitlines()) == 1


def test_check_rejects_an_artifact_whose_mode_is_not_analysis(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(report_path, mode="design")

    passed, message = mma.check_release_gate(report_path, mma._REPO_ROOT, mma._AGENTS_DIR)

    assert not passed and "mode" in message and "analysis" in message


# --- MAGI run 1 on MS2 -------------------------------------------------------------


def test_every_mapped_cause_is_a_column_of_the_artifact():
    # A renamed CAUSES label must not let a mapped failure be tallied and then dropped.
    assert set(mma.MAGI_RS_CAUSE_MAP.values()) <= set(mma.CAUSES)


def test_the_map_covers_every_cause_of_the_real_derived_fixture():
    report = json.loads(_MAGI_RS_FIXTURE.read_text(encoding="utf-8"))
    found = {
        record["cause"]
        for records in report["magi_rs"]["extraction_failures"].values()
        for record in records
    }
    assert found and found <= set(mma.MAGI_RS_CAUSE_MAP)


def test_a_tallied_cause_outside_the_artifact_columns_raises(monkeypatch):
    _fill_tally_all_ok()
    mma.tally["caspar"]["renamed_cause"] += 1
    with pytest.raises(RuntimeError, match="renamed_cause"):
        mma.build_artifact(
            mma._REPO_ROOT, mma._AGENTS_DIR, {"ollama": 1, "claude": 0}, mode="analysis"
        )


def test_prepare_workspace_turns_a_timeout_into_one_line_naming_it(monkeypatch, tmp_path):
    started = _fake_popen(monkeypatch, hangs=True)
    monkeypatch.setattr(mma, "windows_kill_tree", lambda pid: None)
    with pytest.raises(RuntimeError, match=str(mma.OLLAMA_INIT_TIMEOUT_SECONDS)) as exc:
        mma.prepare_workspace(tmp_path)
    assert started[0].timeouts[0] == mma.OLLAMA_INIT_TIMEOUT_SECONDS
    assert "\n" not in str(exc.value)


def test_a_timed_out_init_has_its_whole_tree_killed_on_windows(monkeypatch, tmp_path):
    # windows_kill_tree is best-effort: it is a no-op here and the child only dies on kill().
    started = _fake_popen(monkeypatch, hangs=True)
    killed = []
    monkeypatch.setattr(mma, "windows_kill_tree", killed.append)
    monkeypatch.setattr(mma.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match=str(mma.OLLAMA_INIT_TIMEOUT_SECONDS)):
        mma.prepare_workspace(tmp_path)
    assert killed == [started[0].pid] and started[0].killed is True
    assert started[0].wait_timeouts == [mma.PROC_WAIT_REAP_TIMEOUT]


def test_a_timed_out_init_is_killed_directly_off_windows(monkeypatch, tmp_path):
    started = _fake_popen(monkeypatch, hangs=True)
    _patch_posix_group_kill(monkeypatch)  # never a real killpg on the fake pid
    with pytest.raises(RuntimeError):
        mma.prepare_workspace(tmp_path)
    assert started[0].killed is True


def test_an_init_that_survives_the_kill_warns_and_still_raises_without_hanging(
    monkeypatch, tmp_path, capsys
):
    started = _fake_popen(monkeypatch, hangs=True, immortal=True)
    monkeypatch.setattr(mma, "windows_kill_tree", lambda pid: None)
    with pytest.raises(RuntimeError, match=str(mma.OLLAMA_INIT_TIMEOUT_SECONDS)) as exc:
        mma.prepare_workspace(tmp_path)
    assert "\n" not in str(exc.value)
    assert started[0].wait_timeouts == [mma.PROC_WAIT_REAP_TIMEOUT]
    err = capsys.readouterr().err
    assert "WARNING" in err and "4242" in err


def test_an_interrupt_kills_the_init_tree_and_propagates(monkeypatch, tmp_path):
    started = _fake_popen(monkeypatch, interrupt=True)
    killed = []
    monkeypatch.setattr(mma, "windows_kill_tree", killed.append)
    monkeypatch.setattr(mma.sys, "platform", "win32")
    with pytest.raises(KeyboardInterrupt):
        mma.prepare_workspace(tmp_path)
    assert killed == [started[0].pid] and started[0].killed is True


# --- MAGI run 2 wave ---------------------------------------------------------------


def test_a_magi_rs_report_without_extraction_failures_is_a_measurement_error(tmp_path):
    # G2: absence must not read as "no failures"; a seat would be scored clean on silence.
    def mutate(report):
        del report["magi_rs"]["extraction_failures"]

    path = _copy_magi_rs_report(tmp_path, mutate)
    with pytest.raises(RuntimeError, match="extraction_failures") as exc:
        mma.measure_run_report(path)
    assert str(path) in str(exc.value) and "\n" not in str(exc.value)


def test_a_report_without_a_magi_rs_block_is_still_measured(tmp_path):
    path = _write_run_report(tmp_path, agents=mma.AGENT_NAMES)
    mma.measure_run_report(path)
    assert mma.tally["caspar"][mma.OK_TALLY_KEY] == 1


def _patch_posix_group_kill(monkeypatch):
    """Platform patched to linux; ``os.killpg``/``signal.SIGKILL`` recorded (absent on Windows)."""
    groups = []
    monkeypatch.setattr(mma.sys, "platform", "linux")
    monkeypatch.setattr(mma.os, "killpg", lambda pid, sig: groups.append((pid, sig)), raising=False)
    monkeypatch.setattr(mma.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(mma, "windows_kill_tree", lambda pid: pytest.fail("not on POSIX"))
    return groups


def test_a_timed_out_init_has_its_process_group_killed_on_posix(monkeypatch, tmp_path):
    # G1: the magi-rs init grandchild must not survive the parent.
    started = _fake_popen(monkeypatch, hangs=True)
    groups = _patch_posix_group_kill(monkeypatch)
    with pytest.raises(RuntimeError):
        mma.prepare_workspace(tmp_path)
    assert groups == [(started[0].pid, 9)] and started[0].killed is True


def test_an_interrupt_kills_the_process_group_on_posix(monkeypatch, tmp_path):
    started = _fake_popen(monkeypatch, interrupt=True)
    groups = _patch_posix_group_kill(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        mma.prepare_workspace(tmp_path)
    assert groups == [(started[0].pid, 9)]


def test_a_missing_process_group_is_tolerated(monkeypatch, tmp_path):
    started = _fake_popen(monkeypatch, hangs=True)
    _patch_posix_group_kill(monkeypatch)

    def gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(mma.os, "killpg", gone, raising=False)
    with pytest.raises(RuntimeError, match=str(mma.OLLAMA_INIT_TIMEOUT_SECONDS)):
        mma.prepare_workspace(tmp_path)
    assert started[0].killed is True


def test_the_init_starts_its_own_session_on_posix_only(monkeypatch, tmp_path):
    started = _fake_popen(monkeypatch)
    monkeypatch.setattr(mma.sys, "platform", "linux")
    mma.prepare_workspace(tmp_path)
    assert started[0].kwargs.get("start_new_session") is True
    started = _fake_popen(monkeypatch)
    monkeypatch.setattr(mma.sys, "platform", "win32")
    mma.prepare_workspace(tmp_path)
    assert started[0].kwargs.get("start_new_session") is False


@pytest.mark.parametrize("value", [None, [], "x", 3])
def test_a_non_dict_extraction_failures_is_a_measurement_error(tmp_path, value):
    # G2 re-review: a missing tally reads as null in a real report, and null must not score clean.
    def mutate(report):
        report["magi_rs"]["extraction_failures"] = value

    with pytest.raises(RuntimeError, match="extraction_failures"):
        mma.measure_run_report(_copy_magi_rs_report(tmp_path, mutate))


def test_a_real_report_built_from_an_envelope_without_the_tally_is_rejected(tmp_path):
    # Produced through the plugin's own build_report_block, as a real report would be.
    import run_magi
    from magi_rs_result import ConsultResult

    fixture = _MAGI_RS_FIXTURE.parent.parent / "magi-rs-consult" / "three-agents.json"
    envelope = json.loads(fixture.read_text(encoding="utf-8"))
    envelope["consult"].pop("extraction_failures", None)
    result = ConsultResult.from_json(json.dumps(envelope))
    report = {
        "agents": [{"agent": name} for name in mma.AGENT_NAMES],
        **run_magi.build_report_block(result, timeout_seconds=1),
    }
    assert report["magi_rs"]["extraction_failures"] is None
    path = tmp_path / mma.RUN_REPORT_FILENAME
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="extraction_failures"):
        mma.measure_run_report(path)


def test_empty_tallies_stay_a_legitimate_clean_run(tmp_path):
    def mutate(report):
        report["magi_rs"]["extraction_failures"] = {"caspar": [], "melchior": [], "balthasar": []}

    mma.measure_run_report(_copy_magi_rs_report(tmp_path, mutate))
    assert mma.tally["caspar"][mma.OK_TALLY_KEY] == 1
    mma.reset_tally()

    def mutate_empty(report):
        report["magi_rs"]["extraction_failures"] = {}

    mma.measure_run_report(_copy_magi_rs_report(tmp_path, mutate_empty))
    assert mma.tally["caspar"][mma.OK_TALLY_KEY] == 1


def test_the_real_derived_fixture_still_measures(tmp_path):
    mma.measure_run_report(_copy_magi_rs_report(tmp_path))
    assert mma.tally["melchior"]["missing_markers"] == 1
