# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-13
"""Suite for the start-up guard of the prompt contract (R9, MS2).

It covers what the anchoring test **cannot see**: the **user's installation**. The
anchoring test runs in the developer's repo; the **stale-copy** bug (``mklink /D``
degrading to a copy on Windows) produces old prompts with a new parser **on the user's
machine**, where no test ever reaches.
"""

import json

import pytest

from prompt_guard import AgentPromptGuard, PromptContractError
from validate import ValidationError
from verdict_markers import VERDICT_CLOSE, VERDICT_OPEN, VerdictSentinel

GOOD = f"prose\n{VERDICT_OPEN}\n{{ ...your 7-key JSON object... }}\n{VERDICT_CLOSE}\n"


def _agents(tmp_path, **overrides):
    """Create a healthy prompt directory, with whatever overrides are asked for."""
    directory = tmp_path / "agents"
    directory.mkdir()
    for name in ("melchior", "balthasar", "caspar"):
        (directory / f"{name}.md").write_text(overrides.get(name, GOOD), encoding="utf-8")
    return directory


class TestErrorIsNotRetryable:
    def test_PromptContractError_is_a_SIBLING_of_ValidationError_not_a_child(self):
        """``[CRITICAL]`` from Checkpoint 2: if it inherited, **the retry would eat it**.

        The orchestrator's retry guard catches ``(ValidationError, JSONDecodeError)``. A
        stale prompt **is not fixed by retrying** -- the file does not change by calling
        the model again. It is a **fail-closed** event: abort.

        It is exactly the case of the locked derogation in ``CLAUDE.local.md`` section 0.2
        (precedent: ``InvalidInputError``). **Rule: inherit from ValidationError if the
        retry fixes it; from Exception if it does not.**
        """
        assert issubclass(PromptContractError, Exception)
        assert not issubclass(PromptContractError, ValidationError)


class TestAgentPromptGuard:
    def test_a_healthy_prompt_set_passes(self, tmp_path):
        AgentPromptGuard(_agents(tmp_path), VerdictSentinel()).check()

    def test_a_stale_prompt_without_markers_is_FATAL(self, tmp_path):
        directory = _agents(tmp_path, caspar="Respond with ONLY a JSON object.\n")
        with pytest.raises(PromptContractError, match="caspar.md"):
            AgentPromptGuard(directory, VerdictSentinel()).check()

    def test_two_marker_pairs_are_FATAL(self, tmp_path):
        """The user documents the format twice -> the model sees **two examples**."""
        directory = _agents(tmp_path, caspar=GOOD + GOOD)
        with pytest.raises(PromptContractError, match="2 open"):
            AgentPromptGuard(directory, VerdictSentinel()).check()

    def test_a_VALID_verdict_between_the_markers_is_FATAL(self, tmp_path):
        """The LAST fabrication path, and it lives where no test of ours reaches.

        A user "improves" the prompt by putting a complete example **between the markers**
        and **reinstates variant 1 on THEIR machine**: the model copies that block, produces
        exactly ONE delimited block, it validates... and it fabricates. Neither the canary
        (it is not the shipped example) nor the anchoring test (it runs in OUR repo) sees it.
        """
        verdict = json.dumps(
            {
                "agent": "caspar",
                "verdict": "approve",
                "confidence": 0.85,
                "summary": "s",
                "reasoning": "r",
                "findings": [],
                "recommendation": "x",
            }
        )
        directory = _agents(tmp_path, caspar=f"{VERDICT_OPEN}\n{verdict}\n{VERDICT_CLOSE}\n")
        with pytest.raises(PromptContractError, match="valid verdict"):
            AgentPromptGuard(directory, VerdictSentinel()).check()

    def test_a_harmless_placeholder_that_happens_to_be_json_PASSES(self, tmp_path):
        """``{}`` is valid JSON and **cannot fabricate anything** (it lacks the 7 keys).

        Aborting over it would punish the user for something harmless. The right question is
        not *"is this valid JSON?"* but *"would this, copied, be accepted as a verdict?"*.
        """
        directory = _agents(tmp_path, caspar=f"{VERDICT_OPEN}\n{{}}\n{VERDICT_CLOSE}\n")
        AgentPromptGuard(directory, VerdictSentinel()).check()

    def test_a_file_with_BOM_does_NOT_trigger_a_false_FATAL(self, tmp_path):
        """The BOM is resolved in the **encoding layer** (``utf-8-sig``), not by relaxing
        the predicate. That is why the guard can be STRICT without false FATALs."""
        directory = _agents(tmp_path)
        (directory / "caspar.md").write_text(GOOD, encoding="utf-8-sig")
        AgentPromptGuard(directory, VerdictSentinel()).check()

    def test_an_invisible_inside_OUR_marker_is_corruption_and_is_FATAL(self, tmp_path):
        """The reverse of the permissive one: the MODEL's output with that invisible IS
        accepted.

        *Different trust domains, different strictness.*
        """
        corrupted = GOOD.replace(VERDICT_OPEN, "<MAGI​_VERDICT>")
        directory = _agents(tmp_path, caspar=corrupted)
        with pytest.raises(PromptContractError):
            AgentPromptGuard(directory, VerdictSentinel()).check()

    def test_a_missing_file_is_FATAL(self, tmp_path):
        directory = _agents(tmp_path)
        (directory / "caspar.md").unlink()
        with pytest.raises(PromptContractError, match="caspar.md"):
            AgentPromptGuard(directory, VerdictSentinel()).check()

    def test_the_SHIPPED_prompts_pass_the_guard(self):
        """The guard runs against the **real** prompts, not just against fixtures."""
        from pathlib import Path

        agents = Path(__file__).parent.parent / "skills" / "magi" / "agents"
        AgentPromptGuard(agents, VerdictSentinel()).check()


class TestTheGuardIsACTUALLYWired:
    """The most important guard of MS2 was **implemented, tested and documented**...
    **and it was never called**. Dead code.

    No test in the suite would have caught it: the guard's unit tests passed (they exercised
    the class), and the orchestrator's did too (they never invoked it). It was found by a
    **documentation audit**, on noticing that the FAQ described a guard that did not run.

    The lesson, and the reason this test exists: **"the class works" and "the system uses
    it" are two different claims, and only the second one protects anybody.**
    """

    def test_run_magi_main_invokes_the_guard_before_spending_a_token(self):
        import inspect

        import run_magi

        source = inspect.getsource(run_magi.main)
        assert "AgentPromptGuard" in source, "the guard is NOT invoked from main(): it is dead code"
        assert ".check()" in source

    def test_the_guard_runs_BEFORE_the_backend_is_selected(self):
        """It must abort **before spending a token**, not after the preflight."""
        import inspect

        import run_magi

        source = inspect.getsource(run_magi.main)
        assert source.index("AgentPromptGuard") < source.index("select_backend")

    def test_PromptContractError_is_caught_and_exits_nonzero(self):
        import inspect

        import run_magi

        source = inspect.getsource(run_magi.main)
        assert "except PromptContractError" in source
        assert "sys.exit(1)" in source


class TestCheckPromptsDryRun:
    """MAGI gate (Balthasar, cycles 4-7): customising a prompt should not be a coin flip.

    The guard is strict on purpose -- a prompt with a fabricable verdict between the markers
    reintroduces fabrication in the user's own install -- but until now the only way to learn
    that your edit was rejected was to START A RUN and have it abort. ``--check-prompts``
    validates a prompt directory and exits: cheap, offline, and it costs no tokens.
    """

    def test_a_good_agents_dir_passes_and_exits_zero(self, tmp_path, capsys):
        """The shipped prompts, which are what the fixture seeds, must pass."""
        import run_magi

        with pytest.raises(SystemExit) as exc:
            run_magi.check_prompts(str(tmp_path))

        assert exc.value.code == 0
        assert "OK" in capsys.readouterr().out

    def test_a_dangerous_prompt_is_REPORTED_and_exits_nonzero(self, tmp_path, capsys):
        """A verdict between the markers is exactly what the guard exists to refuse."""
        import json as _json

        import run_magi

        verdict = _json.dumps(
            {
                "agent": "caspar",
                "verdict": "approve",
                "confidence": 0.9,
                "summary": "s",
                "reasoning": "r",
                "findings": [],
                "recommendation": "rec",
            }
        )
        (tmp_path / "caspar.md").write_text(
            f"## Output format\n<MAGI_VERDICT>\n{verdict}\n</MAGI_VERDICT>\n", encoding="utf-8"
        )

        with pytest.raises(SystemExit) as exc:
            run_magi.check_prompts(str(tmp_path))

        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "caspar.md" in err
        assert "faq-prompt-guard" in err, "a FATAL that aborts owes the reader somewhere to go"

    def test_the_ABBREVIATION_reaches_the_dry_run(self):
        """MAGI gate (Balthasar, cycle 8): the first version screened raw ``sys.argv``.

        That duplicated the flag name as a magic string and bypassed argparse's own
        abbreviation handling: ``--check`` is a valid abbreviation of ``--check-prompts`` (no
        other ``--check*`` option exists), so argparse would accept it, expand it -- and the
        raw scan would miss it, silently giving the user a normal run they never asked for.
        The flag now flows through argparse, which is what knows about abbreviations.
        """
        import run_magi

        args = run_magi.parse_args(["--check"])

        assert args.check_prompts is True
        assert args.mode is None, "a dry run needs neither a mode nor an input"

    def test_a_normal_run_still_REQUIRES_mode_and_input(self):
        """Making the positionals optional at parse time must not make them optional."""
        import run_magi

        with pytest.raises(SystemExit):
            run_magi.parse_args([])


class TestTheMarkerCountMessageKnowsItsAudience:
    """MAGI gate (Balthasar, cycle 9): one message for two audiences misdirects one of them."""

    def test_ZERO_markers_reads_as_a_stale_install(self, tmp_path):
        """Prompts that predate the sentinel: reinstalling IS the fix."""
        from prompt_guard import AgentPromptGuard, PromptContractError
        from verdict_markers import VerdictSentinel

        (tmp_path / "caspar.md").write_text("an old prompt, no markers", encoding="utf-8")

        with pytest.raises(PromptContractError, match="Reinstall the plugin"):
            AgentPromptGuard(tmp_path, VerdictSentinel()).check()

    def test_TOO_MANY_markers_reads_as_a_customization(self, tmp_path):
        """Someone who documented the format twice. Telling THEM to reinstall would throw

        their work away -- and would not even fix it.
        """
        from prompt_guard import AgentPromptGuard, PromptContractError
        from verdict_markers import VerdictSentinel

        twice = (
            "## Output format\n<MAGI_VERDICT>\n{ ...placeholder... }\n</MAGI_VERDICT>\n\n"
            "For example:\n<MAGI_VERDICT>\n{ ...placeholder... }\n</MAGI_VERDICT>\n"
        )
        (tmp_path / "caspar.md").write_text(twice, encoding="utf-8")

        with pytest.raises(PromptContractError) as exc:
            AgentPromptGuard(tmp_path, VerdictSentinel()).check()

        message = str(exc.value)
        assert "customised" in message
        assert "Reinstall" not in message, "do not tell a customizer to throw their work away"
