# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-13
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


def test_spy_tallies_invalid_json_inside_markers_and_reraises():
    """Markers present, but the body between them is not valid JSON (R7 content drift)."""
    mma.install_spy()
    sentinel = VerdictSentinel()

    with mma.agent_context("balthasar"):
        with pytest.raises(json.JSONDecodeError):
            sentinel.extract("<MAGI_VERDICT>\nnot valid json at all\n</MAGI_VERDICT>")

    assert mma.tally["balthasar"]["invalid_json"] == 1
    # The marker layer itself succeeded -- only the content tally fires.
    assert "missing_markers" not in mma.tally["balthasar"]


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
    """El fallo que el instrumento existe para contar, y que NO podia ver.

    ``launch_agent`` reescribe ``{agent}.raw.json`` en CADA intento, asi que el archivo
    solo guarda el ULTIMO. Un mago que omite las marcas y acierta en el reintento dejaba
    un raw impecable -> el artefacto salia **verde** con una omision real dentro. Con
    ``max_attempts=2`` y una tasa real del 5 %, el artefacto habria reportado ~0.25 %:
    sistematicamente optimista, justo en el numero del que cuelga el criterio de exito.
    """
    for agent in mma.AGENT_NAMES:  # el raw del ULTIMO intento: limpio, del retry que acerto
        _write_raw(tmp_path, agent, f'<MAGI_VERDICT>\n{{"agent": "{agent}"}}\n</MAGI_VERDICT>')
    _write_run_report(
        tmp_path,
        agents=mma.AGENT_NAMES,
        extraction_failures={"caspar": {"missing_markers": 1}},
    )
    mma.install_spy()

    mma.measure_output_dir(tmp_path)

    assert mma.tally["caspar"]["missing_markers"] == 1, "la omision del 1er intento SE VE"
    assert mma.tally["caspar"][mma.OK_TALLY_KEY] == 1, "y el reintento que acerto tambien"
    artifact = mma.build_artifact(mma._REPO_ROOT, mma._AGENTS_DIR, {"ollama": 1, "claude": 0})
    assert artifact["verdict"] == "red", "un run con una omision NO es un run verde"


def test_the_report_carries_the_causes_the_spy_STRUCTURALLY_cannot_see(tmp_path):
    """``EchoedExampleRejected`` y ``AgentIdentityError`` se lanzan DESPUES del parser.

    El spy envuelve ``VerdictSentinel.extract``, asi que esos dos nunca pasan por el: un
    artefacto construido solo con el spy certificaba 4 de las 6 causas. El contador del
    run las tiene todas -- porque las clasifica donde se deciden.
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

    artifact = mma.build_artifact(mma._REPO_ROOT, mma._AGENTS_DIR, {"ollama": 1, "claude": 0})
    assert artifact["per_seat"]["caspar"]["echoed_example"] == 1
    assert artifact["per_seat"]["melchior"]["agent_identity"] == 2
    assert artifact["verdict"] == "red"


def test_a_run_that_died_without_a_report_falls_back_to_the_raw_files(tmp_path):
    """Un run que muere bajo el suelo de 2 magos no escribe reporte.

    Los raws que alcanzo a producir siguen siendo muestras validas: se miden con el
    parser real, como antes. Lo que NUNCA se hace es fabricar un cero.
    """
    _write_raw(tmp_path, "caspar", "el modelo se olvido de las marcas")
    mma.install_spy()

    mma.measure_output_dir(tmp_path)

    assert mma.tally["caspar"]["missing_markers"] == 1
    assert "melchior" not in mma.tally


def test_a_run_measured_by_the_BLIND_fallback_can_NEVER_certify_green(tmp_path):
    """El camino ciego no puede firmar el release -- ni aunque no vea ni un fallo.

    El fallback re-parsea los raws, que solo guardan el ULTIMO intento de cada mago: es
    **exactamente** el instrumento optimista que este gate acaba de eliminar. Peor: el
    ``ok`` del fallback lo firma el sentinel + ``json.loads``, sin canario ni identidad,
    asi que cuenta como bueno un veredicto que el run real habria **rechazado**.

    El WARNING a stderr no basta: ``make release-check`` corre ``check``, que lee **solo el
    artefacto** -- y para entonces el warning ya no existe. Lo que no queda escrito en el
    artefacto, no gobierna nada.
    """
    for agent in mma.AGENT_NAMES:  # tres raws impecables, ningun reporte
        _write_raw(tmp_path, agent, f'<MAGI_VERDICT>\n{{"agent": "{agent}"}}\n</MAGI_VERDICT>')
    mma.install_spy()

    mma.measure_output_dir(tmp_path)
    artifact = mma.build_artifact(mma._REPO_ROOT, mma._AGENTS_DIR, {"ollama": 1, "claude": 0})

    assert artifact["fallback_measured"] == 1, "el artefacto DICE que midio a ciegas"
    assert artifact["verdict"] != "green", "una medicion ciega no certifica nada"


def test_check_rejects_an_artifact_that_was_measured_blind(tmp_path):
    """Y el gate lo rechaza leyendo el artefacto, que es lo unico que ve."""
    _write_report(tmp_path / "r.json", fallback_measured=1)

    passed, message = mma.check_release_gate(tmp_path / "r.json", mma._REPO_ROOT, mma._AGENTS_DIR)

    assert not passed
    assert "blind" in message.lower() or "fallback" in message.lower()


def test_every_cause_of_the_retry_contract_has_a_column_in_the_artifact():
    """El vocabulario de causas es UNO: el de ``FEEDBACK_TEMPLATES`` (R12).

    Si el artefacto enumerase sus propias causas, una causa nueva en el contrato de
    reintento no tendria columna -- y un fallo real se contaria como cero. Que es
    exactamente la clase de ceguera que este ciclo esta arreglando.
    """
    from retry_feedback import FEEDBACK_TEMPLATES

    for agent in mma.AGENT_NAMES:
        mma.tally[agent][mma.OK_TALLY_KEY] = 1

    artifact = mma.build_artifact(mma._REPO_ROOT, mma._AGENTS_DIR, {"ollama": 1, "claude": 0})

    for agent in mma.AGENT_NAMES:
        assert set(artifact["per_seat"][agent]) == {mma.OK_TALLY_KEY, *FEEDBACK_TEMPLATES}


# --- build_artifact ------------------------------------------------------------------


def _fill_tally_all_ok() -> None:
    for agent in mma.AGENT_NAMES:
        mma.tally[agent][mma.OK_TALLY_KEY] = 3


def test_build_artifact_includes_git_sha_and_prompts_sha256():
    _fill_tally_all_ok()

    artifact = mma.build_artifact(mma._REPO_ROOT, mma._AGENTS_DIR, {"ollama": 1, "claude": 1})

    assert artifact["git_sha"] == mma._git_head_sha(mma._REPO_ROOT)
    assert artifact["prompts_sha256"] == mma._prompts_sha256(mma._AGENTS_DIR)
    assert len(artifact["prompts_sha256"]) == 64  # hex sha256
    assert artifact["verdict"] == "green"
    assert artifact["runs"] == {"ollama": 1, "claude": 1}


def test_build_artifact_verdict_red_when_any_failure_present():
    _fill_tally_all_ok()
    mma.tally["caspar"]["missing_markers"] = 1

    artifact = mma.build_artifact(mma._REPO_ROOT, mma._AGENTS_DIR, {"ollama": 1, "claude": 0})

    assert artifact["verdict"] == "red"
    assert artifact["per_seat"]["caspar"]["missing_markers"] == 1


def test_build_artifact_raises_when_a_seat_has_zero_data():
    mma.tally["caspar"][mma.OK_TALLY_KEY] = 3
    mma.tally["melchior"][mma.OK_TALLY_KEY] = 3
    # balthasar never appears at all.

    with pytest.raises(RuntimeError, match="balthasar"):
        mma.build_artifact(mma._REPO_ROOT, mma._AGENTS_DIR, {"ollama": 1, "claude": 0})


# --- check_release_gate --------------------------------------------------------------


def _write_report(path: Path, **overrides) -> None:
    report = {
        "git_sha": mma._git_head_sha(mma._REPO_ROOT),
        "prompts_sha256": mma._prompts_sha256(mma._AGENTS_DIR),
        "measured_at": "2026-07-13T00:00:00Z",
        "runs": {"ollama": 5, "claude": 2},
        "fallback_measured": 0,
        "per_seat": {a: {"ok": 5, "missing_markers": 0} for a in mma.AGENT_NAMES},
        "verdict": "green",
    }
    report.update(overrides)
    path.write_text(json.dumps(report), encoding="utf-8")


def test_check_rejects_an_artifact_with_no_blindness_field_at_all(tmp_path):
    """Un gate fail-closed no infiere una garantia de un SILENCIO.

    Leer un campo ausente como *"no hubo ceguera"* es exactamente la forma de fail-open que
    este proyecto ya pago tres veces: la ausencia de evidencia convertida en evidencia.
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
