"""read_served_models reports the model that ACTUALLY served each agent.

Every Claude model ID in ``models`` is a floating alias, so a report that only
records what was *requested* attributes nothing once the alias moves. The
``claude -p --output-format json`` envelope carries the served identity in
``modelUsage``; these tests pin that read, and pin that it never guesses.
"""

import json

import pytest
from served_model import read_served_models


def _write_envelope(directory, agent: str, payload) -> None:
    path = directory / f"{agent}.raw.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_reads_the_served_model_from_model_usage(tmp_path):
    _write_envelope(
        tmp_path,
        "caspar",
        {
            "result": "{}",
            "modelUsage": {
                "claude-opus-5": {
                    "outputTokens": 10488,
                    "canonicalModel": "claude-opus-5",
                    "provider": "firstParty",
                }
            },
        },
    )

    assert read_served_models(str(tmp_path), ["caspar"]) == {"caspar": ["claude-opus-5"]}


def test_canonical_model_wins_over_the_key(tmp_path):
    """The key is how the CLI labelled the request; ``canonicalModel`` is the identity.

    When they differ, the alias has been resolved to something more specific --
    which is the entire reason this module exists.
    """
    _write_envelope(
        tmp_path,
        "melchior",
        {"modelUsage": {"claude-opus-5": {"canonicalModel": "claude-opus-5-20260715"}}},
    )

    assert read_served_models(str(tmp_path), ["melchior"]) == {
        "melchior": ["claude-opus-5-20260715"]
    }


def test_two_served_models_are_both_recorded_never_collapsed(tmp_path):
    """A server-side fallback mid-turn bills two models. Record both.

    Picking "the real one" would mean choosing among candidates on a heuristic
    -- the exact failure shape this project already paid for in the verdict
    parser. Two models served the run; the artifact says so.
    """
    _write_envelope(
        tmp_path,
        "balthasar",
        {
            "modelUsage": {
                "claude-opus-5": {"canonicalModel": "claude-opus-5", "outputTokens": 10},
                "claude-opus-4-8": {"canonicalModel": "claude-opus-4-8", "outputTokens": 9000},
            }
        },
    )

    assert read_served_models(str(tmp_path), ["balthasar"]) == {
        "balthasar": ["claude-opus-4-8", "claude-opus-5"]
    }, "sorted, both present -- no most-tokens heuristic"


def test_entry_without_canonical_model_falls_back_to_the_key(tmp_path):
    _write_envelope(tmp_path, "caspar", {"modelUsage": {"claude-haiku-4-5": {"outputTokens": 4}}})

    assert read_served_models(str(tmp_path), ["caspar"]) == {"caspar": ["claude-haiku-4-5"]}


@pytest.mark.parametrize(
    "payload,reason",
    [
        ({"result": "{}"}, "no modelUsage field at all (an Ollama envelope)"),
        ({"modelUsage": {}}, "modelUsage present but empty"),
        ({"modelUsage": "claude-opus-5"}, "modelUsage is not a mapping"),
        ([1, 2, 3], "the envelope is not an object"),
    ],
)
def test_unusable_envelope_omits_the_agent(tmp_path, payload, reason):
    """Fail-soft: attribution is telemetry, and telemetry never kills a verdict.

    An omitted agent says "not known"; a fabricated one would say something
    false about which model judged the input.
    """
    _write_envelope(tmp_path, "caspar", payload)

    assert read_served_models(str(tmp_path), ["caspar"]) == {}, reason


def test_missing_envelope_omits_the_agent(tmp_path):
    """A mage that died before writing its envelope has no served model to read."""
    assert read_served_models(str(tmp_path), ["caspar", "melchior"]) == {}


def test_corrupt_json_omits_the_agent(tmp_path):
    (tmp_path / "caspar.raw.json").write_text("{not json", encoding="utf-8")

    assert read_served_models(str(tmp_path), ["caspar"]) == {}


def test_one_unusable_agent_does_not_hide_the_others(tmp_path):
    _write_envelope(tmp_path, "caspar", {"modelUsage": {"claude-opus-5": {}}})
    (tmp_path / "melchior.raw.json").write_text("{not json", encoding="utf-8")

    assert read_served_models(str(tmp_path), ["caspar", "melchior", "balthasar"]) == {
        "caspar": ["claude-opus-5"]
    }
