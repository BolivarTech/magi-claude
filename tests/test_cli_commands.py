import pytest

import run_magi
from cli_commands import INIT_FLAG, ROTATE_KEY_FLAG, VAULT_SET_FLAG, YES_FLAG, command
from magi_key import REGENERATION_GUIDANCE
from magi_rs_preflight import ProjectPreflight
from magi_rs_trio import _VAULT_HINT


def test_a_command_is_named_one_way_everywhere():
    # [X8] `run_magi.py <flag>`: the one form every operator-facing message uses.
    assert command(INIT_FLAG) == "run_magi.py --ollama-init"
    assert command(VAULT_SET_FLAG, "<NAME>") == "run_magi.py --ollama-vault-set <NAME>"
    assert command(ROTATE_KEY_FLAG) == "run_magi.py --ollama-rotate-key"


def test_the_regeneration_guidance_names_its_commands_in_that_form():
    assert "run_magi.py --ollama-init" in REGENERATION_GUIDANCE
    assert "run_magi.py --ollama-vault-set <NAME>" in REGENERATION_GUIDANCE


def test_the_vault_hint_names_the_command_in_that_form():
    assert "run_magi.py --ollama-vault-set <NAME>" in _VAULT_HINT


def test_an_uninitialised_project_names_the_init_command_in_that_form(tmp_path):
    with pytest.raises(Exception) as exc:
        ProjectPreflight.require_initialised(str(tmp_path))
    assert "run_magi.py --ollama-init" in str(exc.value)


def test_every_flag_constant_is_accepted_by_the_parser():
    # the constants and argparse cannot drift: the parser is built from the same literals.
    assert run_magi.parse_args([INIT_FLAG]).ollama_init is True
    assert (
        run_magi.parse_args([VAULT_SET_FLAG, "OPENAI_API_KEY"]).ollama_vault_set == "OPENAI_API_KEY"
    )
    rotate = run_magi.parse_args([ROTATE_KEY_FLAG, YES_FLAG])
    assert rotate.ollama_rotate_key is True and rotate.yes is True


def test_the_dest_constants_are_the_attributes_the_parser_sets():
    # [M8] the dispatcher reads args.ollama_*; the constants pin those names explicitly.
    from cli_commands import INIT_DEST, ROTATE_KEY_DEST, VAULT_SET_DEST

    parsed = run_magi.parse_args([INIT_FLAG])
    assert getattr(parsed, INIT_DEST) is True
    assert getattr(run_magi.parse_args([VAULT_SET_FLAG, "OPENAI_API_KEY"]), VAULT_SET_DEST)
    assert getattr(run_magi.parse_args([ROTATE_KEY_FLAG]), ROTATE_KEY_DEST) is True
