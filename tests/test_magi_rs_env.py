# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-09-18
"""Tests for the allowlisted child environment of magi-rs (MS3 Task 1)."""

import pytest

from magi_rs_env import EnvironmentPolicy


def test_the_built_environment_holds_NOTHING_beyond_the_allowlist():
    # THE test of this task, and the one the first draft was missing. Asserting
    # `"MAGI_PASSPHRASE" not in built` passes for an implementation that copies
    # the WHOLE parent environment minus that one name -- a denylist wearing an
    # allowlist's costume, and a leak for every variable nobody thought of.
    # Set EQUALITY is what makes the guarantee real.
    parent = {
        "PATH": "/usr/bin",
        "MAGI_PASSPHRASE": "hunter2",
        "ANTHROPIC_API_KEY": "sk-x",
        "SOME_FUTURE_SECRET": "y",
        "AWS_SECRET_ACCESS_KEY": "z",
    }
    assert set(EnvironmentPolicy.build(parent)) == {"PATH"}


def test_an_undeclared_LC_variable_does_NOT_travel():
    # THE test this enumeration was missing, and the gate was right to call it
    # critical: the plan REPLACED a `LC_` prefix rule with an explicit list
    # precisely because a prefix rule is the wildcard hole it refuses for
    # `OLLAMA_` -- and then shipped no test for it. An implementer who writes
    # `name.startswith("LC_")` because it is shorter passes every other test in
    # this file. This one fails.
    built = EnvironmentPolicy.build(
        {"LC_ALL": "C", "LC_PAPER": "x", "LC_TELEPHONE": "y", "PATH": "/usr/bin"}
    )
    assert built["LC_ALL"] == "C"  # declared -> inherited
    assert "LC_PAPER" not in built  # not declared -> does not travel
    assert "LC_TELEPHONE" not in built


def test_the_passphrase_never_reaches_the_child():
    # R7/BDD-8a: the plugin does not read MAGI_PASSPHRASE -- but the CHILD does,
    # in its first statement. An operator who exported it in their shell would
    # open the vault without meaning to, and the same command would take a
    # different execution path depending on who launched it.
    built = EnvironmentPolicy.build({"MAGI_PASSPHRASE": "hunter2", "PATH": "/usr/bin"})
    assert "MAGI_PASSPHRASE" not in built


@pytest.mark.parametrize(
    "name",
    [
        "PATH",
        "HOME",
        "USERPROFILE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "SYSTEMROOT",
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "LC_NUMERIC",
        "LC_TIME",
        "LC_COLLATE",
    ],
)
def test_every_platform_variable_is_inherited(name):
    # PATH last-wins in a dict literal would clobber the parametrized entry
    # when name == "PATH"; the declared variable goes last on purpose.
    built = EnvironmentPolicy.build({"PATH": "/usr/bin", name: "value"})
    assert built[name] == "value"


@pytest.mark.parametrize(
    "name",
    ["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"],
)
def test_proxy_variables_are_inherited_in_both_cases(name):
    built = EnvironmentPolicy.build({name: "http://proxy:8080", "PATH": "/usr/bin"})
    assert built[name] == "http://proxy:8080"


@pytest.mark.parametrize(
    "name",
    [
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
    ],
)
def test_every_ca_variable_is_inherited(name):
    # The allowlist's own risk is coming up SHORT, and this is the case that
    # bites: behind a corporate proxy with its own CA, omitting these breaks
    # ALL HTTPS with a certificate error that looks nothing like a missing
    # environment variable.
    built = EnvironmentPolicy.build({name: "/etc/ca.pem", "PATH": "/usr/bin"})
    assert built[name] == "/etc/ca.pem"


def test_the_ollama_host_is_inherited_but_no_other_ollama_variable_is():
    # Enumerated, NOT a wildcard: OLLAMA_* would silently re-open the door
    # every time upstream adds a variable, which is the denylist failure mode
    # wearing an allowlist costume.
    built = EnvironmentPolicy.build(
        {"OLLAMA_HOST": "http://nas:11434", "OLLAMA_API_KEY": "secret", "PATH": "/usr/bin"}
    )
    assert built["OLLAMA_HOST"] == "http://nas:11434"
    assert "OLLAMA_API_KEY" not in built


def test_the_dynamic_linker_variables_are_refused_on_purpose():
    # A code-injection vector: they load arbitrary shared objects into a binary
    # we JUST verified by digest, voiding R4's guarantee.
    built = EnvironmentPolicy.build(
        {
            "LD_PRELOAD": "/tmp/evil.so",
            "LD_LIBRARY_PATH": "/tmp",
            "DYLD_LIBRARY_PATH": "/tmp",
            "PATH": "/usr/bin",
        }
    )
    assert "LD_PRELOAD" not in built
    assert "LD_LIBRARY_PATH" not in built
    assert "DYLD_LIBRARY_PATH" not in built


def test_the_withheld_names_are_listed_so_a_child_failure_is_diagnosable():
    # R7: an allowlist's failure mode is a certificate or host error that looks
    # nothing like "a variable was missing". The policy therefore names what it
    # withheld -- names only, never values -- for the error message to carry.
    parent = {"PATH": "/usr/bin", "CORP_CA_HINT": "/etc/corp.pem", "MAGI_PASSPHRASE": "hunter2"}
    withheld = EnvironmentPolicy.withheld(parent)
    assert withheld == ["CORP_CA_HINT", "MAGI_PASSPHRASE"]
    assert "hunter2" not in " ".join(withheld)
