# Author: Julian Bolivar
# Version: 8.0.0
# Date: 2026-11-21

"""Checks that the published documentation describes the v8 contract.

The documents under test ship with every installation, so these tests pin the
flags they mention, the commands they must cover and the items the open-weight
guide has to carry.
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DOCS = tuple(
    _ROOT / p
    for p in (
        "README.md",
        "docs/ollama-backend.md",
        "skills/magi/SKILL.md",
        "docs/MAGI-System-Documentation.md",
    )
)
_FLAG = re.compile(r"(--ollama[-a-z]*|--commit|--from|--to|--base|--head|--uncommitted|--yes)\b")


def _published_flags() -> set[str]:
    """Return every v8 flag the published documents mention."""
    flags: set[str] = set()
    for doc in _DOCS:
        flags |= set(_FLAG.findall(doc.read_text(encoding="utf-8")))
    return flags


def test_every_flag_the_docs_mention_exists() -> None:
    known = {
        "--ollama",
        "--ollama-init",
        "--ollama-vault-set",
        "--ollama-rotate-key",
        "--yes",
        "--commit",
        "--from",
        "--to",
        "--base",
        "--head",
        "--uncommitted",
    }
    assert _published_flags() <= known


def test_the_docs_mention_every_new_command() -> None:
    new = {"--ollama-vault-set", "--ollama-rotate-key", "--uncommitted"}
    assert new <= _published_flags()


def test_the_changelog_marks_the_breaking_input_change() -> None:
    text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [8.0.0]" in text
    entry = text.split("## [8.0.0]", 1)[1].split("\n## [", 1)[0]
    assert "Breaking" in entry and "code-review --ollama" in entry


def test_no_published_doc_tells_the_user_to_pass_no_memory() -> None:
    for doc in _DOCS:
        assert "--no-memory" not in doc.read_text(encoding="utf-8")


_GUIDE = _ROOT / "docs" / "ollama-backend.md"
_MANDATORY = (
    ".magi/.secret/magi.key",  # where the key lives
    "plain text",  # what the key protects and what it does not
    "no recovery",  # a lost key
    "embedder",  # prerequisite of code-review --ollama
    "ollama.com",  # egress of clef:cloud
    "clef-flash",  # defaults written by --ollama-init
    "nimble",
    "7.x",  # regenerating a 7.x .magi/
    "adr/",  # gate framing lives only in user-facing docs/ and adr/
)


def test_the_ollama_guide_carries_every_mandatory_item() -> None:
    text = _GUIDE.read_text(encoding="utf-8")
    assert [anchor for anchor in _MANDATORY if anchor not in text] == []


def test_the_docs_state_that_consult_uses_the_key_only_for_the_vault() -> None:
    assert "only to open the vault" in _GUIDE.read_text(encoding="utf-8")
