# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-23
"""Stable, structured identity for MAGI findings.

Ports panóptico's title-independent finding ID: a finding's identity is a
SHA-256 of its normalized file path, line, and category — never its
(LLM-generated, run-to-run-unstable) title. Enables cross-agent merge and
idempotent dedup. Pure stdlib.
"""

from __future__ import annotations

import hashlib

#: Controlled finding-category vocabulary (panóptico's 16 slugs). Unknown
#: values fail soft to ``other`` (see :func:`normalize_category`).
CATEGORY_SLUGS: tuple[str, ...] = (
    "buffer-overflow",
    "null-deref",
    "resource-leak",
    "unvalidated-input",
    "race-condition",
    "error-handling",
    "hardcoded-secret",
    "integer-overflow",
    "injection",
    "logic-error",
    "type-mismatch",
    "deprecated-api",
    "performance",
    "style",
    "documentation",
    "other",
)
_CATEGORY_SET = frozenset(CATEGORY_SLUGS)
DEFAULT_CATEGORY = "other"
_FINDING_ID_HEX_LEN = 16


def normalize_path(path: str) -> str:
    """Canonicalize *path* for stable identity.

    Backslashes -> slashes, strip leading ``./`` / ``.\\``, collapse repeated
    slashes. Pure string transform so the same physical path always hashes
    identically regardless of OS separator or redundant prefixes.
    """
    p = path.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    while "//" in p:
        p = p.replace("//", "/")
    return p


def normalize_category(value: str | None) -> str:
    """Map *value* to a known :data:`CATEGORY_SLUGS` member, else ``other``.

    Fail-soft (mirrors panóptico's ``#[serde(other)]``): a missing or
    LLM-invented category never breaks a finding — it degrades to ``other``.
    """
    if not isinstance(value, str):
        return DEFAULT_CATEGORY
    slug = value.strip().lower().replace("_", "-").replace(" ", "-")
    return slug if slug in _CATEGORY_SET else DEFAULT_CATEGORY


def generate_finding_id(file: str, line: int, category: str) -> str:
    """Return ``SHA-256(normalize(file):line:normalize_category(category))[:16]``.

    Title-independent by construction, so the id is stable across runs even
    when the LLM rewords the title.
    """
    payload = f"{normalize_path(file)}:{int(line)}:{normalize_category(category)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_FINDING_ID_HEX_LEN]
