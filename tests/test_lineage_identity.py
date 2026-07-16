# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-16
"""Tests for the pure architecture-to-vendor lineage identity guard (Task 4).

Covers ``LineageIdentityGuard.family_verdict`` (fail-open on unmapped/None
architecture, ok/contradiction on mapped ones) and
``LineageIdentityGuard.digest_collision`` (pairwise digest comparison, ``None``
treated as non-comparable), plus a consistency check that the map's vendor
vocabulary matches the TOML ``lineage`` strings declared in
``ollama_config.DEFAULT_MODELS``.
"""

from hypothesis import given
from hypothesis import strategies as st

from lineage_identity import LineageIdentityGuard


def test_family_ok_when_vendor_matches():
    assert LineageIdentityGuard().family_verdict("deepseek4", "deepseek") == "ok"


def test_family_contradiction_when_vendor_differs():
    assert LineageIdentityGuard().family_verdict("deepseek4", "acme") == "contradiction"


def test_family_unknown_fails_open_for_unmapped_architecture():
    assert LineageIdentityGuard().family_verdict("some-selfhosted-arch", "acme") == "unknown"


def test_family_unknown_for_ambiguous_base_architecture():
    assert LineageIdentityGuard().family_verdict("llama", "acme") == "unknown"


def test_family_unknown_for_none_architecture():
    assert LineageIdentityGuard().family_verdict(None, "acme") == "unknown"


def test_digest_collision_detects_same_digest():
    assert LineageIdentityGuard().digest_collision(["sha256:a", "sha256:b", "sha256:a"]) == (0, 2)


def test_digest_collision_none_when_all_distinct():
    assert LineageIdentityGuard().digest_collision(["sha256:a", "sha256:b", "sha256:c"]) is None


def test_digest_collision_ignores_none_as_non_comparable():
    assert LineageIdentityGuard().digest_collision([None, "sha256:a", None]) is None


def test_map_vendors_match_declared_lineage_vocabulary():
    # A map vendor for a trio model's architecture MUST equal that model's declared
    # lineage, else family_verdict would raise a FALSE "contradiction".
    from ollama_config import DEFAULT_MODELS

    g = LineageIdentityGuard()
    arch_of = {
        "qwen3.5:397b-cloud": "qwen3.5",
        "kimi-k2.6:cloud": "kimi-k2",
        "deepseek-v4-pro:cloud": "deepseek4",
    }  # test-local: the spike-confirmed trio arches
    for spec in DEFAULT_MODELS.values():
        arch = arch_of.get(spec.model)
        if arch and arch in g.ARCHITECTURE_VENDOR:
            assert g.ARCHITECTURE_VENDOR[arch] == spec.lineage


@given(arch=st.text(min_size=1), lineage=st.text(min_size=1))
def test_property_unmapped_architecture_never_contradicts(arch, lineage):
    g = LineageIdentityGuard()
    if arch not in g.ARCHITECTURE_VENDOR:
        assert g.family_verdict(arch, lineage) == "unknown"
