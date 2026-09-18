# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-09-18
"""Tests for the read-only magi-rs release pin (MS2 Task 1)."""

import json
import os

import pytest

from magi_rs_pin import Pin, PinFormatError, UnsupportedPlatformError

_SHIPPED_PIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "skills",
    "magi",
    "magi-rs.pin.json",
)


def test_the_shipped_pin_file_loads_and_names_the_pinned_version():
    assert Pin.load(_SHIPPED_PIN).version == "0.19.1"


@pytest.mark.parametrize("field", ["archive_sha256", "executable_sha256"])
def test_every_asset_in_the_shipped_pin_carries_two_64_hex_digests(field):
    for asset in Pin.load(_SHIPPED_PIN).assets.values():
        digest = getattr(asset, field)
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)


def test_windows_amd64_selects_the_zip():
    asset = Pin.load(_SHIPPED_PIN).asset_for("Windows", "AMD64")
    assert asset.name.endswith("windows-x86_64.zip")
    assert asset.archive_kind == "zip"


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Linux", "x86_64", "linux-x86_64.tar.gz"),
        ("Linux", "aarch64", "linux-aarch64-rpi5.tar.gz"),
        ("Darwin", "arm64", "macos-universal2.tar.gz"),
        ("Darwin", "x86_64", "macos-universal2.tar.gz"),
    ],
)
def test_each_supported_platform_selects_its_own_tarball(system, machine, expected):
    asset = Pin.load(_SHIPPED_PIN).asset_for(system, machine)
    assert asset.name.endswith(expected)
    assert asset.archive_kind == "tar.gz"


def test_an_unsupported_platform_raises_naming_it_and_never_falls_back():
    with pytest.raises(UnsupportedPlatformError) as exc:
        Pin.load(_SHIPPED_PIN).asset_for("Linux", "riscv64")
    message = str(exc.value)
    assert "Linux" in message and "riscv64" in message


def test_the_pin_errors_do_not_inherit_from_validationerror():
    # Locked derogation: a fail-closed event must not be swallowed by the
    # orchestrator's `except (ValidationError, json.JSONDecodeError)`.
    from synthesize import ValidationError

    assert not issubclass(UnsupportedPlatformError, ValidationError)
    assert not issubclass(PinFormatError, ValidationError)
    assert not issubclass(PinFormatError, json.JSONDecodeError)


def test_the_download_url_points_at_the_pinned_release_asset():
    # Untested until now, and a defect here does not fail loudly -- it fails as
    # a 404 that reads like "the release is gone".
    pin = Pin.load(_SHIPPED_PIN)
    asset = pin.asset_for("Windows", "AMD64")
    url = pin.download_url(asset)
    assert url == (
        f"https://github.com/BolivarTech/magi/releases/download/v{pin.version}/{asset.name}"
    )


def test_a_pin_missing_a_required_field_is_rejected_at_load_time(tmp_path):
    broken = tmp_path / "pin.json"
    broken.write_text(json.dumps({"version": "0.19.1"}), encoding="utf-8")
    with pytest.raises(PinFormatError):
        Pin.load(str(broken))


@pytest.mark.parametrize(
    "document",
    [
        ["not", "an", "object"],
        {"version": "0.19.1", "repository": "r", "assets": "not an object"},
        {"version": "0.19.1", "repository": "r", "assets": {"windows-x86_64": "not an object"}},
        {"version": "0.19.1", "repository": "r", "assets": {"windows-x86_64": {"name": "x"}}},
    ],
)
def test_a_pin_of_the_wrong_shape_is_rejected_at_load_time(tmp_path, document):
    broken = tmp_path / "pin.json"
    broken.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PinFormatError):
        Pin.load(str(broken))


def test_a_pin_that_is_not_json_is_rejected_without_leaking_the_decoder_error(tmp_path):
    broken = tmp_path / "pin.json"
    broken.write_text("{ not json", encoding="utf-8")
    with pytest.raises(PinFormatError):
        Pin.load(str(broken))
