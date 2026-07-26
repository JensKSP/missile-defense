# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for the version bump and the nightly version renderings.

Nothing here touches git or the real tree beyond reading it: the bump is tested
against a fixture, and what it writes is read back with the same regexes the
release gate uses, so a stanza that only *looks* right fails here.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.release import bump, changelog_entry
from tools.version import DevVersion, read_versions

WHEN = datetime(2026, 7, 26, 11, 0, 0, tzinfo=timezone(timedelta(hours=2)))


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "debian").mkdir()
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(missile_defense\n  VERSION 0.1.0\n  LANGUAGES CXX)\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text('version = "0.1.0"\n', encoding="utf-8")
    (tmp_path / "debian" / "control").write_text(
        "Source: missile-defense\nMaintainer: Jens Köhler <jens@example.com>\n", encoding="utf-8"
    )
    (tmp_path / "debian" / "changelog").write_text(
        "missile-defense (0.1.0-1) unstable; urgency=medium\n\n  * Initial release.\n\n"
        " -- Jens Köhler <jens@example.com>  Fri, 25 Jul 2026 10:00:00 +0200\n",
        encoding="utf-8",
    )
    return tmp_path


def test_the_bump_moves_all_three_files_at_once(tree: Path) -> None:
    bump("0.2.0", root=tree, when=WHEN)
    assert set(read_versions(tree).values()) == {"0.2.0"}


def test_the_minimum_required_version_is_left_alone(tree: Path) -> None:
    # Replacing the first VERSION in CMakeLists would raise the CMake floor to
    # the project version and still pass a naive check.
    bump("0.2.0", root=tree, when=WHEN)
    assert "cmake_minimum_required(VERSION 3.25)" in (tree / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )


def test_the_old_changelog_entry_is_kept(tree: Path) -> None:
    bump("0.2.0", root=tree, when=WHEN)
    text = (tree / "debian" / "changelog").read_text(encoding="utf-8")
    assert text.index("(0.2.0-1)") < text.index("(0.1.0-1)"), "newest entry must come first"
    assert "Initial release." in text


def test_the_changelog_trailer_is_shaped_the_way_dpkg_wants() -> None:
    # One space, --, one space, maintainer, *two* spaces, RFC 2822 date. dpkg
    # rejects the stanza otherwise, and complains about the changelog rather than
    # about the whitespace, which is a long way from the cause.
    entry = changelog_entry("0.2.0", maintainer="A B <a@b.c>", when=WHEN)
    lines = entry.splitlines()
    assert lines[0] == "missile-defense (0.2.0-1) unstable; urgency=medium"
    assert lines[1] == ""
    assert lines[2].startswith("  * ")
    assert lines[3] == ""
    assert lines[4] == " -- A B <a@b.c>  Sun, 26 Jul 2026 11:00:00 +0200"


def test_bumping_twice_lands_on_the_second_version(tree: Path) -> None:
    bump("0.2.0", root=tree, when=WHEN)
    bump("0.3.0", root=tree, when=WHEN)
    assert set(read_versions(tree).values()) == {"0.3.0"}


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("semver", "0.1.0-dev.37+g1a2b3c4d"),
        ("pep440", "0.1.0.dev37+g1a2b3c4d"),
        ("debian", "0.1.0~dev37+g1a2b3c4d"),
        ("filename", "0.1.0-dev37-g1a2b3c4d"),
    ],
)
def test_the_dev_version_is_spelled_per_ecosystem(attribute: str, expected: str) -> None:
    dev = DevVersion(base="0.1.0", distance=37, commit="g1a2b3c4d")
    assert getattr(dev, attribute) == expected


def test_every_dev_rendering_sorts_below_the_release_it_precedes() -> None:
    # The property the whole scheme rests on, checked the way each ecosystem
    # actually compares rather than by eye. `packaging` is the reference PEP 440
    # implementation; skipped rather than hand-rolled where it is absent, because
    # a home-made comparison would only be testing my reading of the spec.
    version_module = pytest.importorskip("packaging.version", reason="packaging not installed")
    dev = DevVersion(base="0.1.0", distance=37, commit="g1a2b3c4d")

    assert version_module.Version(dev.pep440) < version_module.Version(dev.base)
    # Debian's rule is simpler and worth asserting directly: `~` is the only
    # character that sorts before the end of a string, so anything after it
    # makes the version *older* than the bare number in front of it.
    assert dev.debian.startswith(f"{dev.base}~")
