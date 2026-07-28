# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for the pre-release version check.

The first test is the one with teeth: it asserts the *real* tree agrees with
itself, so bumping the version in two of the three files and forgetting the third
fails here rather than at the top of a release.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.version import check, read_versions


def test_this_repository_declares_one_version() -> None:
    versions = read_versions()
    assert len(set(versions.values())) == 1, versions


def _tree(root: Path, cmake: str, py: str, changelog: str, package: str | None = None) -> Path:
    (root / "debian").mkdir(parents=True)
    (root / "python" / "md").mkdir(parents=True)
    (root / "CMakeLists.txt").write_text(
        f"project(missile_defense\n  VERSION {cmake}\n  LANGUAGES CXX)\n", encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(f'version = "{py}"\n', encoding="utf-8")
    (root / "debian" / "changelog").write_text(
        f"missile-defense ({changelog}-1) unstable; urgency=medium\n", encoding="utf-8"
    )
    # Defaults to agreeing with pyproject, since they describe the same package;
    # the tests that care about it being left behind pass it explicitly.
    (root / "python" / "md" / "__init__.py").write_text(
        f'__version__ = "{py if package is None else package}"\n', encoding="utf-8"
    )
    return root


def test_agreement_passes(tmp_path: Path) -> None:
    _tree(tmp_path, "1.2.3", "1.2.3", "1.2.3")
    assert check(root=tmp_path) == 0


def test_one_file_left_behind_is_caught(tmp_path: Path) -> None:
    # The realistic mistake: CMakeLists and pyproject bumped, changelog forgotten.
    _tree(tmp_path, "1.2.3", "1.2.3", "1.2.2")
    assert check(root=tmp_path) == 1


def test_the_packages_own_version_left_behind_is_caught(tmp_path: Path) -> None:
    # `md.__version__` is the one nothing builds from, so a stale value survives
    # every build and every test and is discovered by a user reading the training
    # trainer's About box. It has to be checked here or it is not checked at all.
    _tree(tmp_path, "1.2.3", "1.2.3", "1.2.3", package="1.2.2")
    assert check(root=tmp_path) == 1


def test_a_tag_that_does_not_match_the_tree_is_caught(tmp_path: Path) -> None:
    # Every file agrees, so nothing is wrong with the tree — the tag is wrong,
    # and pushing it would publish artifacts named after the other version.
    _tree(tmp_path, "1.2.3", "1.2.3", "1.2.3")
    assert check("v9.9.9", root=tmp_path) == 1
    assert check("v1.2.3", root=tmp_path) == 0


def test_the_leading_v_is_optional(tmp_path: Path) -> None:
    _tree(tmp_path, "1.2.3", "1.2.3", "1.2.3")
    assert check("1.2.3", root=tmp_path) == 0


def test_cmakes_minimum_required_is_not_mistaken_for_the_version(tmp_path: Path) -> None:
    # `cmake_minimum_required(VERSION 3.25)` is the first VERSION in the file and
    # is not the project's. Matching it would report 3.25 and agree with nothing.
    _tree(tmp_path, "1.2.3", "1.2.3", "1.2.3")
    cmakelists = tmp_path / "CMakeLists.txt"
    cmakelists.write_text(
        "cmake_minimum_required(VERSION 3.25)\n" + cmakelists.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert read_versions(tmp_path)["CMakeLists.txt"] == "1.2.3"


def test_a_missing_file_says_so(tmp_path: Path) -> None:
    _tree(tmp_path, "1.2.3", "1.2.3", "1.2.3")
    (tmp_path / "pyproject.toml").unlink()
    with pytest.raises(SystemExit, match="pyproject.toml"):
        read_versions(tmp_path)
