# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""What `pip install missile-defense` promises, checked without installing it.

The wheel is built by CI and by anyone who runs `pip install .`, and the ways it
goes quietly wrong are all statements in ``pyproject.toml`` that no longer match
the tree: an entry point naming a function that has been renamed, or a shim that
imports the very thing it exists to check for.

Both are cheap to assert from the source, and neither needs a compiler.
"""

from __future__ import annotations

import ast
import importlib
import tomllib
from pathlib import Path

import md

ROOT = Path(md.__file__).parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_every_console_script_points_at_something_that_exists() -> None:
    """A renamed function turns into a `pip install` that produces a dead command."""
    scripts = PYPROJECT["project"]["scripts"]
    assert set(scripts) == {"md-train", "md-console", "md-multiseed"}
    for name, target in scripts.items():
        module_name, _, attribute = target.partition(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute, None)), f"{name} -> {target}"


def test_the_trainer_shim_does_not_import_what_it_checks_for() -> None:
    """`md.cli` exists to *explain* a missing torch, so it must load without one.

    Importing torch at module level would make the explanation unreachable: the
    traceback it is meant to replace would happen first.
    """
    source = (ROOT / "python" / "md" / "cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:  # module level only — the lazy import inside is the point
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("torch") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("torch")


def test_neither_heavy_half_is_a_hard_dependency() -> None:
    """docs/PACKAGING.md's central promise, as a test rather than a paragraph.

    torch would drag a multi-gigabyte vendor wheel behind the environment, and
    PySide6 is LGPLv3 where this project is MIT. Both belong in extras.
    """
    required = " ".join(PYPROJECT["project"]["dependencies"]).lower()
    assert "torch" not in required
    assert "pyside6" not in required

    extras = PYPROJECT["project"]["optional-dependencies"]
    assert any("torch" in item for item in extras["train"])
    assert any("PySide6" in item for item in extras["console"])
    assert any("nvidia-ml-py" in item for item in extras["console"])
    assert any(item.startswith("amdsmi;") for item in extras["console"])


def test_the_wheel_builds_only_the_headless_half() -> None:
    """A NumPy array must not cost a Vulkan SDK — the game ships as a package."""
    defines = PYPROJECT["tool"]["scikit-build"]["cmake"]["define"]
    assert defines["MD_BUILD_APP"] == "OFF"
    assert defines["MD_BUILD_BINDINGS"] == "ON"


def test_the_extension_is_installed_rather_than_left_in_the_build_tree() -> None:
    """The blocker docs/PACKAGING.md named: no install() rule, nothing to package."""
    cmake = (ROOT / "bindings" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "install(TARGETS _md_native" in cmake
    assert "MD_PYTHON_INSTALL_DIR" in cmake


# ---- the two products --------------------------------------------------------
# The game and the training console are one source package and two different
# products, and the negative half of that matters more than the positive one: the
# promise is that installing the game brings no Python with it. These read the
# packaging declarations, which is where that promise is actually kept or broken.


def _debian_stanzas() -> dict[str, dict[str, str]]:
    """`debian/control` as {package name: {field: value}}.

    A hand-rolled parser rather than python-debian, which is not a dependency of
    this project and would be a heavy one to add for six assertions. Continuation
    lines are folded into their field; that is all these tests need.
    """
    text = (ROOT / "debian" / "control").read_text(encoding="utf-8")
    stanzas: dict[str, dict[str, str]] = {}
    current: dict[str, str] = {}
    field = ""
    for line in text.splitlines():
        if not line.strip():
            current, field = {}, ""
            continue
        if line[0].isspace():  # a continuation of the previous field
            if field:
                current[field] += " " + line.strip()
            continue
        name, _, value = line.partition(":")
        field = name.strip().lower()
        current[field] = value.strip()
        if field == "package":
            stanzas[value.strip()] = current
    return stanzas


def test_debian_builds_three_binary_packages_from_one_source() -> None:
    """The game, the Python half, and the console — separately installable."""
    assert set(_debian_stanzas()) == {
        "missile-defense",
        "python3-md",
        "missile-defense-training",
    }


def test_the_game_package_pulls_in_no_python_at_all() -> None:
    """The central promise, as a test rather than a paragraph.

    Someone who wants the game must not be made to install an interpreter, torch,
    or an LGPL Qt binding for Python. If this ever fails, the two products have
    become one.
    """
    game = _debian_stanzas()["missile-defense"]
    relations = " ".join(
        game.get(field, "") for field in ("depends", "recommends", "suggests", "pre-depends")
    ).lower()
    for forbidden in ("python3-md", "python3:any", "pyside", "torch"):
        assert forbidden not in relations, f"the game package relates to {forbidden}"


def test_the_console_package_carries_the_dependencies_the_game_refuses() -> None:
    """And it is the console that depends on the game, never the other way round."""
    console = _debian_stanzas()["missile-defense-training"]
    relations = f"{console.get('depends', '')} {console.get('recommends', '')}".lower()
    assert "python3-md" in relations, "the console does not depend on the Python half"
    assert "pyside6" in relations, "the console does not depend on PySide6"


def test_each_package_installs_a_disjoint_set_of_paths() -> None:
    """Two products cannot both own a file, and dpkg refuses if they try."""
    manifests = {
        name: set((ROOT / "debian" / f"{name}.install").read_text(encoding="utf-8").split())
        for name in ("python3-md", "missile-defense-training")
    }
    assert not manifests["python3-md"] & manifests["missile-defense-training"]
    # The console's entry point and its menu entry, which are what make it
    # reachable from an install rather than only from a checkout.
    console = " ".join(manifests["missile-defense-training"])
    assert "md-console" in console
    assert "missile-defense-training.desktop" in console


def test_the_console_has_a_desktop_entry_of_its_own() -> None:
    """A separate product gets a separate launcher, or it is not discoverable."""
    entry = (ROOT / "packaging" / "missile-defense-training.desktop").read_text(encoding="utf-8")
    assert "Exec=md-console" in entry
    # Not a game: it belongs in the development/science menu, not beside the
    # arcade cabinet, or someone looking for the game finds two of them.
    categories = next(line for line in entry.splitlines() if line.startswith("Categories="))
    assert "Game" not in categories
