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
    assert set(scripts) == {"md-train", "md-console"}
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
