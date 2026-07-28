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
import sys
import tomllib
from pathlib import Path

import missile_defense
import pytest

ROOT = Path(missile_defense.__file__).parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_every_console_script_points_at_something_that_exists() -> None:
    """A renamed function turns into a `pip install` that produces a dead command."""
    scripts = PYPROJECT["project"]["scripts"]
    assert set(scripts) == {"missile-defense-train", "missile-defense-trainer"}
    for name, target in scripts.items():
        module_name, _, attribute = target.partition(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute, None)), f"{name} -> {target}"


def test_the_trainer_shim_does_not_import_what_it_checks_for() -> None:
    """The CLI exists to *explain* a missing torch, so it must load without one.

    That is `missile_defense.training.cli`.

    Importing torch at module level would make the explanation unreachable: the
    traceback it is meant to replace would happen first.
    """
    source = (ROOT / "python" / "missile_defense" / "training" / "cli.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    for node in tree.body:  # module level only — the lazy import inside is the point
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("torch") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("torch")


def test_the_advice_for_a_missing_torch_is_advice_that_works_here() -> None:
    """A distribution interpreter refuses `pip install`, so it must not be told to.

    This is the message someone sees when they skipped the setup step, which
    makes it the one place a wrong sentence costs the most. Both machines are
    described here rather than whichever one the tests happen to run on.
    """
    from missile_defense.training.cli import DEBIAN_README, explain_missing

    recipe = "python3 -m venv --system-site-packages"

    managed = explain_missing("torch", "torch", "Training", managed=True)
    assert "missile-defense-trainer" in managed, (
        "the packaged answer is the trainer's own installer"
    )
    assert recipe in managed
    assert DEBIAN_README in managed
    assert f"{sys.executable} -m pip install" not in managed, (
        "an externally managed interpreter was told to install into itself"
    )

    unmanaged = explain_missing("torch", "torch", "Training", managed=False)
    assert f"{sys.executable} -m pip install torch" in unmanaged
    assert recipe not in unmanaged


def test_the_debian_readme_the_message_points_at_is_shipped() -> None:
    """It was cited for months and never written; debhelper installs it by name.

    `debian/README.Debian` would land in the *first* binary package — the game,
    which has no Python in it at all. The per-package spelling is what puts it
    where the command that names it can be run.
    """
    readme = ROOT / "debian" / "missile-defense-trainer.README.Debian"
    assert readme.is_file(), "missile-defense-train points at a file the package does not ship"
    text = readme.read_text(encoding="utf-8")
    assert "--system-site-packages" in text, "the venv recipe lost the part that matters"
    assert "PEP 668" in text


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
    assert any("PySide6" in item for item in extras["trainer"])
    assert any("nvidia-ml-py" in item for item in extras["trainer"])
    assert any(item.startswith("amdsmi;") for item in extras["trainer"])


def test_the_extension_is_not_tied_to_the_interpreter_that_built_it() -> None:
    """`nanobind_add_module(... STABLE_ABI ...)` is a request, not a result.

    nanobind grants it only when `Python::SABIModule` exists, which needs
    `Development.SABIModule` among the components `find_package(Python)` asks
    for — and when it does not, it falls back to a version-tagged module without
    a word. That is what happened here: every build worked, on the machine that
    built it, while the packages shipped an extension that a distribution Python
    upgrade would break and that no other machine's interpreter could load.

    Both halves are asserted, because either alone can regress: the declaration
    that makes it possible, and the artifact that proves it happened.
    """
    cmake = (ROOT / "bindings" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "Development.SABIModule" in cmake, "STABLE_ABI cannot be honoured without it"

    native = pytest.importorskip(
        "missile_defense._md_native", reason="the native binding is not built"
    )
    # `.pyd` on Windows, `.abi3.so` elsewhere — CPython's limited-API name is
    # the untagged one there, so "abi3 is in the filename" is not the test.
    expected = ".pyd" if sys.platform == "win32" else ".abi3.so"
    assert Path(native.__file__).name == f"_md_native{expected}", (
        f"built for one interpreter rather than the stable ABI: {native.__file__}"
    )


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
# The game and the training trainer are one source package and two different
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


def test_debian_builds_two_binary_packages_from_one_source() -> None:
    """The two products, and nothing between them.

    There used to be a third, `python3-md`, holding the Python half so it could be
    installed on its own. It is gone: the audience for a headless environment
    without the trainer is served by `pip install missile-defense`, and a third
    name was a third thing to explain for a split nobody was using.
    """
    assert set(_debian_stanzas()) == {"missile-defense", "missile-defense-trainer"}


def test_the_game_package_pulls_in_no_python_at_all() -> None:
    """The central promise, as a test rather than a paragraph.

    Someone who wants the game must not be made to install an interpreter, torch,
    or an LGPL Qt binding for Python. If this ever fails, the two products have
    become one.
    """
    game = _debian_stanzas()["missile-defense"]
    # `suggests` is deliberately not in this list. apt does not install Suggests,
    # so naming the trainer there costs a game-only install nothing — and it is
    # the only thing that tells such an install the other half exists. What it
    # must not become is a route to Python by another name, so it is pinned to
    # exactly one package below rather than merely scanned.
    installing = " ".join(
        game.get(field, "") for field in ("depends", "recommends", "pre-depends")
    ).lower()
    for forbidden in ("missile-defense-trainer", "python3", "pyside", "torch"):
        assert forbidden not in installing, f"the game package pulls in {forbidden}"
    assert game.get("suggests", "").strip() == "missile-defense-trainer"


def test_the_trainer_package_carries_the_dependencies_the_game_refuses() -> None:
    """And it is the trainer that depends on the game, never the other way round."""
    trainer = _debian_stanzas()["missile-defense-trainer"]
    relations = f"{trainer.get('depends', '')} {trainer.get('recommends', '')}".lower()
    assert "numpy" in relations, "the trainer does not depend on numpy"
    assert "pyside6" in relations, "the trainer does not depend on PySide6"
    # It carries a compiled extension now that the binding has no package of its
    # own, so `all` would be a package that installs on an architecture it cannot
    # run on.
    assert trainer.get("architecture", "").strip() == "any"


def test_the_trainer_package_can_build_the_runtime_it_offers_to_build() -> None:
    """The install-a-runtime button shells out to `python3 -m venv`.

    Debian ships `venv` separately and `${python3:Depends}` does not pull it, so
    a machine can satisfy every declared dependency and still fail on the first
    command of the one setup step this package exists to perform. Asserted from
    both ends — the declaration and the code that needs it — because either one
    moving alone is the bug.
    """
    trainer = _debian_stanzas()["missile-defense-trainer"]
    assert "python3-venv" in trainer.get("depends", ""), (
        "the trainer offers to build a virtualenv without depending on venv"
    )
    assert '"-m", "venv"' in (
        ROOT / "python" / "missile_defense" / "runs" / "runtime.py"
    ).read_text(encoding="utf-8")


def test_each_package_installs_a_disjoint_set_of_paths() -> None:
    """Two products cannot both own a file, and dpkg refuses if they try."""
    manifests = {
        name: set((ROOT / "debian" / f"{name}.install").read_text(encoding="utf-8").split())
        for name in ("missile-defense", "missile-defense-trainer")
    }
    assert not manifests["missile-defense"] & manifests["missile-defense-trainer"]
    # The trainer's entry point and its menu entry, which are what make it
    # reachable from an install rather than only from a checkout — and the Python
    # package, which moved here when `python3-md` was dissolved.
    trainer = " ".join(manifests["missile-defense-trainer"])
    assert "missile-defense-trainer" in trainer
    assert "dist-packages/missile_defense" in trainer
    assert "missile-defense-trainer.desktop" in trainer


def test_every_platform_has_a_way_to_launch_the_installed_trainer() -> None:
    """Three launchers, because three platforms answer "where is `md`?" differently.

    Linux hands the package to an interpreter the distribution owns, so a bare
    `exec python3 -m missile_defense.ui` is enough. Windows and macOS do not have that: the
    interpreter belongs to the user there and cannot be told at packaging time
    where the payload went, so both launchers have to set the import path
    themselves — from their own location, which is the only thing they know.

    A launcher that forgets is a trainer that starts and cannot import itself,
    and that is invisible until someone installs it.
    """
    templates = {
        "launcher.in": "@MD_LAUNCHER_PYTHON@ -m @MD_LAUNCHER_MODULE@",
        "launcher.cmd.in": "%~dp0",
        "trainer-bundle-launcher.in": "$here/../Resources",
    }
    for name, marker in templates.items():
        text = (ROOT / "packaging" / name).read_text(encoding="utf-8")
        assert marker in text, f"{name} lost the part that makes it work"
        assert "@MD_LAUNCHER_MODULE@" in text, f"{name} does not name a module to run"
    # The two that ship outside a distribution's control must not assume the
    # interpreter can already find `md`.
    for name in ("launcher.cmd.in", "trainer-bundle-launcher.in"):
        assert "PYTHONPATH" in (ROOT / "packaging" / name).read_text(encoding="utf-8")


def test_the_launchers_that_follow_a_users_python_check_it_first() -> None:
    """On Windows and macOS the interpreter is not ours, and may not be there.

    Both launchers exec a `python` they do not control. Windows answers a
    missing one with a Microsoft Store alias that returns 9009; macOS answers it
    from the Finder with no terminal at all, so the bundle bounces once and
    quits. Neither is a message, and both are somebody's first impression of a
    trainer they just installed. The Linux launcher is deliberately exempt: its
    package depends on the interpreter, so absence is not a state it can be in.
    """
    for name in ("launcher.cmd.in", "trainer-bundle-launcher.in"):
        text = (ROOT / "packaging" / name).read_text(encoding="utf-8")
        assert "version_info >= (3, 11)" in text, f"{name} execs a Python it never checked"
        assert "pip install PySide6" in text, f"{name} does not say what would fix it"
    # And it has to *say* it where the failure happens: a trainer window that
    # closes on exit, or a Finder launch that has no console at all.
    assert "pause" in (ROOT / "packaging" / "launcher.cmd.in").read_text(encoding="utf-8")
    assert "osascript" in (ROOT / "packaging" / "trainer-bundle-launcher.in").read_text(
        encoding="utf-8"
    )


def test_the_macos_trainer_is_a_separate_application() -> None:
    """A second `.app` in the disk image, and distinguishable from the game.

    The DMG has no checkboxes — the choice is which icon you drag — so "the
    trainer is optional on macOS" is only true if it is genuinely a second
    application. Sharing the game's bundle identifier would make macOS treat
    them as one, which is the failure this guards against.
    """
    trainer = (ROOT / "packaging" / "trainer.Info.plist.in").read_text(encoding="utf-8")
    game = (ROOT / "app" / "Info.plist.in").read_text(encoding="utf-8")
    assert "de.koehler-speyer.missile-defense-trainer" in trainer
    assert "@MACOSX_BUNDLE_GUI_IDENTIFIER@" in game  # the game's is set from CMake
    assert "developer-tools" in trainer, "the trainer is filed as a game"
    assert "arcade-games" in game
    # Without these two the bundle is a folder with a script in it: the Finder
    # will not launch it, and it never becomes a foreground GUI process.
    assert "CFBundleExecutable" in trainer
    assert "NSPrincipalClass" in trainer


def test_the_installer_offers_the_trainer_without_preselecting_it() -> None:
    """The Windows half of the same promise, read off the CPack declaration.

    `game` is required and `python` is offered unticked, so someone who came for
    a game can decline an interpreter without reading a manual — the same split
    `debian/control` keeps by producing separate binaries.
    """
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "set(CPACK_COMPONENT_GAME_REQUIRED ON)" in cmake
    assert "set(CPACK_COMPONENT_PYTHON_DISABLED ON)" in cmake
    # And the game's own install rules are tagged, or `--component game` would
    # stage nothing and every claim made about that tree would be vacuous.
    app_cmake = (ROOT / "app" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert app_cmake.count("COMPONENT game") >= 8


def test_a_bundled_model_would_ship_with_the_game_and_not_with_the_trainer() -> None:
    """The agent is `game`, not `python`, and that is the whole point.

    A `.mdp` is data (docs/API.md §7) and `md::agent::Policy` reads it natively,
    so playing against a learned agent needs no interpreter. Tagging it `python`
    would have made the pretrained agent a feature of the *training* package —
    exactly the coupling the format exists to remove.

    Asserted against the install rules rather than against a staged tree,
    because no model ships yet: every checkpoint predates the observation
    encoding's last change. This is the rule waiting for its payload.
    """
    cmake = (ROOT / "app" / "CMakeLists.txt").read_text(encoding="utf-8")
    block = cmake[cmake.index("file(GLOB MD_MODELS") : cmake.index("# ---- Install / packaging")]
    assert "models/*.mdp" in block, "every bundled model, not one named file"
    assert block.count("COMPONENT game") == 3, "one install rule per platform, all `game`"
    assert "COMPONENT python" not in block
    # Optional, or a checkout with no models would fail to configure. The guard
    # is the glob's own result rather than a file test now that there are three.
    assert "if(MD_MODELS)" in block


def test_the_trainer_has_a_desktop_entry_of_its_own() -> None:
    """A separate product gets a separate launcher, or it is not discoverable."""
    entry = (ROOT / "packaging" / "missile-defense-trainer.desktop").read_text(encoding="utf-8")
    assert "Exec=missile-defense-trainer" in entry
    # Not a game: it belongs in the development/science menu, not beside the
    # arcade cabinet, or someone looking for the game finds two of them.
    categories = next(line for line in entry.splitlines() if line.startswith("Categories="))
    assert "Game" not in categories
