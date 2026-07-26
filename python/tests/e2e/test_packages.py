# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The two packages are two different products, asserted from staged install trees.

`test_packaging.py` reads the *declarations* — `debian/control`, `pyproject.toml`,
the `.install` manifests — and that catches a promise broken on paper. It cannot
catch one broken in the binary: a game that finds a console it was never packaged
with, or an installed console the game cannot see. Only running the shipped
executables out of a staged tree does that, which is why these are e2e.

**The negative half matters more than the positive one.** Someone who wants the
game must not be made to install an interpreter, and the way that promise dies is
not a dependency appearing in `debian/control` — someone would notice — but the
game quietly resolving a console through `PATH` on the developer's machine and
nobody ever testing the machine where there is none.

The trees are staged with `cmake --install`, which is the same code path the
Debian packages use (`debian/rules` runs it twice with different components).
That is deliberate: a test that staged the tree by copying files by hand would
keep passing after the install rules stopped producing that layout.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from .harness import (
    PROJECT_ROOT,
    app_environ,
    assert_clean,
    needs_display,
    needs_qt,
    run_app,
)

pytestmark = pytest.mark.e2e

#: Where the game binary lands in a staged tree, by app/CMakeLists.txt's
#: `install(PROGRAMS ... DESTINATION games RENAME missile-defense)`.
STAGED_GAME = Path("games") / "missile-defense"

#: And the console launcher, by the top-level `MD_INSTALL_PYTHON_PACKAGE` block.
STAGED_CONSOLE = Path("bin") / "md-console"

#: Where the `md` package goes, spelled the way `debian/rules` spells it.
#: `MD_PYTHON_INSTALL_DIR` has no default of its own outside `bindings/`, and a
#: staging test that let it fall back would install the package's *contents*
#: into the prefix root and then assert against a layout nothing ships.
PACKAGE_DESTINATION = Path("lib") / "python3" / "dist-packages" / "md"


def _build_directory() -> Path | None:
    """A configured build tree to stage the game out of, best first."""
    for preset in ("release", "debug"):
        candidate = PROJECT_ROOT / "build" / preset
        if (candidate / "CMakeCache.txt").exists():
            return candidate
    return None


needs_build_tree = pytest.mark.skipif(
    _build_directory() is None,
    reason="no configured build tree to stage from — cmake --preset release",
)
needs_cmake = pytest.mark.skipif(shutil.which("cmake") is None, reason="cmake is not installed")


def _install(prefix: Path, build: Path, *, component: str | None = None) -> None:
    command = ["cmake", "--install", str(build), "--prefix", str(prefix)]
    if component is not None:
        command += ["--component", component]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, f"staging failed:\n{result.stdout}\n{result.stderr}"


@pytest.fixture(scope="module")
def game_only_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """What `apt install missile-defense` alone puts on a machine.

    Module-scoped because staging is the slow part and nothing here mutates the
    tree — every test reads it and runs binaries out of it.
    """
    build = _build_directory()
    assert build is not None
    prefix = tmp_path_factory.mktemp("game-only")
    # `--component game` and not a plain install: this build tree also built the
    # bindings, and a plain install would have staged the native extension into
    # a tree calling itself game-only. That is exactly the mistake the component
    # tags exist to prevent, and it is how this test found them missing.
    _install(prefix, build, component="game")
    return prefix


@pytest.fixture(scope="module")
def full_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The game *and* the training console, in one filesystem.

    Staged the way the Debian source package does it — the game from the
    ordinary build, the Python payload from a second configure with
    `MD_INSTALL_PYTHON_PACKAGE=ON` — rather than by copying files into place, so
    a change to either install rule shows up here.

    The second configure builds nothing: with the app, the bindings and the
    tests off there is no compiler work left to do, only `install()` rules to
    generate. It costs a couple of seconds.
    """
    build = _build_directory()
    assert build is not None
    prefix = tmp_path_factory.mktemp("full")
    _install(prefix, build, component="game")

    python_build = prefix.parent / "python-build"
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(PROJECT_ROOT),
            "-B",
            str(python_build),
            "-DMD_INSTALL_PYTHON_PACKAGE=ON",
            f"-DMD_PYTHON_INSTALL_DIR={PACKAGE_DESTINATION}",
            "-DMD_BUILD_APP=OFF",
            "-DMD_BUILD_BINDINGS=OFF",
            "-DMD_BUILD_TOOLS=OFF",
            "-DBUILD_TESTING=OFF",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert configure.returncode == 0, f"configure failed:\n{configure.stdout}\n{configure.stderr}"
    _install(prefix, python_build, component="python")
    return prefix


def _pathless_environ(sandbox: Path, tree: Path | None) -> dict[str, str]:
    """The environment of a machine with no Python and no console anywhere.

    `PATH` is reduced to the staged tree's own directories plus the few system
    ones the *graphics stack* needs, because the game still has to find Xvfb's
    display and its Vulkan driver. What it must not find is an interpreter, so
    `/usr/bin` — where every distribution keeps `python3` — is left out, and
    `MD_CONSOLE` is cleared in case the developer running this has one set.
    """
    env = app_environ(sandbox)
    directories = [str(tree / "games"), str(tree / "bin")] if tree is not None else []
    env["PATH"] = os.pathsep.join([*directories, "/usr/games"])
    env.pop("MD_CONSOLE", None)
    env.pop("MD_APP", None)
    return env


# ---- the negative half -------------------------------------------------------


@needs_cmake
@needs_build_tree
def test_the_game_only_tree_contains_no_python_and_no_console(game_only_tree: Path) -> None:
    """Nothing Python-shaped may be installed by the game alone.

    This is the cheap half and it runs everywhere, including on a machine with
    no display: if a `.py` file has appeared under the game's prefix, the two
    products have started to merge and the rest of this file is moot.
    """
    staged = {
        path.relative_to(game_only_tree) for path in game_only_tree.rglob("*") if path.is_file()
    }
    assert STAGED_GAME in staged, f"the game itself is missing from {sorted(staged)}"
    assert not [path for path in staged if path.suffix in (".py", ".pyi")]
    assert not [path for path in staged if path.name.startswith("md-")]
    assert not [path for path in staged if "training" in path.name]


@needs_cmake
@needs_build_tree
@needs_display
def test_a_game_only_install_does_not_offer_to_train(game_only_tree: Path, tmp_path: Path) -> None:
    """The promise, from the outside: no console on the machine, no TRAIN AI.

    Run out of the staged tree rather than the build tree, because the build
    tree *is* a checkout — the game's third lookup would find `python/md/ui`
    right there and correctly offer training. An installed game has no checkout
    above it, and that is the case this asserts.
    """
    run = run_app(
        frames=90,
        sandbox=tmp_path,
        binary=game_only_tree / STAGED_GAME,
        environ=_pathless_environ(tmp_path, None),
    )
    assert_clean(run)
    assert run.report.get("can_train") is False
    assert "TRAIN AI" not in run.menu
    # And the rest of the menu is intact — an entry vanishing is a bug, a menu
    # collapsing to nothing is a different and much worse one.
    assert "START" in run.menu
    assert "WATCH AI" in run.menu


# ---- the positive half -------------------------------------------------------


@needs_cmake
@needs_build_tree
def test_the_full_tree_carries_the_console_and_its_menu_entry(full_tree: Path) -> None:
    launcher = full_tree / STAGED_CONSOLE
    assert launcher.exists(), "no md-console launcher in the full tree"
    assert os.access(launcher, os.X_OK), "the console launcher is not executable"
    assert (full_tree / "share" / "applications" / "missile-defense-training.desktop").exists()
    # The package itself, so the launcher has something to run.
    assert (full_tree / PACKAGE_DESTINATION / "ui" / "app.py").exists()


@needs_cmake
@needs_build_tree
@needs_display
def test_the_full_install_offers_to_train(full_tree: Path, tmp_path: Path) -> None:
    """The same binary, the same stripped PATH — and now the entry is there.

    Paired with the negative test on purpose: the two differ in exactly one
    thing, which is whether `md-console` is on the machine. If both passed with
    the entry present, or both with it absent, the lookup would not be reading
    what it claims to read.
    """
    run = run_app(
        frames=90,
        sandbox=tmp_path,
        binary=full_tree / STAGED_GAME,
        environ=_pathless_environ(tmp_path, full_tree),
    )
    assert_clean(run)
    assert run.report.get("can_train") is True
    assert "TRAIN AI" in run.menu


@needs_cmake
@needs_build_tree
@needs_qt
def test_the_staged_console_starts_and_exits_cleanly(full_tree: Path, tmp_path: Path) -> None:
    """The staged launcher, offscreen, all the way to a built window.

    Run as the *shell wrapper* rather than as `python -m md.ui`, because the
    wrapper is what the package ships and the ways it breaks are all invisible
    to a file-existence check: the wrong interpreter, an import path that does
    not reach the staged package, a module that is not executable as `-m`.

    `--self-test` is the console's `--report`: it builds the window, reads the
    run directory once and prints a line. Without it the console would sit in
    its event loop until something closed it, and the only thing this test could
    assert is that it did not exit — which is also what a hang looks like.
    """
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(full_tree / PACKAGE_DESTINATION.parent), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["HOME"] = str(tmp_path)
    env["MD_RUNS_DIR"] = str(tmp_path / "runs")
    result = subprocess.run(
        [str(full_tree / STAGED_CONSOLE), "--self-test"],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd=str(tmp_path),
        check=False,
    )
    assert result.returncode == 0, f"the staged console failed:\n{result.stdout}\n{result.stderr}"
    assert json.loads(result.stdout.splitlines()[-1])["ok"] is True
