# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
"""Create the development venv with the trainer and its telemetry backends.

Run from a fresh checkout with::

    python3 -m tools.bootstrap

PyTorch remains separate because its correct wheel depends on the host GPU and
driver. The trainer is small enough to install consistently, including the
NVIDIA binding everywhere and AMD SMI on Linux.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path

from ._util import PROJECT_ROOT

#: `build` is here for one test: `tests/e2e/test_wheel_install.py` installs the
#: wheel this repository produces into a fresh interpreter, which is the only
#: check in the suite that runs the *shipped* package rather than the source
#: tree. Without it that whole file skips, and the packaging bugs it exists to
#: catch — a module left out, a binding not copied, a renamed entry point —
#: reach a release instead.
DEV_TOOLS = ("poethepoet", "ruff", "pytest", "mypy", "pyright", "build")

#: What `poe bindings` needs, and the reason a fresh checkout could not train.
#:
#: nanobind is already in `[build-system].requires`, which covers building the
#: *wheel* — pip creates an isolated environment for that and installs it there.
#: It does not cover `cmake --build --target _md_native`, which is what `poe
#: bindings` runs against this venv. Without it CMake simply does not create the
#: target, so the build fails with `unknown target` and `missile_defense._md_native` is never
#: written beside the package.
#:
#: The visible cost of that was two steps away and looked like something else
#: entirely: the trainer's runtime installer downloads five gigabytes of CUDA
#: torch, health-checks the result by importing the binding, finds none, and
#: reports the install as failed. So a bootstrapped trainer could watch runs and
#: never start one — which is exactly the promise `[trainer]` is supposed to keep.
BUILD_TOOLS = ("nanobind>=2.0",)

#: Constraints the *gate* needs, over and above what the package needs to run.
#:
#: `numpy<2.5` is a typing requirement rather than a runtime one. numpy 2.5
#: dropped Python 3.11 and writes PEP 695 `type` statements in its own stubs,
#: which mypy parses only when targeting 3.12+ — so a 2.5 numpy beside
#: `python_version = "3.11"` fails inside a stub file nothing in this repository
#: can edit. CI pins the same bound for the same reason; pinning it here means a
#: freshly bootstrapped venv passes `poe check` instead of failing on somebody
#: else's file.
#:
#: Pillow is what `tools/make_icon.py` needs, and installing it is the difference
#: between that script being type-checked and being waved through.
GATE_PINS = ("numpy<2.5", "pillow")


def venv_python(venv: Path, *, platform: str = sys.platform) -> Path:
    """The interpreter a venv creates on this platform."""
    return venv / ("Scripts/python.exe" if platform == "win32" else "bin/python")


def install_command(venv: Path, *, root: Path = PROJECT_ROOT) -> list[str]:
    """The one pip invocation that defines a ready development trainer."""
    return [
        str(venv_python(venv)),
        "-m",
        "pip",
        "install",
        "--editable",
        f"{root}[trainer]",
        *DEV_TOOLS,
        *BUILD_TOOLS,
        *GATE_PINS,
    ]


def windows_cxx_toolchain(environ: Mapping[str, str] | None = None) -> str | None:
    """Where MSVC is on this machine, or ``None`` when there is none.

    ``pip install --editable .`` is not a pure-Python install here: the build
    backend is scikit-build-core, which runs the CMake tree — and CMake needs a
    C++ compiler for its *configure* step whether or not a target is built. On a
    Windows box without MSVC that fails several hundred lines into a log whose
    only clue is ``CMAKE_CXX_COMPILER``, which is not something a person can act
    on. Asking first turns that into a sentence.

    Three ways of being inside a toolchain, cheapest first: already in a
    developer shell, ``cl.exe`` on ``PATH``, or vswhere reporting an install
    with the C++ workload. **MinGW deliberately does not count.** An extension
    has to share an ABI with the interpreter importing it, and the interpreter
    the trainer runs under is MSVC-built — an MSYS2 clang on ``PATH`` is the
    wrong compiler for this job, not a substitute for the right one.
    """
    env = os.environ if environ is None else environ
    inside = env.get("VCINSTALLDIR")
    if inside:
        return inside
    on_path = shutil.which("cl", path=env.get("PATH"))
    if on_path:
        return on_path
    vswhere = (
        Path(env.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        / "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    if not vswhere.is_file():
        return None
    probe = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.stdout.strip() or None


def trainer_requirements(root: Path = PROJECT_ROOT) -> list[str]:
    """What the package and its ``[trainer]`` extra declare, read from pyproject.

    Read rather than restated. A second copy of the list here would be a second
    thing to keep current, and the way that goes wrong is not abstract: `numpy`
    is a *base* dependency, so a trainer assembled from a hand-written list that
    forgot it starts, imports `missile_defense.runs.league`, and dies in `policy_format` with a
    traceback — which is the exact failure this whole path exists to avoid.
    """
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    extras = project.get("optional-dependencies", {})
    return [*project.get("dependencies", ()), *extras.get("trainer", ())]


def dependency_command(venv: Path, *, root: Path = PROJECT_ROOT) -> list[str]:
    """Everything :func:`install_command` would install, minus the project itself.

    The fallback for a machine that cannot compile the extension. What it leaves
    out is `md` — supplied instead by :func:`link_checkout`, which is the half of
    an editable install that needs no compiler.
    """
    return [
        str(venv_python(venv)),
        "-m",
        "pip",
        "install",
        *trainer_requirements(root),
        *DEV_TOOLS,
        *BUILD_TOOLS,
        *GATE_PINS,
    ]


def site_packages(venv: Path, *, platform: str = sys.platform) -> Path:
    """The venv's ``site-packages``, wherever this platform puts it."""
    if platform == "win32":
        return venv / "Lib" / "site-packages"
    libs = sorted((venv / "lib").glob("python3*/site-packages"))
    if not libs:
        raise FileNotFoundError(f"no site-packages under {venv / 'lib'}")
    return libs[0]


def link_checkout(venv: Path, *, root: Path = PROJECT_ROOT, platform: str = sys.platform) -> Path:
    """Put this checkout's ``python/`` on the venv's import path.

    Exactly what an editable install does for a pure-Python package, by exactly
    the same mechanism — a ``.pth`` file — and without the build step that needs
    a compiler. Edits to the sources are live here just as they would be.
    """
    marker = site_packages(venv, platform=platform) / "md-checkout.pth"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{root / 'python'}\n", encoding="utf-8")
    return marker


def activate_hint(venv: Path, *, platform: str = sys.platform) -> str:
    """How to enter the environment, in this platform's own shell.

    `source .venv/bin/activate` is not a thing on Windows, and printing it to
    someone who has just run this on Windows is a small lie at the exact moment
    they are looking for instructions.
    """
    if platform == "win32":
        return f"{venv / 'Scripts' / 'Activate.ps1'}   (PowerShell)"
    return f"source {venv / 'bin' / 'activate'}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a development venv with tooling and trainer dependencies."
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=PROJECT_ROOT / ".venv",
        help="environment to create or update (default: .venv)",
    )
    parser.add_argument(
        "--no-extension",
        action="store_true",
        help="install the dependencies and link this checkout, but do not build "
        "_md_native (implied on Windows when no MSVC toolchain is found)",
    )
    args = parser.parse_args(argv)
    venv = args.venv.resolve()

    # Decided before the venv exists, so nothing is half-built when the answer
    # is "this machine cannot".
    skip = args.no_extension
    if not skip and sys.platform == "win32" and windows_cxx_toolchain() is None:
        skip = True
        print(
            "No MSVC toolchain found, so the simulation binding cannot be built here.\n"
            "Setting up everything else instead — the trainer will start, browse runs\n"
            "and play replays; it will refuse to *start* a run and say why.\n"
            "\n"
            "To get the rest, install the C++ build tools and re-run this:\n"
            "    winget install -e --id Microsoft.VisualStudio.2022.BuildTools "
            '--override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools"\n',
            file=sys.stderr,
        )

    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    if skip:
        subprocess.run(dependency_command(venv), check=True)
        link_checkout(venv)
    else:
        subprocess.run(install_command(venv), check=True)

    print(f"ready: {venv}")
    print(f"activate it with: {activate_hint(venv)}")
    if skip:
        print("without _md_native: `poe bindings` once you have a compiler")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
