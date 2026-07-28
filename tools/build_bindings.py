# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Build the `_md_native` extension and put it beside the Python package.

CMake writes the module into the build tree, but `missile_defense.sim.env` imports it
as ``missile_defense._md_native`` — so it has to sit next to
``python/missile_defense/``. Copying it there (it is gitignored) keeps ``pytest``
and an interactive session working straight from
a source checkout, with no install step and no PYTHONPATH juggling.

This used to be twice the size, because Windows had two Pythons: MSYS2's, which
built the game, and a native one, which was the only thing that could import
torch. Choosing between them — and choosing the matching preset — was most of
the file, and getting it wrong produced a module that built, copied, printed
"copied", and could never be imported. Windows is MSVC throughout now, so there
is one interpreter, one ABI and one preset, and all of that is gone.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import _util
from .bootstrap import windows_cxx_toolchain


def msvc_environment() -> dict[str, str] | None:
    """The environment a Developer Command Prompt would have, or ``None``.

    CMake's Ninja generator finds the compiler on ``PATH`` rather than locating
    an installation itself, so ``cl.exe`` has to be there — and the Build Tools
    put nothing on ``PATH`` until their shell has run. That used to be an error
    message telling you to open a different window and start again.

    It is cheaper to just enter the environment: ``vcvars64.bat`` prints what it
    sets, so running it once and keeping the result gives this process the same
    variables. Captured rather than shelled through, so the build command itself
    is never handed to `cmd` to re-parse.

    ``None`` means no action is needed or none is possible — either MSVC is
    already reachable, or there is no installation to enter.
    """
    if sys.platform != "win32" or shutil.which("cl") is not None:
        return None
    root = windows_cxx_toolchain()
    if root is None:
        return None
    vcvars = Path(root) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.is_file():
        return None  # `windows_cxx_toolchain` found cl.exe or VCINSTALLDIR, not a root
    # `shell=True`, and it has to be. Passing `["cmd", "/c", 'call "..." && set']`
    # as a list looks tidier and does not work: `subprocess` runs the argument
    # through `list2cmdline`, which quotes the third element because it contains
    # spaces, and `cmd` then receives escaped quotes and reports the batch file
    # "is either misspelled or could not be found" — naming the path it was just
    # handed, which is the least helpful possible way to say "quoting".
    #
    # Nothing here comes from a person: `vcvars` is a fixed filename under a
    # directory vswhere reported.
    probe = subprocess.run(
        f'"{vcvars}" >nul && set',
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return None
    found: dict[str, str] = {}
    for line in probe.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            found[key] = value
    return found or None


def missing_build_tools() -> str | None:
    """What this build needs and cannot reach, as a sentence — or ``None``.

    Reached only after :func:`msvc_environment` has had its go, so on Windows
    this fires when there is no Visual Studio at all rather than merely the
    wrong shell. The way it used to surface was a `FileNotFoundError` traceback
    out of `subprocess` naming no tool at all.
    """
    needed = ["cmake", "ninja", "cl"] if sys.platform == "win32" else ["cmake", "ninja"]
    absent = [tool for tool in needed if shutil.which(tool) is None]
    if not absent:
        return None
    listed = ", ".join(absent)
    if sys.platform != "win32":
        return f"error: not on PATH: {listed}."
    return (
        f"error: not on PATH: {listed}.\n"
        "The Windows build needs MSVC, CMake and Ninja. Install them with:\n"
        "    winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "
        '"--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools"\n'
        "    winget install -e --id Kitware.CMake\n"
        "    winget install -e --id Ninja-build.Ninja"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # `release` on every platform. There used to be a Windows-only preset here,
    # because the game was MinGW and only the extension was MSVC; with one
    # compiler the ordinary preset builds both.
    parser.add_argument("preset", nargs="?", default="release", help="CMake configure preset")
    parser.add_argument(
        "--python",
        default=None,
        help="Interpreter to build against. Defaults to the running one.",
    )
    parsed = parser.parse_args(sys.argv[1:] if argv is None else argv)
    preset = parsed.preset
    # The running interpreter is the right default now that there is only one
    # kind on Windows: whoever runs `poe bindings` is in the venv the module is
    # imported from.
    python = parsed.python or sys.executable

    # Enter the developer environment rather than demanding one. Applied to this
    # process so both CMake calls below inherit it; a `dict` from `vcvars64.bat`
    # rather than a shell wrapper, so nothing re-parses the build command.
    if (developer := msvc_environment()) is not None:
        os.environ.update(developer)

    if (unreachable := missing_build_tools()) is not None:
        print(unreachable, file=sys.stderr)
        return 1

    # Point CMake at *that* interpreter: nanobind's headers come from the
    # environment the extension will be imported into, so a mismatch here shows up
    # much later as an import error.
    _util.run(
        [
            "cmake",
            "--preset",
            preset,
            "-DMD_BUILD_BINDINGS=ON",
            f"-DPython_EXECUTABLE={python}",
        ]
    )
    # `md_native_beside_package`, not `_md_native`: building the module is half
    # the job, and the other half — putting it next to python/missile_defense/,
    # where `missile_defense.sim.env` imports it from — is a declared CMake output now
    # rather than something this
    # script does to the build tree afterwards (bindings/CMakeLists.txt says how,
    # including which stale names a build supersedes and which belong to another
    # toolchain and are left alone). One owner, and `cmake --build` on its own
    # leaves an importable checkout.
    built = _util.run(
        ["cmake", "--build", "--preset", preset, "--target", "md_native_beside_package"],
        check=False,
    )
    if built.returncode != 0:
        # Most often not a compile error at all: CMake skips the extension when
        # Python's development files or nanobind are missing, and then the target
        # does not exist, which ninja reports as `unknown target` several lines
        # away from anything that names the cause.
        print(
            "error: the extension was not built — CMake skips it when Python "
            "development files or nanobind are missing (pip install nanobind).",
            file=sys.stderr,
        )
        return 1

    # A build step that produces an unloadable artefact and prints "copied" has
    # not done its job. One import, in the interpreter it was built for — the
    # same check `missile_defense.runs.runtime` makes before an install, and for the same reason:
    # the file being on the path proved nothing about it loading.
    check = subprocess.run(
        [python, "-c", "import missile_defense._md_native"],
        cwd=_util.PROJECT_ROOT / "python",
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode != 0:
        print(
            f"error: built and copied, but {python} cannot import it:\n{check.stderr.strip()}",
            file=sys.stderr,
        )
        return 1
    print(f"verified: {python} imports missile_defense._md_native")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
