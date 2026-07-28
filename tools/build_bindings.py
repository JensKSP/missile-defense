# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Build the `_md_native` extension and put it beside the Python package.

CMake writes the module into the build tree, but `missile_defense.sim.env` imports it
as ``missile_defense._md_native`` — so it has to sit next to
``python/missile_defense/``. Copying it there (it is gitignored) keeps ``pytest``
and an interactive session working straight from
a source checkout, with no install step and no PYTHONPATH juggling.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from . import _util, launch
from .bootstrap import windows_cxx_toolchain

#: The preset that builds against a native Windows CPython rather than MSYS2's.
WIN_NATIVE = "win-native"


def platform_tag(interpreter: str) -> str:
    """``sysconfig.get_platform()`` for ``interpreter``, or ``""`` if it will not run.

    The one word that tells an MSYS2 Python from a native one: `mingw_x86_64`
    against `win-amd64`. They report the *same version* — 3.14 either way — so
    nothing about the version, the path or the name distinguishes them, which is
    exactly why a build against the wrong one looks right until it is imported.
    """
    probe = subprocess.run(
        [interpreter, "-c", "import sysconfig; print(sysconfig.get_platform())"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.stdout.strip() if probe.returncode == 0 else ""


def is_mingw(interpreter: str) -> bool:
    """Whether ``interpreter`` is an MSYS2/MinGW Python."""
    return platform_tag(interpreter).startswith("mingw")


def native_interpreter(*, exclude: str) -> str | None:
    """A native Windows CPython on this machine, or ``None`` if there is none.

    The same candidate list `poe train` and `poe ui` search, for the same reason:
    the interpreter worth building for is the one that will actually import the
    result, and on Windows that is never the MSYS2 one — torch publishes no
    MinGW wheel, so a module built against MSYS2's Python can never be the module
    a training run loads.
    """
    for candidate in launch.candidates():
        if candidate != exclude and not is_mingw(candidate):
            return candidate
    return None


#: How to enter an environment where MSVC, CMake and Ninja are all reachable.
#: The Build Tools ship the last two, but only the developer shell puts any of
#: them on `PATH` — which is why CI runs its native build inside one.
DEV_SHELL_HINT = (
    r'    "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools'
    r'\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64'
)


def missing_build_tools(preset: str, python: str) -> str | None:
    """What this build needs and cannot reach, as a sentence — or ``None``.

    Every preset here uses the Ninja generator, so CMake finds the compiler on
    ``PATH`` rather than locating an installation itself: `cl.exe` has to be
    there, and so do `cmake` and `ninja`. Outside a developer shell none of the
    three is, and the way that surfaced was a `FileNotFoundError` traceback out
    of `subprocess` naming no tool at all.
    """
    native = sys.platform == "win32" and not is_mingw(python)
    needed = ["cmake", "ninja", "cl"] if native else ["cmake", "ninja"]
    absent = [tool for tool in needed if shutil.which(tool) is None]
    if not absent:
        return None
    listed = ", ".join(absent)
    if not native:
        return f"error: not on PATH: {listed}. Build from the MSYS2 CLANG64 shell."
    return (
        f"error: not on PATH: {listed}.\n"
        f"Building for {python} uses the Ninja generator with MSVC, which finds "
        "its tools on PATH. The Build Tools ship CMake and Ninja but put nothing "
        "on PATH until you enter their shell:\n"
        f"{DEV_SHELL_HINT}\n"
        "Run that in a Command Prompt — not PowerShell, where a .bat sets its "
        "variables in a child process that then exits, leaving the session it "
        "was meant to prepare untouched — then this again from the same window."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", nargs="?", default="release", help="CMake configure preset")
    parser.add_argument(
        "--python",
        default=None,
        help=(
            "Interpreter to build against. Defaults to the running one, except "
            "under MSYS2 on Windows, where it defaults to a native CPython: torch "
            "ships no MinGW wheel, so training needs a module built for that ABI."
        ),
    )
    parsed = parser.parse_args(sys.argv[1:] if argv is None else argv)
    preset = parsed.preset
    python = parsed.python

    # Choosing the interpreter, when nobody said which. `sys.executable` is the
    # obvious answer and on Windows it is the wrong one often enough to matter:
    # run through `poe` from the CLANG64 shell, it is MSYS2's Python, and a
    # module built for that cannot be loaded by the interpreter that has torch.
    # The build succeeded, printed "copied", and left something no training run
    # could ever import — which is how 2026-07-28 was spent.
    if python is None:
        python = sys.executable
        if sys.platform == "win32" and is_mingw(python):
            found = native_interpreter(exclude=python)
            if found is None:
                print(
                    "error: this is MSYS2's Python, and a module built for it cannot "
                    "be imported by anything that can train — torch publishes no "
                    "MinGW wheel.\nNo native CPython was found to build for instead. "
                    "Install one from python.org, or name it with --python.",
                    file=sys.stderr,
                )
                return 1
            # stderr, with the errors it may be followed by: stdout is buffered
            # and stderr is not, so a note about the choice printed to the other
            # stream arrives *after* the message explaining why the choice
            # failed, which reads as two unrelated events.
            print(f"note: building for {found} rather than MSYS2's {python}", file=sys.stderr)
            python = found

    # The preset follows the *target* interpreter, never whoever is running this
    # script. Both routes reach here and only one of them is informative: from
    # the CLANG64 shell the runner is MSYS2's Python and the target was just
    # discovered, while from the venv the runner is already the target. Keying
    # this off the runner built a native module with the MSYS2 preset — the same
    # mismatch as before, in the other direction.
    if sys.platform == "win32" and preset == parser.get_default("preset") and not is_mingw(python):
        preset = WIN_NATIVE

    # A native module needs MSVC, and CMake says so several hundred lines in.
    if sys.platform == "win32" and not is_mingw(python) and windows_cxx_toolchain() is None:
        print(
            f"error: {python} is a native CPython, so its extension needs the MSVC "
            "toolchain, and none was found. Install it with the Visual Studio "
            "Installer — workload 'Desktop development with C++' — then run this "
            "from a Developer Command Prompt.",
            file=sys.stderr,
        )
        return 1

    if (unreachable := missing_build_tools(preset, python)) is not None:
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
    # the file being on the path proved nothing about it loading. The failure
    # this catches is not exotic; it is what a MinGW build in a native CPython
    # does, and it surfaced two steps later as a training runtime that would not
    # install.
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
