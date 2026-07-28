# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Build (if needed) and launch the game, passing through any extra arguments."""

from __future__ import annotations

import subprocess
import sys

from . import _util


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    binary = _util.app_binary()
    if not binary.exists():
        # Through `tools.cmake`, which enters the developer environment first:
        # CMake's Ninja generator finds the compiler on PATH and the Build Tools
        # put nothing there until their shell has run, so a plain `cmake` here
        # failed with no compiler in any ordinary terminal.
        _util.run([sys.executable, "-m", "tools.cmake", "cmake", "--build", "--preset", "release"])
    # No platform is imposed: `poe app` should start the game the way installing
    # it does, on the session the developer is actually running. It used to force
    # xcb, which made every local run an XWayland run and hid how the shipped
    # game behaves from the person most likely to notice.
    #
    # Screenshots are the one case that still needs an X11 window — `tools/capture`
    # asks for xcb itself, where the requirement belongs.
    #
    # `launch_environment` adds only what a *build-tree* binary needs and cannot
    # find for itself: on Windows, the Qt kit it was linked against. An installed
    # copy has those DLLs beside the exe and this changes nothing.
    return subprocess.call([str(binary), *args], env=_util.launch_environment())


if __name__ == "__main__":
    raise SystemExit(main())
