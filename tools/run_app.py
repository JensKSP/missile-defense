# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Build (if needed) and launch the game, passing through any extra arguments."""

from __future__ import annotations

import os
import subprocess
import sys

from . import _util


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    binary = _util.app_binary()
    if not binary.exists():
        _util.run(["cmake", "--build", "--preset", "release"])
    # No platform is imposed: `poe app` should start the game the way installing
    # it does, on the session the developer is actually running. It used to force
    # xcb, which made every local run an XWayland run and hid how the shipped
    # game behaves from the person most likely to notice.
    #
    # Screenshots are the one case that still needs an X11 window — `tools/capture`
    # asks for xcb itself, where the requirement belongs.
    return subprocess.call([str(binary), *args], env=dict(os.environ))


if __name__ == "__main__":
    raise SystemExit(main())
