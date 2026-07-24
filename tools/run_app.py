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
    env = dict(os.environ)
    if sys.platform == "linux":
        # Force Qt's xcb platform so the window is an X11 window (screenshot-able).
        env.setdefault("QT_QPA_PLATFORM", "xcb")
    return subprocess.call([str(binary), *args], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
