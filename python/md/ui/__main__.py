# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""``python -m md.ui`` — the console's entry point, and the one place that can
still say something useful when Qt is not installed.

The console is an *optional* component: PySide6 and Qt Charts are LGPLv3 while
this project is MIT, so they are never a dependency of the game or the ``.deb``
(docs/ROADMAP.md, M8). A missing PySide6 is therefore a normal state to explain,
not a traceback to dump.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from .app import main as run
    except ModuleNotFoundError as error:
        if error.name is not None and not error.name.startswith("PySide6"):
            raise
        print(
            "The training console needs PySide6 (Qt Charts comes with it):\n"
            f"    {sys.executable} -m pip install PySide6\n"
            "It is optional, and deliberately not a dependency of the game.",
            file=sys.stderr,
        )
        return 1
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
