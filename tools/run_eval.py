# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Build (if needed) and run the scripted-baseline benchmark (M4).

Prints the metrics the learned agent is measured against. Extra arguments are
passed straight through, e.g. ``poe eval --per-episode --seeds 64``.
"""

from __future__ import annotations

import subprocess
import sys

from . import _util


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    binary = _util.PROJECT_ROOT / "build" / "release" / "agent" / f"md_agent_eval{_util.EXE}"
    if not binary.exists():
        _util.run([sys.executable, "-m", "tools.cmake", "cmake", "--build", "--preset", "release"])
    return subprocess.call([str(binary), *args])


if __name__ == "__main__":
    raise SystemExit(main())
