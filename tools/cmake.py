# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Run ``cmake``/``ctest``/``cpack`` with the compiler already reachable.

Every C++ task goes through here rather than naming the tool directly, and the
reason is Windows. CMake's Ninja generator finds the compiler on ``PATH``
instead of locating an installation itself, and the Visual Studio Build Tools
put *nothing* on ``PATH`` until their own shell has run. So a plain
``cmake --preset release`` from any ordinary terminal fails with "no
CMAKE_CXX_COMPILER could be found" — which names the variable rather than the
cause, and is not something a person can act on.

``vcvars64.bat`` prints what it sets, so this runs it once, keeps the result and
hands it to the child. Entering the developer environment is cheaper than
telling somebody to open a different window, and it means `poe build`, `poe
test` and `poe app` behave the same on Windows as they do everywhere else.

A no-op off Windows, and a no-op on Windows when the compiler is already
reachable — a Developer Command Prompt, or CI, where the setup action has
already done it.
"""

from __future__ import annotations

import os
import sys

from . import _util
from .build_bindings import msvc_environment

#: What may be run. Named rather than "whatever argv[0] is", so this cannot
#: become a general-purpose way to launch a process with a modified environment.
TOOLS = ("cmake", "ctest", "cpack")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] not in TOOLS:
        print(
            f"usage: python -m tools.cmake <{'|'.join(TOOLS)}> [args...]",
            file=sys.stderr,
        )
        return 2

    if (developer := msvc_environment()) is not None:
        os.environ.update(developer)

    # `check=False` and the child's own code: these wrap build and test
    # commands, whose failures are the caller's news to report. Raising here
    # would replace a compiler's diagnostics with a Python traceback.
    return _util.run(list(args), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
