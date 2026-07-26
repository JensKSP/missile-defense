# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The installed commands' front door — the one place that can explain a missing
optional dependency instead of raising it.

``md-train`` and ``md-console`` are entry points of a package whose heavy halves
are deliberately **not** dependencies (docs/PACKAGING.md): torch because Debian's
copy is CPU-only and the vendor wheels are gigabytes, PySide6 because it is
LGPLv3 while this project is MIT. The price of that choice is that the absence
has to be *explained*, and a traceback out of the middle of an import is not an
explanation — it reads as a broken installation rather than as an optional piece
someone has not installed yet.

So the console has ``md.ui.__main__`` and the trainer has this. Neither imports
anything heavy until it has checked; ``find_spec`` locates a module without
importing it, so the check costs nothing when the answer is yes.
"""

from __future__ import annotations

import importlib.util
import sys

#: What each command needs, and what the message tells you to install.
TRAIN_REQUIRES = "torch"


def _missing(module: str, package: str, what: str) -> str:
    return (
        f"{what} needs {package}, which is not installed for this interpreter:\n"
        f"    {sys.executable} -m pip install {package}\n"
        f"It is deliberately not a hard dependency — see docs/PACKAGING.md — so "
        f"`pip install missile-defense` stays cheap for anyone who only wants the "
        f"environment. On Debian the venv recipe is in README.Debian.\n"
        f"(looked for the module `{module}`)"
    )


def train(argv: list[str] | None = None) -> int:
    """``md-train`` — the training loop, if this interpreter can run one."""
    if importlib.util.find_spec("torch") is None:
        print(_missing("torch", TRAIN_REQUIRES, "Training"), file=sys.stderr)
        return 1
    from .train import main

    return main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(train())
