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
import sysconfig
from pathlib import Path

#: What each command needs, and what the message tells you to install.
TRAIN_REQUIRES = "torch"

#: Where the packaged console explains the same thing at length, and the only
#: path worth printing: a bare "README.Debian" is not something anyone can open.
DEBIAN_README = "/usr/share/doc/missile-defense-training/README.Debian"


def externally_managed(interpreter_stdlib: str | None = None) -> bool:
    """Whether PEP 668 forbids installing into this interpreter.

    Debian 12 and later mark the system Python this way, which makes the obvious
    advice — ``pip install torch`` — the one command that is guaranteed to fail
    on the platform this project packages for. Telling someone to run it there
    and letting pip refuse is worse than saying nothing: it reads as the program
    not knowing its own system.

    Takes the stdlib directory as an argument so a test can describe either kind
    of machine without being run on one.
    """
    stdlib = sysconfig.get_path("stdlib") if interpreter_stdlib is None else interpreter_stdlib
    return (Path(stdlib) / "EXTERNALLY-MANAGED").exists()


def explain_missing(module: str, package: str, what: str, *, managed: bool | None = None) -> str:
    """Why the command cannot run, and the fix that works *here*.

    Two different machines are being addressed. On one, ``pip install`` into the
    running interpreter is the answer. On the other it is refused by design, and
    the answer is a virtualenv — or, for someone who installed the packages
    rather than the wheel, the console's own install-a-runtime button, which is
    the whole reason that button exists.
    """
    if managed is None:
        managed = externally_managed()
    if managed:
        fix = (
            f"This interpreter is externally managed (PEP 668), so it will refuse a\n"
            f"`pip install` — correctly. Use the training console, which installs a\n"
            f"runtime for you:\n"
            f"    md-console\n"
            f"or build one yourself:\n"
            f"    python3 -m venv --system-site-packages ~/.venvs/md\n"
            f"    ~/.venvs/md/bin/pip install {package}\n"
            f"    ~/.venvs/md/bin/md-train\n"
            f"The long version, including the GPU builds, is in {DEBIAN_README}."
        )
    else:
        fix = (
            f"    {sys.executable} -m pip install {package}\n"
            f"It is deliberately not a hard dependency — see docs/PACKAGING.md — so "
            f"`pip install missile-defense` stays cheap for anyone who only wants the "
            f"environment."
        )
    return (
        f"{what} needs {package}, which is not installed for this interpreter:\n"
        f"{fix}\n"
        f"(looked for the module `{module}`)"
    )


def train(argv: list[str] | None = None) -> int:
    """``md-train`` — the training loop, if this interpreter can run one."""
    if importlib.util.find_spec("torch") is None:
        print(explain_missing("torch", TRAIN_REQUIRES, "Training"), file=sys.stderr)
        return 1
    from .train import main

    return main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(train())
