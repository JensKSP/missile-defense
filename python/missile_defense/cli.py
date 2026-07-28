# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The installed commands' front door — the one place that can explain a missing
optional dependency instead of raising it.

``missile-defense-train`` and ``missile-defense-trainer`` are entry points of a
package whose heavy halves are deliberately **not** dependencies
(docs/PACKAGING.md): torch because Debian's copy is CPU-only and the vendor
wheels are gigabytes, PySide6 because it is LGPLv3 while this project is MIT. The
price of that choice is that the absence has to be *explained*, and a traceback
out of the middle of an import is not an explanation — it reads as a broken
installation rather than as an optional piece someone has not installed yet.

So the trainer window has ``missile_defense.ui.__main__`` and the training command has this.
Neither imports anything heavy until it has checked; ``find_spec`` locates a
module without importing it, so the check costs nothing when the answer is yes.
"""

from __future__ import annotations

import importlib.util
import sys
import sysconfig
from pathlib import Path

#: What each command needs, and what the message tells you to install.
TRAIN_REQUIRES = "torch"

#: Where the packaged trainer explains the same thing at length, and the only
#: path worth printing: a bare "README.Debian" is not something anyone can open.
DEBIAN_README = "/usr/share/doc/missile-defense-trainer/README.Debian"

#: The flag that turns the training command into the multi-seed experiment
#: runner. It used to be a command of its own, ``md-multiseed``, which put a
#: third name on ``PATH`` for what is the trainer run several times over — and
#: one that never reached a packaged user at all, since only the wheel installed
#: it. As a flag it ships everywhere the trainer does.
MULTISEED_FLAG = "--multiseed"


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
    rather than the wheel, the trainer's own install-a-runtime button, which is
    the whole reason that button exists.
    """
    if managed is None:
        managed = externally_managed()
    if managed:
        fix = (
            f"This interpreter is externally managed (PEP 668), so it will refuse a\n"
            f"`pip install` — correctly. Use the trainer, which installs a\n"
            f"runtime for you:\n"
            f"    missile-defense-trainer\n"
            f"or build one yourself:\n"
            f"    python3 -m venv --system-site-packages ~/.venvs/md\n"
            f"    ~/.venvs/md/bin/pip install {package}\n"
            f"    ~/.venvs/md/bin/missile-defense-train\n"
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


def split_multiseed(argv: list[str]) -> list[str] | None:
    """The multi-seed arguments, or ``None`` when this is an ordinary run.

    ``--multiseed`` counts only *before* the ``--`` separator. Everything after
    it belongs to the child trainers (missile_defense.multiseed), and a run that passes the
    literal string down to them — ``-- --multiseed`` — must reach missile_defense.train and be
    rejected there rather than silently re-entering the experiment runner here.
    """
    end = argv.index("--") if "--" in argv else len(argv)
    if MULTISEED_FLAG not in argv[:end]:
        return None
    rest = list(argv)
    rest.pop(rest.index(MULTISEED_FLAG))
    return rest


def train(argv: list[str] | None = None) -> int:
    """``missile-defense-train`` — the training loop, if this interpreter can run one."""
    arguments = sys.argv[1:] if argv is None else argv

    # Dispatched *before* the torch check, and that ordering is the point: the
    # multi-seed runner starts one trainer process per seed and takes `--python`
    # to say which interpreter runs them, so the process reading this argv never
    # imports torch and must not be refused for lacking it. Gating it here would
    # have broken the one workflow the flag exists for — driving GPU runs in a
    # managed runtime from a system Python that has no torch of its own.
    multiseed = split_multiseed(arguments)
    if multiseed is not None:
        from .multiseed import main as multiseed_main

        return multiseed_main(multiseed)

    if importlib.util.find_spec("torch") is None:
        print(explain_missing("torch", TRAIN_REQUIRES, "Training"), file=sys.stderr)
        return 1
    from .train import main

    return main(arguments)


if __name__ == "__main__":
    raise SystemExit(train())
