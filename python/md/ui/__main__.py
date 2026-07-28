# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""``python -m md.ui`` — the trainer's entry point, and the one place that can
still say something useful when what it needs is not installed.

The trainer is an *optional* component: PySide6 and Qt Charts are LGPLv3 while
this project is MIT, so they are never a dependency of the game or the ``.deb``
(docs/ROADMAP.md, M8). A missing package is therefore a normal state to explain,
not a traceback to dump.

That used to be true of PySide6 alone. Everything else went out as a stack
trace, and on 2026-07-28 the one that did was ``numpy`` — a *base* dependency of
``md``, reached through ``md.league`` → ``md.policy_format``, and absent on any
machine that got the trainer from the Windows ZIP rather than from ``pip``.
Started from the game, which is a GUI-subsystem binary with no console attached,
that trace had nowhere to appear at all: the menu entry did nothing, silently.
Whatever is missing now gets named.
"""

from __future__ import annotations

import sys

#: Packages this entry point can name a fix for, most specific prefix first.
#:
#: The apt column is only consulted on an externally managed interpreter, where
#: `pip install` is refused by design and naming it would waste somebody's
#: evening. A package absent from this table still gets a sentence — it just
#: cannot get a distribution package name, because guessing one is worse than
#: saying `pip install <name>`.
_APT_PACKAGES = {
    "PySide6": "python3-pyside6.qtcharts python3-pyside6.qtwidgets",
    "numpy": "python3-numpy",
    "psutil": "python3-psutil",
}


def package_of(error: ModuleNotFoundError) -> str:
    """The top-level package to install for a failed import, or ``""``.

    A dotted import reports the dotted name — a missing ``PySide6.QtCharts`` is
    fixed by installing ``PySide6``, and quoting the submodule in a `pip install`
    would produce a command that cannot succeed.
    """
    return (error.name or "").split(".")[0]


def explain(missing: str, *, managed: bool | None = None, interpreter: str | None = None) -> str:
    """Why the trainer cannot start, and the one command that changes it.

    Both switches are injected so this is testable as itself. A test that had to
    run on Debian to check the apt branch, and on Windows to check the pip one,
    could not be written at all — and the branch nobody can test is the branch
    that names the wrong package.
    """
    if managed is None:
        # The same PEP 668 split as md.cli: on a distribution interpreter the
        # `pip install` is refused by design, and the packages are the answer.
        from ..cli import externally_managed  # noqa: PLC0415 — only on the failure path

        managed = externally_managed()
    if managed:
        apt = _APT_PACKAGES.get(missing, f"python3-{missing.lower()}")
        fix = (
            f"    sudo apt install {apt}\n"
            "(this interpreter is externally managed, so it refuses a `pip install`)"
        )
    else:
        python = sys.executable if interpreter is None else interpreter
        fix = f"    {python} -m pip install {missing}"
    aside = (
        "It is optional, and deliberately not a dependency of the game."
        if missing.startswith("PySide6")
        else "The trainer needs it; the game itself does not."
    )
    return f"The trainer needs {missing}, which this interpreter does not have:\n{fix}\n{aside}"


def main() -> int:
    try:
        from .app import main as run
    except ModuleNotFoundError as error:
        missing = package_of(error)
        if not missing:
            raise  # nothing to name, so nothing this can say that the trace cannot
        print(explain(missing), file=sys.stderr)
        return 1
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
