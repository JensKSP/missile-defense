# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""One Windows detail, in one place: children of a window must not open consoles.

Everything the console starts — a training run, a pip install into the managed
runtime, the game — is started from a windowed application, and every one of them
is a console program whose output is already being read through a pipe. Windows
allocates a console for such a child anyway, so a person watching the progress
pane in the trainer also gets a black command window per run: it steals focus
when it appears, shows nothing that is not already in the pane, and closing it
kills the run.

``CREATE_NO_WINDOW`` is the flag that says "console, but no window for it". The
pipes keep working — this suppresses the *window*, not the streams — and on every
other platform there is nothing to suppress, so the flags are zero.

Its own module because both callers need it and neither can import the other:
:mod:`missile_defense.runs.runner` already imports
:mod:`missile_defense.runs.runtime`, so the shared piece has to sit under both.
"""

from __future__ import annotations

import subprocess
import sys

#: Absent from `subprocess` on every platform but Windows, hence the `getattr`
#: rather than a bare attribute: naming it directly would make this module fail
#: to import on Linux, where it is still perfectly meaningful (as zero).
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def creation_flags(platform: str = sys.platform) -> int:
    """``creationflags`` for a child that should not open a console window.

    ``platform`` is a parameter so both branches are testable from either OS —
    the same reason :func:`missile_defense.runs.runner.launch_environ` takes one.
    """
    return CREATE_NO_WINDOW if platform == "win32" else 0
