# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""What the console looks like to the desktop: its icon, and who it says it is.

No Qt in here, deliberately, and the type-checker lists in `pyproject.toml` are
the reason to keep it that way — this module is checked everywhere, including the
one environment with no Qt in it. It is also what the facts need: the taskbar
identity has to be claimed *before* a `QApplication` exists, and the icon's path
is wanted on the failure path too, where the widget toolkit is exactly what could
not be imported.

The icon itself travels inside the package (`ui/icon.png`, drawn by
`tools/make_icon.py`). It has to: on Windows and macOS the console is a wheel,
so there is no install prefix beside it to look in, and `packaging/` exists only
in a checkout. A copy in the wheel is a few kilobytes and works everywhere the
console does.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: The console's icon, drawn by `tools/make_icon.py` and shipped in the wheel.
ICON = Path(__file__).with_name("icon.png")

#: The Linux desktop entry this application is, minus the `.desktop`.
#:
#: Handed to Qt so a Wayland compositor can match the window to the entry — and
#: therefore to `packaging/missile-defense-trainer.desktop`'s `Icon=` — instead
#: of falling back to the interpreter's identity. Without it a
#: `python3 -m missile_defense.ui` window is, as far as the shell is concerned,
#: a window belonging to `python3`.
DESKTOP_ENTRY = "missile-defense-trainer"

#: Windows' equivalent of the same idea, and the reason the taskbar shows the
#: python icon without it.
#:
#: Explorer groups windows and picks a taskbar icon by AppUserModelID, which a
#: process inherits from whatever launched it — `python.exe`, or the game. Both
#: are wrong here: the first shows Python's icon on our window, the second hides
#: the console *inside the game's taskbar button*, where a person looking for the
#: window they just opened cannot find it. Distinct from the game's own id
#: (`de.koehler-speyer.missile-defense`) for that second reason.
APP_ID = "de.koehler-speyer.missile-defense.trainer"


def claim_taskbar_identity(platform: str = sys.platform) -> bool:
    """Tell Windows this process is the trainer, not the thing that started it.

    Returns whether the call was made, so a test can see the platform switch
    without a Windows machine. Best-effort by design: this is cosmetic, and a
    console that refuses to start because a shell32 call failed would be a far
    worse bug than a wrong icon.
    """
    if platform != "win32":
        return False  # every other desktop uses the .desktop entry above
    try:
        import ctypes  # noqa: PLC0415 — Windows-only, and only on this path

        # Reached through `getattr`, not as `ctypes.windll`, and the difference
        # is the gate. `windll` is defined only in the Windows build of the
        # module, so on Linux — which is where every checker in this project
        # runs — the attribute does not exist and mypy fails the whole gate on a
        # line that is correct on the platform it executes on. The comment that
        # used to be here said "the checkers run on a machine where the
        # attribute is real"; they do not, and that is the asymmetry AGENTS.md
        # opens with.
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return False
        windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except (AttributeError, OSError):
        return False
    return True
