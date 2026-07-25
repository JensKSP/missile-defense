# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Pausing, resuming and stopping a run from outside it.

The training loop owns the process; everything else asks politely, through a
file in the run directory that the loop checks once per update. An update is
seconds, so that granularity is plenty, and a file works identically on Windows
and Linux, survives a console crash, and can be produced by a shell:

    touch runs/STOP        # finish this update, checkpoint, flush, exit
    touch runs/PAUSE       # block between updates
    rm runs/PAUSE          # carry on

The console's buttons write exactly these files. That ordering matters: the
mechanism has to work without the UI, or the UI stops being a convenience and
starts being the only way in (docs/ROADMAP.md, M8).

**Why marker files rather than the `control.json` the plan sketched.** A JSON
file has to be parsed, and a reader can catch the writer mid-write — the same
torn read `md.ui.sources` goes to trouble to avoid, except that here the damage
is an unreadable *command* rather than one wrong data point. The existence of a
file needs no parsing and no protocol, and `touch` produces it. If a later phase
needs to pass a *value* rather than a signal, that is the moment to add a parsed
file, written to a temporary name and renamed into place.

Each is a **state**, not an event: paused stays paused while the file is there,
which is also what makes it inspectable — `ls runs/` tells you why nothing is
happening. Both are cleared when a run starts, so a stale ``STOP`` from last week
cannot kill a fresh run before its first update.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

#: Marker file names, relative to a run's ``--out-dir``.
PAUSE_NAME = "PAUSE"
STOP_NAME = "STOP"


class Control:
    """The control files for one run directory.

    Both sides use this: the trainer to ask *"should I still be running?"*, the
    console to say *"no"*. Neither imports the other.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    @property
    def pause_file(self) -> Path:
        return self.run_dir / PAUSE_NAME

    @property
    def stop_file(self) -> Path:
        return self.run_dir / STOP_NAME

    # ---- what the trainer asks --------------------------------------------
    def paused(self) -> bool:
        return self.pause_file.exists()

    def stopping(self) -> bool:
        return self.stop_file.exists()

    def wait_while_paused(
        self,
        *,
        interval: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
    ) -> bool:
        """Block until the pause is lifted. Returns True if a stop arrived first.

        The loop blocks *between* updates rather than being suspended by the OS,
        so it keeps its allocations and its place and resumes exactly where it
        was. A SIGSTOP would freeze it mid-update in a state nothing can reason
        about.
        """
        while self.paused():
            if self.stopping():
                return True
            sleep(interval)
        return self.stopping()

    # ---- what the console (or a shell) says --------------------------------
    def request_pause(self) -> None:
        self._write(self.pause_file)

    def resume(self) -> None:
        self.pause_file.unlink(missing_ok=True)

    def request_stop(self) -> None:
        self._write(self.stop_file)

    def clear(self) -> None:
        """Forget every request — what a starting or finishing run does."""
        self.pause_file.unlink(missing_ok=True)
        self.stop_file.unlink(missing_ok=True)

    def _write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
