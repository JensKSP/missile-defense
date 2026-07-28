# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Pausing, resuming and stopping a run from outside it.

The training loop owns the process; everything else asks politely, through a
file in the run directory that the loop checks once per update. An update is
seconds, so that granularity is plenty, and a file works identically on Windows
and Linux, survives a trainer crash, and can be produced by a shell:

    touch runs/STOP        # finish this update, checkpoint, flush, exit
    touch runs/PAUSE       # block between updates
    rm runs/PAUSE          # carry on

The trainer's buttons write exactly these files. That ordering matters: the
mechanism has to work without the UI, or the UI stops being a convenience and
starts being the only way in (docs/ROADMAP.md, M8).

**Why marker files rather than the `control.json` the plan sketched.** A JSON
file has to be parsed, and a reader can catch the writer mid-write — the same
torn read `missile_defense.ui.sources` goes to trouble to avoid, except that here the damage
is an unreadable *command* rather than one wrong data point. The existence of a
file needs no parsing and no protocol, and `touch` produces it.

Each is a **state**, not an event: paused stays paused while the file is there,
which is also what makes it inspectable — `ls runs/` tells you why nothing is
happening. Both are cleared when a run starts, so a stale ``STOP`` from last week
cannot kill a fresh run before its first update.

Passing a *value* is the one thing a marker cannot do, so ``TUNING.json`` sits
beside them for the settings a run can change while it is going:

    cat runs/TUNING.json                          # what the loop is using
    echo '{"eval_every": 10}' > runs/TUNING.json  # score it five times as often

It is written to a temporary name and renamed into place, so a reader never
catches it half-written, and an unreadable file is treated as no file at all: the
loop falls back to what it was started with rather than dying over a typo. A
starting run publishes its own command-line values here, which is what makes the
file the answer to *"what is this run actually using?"* rather than a pile of
overrides whose absence means something else.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

#: Marker file names, relative to a run's ``--out-dir``.
#: The PID of the training process that owns this run. The trainer reads it to
#: answer "is
#: anything actually running in here?" rather than guessing from how long ago a
#: file was touched — a guess that lags by however long the threshold is and,
#: worse, calls a *slow* run dead.
#:
#: **Deliberately never deleted.** Removing it on a clean exit sounds tidier and
#: is worse: the file's absence would then mean either "an old trainer that never
#: wrote one" or "a new one that finished", and the trainer cannot tell those
#: apart — so it would fall back to the timestamp and go on claiming a finished
#: run was live for another ninety seconds, which is the bug this replaces. A
#: dead PID says "finished" immediately and says it for ever.
#:
#: The cost is PID reuse: if the operating system hands this number to something
#: else, a finished run reads as live again. That needs a wrap-around of the PID
#: space between a run ending and someone looking, and the consequence is a
#: stale label rather than a wrong action.
RUNNING_NAME = "RUNNING"

PAUSE_NAME = "PAUSE"
STOP_NAME = "STOP"
#: The one control that carries numbers. Upper case like the markers, because it
#: is a control file and belongs with them in a listing.
TUNING_NAME = "TUNING.json"


class Control:
    """The control files for one run directory.

    Both sides use this: the trainer to ask *"should I still be running?"*, the
    trainer to say *"no"*. Neither imports the other.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    @property
    def pause_file(self) -> Path:
        return self.run_dir / PAUSE_NAME

    @property
    def running_file(self) -> Path:
        return self.run_dir / RUNNING_NAME

    def claim(self) -> None:
        """Record that this process owns the run. Call once at start-up."""
        self.running_file.parent.mkdir(parents=True, exist_ok=True)
        self.running_file.write_text(f"{os.getpid()}\n", encoding="utf-8")

    def owner(self) -> int | None:
        """The PID that claimed this run, or None if nobody has."""
        try:
            return int(self.running_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def running(self) -> bool:
        """Whether a live process owns this run.

        The PID is checked, not merely the file: a trainer killed with SIGKILL,
        or a machine that lost power, leaves the marker behind. A marker whose
        process is gone is a leftover, and reporting it as a running run would
        be the same lie in the other direction.
        """
        pid = self.owner()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)  # signal 0 asks "does it exist?" and sends nothing
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # alive, just not ours to signal
        except OSError:
            return False
        return True

    @property
    def stop_file(self) -> Path:
        return self.run_dir / STOP_NAME

    @property
    def tuning_file(self) -> Path:
        return self.run_dir / TUNING_NAME

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

    # ---- what the trainer (or a shell) says --------------------------------
    def request_pause(self) -> None:
        self._write(self.pause_file)

    def resume(self) -> None:
        self.pause_file.unlink(missing_ok=True)

    def request_stop(self) -> None:
        self._write(self.stop_file)

    def clear(self) -> None:
        """Forget every request — what a starting or finishing run does.

        The tuning file is not a request and is not cleared here: it describes
        what a run *is using*, and a starting run overwrites it with its own
        values (:meth:`publish_tuning`) rather than finding it gone.
        """
        self.pause_file.unlink(missing_ok=True)
        self.stop_file.unlink(missing_ok=True)

    def _write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    # ---- values, rather than signals ---------------------------------------
    def tuning(self) -> dict[str, int]:
        """The live settings this run directory carries, if any.

        Unreadable is the same as absent. A file caught mid-rename, truncated by
        a full disk or edited into invalid JSON by hand must never take down a
        run that is hours old — the caller has its own value to fall back on.
        """
        try:
            loaded: object = json.loads(self.tuning_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(loaded, dict):
            return {}
        return {
            name: value
            for name, value in cast("dict[str, object]", loaded).items()
            # bool is an int in Python, and nothing here is a flag.
            if isinstance(value, int) and not isinstance(value, bool)
        }

    def tuned(self, name: str, fallback: int) -> int:
        """One live setting, or what the run was started with."""
        return self.tuning().get(name, fallback)

    def publish_tuning(self, values: Mapping[str, int]) -> None:
        """Say what this run is using — what a starting run does.

        Overwrites: the command line that just started a run is the truth about
        it, and a leftover file from the last run in the same directory must not
        quietly outrank a flag someone typed.
        """
        self._write_tuning(dict(values))

    def tune(self, name: str, value: int) -> None:
        """Change one setting, leaving the others in the file alone."""
        values = self.tuning()
        values[name] = int(value)
        self._write_tuning(values)

    def _write_tuning(self, values: dict[str, int]) -> None:
        """Whole file at a time, through a rename, so no reader sees it torn."""
        self.tuning_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.tuning_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.tuning_file)
