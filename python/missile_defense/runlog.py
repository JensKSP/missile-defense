# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""A run's own copy of what it printed. No torch, no Qt — both sides read it.

The trainer could show a log pane only for a run *it* had started, because that
is the only run whose stdout it holds a pipe to. A run started from a terminal —
which is the case the whole out-of-process design exists to support
(docs/ROADMAP.md, M8) — printed into that terminal and nowhere else, so
attaching to it gave you curves with no words beside them. And words are what
carry the eval blocks, the "recorded …" lines and, when it dies, the traceback.

So the run writes its own. ``runs/train.log`` joins ``metrics.csv`` and the
recordings as one more artifact that *is* the interface: the trainer tails it
exactly as it tails a CSV, and gets the same pane whether or not it started the
run. `tee` would have done this for one shell on one platform; a file the
trainer always writes does it for every way a run can be launched, including a
double-clicked shortcut with no console at all.

**A copy, not a redirect.** The terminal still gets every line — someone
watching a run in a shell must not have to `tail -f` to see it. The cost is that
a line is written twice, which for a few hundred bytes an update is nothing.

Line-buffered and flushed per line, for the same reason ``metrics.csv`` is: a
reader tailing it is reading *while* it is written, and a half-flushed buffer is
a log pane that stops one update behind the curve.
"""

from __future__ import annotations

import contextlib
import io
import sys
from collections.abc import Generator
from pathlib import Path
from typing import IO, TextIO

#: Beside metrics.csv in the run directory.
FILENAME = "train.log"


class Tee(io.TextIOBase):
    """Writes to two streams, and only ever *owns* the second.

    A stream that closed the terminal it was wrapping would take the rest of the
    process's output with it, so ``close`` closes the file alone.
    """

    def __init__(self, console: TextIO, log: IO[str]) -> None:
        self._console = console
        self._log = log

    def write(self, text: str, /) -> int:
        written = self._console.write(text)
        self._console.flush()
        self._log.write(text)
        # Per line rather than per buffer: something is tailing this while it is
        # being written, and an unflushed line is a pane that lags the curve.
        self._log.flush()
        return written

    def flush(self) -> None:
        self._console.flush()
        self._log.flush()

    def isatty(self) -> bool:
        # Whatever the terminal says. A progress bar asking "am I being watched"
        # must not get a different answer because a log file exists.
        return self._console.isatty()

    def fileno(self) -> int:
        """The terminal's descriptor — a subprocess inheriting stdout gets it.

        Not the log's: a child told to write here should reach the same place
        its parent's ``print`` does. Its output then bypasses the copy, which is
        the honest trade — this class tees one process's ``print``, and the only
        alternative is a pipe and a pump thread for something no run does.
        """
        return self._console.fileno()

    def close(self) -> None:
        self._log.close()


@contextlib.contextmanager
def teed(run_dir: Path, *, filename: str = FILENAME) -> Generator[Path | None]:
    """Copy ``stdout`` into ``<run_dir>/train.log`` for the duration.

    Appended, not truncated: a resumed run continues one story, exactly as
    ``metrics.csv`` does, and a run that crashed leaves its last words for the
    one that follows it to be read against.

    Yields the log's path, or ``None`` when it could not be opened — a
    read-only or full directory is a reason to lose the *copy*, never a reason
    to lose the run. Restoring the original stream is unconditional, so an
    exception on the way out still leaves the process printing normally.
    """
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        # newline="" so a line ends in exactly what it was given, on every
        # platform. A run directory is a thing you sync between machines — the
        # trainer is meant to open one copied off the Debian box — and a file
        # whose line endings depend on which machine wrote it is a file that
        # reads differently on the other one. metrics.csv is opened the same way.
        handle: IO[str] = (run_dir / filename).open(
            "a", encoding="utf-8", errors="replace", newline=""
        )
    except OSError:
        yield None
        return

    original = sys.stdout
    sys.stdout = Tee(original, handle)
    try:
        yield run_dir / filename
    finally:
        sys.stdout = original
        handle.close()
