# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Whether a run is going — asked of the operating system, not of a clock.

The console used to answer this by timing how long ago `metrics.csv` was last
written, with a ninety-second threshold. That is wrong in both directions, and
the second one is the dangerous half:

* a finished run went on reporting itself as live for up to ninety seconds, and
* **a slow run looked dead** — an update that takes longer than the threshold is
  a perfectly healthy large model on a modest card, and calling it stopped
  invites someone to start a second trainer into the same directory.

So the trainer names its PID in a `RUNNING` file and the console asks the OS.
"""

from __future__ import annotations

import os
from pathlib import Path

from md.control import Control


def test_a_run_nobody_claimed_is_not_running(tmp_path: Path) -> None:
    assert Control(tmp_path).owner() is None
    assert not Control(tmp_path).running()


def test_claiming_names_this_process(tmp_path: Path) -> None:
    control = Control(tmp_path)
    control.claim()
    assert control.owner() == os.getpid()
    assert control.running()


def test_the_marker_outlives_the_run_and_still_reads_as_finished(tmp_path: Path) -> None:
    """Why the marker is never deleted.

    Removing it on a clean exit would make its absence ambiguous — an old trainer
    that never wrote one, or a new one that finished — and the console would fall
    back to the timestamp and go on calling the finished run live for another
    ninety seconds. That is the bug this replaces. Keeping the file and checking
    the PID answers immediately and keeps answering.
    """
    control = Control(tmp_path)
    control.claim()
    control.running_file.write_text("2147483646\n", encoding="utf-8")  # as if it had exited
    assert control.owner() is not None, "the marker stays, so absence is unambiguous"
    assert not control.running()


def test_a_marker_left_by_a_dead_process_is_not_a_running_run(tmp_path: Path) -> None:
    """The case a plain "does the file exist?" gets wrong.

    A trainer killed outright — SIGKILL, or the machine losing power — never
    reaches its own cleanup, so the marker outlives it. Reporting that as a
    running run is the same lie as the timestamp told, pointing the other way.
    """
    control = Control(tmp_path)
    # A PID that cannot be running: the kernel's maximum is far below this, and
    # picking a number rather than spawning and killing something keeps the test
    # free of races.
    control.running_file.write_text("2147483646\n", encoding="utf-8")
    assert control.owner() == 2147483646
    assert not control.running()


def test_an_unreadable_marker_is_not_a_running_run(tmp_path: Path) -> None:
    # Half-written, truncated, or scribbled on by hand. None of those is proof
    # that a trainer is alive, and guessing "probably yes" is how a second one
    # gets started into the same directory.
    control = Control(tmp_path)
    control.running_file.write_text("not a pid\n", encoding="utf-8")
    assert control.owner() is None
    assert not control.running()
