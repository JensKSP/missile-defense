# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for the e2e harness itself — the part that reports an exit status.

Deliberately *not* marked `e2e`: nothing here starts the game or needs a display,
so these run in the fast gate. That matters, because the bug they cover was found
by a red app-e2e job that only runs on Linux with xvfb installed, and a guard
against it that only runs there would be no better than the thing it guards.

The harness used to report `xvfb-run`'s exit status as if it were the game's.
When the wrapper's temp-directory cleanup lost a race it exited 5, discarding the
game's real 2, and a test asserting the game refuses a bad argument failed while
the game was refusing it correctly. The masking works in both directions: it can
hide a genuine non-zero exit behind a 0 just as easily.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from .harness import SH, STATUS_ENV, STATUS_SHIM, _exit_status

needs_sh = pytest.mark.skipif(SH is None, reason="no POSIX shell here")


@needs_sh
def test_the_shim_records_what_the_command_returned(tmp_path: Path) -> None:
    # The shim is a string of shell threaded through `sh -c`, so its quoting is
    # exactly the kind of thing that looks right and silently records nothing.
    # Run it for real rather than trusting the string by eye.
    status = tmp_path / "status"
    result = subprocess.run(
        [str(SH), "-c", STATUS_SHIM, "sh", sys.executable, "-c", "raise SystemExit(2)"],
        env={**os.environ, STATUS_ENV: str(status)},
        capture_output=True,
        check=False,
    )
    assert status.read_text(encoding="utf-8").strip() == "2"
    # And the normal path is unchanged: the shim re-raises the status it saw, so
    # a run whose wrapper cleans up properly returns what it always did.
    assert result.returncode == 2


@needs_sh
def test_the_shim_records_success_too(tmp_path: Path) -> None:
    status = tmp_path / "status"
    subprocess.run(
        [str(SH), "-c", STATUS_SHIM, "sh", sys.executable, "-c", ""],
        env={**os.environ, STATUS_ENV: str(status)},
        capture_output=True,
        check=False,
    )
    assert status.read_text(encoding="utf-8").strip() == "0"


def test_the_recorded_status_beats_the_wrappers(tmp_path: Path) -> None:
    """The exact failure: the game returned 2, `xvfb-run` then exited 5."""
    status = tmp_path / "status"
    status.write_text("2", encoding="utf-8")
    assert _exit_status(5, status) == 2


def test_a_missing_record_falls_back_to_the_wrapper(tmp_path: Path) -> None:
    # The game never ran — xvfb-run could not start a server, or the shell was
    # killed. There is no game exit status, and inventing one would hide it.
    assert _exit_status(5, tmp_path / "absent") == 5


def test_an_unreadable_record_falls_back_to_the_wrapper(tmp_path: Path) -> None:
    status = tmp_path / "status"
    status.write_text("", encoding="utf-8")
    assert _exit_status(3, status) == 3
    status.write_text("not a number", encoding="utf-8")
    assert _exit_status(3, status) == 3


def test_without_the_wrapper_the_returncode_is_the_answer() -> None:
    # Windows and macOS render to a hidden window with nothing in front of the
    # game, so there is no wrapper to mask anything and no shim to consult.
    assert _exit_status(2, None) == 2
