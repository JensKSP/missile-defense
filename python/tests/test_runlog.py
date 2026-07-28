# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for a run's own copy of what it printed.

``missile_defense.runs.runlog`` is why the trainer can show a log pane for a run started from a
terminal (docs/ROADMAP.md, M8). It imports neither torch nor Qt, so what it
promises — the terminal still gets every line, the file gets them too, and
neither the run nor the process survives losing the other — is checkable here.
"""

from __future__ import annotations

import sys
from pathlib import Path

from missile_defense.runs import runlog


def test_the_terminal_still_gets_every_line(capsys, tmp_path: Path) -> None:
    """A copy, not a redirect: watching a run in a shell must still work."""
    with runlog.teed(tmp_path):
        print("update 1 | return 4.87")
    assert "update 1 | return 4.87" in capsys.readouterr().out


def test_the_file_gets_them_too(tmp_path: Path) -> None:
    with runlog.teed(tmp_path) as path:
        print("update 1 | return 4.87")
        print("  recorded update-00025.mdr")
    assert path == tmp_path / runlog.FILENAME
    assert path is not None
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ["update 1 | return 4.87", "  recorded update-00025.mdr"]


def test_a_line_is_readable_before_the_run_ends(tmp_path: Path) -> None:
    # Something is tailing this while it is written; an unflushed line is a log
    # pane one update behind the curve.
    with runlog.teed(tmp_path) as path:
        print("update 1")
        assert path is not None
        assert path.read_text(encoding="utf-8") == "update 1\n"


def test_a_resumed_run_continues_the_same_story(tmp_path: Path) -> None:
    """Appended, like metrics.csv — the history stays whole across a resume."""
    with runlog.teed(tmp_path):
        print("first run")
    with runlog.teed(tmp_path):
        print("second run")
    text = (tmp_path / runlog.FILENAME).read_text(encoding="utf-8")
    assert text.splitlines() == ["first run", "second run"]


def test_stdout_is_put_back_afterwards(tmp_path: Path) -> None:
    original = sys.stdout
    with runlog.teed(tmp_path):
        assert sys.stdout is not original
    assert sys.stdout is original


def test_stdout_is_put_back_even_when_the_run_raises(tmp_path: Path) -> None:
    original = sys.stdout
    try:
        with runlog.teed(tmp_path):
            raise RuntimeError("out of memory")
    except RuntimeError:
        pass
    assert sys.stdout is original


def test_an_unwritable_directory_costs_the_copy_not_the_run(tmp_path: Path) -> None:
    # A read-only or full directory is a reason to lose the log, never a reason
    # to lose hours of training.
    blocked = tmp_path / "file-not-a-directory"
    blocked.write_text("", encoding="utf-8")
    with runlog.teed(blocked / "runs") as path:
        print("the run carries on")
    assert path is None


def test_the_run_directory_is_created_if_it_is_not_there(tmp_path: Path) -> None:
    target = tmp_path / "runs-3"
    with runlog.teed(target) as path:
        print("hello")
    assert path is not None and path.exists()


def test_closing_the_tee_does_not_close_the_terminal(tmp_path: Path) -> None:
    """Closing the wrapped stream would take the rest of the process with it."""
    original = sys.stdout
    with runlog.teed(tmp_path):
        pass
    assert not original.closed
    print("still printing")


def test_the_tee_reports_the_terminal_s_own_tty_answer(tmp_path: Path) -> None:
    # A progress bar asking "am I being watched" must not get a different answer
    # because a log file happens to exist.
    with runlog.teed(tmp_path):
        assert sys.stdout.isatty() == sys.__stdout__.isatty()
