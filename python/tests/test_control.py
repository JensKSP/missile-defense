# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for the control files — the half of the console that is not the console.

`md.control` is what makes `touch runs/STOP` and the Stop button the same act.
It holds no torch and no Qt, so the protocol is testable on its own, which is the
point: the mechanism has to work before any button does.
"""

from __future__ import annotations

from pathlib import Path

from md.control import PAUSE_NAME, STOP_NAME, Control


def test_a_quiet_directory_asks_for_nothing(tmp_path: Path) -> None:
    control = Control(tmp_path)
    assert not control.paused()
    assert not control.stopping()


def test_the_files_are_the_ones_a_shell_would_touch(tmp_path: Path) -> None:
    # The names are the interface. A run is stopped from a terminal with
    # `touch runs/STOP`, and the console is only a convenience over that.
    control = Control(tmp_path)
    assert control.pause_file == tmp_path / "PAUSE"
    assert control.stop_file == tmp_path / "STOP"

    (tmp_path / "STOP").touch()
    assert control.stopping()


def test_pausing_and_resuming_is_the_file_appearing_and_going(tmp_path: Path) -> None:
    control = Control(tmp_path)
    control.request_pause()
    assert control.paused()
    control.resume()
    assert not control.paused()


def test_a_request_creates_the_run_directory_if_it_is_not_there_yet(tmp_path: Path) -> None:
    # The console can be pointed at a directory the trainer has not made yet.
    control = Control(tmp_path / "runs")
    control.request_stop()
    assert (tmp_path / "runs" / "STOP").exists()


def test_waiting_returns_when_the_pause_is_lifted(tmp_path: Path) -> None:
    control = Control(tmp_path)
    control.request_pause()
    ticks: list[float] = []

    def sleep(seconds: float) -> None:
        ticks.append(seconds)
        if len(ticks) == 3:  # something outside deletes the file
            control.resume()

    assert control.wait_while_paused(sleep=sleep) is False
    assert len(ticks) == 3


def test_a_stop_gets_a_paused_run_moving_again(tmp_path: Path) -> None:
    # Otherwise Stop would appear to do nothing on a paused run, and the only way
    # out would be the kill the graceful stop exists to avoid.
    control = Control(tmp_path)
    control.request_pause()

    def sleep(_: float) -> None:
        control.request_stop()

    assert control.wait_while_paused(sleep=sleep) is True
    assert control.paused()  # still paused; the loop breaks out on the stop


def test_clearing_forgets_both_so_a_new_run_is_not_born_stopped(tmp_path: Path) -> None:
    control = Control(tmp_path)
    control.request_pause()
    control.request_stop()
    control.clear()
    assert not control.paused()
    assert not control.stopping()
    control.clear()  # and clearing nothing is not an error


def test_names_are_upper_case_so_they_stand_out_in_a_listing() -> None:
    assert (PAUSE_NAME, STOP_NAME) == ("PAUSE", "STOP")
