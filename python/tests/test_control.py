# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for the control files — the half of the trainer that is not the trainer.

`md.control` is what makes `touch runs/STOP` and the Stop button the same act.
It holds no torch and no Qt, so the protocol is testable on its own, which is the
point: the mechanism has to work before any button does.
"""

from __future__ import annotations

import json
from pathlib import Path

from md.control import PAUSE_NAME, STOP_NAME, TUNING_NAME, Control


def test_a_quiet_directory_asks_for_nothing(tmp_path: Path) -> None:
    control = Control(tmp_path)
    assert not control.paused()
    assert not control.stopping()


def test_the_files_are_the_ones_a_shell_would_touch(tmp_path: Path) -> None:
    # The names are the interface. A run is stopped from a terminal with
    # `touch runs/STOP`, and the trainer is only a convenience over that.
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
    # The trainer can be pointed at a directory the trainer has not made yet.
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
    assert (PAUSE_NAME, STOP_NAME, TUNING_NAME) == ("PAUSE", "STOP", "TUNING.json")


# ---- the one control that carries a value ------------------------------------
def test_a_setting_survives_the_round_trip_as_json_anyone_can_read(tmp_path: Path) -> None:
    # Readable and writable with `cat` and `echo` for the same reason the markers
    # are `touch`-able: the trainer must not be the only way in.
    control = Control(tmp_path)
    control.tune("eval_every", 10)
    assert control.tuned("eval_every", 50) == 10
    assert json.loads((tmp_path / TUNING_NAME).read_text(encoding="utf-8")) == {"eval_every": 10}


def test_an_unset_run_uses_what_it_was_started_with(tmp_path: Path) -> None:
    assert Control(tmp_path).tuned("eval_every", 50) == 50


def test_a_mangled_file_is_the_same_as_no_file(tmp_path: Path) -> None:
    # An hours-old run must not die because a hand-edit left a trailing comma —
    # and the value it falls back to is the one it was started with anyway.
    (tmp_path / TUNING_NAME).write_text("{eval_every: 10,}", encoding="utf-8")
    control = Control(tmp_path)
    assert control.tuning() == {}
    assert control.tuned("eval_every", 50) == 50


def test_values_that_are_not_whole_numbers_are_ignored(tmp_path: Path) -> None:
    # A cadence is a count of updates. Anything else in the file is somebody
    # else's key or a typo, and either way it is not an interval.
    (tmp_path / TUNING_NAME).write_text(
        json.dumps({"eval_every": "ten", "record_every": 25, "flag": True}), encoding="utf-8"
    )
    assert Control(tmp_path).tuning() == {"record_every": 25}


def test_publishing_states_the_whole_truth_about_a_starting_run(tmp_path: Path) -> None:
    # What a run does at startup: yesterday's leftover must not outrank the flag
    # that was just typed.
    control = Control(tmp_path)
    control.tune("eval_every", 200)
    control.publish_tuning({"eval_every": 10})
    assert control.tuning() == {"eval_every": 10}


def test_tuning_one_setting_leaves_the_others_alone(tmp_path: Path) -> None:
    control = Control(tmp_path)
    control.publish_tuning({"eval_every": 50, "record_every": 25})
    control.tune("eval_every", 10)
    assert control.tuning() == {"eval_every": 10, "record_every": 25}


def test_a_write_leaves_nothing_half_written_behind(tmp_path: Path) -> None:
    # The rename is the point: a reader polling once a second must never catch
    # this file mid-write, and the temporary it goes through is not left lying
    # in the run directory afterwards.
    control = Control(tmp_path)
    control.tune("eval_every", 10)
    assert sorted(path.name for path in tmp_path.iterdir()) == [TUNING_NAME]


def test_tuning_is_not_a_request_and_survives_clearing(tmp_path: Path) -> None:
    # PAUSE and STOP are things you ask for; this describes what the run is
    # using. A finishing run clears the questions, not the answer.
    control = Control(tmp_path)
    control.tune("eval_every", 10)
    control.clear()
    assert control.tuned("eval_every", 50) == 10
