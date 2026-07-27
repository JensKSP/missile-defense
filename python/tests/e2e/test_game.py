# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The game, driven as a user drives it: started, played, watched, closed.

Everything here runs the **real binary** in its own process, renders real frames
through a real Vulkan device, and asserts on what the run reported. Nothing in
the tree checked any of this before: `core/tests/e2e/` drives a whole simulation,
which is a different claim from "the game starts".

The assertions are deliberately about what must hold *regardless* of timing — it
advanced, it did not crash, it exited cleanly, the driver was the one asked for.
A live game is paced by a wall clock, so the exact tick it stops on is not a
fact worth asserting. The one place that changes is a replay, which has a fixed
length and therefore exactly one right answer; that is `test_replay.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .harness import assert_clean, needs_app, needs_display, run_app

pytestmark = [pytest.mark.e2e, needs_app, needs_display]


def test_the_game_starts_shows_its_menu_and_exits(tmp_path: Path) -> None:
    # The cheapest possible claim, and the one that would have been worth having
    # first: the binary runs, creates a Vulkan device, renders, and comes back.
    run = run_app(frames=90, sandbox=tmp_path)
    assert_clean(run)
    assert run.state == "menu"
    assert run.frames >= 90


def test_a_frame_budget_is_an_upper_bound_that_actually_binds(tmp_path: Path) -> None:
    # The flag exists so a hung window fails a test instead of hanging a job, so
    # "it stopped at roughly the budget" is the property worth checking. Roughly,
    # not exactly: the close is queued and lands on the following frame.
    run = run_app(frames=30, sandbox=tmp_path)
    assert_clean(run)
    assert 30 <= run.frames <= 35


def test_play_boots_straight_into_a_game_that_advances(tmp_path: Path) -> None:
    run = run_app("--play", frames=240, sandbox=tmp_path)
    assert_clean(run)
    assert run.mode == "play"
    assert run.state == "playing"
    # A human is not moving the mouse, so nothing is fired and nothing is scored;
    # what must be true is that the simulation ran at all.
    assert run.ticks > 0


def test_watch_puts_the_scripted_agent_at_the_controls_and_it_scores(tmp_path: Path) -> None:
    # The M4 baseline through the shipped game rather than through `poe eval`:
    # same Action, same Sim::step, but driven by the app's frame loop. It kills
    # things within a couple of seconds of play, so a zero score here means the
    # agent is not actually driving.
    run = run_app("--watch", frames=600, sandbox=tmp_path)
    assert_clean(run)
    assert run.mode == "watch"
    assert run.ticks > 0
    assert run.score > 0
    assert run.cities_left == 6  # nothing has reached the ground this early


def test_a_watched_game_does_not_touch_the_high_score_table(tmp_path: Path) -> None:
    # The rule that makes the table worth having (`ai_assisted_` is sticky), and
    # it is only observable from outside: the agent plays, the game ends, and no
    # initials entry may be waiting for a human who is not there.
    run = run_app("--watch", frames=600, sandbox=tmp_path)
    assert_clean(run)
    assert run.state != "enter-score"


def test_an_unreadable_recording_is_reported_and_does_not_take_the_game_down(
    tmp_path: Path,
) -> None:
    # A recording is a build-local artifact, so meeting a stale or truncated one
    # is a normal event. It must degrade to the menu, not to a crash.
    junk = tmp_path / "not-a-recording.mdr"
    junk.write_bytes(b"this is not a recording")
    run = run_app("--replay", str(junk), frames=60, sandbox=tmp_path)
    assert_clean(run)
    assert run.state == "menu"
    assert "could not read the recording" in run.output
