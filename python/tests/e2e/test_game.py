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

import os
from pathlib import Path

import pytest

from .harness import (
    PROJECT_ROOT,
    app_environ,
    assert_clean,
    needs_app,
    needs_display,
    run_app,
)

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


def test_the_game_survives_a_wayland_session(tmp_path: Path) -> None:
    """The desktop entry runs `missile-defense` with no environment at all.

    On a Wayland session that used to be a segfault during start-up, inside
    `QVulkanWindowPrivate::releaseSwapChain()` — reproducibly, on both the NVIDIA
    driver and lavapipe, so it is Qt's Vulkan/Wayland path rather than one
    vendor's. Wayland is the default session on current KDE and GNOME, so every
    one of those users met a crash instead of the game.

    Skipped where there is no Wayland session to fail on, which includes CI.

    **`assert_clean`, like every other run here, and that is newer than the
    workaround.** Releasing the instance leaves the `VkSurfaceKHR` with nobody to
    destroy it, and for a while this test carried an exception for the
    `VUID-vkDestroyInstance-instance-00629` that leak reports — the one allowed
    allow-list entry in the suite. `GameWindow::event` now destroys that surface
    itself, once the base class has taken the platform window and the swapchain
    apart (docs/WAYLAND.md), so the exception has been deleted rather than
    documented. If 00629 comes back, the reclaim stopped happening and this says
    so with no interpretation required.
    """
    if not os.environ.get("WAYLAND_DISPLAY"):
        pytest.skip("no Wayland session here — nothing for the fallback to prevent")
    environ = app_environ(tmp_path)
    del environ["QT_QPA_PLATFORM"]  # exactly what the .desktop file provides
    run = run_app("--play", frames=120, sandbox=tmp_path, environ=environ)
    assert_clean(run)
    assert run.frames >= 120


def test_every_bundled_model_plays(tmp_path: Path) -> None:
    """The game ships one policy at three stages of its training.

    A player is meant to watch the same network go from firing wildly to beating
    the hand-written expert, so all three have to load and play — not just the
    one the menu happens to open first.
    """
    models = sorted((PROJECT_ROOT / "models").glob("*.mdp"))
    if not models:
        pytest.skip("no bundled models in this tree")
    for model in models:
        run = run_app("--watch-model", str(model), frames=120, sandbox=tmp_path)
        assert_clean(run)
        assert run.ticks > 0, f"{model.name} loaded but never played"
