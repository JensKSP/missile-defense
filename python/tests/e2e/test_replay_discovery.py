# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The game finds the recordings the trainer can see — all of them.

The browser used to scan only the top of the runs directory, while every run
the trainer creates keeps its episodes one level down, in a directory of its
own. So for anyone who had used the trainer — which is everyone the feature is
*for* — REPLAYS was empty, and there was nothing on screen to suggest why.

Discovery is asserted through `--report` rather than a screenshot: the count is
the claim, and a count is checkable. Two layouts are exercised because both are
real, and the deliberately-not-recursive rule is checked too, because "find my
recordings" and "walk my home directory" are one careless line apart.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from .harness import assert_clean, needs_app, needs_display, run_app

pytestmark = [pytest.mark.e2e, needs_app, needs_display]


def _recording(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


@pytest.fixture(scope="module")
def episode(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real recording, reused as the body of every file below.

    What is under test is *discovery*, not playback — the same bytes in five
    places is the cleanest way to make the count the only variable.
    """
    import numpy as np
    from missile_defense.env import VecEnv

    out = tmp_path_factory.mktemp("episode") / "update-00025.mdr"
    # A recording is only written when an episode *ends*, so this steps until one
    # does rather than for a round number of steps. 120 steps happened to finish
    # an episode on one machine and not on a CI runner, where every test in this
    # file then failed on a fixture that had nothing to do with discovery.
    env = VecEnv(1, seed=99, max_ticks=600)
    env.record(0)
    rng = np.random.default_rng(5)
    for _ in range(400):
        _, _, terminated, truncated, _ = env.step(rng.integers(0, 900, size=1))
        if bool(terminated[0]) or bool(truncated[0]):
            break
    else:  # `max_ticks` guarantees truncation long before this
        pytest.fail("no episode finished, so there is nothing to discover")
    assert env.save_recording(0, out, update=25, label="episode")
    return out


def test_an_empty_library_finds_nothing_and_does_not_crash(tmp_path: Path) -> None:
    run = run_app(frames=60, sandbox=tmp_path)
    assert_clean(run)
    assert run.report["replays"] == 0


def test_recordings_inside_managed_runs_are_found(episode: Path, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _recording(episode, runs / "loose.mdr")  # the flat layout, `--out runs/`
    _recording(episode, runs / "run-alpha" / "update-00025.mdr")  # the trainer's
    _recording(episode, runs / "run-beta" / "update-00080.mdr")

    run = run_app(frames=60, sandbox=tmp_path)
    assert_clean(run)
    assert run.report["replays"] == 3, run.report


def test_discovery_stops_one_level_down(episode: Path, tmp_path: Path) -> None:
    # "Find my recordings" and "walk my home directory" are one careless line
    # apart. A runs directory is a place a person points at, not a tree to crawl.
    runs = tmp_path / "runs"
    _recording(episode, runs / "run-alpha" / "update-00025.mdr")
    _recording(episode, runs / "run-alpha" / "deeper" / "update-00050.mdr")

    run = run_app(frames=60, sandbox=tmp_path)
    assert_clean(run)
    assert run.report["replays"] == 1, run.report


def test_a_symlinked_run_directory_is_not_followed(episode: Path, tmp_path: Path) -> None:
    # The containment rule. A link is somebody else's directory wearing a name
    # inside this one, and following it makes the browser's contents unbounded.
    runs = tmp_path / "runs"
    outside = tmp_path / "elsewhere"
    _recording(episode, outside / "update-00025.mdr")
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "linked").symlink_to(outside, target_is_directory=True)

    run = run_app(frames=60, sandbox=tmp_path)
    assert_clean(run)
    assert run.report["replays"] == 0, run.report
