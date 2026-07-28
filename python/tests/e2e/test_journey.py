# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The crossings: where one program's output is another program's input.

Each test here spans a boundary that no single component's tests can see across.
They are the ones worth having most, because every integration failure this
project can produce lives in exactly these seams:

* a Python trainer writes a recording, and a C++ binary has to play it;
* a run directory is written by one process and read by another;
* the trainer has to be able to launch the game on a file it found.

The first is the strongest single assertion in the suite. A recording is
`(seed, config, action indices)` and nothing else, so replaying it correctly
means the C++ simulation reproduced, tick for tick, what the Python trainer's
batched environment did — through a file, across a language boundary, in a
separate process. If anything in the simulation, the encoding or the recorder
drifts, the score comes out different.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .harness import (
    assert_clean,
    needs_app,
    needs_display,
    needs_native,
    needs_torch,
    recordings,
    run_app,
)

pytestmark = [pytest.mark.e2e, needs_torch, needs_native]


@needs_app
@needs_display
def test_a_recorded_episode_replays_in_the_game(trained_run: Path, tmp_path: Path) -> None:
    episode = recordings(trained_run)[-1]
    run = run_app("--replay", str(episode), until_done=True, frames=20_000, sandbox=tmp_path)
    assert_clean(run)
    assert run.mode == "replay"
    # --until-done means it ran out of recording rather than out of budget.
    assert run.state in ("game-over", "enter-score")
    assert run.ticks > 0


@needs_app
@needs_display
def test_the_same_recording_replays_the_same_way_every_time(
    trained_run: Path, tmp_path: Path
) -> None:
    # The determinism claim, made across the process boundary rather than inside
    # one. Frames are paced by a wall clock and are *expected* to differ; the
    # simulation is not, and every number that comes out of it must be identical.
    episode = recordings(trained_run)[-1]
    first = run_app("--replay", str(episode), until_done=True, frames=20_000, sandbox=tmp_path)
    second = run_app("--replay", str(episode), until_done=True, frames=20_000, sandbox=tmp_path)
    assert_clean(first)
    assert_clean(second)
    assert (first.ticks, first.score, first.cities_left) == (
        second.ticks,
        second.score,
        second.cities_left,
    )


@needs_app
@needs_display
def test_a_replayed_episode_is_not_eligible_for_the_high_score_table(
    trained_run: Path, tmp_path: Path
) -> None:
    # A recording is a policy playing, so it is agent-assisted by construction and
    # must never divert to initials entry — which is also what lets it be replayed
    # unattended at all.
    episode = recordings(trained_run)[-1]
    run = run_app("--replay", str(episode), until_done=True, frames=20_000, sandbox=tmp_path)
    assert_clean(run)
    assert run.state != "enter-score"


def test_the_trainer_finds_everything_the_trainer_left(trained_run: Path) -> None:
    # No Qt here on purpose: `missile_defense.runs.sources` is the half of the trainer that has
    # none, and it is the half that has to agree with the trainer about what a run
    # directory contains.
    from missile_defense.runs import sources  # noqa: PLC0415 — imported after the run exists

    metrics = sources.metrics_tail(trained_run).poll()
    evals = sources.evals_tail(trained_run).poll()
    assert metrics.rows, "the trainer read no metrics from a run that wrote some"
    assert evals.rows, "the trainer read no evaluations from a run that wrote some"
    assert evals.rows[-1].mean_score > 0.0
    assert sources.list_checkpoints(trained_run)
    assert sources.list_recordings(trained_run)


def test_the_trainer_reads_a_growing_file_without_rereading_it(trained_run: Path) -> None:
    # The tail remembers where it stopped, which is the difference between a
    # trainer that stays cheap over a run of thousands of updates and one that
    # reparses the whole file every second, forever.
    from missile_defense.runs import sources  # noqa: PLC0415

    tail = sources.metrics_tail(trained_run)
    first = tail.poll()
    assert first.rows
    assert not tail.poll().rows, "the same rows were handed out twice"


def test_the_trainer_finds_the_run_inside_a_container_directory(trained_run: Path) -> None:
    # Runs pile up one --out-dir each, so `runs/` stops being a run and becomes a
    # container of them. Pointing the trainer one level too high is the most
    # common way to see an empty window, and it is supposed to say which
    # directories below it do hold one.
    from missile_defense.runs import sources  # noqa: PLC0415

    assert trained_run in sources.find_runs(trained_run.parent)
