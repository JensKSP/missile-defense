# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""`--silent` has to be silent from the first sample, not from the first frame.

The harness starts every game in this suite with `--silent` so that running the
tests does not come out of the speakers of whoever is at the machine. It did
anyway, and the reason was an ordering one rather than a missing flag:

* `AudioEngine`'s constructor opens the playback device **and starts it**, and
  both of its switches defaulted to on;
* that constructor runs as a member of `GameWindow`, whose own constructor then
  reads the stored preferences and probes the machine for Python interpreters —
  which launches processes;
* and `main` only reached `--silent` afterwards, in the same argument loop that
  handles `--match` and `--replay`, which the harness writes *before* it.

So every run played music from somewhere inside construction until the flag was
finally noticed. One run is a blip. A suite that opens the game a hundred times
is a concert.

`--report` now carries `audible`, which asks the engine rather than the stored
preference: the preferences said "off" the whole time the device was mixing.
"""

from __future__ import annotations

from pathlib import Path

from .harness import needs_app, run_app


@needs_app
def test_a_silent_run_never_reaches_the_speakers(tmp_path: Path) -> None:
    # `run_app` passes `--silent`, so this is the harness's own promise under
    # test. The report is written at the last frame, but `audible` is a property
    # of the engine for the whole run: nothing switches it on and back off.
    run = run_app(sandbox=tmp_path, frames=60)
    assert run.report["audible"] is False, (
        "a --silent run had sound enabled — the tests are audible again"
    )


@needs_app
def test_a_silent_run_that_actually_plays_stays_silent(tmp_path: Path) -> None:
    # The menu is the easy case. `--play` boots straight into a game, so shots,
    # detonations and a wave start all reach `handle_events` — the path that
    # queues voices — while the report still has to come back quiet.
    run = run_app("--play", sandbox=tmp_path, frames=60)
    assert run.report["audible"] is False
