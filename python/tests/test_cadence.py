# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The evaluation and recording cadence: dense early, settled at the interval."""

from __future__ import annotations

from missile_defense.runs.cadence import gap_at, is_due, schedule


def test_the_gap_reaches_the_configured_interval_at_the_ramp_end() -> None:
    assert gap_at(120, interval=20, ramp_until=120) == 20
    assert gap_at(600, interval=20, ramp_until=120) == 20, "and never exceeds it"


def test_the_gap_starts_at_one_update() -> None:
    # Not zero, and not a fraction: the first samples are every update.
    assert gap_at(1, interval=20, ramp_until=120) == 1


def test_the_gap_grows_monotonically() -> None:
    gaps = [gap_at(u, interval=20, ramp_until=120) for u in range(1, 200)]
    assert gaps == sorted(gaps)


def test_no_ramp_means_the_plain_fixed_interval() -> None:
    """The behaviour every run had before the ramp existed."""
    assert schedule(interval=20, ramp_until=0, last=100) == [1, 21, 41, 61, 81]


def test_an_interval_of_zero_never_samples() -> None:
    # `--eval-every 0` disables evaluation, and the ramp must not resurrect it.
    assert schedule(interval=0, ramp_until=120, last=500) == []
    assert not is_due(1, interval=0, ramp_until=120)


def test_the_ramp_is_denser_early_and_identical_late() -> None:
    ramped = schedule(interval=20, ramp_until=120, last=400)
    fixed = schedule(interval=20, ramp_until=0, last=400)
    early_ramped = [u for u in ramped if u <= 120]
    early_fixed = [u for u in fixed if u <= 120]
    assert len(early_ramped) > 3 * len(early_fixed)
    # Past the ramp both step by the interval; only the phase differs.
    tail = [b - a for a, b in zip(ramped, ramped[1:], strict=False) if a > 150]
    assert set(tail) == {20}


def test_is_due_agrees_with_the_schedule_it_describes() -> None:
    points = set(schedule(interval=20, ramp_until=120, last=300))
    for update in range(1, 301):
        assert is_due(update, interval=20, ramp_until=120) == (update in points)


def test_a_resumed_run_rejoins_the_schedule_rather_than_restarting_it() -> None:
    """The property that makes the ramp safe to resume into.

    `is_due` reads only the update number, so a run continued at update 650 is
    asked the same question the original would have been. A cadence that carried
    its own counter would start the ramp again there, and the chart would grow a
    second dense patch in the middle for no reason anyone could see.
    """
    for update in (300, 650, 651, 1000):
        assert is_due(update, interval=20, ramp_until=120) == (
            update in set(schedule(interval=20, ramp_until=120, last=update))
        )
