# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The arithmetic behind the analysis view, with no display anywhere near it.

Everything the analysis screen puts on the glass is derived here first, for the
same reason `missile_defense.ui.sources` exists: a number that is wrong is wrong whether or
not a window is open, and a widget is a terrible place to keep a calculation.

The cases that matter are all about *absence*. A run written by an older trainer
has none of these columns; a run that has just started has an `evals.csv` with a
header and nothing under it; a policy that has fired no shots at all has a
histogram of five zeroes, which is a real state and not a missing one. Each has
to come out as an empty state rather than as a plausible-looking zero.
"""

from __future__ import annotations

import pytest
from missile_defense.ui import stats
from missile_defense.ui.sources import EvalRow


def row(**fields: float | int | str | None) -> EvalRow:
    """An eval row with only the fields a test cares about set.

    Everything else stays `None`, which is what an older run's CSV produces —
    so a test that forgets a column is testing the degraded case by accident
    rather than passing on data no run ever wrote.
    """
    defaults: dict[str, float | int | str | None] = {
        "update": 100,
        "mean_score": 50_000.0,
        "min_score": None,
        "max_score": None,
        "mean_wave": None,
        "mean_cities_left": None,
        "mean_accuracy": None,
        "survived": None,
        "episodes": 32,
    }
    defaults.update(fields)
    return EvalRow(**defaults)  # type: ignore[arg-type]


FULL = {
    "mean_ticks": 11_633.2,
    "mean_waves_cleared": 12.156,
    "mean_wave": 13.156,
    "mean_cities_left": 0.0,
    "mean_cities_lost": 11.906,
    "mean_bases_left": 1.156,
    "mean_bases_lost": 4.906,
    "mean_ammo_left": 2.03,
    "mean_bonus_cities": 5.906,
    "mean_mirv_splits": 6.656,
    "mean_shots": 341.72,
    "mean_kills": 245.94,
    "mean_hits": 222.56,
    "mean_hit_rate": 0.6495,
    "shots_0kill": 3807,
    "shots_1kill": 6394,
    "shots_2kill": 708,
    "shots_3kill": 20,
    "shots_4plus": 0,
}


# ---- the kills-per-shot distribution ----------------------------------------


def test_the_distribution_has_a_bin_per_outcome_in_order() -> None:
    found = stats.distribution(row(**FULL))
    assert found is not None
    assert [b.label for b in found.bins] == ["0", "1", "2", "3", "4+"]
    assert [b.shots for b in found.bins] == [3807, 6394, 708, 20, 0]


def test_each_bin_carries_its_share_of_every_shot_fired() -> None:
    found = stats.distribution(row(**FULL))
    assert found is not None
    assert found.shots == 3807 + 6394 + 708 + 20
    assert found.bins[0].share == pytest.approx(3807 / found.shots)
    assert sum(b.share for b in found.bins) == pytest.approx(1.0)


def test_the_wasted_share_is_the_zero_kill_bin() -> None:
    """`shots_0kill` is ammunition that hit nothing — the number to watch."""
    found = stats.distribution(row(**FULL))
    assert found is not None
    assert found.wasted_share == pytest.approx(3807 / found.shots)
    assert found.hit_share == pytest.approx(1.0 - found.wasted_share)


def test_mean_kills_per_shot_weights_the_bins_by_their_kill_count() -> None:
    # The 4+ bin is counted as exactly 4, which understates it — so the value is
    # a floor, and the docstring says so rather than the number pretending.
    found = stats.distribution(row(**FULL))
    assert found is not None
    total = (6394 * 1) + (708 * 2) + (20 * 3) + (0 * 4)
    assert found.mean_kills_per_shot == pytest.approx(total / found.shots)


def test_a_run_with_no_histogram_has_no_distribution() -> None:
    """An older run: the columns are simply not there, and inventing zeroes
    would draw an empty chart that looks like a policy which never fires."""
    assert stats.distribution(row()) is None


def test_a_policy_that_never_fired_is_an_empty_distribution_not_a_missing_one() -> None:
    """Five real zeroes. Distinct from the case above, and it has to stay so:
    "this run predates the histogram" and "this policy did not shoot" are
    different findings and the screen must not say the same thing about both."""
    found = stats.distribution(
        row(shots_0kill=0, shots_1kill=0, shots_2kill=0, shots_3kill=0, shots_4plus=0)
    )
    assert found is not None
    assert found.shots == 0
    assert found.wasted_share is None
    assert found.mean_kills_per_shot is None
    assert all(b.share is None for b in found.bins)


def test_a_partial_histogram_is_read_as_far_as_it_goes() -> None:
    """A column added later must not invalidate the ones already there."""
    found = stats.distribution(row(shots_0kill=10, shots_1kill=90))
    assert found is not None
    assert found.shots == 100
    assert [b.shots for b in found.bins] == [10, 90, 0, 0, 0]


# ---- the summary card --------------------------------------------------------


def test_the_summary_covers_survival_damage_and_spend() -> None:
    found = {stat.key: stat for stat in stats.summary(row(**FULL))}
    # The three questions the card exists to answer, in the order it asks them.
    assert "score" in found and "survived" in found and "wave" in found
    assert "cities_lost" in found and "bases_lost" in found
    assert "shots" in found and "hit_rate" in found and "wasted" in found


def test_survival_time_is_reported_as_time_and_not_as_ticks() -> None:
    """The simulation counts ticks; a person counts minutes. 60 Hz, so 11,633
    ticks is 3:14 — and "3:14" is a fact about the run that "11633" is not."""
    found = {stat.key: stat for stat in stats.summary(row(**FULL))}
    assert found["survived"].text == "3:14"
    # Rounded rather than truncated, so a run half a second from the next minute
    # is not reported as being a whole second short of where it got.
    assert {s.key: s.text for s in stats.summary(row(mean_ticks=59.5 * 60))}["survived"] == "1:00"


def test_wasted_ammunition_is_counted_in_shots_not_in_percent() -> None:
    """`mean_shots` minus `mean_hits`: how many interceptors bought nothing.

    A percentage is already on the card as the hit rate; the absolute number is
    the one that says whether a run is *out of ammo* by wave nine.
    """
    found = {stat.key: stat for stat in stats.summary(row(**FULL))}
    assert found["wasted"].value == pytest.approx(341.72 - 222.56)


def test_a_stat_with_no_column_behind_it_says_so() -> None:
    found = {stat.key: stat for stat in stats.summary(row())}
    assert found["hit_rate"].value is None
    assert found["hit_rate"].text == stats.ABSENT


def test_the_summary_is_empty_for_no_row_at_all() -> None:
    assert stats.summary(None) == ()


# ---- the curves --------------------------------------------------------------


def test_a_curve_is_the_named_field_against_the_update() -> None:
    rows = [row(update=10, mean_ticks=100.0), row(update=20, mean_ticks=200.0)]
    assert stats.series(rows, "mean_ticks") == [(10, 100.0), (20, 200.0)]


def test_a_missing_measurement_is_a_gap_and_not_a_zero() -> None:
    """The same rule the score curve already follows: a run that gained a column
    halfway through has a hole in that curve, and drawing the hole as zero would
    invent a collapse the policy never had."""
    rows = [row(update=10), row(update=20, mean_ticks=200.0)]
    assert stats.series(rows, "mean_ticks") == [(10, None), (20, 200.0)]


def test_a_curve_nobody_has_any_data_for_is_empty() -> None:
    assert stats.series([row(update=10), row(update=20)], "mean_ticks") == []


def test_the_curves_offered_are_the_ones_that_explain_a_plateau() -> None:
    """Not every column: the view has four slots and the question it answers is
    "why did this stop improving?" — which is survival and damage, not spend."""
    keys = [curve.key for curve in stats.CURVES]
    assert keys == ["mean_ticks", "mean_waves_cleared", "mean_cities_lost", "mean_bases_lost"]
    for curve in stats.CURVES:
        assert curve.title and curve.colour and curve.value_format


# ---- comparing two runs ------------------------------------------------------


def test_a_comparison_is_the_difference_and_which_way_is_better() -> None:
    """Lower is better for the two damage stats and worse for everything else,
    and the card cannot colour a delta without being told which."""
    mine = row(**FULL)
    theirs = row(**{**FULL, "mean_score": 40_000.0, "mean_cities_lost": 13.0})
    found = {c.key: c for c in stats.compare(mine, theirs)}
    assert found["score"].delta == pytest.approx(10_000.0)
    assert found["score"].better is True
    # 11.906 lost against their 13.0 — fewer cities lost, so this run is ahead
    # even though the delta is negative.
    assert found["cities_lost"].delta == pytest.approx(11.906 - 13.0)
    assert found["cities_lost"].better is True


def test_a_delta_is_printed_the_way_its_own_stat_is_printed() -> None:
    """Three separate ways this went wrong on screen before it was fixed."""
    mine = row(**FULL)
    theirs = row(**{**FULL, "mean_score": 15_869.0, "mean_ticks": 5_313.2, "mean_hit_rate": 0.446})
    found = {c.key: c.text for c in stats.compare(mine, theirs)}
    # Not `+3.413e+04`: a general float format turns a scoreboard into notation.
    assert found["score"] == "+34,131"
    # A difference between two durations is a duration, not a tick count.
    assert found["survived"] == "+1:45"
    # And a difference between two percentages is *percentage points*. "+20.4%"
    # would be read as a fifth again, which is a different and wrong claim.
    assert found["hit_rate"] == "+20.3 pp"


def test_an_exact_tie_says_so_rather_than_showing_a_signed_zero() -> None:
    found = {c.key: c for c in stats.compare(row(**FULL), row(**FULL))}
    assert found["score"].text == stats.UNCHANGED
    assert found["score"].better is None


def test_a_comparison_against_a_run_missing_the_column_is_no_comparison() -> None:
    found = {c.key: c for c in stats.compare(row(**FULL), row())}
    assert found["hit_rate"].delta is None
    assert found["hit_rate"].better is None


def test_there_is_nothing_to_compare_without_a_second_run() -> None:
    assert stats.compare(row(**FULL), None) == ()
