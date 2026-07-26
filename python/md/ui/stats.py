# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""What a run's statistics *mean*, derived once. No Qt in here.

`evals.csv` gained the full per-episode statistics in M8 — how long episodes
survived, how much damage they took, how the ammunition was spent — and until
now nothing read them. This module is the half of that view which is arithmetic
rather than pixels: bins and shares, a summary card's worth of stats, the curves
worth drawing, and the deltas against another run.

**Absence is the interesting case, and there are three of them.** They look
alike from a distance and mean entirely different things, so each gets its own
answer rather than a zero:

* *The column is not there.* A run written before these statistics existed. The
  honest answer is nothing at all — an empty state naming what would fill it.
* *The column is there and empty.* The trainer writes a row before the first
  evaluation finishes. Same treatment, different cause.
* *The measurement is genuinely zero.* A policy that fired no shots has a
  histogram of five zeroes, and that is a **finding**. Reporting it as "no data"
  would hide the most alarming thing a run can tell you.

Everything is a pure function of rows the tail already parsed, so the numbers on
the most-read screen in the console are covered by pytest rather than by eye.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import theme
from .sources import EvalRow

#: What a stat shows when nothing behind it exists. An em dash rather than "0",
#: because the difference between "did not happen" and "was not recorded" is the
#: whole reason a plateau is diagnosable at all.
ABSENT = "—"

#: The simulation's fixed timestep. `mean_ticks` counts ticks; people count
#: minutes, and 11,633 is not a duration anyone can feel.
TICKS_PER_SECOND = 60.0

#: The histogram's last bin is open-ended: a shot that killed five threats is
#: counted here as four. Everything derived from it is therefore a *floor*, and
#: the label says `4+` so the screen never implies otherwise.
BIN_LABELS = ("0", "1", "2", "3", "4+")
BIN_FIELDS = ("shots_0kill", "shots_1kill", "shots_2kill", "shots_3kill", "shots_4plus")


# ---- the kills-per-shot distribution ----------------------------------------


@dataclass(frozen=True)
class Bin:
    """One column of the distribution: shots that killed exactly this many."""

    label: str
    #: How many threats a shot in this bin killed. The last bin means "at least".
    kills: int
    shots: int
    #: Of every shot fired. ``None`` only when nothing was fired at all, which
    #: is a share that does not exist rather than one that is zero.
    share: float | None


@dataclass(frozen=True)
class Distribution:
    """How the ammunition was spent, over the whole seed set.

    The single clearest read on "is it catching clusters?" — a policy that fires
    one interceptor per incoming warhead and a policy that waits for two to
    overlap have the same score curve for a long time and completely different
    distributions.
    """

    bins: tuple[Bin, ...]
    shots: int
    wasted_share: float | None
    hit_share: float | None
    #: A floor, because the last bin is open-ended. See :data:`BIN_LABELS`.
    mean_kills_per_shot: float | None


def distribution(row: EvalRow | None) -> Distribution | None:
    """The histogram from ``row``, or ``None`` when that run never wrote one."""
    if row is None:
        return None
    counts = [getattr(row, field) for field in BIN_FIELDS]
    if all(count is None for count in counts):
        return None  # an older run: the columns do not exist
    # A column added later must not invalidate the ones already present, so a
    # missing bin is zero *within* a histogram that does exist.
    filled = [0 if count is None else int(count) for count in counts]
    total = sum(filled)
    bins = tuple(
        Bin(
            label=label,
            kills=index,
            shots=shots,
            share=None if total == 0 else shots / total,
        )
        for index, (label, shots) in enumerate(zip(BIN_LABELS, filled, strict=True))
    )
    if total == 0:
        return Distribution(bins, 0, None, None, None)
    wasted = filled[0] / total
    kills = sum(index * shots for index, shots in enumerate(filled))
    return Distribution(bins, total, wasted, 1.0 - wasted, kills / total)


# ---- the summary card --------------------------------------------------------


@dataclass(frozen=True)
class Stat:
    """One number on the card, already formatted and already explained."""

    key: str
    label: str
    value: float | None
    text: str
    #: What the number is, in the fewest words that make it unambiguous. The
    #: card is read by someone deciding whether a run is worth continuing, and
    #: "cities lost 11.9" invites "out of how many?" unless it says.
    note: str
    #: How :attr:`text` was produced, kept so a *difference* from another run can
    #: be printed the same way. Without it a delta falls back to a general float
    #: format and a 48,240-point lead comes out as `+4.824e+04`.
    spec: str = "{:,.0f}"


#: The one stat whose text is not a format string. A *difference* between two
#: durations is still a duration, so a delta of 6,320 ticks prints as `1:45`
#: rather than as a tick count nobody converts in their head.
DURATION_SPEC = "duration"


def _duration(ticks: float) -> str:
    """``11633`` ticks → ``3:14``."""
    seconds = int(round(ticks / TICKS_PER_SECOND))
    return f"{seconds // 60}:{seconds % 60:02d}"


def summary(row: EvalRow | None) -> tuple[Stat, ...]:
    """The full latest-eval stat block, in the order the card asks its questions.

    Three groups, and the order is the argument: **what it achieved** (score,
    how long it lasted, how far it got), **what it cost** (cities and bases),
    **how it shot** (spend, hit rate, waste). A plateau shows up as the first
    group flat while one of the other two moves.
    """
    if row is None:
        return ()

    wasted = (
        None if row.mean_shots is None or row.mean_hits is None else row.mean_shots - row.mean_hits
    )

    def stat(key: str, label: str, value: float | None, spec: str, note: str) -> Stat:
        text = ABSENT if value is None else spec.format(value)
        return Stat(key, label, value, text, note, spec)

    return (
        stat(
            "score", "Score", row.mean_score, "{:,.0f}", f"mean of {row.episodes or '?'} episodes"
        ),
        Stat(
            "survived",
            "Survived",
            row.mean_ticks,
            ABSENT if row.mean_ticks is None else _duration(row.mean_ticks),
            "mean episode length, at 60 Hz",
            DURATION_SPEC,
        ),
        stat("wave", "Wave reached", row.mean_wave, "{:.1f}", "last level entered"),
        stat("waves_cleared", "Waves cleared", row.mean_waves_cleared, "{:.1f}", "fully survived"),
        stat("cities_lost", "Cities lost", row.mean_cities_lost, "{:.1f}", "of 6, plus rebuilt"),
        stat("cities_left", "Cities left", row.mean_cities_left, "{:.1f}", "standing at the end"),
        stat("bases_lost", "Bases lost", row.mean_bases_lost, "{:.1f}", "of 3"),
        stat("bonus_cities", "Cities rebuilt", row.mean_bonus_cities, "{:.1f}", "earned by score"),
        stat("shots", "Shots fired", row.mean_shots, "{:,.0f}", "interceptors launched"),
        stat("kills", "Kills", row.mean_kills, "{:,.0f}", "threats destroyed"),
        stat("hit_rate", "Hit rate", row.mean_hit_rate, "{:.1%}", "shots that killed anything"),
        stat("wasted", "Wasted", wasted, "{:,.0f}", "shots that killed nothing"),
        stat("ammo_left", "Ammo left", row.mean_ammo_left, "{:.1f}", "unspent at the end"),
        stat("mirv_splits", "MIRV splits", row.mean_mirv_splits, "{:.1f}", "faced per episode"),
    )


# ---- the curves --------------------------------------------------------------


@dataclass(frozen=True)
class Curve:
    """A field worth plotting against the update number, and how to draw it."""

    key: str
    title: str
    colour: str
    value_format: str


#: The four that answer "why did this stop improving?", which is the question a
#: score curve raises and cannot answer. Deliberately not every column: survival
#: and damage explain a plateau, spend explains a *style*, and the distribution
#: below already covers spend far better than a fifth line would.
CURVES: tuple[Curve, ...] = (
    Curve("mean_ticks", "ticks survived", theme.INTERCEPTOR, "%.0f"),
    Curve("mean_waves_cleared", "waves cleared", theme.SMART, "%.1f"),
    Curve("mean_cities_lost", "cities lost", theme.THREAT, "%.1f"),
    Curve("mean_bases_lost", "bases lost", theme.MIRV, "%.1f"),
)


def series(rows: Sequence[EvalRow], key: str) -> list[tuple[int, float | None]]:
    """``[(update, value), …]`` for one field, with gaps preserved.

    Empty when *no* row has the field, so the chart shows its empty state rather
    than an axis with nothing on it. A gap inside a curve that does exist stays
    a gap: a column that appeared halfway through a run leaves a hole, and
    drawing the hole as zero would invent a collapse the policy never had.
    """
    points = [(row.update, getattr(row, key, None)) for row in rows]
    if all(value is None for _, value in points):
        return []
    return points


# ---- comparing two runs ------------------------------------------------------

#: Stats where a *smaller* number is the better one. Without this the card
#: would paint "one fewer city lost" in the colour it uses for a regression.
LOWER_IS_BETTER = frozenset({"cities_lost", "bases_lost", "wasted", "shots"})


@dataclass(frozen=True)
class Comparison:
    """One stat held against another run's, and which way that lands."""

    key: str
    label: str
    delta: float | None
    #: ``None`` when there is nothing to say — either run may lack the column,
    #: and a comparison against a missing number is not a tie.
    better: bool | None
    text: str


#: An exact tie, spelled out. "+0" reads as a rounding artefact; two runs losing
#: the same number of cities is a real and occasionally interesting finding.
UNCHANGED = "no change"


def _delta_text(delta: float, spec: str) -> str:
    """A difference, printed the way the stat itself is printed.

    Three cases, and each is a bug that was on the screen before it was fixed:
    a duration difference is a duration (`+1:45`, not `+6320`); a percentage
    difference is *percentage points* and saying `+20.4%` invites reading it as
    a fifth again; and everything else must not fall back to a general float
    format, where a 48,240-point lead comes out as `+4.824e+04`.
    """
    if delta == 0:
        return UNCHANGED
    sign = "+" if delta > 0 else "−"
    if spec == DURATION_SPEC:
        return f"{sign}{_duration(abs(delta))}"
    if spec.endswith("%}"):
        return f"{sign}{abs(delta) * 100:.1f} pp"
    return f"{sign}{spec.format(abs(delta))}"


def compare(row: EvalRow | None, other: EvalRow | None) -> tuple[Comparison, ...]:
    """``row`` against ``other``, stat by stat.

    Both sides go through :func:`summary`, so the comparison can only ever be
    between two things the card already shows — there is no second definition of
    "wasted" that could drift from the first.
    """
    if row is None or other is None:
        return ()
    theirs = {stat.key: stat.value for stat in summary(other)}
    found: list[Comparison] = []
    for stat in summary(row):
        mine = stat.value
        yours = theirs.get(stat.key)
        if mine is None or yours is None:
            found.append(Comparison(stat.key, stat.label, None, None, ABSENT))
            continue
        delta = mine - yours
        # An exact tie is neither ahead nor behind, and saying "better" of it
        # would be the card taking a side it has no evidence for.
        better = None if delta == 0 else (delta < 0) == (stat.key in LOWER_IS_BETTER)
        found.append(Comparison(stat.key, stat.label, delta, better, _delta_text(delta, stat.spec)))
    return tuple(found)
