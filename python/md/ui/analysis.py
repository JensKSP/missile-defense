# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# pyright: reportMissingImports=false
"""The analysis screen: what the policy actually *did*, not just what it scored.

A score curve says a run stopped improving. It cannot say why, and "why" is the
only question worth asking of a plateau. The statistics that answer it have been
in `evals.csv` since M8 and until now nothing read them:

* **Survival and damage** — how long an episode lasted, how far it got, how many
  cities and bases it lost getting there. A run whose score has flattened while
  its survival time is still climbing is learning something; one where both are
  flat has stopped.
* **How the ammunition was spent** — shots fired, hit rate, and the shots that
  killed nothing. Two policies with the same score can be spending three times
  the ammunition to get it, and only this says so.
* **The kills-per-shot distribution** — the clearest single read on whether the
  policy has learned to wait for a cluster or is still trading one interceptor
  for one warhead.

Everything numeric is computed in :mod:`md.ui.stats`, which has no Qt in it and
is therefore under test. This file places widgets and does no arithmetic beyond
laying them out — the same division `charts.py` and `sources.py` already keep.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from . import sources, stats, theme
from .charts import BarView, CurveView
from .sources import EvalRow

#: What the screen says before a run has scored itself even once. Every empty
#: state in this trainer names the flag that would change it, and this one is
#: the most likely to be met: a fresh run has no evaluation for `--eval-every`
#: updates, which can be several minutes.
NO_EVALS = (
    "No evaluation yet — the run scores itself every --eval-every updates,\n"
    "and every number on this screen comes from that."
)

#: And what it says for a run that predates the statistics themselves. A
#: different sentence from the one above on purpose: "not yet" and "not ever"
#: send you to different places.
NO_STATISTICS = (
    "This run's evals.csv has no per-episode statistics in it.\n"
    "Runs started before they existed keep their score curve and lose this screen."
)

NO_DISTRIBUTION = (
    "No kills-per-shot histogram in this run.\n"
    "It is written from the blast events, so any run since M8 has one."
)

#: A policy that fired nothing at all. Rare, alarming, and *not* missing data —
#: an empty state that said "no data" here would hide the finding.
NEVER_FIRED = "This policy fired no shots at all in the last evaluation."

#: How many tiles fit across before the row wraps. Four keeps each one wide
#: enough for a thousands-separated score without the value reflowing.
TILE_COLUMNS = 4


class StatTile(QFrame):
    """One number, its name, what it means, and how it compares.

    The comparison line is why this is not a `QLabel` with a formatted string:
    it appears and disappears as a second run is picked, and it is coloured by
    *direction × whether up is good*, which the tile cannot work out for itself
    (fewer cities lost is better; fewer shots fired usually is too; a lower
    score never is). :data:`md.ui.stats.LOWER_IS_BETTER` holds that knowledge.
    """

    def __init__(self, stat: stats.Stat) -> None:
        super().__init__()
        self.setProperty("role", "tile")
        self.key = stat.key
        column = QVBoxLayout(self)
        column.setContentsMargins(10, 8, 10, 8)
        column.setSpacing(1)

        self._caption = QLabel(stat.label)
        self._caption.setProperty("role", "caption")
        self._value = QLabel(stat.text)
        self._value.setProperty("role", "value")
        self._note = QLabel(stat.note)
        self._note.setProperty("role", "note")
        self._delta = QLabel("")
        self._delta.setProperty("role", "note")
        self._delta.setVisible(False)

        for widget in (self._caption, self._value, self._note, self._delta):
            column.addWidget(widget)

    def show_stat(self, stat: stats.Stat) -> None:
        self._value.setText(stat.text)
        self._note.setText(stat.note)

    def show_comparison(self, comparison: stats.Comparison | None, name: str) -> None:
        if comparison is None or comparison.delta is None:
            self._delta.setVisible(False)
            return
        colour = (
            theme.MUTED
            if comparison.better is None
            else (theme.AHEAD if comparison.better else theme.BEHIND)
        )
        # The other run is named in the line rather than only in the picker: a
        # bare "+1,200" is unreadable a minute after the selection was made.
        self._delta.setText(f"{comparison.text} vs {name}")
        self._delta.setStyleSheet(f"color: {colour};")
        self._delta.setVisible(True)


class AnalysisView(QWidget):
    """One screen: the tiles, the distribution, and the four cause curves.

    Live, dark and empty-state-handled like every other panel — it is fed by the
    same tick that feeds the dashboard, from rows the same tail already parsed,
    so opening it costs no extra file reading and it never shows a stale number
    beside a fresh one.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tiles: dict[str, StatTile] = {}
        self._compare_name = ""

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)

        self._empty = QLabel(NO_EVALS)
        self._empty.setProperty("role", "placeholder")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self._empty)

        self._body = QWidget()
        body = QVBoxLayout(self._body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)
        column.addWidget(self._body, stretch=1)
        self._body.setVisible(False)

        # Why the comparison is showing nothing, when it is showing nothing.
        # A panel that silently stays blank leaves a person deciding between
        # "broken" and "I misunderstand this" — and both answers are wrong.
        self._note = QLabel("")
        self._note.setProperty("role", "caption")
        self._note.setWordWrap(True)
        self._note.setVisible(False)
        body.addWidget(self._note)

        self._tile_grid = QGridLayout()
        self._tile_grid.setSpacing(8)
        body.addLayout(self._tile_grid)

        lower = QHBoxLayout()
        lower.setSpacing(8)
        body.addLayout(lower, stretch=1)

        # The distribution gets its own column and a third of the width: it is
        # one chart answering one question, and squeezing it in beside four
        # curves would make five plots that are all too small to read.
        self._distribution = BarView("kills per shot", theme.CITY)
        self._distribution.setMinimumHeight(220)
        lower.addWidget(self._distribution, stretch=2)

        curves = QGridLayout()
        curves.setSpacing(8)
        lower.addLayout(curves, stretch=3)
        self._curves: dict[str, CurveView] = {}
        for index, curve in enumerate(stats.CURVES):
            view = CurveView(
                curve.title,
                curve.colour,
                value_format=curve.value_format,
                markers=True,
                from_zero=True,
            )
            view.setMinimumHeight(105)
            view.set_placeholder(NO_EVALS)
            self._curves[curve.key] = view
            curves.addWidget(view, index // 2, index % 2)

    # ---- feeding it ----------------------------------------------------------

    def show_rows(self, rows: Sequence[EvalRow], compare: Sequence[EvalRow] = ()) -> None:
        """Redraw from the run's evaluations, newest last.

        Given the whole series rather than the newest row: the tiles want the
        last one, the curves want all of them, and handing over both separately
        is how the two end up describing different updates.
        """
        latest = rows[-1] if rows else None
        self._empty.setText(NO_EVALS if latest is None else NO_STATISTICS)
        summary = stats.summary(latest)
        # A row with none of the statistics in it is a run that predates them,
        # and every tile would say "—". Better to say that once, in a sentence.
        measured = [stat for stat in summary if stat.value is not None]
        usable = latest is not None and len(measured) > 1  # the score is always there
        self._empty.setVisible(not usable)
        self._body.setVisible(usable)
        if not usable:
            return

        self._show_tiles(summary)
        other = compare[-1] if compare else None
        comparable = (
            latest is not None
            and other is not None
            and sources.matching_eval_protocol(latest, other)
        )
        self._note.setText(self._comparison_note(other, comparable=comparable))
        self._note.setVisible(bool(self._note.text()))

        # Nothing is overlaid across protocols, deliberately: two runs measured
        # over different seed sets, cadences or caps are not two numbers, and a
        # delta between them would be a claim the data does not support. The
        # note above says so, which is the part that used to be missing.
        held = other if comparable else None
        comparisons = {c.key: c for c in stats.compare(latest, held)}
        for key, tile in self._tiles.items():
            tile.show_comparison(comparisons.get(key), self._compare_name)

        self._show_distribution(latest, held)
        self._show_curves(rows, compare if comparable else ())

    def _comparison_note(self, other: EvalRow | None, *, comparable: bool) -> str:
        """One sentence naming what the comparison is, or why there is none."""
        if not self._compare_name:
            return ""
        if other is None:
            return (
                f"{self._compare_name} has no evaluated checkpoint yet, so there is "
                "nothing to compare against. Evaluate it and this fills in."
            )
        if not comparable:
            return (
                f"{self._compare_name} was evaluated under a different protocol — "
                "a different seed set, decision cadence or tick cap — so its numbers "
                "are not comparable with these and nothing is overlaid. "
                "Re-evaluate either run under the canonical protocol to compare them."
            )
        return f"held against {self._compare_name}"

    def set_comparison(self, name: str) -> None:
        """Name the run being held against this one; empty clears it."""
        self._compare_name = name
        for view in self._curves.values():
            if name:
                view.set_comparison(name)
            else:
                view.clear_comparison()

    # ---- the three halves ----------------------------------------------------

    def _show_tiles(self, summary: Sequence[stats.Stat]) -> None:
        for index, stat in enumerate(summary):
            tile = self._tiles.get(stat.key)
            if tile is None:
                tile = StatTile(stat)
                self._tiles[stat.key] = tile
                self._tile_grid.addWidget(tile, index // TILE_COLUMNS, index % TILE_COLUMNS)
            else:
                tile.show_stat(stat)

    def _show_distribution(self, latest: EvalRow | None, other: EvalRow | None) -> None:
        found = stats.distribution(latest)
        if found is None:
            self._distribution.set_placeholder(NO_DISTRIBUTION)
            self._distribution.set_note("")
            return
        if found.shots == 0:
            self._distribution.set_placeholder(NEVER_FIRED)
            self._distribution.set_note("")
            return
        self._distribution.set_placeholder("")
        self._distribution.show_shares([b.label for b in found.bins], [b.share for b in found.bins])
        # The two numbers the shape implies but does not state, and the floor
        # caveat travels with the mean rather than being left in a docstring.
        self._distribution.set_note(
            f"{found.shots:,} shots · {found.wasted_share:.0%} killed nothing · "
            f"≥{found.mean_kills_per_shot:.2f} kills/shot"
        )
        against = stats.distribution(other)
        self._distribution.set_comparison(
            self._compare_name,
            None
            if against is None or against.shots == 0 or not self._compare_name
            else [b.share for b in against.bins],
        )

    def _show_curves(self, rows: Sequence[EvalRow], compare: Sequence[EvalRow]) -> None:
        for curve in stats.CURVES:
            view = self._curves[curve.key]
            view.clear()
            points = stats.series(rows, curve.key)
            if not points:
                view.set_placeholder(NO_STATISTICS)
                continue
            view.set_placeholder("")
            view.extend(points)
            if not self._compare_name:
                continue
            view.set_comparison(self._compare_name)
            for update, value in stats.series(compare, curve.key):
                view.append_comparison(update, value)
