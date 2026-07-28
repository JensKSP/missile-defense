# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The statistics screen, drawn from a run an actual trainer produced.

`test_ui_stats.py` covers the arithmetic against hand-written rows and is where
the edge cases live. What it cannot cover is the join: that the columns
`missile_defense.training.train` writes are the columns `missile_defense.ui.stats` reads,
spelled the same way. A rename on either side leaves both halves passing their own tests and puts a
screen full of em dashes in front of a user, and the only thing that catches
that is a real `evals.csv` written by a real run in another process.

So these are deliberately assertions about *populated*, not about values. The
tiny run trains for seconds and its policy is terrible; what must hold is that
every panel found something rather than falling back to an empty state.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from .harness import needs_native, needs_qt, needs_torch

# Only for the annotations below, which `from __future__ import annotations` keeps
# as strings: naming these types costs nothing at run time and the import never
# happens on a machine without the optional dependency that provides them.
if TYPE_CHECKING:
    from collections.abc import Iterator

    from missile_defense.ui.analysis import AnalysisView

pytestmark = [pytest.mark.e2e, needs_qt]


@pytest.fixture
def analysis(qt_app: object, trained_run: Path) -> Iterator[AnalysisView]:
    """A real trainer window, opened on a real run, showing the statistics."""
    from missile_defense.ui.app import Trainer  # noqa: PLC0415 — optional dependency

    window = Trainer(trained_run)
    window.resize(1400, 900)
    window._tick()
    yield window._analysis
    window.close()


@needs_torch
@needs_native
def test_the_statistics_screen_fills_from_a_real_run(analysis: AnalysisView) -> None:
    """The join, asserted: the trainer's columns reach the trainer's tiles.

    `_body` is visible only when more than the score came through, so this fails
    on exactly the case that matters — a run whose statistics the trainer could
    not find, for any reason including a renamed column.
    """
    # `isHidden`, not `isVisible`: nothing is "visible" while the window itself
    # has never been shown, and these run offscreen with no window at all — so
    # `isVisible()` would be False here whatever the screen decided, and every
    # assertion below it would pass by accident.
    assert not analysis._body.isHidden(), "the statistics screen fell back to its empty state"
    from missile_defense.ui import stats  # noqa: PLC0415 — optional dependency

    shown = {key: tile for key, tile in analysis._tiles.items()}
    # The ones the human asked for by name, and the ones that make a plateau
    # diagnosable. A tile that exists but says "—" is the failure here.
    for key in ("score", "survived", "wave", "shots", "hit_rate", "wasted"):
        assert key in shown, f"no {key} tile"
        assert shown[key]._value.text() != stats.ABSENT, f"{key} came through empty"


@needs_torch
@needs_native
def test_the_kills_per_shot_distribution_is_populated(analysis: AnalysisView) -> None:
    """Five bars with a real total under them, not the empty state.

    The histogram is the one piece of new *core* instrumentation behind this
    screen (it is binned in C++ at blast expiry), so it crosses the most
    boundaries of anything here: sim → bindings → trainer → CSV → trainer.
    """
    chart = analysis._distribution
    assert chart._placeholder.isHidden(), chart._placeholder.text()
    assert chart._set.count() == 5
    # A tiny run still fires hundreds of interceptors, so an all-zero histogram
    # means the bins never reached the CSV — which a header check would pass.
    assert sum(chart._set.at(index) for index in range(5)) > 0
    assert "shots" in chart._stats.text()


@needs_torch
@needs_native
def test_the_cause_curves_are_populated(analysis: AnalysisView) -> None:
    from missile_defense.ui import stats  # noqa: PLC0415 — optional dependency

    for curve in stats.CURVES:
        view = analysis._curves[curve.key]
        assert view._placeholder.isHidden(), f"{curve.key}: {view._placeholder.text()}"
        assert view._xs, f"{curve.key} has no points"


@needs_torch
@needs_native
def test_a_second_run_overlays_the_distribution_and_the_curves(
    qt_app: object,  # noqa: ARG001 — the QApplication has to exist
    trained_run: Path,
    tmp_path: Path,
) -> None:
    """The compare picker, from the outside.

    A copy of the run rather than a second training run: this is testing that a
    comparison *reaches* both the bars and the curves, and the numbers being
    identical makes a failure unambiguous — every delta must be "no change", and
    anything else means the two sides are not reading the same file.
    """
    from missile_defense.ui import stats  # noqa: PLC0415 — optional dependency
    from missile_defense.ui.app import Trainer  # noqa: PLC0415 — optional dependency

    other = tmp_path / "other-run"
    shutil.copytree(trained_run, other)

    window = Trainer(trained_run)
    window.resize(1400, 900)
    window._tick()
    try:
        window._compare_with(other)
        window._tick()
        analysis = window._analysis
        assert analysis._compare_name == other.name
        # The second bar set is only appended when a comparison is drawn.
        assert len(analysis._distribution._series.barSets()) == 2
        for curve in stats.CURVES:
            assert analysis._curves[curve.key]._compare_xs, f"{curve.key} has no comparison"
        # And the deltas, which are the point of comparing at all.
        assert not analysis._tiles["score"]._delta.isHidden()
        assert stats.UNCHANGED in analysis._tiles["score"]._delta.text()
    finally:
        window.close()


def test_a_comparison_that_shows_nothing_says_why(
    qt_app: object,  # noqa: ARG001 — the QApplication has to exist
    trained_run: Path,
    tmp_path: Path,
) -> None:
    """The bug a user actually hit: "I don't see anything — maybe I don't understand".

    Two runs evaluated under different protocols are not two numbers, so nothing
    is overlaid — correctly. What was missing is the sentence saying so. A panel
    that silently stays blank leaves a person deciding between "broken" and "I
    am holding it wrong", and both answers are wrong.
    """
    from missile_defense.runs import sources  # noqa: PLC0415 — optional dependency
    from missile_defense.ui.app import Trainer  # noqa: PLC0415 — optional dependency

    other = tmp_path / "other-protocol"
    shutil.copytree(trained_run, other)
    # Rewrite the comparison run's evaluations under a protocol nothing here
    # uses. `matching_eval_protocol` compares the whole tuple, so one field is
    # all it takes — and one field is exactly the realistic accident.
    evals = other / sources.EVALS_NAME
    with evals.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    assert "seed_count" in fields, fields
    for row in rows:
        row["seed_count"] = "3"  # not the canonical count
    with evals.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    window = Trainer(trained_run)
    window.resize(1400, 900)
    window._tick()
    try:
        window._compare_with(other)
        window._tick()
        analysis = window._analysis

        assert not analysis._note.isHidden(), "an empty comparison explained nothing"
        note = analysis._note.text()
        assert other.name in note
        assert "different protocol" in note
        # And nothing is drawn, which is the *correct* half of the old behaviour.
        assert len(analysis._distribution._series.barSets()) == 1
        assert analysis._tiles["score"]._delta.isHidden()
    finally:
        window.close()


def test_a_comparable_run_says_that_too(
    qt_app: object,  # noqa: ARG001 — the QApplication has to exist
    trained_run: Path,
    tmp_path: Path,
) -> None:
    # The other half of the rule: the note is a statement of what the panel is
    # doing, not an error channel that only appears when something is wrong.
    from missile_defense.ui.app import Trainer  # noqa: PLC0415 — optional dependency

    other = tmp_path / "same-protocol"
    shutil.copytree(trained_run, other)

    window = Trainer(trained_run)
    window.resize(1400, 900)
    window._tick()
    try:
        window._compare_with(other)
        window._tick()
        assert other.name in window._analysis._note.text()
        assert "held against" in window._analysis._note.text()
    finally:
        window.close()
