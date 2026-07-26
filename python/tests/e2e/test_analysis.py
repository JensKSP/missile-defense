# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The statistics screen, drawn from a run an actual trainer produced.

`test_ui_stats.py` covers the arithmetic against hand-written rows and is where
the edge cases live. What it cannot cover is the join: that the columns
`md.train` writes are the columns `md.ui.stats` reads, spelled the same way. A
rename on either side leaves both halves passing their own tests and puts a
screen full of em dashes in front of a user, and the only thing that catches
that is a real `evals.csv` written by a real run in another process.

So these are deliberately assertions about *populated*, not about values. The
tiny run trains for seconds and its policy is terrible; what must hold is that
every panel found something rather than falling back to an empty state.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from .harness import needs_native, needs_qt, needs_torch

pytestmark = [pytest.mark.e2e, needs_qt]


@pytest.fixture
def analysis(qt_app: object, trained_run: Path):  # noqa: ANN201 — PySide6 is optional
    """A real console window, opened on a real run, showing the statistics."""
    from md.ui.app import Console  # noqa: PLC0415 — optional dependency

    window = Console(trained_run)
    window.resize(1400, 900)
    window._tick()
    yield window._analysis
    window.close()


@needs_torch
@needs_native
def test_the_statistics_screen_fills_from_a_real_run(analysis) -> None:  # noqa: ANN001
    """The join, asserted: the trainer's columns reach the console's tiles.

    `_body` is visible only when more than the score came through, so this fails
    on exactly the case that matters — a run whose statistics the console could
    not find, for any reason including a renamed column.
    """
    # `isHidden`, not `isVisible`: nothing is "visible" while the window itself
    # has never been shown, and these run offscreen with no window at all — so
    # `isVisible()` would be False here whatever the screen decided, and every
    # assertion below it would pass by accident.
    assert not analysis._body.isHidden(), "the statistics screen fell back to its empty state"
    from md.ui import stats  # noqa: PLC0415 — optional dependency

    shown = {key: tile for key, tile in analysis._tiles.items()}
    # The ones the human asked for by name, and the ones that make a plateau
    # diagnosable. A tile that exists but says "—" is the failure here.
    for key in ("score", "survived", "wave", "shots", "hit_rate", "wasted"):
        assert key in shown, f"no {key} tile"
        assert shown[key]._value.text() != stats.ABSENT, f"{key} came through empty"


@needs_torch
@needs_native
def test_the_kills_per_shot_distribution_is_populated(analysis) -> None:  # noqa: ANN001
    """Five bars with a real total under them, not the empty state.

    The histogram is the one piece of new *core* instrumentation behind this
    screen (it is binned in C++ at blast expiry), so it crosses the most
    boundaries of anything here: sim → bindings → trainer → CSV → console.
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
def test_the_cause_curves_are_populated(analysis) -> None:  # noqa: ANN001
    from md.ui import stats  # noqa: PLC0415 — optional dependency

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
    from md.ui import stats  # noqa: PLC0415 — optional dependency
    from md.ui.app import Console  # noqa: PLC0415 — optional dependency

    other = tmp_path / "other-run"
    shutil.copytree(trained_run, other)

    window = Console(trained_run)
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
