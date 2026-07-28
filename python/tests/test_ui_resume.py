# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""What a resumed run does to the things that draw it.

`--resume` restarts from a checkpoint, so the trainer writes update 651 again
after having already written 661. `metrics.csv` then holds a *backwards* step
and two rows for each repeated update — real output of a supported flag, not a
corrupt file.

Everything here is about that one fact, because it broke two things at once and
they looked unrelated: the curves drew a straight line running backwards from
the newest point to where the resume began, and the mean-return tile showed a
dash while the chart beneath it was drawing a curve.

The rules are written as properties of *any* rewound sequence rather than as the
shape of `runs/deadline-1330`, which is where they were found.
"""

from __future__ import annotations

import pytest
from missile_defense.ui.sources import Latest

try:  # the widgets, where PySide6 is installed — the trainer is optional
    from missile_defense.ui.charts import CurveView
    from PySide6.QtWidgets import QApplication
except ModuleNotFoundError as error:  # pragma: no cover - depends on the machine
    if error.name is None or not error.name.startswith("PySide6"):
        raise
    CurveView = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]

needs_qt = pytest.mark.skipif(CurveView is None, reason="PySide6 is not installed")


# ---- the tile ---------------------------------------------------------------


def test_a_gap_does_not_erase_the_last_real_measurement() -> None:
    # The trainer writes `nan` until a run's first episodes finish. A dash claims
    # no measurement exists; one does, it is just not from this update.
    latest = Latest()
    latest.offer(650, 712.9)
    latest.offer(651, None)
    latest.offer(652, None)
    assert latest.value == 712.9
    assert latest.update == 650


def test_a_stale_number_says_how_stale() -> None:
    latest = Latest()
    latest.offer(650, 712.9)
    assert latest.note("shaped", current_update=650) == "shaped"
    assert latest.note("shaped", current_update=652) == "shaped · last at update 650"


def test_no_measurement_at_all_is_left_as_a_dash() -> None:
    # Nothing to show and nothing to date — the note must not invent an age.
    latest = Latest()
    latest.offer(1, None)
    assert latest.value is None
    assert latest.note("shaped", current_update=1) == "shaped"


# ---- the curves -------------------------------------------------------------


@pytest.fixture(scope="module")
def qt_app():  # type: ignore[no-untyped-def]
    if QApplication is None:  # pragma: no cover - skipped by needs_qt
        pytest.skip("PySide6 is not installed")
    yield QApplication.instance() or QApplication([])


def _curve():  # type: ignore[no-untyped-def]
    return CurveView(title="t", colour="#ffffff")


@needs_qt
def test_a_curve_never_runs_backwards(qt_app) -> None:  # type: ignore[no-untyped-def]
    """The property the backwards line violated, stated once for any input."""
    curve = _curve()
    for update in (*range(645, 662), 651, 652):
        curve.append(update, float(update))
    xs = [x for x, _ in curve.points]
    assert xs == sorted(xs), "the curve stepped backwards"


@needs_qt
def test_a_resume_discards_the_branch_it_abandoned(qt_app) -> None:  # type: ignore[no-untyped-def]
    curve = _curve()
    for update in range(645, 662):  # the branch that was thrown away
        curve.append(update, 1.0)
    curve.append(651, 2.0)  # resumed from the checkpoint at 650
    curve.append(652, 2.0)
    assert [x for x, _ in curve.points] == [*range(645, 651), 651, 652]
    # The superseded values are gone, not merely reordered.
    assert [y for _, y in curve.points][-2:] == [2.0, 2.0]


@needs_qt
def test_the_seam_is_recorded_rather_than_closed_over(qt_app) -> None:  # type: ignore[no-untyped-def]
    curve = _curve()
    for update in range(645, 662):
        curve.append(update, 1.0)
    curve.append(651, 2.0)
    assert curve.resumes == (651,)


@needs_qt
def test_a_run_that_never_rewinds_records_no_seam(qt_app) -> None:  # type: ignore[no-untyped-def]
    curve = _curve()
    for update in range(1, 50):
        curve.append(update, float(update))
    assert curve.resumes == ()
    assert [x for x, _ in curve.points] == list(range(1, 50))


@needs_qt
def test_the_axis_rescales_when_a_discarded_branch_held_the_extreme(qt_app) -> None:  # type: ignore[no-untyped-def]
    """A y-axis still scaled to a deleted peak leaves the live curve flat."""
    curve = _curve()
    curve.append(1, 1.0)
    curve.append(2, 1000.0)  # the peak, on the branch about to be abandoned
    curve.append(2, 2.0)
    assert [y for _, y in curve.points] == [1.0, 2.0]
    assert curve.y_range == (1.0, 2.0)


@needs_qt
def test_repeated_resumes_each_leave_a_seam(qt_app) -> None:  # type: ignore[no-untyped-def]
    curve = _curve()
    for update in range(1, 11):
        curve.append(update, 1.0)
    curve.append(5, 2.0)
    for update in range(6, 11):
        curve.append(update, 2.0)
    curve.append(8, 3.0)
    assert curve.resumes == (5, 8)
    xs = [x for x, _ in curve.points]
    assert xs == sorted(xs)
