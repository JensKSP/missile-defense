# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# pyright: reportMissingImports=false
"""One curve, drawn. The only module that knows the charting library exists.

Qt Charts ships with PySide6 under the same LGPLv3, so this costs nothing to
install and nothing to vet — and at one point every few seconds, the performance
argument that would push toward pyqtgraph does not arise. Everything still goes
through this wrapper (``append`` / ``set_baseline`` / ``clear``), so swapping the
library later touches this file and no other (docs/ROADMAP.md, M8, risk 1).

The baseline is the reason this is a wrapper and not a bare ``QChartView``: a
horizontal line at 113,834 is what turns a number going up into "am I winning
yet", and it has to keep spanning the plot as the run grows.

A curve can carry a **second run** as well (M8 phase 5). Overlaid rather than in
a second chart beside it: two plots with independent axes make you compare by
eye across a gap, and the question being asked — "did that change help?" — is
answered by whether one line is above the other. It is drawn in the same colour
at a third of the opacity, so it reads as *this metric, the other run* rather
than as a fourth thing on the plot.

Two things are written over the plot rather than beside it, because both are
about the curve and a curve read from across the room has no room for a legend:
a **footnote** in the corner with the statistics the shape alone does not give
you (`md.ui.sources.curve_note`), and a **readout** that follows the pointer
with the value it is over. The arithmetic for both is in `sources`, which has no
Qt in it and is therefore tested; this file only places the labels.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Sequence

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QEvent, QMargins, QPointF, Qt
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen, QResizeEvent
from PySide6.QtWidgets import QLabel, QWidget

from . import sources, theme

#: Out of 255. Enough to follow the line, faint enough that the attached run is
#: unambiguously the subject.
COMPARISON_ALPHA = 90

#: How much of a chart's width the corner footnote may take before it starts
#: dropping its last statistic. The title has the rest of that row.
STATS_WIDTH = 0.58


class CurveView(QChartView):
    """A single metric against the update number, with an optional baseline.

    ``None`` values are gaps, not zeroes: the trainer writes ``nan`` for the mean
    return until the first episodes finish, and drawing that as zero would invent
    a dip the run never had.
    """

    def __init__(
        self,
        title: str,
        colour: str,
        *,
        value_format: str = "%.2f",
        markers: bool = False,
        series_name: str = "",
        from_zero: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # A score is a distance from nothing, so its axis starts at nothing: the
        # gap between the curve and the baseline is the thing being read, and an
        # axis that crops to the data would hide how big it is.
        self._from_zero = from_zero
        self._value_format = value_format
        self._x_min = 0.0
        self._x_max = 0.0
        self._y_min = 0.0
        self._y_max = 0.0
        self._count = 0
        self._last: float | None = None
        self._baseline: float | None = None
        self._x_hint = 0.0
        # The points, again, beside the series that draws them. Qt hands its own
        # copy back as a list of QPointF, which is a copy of the whole run every
        # time the pointer moves; these are what the footnote is computed from
        # and what a hover is snapped to, and both want plain floats.
        self._xs: list[float] = []
        self._ys: list[float] = []
        self._compare_xs: list[float] = []
        self._compare_ys: list[float] = []
        #: The footnote at full length, before it is trimmed to the chart's width.
        self._full_stats = ""

        self._series = QLineSeries()
        self._series.setName(series_name or title)
        self._series.setPen(QPen(QColor(colour), 2.0))
        self._series.setPointsVisible(markers)

        # The same metric from another run. Same hue, a third of the opacity: a
        # different colour would read as a different measurement.
        faded = QColor(colour)
        faded.setAlpha(COMPARISON_ALPHA)
        self._compare = QLineSeries()
        self._compare.setPen(QPen(faded, 1.6))
        self._compare.setPointsVisible(markers)
        self._compare.setVisible(False)
        self._compare_min = 0.0
        self._compare_max = 0.0
        self._compare_count = 0

        # Drawn first so the curve sits on top of it, and dashed so it reads as a
        # target rather than as another measurement.
        self._baseline_series = QLineSeries()
        pen = QPen(QColor(theme.BASELINE), 1.4)
        pen.setStyle(Qt.PenStyle.DashLine)
        self._baseline_series.setPen(pen)
        self._baseline_series.setVisible(False)

        # Where the pointer is, in the plot's own units — a vertical hairline at
        # the point the readout is reporting. Without it the chip says "update
        # 812" and you have to trust it; with it, you can see which point it means.
        self._crosshair = QLineSeries()
        hair = QColor(theme.MUTED)
        hair.setAlpha(110)
        pen = QPen(hair, 1.0)
        pen.setStyle(Qt.PenStyle.DashLine)
        self._crosshair.setPen(pen)
        self._crosshair.setVisible(False)

        self._chart = QChart()
        self._chart.setBackgroundBrush(QColor(theme.PANEL))
        self._chart.setBackgroundRoundness(6.0)
        self._chart.setPlotAreaBackgroundVisible(False)
        self._chart.setMargins(QMargins(6, 6, 10, 4))
        self._chart.setTitle(title.upper())
        self._chart.setTitleBrush(QColor(theme.MUTED))
        self._chart.setTitleFont(_caption_font())
        self._chart.legend().setVisible(False)
        self._chart.legend().setAlignment(Qt.AlignmentFlag.AlignTop)
        self._chart.legend().setLabelColor(QColor(theme.MUTED))
        self._chart.legend().setFont(_caption_font())
        self._chart.addSeries(self._baseline_series)
        self._chart.addSeries(self._crosshair)
        self._chart.addSeries(self._compare)  # under the attached run, not over it
        self._chart.addSeries(self._series)

        self._x_axis = _axis("%d")
        self._y_axis = _axis(value_format)
        self._chart.addAxis(self._x_axis, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(self._y_axis, Qt.AlignmentFlag.AlignLeft)
        for series in (self._series, self._baseline_series, self._compare, self._crosshair):
            series.attachAxis(self._x_axis)
            series.attachAxis(self._y_axis)
        self._show_compare_marker(False)
        for marker in self._chart.legend().markers(self._crosshair):
            marker.setVisible(False)  # the pointer is not one of the run's series

        self.setChart(self._chart)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setFrameShape(QChartView.Shape.NoFrame)
        # Moves with no button held are what a hover is; a QGraphicsView delivers
        # them through its viewport, so both have to be asked for them.
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        # An empty panel says nothing; this says what is missing and what to do.
        self._placeholder = QLabel("", self)
        self._placeholder.setProperty("role", "placeholder")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # The footnote in the corner, and the chip that follows the pointer. Both
        # are children of the view, drawn over the plot; both are transparent to
        # the mouse, or the chip would steal the very moves that position it.
        self._stats = _overlay("stat", self)
        self._readout = _overlay("readout", self)
        self._rescale()
        self._refresh_stats()

    # ---- data ---------------------------------------------------------------
    def append(self, x: float, y: float | None) -> None:
        if y is None:
            return
        self._series.append(x, y)
        self._xs.append(x)
        self._ys.append(y)
        self._last = y
        self._count += 1
        if self._count == 1:
            self._x_min = self._x_max = x
            self._y_min = self._y_max = y
        else:
            self._x_min, self._x_max = min(self._x_min, x), max(self._x_max, x)
            self._y_min, self._y_max = min(self._y_min, y), max(self._y_max, y)
        self._rescale()
        self._refresh_stats()

    def extend(self, points: Iterable[tuple[float, float | None]]) -> None:
        for x, y in points:
            self.append(x, y)

    # ---- the other run ------------------------------------------------------
    def set_comparison(self, name: str) -> None:
        """Start overlaying a second run's points, labelled ``name``.

        Idempotent on the name so the poll can call it every tick, and it drops
        whatever was there — choosing a different run to compare against is the
        same act as choosing none and then that one.
        """
        self.clear_comparison()
        self._compare.setName(name)
        self._compare.setVisible(True)
        self._show_compare_marker(True)

    def append_comparison(self, x: float, y: float | None) -> None:
        if y is None:
            return
        self._compare.append(x, y)
        self._compare_xs.append(x)
        self._compare_ys.append(y)
        self._compare_count += 1
        if self._compare_count == 1:
            self._compare_min = self._compare_max = y
        else:
            self._compare_min = min(self._compare_min, y)
            self._compare_max = max(self._compare_max, y)
        self._rescale()

    def clear_comparison(self) -> None:
        self._compare.clear()
        self._compare_xs.clear()
        self._compare_ys.clear()
        self._compare_count = 0
        self._rescale()

    def hide_comparison(self) -> None:
        """No second run selected — and no ghost of the last one in the legend."""
        self.clear_comparison()
        self._compare.setVisible(False)
        self._show_compare_marker(False)

    def _show_compare_marker(self, shown: bool) -> None:
        for marker in self._chart.legend().markers(self._compare):
            marker.setVisible(shown)

    def set_baseline(self, value: float, label: str) -> None:
        """Draw a horizontal reference line — the number the run is chasing."""
        self._baseline = value
        self._baseline_series.setName(label)
        self._baseline_series.setVisible(True)
        self._chart.legend().setVisible(True)
        self._rescale()

    def set_x_extent(self, x: float) -> None:
        """Stretch the axis to ``x`` even where this curve has no points yet.

        So the sparse evaluation chart shares an x-axis with the dense ones
        beside it: "nothing scored yet, over 117 updates" is a different
        statement from "nothing scored yet, over ten".
        """
        if x > self._x_hint:
            self._x_hint = x
            self._rescale()

    def clear(self) -> None:
        """Forget every point — the run's file was replaced by a new one."""
        self._series.clear()
        self._xs.clear()
        self._ys.clear()
        self._count = 0
        self._last = None
        self._hide_readout()
        self._rescale()
        self._refresh_stats()

    def set_placeholder(self, text: str) -> None:
        """What to say instead of an empty plot. Empty text says nothing at all."""
        if text != self._placeholder.text():
            self._placeholder.setText(text)
            self._centre_placeholder()
        self._placeholder.setVisible(bool(text) and self._count == 0)

    @property
    def last_value(self) -> float | None:
        return self._last

    # ---- the footnote -------------------------------------------------------
    def _refresh_stats(self) -> None:
        """Recompute the corner statistics; lay them out only when they changed."""
        text = sources.curve_note(self._ys, self._value_format)
        if text != self._full_stats:
            self._full_stats = text
            self._place_stats()

    def _place_stats(self) -> None:
        """Over the plot's top right, shortened rather than run into the title.

        The footnote is the first thing to give way when the chart is narrow —
        the last statistic is dropped, then the whole label. A curve whose title
        is half-covered is worse off than a curve with no footnote.
        """
        parts = self._full_stats.split(" · ") if self._full_stats else []
        room = int(self.width() * STATS_WIDTH)
        while parts and self._stats.fontMetrics().horizontalAdvance(" · ".join(parts)) > room:
            parts.pop()
        self._stats.setText(" · ".join(parts))
        self._stats.adjustSize()
        self._stats.setVisible(bool(parts))
        # Against the plot's own right edge rather than the widget's, so it lines
        # up with the axis it is describing and stays on the panel.
        area = self._chart.plotArea()
        if area.isEmpty():  # before the first layout there is no plot to sit over
            self._stats.move(max(self.width() - self._stats.width() - 12, 0), 4)
            return
        corner = self.mapFromScene(self._chart.mapToScene(area.topRight()))
        self._stats.move(
            max(corner.x() - self._stats.width(), 4),
            max(corner.y() - self._stats.height() - 4, 2),
        )

    # ---- the pointer --------------------------------------------------------
    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt's name
        super().mouseMoveEvent(event)
        self._read_at(event.position())

    def leaveEvent(self, event: QEvent) -> None:  # noqa: N802 — Qt's name
        super().leaveEvent(event)
        self._hide_readout()

    def _read_at(self, position: QPointF) -> None:
        """Report the recorded point nearest the pointer — and only inside the plot.

        Over the axes, the title or an empty chart there is nothing to report, and
        a chip that lingers there would be reporting the last place it *was*.
        """
        if not self._xs:
            self._hide_readout()
            return
        point = self._chart.mapFromScene(self.mapToScene(position.toPoint()))
        if not self._chart.plotArea().contains(point):
            self._hide_readout()
            return
        index = _nearest(self._xs, self._chart.mapToValue(point, self._series).x())
        x, y = self._xs[index], self._ys[index]
        # The other run at the same update, when there is one: "did that change
        # help?" is asked of a place on the curve, not only of the two curves.
        other = self._compare_ys[_nearest(self._compare_xs, x)] if self._compare_xs else None
        self._readout.setText(
            sources.readout_note(x, y, self._value_format, self._compare.name(), other)
        )
        self._readout.adjustSize()
        self._readout.move(*self._chip_at(position))
        self._readout.setVisible(True)
        self._crosshair.replace([QPointF(x, self._y_axis.min()), QPointF(x, self._y_axis.max())])
        self._crosshair.setVisible(True)

    def _chip_at(self, position: QPointF) -> tuple[int, int]:
        """Up and to the right of the pointer, but never off the widget."""
        size = self._readout.size()
        x = min(int(position.x()) + 14, self.width() - size.width() - 6)
        y = int(position.y()) - size.height() - 8
        if y < 4:  # against the top edge it goes under the pointer instead
            y = int(position.y()) + 16
        return max(x, 6), y

    def _hide_readout(self) -> None:
        self._readout.setVisible(False)
        self._crosshair.setVisible(False)

    # ---- layout -------------------------------------------------------------
    def _rescale(self) -> None:
        """Fit both axes around the data, the baseline included.

        The baseline has to be inside the y range or the whole point of it is
        invisible — early in a run the curve is nowhere near 113,834, and that gap
        *is* the information.
        """
        x_min, x_max = (0.0, 10.0) if self._count == 0 else (self._x_min, self._x_max)
        if self._x_hint:  # a hinted axis spans the whole run, points or not
            x_min, x_max = 0.0, max(x_max, self._x_hint)
        if x_max - x_min < 1.0:  # a single update is not a range
            x_max = x_min + 1.0
        x_low, x_high = _fit(self._x_axis, x_min, x_max, step_floor=1.0)

        seen = [self._y_min, self._y_max] if self._count else []
        if self._compare_count:
            # The comparison shares the axis, so a run that went somewhere this
            # one did not must still fit — that difference is the whole point.
            seen += [self._compare_min, self._compare_max]
        if self._baseline is not None:
            seen.append(self._baseline)
        if self._from_zero:
            seen.append(0.0)
        if seen:
            _fit(self._y_axis, min(seen), max(seen))
        else:
            _fit(self._y_axis, 0.0, 1.0)

        if self._baseline is not None:
            self._baseline_series.replace(
                [QPointF(x_low, self._baseline), QPointF(x_high, self._baseline)]
            )
        self._placeholder.setVisible(bool(self._placeholder.text()) and self._count == 0)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 — Qt's name
        super().resizeEvent(event)
        self._centre_placeholder()
        self._place_stats()  # a narrower chart may have room for one statistic less
        self._hide_readout()  # the plot moved out from under the pointer

    def _centre_placeholder(self) -> None:
        """A card in the middle of the plot, sized to its text, not a curtain."""
        self._placeholder.adjustSize()
        if self._placeholder.width() > self.width() - 24:  # narrow chart, long text
            self._placeholder.setWordWrap(True)
            self._placeholder.setFixedWidth(max(self.width() - 24, 80))
            self._placeholder.adjustSize()
        size = self._placeholder.size()
        self._placeholder.move(
            max((self.width() - size.width()) // 2, 0),
            max((self.height() - size.height()) // 2, 0),
        )


def _fit(
    axis: QValueAxis, low: float, high: float, *, step_floor: float = 0.0
) -> tuple[float, float]:
    """Range ``axis`` over [low, high] with round numbers on the gridlines.

    Qt's default splits whatever range it is given into equal parts, which reads
    as 17134 / 17585 / 18036 — arithmetically correct and unreadable at a glance.
    Rounding the range outward to a 1/2/5 step instead costs a little empty space
    and buys labels you can compare without doing arithmetic.
    """
    span = high - low
    if span <= 0:  # one point, or a metric that has not moved at all
        span = max(abs(high), 1.0) * 0.2
        low, high = low - (span / 2), high + (span / 2)
    step = max(_nice_step(high - low), step_floor)
    # Only the low end is rounded onto a gridline. Rounding the high end too
    # would leave the newest point in the middle of the plot, which is the half
    # you actually watch.
    low = math.floor(low / step) * step
    high += (high - low) * 0.02
    # Anchor and interval *before* the tick type: a dynamic axis whose interval is
    # still the default zero hangs Qt outright, laying out an endless row of ticks.
    axis.setTickAnchor(low)
    axis.setTickInterval(step)
    axis.setTickType(QValueAxis.TickType.TicksDynamic)
    axis.setRange(low, high)
    return low, high


def _nice_step(span: float) -> float:
    """A gridline interval of 1, 2 or 5 times a power of ten, near span / 4."""
    rough = span / 4.0
    magnitude = 10.0 ** math.floor(math.log10(rough))
    for factor in (1.0, 2.0, 5.0):
        if rough <= factor * magnitude:
            return factor * magnitude
    return 10.0 * magnitude


def _axis(label_format: str) -> QValueAxis:
    axis = QValueAxis()
    axis.setLabelFormat(label_format)
    axis.setLabelsColor(QColor(theme.MUTED))
    axis.setLabelsFont(_caption_font())
    axis.setGridLinePen(QPen(QColor(theme.GRID), 1.0))
    axis.setLinePen(QPen(QColor(theme.EDGE), 1.0))
    axis.setMinorGridLineVisible(False)
    return axis


def _nearest(xs: Sequence[float], x: float) -> int:
    """Index of the point closest to ``x``.

    A binary search rather than a scan: this runs on every mouse move, over every
    update a run has produced, on four charts at once.
    """
    index = bisect.bisect_left(xs, x)
    if index <= 0:
        return 0
    if index >= len(xs):
        return len(xs) - 1
    return index if xs[index] - x < x - xs[index - 1] else index - 1


def _overlay(role: str, parent: QWidget) -> QLabel:
    """A label drawn over the plot that the pointer passes straight through.

    Transparent to the mouse because both of these follow or sit near the pointer,
    and a child widget under it would swallow the very moves that place it.
    """
    label = QLabel("", parent)
    label.setProperty("role", role)
    label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    label.setVisible(False)
    return label


def _caption_font() -> QFont:
    font = QFont()
    font.setPointSizeF(8.0)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.6)
    return font
