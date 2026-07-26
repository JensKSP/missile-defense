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
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QMargins, QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QResizeEvent
from PySide6.QtWidgets import QLabel, QWidget

from . import theme


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
        self._x_min = 0.0
        self._x_max = 0.0
        self._y_min = 0.0
        self._y_max = 0.0
        self._count = 0
        self._last: float | None = None
        self._baseline: float | None = None
        self._x_hint = 0.0

        self._series = QLineSeries()
        self._series.setName(series_name or title)
        self._series.setPen(QPen(QColor(colour), 2.0))
        self._series.setPointsVisible(markers)

        # Drawn first so the curve sits on top of it, and dashed so it reads as a
        # target rather than as another measurement.
        self._baseline_series = QLineSeries()
        pen = QPen(QColor(theme.BASELINE), 1.4)
        pen.setStyle(Qt.PenStyle.DashLine)
        self._baseline_series.setPen(pen)
        self._baseline_series.setVisible(False)

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
        self._chart.addSeries(self._series)

        self._x_axis = _axis("%d")
        self._y_axis = _axis(value_format)
        self._chart.addAxis(self._x_axis, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(self._y_axis, Qt.AlignmentFlag.AlignLeft)
        for series in (self._series, self._baseline_series):
            series.attachAxis(self._x_axis)
            series.attachAxis(self._y_axis)

        self.setChart(self._chart)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setFrameShape(QChartView.Shape.NoFrame)

        # An empty panel says nothing; this says what is missing and what to do.
        self._placeholder = QLabel("", self)
        self._placeholder.setProperty("role", "placeholder")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._rescale()

    # ---- data ---------------------------------------------------------------
    def append(self, x: float, y: float | None) -> None:
        if y is None:
            return
        self._series.append(x, y)
        self._last = y
        self._count += 1
        if self._count == 1:
            self._x_min = self._x_max = x
            self._y_min = self._y_max = y
        else:
            self._x_min, self._x_max = min(self._x_min, x), max(self._x_max, x)
            self._y_min, self._y_max = min(self._y_min, y), max(self._y_max, y)
        self._rescale()

    def extend(self, points: Iterable[tuple[float, float | None]]) -> None:
        for x, y in points:
            self.append(x, y)

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
        self._count = 0
        self._last = None
        self._rescale()

    def set_placeholder(self, text: str) -> None:
        """What to say instead of an empty plot. Empty text says nothing at all."""
        if text != self._placeholder.text():
            self._placeholder.setText(text)
            self._centre_placeholder()
        self._placeholder.setVisible(bool(text) and self._count == 0)

    @property
    def last_value(self) -> float | None:
        return self._last

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


def _caption_font() -> QFont:
    font = QFont()
    font.setPointSizeF(8.0)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.6)
    return font
