# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# pyright: reportMissingImports=false
"""The console window: the curve, the yardstick, and the episodes to watch.

    poe ui                 # attach to ./runs
    poe ui -- path/to/run  # attach to somewhere else

This phase is **read-only** (docs/ROADMAP.md, M8, phase 1). It attaches to a run
started from a terminal and cannot start one, which is the whole point: training
happens in its own process, so a UI crash costs you nothing and the console can
be opened on a directory synced from another machine.

One screen, because the run is the subject and a tab bar makes you hunt for the
thing you came to look at. The score curve gets the space; the diagnostics and
the episode list are strips around it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import sources, theme
from .charts import CurveView
from .runner import AppNotFound, ReplayLauncher
from .sources import BASELINE_MEAN_SCORE, EvalRow, MetricRow, Recording

#: An update takes seconds, so a second is a smooth refresh and not a busy loop.
POLL_MS = 1000
#: Recordings arrive every `--record-every` updates; scanning a directory every
#: poll would be wasted work.
RESCAN_EVERY = 3
#: If metrics.csv has not moved in this long, the run is not running.
LIVE_AFTER_S = 90.0

#: Empty states. Each says what is missing and what would fill it.
WAITING = "Waiting for metrics.csv.\nStart a run with `poe train` and it appears here."
NO_EPISODES = "No finished episodes yet.\nAn episode is thousands of ticks long."
NO_RECORDINGS = (
    "No recordings yet. The trainer writes a watchable episode every --record-every updates."
)
NO_EVALS = (
    "No evaluation scored yet.\n"
    "The trainer scores the policy on the 32 canonical seeds every\n"
    "--eval-every updates and appends it to evals.csv."
)


class StatTile(QFrame):
    """One number, large enough to read from across the room."""

    def __init__(self, caption: str, note: str = "") -> None:
        super().__init__()
        self.setProperty("role", "tile")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        self._caption = QLabel(caption)
        self._caption.setProperty("role", "caption")
        self._value = QLabel("—")
        self._value.setProperty("role", "value")
        self._note = QLabel(note)
        self._note.setProperty("role", "note")
        for widget in (self._caption, self._value, self._note):
            layout.addWidget(widget)
        self._colour = ""

    def set_value(self, text: str, colour: str = theme.TEXT) -> None:
        self._value.setText(text)
        if colour != self._colour:  # restyling every tick would repaint the world
            self._value.setStyleSheet(f"color: {colour};")
            self._colour = colour

    def set_note(self, text: str) -> None:
        self._note.setText(text)


class Console(QMainWindow):
    """Everything the console is, in one window."""

    def __init__(self, run_dir: Path) -> None:
        super().__init__()
        self._run_dir = run_dir
        self._metrics = sources.metrics_tail(run_dir)
        self._evals = sources.evals_tail(run_dir)
        self._launcher = ReplayLauncher()
        #: None until the first scan, so "still empty" is distinguishable from
        #: "not looked yet" — otherwise the empty state never gets drawn.
        self._listed: list[Recording] | None = None
        self._ticks = 0
        self._updates = 0
        self._last_metric: MetricRow | None = None
        self._last_eval: EvalRow | None = None

        self.setWindowTitle(f"Missile Command — training console · {run_dir}")
        self.setCentralWidget(self._build())
        self.statusBar().showMessage(f"watching {run_dir / sources.METRICS_NAME}")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(POLL_MS)
        self._tick()

    # ---- construction -------------------------------------------------------
    def _build(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 12, 14, 8)
        layout.setSpacing(10)
        layout.addLayout(self._header())
        layout.addLayout(self._tiles())

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._plots())
        split.addWidget(self._recordings())
        split.setStretchFactor(0, 1)
        split.setSizes([980, 300])
        layout.addWidget(split, stretch=1)
        return root

    def _header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        title = QLabel("MISSILE COMMAND · TRAINING CONSOLE")
        title.setProperty("role", "title")
        path = QLabel(str(self._run_dir.resolve()))
        path.setProperty("role", "note")
        self._status = QLabel("NO RUN")
        self._status.setProperty("role", "caption")
        row.addWidget(title)
        row.addSpacing(12)
        row.addWidget(path)
        row.addStretch(1)
        row.addWidget(self._status)
        return row

    def _tiles(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        self._tile_update = StatTile("update", "no run attached")
        self._tile_score = StatTile("eval score", f"baseline {BASELINE_MEAN_SCORE:,.0f}")
        self._tile_return = StatTile("mean return", "shaped, scaled — not a score")
        self._tile_entropy = StatTile("entropy", "how undecided the policy is")
        for tile in (self._tile_update, self._tile_score, self._tile_return, self._tile_entropy):
            row.addWidget(tile)
        return row

    def _plots(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # The hero: the only curve in the same units as the scripted agent, so
        # the only one the baseline can honestly be drawn across.
        self._score = CurveView(
            "score on the canonical seeds",
            theme.SCORE,
            value_format="%.0f",
            markers=True,  # an eval every 50 updates is dots, not a dense line
            series_name="learned policy",
            from_zero=True,
        )
        self._score.set_baseline(
            BASELINE_MEAN_SCORE, f"scripted baseline {BASELINE_MEAN_SCORE:,.0f}"
        )
        self._score.set_placeholder(NO_EVALS)
        layout.addWidget(self._score, stretch=3)

        strip = QHBoxLayout()
        strip.setSpacing(10)
        self._return = CurveView("mean return", theme.RETURN, value_format="%.1f")
        self._entropy = CurveView("entropy", theme.ENTROPY, value_format="%.2f")
        self._value = CurveView("value loss", theme.VALUE, value_format="%.3g")
        for curve in (self._return, self._entropy, self._value):
            curve.setMinimumHeight(150)
            strip.addWidget(curve)
        layout.addLayout(strip, stretch=2)
        return panel

    def _recordings(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        caption = QLabel("recordings")
        caption.setProperty("role", "caption")
        layout.addWidget(caption)
        self._list = QListWidget()
        self._list.itemActivated.connect(self._open)
        self._list.itemDoubleClicked.connect(self._open)
        layout.addWidget(self._list, stretch=1)
        # A greyed-out list row is painted from the disabled palette, which under
        # this stylesheet is invisible; an empty state has to be a real widget.
        self._no_recordings = QLabel(NO_RECORDINGS)
        self._no_recordings.setProperty("role", "placeholder")
        self._no_recordings.setWordWrap(True)
        self._no_recordings.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._no_recordings.setVisible(False)
        layout.addWidget(self._no_recordings, stretch=1)
        note = QLabel("double-click to watch an episode")
        note.setProperty("role", "note")
        layout.addWidget(note)
        return panel

    # ---- the poll -----------------------------------------------------------
    def _tick(self) -> None:
        self._ticks += 1
        self._read_metrics()
        self._read_evals()
        if self._ticks % RESCAN_EVERY == 1:
            self._refresh_recordings()
        self._refresh_status()

    def _read_metrics(self) -> None:
        batch = self._metrics.poll()
        if batch.restarted:  # a different run writes into the same file
            for curve in (self._return, self._entropy, self._value):
                curve.clear()
            self._updates = 0
            self._last_metric = None
            for tile in (self._tile_update, self._tile_return, self._tile_entropy):
                tile.set_value("—")
        for row in batch.rows:
            self._return.append(row.update, row.mean_return)
            self._entropy.append(row.update, row.entropy)
            self._value.append(row.update, row.value_loss)
            self._updates += 1
        if not batch.rows:
            return
        row = self._last_metric = batch.rows[-1]
        self._score.set_x_extent(row.update)  # the eval chart spans the run too
        self._tile_update.set_value(f"{row.update:,}")
        self._tile_update.set_note(f"{row.samples:,} samples")
        self._tile_return.set_value(_number(row.mean_return, "{:,.1f}"))
        self._tile_entropy.set_value(_number(row.entropy, "{:.3f}"))

    def _read_evals(self) -> None:
        batch = self._evals.poll()
        if batch.restarted:
            self._score.clear()
            self._last_eval = None
            self._tile_score.set_value("—")
        for row in batch.rows:
            self._score.append(row.update, row.mean_score)
        if not batch.rows:
            return
        row = self._last_eval = batch.rows[-1]
        delta = row.mean_score - BASELINE_MEAN_SCORE
        ahead = delta > 0
        self._tile_score.set_value(f"{row.mean_score:,.0f}", theme.AHEAD if ahead else theme.BEHIND)
        self._tile_score.set_note(
            f"{abs(delta):,.0f} {'ahead of' if ahead else 'behind'} baseline · update {row.update}"
        )

    def _refresh_recordings(self) -> None:
        found = sources.list_recordings(self._run_dir)
        if self._listed is not None and _same(found, self._listed):
            return  # nothing new; leave the selection and scroll position alone
        selected = self._selected_path()
        self._listed = found
        self._list.clear()
        self._list.setVisible(bool(found))
        self._no_recordings.setVisible(not found)
        if not found:
            return
        now = time.time()
        for recording in found:
            item = QListWidgetItem(
                f"{recording.name}\n"
                f"{sources.human_age(now - recording.modified)} ago · "
                f"{sources.human_size(recording.size)}"
            )
            item.setData(Qt.ItemDataRole.UserRole, str(recording.path))
            self._list.addItem(item)
            if recording.path == selected:
                self._list.setCurrentItem(item)

    def _refresh_status(self) -> None:
        modified = sources.last_modified(self._run_dir / sources.METRICS_NAME)
        # Two different kinds of empty, and saying which is the whole job of an
        # empty state: no run at all, versus a run whose episodes have not ended.
        self._return.set_placeholder(WAITING if modified is None else NO_EPISODES)
        for curve in (self._entropy, self._value):
            curve.set_placeholder("" if modified is not None else WAITING)
        if modified is None:
            self._set_status("NO RUN", theme.MUTED)
            self.statusBar().showMessage(
                f"no {sources.METRICS_NAME} in {self._run_dir} yet — start a run with `poe train`"
            )
            return
        age = max(time.time() - modified, 0.0)
        live = age < LIVE_AFTER_S
        self._set_status("LIVE" if live else "IDLE", theme.AHEAD if live else theme.MUTED)
        self.statusBar().showMessage(
            f"{self._updates:,} updates · last write {sources.human_age(age)} ago"
            + (
                f" · {self._launcher.running} replay window(s) open"
                if self._launcher.running
                else ""
            )
        )

    def _set_status(self, text: str, colour: str) -> None:
        if self._status.text() != text:
            self._status.setText(text)
            self._status.setStyleSheet(f"color: {colour}; font-weight: 600;")

    # ---- opening an episode -------------------------------------------------
    def _selected_path(self) -> Path | None:
        # selectedItems() rather than currentItem(), which is only *sometimes*
        # a widget and whose stubs say it always is.
        selected = self._list.selectedItems()
        if not selected:
            return None
        data = selected[0].data(Qt.ItemDataRole.UserRole)
        return Path(str(data)) if data else None

    def _open(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        try:
            self._launcher.launch(Path(str(data)))
        except AppNotFound as error:
            QMessageBox.warning(self, "The game is not built", str(error))
            return
        self.statusBar().showMessage(f"playing {Path(str(data)).name}")


def _same(left: list[Recording], right: list[Recording]) -> bool:
    """Same files, same timestamps — so the list is left exactly as it is."""
    return [(r.path, r.modified) for r in left] == [(r.path, r.modified) for r in right]


def _number(value: float | None, spec: str) -> str:
    """A measurement the trainer has not produced yet is a dash, never a zero."""
    return "—" if value is None else spec.format(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch a Missile Command training run.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=Path("runs"),
        help="the run's --out-dir (default: runs)",
    )
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Missile Command training console")
    app.setStyleSheet(theme.stylesheet())
    window = Console(args.run_dir)
    # Roomy, but never bigger than the desktop it opens on — a window whose
    # status line is off-screen is a window with a bug in it.
    available = app.primaryScreen().availableSize()
    window.resize(min(1360, available.width() - 60), min(860, available.height() - 60))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
