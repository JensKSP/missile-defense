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
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import modelcard, paths
from ..control import Control
from . import sources, theme
from .charts import CurveView
from .forms import ParameterDialog
from .meters import SystemPanel
from .model import ModelPanel
from .params import read_params
from .runner import (
    PACKAGE_PATH,
    PROJECT_ROOT,
    AppNotFound,
    ReplayLauncher,
    TrainingRun,
    can_train,
    training_python,
)
from .runtime_dialog import RuntimeDialog
from .sources import BASELINE_MEAN_SCORE, EvalRow, MetricRow, Recording

#: Where the trainer's dataclasses live, for the parameter form's tooltips.
TRAINER_SOURCES = PACKAGE_PATH / "md"

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
#: The compare picker's first entry, and its default. Comparing is something you
#: ask for; the window's subject is one run.
NO_COMPARISON = "no comparison"

#: The pill in the corner, by what the run is doing. Read from across the room,
#: so it is one word and a colour.
STATUS = {
    "none": ("NO RUN", theme.MUTED),
    "idle": ("IDLE", theme.MUTED),
    "live": ("LIVE", theme.AHEAD),
    "paused": ("PAUSED", theme.AMBER),
    "stopping": ("STOPPING", theme.AMBER),
}


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
        # The best this number has been, under what it is now. A run regresses as
        # well as improves — PPO peaks and then falls back — so "is this the best
        # it has managed?" is a real question, and the only other thing that can
        # answer it is remembering where the curve's high point was.
        self._peak = QLabel()
        self._peak.setProperty("role", "note")
        self._peak.setVisible(False)
        self._note = QLabel(note)
        self._note.setProperty("role", "note")
        # The same number from the run being compared against. A line of its own
        # and only when comparing, so the tile stays one number the rest of the
        # time — which is what makes it readable from across the room.
        self._compare = QLabel()
        self._compare.setProperty("role", "note")
        self._compare.setVisible(False)
        for widget in (self._caption, self._value, self._peak, self._note, self._compare):
            layout.addWidget(widget)
        # Packed to the top, so a tile with no peak line still has its caption and
        # its number on the same baselines as the ones beside it.
        layout.addStretch(1)
        self._colour = ""

    def set_value(self, text: str, colour: str = theme.TEXT) -> None:
        self._value.setText(text)
        if colour != self._colour:  # restyling every tick would repaint the world
            self._value.setStyleSheet(f"color: {colour};")
            self._colour = colour

    def set_note(self, text: str) -> None:
        self._note.setText(text)

    def set_peak(self, text: str) -> None:
        if text != self._peak.text():
            self._peak.setText(text)
        self._peak.setVisible(bool(text))

    def set_compare(self, text: str) -> None:
        if text != self._compare.text():
            self._compare.setText(text)
        self._compare.setVisible(bool(text))


class Console(QMainWindow):
    """Everything the console is, in one window."""

    def __init__(self, run_dir: Path) -> None:
        super().__init__()
        self._launcher = ReplayLauncher()
        self._ticks = 0

        self.setCentralWidget(self._build())
        self._attach(run_dir)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(POLL_MS)
        self._tick()

    def _attach(self, run_dir: Path) -> None:
        """Point the whole window at a run directory — including a fresh one.

        Everything the console knows comes from that directory, so re-attaching
        is the same act as starting up. Reset is exactly this, aimed somewhere
        new; nothing is deleted.
        """
        self._run_dir = run_dir
        self._metrics = sources.metrics_tail(run_dir)
        self._evals = sources.evals_tail(run_dir)
        #: What the run has printed. A run this console started comes down a
        #: pipe; every other one writes this file itself (md.runlog), which is
        #: what gives a terminal-started run a log pane at all.
        self._log_file = sources.log_tail(run_dir)
        self._control = Control(run_dir)
        #: A run *this* console started, kept after it exits so the window knows
        #: the difference between "quiet for now" and "over". Dropped on
        #: re-attach: that run carries on, it is simply no longer this screen's.
        self._run: TrainingRun | None = None
        self._reported_exit = False
        #: What the picker is showing, so it is only rebuilt when it changed.
        self._choices: list[Path] = []
        self._compare_choices: list[Path] = []
        #: None until the first scan, so "still empty" is distinguishable from
        #: "not looked yet" — otherwise the empty state never gets drawn.
        self._listed: list[Recording] | None = None
        self._updates = 0
        self._last_metric: MetricRow | None = None
        self._last_eval: EvalRow | None = None
        #: The high-water mark of each headline number, so a tile says what the
        #: run has *managed* and not only what it is doing this second. Per run,
        #: so they are made here with everything else that re-attaching resets.
        self._peak_score = sources.Peak()
        self._peak_return = sources.Peak()
        self._peak_entropy = sources.Peak()
        #: Every eval, by the update it scored. A checkpoint is described by the
        #: evaluation at *its* update, which is not always the newest one.
        self._eval_rows: dict[int, EvalRow] = {}
        for curve in (self._score, self._return, self._entropy, self._value):
            curve.clear()
        for tile in (self._tile_update, self._tile_score, self._tile_return, self._tile_entropy):
            tile.set_value("—")
            tile.set_peak("")

        self.setWindowTitle(f"Missile Command — training console · {run_dir}")
        self._refresh_model()  # not on the next rescan: it would be the old run's
        self._compare_with(None)  # a comparison is against *this* run, not the last
        self._refresh_picker()
        self.statusBar().showMessage(f"watching {run_dir / sources.METRICS_NAME}")

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
        split.addWidget(self._side())
        split.setStretchFactor(0, 1)
        split.setSizes([980, 320])
        layout.addWidget(split, stretch=1)
        layout.addWidget(self._log_pane())
        return root

    def _header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        title = QLabel("MISSILE COMMAND · TRAINING CONSOLE")
        title.setProperty("role", "title")
        # Runs pile up one directory per experiment, so which one you are looking
        # at is a thing you change often — often enough that it belongs in the
        # window rather than in the command that started it.
        self._picker = QComboBox()
        self._picker.setMinimumWidth(220)
        self._picker.currentIndexChanged.connect(self._picked)
        # And which run it is being *held against*. Beside the picker rather than
        # behind a button: "did that change help?" is asked of the same two runs
        # repeatedly, and the answer is the two curves on one plot.
        versus = QLabel("vs")
        versus.setProperty("role", "caption")
        self._compare_picker = QComboBox()
        self._compare_picker.setMinimumWidth(180)
        self._compare_picker.currentIndexChanged.connect(self._compare_picked)
        self._status = QLabel("NO RUN")
        self._status.setProperty("role", "caption")
        row.addWidget(title)
        row.addSpacing(12)
        row.addWidget(self._picker)
        row.addSpacing(6)
        row.addWidget(versus)
        row.addWidget(self._compare_picker)
        row.addStretch(1)
        row.addLayout(self._controls())
        row.addSpacing(14)
        row.addWidget(self._status)
        return row

    def _controls(self) -> QHBoxLayout:
        """Three affordances, not a dashboard of them.

        One primary button that changes meaning, Stop beside it, and Reset kept
        at arm's length because it is the one that abandons a run.
        """
        row = QHBoxLayout()
        row.setSpacing(6)
        self._primary = QPushButton("Start")
        self._primary.setProperty("role", "primary")
        self._primary.clicked.connect(self._primary_pressed)
        self._stop = QPushButton("Stop")
        self._stop.clicked.connect(self._stop_pressed)
        self._reset = QPushButton("Reset…")
        self._reset.clicked.connect(self._reset_pressed)
        self._log_toggle = QPushButton("Log")
        self._log_toggle.setCheckable(True)
        self._log_toggle.toggled.connect(self._show_log)
        for button in (self._primary, self._stop, self._reset, self._log_toggle):
            row.addWidget(button)
        return row

    def _log_pane(self) -> QWidget:
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.setFixedHeight(150)
        self._log.setVisible(False)
        return self._log

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

    def _side(self) -> QWidget:
        """The right-hand column: what the run has produced, and what it is.

        Episodes above, the network below, the machine under both — three
        questions asked a few times a run rather than watched, so none of them
        is allowed to take space from the curve.

        The machine's row is three meters however long the run has been going,
        so it is pinned at the foot. The other two share what is left through a
        splitter: a network has as many layers as it has, but a run accumulates
        episodes for hours, and how many of them you want to see at once is a
        judgement only the person watching can make.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self._model = ModelPanel()
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self._recordings())
        split.addWidget(self._model)
        split.setStretchFactor(0, 1)  # a taller window is more episodes, not more layers
        split.setStretchFactor(1, 0)
        split.setSizes([380, 180])
        # The machine's own row goes at the foot of this column rather than in
        # the tiles: this side had the space, and the curve is not allowed to
        # lose any.
        self._system = SystemPanel()
        layout.addWidget(split, stretch=1)
        layout.addWidget(self._system)
        return panel

    def _recordings(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # The buttons live in the caption's own row rather than under the list:
        # this column's height is the scarce thing, and a strip of controls
        # across the bottom would cost an episode or two of it.
        head = QHBoxLayout()
        head.setSpacing(6)
        caption = QLabel("recordings")
        caption.setProperty("role", "caption")
        head.addWidget(caption)
        head.addStretch(1)
        # Double-click has always played an episode and still does — it was just
        # never an affordance you could *see*. These are, and Delete is the one
        # thing you could previously only do from a file manager.
        self._play = QPushButton("▶ Play")
        self._play.setToolTip("Open the selected episode in the game (or double-click it)")
        self._play.clicked.connect(self._play_selected)
        self._delete = QPushButton("Delete…")
        self._delete.setToolTip("Delete the selected recording from this run directory")
        for button in (self._play, self._delete):
            button.setProperty("role", "compact")
            head.addWidget(button)
        self._delete.clicked.connect(self._delete_selected)
        layout.addLayout(head)

        self._list = QListWidget()
        self._list.itemActivated.connect(self._open)
        self._list.itemDoubleClicked.connect(self._open)
        self._list.itemSelectionChanged.connect(self._selection_changed)
        # Four rows, so a network with many layers below cannot squeeze the list
        # down to a scrollbar with one episode in it.
        self._list.setMinimumHeight(140)
        layout.addWidget(self._list, stretch=1)
        # A greyed-out list row is painted from the disabled palette, which under
        # this stylesheet is invisible; an empty state has to be a real widget.
        self._no_recordings = QLabel(NO_RECORDINGS)
        self._no_recordings.setProperty("role", "placeholder")
        self._no_recordings.setWordWrap(True)
        self._no_recordings.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._no_recordings.setVisible(False)
        layout.addWidget(self._no_recordings, stretch=1)
        return panel

    # ---- the poll -----------------------------------------------------------
    def _tick(self) -> None:
        self._ticks += 1
        self._read_metrics()
        self._read_evals()
        self._read_comparison()
        self._read_log()
        self._system.refresh()
        if self._ticks % RESCAN_EVERY == 1:
            self._refresh_recordings()
            self._refresh_model()
            self._refresh_picker()  # a new run directory can appear at any time
        self._refresh_status()

    def _read_metrics(self) -> None:
        batch = self._metrics.poll()
        if batch.restarted:  # a different run writes into the same file
            for curve in (self._return, self._entropy, self._value):
                curve.clear()
            self._updates = 0
            self._last_metric = None
            self._peak_return.clear()
            self._peak_entropy.clear()
            for tile in (self._tile_update, self._tile_return, self._tile_entropy):
                tile.set_value("—")
                tile.set_peak("")
        for row in batch.rows:
            self._return.append(row.update, row.mean_return)
            self._entropy.append(row.update, row.entropy)
            self._value.append(row.update, row.value_loss)
            # Every row, not only the last of the batch: a poll can carry a
            # hundred updates, and the peak is often not in the newest one.
            self._peak_return.offer(row.update, row.mean_return)
            self._peak_entropy.offer(row.update, row.entropy)
            self._updates += 1
        if not batch.rows:
            return
        row = self._last_metric = batch.rows[-1]
        self._score.set_x_extent(row.update)  # the eval chart spans the run too
        self._tile_update.set_value(f"{row.update:,}")
        self._tile_update.set_note(f"{row.samples:,} samples")
        self._tile_return.set_value(_number(row.mean_return, "{:,.1f}"))
        self._tile_entropy.set_value(_number(row.entropy, "{:.3f}"))
        self._tile_return.set_peak(sources.peak_note(self._peak_return, "{:,.1f}"))
        self._tile_entropy.set_peak(sources.peak_note(self._peak_entropy, "{:.3f}"))

    def _read_evals(self) -> None:
        batch = self._evals.poll()
        if batch.restarted:
            self._score.clear()
            self._last_eval = None
            self._eval_rows.clear()
            self._peak_score.clear()
            self._tile_score.set_value("—")
            self._tile_score.set_peak("")
        for row in batch.rows:
            self._score.append(row.update, row.mean_score)
            self._eval_rows[row.update] = row
            self._peak_score.offer(row.update, row.mean_score)
        if not batch.rows:
            return
        self._tile_score.set_peak(sources.peak_note(self._peak_score, "{:,.0f}"))
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
            self._selection_changed()
            return
        now = time.time()
        for recording in found:
            # One line, not two. This column is the scarce space in the window
            # and a run leaves an episode every few minutes, so how *many* you
            # can see at once is worth more than the second line's air.
            item = QListWidgetItem(
                f"{recording.name}   {sources.human_age(now - recording.modified)} · "
                f"{sources.human_size(recording.size)}"
            )
            item.setData(Qt.ItemDataRole.UserRole, str(recording.path))
            self._list.addItem(item)
            if recording.path == selected:
                self._list.setCurrentItem(item)
        self._selection_changed()

    def _refresh_model(self) -> None:
        """Re-read the model card and the checkpoints.

        Both appear *during* a run — the card when the trainer starts, a
        checkpoint every `--checkpoint-every` updates — so this is a poll rather
        than something done once on attach. It is a small JSON and one directory
        listing, and the panel repaints only when the text actually changed.
        """
        self._model.show_run(
            modelcard.read(self._run_dir),
            sources.list_checkpoints(self._run_dir),
            self._eval_rows,
        )

    def _refresh_picker(self) -> None:
        """Rebuild the run list, without disturbing anyone reading it.

        Rebuilding fires ``currentIndexChanged``, which would re-attach on every
        tick, so the signals are blocked while the items are replaced — and an
        open dropdown is left alone entirely rather than being yanked out from
        under the pointer.
        """
        if self._picker.view().isVisible() or self._compare_picker.view().isVisible():
            return
        choices = sources.run_choices(self._run_dir)
        current = self._run_dir.resolve()
        if choices != self._choices:
            self._choices = choices
            self._picker.blockSignals(True)
            self._picker.clear()
            for path in choices:
                self._picker.addItem(path.name, str(path))
                self._picker.setItemData(
                    self._picker.count() - 1, str(path), Qt.ItemDataRole.ToolTipRole
                )
                if path == current:
                    self._picker.setCurrentIndex(self._picker.count() - 1)
            self._picker.setToolTip(str(current))
            self._picker.blockSignals(False)

        # Everything except the run it would be compared with — a run held
        # against itself is two identical curves and no information.
        others = [path for path in choices if path != current]
        if others == self._compare_choices:
            return
        self._compare_choices = others
        selected = self._compare_dir
        self._compare_picker.blockSignals(True)
        self._compare_picker.clear()
        self._compare_picker.addItem(NO_COMPARISON, "")
        for path in others:
            self._compare_picker.addItem(path.name, str(path))
            self._compare_picker.setItemData(
                self._compare_picker.count() - 1, str(path), Qt.ItemDataRole.ToolTipRole
            )
            if path == selected:
                self._compare_picker.setCurrentIndex(self._compare_picker.count() - 1)
        self._compare_picker.setEnabled(bool(others))
        self._compare_picker.blockSignals(False)
        if selected is not None and selected not in others:
            # The run it was being held against is gone, or has become the
            # attached one. Say so by drawing nothing rather than by leaving a
            # curve that no longer has a name in the picker.
            self._compare_with(None)

    def _picked(self, index: int) -> None:
        chosen = self._picker.itemData(index)
        if chosen and Path(str(chosen)) != self._run_dir.resolve():
            self._attach(Path(str(chosen)))

    # ---- the other run ------------------------------------------------------
    def _compare_with(self, run_dir: Path | None) -> None:
        """Hold the attached run against another one, or against nothing.

        Only the two curve files are read: an experiment is worth comparing on
        what it *scored* and how it got there, and the rest of the window stays
        about the run you are actually driving.
        """
        self._compare_dir = run_dir
        self._compare_metrics = None if run_dir is None else sources.metrics_tail(run_dir)
        self._compare_evals = None if run_dir is None else sources.evals_tail(run_dir)
        for curve in (self._score, self._return, self._entropy, self._value):
            if run_dir is None:
                curve.hide_comparison()
            else:
                curve.set_comparison(run_dir.name)
        for tile in (self._tile_update, self._tile_score, self._tile_return, self._tile_entropy):
            tile.set_compare("")

    def _compare_picked(self, index: int) -> None:
        chosen = self._compare_picker.itemData(index)
        self._compare_with(Path(str(chosen)) if chosen else None)

    def _read_comparison(self) -> None:
        """Poll the other run — it may be live too, and often is."""
        if self._compare_metrics is None or self._compare_evals is None:
            return
        metrics = self._compare_metrics.poll()
        if metrics.restarted:
            for curve in (self._return, self._entropy, self._value):
                curve.clear_comparison()
        for row in metrics.rows:
            self._return.append_comparison(row.update, row.mean_return)
            self._entropy.append_comparison(row.update, row.entropy)
            self._value.append_comparison(row.update, row.value_loss)
        if metrics.rows:
            last = metrics.rows[-1]
            self._tile_update.set_compare(f"{self._name()} {last.update:,}")
            self._tile_return.set_compare(_compare_note(self._name(), last.mean_return, "{:,.1f}"))
            self._tile_entropy.set_compare(_compare_note(self._name(), last.entropy, "{:.3f}"))

        evals = self._compare_evals.poll()
        if evals.restarted:
            self._score.clear_comparison()
        for row in evals.rows:
            self._score.append_comparison(row.update, row.mean_score)
        if evals.rows:
            # Its *latest* score, not its best: the tile above shows this run's
            # latest, and two tiles reporting different statistics under the same
            # caption is a comparison you cannot make.
            last = evals.rows[-1]
            self._tile_score.set_compare(f"{self._name()} {last.mean_score:,.0f}")

    def _name(self) -> str:
        return "—" if self._compare_dir is None else self._compare_dir.name

    def _refresh_status(self) -> None:
        modified = sources.last_modified(self._run_dir / sources.METRICS_NAME)
        # Two different kinds of empty, and saying which is the whole job of an
        # empty state: no run at all, versus a run whose episodes have not ended.
        self._return.set_placeholder(WAITING if modified is None else NO_EPISODES)
        for curve in (self._entropy, self._value):
            curve.set_placeholder("" if modified is not None else WAITING)

        state = self._state(modified)
        self._set_status(*STATUS[state])
        # Idle with nothing that could train is the fourth meaning of the primary
        # button: it offers to fix that rather than being a dead control with an
        # explanation on it. Watching a run from a machine with no torch stays a
        # supported way to use this — only Start was ever gated.
        idle_label = "Start" if can_train() else "Set up training…"
        self._primary.setText({"paused": "Resume", "live": "Pause"}.get(state, idle_label))
        self._primary.setEnabled(state != "stopping")
        self._stop.setEnabled(state in ("live", "paused"))

        if modified is None:
            self.statusBar().showMessage(self._nothing_here())
            return
        age = max(time.time() - modified, 0.0)
        windows = self._launcher.running
        replays = f" · {windows} replay window(s) open" if windows else ""
        self.statusBar().showMessage(
            f"{self._updates:,} updates · last write {sources.human_age(age)}{replays}"
        )

    def _nothing_here(self) -> str:
        """Why this directory is empty — and where the runs actually are.

        Runs pile up one directory per experiment, so "no metrics.csv" usually
        means the console is aimed one level too high. Saying which directories
        do hold a run turns a dead end into the next command to type.
        """
        inside = sources.find_runs(self._run_dir)
        if not inside:
            # "Press Start" is the wrong next step on a machine that has nothing
            # to start one *with*; the button says so too, and the two must agree.
            next_step = (
                "press Start, or run `poe train` in a terminal"
                if can_train()
                else "press Set up training to install PyTorch"
            )
            return f"no {sources.METRICS_NAME} in {self._run_dir} yet — {next_step}"
        names = ", ".join(run.name for run in inside[:4])
        more = f" (+{len(inside) - 4} more)" if len(inside) > 4 else ""
        return (
            f"no run in {self._run_dir} itself · {len(inside)} inside it: {names}{more} "
            f"— open one with `poe ui -- {inside[0].as_posix()}`"
        )

    def _state(self, modified: float | None) -> str:
        """What the run is doing, from the files alone.

        Which is why it works for a run this console never started: the control
        files and the metrics timestamp are all it reads.
        """
        if self._control.stopping():
            return "stopping"
        if self._control.paused():
            return "paused"
        if self._run is not None:
            # Our own child: no guessing needed, and no pretending a run that
            # just exited is live because its last line is thirty seconds old.
            return "idle" if self._run.finished else "live"
        if modified is None:
            return "none"
        return "live" if (time.time() - modified) < LIVE_AFTER_S else "idle"

    def _set_status(self, text: str, colour: str) -> None:
        if self._status.text() != text:
            self._status.setText(text)
            self._status.setStyleSheet(f"color: {colour}; font-weight: 600;")

    # ---- control ------------------------------------------------------------
    def _primary_pressed(self) -> None:
        """One button, three meanings — whichever the run's state makes sensible."""
        if self._control.paused():
            self._control.resume()
            self.statusBar().showMessage("resuming — the loop picks it up within an update")
        elif self._state(sources.last_modified(self._run_dir / sources.METRICS_NAME)) == "live":
            self._control.request_pause()
            self.statusBar().showMessage(
                f"pausing after the current update — {self._control.pause_file}"
            )
        elif not can_train():
            self._set_up_runtime()
        else:
            self._start()

    def _set_up_runtime(self) -> None:
        """Install the runtime, then carry straight on into the new-run dialog.

        Setup is a means, not a destination: someone who pressed the button
        wanted to train, so a successful install continues to the thing they were
        actually after rather than returning them to a window with a new button
        on it.
        """
        dialog = RuntimeDialog(parent=self)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        self._refresh_status()  # the button's meaning has probably just changed
        if accepted and can_train():
            self._start()

    def _start(self) -> None:
        out_dir = self._run_dir.resolve()
        dialog = ParameterDialog(
            read_params(TRAINER_SOURCES),
            python=training_python(),
            out_dir=out_dir,
            # What is already in this directory, so continuing a run is a choice
            # from a list rather than a path typed from memory.
            checkpoints=sources.list_checkpoints(out_dir),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._control.clear()  # a STOP left by the last run would end this one at once
        command = dialog.command()
        self._log.clear()
        self._append_log(f"$ {' '.join(command)}")
        self._log_toggle.setChecked(True)
        self._run = TrainingRun(command, cwd=PROJECT_ROOT)
        self._reported_exit = False

    def _stop_pressed(self) -> None:
        self._control.request_stop()
        self.statusBar().showMessage(
            "stop requested — the run finishes this update, writes a final checkpoint and exits"
        )

    def _reset_pressed(self) -> None:
        """Start over somewhere new. Destructive only in the sense of moving on."""
        target = sources.next_run_dir(self._run_dir)
        answer = QMessageBox.question(
            self,
            "Start a fresh run directory?",
            f"The console will attach to <b>{target}</b> and the next Start writes there."
            f"<br><br>Nothing in <b>{self._run_dir}</b> is deleted — its checkpoints, "
            "recordings and metrics stay exactly where they are. Stop the run there "
            "first if it is still going.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Ok:
            self._attach(target)

    def _show_log(self, shown: bool) -> None:
        self._log.setVisible(shown)

    def _append_log(self, text: str) -> None:
        self._log.appendPlainText(text)

    def _read_log(self) -> None:
        """Drain what the run has printed, and notice when it is over.

        Two sources, never both: our own child's pipe if we have one, otherwise
        the file the trainer writes. Reading both would double every line of a
        run this console started, since the trainer tees rather than redirects.
        """
        if self._run is None:
            batch = self._log_file.poll()
            if batch.restarted:
                self._log.clear()  # a fresh run in this directory, from its first line
            for line in batch.rows:
                self._append_log(line)
            return
        for line in self._run.drain():
            self._append_log(line)
        code = self._run.exit_code()
        if code is None or self._reported_exit:
            return
        self._reported_exit = True
        self._append_log(f"— the run exited with code {code} —")
        if code != 0:
            # A run that died on its first line is the case where the log is the
            # only thing that can tell you why, so it opens itself.
            self._log_toggle.setChecked(True)

    # ---- opening an episode -------------------------------------------------
    def _selected_path(self) -> Path | None:
        # selectedItems() rather than currentItem(), which is only *sometimes*
        # a widget and whose stubs say it always is.
        selected = self._list.selectedItems()
        if not selected:
            return None
        data = selected[0].data(Qt.ItemDataRole.UserRole)
        return Path(str(data)) if data else None

    def _selection_changed(self) -> None:
        """Play and Delete act on a selection, so they are off without one."""
        chosen = self._selected_path() is not None
        self._play.setEnabled(chosen)
        self._delete.setEnabled(chosen)

    def _open(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if data:
            self._play_recording(Path(str(data)))

    def _play_selected(self) -> None:
        path = self._selected_path()
        if path is not None:
            self._play_recording(path)

    def _play_recording(self, path: Path) -> None:
        try:
            self._launcher.launch(path)
        except AppNotFound as error:
            QMessageBox.warning(self, "The game is not built", str(error))
            return
        self.statusBar().showMessage(f"playing {path.name}")

    def _delete_selected(self) -> None:
        """Remove one episode from the run directory.

        It confirms, because this is the console's only destructive act on a
        file — everything else it writes is a control marker. A recording is
        cheap to regenerate only while the run that wrote it is still going, so
        the dialog names the file rather than asking "are you sure".
        """
        path = self._selected_path()
        if path is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete this recording?",
            f"<b>{path.name}</b> will be removed from {path.parent}."
            "<br><br>The trainer writes a new episode every --record-every "
            "updates; one from a finished run does not come back.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Ok:
            return
        try:
            path.unlink()
        except OSError as error:
            # Windows holds a lock on a file the game has open, and the honest
            # answer is which file and why — not a traceback into the log pane.
            QMessageBox.warning(self, "The recording could not be deleted", str(error))
            return
        self._listed = None  # force the rescan to notice, mtimes unchanged or not
        self._refresh_recordings()
        self.statusBar().showMessage(f"deleted {path.name}")


def _same(left: list[Recording], right: list[Recording]) -> bool:
    """Same files, same timestamps — so the list is left exactly as it is."""
    return [(r.path, r.modified) for r in left] == [(r.path, r.modified) for r in right]


def _number(value: float | None, spec: str) -> str:
    """A measurement the trainer has not produced yet is a dash, never a zero."""
    return "—" if value is None else spec.format(value)


def _compare_note(name: str, value: float | None, spec: str) -> str:
    """``runs-2 13.4`` — the other run's value under this one's."""
    return f"{name} {_number(value, spec)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch a Missile Command training run.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=None,
        help="the run's --out-dir (default: ./runs in a checkout, else the user data dir)",
    )
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Missile Command training console")
    app.setStyleSheet(theme.stylesheet())
    window = Console(paths.runs_dir(args.run_dir))
    # Roomy, but never bigger than the desktop it opens on — a window whose
    # status line is off-screen is a window with a bug in it.
    available = app.primaryScreen().availableSize()
    window.resize(min(1360, available.width() - 60), min(860, available.height() - 60))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
