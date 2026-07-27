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

The run is the subject, so the score curve gets the space and the diagnostics and
the episode list are strips around it.

**Two tabs, and only two** (docs/ROADMAP.md, M8, amended 2026-07-26 at the
human's request; the original rule was one screen and no tab bar). **TRAINING**
is the screen above — what you watch while a run goes. **STATISTICS**
(:mod:`md.ui.analysis`) is what you read when it stops improving: the full
per-episode stat block, the kills-per-shot distribution, and the curves that say
*why*. They are two activities rather than two views, and everything *around* the
plots is shared, so switching a tab never changes which run is on screen — which
is the hunting the original rule was written against.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import library as run_library
from .. import modelcard, paths
from ..control import Control
from . import about, sources, theme
from .analysis import AnalysisView
from .charts import CurveView
from .forms import ParameterDialog
from .league import LeagueView, PromoteDialog
from .library import LibraryView
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
    "The trainer scores the policy on the 32 validation seeds every\n"
    "--eval-every updates and appends it to evals.csv."
)
#: The compare picker's first entry, and its default. Comparing is something you
#: ask for; the window's subject is one run.
NO_COMPARISON = "no comparison"

#: The stops the eval slider offers, in updates between scores. Discrete and
#: roughly logarithmic, because "every 37 updates" is not a thought anyone has —
#: a continuous slider would invite one and make the useful end, 1 to 25,
#: unhittable. The first stop is the trainer's own "never score", spelled out.
EVAL_EVERY_STOPS = (0, 1, 2, 5, 10, 25, 50, 100, 250, 500)
#: What the slider falls back to when nothing has published a value, if the
#: trainer's own default cannot be read beside this console.
EVAL_EVERY_FALLBACK = 10
EVAL_EVERY_HELP = (
    "How often the run scores itself on the 32 validation seeds.\n\n"
    "Drag it while the run goes: the trainer re-reads TUNING.json every update, so a "
    "new interval takes effect from the next one — no restart, no lost checkpoint.\n\n"
    "Often early, when the policy changes shape every few updates; less often later, "
    "when an eval plays 32 full-length episodes to repeat what the last one said."
)
#: Why the slider is greyed out. A run started before this existed — or no run at
#: all — publishes nothing, and a control that wrote into the void is worse than
#: one that says it has nothing to drive.
EVAL_EVERY_UNPUBLISHED = (
    "No run here is publishing an eval interval.\n\n"
    "A run writes TUNING.json when it starts, so this slider drives any run started "
    "since — including one started from a terminal."
)

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
        #: The directory the library lists, decided **once** from what the
        #: console was opened on and never re-derived. Derived, because the two
        #: differ exactly when it matters — `poe ui -- runs/amber-anvil` opens
        #: one run and its library is `runs/`, while `poe ui -- runs` opens a
        #: directory of them and is its own library. Once, because the answer
        #: has to survive the run directory ceasing to be a run: a new run
        #: nobody has started yet and a run somebody has just deleted both look
        #: exactly like "a directory that is not a run", and re-deriving would
        #: quietly move the library to the wrong level of the tree.
        self._library_dir = run_dir.parent if run_library.load_run(run_dir) is not None else run_dir
        # Land on the library when the directory holds *several* runs, and on
        # the run itself when it holds one. Both are what the person meant:
        # `poe ui` on an experiment directory wants the list, and
        # `poe ui -- runs/amber-anvil` wants that run.
        self._library.attach(self._library_root())
        self._league.refresh()
        self._pages.setCurrentIndex(0 if run_library.load_run(run_dir) is None else 1)

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
        #: A display name typed into the New Run dialog and not yet on disk.
        #: Cleared here: attaching somewhere else means it was for a directory
        #: this window is no longer looking at.
        self._pending_name = ""
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
        #: The newest contiguous protocol segment, which is the only set a
        #: single curve and peak can compare without joining unlike scores.
        self._display_eval_rows: list[EvalRow] = []
        #: The comparison run's evaluations, kept rather than only plotted. The
        #: score curve consumes each row as it arrives; the statistics view needs
        #: the *series* and the latest row together, and re-reading the file a
        #: second time would be two tails on one path disagreeing about EOF.
        self._compare_eval_rows: list[EvalRow] = []
        for curve in (self._score, self._return, self._entropy, self._value):
            curve.clear()
        self._score.set_baseline(None)
        for tile in (self._tile_update, self._tile_score, self._tile_return, self._tile_entropy):
            tile.set_value("—")
            tile.set_peak("")

        self.setWindowTitle(f"Missile Defense — training console · {run_dir}")
        self._refresh_model()  # not on the next rescan: it would be the old run's
        self._compare_with(None)  # a comparison is against *this* run, not the last
        self._read_tuning()  # the box describes the run it is aimed at
        self._refresh_picker()
        self.statusBar().showMessage(f"watching {run_dir / sources.METRICS_NAME}")

    # ---- construction -------------------------------------------------------
    def _build(self) -> QWidget:
        """Two levels, not two tabs.

        The **library** lists every run and every promoted model; the **run**
        screen is about one of them. That is a navigation step and not a peer
        view — the run picker in the header cannot answer "which of these eleven
        is worth my attention", because a dropdown shows names where the
        question needs scores, sizes and states side by side.

        The library is the landing view (docs/ROADMAP.md, M8): being dropped
        into whichever run happened to sort first is how you read the wrong
        curve for a minute.
        """
        self._pages = QStackedWidget()
        self._pages.addWidget(self._library_page())
        self._pages.addWidget(self._run_page())
        return self._pages

    def _library_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        title = QLabel("MISSILE DEFENSE · TRAINING CONSOLE")
        title.setProperty("role", "title")
        layout.addWidget(title)

        split = QSplitter(Qt.Orientation.Horizontal)
        self._library = LibraryView(on_new_run=self._new_run_from_library)
        self._library.opened.connect(self._open_run)
        self._league = LeagueView()
        self._league.watch.connect(self._watch_model)
        self._league.show_match.connect(self._watch_match)
        split.addWidget(self._library)
        split.addWidget(self._league)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([760, 520])
        layout.addWidget(split, stretch=1)
        return page

    def _run_page(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 12, 14, 8)
        layout.setSpacing(10)
        layout.addLayout(self._header())
        layout.addLayout(self._tiles())

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._main_tabs())
        split.addWidget(self._side())
        split.setStretchFactor(0, 1)
        split.setSizes([980, 320])
        layout.addWidget(split, stretch=1)
        layout.addWidget(self._log_pane())
        return root

    # ---- moving between the two levels ---------------------------------------

    def _show_library(self) -> None:
        """Back to the list, and re-read it: a run may have finished meanwhile."""
        self._library.attach(self._library_root())
        self._league.refresh()
        self._pages.setCurrentIndex(0)
        # Focus lands on the list, not wherever Qt left it. Arriving on a screen
        # with the keyboard parked on nothing means the first Tab is spent
        # finding out where you are, every single time.
        self._library.table.focus_list()

    def _open_run(self, run_dir: Path) -> None:
        self._attach(run_dir)
        self._pages.setCurrentIndex(1)
        self._tick()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 — Qt's name
        """Escape goes back to the library, from anywhere on a run screen.

        The one navigation this window has, and it was reachable only by
        clicking a button in the corner. Escape is what every other back
        control in both binaries answers to, including the game's.
        """
        if event.key() == Qt.Key.Key_Escape and self._pages.currentIndex() == 1:
            self._show_library()
            return
        super().keyPressEvent(event)

    def _new_run_from_library(self) -> None:
        """Name a run, then start it, in a directory of its own.

        Fresh rather than "the one that sorted first": a New Run button that
        silently reset an existing directory would be the most destructive
        control in the window, and Reset already exists for that with a
        confirmation in front of it.

        **The name is asked for before anything else**, because it is the one
        decision the parameter form cannot carry: a run is identified by its
        directory for the rest of its life, and a generated `amber-anvil` is a
        handle rather than an answer to "which experiment is this?". Prefilled
        with a generated one all the same, so Enter is still a whole answer.
        """
        root = self._library_root()
        existing = [run.name for run in run_library.discover(root)]
        suggested = run_library.default_name(existing)
        name, accepted = QInputDialog.getText(
            self, "New run", "A name for this run:", text=suggested
        )
        if not accepted:
            return
        # An emptied field means "I did not care after all", which is what the
        # suggestion was for — not a run called nothing.
        name = name.strip() or suggested
        target = run_library.new_run_dir(root, name)
        self._open_run(target)
        # The typed name only becomes a *display* name when the directory could
        # not be called that: `entity-3-seed` needs no second copy of itself,
        # and `Entity policy, 3 seeds` does. Written when the run starts rather
        # than now, so a parameter dialog somebody cancels leaves nothing on
        # disk at all — see `_start`.
        self._pending_name = name if name != target.name else ""
        self._primary_pressed()

    def _library_root(self) -> Path:
        """Where the library looks: the directory *containing* runs.

        Settled in `__init__` from what the console was opened on rather than
        from `paths.runs_dir()`, which would show a list of somebody else's runs.
        """
        return self._library_dir

    def _watch_model(self, policy: Path) -> None:
        """Open the game on a promoted model — the league's `Watch it play`.

        `--watch-model` rather than a recording: this is the model as it plays
        *now*, against a fresh seed, which is the question a league table
        provokes and a stored episode cannot answer.
        """
        try:
            self._launcher.launch_model(policy)
        except AppNotFound as error:
            QMessageBox.warning(self, "The game is not built", str(error))

    def _watch_match(self, manifest: Path) -> None:
        """Open a head-to-head split-screen — the league's `Watch the match`."""
        try:
            self._launcher.launch_match(manifest)
        except AppNotFound as error:
            QMessageBox.warning(self, "The game is not built", str(error))

    def _header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        title = QLabel("MISSILE DEFENSE · TRAINING CONSOLE")
        title.setProperty("role", "title")
        # The version, always on screen rather than behind a menu: "which build
        # is this?" is the first question of every bug report, and the console is
        # the half most often installed from a package by someone with no
        # checkout to read it out of. Pressing it opens the rest — author,
        # licence, and the LGPL libraries this MIT program runs on, which a user
        # should be able to learn from the program and not only from a file in a
        # repository they have never opened.
        self._about = QPushButton(f"v{about.version()}")
        self._about.setProperty("role", "version")
        self._about.setToolTip("About Missile Defense")
        self._about.setCursor(Qt.CursorShape.PointingHandCursor)
        self._about.clicked.connect(self._show_about)
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
        # Back to the list. First in the row because it is a *level*, not an
        # action on this run — the same place a browser puts one.
        self._back = QPushButton("‹ &Library")
        self._back.setToolTip("Every run and every promoted model (Escape)")
        self._back.clicked.connect(self._show_library)
        row.addWidget(self._back)
        row.addSpacing(10)
        row.addWidget(title)
        row.addWidget(self._about)
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
        at arm's length because it is the one that abandons a run. The eval
        interval sits with them because it is the same kind of thing: something
        you do *to* the run that is going, not a parameter you chose before it.
        """
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addLayout(self._eval_control())
        row.addSpacing(10)
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

    def _eval_control(self) -> QHBoxLayout:
        """The one hyperparameter that belongs on the screen you leave open.

        Every other knob is chosen before a run and lives in the start dialog.
        This one has no right answer for a whole run: you want the yardstick
        constantly while the policy is still finding its shape, and hardly at all
        once each eval costs more than the update it interrupts. Deciding that
        from the curve, without stopping, is the point.

        A slider rather than a number box, because it is a *dial*: the question
        is "more often or less often than now", asked while looking at the curve,
        and the answer is a direction. Its stops are the intervals worth having
        (:data:`EVAL_EVERY_STOPS`), so there is nothing to mistype and no way to
        land on a number nobody meant.

        It writes :mod:`md.control`'s tuning file and nothing else — so it drives
        a run this console never started, and a terminal can do the same with
        ``echo``. Nothing here imports the trainer.
        """
        row = QHBoxLayout()
        row.setSpacing(6)
        caption = QLabel("eval every")
        caption.setProperty("role", "caption")
        #: The stops, which a run is allowed to add to: one started with
        #: ``--eval-every 30`` must be shown where it actually is.
        self._eval_stops: list[int] = list(EVAL_EVERY_STOPS)
        self._eval_every = QSlider(Qt.Orientation.Horizontal)
        self._eval_every.setFixedWidth(116)
        self._eval_every.setRange(0, len(self._eval_stops) - 1)
        self._eval_every.setPageStep(1)  # a click beside the handle is one stop
        self._eval_every.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._eval_every.setTickInterval(1)
        self._eval_every.setToolTip(EVAL_EVERY_HELP)
        # The number, because a slider on its own says "more" and "less" and
        # never says *what*. Monospace and fixed-width, so the row does not
        # twitch as the value changes under the handle.
        self._eval_readout = QLabel()
        self._eval_readout.setProperty("role", "readout")
        self._eval_readout.setMinimumWidth(58)
        self._eval_readout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Seeded before the signals are connected: a slider that wrote a tuning
        # file while it was being built would tune a directory nobody looked at.
        self._show_eval_interval(_default_eval_every())
        # Dragging is one decision, not the forty values it passes through, so
        # the write waits for the handle to be let go. Arrow keys and clicks are
        # already one decision each and take effect at once.
        self._eval_every.valueChanged.connect(self._eval_every_moved)
        self._eval_every.sliderReleased.connect(self._eval_every_settled)
        row.addWidget(caption)
        row.addWidget(self._eval_every)
        row.addWidget(self._eval_readout)
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
        self._tile_score = StatTile("eval score", "validation or held-out benchmark")
        self._tile_return = StatTile("mean return", "shaped, scaled — not a score")
        self._tile_entropy = StatTile("entropy", "how undecided the policy is")
        for tile in (self._tile_update, self._tile_score, self._tile_return, self._tile_entropy):
            row.addWidget(tile)
        return row

    def _main_tabs(self) -> QWidget:
        """Two views of the same run, sharing everything around them.

        The tiles, the picker, the episode list and the log stay put; only the
        plot area changes, because the two views answer questions at different
        moments. **Training** is what you watch while a run goes — is it
        learning, is it stable. **Statistics** is what you open when it stops
        improving and you want to know why, and it is the wrong shape for a
        glance: fourteen numbers and a distribution reward being read, not
        monitored.

        A tab rather than a second window, so the comparison picked in the
        header applies to both and there is only ever one answer to "which run
        am I looking at".
        """
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._plots(), "TRAINING")
        self._analysis = AnalysisView()
        tabs.addTab(self._analysis, "STATISTICS")
        return tabs

    def _plots(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # The hero. Its rows carry protocol metadata; the held-out baseline is
        # drawn only when every plotted score actually used that protocol.
        self._score = CurveView(
            "policy evaluation score",
            theme.SCORE,
            value_format="%.0f",
            markers=True,  # an eval every --eval-every updates is dots, not a line
            series_name="learned policy",
            from_zero=True,
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
        # Promotion sits under the model panel because that panel is already
        # "what this run has produced" — and because promoting is something you
        # do *to* the thing described right above it, not a control that belongs
        # in the bar with Start and Stop.
        self._promote = QPushButton("Enter Model League…")
        self._promote.setToolTip(
            "Copy this run's best checkpoint into the league as a .mdp, where it "
            "outlives the run — and where the game finds it, under WATCH AI → MODELS"
        )
        self._promote.clicked.connect(self._promote_run)
        layout.addWidget(split, stretch=1)
        layout.addWidget(self._promote)
        self._system = SystemPanel()
        layout.addWidget(self._system)
        return panel

    def _promote_run(self) -> None:
        """Open the promotion dialog on the attached run.

        Re-read rather than cached: the run may have written three more
        checkpoints since the library last listed it, and the dialog's whole job
        is to offer the right one.
        """
        run = run_library.load_run(self._run_dir)
        if run is None:
            QMessageBox.information(
                self,
                "Nothing to promote",
                f"No run in {self._run_dir} yet — a run is promoted from its checkpoints, "
                "and this directory has none.",
            )
            return
        dialog = PromoteDialog(run, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.promoted is not None:
            self._league.refresh()
            # Says where it went *and* that the game can now play it. Promotion
            # is also the install step — the game scans this directory and
            # offers everything in it under WATCH AI -> MODELS — and a person
            # who is not told that has no reason to go and look.
            self.statusBar().showMessage(
                f"promoted {dialog.promoted.name} — the game can now play it "
                f"from WATCH AI → MODELS ({dialog.promoted.policy})"
            )

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
        self._read_tuning()
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
            self._score.set_baseline(None)
            self._rewind_comparison_evals()
            self._last_eval = None
            self._eval_rows.clear()
            self._display_eval_rows.clear()
            self._peak_score.clear()
            self._tile_score.set_value("—")
            self._tile_score.set_peak("")
            self._refresh_analysis()
        for row in batch.rows:
            if self._last_eval is not None and not sources.same_eval_series(self._last_eval, row):
                # A line between unlike protocols is invented evidence. Show
                # only the newest contiguous protocol segment and start its peak
                # again; the CSV still retains every historical row.
                self._score.clear()
                self._score.set_baseline(None)
                self._rewind_comparison_evals()
                self._display_eval_rows.clear()
                self._peak_score.clear()
            self._score.append(row.update, row.mean_score)
            self._eval_rows[row.update] = row
            self._display_eval_rows.append(row)
            self._peak_score.offer(row.update, row.mean_score)
            self._last_eval = row
        if not batch.rows:
            return
        self._tile_score.set_peak(sources.peak_note(self._peak_score, "{:,.0f}"))
        row = batch.rows[-1]

        all_canonical = bool(self._display_eval_rows) and all(
            sources.is_canonical_benchmark(item) for item in self._display_eval_rows
        )
        if all_canonical:
            self._score.set_baseline(
                BASELINE_MEAN_SCORE, f"scripted baseline {BASELINE_MEAN_SCORE:,.0f}"
            )
        else:
            self._score.set_baseline(None)

        if sources.is_canonical_benchmark(row):
            delta = row.mean_score - BASELINE_MEAN_SCORE
            ahead = delta > 0
            self._tile_score.set_value(
                f"{row.mean_score:,.0f}", theme.AHEAD if ahead else theme.BEHIND
            )
            self._tile_score.set_note(
                f"{abs(delta):,.0f} {'ahead of' if ahead else 'behind'} baseline "
                f"· update {row.update}"
            )
        else:
            self._tile_score.set_value(f"{row.mean_score:,.0f}")
            self._tile_score.set_note(f"{sources.eval_protocol_note(row)} · update {row.update}")
        self._refresh_analysis()

    def _refresh_analysis(self) -> None:
        """Hand the statistics view the rows both curves are already drawn from.

        Only on a change, never on the timer: the view rebuilds fourteen tiles,
        five bars and four curves, and a run evaluates every few minutes. Doing
        it on every poll would be a second of work per second for a screen whose
        numbers moved once an hour.
        """
        self._analysis.show_rows(self._display_eval_rows, self._compare_eval_rows)

    def _read_tuning(self) -> None:
        """Show what the run is on, without arguing with the hand on the slider.

        The file is the shared state, not this widget: a value set from a
        terminal, or by a run publishing what it was started with, has to land
        here too, or the slider would describe the last thing *it* said rather
        than the run.
        """
        published = self._control.tuning().get("eval_every")
        for widget in (self._eval_every, self._eval_readout):
            widget.setEnabled(published is not None)
        if published is None:
            self._eval_every.setToolTip(EVAL_EVERY_UNPUBLISHED)
            return
        self._eval_every.setToolTip(EVAL_EVERY_HELP)
        # Anything below zero is the trainer's "no evaluation" too, and the
        # slider has one stop for that rather than a negative one nobody asked
        # for. Mid-drag the hand wins; the next poll is a second away.
        if max(published, 0) == self._eval_shown() or self._eval_every.isSliderDown():
            return
        self._show_eval_interval(max(published, 0))

    def _eval_shown(self) -> int:
        """The interval the handle is currently sitting on."""
        return self._eval_stops[self._eval_every.value()]

    def _show_eval_interval(self, updates: int) -> None:
        """Move the handle to an interval without calling it a decision."""
        # A run may have been started with an interval between two stops.
        # Snapping the handle to a neighbour would be describing it wrongly, so
        # the scale gains a stop instead — and loses it again at the next value.
        stops = sorted({*EVAL_EVERY_STOPS, updates})
        if stops != self._eval_stops:
            self._eval_stops = stops
            self._eval_every.setMaximum(len(stops) - 1)
        self._eval_every.blockSignals(True)  # or reading would write it back
        try:
            self._eval_every.setValue(self._eval_stops.index(updates))
        finally:
            self._eval_every.blockSignals(False)
        self._eval_readout.setText(_eval_label(updates))

    def _eval_every_moved(self, index: int) -> None:
        """The handle is somewhere new — say so, and decide if that was a choice."""
        self._eval_readout.setText(_eval_label(self._eval_stops[index]))
        if not self._eval_every.isSliderDown():
            self._eval_every_settled()  # a click or an arrow key: already final

    def _eval_every_settled(self) -> None:
        updates = self._eval_shown()
        self._control.tune("eval_every", updates)
        cadence = "no longer evaluating" if updates == 0 else f"evaluating every {updates} updates"
        self.statusBar().showMessage(
            f"{cadence} — from the run's next update ({self._control.tuning_file})"
        )

    def _rewind_comparison_evals(self) -> None:
        """Reconsider the other run when this run's score protocol changes."""

        self._score.clear_comparison()
        self._tile_score.set_compare("")
        if self._compare_evals is not None:
            self._compare_evals.rewind()

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
        if run_dir is not None:
            # Said now, not when the first row arrives: picking a run and seeing
            # every tile stay blank reads as a broken feature, and the wait is
            # longest exactly when the other run has nothing evaluated yet.
            self._tile_score.set_compare(f"{self._name()} · reading…")
        self._compare_eval_rows.clear()
        self._analysis.set_comparison("" if run_dir is None else run_dir.name)
        self._refresh_analysis()

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

        # Do not consume the comparison scores before there is a primary
        # protocol to match them against. Otherwise selecting a comparison
        # while this run is still waiting for its first evaluation advances the
        # tail to EOF and the matching curve never appears.
        if self._last_eval is None:
            return
        evals = self._compare_evals.poll()
        if evals.restarted:
            self._score.clear_comparison()
            self._compare_eval_rows.clear()
        for row in evals.rows:
            self._compare_eval_rows.append(row)
            if self._last_eval is not None and sources.matching_eval_protocol(self._last_eval, row):
                self._score.append_comparison(row.update, row.mean_score)
        if evals.rows:
            self._refresh_analysis()
            # Its *latest* score, not its best: the tile above shows this run's
            # latest, and two tiles reporting different statistics under the same
            # caption is a comparison you cannot make.
            last = evals.rows[-1]
            if self._last_eval is not None and sources.matching_eval_protocol(
                self._last_eval, last
            ):
                self._tile_score.set_compare(f"{self._name()} {last.mean_score:,.0f}")
            else:
                # Nothing overlaid, and the reason on the tile. The full
                # sentence is on the STATISTICS tab, which has room for it; what
                # matters here is that the blank is explained rather than bare.
                self._score.clear_comparison()
                self._tile_score.set_compare(f"{self._name()} · not comparable")

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
        own = self._run
        if own is not None and own.finished:
            # Our own child, and it is over: no guessing needed, no pretending a
            # run that just exited is live because its last line is thirty
            # seconds old, and no marker file outliving the process it was for.
            return "idle"
        live = own is not None or (modified is not None and time.time() - modified < LIVE_AFTER_S)
        if self._control.stopping():
            # Qualified by liveness, because a stop is obeyed within one update:
            # a STOP still sitting in a directory nothing is writing to is a
            # leftover rather than a state. Pressing Stop in a second window
            # just after a run ended writes exactly that — and reporting it for
            # ever would disable the Start button that clears it, which is a
            # console wedged by its own status line.
            return "stopping" if live else "idle"
        if self._control.paused():
            # *Not* qualified the same way: a paused run writes nothing at all,
            # so a metrics.csv that has stopped moving is what being paused
            # looks like from out here. Only our own exited child (above) is
            # proof enough to override the file.
            return "paused"
        if own is not None:
            return "live"
        if modified is None:
            return "none"
        return "live" if live else "idle"

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
        if self._pending_name:
            # Now that there is a run to name. `rename` creates the directory if
            # the trainer has not got there yet, and writes atomically because
            # the trainer is writing everything else in it from this moment on.
            run_library.rename(out_dir, self._pending_name)
            self._pending_name = ""

    def _stop_pressed(self) -> None:
        self._control.request_stop()
        self.statusBar().showMessage(
            "stop requested — the run finishes this update, writes a final checkpoint and exits"
        )

    def _reset_pressed(self) -> None:
        """Start over somewhere new. Destructive only in the sense of moving on.

        Named here rather than only in the library, because this is the other
        way a run directory comes into existence and `high-delta-3` is exactly
        the name nobody can tell from `high-delta-2` a fortnight later. The
        suggestion is still that name, so the quick answer is Enter.
        """
        suggested = sources.next_run_dir(self._run_dir).name
        name, accepted = QInputDialog.getText(
            self,
            "Start a fresh run directory?",
            "A name for the new run. The next Start writes there instead.\n\n"
            f"Nothing in {self._run_dir.name} is deleted — its checkpoints,\n"
            "recordings and metrics stay where they are. Stop the run there\n"
            "first if it is still going.",
            text=suggested,
        )
        if not accepted:
            return
        name = name.strip() or suggested
        target = run_library.new_run_dir(self._library_root(), name)
        self._attach(target)
        # After attaching, which clears it: this name is for the directory the
        # window is looking at *now*.
        self._pending_name = name if name != target.name else ""

    def _show_about(self) -> None:
        """Who wrote this, which build it is, and what it is standing on.

        A plain box rather than a designed screen: it is opened once, read once,
        and closed, so the space it deserves is the space its text takes. The
        component list is the part that has to be here — the console runs on
        PySide6 and Qt Charts under the LGPL, and this is where a user meets that
        fact (:mod:`md.ui.about` has the reasoning).
        """
        box = QMessageBox(self)
        box.setWindowTitle("About Missile Defense")
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setText(self._about_text())
        box.setIcon(QMessageBox.Icon.NoIcon)
        box.exec()

    def _about_text(self) -> str:
        """What the box says — separate from showing it, so it can be asserted.

        `QMessageBox.exec()` blocks until someone closes it, which no test can
        do, so a test that drove `_show_about` would hang rather than fail. This
        is the seam: the text is checkable, and what is left is Qt's own job.
        """
        return about.summary()

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
        # Whatever control files are still in the directory were requests to
        # this process, and it has gone. The trainer clears them itself on the
        # way out (md.train), but a Stop pressed while it was already writing
        # its last checkpoint lands *after* that clear and has nobody left to
        # obey it — so the run that made the request is also what ends it.
        self._control.clear()
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


def _eval_label(updates: int) -> str:
    """What the slider is pointing at, in the words the trainer uses."""
    return "off" if updates <= 0 else f"{updates} upd"


def _default_eval_every() -> int:
    """What the trainer would use, read out of its own source (:mod:`md.ui.params`).

    For the box before any run has published a value: showing the number the next
    Start would use beats inventing one here and drifting from the dataclass.
    """
    for field in read_params(TRAINER_SOURCES):
        if field.name == "eval_every" and field.default.isdigit():
            return int(field.default)
    return EVAL_EVERY_FALLBACK  # no trainer beside this console


def _number(value: float | None, spec: str) -> str:
    """A measurement the trainer has not produced yet is a dash, never a zero."""
    return "—" if value is None else spec.format(value)


def _compare_note(name: str, value: float | None, spec: str) -> str:
    """``runs-2 13.4`` — the other run's value under this one's."""
    return f"{name} {_number(value, spec)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch a Missile Defense training run.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=None,
        help="the run's --out-dir (default: ./runs in a checkout, else the user data dir)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="build the window, read the run once, print one JSON line and exit",
    )
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Missile Defense training console")
    app.setStyleSheet(theme.stylesheet())
    window = Console(paths.runs_dir(args.run_dir))
    if args.self_test:
        # The console's answer to the game's `--report`, and it exists for the
        # same reason: an exit code cannot tell "started, read the run, drew it"
        # from "printed a usage message". A packaging test needs to know that the
        # *staged* launcher found its interpreter, its import path, PySide6 and
        # the run directory — and that is four separate ways to fail that all
        # look alike from outside (python/tests/e2e/test_packages.py).
        window.resize(1280, 800)
        window._tick()
        print(json.dumps({"ok": True, "run_dir": str(window._run_dir), "updates": window._updates}))
        return 0
    # Roomy, but never bigger than the desktop it opens on — a window whose
    # status line is off-screen is a window with a bug in it.
    available = app.primaryScreen().availableSize()
    window.resize(min(1360, available.width() - 60), min(860, available.height() - 60))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
