# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# pyright: reportMissingImports=false
"""What a run was started with — the knobs, beside the curves they produced.

The trainer has written `config.json` into every run directory for as long as
there have been runs, and until now nothing in the trainer read it. So the
answer to "what was this one trained with?" — asked of exactly the run that
turned out interesting, usually a fortnight later — was to leave the window,
find the directory, and read JSON.

Two things make this more than a formatted dump of that file:

* **What this run *changed*.** Twenty-six numbers are unreadable. The four that
  differ from the trainer's own defaults are the run, and they are marked and
  counted (:func:`missile_defense.ui.params.settings_of` does that arithmetic, Qt-free).
* **Why each knob is there.** Each row's tooltip is the sentence written beside
  the field in `TrainConfig`, `PPOConfig` or `Shaping` — the same text the Start
  dialog shows, from the same source, so the two cannot drift.

Read-only on purpose, and *reachable at any time* — from the button beside Log
on the run screen, and from the library without opening the run at all. A run's
settings are what it **was** started with; the one knob that can still be changed
while a run goes is the eval interval, and it lives on the bar with Start and
Stop where it can be turned while watching the curve. A form here would be a
second, lying way to set the rest.

A button and a window you close again, rather than a panel or a third tab: this
is asked rarely, in the middle of doing something else, and answered by reading
rather than by watching. Anything permanent would take space from the curve
every minute of a run to serve a glance twice a week.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..runs import runconfig
from . import params as params_module
from . import theme
from .params import Setting, read_params, settings_of
from .reward import Formula, Term, formula_of

#: A run directory with no `config.json`. Names the file, because a run that
#: began before the trainer wrote one has everything else and only this missing —
#: which is a different situation from a directory that holds no run.
NOTHING_STORED = (
    "No config.json in this run.\n"
    "The trainer writes one at start-up, so a run that began before it existed\n"
    "has none — and one somebody synced without it keeps its curves regardless."
)

COLUMNS = ("Setting", "Value", "Default")

#: The headings come from `missile_defense.ui.params.arrange` now — the same
#: domains and groups the parameter dialog is laid out in, so one run is
#: described one way on both screens. This used to be a table of the four raw
#: keys `missile_defense.runs.runconfig` writes (`train`, `ppo`, `shaping`,
#: `schedule`), which named the storage rather than the decisions.


class RewardView(QWidget):
    """The reward formula of a run, with that run's numbers in it.

    Sits above the table rather than replacing any of it. The table answers
    "what was set"; this answers "what was the agent paid for", which is the
    question the seven `Shaping` rows are least able to answer on their own —
    they do not show that three of them are summed into one potential, that
    `gamma` discounts that potential and not the return, or that two of them are
    switched off.

    A zero-weighted term is drawn muted rather than dropped. "Was this run
    penalised for wasted shots?" is a question about a term that is not there,
    and a missing line answers it with silence.
    """

    def __init__(self) -> None:
        super().__init__()
        self._column = QVBoxLayout(self)
        self._column.setContentsMargins(0, 0, 0, 0)
        self._column.setSpacing(4)
        self._shown: object | None = None

    def show_formula(self, formula: Formula | None) -> None:
        """Draw one run's reward. Cheap to call on every rescan."""
        key = _formula_key(formula)
        if key == self._shown:
            return
        self._shown = key

        while (item := self._column.takeAt(0)) is not None:
            if (widget := item.widget()) is not None:
                widget.deleteLater()
        self.setVisible(formula is not None)
        if formula is None:
            return

        self._add(formula.total, role="formula")
        if formula.shaped and formula.phi:
            self._add(formula.phi, role="formula")
            if formula.gamma:
                self._add(f"γ = {formula.gamma}", role="note", colour=theme.MUTED)
        for term in (*formula.potential, *formula.priced):
            self._add_term(term)
        for note in formula.notes:
            self._add(note, role="note", colour=theme.MUTED)

    def _add(self, text: str, *, role: str, colour: str | None = None) -> QLabel:
        label = QLabel(text)
        label.setProperty("role", role)
        label.setWordWrap(True)
        if colour is not None:
            label.setStyleSheet(f"color: {colour};")
        self._column.addWidget(label)
        return label

    def _add_term(self, term: Term) -> None:
        # An inactive term keeps its row and loses its colour: present, priced at
        # nothing, and said so — rather than absent and ambiguous.
        label = self._add(term.line, role="note", colour=None if term.active else theme.MUTED)
        label.setToolTip(term.why)


def _formula_key(formula: Formula | None) -> object | None:
    """What is on screen, so an unchanged run is not redrawn every rescan."""
    if formula is None:
        return None
    return (
        formula.shaped,
        formula.gamma,
        tuple((t.name, t.weight, t.active) for t in (*formula.potential, *formula.priced)),
    )


class ConfigPanel(QWidget):
    """A run's settings as a table, with the changed ones marked."""

    def __init__(self) -> None:
        super().__init__()
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)

        self._summary = QLabel()
        self._summary.setProperty("role", "note")
        self._summary.setWordWrap(True)
        column.addWidget(self._summary)

        self._reward = RewardView()
        column.addWidget(self._reward)

        self._empty = QLabel(NOTHING_STORED)
        self._empty.setProperty("role", "placeholder")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self._empty)

        self._table = QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels(list(COLUMNS))
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        column.addWidget(self._table, stretch=1)

        #: What is on screen, so an unchanged run is not redrawn every rescan —
        #: a table that rebuilds once a second loses your scroll position and
        #: your selection with it.
        self.settings: tuple[Setting, ...] = ()
        self._shown: tuple[object, ...] | None = None
        self.show_settings(None, ())

    def show_settings(
        self, config: runconfig.RunConfig | None, settings: Sequence[Setting]
    ) -> None:
        """Draw a run's stored settings. Cheap to call on every rescan."""
        key = (str(config.path) if config else "", tuple(map(_key, settings)))
        if key == self._shown:
            return
        self._shown = key
        self.settings = tuple(settings)

        self._empty.setVisible(not settings)
        self._table.setVisible(bool(settings))
        self._summary.setText(_summary(config, settings))
        self._reward.show_formula(formula_of(settings))
        self._table.setRowCount(0)
        if not settings:
            return

        # Under the dialog's own headings, not `config.json`'s four raw keys.
        # The run was configured under *Scale*, *Human handicap*, *Annealing*;
        # reading it back under `train` / `ppo` / `shaping` made the reader
        # translate between two vocabularies for one run.
        domain = ""
        for domain_title, group_name, members in params_module.arrange(settings):
            if domain_title != domain:
                domain = domain_title
                self._add_header(domain_title.upper(), lead=True)
            self._add_header(group_name)
            for setting in members:
                self._add_setting(setting)

    def _add_header(self, label: str, *, lead: bool = False) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        item = QTableWidgetItem(label if lead else f"    {label}")
        # The domain is the louder of the two: it is which of the three questions
        # you are reading, where the group beneath it is only which handful.
        item.setForeground(QColor(theme.TEXT if lead else theme.MUTED))
        self._table.setItem(row, 0, item)
        self._table.setSpan(row, 0, 1, len(COLUMNS))

    def _add_setting(self, setting: Setting) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        # A changed value is the only thing coloured, because it is the only
        # thing being looked for: what is this run, as against every other one.
        cells = (
            setting.name.replace("_", " "),
            setting.value or runconfig.NOTHING,
            setting.default if setting.changed else "",
        )
        for index, text in enumerate(cells):
            item = QTableWidgetItem(text)
            # The reasoning where there is one, and the value in full where the
            # column had to cut it off — `out_dir` is a path, and a truncated
            # path is exactly the value somebody opened this to read.
            item.setToolTip(setting.help or f"{setting.name} = {setting.value}")
            if index == 1 and setting.changed:
                item.setForeground(QColor(theme.AMBER))
            if index == 2:
                item.setForeground(QColor(theme.MUTED))
            self._table.setItem(row, index, item)


def _key(setting: Setting) -> tuple[str, str, str, bool]:
    return (setting.group, setting.name, setting.value, setting.changed)


def _summary(config: runconfig.RunConfig | None, settings: Sequence[Setting]) -> str:
    """Above the table: what this run changed, then where it was read from.

    That order, because a path is what you check and a count is what you came
    for — and a run directory's path is long enough to push the sentence worth
    reading off the top of the window.
    """
    if config is None or not settings:
        return ""
    changed = sum(1 for setting in settings if setting.changed)
    # Said either way: a run on the stock defaults is a fact worth stating, and
    # an absent line reads as a panel that failed rather than as an answer.
    lines = [
        f"{changed} setting{'' if changed == 1 else 's'} changed from the trainer's defaults"
        if changed
        else "The trainer's own defaults throughout."
    ]
    if config.resumed_from is not None:
        # Then `updates` is a count of *additional* updates and the run's first
        # update number is somebody else's last plus one. Named relative to the
        # run when it is the run's own checkpoint, which it usually is.
        lines.append(f"Continued from {_within(config.resumed_from, config.path.parent)}.")
    lines.append(str(config.path))
    return "\n".join(lines)


def _within(path: Path, run_dir: Path) -> str:
    """``checkpoints/policy-final.pt`` for this run's own file, else the full path."""
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


class ConfigDialog(QDialog):
    """The table as a window you open and close again.

    Both places that answer "what was this trained with?" open this: the button
    beside *Log* on the run screen, and the library's own, where the question is
    asked *while comparing* — which of these eleven was the one with the wider
    rollout — and opening each run to find out is four clicks per answer.

    A dialog rather than a panel or a third tab, because of how the question is
    asked: rarely, in the middle of doing something else, and answered by
    reading rather than watching. A permanent view of it would take space from
    the curve every minute of the run to serve a glance twice a week.
    """

    def __init__(
        self,
        name: str,
        config: runconfig.RunConfig | None,
        settings: Sequence[Setting],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Parameters — {name}")
        self.resize(620, 560)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)

        self.panel = ConfigPanel()
        self.panel.show_settings(config, settings)
        layout.addWidget(self.panel, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


def settings_for(run_dir: Path, trainer: Path) -> tuple[runconfig.RunConfig | None, list[Setting]]:
    """Read a run's settings and pair them with the trainer's own reasoning.

    Both halves in one call because both callers want both, and because the
    training loop's source may not be beside this trainer at all — an installed trainer
    watching a synced directory still shows every value, just without the
    tooltips and the comparison against defaults.
    """
    config = runconfig.read(run_dir)
    return config, settings_of(config, read_params(trainer))
