# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# pyright: reportMissingImports=false
"""The Model League: promoted policies, and the dialog that puts one there.

A run is a *process*; a model is a *thing you keep*. The library lists the first
and this lists the second, and they are separate screens because they are
answered at different moments — you look at runs while training and at models
when deciding what to ship or what to beat.

**Promotion is where the two meet, and it is deliberately a decision.** The plan
could have promoted the best checkpoint automatically at the end of every run;
that would fill the league with fifty entries nobody chose and make the table
useless for the one thing it is for. So it is a dialog, it defaults to the best
*evaluated checkpoint that still exists on disk* (:func:`md.library.best_evaluated_checkpoint`
— the best score often has no checkpoint), it names the model after the run, and
it shows exactly what will happen before it happens.

Every failure path is :mod:`md.league`'s: an unexportable architecture, a
checkpoint from a different simulator, a full disk. They arrive here as a
message and leave the league untouched, because promotion stages and validates
before it renames.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import league, library
from . import sources

NOTHING_PROMOTED = (
    "No models promoted yet.\n\n"
    "Open a stopped run and press Enter Model League to put its best\n"
    "checkpoint here, where it outlives the run that produced it."
)

#: What a model row says. `Score` is the best *canonical* result — a quick match
#: never ranks, so a model with only those reads as unranked rather than as
#: whatever its warm-up happened to score.
COLUMNS = ("Model", "From", "Updates", "Score", "Results")


class PromoteDialog(QDialog):
    """Turn a run's checkpoint into a league entry, showing what will happen.

    The three things it has to get right, in the order they bite:

    1. **Which checkpoint.** Defaults to the best evaluated one *that exists* —
       the best score frequently has none, because checkpoints and evaluations
       are written on different cadences.
    2. **What it will be called.** Defaulted from the run's own name and the
       update, since a run can be promoted twice and two identical rows in a
       league table are the confusion naming was meant to end.
    3. **That it may fail.** Export can refuse — an `entity` policy has no
       native forward pass — and the refusal has to arrive as a sentence rather
       than as a league entry the game cannot load.
    """

    def __init__(self, run: library.Run, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Enter Model League — {run.name}")
        self.setMinimumWidth(460)
        self._run = run
        self.promoted: league.Model | None = None

        column = QVBoxLayout(self)
        column.setSpacing(10)
        form = QFormLayout()
        form.setSpacing(8)

        self._checkpoint = QComboBox()
        best = library.best_evaluated_checkpoint(run.path)
        checkpoints = sources.list_checkpoints(run.path)
        scores = {row.update: row for row in _eval_rows(run.path)}
        for entry in checkpoints:
            row = scores.get(entry.iteration) if entry.iteration is not None else None
            label = entry.name
            if row is not None:
                label += f"   scored {row.mean_score:,.0f}"
            self._checkpoint.addItem(label, str(entry.path))
        if best is not None:
            index = self._checkpoint.findData(str(best[0]))
            if index >= 0:
                self._checkpoint.setCurrentIndex(index)
        form.addRow("Checkpoint", self._checkpoint)

        self._name = QLineEdit()
        default = library.model_name_for(run, best[0]) if best is not None else run.name
        self._name.setText(default)
        form.addRow("Name", self._name)

        self._note = QLineEdit()
        self._note.setPlaceholderText("what was different about this one")
        form.addRow("Note", self._note)
        column.addLayout(form)

        self._explain = QLabel()
        self._explain.setProperty("role", "note")
        self._explain.setWordWrap(True)
        self._explain.setText(
            "The checkpoint is converted to a data-only .mdp and copied into the "
            "league, so the model survives this run being cleaned up or archived. "
            "It is read back before anything is written; if it cannot be, nothing is."
        )
        column.addWidget(self._explain)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Promote")
        buttons.accepted.connect(self._promote)
        buttons.rejected.connect(self.reject)
        column.addWidget(buttons)

        self._ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setEnabled(self._checkpoint.count() > 0)
        if self._checkpoint.count() == 0:
            self._explain.setText(
                "This run has no checkpoints. The trainer writes one every "
                "--checkpoint-every updates; there is nothing to promote yet."
            )

    def _promote(self) -> None:
        chosen = self._checkpoint.currentData()
        if not chosen:
            return
        name = self._name.text().strip() or self._run.name
        plan = league.Promotion(
            checkpoint=Path(str(chosen)),
            display_name=name,
            source_run=self._run.run_id,
            trained_updates=self._run.updates,
            note=self._note.text().strip(),
        )
        try:
            self.promoted = league.promote(plan)
        except league.LeagueError as error:
            # The league is untouched — `promote` stages and validates before it
            # renames — so the dialog stays open and the choice can be changed.
            QMessageBox.warning(self, "Could not promote", str(error))
            return
        self.accept()


class LeagueView(QWidget):
    """The table of promoted models."""

    watch = Signal(Path)

    def __init__(self) -> None:
        super().__init__()
        self._models: list[league.Model] = []

        column = QVBoxLayout(self)
        column.setContentsMargins(10, 0, 0, 0)
        column.setSpacing(8)

        heading = QHBoxLayout()
        caption = QLabel("MODEL LEAGUE")
        caption.setProperty("role", "caption")
        heading.addWidget(caption)
        heading.addStretch(1)
        self._import = QPushButton("Import .mdp…")
        self._import.clicked.connect(self._import_policy)
        heading.addWidget(self._import)
        column.addLayout(heading)

        self._empty = QLabel(NOTHING_PROMOTED)
        self._empty.setProperty("role", "placeholder")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self._empty)

        self._table = QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels(list(COLUMNS))
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setShowGrid(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for index in range(1, len(COLUMNS)):
            header.setSectionResizeMode(index, QHeaderView.ResizeMode.ResizeToContents)
        column.addWidget(self._table, stretch=1)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self._watch = QPushButton("Watch it play")
        self._watch.setProperty("role", "primary")
        self._watch.clicked.connect(self._watch_selected)
        self._rename = QPushButton("Rename…")
        self._rename.clicked.connect(self._rename_selected)
        for button in (self._watch, self._rename):
            actions.addWidget(button)
        actions.addStretch(1)
        column.addLayout(actions)

        self._table.itemSelectionChanged.connect(self._selection_changed)
        self._selection_changed()

    def refresh(self) -> None:
        self.show_models(league.models())

    def show_models(self, models: Sequence[league.Model]) -> None:
        chosen = self.selected()
        self._models = list(models)
        self._empty.setVisible(not self._models)
        self._table.setVisible(bool(self._models))
        self._table.setRowCount(len(self._models))
        for row, model in enumerate(self._models):
            best = model.best_result
            cells = (
                model.name,
                model.source_run or "imported",
                f"{model.trained_updates:,}" if model.trained_updates else "—",
                _score_text(best),
                f"{len(model.results)}",
            )
            for index, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if index in (2, 3, 4):
                    # The flag itself, not `int(...)`: the integer overload is
                    # deprecated in Qt 6 and warns on every cell.
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self._table.setItem(row, index, item)
            if chosen is not None and model.path == chosen.path:
                self._table.selectRow(row)
        self._selection_changed()

    def selected(self) -> league.Model | None:
        rows = {index.row() for index in self._table.selectedIndexes()}
        if len(rows) != 1:
            return None
        row = rows.pop()
        return self._models[row] if 0 <= row < len(self._models) else None

    def _selection_changed(self) -> None:
        model = self.selected()
        for button in (self._watch, self._rename):
            button.setEnabled(model is not None)

    def _watch_selected(self) -> None:
        model = self.selected()
        if model is not None:
            self.watch.emit(model.policy)

    def _rename_selected(self) -> None:
        model = self.selected()
        if model is None:
            return
        name, accepted = QInputDialog.getText(
            self, "Rename model", f"A name for {model.model_id}:", text=model.name
        )
        if accepted:
            league.rename(model, name)
            self.refresh()

    def _import_policy(self) -> None:
        from PySide6.QtWidgets import QFileDialog  # noqa: PLC0415 — only this path

        chosen, _ = QFileDialog.getOpenFileName(
            self, "Import a policy", "", "Missile Defense policy (*.mdp)"
        )
        if not chosen:
            return
        try:
            league.import_policy(Path(chosen), "")
        except league.LeagueError as error:
            # Validated before anything is written, so the league is untouched.
            QMessageBox.warning(self, "Could not import", str(error))
            return
        self.refresh()


def _score_text(best: Mapping[str, object] | None) -> str:
    """A ranked score, or the word that says there is not one.

    "unranked" and not a number: a model whose only results are quick matches
    has no benchmark score, and showing its warm-up in the same column as one
    would put a four-seed run above a thirty-two-seed one.
    """
    if best is None:
        return "unranked"
    value = best.get("mean_score")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "unranked"
    return f"{float(value):,.0f}"


def _eval_rows(run: Path) -> list[sources.EvalRow]:
    return list(sources.evals_tail(run).poll().rows)
