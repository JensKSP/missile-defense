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

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QThread, Signal
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
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import league, library, policy_format
from . import sources
from .runner import training_python

if TYPE_CHECKING:  # pulls in the native binding; annotations only
    from ..tournament import Protocol

NOTHING_PROMOTED = (
    "No models promoted yet.\n\n"
    "Open a stopped run and press Enter Model League to put its best\n"
    "checkpoint here, where it outlives the run that produced it."
)

#: What a model row says. `Score` is the best *canonical* result — a quick match
#: never ranks, so a model with only those reads as unranked rather than as
#: whatever its warm-up happened to score.
COLUMNS = ("Model", "From", "Updates", "Score", "Results")


def ask_about_clash(parent: QWidget, existing: league.Model) -> league.Model | None:
    """What to do about a name another model already has: replace it, or rename.

    Returns the model to replace, or ``None`` for "let me type a different
    name" — including when the question is dismissed, because backing out of a
    destructive choice should land on the harmless one.

    The question is asked *here*, in front of the person, rather than resolved
    by a rule. Both answers are reasonable and which is right is not knowable
    from the directory: promoting a better checkpoint of the same experiment
    wants a replacement, and two experiments that happen to share a working
    title want two names. What the dialog owes them is the consequence of the
    destructive one, which is that the old model's results do not survive it.
    """
    scored = len([result for result in existing.results if result.get("canonical")])
    measured = (
        f" It has {scored} ranked result{'' if scored == 1 else 's'}, which a replacement discards"
        " — they were measured on the weights that would be going."
        if scored
        else ""
    )
    question = QMessageBox(parent)
    question.setIcon(QMessageBox.Icon.Warning)
    question.setWindowTitle("That name is taken")
    # Plain text, never rich: a display name is typed by a person and one
    # containing a `<` must not be able to swallow the sentence after it.
    question.setText(f"The league already has a model called {existing.name}.")
    question.setInformativeText(
        f"Names have to be unique — the game's MODELS menu shows nothing else, so two "
        f"of them cannot be told apart.{measured}"
    )
    replace = question.addButton("Replace it", QMessageBox.ButtonRole.DestructiveRole)
    rename = question.addButton("Use another name", QMessageBox.ButtonRole.AcceptRole)
    question.setDefaultButton(rename)
    question.exec()
    return existing if question.clickedButton() is replace else None


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
        # The training runtime's interpreter, because opening a `.pt` needs
        # torch and this console is built never to have it (`league._export`).
        # A checkout with torch beside it spawns nothing.
        python = training_python()
        try:
            self.promoted = league.promote(plan, python=python)
        except league.DuplicateName as clash:
            # Refused before the checkpoint was read, so this costs nothing and
            # the answer is one of two things the person is best placed to pick.
            replace = ask_about_clash(self, clash.existing)
            if replace is None:
                self._retype_the_name()
                return
            try:
                self.promoted = league.promote(plan, replace=replace, python=python)
            except league.LeagueError as error:
                QMessageBox.warning(self, "Could not promote", str(error))
                return
        except league.LeagueError as error:
            # The league is untouched — `promote` stages and validates before it
            # renames — so the dialog stays open and the choice can be changed.
            QMessageBox.warning(self, "Could not promote", str(error))
            return
        self.accept()

    def _retype_the_name(self) -> None:
        """Put the cursor where the work is. The dialog is still open, the name
        is still wrong, and hunting for the field is a step nobody should take."""
        self._name.setFocus()
        self._name.selectAll()


class LeagueView(QWidget):
    """The table of promoted models."""

    watch = Signal(Path)
    #: A match manifest to open in the game, split-screen (`md_app --match`).
    show_match = Signal(Path)
    #: A policy and a seed to open *while a contest is running* — the peek.
    peek = Signal(Path, int)
    #: ...or both sides of a head-to-head, recorded on the seed in flight.
    peek_pair = Signal(Path, Path)

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
        self._import = QPushButton("&Import .mdp…")
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
        self._watch = QPushButton("&Watch it play")
        self._watch.setProperty("role", "primary")
        self._watch.clicked.connect(self._watch_selected)
        self._evaluate = QPushButton("&Evaluate")
        self._evaluate.setToolTip(
            "Score this model over the canonical held-out seeds — the only "
            "protocol the league ranks on"
        )
        self._evaluate.clicked.connect(self._evaluate_selected)
        self._versus = QPushButton("&Head-to-head…")
        self._versus.setToolTip(
            "Play two models over the *same* seeds, then watch one of those episodes side by side"
        )
        self._versus.clicked.connect(self._head_to_head)
        self._rename = QPushButton("Re&name…")
        self._rename.clicked.connect(self._rename_selected)
        self._export = QPushButton("E&xport…")
        self._export.setToolTip(
            "Save this model as a .mdp somebody else can import — or the game "
            "can play with no Python anywhere"
        )
        self._export.clicked.connect(self._export_selected)
        self._delete = QPushButton("&Delete…")
        self._delete.setToolTip(
            "Remove this model from the league, and so from the game's MODELS menu"
        )
        self._delete.clicked.connect(self._delete_selected)
        for button in (
            self._watch,
            self._evaluate,
            self._versus,
            self._rename,
            self._export,
            self._delete,
        ):
            actions.addWidget(button)
        actions.addStretch(1)
        column.addLayout(actions)

        self._table.itemSelectionChanged.connect(self._selection_changed)
        self._table.itemActivated.connect(self._activated)
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

    def _activated(self, _item: QTableWidgetItem) -> None:
        """Enter on the selected row does what the primary button does."""
        self._watch_selected()

    def _selection_changed(self) -> None:
        model = self.selected()
        for button in (self._watch, self._evaluate, self._rename, self._export, self._delete):
            button.setEnabled(model is not None)
        # A contest needs an opponent, and a league of one has none. Disabled
        # with the reason in the tooltip rather than hidden: a button that
        # appears only sometimes is a feature people never find.
        self._versus.setEnabled(model is not None and len(self._models) > 1)
        if model is not None and len(self._models) < 2:
            self._versus.setToolTip("Promote a second model — a head-to-head needs two contestants")

    def _watch_selected(self) -> None:
        model = self.selected()
        if model is not None:
            self.watch.emit(model.policy)

    def _evaluate_selected(self) -> None:
        model = self.selected()
        if model is None:
            return
        dialog = ContestDialog(model, parent=self)
        dialog.peek.connect(self.peek)  # one contestant, so one window
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # The result is already recorded on the model — `evaluate_model`
            # writes it — so the table only has to be re-read.
            self.refresh()

    def _head_to_head(self) -> None:
        model = self.selected()
        if model is None:
            return
        opponents = [other for other in self._models if other.path != model.path]
        if not opponents:
            return
        names = [other.name for other in opponents]
        chosen, ok = QInputDialog.getItem(
            self, "Head-to-head", f"Play {model.name} against:", names, 0, False
        )
        if not ok:
            return
        opponent = opponents[names.index(chosen)]

        dialog = ContestDialog(model, opponent, parent=self)
        dialog.peek.connect(self.peek)
        dialog.peek_pair.connect(self.peek_pair)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.refresh()
        match = dialog.result_of
        if match is None:
            return
        self._offer_the_match(match)

    def _offer_the_match(self, match: object) -> None:
        """A finished contest, and the one thing worth doing with it next.

        Two mean scores answer *which* model is better and say nothing at all
        about *how*. The recordings that answer that are cheap — one episode
        each, on a seed both already played — but only if someone asks for them
        here, while the result is on screen.
        """
        from .. import tournament  # noqa: PLC0415 — needs the native binding

        assert isinstance(match, tournament.Match)
        summary = (
            f"{match.left.display_name}: {match.left.mean_score:,.0f}\n"
            f"{match.right.display_name}: {match.right.mean_score:,.0f}"
        )
        answer = QMessageBox.question(
            self,
            "Watch the match?",
            f"{summary}\n\nRecord one shared seed from each and open them side "
            "by side in the game?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        directory = league.matches_dir() / f"{match.left.model_id}-{match.right.model_id}"
        try:
            recordings = tournament.record_pair(match, directory)
            manifest = tournament.write_manifest(
                match, directory / "match.json", {k: Path(v.name) for k, v in recordings.items()}
            )
        except Exception as error:  # noqa: BLE001 — a dialog is the error channel
            QMessageBox.warning(self, "The match could not be recorded", str(error))
            return
        self.show_match.emit(manifest)

    def _rename_selected(self) -> None:
        """Ask until the name is one no other model has, or until they stop asking.

        A loop rather than a warning-and-close: the answer to "that one is
        taken" is another name, and a dialog that discards what was typed makes
        the person start from the beginning to change one word.
        """
        model = self.selected()
        if model is None:
            return
        proposed = model.name
        while True:
            name, accepted = QInputDialog.getText(
                self, "Rename model", f"A name for {model.model_id}:", text=proposed
            )
            if not accepted:
                return
            try:
                league.rename(model, name)
            except league.DuplicateName as clash:
                QMessageBox.warning(
                    self,
                    "That name is taken",
                    f"{clash.existing.name} already has it. Names have to be unique — "
                    "the game's MODELS menu shows nothing else, so two of them cannot "
                    "be told apart.",
                )
                proposed = name
                continue
            self.refresh()
            return

    def _delete_selected(self) -> None:
        """Remove a model from the league — which is also how it leaves the game.

        The one irreversible thing this screen does, and the only route out of
        the MODELS menu, so it says both: what is about to go, and that `Export…`
        is the version of this you can undo. A promoted model is frequently the
        *only* surviving artifact of a run that has since been cleaned up.
        """
        model = self.selected()
        if model is None:
            return
        named = model.name if model.name == model.model_id else f"{model.name} ({model.model_id})"
        ranked = len([result for result in model.results if result.get("canonical")])
        scores = f" and {ranked} ranked result{'' if ranked == 1 else 's'}" if ranked else ""
        answer = QMessageBox.question(
            self,
            "Delete this model?",
            f"{named}{scores} will be removed from the league and from the game's "
            "MODELS menu. There is no other copy — the run it came from may not "
            "even exist any more.\n\nThis cannot be undone. Export… writes a .mdp "
            "first if there is any chance you will want it back.\n\nRecorded "
            "matches are kept: a match is evidence about the other model too.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            freed = league.delete(model)
        except (league.LeagueError, OSError) as error:
            QMessageBox.warning(self, "Could not delete", str(error))
            return
        self.refresh()
        QMessageBox.information(
            self, "Deleted", f"{model.name} is gone; {sources.human_size(freed)} freed."
        )

    def _export_selected(self) -> None:
        """Copy a model out of the league, byte for byte.

        The other half of Import, and it was missing — you could take a `.mdp`
        in and never get one out, which makes the league a place models go to
        rather than a place they live. A copy and not a re-export: the file in
        the league is already the portable format, and rewriting it would risk
        producing something subtly different from the thing that was scored.
        """
        import shutil  # noqa: PLC0415 — only this path

        from PySide6.QtWidgets import QFileDialog  # noqa: PLC0415 — only this path

        model = self.selected()
        if model is None:
            return
        suggested = str(Path.home() / f"{_file_stem(model.name)}.mdp")
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Export this model", suggested, "Missile Defense policy (*.mdp)"
        )
        if not chosen:
            return
        destination = Path(chosen)
        try:
            shutil.copyfile(model.policy, destination)
            # Read back before claiming success. A full disk truncates happily,
            # and a `.mdp` that only *looks* written is the one failure that
            # surfaces on somebody else's machine rather than this one.
            policy_format.read(destination)
        except (OSError, policy_format.PolicyFormatError) as error:
            QMessageBox.warning(self, "Could not export", str(error))
            return
        QMessageBox.information(self, "Exported", f"{model.name} written to {destination}.")

    def _import_policy(self) -> None:
        from PySide6.QtWidgets import QFileDialog  # noqa: PLC0415 — only this path

        chosen, _ = QFileDialog.getOpenFileName(
            self, "Import a policy", "", "Missile Defense policy (*.mdp)"
        )
        if not chosen:
            return
        source = Path(chosen)
        name = ""
        while True:
            try:
                league.import_policy(source, name)
            except league.DuplicateName as clash:
                # The commonest clash there is: an `.mdp` carries whatever its
                # author called it, and re-importing one you already have is the
                # ordinary case rather than the odd one.
                replace = ask_about_clash(self, clash.existing)
                if replace is not None:
                    try:
                        league.import_policy(source, name, replace=replace)
                    except league.LeagueError as error:
                        QMessageBox.warning(self, "Could not import", str(error))
                        return
                    break
                typed, accepted = QInputDialog.getText(
                    self,
                    "Import under another name",
                    "A name for the imported model:",
                    text=clash.existing.name,
                )
                if not accepted:
                    return
                name = typed
                continue
            except league.LeagueError as error:
                # Validated before anything is written, so the league is untouched.
                QMessageBox.warning(self, "Could not import", str(error))
                return
            break
        self.refresh()


def _file_stem(name: str) -> str:
    """A display name as a filename: spaces to dashes, nothing exotic kept.

    Only a *suggestion* in a save dialog — the person can type whatever they
    like — but a default containing a slash is one that fails on Enter.
    """
    kept = [c if c.isalnum() else "-" for c in name.strip().lower()]
    stem = "".join(kept).strip("-")
    while "--" in stem:
        stem = stem.replace("--", "-")
    return stem or "policy"


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


class _Contest(QThread):
    """One evaluation or one head-to-head, off the event loop.

    Both are minutes of pure computation with no I/O to wait on, so they would
    freeze the window solid — and a frozen window during the *one* operation
    that takes long enough to notice is how a person concludes the program has
    crashed. `md.tournament` already reports progress and takes its seed list
    once, so all this adds is a thread and a way to stop.
    """

    progress = Signal(int, int, int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        left: league.Model,
        right: league.Model | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._left = left
        self._right = right
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Stop at the next seed boundary.

        Nothing is recorded until a contest finishes, so a cancellation leaves
        the league exactly as it was — which is why this can be a flag rather
        than an unwind.
        """
        self._cancelled.set()

    def run(self) -> None:
        from .. import tournament  # noqa: PLC0415 — needs the native binding

        def report(index: int, contestants: int, done: int, total: int) -> None:
            if self._cancelled.is_set():
                raise _Cancelled
            self.progress.emit(index, contestants, done, total)

        try:
            if self._right is None:
                self.done.emit(tournament.evaluate_model(self._left, progress=report))
            else:
                self.done.emit(tournament.head_to_head(self._left, self._right, progress=report))
        except _Cancelled:
            self.done.emit(None)
        except Exception as error:  # noqa: BLE001 — a dialog is the error channel
            self.failed.emit(str(error))


class _Cancelled(Exception):
    """Raised inside the worker to unwind out of `md.tournament`'s progress hook."""


class _PeekRecorder(QThread):
    """Record both contestants on one seed, so a peek can be side by side.

    A head-to-head plays one model's whole block and then the other's, so at any
    moment only one of them is *being* computed — and one half of a comparison
    is not the thing anybody pressed the button for. Both episodes are recorded
    here instead, which costs a couple of seconds now that inference is native
    and is exact rather than approximate: the same policies on the same seed are
    the same episodes the contest is scoring.

    Off the event loop for those seconds, because the contest's own progress is
    still arriving and a frozen dialog is how a person concludes it has hung.
    """

    done = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        left: league.Model,
        right: league.Model,
        seed: int,
        directory: Path,
        max_ticks: int,
        frame_skip: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._left = left
        self._right = right
        self._seed = seed
        self._directory = directory
        self._max_ticks = max_ticks
        self._frame_skip = frame_skip

    def run(self) -> None:
        from .. import tournament  # noqa: PLC0415 — needs the native binding

        try:
            sides = tuple(
                tournament.record_episode(
                    model,
                    self._seed,
                    self._directory / f"{side}.mdr",
                    max_ticks=self._max_ticks,
                    frame_skip=self._frame_skip,
                )
                for side, model in (("left", self._left), ("right", self._right))
            )
        except Exception as error:  # noqa: BLE001 — a dialog is the error channel
            self.failed.emit(str(error))
            return
        self.done.emit(sides)


class ContestDialog(QDialog):
    """Run a canonical evaluation, or two models against the same seeds.

    Modal and cancellable, with the protocol stated before it starts: what makes
    a league table worth anything is that every row was measured the same way,
    and a person who cannot see which protocol is about to run has no way to
    know whether the number they are about to add belongs in it.

    **You can look inside it.** A contest is a couple of minutes of numbers
    going up, and the interesting question — *what is it actually doing on this
    seed?* — has no answer in a progress bar. `Watch this seed` opens the game
    on the model and seed being played right now. Off unless asked for: it is a
    second process and a second window, and the point of running a contest is
    usually the score at the end.

    The spectator is a **copy**, not a view. Both the evaluator and the game are
    deterministic, so the same policy on the same seed is the same episode tick
    for tick — the game computes its own and the contest never waits for it.
    Close the window and the contest carries on; it never knew.
    """

    #: A model and the seed it is on — one contestant, so one window.
    peek = Signal(Path, int)
    #: Two recordings of the seed a head-to-head is on, for the split screen.
    peek_pair = Signal(Path, Path)

    def __init__(
        self,
        left: league.Model,
        right: league.Model | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.result_of: object | None = None
        self._left = left
        self._right = right
        self._worker: _Contest | None = None
        self._playing: tuple[league.Model, int] | None = None
        self._seed_list: list[int] | None = None
        self._recorder: _PeekRecorder | None = None

        self.setWindowTitle("Head-to-head" if right is not None else "Evaluate")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)

        title = QLabel(f"{left.name} vs {right.name}" if right is not None else left.name)
        title.setProperty("role", "value")
        layout.addWidget(title)

        note = QLabel(self._protocol_note())
        note.setWordWrap(True)
        note.setProperty("role", "note")
        layout.addWidget(note)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)  # indeterminate until the first report arrives
        layout.addWidget(self._bar)

        self._status = QLabel("starting…")
        self._status.setProperty("role", "note")
        layout.addWidget(self._status)

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self._watch = self._buttons.addButton(
            "&Watch this seed", QDialogButtonBox.ButtonRole.ActionRole
        )
        self._watch.setToolTip(
            "Open the game on the seed being played right now — both models side "
            "by side in a head-to-head. It plays its own copy, so the contest "
            "keeps running and closing it changes nothing"
        )
        self._watch.setEnabled(False)  # until the first report says what is in flight
        self._watch.clicked.connect(self._peek)
        self._buttons.rejected.connect(self._cancel)
        layout.addWidget(self._buttons)

    def _protocol(self) -> Protocol:
        """The protocol the contest is running. One lookup, used three ways."""
        from .. import tournament  # noqa: PLC0415 — needs the native binding

        return tournament.canonical_protocol()

    @property
    def _seeds(self) -> Sequence[int]:
        """The list the contest is playing, in the order it plays it.

        Derived here rather than reported by the worker: it is the same
        `canonical_protocol` the contest itself asked for, and a second channel
        carrying seeds across a thread boundary would be a second thing that
        could disagree about which episode is on screen.
        """
        from .. import tournament  # noqa: PLC0415 — needs the native binding

        if self._seed_list is None:
            self._seed_list = tournament.canonical_protocol().seeds()
        return self._seed_list

    def _protocol_note(self) -> str:
        from .. import tournament  # noqa: PLC0415 — needs the native binding

        protocol = tournament.canonical_protocol()
        seeds = len(protocol.seeds())
        both = " Both models play the same seeds, taken once." if self._right else ""
        return (
            f"The canonical protocol: {seeds} held-out seeds, decision every "
            f"{protocol.frame_skip} ticks, capped at {protocol.max_ticks:,}."
            f"{both} Nothing is recorded until it finishes, so cancelling leaves "
            "the league unchanged."
        )

    def exec(self) -> int:
        self._worker = _Contest(self._left, self._right, self)
        self._worker.progress.connect(self._advance)
        self._worker.done.connect(self._finished)
        self._worker.failed.connect(self._blame)
        self._worker.start()
        return super().exec()

    def _advance(self, index: int, contestants: int, done: int, total: int) -> None:
        self._bar.setRange(0, total * contestants)
        self._bar.setValue((index * total) + done)
        model = self._left if index == 0 else (self._right or self._left)
        who = model.name
        self._status.setText(f"playing {who} — {done} of {total} seeds")
        # `done` is how many are *in*, so the one being played is the next along.
        # At the end of a contestant's block there is none, and the button says
        # so by going quiet rather than opening the wrong episode.
        in_flight = self._seeds[done] if done < len(self._seeds) else None
        self._playing = None if in_flight is None else (model, in_flight)
        self._watch.setEnabled(self._playing is not None)

    def _peek(self) -> None:
        """Open the episode being played — side by side when there are two of them.

        A contest computes one contestant at a time, so the *other* side of this
        seed has not been played yet (or was played minutes ago and not kept).
        Both are recorded here before the window opens, which is the difference
        between watching a comparison and watching half of one.
        """
        if self._playing is None:
            return
        model, seed = self._playing
        if self._right is None:
            self.peek.emit(model.policy, seed)
            return

        self._watch.setEnabled(False)
        self._watch.setText("recording…")
        protocol = self._protocol()
        self._recorder = _PeekRecorder(
            self._left,
            self._right,
            seed,
            league.matches_dir() / f"peek-{self._left.model_id}-{self._right.model_id}",
            protocol.max_ticks,
            protocol.frame_skip,
            self,
        )
        self._recorder.done.connect(self._recorded)
        self._recorder.failed.connect(self._not_recorded)
        self._recorder.start()

    def _recorded(self, sides: object) -> None:
        self._restore_watch()
        assert isinstance(sides, tuple)
        left, right = sides
        assert isinstance(left, Path)
        assert isinstance(right, Path)
        self.peek_pair.emit(left, right)

    def _not_recorded(self, message: str) -> None:
        self._restore_watch()
        QMessageBox.warning(self, "The seed could not be recorded", message)

    def _restore_watch(self) -> None:
        self._watch.setText("&Watch this seed")
        # Enabled only if something is still in flight: a contest that finished
        # while the recording ran has no seed to open.
        self._watch.setEnabled(self._playing is not None)

    def _finished(self, outcome: object) -> None:
        self.result_of = outcome
        self.accept() if outcome is not None else self.reject()

    def _blame(self, message: str) -> None:
        QMessageBox.warning(self, "The contest could not be run", message)
        self.reject()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._status.setText("stopping at the next seed…")
        self._buttons.setEnabled(False)
