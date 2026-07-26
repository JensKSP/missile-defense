# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# pyright: reportMissingImports=false
"""The run library: every run there is, before you have picked one.

**A level above the dashboard, not a tab beside it.** The dashboard's subject is
*one run*; this screen's subject is *all of them*, and the question it answers —
"which of these eleven is worth my attention, and which can go?" — is not one the
run picker in the header can answer, because a dropdown shows names and this has
to show scores, sizes and states side by side.

That is also why it is the landing view. Opening the console on `runs/` with
eleven experiments in it and being dropped into whichever one sorted first is
how you end up reading the wrong curve for a minute.

Everything numeric comes from :mod:`md.library`, which is Qt-free and tested.
This file lays out rows and turns clicks into calls.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import library
from . import sources, theme

#: What the screen says on a machine that has never trained anything. Names the
#: thing that would change it, like every other empty state in this console.
NOTHING_YET = (
    "No runs here yet.\n\n"
    "Start one with the button above, or point the console at a directory that\n"
    "already has some:  poe ui -- path/to/runs"
)

#: The columns, in the order the questions get asked: what is it, is it going,
#: how far did it get, how good was it, what does it cost.
COLUMNS = ("Run", "State", "Updates", "Best score", "Episodes", "Disk", "Note")


class RunTable(QWidget):
    """The list itself. Emits a path when a row is chosen."""

    opened = Signal(Path)
    renamed = Signal(Path)
    #: A run's contents changed on disk — cleaned up, or archived and removed.
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._runs: list[library.Run] = []
        #: Where these runs live. Storage operations refuse to touch anything
        #: outside it, so the table has to know rather than infer from a row.
        self._root: Path | None = None

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)

        self._empty = QLabel(NOTHING_YET)
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
        self._table.setAlternatingRowColors(False)
        # A double-click opens: the same gesture the recordings list already
        # uses, so there is one way to mean "this one" in the whole window.
        self._table.doubleClicked.connect(self._open_selected)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        for index in range(1, len(COLUMNS) - 1):
            header.setSectionResizeMode(index, QHeaderView.ResizeMode.ResizeToContents)
        column.addWidget(self._table, stretch=1)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self._open = QPushButton("Open")
        self._open.setProperty("role", "primary")
        self._open.clicked.connect(self._open_selected)
        self._rename = QPushButton("Rename…")
        self._rename.clicked.connect(self._rename_selected)
        self._note = QPushButton("Note…")
        self._note.clicked.connect(self._note_selected)
        self._storage = QPushButton("Storage…")
        self._storage.setToolTip(
            "What this run costs on disk, and how to clean it up, archive it, "
            "or archive it and remove it"
        )
        self._storage.clicked.connect(self._storage_selected)
        for button in (self._open, self._rename, self._note, self._storage):
            actions.addWidget(button)
        actions.addStretch(1)
        self._summary = QLabel()
        self._summary.setProperty("role", "note")
        actions.addWidget(self._summary)
        column.addLayout(actions)

        self._table.itemSelectionChanged.connect(self._selection_changed)
        self._selection_changed()

    # ---- feeding it ----------------------------------------------------------

    def set_root(self, root: Path) -> None:
        self._root = root

    def show_runs(self, runs: Sequence[library.Run]) -> None:
        """Redraw. Keeps the selection on the same *run* if it is still there.

        By path and not by row index: a run that finished while you were reading
        moves in the list — it is sorted by last activity — and a selection that
        followed the index would silently jump to a different run under you.
        """
        chosen = self.selected()
        self._runs = list(runs)
        self._empty.setVisible(not self._runs)
        self._table.setVisible(bool(self._runs))

        self._table.setRowCount(len(self._runs))
        for row, run in enumerate(self._runs):
            for index, text in enumerate(self._cells(run)):
                item = QTableWidgetItem(text)
                if index in (2, 3, 5):  # the numbers, right-aligned to compare
                    # The flag itself, not `int(...)`: the integer overload is
                    # deprecated in Qt 6 and warns on every cell.
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if index == 1:
                    # The one coloured cell: "is this still going" is the thing
                    # you scan the column for, and a word alone does not scan.
                    item.setForeground(QColor(theme.AHEAD if run.live else theme.MUTED))
                self._table.setItem(row, index, item)
            if chosen is not None and run.path == chosen.path:
                self._table.selectRow(row)

        total = sum(run.storage.total for run in self._runs)
        self._summary.setText(
            f"{len(self._runs)} run{'' if len(self._runs) == 1 else 's'} · "
            f"{sources.human_size(total)}"
        )
        self._selection_changed()

    @staticmethod
    def _cells(run: library.Run) -> tuple[str, ...]:
        # The id in brackets after a display name, because the id is what a
        # `--resume` or a path needs and the name is what a person remembers.
        name = run.name if run.name == run.run_id else f"{run.name}  ({run.run_id})"
        return (
            name,
            "live" if run.live else "stopped",
            f"{run.updates:,}",
            "—" if run.best_score is None else f"{run.best_score:,.0f}",
            f"{run.recordings}",
            sources.human_size(run.storage.total),
            run.note,
        )

    def selected(self) -> library.Run | None:
        rows = {index.row() for index in self._table.selectedIndexes()}
        if len(rows) != 1:
            return None
        row = rows.pop()
        return self._runs[row] if 0 <= row < len(self._runs) else None

    # ---- actions -------------------------------------------------------------

    def _selection_changed(self) -> None:
        run = self.selected()
        for button in (self._open, self._rename, self._note):
            button.setEnabled(run is not None)

    def _open_selected(self) -> None:
        run = self.selected()
        if run is not None:
            self.opened.emit(run.path)

    def _rename_selected(self) -> None:
        run = self.selected()
        if run is None:
            return
        name, accepted = QInputDialog.getText(
            self, "Rename run", f"A name for {run.run_id}:", text=run.name
        )
        if accepted:
            library.rename(run.path, name)
            self.renamed.emit(run.path)

    def _storage_selected(self) -> None:
        run = self.selected()
        if run is None or self._root is None:
            return
        from .storage import StorageDialog  # noqa: PLC0415 — optional dependency

        dialog = StorageDialog(run, self._root, self)
        dialog.exec()
        if dialog.changed:
            self.changed.emit()

    def _note_selected(self) -> None:
        run = self.selected()
        if run is None:
            return
        note, accepted = QInputDialog.getText(
            self, "Note", f"What is worth remembering about {run.name}?", text=run.note
        )
        if accepted:
            library.set_note(run.path, note)
            self.renamed.emit(run.path)


class LibraryView(QWidget):
    """The landing screen: the run table, with the league beside it."""

    opened = Signal(Path)

    def __init__(self, on_new_run: Callable[[], None] | None = None) -> None:
        super().__init__()
        self._root: Path | None = None

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 10, 0)
        column.setSpacing(10)

        heading = QHBoxLayout()
        caption = QLabel("RUNS")
        caption.setProperty("role", "caption")
        heading.addWidget(caption)
        heading.addStretch(1)
        self._restore = QPushButton("Restore…")
        self._restore.setToolTip("Put an archived run back into this library")
        self._restore.clicked.connect(self._restore_archive)
        heading.addWidget(self._restore)
        if on_new_run is not None:
            new_run = QPushButton("New run…")
            new_run.setProperty("role", "primary")
            new_run.clicked.connect(on_new_run)
            heading.addWidget(new_run)
        column.addLayout(heading)

        self.table = RunTable()
        self.table.opened.connect(self.opened)
        self.table.renamed.connect(self._renamed)
        self.table.changed.connect(self.refresh)
        column.addWidget(self.table, stretch=1)

    def _restore_archive(self) -> None:
        if self._root is None:
            return
        from .storage import restore  # noqa: PLC0415 — optional dependency

        if restore(self._root, self) is not None:
            self.refresh()

    def _renamed(self, _path: Path) -> None:
        """A name or a note changed; the row it is in has to be redrawn."""
        self.refresh()

    def attach(self, root: Path) -> None:
        """Point the library at a directory of runs and read it."""
        self._root = root
        self.table.set_root(root)
        self.refresh()

    def refresh(self) -> None:
        if self._root is None:
            return
        self.table.show_runs(library.discover(self._root))
