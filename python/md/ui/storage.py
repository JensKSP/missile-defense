# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# pyright: reportMissingImports=false
"""What a run costs on disk, and the three things you can do about it.

Training fills a disk faster than anything else here: a long run is gigabytes
of checkpoints, most of which nobody will ever load again. The backends for
tidying that up have existed since Task 9 — :func:`md.archive.plan_cleanup`,
:func:`md.archive.create_archive`, :func:`md.archive.restore_archive` — and
none of them were reachable without a Python prompt, which is the same as not
existing for the person the console is for.

**Nothing here destroys anything without showing the list first.** A cleanup
plan is computed, displayed, and then executed *as displayed*: recomputing at
the moment of the click would be a different answer from the one that was
agreed to. Archive-and-remove writes, verifies, and only then deletes — the one
outcome nobody recovers from is a run archived, deleted, and found unreadable.

The long operations run on a worker thread. Hashing and deflating a few
gigabytes is tens of seconds, and a window that stops repainting for tens of
seconds is indistinguishable from one that has crashed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import archive, library
from . import sources

T = TypeVar("T")

#: Shown when a run has nothing a cleanup would take.
NOTHING_TO_CLEAN = (
    "Nothing to remove. Every checkpoint here is either the newest, the best, "
    "or pinned — and the summary files are kilobytes worth keeping."
)

#: Why archiving takes as long as it does — the modal's one line of explanation.
#: A progress bar that cannot say how far along it is has to say what it is
#: doing instead, or a long operation is indistinguishable from a hung one.
HASHING_NOTE = (
    "Every file is hashed on the way in, which is what makes the archive verifiable later."
)


class _Work(QThread):
    """One long filesystem operation, off the event loop.

    Deliberately a bare callable rather than a job type per operation: archiving
    and restoring differ only in which function is called, and three near-copies
    of a QThread is how the error handling in two of them goes stale.
    """

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, work: Callable[[], object], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._work = work

    def run(self) -> None:
        try:
            self.done.emit(self._work())
        except (archive.ArchiveError, OSError) as error:
            self.failed.emit(str(error))


class _Busy(QDialog):
    """A modal "this is happening" with no cancel, because there is no safe one.

    An archive half-written is deleted by :func:`md.archive.create_archive`'s own
    temporary-and-rename; an archive half-verified has changed nothing. But a
    *removal* interrupted between two files is a run in a state nobody can
    reason about, so the honest interface is to say what is happening and let it
    finish rather than to offer a stop that only sometimes stops.
    """

    def __init__(
        self,
        what: str,
        work: Callable[[], object],
        parent: QWidget | None = None,
        note_text: str = HASHING_NOTE,
    ) -> None:
        super().__init__(parent)
        self.outcome: object | None = None
        self.error = ""

        self.setWindowTitle(what)
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)

        label = QLabel(f"{what}…")
        label.setProperty("role", "value")
        layout.addWidget(label)

        bar = QProgressBar()
        bar.setRange(0, 0)
        layout.addWidget(bar)

        note = QLabel(note_text)
        note.setWordWrap(True)
        note.setProperty("role", "note")
        layout.addWidget(note)

        self._worker = _Work(work, self)
        self._worker.done.connect(self._finished)
        self._worker.failed.connect(self._blame)

    def exec(self) -> int:
        self._worker.start()
        return super().exec()

    def _finished(self, outcome: object) -> None:
        self.outcome = outcome
        self.accept()

    def _blame(self, message: str) -> None:
        self.error = message
        self.reject()


def run_work(
    what: str, work: Callable[[], T], parent: QWidget | None, note: str = HASHING_NOTE
) -> T | None:
    """Run ``work`` behind a modal, or report why it could not be done.

    Generic so a caller keeps its own return type: the signal that carries the
    result across the thread boundary can only be `object`, and every call site
    casting it back by hand is how one of them ends up casting it wrong.

    ``note`` is the line under the progress bar. It has a default because three
    of the four callers are archiving; a caller that is doing something else has
    to say so, or the modal explains an operation that is not happening.
    """
    busy = _Busy(what, work, parent, note)
    if busy.exec() == QDialog.DialogCode.Accepted:
        return cast(T, busy.outcome)
    if busy.error:
        QMessageBox.warning(parent, f"{what} failed", busy.error)
    return None


class StorageDialog(QDialog):
    """What one run costs, and the two ways to get it back."""

    def __init__(self, run: library.Run, root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._run = run
        self._root = root
        self.changed = False

        self.setWindowTitle(f"Storage — {run.name}")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)

        usage = library.storage_of(run.path)
        breakdown = QLabel(
            f"{sources.human_size(usage.total)} in total — "
            f"{sources.human_size(usage.checkpoints)} of checkpoints, "
            f"{sources.human_size(usage.recordings)} of recordings, "
            f"{sources.human_size(usage.other)} of everything else."
        )
        breakdown.setWordWrap(True)
        breakdown.setProperty("role", "value")
        layout.addWidget(breakdown)

        # The pins by absolute path, which is what `plan_cleanup` matches on:
        # `LIBRARY.json` stores bare names, and a name is not a file.
        pinned = [run.path / name for name in library.read_metadata(run.path).pinned]
        self._plan = archive.plan_cleanup(run, pinned)
        caption = QLabel(self._cleanup_caption())
        caption.setWordWrap(True)
        caption.setProperty("role", "note")
        layout.addWidget(caption)

        # The list, always — this is the "shown before it is executed" half, and
        # a count with no names is not a thing anyone can agree to.
        self._list = QListWidget()
        self._list.addItems(archive.entries_of(self._plan.remove))
        self._list.setMaximumHeight(160)
        self._list.setVisible(not self._plan.empty)
        layout.addWidget(self._list)

        options = QHBoxLayout()
        options.setSpacing(10)
        self._with_checkpoints = QCheckBox("Archive checkpoints")
        self._with_checkpoints.setChecked(True)
        self._with_recordings = QCheckBox("Archive recordings")
        self._with_recordings.setChecked(True)
        options.addWidget(self._with_checkpoints)
        options.addWidget(self._with_recordings)
        options.addStretch(1)
        layout.addLayout(options)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self._clean = QPushButton(f"Clean up ({archive.describe(self._plan)})")
        self._clean.setEnabled(not self._plan.empty)
        self._clean.clicked.connect(self._clean_up)
        self._archive = QPushButton("Archive…")
        self._archive.clicked.connect(lambda: self._archive_run(remove=False))
        self._move = QPushButton("Archive and remove…")
        self._move.clicked.connect(lambda: self._archive_run(remove=True))
        for button in (self._clean, self._archive, self._move):
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _cleanup_caption(self) -> str:
        if self._plan.empty:
            return NOTHING_TO_CLEAN
        return (
            f"Cleaning up removes these {len(self._plan.remove)} checkpoints and "
            f"frees {sources.human_size(self._plan.reclaim_bytes)}. The newest, the "
            "best-evaluated and every pinned one are kept, and so is every summary "
            "file the console draws from."
        )

    def _selection(self) -> archive.ArchiveSelection:
        return archive.ArchiveSelection(
            checkpoints=self._with_checkpoints.isChecked(),
            recordings=self._with_recordings.isChecked(),
        )

    def _clean_up(self) -> None:
        answer = QMessageBox.question(
            self,
            "Clean up this run?",
            f"{len(self._plan.remove)} checkpoints will be deleted, freeing "
            f"{sources.human_size(self._plan.reclaim_bytes)}. This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        # The plan computed when this dialog opened, not a fresh one: executed
        # *as shown* is the whole promise of showing it.
        plan = self._plan
        removed = run_work("Cleaning up", lambda: archive.apply_cleanup(plan, self._root), self)
        if removed is None:
            return
        self.changed = True
        QMessageBox.information(self, "Cleaned up", f"Removed {removed} files.")
        self.accept()

    def _archive_run(self, *, remove: bool) -> None:
        suggested = str(Path.home() / f"{self._run.name}.zip")
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Archive to", suggested, "ZIP archive (*.zip)"
        )
        if not chosen:
            return
        destination = Path(chosen)
        selection = self._selection()

        if remove:
            answer = QMessageBox.question(
                self,
                "Archive and remove?",
                f"{self._run.name} will be written to {destination.name}, the archive "
                "verified, and only then the original deleted. If verification fails "
                "nothing is removed.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            outcome = run_work(
                "Archiving and removing",
                lambda: archive.archive_and_remove(
                    self._run, destination, root=self._root, selection=selection
                ),
                self,
            )
            if outcome is None:
                return
            _, freed = outcome
            self.changed = True
            QMessageBox.information(
                self,
                "Archived",
                f"Written to {destination} and verified; {sources.human_size(freed)} freed.",
            )
            self.accept()
            return

        written = run_work(
            "Archiving", lambda: archive.create_archive(self._run, destination, selection), self
        )
        if written is None:
            return
        QMessageBox.information(self, "Archived", f"Written to {written}.")


def restore(root: Path, parent: QWidget | None = None) -> Path | None:
    """Ask for an archive and put it back, verifying before anything is written.

    Verification first, always: an archive that has rotted, been truncated in
    transit, or was never one is caught before a single file lands in the
    library. Zip-slip, absolute paths and symlinks are *refused* rather than
    sanitised — see :func:`md.archive.verify_archive`.
    """
    chosen, _ = QFileDialog.getOpenFileName(
        parent, "Restore a run", str(Path.home()), "ZIP archive (*.zip)"
    )
    if not chosen:
        return None
    source = Path(chosen)
    try:
        manifest = archive.verify_archive(source)
    except (archive.ArchiveError, OSError) as error:
        QMessageBox.warning(parent, "That archive cannot be restored", str(error))
        return None
    # Into a directory named for the run it holds, never over an existing one:
    # a half-merged run has `metrics.csv` from one and checkpoints from another,
    # and nothing downstream would notice.
    destination = root / manifest.run_id
    if destination.exists():
        QMessageBox.warning(
            parent,
            "That run is already here",
            f"{destination.name} already exists in this library. Rename or move it "
            "first — restoring on top of it would mix two runs together.",
        )
        return None
    outcome = run_work("Restoring", lambda: archive.restore_archive(source, destination), parent)
    if outcome is None:
        return None
    restored = outcome
    QMessageBox.information(parent, "Restored", f"{restored.name} is back in the library.")
    return restored
