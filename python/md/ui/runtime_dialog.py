# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# pyright: reportMissingImports=false
"""One-click setup for the training runtime.

The console could always watch a run and could only start one where torch was
already importable, which made "open a terminal and pip install torch" the first
instruction an installed copy had to give. This is the dialog that replaces it.

All the thinking is in :mod:`md.runtime`; this is the surface. It shows what
would be installed *before* installing it — the backend, why that one, where it
comes from and roughly how large it is — because a multi-gigabyte download that
starts the moment you open a window is not a choice you were offered.

Three things are deliberate:

* **The install runs on a thread.** pip takes minutes, and a dialog that stops
  repainting for minutes is indistinguishable from one that has hung. Cancel has
  to keep working, which it cannot do from the thread doing the work.
* **Progress is honest, not decorative.** pip reports no total this can turn into
  a percentage, so the bar is indeterminate and the *log* is the real progress
  report. It sits behind a disclosure, because most people want "installing…" and
  the ones who want the resolver's output want all of it.
* **Nothing is offered that cannot work.** The backend list is what
  :mod:`md.runtime` says this platform can install — an AMD card on Windows is
  offered the CPU build, not a ROCm install that downloads and then fails.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import runtime
from ..runtime import Backend, Runtime, RuntimePlan, RuntimeStatus, SystemInfo

#: Wide enough for a pip line without wrapping every one of them.
DIALOG_WIDTH = 640

#: What the dialog opens saying, before a backend has been chosen for you.
INTRO = (
    "Training needs PyTorch. This installs a copy the console manages itself, "
    "in your data directory — it does not touch your system Python, and it can "
    "be removed again from here."
)


class _Install(QThread):
    """Runs one install off the event loop, reporting as it goes."""

    line = Signal(str)
    done = Signal(object)

    def __init__(self, store: Runtime, plan: RuntimePlan, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = store
        self._plan = plan
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Ask the install to stop at the next step boundary.

        Not a kill: a half-killed pip leaves a directory nobody can reason about.
        :meth:`md.runtime.Runtime.install` checks between steps and deletes the
        directory it was filling, which is a stop that leaves no trace.
        """
        self._cancel.set()

    def run(self) -> None:
        status = self._store.install(
            self._plan,
            on_output=self.line.emit,
            cancel=self._cancel.is_set,
        )
        self.done.emit(status)


class RuntimeDialog(QDialog):
    """Install, repair or remove the training runtime."""

    def __init__(
        self,
        store: Runtime | None = None,
        *,
        system: SystemInfo | None = None,
        probes: Sequence[runtime.BackendProbe] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = Runtime() if store is None else store
        self._system = SystemInfo.here() if system is None else system
        self._probes = tuple(runtime.default_probes() if probes is None else probes)
        self._worker: _Install | None = None

        self.setWindowTitle("Training runtime")
        self.setMinimumWidth(DIALOG_WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)

        intro = QLabel(INTRO)
        intro.setWordWrap(True)
        intro.setProperty("role", "note")
        layout.addWidget(intro)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setProperty("role", "value")
        layout.addWidget(self._status_label)

        layout.addWidget(self._choice_panel())

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # pip has no total to turn into a percentage
        self._progress.setVisible(False)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        layout.addWidget(self._log_disclosure())
        layout.addLayout(self._buttons())

        self._refresh()

    # ---- construction -------------------------------------------------------

    def _choice_panel(self) -> QWidget:
        """What would be installed: the backend, its reason, size and source."""
        frame = QFrame()
        frame.setProperty("role", "panel")
        form = QFormLayout(frame)
        form.setContentsMargins(14, 12, 14, 12)
        form.setSpacing(8)

        self._backend = QComboBox()
        for backend in self._installable():
            self._backend.addItem(backend.label, backend.name)
        self._backend.currentIndexChanged.connect(self._backend_changed)
        form.addRow("Build", self._backend)

        self._why = QLabel()
        self._why.setWordWrap(True)
        self._why.setProperty("role", "note")
        form.addRow("", self._why)

        self._source = QLabel()
        self._source.setProperty("role", "note")
        self._source.setWordWrap(True)
        form.addRow("From", self._source)
        return frame

    def _log_disclosure(self) -> QWidget:
        container = QWidget()
        box = QVBoxLayout(container)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)

        self._log_toggle = QToolButton()
        self._log_toggle.setText("Details")
        self._log_toggle.setCheckable(True)
        self._log_toggle.toggled.connect(lambda shown: self._log.setVisible(shown))
        box.addWidget(self._log_toggle)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.setFixedHeight(160)
        self._log.setVisible(False)
        box.addWidget(self._log)
        return container

    def _buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        self._primary = QPushButton()
        self._primary.setProperty("role", "primary")
        self._primary.clicked.connect(self._primary_pressed)
        self._remove = QPushButton("Remove")
        self._remove.clicked.connect(self._remove_pressed)
        self._close = QPushButton("Close")
        self._close.clicked.connect(self.reject)
        row.addWidget(self._primary)
        row.addWidget(self._remove)
        row.addStretch(1)
        row.addWidget(self._close)
        return row

    # ---- state --------------------------------------------------------------

    def _installable(self) -> tuple[Backend, ...]:
        """The builds that exist for this platform, best first."""
        return tuple(b for b in runtime.BACKENDS if self._system.platform in b.platforms)

    def _selected(self) -> Backend:
        return runtime.backend_for(str(self._backend.currentData()))

    def _plan(self) -> RuntimePlan:
        """The recommendation, overridden by whatever the picker now says.

        The recommendation decides the *default*; someone who knows their driver
        is too old for the CUDA wheel has to be able to pick CPU without editing
        anything.
        """
        recommended = runtime.recommend(self._system, self._probes, root=self._store.root)
        chosen = self._selected()
        if chosen.name == recommended.backend:
            return recommended
        return RuntimePlan(
            backend=chosen.name,
            python=recommended.python,
            target=recommended.target.with_name(
                recommended.target.name.replace(recommended.backend, chosen.name, 1)
            ),
            packages=chosen.packages,
            index_url=chosen.index_url,
        )

    def _refresh(self, note: str = "") -> None:
        """Re-read the store and make every control say what it now means.

        ``note`` overrides the headline with what just happened. After a failed
        or cancelled install the store is back to ``absent``, and "Not installed
        yet." — true, but the reason thrown away — is the wrong thing to say to
        someone who just watched it try.

        `verify`, not `status`. This dialog is the only way out of a runtime that
        has stopped working, so it is the last place that may believe a manifest:
        a store that says ready while the console's own background check says
        otherwise is a window offering *Start a run* beside a button that says
        *Set up training*, and no way to reconcile them. The result is cached
        against that manifest and the console has usually paid for it already, so
        agreeing with it costs nothing.
        """
        status = self._store.verify()
        self._status_label.setText(note or self._headline(status))
        self._remove.setEnabled(status.removable)

        if status.ready:
            self._primary.setText("Start a run")
        elif status.repairable:
            self._primary.setText("Repair and install")
        else:
            self._primary.setText("Install")
        self._primary.setEnabled(True)
        self._backend.setEnabled(not status.ready)
        self._backend_changed()

        if status.ready:
            # Pre-select what is installed, so the picker describes reality
            # rather than what would be installed if it were not.
            index = self._backend.findData(status.backend)
            if index >= 0:
                self._backend.setCurrentIndex(index)

    @staticmethod
    def _headline(status: RuntimeStatus) -> str:
        if status.ready:
            return f"Installed and working — {status.detail}"
        if status.repairable:
            return f"Needs attention — {status.detail}"
        return "Not installed yet."

    def _backend_changed(self) -> None:
        backend = self._selected()
        self._why.setText(backend.detail)
        self._source.setText(f"{backend.index_url}   ·   about {backend.gigabytes:.1f} GB")

    # ---- actions ------------------------------------------------------------

    def _primary_pressed(self) -> None:
        if self._worker is not None:  # the button is Cancel while installing
            self._worker.cancel()
            self._primary.setEnabled(False)
            self._primary.setText("Cancelling…")
            return
        status = self._store.verify()  # what the label was drawn from, cached
        if status.ready:
            self.accept()  # straight on to the new-run dialog
            return
        if status.repairable:
            self._store.repair()
        self._start_install()

    def _start_install(self) -> None:
        plan = self._plan()
        self._log.clear()
        self._append(f"installing the {self._selected().label} build into {plan.target}")
        self._progress.setVisible(True)
        self._remove.setEnabled(False)
        self._close.setEnabled(False)
        self._backend.setEnabled(False)
        self._primary.setText("Cancel")

        self._worker = _Install(self._store, plan, self)
        self._worker.line.connect(self._append)
        self._worker.done.connect(self._finished)
        self._worker.start()

    def _finished(self, status: RuntimeStatus) -> None:
        self._worker = None
        self._progress.setVisible(False)
        self._close.setEnabled(True)
        self._append(status.detail)
        if not status.ready:
            # A failure is the one time the log is worth more than the summary,
            # so it opens itself rather than waiting to be asked for.
            self._log_toggle.setChecked(True)
        self._refresh("" if status.ready else status.detail)

    def _remove_pressed(self) -> None:
        self._store.remove()
        self._append("removed")
        self._refresh()

    def _append(self, line: str) -> None:
        self._log.appendPlainText(line)

    def reject(self) -> None:
        """Closing mid-install cancels it rather than orphaning the thread."""
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(30_000)
            self._worker = None
        super().reject()
