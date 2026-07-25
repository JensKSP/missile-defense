# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# pyright: reportMissingImports=false
"""The parameter dialog — the one surface that is not the main screen.

Twenty-odd hyperparameters with good, reasoned defaults do not belong on the
screen you leave open all day, and they do not belong in a wall of boxes either.
Four fields change a run's character (envs, steps, updates, learning rate); the
rest sit behind *Advanced*, and every one of them carries as its tooltip the
sentence already written beside it in the code (:mod:`md.ui.params`).

Two things here are deliberate:

* **Only changed values are passed.** A field left alone is left to the
  dataclass, so the command line reads as the difference from the defaults.
* **The command line is shown.** The console must not become the only way to
  start a run — you can read it off the dialog and type it into a terminal.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import params as params_module
from .params import Param

#: Wide enough for the four headline fields and their explanations.
DIALOG_WIDTH = 620
#: `--updates 2000000000` is nonsense, but a spin box needs *some* ceiling.
SPIN_MAX = 2_000_000_000


class ParameterDialog(QDialog):
    """Configure a run, then start it."""

    def __init__(
        self,
        fields: list[Param],
        *,
        python: str,
        out_dir: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._fields = fields
        self._python = python
        self._out_dir = out_dir
        self._editors: dict[str, QWidget] = {}

        self.setWindowTitle("Start a training run")
        self.setMinimumWidth(DIALOG_WIDTH)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)

        if not fields:
            # An installed console watching a synced directory has no trainer to
            # read; say that, rather than showing an empty form.
            missing = QLabel(
                "The trainer's source is not beside this console, so its "
                "parameters cannot be read. A run started here would use the "
                "defaults."
            )
            missing.setWordWrap(True)
            missing.setProperty("role", "note")
            layout.addWidget(missing)

        headline = [field for field in fields if field.headline]
        advanced = [field for field in fields if not field.headline]
        if headline:
            layout.addWidget(self._group(headline))
        if advanced:
            layout.addWidget(self._advanced(advanced))

        self._preview = QLabel()
        self._preview.setProperty("role", "preview")
        self._preview.setWordWrap(True)
        self._preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._preview)

        # The command alone is not quite runnable: `md` is imported from the
        # checkout rather than installed, which the console arranges for its
        # child. Say so, or "type it in a terminal yourself" is not true.
        note = QLabel("run from the project root with python/ on PYTHONPATH")
        note.setProperty("role", "note")
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        start = buttons.button(QDialogButtonBox.StandardButton.Ok)
        start.setText("Start run")
        start.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh_preview()

    # ---- construction -------------------------------------------------------
    def _group(self, fields: list[Param]) -> QWidget:
        frame = QFrame()
        frame.setProperty("role", "panel")
        form = QFormLayout(frame)
        form.setContentsMargins(14, 12, 14, 12)
        form.setSpacing(8)
        for field in fields:
            label = QLabel(field.name.replace("_", " "))
            editor = self._editor(field)
            for widget in (label, editor):
                widget.setToolTip(field.help or f"{field.owner}.{field.name}")
            self._editors[field.name] = editor
            form.addRow(label, editor)
        return frame

    def _advanced(self, fields: list[Param]) -> QWidget:
        container = QWidget()
        box = QVBoxLayout(container)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)

        toggle = QToolButton()
        toggle.setText(f"Advanced  ({len(fields)} more)")
        toggle.setCheckable(True)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.ArrowType.RightArrow)
        box.addWidget(toggle, alignment=Qt.AlignmentFlag.AlignLeft)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._group(fields))
        scroll.setMaximumHeight(320)
        scroll.setVisible(False)
        box.addWidget(scroll)

        def reveal(shown: bool) -> None:
            scroll.setVisible(shown)
            toggle.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)
            self.adjustSize()

        toggle.toggled.connect(reveal)
        return container

    def _editor(self, field: Param) -> QWidget:
        if field.kind == "int":
            spin = QSpinBox()
            spin.setRange(0, SPIN_MAX)
            # Grouped, but in the console's own convention rather than the
            # machine's: under a German locale Qt renders 1024 as "1.024", which
            # next to a command line reading `--envs 1024` is a puzzle.
            spin.setLocale(QLocale("en_US"))
            spin.setGroupSeparatorShown(True)
            spin.setValue(int(field.default or 0))
            spin.valueChanged.connect(self._refresh_preview)
            return spin
        edit = QLineEdit(field.default)
        if field.kind == "float":
            # Scientific notation on purpose: 3e-4 is how a learning rate is
            # written and read; 0.000300 in a spin box is neither.
            edit.setValidator(QDoubleValidator())
        edit.setPlaceholderText("auto" if not field.default else field.default)
        edit.textChanged.connect(self._refresh_preview)
        return edit

    # ---- what came out of it ------------------------------------------------
    def values(self) -> dict[str, str]:
        """Only the fields whose value differs from the trainer's default."""
        changed: dict[str, str] = {}
        for field in self._fields:
            editor = self._editors.get(field.name)
            if editor is None:
                continue
            value = (
                str(editor.value())
                if isinstance(editor, QSpinBox)
                else editor.text().strip()
                if isinstance(editor, QLineEdit)
                else ""
            )
            if value and value != field.default:
                changed[field.name] = value
        return changed

    def command(self) -> list[str]:
        return params_module.command_line(self._python, self.values(), out_dir=self._out_dir)

    def _refresh_preview(self) -> None:
        self._preview.setText(" ".join(self.command()))
