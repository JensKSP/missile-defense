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

Three things here are deliberate:

* **Only changed values are passed.** A field left alone is left to the
  dataclass, so the command line reads as the difference from the defaults.
* **The command line is shown.** The console must not become the only way to
  start a run — you can read it off the dialog and type it into a terminal.
* **Resuming is a picker, not a field.** ``--resume`` takes a path to a file
  that already exists, so a text box would only be a way to mistype one. It is
  also the first question about a run rather than a twenty-first — start over,
  or carry on? — so it sits above the parameters instead of in with them.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
from .sources import Checkpoint, human_age, human_size

#: Wide enough for the four headline fields and their explanations.
DIALOG_WIDTH = 620
#: `--updates 2000000000` is nonsense, but a spin box needs *some* ceiling.
SPIN_MAX = 2_000_000_000

#: The picker's first entry — and its default, because "carry on" is the choice
#: that has to be made deliberately.
FROM_SCRATCH = "start from scratch"

#: Why a resume is not the same as loading weights, in the one place someone is
#: about to do it. The same reasoning is in docs/TRAINING.md.
RESUME_HELP = (
    "Continue a run from this checkpoint — weights, optimizer and iteration.\n\n"
    "The optimizer is the part that matters: Adam carries momentum estimates, "
    "and restarting without them makes the next few updates behave unlike the "
    "ones before, which looks like a kink in the curve rather than the artefact "
    "it is.\n\n"
    "metrics.csv is appended, so the history stays whole."
)


def _read(editor: QWidget) -> str:
    """One string out of whichever editor a field was given.

    One place, so adding an editor type cannot leave `values()` silently
    returning "" for it — which would look like "unchanged" and quietly drop the
    setting.
    """
    if isinstance(editor, QSpinBox):
        return str(editor.value())
    if isinstance(editor, QCheckBox):
        return "True" if editor.isChecked() else "False"
    if isinstance(editor, QComboBox):
        return editor.currentText().strip()
    if isinstance(editor, QLineEdit):
        return editor.text().strip()
    return ""


class ParameterDialog(QDialog):
    """Configure a run, then start it."""

    def __init__(
        self,
        fields: list[Param],
        *,
        python: str,
        out_dir: Path,
        checkpoints: Sequence[Checkpoint] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._fields = fields
        self._python = python
        self._out_dir = out_dir
        self._editors: dict[str, QWidget] = {}
        #: What each editor showed when it was built. `values()` compares
        #: against this rather than the dataclass's source text, because the two
        #: differ for optional fields — see the note there.
        self._initial: dict[str, str] = {}
        self._resume: QComboBox | None = None

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

        # No checkpoints means no row at all rather than a disabled one: this is
        # a dialog you dismiss, not a panel you leave open, and "you cannot
        # continue a run that has not happened" is not news.
        if checkpoints:
            layout.addWidget(self._resume_row(checkpoints))

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
    def _resume_row(self, checkpoints: Sequence[Checkpoint]) -> QWidget:
        """The checkpoints this run directory already holds, newest first."""
        frame = QFrame()
        frame.setProperty("role", "panel")
        form = QFormLayout(frame)
        form.setContentsMargins(14, 12, 14, 12)
        form.setSpacing(8)

        self._resume = QComboBox()
        self._resume.addItem(FROM_SCRATCH, "")
        now = time.time()
        for checkpoint in checkpoints:
            self._resume.addItem(
                f"{checkpoint.name}   {human_age(now - checkpoint.modified)} · "
                f"{human_size(checkpoint.size)}",
                str(checkpoint.path),
            )
        self._resume.currentIndexChanged.connect(self._refresh_preview)

        label = QLabel("continue from")
        for widget in (label, self._resume):
            widget.setToolTip(RESUME_HELP)
        form.addRow(label, self._resume)
        return frame

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
            self._initial[field.name] = _read(editor)
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
        """The narrowest control the field's type allows.

        Narrow on purpose. A run is hours, and a value that cannot be entered
        costs nothing where a value that can be mistyped costs the run: a
        dropdown cannot be misspelled `entty`, and a spin box that stops at 1
        cannot be given `--envs 0`. The bounds in `md.ui.params.BOUNDS` are the
        range in which a number is a *decision* rather than a typo, and they are
        deliberately generous at the top — the job is to catch a slipped decimal
        place, not to have an opinion about someone's experiment.
        """
        if field.choices:
            choose = QComboBox()
            choose.addItems(list(field.choices))
            if field.default.strip("'\"") in field.choices:
                choose.setCurrentText(field.default.strip("'\""))
            choose.currentTextChanged.connect(self._refresh_preview)
            return choose
        if field.kind == "bool":
            check = QCheckBox()
            check.setChecked(field.default.strip() == "True")
            check.toggled.connect(self._refresh_preview)
            return check
        if field.kind == "int":
            spin = QSpinBox()
            low, high = field.bounds or (0, SPIN_MAX)
            spin.setRange(int(low), min(int(high), SPIN_MAX))
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
            # written and read; 0.000300 in a spin box is neither. So a
            # validator rather than a spin box — and it carries the bounds, so
            # the range is enforced without giving up the notation.
            validator = QDoubleValidator()
            validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
            if field.bounds is not None:
                validator.setRange(field.bounds[0], field.bounds[1], 12)
            edit.setValidator(validator)
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
            value = _read(editor)
            # Against what the editor was *built* showing, not against the
            # dataclass's source text. They differ for an optional field: a
            # `int | None = None` gets a spin box sitting on 0, and comparing
            # "0" with "" made the dialog emit `--schedule-updates 0` for a
            # field nobody had touched.
            if value and value != self._initial.get(field.name, ""):
                changed[field.name] = value
        return changed

    def resume(self) -> Path | None:
        """The checkpoint to continue from, or ``None`` for a fresh run."""
        if self._resume is None:
            return None
        chosen = self._resume.currentData()
        return Path(str(chosen)) if chosen else None

    def command(self) -> list[str]:
        return params_module.command_line(
            self._python, self.values(), out_dir=self._out_dir, resume=self.resume()
        )

    def _refresh_preview(self) -> None:
        self._preview.setText(" ".join(self.command()))
