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
* **The command line is shown.** The trainer must not become the only way to
  start a run — you can read it off the dialog and type it into a terminal.
* **Resuming is a picker, not a field.** ``--resume`` takes a path to a file
  that already exists, so a text box would only be a way to mistype one. It is
  also the first question about a run rather than a twenty-first — start over,
  or carry on? — so it sits above the parameters instead of in with them.
* **Continuing a run opens on that run's own settings.** They arrive as
  ``initial`` (from its ``config.json``, via :mod:`md.runconfig`) and are poured
  in the way a preset is. Retyping them from memory was the step that used to
  cost a run: the trainer rejects a resume whose architecture, hidden size or
  annealing schedule disagrees with the checkpoint, and quietly accepts a
  different rollout length as a different experiment under the same run's name.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
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
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import footprint, presets
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

#: What the picker says when no named set is loaded — which is the state the
#: dialog opens in, and the state it returns to the moment a value is edited.
NO_PRESET_SUMMARY = "The trainer's own defaults, plus whatever you change below."

#: The one place someone meets presets, so it says what they are and what the
#: three built-in ones mean. The same reasoning is in docs/TRAINING.md.
PRESET_HELP = (
    "A named set of training options.\n\n"
    "fast — throughput first: saturates the card, finishes in minutes, and is for "
    "checking the loop turns rather than for producing a policy.\n"
    "good — the recipe that produced the bundled model (23,067 on the held-out "
    "block). The one preset with a measured result behind it.\n"
    "best — that recipe scaled up along the axes the documentation recommends. A "
    "considered bet on a long run, not a measured result.\n\n"
    "Save as… keeps what is in this form under a name of your own. The three "
    "above cannot be changed or deleted — take a copy instead."
)

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


def free_vram() -> int | None:
    """Bytes free on the training card, or ``None`` when there is nothing to ask.

    Through the trainer's own vendor probes, so the dialog agrees with the number
    the System panel is showing a few centimetres away. Every failure is the same
    failure — no card, no driver, no binding, a vendor library raising its own
    exception hierarchy — and all of them mean "do not claim to know".
    """
    try:
        from .system import find_gpu_probe  # noqa: PLC0415 — optional, and slow to import

        probe = find_gpu_probe()
        sample = probe.sample() if probe is not None else None
    except Exception:  # noqa: BLE001 — every vendor error means the same here
        return None
    if sample is None or sample.memory_total is None or sample.memory_used is None:
        return None
    return max(sample.memory_total - sample.memory_used, 0)


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


def _write(editor: QWidget, value: str) -> None:
    """Put ``value`` into whichever editor a field was given — the inverse of
    :func:`_read`, and the reason applying a preset is four lines rather than a
    type switch at the call site.

    A value an editor cannot hold is *dropped*, not forced: a spin box asked for
    "auto" keeps what it had. Presets come from a file a person may have edited,
    and a form that silently turned a typo into ``--envs 0`` would start the run
    anyway. The command-line preview underneath always shows what will actually
    be used.
    """
    if isinstance(editor, QSpinBox):
        try:
            editor.setValue(int(value))
        except ValueError:
            pass
        return
    if isinstance(editor, QCheckBox):
        editor.setChecked(value.strip() == "True")
        return
    if isinstance(editor, QComboBox):
        index = editor.findText(value.strip())
        if index >= 0:
            editor.setCurrentIndex(index)
        return
    if isinstance(editor, QLineEdit):
        editor.setText(value)


def _shown(field: Param) -> str:
    """A field's default as a box can show it, or "" when it cannot show it.

    Empty means the trainer's own default stands, which the placeholder reads as
    *auto* — and that is the truth about a numeric default this trainer failed to
    read. `md.ui.params` follows a named constant to its value, so the usual
    answer is the number; this is what is left when a default is an expression
    nothing here can evaluate. Showing the expression instead would offer
    ``--aim-trail CANONICAL_AIM_TRAIL`` to anyone who then edited the field, and
    the trainer rejects that at the point where it has already been started.
    """
    if field.kind not in ("int", "float"):
        return field.default
    try:
        float(field.default)
    except ValueError:
        return ""
    return field.default


class ParameterDialog(QDialog):
    """Configure a run, then start it."""

    def __init__(
        self,
        fields: list[Param],
        *,
        python: str,
        out_dir: Path,
        checkpoints: Sequence[Checkpoint] = (),
        initial: Mapping[str, str] | None = None,
        resume: Path | None = None,
        presets_file: Path | None = None,
        free_vram_bytes: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._fields = fields
        self._python = python
        self._out_dir = out_dir
        self._presets_file = presets_file
        #: Read once. A dialog is open for a minute, and a number that moved
        #: while you were choosing would only ever move for the worse.
        self._free_vram = free_vram() if free_vram_bytes is None else free_vram_bytes
        self._editors: dict[str, QWidget] = {}
        #: What each editor showed when it was built. `values()` compares
        #: against this rather than the dataclass's source text, because the two
        #: differ for optional fields — see the note there.
        self._initial: dict[str, str] = {}
        self._resume: QComboBox | None = None
        #: True while a preset is being poured into the editors, so their change
        #: signals do not read that as the person typing and flip the picker to
        #: *custom* halfway through applying it.
        self._applying = False

        self.setWindowTitle("Start a training run")
        self.setMinimumWidth(DIALOG_WIDTH)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)

        if not fields:
            # An installed trainer watching a synced directory has no md.train
            # to read; say that, rather than showing an empty form.
            missing = QLabel(
                "The training loop's source is not beside this trainer, so its "
                "parameters cannot be read. A run started here would use the "
                "defaults."
            )
            missing.setWordWrap(True)
            missing.setProperty("role", "note")
            layout.addWidget(missing)

        # Above the fields, because it fills them in: a preset is the answer to
        # "what should these be?", and reading it after the twenty boxes it
        # sets would be reading the answer after the exam.
        if fields:
            layout.addWidget(self._preset_row())

        # No checkpoints means no row at all rather than a disabled one: this is
        # a dialog you dismiss, not a panel you leave open, and "you cannot
        # continue a run that has not happened" is not news.
        if checkpoints:
            layout.addWidget(self._resume_row(checkpoints, resume))

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

        # What the run will ask of the card, under the command that asks for it.
        # Only where there are fields to read it from — with no trainer source
        # there is nothing to estimate from and nothing to change if it is wrong.
        self._memory: QLabel | None = None
        if fields:
            self._memory = QLabel()
            self._memory.setProperty("role", "note")
            self._memory.setWordWrap(True)
            layout.addWidget(self._memory)

        # The command alone is not quite runnable: `md` is imported from the
        # checkout rather than installed, which the trainer arranges for its
        # child. Say so, or "type it in a terminal yourself" is not true.
        note = QLabel("run from the project root with python/ on PYTHONPATH")
        note.setProperty("role", "note")
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._go = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._go.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        # Last, so every editor exists to pour into. Not before the preview
        # either: what this fills in is exactly what the preview has to show.
        self._apply_initial(initial or {})
        self._refresh_preview()

    # ---- presets ------------------------------------------------------------
    def _preset_row(self) -> QWidget:
        """Pick a named set of options, and save, update or delete your own."""
        frame = QFrame()
        frame.setProperty("role", "panel")
        box = QVBoxLayout(frame)
        box.setContentsMargins(14, 12, 14, 12)
        box.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(8)
        label = QLabel("preset")
        label.setToolTip(PRESET_HELP)
        self._presets = QComboBox()
        self._presets.setToolTip(PRESET_HELP)
        self._presets.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.addWidget(label)
        row.addWidget(self._presets, stretch=1)

        self._save_preset = QPushButton("Save as…")
        self._save_preset.setToolTip("Save what is in this form under a name of your own")
        self._save_preset.clicked.connect(self._save_preset_as)
        self._update_preset = QPushButton("Update")
        self._update_preset.setToolTip("Replace the selected preset with what is in this form")
        self._update_preset.clicked.connect(self._update_selected_preset)
        self._delete_preset = QPushButton("Delete")
        self._delete_preset.setToolTip("Forget the selected preset")
        self._delete_preset.clicked.connect(self._delete_selected_preset)
        for button in (self._save_preset, self._update_preset, self._delete_preset):
            row.addWidget(button)
        box.addLayout(row)

        # What the selected preset is *for*. A name alone cannot carry "this one
        # has a measured score behind it and that one is a considered bet".
        self._preset_summary = QLabel()
        self._preset_summary.setProperty("role", "note")
        self._preset_summary.setWordWrap(True)
        box.addWidget(self._preset_summary)

        self._reload_presets()
        self._presets.currentIndexChanged.connect(self._preset_chosen)
        return frame

    def _reload_presets(self, *, select: str = presets.CUSTOM) -> None:
        """Rebuild the picker from the file, leaving ``select`` chosen.

        From the file every time rather than from a list held here: the trainer
        is not the only thing that may have written it, and a picker that showed
        a preset somebody deleted in an editor would offer to start a run from
        options that no longer exist.
        """
        self._saved = presets.all_presets(self._presets_file)
        blocked = self._presets.blockSignals(True)
        self._presets.clear()
        self._presets.addItem(presets.CUSTOM, "")
        for preset in self._saved:
            self._presets.addItem(preset.name, preset.name)
        index = self._presets.findData(select)
        self._presets.setCurrentIndex(max(index, 0))
        self._presets.blockSignals(blocked)
        self._refresh_preset_state()

    def _selected_preset(self) -> presets.Preset | None:
        name = str(self._presets.currentData() or "")
        return next((preset for preset in self._saved if preset.name == name), None)

    def _refresh_preset_state(self) -> None:
        """The summary line, and which of the three buttons make sense."""
        preset = self._selected_preset()
        if preset is None:
            self._preset_summary.setText(NO_PRESET_SUMMARY)
        elif preset.builtin:
            self._preset_summary.setText(f"{preset.summary}  (ships with the trainer)")
        else:
            self._preset_summary.setText(preset.summary or "Saved from this form.")
        # Built-ins are read-only: their names are quoted in the documentation,
        # so "good" has to keep meaning what is written there. Save as… always
        # works, and is how you take a copy of one and change it.
        editable = preset is not None and not preset.builtin
        self._update_preset.setEnabled(editable)
        self._delete_preset.setEnabled(editable)

    def _preset_chosen(self) -> None:
        """Pour the selection into the form. *custom* leaves it alone."""
        preset = self._selected_preset()
        self._refresh_preset_state()
        if preset is None:
            return
        self._applying = True
        try:
            for field in self._fields:
                editor = self._editors.get(field.name)
                if editor is None:
                    continue
                # Fields the preset does not mention go back to the trainer's
                # own default, so what is on screen is the preset and not the
                # preset over the residue of whatever was picked before it.
                _write(editor, str(preset.options.get(field.name, self._initial[field.name])))
        finally:
            self._applying = False
        self._refresh_preview()

    def _touched(self) -> None:
        """A field was edited by hand, so no named preset is on screen any more."""
        if self._applying or not hasattr(self, "_presets"):
            return
        if self._presets.currentIndex() != 0:
            blocked = self._presets.blockSignals(True)
            self._presets.setCurrentIndex(0)
            self._presets.blockSignals(blocked)
            self._refresh_preset_state()

    def _save_preset_as(self) -> None:
        selected = self._selected_preset()
        suggested = "" if selected is None or selected.builtin else selected.name
        name, accepted = QInputDialog.getText(
            self,
            "Save these options",
            "A name for this set of training options.\n\n"
            "Only the values that differ from the trainer's defaults are saved,\n"
            "so the preset stays readable as the command line below.",
            text=suggested,
        )
        if not accepted:
            return
        summary, _ = QInputDialog.getText(
            self, "Save these options", "One line on what it is for (optional)."
        )
        self._store(name, summary)

    def _update_selected_preset(self) -> None:
        preset = self._selected_preset()
        if preset is not None:
            self._store(preset.name, preset.summary)

    def _store(self, name: str, summary: str) -> None:
        try:
            saved = presets.save(name, self.values(), summary=summary, path=self._presets_file)
        except presets.PresetError as error:
            QMessageBox.warning(self, "That name will not do", str(error))
            return
        self._reload_presets(select=saved.name)

    def _delete_selected_preset(self) -> None:
        preset = self._selected_preset()
        if preset is None:
            return
        confirm = QMessageBox.question(
            self,
            "Forget this preset?",
            f"Delete '{preset.name}'?\n\nThe values stay in the form; only the name goes.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            presets.delete(preset.name, self._presets_file)
        except presets.PresetError as error:
            QMessageBox.warning(self, "Cannot delete that", str(error))
            return
        self._reload_presets()

    # ---- construction -------------------------------------------------------
    def _resume_row(self, checkpoints: Sequence[Checkpoint], chosen: Path | None) -> QWidget:
        """The checkpoints this run directory already holds, newest first.

        ``chosen`` is pre-picked where there is one — pressing *Continue* on a
        stopped run should not then have to be told, in a dropdown, which run it
        meant. *Start from scratch* stays the first entry, so training over a
        directory that already has checkpoints is still one click away and still
        a thing you choose rather than a thing that happens.
        """
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
        if chosen is not None:
            index = self._resume.findData(str(chosen))
            if index > 0:
                self._resume.setCurrentIndex(index)
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
            choose.currentTextChanged.connect(self._edited)
            return choose
        if field.kind == "bool":
            check = QCheckBox()
            check.setChecked(field.default.strip() == "True")
            check.toggled.connect(self._edited)
            return check
        if field.kind == "int":
            spin = QSpinBox()
            low, high = field.bounds or (0, SPIN_MAX)
            spin.setRange(int(low), min(int(high), SPIN_MAX))
            # Grouped, but in the trainer's own convention rather than the
            # machine's: under a German locale Qt renders 1024 as "1.024", which
            # next to a command line reading `--envs 1024` is a puzzle.
            spin.setLocale(QLocale("en_US"))
            spin.setGroupSeparatorShown(True)
            # Through `_write`, which drops what it cannot hold, rather than
            # `int(...)`, which raised. `md.ui.params` follows a named default to
            # its value, so this should not happen — but when it does, it happens
            # in a Qt slot, and PySide6 prints the traceback and carries on. That
            # is a Start button that does nothing, with the reason on a terminal
            # nobody has open. A box showing the wrong number is recoverable;
            # `values()` compares against what the editor was built showing, so
            # an untouched field is still not passed to the trainer.
            _write(spin, field.default)
            spin.valueChanged.connect(self._edited)
            return spin
        edit = QLineEdit(_shown(field))
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
        placeholder = _shown(field)
        edit.setPlaceholderText(placeholder or "auto")
        edit.textChanged.connect(self._edited)
        return edit

    def _apply_initial(self, initial: Mapping[str, str]) -> None:
        """Fill the form in from a run that has already answered these questions.

        Poured in the way a preset is, and for the same reason — but *without*
        being recorded as what the editors started on, so `values()` still
        reports them as set and the command line restates them. That matters:
        the command shown here has to be one you can paste into a terminal, and
        `--resume` alone would be a run whose settings came from a file rather
        than from anything on screen.
        """
        if not initial:
            return
        self._applying = True
        try:
            for name, value in initial.items():
                editor = self._editors.get(name)
                if editor is not None:
                    _write(editor, value)
        finally:
            self._applying = False

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

    def _edited(self) -> None:
        """Any editor changing: the preview follows, and a preset stops claiming
        to be what is on screen the moment a value stops matching it."""
        self._touched()
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        self._preview.setText(" ".join(self.command()))
        self._refresh_memory()
        # Two different acts behind one button, so it says which it is about to
        # do. Re-read from the picker rather than set once, because continuing
        # is a choice that can be changed while the dialog is open.
        continuing = self.resume() is not None
        self.setWindowTitle("Continue a training run" if continuing else "Start a training run")
        self._go.setText("Continue run" if continuing else "Start run")

    # ---- what it will cost --------------------------------------------------
    def _number(self, name: str, fallback: int) -> int:
        """One field's effective value — what it shows, not what it overrides."""
        editor = self._editors.get(name)
        if editor is None:
            return fallback
        try:
            return int(_read(editor))
        except ValueError:
            return fallback

    def _refresh_memory(self) -> None:
        """Say what this run will ask of the GPU, before it asks for it.

        The estimate is worth showing even when it fits: 17 GiB is the difference
        between "start it and go to bed" and "start it and stop using the
        machine", and nothing else on this dialog hints that the choice was made.
        Hidden entirely without a card to compare against — an estimate with
        nothing to measure it against is trivia.
        """
        if self._memory is None:
            return
        shape = {
            "envs": self._number("envs", 1024),
            "steps": self._number("steps", 256),
            "minibatches": self._number("minibatches", 8),
            "architecture": _read(self._editors["architecture"])
            if "architecture" in self._editors
            else "mlp",
        }
        estimate = footprint.estimate_gib(**shape)
        free = self._free_vram
        if free is None:
            self._memory.setText(f"≈ {estimate:.1f} GiB of GPU memory")
            self._memory.setProperty("role", "note")
        elif footprint.fits_in(free, **shape):
            self._memory.setText(
                f"≈ {estimate:.1f} GiB of GPU memory · {free / footprint.GIB:.1f} GiB free"
            )
            self._memory.setProperty("role", "note")
        else:
            # Not disabled, only warned: the estimate is a model, the card may be
            # freed by closing the game, and someone who knows better than this
            # dialog must still be able to press Start.
            self._memory.setText(
                f"⚠ ≈ {estimate:.1f} GiB of GPU memory, but only "
                f"{free / footprint.GIB:.1f} GiB is free — this run is likely to "
                f"run out. More --minibatches costs nothing: same data, smaller pieces."
            )
            self._memory.setProperty("role", "warning")
        # Qt caches the styled look; a role that changed has to be re-polished.
        style = self._memory.style()
        style.unpolish(self._memory)
        style.polish(self._memory)
