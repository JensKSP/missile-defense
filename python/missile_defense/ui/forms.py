# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# pyright: reportMissingImports=false
"""The parameter dialog — the one surface that is not the main screen.

Thirty-odd hyperparameters with good, reasoned defaults do not belong on the
screen you leave open all day, and they do not belong in a wall of boxes either.
This is laid out as a control station rather than a form:

* **Three tabs, named for the decision** — what the agent is paid for, how it
  learns, how big and how long. Not one per config class: someone starting a run
  chooses between those three questions, and only afterwards cares that they
  happen to be `Shaping`, `PPOConfig` and `TrainConfig`. The split lives in
  :data:`missile_defense.ui.params.GROUPS`, and a test holds it to the real
  dataclasses in both directions, so a field nobody placed fails rather than
  silently vanishing from the dialog.
* **Essentials open, the rest folded — and a closed fold still shows its
  values.** A fold that hides what it holds is the same wall of boxes with a
  button in front of it. None holds more than five, so opening one costs a
  glance; the single "everything else" drawer this replaced held thirty-three.
* **A bounded number gets a slider and a readout.** Position says where in the
  range you are, which identical spin boxes never did, and the box beside it
  keeps `3e-4` typeable. Three scales, because one does not fit — see
  :class:`ValueSlider`.
* **One status strip carries every explanation.** The `#:` sentence beside each
  field in the trainer's source, shown for whatever is pointed at *or focused*.
  Under each row instead, thirty-odd paragraphs were most of the dialog's height,
  and the reasoning is only ever wanted one at a time.
* **A filter, which is what makes folding safe.** Grouping only helps if you
  never have to remember which group something is in — so a search reveals a
  match wherever it lives, opening its fold and switching to its tab.

Four more things are deliberate:

* **Only changed values are passed.** A field left alone is left to the
  dataclass, so the command line reads as the difference from the defaults.
* **The command line is shown.** The trainer must not become the only way to
  start a run — you can read it off the dialog and type it into a terminal.
* **Resuming is a picker, not a field.** ``--resume`` takes a path to a file
  that already exists, so a text box would only be a way to mistype one. It is
  also the first question about a run rather than a twenty-first — start over,
  or carry on? — so it sits above the parameters instead of in with them.
* **Continuing a run opens on that run's own settings.** They arrive as
  ``initial`` (from its ``config.json``, via :mod:`missile_defense.runs.runconfig`) and are poured
  in the way a preset is. Retyping them from memory was the step that used to
  cost a run: the trainer rejects a resume whose architecture, hidden size or
  annealing schedule disagrees with the checkpoint, and quietly accepts a
  different rollout length as a different experiment under the same run's name.
  Those four are now *disabled* while a checkpoint is selected
  (:data:`LOCKED_ON_RESUME`), so a rejection that used to arrive after Start is
  a control that visibly is not yours to move.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from PySide6.QtCore import QEvent, QLocale, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QDoubleValidator, QPainter
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
    QSlider,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..runs import footprint, presets
from ..runs.sources import Checkpoint, human_age, human_size
from . import params as params_module
from . import reward as reward_module
from . import theme
from .params import GLOSSARY, GROUPS, Group, Param

#: Wide enough for a label, a slider with room to aim, and its readout.
DIALOG_WIDTH = 680
#: The numeric box beside every slider. Fixed, so the sliders in a panel all
#: start and end at the same x and can be compared by eye.
READOUT_WIDTH = 92
#: What the status strip says when nothing is under the pointer or the caret.
HELP_IDLE = "Point at a parameter, or tab to it, for what it does."

#: The two headings on the Objective panel, and the one word each that changes
#: how a run may be read. Both sentences come from `missile_defense.ui.reward`,
#: which is where the distinction is defined — this is a lookup, not a restating.
REWARD_CAPTIONS: dict[str, tuple[str, str, str, str]] = {
    "Potential terms": (
        "Potential terms",
        "cannot change the optimum",
        "tag-invariant",
        reward_module.INVARIANT,
    ),
    "Priced events": (
        "Priced events",
        "changes the objective",
        "tag-objective",
        reward_module.OBJECTIVE,
    ),
}


def _as_number(text: str) -> float:
    """``text`` as a weight, or 0.0 — an unparseable box contributes nothing."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


#: What a resume cannot change, because the checkpoint already decided it.
#:
#: The first three the trainer *rejects* outright — a state dict of a different
#: shape will not load, and the error names two tensors rather than the setting.
#: `steps` it accepts, silently, as a different experiment written into the same
#: run's directory, which is the worse of the two failures. Both used to happen
#: after Start.
LOCKED_ON_RESUME = ("architecture", "hidden", "schedule_updates", "steps")

RESUME_LOCK_HELP = (
    "Fixed by the checkpoint you are continuing from. The trainer would reject a "
    "resume that disagreed — or, for the rollout length, quietly treat it as a "
    "different experiment under this run's name."
)

#: The two events the status strip listens for. Spelled once, because
#: `QEvent.Type.Enter` is a mouthful at every use and misspelling one silently
#: means a strip that never updates.
_ENTER = QEvent.Type.Enter
_FOCUS_IN = QEvent.Type.FocusIn
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


#: Positions on every slider. Fine enough that dragging feels continuous, coarse
#: enough that the readout beside it does not flicker through digits nobody asked
#: for.
SLIDER_STEPS = 1000


class BalanceBar(QWidget):
    """The three potential weights drawn as the proportions they actually are.

    What a battery is worth *relative to* a city is the decision; the absolute
    numbers are a scale nobody reads. Three spin boxes showing 200, 100 and 5
    say that only to someone who divides them, and the ratio is the thing that
    changes what the agent protects — so it is drawn.

    The game's own colours, so the bar reads as the board: batteries amber
    (the HUD), cities blue (the cities), ammunition the interceptor's paler
    blue.
    """

    #: In the order the potential is written, with the colour each is drawn in.
    TERMS = (
        ("base_weight", "batteries", theme.AMBER),
        ("city_weight", "cities", theme.CITY),
        ("ammo_weight", "ammo", theme.INTERCEPTOR),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._weights: dict[str, float] = {}
        self.setMinimumHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def show_weights(self, weights: Mapping[str, float]) -> None:
        if dict(weights) == self._weights:
            return
        self._weights = dict(weights)
        # Set here rather than while painting. A tooltip is state, and one
        # assigned in `paintEvent` does not exist until the widget has been
        # drawn — so it was empty for exactly as long as nobody had looked at
        # the bar, which is the moment somebody would ask it what it means.
        total = sum(max(value, 0.0) for value in self._weights.values())
        if total <= 0.0:
            self.setToolTip("Every potential weight is zero: the shaped term pays nothing.")
        else:
            self.setToolTip(
                "  ·  ".join(
                    f"{label} {100 * max(self._weights.get(name, 0.0), 0.0) / total:.0f}%"
                    for name, label, _ in self.TERMS
                )
            )
        self.update()

    def paintEvent(self, event: object) -> None:  # noqa: N802 — Qt's spelling
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(QColor(theme.EDGE))
        painter.setBrush(QColor(theme.NIGHT))
        painter.drawRoundedRect(rect, 4, 4)

        total = sum(max(self._weights.get(name, 0.0), 0.0) for name, _, _ in self.TERMS)
        if total <= 0.0:
            painter.setPen(QColor(theme.MUTED))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "every weight is zero")
            return

        painter.setPen(Qt.PenStyle.NoPen)
        x = float(rect.left() + 1)
        inner = rect.width() - 2
        for name, label, colour in self.TERMS:
            share = max(self._weights.get(name, 0.0), 0.0) / total
            width = share * inner
            if width <= 0.0:
                continue
            slice_rect = QRectF(x, rect.top() + 1, width, rect.height() - 1)
            painter.setBrush(QColor(colour))
            painter.drawRect(slice_rect)
            # Only where the label fits: a clipped word is worse than no word,
            # and the tooltip carries the full reading anyway.
            if width > 58:
                painter.setPen(QColor(theme.NIGHT))
                painter.drawText(slice_rect, Qt.AlignmentFlag.AlignCenter, label)
                painter.setPen(Qt.PenStyle.NoPen)
            x += width


class ValueSlider(QWidget):
    """A slider and a typeable readout for one bounded number.

    The slider is for *reaching* a value — position carries meaning across a
    known range, which twenty identical spin boxes never did. The readout is for
    *stating* one, because a training parameter is often a number you were told
    rather than one you feel for, and `3e-4` has to remain typeable.

    Three scales, because one does not fit (`missile_defense.ui.params.SCALE`):

    * **linear** for a ratio in [0, 1] — gamma, clip, the entropy bonus.
    * **decade** for a range spanning orders of magnitude. `learning_rate` runs
      1e-8 to 1; linear, everything below 0.001 shares the first pixel.
    * **log** for a range that starts at a real zero, which has no logarithm.
      The reward weights run 0 to 10,000 and `ammo_weight` defaults to 5.0.

    Qt's slider is integer-only, so all three are a mapping over
    :data:`SLIDER_STEPS` positions rather than a widget property.
    """

    changed = Signal()

    def __init__(self, field: Param, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._field = field
        bounds = field.bounds
        # Guarded rather than assumed: `read_params` reads whatever the trainer
        # has, and a field that gained no bound yet must still get an editor.
        self._low, self._high = bounds if bounds is not None else (0.0, 1.0)
        self._scale = field.scale
        self._int = field.kind == "int"

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, SLIDER_STEPS)
        self._slider.setAccessibleName(field.name.replace("_", " "))
        row.addWidget(self._slider, stretch=1)

        self._readout = QLineEdit()
        self._readout.setFixedWidth(READOUT_WIDTH)
        self._readout.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._readout.setProperty("role", "readout")
        validator = QDoubleValidator(self._low, self._high, 12)
        # Scientific notation on purpose: 3e-4 is how a learning rate is written
        # and read, where 0.000300 is neither.
        validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        self._readout.setValidator(validator)
        row.addWidget(self._readout)

        self.set_value(field.default)
        self._slider.valueChanged.connect(self._from_slider)
        self._readout.editingFinished.connect(self._from_readout)

    # ---- the three mappings -------------------------------------------------
    def _to_position(self, value: float) -> int:
        low, high = self._low, self._high
        if self._scale == "decade":
            lo, hi = math.log10(max(low, 1e-12)), math.log10(max(high, 1e-12))
            if value <= 0.0:
                return 0
            fraction = (math.log10(value) - lo) / (hi - lo) if hi > lo else 0.0
        elif self._scale == "log":
            span = math.log10(1.0 + high - low)
            fraction = math.log10(1.0 + max(value - low, 0.0)) / span if span > 0 else 0.0
        else:
            fraction = (value - low) / (high - low) if high > low else 0.0
        return int(round(min(max(fraction, 0.0), 1.0) * SLIDER_STEPS))

    def _from_position(self, position: int) -> float:
        fraction = position / SLIDER_STEPS
        low, high = self._low, self._high
        if self._scale == "decade":
            lo, hi = math.log10(max(low, 1e-12)), math.log10(max(high, 1e-12))
            value = 10.0 ** (lo + fraction * (hi - lo))
            return float(f"{value:.2g}")  # a learning rate is two digits, not sixteen
        if self._scale == "log":
            value = low + (10.0 ** (fraction * math.log10(1.0 + high - low)) - 1.0)
        else:
            value = low + fraction * (high - low)
        if self._int:
            return float(round(value))
        return float(f"{value:.4g}")

    # ---- the editor protocol ------------------------------------------------
    def value(self) -> str:
        return self._readout.text().strip()

    def set_value(self, text: str) -> None:
        """Show ``text``, or leave the control alone when it is not a number.

        Dropped rather than forced, exactly as the other editors do: a preset is
        a file somebody may have hand-edited, and turning a typo into a number is
        worse than ignoring it. The command-line preview always shows what will
        actually be used.
        """
        try:
            number = float(text)
        except (TypeError, ValueError):
            return
        number = min(max(number, self._low), self._high)
        self._show(number)
        # Announced like any other change. Pouring a preset in is not a person
        # editing, but that distinction is the dialog's to make and it already
        # makes it: `_touched` returns early while `_applying` is set. Staying
        # silent here instead would mean a value written from outside those
        # guards never reached the preview — which is a form showing one thing
        # and a command line saying another.
        self.changed.emit()

    def _show(self, number: float) -> None:
        blocked = self._slider.blockSignals(True)
        self._slider.setValue(self._to_position(number))
        self._slider.blockSignals(blocked)
        self._readout.setText(_number_text(number, self._int))

    def _from_slider(self, position: int) -> None:
        self._readout.setText(_number_text(self._from_position(position), self._int))
        self.changed.emit()

    def _from_readout(self) -> None:
        try:
            number = float(self._readout.text())
        except ValueError:
            return
        self._show(min(max(number, self._low), self._high))
        self.changed.emit()

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 — Qt's spelling
        super().setEnabled(enabled)
        self._slider.setEnabled(enabled)
        self._readout.setEnabled(enabled)


def _number_text(number: float, is_int: bool) -> str:
    """A number as a person writes it: no trailing zeros, exponent when tiny."""
    if is_int:
        return str(int(round(number)))
    if number != 0.0 and (abs(number) < 1e-3 or abs(number) >= 1e6):
        return f"{number:.2g}"
    return f"{number:g}"


def _read(editor: QWidget) -> str:
    """One string out of whichever editor a field was given.

    One place, so adding an editor type cannot leave `values()` silently
    returning "" for it — which would look like "unchanged" and quietly drop the
    setting.
    """
    if isinstance(editor, ValueSlider):
        return editor.value()
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
    if isinstance(editor, ValueSlider):
        editor.set_value(value)
        return
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
    read. `missile_defense.ui.params` follows a named constant to its value, so the usual
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
        #: Keyed by `Param.key` — owner *and* name. `gamma` is a field of two
        #: config classes, and keying by name alone gave them one editor between
        #: them, so only the last one read. `Shaping.gamma` is derived now and has
        #: no editor at all, but the key stays honest about what identifies a field.
        self._editors: dict[tuple[str, str], QWidget] = {}
        #: What each editor showed when it was built. `values()` compares
        #: against this rather than the dataclass's source text, because the two
        #: differ for optional fields — see the note there.
        self._initial: dict[tuple[str, str], str] = {}
        #: Every row, so the filter can hide one and the folds can be told which
        #: of their children matched.
        self._rows: dict[tuple[str, str], QWidget] = {}
        #: Each fold, with the fields it holds, so a closed one can show its
        #: values and say how many of them you have changed.
        self._folds: list[tuple[Group, QToolButton, QWidget, QLabel]] = []
        #: Which widget explains which field, for the status strip. Keyed by the
        #: widget because that is what an event arrives on.
        self._explains: dict[QWidget, tuple[Param, str]] = {}
        self._labels: dict[tuple[str, str], QLabel] = {}
        #: The Objective panel's three extras. `None` where there is no reward
        #: group to draw them for — an installed trainer with no trainer source
        #: beside it has no fields at all.
        self._balance: BalanceBar | None = None
        self._equation: QLabel | None = None
        self._equation_note: QLabel | None = None
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
            # An installed trainer watching a synced directory has no missile_defense.training
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

        if fields:
            layout.addWidget(self._find_row())
            layout.addWidget(self._tabs(fields), stretch=1)
            # One place the dialog explains itself, for parameters and for
            # jargon alike. A strip rather than a line under every row: thirty-odd
            # parameters carrying thirty-odd paragraphs is most of the height of
            # the dialog, and the reasoning is only ever wanted one at a time.
            layout.addWidget(self._help_strip())

        self._preview = QLabel()
        self._preview.setProperty("role", "preview")
        self._preview.setWordWrap(True)
        self._preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._preview)

        # What the run will ask of the card, under the command that asks for it.
        # Only where there are fields to read it from — with no trainer source
        # there is nothing to estimate from and nothing to change if it is wrong.
        self._memory: QLabel | None = None
        self._schedule: QLabel | None = None
        if fields:
            self._schedule = QLabel()
            self._schedule.setProperty("role", "note")
            layout.addWidget(self._schedule)
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
                editor = self._editors.get(field.key)
                if editor is None:
                    continue
                # Fields the preset does not mention go back to the trainer's
                # own default, so what is on screen is the preset and not the
                # preset over the residue of whatever was picked before it.
                _write(editor, str(preset.options.get(field.name, self._initial[field.key])))
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

    # ---- the three panels ---------------------------------------------------
    def _find_row(self) -> QWidget:
        """Reveal a parameter wherever it lives — across folds and across tabs.

        This is what makes folding safe. Grouping thirty-odd parameters into
        named sections is only an improvement if you never have to *remember*
        which section something is in, and a search box answers that better than
        any arrangement can.
        """
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(8)

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Find a parameter…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        box.addWidget(self._filter, stretch=1)

        glossary = QPushButton("Glossary")
        glossary.setToolTip("What the abbreviations on this dialog mean")
        glossary.clicked.connect(self._show_glossary)
        box.addWidget(glossary)
        return row

    def _tabs(self, fields: list[Param]) -> QWidget:
        """One tab per decision, not per config class.

        Somebody starting a run chooses between *what it is paid for*, *how it
        learns* and *how big and how long* — and only afterwards cares that those
        happen to be `Shaping`, `PPOConfig` and `TrainConfig`.
        """
        by_name = {field.name: field for field in fields if not field.derived}
        tabs = QTabWidget()
        #: Which domain each tab shows, so the filter can bring you to a match
        #: that is on a tab you are not looking at.
        self._tab_domains: list[str] = []
        for domain, title, subtitle in params_module.DOMAINS:
            groups = [g for g in GROUPS if g.domain == domain]
            page = self._panel(groups, by_name)
            if page is not None:
                tabs.addTab(page, title)
                tabs.setTabToolTip(tabs.count() - 1, subtitle)
                self._tab_domains.append(domain)
        self._tabs_widget = tabs
        return tabs

    def _panel(self, groups: list[Group], by_name: dict[str, Param]) -> QWidget | None:
        """One tab: its essential groups open, the rest folded."""
        present = [g for g in groups if any(name in by_name for name in g.fields)]
        if not present:
            return None

        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(2, 8, 2, 2)
        box.setSpacing(8)
        for group in present:
            members = [by_name[name] for name in group.fields if name in by_name]
            # The Objective panel is the one that says what the numbers *mean*.
            # Three weights whose ratio decides what the agent protects, and two
            # terms that are not optimality-neutral where the others provably
            # are — a distinction `missile_defense.ui.reward` has always drawn
            # for a *finished* run and nothing drew for the person choosing them.
            caption = REWARD_CAPTIONS.get(group.name)
            if caption is not None:
                box.addWidget(self._caption(*caption))
            if group.essential:
                box.addWidget(self._rows_frame(members))
            else:
                box.addWidget(self._fold(group, members))
            if group.name == "Potential terms":
                self._balance = BalanceBar()
                box.addWidget(self._balance)
        if any(g.domain == "reward" for g in present):
            box.addWidget(self._equation_panel())
        box.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        return scroll

    def _caption(self, title: str, tag: str, role: str, why: str) -> QWidget:
        """A group heading with the one word that changes how a run is read."""
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(2, 4, 2, 0)
        box.setSpacing(8)
        label = QLabel(title.upper())
        label.setProperty("role", "caption")
        box.addWidget(label)
        chip = QLabel(tag)
        chip.setProperty("role", role)
        chip.setToolTip(why)
        box.addWidget(chip)
        box.addStretch(1)
        for widget in (row, label, chip):
            widget.setToolTip(why)
        return row

    def _equation_panel(self) -> QWidget:
        """The reward as its formula, with the run's own numbers in it.

        Seven boxes are faithful and nearly useless: they do not say that three
        of them are summed into a potential, that one discounts that potential
        rather than the return, or that two are switched off. The equation says
        all of it in two lines, and `missile_defense.ui.reward` already knew how
        to write it — for a finished run, where nobody was choosing anything.
        """
        frame = QFrame()
        frame.setProperty("role", "panel")
        box = QVBoxLayout(frame)
        box.setContentsMargins(14, 10, 14, 10)
        box.setSpacing(6)
        caption = QLabel("WHAT THE AGENT IS PAID")
        caption.setProperty("role", "caption")
        box.addWidget(caption)
        self._equation = QLabel()
        self._equation.setProperty("role", "formula")
        self._equation.setWordWrap(True)
        self._equation.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.addWidget(self._equation)
        self._equation_note = QLabel()
        self._equation_note.setProperty("role", "note")
        self._equation_note.setWordWrap(True)
        box.addWidget(self._equation_note)
        return frame

    def _refresh_reward(self) -> None:
        """Redraw the equation and the balance bar from what is on screen."""
        if self._equation is None:
            return
        shown = {
            name: _read(editor)
            for (owner, name), editor in self._editors.items()
            if owner == "Shaping"
        }
        # The discount is derived rather than offered, so it is not among the
        # editors — but it is part of the formula, and reading it from the
        # control that owns it is what keeps the two in step.
        shown["gamma"] = self._text_named("gamma", "")
        formula = reward_module.formula_being_configured(shown)
        if formula is None:
            self._equation.setText("")
            self._equation_note.setText("")
            return
        lines = [formula.phi, formula.total] if formula.shaped else [formula.total]
        self._equation.setText("\n".join(line for line in lines if line))
        self._equation_note.setText("  ".join(formula.notes))
        if self._balance is not None:
            self._balance.show_weights(
                {name: _as_number(shown.get(name, "")) for name, _, _ in BalanceBar.TERMS}
            )

    def _rows_frame(self, fields: list[Param]) -> QWidget:
        frame = QFrame()
        frame.setProperty("role", "panel")
        form = QFormLayout(frame)
        form.setContentsMargins(14, 10, 14, 10)
        form.setSpacing(6)
        for field in fields:
            label, editor = self._row(field)
            form.addRow(label, editor)
        return frame

    def _fold(self, group: Group, fields: list[Param]) -> QWidget:
        """A named section that folds, and still shows its values when closed.

        The digest on the summary line is the point. A fold that hides what it
        holds is the same wall of boxes as before, only quieter — so a closed
        group reads `trail 0.84 · delay 3` and gains a count when you change one
        of its values.
        """
        container = QWidget()
        box = QVBoxLayout(container)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)

        head = QWidget()
        head_box = QHBoxLayout(head)
        head_box.setContentsMargins(0, 0, 0, 0)
        head_box.setSpacing(8)

        toggle = QToolButton()
        toggle.setText(group.name)
        toggle.setCheckable(True)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.ArrowType.RightArrow)
        toggle.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        head_box.addWidget(toggle)

        digest = QLabel()
        digest.setProperty("role", "stat")
        digest.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        head_box.addWidget(digest, stretch=1)
        box.addWidget(head)

        body = self._rows_frame(fields)
        body.setVisible(False)
        box.addWidget(body)

        def reveal(shown: bool) -> None:
            body.setVisible(shown)
            toggle.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)

        toggle.toggled.connect(reveal)
        self._folds.append((group, toggle, body, digest))
        return container

    def _row(self, field: Param) -> tuple[QLabel, QWidget]:
        """One parameter: its name, its editor, and its reasoning on demand."""
        label = QLabel(field.name.replace("_", " "))
        editor = self._editor(field)
        reason = field.help or f"{field.owner}.{field.name}"
        for widget in (label, editor):
            widget.setToolTip(reason)
        self._editors[field.key] = editor
        self._initial[field.key] = _read(editor)
        self._rows[field.key] = editor
        self._explains[editor] = (field, reason)
        label.installEventFilter(self)
        editor.installEventFilter(self)
        self._labels[field.key] = label
        return label, editor

    def _reveal_tab_holding(self, matched: set[tuple[str, str]]) -> None:
        """Switch to the first tab that holds a match, unless one already does."""
        domains = {
            group.domain
            for key in matched
            if (group := params_module.group_of(key[1], key[0])) is not None
        }
        if not domains:
            return
        current = self._tabs_widget.currentIndex()
        if 0 <= current < len(self._tab_domains) and self._tab_domains[current] in domains:
            return  # what you are looking at already answers the query
        for index, domain in enumerate(self._tab_domains):
            if domain in domains:
                self._tabs_widget.setCurrentIndex(index)
                return

    def _help_strip(self) -> QWidget:
        frame = QFrame()
        frame.setProperty("role", "panel")
        box = QHBoxLayout(frame)
        box.setContentsMargins(12, 8, 12, 8)
        box.setSpacing(10)
        self._help_key = QLabel("—")
        self._help_key.setProperty("role", "stat")
        self._help_key.setAlignment(Qt.AlignmentFlag.AlignTop)
        box.addWidget(self._help_key)
        self._help_text = QLabel(HELP_IDLE)
        self._help_text.setProperty("role", "note")
        self._help_text.setWordWrap(True)
        self._help_text.setMinimumHeight(34)
        box.addWidget(self._help_text, stretch=1)
        return frame

    def eventFilter(self, watched: object, event: object) -> bool:  # noqa: N802 — Qt's spelling
        """Feed the status strip from whatever is pointed at or focused.

        Focus as well as the pointer, deliberately: a strip only the mouse can
        reach is a tooltip with extra steps, and the whole reason the reasoning
        moved here was to be readable without hovering thirty-odd rows.
        """
        kind = getattr(event, "type", lambda: None)()
        if kind in (_ENTER, _FOCUS_IN):
            found = self._explains.get(watched)  # type: ignore[arg-type]
            if found is None:
                # A label shares its row's explanation; find it by position.
                for key, label in self._labels.items():
                    if label is watched:
                        found = self._explains.get(self._rows[key])
                        break
            if found is not None:
                self._explain(*found)
        return super().eventFilter(watched, event)  # type: ignore[arg-type]

    def _explain(self, field: Param, reason: str) -> None:
        default = field.default or "auto"
        self._help_key.setText(f"{field.flag} · default {default}")
        self._help_text.setText(reason)

    def _show_glossary(self) -> None:
        """Every abbreviation the dialog uses, in the game's terms.

        A dialog rather than a hover: the terms are worth reading through once,
        and a definition only reachable by pointing at the right word is one most
        people never find.
        """
        body = "\n\n".join(f"{term}\n    {text}" for term, text in GLOSSARY.items())
        box = QMessageBox(self)
        box.setWindowTitle("Glossary")
        box.setText("What the words on this dialog mean.")
        box.setDetailedText(body)
        box.exec()

    def _apply_filter(self, query: str) -> None:
        """Show only what matches, opening any fold that holds a match."""
        text = query.strip().lower()
        matched: set[tuple[str, str]] = set()
        for key, editor in self._rows.items():
            name = key[1]
            hit = (
                not text
                or text in name.lower()
                or any(text in flag.lower() for flag in params_module.flags_for(name))
            )
            if hit:
                matched.add(key)
            editor.setVisible(hit)
            label = self._labels.get(key)
            if label is not None:
                label.setVisible(hit)
        for group, toggle, _, _ in self._folds:
            # From the set, not from `isVisible()`: a row inside a closed fold
            # reports invisible however it was just set, because Qt answers for
            # the whole ancestor chain. Asking the widget produced a filter that
            # could never open the fold holding the match.
            holds = any(key in matched for key in self._rows if key[1] in group.fields)
            toggle.setVisible(holds or not text)
            # `reveal` follows the toggle, so the body needs no separate handling
            # — and doing it here as well is how the two disagreed.
            toggle.setChecked(bool(text) and holds)
        # And come to the match rather than waiting to be found. Searching for a
        # parameter and being shown nothing, because it lives on a tab you are
        # not looking at, is the hunt this box exists to end.
        if text and matched:
            self._reveal_tab_holding(matched)

    def _editor(self, field: Param) -> QWidget:
        """The narrowest control the field's type allows.

        Narrow on purpose. A run is hours, and a value that cannot be entered
        costs nothing where a value that can be mistyped costs the run: a
        dropdown cannot be misspelled `entty`, and a spin box that stops at 1
        cannot be given `--envs 0`. The bounds in `missile_defense.ui.params.BOUNDS` are the
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
        # A bounded number gets a slider and a readout: position says where in
        # the range you are, which twenty identical boxes never did, and the box
        # beside it keeps `3e-4` typeable. Only where a bound exists — a slider
        # over an unknown range is a control with no meaning.
        if field.bounds is not None and field.kind in ("int", "float"):
            try:
                float(field.default)
            except ValueError:
                pass  # a default this trainer could not resolve; fall through
            else:
                slider = ValueSlider(field)
                slider.changed.connect(self._edited)
                return slider
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
            # `int(...)`, which raised. `missile_defense.ui.params` follows a named default to
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
                editor = self._editor_named(name)
                if editor is not None:
                    _write(editor, value)
        finally:
            self._applying = False

    def _editor_named(self, name: str) -> QWidget | None:
        """The editor for a field, found by name alone.

        Presets and a run's stored `config.json` speak names, not owners. Since
        the one name shared by two classes is derived and has a single editor,
        the first match is unambiguous.
        """
        for (_, field_name), editor in self._editors.items():
            if field_name == name:
                return editor
        return None

    # ---- what came out of it ------------------------------------------------
    def values(self) -> dict[str, str]:
        """Only the fields whose value differs from the trainer's default."""
        changed: dict[str, str] = {}
        for field in self._fields:
            editor = self._editors.get(field.key)
            if editor is None:
                continue  # derived, like `Shaping.gamma`, or not offered here
            value = _read(editor)
            # Against what the editor was *built* showing, not against the
            # dataclass's source text. They differ for an optional field: a
            # `int | None = None` gets a spin box sitting on 0, and comparing
            # "0" with "" made the dialog emit `--schedule-updates 0` for a
            # field nobody had touched.
            if value and value != self._initial.get(field.key, ""):
                # Keyed by *name*, because that is what `command_line` and the
                # preset file speak, and because the one name owned by two
                # classes — `gamma` — now has exactly one editor between them.
                # `flags_for` writes it to both flags.
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

    def _refresh_folds(self) -> None:
        """Each closed fold shows its values, and says how many you changed.

        Without this a fold hides what it holds, which is the wall of boxes again
        with a button in front of it.
        """
        changed = self.values()
        for group, toggle, _, digest in self._folds:
            present = [name for name in group.fields if self._editor_named(name) is not None]
            touched = sum(1 for name in present if name in changed)
            digest.setText(
                " · ".join(
                    f"{name.replace('_', ' ')} {self._text_named(name, '')}" or name
                    for name in present[:3]
                )
            )
            toggle.setText(f"{group.name}  ({touched} changed)" if touched else group.name)

    def _refresh_resume_locks(self) -> None:
        """Fields a resume cannot change are locked to the checkpoint's values.

        The trainer *rejects* a resume whose architecture, hidden size or
        annealing horizon disagrees with the checkpoint, and quietly accepts a
        different rollout length as a different experiment under the same run's
        name. Both of those land after Start has been pressed. Disabling them
        while a checkpoint is selected turns a late rejection into a control that
        visibly is not yours to move.
        """
        continuing = self.resume() is not None
        for name in LOCKED_ON_RESUME:
            editor = self._editor_named(name)
            if editor is None:
                continue
            editor.setEnabled(not continuing)
            editor.setToolTip(
                RESUME_LOCK_HELP if continuing else self._explains.get(editor, (None, ""))[1]
            )

    def _refresh_shaping(self) -> None:
        """Shaping off greys every term it governs, because it ignores them all.

        `VecEnv.step` nests the whole block — the potential *and* both priced
        events — inside `if self._shaping.enabled`, so an unshaped run is paid
        the game score and nothing else. Leaving seven live controls under a
        switch that discards them is the form claiming they still matter.
        """
        editor = self._editor_named("enabled")
        if editor is None:
            return
        on = _read(editor) == "True"
        for (owner, name), row in self._rows.items():
            if owner != "Shaping" or name == "enabled":
                continue
            row.setEnabled(on)
            label = self._labels.get((owner, name))
            if label is not None:
                label.setEnabled(on)
        if self._balance is not None:
            self._balance.setEnabled(on)

    def _refresh_preview(self) -> None:
        self._preview.setText(" ".join(self.command()))
        self._refresh_folds()
        self._refresh_shaping()
        self._refresh_reward()
        self._refresh_resume_locks()
        self._refresh_schedule()
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
        editor = self._editor_named(name)
        if editor is None:
            return fallback
        try:
            return int(float(_read(editor)))
        except ValueError:
            return fallback

    def _text_named(self, name: str, fallback: str) -> str:
        editor = self._editor_named(name)
        return _read(editor) if editor is not None else fallback

    def _refresh_schedule(self) -> None:
        """What the cadence settings actually produce, over this many updates.

        `updates` interacts with three separate intervals and nothing showed the
        result, so "evaluate every 10" was a number with no consequence attached.
        Stated as a count rather than as a warning: the arithmetic is the point,
        and the reader can see for themselves that 2,000 updates at
        `checkpoint_every 100` is twenty checkpoints.
        """
        if self._schedule is None:
            return
        updates = self._number("updates", 0)
        if updates <= 0:
            self._schedule.setText("")
            return

        def count(name: str) -> int:
            every = self._number(name, 0)
            return 0 if every <= 0 else updates // every

        parts = [
            f"{count('eval_every')} evals",
            f"{count('record_every')} recordings",
            f"{count('checkpoint_every')} checkpoints",
        ]
        self._schedule.setText(f"{updates:,} updates → " + " · ".join(parts))

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
            "architecture": self._text_named("architecture", "mlp"),
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
