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

**It is the Start dialog, read-only.** That is the whole design: "what was this
trained with?" and "what shall I train with?" are the same thirty-odd values
under the same headings, and this used to answer the first with a three-column
table while the other answered the second with a form — so the reader translated
between two layouts for one run. Same tabs, same folds, sliders sitting where the
run left them, the same reward equation with that run's own numbers in it, and
nothing that can be moved.

Two things it adds over a formatted dump of the file:

* **What this run *changed*.** Thirty numbers are unreadable. The few that differ
  from the trainer's own defaults are the run, and they are counted above
  (:func:`missile_defense.ui.params.settings_of` does that arithmetic, Qt-free).
* **Settings this trainer has never heard of are still shown.** A run from a
  newer training loop carries knobs with no control here; they are listed rather
  than dropped, because answering "what was this trained with?" with "the part I
  recognise" is not answering it.

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
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..runs import runconfig
from .forms import ParameterDialog
from .params import TRAINER_SOURCES, Param, Setting, read_params, settings_of

#: A run directory with no `config.json`. Names the file, because a run that
#: began before the trainer wrote one has everything else and only this missing —
#: which is a different situation from a directory that holds no run.
NOTHING_STORED = (
    "No config.json in this run.\n"
    "The trainer writes one at start-up, so a run that began before it existed\n"
    "has none — and one somebody synced without it keeps its curves regardless."
)


def _summary(config: runconfig.RunConfig | None, settings: Sequence[Setting]) -> str:
    """Above the parameters: what this run changed, then where it was read from.

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
        self.resize(720, 760)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)

        # What this run changed, whether it was continued, and where the file was
        # read from. The parameters below show every value; this says which of
        # them are *this run's* rather than the trainer's, which is the question
        # somebody opens the dialog with.
        summary = _summary(config, settings)
        if summary:
            note = QLabel(summary)
            note.setProperty("role", "note")
            note.setWordWrap(True)
            layout.addWidget(note)

        #: What this dialog is showing, as `missile_defense.ui.params` read it.
        #: Kept beside the widgets because "which values did this run use" is a
        #: question worth answering without walking a widget tree — the tests
        #: ask it, and so did the table this replaced.
        self.settings: tuple[Setting, ...] = tuple(settings)

        # The run screen's own dialog, read-only. "What was this trained with?"
        # and "what shall I train with?" are the same values under the same
        # headings, and answering them in two layouts — a form there, a
        # three-column table here — made the reader translate between them. Same
        # tabs, same folds, sliders sitting at the values the run used, the same
        # equation with that run's numbers in it. Nothing editable.
        fields = read_params(TRAINER_SOURCES)
        self.parameters = ParameterDialog(
            fields,
            python="python",
            # The run's own directory, so the command line underneath is the one
            # that produced *this* run and can be pasted to reproduce it. An
            # empty path made it read `--out-dir .`, which is nowhere.
            out_dir=config.path.parent if config is not None else Path(),
            initial={setting.name: setting.value for setting in settings},
            read_only=True,
            title=f"Parameters — {name}",
            embedded=True,
            parent=self,
        )
        # Embedded rather than opened: it is a `QDialog`, but here it is the
        # body of this one. `setWindowFlags(Widget)` is what lets a dialog be a
        # child widget without trying to be a window of its own.
        self.parameters.setWindowFlags(Qt.WindowType.Widget)
        layout.addWidget(self.parameters, stretch=1)

        # Anything the dialog has no control for is still a value this run was
        # trained with, and a trainer that showed only the part it recognised
        # would be answering the question dishonestly. A newer training loop's
        # knobs, and `schedule`'s bookkeeping, land here.
        unknown = [s for s in settings if not _has_field(s, fields)]
        if unknown:
            also = QLabel(
                "Also recorded, with no control on this trainer:  "
                + " · ".join(f"{s.name} {s.value or runconfig.NOTHING}" for s in unknown)
            )
            also.setProperty("role", "note")
            also.setWordWrap(True)
            layout.addWidget(also)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


def _has_field(setting: Setting, fields: Sequence[Param]) -> bool:
    """Whether the parameter dialog has a control that shows this setting.

    By name alone, because that is what the dialog's editors are keyed by from
    the outside — and the one name owned by two config classes is derived, so it
    has exactly one control between them.
    """
    return any(field.name == setting.name and not field.derived for field in fields)


def settings_for(run_dir: Path, trainer: Path) -> tuple[runconfig.RunConfig | None, list[Setting]]:
    """Read a run's settings and pair them with the trainer's own reasoning.

    Both halves in one call because both callers want both, and because the
    training loop's source may not be beside this trainer at all — an installed trainer
    watching a synced directory still shows every value, just without the
    tooltips and the comparison against defaults.
    """
    config = runconfig.read(run_dir)
    return config, settings_of(config, read_params(trainer))
