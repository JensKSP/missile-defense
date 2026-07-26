# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# pyright: reportMissingImports=false
"""What the run is training: the network, its layers, and the policy on disk.

M8 phase 4's other half (docs/ROADMAP.md) — parameter count, layer shapes,
observation and action sizes, which checkpoint is newest and what it scored.

Everything here is read out of ``runs/model.json`` and the ``checkpoints``
directory. Nothing loads a ``.pt``, because loading one needs torch and the
console must never import it; the trainer writes the description instead, which
is the roadmap's own answer to that rule (:mod:`md.modelcard`).

It sits under the recordings list because that is the column with the room, and
because the two answer neighbouring questions: what has this policy *done*, and
what *is* it. Both are read a few times a run rather than watched, so neither is
allowed to take space from the curve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .. import modelcard
from ..modelcard import ModelCard
from .sources import Checkpoint, EvalRow, checkpoint_note

#: No card in the run directory. Says which file, because a run started before
#: the trainer wrote one has everything else and only this missing — and that is
#: a different situation from a run that has not started.
NO_CARD = (
    "No model.json in this run.\n"
    "The trainer writes one at start-up, beside config.json, so a run that "
    "began before it existed has none."
)


class ModelPanel(QWidget):
    """The network, or a line saying why it is not known yet."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)
        caption = QLabel("model")
        caption.setProperty("role", "caption")
        layout.addWidget(caption)

        self._headline = QLabel()
        self._wiring = QLabel()
        self._wiring.setProperty("role", "note")
        self._layers = QLabel()
        self._layers.setProperty("role", "preview")
        self._empty = QLabel(NO_CARD)
        self._empty.setProperty("role", "placeholder")
        self._empty.setWordWrap(True)
        self._note = QLabel()
        self._note.setProperty("role", "note")
        self._note.setWordWrap(True)
        for widget in (self._headline, self._wiring, self._layers, self._empty, self._note):
            layout.addWidget(widget)

        #: What is currently on screen, so an unchanged run is not repainted
        #: every rescan — a panel that flickers once a second is one you learn
        #: to look away from.
        self._shown: tuple[str, ...] | None = None
        self.show_run(None, (), {})

    def show_run(
        self,
        card: ModelCard | None,
        checkpoints: Sequence[Checkpoint],
        evals: Mapping[int, EvalRow],
    ) -> None:
        """Draw ``card`` and the checkpoints beside it. Cheap to call every poll."""
        note = checkpoint_note(checkpoints, evals)
        if card is None:
            self._paint(("", "", "", note))
            return
        self._paint(
            (
                modelcard.headline(card),
                modelcard.wiring(card),
                "\n".join(modelcard.layer_table(card)),
                note,
            )
        )

    def _paint(self, texts: tuple[str, str, str, str]) -> None:
        if texts == self._shown:
            return
        self._shown = texts
        headline, wiring, layers, note = texts
        described = bool(headline)
        self._headline.setText(headline)
        self._wiring.setText(wiring)
        self._layers.setText(layers)
        self._note.setText(note)
        for widget in (self._headline, self._wiring):
            widget.setVisible(described)
        self._layers.setVisible(bool(layers))
        self._empty.setVisible(not described)
