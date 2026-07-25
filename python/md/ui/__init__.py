# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The training console — a window onto a run, never a trainer.

Launch it with ``poe ui``. It attaches to whatever a run has already written to
``runs/``: the curves come from ``metrics.csv``, the yardstick from ``evals.csv``,
and the episode list from the ``.mdr`` files beside them. Nothing here steps a
simulation, loads a checkpoint or imports torch — a test enforces that last one,
because a console that quietly grew a trainer inside it would take the run down
with the window (docs/ROADMAP.md, M8).

The submodules split on a line worth keeping:

* :mod:`md.ui.sources` and :mod:`md.ui.runner` are **Qt-free**, so pytest drives
  them with no display.
* :mod:`md.ui.charts` and :mod:`md.ui.app` are the widgets, and :mod:`md.ui.theme`
  is the palette they wear — lifted from the game, so the console looks like the
  same product seen from the other side.

Importing this package pulls in none of them: an ``import md.ui`` must stay cheap
and Qt-less.
"""

__all__: list[str] = []
