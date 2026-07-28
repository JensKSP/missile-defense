# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Runs on disk: where they live, what they wrote, and how to talk to one.

No torch and no Qt. That is not an accident of what happens to be in here — it
is what lets the trainer window attach to a directory synced from another
machine, browse it, plot it and replay it on a laptop that could never have
produced it.

The artifacts *are* the interface (docs/ROADMAP.md, M8). Nothing in this layer
calls into a running trainer; it reads ``metrics.csv``, ``evals.csv``,
``train.log`` and the ``.mdr`` recordings, and it asks a run to stop by writing a
file (:mod:`~missile_defense.runs.control`) rather than by signalling a process.

:mod:`~missile_defense.runs.sources` and :mod:`~missile_defense.runs.runner` used
to live under ``ui/`` despite being Qt-free, and ``archive`` and ``library``
imported them from there — a library layer reaching up into the window layer.
Nothing failed, because the modules were importable either way; it just meant
``ui/`` could not be a boundary. Here they are below the window, where the things
that use them already were.
"""

__all__: list[str] = []
