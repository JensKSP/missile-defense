# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The learning half: PPO, the training loop, and the multi-seed runner.

    python -m missile_defense.training --updates 750

This is the only layer that imports torch, and it is why torch is not a
dependency of anything else: :mod:`missile_defense.sim` and
:mod:`missile_defense.runs` are complete without it, so watching, comparing and
replaying runs works on a machine that has none.

Importing this package does *not* import torch. The entry point in
:mod:`~missile_defense.training.cli` checks for it first and explains its absence
instead of raising, which is the difference between an optional component and a
broken install (docs/PACKAGING.md).

A ``__main__`` rather than a module anyone has to name: ``python -m
missile_defense.training`` is the command, so nothing outside this package ever
has to know whether the loop lives in ``train.py``. The multi-seed runner spawns
its children that way for the same reason.
"""

__all__: list[str] = []
