# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The simulation, and the file formats that describe it. numpy only.

    from missile_defense.sim import VecEnv
    env = VecEnv(num_envs=1024)

This is the bottom layer and the one with reuse value on its own: it needs
neither torch nor Qt, so it runs on a headless box that has nothing but numpy.
Everything above it — :mod:`missile_defense.runs`, :mod:`missile_defense.training`,
:mod:`missile_defense.ui` — may import from here, and nothing here may import
from them. That direction is the whole point of the split, and it is checkable by
reading the imports rather than by trusting a convention.

:mod:`~missile_defense.sim.export_policy` sits here rather than under `training`
even though it reads a PyTorch checkpoint, because what it *is* is the writer for
the format :mod:`~missile_defense.sim.policy_format` reads. Its torch import is
deferred into the one function that converts, so importing this package still
costs nothing on a machine that has no torch.

``VecEnv`` is re-exported because it is what almost every caller wants and
``missile_defense.sim.env.VecEnv`` is a mouthful. Nothing else is: reaching for
the evaluation protocol or the policy format is a deliberate act and should read
like one.
"""

from .env import VecEnv

__all__ = ["VecEnv"]
