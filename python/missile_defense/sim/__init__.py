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

**Lazily**, and that is not a micro-optimisation. Importing ``env`` imports
``_md_native``, and a trainer without the binding must still start, browse and
replay — it just cannot train (CMakeLists.txt says the same thing where it
refuses to package one). An eager re-export here put the extension behind every
import in this layer: ``runs.sources`` asks for ``sim.benchmark``, which is pure
Python and reads a CSV, and got an ``ImportError`` about a missing ``.so``
instead. The staged-install test caught it; nothing else would have, because a
development checkout always has the binding.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the type checkers need the real thing; the interpreter does not
    from .env import VecEnv

__all__ = ["VecEnv"]


def __getattr__(name: str) -> object:
    """PEP 562: resolve ``VecEnv`` on first use rather than on import."""
    if name == "VecEnv":
        from .env import VecEnv as _VecEnv  # noqa: PLC0415 — the whole point

        return _VecEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
