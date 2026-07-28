# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""missile-defense — the deterministic C++ simulation as an RL environment.

    from missile_defense.sim import VecEnv
    env = VecEnv(num_envs=1024)
    obs = env.reset(0)
    obs, reward, terminated, truncated, info = env.step(actions)

Four layers, and each may only import from the ones below it:

* :mod:`missile_defense.sim` — the simulation and the formats that describe it.
  numpy, and nothing else.
* :mod:`missile_defense.runs` — runs on disk: discovery, artifacts, control, and
  the managed training runtime. Still no torch and no Qt, which is what lets a
  laptop attach to a directory synced off a training box.
* :mod:`missile_defense.training` — PPO and the loop. The only layer that
  imports torch.
* :mod:`missile_defense.ui` — the trainer window. The only layer that imports
  PySide6.

The direction is the point: it is readable off the imports, so a violation shows
up while writing rather than as a package that drags Qt onto a headless machine.

The heavy lifting is in ``_md_native``, a nanobind extension around ``md::core``:
observations are written directly into this package's NumPy buffers and the batch
step releases the GIL, so the C++ worker pool runs in parallel. An ordinary build
produces it and places it beside this package::

    cmake --preset release && cmake --build --preset release

See ``docs/API.md`` for the observation contract, action space and reward spec.
"""

__version__ = "0.9.0"

__all__ = ["__version__"]
