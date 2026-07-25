# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""missile-defense — the deterministic C++ simulation as an RL environment.

    from md.env import VecEnv
    env = VecEnv(num_envs=1024)
    obs = env.reset(0)
    obs, reward, terminated, truncated, info = env.step(actions)

The heavy lifting is in ``_md_native``, a nanobind extension around ``md::core``:
observations are written directly into this package's NumPy buffers and the batch
step releases the GIL, so the C++ worker pool runs in parallel. Build it with::

    cmake --preset release -DMD_BUILD_BINDINGS=ON

See ``docs/API.md`` for the observation contract, action space and reward spec.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
