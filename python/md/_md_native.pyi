# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Typed interface to the nanobind extension — see ``bindings/module.cpp``.

Hand-written rather than generated. Without it the extension is opaque to the
strict type-checkers, and *everything* derived from it in ``env.py`` degrades to
``Unknown`` — which is exactly the layer where a wrong dtype or a transposed
shape would otherwise go unnoticed.

Only what ``module.cpp`` actually exports is declared here; keep the two in step.
The array parameters are the caller-owned buffers the extension writes **in
place**, so they are annotated with the dtype the C++ side reinterprets them as.
"""

import numpy as np
import numpy.typing as npt

__version__: str

# Simulation capacities (md::max_cities, md::base_count, md::max_threats).
MAX_CITIES: int
BASE_COUNT: int
MAX_THREATS: int

class EpisodeResult:
    """Outcome of one episode (md::agent::EpisodeResult)."""

    @property
    def seed(self) -> int: ...
    @property
    def score(self) -> int: ...
    @property
    def wave_reached(self) -> int: ...
    @property
    def cities_left(self) -> int: ...
    @property
    def ticks(self) -> int: ...
    @property
    def shots(self) -> int: ...
    @property
    def kills(self) -> int: ...
    @property
    def terminated(self) -> bool: ...
    @property
    def accuracy(self) -> float: ...

class Summary:
    """Aggregate over a seed set (md::agent::Summary)."""

    @property
    def episodes(self) -> int: ...
    @property
    def mean_score(self) -> float: ...
    @property
    def mean_wave(self) -> float: ...
    @property
    def mean_cities_left(self) -> float: ...
    @property
    def mean_accuracy(self) -> float: ...
    @property
    def min_score(self) -> int: ...
    @property
    def max_score(self) -> int: ...
    @property
    def survived(self) -> int: ...

def default_seeds(count: int = ...) -> list[int]:
    """The canonical evaluation seeds."""

def summarize(episodes: list[EpisodeResult]) -> Summary:
    """Aggregate outcomes with the same function the scripted baseline uses."""

class Config:
    """Tunable simulation constants (md::Config)."""

    world_width: float
    world_height: float
    dt: float
    ammo_per_base: int
    base_cooldown: float
    aim_max_speed: float
    fire_interval: float
    bonus_city_score: int
    score_per_kill: int
    score_per_unused_interceptor: int
    score_per_surviving_city: int
    def __init__(self) -> None: ...

class ObsSpec:
    """How much of the field the observation exposes (md::ObsSpec)."""

    threats: int
    interceptors: int
    blasts: int
    @property
    def size(self) -> int:
        """Floats per observation."""

    def __init__(self) -> None: ...

class VecEnv:
    """A batch of independent simulations with RL conventions applied."""

    def __init__(
        self,
        num_envs: int,
        config: Config = ...,
        obs_spec: ObsSpec = ...,
        threads: int = ...,
        frame_skip: int = ...,
        max_ticks: int = ...,
    ) -> None: ...
    @property
    def num_envs(self) -> int: ...
    @property
    def obs_size(self) -> int: ...
    @property
    def action_count(self) -> int: ...
    @property
    def threads(self) -> int: ...
    @property
    def frame_skip(self) -> int: ...
    def reset(self, seed: int, obs: npt.NDArray[np.float32]) -> None:
        """Seed every env and fill ``obs`` in place."""

    def step(
        self,
        actions: npt.NDArray[np.int32],
        obs: npt.NDArray[np.float32],
        final_obs: npt.NDArray[np.float32],
        rewards: npt.NDArray[np.float32],
        terminated: npt.NDArray[np.bool_],
        truncated: npt.NDArray[np.bool_],
    ) -> None:
        """Advance every env by ``frame_skip`` ticks, writing all outputs in place."""

    def action_masks(self, mask: npt.NDArray[np.bool_]) -> None:
        """Fill ``mask`` with which actions are legal per env."""

    def reset_seeds(self, seeds: list[int], obs: npt.NDArray[np.float32]) -> None:
        """Seed each env explicitly — for evaluating on the canonical seed set."""

    def take_episode_result(self, index: int) -> EpisodeResult | None: ...
    def record(self, index: int, on: bool = ...) -> None:
        """Log this env's actions so the episode can be watched in the app."""

    def is_recording(self, index: int) -> bool: ...
    def save_recording(self, index: int, path: str, update: int = ..., label: str = ...) -> bool:
        """Write the last completed episode for ``index``; False if none is ready."""
