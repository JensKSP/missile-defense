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
    def waves_cleared(self) -> int: ...
    @property
    def ticks(self) -> int: ...
    @property
    def cities_left(self) -> int: ...
    @property
    def cities_lost(self) -> int: ...
    @property
    def bases_left(self) -> int: ...
    @property
    def bases_lost(self) -> int: ...
    @property
    def ammo_left(self) -> int:
        """Unfired interceptors in surviving batteries at episode end."""

    @property
    def bonus_cities(self) -> int: ...
    @property
    def mirv_splits(self) -> int: ...
    @property
    def shots(self) -> int: ...
    @property
    def kills(self) -> int: ...
    @property
    def kills_per_shot(self) -> list[int]:
        """Interceptors binned by threats each destroyed: [0, 1, 2, 3, 4+]."""

    @property
    def terminated(self) -> bool: ...
    @property
    def wasted(self) -> int:
        """Interceptors whose blast destroyed nothing (== kills_per_shot[0])."""

    @property
    def hits(self) -> int:
        """Interceptors whose blast destroyed at least one threat."""

    @property
    def accuracy(self) -> float:
        """Kills per interceptor spent (the documented yardstick)."""

    @property
    def hit_rate(self) -> float:
        """Fraction of resolved interceptors that destroyed at least one threat."""

class Summary:
    """Aggregate over a seed set (md::agent::Summary)."""

    @property
    def episodes(self) -> int: ...
    @property
    def mean_score(self) -> float: ...
    @property
    def mean_wave(self) -> float: ...
    @property
    def mean_waves_cleared(self) -> float: ...
    @property
    def mean_ticks(self) -> float: ...
    @property
    def mean_cities_left(self) -> float: ...
    @property
    def mean_cities_lost(self) -> float: ...
    @property
    def mean_bases_left(self) -> float: ...
    @property
    def mean_bases_lost(self) -> float: ...
    @property
    def mean_ammo_left(self) -> float: ...
    @property
    def mean_bonus_cities(self) -> float: ...
    @property
    def mean_mirv_splits(self) -> float: ...
    @property
    def mean_shots(self) -> float: ...
    @property
    def mean_kills(self) -> float: ...
    @property
    def mean_hits(self) -> float: ...
    @property
    def mean_accuracy(self) -> float: ...
    @property
    def mean_hit_rate(self) -> float: ...
    @property
    def min_score(self) -> int: ...
    @property
    def max_score(self) -> int: ...
    @property
    def survived(self) -> int: ...
    @property
    def kills_per_shot(self) -> list[int]:
        """Kills-per-shot histogram summed over the seed set: [0, 1, 2, 3, 4+]."""

def default_seeds(count: int = ...) -> list[int]:
    """The fixed deterministic seed stream; protocols select disjoint blocks."""

def summarize(episodes: list[EpisodeResult]) -> Summary:
    """Aggregate outcomes with the same function the scripted baseline uses."""

class LoadedPolicy:
    """A promoted `.mdp`, ready to play, with inference in C++.

    `md.export_policy.evaluate` is the reference forward pass and stays the
    definition of the format; this is the implementation the *game* runs, and
    the only one fast enough to score a model over a whole seed block.
    """

    def __init__(self, path: str) -> None:
        """Read and validate a .mdp. Raises if this build cannot run it."""

    @property
    def observation_size(self) -> int: ...
    @property
    def action_count(self) -> int: ...
    @property
    def architecture(self) -> str: ...
    @property
    def display_name(self) -> str: ...
    def play(
        self,
        seed: int,
        max_ticks: int = ...,
        decision_interval: int = ...,
        aim_trail: float = ...,
        reaction_delay: int = ...,
    ) -> EpisodeResult:
        """Play one seed to termination or the cap, under the published handicap.

        The handicap defaults to the canonical one, the same way `md_agent_eval`
        does: a score earned without it is not comparable with the ladder, so
        opting out (``aim_trail=0.0, reaction_delay=0``) has to be asked for.
        """

    def act(self, observation: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]) -> int:
        """The action this policy would take, masked to the legal ones."""

class Config:
    """Tunable simulation constants (md::Config)."""

    world_width: float
    world_height: float
    dt: float
    ammo_per_base: int
    base_cooldown: float
    aim_max_speed: float
    fire_interval: float
    decision_interval: int
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
        aim_trail: float = ...,
        reaction_delay: int = ...,
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
    @property
    def aim_trail(self) -> float: ...
    @property
    def reaction_delay(self) -> int: ...
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
        """Advance by up to ``frame_skip`` ticks without crossing ``max_ticks``.

        Rewards and event features aggregate every simulation tick in the window.
        """

    def action_masks(self, mask: npt.NDArray[np.bool_]) -> None:
        """Fill ``mask`` with which actions are legal per env."""

    def shot_stats(self, wasted: npt.NDArray[np.int32], multi_kills: npt.NDArray[np.int32]) -> None:
        """How the last step spent its ammunition, per env.

        ``wasted`` counts blasts that expired having killed nothing; ``multi_kills``
        counts kills beyond a blast's first. The simulation reports both and prices
        neither — see :class:`md.env.Shaping`.
        """

    def reset_seeds(self, seeds: list[int], obs: npt.NDArray[np.float32]) -> None:
        """Seed each env explicitly — for evaluating on the canonical seed set."""

    def take_episode_result(self, index: int) -> EpisodeResult | None: ...
    def record(self, index: int, on: bool = ...) -> None:
        """Log this env's actions from episode tick 0 so it can be watched in the app."""

    def is_recording(self, index: int) -> bool: ...
    def save_recording(self, index: int, path: str, update: int = ..., label: str = ...) -> bool:
        """Write the last completed episode for ``index``; False if none is ready."""
