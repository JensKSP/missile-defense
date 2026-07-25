# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Vectorised reinforcement-learning environment over the C++ simulation.

The batch is stepped in C++ across a worker pool with the GIL released, writing
observations straight into the NumPy arrays held here — a rollout never copies a
batch and never allocates per step.

See ``docs/API.md`` for the observation contract (raw state only, never analysis),
the discrete action space, and the reward specification implemented below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _md_native as _native

Config = _native.Config
ObsSpec = _native.ObsSpec


@dataclass(frozen=True)
class Shaping:
    """Potential-based reward shaping (Ng, Harada & Russell, 1999).

    ``r' = r + gamma * phi(s') - phi(s)`` with
    ``phi(s) = city_weight * live_cities + ammo_weight * total_ammo``.

    The weights are the *same* ones the end-of-wave bonus already pays (100 per
    surviving city, 5 per unused interceptor); shaping merely delivers them at the
    moment they are earned. Losing a city costs ~100 immediately instead of at the
    next wave boundary — which, at 60 Hz, discounting would otherwise render
    mathematically invisible. Being a potential difference, it provably leaves the
    optimal policy unchanged.
    """

    city_weight: float = 100.0
    ammo_weight: float = 5.0
    gamma: float = 0.997
    enabled: bool = True

    @property
    def scale(self) -> float:
        """Divide rewards by this so the value head sees O(1) targets."""
        return 100.0


class VecEnv:
    """A batch of independent Missile Command environments.

    Follows the Gymnasium vector convention: ``step`` auto-resets a finished
    environment so ``obs`` always holds a live state, and returns the final
    observation separately in ``info`` — a truncated return has to bootstrap from
    it, and conflating truncation with termination would teach the agent that
    running out of clock is as bad as losing.
    """

    def __init__(
        self,
        num_envs: int = 1024,
        *,
        config: Config | None = None,
        obs_spec: ObsSpec | None = None,
        threads: int = 0,
        frame_skip: int = 4,
        max_ticks: int = 120_000,
        shaping: Shaping | None = None,
        seed: int = 0,
    ) -> None:
        self._config = config if config is not None else Config()
        self._spec = obs_spec if obs_spec is not None else ObsSpec()
        self._shaping = shaping if shaping is not None else Shaping()
        self._native = _native.VecEnv(
            num_envs=num_envs,
            config=self._config,
            obs_spec=self._spec,
            threads=threads,
            frame_skip=frame_skip,
            max_ticks=max_ticks,
        )
        n, k = self.num_envs, self.obs_size
        # Allocated once; C++ writes into these in place, for the process lifetime.
        self._obs = np.zeros((n, k), dtype=np.float32)
        self._final_obs = np.zeros((n, k), dtype=np.float32)
        self._rewards = np.zeros(n, dtype=np.float32)
        self._terminated = np.zeros(n, dtype=bool)
        self._truncated = np.zeros(n, dtype=bool)
        self._mask = np.zeros((n, self.action_count), dtype=bool)
        self._potential = np.zeros(n, dtype=np.float32)
        self._seed = seed
        self.reset(seed)

    # ---- shape -----------------------------------------------------------
    @property
    def num_envs(self) -> int:
        return int(self._native.num_envs)

    @property
    def obs_size(self) -> int:
        return int(self._native.obs_size)

    @property
    def action_count(self) -> int:
        return int(self._native.action_count)

    @property
    def frame_skip(self) -> int:
        return int(self._native.frame_skip)

    @property
    def threads(self) -> int:
        return int(self._native.threads)

    # ---- the potential, read back out of the observation ------------------
    def _phi(self) -> np.ndarray:
        """phi(s) for every env, recovered from the observation itself.

        The city-alive flags and per-battery ammo are already in the observation
        (that is the point of the fairness rule), so shaping needs no extra
        simulation state — just the layout offsets.
        """
        spec = self._spec
        base = (
            spec.threats * 9
            + spec.interceptors * 7
            + spec.blasts * 4
        )
        bases = self._obs[:, base : base + _native.BASE_COUNT * 4].reshape(
            self.num_envs, _native.BASE_COUNT, 4
        )
        cities_at = base + _native.BASE_COUNT * 4
        cities = self._obs[:, cities_at : cities_at + _native.MAX_CITIES * 2].reshape(
            self.num_envs, _native.MAX_CITIES, 2
        )
        live_cities = cities[:, :, 0].sum(axis=1)
        # ammo is stored as a fraction of a full battery
        ammo = (bases[:, :, 2] * float(self._config.ammo_per_base)).sum(axis=1)
        s = self._shaping
        return (s.city_weight * live_cities + s.ammo_weight * ammo).astype(np.float32)

    # ---- the loop ---------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._seed = seed
        self._native.reset(self._seed, self._obs)
        self._potential = self._phi()
        return self._obs

    def action_masks(self) -> np.ndarray:
        """Which actions are legal per env — apply before sampling."""
        self._native.action_masks(self._mask)
        return self._mask

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        actions = np.ascontiguousarray(actions, dtype=np.int32)
        self._native.step(
            actions,
            self._obs,
            self._final_obs,
            self._rewards,
            self._terminated,
            self._truncated,
        )
        reward = self._rewards
        if self._shaping.enabled:
            phi_next = self._phi()
            done = self._terminated | self._truncated
            # A finished episode's phi(s') belongs to the *new* episode, so the
            # shaping term would compare across a discontinuity. Zero it there.
            delta = np.where(done, 0.0, self._shaping.gamma * phi_next - self._potential)
            reward = reward + delta.astype(np.float32)
            self._potential = phi_next
        info = {"final_observation": self._final_obs}
        return (
            self._obs,
            reward / self._shaping.scale,
            self._terminated,
            self._truncated,
            info,
        )
