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

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from . import _md_native as _native

Config = _native.Config
ObsSpec = _native.ObsSpec

# The batch arrays, named once so the buffer contract reads the same everywhere:
# observations are float32, the episode flags are bool, actions are int32.
Observations = npt.NDArray[np.float32]
Rewards = npt.NDArray[np.float32]
Flags = npt.NDArray[np.bool_]
Actions = npt.NDArray[np.int32]


@dataclass(frozen=True)
class Shaping:
    """The training reward: potential-based shaping, plus two priced events.

    ``r' = r + gamma * phi(s') - phi(s) - waste_penalty * wasted
                                       + multikill_bonus * multi_kills``
    ``phi(s) = base_weight * live_bases + city_weight * live_cities
               + ammo_weight * total_ammo``

    **The two halves behave completely differently, and it matters.**

    The `phi` terms are potential-based (Ng, Harada & Russell, 1999), so they
    provably leave the optimal policy unchanged — they only change how *fast* it
    is found. City and ammo use the same weights the end-of-wave bonus already
    pays, so shaping merely delivers them the moment they are earned rather than
    at the next wave boundary, where 60 Hz discounting would render them nearly
    invisible. Batteries are in there for the same reason and were previously
    missing entirely: losing one costs a third of your firepower for the rest of
    the wave and used to be worth exactly nothing.

    ``waste_penalty`` and ``multikill_bonus`` are **not** potential terms. They
    genuinely change the objective, which is the only way to change what the
    policy converges to — and the reason they must be judged on the 32-seed score
    rather than on themselves. Neither touches the score, so
    ``md::agent::evaluate`` and the 113,834 baseline are unaffected.
    """

    city_weight: float = 100.0
    ammo_weight: float = 5.0
    #: A destroyed battery stops firing for the rest of the wave and revives at the
    #: next one, so unlike a city the loss is temporary — but it was priced at
    #: *nothing*, which made losing a third of your firepower mid-wave free. The
    #: weight is above a city's on purpose: protecting the guns is what protects
    #: the cities. Being a potential term it cannot change the optimal policy, only
    #: how quickly the loss becomes visible.
    base_weight: float = 200.0
    #: Must match `PPOConfig.gamma` — this discounts the potential term, that one
    #: discounts the return, and the invariance proof assumes they are the same.
    gamma: float = 0.999
    enabled: bool = True

    #: Charged when an interceptor's blast expires having destroyed nothing. Unlike
    #: the terms above this is *not* potential-based, so it genuinely changes the
    #: objective — which is the point, and the reason to judge it on the 32-seed
    #: score rather than on itself.
    #:
    #: **Off by default, on the evidence.** Set to 10 it moved break-even accuracy
    #: from 20% to 60%, and across 800 updates the policy's accuracy did not move
    #: at all: 0.257 at update 50, 0.242 at update 800. It paid the penalty rather
    #: than avoiding it — rationally, because a landed warhead costs 100-200 in
    #: potential while a wasted shot costs 15, so firing on a poor chance is still
    #: the better bet. The knob works; the theory behind it did not. Kept because
    #: it is one flag to re-test once something else has changed.
    waste_penalty: float = 0.0
    #: Also off by default, and for a better reason than the above: the score now
    #: pays for multi-kills properly by itself. A blast catching two warheads at
    #: the wave-11 multiplier is worth 300, and ammunition is finite, so the
    #: incentive to catch clusters is already in the objective. Adding a bonus on
    #: top only distorts it — and it is the one term that rewards *waiting*, which
    #: an agent can overlearn into holding fire while the cities burn.
    multikill_bonus: float = 0.0

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
        self._obs: Observations = np.zeros((n, k), dtype=np.float32)
        self._final_obs: Observations = np.zeros((n, k), dtype=np.float32)
        self._rewards: Rewards = np.zeros(n, dtype=np.float32)
        self._terminated: Flags = np.zeros(n, dtype=np.bool_)
        self._truncated: Flags = np.zeros(n, dtype=np.bool_)
        self._mask: Flags = np.zeros((n, self.action_count), dtype=np.bool_)
        self._potential: Rewards = np.zeros(n, dtype=np.float32)
        self._wasted: npt.NDArray[np.int32] = np.zeros(n, dtype=np.int32)
        self._multi_kills: npt.NDArray[np.int32] = np.zeros(n, dtype=np.int32)
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

    @property
    def observations(self) -> Observations:
        """The live observation batch — the same array every step writes into."""
        return self._obs

    @property
    def spec(self) -> ObsSpec:
        """How many threat / interceptor / blast slots the observation carries.

        Public because a policy that reads the observation *as entities* rather
        than as a flat vector needs the block sizes to slice it.
        """
        return self._spec

    # ---- the potential, read back out of the observation ------------------
    def _phi(self, obs: Observations | None = None) -> Rewards:
        """phi(s) for every env, recovered from the observation itself.

        The city-alive flags and per-battery ammo are already in the observation
        (that is the point of the fairness rule), so shaping needs no extra
        simulation state — just the layout offsets.

        Defaults to the live batch. `step` also needs it for the *final*
        observation of a truncated episode, which is no longer in `_obs`.
        """
        source = self._obs if obs is None else obs
        spec = self._spec
        base = spec.threats * 9 + spec.interceptors * 7 + spec.blasts * 4
        bases = source[:, base : base + _native.BASE_COUNT * 4].reshape(
            self.num_envs, _native.BASE_COUNT, 4
        )
        cities_at = base + _native.BASE_COUNT * 4
        cities = source[:, cities_at : cities_at + _native.MAX_CITIES * 2].reshape(
            self.num_envs, _native.MAX_CITIES, 2
        )
        live_cities = cities[:, :, 0].sum(axis=1)
        # Feature 0 of a battery is its alive flag; a battery knocked out mid-wave
        # reads zero here and revives at the next wave start.
        live_bases = bases[:, :, 0].sum(axis=1)
        # ammo is stored as a fraction of a full battery
        ammo = (bases[:, :, 2] * float(self._config.ammo_per_base)).sum(axis=1)
        s = self._shaping
        phi = (s.base_weight * live_bases) + (s.city_weight * live_cities) + (s.ammo_weight * ammo)
        # NumPy's reductions above are typed loosely enough that the sum comes back
        # as Any; the cast states the dtype astype() has actually produced.
        return cast(Rewards, phi.astype(np.float32))

    # ---- recording, for watching a policy in the app -----------------------
    def record(self, index: int, on: bool = True) -> None:
        """Log environment ``index``'s actions so its episodes can be watched.

        Record one environment, not the batch: a training run wants the occasional
        watchable episode, not a copy of every rollout. The log is four bytes per
        agent step, so leaving one recording costs nothing next to the forward pass.
        """
        self._native.record(index, on)

    def is_recording(self, index: int) -> bool:
        return bool(self._native.is_recording(index))

    def save_recording(
        self,
        index: int,
        path: str | os.PathLike[str],
        *,
        update: int = 0,
        label: str = "",
    ) -> bool:
        """Write the last *completed* episode for ``index`` to ``path``.

        Returns False when no episode has finished since the last call — episodes
        are handed over whole, so a partial log is never written. ``update`` and
        ``label`` are stamped into the file so the app can say which point in
        training it is showing.
        """
        return bool(self._native.save_recording(index, str(path), update, label))

    def reset_seeds(self, seeds: Sequence[int]) -> Observations:
        """Seed each environment explicitly — for the canonical evaluation set.

        ``reset`` derives env *i*'s seed as ``seed + i``; the evaluation seeds are
        not an arithmetic run, so they have to be named.
        """
        self._native.reset_seeds(list(seeds), self._obs)
        self._potential = self._phi()
        return self._obs

    def take_episode_result(self, index: int) -> _native.EpisodeResult | None:
        """Outcome of the last episode this env finished, or None.

        Reported in the same shape as the scripted baseline's, so both can be
        aggregated by ``md.eval``'s (that is, the C++) ``summarize``.
        """
        return self._native.take_episode_result(index)

    # ---- the loop ---------------------------------------------------------
    def reset(self, seed: int | None = None) -> Observations:
        if seed is not None:
            self._seed = seed
        self._native.reset(self._seed, self._obs)
        self._potential = self._phi()
        return self._obs

    def action_masks(self) -> Flags:
        """Which actions are legal per env — apply before sampling."""
        self._native.action_masks(self._mask)
        return self._mask

    def step(
        self, actions: npt.ArrayLike
    ) -> tuple[Observations, Rewards, Flags, Flags, dict[str, Observations]]:
        batch: Actions = np.ascontiguousarray(actions, dtype=np.int32)
        self._native.step(
            batch,
            self._obs,
            self._final_obs,
            self._rewards,
            self._terminated,
            self._truncated,
        )
        reward: Rewards = self._rewards
        if self._shaping.enabled:
            phi_next = self._phi()
            gamma = self._shaping.gamma
            # Three cases, because after an episode ends `_obs` already holds the
            # *next* episode's first state and phi(s') must not be read from it.
            #
            #   still running — the ordinary term, gamma * phi(s') - phi(s).
            #   terminated    — phi of an absorbing state is 0 *by definition*
            #                   (Ng, Harada & Russell 1999), so the term is
            #                   -phi(s). Zeroing it instead would leave the total
            #                   shaping policy-dependent, and the invariance
            #                   proof — the entire reason this is safe to add —
            #                   would no longer hold.
            #   truncated     — the clock ran out but the state was alive and
            #                   still worth something, so bootstrap it from the
            #                   final observation rather than treating it as death.
            delta: Rewards = ((gamma * phi_next) - self._potential).astype(np.float32)
            if self._truncated.any():
                phi_final = self._phi(self._final_obs)
                delta = np.where(
                    self._truncated, (gamma * phi_final) - self._potential, delta
                ).astype(np.float32)
            delta = np.where(self._terminated, -self._potential, delta).astype(np.float32)
            reward = (reward + delta).astype(np.float32)
            self._potential = phi_next

            # Not potential terms, and deliberately so: these are the two that
            # actually move the objective. Everything above is optimality-neutral
            # and only changes how fast the policy gets where it was already going.
            if self._shaping.waste_penalty or self._shaping.multikill_bonus:
                self._native.shot_stats(self._wasted, self._multi_kills)
                reward = (
                    reward
                    - (self._shaping.waste_penalty * self._wasted)
                    + (self._shaping.multikill_bonus * self._multi_kills)
                ).astype(np.float32)
        scaled: Rewards = (reward / self._shaping.scale).astype(np.float32)
        info: dict[str, Observations] = {"final_observation": self._final_obs}
        return (self._obs, scaled, self._terminated, self._truncated, info)
