# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for the Python side of the RL environment.

The batch mechanics themselves are pinned in C++ (``bindings/tests``). What is
only testable from here is the layer Python adds: the NumPy buffer contract, and
the potential-based shaping — whose ``_phi`` reads the observation by hand-computed
offsets, so it silently breaks if the observation layout ever moves.
"""

from __future__ import annotations

import numpy as np
import pytest

_native = pytest.importorskip(
    "md.env", reason="the _md_native extension is not built (cmake -DMD_BUILD_BINDINGS=ON)"
)

from md.env import Shaping, VecEnv  # noqa: E402


@pytest.fixture
def env() -> VecEnv:
    return VecEnv(num_envs=8, threads=1, frame_skip=4, max_ticks=400, seed=0)


def test_reset_returns_the_batch_shaped_observation(env: VecEnv) -> None:
    obs = env.reset(0)
    assert obs.shape == (env.num_envs, env.obs_size)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_step_returns_the_gymnasium_five_tuple(env: VecEnv) -> None:
    actions = np.zeros(env.num_envs, dtype=np.int32)
    obs, reward, terminated, truncated, info = env.step(actions)

    assert obs.shape == (env.num_envs, env.obs_size)
    assert reward.shape == (env.num_envs,)
    assert terminated.shape == (env.num_envs,)
    assert truncated.shape == (env.num_envs,)
    assert terminated.dtype == np.bool_
    assert truncated.dtype == np.bool_
    assert info["final_observation"].shape == (env.num_envs, env.obs_size)
    assert not (terminated & truncated).any()


def test_buffers_are_reused_rather_than_reallocated(env: VecEnv) -> None:
    # The zero-copy claim: C++ writes into arrays this object owns for the process
    # lifetime, so a rollout never allocates a batch per step.
    first = env.reset(0)
    obs, _, _, _, _ = env.step(np.zeros(env.num_envs, dtype=np.int32))
    assert obs is first


def test_action_masks_are_per_env_and_allow_noop(env: VecEnv) -> None:
    mask = env.action_masks()
    assert mask.shape == (env.num_envs, env.action_count)
    assert mask.dtype == np.bool_
    # Index 0 is NoOp: always available, whatever the state.
    assert mask[:, 0].all()


def test_masked_sampling_drives_a_rollout(env: VecEnv) -> None:
    rng = np.random.default_rng(0)
    env.reset(0)
    for _ in range(50):
        mask = env.action_masks()
        actions = np.array([rng.choice(np.flatnonzero(row)) for row in mask], dtype=np.int32)
        obs, reward, _, _, _ = env.step(actions)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward).all()


def test_shaping_reads_cities_and_ammo_from_the_observation() -> None:
    # `_phi` locates the battery/city block by hand-computed offsets. At reset the
    # potential must equal a full board: six cities plus every battery's ammo.
    env = VecEnv(num_envs=4, threads=1, shaping=Shaping(), seed=0)
    env.reset(0)
    phi = env._phi()

    config = env._config
    expected = (100.0 * 6) + (5.0 * 3 * config.ammo_per_base)
    assert phi.shape == (env.num_envs,)
    np.testing.assert_allclose(phi, expected, rtol=1e-4)


def test_shaping_is_off_when_disabled() -> None:
    # With shaping disabled the reward is the raw score delta over the window,
    # scaled — no potential term. A no-op opening tick scores nothing.
    env = VecEnv(num_envs=4, threads=1, shaping=Shaping(enabled=False), seed=0)
    env.reset(0)
    _, reward, _, _, _ = env.step(np.zeros(env.num_envs, dtype=np.int32))
    np.testing.assert_array_equal(reward, np.zeros(env.num_envs, dtype=np.float32))


def test_a_batch_is_reproducible_from_its_seed() -> None:
    def rollout() -> tuple[np.ndarray, np.ndarray]:
        env = VecEnv(num_envs=8, threads=1, frame_skip=4, seed=0)
        env.reset(123)
        rng = np.random.default_rng(7)
        total = np.zeros(env.num_envs, dtype=np.float64)
        for _ in range(25):
            actions = rng.integers(0, env.action_count, size=env.num_envs, dtype=np.int32)
            obs, reward, _, _, _ = env.step(actions)
            total += reward
        return obs.copy(), total

    first_obs, first_total = rollout()
    second_obs, second_total = rollout()
    np.testing.assert_array_equal(first_obs, second_obs)
    np.testing.assert_array_equal(first_total, second_total)
