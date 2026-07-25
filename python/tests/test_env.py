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

# Guard on the extension, not on md.env. Importing md.env without the extension
# raises ImportError ("cannot import name _md_native"), and importorskip only
# turns ModuleNotFoundError into a skip — so asking it for md.env made this file a
# collection *error* on every machine that has no bindings build, which is every
# CI runner. It passed locally only because a dev checkout has one.
pytest.importorskip(
    "md._md_native", reason="the _md_native extension is not built (cmake -DMD_BUILD_BINDINGS=ON)"
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


def _noop_episode(max_ticks: int, *, enabled: bool) -> tuple[list[float], VecEnv, float]:
    """Play one env to the end on NoOp alone.

    Returns its per-step rewards, the env, and phi(s) as it entered the *final*
    step — which cannot be recovered afterwards, because `_potential` has by then
    moved on to the episode that auto-reset in its place.

    Shaping cannot affect the simulation, so the same seed and the same actions
    give the same trajectory with it on or off. That is what lets a test isolate
    the shaping term exactly, by differencing the two reward streams.
    """
    env = VecEnv(
        num_envs=1,
        threads=1,
        frame_skip=4,
        max_ticks=max_ticks,
        shaping=Shaping(enabled=enabled),
        seed=0,
    )
    env.reset(0)
    rewards: list[float] = []
    for _ in range(50_000):
        phi_before = float(env._phi()[0])
        _, reward, terminated, truncated, _ = env.step(np.zeros(1, dtype=np.int32))
        rewards.append(float(reward[0]))
        if terminated[0] or truncated[0]:
            return rewards, env, phi_before
    raise AssertionError("the episode never ended")


def test_terminal_shaping_charges_the_whole_potential() -> None:
    # phi of an absorbing state is zero by definition, so the transition into
    # game over must charge -phi(s). Zeroing it instead makes dying free, and the
    # Ng et al. invariance — the entire reason shaping is safe to add — is lost.
    shaped, env, phi_last = _noop_episode(200_000, enabled=True)
    plain, _, _ = _noop_episode(200_000, enabled=False)
    assert env._terminated[0] and not env._truncated[0]
    assert len(shaped) == len(plain)

    charged = (shaped[-1] - plain[-1]) * Shaping().scale
    assert charged == pytest.approx(-phi_last, rel=1e-4)


def test_truncation_shaping_bootstraps_from_the_final_observation() -> None:
    # A time-limit cutoff is not death: the state was alive and still worth
    # something. The final observation holds it — `obs` already holds the next
    # episode — so the term is gamma * phi(final) - phi(s), not zero and not
    # -phi(s).
    shaped, env, phi_last = _noop_episode(400, enabled=True)
    plain, _, _ = _noop_episode(400, enabled=False)
    assert env._truncated[0] and not env._terminated[0]

    phi_final = float(env._phi(env._final_obs)[0])
    expected = (Shaping().gamma * phi_final) - phi_last
    charged = (shaped[-1] - plain[-1]) * Shaping().scale
    assert charged == pytest.approx(expected, rel=1e-4)


def test_shaping_telescopes_to_minus_the_initial_potential() -> None:
    # The property the whole design rests on (Ng, Harada & Russell 1999): summed
    # over an episode and discounted, potential-based shaping contributes exactly
    # -phi(s_0) — a constant that cannot depend on the policy, which is why it
    # cannot change the optimal one. Any other terminal handling breaks this.
    shaped, env, _ = _noop_episode(200_000, enabled=True)
    plain, _, _ = _noop_episode(200_000, enabled=False)
    shaping = Shaping()

    total = sum(
        (shaping.gamma**t) * (s - p) * shaping.scale
        for t, (s, p) in enumerate(zip(shaped, plain, strict=True))
    )
    phi_start = (100.0 * 6) + (5.0 * 3 * env._config.ammo_per_base)
    assert total == pytest.approx(-phi_start, rel=1e-3)


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


def test_default_seeds_are_the_canonical_set() -> None:
    from md.eval import default_seeds

    seeds = default_seeds()
    assert len(seeds) == 32
    assert len(set(seeds)) == 32  # a repeated seed would double-count an episode
    assert default_seeds() == seeds  # fixed, so results compare across time
    assert default_seeds(4) == seeds[:4]


def test_evaluate_scores_a_policy_on_the_shared_protocol() -> None:
    # The comparison M6 rests on: every canonical seed played exactly once, and
    # aggregated by the same C++ summarize() the scripted baseline goes through.
    from md.eval import evaluate

    rng = np.random.default_rng(0)

    def policy(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.array([rng.choice(np.flatnonzero(r)) for r in mask], dtype=np.int32)

    summary = evaluate(policy, seeds=[1, 2, 3, 4], max_ticks=900)
    assert summary.episodes == 4
    assert summary.min_score <= summary.mean_score <= summary.max_score
    assert 0 <= summary.survived <= 4
