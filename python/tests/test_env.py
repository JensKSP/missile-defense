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

# Guard on the extension, not on missile_defense.sim.env. Importing that module
# without the extension raises ImportError ("cannot import name _md_native"), and importorskip only
# turns ModuleNotFoundError into a skip — so asking it for missile_defense.sim.env made this file a
# collection *error* on every machine that has no bindings build, which is every
# CI runner. It passed locally only because a dev checkout has one.
pytest.importorskip(
    "missile_defense._md_native",
    reason="the _md_native extension is not built (cmake -DMD_BUILD_BINDINGS=ON)",
)

from missile_defense.sim.env import ObsSpec, Shaping, VecEnv  # noqa: E402


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


def test_recording_cannot_start_part_way_through_an_episode() -> None:
    env = VecEnv(num_envs=2, threads=1, frame_skip=4, max_ticks=8, seed=31)
    actions = np.zeros(env.num_envs, dtype=np.int32)
    _, _, _, truncated, _ = env.step(actions)
    assert not truncated.any()

    with pytest.raises(
        RuntimeError,
        match=r"target environment is at episode tick 0; call reset\(\) before record\(\)",
    ):
        env.record(0)
    assert not env.is_recording(0)

    # The rejected request must not reset the live batch.
    _, _, _, truncated, _ = env.step(actions)
    assert truncated.all()

    env.reset(31)
    env.record(0)
    assert env.is_recording(0)


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


def test_shaping_reads_bases_cities_and_ammo_from_the_observation() -> None:
    # `_phi` locates the battery/city block by hand-computed offsets. At reset the
    # potential must equal a full board: three live batteries, six cities, and
    # every battery's ammo.
    env = VecEnv(num_envs=4, threads=1, shaping=Shaping(), seed=0)
    env.reset(0)
    phi = env._phi()

    config = env._config
    shaping = Shaping()
    expected = (
        (shaping.base_weight * 3)
        + (shaping.city_weight * 6)
        + (shaping.ammo_weight * 3 * config.ammo_per_base)
    )
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
    phi_start = (
        (shaping.base_weight * 3)
        + (shaping.city_weight * 6)
        + (shaping.ammo_weight * 3 * env._config.ammo_per_base)
    )
    assert total == pytest.approx(-phi_start, rel=1e-3)


def test_a_shot_that_kills_nothing_is_charged_the_waste_penalty() -> None:
    # The measured failure this exists for: 72% of the learned policy's
    # interceptors detonated in a zone something else already covered. Firing at
    # an empty sky is the cleanest instance — nothing can die, so the blast is
    # wasted by construction and the penalty must land exactly once per blast.
    # Two envs, same seed and same actions, differing only in the penalty: the
    # reward gap is then exactly the penalty term, with no need to model the score
    # delta or the potential difference.
    penalty = 10.0
    plain = VecEnv(num_envs=1, threads=1, frame_skip=4, seed=0, shaping=Shaping(waste_penalty=0.0))
    priced = VecEnv(
        num_envs=1, threads=1, frame_skip=4, seed=0, shaping=Shaping(waste_penalty=penalty)
    )
    plain.reset(0)
    priced.reset(0)

    seen = 0
    for _ in range(400):
        mask = plain.action_masks()
        legal = np.flatnonzero(mask[0])
        # Fire whenever anything is legal; early shots mostly hit empty sky.
        action = np.array([legal[-1] if len(legal) > 1 else 0], dtype=np.int32)
        _, without, terminated, truncated, _ = plain.step(action)
        _, priced_reward, _, _, _ = priced.step(action)

        wasted = int(priced._wasted[0])
        seen += wasted
        charged = (float(priced_reward[0]) - float(without[0])) * Shaping().scale
        assert charged == pytest.approx(-penalty * wasted, abs=1e-2)
        if terminated[0] or truncated[0]:
            break

    assert seen > 0, "no blast expired without a kill — the test never exercised the penalty"


def test_the_counters_are_zero_when_nothing_is_fired() -> None:
    # NoOp cannot detonate anything, so neither counter may ever tick — otherwise
    # the penalty would be charged against episodes that never took a shot.
    env = VecEnv(num_envs=4, threads=1, frame_skip=4, seed=0)
    env.reset(0)
    for _ in range(200):
        env.step(np.zeros(4, dtype=np.int32))
        assert int(env._wasted.sum()) == 0
        assert int(env._multi_kills.sum()) == 0


def test_the_event_terms_are_off_when_their_weights_are_zero() -> None:
    # With both weights zero the reward must be exactly the potential-shaped one,
    # so the two arms of an experiment differ only by what they are meant to.
    rng = np.random.default_rng(0)
    plain = VecEnv(
        num_envs=2,
        threads=1,
        frame_skip=4,
        seed=0,
        shaping=Shaping(waste_penalty=0.0, multikill_bonus=0.0),
    )
    priced = VecEnv(
        num_envs=2,
        threads=1,
        frame_skip=4,
        seed=0,
        shaping=Shaping(waste_penalty=25.0, multikill_bonus=25.0),
    )
    plain.reset(0)
    priced.reset(0)

    differed = False
    for _ in range(300):
        mask = plain.action_masks()
        actions = np.array([rng.choice(np.flatnonzero(row)) for row in mask], dtype=np.int32)
        _, a, _, _, _ = plain.step(actions)
        _, b, _, _, _ = priced.step(actions)
        if int(priced._wasted.sum()) or int(priced._multi_kills.sum()):
            differed = True
        else:
            np.testing.assert_allclose(a, b, rtol=1e-5)
    assert differed, "the priced env never fired, so the comparison proved nothing"


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


def test_validation_and_canonical_seeds_are_fixed_disjoint_splits() -> None:
    from missile_defense import _md_native
    from missile_defense.sim.eval import default_seeds, validation_seeds

    validation = validation_seeds()
    canonical = default_seeds()
    assert len(validation) == len(canonical) == 32
    assert len(set(validation)) == len(set(canonical)) == 32
    assert set(validation).isdisjoint(canonical)
    # The historical stream prefix remains validation; canonical starts at 32.
    assert validation == list(_md_native.default_seeds(32))
    assert canonical == list(_md_native.default_seeds(64))[32:]
    assert validation_seeds(4) == validation[:4]
    assert default_seeds(4) == canonical[:4]
    assert (validation_seeds(), default_seeds()) == (validation, canonical)


def test_evaluate_scores_a_policy_on_the_shared_protocol() -> None:
    # The comparison M6 rests on: every canonical seed played exactly once, and
    # aggregated by the same C++ summarize() the scripted baseline goes through.
    from missile_defense.sim.eval import evaluate

    rng = np.random.default_rng(0)

    def policy(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.array([rng.choice(np.flatnonzero(r)) for r in mask], dtype=np.int32)

    summary = evaluate(policy, seeds=[1, 2, 3, 4], max_ticks=900)
    assert summary.episodes == 4
    assert summary.min_score <= summary.mean_score <= summary.max_score
    assert 0 <= summary.survived <= 4


def test_evaluate_rejects_a_truncated_observation_spec() -> None:
    from missile_defense.sim.eval import evaluate

    spec = ObsSpec()
    spec.threats -= 1

    def policy(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.zeros(obs.shape[0], dtype=np.int32)

    with pytest.raises(ValueError, match="full-capacity ObsSpec"):
        evaluate(policy, seeds=[1], obs_spec=spec, max_ticks=4)


def test_evaluate_reports_the_full_statistics() -> None:
    # The complete per-run stat set the trainer will draw on: survival, damage,
    # ammunition spend and the kills-per-shot distribution — all aggregated by the
    # same C++ summarize() the scripted baseline goes through.
    from missile_defense.sim.eval import evaluate, format_summary

    rng = np.random.default_rng(1)

    def policy(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.array([rng.choice(np.flatnonzero(r)) for r in mask], dtype=np.int32)

    summary = evaluate(policy, seeds=[1, 2, 3, 4], max_ticks=1500)
    hist = list(summary.kills_per_shot)
    assert len(hist) == 5  # 0, 1, 2, 3, 4+
    assert summary.mean_ticks > 0
    assert 0.0 <= summary.mean_hit_rate <= 1.0
    assert summary.mean_hits <= summary.mean_shots
    assert summary.mean_cities_lost >= 0.0
    assert summary.mean_bases_left <= 3.0
    # Ammo held in reserve at the end — the "shots left" ledger entry. Still
    # recorded, no longer printed: it is ~0 in any episode that ran to the end,
    # because the game ends when the cities do and the ammunition is what was
    # spent defending them.
    assert summary.mean_ammo_left >= 0.0

    # What the score was made of. The three credits are the answer `mean_ammo_left`
    # and `mean_cities_left` could not give, and they reconcile with the total —
    # the same invariant the C++ unit tests assert per tick, checked here across
    # the Python path that aggregates them.
    assert (
        summary.mean_kill_credit + summary.mean_city_credit + summary.mean_ammo_credit
        == pytest.approx(summary.mean_score)
    )
    assert summary.kill_share + summary.city_share + summary.ammo_share == pytest.approx(1.0)
    assert summary.mean_shots_per_wave >= 0.0

    text = format_summary(summary)
    for label in ("survived", "last wave", "cities", "bases", "score from", "shots per wave"):
        assert label in text
    assert "kills per shot" in text


def test_episode_result_histogram_reconciles_with_kills() -> None:
    # bin 0 is the wasted shots, bins 1.. the hits, and — barring a blast in the
    # capped top bin — the weighted histogram is the kill count exactly.
    rng = np.random.default_rng(2)
    env = VecEnv(num_envs=1, threads=1, frame_skip=4, max_ticks=2000, shaping=None, seed=5)
    env.reset_seeds([5])
    result = None
    while result is None:
        mask = env.action_masks()
        actions = np.array([rng.choice(np.flatnonzero(mask[0]))], dtype=np.int32)
        _, _, terminated, truncated, _ = env.step(actions)
        if terminated[0] or truncated[0]:
            result = env.take_episode_result(0)

    hist = list(result.kills_per_shot)
    assert result.wasted == hist[0]
    assert result.hits == sum(hist[1:])
    assert result.hits + result.wasted <= result.shots
    if hist[-1] == 0:
        assert sum(i * count for i, count in enumerate(hist)) == result.kills
