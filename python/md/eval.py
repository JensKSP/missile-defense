# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Score a policy on the M4 evaluation protocol.

The point of this module is that "beat the baseline" stays a concrete claim.
Routine learned-policy scores use the historical validation prefix; the scripted
yardstick and one final learned-policy score use the next, held-out seed block.
Both sides fill the same per-episode records and call the same C++ ``summarize``
function. Nothing about the comparison is reimplemented on the Python side,
because a reimplementation is exactly where the two would drift.

The first 32 deterministic seeds were historically used during training, so
they are the validation split. The next disjoint 32 values are the held-out
canonical benchmark; choosing ``policy-best.pt`` must never inspect them.

    from md.eval import evaluate, default_seeds

    summary = evaluate(lambda obs, mask: policy(obs, mask))
    print(summary.mean_score, summary.mean_wave)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from . import _md_native as _native
from .benchmark import (
    CANONICAL_AIM_TRAIL,
    CANONICAL_REACTION_DELAY,
    CANONICAL_SEED_OFFSET,
    SEEDS_PER_SPLIT,
    VALIDATION_SEED_OFFSET,
)
from .env import Actions, Flags, Observations, ObsSpec, VecEnv

EpisodeResult = _native.EpisodeResult
Summary = _native.Summary

#: Chooses an action index per environment, given the batch and its action mask.
Policy = Callable[[Observations, Flags], Actions]


def default_seeds(count: int = SEEDS_PER_SPLIT) -> list[int]:
    """The held-out canonical seeds. Never tune or select checkpoints on these."""
    if count < 0:
        raise ValueError("canonical seed count must not be negative")
    pool = _native.default_seeds(CANONICAL_SEED_OFFSET + count)
    return list(pool[CANONICAL_SEED_OFFSET:])


def validation_seeds(count: int = SEEDS_PER_SPLIT) -> list[int]:
    """The historical prefix used for routine evaluation and selection."""
    if count < 0:
        raise ValueError("validation seed count must not be negative")
    pool = _native.default_seeds(VALIDATION_SEED_OFFSET + count)
    return list(pool[VALIDATION_SEED_OFFSET:])


def evaluate(
    policy: Policy,
    *,
    seeds: Sequence[int] | None = None,
    config: object | None = None,
    obs_spec: ObsSpec | None = None,
    frame_skip: int = 4,
    max_ticks: int = 120_000,
    aim_trail: float = CANONICAL_AIM_TRAIL,
    reaction_delay: int = CANONICAL_REACTION_DELAY,
    threads: int = 0,
) -> Summary:
    """Play every seed once under `policy` and aggregate.

    One environment per seed, so each is played exactly once — with auto-reset a
    fast-finishing seed would otherwise start a second episode and be counted
    twice. Environments that have finished keep stepping (their actions are
    ignored) until the slowest one is done; only the first result per env is kept.

    The canonical scripted baseline sees the full simulation, so this protocol
    rejects reduced observation specs rather than attaching a comparable-looking
    score to an information-handicapped policy. ``frame_skip`` is the authoritative
    decision cadence; any value already present in ``config.decision_interval`` is
    replaced in the environment's private copy.
    """
    seed_list = list(default_seeds() if seeds is None else seeds)
    if not seed_list:
        raise ValueError("evaluate needs at least one seed")

    spec = ObsSpec() if obs_spec is None else obs_spec
    canonical = ObsSpec()
    if (
        spec.threats != canonical.threats
        or spec.interceptors != canonical.interceptors
        or spec.blasts != canonical.blasts
    ):
        raise ValueError(
            "policy evaluation requires the default full-capacity ObsSpec; "
            "a truncated policy cannot be compared with the full-state scripted baseline"
        )

    env = VecEnv(
        num_envs=len(seed_list),
        config=config,  # type: ignore[arg-type]
        obs_spec=spec,
        threads=threads,
        frame_skip=frame_skip,
        max_ticks=max_ticks,
        aim_trail=aim_trail,
        reaction_delay=reaction_delay,
        shaping=None,
        seed=seed_list[0],
    )
    env.reset_seeds(seed_list)

    results: list[EpisodeResult | None] = [None] * len(seed_list)
    remaining = len(seed_list)
    while remaining > 0:
        actions = policy(env.observations, env.action_masks())
        _, _, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated
        for index in np.flatnonzero(done):
            i = int(index)
            if results[i] is not None:
                continue  # already measured; this env is only still running to keep pace
            outcome = env.take_episode_result(i)
            if outcome is not None:
                results[i] = outcome
                remaining -= 1

    return _native.summarize([r for r in results if r is not None])


#: Simulation tick rate (`Config::dt` = 1/60 s), for reporting survival in seconds.
TICKS_PER_SECOND = 60.0


def format_summary(summary: Summary) -> str:
    """The full per-run statistics, the same shape `poe eval` prints so the
    scripted baseline and a learned policy can be read side by side.

    Averages are per episode over the seed set; the kills-per-shot line is the
    histogram summed over *every* interceptor in the evaluation, which is the one
    that answers "is it catching clusters or wasting shots?" as a distribution.
    """
    hist = list(summary.kills_per_shot)
    total_shots = sum(hist) or 1
    dist = "  ".join(
        f"{label}:{count} ({100 * count / total_shots:.0f}%)"
        for label, count in zip(("0", "1", "2", "3", "4+"), hist, strict=False)
    )
    seconds = summary.mean_ticks / TICKS_PER_SECOND
    return "\n".join(
        [
            f"mean score       {summary.mean_score:10.1f}   "
            f"[{summary.min_score} .. {summary.max_score}]",
            f"survived         {summary.mean_ticks:10.0f} ticks ({seconds:.1f}s)   "
            f"{summary.survived}/{summary.episodes} reached the cap",
            f"last wave        {summary.mean_wave:10.2f}   "
            f"({summary.mean_waves_cleared:.2f} cleared)",
            f"cities           {summary.mean_cities_left:10.2f} left   "
            f"{summary.mean_cities_lost:.2f} lost   {summary.mean_bonus_cities:.2f} rebuilt",
            f"bases            {summary.mean_bases_left:10.2f} left   "
            f"{summary.mean_bases_lost:.2f} lost",
            f"ammo unfired     {summary.mean_ammo_left:10.2f}   "
            f"(interceptors still loaded at the end)",
            f"targets killed   {summary.mean_kills:10.2f}   "
            f"({summary.mean_mirv_splits:.2f} MIRV splits)",
            f"shots fired      {summary.mean_shots:10.2f}   {summary.mean_hits:.2f} hit "
            f"({100 * summary.mean_hit_rate:.0f}%)   {summary.mean_accuracy:.2f} kills/shot",
            f"kills per shot   {dist}",
        ]
    )
