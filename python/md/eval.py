# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Score a policy on the M4 evaluation protocol.

The point of this module is that "beat the baseline" stays a concrete claim. The
scripted agent's numbers in ``docs/ROADMAP.md`` come from ``md::agent::evaluate``
over ``default_seeds``; a learned policy is measured here over *the same seeds*,
with the per-episode fields filled by the same code path and aggregated by the
same ``summarize`` function. Nothing about the comparison is reimplemented on the
Python side, because a reimplementation is exactly where the two would drift.

    from md.eval import evaluate, default_seeds

    summary = evaluate(lambda obs, mask: policy(obs, mask))
    print(summary.mean_score, summary.mean_wave)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from . import _md_native as _native
from .env import Actions, Flags, Observations, VecEnv

EpisodeResult = _native.EpisodeResult
Summary = _native.Summary

#: Chooses an action index per environment, given the batch and its action mask.
Policy = Callable[[Observations, Flags], Actions]


def default_seeds(count: int = 32) -> list[int]:
    """The canonical evaluation seeds. Never tune against these."""
    return list(_native.default_seeds(count))


def evaluate(
    policy: Policy,
    *,
    seeds: Sequence[int] | None = None,
    config: object | None = None,
    frame_skip: int = 4,
    max_ticks: int = 120_000,
    threads: int = 0,
) -> Summary:
    """Play every seed once under `policy` and aggregate.

    One environment per seed, so each is played exactly once — with auto-reset a
    fast-finishing seed would otherwise start a second episode and be counted
    twice. Environments that have finished keep stepping (their actions are
    ignored) until the slowest one is done; only the first result per env is kept.
    """
    seed_list = list(default_seeds() if seeds is None else seeds)
    if not seed_list:
        raise ValueError("evaluate needs at least one seed")

    env = VecEnv(
        num_envs=len(seed_list),
        config=config,  # type: ignore[arg-type]
        threads=threads,
        frame_skip=frame_skip,
        max_ticks=max_ticks,
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


def format_summary(summary: Summary) -> str:
    """The same shape `poe eval` prints, so the two can be read side by side."""
    return "\n".join(
        [
            f"mean score      {summary.mean_score:10.1f}   "
            f"[{summary.min_score} .. {summary.max_score}]",
            f"mean wave       {summary.mean_wave:10.2f}",
            f"mean cities left{summary.mean_cities_left:10.2f}",
            f"kills per shot  {summary.mean_accuracy:10.2f}",
            f"survived cap    {summary.survived:10d} / {summary.episodes}",
        ]
    )
