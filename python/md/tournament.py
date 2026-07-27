# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Scoring league models against each other, fairly, or not at all.

**Fairness is a property of the whole orchestration, which is why it lives in
one module.** Every individual piece is already trustworthy — the seeds come
from one C++ generator, the episodes from one `run_episode`, the aggregation
from one `summarize` — and a contest can still be unfair if one contestant got
thirty seeds and another got thirty-two, or if a run that was cancelled halfway
replaced a complete result.

So this module enforces three rules, and they are the whole reason it exists:

1. **Every contestant gets the identical seed set**, taken once and handed to
   each in turn. Not "the same protocol" — the same list.
2. **A ranking only appears when every seed is in.** A partial evaluation is
   reported as partial and never written as a result.
3. **Only a complete canonical evaluation ranks.** A quick head-to-head over
   four seeds is useful and is labelled unranked; `md.benchmark` owns what
   canonical means and this asks it rather than reimplementing it.

**Inference is native, and that is what makes this usable at all.**
:func:`md.export_policy.evaluate` is the reference forward pass — it defines the
`.mdp` format and `python/tests/e2e/test_parity.py` holds it and the C++ side to
the same action, decision for decision — but it runs one observation at a time
from Python, at about 17 ms each for the relational architecture. A canonical
block is 32 seeds of up to 30,000 decisions, so that is *hours* per contestant,
against a progress bar with nothing to report; the head-to-head that produced
this note never finished a single seed in two minutes. The same policy through
`md_native.LoadedPolicy` is the code the game runs at sixty frames a second, and
a contestant's canonical block takes about a minute and a half.

Episodes are played **one seed at a time** rather than as a batch. It costs
nothing — a finished environment in a batch keeps stepping until the slowest one
is done — and it buys the two things a person in front of a progress bar needs:
a count that moves, and a cancel that lands on a seed boundary rather than
mid-episode.

The scripted baseline is a **published constant** rather than a contestant that
is re-run: `md.benchmark.CANONICAL_BASELINE_MEAN_SCORE` was measured on this
exact protocol, and re-measuring it every tournament would spend four minutes to
reproduce a number that is already the definition of the yardstick.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from . import benchmark, league

if TYPE_CHECKING:  # these pull in the native extension; annotations only
    from ._md_native import EpisodeResult, LoadedPolicy, Summary

#: How many seeds a *quick* match uses. Small enough to answer "is this one
#: obviously worse?" in under a minute, and labelled unranked so it can never be
#: mistaken for the benchmark.
QUICK_SEEDS = 4

#: What a quick match caps an episode at. The canonical cap is 120,000 ticks —
#: over half an hour of simulated play per seed — and a quick match that used it
#: would not be quick.
QUICK_MAX_TICKS = 20_000


class TournamentError(Exception):
    """A contest that could not be run, and why."""


class Cancelled(Exception):  # noqa: N818 — a control-flow signal, not a fault
    """Raised through the progress callback to stop a tournament.

    A cancelled tournament writes nothing. That is rule 2 in this module's
    docstring, and it is enforced by simply never reaching the write.
    """


@dataclass(frozen=True)
class Protocol:
    """Which seeds, how long, and whether the result may rank.

    A value rather than a set of arguments, because it is recorded *into* the
    result: a score without the protocol that produced it cannot be compared
    with anything, which is the mistake `md.benchmark` exists to prevent.
    """

    seed_split: str
    seed_offset: int
    seed_count: int
    frame_skip: int
    max_ticks: int
    inference_device: str = benchmark.CANONICAL_INFERENCE_DEVICE
    #: The human handicap in force. Part of the protocol because a score earned
    #: against an agent that never mis-clicks is not the same claim as one
    #: earned against a handicapped one, and a table that mixed them would be
    #: comparing two different games.
    aim_trail: float = benchmark.CANONICAL_AIM_TRAIL

    @property
    def canonical(self) -> bool:
        """Whether a *complete* result under this protocol may be ranked."""
        return benchmark.canonical_baseline_comparable(
            seed_split=self.seed_split,
            seed_offset=self.seed_offset,
            seed_count=self.seed_count,
            frame_skip=self.frame_skip,
            aim_trail=self.aim_trail,
            max_ticks=self.max_ticks,
            inference_device=self.inference_device,
        )

    def seeds(self) -> list[int]:
        """The seed list itself, so every contestant is handed the same one."""
        from . import eval as md_eval  # noqa: PLC0415 — needs the native binding

        # The canonical block goes through `md.eval` rather than the raw
        # generator, so a tournament and `poe eval` cannot disagree about which
        # 32 seeds those are.
        if self.seed_offset == benchmark.CANONICAL_SEED_OFFSET:
            return md_eval.default_seeds(self.seed_count)
        if self.seed_offset == benchmark.VALIDATION_SEED_OFFSET:
            return md_eval.validation_seeds(self.seed_count)
        raise TournamentError(
            f"seed offset {self.seed_offset} is neither the validation nor the "
            "canonical block; there is no protocol that means"
        )


def canonical_protocol() -> Protocol:
    """The published benchmark. The only protocol whose results rank."""
    return Protocol(
        seed_split=benchmark.CANONICAL_SPLIT,
        seed_offset=benchmark.CANONICAL_SEED_OFFSET,
        seed_count=benchmark.SEEDS_PER_SPLIT,
        frame_skip=benchmark.CANONICAL_FRAME_SKIP,
        max_ticks=benchmark.CANONICAL_MAX_TICKS,
    )


def quick_protocol() -> Protocol:
    """A few seeds and a short cap. Useful, and never ranked."""
    return Protocol(
        seed_split=benchmark.CANONICAL_SPLIT,
        seed_offset=benchmark.CANONICAL_SEED_OFFSET,
        seed_count=QUICK_SEEDS,
        frame_skip=benchmark.CANONICAL_FRAME_SKIP,
        max_ticks=QUICK_MAX_TICKS,
    )


@dataclass(frozen=True)
class Result:
    """One contestant's score under one protocol."""

    model_id: str
    display_name: str
    protocol: Protocol
    mean_score: float
    mean_wave: float
    mean_ticks: float
    episodes: int
    #: True only when the protocol is canonical **and** every seed came in.
    #: The league ranks on this and nothing else.
    canonical: bool
    #: Against the published scripted baseline, when the result is comparable
    #: with it at all. `None` for a quick match, deliberately: a number next to
    #: a yardstick it was not measured against is worse than no number.
    versus_baseline: float | None
    finished_at: float = field(default_factory=time.time)

    def as_record(self) -> dict[str, object]:
        """What :func:`md.league.record_result` stores."""
        return {
            "protocol": {
                "seed_split": self.protocol.seed_split,
                "seed_offset": self.protocol.seed_offset,
                "seed_count": self.protocol.seed_count,
                "frame_skip": self.protocol.frame_skip,
                "max_ticks": self.protocol.max_ticks,
                "inference_device": self.protocol.inference_device,
            },
            "mean_score": self.mean_score,
            "mean_wave": self.mean_wave,
            "mean_ticks": self.mean_ticks,
            "episodes": self.episodes,
            "canonical": self.canonical,
            "versus_baseline": self.versus_baseline,
        }


#: Called with (contestant index, contestants, seeds done, seeds total). Raise
#: :class:`Cancelled` from it to stop; nothing will have been written.
Progress = Callable[[int, int, int, int], None]


def load_policy(model: league.Model) -> LoadedPolicy:
    """A league model's weights, loaded once, ready to play.

    The failure this catches is the one a person actually meets: a `.mdp`
    trained against an older observation, which parses and cannot be run. The
    native loader refuses it by name and by number, and that arrives here as a
    :class:`TournamentError` rather than as a crash half-way through a contest.
    """
    from ._md_native import LoadedPolicy as Native  # noqa: PLC0415 — the native binding

    try:
        return Native(str(model.policy))
    except (RuntimeError, ValueError, OSError) as error:
        raise TournamentError(f"{model.name}: {error}") from error


def _summarize(episodes: Sequence[EpisodeResult]) -> Summary:
    """The same aggregation the scripted baseline is published with."""
    from ._md_native import summarize  # noqa: PLC0415 — the native binding

    return summarize(list(episodes))


def evaluate_model(
    model: league.Model,
    protocol: Protocol | None = None,
    *,
    seeds: Sequence[int] | None = None,
    progress: Progress | None = None,
    index: int = 0,
    contestants: int = 1,
) -> Result:
    """Play ``model`` over ``protocol``'s seeds and score it.

    ``seeds`` overrides the protocol's own list, which is how a head-to-head
    guarantees rule 1 — it takes the list once and hands the *same* one to each
    contestant, rather than trusting two calls to derive it identically.

    ``progress`` is called after **every seed**, with the count of finished
    ones. That is a real number and not a liveness tick: it is what tells
    somebody whether to wait, and it is where a cancellation lands.
    """
    chosen = canonical_protocol() if protocol is None else protocol
    seed_list = list(chosen.seeds() if seeds is None else seeds)
    if not seed_list:
        raise TournamentError("a tournament needs at least one seed")

    runner = load_policy(model)
    if progress is not None:
        progress(index, contestants, 0, len(seed_list))

    episodes: list[EpisodeResult] = []
    for done, seed in enumerate(seed_list, start=1):
        episodes.append(runner.play(seed, chosen.max_ticks, chosen.frame_skip))
        if progress is not None:
            # After the episode, so the count is of seeds that are *in*. A
            # cancellation raises out of here, and rule 2 makes that safe:
            # nothing is recorded until every seed has been played.
            progress(index, contestants, done, len(seed_list))

    summary = _summarize(episodes)

    # Rule 2: a ranking only appears when every seed is in. `summarize` counts
    # what it was given, so a short count is a short evaluation however it got
    # that way — a crash, a cancellation, an environment that never finished.
    complete = summary.episodes == len(seed_list)
    canonical = complete and chosen.canonical
    return Result(
        model_id=model.model_id,
        display_name=model.name,
        protocol=chosen,
        mean_score=float(summary.mean_score),
        mean_wave=float(summary.mean_wave),
        mean_ticks=float(summary.mean_ticks),
        episodes=int(summary.episodes),
        canonical=canonical,
        versus_baseline=(
            float(summary.mean_score) - benchmark.CANONICAL_BASELINE_MEAN_SCORE
            if canonical
            else None
        ),
    )


def rank(models: Sequence[league.Model]) -> list[tuple[league.Model, float | None]]:
    """The league table: best canonical score first, unranked models last.

    Rule 3, as a sort. A model whose only results are quick matches sorts to the
    bottom with no score rather than with its best quick number — which would
    put a four-seed warm-up above a thirty-two-seed benchmark.
    """
    scored: list[tuple[league.Model, float | None]] = []
    for model in models:
        best = model.best_result
        # `league._number` semantics inline: a results file is on disk and a
        # malformed entry must not stop a table from being drawn.
        raw = None if best is None else best.get("mean_score")
        score = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
        scored.append((model, score))
    scored.sort(key=lambda pair: (pair[1] is not None, pair[1] or 0.0), reverse=True)
    return scored


@dataclass(frozen=True)
class Match:
    """Two contestants over one shared seed set."""

    left: Result
    right: Result
    seeds: tuple[int, ...]

    @property
    def winner(self) -> Result | None:
        """The higher mean score, or ``None`` for a tie."""
        if self.left.mean_score == self.right.mean_score:
            return None
        return self.left if self.left.mean_score > self.right.mean_score else self.right

    @property
    def ranked(self) -> bool:
        """Whether this match may move the league table."""
        return self.left.canonical and self.right.canonical


def head_to_head(
    left: league.Model,
    right: league.Model,
    protocol: Protocol | None = None,
    *,
    progress: Progress | None = None,
    record: bool = True,
) -> Match:
    """Play two models over the *same* seeds and record both results.

    The seed list is taken **once**, here, and handed to both. That is rule 1,
    and it is a stronger promise than "both used the canonical protocol": two
    derivations of the same list are two chances to differ.

    Nothing is written until both sides are done. A cancellation part-way
    through therefore leaves the league exactly as it was — which is rule 2, and
    it is why the two `record_result` calls are at the bottom rather than after
    each contestant.
    """
    chosen = canonical_protocol() if protocol is None else protocol
    seeds = chosen.seeds()

    left_result = evaluate_model(
        left, chosen, seeds=seeds, progress=progress, index=0, contestants=2
    )
    right_result = evaluate_model(
        right, chosen, seeds=seeds, progress=progress, index=1, contestants=2
    )

    if record:
        league.record_result(left, left_result.as_record())
        league.record_result(right, right_result.as_record())
    return Match(left_result, right_result, tuple(seeds))


# ---- match manifests ---------------------------------------------------------


def write_manifest(match: Match, destination: Path, recordings: dict[str, Path]) -> Path:
    """One manifest per paired seed, for the split-screen spectator (Task 8).

    Both sides, the seed they shared, and the scores the tournament recorded —
    so the game can assert, when it plays the manifest back, that what it
    rendered is what was scored. A spectator that quietly showed a different
    episode than the table claims would be worse than no spectator.
    """
    import json  # noqa: PLC0415

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "seeds": list(match.seeds),
        "left": {
            "model_id": match.left.model_id,
            "display_name": match.left.display_name,
            "mean_score": match.left.mean_score,
            "recording": str(recordings.get("left", "")),
        },
        "right": {
            "model_id": match.right.model_id,
            "display_name": match.right.display_name,
            "mean_score": match.right.mean_score,
            "recording": str(recordings.get("right", "")),
        },
        "ranked": match.ranked,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


# ---- paired recordings -------------------------------------------------------


def record_episode(
    model: league.Model,
    seed: int,
    destination: Path,
    *,
    max_ticks: int = 120_000,
    frame_skip: int = 4,
) -> Path:
    """Play one seed under one model and save the episode as a recording.

    The other half of :func:`write_manifest`. A manifest names two recordings;
    without this nothing produced them, and a "paired match" was a file format
    with no way to fill it in.

    The seed is set with ``reset_seeds`` rather than the constructor's, because
    the constructor's is a *starting point* a `VecEnv` derives per-episode seeds
    from — and a match is only a match if both sides played the seed the
    manifest claims. The saved recording carries that seed in its header, which
    is what `MatchPlayer` checks before it will pair two files.
    """
    from .env import VecEnv  # noqa: PLC0415

    # `VecEnv` and not `LoadedPolicy.play`, because what is wanted here is the
    # *recording* — the action log a `.mdr` is made of — and that is the
    # environment's to write. Only the inference comes from the native policy,
    # which is the difference between a minute of waiting and an hour of it.
    runner = load_policy(model)

    env = VecEnv(num_envs=1, frame_skip=frame_skip, max_ticks=max_ticks, seed=seed)
    env.reset_seeds([seed])
    env.record(0)

    import numpy as np  # noqa: PLC0415

    while True:
        observation = env.observations[0]
        mask = env.action_masks()[0]
        action = runner.act(observation, mask)
        _, _, terminated, truncated, _ = env.step(np.array([action], dtype=np.int32))
        if bool(terminated[0]) or bool(truncated[0]):
            break

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not env.save_recording(0, destination, label=model.name):
        raise TournamentError(f"{model.name}: the episode could not be saved to {destination}")
    return destination


def record_pair(
    match: Match,
    directory: Path,
    *,
    seed: int | None = None,
    max_ticks: int = 120_000,
    root: Path | None = None,
) -> dict[str, Path]:
    """Record both sides of ``match`` on one shared seed.

    Returns what :func:`write_manifest` wants. The seed defaults to the first of
    the match's own — the same list rule 1 handed to both contestants — so the
    episode on screen is drawn from the set that produced the scores beside it.
    """
    chosen = match.seeds[0] if seed is None else seed
    if chosen not in match.seeds:
        raise TournamentError(
            f"seed {chosen} is not one this match was played on; a recording of it "
            "would not be an episode either score was measured over"
        )
    recordings: dict[str, Path] = {}
    for side, result in (("left", match.left), ("right", match.right)):
        model = league.find(result.model_id, root)
        if model is None:
            raise TournamentError(
                f"the {side} model ({result.model_id}) is no longer in the league, "
                "so its side of this match cannot be recorded"
            )
        recordings[side] = record_episode(
            model, chosen, directory / f"{side}.mdr", max_ticks=max_ticks
        )
    return recordings
