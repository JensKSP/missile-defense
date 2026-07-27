# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: OpenAI Codex
"""Authoritative metadata for policy evaluation.

Training-time validation and the final benchmark deliberately use different seed
splits.  Keeping the protocol constants here gives the trainer and the console one
lightweight source of truth without making either import the other (or PyTorch).
"""

from __future__ import annotations

from dataclasses import dataclass

CANONICAL_SPLIT = "canonical"
VALIDATION_SPLIT = "validation"

# The prefix was historically used for routine checkpoint selection, so it is
# validation data, not honestly held out. The following untouched block is the
# canonical benchmark.
SEEDS_PER_SPLIT = 32
VALIDATION_SEED_OFFSET = 0
CANONICAL_SEED_OFFSET = SEEDS_PER_SPLIT

# Final checkpoint scoring is pinned to this protocol. Training-time validation
# records its own cadence and tick cap because users may deliberately change them.
CANONICAL_FRAME_SKIP = 4
CANONICAL_MAX_TICKS = 120_000
CANONICAL_INFERENCE_DEVICE = "cpu"


@dataclass(frozen=True)
class Baseline:
    """One rung of the scripted ladder — what ``--skill <name>`` scores.

    ``mean_score`` is exact rather than the one-decimal figure the docs quote:
    it is a mean of integer scores over 32 episodes, so it lands on a
    thirty-second and comparing against a rounded copy of it would put a policy
    on the wrong side of a rung it had just cleared.
    """

    #: What `md_agent_eval --skill` and `md_app --watch-scripted` call it.
    skill: str
    mean_score: float

    @property
    def label(self) -> str:
        """LOW / MEDIUM / HIGH — the ladder as the console and the HUD name it."""
        return self.skill.upper()


@dataclass(frozen=True)
class Ladder:
    """The scripted agent at three settings, measured on **one** seed block.

    A ladder belongs to the block it was played on and to no other. The two
    blocks are close but not equal — HIGH is 98,542 canonical and 98,170 on
    validation — so a score may only ever be read against the ladder from its
    own block. That is the whole reason this is a type rather than a module
    constant: the wrong ladder is a plausible-looking lie, and passing one
    around by hand is how it would get told.
    """

    #: The seed split it was measured on, in the words `evals.csv` uses.
    block: str
    #: Ascending, and one per `md::agent::Skill`.
    rungs: tuple[Baseline, ...]

    def __bool__(self) -> bool:
        """False for the empty ladder — "nothing here may be compared"."""
        return bool(self.rungs)


# Scripted Heuristic, Config defaults, 120,000-tick cap, decision interval /
# frame skip 4, over each block's 32 seeds. Both verified against md_agent_eval
# by the application-level evaluator test.
#
# Three rungs rather than one, in ascending order, because a single yardstick a
# learner cannot reach yet says nothing about progress: LOW is the first target,
# MEDIUM the normal one for a trained policy, HIGH the expert challenge. They
# are one agent at three settings (`md::agent::Skill`) — each step down removes
# one identifiable behaviour rather than tuning a magic number, so the gaps are
# attributable. See docs/ROADMAP.md.
CANONICAL_LADDER = Ladder(
    CANONICAL_SPLIT,
    (
        Baseline("low", 19_585.46875),
        Baseline("medium", 63_295.625),
        Baseline("high", 98_542.34375),
    ),
)

# The same three agents on the validation block, which is what a run scores
# itself against every `--eval-every` updates. Not a claim about anything — the
# published benchmark is the canonical one below — but a run spends hours
# producing validation scores and nothing else, and a chart that can say
# "MEDIUM, on this block" for those hours beats one that stays blank until a
# final benchmark exists.
VALIDATION_LADDER = Ladder(
    VALIDATION_SPLIT,
    (
        Baseline("low", 19_049.6875),
        Baseline("medium", 60_339.0625),
        Baseline("high", 98_170.15625),
    ),
)

#: Nothing measured under this row's protocol. Not an error — an honest refusal.
NO_LADDER = Ladder("", ())

#: **The** published baseline: the top canonical rung, and the number the docs
#: quote. Lower rungs and the validation block are progress markers, never "the
#: baseline" unqualified.
CANONICAL_BASELINE_MEAN_SCORE = CANONICAL_LADDER.rungs[-1].mean_score


def ladder_standing(score: float, ladder: Ladder) -> tuple[Baseline | None, Baseline | None]:
    """The highest rung ``score`` clears on ``ladder``, and the next to aim at.

    Either end can be ``None``, and both answers are information: nothing is
    cleared below LOW, and above HIGH there is nothing left to aim at.
    """

    cleared: Baseline | None = None
    for baseline in ladder.rungs:
        if score < baseline.mean_score:
            return cleared, baseline
        cleared = baseline
    return cleared, None


def canonical_baseline_comparable(
    *,
    seed_split: str | None,
    seed_offset: int | None,
    seed_count: int | None,
    frame_skip: int | None,
    max_ticks: int | None,
    inference_device: str | None,
) -> bool:
    """Whether a learned-policy row used the published canonical protocol."""

    return (
        seed_split == CANONICAL_SPLIT
        and seed_offset == CANONICAL_SEED_OFFSET
        and seed_count == SEEDS_PER_SPLIT
        and frame_skip == CANONICAL_FRAME_SKIP
        and max_ticks == CANONICAL_MAX_TICKS
        and inference_device == CANONICAL_INFERENCE_DEVICE
    )


def validation_ladder_comparable(
    *,
    seed_split: str | None,
    seed_offset: int | None,
    seed_count: int | None,
    frame_skip: int | None,
    max_ticks: int | None,
) -> bool:
    """Whether a row was scored on the block the validation ladder was played on.

    The cadence and the cap are checked because both change what a score *is*
    and a run may deliberately move them. The inference backend is deliberately
    **not**: it is part of the published canonical claim, but a validation score
    is a diagnostic, the scripted agent has no inference backend at all, and
    pinning CPU here would blank the ladder for every GPU run — which is nearly
    all of them.
    """

    return (
        seed_split == VALIDATION_SPLIT
        and seed_offset == VALIDATION_SEED_OFFSET
        and seed_count == SEEDS_PER_SPLIT
        # The protocol's cadence and cap, shared by both blocks; only the seeds
        # and the published-claim rules differ between them.
        and frame_skip == CANONICAL_FRAME_SKIP
        and max_ticks == CANONICAL_MAX_TICKS
    )


def ladder_for(
    *,
    seed_split: str | None,
    seed_offset: int | None,
    seed_count: int | None,
    frame_skip: int | None,
    max_ticks: int | None,
    inference_device: str | None,
) -> Ladder:
    """The ladder a row scored under this protocol may be read against.

    :data:`NO_LADDER` when the scripted agent has not been measured under it —
    an old row with no protocol recorded, a different seed count, a changed
    cadence. Drawing a rung there would compare two different measurements.
    """

    if canonical_baseline_comparable(
        seed_split=seed_split,
        seed_offset=seed_offset,
        seed_count=seed_count,
        frame_skip=frame_skip,
        max_ticks=max_ticks,
        inference_device=inference_device,
    ):
        return CANONICAL_LADDER
    if validation_ladder_comparable(
        seed_split=seed_split,
        seed_offset=seed_offset,
        seed_count=seed_count,
        frame_skip=frame_skip,
        max_ticks=max_ticks,
    ):
        return VALIDATION_LADDER
    return NO_LADDER
