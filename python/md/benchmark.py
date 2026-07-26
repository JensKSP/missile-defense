# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: OpenAI Codex
"""Authoritative metadata for policy evaluation.

Training-time validation and the final benchmark deliberately use different seed
splits.  Keeping the protocol constants here gives the trainer and the console one
lightweight source of truth without making either import the other (or PyTorch).
"""

from __future__ import annotations

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

# Scripted Heuristic, canonical seed block, Config defaults, 120,000-tick cap,
# decision interval / frame skip 4. Verified against md_agent_eval by the
# application-level evaluator test.
CANONICAL_BASELINE_MEAN_SCORE = 98_542.34375


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
