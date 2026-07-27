# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The human handicap, and the rule that there is only one of it.

The handicap is defined twice — `md::protocol::aim_trail` in C++ and
`md.benchmark.CANONICAL_AIM_TRAIL` in Python — because the trainer and the
console never read the header, and the game never reads the Python. Two copies
are fine; two copies that disagree are a ladder whose rungs mean different things
depending on which program you asked, which is the failure this file exists to
make impossible.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from md.benchmark import (
    CANONICAL_AIM_TRAIL,
    CANONICAL_FRAME_SKIP,
    CANONICAL_INFERENCE_DEVICE,
    CANONICAL_MAX_TICKS,
    CANONICAL_REACTION_DELAY,
    CANONICAL_SEED_OFFSET,
    CANONICAL_SPLIT,
    SEEDS_PER_SPLIT,
    canonical_baseline_comparable,
)

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "core" / "include" / "md" / "protocol.hpp"


def _cpp_constant() -> float:
    """The C++ side's value, read from the generated header.

    Deliberately textual, and deliberately not through the built extension: the
    question is whether the two *files* agree, and a test that imported a binding
    would pass against a stale build and fail confusingly against a missing one.
    """
    source = HEADER.read_text(encoding="utf-8")
    match = re.search(r"aim_trail\s*=\s*([0-9.]+)f?\s*;", source)
    assert match is not None, f"no aim_trail in {HEADER}"
    return float(match.group(1))


def test_both_generated_copies_of_the_handicap_agree() -> None:
    assert _cpp_constant() == CANONICAL_AIM_TRAIL


def test_the_generated_files_match_their_source() -> None:
    """The real guarantee: neither copy can be edited into disagreement.

    `protocol.toml` is the source and both files are generated from it, so this
    is what makes the duplication mechanical instead of remembered. It is the
    same check `poe check` runs; having it here too means a plain `poe pytest`
    catches a hand-edited constant.
    """
    from tools.protocol import main as generate

    assert generate(["--check"]) == 0, "run `poe protocol` and commit the result"


def _protocol(**overrides: object) -> bool:
    fields: dict[str, object] = {
        "seed_split": CANONICAL_SPLIT,
        "seed_offset": CANONICAL_SEED_OFFSET,
        "seed_count": SEEDS_PER_SPLIT,
        "frame_skip": CANONICAL_FRAME_SKIP,
        "max_ticks": CANONICAL_MAX_TICKS,
        "inference_device": CANONICAL_INFERENCE_DEVICE,
        "aim_trail": CANONICAL_AIM_TRAIL,
        "reaction_delay": CANONICAL_REACTION_DELAY,
    }
    fields.update(overrides)
    return canonical_baseline_comparable(**fields)  # type: ignore[arg-type]


def test_the_canonical_protocol_includes_the_handicap() -> None:
    assert _protocol()
    # Both halves, because a score earned with only one of them is a score from
    # a different game.
    assert not _protocol(aim_trail=0.0)
    assert not _protocol(aim_trail=CANONICAL_AIM_TRAIL + 0.01)
    assert not _protocol(reaction_delay=0)
    assert not _protocol(reaction_delay=CANONICAL_REACTION_DELAY + 1)


def test_the_league_plays_under_the_handicap_it_records() -> None:
    """The half that was missing, and could not be seen from either side alone.

    `md_agent_eval` wraps its driver in `HandicappedDriver` by default; the
    binding the league scores through built a bare `PolicyDriver` and did not.
    So a league result carried `aim_trail` and `reaction_delay` in its protocol
    and had been measured under neither — two implementations of "play a seed"
    that the docstring on each claimed were the same code.

    A bundled model and a short cap: what is asserted is which driver is used,
    not what the model scores.
    """
    pytest.importorskip(
        "md._md_native",
        reason="the _md_native extension is not built (cmake -DMD_BUILD_BINDINGS=ON)",
    )
    from md._md_native import LoadedPolicy  # noqa: PLC0415 — the native binding

    model = ROOT / "models" / "learned-high.mdp"
    if not model.exists():
        pytest.skip(f"{model.name} is not in this checkout")
    policy = LoadedPolicy(str(model))
    seed = 10_791_363_909_057_970_626  # the canonical block's first

    default = policy.play(seed, 1_200, CANONICAL_FRAME_SKIP).score
    handicapped = policy.play(
        seed,
        1_200,
        CANONICAL_FRAME_SKIP,
        CANONICAL_AIM_TRAIL,
        CANONICAL_REACTION_DELAY,
    ).score
    unhandicapped = policy.play(seed, 1_200, CANONICAL_FRAME_SKIP, 0.0, 0).score

    assert default == handicapped, "the default is not the published protocol"
    assert default != unhandicapped, "the handicap is being asked for and ignored"


def test_a_score_from_before_the_handicap_is_not_assumed_comparable() -> None:
    """The rule that keeps old numbers out of new tables.

    Every evaluation written before the handicap existed carries no reaction
    delay at all. Those runs were scored against an agent that never mis-clicked
    and never forgot a shot; treating a missing field as "probably canonical"
    would put a 90,866 beside a 9,000 as though they answered the same question.
    """
    assert not _protocol(aim_trail=None)
    assert not _protocol(reaction_delay=None)
