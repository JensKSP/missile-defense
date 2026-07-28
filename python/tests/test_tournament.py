# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The contest is fair and complete, or it is not recorded.

Three rules, and each has a test that fails loudly if it stops holding:

1. every contestant gets the **identical** seed list, not merely the same
   protocol;
2. a ranking appears only when **every** seed is in;
3. only a **complete canonical** evaluation ranks.

The evaluation itself is stubbed here. Playing 32 seeds twice takes minutes and
proves something `test_promotion.py` and `test_parity.py` already prove — that
the policy plays. What is under test is the *orchestration*, and the orchestration
is exactly the part that can be unfair while every piece it calls is correct.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from missile_defense import benchmark, league, policy_format, tournament

OBS = 6
ACTIONS = 4
HIDDEN = 3


def fixture_policy() -> policy_format.NativePolicy:
    rng = np.random.default_rng(3)

    def normal(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.1).astype(np.float32)

    return policy_format.NativePolicy(
        schema=policy_format.SCHEMA,
        observation_size=OBS,
        action_count=ACTIONS,
        architecture="mlp",
        tensors=(
            policy_format.Tensor("trunk.0.weight", (HIDDEN, OBS), normal(HIDDEN, OBS)),
            policy_format.Tensor("trunk.0.bias", (HIDDEN,), normal(HIDDEN)),
            policy_format.Tensor("trunk.2.weight", (HIDDEN, HIDDEN), normal(HIDDEN, HIDDEN)),
            policy_format.Tensor("trunk.2.bias", (HIDDEN,), normal(HIDDEN)),
            policy_format.Tensor("policy_head.weight", (ACTIONS, HIDDEN), normal(ACTIONS, HIDDEN)),
            policy_format.Tensor("policy_head.bias", (ACTIONS,), normal(ACTIONS)),
            policy_format.Tensor("value_head.weight", (1, HIDDEN), normal(1, HIDDEN)),
            policy_format.Tensor("value_head.bias", (1,), normal(1)),
        ),
        metadata={},
    )


def make_model(root: Path, model_id: str, name: str) -> league.Model:
    directory = root / model_id
    directory.mkdir(parents=True, exist_ok=True)
    policy_format.write(directory / league.POLICY_NAME, fixture_policy())
    (directory / league.CARD_NAME).write_text(f'{{"display_name": "{name}"}}', encoding="utf-8")
    model = league.find(model_id, root)
    assert model is not None
    return model


# ---- the protocol ------------------------------------------------------------


def test_only_the_published_protocol_is_canonical() -> None:
    assert tournament.canonical_protocol().canonical
    assert not tournament.quick_protocol().canonical


def test_a_quick_match_is_short_enough_to_be_worth_having() -> None:
    """It exists to answer "is this obviously worse?" while you wait.

    A quick match that used the canonical 120,000-tick cap would take as long as
    the benchmark and there would be no reason for it to exist.
    """
    quick = tournament.quick_protocol()
    assert quick.seed_count < benchmark.SEEDS_PER_SPLIT
    assert quick.max_ticks < benchmark.CANONICAL_MAX_TICKS


def test_changing_any_part_of_the_protocol_stops_it_ranking() -> None:
    """`missile_defense.benchmark` owns what canonical means; this asks rather than guesses."""
    base = tournament.canonical_protocol()
    for field, value in (
        ("seed_offset", 0),
        ("seed_count", 8),
        ("frame_skip", 1),
        ("max_ticks", 1000),
        ("inference_device", "cuda"),
    ):
        import dataclasses  # noqa: PLC0415

        assert not dataclasses.replace(base, **{field: value}).canonical, field


# ---- rule 1: the same seeds ---------------------------------------------------


@pytest.mark.usefixtures("_native")
def test_both_contestants_are_handed_the_identical_seed_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not "the same protocol" — the same list.

    Two derivations of a seed set are two chances to differ, and a contest where
    one side played a different thirty-two seeds is not a contest. Recorded here
    by capturing what each call actually received.
    """
    root = tmp_path / "models"
    left = make_model(root, "left", "Left")
    right = make_model(root, "right", "Right")

    handed: list[tuple[int, ...]] = []

    def fake(model, protocol, *, seeds=None, **_):  # noqa: ANN001, ANN202
        handed.append(tuple(seeds or ()))
        return tournament.Result(
            model_id=model.model_id,
            display_name=model.name,
            protocol=protocol,
            mean_score=1.0,
            mean_wave=1.0,
            mean_ticks=1.0,
            episodes=len(seeds or ()),
            canonical=True,
            versus_baseline=0.0,
        )

    monkeypatch.setattr(tournament, "evaluate_model", fake)
    match = tournament.head_to_head(left, right, record=False)
    assert len(handed) == 2
    assert handed[0] == handed[1]
    assert handed[0] == match.seeds
    assert len(handed[0]) == benchmark.SEEDS_PER_SPLIT


# ---- rule 2: complete, or nothing ---------------------------------------------


def test_a_partial_evaluation_is_not_canonical() -> None:
    """Fewer episodes than seeds means something did not finish.

    However it happened — a crash, a cancellation, an environment that never
    reported done — the result must not rank, because it is a different (and
    probably easier) sample of the seed set.
    """
    protocol = tournament.canonical_protocol()
    partial = tournament.Result(
        model_id="m",
        display_name="M",
        protocol=protocol,
        mean_score=200_000.0,
        mean_wave=30.0,
        mean_ticks=1.0,
        episodes=protocol.seed_count - 1,
        canonical=False,  # what `evaluate_model` computes for a short count
        versus_baseline=None,
    )
    assert not partial.canonical
    assert partial.as_record()["canonical"] is False


@pytest.mark.usefixtures("_native")
def test_a_cancelled_match_records_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancel during the second contestant; the first must not be written either.

    That is why both `record_result` calls are at the bottom of `head_to_head`
    rather than after each side — a half-recorded match would leave one model
    holding a result from a contest that never completed.
    """
    root = tmp_path / "models"
    left = make_model(root, "left", "Left")
    right = make_model(root, "right", "Right")

    calls = {"n": 0}

    def fake(model, protocol, *, seeds=None, **_):  # noqa: ANN001, ANN202
        calls["n"] += 1
        if calls["n"] == 2:
            raise tournament.Cancelled
        return tournament.Result(
            model_id=model.model_id,
            display_name=model.name,
            protocol=protocol,
            mean_score=1.0,
            mean_wave=1.0,
            mean_ticks=1.0,
            episodes=len(seeds or ()),
            canonical=True,
            versus_baseline=0.0,
        )

    monkeypatch.setattr(tournament, "evaluate_model", fake)
    with pytest.raises(tournament.Cancelled):
        tournament.head_to_head(left, right)

    assert league.find("left", root).results == ()  # type: ignore[union-attr]
    assert league.find("right", root).results == ()  # type: ignore[union-attr]


# ---- rule 3: only canonical ranks ---------------------------------------------


def test_a_quick_result_never_tops_the_table(tmp_path: Path) -> None:
    """A four-seed warm-up above a thirty-two-seed benchmark is the failure."""
    root = tmp_path / "models"
    quick = make_model(root, "quick", "Quick")
    proper = make_model(root, "proper", "Proper")

    league.record_result(quick, {"mean_score": 200_000.0, "canonical": False})
    league.record_result(proper, {"mean_score": 50_000.0, "canonical": True})

    table = tournament.rank(league.models(root))
    assert [model.model_id for model, _ in table] == ["proper", "quick"]
    assert table[0][1] == 50_000.0
    assert table[1][1] is None  # unranked, not "200,000"


def test_a_model_with_no_results_sorts_last_without_a_score(tmp_path: Path) -> None:
    root = tmp_path / "models"
    make_model(root, "fresh", "Fresh")
    scored = make_model(root, "scored", "Scored")
    league.record_result(scored, {"mean_score": 1.0, "canonical": True})
    table = tournament.rank(league.models(root))
    assert table[0][0].model_id == "scored"
    assert table[-1][1] is None


# ---- the result record --------------------------------------------------------


def test_a_result_carries_the_protocol_that_produced_it() -> None:
    """A score without its protocol cannot be compared with anything."""
    protocol = tournament.canonical_protocol()
    record = tournament.Result(
        model_id="m",
        display_name="M",
        protocol=protocol,
        mean_score=100_000.0,
        mean_wave=16.0,
        mean_ticks=120_000.0,
        episodes=protocol.seed_count,
        canonical=True,
        versus_baseline=100_000.0 - benchmark.CANONICAL_BASELINE_MEAN_SCORE,
    ).as_record()
    stored = record["protocol"]
    assert stored["seed_offset"] == benchmark.CANONICAL_SEED_OFFSET  # type: ignore[index]
    assert stored["max_ticks"] == benchmark.CANONICAL_MAX_TICKS  # type: ignore[index]
    assert record["versus_baseline"] > 0  # type: ignore[operator]


def test_a_quick_result_has_no_comparison_with_the_baseline() -> None:
    """A number beside a yardstick it was not measured against is worse than none."""
    result = tournament.Result(
        model_id="m",
        display_name="M",
        protocol=tournament.quick_protocol(),
        mean_score=100_000.0,
        mean_wave=16.0,
        mean_ticks=1.0,
        episodes=tournament.QUICK_SEEDS,
        canonical=False,
        versus_baseline=None,
    )
    assert result.as_record()["versus_baseline"] is None


# ---- matches -----------------------------------------------------------------


def _result(model_id: str, score: float, *, canonical: bool = True) -> tournament.Result:
    return tournament.Result(
        model_id=model_id,
        display_name=model_id.title(),
        protocol=tournament.canonical_protocol(),
        mean_score=score,
        mean_wave=1.0,
        mean_ticks=1.0,
        episodes=benchmark.SEEDS_PER_SPLIT,
        canonical=canonical,
        versus_baseline=None,
    )


def test_the_winner_is_the_higher_score_and_a_tie_has_none() -> None:
    higher = tournament.Match(_result("a", 2.0), _result("b", 1.0), (1, 2))
    assert higher.winner is not None
    assert higher.winner.model_id == "a"
    assert tournament.Match(_result("a", 1.0), _result("b", 1.0), (1, 2)).winner is None


def test_a_match_only_counts_when_both_sides_were_canonical() -> None:
    assert tournament.Match(_result("a", 2.0), _result("b", 1.0), (1,)).ranked
    assert not tournament.Match(_result("a", 2.0), _result("b", 1.0, canonical=False), (1,)).ranked


def test_a_manifest_names_both_sides_and_the_scores_it_claims(tmp_path: Path) -> None:
    """So the spectator can assert what it renders is what was scored.

    A split screen quietly showing a different episode than the table claims
    would be worse than no split screen.
    """
    import json  # noqa: PLC0415

    match = tournament.Match(_result("a", 2.0), _result("b", 1.0), (11, 22))
    written = tournament.write_manifest(
        match,
        tmp_path / "match.json",
        {"left": tmp_path / "a.mdr", "right": tmp_path / "b.mdr"},
    )
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["seeds"] == [11, 22]
    assert payload["left"]["display_name"] == "A"
    assert payload["left"]["mean_score"] == 2.0
    assert payload["right"]["recording"].endswith("b.mdr")
    assert payload["ranked"] is True
    assert not list(tmp_path.glob("*.tmp"))


@pytest.fixture
def _native() -> None:
    pytest.importorskip("missile_defense._md_native", reason="the native binding is not built")


# ---- paired recordings -------------------------------------------------------


def test_a_pair_can_only_be_recorded_on_a_seed_the_match_was_played_on(tmp_path: Path) -> None:
    # A recording of some other seed would not be an episode either score was
    # measured over, which is the one thing the manifest promises a viewer.
    left = make_model(tmp_path, "aaaa", "Amber Anvil")
    right = make_model(tmp_path, "bbbb", "Brisk Harbour")
    protocol = tournament.canonical_protocol()
    match = tournament.Match(
        tournament.Result(left.model_id, left.name, protocol, 100.0, 1.0, 60.0, 1, True, None),
        tournament.Result(right.model_id, right.name, protocol, 90.0, 1.0, 60.0, 1, True, None),
        (7, 8, 9),
    )
    with pytest.raises(tournament.TournamentError, match="not one this match"):
        tournament.record_pair(match, tmp_path / "out", seed=1234)


def test_recording_a_pair_says_which_side_went_missing(tmp_path: Path) -> None:
    # Models can be deleted between a match being played and a recording of it
    # being asked for. Naming the side is the difference between a fixable
    # message and a puzzle — and it is checked before anything is written, so a
    # match with one live side does not leave half a pair on disk.
    right = make_model(tmp_path, "bbbb", "Brisk Harbour")
    protocol = tournament.canonical_protocol()
    match = tournament.Match(
        tournament.Result("gone", "Vanished", protocol, 100.0, 1.0, 60.0, 1, True, None),
        tournament.Result(right.model_id, right.name, protocol, 90.0, 1.0, 60.0, 1, True, None),
        (7,),
    )
    with pytest.raises(tournament.TournamentError, match="left model"):
        tournament.record_pair(match, tmp_path / "out", root=tmp_path)
    assert not (tmp_path / "out").exists()
