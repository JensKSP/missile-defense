# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Promoting a policy into the league, and every way that must not half-happen.

The promise this module makes is narrow and load-bearing: **a league entry either
exists whole or does not exist.** A half-written one is a model the game finds and
refuses at the worst possible moment — mid-tournament, or when someone picks it
from a menu — so every test below is really about the state of the directory
after something went wrong.

The second promise is that promotion *converts*. What lands is an `.mdp`, not the
`.pt`, because the game and the native evaluator read that and never a pickle. A
checkpoint that cannot be exported cannot be promoted, and that refusal is the
intended outcome rather than a limitation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from md import league, policy_format

OBS = 8
ACTIONS = 5
HIDDEN = 4


def fixture_policy(name: str = "Fixture") -> policy_format.NativePolicy:
    rng = np.random.default_rng(7)

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
        metadata={"display_name": name},
    )


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """A real `.pt` in the shape `md.train.save_checkpoint` writes one."""
    torch = pytest.importorskip("torch", reason="torch is not installed")
    from md.ppo import Policy  # noqa: PLC0415 — optional dependency

    torch.manual_seed(1)
    net = Policy(OBS, ACTIONS, HIDDEN)
    path = tmp_path / "policy-00800.pt"
    torch.save(
        {
            "policy": net.state_dict(),
            "obs_size": OBS,
            "action_count": ACTIONS,
            "hidden": HIDDEN,
            "architecture": "mlp",
            "iteration": 800,
        },
        path,
    )
    return path


# ---- identity ----------------------------------------------------------------


def test_an_id_is_derived_from_the_name_so_a_path_is_readable() -> None:
    """Someone reads this in a shell prompt and an archive entry.

    `amber-anvil-800` tells them which model it is; a UUID does not, and the id
    is what every result, match and archive refers to.
    """
    assert league.make_id("Amber Anvil @800") == "amber-anvil-800"
    assert league.make_id("  spaced  out  ") == "spaced-out"


def test_ids_are_made_unique_rather_than_refused() -> None:
    """Promotion happens at the end of three hours of training.

    Failing it over a name collision would be poor manners; suffixing costs
    nothing and keeps the id readable.
    """
    assert league.make_id("run", ["run"]) == "run-2"
    assert league.make_id("run", ["run", "run-2"]) == "run-3"


def test_a_name_with_nothing_usable_in_it_still_produces_an_id() -> None:
    """An id is a directory name; an empty one would be a directory called ``.``."""
    assert league.make_id("!!!") == "model"
    assert league.make_id("") == "model"


# ---- promotion ---------------------------------------------------------------


def test_promotion_converts_the_checkpoint_and_the_game_can_read_it(
    tmp_path: Path, checkpoint: Path
) -> None:
    """What lands is an `.mdp`, never the `.pt`.

    The strongest single assertion here, because it is the one the rest of the
    system depends on: `policy_format.read` is the same reader
    `agent/src/policy.cpp` mirrors, so a file it accepts is one the game loads.
    """
    root = tmp_path / "models"
    model = league.promote(
        league.Promotion(checkpoint, "Amber Anvil @800", source_run="runs-7", trained_updates=800),
        root=root,
    )
    assert model.policy.is_file()
    assert model.policy.name == league.POLICY_NAME
    policy = policy_format.read(model.policy)
    assert policy.observation_size == OBS
    assert policy.action_count == ACTIONS
    # The display name travels *into* the file, which is where the game reads it
    # for the HUD and the league reads it for the table.
    assert policy.metadata["display_name"] == "Amber Anvil @800"


def test_a_promoted_model_records_where_it_came_from(tmp_path: Path, checkpoint: Path) -> None:
    """A model outlives its run, so the card is the only surviving provenance."""
    model = league.promote(
        league.Promotion(checkpoint, "Named", source_run="runs-7", trained_updates=800),
        root=tmp_path / "models",
    )
    assert model.source_run == "runs-7"
    assert model.source_checkpoint == "policy-00800.pt"
    assert model.trained_updates == 800
    assert model.promoted_at > 0


def test_a_promotion_that_fails_leaves_the_league_untouched(tmp_path: Path) -> None:
    """The whole promise, at its sharpest.

    A half-written entry is a model the game finds and refuses mid-tournament.
    So the export happens in a staging directory beside the destination and is
    renamed in only after the result has been read back.
    """
    root = tmp_path / "models"
    junk = tmp_path / "not-a-checkpoint.pt"
    junk.write_text("nope", encoding="utf-8")
    pytest.importorskip("torch", reason="torch is not installed")

    with pytest.raises(league.LeagueError):
        league.promote(league.Promotion(junk, "Doomed"), root=root)

    assert league.models(root) == []
    # And nothing was left lying about for the next promotion to trip over.
    assert not list(root.glob(".*incoming"))


def test_an_architecture_the_game_cannot_run_is_refused(tmp_path: Path) -> None:
    """`entity` trains and has no native forward pass.

    Promoting one would produce a league entry the game accepts and then plays
    as noise, which is worse than a refusal by exactly the amount of time it
    takes somebody to work out why their agent is bad.
    """
    torch = pytest.importorskip("torch", reason="torch is not installed")
    from md.ppo import Policy  # noqa: PLC0415 — optional dependency

    path = tmp_path / "entity.pt"
    torch.save(
        {
            "policy": Policy(OBS, ACTIONS, HIDDEN).state_dict(),
            "obs_size": OBS,
            "action_count": ACTIONS,
            "hidden": HIDDEN,
            "architecture": "entity",
            "iteration": 550,
        },
        path,
    )
    with pytest.raises(league.LeagueError, match="entity"):
        league.promote(league.Promotion(path, "Relational"), root=tmp_path / "models")
    assert league.models(tmp_path / "models") == []


def test_promoting_the_same_run_twice_produces_two_entries(
    tmp_path: Path, checkpoint: Path
) -> None:
    """A run gets promoted again after another thousand updates, and both stay."""
    root = tmp_path / "models"
    first = league.promote(league.Promotion(checkpoint, "Anvil"), root=root)
    second = league.promote(league.Promotion(checkpoint, "Anvil"), root=root)
    assert first.model_id != second.model_id
    assert {model.model_id for model in league.models(root)} == {
        first.model_id,
        second.model_id,
    }


# ---- listing -----------------------------------------------------------------


def test_a_directory_with_a_card_and_no_weights_is_not_a_model(tmp_path: Path) -> None:
    """What an interrupted promotion would have left, before it was made atomic.

    Listing it would put a row on screen that nothing can play.
    """
    root = tmp_path / "models"
    (root / "ghost").mkdir(parents=True)
    (root / "ghost" / league.CARD_NAME).write_text('{"display_name": "Ghost"}', encoding="utf-8")
    assert league.models(root) == []
    assert league.find("ghost", root) is None


def test_an_empty_league_is_empty_and_not_an_error(tmp_path: Path) -> None:
    assert league.models(tmp_path / "never-created") == []


# ---- naming ------------------------------------------------------------------


def test_renaming_a_model_keeps_its_id(tmp_path: Path, checkpoint: Path) -> None:
    """Every result, match and archive refers to the id; only humans read names."""
    root = tmp_path / "models"
    model = league.promote(league.Promotion(checkpoint, "Working Title"), root=root)
    renamed = league.rename(model, "Amber Anvil")
    assert renamed.model_id == model.model_id
    assert renamed.display_name == "Amber Anvil"
    assert league.find(model.model_id, root) is not None


def test_a_model_with_no_display_name_shows_its_id(tmp_path: Path) -> None:
    root = tmp_path / "models"
    (root / "bare").mkdir(parents=True)
    policy_format.write(root / "bare" / league.POLICY_NAME, fixture_policy())
    (root / "bare" / league.CARD_NAME).write_text("{}", encoding="utf-8")
    model = league.find("bare", root)
    assert model is not None
    assert model.name == "bare"


# ---- importing ---------------------------------------------------------------


def test_an_imported_policy_is_validated_before_anything_is_written(tmp_path: Path) -> None:
    """*More* suspect than a promotion, not less — this file may be a download."""
    root = tmp_path / "models"
    junk = tmp_path / "downloaded.mdp"
    junk.write_bytes(b"MDPOLICY" + bytes(40))
    with pytest.raises(league.LeagueError):
        league.import_policy(junk, "Someone Else's", root=root)
    assert league.models(root) == []


def test_a_valid_policy_can_be_imported_and_named(tmp_path: Path) -> None:
    source = tmp_path / "shared.mdp"
    policy_format.write(source, fixture_policy("Their Model"))
    model = league.import_policy(source, "Their Model", root=tmp_path / "models")
    assert model.display_name == "Their Model"
    assert model.policy.is_file()
    assert policy_format.read(model.policy).action_count == ACTIONS


def test_an_import_with_no_name_given_takes_the_one_in_the_file(tmp_path: Path) -> None:
    source = tmp_path / "shared.mdp"
    policy_format.write(source, fixture_policy("Carried Name"))
    model = league.import_policy(source, "", root=tmp_path / "models")
    assert model.display_name == "Carried Name"


# ---- results -----------------------------------------------------------------


def test_results_accumulate_rather_than_replace(tmp_path: Path) -> None:
    """A model's results are its record.

    Overwriting the previous one would make "has this got better since?"
    unanswerable, which is the question a league exists to answer.
    """
    root = tmp_path / "models"
    (root / "m").mkdir(parents=True)
    policy_format.write(root / "m" / league.POLICY_NAME, fixture_policy())
    (root / "m" / league.CARD_NAME).write_text('{"display_name": "M"}', encoding="utf-8")
    model = league.find("m", root)
    assert model is not None

    model = league.record_result(model, {"mean_score": 1000.0, "canonical": True})
    model = league.record_result(model, {"mean_score": 2000.0, "canonical": True})
    assert len(model.results) == 2
    best = model.best_result
    assert best is not None
    assert best["mean_score"] == 2000.0


def test_only_canonical_results_are_ranked(tmp_path: Path) -> None:
    """A quick match is not a benchmark, and must not top the table.

    Task 7 enforces the whole fairness protocol; this is the flag it hangs on,
    checked here so a result written without it can never be mistaken for one.
    """
    root = tmp_path / "models"
    (root / "m").mkdir(parents=True)
    policy_format.write(root / "m" / league.POLICY_NAME, fixture_policy())
    (root / "m" / league.CARD_NAME).write_text("{}", encoding="utf-8")
    model = league.find("m", root)
    assert model is not None

    model = league.record_result(model, {"mean_score": 99_999.0, "canonical": False})
    assert model.best_result is None  # a quick run cannot rank

    model = league.record_result(model, {"mean_score": 100.0, "canonical": True})
    best = model.best_result
    assert best is not None
    assert best["mean_score"] == 100.0


def test_a_recorded_result_is_written_atomically(tmp_path: Path) -> None:
    root = tmp_path / "models"
    (root / "m").mkdir(parents=True)
    policy_format.write(root / "m" / league.POLICY_NAME, fixture_policy())
    (root / "m" / league.CARD_NAME).write_text("{}", encoding="utf-8")
    model = league.find("m", root)
    assert model is not None
    league.record_result(model, {"mean_score": 1.0, "canonical": True})
    assert not list((root / "m").glob("*.tmp"))
    stored = json.loads((root / "m" / league.RESULTS_NAME).read_text(encoding="utf-8"))
    assert stored["results"][0]["recorded_at"] > 0
