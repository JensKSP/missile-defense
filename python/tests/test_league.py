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
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from missile_defense import league, policy_format

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
    """A real `.pt` in the shape `missile_defense.train.save_checkpoint` writes one."""
    torch = pytest.importorskip("torch", reason="torch is not installed")
    from missile_defense.ppo import Policy  # noqa: PLC0415 — optional dependency

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


def test_promoting_with_no_torch_here_says_where_torch_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trainer has no torch, on purpose, and promotion is where that bites.

    Everything else about a run is read from the files it left — which is what
    lets a trainer with no CUDA anywhere near it watch and compare runs — but a
    `.pt` cannot be opened without torch. From a packaged trainer this raised
    `ModuleNotFoundError` out of a Qt slot, which is a button that does nothing.
    A sentence naming the way out is the least it can do.
    """
    from missile_defense import export_policy  # noqa: PLC0415

    def no_torch(*_args: object, **_kwargs: object) -> Path:
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(export_policy, "export_checkpoint", no_torch)
    junk = tmp_path / "policy.pt"
    junk.write_text("nope", encoding="utf-8")

    with pytest.raises(league.LeagueError) as raised:
        league.promote(league.Promotion(junk, "Doomed"), root=tmp_path / "models")
    assert "training runtime" in str(raised.value)
    assert not list((tmp_path / "models").glob(".*incoming"))


def test_promoting_borrows_the_interpreter_it_is_given(tmp_path: Path) -> None:
    """The way out: export in the interpreter that *does* have torch.

    The same one a run is started with. What is asserted is the command and the
    environment, because that is what has to be right for a *bare venv* to run
    this: the runtime has torch and nothing else, `missile_defense` lives in this checkout or
    this installation, so without `PYTHONPATH` the spawned process cannot import
    the module it was told to run. Stood in for by a shell shim, so this needs
    neither torch nor a real checkpoint.
    """
    if os.name == "nt":
        pytest.skip("the shim is a shell script")

    record = tmp_path / "argv"
    shim = tmp_path / "python-shim"
    shim.write_text(
        f'#!/bin/sh\n{{ echo "$@"; echo "PYTHONPATH=$PYTHONPATH"; }} > "{record}"\n'
        'echo "the runtime said no" >&2\nexit 1\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    checkpoint = tmp_path / "policy-00800.pt"
    checkpoint.write_text("not read by the shim", encoding="utf-8")

    with pytest.raises(league.LeagueError) as raised:
        league.promote(
            league.Promotion(checkpoint, "Borrowed"), root=tmp_path / "models", python=str(shim)
        )
    # Whatever the exporter said, said back — it is the sentence written for a
    # person, and wrapping it in one of ours would bury it.
    assert "the runtime said no" in str(raised.value)

    argv, pythonpath = record.read_text(encoding="utf-8").splitlines()
    assert argv.startswith(f"-m missile_defense.export_policy {checkpoint} ")
    assert '"display_name": "Borrowed"' in argv
    package_root = str(Path(league.__file__).resolve().parents[1])
    assert package_root in pythonpath.removeprefix("PYTHONPATH=").split(os.pathsep)
    assert not list((tmp_path / "models").glob(".*incoming"))


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
    from missile_defense.ppo import Policy  # noqa: PLC0415 — optional dependency

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


def test_promoting_the_same_run_twice_keeps_both_under_different_names(
    tmp_path: Path, checkpoint: Path
) -> None:
    """A run gets promoted again after another thousand updates, and both stay."""
    root = tmp_path / "models"
    first = league.promote(league.Promotion(checkpoint, "Anvil @800"), root=root)
    second = league.promote(league.Promotion(checkpoint, "Anvil @1800"), root=root)
    assert first.model_id != second.model_id
    assert {model.model_id for model in league.models(root)} == {
        first.model_id,
        second.model_id,
    }


# ---- unique names ------------------------------------------------------------


def test_a_name_already_in_the_league_is_refused_and_says_which_model(
    tmp_path: Path, checkpoint: Path
) -> None:
    """The name is all anyone sees — in the table, and in the game's MODELS menu.

    Two rows called `Anvil` cannot be told apart, so the second promotion stops
    and hands back the entry it collided with; deciding what to do about it is
    the caller's, and it needs to know what is already there.
    """
    root = tmp_path / "models"
    first = league.promote(league.Promotion(checkpoint, "Anvil"), root=root)

    with pytest.raises(league.DuplicateName) as raised:
        league.promote(league.Promotion(checkpoint, "  anvil  "), root=root)

    assert raised.value.existing.model_id == first.model_id
    assert [model.model_id for model in league.models(root)] == [first.model_id]


def test_a_refused_name_costs_nothing_and_leaves_no_directory(
    tmp_path: Path, checkpoint: Path
) -> None:
    """Refused before the checkpoint is read, so retyping the name is the whole fix."""
    root = tmp_path / "models"
    league.promote(league.Promotion(checkpoint, "Anvil"), root=root)
    with pytest.raises(league.DuplicateName):
        league.promote(league.Promotion(checkpoint, "Anvil"), root=root)
    assert sorted(child.name for child in root.iterdir()) == ["anvil"]


def test_replacing_keeps_the_id_and_drops_the_results(tmp_path: Path, checkpoint: Path) -> None:
    """Overwrite is a real swap, not a second entry with the same name.

    The id stays so a path somebody wrote down still resolves; the results go,
    because they were measured on the weights that just left.
    """
    root = tmp_path / "models"
    first = league.promote(league.Promotion(checkpoint, "Anvil"), root=root)
    league.record_result(first, {"mean_score": 1234.0, "canonical": True})

    replaced = league.promote(
        league.Promotion(checkpoint, "Anvil", note="second attempt"),
        root=root,
        replace=first,
    )
    assert replaced.model_id == first.model_id
    assert replaced.note == "second attempt"
    assert replaced.results == ()  # the old scores belonged to the old weights
    assert [model.model_id for model in league.models(root)] == [first.model_id]
    assert not list(root.glob(".*"))  # nothing left staged or superseded


def test_replacing_a_model_that_is_gone_is_refused(tmp_path: Path, checkpoint: Path) -> None:
    """Two trainers, and the other one deleted it. Better an error than a ghost."""
    root = tmp_path / "models"
    model = league.promote(league.Promotion(checkpoint, "Anvil"), root=root)
    shutil.rmtree(model.path)
    with pytest.raises(league.LeagueError, match="no longer in the league"):
        league.promote(league.Promotion(checkpoint, "Anvil"), root=root, replace=model)


def test_replacing_cannot_take_a_third_models_name(tmp_path: Path, checkpoint: Path) -> None:
    """Renaming onto someone else's name while replacing would recreate the duplicate."""
    root = tmp_path / "models"
    anvil = league.promote(league.Promotion(checkpoint, "Anvil"), root=root)
    beacon = league.promote(league.Promotion(checkpoint, "Beacon"), root=root)
    with pytest.raises(league.DuplicateName) as raised:
        league.promote(league.Promotion(checkpoint, "Beacon"), root=root, replace=anvil)
    assert raised.value.existing.model_id == beacon.model_id
    assert len(league.models(root)) == 2


def test_a_staging_directory_is_never_listed_as_a_model(tmp_path: Path) -> None:
    """A promotion in flight is a whole, loadable model that is not in the league."""
    root = tmp_path / "models"
    staging = root / ".anvil.incoming"
    staging.mkdir(parents=True)
    policy_format.write(staging / league.POLICY_NAME, fixture_policy())
    (staging / league.CARD_NAME).write_text('{"display_name": "Anvil"}', encoding="utf-8")
    assert league.models(root) == []
    assert league.find_by_name("Anvil", root) is None


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


def test_renaming_onto_another_models_name_is_refused(tmp_path: Path, checkpoint: Path) -> None:
    """The back door into duplicate names, closed where promotion's is."""
    root = tmp_path / "models"
    anvil = league.promote(league.Promotion(checkpoint, "Anvil"), root=root)
    beacon = league.promote(league.Promotion(checkpoint, "Beacon"), root=root)
    with pytest.raises(league.DuplicateName) as raised:
        league.rename(beacon, "anvil")
    assert raised.value.existing.model_id == anvil.model_id
    assert league.find(beacon.model_id, root) is not None
    assert (league.find(beacon.model_id, root) or beacon).display_name == "Beacon"


def test_renaming_a_model_to_the_name_it_already_has_is_allowed(
    tmp_path: Path, checkpoint: Path
) -> None:
    """It is not a collision with itself, and a dialog prefilled with the current
    name must not refuse the obvious Enter."""
    root = tmp_path / "models"
    model = league.promote(league.Promotion(checkpoint, "Anvil"), root=root)
    assert league.rename(model, "Anvil").display_name == "Anvil"


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


def test_matches_live_beside_the_league_not_inside_a_model(tmp_path: Path) -> None:
    # A match belongs to two models. Under either one, deleting that model would
    # take the comparison with it — including the half that is about the model
    # still there.
    matches = league.matches_dir(tmp_path / "models")
    assert matches == tmp_path / "matches"
    assert matches.parent == (tmp_path / "models").parent


# ---- removing ----------------------------------------------------------------


def test_deleting_a_model_removes_it_from_the_league_and_the_game(tmp_path: Path) -> None:
    """The league directory *is* the game's MODELS menu, so this is one act.

    There is no second copy anywhere: what is deleted here is gone from both.
    """
    root = tmp_path / "models"
    (root / "m").mkdir(parents=True)
    policy_format.write(root / "m" / league.POLICY_NAME, fixture_policy())
    (root / "m" / league.CARD_NAME).write_text('{"display_name": "M"}', encoding="utf-8")
    model = league.find("m", root)
    assert model is not None

    freed = league.delete(model, root)
    assert freed > 0
    assert league.models(root) == []
    assert not (root / "m").exists()
    assert root.is_dir()  # the league itself survives losing its last model


def test_deleting_leaves_the_recorded_matches_alone(tmp_path: Path) -> None:
    """A match is evidence about *two* models; one leaving must not take it."""
    root = tmp_path / "models"
    (root / "m").mkdir(parents=True)
    policy_format.write(root / "m" / league.POLICY_NAME, fixture_policy())
    (root / "m" / league.CARD_NAME).write_text("{}", encoding="utf-8")
    match = league.matches_dir(root) / "m-other"
    match.mkdir(parents=True)
    (match / "match.json").write_text("{}", encoding="utf-8")

    model = league.find("m", root)
    assert model is not None
    league.delete(model, root)
    assert (match / "match.json").is_file()


def test_deleting_refuses_anything_that_is_not_an_entry_in_this_league(
    tmp_path: Path,
) -> None:
    """`rmtree` on a path that arrived from elsewhere is how other programs have
    deleted a home directory. Both refusals are checked before any of it goes."""
    root = tmp_path / "models"
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "m").mkdir(parents=True)
    policy_format.write(elsewhere / "m" / league.POLICY_NAME, fixture_policy())
    (elsewhere / "m" / league.CARD_NAME).write_text("{}", encoding="utf-8")
    outsider = league.find("m", elsewhere)
    assert outsider is not None

    with pytest.raises(league.LeagueError, match="refusing to remove"):
        league.delete(outsider, root)
    assert (elsewhere / "m").is_dir()

    root.mkdir(parents=True)
    itself = league.Model(path=root, model_id=root.name, display_name="")
    with pytest.raises(league.LeagueError, match="league directory itself"):
        league.delete(itself, root)
    assert root.is_dir()
