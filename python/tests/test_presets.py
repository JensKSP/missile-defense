# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Named training-option sets: the three that ship, and the ones you save.

No Qt in :mod:`md.presets`, so the whole create/read/update/delete cycle is
testable against a real file — including the states a hand-edited one can be in,
which is the case a store like this actually fails on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from md import presets
from md.presets import PresetError


def test_the_three_built_ins_are_the_ones_the_documentation_names() -> None:
    # The names are quoted in docs/TRAINING.md and in the dialog's help, and
    # they are ordered as a person meets them: prove it works, get a policy,
    # then spend real time.
    assert [preset.name for preset in presets.BUILTIN] == ["fast", "good", "best"]
    assert all(preset.builtin for preset in presets.BUILTIN)
    assert all(preset.summary for preset in presets.BUILTIN), "a preset with no stated purpose"


def test_good_is_exactly_the_recipe_that_produced_the_bundled_model() -> None:
    # docs/ROADMAP.md's reference run, which is the only preset with a measured
    # score behind it. Changing any of these makes that number someone else's
    # result rather than this preset's, so it is pinned here.
    good = presets.find("good")
    assert good is not None
    assert good.options == {
        "architecture": "entity",
        "envs": "1024",
        "steps": "256",
        "updates": "1000",
    }


def test_fast_maximises_throughput_and_best_scales_the_good_recipe() -> None:
    fast = presets.find("fast")
    good = presets.find("good")
    best = presets.find("best")
    assert fast is not None and good is not None and best is not None
    # The measured saturation point from docs/NVIDIA.md — `fast` wants the card
    # busy for minutes and is not trying to learn the relational comparison.
    assert fast.options["envs"] == "4096"
    assert fast.options["architecture"] == "mlp"
    assert best.options["architecture"] == "entity"
    # Four times the samples an update, over four times the updates: a longer
    # rollout for late-wave credit, and the run to go with it.
    assert int(best.options["steps"]) > int(good.options["steps"])
    assert int(best.options["updates"]) > int(good.options["updates"])
    batch = int(best.options["envs"]) * int(best.options["steps"])
    assert batch == 4 * int(good.options["envs"]) * int(good.options["steps"])


def test_best_splits_its_bigger_batch_into_smaller_pieces_not_larger_ones() -> None:
    # The regression this preset was born with: it quadrupled the batch and left
    # `minibatches` alone, which quadruples the *minibatch* — and on the
    # relational architecture that is what peak memory follows. It ran out of
    # memory on a 32 GiB card before the first update finished. See md.footprint.
    good = presets.find("good")
    best = presets.find("best")
    assert good is not None and best is not None

    def minibatch(preset: presets.Preset) -> int:
        options = preset.options
        batch = int(options.get("envs", 1024)) * int(options.get("steps", 256))
        return batch // int(options.get("minibatches", 8))

    assert minibatch(best) < minibatch(good), "best asks for a bigger minibatch than good"


def test_a_saved_preset_survives_a_round_trip(tmp_path: Path) -> None:
    file = tmp_path / "presets.json"
    saved = presets.save("overnight", {"envs": "8192", "updates": "6000"}, summary="big", path=file)
    assert saved.name == "overnight"
    assert not saved.builtin

    reloaded = presets.find("overnight", file)
    assert reloaded is not None
    assert reloaded.options == {"envs": "8192", "updates": "6000"}
    assert reloaded.summary == "big"
    # And it is listed after the built-ins, which never enter the file.
    assert [preset.name for preset in presets.all_presets(file)][-1] == "overnight"
    stored = json.loads(file.read_text(encoding="utf-8"))
    assert [record["name"] for record in stored] == ["overnight"]


def test_saving_a_name_that_exists_replaces_it_rather_than_doubling_it(tmp_path: Path) -> None:
    # Save as… over an existing name and Update are the same act; a store that
    # distinguished them could only fail in a second way.
    file = tmp_path / "presets.json"
    presets.save("mine", {"envs": "2048"}, path=file)
    presets.save("mine", {"envs": "4096"}, summary="wider", path=file)

    saved = presets.user_presets(file)
    assert [preset.name for preset in saved] == ["mine"]
    assert saved[0].options == {"envs": "4096"}
    assert saved[0].summary == "wider"


def test_a_built_in_name_can_be_neither_taken_nor_deleted(tmp_path: Path) -> None:
    # Their names are quoted in the documentation, so "good" has to keep meaning
    # what is written there.
    file = tmp_path / "presets.json"
    with pytest.raises(PresetError, match="built-in"):
        presets.save("good", {"envs": "16384"}, path=file)
    with pytest.raises(PresetError, match="ships with the trainer"):
        presets.delete("fast", file)
    assert presets.find("good", file) == presets.find("good")


def test_deleting_forgets_only_that_one(tmp_path: Path) -> None:
    file = tmp_path / "presets.json"
    presets.save("keep", {"envs": "1024"}, path=file)
    presets.save("drop", {"envs": "2048"}, path=file)

    presets.delete("drop", file)
    assert [preset.name for preset in presets.user_presets(file)] == ["keep"]
    with pytest.raises(PresetError, match="no saved preset"):
        presets.delete("drop", file)


def test_a_name_is_trimmed_and_must_say_something(tmp_path: Path) -> None:
    file = tmp_path / "presets.json"
    assert presets.save("  spaced   out  ", {"envs": "1024"}, path=file).name == "spaced out"
    for rejected in ("", "   ", "\t\n"):
        with pytest.raises(PresetError, match="needs a name"):
            presets.save(rejected, {"envs": "1024"}, path=file)
    # The picker's own first entry is not a preset and cannot become one.
    with pytest.raises(PresetError, match="no preset at all"):
        presets.save("Custom", {"envs": "1024"}, path=file)


def test_a_missing_or_broken_file_is_no_presets_rather_than_no_trainer(tmp_path: Path) -> None:
    # This is a convenience over the flags. A file somebody edited into nonsense
    # must not be the reason a run cannot be started, so every one of these is
    # "there are no saved presets" and the built-ins still list.
    missing = tmp_path / "gone.json"
    assert presets.user_presets(missing) == []
    assert len(presets.all_presets(missing)) == len(presets.BUILTIN)

    for content in ("", "not json at all", "{}", '"a string"', "[1, 2, 3]"):
        broken = tmp_path / "broken.json"
        broken.write_text(content, encoding="utf-8")
        assert presets.user_presets(broken) == []


def test_hand_edited_records_are_narrowed_field_by_field(tmp_path: Path) -> None:
    file = tmp_path / "presets.json"
    file.write_text(
        json.dumps(
            [
                {"name": "no options"},  # dropped: nothing to apply
                {"options": {"envs": "1024"}},  # dropped: nothing to call it
                {"name": "good", "options": {"envs": "1"}},  # dropped: reserved name
                {
                    "name": "typed",
                    # JSON's own types, and a nested value that is not a flag.
                    "options": {"envs": 4096, "learning_rate": 0.0003, "list": [1], "on": True},
                },
            ]
        ),
        encoding="utf-8",
    )
    saved = presets.user_presets(file)
    assert [preset.name for preset in saved] == ["typed"]
    # Numbers become the strings the command line takes, `list` is dropped, and
    # a boolean arrives in Python's spelling rather than JSON's.
    assert saved[0].options == {"envs": "4096", "learning_rate": "0.0003", "on": "True"}
    # And the built-in is still the built-in.
    good = presets.find("good", file)
    assert good is not None and good.builtin and good.options["envs"] == "1024"


def test_the_file_is_replaced_atomically_and_left_readable(tmp_path: Path) -> None:
    # It is meant to be opened in an editor and copied between machines, and the
    # trainer may be reading it to draw the picker while a save lands.
    file = tmp_path / "nested" / "presets.json"
    presets.save("mine", {"envs": "1024"}, summary="one line", path=file)
    text = file.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\n  " in text, "the presets file is not indented for a human"
    assert not list(file.parent.glob(".*.new")), "a temporary file was left behind"
