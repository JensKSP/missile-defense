# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for the model card — the trainer's description of what it is training.

``md.modelcard`` sits on the boundary: the trainer writes it, the console reads
it, and neither imports the other (docs/ROADMAP.md, M8). It takes *shapes* rather
than tensors, which is what lets these run with no torch installed — the same
reason the console can read the file at all.
"""

from __future__ import annotations

import json
from pathlib import Path

from md import modelcard
from md.modelcard import ModelCard, Tensor, describe, headline, layer_table, read, wiring, write

#: The state dict of the default MLP policy, as `policy-00800.pt` carries it.
MLP = {
    "trunk.0.weight": (512, 1895),
    "trunk.0.bias": (512,),
    "trunk.2.weight": (512, 512),
    "trunk.2.bias": (512,),
    "policy_head.weight": (385, 512),
    "policy_head.bias": (385,),
    "value_head.weight": (1, 512),
    "value_head.bias": (1,),
}


def _card() -> ModelCard:
    return describe(MLP, architecture="mlp", obs_size=1895, action_count=385, hidden=512)


# ---- counting ---------------------------------------------------------------


def test_the_parameter_count_is_every_element_of_every_tensor() -> None:
    # 1,431,426 — checked against `sum(p.numel() for p in policy.parameters())`
    # on a real checkpoint, which is the number this is standing in for.
    assert _card().parameters == 1_431_426


def test_a_bias_is_folded_into_the_layer_it_belongs_to() -> None:
    """Eight state-dict entries are four layers; a table of eight reads as noise."""
    blocks = _card().blocks()
    assert [block.name for block in blocks] == [
        "trunk.0",
        "trunk.2",
        "policy_head",
        "value_head",
    ]
    assert blocks[0].parameters == 512 * 1895 + 512


def test_layers_stay_in_the_order_the_data_flows_through() -> None:
    """Sorted by name, `policy_head` would come before `trunk`, which is nonsense."""
    assert [block.name for block in _card().blocks()][0] == "trunk.0"


def test_a_weight_reads_in_to_out_rather_than_out_by_in() -> None:
    """Torch stores `(out, in)`; nobody reads a network backwards."""
    assert _card().blocks()[0].wiring == "1895 → 512"


def test_a_one_dimensional_layer_shows_its_extent() -> None:
    card = describe(
        {"norm.weight": (512,), "norm.bias": (512,)},
        architecture="entity",
        obs_size=1895,
        action_count=385,
        hidden=512,
    )
    (block,) = card.blocks()
    assert block.wiring == "512"
    assert block.parameters == 1024


def test_a_tensor_that_belongs_to_no_layer_keeps_its_own_name() -> None:
    card = describe(
        {"log_std": (385,)}, architecture="mlp", obs_size=1895, action_count=385, hidden=512
    )
    assert [block.name for block in card.blocks()] == ["log_std"]


# ---- what it reads like ------------------------------------------------------


def test_the_headline_is_the_architecture_and_the_size() -> None:
    assert headline(_card()) == "mlp · 1,431,426 parameters"


def test_the_wiring_line_is_the_shape_of_the_problem() -> None:
    assert wiring(_card()) == "1,895 observations → 385 actions"


def test_the_layer_table_lines_up() -> None:
    lines = layer_table(_card())
    assert len(lines) == 4
    assert len({len(line) for line in lines}) == 1, lines
    assert lines[0].startswith("trunk.0")
    assert "970,752" in lines[0]


def test_a_card_with_no_tensors_has_no_table() -> None:
    empty = ModelCard("mlp", 1895, 385, 512, ())
    assert layer_table(empty) == []
    assert empty.parameters == 0


# ---- the file ----------------------------------------------------------------


def test_a_written_card_reads_back_the_same(tmp_path: Path) -> None:
    write(tmp_path, _card())
    restored = read(tmp_path)
    assert restored == _card()


def test_the_file_lands_beside_config_json(tmp_path: Path) -> None:
    path = write(tmp_path, _card())
    assert path == tmp_path / modelcard.FILENAME


def test_the_file_carries_the_derived_count_for_a_human_reading_it(tmp_path: Path) -> None:
    write(tmp_path, _card())
    payload = json.loads((tmp_path / modelcard.FILENAME).read_text(encoding="utf-8"))
    assert payload["parameters"] == 1_431_426
    assert payload["tensors"][0] == {"name": "trunk.0.weight", "shape": [512, 1895]}


def test_a_run_without_a_card_reads_as_nothing_to_show(tmp_path: Path) -> None:
    """A run started before this file existed, which the panel explains."""
    assert read(tmp_path) is None


def test_a_half_written_card_reads_as_nothing_to_show(tmp_path: Path) -> None:
    (tmp_path / modelcard.FILENAME).write_text('{"architecture": "ml', encoding="utf-8")
    assert read(tmp_path) is None


def test_a_card_missing_a_field_reads_as_nothing_to_show(tmp_path: Path) -> None:
    (tmp_path / modelcard.FILENAME).write_text('{"architecture": "mlp"}', encoding="utf-8")
    assert read(tmp_path) is None


def test_the_directory_is_created_if_the_run_has_not_written_yet(tmp_path: Path) -> None:
    target = tmp_path / "runs-7"
    write(target, _card())
    assert (target / modelcard.FILENAME).exists()


def test_shapes_survive_the_round_trip_as_tuples(tmp_path: Path) -> None:
    """JSON has only lists, and a shape compared against a tuple must still match."""
    write(tmp_path, _card())
    restored = read(tmp_path)
    assert restored is not None
    assert Tensor("trunk.0.weight", (512, 1895)) in restored.tensors
