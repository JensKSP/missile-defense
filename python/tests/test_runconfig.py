# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Reading back what a run was started with.

`config.json` is written by the trainer and read by everything that is not the
trainer — the trainer's parameter view, the Start dialog when it continues a run,
and `--show-config`. So it is parsed here, once, with no torch and no Qt, and
against a file a person may well have edited.
"""

from __future__ import annotations

import json
from pathlib import Path

from missile_defense.runs import runconfig

PAYLOAD = {
    "train": {
        "envs": 2048,
        "steps": 512,
        "updates": 4000,
        "device": None,
        "out_dir": "/runs/grave-quarry",
        "resume": None,
        "learning_rate_final": 1e-05,
    },
    "ppo": {"hidden": 512, "learning_rate": 0.0003, "architecture": "entity", "gamma": 0.999},
    "shaping": {"city_weight": 100.0, "enabled": True, "gamma": 0.99},
    "schedule": {"start_update": 1, "end_update": 4000},
}


def _run(tmp_path: Path, payload: object = PAYLOAD) -> Path:
    (tmp_path / runconfig.FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_reads_a_run_directory(tmp_path: Path) -> None:
    config = runconfig.read(_run(tmp_path))

    assert config is not None
    assert config.train["envs"] == 2048
    assert config.ppo["architecture"] == "entity"
    assert config.shaping["city_weight"] == 100.0
    assert config.schedule["end_update"] == 4000


def test_reads_the_file_itself_as_well_as_its_directory(tmp_path: Path) -> None:
    run = _run(tmp_path)

    assert runconfig.read(run / runconfig.FILENAME) == runconfig.read(run)


def test_a_directory_without_one_is_simply_no_config(tmp_path: Path) -> None:
    assert runconfig.read(tmp_path) is None


def test_a_mangled_file_is_no_config_rather_than_an_exception(tmp_path: Path) -> None:
    (tmp_path / runconfig.FILENAME).write_text("{not json", encoding="utf-8")

    assert runconfig.read(tmp_path) is None


def test_missing_groups_read_as_empty_rather_than_raising(tmp_path: Path) -> None:
    config = runconfig.read(_run(tmp_path, {"train": {"envs": 8}}))

    assert config is not None
    assert config.train["envs"] == 8
    assert config.ppo == {}
    assert config.shaping == {}


def test_a_group_that_is_not_a_mapping_is_dropped(tmp_path: Path) -> None:
    config = runconfig.read(_run(tmp_path, {"train": {"envs": 8}, "ppo": [1, 2, 3]}))

    assert config is not None
    assert config.ppo == {}


def test_a_payload_that_is_not_a_mapping_is_no_config(tmp_path: Path) -> None:
    assert runconfig.read(_run(tmp_path, [1, 2, 3])) is None


def test_options_are_the_shape_the_start_dialog_and_presets_use(tmp_path: Path) -> None:
    config = runconfig.read(_run(tmp_path))
    assert config is not None

    options = runconfig.options(config)

    # Field name to string, exactly as `missile_defense.runs.presets` stores them and as
    # `missile_defense.ui.params.command_line` consumes them.
    assert options["envs"] == "2048"
    assert options["architecture"] == "entity"
    assert options["learning_rate"] == "0.0003"
    assert options["enabled"] == "True"
    assert all(isinstance(value, str) for value in options.values())


def test_options_leave_out_what_a_form_must_not_restate(tmp_path: Path) -> None:
    options = runconfig.options(runconfig.read(_run(tmp_path)))

    # The trainer supplies the output directory itself and resuming is a picker,
    # so both would be a second, stale answer to a question already asked.
    assert "out_dir" not in options
    assert "resume" not in options
    # The schedule is derived from the four annealing endpoints; it has no flags.
    assert "start_update" not in options


def test_options_drop_a_value_that_was_not_set(tmp_path: Path) -> None:
    options = runconfig.options(runconfig.read(_run(tmp_path)))

    # `device: null` is "decide at run time". Poured into a form as "None" it
    # would become `--device None`, which is a device nobody has.
    assert "device" not in options


def test_the_reward_weights_win_the_name_they_share_with_ppo(tmp_path: Path) -> None:
    options = runconfig.options(runconfig.read(_run(tmp_path)))

    # `Shaping.gamma` and `PPOConfig.gamma` are two different discounts, and the
    # trainer's flat flag namespace resolves the collision to `--reward-gamma`
    # (missile_defense.ui.params.REWARD_FIELDS). A flat option map has to agree with it.
    assert options["gamma"] == "0.99"


def test_describe_lays_every_group_out_for_a_terminal(tmp_path: Path) -> None:
    config = runconfig.read(_run(tmp_path))
    assert config is not None

    lines = runconfig.describe(config.payload)

    assert any(line.startswith("  train") and "envs=2048" in line for line in lines)
    assert any(line.startswith("  ppo") and "architecture=entity" in line for line in lines)
    assert any("shaping" in line and "city_weight=100.0" in line for line in lines)
    assert all(len(line) <= 96 for line in lines)


def test_describe_wraps_rather_than_running_off_the_screen() -> None:
    payload = {"train": {f"field_{index}": index for index in range(40)}}

    lines = runconfig.describe(payload, width=60)

    assert len(lines) > 1
    assert all(len(line) <= 60 for line in lines)
    # Continuation lines line up under the first value, not under the label.
    assert lines[1].startswith(" " * len(lines[0].split("field_0")[0]))


def test_describe_says_so_when_there_is_nothing_to_say() -> None:
    assert runconfig.describe({}) == []


def test_describe_gives_an_unset_setting_a_word(tmp_path: Path) -> None:
    config = runconfig.read(_run(tmp_path))
    assert config is not None

    lines = runconfig.describe(config.payload)

    # `device=` reads as a line that got cut off. It did not: the run simply
    # never pinned one.
    assert any(f"device={runconfig.NOTHING}" in line for line in lines)


def test_resumed_from_is_the_checkpoint_a_continuation_names(tmp_path: Path) -> None:
    payload = dict(PAYLOAD) | {"train": dict(PAYLOAD["train"]) | {"resume": "/runs/x/p.pt"}}
    config = runconfig.read(_run(tmp_path, payload))
    assert config is not None

    assert config.resumed_from == Path("/runs/x/p.pt")
    assert runconfig.read(_run(tmp_path)) is not None
    fresh = runconfig.read(_run(tmp_path))
    assert fresh is not None and fresh.resumed_from is None
