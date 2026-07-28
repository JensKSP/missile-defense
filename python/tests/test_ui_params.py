# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for reading the trainer's knobs out of the trainer's source.

The trainer cannot import `missile_defense.training.train` — that would pull in torch, which a test
forbids — so the parameter form parses it instead. These tests pin both halves:
the parsing, against a fixture that looks like the real dataclasses, and the
result against the actual `missile_defense/training/train.py`, so a rename over there is caught here
rather than by an empty form.
"""

from __future__ import annotations

from pathlib import Path

import missile_defense
from missile_defense.runs import runconfig
from missile_defense.ui.params import HEADLINE, Param, command_line, read_params, settings_of

TRAINER = Path(missile_defense.__file__).parent

FIXTURE = '''
"""A stand-in for the real thing."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Everything worth changing."""

    #: Environments stepped in parallel. Throughput keeps climbing to a few
    #: thousand; the ceiling is usually GPU memory.
    envs: int = 1024
    steps: int = 128
    #: "cuda", "cpu", or None to pick automatically.
    device: str | None = None
    out_dir: Path = Path("runs")
    resume: Path | None = None


@dataclass(frozen=True)
class PPOConfig:
    #: Adam's step size.
    learning_rate: float = 3.0e-4
'''


def _fixture(tmp_path: Path) -> Path:
    """A package root with the two config modules where the layers put them.

    Spelled out rather than read from `params.SOURCES` — a fixture that shares
    the implementation's idea of the layout cannot catch it being wrong, which is
    exactly what these tests are for.
    """
    (tmp_path / "training").mkdir(exist_ok=True)
    (tmp_path / "training" / "train.py").write_text(FIXTURE, encoding="utf-8")
    (tmp_path / "training" / "ppo.py").write_text(FIXTURE, encoding="utf-8")
    return tmp_path


def test_a_field_carries_the_reasoning_written_beside_it(tmp_path: Path) -> None:
    fields = {field.name: field for field in read_params(_fixture(tmp_path))}
    assert fields["envs"].help.startswith("Environments stepped in parallel.")
    assert "GPU memory" in fields["envs"].help  # the whole run of #: lines, joined
    assert fields["envs"].default == "1024"
    assert fields["envs"].kind == "int"


def test_a_field_with_no_comment_simply_has_no_tooltip(tmp_path: Path) -> None:
    fields = {field.name: field for field in read_params(_fixture(tmp_path))}
    assert fields["steps"].help == ""


def test_the_run_directory_is_not_a_parameter(tmp_path: Path) -> None:
    # The trainer supplies it, and resuming belongs to a checkpoint browser.
    names = {field.name for field in read_params(_fixture(tmp_path))}
    assert "out_dir" not in names
    assert "resume" not in names


def test_an_optional_default_reads_as_empty(tmp_path: Path) -> None:
    fields = {field.name: field for field in read_params(_fixture(tmp_path))}
    assert fields["device"].default == ""  # shown as a placeholder, not "None"
    assert fields["device"].kind == "text"


def test_floats_are_told_apart_from_ints(tmp_path: Path) -> None:
    fields = {field.name: field for field in read_params(_fixture(tmp_path))}
    assert fields["learning_rate"].kind == "float"
    assert float(fields["learning_rate"].default) == 3.0e-4


def test_no_md_train_beside_the_trainer_is_not_an_error(tmp_path: Path) -> None:
    assert read_params(tmp_path / "nothing-here") == []


def test_a_default_that_names_a_constant_shows_the_number_it_stands_for(tmp_path: Path) -> None:
    # `reaction_delay: int = CANONICAL_REACTION_DELAY` is the trainer refusing to
    # retype the published protocol, and it is two modules away from the number.
    # The form still has to show a number: a spin box cannot hold a name, and one
    # that was handed one took the whole parameter dialog down with it.
    # And the chain now crosses a layer: `training/train.py` reaches `sim/` with a
    # two-dot import, `sim/benchmark.py` reaches its neighbour with one. Following
    # only single-dot imports resolved neither and left the *name* in the form.
    (tmp_path / "training").mkdir(exist_ok=True)
    (tmp_path / "sim").mkdir(exist_ok=True)
    (tmp_path / "training" / "train.py").write_text(
        "from dataclasses import dataclass\n"
        "from ..sim.benchmark import CANONICAL_REACTION_DELAY\n\n\n"
        "@dataclass\n"
        "class TrainConfig:\n"
        "    reaction_delay: int = CANONICAL_REACTION_DELAY\n"
        "    nowhere: int = MISSING\n",
        encoding="utf-8",
    )
    (tmp_path / "sim" / "benchmark.py").write_text(
        "from .protocol import REACTION_DELAY\n\nCANONICAL_REACTION_DELAY = REACTION_DELAY\n",
        encoding="utf-8",
    )
    (tmp_path / "sim" / "protocol.py").write_text("REACTION_DELAY: int = 3\n", encoding="utf-8")

    fields = {field.name: field for field in read_params(tmp_path)}
    assert fields["reaction_delay"].default == "3"
    # And a name that leads nowhere is left as it is written rather than guessed
    # at: wrong is worse than unresolved, and the form falls back on its own.
    assert fields["nowhere"].default == "MISSING"


def test_the_handicap_the_real_trainer_defaults_to_reads_as_its_value() -> None:
    # The chain that actually matters: `missile_defense.training.train` →
    # `missile_defense.sim.benchmark` → the
    # generated `missile_defense.sim.protocol`, followed without importing any of them.
    from missile_defense.sim.benchmark import (  # noqa: PLC0415
        CANONICAL_AIM_TRAIL,
        CANONICAL_REACTION_DELAY,
    )

    fields = {field.name: field for field in read_params(TRAINER)}
    assert float(fields["aim_trail"].default) == CANONICAL_AIM_TRAIL
    assert int(fields["reaction_delay"].default) == CANONICAL_REACTION_DELAY


def test_the_real_trainer_still_has_the_headline_four() -> None:
    # The four the form promotes are named by string; if one is renamed in the
    # trainer, this is where it is noticed.
    fields = {field.name: field for field in read_params(TRAINER)}
    for name in HEADLINE:
        assert name in fields, f"{name} is gone from the trainer's config"
        assert fields[name].headline
    assert fields["envs"].owner == "TrainConfig"
    assert fields["learning_rate"].owner == "PPOConfig"
    assert fields["entropy_coef"].help  # the advanced ones are explained too


def test_every_field_is_reachable_from_the_command_line() -> None:
    # A field the form offers but `missile_defense.training.train` has no flag for would silently do
    # nothing. The two config classes reach the CLI by different routes, so both
    # are checked: TrainConfig by explicit flags, PPOConfig by the loop that
    # generates one per dataclass field.
    source = (TRAINER / "training" / "train.py").read_text(encoding="utf-8")
    assert "for field in dataclasses.fields(PPOConfig)" in source
    for field in read_params(TRAINER):
        if field.owner == "TrainConfig":
            assert field.flag in source, f"{field.flag} has no argparse flag"


def test_the_command_line_carries_only_what_changed() -> None:
    command = command_line("python", {"envs": "2048"}, out_dir=Path("runs"))
    assert command == [
        "python",
        "-u",
        "-m",
        "missile_defense.training",
        "--envs",
        "2048",
        "--out-dir",
        "runs",
    ]


def test_a_flag_is_the_field_name_with_dashes() -> None:
    assert Param("entropy_coef", "float", "0.01", "", "PPOConfig").flag == "--entropy-coef"


def test_resuming_adds_the_checkpoint_last() -> None:
    # Last so the command reads as "this run, continued from there" rather than
    # burying the one flag that changes what the run *is* among the tuning.
    command = command_line(
        "python", {}, out_dir=Path("runs"), resume=Path("runs/checkpoints/policy-00400.pt")
    )
    assert command[-2:] == ["--resume", str(Path("runs/checkpoints/policy-00400.pt"))]


def test_a_fresh_run_carries_no_resume_flag() -> None:
    assert "--resume" not in command_line("python", {}, out_dir=Path("runs"))


def test_resume_is_not_offered_as_a_text_field() -> None:
    # It is a file that exists, so the form gives it a picker; a box you can
    # mistype a path into is the thing being avoided (missile_defense.ui.forms).
    assert "resume" not in {field.name for field in read_params(TRAINER)}


# ---- the third group, and the guards on it -----------------------------------


def test_the_reward_weights_are_offered_too() -> None:
    """`Shaping` was the one group with no flag at all.

    Changing what the agent is *paid for* meant editing the source and
    rebuilding, which put the most consequential knobs in the project out of
    reach of anyone not editing it.
    """
    from missile_defense.ui.params import read_params as read  # noqa: PLC0415

    fields = {field.name: field for field in read(Path("python/missile_defense"))}
    for name in ("city_weight", "base_weight", "waste_penalty", "multikill_bonus"):
        assert name in fields, name
        assert fields[name].owner == "Shaping"


def test_the_reward_flags_are_prefixed_so_the_two_gammas_do_not_collide() -> None:
    """`Shaping.gamma` and `PPOConfig.gamma` are different discounts."""
    from missile_defense.ui.params import read_params as read  # noqa: PLC0415

    fields = [field for field in read(Path("python/missile_defense")) if field.name == "gamma"]
    assert {field.flag for field in fields} == {"--gamma", "--reward-gamma"}


def test_every_offered_choice_is_one_the_trainer_accepts() -> None:
    """A dropdown cannot be misspelled — but it can be *wrong*.

    The values here and the ones `missile_defense.training.ppo.build_policy`
    implements are two lists
    that would drift silently, and the symptom would be a run that dies on its
    first update after the parameter dialog offered the option.
    """
    from missile_defense.sim.policy_format import ARCHITECTURES  # noqa: PLC0415
    from missile_defense.ui.params import CHOICES  # noqa: PLC0415

    # Every architecture the trainer offers must be one the trainer can build.
    # `missile_defense.training.ppo` is not importable without torch, so the
    # format's own table — which
    # is generated from the same set and *is* importable — stands in for it.
    assert set(CHOICES["architecture"]) == set(ARCHITECTURES)


def test_every_bound_belongs_to_a_field_that_exists() -> None:
    """A bound on a renamed field is a bound that silently stops applying."""
    from missile_defense.ui.params import BOUNDS  # noqa: PLC0415
    from missile_defense.ui.params import read_params as read  # noqa: PLC0415

    names = {field.name for field in read(Path("python/missile_defense"))}
    assert set(BOUNDS) <= names, sorted(set(BOUNDS) - names)


def test_no_bound_excludes_its_own_default() -> None:
    """The range has to admit the value the trainer would have used anyway.

    Otherwise the dialog opens on a field it immediately considers invalid,
    which is the most confusing possible first impression.
    """
    from missile_defense.ui.params import read_params as read  # noqa: PLC0415

    for field in read(Path("python/missile_defense")):
        if field.bounds is None or not field.default:
            continue
        try:
            value = float(field.default)
        except ValueError:
            continue
        low, high = field.bounds
        assert low <= value <= high, f"{field.name}={value} outside {field.bounds}"


def test_the_prefixed_field_list_matches_the_real_dataclass() -> None:
    """`REWARD_FIELDS` is stated, so it can drift — and a drifted name means a
    Start button that emits a flag the trainer rejects."""
    from missile_defense.ui.params import REWARD_FIELDS  # noqa: PLC0415
    from missile_defense.ui.params import read_params as read  # noqa: PLC0415

    shaping = {f.name for f in read(Path("python/missile_defense")) if f.owner == "Shaping"}
    assert shaping == set(REWARD_FIELDS)


def test_a_reward_weight_reaches_the_command_line_under_its_real_flag() -> None:
    """The bug this guards: `command_line` rebuilt flags from the field name and
    so emitted `--city-weight`, which `missile_defense.training.train` does not accept."""
    from missile_defense.ui.params import command_line  # noqa: PLC0415

    command = command_line("python3", {"city_weight": "250", "envs": "512"}, out_dir=Path("/tmp/x"))
    assert "--reward-city-weight" in command
    assert "--city-weight" not in command
    assert "--envs" in command  # unprefixed groups are untouched


def test_the_discount_is_written_to_both_config_classes() -> None:
    """`gamma` names a field of `PPOConfig` *and* one of `Shaping`.

    They have to agree: one discounts the return and the other the potential,
    and `missile_defense.sim.env.Shaping` says the invariance proof assumes they
    are the same. Since the dialog's editors are keyed by field name there has
    only ever been one box for both, so the one value it produces must reach
    both flags.

    The bug this guards: `flag_for` resolved the name through `REWARD_FIELDS`,
    which contains `gamma`, so the command carried `--reward-gamma` alone.
    `--gamma` was unreachable from the trainer entirely, and every run whose
    discount was set here trained with the potential discounted at one rate and
    the return at another — the one state the proof excludes, reached silently.
    """
    command = command_line("python", {"gamma": "0.997"}, out_dir=Path("runs"))

    assert "--gamma" in command
    assert "--reward-gamma" in command
    # Both carry the value the form was given, rather than one of them keeping
    # the dataclass default — which is the disagreement itself.
    assert command[command.index("--gamma") + 1] == "0.997"
    assert command[command.index("--reward-gamma") + 1] == "0.997"


def test_a_field_owned_by_one_class_still_writes_one_flag() -> None:
    """The shared-flag rule is for `gamma` and must not leak to its neighbours."""
    from missile_defense.ui.params import flags_for  # noqa: PLC0415

    assert flags_for("envs") == ("--envs",)
    assert flags_for("city_weight") == ("--reward-city-weight",)
    assert flags_for("gamma") == ("--gamma", "--reward-gamma")


# ---- pairing a finished run with the reasoning behind its knobs ---------------


def _stored(tmp_path: Path, payload: object) -> runconfig.RunConfig:
    import json

    (tmp_path / runconfig.FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    config = runconfig.read(tmp_path)
    assert config is not None
    return config


def test_a_stored_setting_carries_the_reasoning_written_beside_the_field(
    tmp_path: Path,
) -> None:
    config = _stored(tmp_path, {"train": {"envs": 4096}})

    settings = {setting.name: setting for setting in settings_of(config, read_params(TRAINER))}

    assert settings["envs"].value == "4096"
    assert settings["envs"].changed  # against the trainer's own 1024
    assert "Environments stepped in parallel" in settings["envs"].help


def test_a_run_that_kept_a_default_is_not_marked_as_having_changed_it(
    tmp_path: Path,
) -> None:
    config = _stored(tmp_path, {"ppo": {"learning_rate": 0.0003}})

    settings = {setting.name: setting for setting in settings_of(config, read_params(TRAINER))}

    # The trainer's source spells it `3.0e-4` and its own config.json spells it
    # `0.0003`. Comparing those as text marks every run as having changed it.
    assert not settings["learning_rate"].changed


def test_a_knob_this_trainer_has_never_heard_of_is_still_shown(tmp_path: Path) -> None:
    """A run trained by a newer trainer is still a run somebody has to read."""
    config = _stored(tmp_path, {"train": {"curriculum": "waves"}})

    settings = settings_of(config, read_params(TRAINER))

    assert [(s.name, s.value, s.changed) for s in settings] == [("curriculum", "waves", False)]


def test_settings_keep_the_order_and_grouping_the_trainer_wrote(tmp_path: Path) -> None:
    config = _stored(tmp_path, {"train": {"envs": 8, "steps": 4}, "schedule": {"start_update": 1}})

    settings = settings_of(config, read_params(TRAINER))

    assert [(s.group, s.name) for s in settings] == [
        ("train", "envs"),
        ("train", "steps"),
        ("schedule", "start_update"),
    ]


def test_nothing_stored_is_nothing_to_show() -> None:
    assert settings_of(None, read_params(TRAINER)) == []


# ---- the dialog's layout, held to the trainer's actual fields -----------------


def test_every_settable_field_is_placed_in_exactly_one_group() -> None:
    """The failure this catches is silent: a field nobody placed is a field the
    dialog never shows, and the trainer's default stands with no way to change it.

    Checked in both directions, because both drift. A field added to the trainer
    and not to `GROUPS` disappears from the form; a name left in `GROUPS` after a
    rename points at nothing and quietly shrinks a panel.
    """
    from missile_defense.ui.params import GROUPS  # noqa: PLC0415
    from missile_defense.ui.params import read_params as read  # noqa: PLC0415

    fields = [f for f in read(Path("python/missile_defense")) if not f.derived]
    placed: list[str] = [name for group in GROUPS for name in group.fields]

    assert len(placed) == len(set(placed)), "a field is in two groups"
    assert set(placed) == {f.name for f in fields}, (
        f"unplaced: {sorted({f.name for f in fields} - set(placed))}  "
        f"stale: {sorted(set(placed) - {f.name for f in fields})}"
    )


def test_every_group_belongs_to_a_real_domain() -> None:
    from missile_defense.ui.params import DOMAINS, GROUPS  # noqa: PLC0415

    known = {key for key, _, _ in DOMAINS}
    for group in GROUPS:
        assert group.domain in known, f"{group.name} is in domain {group.domain!r}"


def test_no_folded_group_is_large_enough_to_be_a_wall_of_boxes() -> None:
    """The whole point of the folds: opening one costs a glance.

    The bound is on *folded* groups only. An essential group is the panel's own
    surface — it is not hiding anything, and holding it to the same size would
    only push a field people need into a drawer to satisfy a number.
    """
    from missile_defense.ui.params import GROUPS  # noqa: PLC0415

    for group in GROUPS:
        if group.essential:
            continue
        assert len(group.fields) <= 5, f"{group.name} has {len(group.fields)} fields"


def test_the_shaping_discount_is_derived_rather_than_offered() -> None:
    """Two editors for one number is how they end up disagreeing.

    `Shaping.gamma` must equal `PPOConfig.gamma` — both dataclasses say so and
    the invariance proof assumes it — so the form offers the PPO one and derives
    the other. It still reaches the command line under its own flag.
    """
    from missile_defense.ui.params import flags_for  # noqa: PLC0415
    from missile_defense.ui.params import read_params as read  # noqa: PLC0415

    gammas = {f.owner: f for f in read(Path("python/missile_defense")) if f.name == "gamma"}
    assert gammas["Shaping"].derived
    assert not gammas["PPOConfig"].derived
    assert gammas["Shaping"].group is None  # no panel row of its own
    assert gammas["PPOConfig"].group is not None
    assert flags_for("gamma") == ("--gamma", "--reward-gamma")


def test_a_field_is_identified_by_its_owner_as_well_as_its_name() -> None:
    """`gamma` names a field of two config classes. Keying editors by name alone
    gave them one box between them, and only the last one read."""
    from missile_defense.ui.params import read_params as read  # noqa: PLC0415

    fields = read(Path("python/missile_defense"))
    assert len({f.key for f in fields}) == len(fields)
    assert len({f.name for f in fields}) < len(fields)  # the collision is real


def test_every_bounded_field_can_be_reached_by_its_slider() -> None:
    """A log slider cannot represent a negative number and a decade slider cannot
    represent zero. A field whose scale disagrees with its bounds would open on a
    control that cannot show its own default."""
    from missile_defense.ui.params import SCALE  # noqa: PLC0415
    from missile_defense.ui.params import read_params as read  # noqa: PLC0415

    for field in read(Path("python/missile_defense")):
        scale = SCALE.get(field.name, "linear")
        if scale == "linear" or field.bounds is None:
            continue
        low, high = field.bounds
        assert high > 0, f"{field.name}: {scale} needs a positive maximum"
        if scale == "decade":
            assert low > 0, f"{field.name}: a decade slider cannot start at {low}"
        else:
            assert low >= 0, f"{field.name}: a log slider cannot start at {low}"


def test_every_scale_names_a_field_that_exists() -> None:
    from missile_defense.ui.params import SCALE  # noqa: PLC0415
    from missile_defense.ui.params import read_params as read  # noqa: PLC0415

    names = {f.name for f in read(Path("python/missile_defense"))}
    assert set(SCALE) <= names, sorted(set(SCALE) - names)


def test_the_glossary_defines_what_it_is_asked_about() -> None:
    """A glossary entry nobody can reach is one nobody maintains. Every term has
    to be referenced by a field's label, its choices, or a group name."""
    from missile_defense.ui.params import CHOICES, GLOSSARY, GROUPS  # noqa: PLC0415
    from missile_defense.ui.params import read_params as read  # noqa: PLC0415

    vocabulary = {f.name for f in read(Path("python/missile_defense"))}
    vocabulary |= {c for values in CHOICES.values() for c in values}
    vocabulary |= {word.lower() for group in GROUPS for word in group.name.split()}
    # The terms that are explanations of the domain rather than of a field.
    conceptual = {"PPO", "GAE", "potential", "shaping", "checkpoint", "policy", "rollout"}

    for term in GLOSSARY:
        assert (
            term in conceptual
            or term in vocabulary
            or term in {n.rsplit("_", 1)[0] for n in vocabulary}
        ), f"{term!r} is defined but nothing in the dialog refers to it"
