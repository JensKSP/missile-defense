# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for reading the trainer's knobs out of the trainer's source.

The console cannot import `md.train` — that would pull in torch, which a test
forbids — so the parameter form parses it instead. These tests pin both halves:
the parsing, against a fixture that looks like the real dataclasses, and the
result against the actual `md/train.py`, so a rename over there is caught here
rather than by an empty form.
"""

from __future__ import annotations

from pathlib import Path

import md
from md.ui.params import HEADLINE, Param, command_line, read_params

TRAINER = Path(md.__file__).parent

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
    (tmp_path / "train.py").write_text(FIXTURE, encoding="utf-8")
    (tmp_path / "ppo.py").write_text(FIXTURE, encoding="utf-8")
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
    # The console supplies it, and resuming belongs to a checkpoint browser.
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


def test_no_trainer_beside_the_console_is_not_an_error(tmp_path: Path) -> None:
    assert read_params(tmp_path / "nothing-here") == []


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
    # A field the form offers but `md.train` has no flag for would silently do
    # nothing. The two config classes reach the CLI by different routes, so both
    # are checked: TrainConfig by explicit flags, PPOConfig by the loop that
    # generates one per dataclass field.
    source = (TRAINER / "train.py").read_text(encoding="utf-8")
    assert "for field in dataclasses.fields(PPOConfig)" in source
    for field in read_params(TRAINER):
        if field.owner == "TrainConfig":
            assert field.flag in source, f"{field.flag} has no argparse flag"


def test_the_command_line_carries_only_what_changed() -> None:
    command = command_line("python", {"envs": "2048"}, out_dir=Path("runs"))
    assert command == ["python", "-u", "-m", "md.train", "--envs", "2048", "--out-dir", "runs"]


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
    # mistype a path into is the thing being avoided (md.ui.forms).
    assert "resume" not in {field.name for field in read_params(TRAINER)}


# ---- the third group, and the guards on it -----------------------------------


def test_the_reward_weights_are_offered_too() -> None:
    """`Shaping` was the one group with no flag at all.

    Changing what the agent is *paid for* meant editing the source and
    rebuilding, which put the most consequential knobs in the project out of
    reach of anyone not editing it.
    """
    from md.ui.params import read_params as read  # noqa: PLC0415

    fields = {field.name: field for field in read(Path("python/md"))}
    for name in ("city_weight", "base_weight", "waste_penalty", "multikill_bonus"):
        assert name in fields, name
        assert fields[name].owner == "Shaping"


def test_the_reward_flags_are_prefixed_so_the_two_gammas_do_not_collide() -> None:
    """`Shaping.gamma` and `PPOConfig.gamma` are different discounts."""
    from md.ui.params import read_params as read  # noqa: PLC0415

    fields = [field for field in read(Path("python/md")) if field.name == "gamma"]
    assert {field.flag for field in fields} == {"--gamma", "--reward-gamma"}


def test_every_offered_choice_is_one_the_trainer_accepts() -> None:
    """A dropdown cannot be misspelled — but it can be *wrong*.

    The values here and the ones `md.ppo.build_policy` implements are two lists
    that would drift silently, and the symptom would be a run that dies on its
    first update after the parameter dialog offered the option.
    """
    from md.policy_format import ARCHITECTURES  # noqa: PLC0415
    from md.ui.params import CHOICES  # noqa: PLC0415

    # Every architecture the console offers must be one the trainer can build.
    # `md.ppo` is not importable without torch, so the format's own table — which
    # is generated from the same set and *is* importable — stands in for it.
    assert set(CHOICES["architecture"]) == set(ARCHITECTURES)


def test_every_bound_belongs_to_a_field_that_exists() -> None:
    """A bound on a renamed field is a bound that silently stops applying."""
    from md.ui.params import BOUNDS  # noqa: PLC0415
    from md.ui.params import read_params as read  # noqa: PLC0415

    names = {field.name for field in read(Path("python/md"))}
    assert set(BOUNDS) <= names, sorted(set(BOUNDS) - names)


def test_no_bound_excludes_its_own_default() -> None:
    """The range has to admit the value the trainer would have used anyway.

    Otherwise the dialog opens on a field it immediately considers invalid,
    which is the most confusing possible first impression.
    """
    from md.ui.params import read_params as read  # noqa: PLC0415

    for field in read(Path("python/md")):
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
    from md.ui.params import REWARD_FIELDS  # noqa: PLC0415
    from md.ui.params import read_params as read  # noqa: PLC0415

    shaping = {f.name for f in read(Path("python/md")) if f.owner == "Shaping"}
    assert shaping == set(REWARD_FIELDS)


def test_a_reward_weight_reaches_the_command_line_under_its_real_flag() -> None:
    """The bug this guards: `command_line` rebuilt flags from the field name and
    so emitted `--city-weight`, which `md.train` does not accept."""
    from md.ui.params import command_line  # noqa: PLC0415

    command = command_line("python3", {"city_weight": "250", "envs": "512"}, out_dir=Path("/tmp/x"))
    assert "--reward-city-weight" in command
    assert "--city-weight" not in command
    assert "--envs" in command  # unprefixed groups are untouched
