# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Continuing a run that stopped, without retyping the run.

A stopped run is the common case, not the exceptional one: `touch runs/STOP`,
a full disk, a reboot, or simply a horizon that turned out to be too short. What
made it expensive was that `--resume` took a checkpoint and nothing else, so
every non-default flag of the original run had to be restated from memory — and
getting one wrong is either a rejected checkpoint or, worse, a continuation that
is quietly a different experiment.

So `--resume` is given a *run* and works the rest out: which checkpoint is the
latest, what the run was configured with, and how much of its schedule is left.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is optional; see docs/TRAINING.md")
pytest.importorskip(
    "md._md_native", reason="the _md_native extension is not built (cmake -DMD_BUILD_BINDINGS=ON)"
)

from md import runconfig  # noqa: E402
from md.env import Shaping  # noqa: E402
from md.ppo import PPOConfig  # noqa: E402
from md.train import (  # noqa: E402
    ResumeError,
    TrainConfig,
    continuation,
    main,
    resolve_checkpoint,
)

SCHEDULE = {
    "start_update": 1,
    "end_update": 1000,
    "learning_rate_start": 3.0e-4,
    "learning_rate_final": 1.0e-5,
    "entropy_coef_start": 0.02,
    "entropy_coef_final": 2.0e-3,
}

CONFIG = {
    "train": {
        "envs": 2048,
        "steps": 512,
        "updates": 1000,
        "frame_skip": 4,
        "max_ticks": 120_000,
        "eval_every": 20,
        "record_every": 25,
        "checkpoint_every": 100,
        "seed": 7,
        "device": None,
        "out_dir": "/somewhere/else",
        "resume": None,
        "learning_rate_final": 1.0e-5,
        "entropy_coef_final": 2.0e-3,
        "schedule_updates": 1000,
    },
    "ppo": {"hidden": 512, "architecture": "entity", "minibatches": 64, "gamma": 0.999},
    "shaping": {"city_weight": 250.0, "multikill_bonus": 3.0},
    "schedule": SCHEDULE,
}


def _checkpoint(run: Path, name: str, iteration: int, *, schedule: object = SCHEDULE) -> Path:
    """A checkpoint with the metadata a resume reads, and no weights worth loading."""
    path = run / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "policy": {},
        "optimizer": {"param_groups": [{"lr": 1.0e-4}]},
        "iteration": iteration,
        "obs_size": 1959,
        "action_count": 121,
        "hidden": 512,
        "architecture": "entity",
    }
    if schedule is not None:
        payload["schedule"] = schedule
    torch.save(payload, path)
    return path


def _run(tmp_path: Path, config: object = CONFIG) -> Path:
    run = tmp_path / "grave-quarry"
    run.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (run / runconfig.FILENAME).write_text(json.dumps(config), encoding="utf-8")
    return run


# ---- finding the checkpoint --------------------------------------------------


def test_a_run_directory_resolves_to_the_checkpoint_it_got_furthest_with(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _checkpoint(run, "policy-00100.pt", 100)
    _checkpoint(run, "policy-00200.pt", 200)
    final = _checkpoint(run, "policy-final.pt", 247)

    assert resolve_checkpoint(run) == final


def test_a_killed_run_resolves_to_its_last_numbered_checkpoint(tmp_path: Path) -> None:
    """No policy-final.pt: the process was killed rather than stopped."""
    run = _run(tmp_path)
    _checkpoint(run, "policy-00100.pt", 100)
    latest = _checkpoint(run, "policy-00200.pt", 200)

    assert resolve_checkpoint(run) == latest


def test_the_validation_best_is_not_mistaken_for_the_latest(tmp_path: Path) -> None:
    """`policy-best.pt` is usually an *earlier* update, and resuming rewinds."""
    run = _run(tmp_path)
    _checkpoint(run, "policy-best.pt", 300)
    latest = _checkpoint(run, "policy-00800.pt", 800)

    assert resolve_checkpoint(run) == latest


def test_the_best_will_do_when_it_is_all_there_is(tmp_path: Path) -> None:
    run = _run(tmp_path)
    best = _checkpoint(run, "policy-best.pt", 300)

    assert resolve_checkpoint(run) == best


def test_the_checkpoints_directory_itself_resolves_too(tmp_path: Path) -> None:
    run = _run(tmp_path)
    final = _checkpoint(run, "policy-final.pt", 60)

    assert resolve_checkpoint(run / "checkpoints") == final


def test_a_checkpoint_file_resolves_to_itself(tmp_path: Path) -> None:
    run = _run(tmp_path)
    one = _checkpoint(run, "policy-00100.pt", 100)
    _checkpoint(run, "policy-final.pt", 200)

    assert resolve_checkpoint(one) == one


def test_a_run_with_no_checkpoints_says_so(tmp_path: Path) -> None:
    run = _run(tmp_path)

    with pytest.raises(ResumeError, match="no checkpoint"):
        resolve_checkpoint(run)


def test_a_path_that_does_not_exist_says_so(tmp_path: Path) -> None:
    with pytest.raises(ResumeError, match="does not exist"):
        resolve_checkpoint(tmp_path / "nowhere")


# ---- what a continuation inherits --------------------------------------------


def test_a_continuation_carries_the_original_run_settings(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _checkpoint(run, "policy-00400.pt", 400)

    carry = continuation(run)

    assert carry.run_dir == run
    assert carry.iteration == 400
    assert carry.train["envs"] == 2048
    assert carry.train["steps"] == 512
    assert carry.train["seed"] == 7
    assert carry.ppo["minibatches"] == 64
    assert carry.ppo["gamma"] == 0.999
    # The two non-potential reward terms genuinely change what the policy
    # converges to, so a continuation that dropped them would be a different
    # experiment wearing the same run's name.
    assert carry.shaping["city_weight"] == 250.0
    assert carry.shaping["multikill_bonus"] == 3.0


def test_a_continuation_does_not_inherit_this_runs_own_identity(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _checkpoint(run, "policy-00400.pt", 400)

    carry = continuation(run)

    # `out_dir` is where the *previous* run wrote, and the resumed checkpoint is
    # decided here rather than restated from a file that may name another one.
    assert "out_dir" not in carry.train
    assert "resume" not in carry.train


def test_the_checkpoint_outranks_the_config_file_on_what_must_match(tmp_path: Path) -> None:
    """A config.json can be stale or hand-edited; the weights cannot be."""
    stale = json.loads(json.dumps(CONFIG))
    stale["ppo"]["architecture"] = "mlp"
    stale["ppo"]["hidden"] = 64
    stale["ppo"]["learning_rate"] = 1.0
    run = _run(tmp_path, stale)
    _checkpoint(run, "policy-00400.pt", 400)

    carry = continuation(run)

    assert carry.ppo["architecture"] == "entity"
    assert carry.ppo["hidden"] == 512
    assert carry.ppo["learning_rate"] == pytest.approx(3.0e-4)
    assert carry.ppo["entropy_coef"] == pytest.approx(0.02)
    assert carry.train["learning_rate_final"] == pytest.approx(1.0e-5)
    assert carry.train["entropy_coef_final"] == pytest.approx(2.0e-3)


def test_the_remaining_updates_finish_the_original_horizon(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _checkpoint(run, "policy-00400.pt", 400)

    assert continuation(run).remaining == 600


def test_a_run_that_reached_its_horizon_has_nothing_left_to_finish(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _checkpoint(run, "policy-final.pt", 1000)

    assert continuation(run).remaining is None


def test_a_checkpoint_without_a_schedule_still_continues(tmp_path: Path) -> None:
    """Checkpoints predating the annealing schedule carry no endpoints."""
    run = _run(tmp_path, None)
    _checkpoint(run, "policy-final.pt", 120, schedule=None)

    carry = continuation(run)

    assert carry.iteration == 120
    assert carry.remaining is None
    assert "learning_rate" not in carry.ppo


def test_a_run_without_a_config_file_still_continues(tmp_path: Path) -> None:
    """A checkpoint synced from another machine, with nothing else beside it."""
    run = _run(tmp_path, None)
    _checkpoint(run, "policy-final.pt", 120)

    carry = continuation(run)

    assert carry.stored is None
    # Nothing to inherit but what the checkpoint itself carries.
    assert carry.ppo["architecture"] == "entity"
    assert carry.train["learning_rate_final"] == pytest.approx(1.0e-5)
    assert "envs" not in carry.train


def test_unknown_fields_in_a_config_file_are_ignored(tmp_path: Path) -> None:
    """A run trained by a newer trainer must not crash an older one."""
    future = json.loads(json.dumps(CONFIG))
    future["train"]["curriculum"] = "waves"
    future["ppo"]["attention_heads"] = 4
    run = _run(tmp_path, future)
    _checkpoint(run, "policy-00400.pt", 400)

    carry = continuation(run)

    assert "curriculum" not in carry.train
    assert "attention_heads" not in carry.ppo


# ---- the command line --------------------------------------------------------


class Started:
    """What `main` handed the training loop, instead of training anything."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.config: TrainConfig | None = None
        self.ppo: PPOConfig | None = None
        self.shaping: Shaping | None = None
        monkeypatch.setattr("md.train.train", self._record)

    def _record(self, config: TrainConfig, ppo: PPOConfig, shaping: Shaping) -> None:
        self.config, self.ppo, self.shaping = config, ppo, shaping


def test_resuming_a_run_writes_into_that_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path)
    _checkpoint(run, "policy-00400.pt", 400)
    started = Started(monkeypatch)

    assert main(["--resume", str(run)]) == 0

    assert started.config is not None
    assert started.ppo is not None and started.shaping is not None
    assert started.config.out_dir == run
    assert started.config.resume == run / "checkpoints" / "policy-00400.pt"
    assert started.config.envs == 2048
    assert started.config.updates == 600  # what is left of the original horizon
    assert started.ppo.architecture == "entity"
    assert started.shaping.city_weight == 250.0


def test_a_flag_on_the_command_line_beats_what_the_run_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path)
    _checkpoint(run, "policy-00400.pt", 400)
    started = Started(monkeypatch)

    assert main(["--resume", str(run), "--updates", "50", "--eval-every", "5"]) == 0

    assert started.config is not None
    assert started.config.updates == 50
    assert started.config.eval_every == 5
    assert started.config.steps == 512  # still inherited


def test_an_explicit_out_dir_still_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Continuing into a fresh directory is how a run is forked rather than extended."""
    run = _run(tmp_path)
    _checkpoint(run, "policy-00400.pt", 400)
    fork = tmp_path / "fork"
    started = Started(monkeypatch)

    assert main(["--resume", str(run), "--out-dir", str(fork)]) == 0

    assert started.config is not None
    assert started.config.out_dir == fork


def test_resuming_something_that_is_not_a_run_fails_with_a_sentence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--resume", str(tmp_path / "nowhere")]) == 2

    assert "does not exist" in capsys.readouterr().err
