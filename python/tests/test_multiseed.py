# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: OpenAI Codex
"""Multi-seed experiments stay fresh and never select on the final benchmark."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest
from missile_defense.sim.benchmark import VALIDATION_SPLIT
from missile_defense.training.multiseed import (
    ExperimentError,
    best_validation_result,
    run_directory,
    run_experiment,
    seed_values,
    select_winner,
    trainer_arguments,
    training_command,
    write_summary,
)

FIELDS = (
    "update",
    "mean_score",
    "seed_split",
    "seed_offset",
    "seed_count",
    "frame_skip",
    "max_ticks",
    "inference_device",
)


def _eval(
    run_dir: Path,
    *,
    score: float,
    update: int,
    split: str = VALIDATION_SPLIT,
    offset: int = 0,
    device: str = "cuda:0",
) -> None:
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    (checkpoints / "policy-best.pt").write_bytes(b"checkpoint")
    with (run_dir / "evals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "update": update,
                "mean_score": score,
                "seed_split": split,
                "seed_offset": offset,
                "seed_count": 32,
                "frame_skip": 4,
                "max_ticks": 120_000,
                "inference_device": device,
            }
        )


def test_seed_schedule_is_explicit_and_contiguous() -> None:
    assert seed_values(1000, 3) == [1000, 1001, 1002]
    with pytest.raises(ExperimentError, match="at least 1"):
        seed_values(0, 0)


def test_command_owns_freshness_arguments(tmp_path: Path) -> None:
    command = training_command(
        "/opt/python",
        tmp_path / "experiment",
        17,
        ["--updates", "500", "--architecture", "entity"],
    )

    assert command[:4] == ["/opt/python", "-u", "-m", "missile_defense.training"]
    assert command[4:8] == [
        "--seed",
        "17",
        "--out-dir",
        str(tmp_path / "experiment" / "seed-000017"),
    ]
    assert "--resume" not in command and "--load" not in command

    for forbidden in ("--resume", "--load=x.pt", "--seed=2", "--out-dir"):
        with pytest.raises(ExperimentError, match="owned by"):
            trainer_arguments([forbidden])


def test_best_result_ignores_canonical_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "seed-000003"
    _eval(run_dir, score=42_000, update=50)
    with (run_dir / "evals.csv").open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writerow(
            {
                "update": 50,
                "mean_score": 999_999,
                "seed_split": "canonical",
                "seed_offset": 32,
                "seed_count": 32,
                "frame_skip": 4,
                "max_ticks": 120_000,
                "inference_device": "cpu",
            }
        )

    result = best_validation_result(3, run_dir)

    assert result.validation_score == 42_000
    assert result.protocol is not None
    assert result.protocol.seed_split == VALIDATION_SPLIT


def test_canonical_only_run_is_not_a_candidate(tmp_path: Path) -> None:
    run_dir = tmp_path / "seed-000004"
    _eval(run_dir, score=999_999, update=50, split="canonical", offset=32)

    result = best_validation_result(4, run_dir)

    assert result.validation_score is None
    assert result.status == "incomplete"
    assert result.error is not None and "do not substitute a canonical" in result.error


def test_selection_requires_matching_validation_protocols(tmp_path: Path) -> None:
    first_dir = tmp_path / "seed-000001"
    second_dir = tmp_path / "seed-000002"
    _eval(first_dir, score=50_000, update=100, offset=0)
    _eval(second_dir, score=60_000, update=100, offset=64)
    results = [
        best_validation_result(1, first_dir),
        best_validation_result(2, second_dir),
    ]

    with pytest.raises(ExperimentError, match="protocols differ"):
        select_winner(results)


def test_summary_selects_highest_validation_score_and_records_no_canonical_use(
    tmp_path: Path,
) -> None:
    first_dir = run_directory(tmp_path, 10)
    second_dir = run_directory(tmp_path, 11)
    _eval(first_dir, score=70_000, update=300)
    _eval(second_dir, score=75_000, update=250)
    results = [
        best_validation_result(10, first_dir),
        best_validation_result(11, second_dir),
    ]

    winner = write_summary(tmp_path, results)

    assert winner is not None and winner.train_seed == 11
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["canonical_benchmark_used"] is False
    assert payload["winner"]["train_seed"] == 11
    assert payload["winner"]["status"] == "selected"


def test_runner_refuses_a_nonempty_experiment_directory(tmp_path: Path) -> None:
    experiment = tmp_path / "used"
    experiment.mkdir()
    (experiment / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(ExperimentError, match="starts from zero"):
        run_experiment(experiment, [0], [], python="/opt/python")


def test_runner_launches_distinct_fresh_runs_and_aggregates_them(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is False
        commands.append(command)
        seed = int(command[command.index("--seed") + 1])
        run_dir = Path(command[command.index("--out-dir") + 1])
        _eval(run_dir, score=50_000 + seed, update=100)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    status = run_experiment(
        experiment,
        [20, 21, 22],
        ["--updates", "100", "--architecture", "entity"],
        python="/opt/python",
    )

    assert status == 0
    assert [command[command.index("--seed") + 1] for command in commands] == [
        "20",
        "21",
        "22",
    ]
    assert len({command[command.index("--out-dir") + 1] for command in commands}) == 3
    manifest = json.loads((experiment / "experiment.json").read_text(encoding="utf-8"))
    assert manifest["exit_codes"] == {"20": 0, "21": 0, "22": 0}
    summary = json.loads((experiment / "summary.json").read_text(encoding="utf-8"))
    assert summary["winner"]["train_seed"] == 22
