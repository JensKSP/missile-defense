# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""A real training run, and every artifact it owes the rest of the system.

This is not checking that PPO learns — `test_ppo.py` does that, in-process and
fast. What it checks is the thing only a subprocess can: that a run started the
way a user starts one produces the files the console tails, the checkpoints the
model panel describes, and the recordings the game plays. Those are a *contract*
between three programs, and a contract is exactly the sort of thing that breaks
without any single component's tests noticing.

The run is deliberately tiny (`harness.TINY_RUN`): four envs, eight updates,
episodes capped at 400 ticks. The cap is load-bearing — a recording is only
written when an episode *finishes*, so an uncapped run of this length would
produce curves and checkpoints but nothing watchable.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest
from md.benchmark import (
    CANONICAL_BASELINE_MEAN_SCORE,
    CANONICAL_FRAME_SKIP,
    CANONICAL_MAX_TICKS,
    CANONICAL_SEED_OFFSET,
    SEEDS_PER_SPLIT,
    VALIDATION_SEED_OFFSET,
    VALIDATION_SPLIT,
)

from .harness import agent_eval, needs_agent_eval, needs_native, needs_torch, recordings

pytestmark = [pytest.mark.e2e, needs_torch, needs_native]


def test_a_run_writes_the_curves_the_console_tails(trained_run: Path) -> None:
    metrics = trained_run / "metrics.csv"
    assert metrics.exists()
    rows = list(csv.DictReader(metrics.read_text(encoding="utf-8").splitlines()))
    assert rows, "metrics.csv has a header and no data"
    # One row per update, and the columns the console actually plots.
    for column in ("update", "return", "entropy", "value_loss", "clip_fraction"):
        assert column in rows[0], f"metrics.csv has no {column} column"


def test_a_run_scores_itself_on_the_validation_seeds(trained_run: Path) -> None:
    # evals.csv exists because the return in metrics.csv is shaped and scaled and
    # has no fixed relationship to a game score (docs/ROADMAP.md, M8). Routine
    # evaluation selects checkpoints on validation and never inspects held-out
    # canonical seeds.
    evals = trained_run / "evals.csv"
    assert evals.exists()
    rows = list(csv.DictReader(evals.read_text(encoding="utf-8").splitlines()))
    assert rows, "evals.csv has no evaluation in it"
    latest = rows[-1]
    assert float(latest["mean_score"]) > 0.0
    assert latest["seed_split"] == VALIDATION_SPLIT
    assert int(latest["seed_offset"]) == VALIDATION_SEED_OFFSET
    assert int(latest["seed_count"]) == SEEDS_PER_SPLIT
    assert int(latest["frame_skip"]) == 4
    assert int(latest["max_ticks"]) == 400
    assert latest["inference_device"] == "cpu"
    # Only the columns the shared C++ `Summary` has always had. Task 11 widens
    # this file considerably, and its own e2e asserts the new ones — this test
    # must not start failing the moment a column is *added*, which is precisely
    # the compatibility the console's by-name column matching also provides.
    for column in ("update", "mean_score", "mean_wave", "mean_accuracy", "episodes"):
        assert column in rows[0], f"evals.csv has no {column} column"


#: The five bins, in the order `md.train.EVAL_COLUMNS` writes them.
EVAL_HISTOGRAM = ("shots_0kill", "shots_1kill", "shots_2kill", "shots_3kill", "shots_4plus")

#: The per-episode means the statistics added, beside the nine original columns.
EVAL_STATISTICS = (
    "mean_ticks",
    "mean_waves_cleared",
    "mean_cities_lost",
    "mean_bases_left",
    "mean_bases_lost",
    "mean_ammo_left",
    "mean_bonus_cities",
    "mean_mirv_splits",
    "mean_shots",
    "mean_kills",
    "mean_hits",
    "mean_hit_rate",
)


def _plausible(row: dict[str, str]) -> None:
    """Assert an evaluation row could have come from a game that was played.

    A column of zeroes is exactly the failure this catches: the wiring can be
    complete from `Sim` through the bindings to the CSV and still deliver
    nothing, and a test that only checked the header would pass on it.
    """
    ticks, shots = float(row["mean_ticks"]), float(row["mean_shots"])
    hits, kills = float(row["mean_hits"]), float(row["mean_kills"])
    histogram = [int(row[name]) for name in EVAL_HISTOGRAM]

    assert ticks > 0, "episodes that lasted no time at all"
    assert shots > 0, "an evaluation in which nothing was ever fired"
    assert kills > 0, "shots were fired and nothing was ever hit"
    # An interceptor cannot hit more often than it was launched, and the rate is
    # a fraction. Both would survive a units mix-up that the presence check misses.
    assert hits <= shots
    assert 0.0 <= float(row["mean_hit_rate"]) <= 1.0
    # Nothing may go backwards, and the counts are bounded by the board.
    assert float(row["mean_waves_cleared"]) >= 0.0
    assert float(row["mean_cities_lost"]) >= 0.0
    assert float(row["mean_bonus_cities"]) >= 0.0
    assert float(row["mean_mirv_splits"]) >= 0.0
    assert float(row["mean_ammo_left"]) >= 0.0
    assert 0.0 <= float(row["mean_bases_left"]) <= 3.0
    # The distribution: every resolved interceptor is in exactly one bin, and bin
    # 0 is the wasted ones. If the histogram were never filled this is what says so.
    assert sum(histogram) > 0, "the kills-per-shot histogram is empty"
    assert all(count >= 0 for count in histogram)


def test_a_runs_evaluation_carries_the_full_statistics(trained_run: Path) -> None:
    # Task 11's claim, from the file end: survival, damage, spend and the
    # kills-per-shot distribution reach the CSV a person opens, with values that
    # could have come from a game. Asserted by reading what a real run wrote,
    # never by calling summarize() — the wiring from Sim through the bindings and
    # the trainer is the part that can break without any unit test noticing.
    evals = trained_run / "evals.csv"
    rows = list(csv.DictReader(evals.read_text(encoding="utf-8").splitlines()))
    assert rows, "evals.csv has no evaluation in it"
    for column in (*EVAL_STATISTICS, *EVAL_HISTOGRAM):
        assert column in rows[0], f"evals.csv has no {column} column"
    _plausible(rows[-1])


@needs_agent_eval
def test_the_evaluator_prints_the_full_statistics_block() -> None:
    # The other end of the same claim, and the one with no Python in it: the
    # scripted baseline's own binary printing the same C++ Summary. Two seeds and
    # a short cap, because this is checking the printout rather than the baseline
    # — the canonical 32-seed number is what `poe eval` is for.
    result = agent_eval("--seeds", "2", "--seed-offset", "0", "--max-ticks", "2000")
    assert result.returncode == 0, result.stderr
    assert "seed stream offset 0" in result.stdout
    for line in ("mean score", "survived", "last wave", "cities", "bases", "ammo unfired"):
        assert line in result.stdout, f"the eval printout has no {line!r} line"
    # The distribution, spelled out — the one statistic that is a shape rather
    # than a number, and the reason the block exists at all.
    assert "kills per shot" in result.stdout
    for label in ("0:", "1:", "2:", "3:", "4+:"):
        assert label in result.stdout, f"the kills-per-shot line has no {label!r} bin"


@needs_agent_eval
def test_the_default_evaluator_is_the_published_held_out_benchmark() -> None:
    result = agent_eval(
        "--seeds",
        str(SEEDS_PER_SPLIT),
        "--frame-skip",
        str(CANONICAL_FRAME_SKIP),
        "--max-ticks",
        str(CANONICAL_MAX_TICKS),
    )
    assert result.returncode == 0, result.stderr
    match = re.search(
        r"mean score\s+([0-9.]+)\s+\[([0-9]+) \.\. ([0-9]+)\]",
        result.stdout,
    )
    assert match is not None
    assert float(match.group(1)) == pytest.approx(CANONICAL_BASELINE_MEAN_SCORE, abs=0.05)
    assert (int(match.group(2)), int(match.group(3))) == (83_525, 108_920)
    assert f"seed stream offset {CANONICAL_SEED_OFFSET}" in result.stdout
    assert re.search(r"last wave\s+15\.75", result.stdout)
    assert re.search(r"shots fired.*1\.09 kills/shot", result.stdout)


def test_a_run_records_what_it_was_started_with(trained_run: Path) -> None:
    # Written by the trainer rather than the console, so a run started from a
    # terminal has one too — which is what makes the console able to describe a
    # run it did not start.
    config = json.loads((trained_run / "config.json").read_text(encoding="utf-8"))
    # Both halves, because a run is only reproducible from the pair: what the
    # loop was told to do, and what PPO was told to do with it.
    assert int(config["train"]["envs"]) == 4
    assert int(config["train"]["updates"]) == 8
    assert config["ppo"]["learning_rate"] > 0


def test_a_run_describes_the_network_it_is_training(trained_run: Path) -> None:
    # The console cannot open a .pt without torch, so the trainer writes this
    # instead and the model panel reads it (md.modelcard).
    card = json.loads((trained_run / "model.json").read_text(encoding="utf-8"))
    assert card["parameters"] > 0
    assert card["tensors"], "model.json lists no tensors"
    assert card["obs_size"] > 0
    assert card["action_count"] > 0
    assert card["architecture"]


def test_a_run_leaves_checkpoints_that_can_be_continued_from(trained_run: Path) -> None:
    checkpoints = sorted((trained_run / "checkpoints").glob("*.pt"))
    names = {path.name for path in checkpoints}
    # policy-final is what --resume takes; policy-best is what promotion will.
    assert "policy-final.pt" in names
    assert "policy-best.pt" in names
    assert any(name.startswith("policy-000") for name in names)


def test_a_run_drops_watchable_episodes(trained_run: Path) -> None:
    episodes = recordings(trained_run)
    assert episodes, "no update-*.mdr was written"
    # ~80 kB an episode is the design claim (four bytes per agent step); the
    # assertion is only that they are whole files rather than empty stubs.
    assert all(path.stat().st_size > 0 for path in episodes)


def test_a_run_logs_itself_so_a_console_can_attach_later(trained_run: Path) -> None:
    # md.runlog: the trainer writes its own log, which is what gives a run
    # started in a terminal a log pane in a console that never started it.
    log = (trained_run / "train.log").read_text(encoding="utf-8")
    assert "update" in log.lower()
