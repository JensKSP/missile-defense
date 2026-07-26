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
from pathlib import Path

import pytest

from .harness import needs_native, needs_torch, recordings

pytestmark = [pytest.mark.e2e, needs_torch, needs_native]


def test_a_run_writes_the_curves_the_console_tails(trained_run: Path) -> None:
    metrics = trained_run / "metrics.csv"
    assert metrics.exists()
    rows = list(csv.DictReader(metrics.read_text(encoding="utf-8").splitlines()))
    assert rows, "metrics.csv has a header and no data"
    # One row per update, and the columns the console actually plots.
    for column in ("update", "return", "entropy", "value_loss", "clip_fraction"):
        assert column in rows[0], f"metrics.csv has no {column} column"


def test_a_run_scores_itself_against_the_canonical_seeds(trained_run: Path) -> None:
    # evals.csv exists because the return in metrics.csv is shaped and scaled and
    # has no fixed relationship to a game score (docs/ROADMAP.md, M8). It is the
    # only column comparable to the 113,834 baseline, so the console plots it.
    evals = trained_run / "evals.csv"
    assert evals.exists()
    rows = list(csv.DictReader(evals.read_text(encoding="utf-8").splitlines()))
    assert rows, "evals.csv has no evaluation in it"
    assert float(rows[-1]["mean_score"]) > 0.0
    # Only the columns the shared C++ `Summary` has always had. Task 11 widens
    # this file considerably, and its own e2e asserts the new ones — this test
    # must not start failing the moment a column is *added*, which is precisely
    # the compatibility the console's by-name column matching also provides.
    for column in ("update", "mean_score", "mean_wave", "mean_accuracy", "episodes"):
        assert column in rows[0], f"evals.csv has no {column} column"


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
