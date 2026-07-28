# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""A real training run, and every artifact it owes the rest of the system.

This is not checking that PPO learns — `test_ppo.py` does that, in-process and
fast. What it checks is the thing only a subprocess can: that a run started the
way a user starts one produces the files the trainer tails, the checkpoints the
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
import subprocess
import sys
from pathlib import Path

import pytest
from md import cadence
from md.benchmark import (
    CANONICAL_BASELINE_MEAN_SCORE,
    CANONICAL_FRAME_SKIP,
    CANONICAL_LADDER,
    CANONICAL_MAX_TICKS,
    CANONICAL_SEED_OFFSET,
    SEEDS_PER_SPLIT,
    VALIDATION_LADDER,
    VALIDATION_SEED_OFFSET,
    VALIDATION_SPLIT,
    Ladder,
)
from md.control import TUNING_NAME, Control

from .harness import (
    PROJECT_ROOT,
    TINY_RUN,
    TRAIN_TIMEOUT_S,
    agent_eval,
    needs_agent_eval,
    needs_native,
    needs_torch,
    recordings,
    start_training,
    train_environ,
    wait_until,
)

pytestmark = [pytest.mark.e2e, needs_torch, needs_native]


def test_a_run_writes_the_curves_the_trainer_tails(trained_run: Path) -> None:
    metrics = trained_run / "metrics.csv"
    assert metrics.exists()
    rows = list(csv.DictReader(metrics.read_text(encoding="utf-8").splitlines()))
    assert rows, "metrics.csv has a header and no data"
    # One row per update, and the columns the trainer actually plots.
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
    # the compatibility the trainer's by-name column matching also provides.
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
    # The spread, the wave the run dies on and the kills per shot, all under the
    # published handicap — which is what the evaluator applies by default, and
    # what moved every one of these numbers. The scripted agent used to reach
    # wave 15.75 and average 1.09 kills a shot when it never mis-clicked and
    # never had to wait; it does not any more, and neither will a policy.
    assert (int(match.group(2)), int(match.group(3))) == (8_040, 20_270)
    assert f"seed stream offset {CANONICAL_SEED_OFFSET}" in result.stdout
    assert re.search(r"last wave\s+7\.16", result.stdout)
    assert re.search(r"shots fired.*0\.73 kills/shot", result.stdout)


@needs_agent_eval
@pytest.mark.parametrize(
    ("ladder", "seed_offset"),
    [
        (CANONICAL_LADDER, CANONICAL_SEED_OFFSET),
        (VALIDATION_LADDER, VALIDATION_SEED_OFFSET),
    ],
    ids=["canonical", "validation"],
)
def test_every_rung_of_a_ladder_is_what_the_scripted_agent_scores_on_that_block(
    ladder: Ladder, seed_offset: int
) -> None:
    # The trainer draws three lines per block and tells a learner which one it
    # has beaten; each rung is a promise about what `--skill <name>` actually
    # does on those seeds. A rung that has drifted from its agent is worse than
    # no rung at all — the trainer would be measuring a run against a number
    # nothing in the project produces. The two blocks are checked separately
    # because they are genuinely different numbers, which is the entire reason
    # a score is only ever read against its own.
    for rung in ladder.rungs:
        result = agent_eval(
            "--skill",
            rung.skill,
            "--seeds",
            str(SEEDS_PER_SPLIT),
            "--seed-offset",
            str(seed_offset),
            "--frame-skip",
            str(CANONICAL_FRAME_SKIP),
            "--max-ticks",
            str(CANONICAL_MAX_TICKS),
        )
        assert result.returncode == 0, result.stderr
        match = re.search(r"mean score\s+([0-9.]+)", result.stdout)
        assert match is not None, f"--skill {rung.skill} printed no mean score"
        assert float(match.group(1)) == pytest.approx(rung.mean_score, abs=0.05), (
            f"the {ladder.block} {rung.label} rung is {rung.mean_score}, "
            f"but --skill {rung.skill} scores {match.group(1)} on that block"
        )

    scores = [rung.mean_score for rung in ladder.rungs]
    assert scores == sorted(scores), f"the {ladder.block} ladder is not a ladder"


@needs_agent_eval
def test_the_published_baseline_is_the_top_canonical_rung() -> None:
    # Everything outside the trainer quotes one number. It has to keep being the
    # top of the canonical ladder, or the two would drift apart silently.
    assert CANONICAL_LADDER.rungs[-1].mean_score == CANONICAL_BASELINE_MEAN_SCORE
    assert CANONICAL_LADDER.rungs[-1].skill == "high"


def test_a_run_records_what_it_was_started_with(trained_run: Path) -> None:
    # Written by the trainer rather than the trainer, so a run started from a
    # terminal has one too — which is what makes the trainer able to describe a
    # run it did not start.
    config = json.loads((trained_run / "config.json").read_text(encoding="utf-8"))
    # Both halves, because a run is only reproducible from the pair: what the
    # loop was told to do, and what PPO was told to do with it.
    assert int(config["train"]["envs"]) == 4
    assert int(config["train"]["updates"]) == 8
    assert config["ppo"]["learning_rate"] > 0


def test_a_run_describes_the_network_it_is_training(trained_run: Path) -> None:
    # The trainer cannot open a .pt without torch, so the trainer writes this
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


def test_a_run_publishes_the_cadence_it_is_evaluating_on(trained_run: Path) -> None:
    # Published rather than merely obeyed: the trainer's eval box, and anyone
    # with `cat`, has to be able to ask a running trainer what it is on — and a
    # value that only exists inside the process cannot be asked.
    published = json.loads((trained_run / TUNING_NAME).read_text(encoding="utf-8"))
    assert published["eval_every"] == int(TINY_RUN["--eval-every"])


def test_a_live_run_takes_a_new_eval_cadence_without_being_restarted(tmp_path: Path) -> None:
    """The claim the tuning file exists to make: no restart, no lost checkpoint.

    Started with evaluation switched off entirely, told to score every 30 updates
    while it is running, and it does. `test_control.py` covers the file; only a
    real loop can show that the loop re-reads it.

    The pause is the synchronisation. A tiny run is a few seconds long, so the
    test stops its clock rather than racing it — which is also the mechanism
    doing exactly what it is for.
    """
    out_dir = tmp_path / "run"
    control = Control(out_dir)
    process = start_training(
        out_dir,
        overrides={"--updates": "60", "--eval-every": "0", "--record-every": "0"},
    )
    try:
        wait_until(
            control.tuning_file.exists,
            timeout=TRAIN_TIMEOUT_S,
            what="the run publishing what it was started with",
        )
        assert control.tuning() == {"eval_every": 0}, "the command line is what a run publishes"

        control.request_pause()
        log = out_dir / "train.log"
        wait_until(
            lambda: log.exists() and "paused after update" in log.read_text(encoding="utf-8"),
            timeout=TRAIN_TIMEOUT_S,
            what="the run pausing between updates",
        )
        control.tune("eval_every", 30)
        control.resume()
        assert process.wait(timeout=TRAIN_TIMEOUT_S) == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()

    assert "eval interval 0 -> 30 updates" in log.read_text(encoding="utf-8"), (
        "the change is not in the log, so the run never noticed it"
    )
    rows = list(csv.DictReader((out_dir / "evals.csv").read_text(encoding="utf-8").splitlines()))
    scored = [int(row["update"]) for row in rows]
    # Not `[30, 60]`: the cadence *ramps* (md.cadence), and the schedule is a
    # function of the update number alone. So a run that adopts a 30-update
    # interval part-way through joins the schedule it would have been on all
    # along — dense over the updates where a policy still changes shape — rather
    # than counting thirty from wherever it happened to be told. Which updates
    # those are is asked of `cadence`, because a list of numbers here would say
    # nothing about why they are those and would drift the moment the ramp did.
    ramp = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))["train"][
        "eval_ramp_until"
    ]
    ramped = cadence.schedule(interval=30, ramp_until=ramp, last=60)
    assert scored, "the run scored nothing at all after being told to score"
    assert scored[0] in ramped and scored == ramped[ramped.index(scored[0]) :], (
        f"a run told to score every 30 updates scored at {scored}, "
        f"which is not the tail of the 30-update ramp {ramped}"
    )
    assert scored[-1] == 60, "the run stopped scoring before its last update"


def test_a_run_logs_itself_so_a_trainer_can_attach_later(trained_run: Path) -> None:
    # md.runlog: the trainer writes its own log, which is what gives a run
    # started in a terminal a log pane in a trainer that never started it.
    log = (trained_run / "train.log").read_text(encoding="utf-8")
    assert "update" in log.lower()


def test_a_stopped_run_is_continued_from_its_own_directory(
    trained_run: Path, tmp_path: Path
) -> None:
    """The whole point of `--resume <run>`: no flag of the original restated.

    The tiny run is deliberately non-default in eight places (`TINY_RUN`), and
    every one of them has to come back — `--envs 4` above all, because a
    continuation that quietly used the default 1,024 would be a different
    experiment wearing the same run's name, and nothing on screen would say so.
    """
    import shutil  # noqa: PLC0415 — this test copies the shared run before writing

    run = tmp_path / "continued"
    shutil.copytree(trained_run, run)
    before = len((run / "metrics.csv").read_text(encoding="utf-8").splitlines())

    result = subprocess.run(
        [sys.executable, "-m", "md.train", "--resume", str(run), "--updates", "1"],
        capture_output=True,
        text=True,
        timeout=TRAIN_TIMEOUT_S,
        env=train_environ(),
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "continuing" in result.stdout
    # Into the same run, not into ./runs: the history stays one story.
    assert len((run / "metrics.csv").read_text(encoding="utf-8").splitlines()) == before + 1
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert int(config["train"]["envs"]) == int(TINY_RUN["--envs"])
    assert int(config["train"]["max_ticks"]) == int(TINY_RUN["--max-ticks"])
    assert config["train"]["resume"].endswith(".pt")


def test_a_continuation_picks_up_at_the_update_the_run_stopped_on(
    trained_run: Path, tmp_path: Path
) -> None:
    import shutil  # noqa: PLC0415 — this test copies the shared run before writing

    run = tmp_path / "numbering"
    shutil.copytree(trained_run, run)

    result = subprocess.run(
        [sys.executable, "-m", "md.train", "--resume", str(run), "--updates", "1"],
        capture_output=True,
        text=True,
        timeout=TRAIN_TIMEOUT_S,
        env=train_environ(),
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    # The run did `--updates 8`, so the continuation is update 9 — not update 1
    # again. Restarting the numbering would draw a curve that folds back on
    # itself and make `metrics.csv` unreadable as one history.
    updates = int(TINY_RUN["--updates"])
    assert f"update {updates + 1:>5}" in result.stdout


def test_the_settings_of_a_run_can_be_read_back_without_opening_the_file(
    trained_run: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "md.train", "--show-config", str(trained_run)],
        capture_output=True,
        text=True,
        timeout=TRAIN_TIMEOUT_S,
        env=train_environ(),
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"envs={TINY_RUN['--envs']}" in result.stdout
    assert "architecture=" in result.stdout


def test_a_run_prints_the_settings_it_resolved_to(trained_run: Path) -> None:
    """In the log as well as the terminal, which is where the trainer reads it."""
    log = (trained_run / "train.log").read_text(encoding="utf-8")

    assert f"envs={TINY_RUN['--envs']}" in log
    assert "shaping" in log
