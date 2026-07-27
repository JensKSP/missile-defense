# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: OpenAI Codex
"""The trainer's validation/benchmark boundary and its recorded protocol."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch", reason="torch is not installed")
pytest.importorskip(
    "md._md_native",
    reason="the _md_native extension is not built (cmake -DMD_BUILD_BINDINGS=ON)",
)

import torch  # noqa: E402
from md import train as trainer  # noqa: E402
from md.benchmark import (  # noqa: E402
    CANONICAL_BASELINE_MEAN_SCORE,
    CANONICAL_FRAME_SKIP,
    CANONICAL_INFERENCE_DEVICE,
    CANONICAL_MAX_TICKS,
    CANONICAL_SEED_OFFSET,
    CANONICAL_SPLIT,
    SEEDS_PER_SPLIT,
    VALIDATION_SEED_OFFSET,
    VALIDATION_SPLIT,
)
from md.eval import default_seeds, validation_seeds  # noqa: E402
from md.ppo import PPOConfig  # noqa: E402
from md.train import TrainConfig  # noqa: E402
from torch import nn  # noqa: E402


def _summary(mean_score: float = 12_345.0) -> SimpleNamespace:
    return SimpleNamespace(
        mean_score=mean_score,
        min_score=10_000,
        max_score=15_000,
        mean_wave=7.5,
        mean_cities_left=2.0,
        mean_accuracy=1.1,
        survived=0,
        episodes=SEEDS_PER_SPLIT,
        mean_ticks=4_000.0,
        mean_waves_cleared=6.5,
        mean_cities_lost=4.0,
        mean_bases_left=2.0,
        mean_bases_lost=1.0,
        mean_ammo_left=3.0,
        mean_bonus_cities=0.5,
        mean_mirv_splits=1.0,
        mean_shots=80.0,
        mean_kills=88.0,
        mean_hits=78.0,
        mean_hit_rate=0.975,
        kills_per_shot=(2, 70, 7, 1, 0),
    )


def test_routine_scoring_uses_validation_and_forwards_the_training_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    marker = object()

    def fake_evaluate(policy: object, **kwargs: object) -> object:
        captured.update(kwargs)
        captured["policy"] = policy
        return marker

    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)
    result = trainer._score(
        nn.Identity(),
        torch.device("cpu"),
        frame_skip=7,
        max_ticks=9_999,
    )

    assert result is marker
    assert captured["seeds"] == validation_seeds()
    assert captured["frame_skip"] == 7
    assert captured["max_ticks"] == 9_999


def test_eval_csv_records_the_full_reproduction_protocol(tmp_path: Path) -> None:
    path = tmp_path / "evals.csv"
    trainer._log_eval(
        path,
        50,
        _summary(),
        seed_split=VALIDATION_SPLIT,
        seed_offset=VALIDATION_SEED_OFFSET,
        seed_count=SEEDS_PER_SPLIT,
        frame_skip=7,
        max_ticks=9_999,
        inference_device="cpu",
    )

    row = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))[0]
    assert row["seed_split"] == VALIDATION_SPLIT
    assert int(row["seed_offset"]) == VALIDATION_SEED_OFFSET
    assert int(row["seed_count"]) == SEEDS_PER_SPLIT
    assert int(row["frame_skip"]) == 7
    assert int(row["max_ticks"]) == 9_999
    assert row["inference_device"] == "cpu"


def test_resuming_an_old_eval_csv_atomically_migrates_its_rows(tmp_path: Path) -> None:
    path = tmp_path / "evals.csv"
    path.write_text("update,mean_score\n1,100\n", encoding="utf-8")

    trainer._log_eval(
        path,
        2,
        _summary(),
        seed_split=VALIDATION_SPLIT,
        seed_offset=VALIDATION_SEED_OFFSET,
        seed_count=SEEDS_PER_SPLIT,
        frame_skip=4,
        max_ticks=120_000,
        inference_device="cpu",
    )

    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 2
    assert rows[0]["update"] == "1"
    assert rows[0]["seed_split"] == ""
    assert rows[0]["seed_offset"] == ""
    assert rows[1]["seed_split"] == VALIDATION_SPLIT
    assert int(rows[1]["seed_offset"]) == VALIDATION_SEED_OFFSET
    assert int(rows[1]["frame_skip"]) == 4


def test_explicit_checkpoint_score_defaults_to_the_pinned_held_out_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def fake_load(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, int]]:
        captured["path"] = path
        captured["device"] = str(device)
        return nn.Identity(), {"iteration": 400}

    def fake_evaluate(policy: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        captured["policy"] = policy
        return _summary(CANONICAL_BASELINE_MEAN_SCORE)

    monkeypatch.setattr(trainer, "load_policy", fake_load)
    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)
    monkeypatch.setattr(trainer, "format_summary", lambda summary: "summary")

    checkpoint = tmp_path / "run" / "checkpoints" / "policy.pt"
    assert trainer.score_checkpoint(checkpoint) == 0

    assert captured["device"] == CANONICAL_INFERENCE_DEVICE
    assert captured["seeds"] == default_seeds()
    assert captured["frame_skip"] == CANONICAL_FRAME_SKIP
    assert captured["max_ticks"] == CANONICAL_MAX_TICKS
    output = capsys.readouterr().out
    assert f"seed offset {CANONICAL_SEED_OFFSET}" in output
    assert f"inference {CANONICAL_INFERENCE_DEVICE}" in output
    assert "held-out canonical benchmark" in output
    row = list(
        csv.DictReader((tmp_path / "run" / "evals.csv").read_text(encoding="utf-8").splitlines())
    )[0]
    assert row["seed_split"] == CANONICAL_SPLIT
    assert int(row["seed_offset"]) == CANONICAL_SEED_OFFSET
    assert int(row["frame_skip"]) == CANONICAL_FRAME_SKIP
    assert row["inference_device"] == CANONICAL_INFERENCE_DEVICE


def test_noncanonical_inference_backend_disables_the_published_delta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        trainer,
        "load_policy",
        lambda path, device: (nn.Identity(), {"iteration": 400}),
    )
    monkeypatch.setattr(
        trainer,
        "evaluate",
        lambda policy, **kwargs: _summary(CANONICAL_BASELINE_MEAN_SCORE + 1_000),
    )
    monkeypatch.setattr(trainer, "format_summary", lambda summary: "summary")

    assert trainer.score_checkpoint(tmp_path / "policy.pt", "cuda") == 0
    output = capsys.readouterr().out
    assert "scripted-baseline comparison disabled" in output
    assert "ahead of" not in output


def test_checkpoint_selection_state_survives_a_protocol_matched_resume() -> None:
    protocol = trainer._validation_protocol(TrainConfig(frame_skip=7, max_ticks=9_999), "cpu")
    record = trainer._best_validation_record(77_500.25, 400, protocol)
    assert record is not None

    payload = {"iteration": 450, "best_validation": record}
    assert trainer._restore_best_validation(payload, expected_protocol=protocol) == (77_500.25, 400)


def test_checkpoint_selection_state_is_not_reused_across_protocols() -> None:
    original = trainer._validation_protocol(TrainConfig(frame_skip=4), "cuda")
    record = trainer._best_validation_record(77_500.25, 400, original)
    assert record is not None

    changed = trainer._validation_protocol(TrainConfig(frame_skip=8), "cuda")
    score, iteration = trainer._restore_best_validation(
        {"iteration": 450, "best_validation": record},
        expected_protocol=changed,
    )
    assert score == float("-inf")
    assert iteration == 0


def test_resume_copies_the_verified_best_policy_into_a_new_run(tmp_path: Path) -> None:
    layout, action_count = trainer._current_environment_schema()
    ppo = PPOConfig(hidden=8)
    policy = trainer.build_policy(ppo.architecture, layout, action_count, ppo.hidden)
    optimizer = torch.optim.Adam(policy.parameters())
    schedule = trainer.LinearSchedule(1, 100, 3e-4, 1e-5, 0.02, 0.002)
    protocol = trainer._validation_protocol(TrainConfig(), "cpu")
    record = trainer._best_validation_record(77_500.25, 40, protocol)
    assert record is not None

    source = tmp_path / "source" / "checkpoints"
    best = source / "policy-best.pt"
    final = source / "policy-final.pt"
    trainer._save(
        policy,
        optimizer,
        40,
        (layout.obs_size, action_count),
        ppo,
        layout,
        schedule,
        best,
        best_validation=record,
    )
    trainer._save(
        policy,
        optimizer,
        50,
        (layout.obs_size, action_count),
        ppo,
        layout,
        schedule,
        final,
        best_validation=record,
    )
    resume_payload = torch.load(final, map_location="cpu", weights_only=True)
    destination = tmp_path / "continuation" / "checkpoints" / "policy-best.pt"

    assert trainer._materialize_restored_best(
        resume_path=final,
        resume_payload=resume_payload,
        destination=destination,
        score=77_500.25,
        iteration=40,
        expected_protocol=protocol,
        expected_layout=layout,
        expected_action_count=action_count,
        expected_ppo=ppo,
    )
    assert destination.read_bytes() == best.read_bytes()


def test_resume_does_not_claim_an_unavailable_earlier_best(tmp_path: Path) -> None:
    layout, action_count = trainer._current_environment_schema()
    ppo = PPOConfig(hidden=8)
    protocol = trainer._validation_protocol(TrainConfig(), "cpu")
    record = trainer._best_validation_record(77_500.25, 40, protocol)
    assert record is not None

    resume = tmp_path / "source" / "checkpoints" / "policy-final.pt"
    payload: dict[str, Any] = {"iteration": 50, "best_validation": record}
    destination = tmp_path / "continuation" / "checkpoints" / "policy-best.pt"

    assert not trainer._materialize_restored_best(
        resume_path=resume,
        resume_payload=payload,
        destination=destination,
        score=77_500.25,
        iteration=40,
        expected_protocol=protocol,
        expected_layout=layout,
        expected_action_count=action_count,
        expected_ppo=ppo,
    )
    assert not destination.exists()


def test_an_out_of_memory_crash_names_the_knob_that_fixes_it() -> None:
    # A CUDA OOM traceback names a tensor nobody chose and a byte count nobody
    # can act on. The knobs that caused it are all in this config, and the
    # cheapest fix — the same data in smaller pieces — is the one nobody guesses,
    # because every instinct says the *batch* is the problem.
    advice = trainer._out_of_memory_advice(
        TrainConfig(envs=4096, steps=512),
        PPOConfig(minibatches=8, architecture="entity"),
    )
    assert "out of GPU memory" in advice
    assert "batch 2,097,152 samples (4,096 envs x 512 steps)" in advice
    assert "minibatch 262,144 (--minibatches 8)" in advice
    assert "--minibatches 16" in advice, "the advice does not offer the cheapest fix"
    # And what it would have taken, so the number is comparable with a card's size.
    assert "Estimated peak for this configuration: 152." in advice


def test_the_handicap_reaches_the_validation_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both halves of the handicap, forwarded from the run's config.

    A policy trained under a limit and scored without it is being asked a
    different question — and the answer flatters it, because the limit is what
    made the task hard. This is also the signature the trainer got wrong once:
    `_score` took `aim_trail` but not `reaction_delay`, and every run configured
    with one died at its first evaluation.
    """
    captured: dict[str, object] = {}

    def fake_evaluate(policy: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)
    trainer._score(
        nn.Identity(),
        torch.device("cpu"),
        frame_skip=4,
        max_ticks=1_000,
        aim_trail=0.5,
        reaction_delay=7,
    )

    assert captured["aim_trail"] == 0.5
    assert captured["reaction_delay"] == 7
