# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: OpenAI Codex
"""Annealing and rollout-horizon behavior owned by the training loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is optional; see docs/TRAINING.md")
pytest.importorskip(
    "missile_defense._md_native",
    reason="the _md_native extension is not built (cmake -DMD_BUILD_BINDINGS=ON)",
)

from missile_defense.training.ppo import ObsLayout, PPOConfig  # noqa: E402
from missile_defense.training.train import (  # noqa: E402
    CheckpointCompatibilityError,
    LinearSchedule,
    TrainConfig,
    _resolve_schedule,
    _save,
    _write_config,
)


def _schedule() -> LinearSchedule:
    return LinearSchedule(
        start_update=1,
        end_update=1000,
        learning_rate_start=3.0e-4,
        learning_rate_final=1.0e-5,
        entropy_coef_start=0.02,
        entropy_coef_final=0.002,
    )


def test_default_rollout_reaches_twice_as_far_through_an_episode() -> None:
    assert TrainConfig().steps == 256


def test_linear_schedule_reaches_both_endpoints_and_clamps() -> None:
    schedule = _schedule()

    assert schedule.at(0) == pytest.approx((3.0e-4, 0.02))
    assert schedule.at(1) == pytest.approx((3.0e-4, 0.02))
    assert schedule.at(1000) == pytest.approx((1.0e-5, 0.002))
    assert schedule.at(1200) == pytest.approx((1.0e-5, 0.002))
    midpoint = schedule.at(500)
    assert 1.0e-5 < midpoint[0] < 3.0e-4
    assert 0.002 < midpoint[1] < 0.02


def test_fresh_schedule_uses_the_run_length_by_default() -> None:
    config = TrainConfig(updates=800)
    ppo = PPOConfig(learning_rate=3.0e-4, entropy_coef=0.02)

    schedule = _resolve_schedule(config, ppo, None, first=1)

    assert schedule.start_update == 1
    assert schedule.end_update == 800
    assert schedule.at(800) == pytest.approx((1.0e-5, 0.002))


def test_resume_restores_the_original_schedule_without_restarting() -> None:
    schedule = _schedule()
    payload = {"iteration": 400, "schedule": schedule.__dict__}
    config = TrainConfig(updates=600)
    ppo = PPOConfig(learning_rate=3.0e-4, entropy_coef=0.02)

    restored = _resolve_schedule(config, ppo, payload, first=401)

    assert restored == schedule
    assert restored.at(401) == pytest.approx(schedule.at(401))
    assert restored.at(401) != pytest.approx(restored.at(1))


def test_resume_rejects_schedule_settings_that_change_the_run() -> None:
    payload = {"iteration": 400, "schedule": _schedule().__dict__}
    config = TrainConfig(updates=600, entropy_coef_final=0.001)

    with pytest.raises(CheckpointCompatibilityError, match="final entropy coefficient is 0.002"):
        _resolve_schedule(config, PPOConfig(), payload, first=401)


def test_resume_rejects_invalid_schedule_metadata() -> None:
    payload = {"iteration": 400, "schedule": _schedule().__dict__ | {"end_update": 0}}

    with pytest.raises(CheckpointCompatibilityError, match="invalid annealing-schedule endpoints"):
        _resolve_schedule(TrainConfig(), PPOConfig(), payload, first=401)


def test_legacy_checkpoint_anneals_from_its_restored_optimizer_rate() -> None:
    payload = {"iteration": 400, "optimizer": {"param_groups": [{"lr": 8.0e-5}]}}
    config = TrainConfig(updates=100, learning_rate_final=1.0e-5)

    schedule = _resolve_schedule(config, PPOConfig(), payload, first=401)

    assert schedule.start_update == 401
    assert schedule.end_update == 500
    assert schedule.at(401)[0] == pytest.approx(8.0e-5)
    assert schedule.at(500)[0] == pytest.approx(1.0e-5)


def test_resolved_schedule_is_written_beside_the_requested_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = TrainConfig(updates=100, out_dir=tmp_path)
    schedule = _resolve_schedule(config, PPOConfig(), None, first=1)

    _write_config(path, config, PPOConfig(), schedule, tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["train"]["steps"] == 256
    assert payload["train"]["schedule_updates"] is None
    assert payload["schedule"]["end_update"] == 100
    assert payload["schedule"]["learning_rate_final"] == pytest.approx(1.0e-5)


def test_checkpoint_carries_the_schedule_needed_for_an_exact_resume(tmp_path: Path) -> None:
    policy = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    schedule = _schedule()
    path = tmp_path / "policy.pt"

    _save(
        policy,
        optimizer,
        400,
        (2, 1),
        PPOConfig(),
        ObsLayout(threats=0, interceptors=0, blasts=0, obs_size=2),
        schedule,
        path,
    )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["iteration"] == 400
    assert payload["schedule"] == schedule.__dict__
    assert _resolve_schedule(TrainConfig(), PPOConfig(), payload, first=401) == schedule
