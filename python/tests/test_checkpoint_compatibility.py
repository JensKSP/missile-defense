# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: OpenAI Codex
"""Checkpoint schema failures should explain themselves before a rollout starts."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="torch is optional; see docs/TRAINING.md")
pytest.importorskip(
    "md._md_native", reason="the _md_native extension is not built (cmake -DMD_BUILD_BINDINGS=ON)"
)

from md.env import VecEnv  # noqa: E402
from md.ppo import PPOConfig, build_policy  # noqa: E402
from md.train import (  # noqa: E402
    CheckpointCompatibilityError,
    TrainConfig,
    _layout,
    load_policy,
    score_checkpoint,
    train,
)


def _payload() -> dict[str, Any]:
    env = VecEnv(num_envs=0, threads=1)
    layout = _layout(env)
    return {
        # Deliberately invalid weights: every incompatibility test must fail
        # before a network is constructed or load_state_dict is reached.
        "policy": {},
        "optimizer": {},
        "iteration": 12,
        "obs_size": env.obs_size,
        "action_count": env.action_count,
        "hidden": 8,
        "architecture": "mlp",
        "layout": dataclasses.asdict(layout),
    }


def _write(path: Path, payload: dict[str, Any]) -> Path:
    torch.save(payload, path)
    return path


def _resume_config(tmp_path: Path, checkpoint: Path) -> TrainConfig:
    return TrainConfig(
        envs=1,
        steps=1,
        updates=0,
        eval_every=0,
        record_every=0,
        checkpoint_every=0,
        device="cpu",
        out_dir=tmp_path / "run",
        resume=checkpoint,
    )


def _old_blast_schema() -> dict[str, Any]:
    payload = _payload()
    # The lifetime-phase feature added one float to each of 64 blast slots.
    payload["obs_size"] = 1895
    payload["layout"]["obs_size"] = 1895
    payload["layout"]["blast_features"] = 4
    return payload


def test_score_rejects_the_old_blast_schema_before_loading_weights(tmp_path: Path) -> None:
    checkpoint = _write(tmp_path / "old-policy.pt", _old_blast_schema())

    with pytest.raises(CheckpointCompatibilityError) as caught:
        score_checkpoint(checkpoint, device_name="cpu")

    message = str(caught.value)
    assert "observation size is 1895, current environment is 1959" in message
    assert "layout.blast_features is 4, current environment is 5" in message
    assert "cannot be scored or resumed safely" in message


def test_load_rejects_an_action_shape_mismatch(tmp_path: Path) -> None:
    payload = _payload()
    payload["action_count"] -= 1
    checkpoint = _write(tmp_path / "wrong-actions.pt", payload)

    with pytest.raises(CheckpointCompatibilityError, match="action count is 384"):
        load_policy(checkpoint)


def test_load_rejects_an_entity_layout_mismatch_even_at_the_same_size(tmp_path: Path) -> None:
    payload = _payload()
    payload["layout"]["blast_features"] = 4
    checkpoint = _write(tmp_path / "wrong-layout.pt", payload)

    with pytest.raises(
        CheckpointCompatibilityError,
        match=r"layout\.blast_features is 4, current environment is 5",
    ):
        load_policy(checkpoint)


def test_training_resume_rejects_the_old_schema_before_building_a_policy(tmp_path: Path) -> None:
    checkpoint = _write(tmp_path / "old-resume.pt", _old_blast_schema())

    with pytest.raises(CheckpointCompatibilityError, match="observation size is 1895"):
        train(_resume_config(tmp_path, checkpoint), PPOConfig(hidden=8))


@pytest.mark.parametrize(
    ("field", "stored", "ppo", "message"),
    [
        (
            "architecture",
            "entity",
            PPOConfig(architecture="mlp", hidden=8),
            "architecture is 'entity', requested PPOConfig is 'mlp'",
        ),
        (
            "hidden",
            16,
            PPOConfig(architecture="mlp", hidden=8),
            "hidden size is 16, requested PPOConfig is 8",
        ),
    ],
)
def test_training_resume_rejects_a_different_policy_configuration(
    tmp_path: Path,
    field: str,
    stored: object,
    ppo: PPOConfig,
    message: str,
) -> None:
    payload = _payload()
    payload[field] = stored
    checkpoint = _write(tmp_path / f"wrong-{field}.pt", payload)

    with pytest.raises(CheckpointCompatibilityError, match=message):
        train(_resume_config(tmp_path, checkpoint), ppo)


def test_load_policy_uses_the_checkpoints_architecture_and_hidden_size(tmp_path: Path) -> None:
    payload = _payload()
    env = VecEnv(num_envs=0, threads=1)
    layout = _layout(env)
    payload["architecture"] = "entity"
    payload["hidden"] = 48
    payload["policy"] = build_policy("entity", layout, env.action_count, 48).state_dict()
    checkpoint = _write(tmp_path / "stored-network.pt", payload)

    policy, loaded_payload = load_policy(checkpoint)

    assert type(policy).__name__ == "EntityPolicy"
    assert policy.context[0].out_features == 48
    assert loaded_payload["hidden"] == 48
