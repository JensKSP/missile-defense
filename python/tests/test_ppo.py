# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for the entity-factored policy.

Only one thing here can fail *silently*, and it is the reason this file exists.
The policy emits a logit per (battery, threat slot) and flattens them into the
discrete action space; the simulation decodes that space as
``index = 1 + battery * threats + slot`` (``md::decode_action``). Get the
flattening wrong and nothing raises — the network simply learns against
permuted actions and converges on something mediocre, which is indistinguishable
from "the hyperparameters need work" until a great deal of time has been spent.

torch has no MinGW wheel, so these skip on the MSYS2 interpreter that runs the
gate here and execute on Linux/CI and on the native Windows interpreter.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch is optional; see docs/TRAINING.md")

from md.ppo import MASKED_LOGIT, EntityPolicy, ObsLayout, build_policy  # noqa: E402

THREATS, INTERCEPTORS, BLASTS, GLOBALS = 4, 3, 2, 5
BATTERIES = 3


def _layout() -> ObsLayout:
    size = (THREATS * 9) + (INTERCEPTORS * 7) + (BLASTS * 4) + GLOBALS
    return ObsLayout(threats=THREATS, interceptors=INTERCEPTORS, blasts=BLASTS, obs_size=size)


@pytest.fixture
def policy() -> EntityPolicy:
    return EntityPolicy(_layout(), 1 + (BATTERIES * THREATS), hidden=16)


def test_logit_order_matches_the_simulations_action_index(policy: EntityPolicy) -> None:
    # index = 1 + battery * threats + slot, exactly as decode_action reads it.
    obs = torch.randn(2, policy.layout.obs_size)
    mask = torch.ones(2, policy.action_count, dtype=torch.bool)
    logits, _ = policy(obs, mask)

    per_threat, summary = policy._features(obs)
    fire = policy.fire_head(per_threat)  # (batch, threats, batteries)
    for battery in range(BATTERIES):
        for slot in range(THREATS):
            index = 1 + (battery * THREATS) + slot
            assert torch.allclose(logits[:, index], fire[:, slot, battery], atol=1e-6)
    assert torch.allclose(logits[:, 0], policy.noop_head(summary).squeeze(-1), atol=1e-6)


def test_illegal_actions_are_masked_out(policy: EntityPolicy) -> None:
    obs = torch.randn(2, policy.layout.obs_size)
    mask = torch.zeros(2, policy.action_count, dtype=torch.bool)
    mask[:, 0] = True  # NoOp only
    logits, _ = policy(obs, mask)

    assert (logits[:, 1:] == MASKED_LOGIT).all()
    assert torch.softmax(logits, dim=-1)[:, 0].min() > 0.999


def test_absent_entities_do_not_reach_the_pooled_context(policy: EntityPolicy) -> None:
    # Padding slots read zero, and presence is feature 0 of every entity block.
    # If the pooling ever stopped masking on it, empty slots would contribute a
    # bias — a bug that shows up only as a policy that ignores sparse states.
    layout = policy.layout
    a = torch.zeros(1, layout.obs_size)
    b = torch.zeros(1, layout.obs_size)
    # Same one live threat in both; b additionally has junk in *absent* slots.
    for obs in (a, b):
        obs[0, 0:9] = torch.tensor([1.0, 0.5, 0.5, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0])
    b[0, layout.interceptors_at + 1 : layout.interceptors_at + 7] = 3.7  # present flag stays 0

    mask = torch.ones(1, policy.action_count, dtype=torch.bool)
    assert torch.allclose(policy(a, mask)[0], policy(b, mask)[0], atol=1e-6)


def test_layout_rejects_an_observation_that_cannot_hold_its_blocks() -> None:
    # A silent slice misalignment would train to plausible mediocrity, so the
    # layout is checked once at construction rather than trusted.
    bad = ObsLayout(threats=128, interceptors=32, blasts=16, obs_size=64)
    with pytest.raises(ValueError, match="no longer matches md::encode"):
        EntityPolicy(bad, 1 + (3 * 128), hidden=8)


def test_build_policy_selects_the_named_architecture() -> None:
    layout = _layout()
    count = 1 + (BATTERIES * THREATS)
    assert type(build_policy("entity", layout, count, 16)).__name__ == "EntityPolicy"
    assert type(build_policy("mlp", layout, count, 16)).__name__ == "Policy"
    with pytest.raises(ValueError, match="unknown architecture"):
        build_policy("transformer", layout, count, 16)
