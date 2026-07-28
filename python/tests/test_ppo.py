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

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    # The real module, so `torch.Tensor` is a type below rather than an
    # attribute of whatever `importorskip` returned.
    import torch
else:
    torch = pytest.importorskip("torch", reason="torch is optional; see docs/TRAINING.md")

from missile_defense.training import ppo as ppo_module  # noqa: E402
from missile_defense.training.auxiliary import AuxiliaryTargets  # noqa: E402
from missile_defense.training.auxiliary import targets as auxiliary_targets  # noqa: E402
from missile_defense.training.ppo import (  # noqa: E402
    MASKED_LOGIT,
    EntityPolicy,
    ObsLayout,
    PPOConfig,
    Rollout,
    build_policy,
    update,
)

THREATS, INTERCEPTORS, BLASTS, GLOBALS = 4, 3, 2, 5
BATTERIES = 3


def _layout() -> ObsLayout:
    size = (THREATS * 9) + (INTERCEPTORS * 7) + (BLASTS * 5) + GLOBALS
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
    b[0, layout.blasts_at + 1 : layout.blasts_at + 5] = 2.4  # including lifetime phase

    mask = torch.ones(1, policy.action_count, dtype=torch.bool)
    assert torch.allclose(policy(a, mask)[0], policy(b, mask)[0], atol=1e-6)


def test_entity_slot_order_does_not_change_relational_features(policy: EntityPolicy) -> None:
    layout = policy.layout
    obs = torch.zeros(1, layout.obs_size)
    obs[0, 0:9] = torch.tensor([1.0, 0.2, 0.3, 0.1, -0.2, 1.0, 0.0, 0.0, 0.0])
    interceptors = obs[:, layout.interceptors_at : layout.blasts_at].view(
        1, INTERCEPTORS, layout.interceptor_features
    )
    interceptors[0, 0] = torch.tensor([1.0, 0.1, 0.2, 0.0, -0.1, 0.3, 0.4])
    interceptors[0, 1] = torch.tensor([1.0, 0.8, 0.7, -0.1, 0.0, 0.6, 0.5])
    blasts = obs[:, layout.blasts_at : layout.globals_at].view(1, BLASTS, layout.blast_features)
    blasts[0, 0] = torch.tensor([1.0, 0.3, 0.4, 0.2, 0.5])
    blasts[0, 1] = torch.tensor([1.0, 0.7, 0.6, 0.1, 0.2])

    permuted = obs.clone()
    permuted_i = permuted[:, layout.interceptors_at : layout.blasts_at].view(
        1, INTERCEPTORS, layout.interceptor_features
    )
    permuted_b = permuted[:, layout.blasts_at : layout.globals_at].view(
        1, BLASTS, layout.blast_features
    )
    permuted_i[:, [0, 1]] = permuted_i[:, [1, 0]].clone()
    permuted_b[:, [0, 1]] = permuted_b[:, [1, 0]].clone()

    features, summary = policy._actor_features(obs)
    permuted_features, permuted_summary = policy._actor_features(permuted)
    assert torch.allclose(features, permuted_features, atol=1e-6)
    assert torch.allclose(summary, permuted_summary, atol=1e-6)


def test_each_threat_gets_its_own_attended_relationship(policy: EntityPolicy) -> None:
    # Make the projections identities so two orthogonal threat queries must
    # select their corresponding interceptor. This distinguishes real
    # per-threat attention from broadcasting one pooled entity summary.
    attention = policy.interceptor_attention
    with torch.no_grad():
        identity = torch.eye(policy.width)
        attention.query.weight.copy_(identity)
        attention.key.weight.copy_(identity)
        attention.value.weight.copy_(identity)
        attention.output.weight.copy_(identity)
        attention.output.bias.zero_()
    encoded_t = torch.zeros(1, THREATS, policy.width)
    encoded_i = torch.zeros(1, INTERCEPTORS, policy.width)
    encoded_t[0, 0, 0] = 10.0
    encoded_t[0, 1, 1] = 10.0
    encoded_i[0, 0, 0] = 10.0
    encoded_i[0, 1, 1] = 10.0
    present = torch.tensor([[True, True, False]])

    related = attention(encoded_t, encoded_i, present)

    assert related.shape == (1, THREATS, policy.width)
    assert related[0, 0, 0] > related[0, 0, 1]
    assert related[0, 1, 1] > related[0, 1, 0]


def test_relational_actor_and_critic_gradients_are_disjoint(policy: EntityPolicy) -> None:
    network = policy
    obs = torch.randn(2, _layout().obs_size)
    # Presence flags need to describe live entities for attention gradients.
    threats = obs[:, : _layout().interceptors_at].view(2, THREATS, 9)
    interceptors = obs[:, _layout().interceptors_at : _layout().blasts_at].view(2, INTERCEPTORS, 7)
    blasts = obs[:, _layout().blasts_at : _layout().globals_at].view(2, BLASTS, 5)
    threats[..., 0] = 1.0
    interceptors[..., 0] = 1.0
    blasts[..., 0] = 1.0
    mask = torch.ones(2, 1 + (BATTERIES * THREATS), dtype=torch.bool)

    logits, values = network(obs, mask)
    values.square().mean().backward()
    actor_parameters = [
        parameter
        for name, parameter in network.named_parameters()
        if not name.startswith("critic_trunk.") and not name.startswith("value_head.")
    ]
    critic_parameters = [
        parameter
        for name, parameter in network.named_parameters()
        if name.startswith("critic_trunk.") or name.startswith("value_head.")
    ]
    assert all(parameter.grad is None for parameter in actor_parameters)
    assert any(parameter.grad is not None for parameter in critic_parameters)

    network.zero_grad(set_to_none=True)
    logits, _ = network(obs, mask)
    logits.square().mean().backward()
    assert all(parameter.grad is None for parameter in critic_parameters)
    assert any(parameter.grad is not None for parameter in actor_parameters)


def test_auxiliary_predictions_reuse_per_threat_features_and_mask_padding(
    policy: EntityPolicy,
) -> None:
    layout = policy.layout
    obs = torch.zeros(2, layout.obs_size)
    threats = obs[:, : layout.interceptors_at].view(2, THREATS, layout.threat_features)
    threats[:, 0] = torch.tensor([1.0, 0.3, 0.4, 0.0, -0.1, 1.0, 0.0, 0.0, 0.0])

    predictions = policy.auxiliary_predictions(obs)

    assert predictions.shape == (2, THREATS, 3)
    assert torch.count_nonzero(predictions[:, 1:]) == 0
    predictions[:, 0].sum().backward()  # type: ignore[no-untyped-call]
    assert policy.auxiliary_head.weight.grad is not None
    assert policy.relation[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in policy.critic_trunk.parameters())


def test_forward_with_auxiliary_matches_separate_interfaces(policy: EntityPolicy) -> None:
    obs = torch.randn(2, policy.layout.obs_size)
    threats = obs[:, : policy.layout.interceptors_at].view(
        2, THREATS, policy.layout.threat_features
    )
    threats[..., 0] = 1.0
    mask = torch.ones(2, policy.action_count, dtype=torch.bool)

    logits, values = policy(obs, mask)
    predictions = policy.auxiliary_predictions(obs)
    combined_logits, combined_values, combined_predictions = policy.forward_with_auxiliary(
        obs, mask
    )

    assert torch.allclose(combined_logits, logits)
    assert torch.allclose(combined_values, values)
    assert torch.allclose(combined_predictions, predictions)


def test_ppo_update_trains_the_auxiliary_head(
    policy: EntityPolicy, monkeypatch: pytest.MonkeyPatch
) -> None:
    rollout = Rollout(
        steps=1,
        num_envs=2,
        obs_size=policy.layout.obs_size,
        action_count=policy.action_count,
        device=torch.device("cpu"),
    )
    threats = rollout.obs[:, :, : policy.layout.interceptors_at].view(
        1, 2, THREATS, policy.layout.threat_features
    )
    threats[:, :, 0, :5] = torch.tensor([1.0, 0.2, 0.6, 0.0, -0.1])
    rollout.masks[:] = True
    with torch.no_grad():
        logits, values = policy(rollout.obs[0], rollout.masks[0])
        distribution = torch.distributions.Categorical(logits=logits)
        rollout.actions[0] = torch.tensor([0, 1])
        rollout.log_probs[0] = distribution.log_prob(  # type: ignore[no-untyped-call]
            rollout.actions[0]
        )
        rollout.values[0] = values

    before = policy.auxiliary_head.weight.detach().clone()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    calls: list[str] = []
    original_targets = auxiliary_targets
    original_forward = EntityPolicy.forward_with_auxiliary

    def tracked_targets(obs: torch.Tensor, layout: ObsLayout) -> AuxiliaryTargets:
        calls.append("targets")
        return original_targets(obs, layout)

    def tracked_forward(
        self: EntityPolicy, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        calls.append("forward")
        return original_forward(self, obs, mask)

    monkeypatch.setattr(ppo_module, "auxiliary_targets", tracked_targets)
    monkeypatch.setattr(EntityPolicy, "forward_with_auxiliary", tracked_forward)
    stats = update(
        policy,
        optimizer,
        rollout,
        advantages=torch.tensor([[1.0, -1.0]]),
        returns=rollout.values.clone(),
        config=PPOConfig(
            hidden=16,
            epochs=1,
            minibatches=1,
            value_coef=0.0,
            entropy_coef=0.0,
            auxiliary_coef=1.0,
            architecture="entity",
        ),
    )

    assert stats["auxiliary_loss"] > 0.0
    assert not torch.equal(policy.auxiliary_head.weight, before)
    assert calls == ["targets", "forward"]


def test_layout_accounts_for_the_blast_lifetime_phase() -> None:
    layout = _layout()
    assert layout.blast_features == 5
    assert layout.globals_at == (THREATS * 9) + (INTERCEPTORS * 7) + (BLASTS * 5)


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
