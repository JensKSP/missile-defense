# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""PPO — the policy, and the update rule.

Written to be *read*. This is a learning project, so the implementation is a
single straightforward PPO rather than a configurable framework: one network, one
buffer, one update function, each with the reasoning written down next to it.

Two things here are specific to this game and worth understanding before tuning
anything:

**Action masking.** Firing an empty battery, or at an empty threat slot, does
nothing. Those actions are masked out *before* the softmax, so the policy never
spends probability mass on them and never has to learn that they are wasted. The
mask comes from the simulation (`env.action_masks()`), not from a heuristic.

**Frame-skip is part of the action.** One agent step holds an action index across
several ticks, re-decoding it each tick — an engagement is a steer-then-fire
macro. So one "step" here is `frame_skip` ticks of game time, which is why the
step counts below look small next to the ticks actually simulated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Actions that the mask forbids get this logit, so softmax gives them ~0
# probability. Not -inf: that produces NaNs if an env ever masks *everything*.
MASKED_LOGIT = -1.0e8


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters, with the reasoning for each default."""

    #: Hidden width of the shared trunk. The observation is ~1900 floats of mostly
    #: padding, so width matters more than depth here.
    hidden: int = 512
    #: Discount. One agent step is frame_skip ticks (~1/15 s), and an episode runs
    #: tens of thousands of ticks, so this has to be close to 1 to see a wave out.
    gamma: float = 0.997
    #: GAE trace decay — the usual bias/variance dial.
    gae_lambda: float = 0.95
    #: PPO's trust region. 0.2 is the standard starting point.
    clip: float = 0.2
    #: Adam's step size. The first thing to halve if the return climbs and then
    #: falls apart, and to raise if a run is simply not moving.
    learning_rate: float = 3.0e-4
    #: Passes over each batch of rollout data.
    epochs: int = 4
    #: Minibatches per epoch.
    minibatches: int = 8
    #: Weight on the value loss.
    value_coef: float = 0.5
    #: Entropy bonus. Missile Command punishes early commitment, so keep some
    #: exploration alive well into training.
    entropy_coef: float = 0.01
    #: Gradient-norm clip.
    max_grad_norm: float = 0.5


class Policy(nn.Module):
    """Shared trunk, then a policy head and a value head.

    Deliberately small and dense: the observation is already the raw state the
    human sees (see ``docs/API.md``), so there is nothing to convolve over — no
    image, no spatial grid, just a flat list of entities.
    """

    def __init__(self, obs_size: int, action_count: int, hidden: int = 512) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, action_count)
        self.value_head = nn.Linear(hidden, 1)

        # Orthogonal init with a small policy gain: start near-uniform over
        # actions, so early updates are not fighting an arbitrary initial bias.
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked logits and the value estimate."""
        features = self.trunk(obs)
        logits = self.policy_head(features)
        logits = torch.where(mask, logits, torch.full_like(logits, MASKED_LOGIT))
        return logits, self.value_head(features).squeeze(-1)

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """Value estimate only. Needs no action mask — masking touches the policy
        logits, never the critic — which is what lets a truncated episode be
        bootstrapped from its final observation without knowing what was legal there.
        """
        return self.value_head(self.trunk(obs)).squeeze(-1)

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions. Returns (action, log-prob, value)."""
        logits, value = self(obs, mask)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        return action, distribution.log_prob(action), value


class Rollout:
    """A fixed-size buffer of on-policy experience, laid out (steps, envs).

    Preallocated once and overwritten each iteration — the same reasoning as the
    environment's NumPy buffers, so a training run does not allocate per step.
    """

    def __init__(
        self, steps: int, num_envs: int, obs_size: int, action_count: int, device: torch.device
    ) -> None:
        self.steps = steps
        self.num_envs = num_envs
        shape = (steps, num_envs)
        self.obs = torch.zeros((*shape, obs_size), dtype=torch.float32, device=device)
        self.masks = torch.zeros((*shape, action_count), dtype=torch.bool, device=device)
        self.actions = torch.zeros(shape, dtype=torch.int64, device=device)
        self.log_probs = torch.zeros(shape, dtype=torch.float32, device=device)
        self.values = torch.zeros(shape, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(shape, dtype=torch.float32, device=device)
        #: 1 where the episode continued into the next step, 0 where it ended.
        self.continues = torch.zeros(shape, dtype=torch.float32, device=device)

    def advantages(
        self, last_value: torch.Tensor, gamma: float, gae_lambda: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generalised advantage estimation, walked backwards through the buffer.

        `continues` is zero at every episode end, truncation included, because the
        next slot holds a *different* episode and the trace must not run across it.

        Truncation is not the same as death, though, and the difference is handled
        before this point: the collector adds `gamma * V(final_observation)` into
        the reward of a truncated step (see `train.py`). Without that, hitting the
        tick cap is valued as though the world ended — which in this game means
        systematically under-valuing exactly the long survivals you are trying to
        train toward.
        """
        advantages = torch.zeros_like(self.rewards)
        running = torch.zeros(self.num_envs, dtype=torch.float32, device=self.rewards.device)
        next_value = last_value
        for step in reversed(range(self.steps)):
            keep = self.continues[step]
            delta = self.rewards[step] + (gamma * next_value * keep) - self.values[step]
            running = delta + (gamma * gae_lambda * keep * running)
            advantages[step] = running
            next_value = self.values[step]
        return advantages, advantages + self.values


def update(
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: PPOConfig,
) -> dict[str, float]:
    """One PPO update over the collected rollout. Returns losses for logging."""
    # Flatten (steps, envs) into one batch of independent samples.
    obs = rollout.obs.flatten(0, 1)
    masks = rollout.masks.flatten(0, 1)
    actions = rollout.actions.flatten()
    old_log_probs = rollout.log_probs.flatten()
    flat_advantages = advantages.flatten()
    flat_returns = returns.flatten()

    batch = obs.shape[0]
    minibatch = max(1, batch // config.minibatches)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0}
    updates = 0

    for _ in range(config.epochs):
        order = torch.randperm(batch, device=obs.device)
        for start in range(0, batch, minibatch):
            index = order[start : start + minibatch]
            # Normalise per minibatch: the reward scale shifts a lot as the policy
            # improves, and this keeps the step size meaningful throughout.
            sample_advantages = flat_advantages[index]
            sample_advantages = (sample_advantages - sample_advantages.mean()) / (
                sample_advantages.std() + 1e-8
            )

            logits, values = policy(obs[index], masks[index])
            distribution = torch.distributions.Categorical(logits=logits)
            log_probs = distribution.log_prob(actions[index])
            entropy = distribution.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs[index])
            unclipped = ratio * sample_advantages
            clipped = torch.clamp(ratio, 1.0 - config.clip, 1.0 + config.clip) * sample_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, flat_returns[index])

            loss = policy_loss + (config.value_coef * value_loss) - (config.entropy_coef * entropy)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            # Detached: these are for logging, and converting a grad-tracking
            # tensor to a scalar keeps the graph alive (and torch warns about it).
            with torch.no_grad():
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["clip_fraction"] += ((ratio - 1.0).abs() > config.clip).float().mean().item()
            updates += 1

    return {key: value / max(1, updates) for key, value in stats.items()}
