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
    #: "mlp" flattens the observation into one vector; "entity" encodes each
    #: threat with *shared* weights and pools the interceptors and blasts into a
    #: context vector. The flat MLP has to learn "is this threat already covered
    #: by an interceptor in flight?" separately for all 128 threat slots, since
    #: no weight is shared between them — and measurably does not: it fires into
    #: an already-covered zone 72% of the time, worse than random's 56%. Sharing
    #: the encoder lets it learn that comparison once.
    architecture: str = "mlp"


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


@dataclass(frozen=True)
class ObsLayout:
    """Where the entity blocks sit in the flat observation.

    The widths mirror ``md::encode`` (docs/API.md §4): a threat is
    ``[present, x, y, vx, vy, one-hot type x4]``, an interceptor
    ``[present, x, y, vx, vy, detonation x, detonation y]``, a blast
    ``[present, x, y, radius]``. Everything after those blocks — bases, cities,
    wave, score — is the per-episode context and is taken as one vector.
    """

    threats: int
    interceptors: int
    blasts: int
    obs_size: int
    threat_features: int = 9
    interceptor_features: int = 7
    blast_features: int = 4

    @property
    def interceptors_at(self) -> int:
        return self.threats * self.threat_features

    @property
    def blasts_at(self) -> int:
        return self.interceptors_at + (self.interceptors * self.interceptor_features)

    @property
    def globals_at(self) -> int:
        return self.blasts_at + (self.blasts * self.blast_features)

    @property
    def globals_width(self) -> int:
        return self.obs_size - self.globals_at

    def validate(self) -> None:
        """Fail loudly if the observation is not the shape assumed here.

        Slicing silently misreads if a block width ever changes, and a policy fed
        misaligned features would train to a plausible-looking mediocrity rather
        than crash. Cheap to check once at construction.
        """
        if self.globals_width <= 0:
            raise ValueError(
                f"observation is {self.obs_size} floats but the entity blocks alone "
                f"need {self.globals_at}; the layout in ObsLayout no longer matches md::encode"
            )


class EntityPolicy(nn.Module):
    """Per-entity encoders with shared weights, instead of one flat trunk.

    The action space is (battery, threat slot), so the network is built in that
    shape: every threat slot goes through *the same* encoder, and the head emits
    one logit per battery for each. Two things follow that the flat MLP cannot
    get. The comparison "is this threat already covered by something in flight?"
    is learned once and applied to all 128 slots, rather than being learned
    separately per slot from whatever data happened to land there. And the
    interceptors and blasts are pooled permutation-invariantly, so their meaning
    does not depend on which array slot swap-and-pop left them in.

    Both matter here specifically: the flat policy fires into an already-covered
    zone 72% of the time — worse than random — which is the single behaviour
    standing between it and the scripted baseline's 1.10 kills per interceptor.
    """

    def __init__(self, layout: ObsLayout, action_count: int, hidden: int = 512) -> None:
        super().__init__()
        layout.validate()
        self.layout = layout
        self.action_count = action_count
        # One battery-column per (action_count - 1) / threats. Derived rather
        # than assumed, so a Config with a different battery count still lines up.
        batteries, remainder = divmod(action_count - 1, layout.threats)
        if remainder or batteries < 1:
            raise ValueError(
                f"action_count {action_count} is not 1 + batteries x {layout.threats} threats"
            )
        self.batteries = batteries

        # Per-entity work runs once per *slot*, so this width, not `hidden`, is
        # what sets the network's cost: the threat path executes 128 times per
        # sample, and its activations are 128x more numerous than a flat trunk's.
        # That is memory traffic rather than arithmetic — the FLOP counts are
        # within 10% of the flat MLP either way — which is why the fix is a narrow
        # per-slot path and a wide context, not fewer layers. Measured against the
        # flat policy on a 4096-sample update: hidden/2 was 6.3x, this is 3.4x.
        #
        # The remaining 3.4x is nearly all waste: 128 threat slots are encoded and
        # typically 2 are occupied. Gathering only the live slots would remove it,
        # at the cost of real complexity here — worth doing once this architecture
        # has earned it, not before.
        width = max(32, hidden // 16)
        self.width = width

        def encoder(in_features: int, out_features: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.Tanh(),
                nn.Linear(out_features, out_features),
                nn.Tanh(),
            )

        self.interceptor_encoder = encoder(layout.interceptor_features, width)
        self.blast_encoder = encoder(layout.blast_features, width)
        self.context = encoder(layout.globals_width + (2 * width), hidden)
        # Conditioning is additive rather than concatenated: concatenating the
        # context onto every threat would widen the per-slot input by `hidden` and
        # pay for it 128 times over, where projecting it once and adding costs one
        # matmul per sample and is the same function class.
        self.threat_in = nn.Linear(layout.threat_features, width)
        self.context_to_threat = nn.Linear(hidden, width)
        self.threat_out = nn.Linear(width, width)
        # One logit per battery for each threat slot, plus the always-legal NoOp.
        self.fire_head = nn.Linear(width, batteries)
        self.noop_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden + width, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        for head, gain in ((self.fire_head, 0.01), (self.noop_head, 0.01), (self.value_head, 1.0)):
            nn.init.orthogonal_(head.weight, gain=gain)
            nn.init.zeros_(head.bias)

    def _features(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (per-threat features, pooled context)."""
        layout = self.layout
        batch = obs.shape[0]
        threats = obs[:, : layout.interceptors_at].view(batch, layout.threats, -1)
        interceptors = obs[:, layout.interceptors_at : layout.blasts_at].view(
            batch, layout.interceptors, -1
        )
        blasts = obs[:, layout.blasts_at : layout.globals_at].view(batch, layout.blasts, -1)
        context = obs[:, layout.globals_at :]

        # Feature 0 is the presence flag; empty slots read zero by construction,
        # so masking before the sum keeps padding out of the pooled vector.
        pooled_i = (self.interceptor_encoder(interceptors) * interceptors[..., :1]).sum(dim=1)
        pooled_b = (self.blast_encoder(blasts) * blasts[..., :1]).sum(dim=1)
        summary = self.context(torch.cat([context, pooled_i, pooled_b], dim=-1))

        # Broadcast-add the projected context into every slot: one matmul for the
        # context, then only slot-width work per threat.
        hidden_threats = torch.tanh(
            self.threat_in(threats) + self.context_to_threat(summary).unsqueeze(1)
        )
        per_threat = torch.tanh(self.threat_out(hidden_threats))
        per_threat = per_threat * threats[..., :1]
        return per_threat, summary

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked logits and the value estimate."""
        per_threat, summary = self._features(obs)
        # (batch, threats, batteries) -> battery-major, matching decode_action's
        # index = 1 + battery * threats + slot.
        fire = self.fire_head(per_threat).permute(0, 2, 1).reshape(obs.shape[0], -1)
        logits = torch.cat([self.noop_head(summary), fire], dim=-1)
        logits = torch.where(mask, logits, torch.full_like(logits, MASKED_LOGIT))
        return logits, self._value(per_threat, summary)

    def _value(self, per_threat: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        return self.value_head(torch.cat([summary, per_threat.sum(dim=1)], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """Value estimate only — no mask, same contract as `Policy.value`."""
        per_threat, summary = self._features(obs)
        return self._value(per_threat, summary)

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions. Returns (action, log-prob, value)."""
        logits, value = self(obs, mask)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        return action, distribution.log_prob(action), value


def build_policy(architecture: str, layout: ObsLayout, action_count: int, hidden: int) -> nn.Module:
    """Construct the policy named by `PPOConfig.architecture`."""
    if architecture == "mlp":
        return Policy(layout.obs_size, action_count, hidden)
    if architecture == "entity":
        return EntityPolicy(layout, action_count, hidden)
    raise ValueError(f"unknown architecture {architecture!r}; expected 'mlp' or 'entity'")


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
