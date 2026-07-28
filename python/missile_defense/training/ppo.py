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

from .auxiliary import targets as auxiliary_targets

# Actions that the mask forbids get this logit, so softmax gives them ~0
# probability. Not -inf: that produces NaNs if an env ever masks *everything*.
MASKED_LOGIT = -1.0e8


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters, with the reasoning for each default."""

    #: Hidden width of the MLP trunk, or of the relational policy's episode
    #: context and independent critic. Entity-slot encoders stay narrow.
    hidden: int = 512
    #: Discount, and it has to match `Shaping.gamma`. One agent step is frame_skip
    #: ticks (~1/15 s), so 1/(1-gamma) reads directly as an horizon: 0.997 is 333
    #: steps ~ 22 s, barely one wave, and 0.999 is 1000 steps ~ 66 s.
    #:
    #: 0.997 was chosen to span a wave, which was right when every wave scored the
    #: same. It is not right now: the arcade multiplier makes wave 11 worth six
    #: times wave 1, so what a shot is *worth* depends on surviving several waves
    #: further, and an agent that cannot see that far cannot be asked to value it.
    gamma: float = 0.999
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
    #: Clips how far the critic may move per update, PPO-style: the value loss is
    #: the larger of the plain and the clipped error, so a big target change (the
    #: return grows several-fold as the policy improves) cannot yank the value head
    #: — and, through the shared trunk, the *policy* features — in one step. This is
    #: the mechanism behind a run that peaks then regresses. 0.2 matches `clip`,
    #: and is well-sized here because the reward `scale` keeps returns O(1).
    value_clip: float = 0.2
    #: Entropy bonus. Missile Command punishes early commitment, so keep some
    #: exploration alive well into training. Raised from 0.01: a run was collapsing
    #: to entropy ~0.4 (committing to one tactic) and regressing after its peak;
    #: more pressure to keep exploring is the first defence against that.
    entropy_coef: float = 0.02
    #: Weight on three training-only relational predictions: time-to-impact,
    #: existing coverage, and local threat density. The targets are derived only
    #: from the raw observation, never added to it, and are available on the
    #: relational entity architecture. This supplies a dense signal before sparse
    #: score and survival rewards can distinguish good tactical representations.
    auxiliary_coef: float = 0.1
    #: Gradient-norm clip.
    max_grad_norm: float = 0.5
    #: "mlp" is the checkpoint-compatible flat network; "entity" is the new
    #: relational actor with shared threat encoding, per-threat cross-attention
    #: over interceptors/blasts, and a separate critic. The flat MLP must learn
    #: "is this threat already covered?" separately for all 128 threat slots and
    #: measurably does not; the relational path learns that comparison once.
    architecture: str = "mlp"


class Policy(nn.Module):
    """Shared trunk, then a policy head and a value head.

    Deliberately small and dense: the observation is already the raw state the
    human sees (see ``docs/API.md``), so there is nothing to convolve over — no
    image, no spatial grid, just a flat list of entities. Kept unchanged so
    existing ``architecture="mlp"`` checkpoints remain loadable; the relational
    architecture below is the new actor/critic-separated training path.
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
    ``[present, x, y, radius, lifetime phase]``. Everything after those blocks —
    bases, cities, wave, score — is the per-episode context and is taken as one
    vector.
    """

    threats: int
    interceptors: int
    blasts: int
    obs_size: int
    threat_features: int = 9
    interceptor_features: int = 7
    blast_features: int = 5

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


class _CrossAttention(nn.Module):
    """One-head cross-attention from every threat to another entity set.

    PyTorch's fused scaled-dot-product implementation avoids materialising the
    large ``batch x threats x entities`` score tensor on supported GPUs. Presence
    masks exclude padding; an entirely empty entity set produces an exact zero.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width)

    def forward(
        self,
        threats: torch.Tensor,
        entities: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query(threats).unsqueeze(1)
        key = self.key(entities).unsqueeze(1)
        value = self.value(entities).unsqueeze(1)
        # True entries participate. The singleton head/query dimensions make the
        # entity mask broadcast to every threat without allocating a copy.
        mask = present[:, None, None, :]
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        has_entity = present.any(dim=-1, keepdim=True).unsqueeze(-1)
        return self.output(attended.squeeze(1)) * has_entity


class EntityPolicy(nn.Module):
    """Relational actor with an entirely independent critic.

    The action space is (battery, threat slot), so every threat passes through the
    same encoder and emits one logit per battery. Each encoded threat separately
    attends to live interceptors and blasts. Thus "is this threat already
    covered?" is a direct relation, learned once for all slots, rather than a fact
    compressed into one global pool or relearned per slot by a flat MLP.

    The critic is a separate flat MLP. It shares no actor-side encoder, attention
    projection, context, or relation feature, so value-loss gradients cannot
    corrupt the action representation.
    """

    def __init__(self, layout: ObsLayout, action_count: int, hidden: int = 512) -> None:
        super().__init__()
        layout.validate()
        self.layout = layout
        self.action_count = action_count
        batteries, remainder = divmod(action_count - 1, layout.threats)
        if remainder or batteries < 1:
            raise ValueError(
                f"action_count {action_count} is not 1 + batteries x {layout.threats} threats"
            )
        self.batteries = batteries

        # This path runs once per entity slot, so keep it narrow. Attention then
        # relates every candidate threat to the live interceptor/blast sets.
        width = max(32, hidden // 16)
        self.width = width
        self.threat_encoder = self._encoder(layout.threat_features, width)
        self.interceptor_encoder = self._encoder(layout.interceptor_features, width)
        self.blast_encoder = self._encoder(layout.blast_features, width)
        self.interceptor_attention = _CrossAttention(width)
        self.blast_attention = _CrossAttention(width)

        # The pool carries episode-level context (bases, cities, wave, score) and
        # supports NoOp. Pairwise facts stay in the attention outputs.
        self.actor_context = self._encoder(layout.globals_width + (2 * width), hidden)
        self.context_to_threat = nn.Linear(hidden, width)
        self.relation = self._encoder(4 * width, width)

        self.fire_head = nn.Linear(width, batteries)
        self.noop_head = nn.Linear(hidden, 1)
        self.auxiliary_head = nn.Linear(width, 3)

        # No actor feature enters this path.
        self.critic_trunk = self._encoder(layout.obs_size, hidden)
        self.value_head = nn.Linear(hidden, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for head, gain in (
            (self.fire_head, 0.01),
            (self.noop_head, 0.01),
            (self.auxiliary_head, 0.01),
            (self.value_head, 1.0),
        ):
            nn.init.orthogonal_(head.weight, gain=gain)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _encoder(in_features: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh(),
            nn.Linear(out_features, out_features),
            nn.Tanh(),
        )

    @property
    def context(self) -> nn.Sequential:
        """Backward-compatible name for the actor context encoder."""
        return self.actor_context

    @staticmethod
    def _masked_mean(encoded: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """Permutation-invariant mean whose empty-set value is exactly zero."""
        weights = present.unsqueeze(-1).to(encoded.dtype)
        count = weights.sum(dim=1).clamp_min(1.0)
        return (encoded * weights).sum(dim=1) / count

    def _split(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        layout = self.layout
        batch = obs.shape[0]
        threats = obs[:, : layout.interceptors_at].view(batch, layout.threats, -1)
        interceptors = obs[:, layout.interceptors_at : layout.blasts_at].view(
            batch, layout.interceptors, -1
        )
        blasts = obs[:, layout.blasts_at : layout.globals_at].view(batch, layout.blasts, -1)
        context = obs[:, layout.globals_at :]
        return threats, interceptors, blasts, context

    def _actor_features(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return relational per-threat features and the actor-only summary."""
        threats, interceptors, blasts, context = self._split(obs)
        threat_present = threats[..., 0].bool()
        interceptor_present = interceptors[..., 0].bool()
        blast_present = blasts[..., 0].bool()

        encoded_t = self.threat_encoder(threats)
        encoded_i = self.interceptor_encoder(interceptors)
        encoded_b = self.blast_encoder(blasts)
        pooled_i = self._masked_mean(encoded_i, interceptor_present)
        pooled_b = self._masked_mean(encoded_b, blast_present)
        summary = self.actor_context(torch.cat([context, pooled_i, pooled_b], dim=-1))

        related_i = self.interceptor_attention(encoded_t, encoded_i, interceptor_present)
        related_b = self.blast_attention(encoded_t, encoded_b, blast_present)
        episode = self.context_to_threat(summary).unsqueeze(1).expand(-1, self.layout.threats, -1)
        per_threat = self.relation(torch.cat([encoded_t, related_i, related_b, episode], dim=-1))
        per_threat = per_threat * threat_present.unsqueeze(-1)
        return per_threat, summary

    # Kept as a narrow compatibility shim for tests and analysis notebooks that
    # inspected the old entity policy's shared feature function.
    def _features(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._actor_features(obs)

    def _logits(
        self,
        per_threat: torch.Tensor,
        summary: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # (batch, threats, batteries) -> battery-major, matching decode_action's
        # index = 1 + battery * threats + slot.
        fire = self.fire_head(per_threat).permute(0, 2, 1).flatten(1)
        logits = torch.cat([self.noop_head(summary), fire], dim=-1)
        return torch.where(mask, logits, torch.full_like(logits, MASKED_LOGIT))

    def _auxiliary(self, obs: torch.Tensor, per_threat: torch.Tensor) -> torch.Tensor:
        present = obs[:, : self.layout.interceptors_at].view(
            obs.shape[0], self.layout.threats, self.layout.threat_features
        )[..., :1]
        return self.auxiliary_head(per_threat) * present

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked logits and the independent value estimate."""
        per_threat, summary = self._actor_features(obs)
        return self._logits(per_threat, summary, mask), self._value(obs)

    def forward_with_auxiliary(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits, value, and auxiliary predictions from one actor pass."""
        per_threat, summary = self._actor_features(obs)
        return (
            self._logits(per_threat, summary, mask),
            self._value(obs),
            self._auxiliary(obs, per_threat),
        )

    def auxiliary_predictions(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict per-threat [time-to-impact, coverage, cluster-density].

        The head deliberately reuses the actor's relational features, so these
        observable auxiliary tasks teach the representation used to choose fire
        actions. Padding rows are zero because `_actor_features` masks them.
        """
        per_threat, _ = self._actor_features(obs)
        return self._auxiliary(obs, per_threat)

    def _value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.critic_trunk(obs)).squeeze(-1)

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """Value estimate only; skips all actor encoders and attention."""
        return self._value(obs)

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
    policy: nn.Module,
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
    old_values = rollout.values.flatten()
    flat_advantages = advantages.flatten()
    flat_returns = returns.flatten()

    batch = obs.shape[0]
    minibatch = max(1, batch // config.minibatches)
    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "auxiliary_loss": 0.0,
        "entropy": 0.0,
        "clip_fraction": 0.0,
    }
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

            sample_obs = obs[index]
            if isinstance(policy, EntityPolicy) and config.auxiliary_coef > 0.0:
                # Targets are immutable functions of the observation. Build them
                # before the actor's gradient graph: both operations contain
                # entity-pair tensors, and keeping the forward activations alive
                # while deriving labels needlessly adds their peaks together.
                # This ordering is what lets the documented 1024 x 256 batch fit
                # on a 32 GiB training GPU.
                target = auxiliary_targets(sample_obs, policy.layout)
                logits, values, predictions = policy.forward_with_auxiliary(
                    sample_obs, masks[index]
                )
                present = target.present.unsqueeze(-1)
                auxiliary_loss = ((predictions - target.stacked()).square() * present).sum() / (
                    present.sum().clamp_min(1.0) * predictions.shape[-1]
                )
            else:
                logits, values = policy(sample_obs, masks[index])
                auxiliary_loss = torch.zeros((), device=obs.device)
            distribution = torch.distributions.Categorical(logits=logits)
            log_probs = distribution.log_prob(actions[index])
            entropy = distribution.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs[index])
            unclipped = ratio * sample_advantages
            clipped = torch.clamp(ratio, 1.0 - config.clip, 1.0 + config.clip) * sample_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            # Clipped value loss (the larger of plain and clipped error), so the
            # critic cannot take a huge step when the target has grown — which,
            # sharing a trunk with the policy, would otherwise corrupt the policy
            # features too. Same epsilon as the policy clip; returns are O(1).
            sample_returns = flat_returns[index]
            sample_old_values = old_values[index]
            values_clipped = sample_old_values + torch.clamp(
                values - sample_old_values, -config.value_clip, config.value_clip
            )
            value_loss = torch.max(
                (values - sample_returns).square(), (values_clipped - sample_returns).square()
            ).mean()

            loss = (
                policy_loss
                + (config.value_coef * value_loss)
                + (config.auxiliary_coef * auxiliary_loss)
                - (config.entropy_coef * entropy)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            # Detached: these are for logging, and converting a grad-tracking
            # tensor to a scalar keeps the graph alive (and torch warns about it).
            with torch.no_grad():
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["auxiliary_loss"] += auxiliary_loss.item()
                stats["entropy"] += entropy.item()
                stats["clip_fraction"] += ((ratio - 1.0).abs() > config.clip).float().mean().item()
            updates += 1

    return {key: value / max(1, updates) for key, value in stats.items()}
