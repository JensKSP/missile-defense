# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: OpenAI Codex
"""Training-only tactical targets derived from the raw observation.

These quantities are deliberately *not* appended to the observation. A deployed
policy still receives only what a human sees; the auxiliary loss merely asks its
shared entity representation to recover useful relationships from that state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class Layout(Protocol):
    """The observation offsets needed here, without importing the policy module."""

    threats: int
    interceptors: int
    blasts: int
    interceptors_at: int
    blasts_at: int
    globals_at: int
    threat_features: int
    interceptor_features: int
    blast_features: int


@dataclass(frozen=True)
class AuxiliaryTargets:
    """One target per threat slot; absent slots are masked out by ``present``."""

    present: torch.Tensor
    time_to_impact: torch.Tensor
    coverage: torch.Tensor
    cluster_density: torch.Tensor

    def stacked(self) -> torch.Tensor:
        """Targets in the order expected by a three-output auxiliary head."""
        return torch.stack(
            (self.time_to_impact, self.coverage, self.cluster_density),
            dim=-1,
        )


def targets(obs: torch.Tensor, layout: Layout) -> AuxiliaryTargets:
    """Derive bounded tactical targets from a batch of encoded observations.

    ``coverage`` is a smooth proximity score to an interceptor's chosen
    detonation point or an active blast. It is intentionally not the scripted
    agent's binary judgement: predicting a continuous relationship supplies a
    denser learning signal and does not bake the teacher into the policy.
    """
    batch = obs.shape[0]
    threats = obs[:, : layout.interceptors_at].view(batch, layout.threats, layout.threat_features)
    interceptors = obs[:, layout.interceptors_at : layout.blasts_at].view(
        batch, layout.interceptors, layout.interceptor_features
    )
    blasts = obs[:, layout.blasts_at : layout.globals_at].view(
        batch, layout.blasts, layout.blast_features
    )

    present = threats[..., 0]
    threat_xy = threats[..., 1:3]

    # Height and vertical speed are already normalized. Their ratio is therefore
    # only a monotonic proxy for seconds, which is all an auxiliary regression
    # needs; clamp it so one nearly-horizontal threat cannot dominate the loss.
    height = (threats[..., 2] + 1.0) * 0.5
    falling_speed = (-threats[..., 4]).clamp_min(1.0e-4)
    time_to_impact = (height / falling_speed / 10.0).clamp(0.0, 1.0) * present

    coverage_parts: list[torch.Tensor] = []
    if layout.interceptors > 0:
        interceptor_target = interceptors[..., 5:7]
        delta_i = threat_xy.unsqueeze(2) - interceptor_target.unsqueeze(1)
        distance_i = delta_i.square().sum(dim=-1)
        live_i = interceptors[..., 0].unsqueeze(1)
        coverage_parts.append((torch.exp(-12.0 * distance_i) * live_i).amax(dim=2))
    if layout.blasts > 0:
        blast_xy = blasts[..., 1:3]
        delta_b = threat_xy.unsqueeze(2) - blast_xy.unsqueeze(1)
        distance_b = delta_b.square().sum(dim=-1)
        live_b = blasts[..., 0].unsqueeze(1)
        # A larger visible radius is stronger evidence that a nearby threat is
        # covered. Keep a small floor because a newly spawned blast is already a
        # committed shot even while its rendered radius begins at zero.
        radius = (0.25 + (0.75 * blasts[..., 3])).unsqueeze(1)
        coverage_parts.append((torch.exp(-12.0 * distance_b) * live_b * radius).amax(dim=2))
    coverage = (
        torch.stack(coverage_parts, dim=0).amax(dim=0)
        if coverage_parts
        else torch.zeros_like(present)
    )
    coverage = coverage.clamp(0.0, 1.0) * present

    # A smooth neighbourhood count teaches the representation that two nearby
    # threats may be worth one blast. Remove the self-match, mask padding, and
    # scale into [0, 1] for a loss comparable with the other two heads.
    relative = threat_xy.unsqueeze(2) - threat_xy.unsqueeze(1)
    proximity = torch.exp(-20.0 * relative.square().sum(dim=-1))
    live_pairs = present.unsqueeze(2) * present.unsqueeze(1)
    neighbours = ((proximity * live_pairs).sum(dim=2) - present).clamp_min(0.0)
    cluster_density = (neighbours / 4.0).clamp(0.0, 1.0) * present

    return AuxiliaryTargets(
        present=present,
        time_to_impact=time_to_impact,
        coverage=coverage,
        cluster_density=cluster_density,
    )
