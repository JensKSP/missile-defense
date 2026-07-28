# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: OpenAI Codex
import pytest

# torch is optional — the trainer and the environment do not need it, and the
# gate runs where it is absent. Guarded the way `test_ppo.py` and
# `test_train_eval.py` already guard: a bare `import torch` here took the whole
# collection down, so *every* Python test failed in CI over one missing optional
# dependency, and the error named this file rather than the policy behind it.
torch = pytest.importorskip("torch", reason="torch is optional; see docs/TRAINING.md")

from missile_defense.training.auxiliary import targets  # noqa: E402 — after the skip, by design
from missile_defense.training.ppo import ObsLayout  # noqa: E402


def _layout() -> ObsLayout:
    return ObsLayout(threats=3, interceptors=2, blasts=2, obs_size=(3 * 9) + (2 * 7) + (2 * 5) + 8)


def _obs(layout: ObsLayout) -> torch.Tensor:
    return torch.zeros(1, layout.obs_size)


def test_absent_threat_slots_have_no_auxiliary_target() -> None:
    layout = _layout()
    result = targets(_obs(layout), layout)
    assert result.stacked().shape == (1, layout.threats, 3)
    assert torch.count_nonzero(result.stacked()) == 0


def test_lower_falling_threat_has_less_time_to_impact() -> None:
    layout = _layout()
    obs = _obs(layout)
    threats = obs[:, : layout.interceptors_at].view(1, layout.threats, -1)
    threats[0, 0, :5] = torch.tensor([1.0, 0.0, 0.8, 0.0, -0.1])
    threats[0, 1, :5] = torch.tensor([1.0, 0.0, -0.5, 0.0, -0.1])
    result = targets(obs, layout)
    assert result.time_to_impact[0, 1] < result.time_to_impact[0, 0]


def test_interceptor_target_and_blast_raise_coverage_proximity() -> None:
    layout = _layout()
    obs = _obs(layout)
    threats = obs[:, : layout.interceptors_at].view(1, layout.threats, -1)
    threats[0, 0, :5] = torch.tensor([1.0, 0.0, 0.0, 0.0, -0.1])
    threats[0, 1, :5] = torch.tensor([1.0, 0.9, 0.9, 0.0, -0.1])
    interceptors = obs[:, layout.interceptors_at : layout.blasts_at].view(
        1, layout.interceptors, -1
    )
    interceptors[0, 0, [0, 5, 6]] = torch.tensor([1.0, 0.0, 0.0])
    blasts = obs[:, layout.blasts_at : layout.globals_at].view(1, layout.blasts, -1)
    blasts[0, 0, [0, 1, 2, 3]] = torch.tensor([1.0, 0.05, 0.05, 1.0])
    result = targets(obs, layout)
    assert result.coverage[0, 0] > 0.9
    assert result.coverage[0, 1] < 0.1


def test_nearby_threats_have_cluster_signal() -> None:
    layout = _layout()
    obs = _obs(layout)
    threats = obs[:, : layout.interceptors_at].view(1, layout.threats, -1)
    threats[0, 0, :5] = torch.tensor([1.0, 0.0, 0.0, 0.0, -0.1])
    threats[0, 1, :5] = torch.tensor([1.0, 0.02, 0.02, 0.0, -0.1])
    threats[0, 2, :5] = torch.tensor([1.0, 0.9, 0.9, 0.0, -0.1])
    result = targets(obs, layout)
    assert result.cluster_density[0, 0] > result.cluster_density[0, 2]
