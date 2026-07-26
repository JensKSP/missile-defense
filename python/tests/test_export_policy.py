# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Turning a checkpoint into a `.mdp`, and pinning what it must then compute.

Two claims live here and they are different sizes.

The small one is the conversion: a `.pt` written by `md.train` becomes an `.mdp`
with the same weights, and the ways it can go wrong — a missing tensor, an
architecture nothing implements, a checkpoint that is not one — are refused with
something a person can act on.

The large one is **parity**, and it is why this file writes a JSON fixture that
another language reads. `agent/tests/unit/test_policy.cpp` runs the same
observations through the C++ forward pass and asserts the same logits, the same
value and the same chosen action. Asserting parity inside one process proves the
arithmetic; asserting it across two proves the *file*, which is the thing being
shipped. `test_journey.py` then closes the loop with two whole episodes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from md import export_policy, policy_format

torch = pytest.importorskip("torch", reason="torch is not installed")

OBS = 12
ACTIONS = 7
HIDDEN = 16


def _checkpoint(tmp_path: Path, **overrides: object) -> Path:
    """A checkpoint in exactly the shape `md.train.save_checkpoint` writes one."""
    from md.ppo import Policy  # noqa: PLC0415 — optional dependency

    torch.manual_seed(20260726)
    policy = Policy(OBS, ACTIONS, HIDDEN)
    payload: dict[str, object] = {
        "policy": policy.state_dict(),
        "optimizer": {},
        "iteration": 800,
        "obs_size": OBS,
        "action_count": ACTIONS,
        "hidden": HIDDEN,
        "architecture": "mlp",
    }
    payload.update(overrides)
    path = tmp_path / "policy-00800.pt"
    torch.save(payload, path)
    return path


# ---- the conversion ----------------------------------------------------------


def test_export_produces_a_readable_policy(tmp_path: Path) -> None:
    exported = export_policy.export_checkpoint(
        _checkpoint(tmp_path), tmp_path / "policy.mdp", metadata={"display_name": "Test"}
    )
    policy = policy_format.read(exported)
    assert policy.architecture == "mlp"
    assert policy.observation_size == OBS
    assert policy.action_count == ACTIONS
    assert policy.metadata["display_name"] == "Test"


def test_the_exported_weights_are_the_checkpoint_s_weights(tmp_path: Path) -> None:
    """Bit for bit, and by name — not merely the right shapes.

    A transposed weight matrix has the same shape as the one it should be when
    the layer is square, and `hidden` is square in every trunk this project
    trains. So the check is on values, against the tensors torch itself holds.
    """
    path = _checkpoint(tmp_path)
    stored = torch.load(path, map_location="cpu", weights_only=True)["policy"]
    exported = export_policy.export_checkpoint(path, tmp_path / "policy.mdp", metadata={})
    policy = policy_format.read(exported)
    for name, tensor in stored.items():
        assert np.array_equal(policy.tensor(name).values, tensor.numpy().astype(np.float32))


def test_export_records_where_the_policy_came_from(tmp_path: Path) -> None:
    """Provenance in the file, because a promoted model outlives its run directory."""
    exported = export_policy.export_checkpoint(
        _checkpoint(tmp_path), tmp_path / "policy.mdp", metadata={}
    )
    metadata = policy_format.read(exported).metadata
    assert metadata["trained_updates"] == 800
    assert metadata["source"].endswith("policy-00800.pt")  # type: ignore[union-attr]


def test_export_refuses_an_architecture_nothing_can_run(tmp_path: Path) -> None:
    """`entity` trains but has no native forward pass, so it cannot be exported.

    Failing here is the honest outcome: the alternative is an `.mdp` the game
    accepts and then cannot evaluate, which surfaces as a policy that plays
    like noise rather than as an error anyone can act on.
    """
    path = _checkpoint(tmp_path, architecture="entity")
    with pytest.raises(export_policy.ExportError, match="entity"):
        export_policy.export_checkpoint(path, tmp_path / "policy.mdp", metadata={})


def test_export_refuses_a_checkpoint_missing_a_tensor(tmp_path: Path) -> None:
    from md.ppo import Policy  # noqa: PLC0415 — optional dependency

    state = Policy(OBS, ACTIONS, HIDDEN).state_dict()
    del state["value_head.bias"]
    path = _checkpoint(tmp_path, policy=state)
    with pytest.raises(export_policy.ExportError, match="value_head.bias"):
        export_policy.export_checkpoint(path, tmp_path / "policy.mdp", metadata={})


def test_export_refuses_a_file_that_is_not_a_checkpoint(tmp_path: Path) -> None:
    junk = tmp_path / "notes.txt"
    junk.write_text("this is not a checkpoint", encoding="utf-8")
    with pytest.raises(export_policy.ExportError):
        export_policy.export_checkpoint(junk, tmp_path / "policy.mdp", metadata={})


def test_export_leaves_no_file_behind_when_it_refuses(tmp_path: Path) -> None:
    destination = tmp_path / "policy.mdp"
    with pytest.raises(export_policy.ExportError):
        export_policy.export_checkpoint(
            _checkpoint(tmp_path, architecture="entity"), destination, metadata={}
        )
    assert not destination.exists()


# ---- parity, in this process -------------------------------------------------


def test_the_numpy_forward_pass_matches_torch(tmp_path: Path) -> None:
    """The reference implementation the C++ one is checked against.

    `md.export_policy.evaluate` exists so the parity fixture can be produced
    from an `.mdp` alone, with no torch anywhere — which is what makes it usable
    as the *definition* of what the file computes rather than as a recording of
    what one PyTorch version happened to do. It has to agree with torch first.
    """
    from md.ppo import MASKED_LOGIT  # noqa: PLC0415 — optional dependency

    path = _checkpoint(tmp_path)
    net, _ = _load(path)
    policy = policy_format.read(
        export_policy.export_checkpoint(path, tmp_path / "policy.mdp", metadata={})
    )

    rng = np.random.default_rng(7)
    for _ in range(16):
        obs = rng.standard_normal(OBS).astype(np.float32)
        legal = rng.random(ACTIONS) > 0.35
        legal[0] = True  # the no-op is always available; a policy with no legal
        # action is a bug in the environment, not a case for this function
        with torch.no_grad():
            expected_logits, expected_value = net(
                torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(legal).unsqueeze(0)
            )
        found = export_policy.evaluate(policy, obs, legal)
        # Only the legal entries: torch writes MASKED_LOGIT into the rest and so
        # does this, but a test that compared them would be asserting the
        # sentinel rather than the network.
        assert np.allclose(found.logits[legal], expected_logits[0].numpy()[legal], atol=1e-5)
        assert np.allclose(found.value, expected_value.item(), atol=1e-5)
        assert found.logits[~legal].max(initial=MASKED_LOGIT) <= MASKED_LOGIT
        assert legal[found.action]


def test_the_chosen_action_breaks_ties_towards_the_lowest_index() -> None:
    """Determinism has to survive two equal logits, or parity is a coin flip.

    Both languages take the *first* maximum. Left unstated it would be whatever
    `argmax` and `std::max_element` happen to agree on, which is true today and
    is not a thing to rely on across two standard libraries.
    """
    policy = _flat_policy()
    legal = np.ones(ACTIONS, dtype=bool)
    found = export_policy.evaluate(policy, np.zeros(OBS, dtype=np.float32), legal)
    assert found.action == 0
    legal[0] = False
    assert export_policy.evaluate(policy, np.zeros(OBS, dtype=np.float32), legal).action == 1


def test_an_illegal_action_is_never_chosen_however_good_its_logit() -> None:
    """The masking rule, at its sharpest: the best action is the illegal one."""
    policy = _flat_policy(best=3)
    legal = np.ones(ACTIONS, dtype=bool)
    assert export_policy.evaluate(policy, np.zeros(OBS, dtype=np.float32), legal).action == 3
    legal[3] = False
    assert export_policy.evaluate(policy, np.zeros(OBS, dtype=np.float32), legal).action != 3


# ---- the fixture the C++ tests read -----------------------------------------


def test_the_parity_fixture_pins_every_input_and_output(tmp_path: Path) -> None:
    """What `agent/tests/unit/test_policy.cpp` asserts against.

    Written rather than generated on both sides: a fixture computed twice from
    the same source is two implementations of the same bug, and the point is to
    catch the C++ forward pass disagreeing with the Python one.
    """
    exported = export_policy.export_checkpoint(
        _checkpoint(tmp_path), tmp_path / "policy.mdp", metadata={}
    )
    fixture = export_policy.write_parity_fixture(exported, tmp_path / "parity.json", samples=8)
    payload = json.loads(fixture.read_text(encoding="utf-8"))

    assert payload["policy"] == "policy.mdp"
    assert payload["observation_size"] == OBS
    assert payload["action_count"] == ACTIONS
    assert len(payload["samples"]) == 8
    for sample in payload["samples"]:
        assert len(sample["observation"]) == OBS
        assert len(sample["legal"]) == ACTIONS
        assert len(sample["logits"]) == ACTIONS
        assert isinstance(sample["value"], float)
        assert sample["legal"][sample["action"]] == 1


def test_the_fixture_reproduces_from_the_policy_alone(tmp_path: Path) -> None:
    """No torch, no checkpoint — just the `.mdp`, which is what ships.

    If this ever needed the `.pt` back, the fixture would be pinning the
    checkpoint rather than the file the game loads, and the parity claim would
    have a hole in exactly the place it is meant to cover.
    """
    exported = export_policy.export_checkpoint(
        _checkpoint(tmp_path), tmp_path / "policy.mdp", metadata={}
    )
    first = export_policy.write_parity_fixture(exported, tmp_path / "a.json", samples=4)
    second = export_policy.write_parity_fixture(exported, tmp_path / "b.json", samples=4)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def _flat_policy(best: int | None = None) -> policy_format.NativePolicy:
    """Zero weights, so the logits are exactly the policy head's bias.

    Which makes "which action would be chosen" a fact stated in the fixture
    rather than one that emerges from a forward pass — the right shape for a
    test about tie-breaking and masking.
    """
    bias = np.zeros(ACTIONS, dtype=np.float32)
    if best is not None:
        bias[best] = 1.0
    zeros = [
        policy_format.Tensor("trunk.0.weight", (HIDDEN, OBS), np.zeros((HIDDEN, OBS), np.float32)),
        policy_format.Tensor("trunk.0.bias", (HIDDEN,), np.zeros(HIDDEN, np.float32)),
        policy_format.Tensor(
            "trunk.2.weight", (HIDDEN, HIDDEN), np.zeros((HIDDEN, HIDDEN), np.float32)
        ),
        policy_format.Tensor("trunk.2.bias", (HIDDEN,), np.zeros(HIDDEN, np.float32)),
        policy_format.Tensor(
            "policy_head.weight", (ACTIONS, HIDDEN), np.zeros((ACTIONS, HIDDEN), np.float32)
        ),
        policy_format.Tensor("policy_head.bias", (ACTIONS,), bias),
        policy_format.Tensor("value_head.weight", (1, HIDDEN), np.zeros((1, HIDDEN), np.float32)),
        policy_format.Tensor("value_head.bias", (1,), np.zeros(1, np.float32)),
    ]
    return policy_format.NativePolicy(
        schema=policy_format.SCHEMA,
        observation_size=OBS,
        action_count=ACTIONS,
        architecture="mlp",
        tensors=tuple(zeros),
        metadata={},
    )


def _load(path: Path):  # noqa: ANN202 — torch is optional
    from md.ppo import Policy  # noqa: PLC0415 — optional dependency

    payload = torch.load(path, map_location="cpu", weights_only=True)
    net = Policy(payload["obs_size"], payload["action_count"], payload["hidden"])
    net.load_state_dict(payload["policy"])
    net.eval()
    return net, payload
