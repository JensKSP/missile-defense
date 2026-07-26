# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Checkpoint → `.mdp`, and the reference forward pass both languages must match.

Three things live here, and the third is the reason for the other two.

:func:`export_checkpoint` converts what training wrote into what the game loads.
It is the *only* place a `.pt` is opened outside the trainer, and it opens one
the same way the trainer does — ``weights_only=True``, which is torch's own
refusal to unpickle anything but tensors. Everything after that point in this
project reads `.mdp` and never a pickle again.

:func:`evaluate` is a NumPy implementation of the forward pass, deliberately
written against the `.mdp` rather than against torch. That inversion matters: it
means the fixture below can be produced from the shipped file alone, on a
machine with no torch, so it is the *definition* of what an `.mdp` computes
rather than a recording of what one PyTorch version happened to do on one day.

:func:`write_parity_fixture` writes that definition out as JSON —
observations, legal masks, logits, values, chosen actions — for
``agent/tests/unit/test_policy.cpp`` to assert against. Parity checked inside one
process proves the arithmetic. Parity checked across two, against a file, proves
the thing that actually ships.

## The masking rule, stated once

Both implementations do exactly this and it is written down because leaving it
implicit is how the two drift:

1. compute the logits from the unmasked network;
2. overwrite every illegal action's logit with :data:`MASKED_LOGIT`;
3. take the **first** maximum — ties go to the lowest index.

Step 3 is not incidental. `np.argmax` and `std::max_element` agree on it today,
and "the two standard libraries happen to agree" is not a property to build a
determinism claim on.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from . import policy_format
from .policy_format import NativePolicy, PolicyFormatError, Tensor

#: What an illegal action's logit becomes. The same value `md.ppo` uses during
#: training, so the exported policy behaves as the trained one did — a different
#: sentinel would be a different (if usually equivalent) policy.
MASKED_LOGIT = -1.0e8

#: Architectures with a native forward pass. Training can produce others; the
#: game cannot run them, and saying so at export time is the honest place.
#: `entity` is the notable absence — it trains and has no `agent/src/policy.cpp`
#: implementation, so exporting one would produce a file the game accepts and
#: then evaluates as noise.
EXPORTABLE = ("mlp",)

#: How many decimal digits the parity fixture carries. float32 has about seven,
#: so this round-trips exactly while keeping the file diffable.
FIXTURE_DIGITS = 9

Weights = npt.NDArray[np.float32]
Legal = npt.NDArray[np.bool_]


class ExportError(Exception):
    """This checkpoint cannot become a policy the game could run, and why."""


@dataclass(frozen=True)
class Decision:
    """One forward pass: what the network said and what it would do."""

    logits: Weights
    value: float
    action: int


# ---- the conversion ----------------------------------------------------------


def export_checkpoint(
    checkpoint: Path,
    destination: Path,
    *,
    metadata: Mapping[str, str | int | float],
) -> Path:
    """Convert ``checkpoint`` to an `.mdp` at ``destination``.

    Validated and written atomically by :mod:`md.policy_format`, so a refusal
    leaves nothing behind and a success is a file this build can read back.

    ``metadata`` is merged *under* the provenance this function derives, so a
    caller can name the model without being able to lie about which checkpoint
    it came from.
    """
    import torch  # noqa: PLC0415 — optional, and only the exporter needs it

    try:
        # `weights_only=True` is torch's own refusal to unpickle anything but
        # tensors. It is the whole reason a `.pt` can be read here at all
        # without inheriting the property this format exists to remove.
        payload: dict[str, Any] = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 — torch raises a dozen unrelated types
        raise ExportError(f"{checkpoint}: not a readable checkpoint ({error})") from error
    if "policy" not in payload:
        raise ExportError(f"{checkpoint}: no policy state in this file")

    architecture = str(payload.get("architecture", "mlp"))
    if architecture not in EXPORTABLE:
        raise ExportError(
            f"{checkpoint}: architecture {architecture!r} has no native forward pass, "
            f"so it cannot be exported (this build exports {', '.join(EXPORTABLE)}). "
            "It can still be trained, evaluated and replayed through Python."
        )

    state = payload["policy"]
    tensors: list[Tensor] = []
    for name, dimensions in policy_format.ARCHITECTURES[architecture]:
        if name not in state:
            raise ExportError(
                f"{checkpoint}: the checkpoint has no {name!r}, which {architecture} needs "
                f"({len(dimensions)} dimensions)"
            )
        values = np.ascontiguousarray(state[name].detach().cpu().numpy(), dtype=np.float32)
        tensors.append(Tensor(name, tuple(int(extent) for extent in values.shape), values))

    provenance: dict[str, str | int | float] = dict(metadata)
    provenance.update(
        {
            "source": str(checkpoint),
            "architecture": architecture,
            "trained_updates": int(payload.get("iteration", 0)),
        }
    )
    policy = NativePolicy(
        schema=policy_format.SCHEMA,
        observation_size=int(payload["obs_size"]),
        action_count=int(payload["action_count"]),
        architecture=architecture,
        tensors=tuple(tensors),
        metadata=provenance,
    )
    try:
        return policy_format.write(destination, policy)
    except PolicyFormatError as error:
        raise ExportError(f"{checkpoint}: {error}") from error


# ---- the reference forward pass ---------------------------------------------


def evaluate(policy: NativePolicy, observation: Weights, legal: Legal) -> Decision:
    """Run ``policy`` on one observation. The definition `policy.cpp` must match.

    Written against the `.mdp` and not against torch, so it needs no torch and
    therefore no checkpoint — see this module's docstring for why that makes it
    the definition rather than a recording.
    """
    if policy.architecture != "mlp":
        raise ExportError(f"no reference forward pass for {policy.architecture!r}")
    if observation.shape != (policy.observation_size,):
        raise ExportError(
            f"observation is {observation.shape}, expected ({policy.observation_size},)"
        )
    if legal.shape != (policy.action_count,):
        raise ExportError(f"legal mask is {legal.shape}, expected ({policy.action_count},)")

    features = np.tanh(_affine(observation, policy, "trunk.0"))
    features = np.tanh(_affine(features, policy, "trunk.2"))
    logits = _affine(features, policy, "policy_head")
    value = float(_affine(features, policy, "value_head")[0])

    # numpy's stubs type `where`/`argmax` with Unknown parameters, so strict
    # mode cannot see through either; the arrays themselves are typed.
    masked: Weights = np.where(  # pyright: ignore[reportUnknownMemberType]
        legal, logits, np.float32(MASKED_LOGIT)
    ).astype(np.float32)
    # `argmax` returns the first maximum, which is the tie-break both sides
    # promise. Stated in a comment rather than assumed, because the C++ side
    # has to reproduce it deliberately.
    chosen = int(np.argmax(masked))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    return Decision(masked, value, chosen)


def _affine(x: Weights, policy: NativePolicy, layer: str) -> Weights:
    """``W @ x + b``, in float32 throughout.

    float32 and not float64: the C++ side has no double anywhere in its forward
    pass, and accumulating in a wider type here would put the two implementations
    a few ULPs apart on every layer — which is invisible until it flips an argmax
    between two near-equal logits, in one seed out of a hundred.
    """
    weight = policy.tensor(f"{layer}.weight").values
    bias = policy.tensor(f"{layer}.bias").values
    return (weight.astype(np.float32) @ x.astype(np.float32) + bias).astype(np.float32)


# ---- the fixture the C++ tests read -----------------------------------------


def write_parity_fixture(
    policy_path: Path,
    destination: Path,
    *,
    samples: int = 16,
    seed: int = 20260726,
) -> Path:
    """Pin what ``policy_path`` computes, as JSON, for the native tests.

    The observations are drawn from a fixed generator rather than from a real
    episode. That is deliberate: a real observation is mostly zeroes (the entity
    slots are empty early on), so a fixture built from one would exercise a
    fraction of the weights and pass with a transposed matrix. Random normals
    touch every input.
    """
    policy = policy_format.read(policy_path)
    rng = np.random.default_rng(seed)
    written: list[dict[str, object]] = []
    for index in range(samples):
        observation = rng.standard_normal(policy.observation_size).astype(np.float32)
        legal = rng.random(policy.action_count) > 0.35
        # A state with nothing legal cannot arise — the no-op always is — and a
        # fixture containing one would pin behaviour nothing has to define.
        legal[0] = True
        decision = evaluate(policy, observation, legal)
        written.append(
            {
                "index": index,
                "observation": _round(observation),
                "legal": [int(flag) for flag in legal],
                "logits": _round(decision.logits),
                "value": round(decision.value, FIXTURE_DIGITS),
                "action": decision.action,
            }
        )

    destination.write_text(
        json.dumps(
            {
                "policy": policy_path.name,
                "schema": policy.schema,
                "architecture": policy.architecture,
                "observation_size": policy.observation_size,
                "action_count": policy.action_count,
                "masked_logit": MASKED_LOGIT,
                "seed": seed,
                "samples": written,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _round(values: Sequence[float] | Weights) -> list[float]:
    return [round(float(value), FIXTURE_DIGITS) for value in values]
