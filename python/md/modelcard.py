# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""What a run is training, written where the console can read it. No torch here.

The console must never import torch — that is a test, not a habit
(docs/ROADMAP.md, M8, risk 3) — and the roadmap's own answer to the rule is that
anything needing model state belongs in ``md.train``, surfaced as an artifact the
console reads. This is that artifact: ``runs/model.json``, beside ``config.json``
and for the same reason. One says what the run was started with, the other says
what it is training.

**One file per run, not one per checkpoint.** Within a run the architecture, the
layer shapes and the observation and action sizes never change; only the
iteration does, and that is already in the checkpoint's *name*. A sidecar beside
every ``policy-*.pt`` would be the same bytes repeated once per hundred updates.

Nothing here imports torch, not even lazily: :func:`describe` takes the *shapes*
out of a state dict, and a shape is a tuple of ints. That keeps this module
importable from either side of the boundary without either side having to think
about it — which is the same trick :mod:`md.control` plays with the pause file.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Beside ``config.json`` in the run directory.
FILENAME = "model.json"

#: State-dict names end in one of these, and both belong to the same layer.
TENSOR_ROLES = ("weight", "bias")


@dataclass(frozen=True)
class Tensor:
    """One entry of the state dict: ``("trunk.0.weight", (512, 1959))``."""

    name: str
    shape: tuple[int, ...]

    @property
    def parameters(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class Block:
    """A layer as a person reads it — the weight, with its bias folded in.

    A state dict lists ``trunk.0.weight`` and ``trunk.0.bias`` separately, which
    is right for loading and wrong for looking at: it doubles the rows and hides
    the one number per row worth reading, the shape of the transform.
    """

    name: str
    shape: tuple[int, ...]
    parameters: int

    @property
    def wiring(self) -> str:
        """``1959 → 512`` for a weight matrix; the bare extent for anything else.

        Torch stores a linear layer's weight as ``(out, in)``, so it reads
        backwards; turning it round is the whole reason this is not just the
        tuple.
        """
        if len(self.shape) == 2:
            return f"{self.shape[1]} → {self.shape[0]}"
        return " × ".join(str(extent) for extent in self.shape)


@dataclass(frozen=True)
class ModelCard:
    """The network a run is training, in the terms M8 phase 4 asks for."""

    architecture: str
    obs_size: int
    action_count: int
    hidden: int
    tensors: tuple[Tensor, ...]

    @property
    def parameters(self) -> int:
        return sum(tensor.parameters for tensor in self.tensors)

    def blocks(self) -> list[Block]:
        """The layers, in state-dict order, ready to be listed.

        Order is preserved rather than sorted: a state dict comes out in
        definition order, which is the order the data flows through — and a
        table of layers sorted by name would be a table of layers in no
        particular order.
        """
        found: dict[str, Block] = {}
        for tensor in self.tensors:
            stem, _, role = tensor.name.rpartition(".")
            key = stem if stem and role in TENSOR_ROLES else tensor.name
            previous = found.get(key)
            if previous is None:
                found[key] = Block(key, tensor.shape, tensor.parameters)
                continue
            # The weight carries the shape worth showing; a bias arriving first
            # (it does not, but nothing here depends on that) must not win.
            shape = tensor.shape if len(tensor.shape) > len(previous.shape) else previous.shape
            found[key] = Block(key, shape, previous.parameters + tensor.parameters)
        return list(found.values())


# ---- what it reads like -----------------------------------------------------
# Pure, so the panel's text is covered by tests rather than by eye — the same
# bargain md.ui.sources makes with its own glanceable formatting.


def headline(card: ModelCard) -> str:
    """``mlp · 1,464,194 parameters`` — the two facts about a network."""
    return f"{card.architecture} · {card.parameters:,} parameters"


def wiring(card: ModelCard) -> str:
    """``1,959 observations → 385 actions`` — the shape of the problem."""
    return f"{card.obs_size:,} observations → {card.action_count:,} actions"


def layer_table(card: ModelCard) -> list[str]:
    """One padded line per layer, widest column first — for a monospaced label.

    Padded here rather than by a table widget because four rows of three columns
    is not a table, it is a paragraph with alignment; and a `QLabel` costs the
    panel nothing when the run has no card to show.
    """
    blocks = card.blocks()
    if not blocks:
        return []
    name = max(len(block.name) for block in blocks)
    shape = max(len(block.wiring) for block in blocks)
    count = max(len(f"{block.parameters:,}") for block in blocks)
    return [
        f"{block.name:<{name}}  {block.wiring:>{shape}}  {block.parameters:>{count},}"
        for block in blocks
    ]


def describe(
    shapes: Mapping[str, Sequence[int]],
    *,
    architecture: str,
    obs_size: int,
    action_count: int,
    hidden: int,
) -> ModelCard:
    """Build a card from a state dict's shapes — ``{name: tuple(t.shape)}``."""
    return ModelCard(
        architecture=architecture,
        obs_size=obs_size,
        action_count=action_count,
        hidden=hidden,
        tensors=tuple(Tensor(name, tuple(shape)) for name, shape in shapes.items()),
    )


def write(run_dir: Path, card: ModelCard) -> Path:
    """Write ``<run_dir>/model.json``. Returns where it went."""
    path = run_dir / FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": card.architecture,
        "obs_size": card.obs_size,
        "action_count": card.action_count,
        "hidden": card.hidden,
        "parameters": card.parameters,  # derived, but this file is also read by eye
        "tensors": [{"name": t.name, "shape": list(t.shape)} for t in card.tensors],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read(run_dir: Path) -> ModelCard | None:
    """The card in ``run_dir``, or ``None`` if there is not a readable one.

    ``None`` covers three states the console shows the same way — no file yet, a
    run written before this file existed, and a file caught mid-write — because
    the answer to all three is "there is nothing to show yet", and the panel
    already says which run it is looking at.
    """
    try:
        # Annotated `Any` rather than left to inference: what is on disk is
        # whatever is on disk, and the `except` below is the type check.
        payload: Any = json.loads((run_dir / FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        tensors = tuple(
            Tensor(str(entry["name"]), tuple(int(extent) for extent in entry["shape"]))
            for entry in payload.get("tensors", ())
        )
        return ModelCard(
            architecture=str(payload["architecture"]),
            obs_size=int(payload["obs_size"]),
            action_count=int(payload["action_count"]),
            hidden=int(payload["hidden"]),
            tensors=tensors,
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        # Every way the file can be something other than a card ends here — a
        # list where an object was expected, a missing key, a shape that is not
        # numbers. There is one answer to all of them and the panel gives it.
        return None
