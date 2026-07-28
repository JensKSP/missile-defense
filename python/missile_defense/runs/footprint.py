# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""How much GPU memory a training configuration needs. No Qt, no torch.

Two terms, and the second one is the trap:

* The **rollout buffer** is `envs × steps` samples, preallocated once
  (:class:`missile_defense.training.ppo.Rollout`). It scales with the batch, and
  it is the term people
  expect.
* The **minibatch** is what the update actually pushes through the network, and
  on the relational architecture it dominates by a factor of sixty. Its entity
  encoders and auxiliary targets build per-sample threat×entity tensors
  (:mod:`missile_defense.training.auxiliary`), so peak memory follows `batch / minibatches` — *not*
  the batch. Doubling `--minibatches` roughly halves the peak while training on
  exactly the same data.

That is why 4,096 envs × 512 steps at the default 8 minibatches falls over on a
32 GiB card while 2,048 × 512 at 64 minibatches — the same number of samples per
update — fits in 17 GiB.

The constants are **measured**, not derived: two updates of `missile_defense.training` on an
RTX 5090, reading `torch.cuda.max_memory_allocated()`. They are a straight line
through those points to within 1%:

| architecture | envs × steps | minibatches | predicted | measured |
|---|---|---|---|---|
| entity | 1024 × 256 | 8 | 18.9 GiB | 18.95 GiB |
| entity | 2048 × 256 | 16 | 21.0 GiB | 20.97 GiB |
| entity | 4096 × 256 | 64 | 16.6 GiB | 16.61 GiB |
| entity | 2048 × 512 | 64 | 16.6 GiB | 16.59 GiB |
| mlp | 1024 × 256 | 8 | 2.8 GiB | 2.81 GiB |
| mlp | 4096 × 128 | 8 | 5.6 GiB | 5.56 GiB |

An estimate, and presented as one: PyTorch's caching allocator *reserves* 10–30%
more than it allocates, and a card is never empty — a desktop session and the
game are a gigabyte or two between them. :data:`HEADROOM` is what turns this
into the "will it fit?" answer a dialog needs.
"""

from __future__ import annotations

#: Floats in one observation, and legal actions in one mask. Fixed by the game's
#: `Config`, and asserted against the real environment by the native tests —
#: this module cannot import `missile_defense.sim.env` to ask, because an installed trainer has
#: the trainer's source but not always its compiled binding.
OBSERVATION_FLOATS = 1959
ACTION_COUNT = 385

#: One sample in the rollout: the observation and mask above, plus the action
#: (int64) and four float32 columns (log prob, value, reward, continues).
ROLLOUT_BYTES_PER_SAMPLE = OBSERVATION_FLOATS * 4 + ACTION_COUNT + 8 + 4 * 4

#: Peak working memory per *minibatch* sample, by architecture. The relational
#: path is sixty times the flat one because it materialises threat×entity pairs;
#: see the table above for where these come from.
WORKING_BYTES_PER_SAMPLE = {
    "mlp": 26_000,
    "entity": 560_000,
}

#: Multiplier applied before comparing with free memory: the allocator's reserve
#: over what it allocates, plus room for the machine to be in use. A run that is
#: predicted to fit with nothing to spare is a run that dies at update 300
#: because somebody opened a browser.
HEADROOM = 1.35

GIB = 1024**3


def estimate_bytes(*, envs: int, steps: int, minibatches: int, architecture: str) -> int:
    """Peak GPU memory one update is expected to allocate, in bytes.

    Unknown architectures fall back to the relational figure, deliberately: a new
    network is far more likely to resemble the expensive one than the flat one,
    and an estimate that is too low is the only kind that costs someone a run.
    """

    batch = max(1, envs) * max(1, steps)
    per_sample = WORKING_BYTES_PER_SAMPLE.get(architecture, WORKING_BYTES_PER_SAMPLE["entity"])
    minibatch = batch // max(1, minibatches)
    return batch * ROLLOUT_BYTES_PER_SAMPLE + minibatch * per_sample


def estimate_gib(*, envs: int, steps: int, minibatches: int, architecture: str) -> float:
    """:func:`estimate_bytes` in the unit a person reads a GPU's size in."""

    return (
        estimate_bytes(envs=envs, steps=steps, minibatches=minibatches, architecture=architecture)
        / GIB
    )


def fits_in(free_bytes: int, *, envs: int, steps: int, minibatches: int, architecture: str) -> bool:
    """Whether that configuration should fit in ``free_bytes``, with headroom."""

    return (
        estimate_bytes(envs=envs, steps=steps, minibatches=minibatches, architecture=architecture)
        * HEADROOM
        <= free_bytes
    )


def advice(*, envs: int, steps: int, minibatches: int, architecture: str) -> str:
    """What to change, in the order of what actually helps.

    The first suggestion is always `--minibatches`, because it is the only one
    that costs nothing: the same samples, the same update, in smaller pieces.
    """

    batch = max(1, envs) * max(1, steps)
    minibatch = batch // max(1, minibatches)
    return (
        f"batch {batch:,} samples ({envs:,} envs x {steps} steps), "
        f"minibatch {minibatch:,} (--minibatches {minibatches}), architecture {architecture}.\n"
        f"Peak memory follows the *minibatch*, not the batch: on the relational "
        f"architecture one sample costs about "
        f"{WORKING_BYTES_PER_SAMPLE.get(architecture, WORKING_BYTES_PER_SAMPLE['entity']) // 1024} "
        f"KiB of working memory.\n"
        f"Try --minibatches {minibatches * 2} first — same data, same update, half the peak. "
        f"Then fewer --envs or --steps, which shrinks the rollout buffer as well."
    )
