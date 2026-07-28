# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""What a training configuration costs in GPU memory, and what it is for.

The model exists because a preset shipped that ran out of memory on the card it
was designed for: the first `best` scaled the batch fourfold and left
`minibatches` at 8, which asks for sixty times more working memory than the
rollout buffer it was being judged by. These tests hold the model to the
measurements in the module docstring, and every built-in preset to the model.
"""

from __future__ import annotations

import pytest
from missile_defense.runs import footprint, presets

#: (architecture, envs, steps, minibatches, measured GiB) — two updates of
#: missile_defense.training.train on an RTX 5090, reading torch.cuda.max_memory_allocated().
MEASURED = [
    ("entity", 1024, 256, 8, 18.95),
    ("entity", 2048, 256, 16, 20.97),
    ("entity", 4096, 256, 64, 16.61),
    ("entity", 2048, 512, 64, 16.59),
    ("mlp", 1024, 256, 8, 2.81),
    ("mlp", 4096, 128, 8, 5.56),
]


@pytest.mark.parametrize(("architecture", "envs", "steps", "minibatches", "measured"), MEASURED)
def test_the_estimate_matches_what_was_measured_on_the_card(
    architecture: str, envs: int, steps: int, minibatches: int, measured: float
) -> None:
    estimate = footprint.estimate_gib(
        envs=envs, steps=steps, minibatches=minibatches, architecture=architecture
    )
    assert estimate == pytest.approx(measured, rel=0.03), (
        f"{architecture} {envs}x{steps} mb{minibatches}: "
        f"estimated {estimate:.2f} GiB against a measured {measured} GiB"
    )


def test_memory_follows_the_minibatch_and_not_the_batch() -> None:
    # The whole point of the module, and the mistake it was written after:
    # quadrupling the batch while quartering the minibatch *reduces* peak memory,
    # which is the opposite of what the batch size alone would suggest.
    good = dict(envs=1024, steps=256, minibatches=8, architecture="entity")
    scaled = dict(envs=2048, steps=512, minibatches=64, architecture="entity")
    assert 4 * (good["envs"] * good["steps"]) == scaled["envs"] * scaled["steps"]  # type: ignore[operator]
    assert footprint.estimate_gib(**scaled) < footprint.estimate_gib(**good)  # type: ignore[arg-type]

    # And the trap itself: the same samples at the default minibatch count is
    # the configuration that died.
    naive = {**scaled, "minibatches": 8}
    assert footprint.estimate_gib(**naive) > 60  # type: ignore[arg-type]


def test_an_unknown_architecture_is_costed_as_the_expensive_one() -> None:
    # An estimate that is too low is the only kind that costs someone a run.
    unknown = footprint.estimate_bytes(
        envs=1024, steps=256, minibatches=8, architecture="whatever-comes-next"
    )
    assert unknown == footprint.estimate_bytes(
        envs=1024, steps=256, minibatches=8, architecture="entity"
    )


def test_fitting_leaves_room_for_the_machine_to_be_in_use() -> None:
    shape = dict(envs=2048, steps=512, minibatches=64, architecture="entity")
    needed = footprint.estimate_bytes(**shape)  # type: ignore[arg-type]
    # Exactly enough is not enough: the allocator reserves more than it
    # allocates, and a desktop session is a gigabyte before anyone opens a game.
    assert not footprint.fits_in(needed, **shape)  # type: ignore[arg-type]
    assert footprint.fits_in(int(needed * footprint.HEADROOM) + 1, **shape)  # type: ignore[arg-type]


def test_pressure_counts_what_the_card_is_already_holding() -> None:
    # The bar is drawn against the card, not against what is free, so memory
    # somebody else is holding has to push the fill along exactly as this run's
    # own allocation would.
    shape = dict(envs=1024, steps=256, minibatches=8, architecture="mlp")
    total = 32 * footprint.GIB
    empty = footprint.pressure(free_bytes=total, total_bytes=total, **shape)  # type: ignore[arg-type]
    busy = footprint.pressure(free_bytes=total - 8 * footprint.GIB, total_bytes=total, **shape)  # type: ignore[arg-type]
    assert busy == pytest.approx(empty + 0.25), "8 of 32 GiB in use is a quarter of the bar"


def test_a_red_bar_and_the_warning_under_it_are_the_same_claim() -> None:
    # `level` colours the bar and `fits_in` writes the sentence. If they disagree
    # the dialog contradicts itself, so they cross at exactly the same point.
    shape = dict(envs=2048, steps=512, minibatches=64, architecture="entity")
    total = 32 * footprint.GIB
    for free_gib in range(0, 33):
        free = free_gib * footprint.GIB
        over = footprint.level(free_bytes=free, total_bytes=total, **shape) == "over"  # type: ignore[arg-type]
        assert over is not footprint.fits_in(free, **shape), (  # type: ignore[arg-type]
            f"at {free_gib} GiB free the bar and the warning disagree"
        )


def test_fitting_and_fitting_comfortably_are_told_apart() -> None:
    # The state the old single line could not express: a run that fits and also
    # takes the whole machine with it.
    shape = dict(envs=1024, steps=256, minibatches=8, architecture="entity")
    total = 32 * footprint.GIB
    assert footprint.level(free_bytes=total, total_bytes=total, **shape) == "tight"  # type: ignore[arg-type]
    # The same run on a card twice the size is simply fine.
    roomy = 80 * footprint.GIB
    assert footprint.level(free_bytes=roomy, total_bytes=roomy, **shape) == "ok"  # type: ignore[arg-type]
    # And the flat architecture on the documented card never leaves green.
    flat = dict(envs=1024, steps=256, minibatches=8, architecture="mlp")
    assert footprint.level(free_bytes=total, total_bytes=total, **flat) == "ok"  # type: ignore[arg-type]


def test_a_card_that_cannot_be_measured_is_not_reported_as_full() -> None:
    # Division by a zero-sized card, which is what a probe that answered with
    # nothing looks like by the time it reaches here.
    shape = dict(envs=1024, steps=256, minibatches=8, architecture="mlp")
    assert footprint.pressure(free_bytes=0, total_bytes=0, **shape) == 0.0  # type: ignore[arg-type]
    assert footprint.level(free_bytes=0, total_bytes=0, **shape) == "ok"  # type: ignore[arg-type]


def test_the_advice_names_the_knob_that_actually_helps() -> None:
    text = footprint.advice(envs=4096, steps=512, minibatches=8, architecture="entity")
    assert "--minibatches 16" in text, "the advice does not offer the cheapest fix"
    assert "2,097,152" in text  # the batch it is talking about
    assert "262,144" in text  # and the minibatch that is the actual problem


@pytest.mark.parametrize("preset", presets.BUILTIN, ids=lambda preset: preset.name)
def test_every_built_in_preset_fits_on_the_card_this_project_documents(
    preset: presets.Preset,
) -> None:
    # 32 GiB (docs/NVIDIA.md), of which the desktop and the game already hold a
    # couple. A shipped preset that cannot run on the documented hardware is the
    # bug this whole module exists to prevent recurring.
    shape = {
        "envs": int(preset.options.get("envs", 1024)),
        "steps": int(preset.options.get("steps", 256)),
        "minibatches": int(preset.options.get("minibatches", 8)),
        "architecture": preset.options.get("architecture", "mlp"),
    }
    free = 30 * footprint.GIB
    estimate = footprint.estimate_gib(**shape)  # type: ignore[arg-type]
    assert footprint.fits_in(free, **shape), (  # type: ignore[arg-type]
        f"preset '{preset.name}' needs about {estimate:.1f} GiB, which does not fit "
        f"in {free / footprint.GIB:.0f} GiB with headroom"
    )
