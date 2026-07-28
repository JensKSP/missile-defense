# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The same policy file chooses the same move in both languages.

**This is Task 2's whole claim, and it can only be made from out here.**
`test_export_policy.py` proves the NumPy forward pass matches torch, and
`agent/tests/unit/test_policy.cpp` proves the C++ one matches a fixture. Both run
inside one process against data one side produced. What neither can show is that
the *file* carries enough, and carries it correctly, for two independently
written implementations to play the same game — and the file is the thing that
ships.

So: export one checkpoint, hand the `.mdp` to `md_agent_eval` and to the Python
evaluator, give both the same seed, and compare the action logs decision for
decision. A single differing index is a divergence that compounds: the two
simulations are in different states from that tick on, and the scores stop being
comparable long before anyone notices the cause.

The logs are compared at *decision* granularity rather than per tick, because
that is the only rate both sides share — the simulation samples an action once
per `Config::decision_interval` and holds it between, so the ticks in between are
ones nothing was asked about.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from .harness import agent_eval_binary, needs_agent_eval, needs_native, needs_torch

pytestmark = [pytest.mark.e2e, needs_torch, needs_native, needs_agent_eval]

#: Long enough that the policy has made hundreds of decisions and the episode has
#: reached real states — waves spawning, blasts expiring, ammo running down —
#: and short enough that the test is seconds rather than minutes. A divergence
#: shows up in the first dozen decisions or not at all.
PARITY_TICKS = 3000

#: One seed, from the validation split. Which seed does not matter; that both
#: sides play the *same* one does.
PARITY_SEED_OFFSET = 0


@pytest.fixture(scope="module")
def exported(trained_run: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real checkpoint from a real run, converted to the file that ships."""
    from missile_defense import export_policy  # noqa: PLC0415 — optional dependency

    checkpoint = trained_run / "checkpoints" / "policy-final.pt"
    saved = sorted(trained_run.rglob("*.pt"))
    assert checkpoint.exists(), f"the run wrote no final checkpoint: {saved}"
    destination = tmp_path_factory.mktemp("exported") / "policy.mdp"
    return export_policy.export_checkpoint(
        checkpoint, destination, metadata={"display_name": "Parity"}
    )


def _native_actions(policy: Path, log: Path) -> list[int]:
    """`md_agent_eval` playing one seed, and the indices it chose."""
    binary = agent_eval_binary()
    assert binary is not None
    result = subprocess.run(
        [
            str(binary),
            "--policy",
            str(policy),
            "--seeds",
            "1",
            "--seed-offset",
            str(PARITY_SEED_OFFSET),
            "--max-ticks",
            str(PARITY_TICKS),
            "--action-log",
            str(log),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"the evaluator failed:\n{result.stdout}\n{result.stderr}"
    return [int(line) for line in log.read_text(encoding="utf-8").split()]


def _python_actions(policy: Path) -> list[int]:
    """The Python evaluator playing the same seed, through the same `.mdp`."""
    import numpy as np  # noqa: PLC0415 — optional dependency
    from missile_defense import eval as md_eval  # noqa: PLC0415 — optional dependency
    from missile_defense import export_policy, policy_format  # noqa: PLC0415 — optional dependency
    from missile_defense.benchmark import (
        CANONICAL_FRAME_SKIP,  # noqa: PLC0415 — optional dependency
    )

    loaded = policy_format.read(policy)
    chosen: list[int] = []

    def act(observations, masks):  # noqa: ANN001, ANN202 — numpy arrays
        actions = np.zeros(len(observations), dtype=np.int32)
        for i, (observation, mask) in enumerate(zip(observations, masks, strict=True)):
            decision = export_policy.evaluate(loaded, observation, mask)
            actions[i] = decision.action
            chosen.append(decision.action)
        return actions

    md_eval.evaluate(
        act,
        seeds=md_eval.validation_seeds(1 + PARITY_SEED_OFFSET)[PARITY_SEED_OFFSET:],
        frame_skip=CANONICAL_FRAME_SKIP,
        max_ticks=PARITY_TICKS,
        threads=1,
    )
    return chosen


def test_both_languages_play_the_same_seed_identically(exported: Path, tmp_path: Path) -> None:
    """Decision for decision, from the same file. The claim Task 2 exists to make."""
    native = _native_actions(exported, tmp_path / "native.log")
    python = _python_actions(exported)

    assert native, "the native evaluator logged no decisions"
    # The two loops stop on slightly different conditions — the native one at the
    # tick cap, the Python one when the environment reports done — so the shorter
    # log is the honest comparison length. What must hold is that they agree for
    # every decision they both made, and that is hundreds of them.
    shared = min(len(native), len(python))
    assert shared > 50, f"only {shared} shared decisions; the comparison proves nothing"
    for index in range(shared):
        assert native[index] == python[index], (
            f"the two implementations diverged at decision {index}: "
            f"native chose {native[index]}, Python chose {python[index]}. "
            f"Everything after this point is a different episode."
        )


def test_the_native_evaluator_names_the_model_it_is_running(exported: Path) -> None:
    """The display name out of the file, not the filename.

    Same fact the game's HUD needs (Task 3 Step 4b): a path is not a name, and
    `policy-final.pt` says nothing about which run produced it.
    """
    binary = agent_eval_binary()
    assert binary is not None
    result = subprocess.run(
        [str(binary), "--policy", str(exported), "--seeds", "1", "--max-ticks", "600"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Parity" in result.stdout


def test_the_native_evaluator_refuses_a_policy_it_cannot_run(tmp_path: Path) -> None:
    """A corrupt or foreign `.mdp` is an error, not a silently worse agent."""
    binary = agent_eval_binary()
    assert binary is not None
    junk = tmp_path / "not-a-policy.mdp"
    junk.write_bytes(b"MDPOLICY" + bytes(64))
    result = subprocess.run(
        [str(binary), "--policy", str(junk), "--seeds", "1", "--max-ticks", "60"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert "not-a-policy.mdp" in result.stderr
