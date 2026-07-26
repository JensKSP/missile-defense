# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Promotion produces something the rest of the system can actually use.

Task 6's e2e claim, and the word doing the work is **native**. `test_league.py`
proves a promoted `policy.mdp` can be read back by the Python that wrote it,
which is a round trip through one implementation. What the league is *for* is
handing a model to the evaluator and to the game, and neither of those has any
Python in it.

So: train briefly, promote from the library's own service, and then make
`md_agent_eval` play a seed with the result. If that works, every consumer
downstream of promotion works, because they all go through the same
`md::agent::Policy`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from .harness import agent_eval_binary, needs_agent_eval, needs_native, needs_torch

pytestmark = [pytest.mark.e2e, needs_torch, needs_native]


@pytest.fixture
def promoted(trained_run: Path, tmp_path: Path):  # noqa: ANN201 — md.league.Model
    """A real run, promoted the way the console promotes one."""
    from md import league, library  # noqa: PLC0415 — optional dependency

    run = library.load_run(trained_run)
    assert run is not None, f"the trainer left no run in {trained_run}"

    best = library.best_evaluated_checkpoint(trained_run)
    checkpoint = best[0] if best else trained_run / "checkpoints" / "policy-final.pt"
    assert checkpoint.exists()

    return league.promote(
        league.Promotion(
            checkpoint=checkpoint,
            display_name=library.model_name_for(run, checkpoint),
            source_run=run.run_id,
            trained_updates=run.updates,
        ),
        root=tmp_path / "models",
    )


@needs_agent_eval
def test_a_promoted_model_is_one_the_native_evaluator_can_play(promoted) -> None:  # noqa: ANN001
    """The claim. A file only Python can read would pass `test_league.py` and
    fail here, which is exactly the gap this test exists to close."""
    binary = agent_eval_binary()
    assert binary is not None
    result = subprocess.run(
        [
            str(binary),
            "--policy",
            str(promoted.policy),
            "--seeds",
            "1",
            "--max-ticks",
            "1200",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    # It played: the summary block is printed from the same C++ `Summary` the
    # scripted baseline uses, so a `mean score` line means an episode ran.
    assert "mean score" in result.stdout
    # And it is named by the name promotion gave it, not by its path.
    assert promoted.display_name.split()[0] in result.stdout


def test_promotion_carries_the_name_into_the_file(promoted) -> None:  # noqa: ANN001
    """Where the game's HUD reads it from, and the league's table too."""
    from md import policy_format  # noqa: PLC0415 — optional dependency

    policy = policy_format.read(promoted.policy)
    assert policy.metadata["display_name"] == promoted.display_name
    # Provenance survives the run being deleted, which is the point of copying.
    assert promoted.source_run
    assert promoted.source_checkpoint.endswith(".pt")


def test_a_promoted_model_survives_its_run_being_removed(trained_run: Path, tmp_path: Path) -> None:
    """The whole reason promotion copies instead of pointing at a checkpoint.

    Cleanup and archiving (Task 9) delete run directories; a league that held
    references would lose its models on the first tidy-up.

    A **copy** of the run is promoted and then deleted, not the shared fixture.
    `trained_run` is session-scoped — it is the input to most of this suite —
    and removing it here made three tests in `test_training.py` fail depending
    on collection order, which is a far more annoying bug than the one this
    test is about.
    """
    import shutil  # noqa: PLC0415

    from md import league, library, policy_format  # noqa: PLC0415

    doomed = tmp_path / "doomed-run"
    shutil.copytree(trained_run, doomed)
    run = library.load_run(doomed)
    assert run is not None
    best = library.best_evaluated_checkpoint(doomed)
    checkpoint = best[0] if best else doomed / "checkpoints" / "policy-final.pt"

    root = tmp_path / "models"
    model = league.promote(
        league.Promotion(checkpoint, "Outlives Its Run", source_run=run.run_id), root=root
    )

    shutil.rmtree(doomed)
    assert not doomed.exists()

    found = league.find(model.model_id, root)
    assert found is not None
    assert policy_format.read(found.policy).action_count > 0
