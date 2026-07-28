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

from .harness import agent_eval_binary, needs_agent_eval, needs_native, needs_qt, needs_torch

pytestmark = [pytest.mark.e2e, needs_torch, needs_native]


@pytest.fixture
def promoted(trained_run: Path, tmp_path: Path):  # noqa: ANN201 — missile_defense.runs.league.Model
    """A real run, promoted the way the trainer promotes one."""
    from missile_defense.runs import league, library  # noqa: PLC0415 — optional dependency

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
    from missile_defense.sim import policy_format  # noqa: PLC0415 — optional dependency

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

    from missile_defense.runs import league, library  # noqa: PLC0415
    from missile_defense.sim import policy_format  # noqa: PLC0415

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


@needs_qt
def test_a_run_can_be_promoted_from_the_run_list(
    qt_app: object,  # noqa: ARG001 — the QApplication has to exist
    trained_run: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The short way in: promote from the list, without opening the run first.

    Promotion used to live only inside a run, which put four clicks between
    "that one won" and the model being in the game — while the comparison that
    decides it happens in the list, where every run's best score is one column.

    The dialog is not shown: `exec` is replaced by the same `_promote` an OK
    press would call, so what is under test is the path from the list to a real
    `.mdp` on disk rather than Qt's ability to draw a form.
    """
    from missile_defense.runs import league
    from missile_defense.ui.library import LibraryView
    from PySide6.QtWidgets import QDialog, QMessageBox

    root = tmp_path / "models"
    monkeypatch.setenv("MD_MODELS_DIR", str(root))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    def straight_to_ok(dialog: object) -> int:
        dialog._promote()  # noqa: SLF001 — standing in for the button press
        return (
            QDialog.DialogCode.Accepted
            if dialog.promoted is not None
            else QDialog.DialogCode.Rejected
        )

    monkeypatch.setattr("missile_defense.ui.league.PromoteDialog.exec", straight_to_ok)

    announced: list[int] = []
    view = LibraryView()
    view.promoted.connect(lambda: announced.append(1))
    try:
        view.attach(trained_run.parent)
        rows = view.table
        for index, run in enumerate(rows._runs):  # noqa: SLF001 — what `selected` reads
            if run.path == trained_run:
                rows._table.selectRow(index)  # noqa: SLF001
        assert rows.selected() is not None, "the trained run is not in the list"
        rows._promote_selected()  # noqa: SLF001 — standing in for the button press
    finally:
        view.close()

    assert announced, "the league was never told to re-read itself"
    installed = league.models(root)
    assert len(installed) == 1
    assert installed[0].source_run == trained_run.name
    assert installed[0].policy.is_file()


@needs_agent_eval
def test_a_league_score_is_the_number_the_published_evaluator_prints(
    promoted,  # noqa: ANN001
) -> None:
    """The league's number and `md_agent_eval`'s are the same number.

    They are now the same *code*: scoring goes through `md_native.LoadedPolicy`,
    which is `md::agent::Policy` driven by `run_episode` — what the binary runs
    and what the game runs. Before that, inference was a NumPy forward pass
    called once per observation from Python, and a canonical block took hours
    per contestant instead of a minute and a half; a head-to-head looked like a
    hang, because a progress bar that cannot count seeds has nothing to show.

    One seed and a short cap: what is asserted is that two implementations agree
    on an episode, not how long an episode is.
    """
    from missile_defense.runs import tournament  # noqa: PLC0415 — optional dependency

    seed = int(tournament.canonical_protocol().seeds()[0])
    runner = tournament.load_policy(promoted)
    episode = runner.play(seed, 1200, 4)

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
            "--per-episode",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    # The per-episode table's first column is the seed and its second the score.
    row = next(line for line in result.stdout.splitlines() if line.startswith(str(seed)))
    assert int(row.split()[1]) == episode.score
    assert int(row.split()[4]) == episode.ticks


def test_evaluating_a_model_counts_the_seeds_it_has_finished(promoted) -> None:  # noqa: ANN001
    """A progress bar with a number in it, which is what makes a contest waitable.

    The old callback passed a hardcoded zero — every report said "0 of 32 seeds"
    for however many hours it took. Reported *after* each episode, so the count
    is of seeds that are in, and so a cancellation raised through it lands on a
    boundary with nothing recorded.
    """
    from missile_defense.runs import tournament  # noqa: PLC0415 — optional dependency

    protocol = tournament.Protocol(
        seed_split=tournament.benchmark.CANONICAL_SPLIT,
        seed_offset=tournament.benchmark.CANONICAL_SEED_OFFSET,
        seed_count=2,
        frame_skip=4,
        max_ticks=600,
    )
    seen: list[tuple[int, int]] = []
    result = tournament.evaluate_model(
        promoted,
        protocol,
        progress=lambda index, contestants, done, total: seen.append((done, total)),
        index=1,
        contestants=2,
    )
    assert seen == [(0, 2), (1, 2), (2, 2)]
    assert result.episodes == 2
    assert not result.canonical  # two seeds is not the published block
