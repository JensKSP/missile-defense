# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Cleaning up, archiving and restoring a real run through the real dialogs.

`md.archive` has been tested since Task 9 and none of it was reachable from the
trainer. What is asserted here is the part unit tests cannot reach: that the
dialog computes a plan, *shows* it, and then executes the plan it showed — and
that a round trip out to a ZIP and back leaves a run the library still lists.

The confirmations are answered by monkeypatching `QMessageBox`, not by clicking:
these are destructive operations, and a test that could be made to pass by a
stray Enter is not evidence of anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .harness import needs_native, needs_qt, needs_torch

pytestmark = [pytest.mark.e2e, needs_qt, needs_native, needs_torch]


@pytest.fixture
def library_copy(trained_run: Path, tmp_path: Path) -> Path:
    """A private copy of the session's run, in a library of its own.

    A copy because everything below deletes things, and the fixture is
    session-scoped: a test that cleaned up the shared run would break every
    later test by collection order rather than by any fault of its own.
    """
    import shutil

    root = tmp_path / "library"
    shutil.copytree(trained_run, root / trained_run.name)
    return root


def _run(root: Path):  # noqa: ANN202 — md.library is an optional-dependency import
    from md import library

    runs = library.discover(root)
    assert runs, f"no runs discovered in {root}"
    return runs[0]


def test_the_dialog_executes_the_plan_it_showed(
    qt_app: object,  # noqa: ARG001 — the QApplication has to exist
    library_copy: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from md.ui.storage import StorageDialog
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    run = _run(library_copy)
    dialog = StorageDialog(run, library_copy)
    try:
        planned = list(dialog._plan.remove)
        # The list on screen is the plan, item for item. A count with no names
        # is not a thing anyone can agree to, and a list that disagreed with
        # what gets deleted would be worse than no list.
        shown = [dialog._list.item(i).text() for i in range(dialog._list.count())]
        assert shown == [path.name for path in planned]

        if not planned:
            pytest.skip("this run has nothing a cleanup would take")

        dialog._clean_up()
        assert dialog.changed
        for path in planned:
            assert not path.exists(), f"{path.name} was shown but survived"
    finally:
        dialog.close()

    # And nothing that matters went with it: the trainer still draws this run.
    after = _run(library_copy)
    assert (after.path / "metrics.csv").is_file()
    assert (after.path / "evals.csv").is_file()


def test_a_run_survives_a_round_trip_through_an_archive(
    qt_app: object,  # noqa: ARG001 — the QApplication has to exist
    library_copy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Archive, remove, restore — the sequence the button offers, end to end.

    The ordering is the feature: written, verified, and only then deleted. What
    this proves is the half a unit test cannot, that the dialog wires those
    three together in that order and that the library sees the result.
    """
    from md import library
    from md.ui import storage
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    archive_path = tmp_path / "archived.zip"
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(archive_path), ""))
    )

    run = _run(library_copy)
    original = run.path
    dialog = storage.StorageDialog(run, library_copy)
    try:
        dialog._archive_run(remove=True)
    finally:
        dialog.close()

    assert archive_path.is_file(), "no archive was written"
    assert not original.exists(), "the original survived an archive-and-remove"
    assert not library.discover(library_copy), "the library still lists a removed run"

    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(archive_path), ""))
    )
    restored = storage.restore(library_copy, None)
    assert restored is not None
    assert library.discover(library_copy), "a restored run is not in the library"
    assert (restored / "metrics.csv").is_file()


def test_restoring_over_an_existing_run_is_refused(
    qt_app: object,  # noqa: ARG001 — the QApplication has to exist
    library_copy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A half-merged run has `metrics.csv` from one and checkpoints from another,
    # and nothing downstream would notice. Refused, and said out loud.
    from md import archive
    from md.ui import storage
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    run = _run(library_copy)
    written = archive.create_archive(run, tmp_path / "same.zip")

    warned: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda _p, _t, text, *a, **k: warned.append(text))
    )
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(written), ""))
    )

    assert storage.restore(library_copy, None) is None
    assert warned, "restoring over an existing run said nothing"
    assert "already exists" in warned[0]


def test_a_model_can_be_exported_and_imported_back(
    qt_app: object,  # noqa: ARG001 — the QApplication has to exist
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The round trip Import implied and Export did not exist for.

    A league you can put models into and never get one out of is a place they
    go rather than a place they live — and the whole point of `.mdp` is that it
    travels. Byte-for-byte, because the file in the league already *is* the
    portable format and re-exporting it could produce something subtly other
    than the thing that was scored.
    """
    import numpy as np
    from md import league, policy_format
    from md.ui.league import LeagueView
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    rng = np.random.default_rng(4)

    def normal(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.1).astype(np.float32)

    policy = policy_format.NativePolicy(
        schema=policy_format.SCHEMA,
        observation_size=6,
        action_count=4,
        architecture="mlp",
        tensors=(
            policy_format.Tensor("trunk.0.weight", (3, 6), normal(3, 6)),
            policy_format.Tensor("trunk.0.bias", (3,), normal(3)),
            policy_format.Tensor("trunk.2.weight", (3, 3), normal(3, 3)),
            policy_format.Tensor("trunk.2.bias", (3,), normal(3)),
            policy_format.Tensor("policy_head.weight", (4, 3), normal(4, 3)),
            policy_format.Tensor("policy_head.bias", (4,), normal(4)),
            policy_format.Tensor("value_head.weight", (1, 3), normal(1, 3)),
            policy_format.Tensor("value_head.bias", (1,), normal(1)),
        ),
        metadata={"display_name": "Amber Anvil"},
    )
    root = tmp_path / "models"
    (root / "aaaa").mkdir(parents=True)
    policy_format.write(root / "aaaa" / league.POLICY_NAME, policy)

    out = tmp_path / "shared" / "amber.mdp"
    out.parent.mkdir()
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), ""))
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    view = LeagueView()
    try:
        view.show_models(league.models(root))
        assert view.selected() is None
        view._table.selectRow(0)
        view._export_selected()
    finally:
        view.close()

    assert out.is_file(), "nothing was exported"
    # Byte for byte, and still a policy this build would run.
    assert out.read_bytes() == (root / "aaaa" / league.POLICY_NAME).read_bytes()
    assert policy_format.read(out) == policy


def _a_policy(name: str) -> object:
    """The smallest thing `policy_format` and the game both accept."""
    import numpy as np
    from md import policy_format

    rng = np.random.default_rng(11)

    def normal(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.1).astype(np.float32)

    return policy_format.NativePolicy(
        schema=policy_format.SCHEMA,
        observation_size=6,
        action_count=4,
        architecture="mlp",
        tensors=(
            policy_format.Tensor("trunk.0.weight", (3, 6), normal(3, 6)),
            policy_format.Tensor("trunk.0.bias", (3,), normal(3)),
            policy_format.Tensor("trunk.2.weight", (3, 3), normal(3, 3)),
            policy_format.Tensor("trunk.2.bias", (3,), normal(3)),
            policy_format.Tensor("policy_head.weight", (4, 3), normal(4, 3)),
            policy_format.Tensor("policy_head.bias", (4,), normal(4)),
            policy_format.Tensor("value_head.weight", (1, 3), normal(1, 3)),
            policy_format.Tensor("value_head.bias", (1,), normal(1)),
        ),
        metadata={"display_name": name},
    )


def test_a_model_can_be_deleted_out_of_the_league(
    qt_app: object,  # noqa: ARG001 — the QApplication has to exist
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The way out of the league, and therefore out of the game's MODELS menu.

    There was no way out at all: models could be promoted and imported and never
    removed, so a league accumulated every experiment anybody ever tried and the
    game's menu listed all of them. Answered through the real confirmation,
    monkeypatched rather than clicked — a destructive action that a stray Enter
    could trigger is not evidence of anything.
    """
    from md import league, policy_format
    from md.ui.league import LeagueView
    from PySide6.QtWidgets import QMessageBox

    root = tmp_path / "models"
    for model_id, name in (("keeper", "Keeper"), ("goner", "Goner")):
        (root / model_id).mkdir(parents=True)
        policy_format.write(root / model_id / league.POLICY_NAME, _a_policy(name))
        (root / model_id / league.CARD_NAME).write_text(
            f'{{"display_name": "{name}"}}', encoding="utf-8"
        )
    # The view deletes through `md.paths`, exactly as the trainer does, so the
    # guard that refuses a model outside the league stays a real guard here.
    monkeypatch.setenv("MD_MODELS_DIR", str(root))
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    view = LeagueView()
    try:
        view.refresh()
        rows = {view._table.item(row, 0).text() for row in range(view._table.rowCount())}
        assert rows == {"Keeper", "Goner"}
        for row in range(view._table.rowCount()):
            if view._table.item(row, 0).text() == "Goner":
                view._table.selectRow(row)
        assert (view.selected() or league.Model(root, "", "")).name == "Goner"
        view._delete_selected()
        assert view._table.rowCount() == 1
    finally:
        view.close()

    assert not (root / "goner").exists(), "the model is still on disk"
    assert (root / "keeper" / league.POLICY_NAME).is_file(), "the wrong model went"


def test_importing_a_name_the_league_already_has_asks_before_replacing(
    qt_app: object,  # noqa: ARG001 — the QApplication has to exist
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two models with one name are two rows nobody can tell apart.

    So the second one stops and asks. The question itself is `ask_about_clash`
    and is patched here to answer *replace*; what this asserts is the wiring
    around it — that the answer reaches `md.league` and produces one entry
    rather than an `-2` nobody chose.
    """
    from md import league, policy_format
    from md.ui import league as league_ui
    from PySide6.QtWidgets import QFileDialog

    root = tmp_path / "models"
    root.mkdir(parents=True)
    shared = tmp_path / "shared.mdp"
    policy_format.write(shared, _a_policy("Amber Anvil"))

    monkeypatch.setenv("MD_MODELS_DIR", str(root))
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(shared), ""))
    )

    asked: list[str] = []

    def answer(_parent: object, existing: league.Model) -> league.Model:
        asked.append(existing.name)
        return existing

    monkeypatch.setattr(league_ui, "ask_about_clash", answer)

    view = league_ui.LeagueView()
    try:
        view._import_policy()  # the first one: no clash, nothing asked
        assert asked == []
        view._import_policy()  # the same file again
    finally:
        view.close()

    assert asked == ["Amber Anvil"], "the second import did not ask"
    installed = league.models(root)
    assert [model.model_id for model in installed] == ["amber-anvil"]
    assert not (root / "amber-anvil-2").exists(), "a duplicate name landed anyway"


def test_peeking_at_a_head_to_head_records_both_sides(
    qt_app: object,  # noqa: ARG001 — the QApplication has to exist
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The peek a running head-to-head offers is a *comparison*, not one half.

    The contest plays one contestant's whole block and then the other's, so at
    the moment somebody asks to see a seed, only one side of it has been
    computed. Both are recorded here — the same policies on the same seed are
    the same episodes, so what opens is what is being scored.
    """
    from md import league, policy_format, tournament
    from md.ui.league import ContestDialog

    root = tmp_path / "models"
    monkeypatch.setenv("MD_MODELS_DIR", str(root))
    monkeypatch.setenv("MD_RUNS_DIR", str(tmp_path / "runs"))
    bundled = Path(__file__).resolve().parents[3] / "models" / "pretrained.mdp"
    if not bundled.is_file():
        pytest.skip("this checkout ships no bundled model to contest with")
    for name in ("Alpha", "Bravo"):
        league.import_policy(bundled, name)
    left, right = league.models(root)[1], league.models(root)[0]

    # A short protocol: what is under test is that two recordings arrive, not
    # how long an episode is.
    brief = tournament.Protocol(
        seed_split=tournament.benchmark.CANONICAL_SPLIT,
        seed_offset=tournament.benchmark.CANONICAL_SEED_OFFSET,
        seed_count=1,
        frame_skip=4,
        max_ticks=600,
    )
    monkeypatch.setattr(ContestDialog, "_protocol", lambda _self: brief)

    seed = brief.seeds()[0]
    opened: list[tuple[Path, Path]] = []
    dialog = ContestDialog(left, right)
    dialog.peek_pair.connect(lambda a, b: opened.append((a, b)))
    try:
        dialog._playing = (left, seed)  # noqa: SLF001 — what a progress report sets
        dialog._peek()  # noqa: SLF001 — standing in for the button press
        recorder = dialog._recorder  # noqa: SLF001
        assert recorder is not None
        assert recorder.wait(120_000), "recording the pair never finished"
        # The thread's `done` signal is delivered on the event loop, not in it.
        from PySide6.QtCore import QCoreApplication

        QCoreApplication.processEvents()
    finally:
        dialog.close()

    assert len(opened) == 1, "the split screen was never asked for"
    left_side, right_side = opened[0]
    for side in (left_side, right_side):
        assert side.is_file(), f"{side} was not recorded"
        assert side.stat().st_size > 0
    assert left_side != right_side
    # The seed is in each recording's header, and `MatchPlayer::pair` refuses two
    # of different seeds — `test_match.py` drives the game itself. What is
    # asserted here is the trainer's half: two episodes of the seed in flight,
    # written and handed on, without the contest's own thread being touched.
    assert seed.to_bytes(8, "little") in left_side.read_bytes()[:64]
    assert seed.to_bytes(8, "little") in right_side.read_bytes()[:64]
    assert policy_format.read(left.policy).observation_size > 0
