# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Cleaning up, archiving and restoring a real run through the real dialogs.

`md.archive` has been tested since Task 9 and none of it was reachable from the
console. What is asserted here is the part unit tests cannot reach: that the
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

    # And nothing that matters went with it: the console still draws this run.
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
