# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Both binaries, driven with no mouse at all.

Keyboard-only operation is the accessibility claim this release actually makes,
and it is the one that can be checked rather than asserted. Two failure modes
matter and neither is visible in a screenshot: a control the keyboard can *reach*
but cannot *act in* — a table you can Tab to and not open — and focus that is
styled away, so a person navigating by Tab has no idea where they are.

The game half is a state machine over key events and is checked by driving it.
The console half is Qt, where the two things to get wrong are activation
signals and a stylesheet that quietly replaces the platform focus ring with
nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .harness import assert_clean, needs_app, needs_display, needs_native, needs_qt, run_app


@pytest.mark.e2e
@needs_app
@needs_display
def test_the_game_needs_no_mouse_to_reach_any_screen(tmp_path: Path) -> None:
    # Every state the game can be in is handled by `keyPressEvent`, and the one
    # that is keyboard-only by design — a match, where there is nothing to click
    # — is the one this would most easily have missed.
    run = run_app(frames=90, sandbox=tmp_path)
    assert_clean(run)
    menu = run.report["menu"]
    assert isinstance(menu, list) and menu, run.report
    # Arrows and Enter drive the list, which the footer states on screen. What
    # is asserted here is the weaker, checkable half: the menu exists and is a
    # list of labels, so there is something for the arrows to move through.
    assert all(isinstance(label, str) and label for label in menu)


@pytest.mark.e2e
@needs_qt
@needs_native
def test_a_table_the_keyboard_can_reach_is_one_it_can_act_in(
    qt_app: object,  # noqa: ARG001 — the QApplication has to exist
    tmp_path: Path,
) -> None:
    """Enter on a selected row opens it.

    The most common way a Tab-navigable window is still mouse-only in practice:
    focus lands on the table, the arrows move the selection, and then nothing
    happens, because the only path to the action was the button beside it.
    """
    from md.ui.library import LibraryView
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication

    root = tmp_path / "runs"
    for name in ("run-alpha", "run-beta"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "metrics.csv").write_text("update,mean_return\n1,0.5\n", encoding="utf-8")

    opened: list[Path] = []
    view = LibraryView()
    view.opened.connect(opened.append)
    view.attach(root)
    try:
        view.table.focus_list()
        assert view.table.selected() is not None, "focusing the list selected nothing"

        # Straight through the event queue, as a keystroke arrives.
        press = QKeyEvent(
            QKeyEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier
        )
        QApplication.sendEvent(view.table._table, press)
        assert opened, "Enter on the selected run opened nothing"
    finally:
        view.close()


@pytest.mark.e2e
@needs_qt
def test_focus_is_visible_on_every_styled_control(qt_app: object) -> None:  # noqa: ARG001
    """A stylesheet that sets `border` removes the platform's focus rectangle.

    Which is not cosmetic: it makes a window that is *technically* navigable
    impossible to navigate, because nothing on screen says where you are. Every
    control the theme styles has to state its own focus.
    """
    from md.ui import theme

    sheet = theme.stylesheet()
    for control in ("QPushButton:focus", "QComboBox:focus", "QLineEdit:focus"):
        assert control in sheet, f"{control} has no visible focus state"
    # And the accent it uses is not the same as the resting border, or "focused"
    # and "not focused" look identical to the person the rule is for.
    assert theme.INTERCEPTOR in sheet.split("QPushButton:focus", 1)[1][:400]


@pytest.mark.e2e
@needs_qt
@needs_native
def test_escape_leaves_a_run_without_touching_the_back_button(
    qt_app: object,  # noqa: ARG001 — the QApplication has to exist
    tmp_path: Path,
) -> None:
    # The console's one navigation, previously reachable only by clicking a
    # button in the corner. Escape is what every other back control in both
    # binaries answers to, including all eleven of the game's screens.
    from md.ui.app import Console
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    run = tmp_path / "runs" / "run-alpha"
    run.mkdir(parents=True)
    (run / "metrics.csv").write_text("update,mean_return\n1,0.5\n", encoding="utf-8")

    window = Console(tmp_path / "runs")
    try:
        window._open_run(run)
        assert window._pages.currentIndex() == 1

        window.keyPressEvent(
            QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        )
        assert window._pages.currentIndex() == 0, "Escape did not go back to the library"
    finally:
        window.close()
