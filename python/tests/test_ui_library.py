# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The way from the run list to the game.

Everything else that opens the game opens it *on* something — a recording, a
promoted model, a match. There was no way to simply go and play, which is the
thing most of these runs exist to improve, so the library screen grew a `Play`
button and the launcher grew the one spawn that carries no target.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from missile_defense.runs.runner import AppNotFound, ReplayLauncher

try:  # the widgets, where PySide6 is installed — the trainer is optional
    from missile_defense.ui.library import LibraryView
    from PySide6.QtWidgets import QApplication
except ModuleNotFoundError as error:  # pragma: no cover - depends on the machine
    if error.name is None or not error.name.startswith("PySide6"):
        raise
    LibraryView = None  # type: ignore[assignment,misc,unused-ignore]
    QApplication = None  # type: ignore[assignment,misc,unused-ignore]

needs_qt = pytest.mark.skipif(LibraryView is None, reason="PySide6 is not installed")


class FakeProcess:
    """A child that never finishes, which is what a running game looks like."""

    def poll(self) -> int | None:
        return None


class FakeSpawn:
    """Records what would have been launched instead of launching it."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], cwd: Path, env: object) -> FakeProcess:
        del cwd, env
        self.calls.append(list(command))
        return FakeProcess()


def _build_app(root: Path) -> Path:
    """A release build of the game, as far as the launcher can tell."""
    directory = root / "build" / "release" / "app"
    directory.mkdir(parents=True)
    binary = directory / ("md_app.exe" if os.name == "nt" else "md_app")
    binary.write_text("", encoding="utf-8")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return binary


def test_playing_launches_the_game_with_nothing_to_open(tmp_path: Path) -> None:
    # The distinguishing property: no flag and no path. Every other entry point
    # appends a target, and one that slipped a stray argument in here would open
    # the game on something instead of on its menu.
    binary = _build_app(tmp_path)
    launcher = ReplayLauncher(root=tmp_path, environ={}, spawn=(spawn := FakeSpawn()))

    launcher.launch_game()

    assert spawn.calls == [[str(binary)]]


def test_playing_still_refuses_when_the_game_is_not_built(tmp_path: Path) -> None:
    # Same failure and the same sentence as every other way of opening it: the
    # button must not fail silently on a checkout that has never been built.
    launcher = ReplayLauncher(root=tmp_path, environ={}, spawn=FakeSpawn())
    with pytest.raises(AppNotFound, match="not built"):
        launcher.launch_game()


def test_a_launched_game_is_counted_so_a_second_one_is_not_started(tmp_path: Path) -> None:
    # What the button reads before deciding to launch. Stacking a duplicate
    # window on top of the one already open is the behaviour this prevents.
    _build_app(tmp_path)
    launcher = ReplayLauncher(root=tmp_path, environ={}, spawn=FakeSpawn())
    assert launcher.running == 0
    launcher.launch_game()
    assert launcher.running == 1


@pytest.fixture(scope="module")
def qt_app():  # type: ignore[no-untyped-def]
    if QApplication is None:  # pragma: no cover - skipped by needs_qt
        pytest.skip("PySide6 is not installed")
    yield QApplication.instance() or QApplication([])


@needs_qt
def test_the_library_offers_play_and_wires_it_up(qt_app: object) -> None:
    del qt_app
    assert LibraryView is not None  # guaranteed by `needs_qt`; stated for the checker
    pressed: list[bool] = []
    view = LibraryView(on_play=lambda: pressed.append(True))
    try:
        view._play.click()
        assert pressed == [True], "the Play button is not connected to anything"
    finally:
        view.deleteLater()


@needs_qt
def test_a_library_without_a_way_to_play_grows_no_button(qt_app: object) -> None:
    # The same shape as `on_new_run`: the screen is constructed without one in
    # tests and in the embedded views, and must not fail for the want of it.
    del qt_app
    assert LibraryView is not None  # guaranteed by `needs_qt`; stated for the checker
    view = LibraryView()
    try:
        assert not hasattr(view, "_play")
    finally:
        view.deleteLater()
