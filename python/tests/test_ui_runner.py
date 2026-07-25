# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for launching the game from the console — without launching anything.

``md.ui.runner`` takes the spawn function as an argument for exactly this: what
matters is the command line it builds and where it looks for the binary, and
neither needs a window to check.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from md.ui.runner import AppNotFound, ReplayLauncher, app_binary, launch_environ

EXE = ".exe" if os.name == "nt" else ""


class FakeProcess:
    """A child that is alive until told otherwise."""

    def __init__(self) -> None:
        self.status: int | None = None

    def poll(self) -> int | None:
        return self.status


class FakeSpawn:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Path, dict[str, str]]] = []
        self.processes: list[FakeProcess] = []

    def __call__(self, command: list[str], cwd: Path, env: dict[str, str]) -> FakeProcess:
        self.calls.append((command, cwd, dict(env)))
        process = FakeProcess()
        self.processes.append(process)
        return process


def _build_app(root: Path, preset: str = "release") -> Path:
    binary = root / "build" / preset / "app" / f"md_app{EXE}"
    binary.parent.mkdir(parents=True)
    binary.write_text("", encoding="utf-8")
    return binary


def test_no_build_anywhere_is_none_not_a_guess(tmp_path: Path) -> None:
    assert app_binary({"PATH": str(tmp_path)}, root=tmp_path) is None


def test_the_release_build_wins_over_the_debug_one(tmp_path: Path) -> None:
    _build_app(tmp_path, "debug")
    release = _build_app(tmp_path, "release")
    assert app_binary({}, root=tmp_path) == release


def test_a_debug_build_is_used_when_that_is_all_there_is(tmp_path: Path) -> None:
    debug = _build_app(tmp_path, "debug")
    assert app_binary({}, root=tmp_path) == debug


def test_md_app_overrides_the_search(tmp_path: Path) -> None:
    _build_app(tmp_path, "release")
    installed = tmp_path / "elsewhere" / f"md_app{EXE}"
    installed.parent.mkdir()
    installed.write_text("", encoding="utf-8")
    assert app_binary({"MD_APP": str(installed)}, root=tmp_path) == installed


def test_an_md_app_pointing_at_nothing_is_not_silently_ignored(tmp_path: Path) -> None:
    # Falling back to the build directory would hide the typo; nothing found says
    # so, and the console explains how to fix it.
    _build_app(tmp_path, "release")
    assert app_binary({"MD_APP": str(tmp_path / "missing")}, root=tmp_path) is None


def test_a_system_install_is_found_on_path(tmp_path: Path) -> None:
    # What `poe deb` leaves behind: no build tree, md_app on PATH.
    installed = tmp_path / "bin" / f"md_app{EXE}"
    installed.parent.mkdir()
    installed.write_text("", encoding="utf-8")
    installed.chmod(0o755)
    found = app_binary({"PATH": str(installed.parent)}, root=tmp_path / "no-checkout")
    # `which` reports the name as PATHEXT spells it, which on Windows is MD_APP.EXE.
    assert found is not None and found.parent == installed.parent


def test_replaying_builds_the_command_the_app_expects(tmp_path: Path) -> None:
    binary = _build_app(tmp_path)
    spawn = FakeSpawn()
    launcher = ReplayLauncher(root=tmp_path, environ={}, spawn=spawn)

    launcher.launch(tmp_path / "runs" / "update-00025.mdr")

    command, cwd, _ = spawn.calls[0]
    assert command == [str(binary), "--replay", str(tmp_path / "runs" / "update-00025.mdr")]
    assert cwd == tmp_path


def test_an_unbuilt_game_says_how_to_build_it(tmp_path: Path) -> None:
    launcher = ReplayLauncher(root=tmp_path, environ={"PATH": str(tmp_path)}, spawn=FakeSpawn())
    with pytest.raises(AppNotFound, match="cmake --build"):
        launcher.launch(tmp_path / "update-00025.mdr")


def test_finished_windows_stop_being_counted(tmp_path: Path) -> None:
    _build_app(tmp_path)
    spawn = FakeSpawn()
    launcher = ReplayLauncher(root=tmp_path, environ={}, spawn=spawn)

    launcher.launch(tmp_path / "a.mdr")
    launcher.launch(tmp_path / "b.mdr")
    assert launcher.running == 2

    spawn.processes[0].status = 0  # the player closed that window
    assert launcher.running == 1


def test_linux_gets_an_x11_window_so_it_can_be_screenshot() -> None:
    env = launch_environ({"PATH": "/usr/bin"}, platform="linux")
    assert env["QT_QPA_PLATFORM"] == "xcb"
    # …unless the caller has already chosen, e.g. a Wayland-only session.
    chosen = launch_environ({"QT_QPA_PLATFORM": "wayland"}, platform="linux")
    assert chosen["QT_QPA_PLATFORM"] == "wayland"


def test_windows_puts_the_msys2_qt_dlls_back_on_path(tmp_path: Path) -> None:
    # The MinGW build finds Qt on PATH, which is set in the CLANG64 shell and
    # nowhere else — including the native interpreter the console runs in.
    (tmp_path / "clang64" / "bin").mkdir(parents=True)
    env = launch_environ({"MSYS2_ROOT": str(tmp_path), "PATH": "C:/Windows"}, platform="win32")
    assert env["PATH"].startswith(str(tmp_path / "clang64" / "bin"))
    assert env["PATH"].endswith("C:/Windows")


def test_no_msys2_no_change(tmp_path: Path) -> None:
    env = launch_environ(
        {"MSYS2_ROOT": str(tmp_path / "nope"), "PATH": "C:/Windows"}, platform="win32"
    )
    assert env["PATH"] == "C:/Windows"
