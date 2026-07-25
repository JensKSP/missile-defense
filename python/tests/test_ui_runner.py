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
import sys
from pathlib import Path

import pytest
from md.ui import runner
from md.ui.runner import (
    PACKAGE_PATH,
    AppNotFound,
    ReplayLauncher,
    TrainingRun,
    app_binary,
    can_train,
    launch_environ,
    training_environ,
    training_python,
)

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
    """A fake build, in the layout *this* machine's build would produce.

    The tests below are about the search order — release before debug, MD_APP
    over both, PATH last — which is the same everywhere. They ask app_binary for
    this platform's answer, so the fake has to be this platform's shape, exactly
    as `EXE` already does for Windows. The layouts themselves are pinned by the
    tests that name a platform outright.
    """
    app_dir = root / "build" / preset / "app"
    if sys.platform == "darwin":
        binary = app_dir / "md_app.app" / "Contents" / "MacOS" / "md_app"
    else:
        binary = app_dir / f"md_app{EXE}"
    binary.parent.mkdir(parents=True)
    binary.write_text("", encoding="utf-8")
    return binary


def _build_bundle(root: Path, preset: str = "release") -> Path:
    """What a macOS build leaves behind: the executable inside the .app bundle.

    Spelled out rather than reusing the runner's own helper — a test that shares
    the implementation's idea of the layout cannot catch it being wrong.
    """
    binary = root / "build" / preset / "app" / "md_app.app" / "Contents" / "MacOS" / "md_app"
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


def test_the_debian_package_installs_it_under_another_name(tmp_path: Path) -> None:
    # /usr/games/missile-defense, not md_app: looking only for the build's name
    # means the console can never open a replay on a machine with the .deb.
    installed = tmp_path / "games" / f"missile-defense{EXE}"
    installed.parent.mkdir()
    installed.write_text("", encoding="utf-8")
    installed.chmod(0o755)
    found = app_binary({"PATH": str(installed.parent)}, root=tmp_path / "no-checkout")
    assert found is not None and found.parent == installed.parent


def test_a_macos_build_is_found_inside_the_app_bundle(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    assert app_binary({}, root=tmp_path, platform="darwin") == bundle


def test_the_bundle_layout_is_not_looked_for_off_macos(tmp_path: Path) -> None:
    # The bundle and the flat executable are different paths. Getting this branch
    # backwards would mean a Linux console ignoring the build sitting next to it.
    _build_bundle(tmp_path)
    assert app_binary({"PATH": str(tmp_path)}, root=tmp_path, platform="linux") is None


def test_a_macos_disk_image_install_is_found_in_applications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Drag-to-Applications is the whole install story on macOS, and nothing inside
    # a bundle is ever on PATH — so without this the console could never open a
    # replay on a machine that has the .dmg but no checkout.
    installed = tmp_path / "md_app.app" / "Contents" / "MacOS" / "md_app"
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner, "MACOS_APP_PATHS", (str(tmp_path),))
    found = app_binary({"PATH": str(tmp_path)}, root=tmp_path / "no-checkout", platform="darwin")
    assert found == installed


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


def test_a_run_starts_in_this_interpreter_unless_told_otherwise() -> None:
    assert training_python({}) == sys.executable
    assert training_python({"MD_PYTHON": "C:/python312/python.exe"}) == "C:/python312/python.exe"


def test_a_spawned_run_can_import_md_without_it_being_installed() -> None:
    # The console is run from a checkout, and so is the trainer it starts.
    env = training_environ({"PYTHONPATH": "/somewhere/else"})
    assert env["PYTHONPATH"].startswith(str(PACKAGE_PATH))
    assert "/somewhere/else" in env["PYTHONPATH"]
    # ...and it is not added twice when the console was itself started that way.
    once = training_environ({"PYTHONPATH": str(PACKAGE_PATH)})
    assert once["PYTHONPATH"] == str(PACKAGE_PATH)


def test_whether_this_interpreter_could_train_at_all() -> None:
    # True where torch is installed, False in the MSYS2 one that runs the gate —
    # either way the answer must not *import* torch (see test_ui_boundary.py).
    assert isinstance(can_train(), bool)
    assert "torch" not in sys.modules


def test_a_run_is_read_line_by_line_and_reports_how_it_ended(tmp_path: Path) -> None:
    # A real child process, because the point of this class is the plumbing:
    # a reader thread, a buffer, and an exit code the UI can notice.
    run = TrainingRun(
        [sys.executable, "-u", "-c", "print('update 1'); print('update 2')"],
        cwd=tmp_path,
    )
    lines = list(run.wait_for_output())
    assert lines == ["update 1", "update 2"]
    assert run.drain() == []  # drained once, not kept forever
    assert run.finished
    assert run.exit_code() == 0


def test_a_run_that_dies_says_so_with_its_own_words(tmp_path: Path) -> None:
    run = TrainingRun(
        [sys.executable, "-u", "-c", "raise SystemExit('no torch here')"],
        cwd=tmp_path,
    )
    lines = list(run.wait_for_output())
    assert "no torch here" in "\n".join(lines)  # stderr shares the stream
    assert run.exit_code() == 1


def test_no_msys2_no_change(tmp_path: Path) -> None:
    env = launch_environ(
        {"MSYS2_ROOT": str(tmp_path / "nope"), "PATH": "C:/Windows"}, platform="win32"
    )
    assert env["PATH"] == "C:/Windows"
