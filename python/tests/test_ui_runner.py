# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for launching the game from the console — without launching anything.

``md.ui.runner`` takes the spawn function as an argument for exactly this: what
matters is the command line it builds and where it looks for the binary, and
neither needs a window to check.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import pytest
from md import runtime
from md.ui import runner
from md.ui.runner import (
    PACKAGE_PATH,
    AppNotFound,
    ReplayLauncher,
    TrainingRun,
    app_binary,
    can_train,
    console_executable,
    find_interpreter,
    launch_environ,
    training_environ,
    training_python,
)

EXE = ".exe" if os.name == "nt" else ""


def _stub_ready_runtime(tmp_path: Path) -> runtime.Runtime:
    """A store with a runtime installed, without installing one.

    It goes through the real :meth:`Runtime.install`, because the state that
    makes a runtime *ready* — the signed manifest and the current marker — is
    exactly what this is testing the reading of. Only the three commands are
    faked.
    """
    root = tmp_path / "runtime"

    def fake(command: list[str], emit: Callable[[str], None]) -> int:
        if "venv" in command:
            interpreter = runtime.venv_python(Path(command[-1]), platform=sys.platform)
            interpreter.parent.mkdir(parents=True, exist_ok=True)
            interpreter.write_text("")
        elif runtime.HEALTH_SCRIPT in command:
            emit(json.dumps({"native": True, "torch": "2.13.0", "device": "cuda"}))
        return 0

    store = runtime.Runtime(root, runner=fake)
    store.install(runtime.recommend(runtime.SystemInfo.here(), [], root=root))
    return store


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


def test_every_way_of_opening_the_game_builds_its_own_command(tmp_path: Path) -> None:
    # Three flags, one spawn. They were three copies of the same twenty lines,
    # and the "is it built?" message — the part a person actually reads — is
    # exactly the kind of thing that drifts between copies.
    binary = _build_app(tmp_path)
    spawn = FakeSpawn()
    launcher = ReplayLauncher(root=tmp_path, environ={}, spawn=spawn)

    launcher.launch_model(tmp_path / "models" / "aaaa" / "policy.mdp")
    launcher.launch_match(tmp_path / "matches" / "a-b" / "match.json")

    assert [call[0][1] for call in spawn.calls] == ["--watch-model", "--match"]
    assert spawn.calls[-1][0] == [
        str(binary),
        "--match",
        str(tmp_path / "matches" / "a-b" / "match.json"),
    ]


def test_an_unbuilt_game_says_how_to_build_it(tmp_path: Path) -> None:
    launcher = ReplayLauncher(root=tmp_path, environ={"PATH": str(tmp_path)}, spawn=FakeSpawn())
    # Every entry point, not just the first: an unbuilt game is the normal state
    # of a console-only install, and each of these is somebody's first click.
    with pytest.raises(AppNotFound, match="cmake --build"):
        launcher.launch_match(tmp_path / "match.json")
    with pytest.raises(AppNotFound, match="cmake --build"):
        launcher.launch_model(tmp_path / "policy.mdp")


def test_an_unbuilt_game_says_how_to_build_it_for_replays(tmp_path: Path) -> None:
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


def _empty_store(tmp_path: Path) -> runtime.Runtime:
    """A runtime store with nothing installed, so these tests do not depend on
    whether the machine running them has one."""
    return runtime.Runtime(tmp_path / "runtime")


def test_a_run_starts_in_this_interpreter_unless_told_otherwise(tmp_path: Path) -> None:
    store = _empty_store(tmp_path)
    assert training_python({}, store=store) == sys.executable
    assert (
        training_python({"MD_PYTHON": "C:/python312/python.exe"}, store=store)
        == "C:/python312/python.exe"
    )


def test_an_explicit_interpreter_is_not_second_guessed(tmp_path: Path) -> None:
    # MD_PYTHON is someone saying which build to train with — the Windows
    # split-interpreter case. It wins over a managed runtime and is not probed.
    found = find_interpreter({"MD_PYTHON": "/opt/py/bin/python"}, store=_empty_store(tmp_path))
    assert found is not None
    assert found.path == "/opt/py/bin/python"
    assert found.source == "MD_PYTHON"


def test_a_managed_runtime_is_what_a_packaged_user_trains_with(tmp_path: Path) -> None:
    # The whole point of md.runtime: an installed copy with no torch anywhere
    # can still start a run, and Start does not depend on how the console itself
    # was installed.
    store = _stub_ready_runtime(tmp_path)
    found = find_interpreter({}, store=store)
    assert found is not None
    assert found.path == str(store.python())
    assert "installed" in found.source
    assert can_train({}, store=store)


def test_a_machine_with_no_runtime_and_no_torch_cannot_train(tmp_path: Path) -> None:
    # The packaged console before setup: no managed runtime, and torch is not
    # importable from the interpreter it is running in either.
    store = _empty_store(tmp_path)
    with mock.patch.object(runner.importlib.util, "find_spec", return_value=None):
        assert not can_train({}, store=store)
        assert find_interpreter({}, store=store) is None


def test_browsing_and_replay_do_not_depend_on_being_able_to_train(tmp_path: Path) -> None:
    # A console attached to a directory synced from the training box must stay
    # fully useful: only Start is gated on an interpreter.
    root = tmp_path / "checkout"
    binary = _build_app(root)
    launcher = ReplayLauncher(root=root, environ={}, spawn=(spawn := FakeSpawn()))
    with mock.patch.object(runner.importlib.util, "find_spec", return_value=None):
        assert not can_train({}, store=_empty_store(tmp_path))
        launcher.launch(tmp_path / "episode.mdr")
    assert spawn.calls[0][0][0] == str(binary)


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
    # Guarded against a torch already imported by an earlier test in the same
    # process (e.g. test_ppo, which runs where torch is installed): the check is
    # that can_train adds nothing, not that the whole session is torch-free.
    had_torch = "torch" in sys.modules
    assert isinstance(can_train(), bool)
    if not had_torch:
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


# ---- finding the console -----------------------------------------------------
# The game offers TRAIN AI only when this lookup resolves, so its answer *is* the
# difference between the two products: a game-only install must find nothing.


def _install_console(root: Path, name: str = "md-console") -> Path:
    console = root / f"{name}{EXE}"
    console.write_text("", encoding="utf-8")
    console.chmod(0o755)
    return console


def test_a_game_only_install_finds_no_console(tmp_path: Path) -> None:
    # The negative half, and the one that matters most: nothing on PATH, no
    # checkout. If this ever starts finding something, the game has stopped being
    # installable on its own.
    assert console_executable({"PATH": str(tmp_path)}, root=tmp_path / "nowhere") is None


def test_an_installed_console_is_found_on_path(tmp_path: Path) -> None:
    console = _install_console(tmp_path)
    found = console_executable({"PATH": str(tmp_path)}, root=tmp_path / "no-checkout")
    assert found == console


def test_md_console_overrides_the_search(tmp_path: Path) -> None:
    # The same escape hatch MD_APP is for the game: someone who knows where it is
    # is not second-guessed.
    elsewhere = _install_console(tmp_path, "console-somewhere")
    found = console_executable({"MD_CONSOLE": str(elsewhere), "PATH": ""}, root=tmp_path)
    assert found == elsewhere


def test_an_md_console_pointing_at_nothing_is_not_silently_ignored(tmp_path: Path) -> None:
    # Falling back to a search would launch a *different* console than the one
    # that was named, which is worse than reporting none.
    assert console_executable({"MD_CONSOLE": str(tmp_path / "missing")}, root=tmp_path) is None


def test_a_checkout_can_run_the_console_it_contains(tmp_path: Path) -> None:
    # A developer has no md-console on PATH but does have the package, and the
    # game should still offer TRAIN AI there — the checkout is an install too.
    package = tmp_path / "python" / "md" / "ui"
    package.mkdir(parents=True)
    (package / "__main__.py").write_text("", encoding="utf-8")
    found = console_executable({"PATH": ""}, root=tmp_path)
    assert found is not None
    assert found.name.startswith("python")
