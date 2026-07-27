# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The Wayland workaround in `GameWindow::event`, held to claims that can fail.

The game releases its Vulkan instance when it receives `Close`. One line, and
without it `QVulkanWindow` cannot survive Qt's own window teardown on Wayland —
which is the default session on current KDE and GNOME, so it is most of the
desktop. The defect is Qt's (QTBUG-123214, reported 2024-03-12, still untriaged,
`QWindowPrivate::destroy()` unchanged through the dev branch).

A workaround with no test decays into folklore: nobody dares remove it and
nobody can say what it does. So three things are asserted separately, and each
can be wrong on its own.

* `test_qt_is_still_broken` — the cause exists. A bare `QVulkanWindow` with no
  line of this project still dies. When Qt is fixed this fails and says to
  delete the workaround.
* `test_the_workaround_works_on_its_own` — the effect is the workaround's. The
  same bare window, with only that one line added, survives. This is what
  separates a fix from a coincidence.
* `test_the_game_exits_cleanly_on_wayland` — the two meet in the real binary.

None of this can be observed without a compositor, so all three skip elsewhere;
CI is in that state and Jens's desktop is not.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from .harness import PROJECT_ROOT, app_binary, app_environ, needs_app

pytestmark = [pytest.mark.e2e, needs_app]

#: How many runs stand behind a claim of "does not crash". The failure being
#: guarded against is intermittent by nature, and one clean run is thin evidence
#: for a negative. Kept small because each run is about a second.
RUNS = 5


def _witness() -> Path | None:
    """The bare-QVulkanWindow witness, preferring a build without sanitizers.

    Release first, unlike the rest of this suite: the debug build carries
    AddressSanitizer, whose quarantine keeps the freed block mapped and turns
    the crash into a pass. A sanitised witness can only ever skip.
    """
    for build in ("build/release/app", "build/debug/app"):
        candidate = PROJECT_ROOT / build / "md_wayland_teardown"
        if candidate.exists():
            return candidate
    return None


def _wayland_env(tmp_path: Path) -> dict[str, str]:
    env = app_environ(tmp_path)
    env["QT_QPA_PLATFORM"] = "wayland"
    # LeakSanitizer would turn every run into exit 1 over libdbus allocations and
    # drown out the only number these tests read.
    env["ASAN_OPTIONS"] = f"{env.get('ASAN_OPTIONS', '')}:detect_leaks=0".lstrip(":")
    return env


def _require_wayland() -> None:
    if not os.environ.get("WAYLAND_DISPLAY"):
        # Not a failure, and not something a virtual X server can stand in for:
        # the defect is in Qt's Wayland platform plugin, so without a compositor
        # there is nothing to observe either way.
        pytest.skip("no Wayland session — this can only be observed under a compositor")


def _run_witness(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    witness = _witness()
    if witness is None:
        pytest.skip("md_wayland_teardown not built")
    _require_wayland()
    result = subprocess.run(
        [str(witness), *args],
        capture_output=True,
        text=True,
        timeout=120.0,
        env=_wayland_env(tmp_path),
        check=False,
    )
    reported = next(
        (json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")),
        None,
    )
    assert reported is not None, (
        f"the witness never reached a rendered frame, so it says nothing about "
        f"teardown:\n{result.stdout}\n{result.stderr}"
    )
    if reported.get("sanitized"):
        # See `_witness`: an instrumented run that survives proves nothing.
        pytest.skip(
            "built with AddressSanitizer, whose quarantine masks the use-after-free; "
            "build without it to observe this"
        )
    return result


def test_qt_is_still_broken(tmp_path: Path) -> None:
    """Fails the day Qt is fixed, which is the only time it should."""
    result = _run_witness([], tmp_path)
    # Negative on POSIX means killed by a signal; SIGSEGV is 11.
    assert result.returncode < 0, (
        f"a bare QVulkanWindow survived Qt's teardown under Wayland "
        f"(exit {result.returncode}) — QTBUG-123214 appears fixed in this Qt. "
        f"Delete the Close handling in GameWindow::event and this file with it.\n"
        f"{result.stdout}\n{result.stderr}"
    )


def test_the_workaround_works_on_its_own(tmp_path: Path) -> None:
    """The same window, one line added, survives — with the game held out of it."""
    for attempt in range(RUNS):
        result = _run_witness(["--detach"], tmp_path)
        assert result.returncode == 0, (
            f"releasing the Vulkan instance on Close did not save the bare "
            f"witness on run {attempt + 1} (exit {result.returncode}) — the "
            f"workaround in GameWindow::event rests on this and is no longer "
            f"earning its place.\n{result.stdout}\n{result.stderr}"
        )


def test_the_game_exits_cleanly_on_wayland(tmp_path: Path) -> None:
    """The claim that actually matters to whoever installs this.

    Deliberately the real binary and the real platform: the witness can only
    show that the technique works, not that the game still applies it.
    """
    _require_wayland()
    binary = app_binary()
    assert binary is not None, "needs_app should have skipped this"

    for attempt in range(RUNS):
        result = subprocess.run(
            [str(binary), "--play", "--frames", "90", "--report"],
            capture_output=True,
            text=True,
            timeout=120.0,
            env=_wayland_env(tmp_path),
            check=False,
        )
        assert result.returncode == 0, (
            f"the game did not exit cleanly on native Wayland on run "
            f"{attempt + 1} (exit {result.returncode}).\n{result.stdout}\n{result.stderr}"
        )
