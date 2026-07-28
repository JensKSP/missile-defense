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
nobody can say what it does. So four things are asserted separately, and each
can be wrong on its own.

* `test_qt_is_still_broken` — the cause exists. A bare `QVulkanWindow` with no
  line of this project still dies. When Qt is fixed this fails and says to
  delete the workaround.
* `test_the_workaround_works_on_its_own` — the effect is the workaround's. The
  same bare window, with only that one line added, survives. This is what
  separates a fix from a coincidence.
* `test_the_orphaned_surface_can_be_reclaimed` — and it costs no leaked handle.
  Detaching leaves the `VkSurfaceKHR` with nobody to destroy it; destroying it
  by hand, after the base class has taken the platform window and the swapchain
  apart, still survives. Its own claim because the same call one step earlier
  crashes as reliably as no workaround at all.
* `test_the_game_exits_cleanly_on_wayland` — the three meet in the real binary.

None of this can be observed without a compositor, so all four skip elsewhere;
CI is in that state and Jens's desktop is not.

**And they are opt-in even where they can run.** There is no Xvfb for Wayland —
see `MD_E2E_WAYLAND` in the harness — so each of these puts a real window on the
screen of whoever is at the machine and takes the focus with it, twenty times
over a full run. A suite that does that to somebody mid-sentence is a suite they
stop running, so `poe test-app` skips them with the reason, and `poe
test-wayland` is the deliberate act of asking for them.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from .harness import PROJECT_ROOT, app_binary, app_environ, needs_app, needs_wayland

pytestmark = [pytest.mark.e2e, pytest.mark.wayland, needs_app, needs_wayland]

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


def _run_witness(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    witness = _witness()
    if witness is None:
        pytest.skip("md_wayland_teardown not built")
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


def test_the_orphaned_surface_can_be_reclaimed(tmp_path: Path) -> None:
    """The workaround need not cost a leaked `VkSurfaceKHR`, and does not.

    Two claims in one, because a run that skipped the call would otherwise look
    exactly like a run that made it and survived: the witness reports whether it
    actually destroyed the surface, and the exit status says whether doing so
    took the process down with it.
    """
    for attempt in range(RUNS):
        result = _run_witness(["--reclaim"], tmp_path)
        assert result.returncode == 0, (
            f"destroying the orphaned surface after teardown killed the witness "
            f"on run {attempt + 1} (exit {result.returncode}) — GameWindow::event "
            f"does this, so the game is crashing on Wayland too.\n"
            f"{result.stdout}\n{result.stderr}"
        )
        assert '"reclaimed": true' in result.stdout, (
            f"the witness survived without ever destroying the surface, so this "
            f"run says nothing about whether destroying it is safe.\n{result.stdout}"
        )


def test_the_game_exits_cleanly_on_wayland(tmp_path: Path) -> None:
    """The claim that actually matters to whoever installs this.

    Deliberately the real binary and the real platform: the witness can only
    show that the technique works, not that the game still applies it.

    **`--silent` is not optional here.** This is the one place in the suite that
    builds its own game command line rather than going through `run_app`, and it
    is therefore the one place that can lose the harness's promise that a test
    run does not come out of the speakers of whoever is at the machine
    (test_silence.py). It had lost it: five runs of real gameplay, audible, on
    the desktop the tests are running on.
    """
    binary = app_binary()
    assert binary is not None, "needs_app should have skipped this"

    for attempt in range(RUNS):
        result = subprocess.run(
            [str(binary), "--play", "--frames", "90", "--report", "--silent"],
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
        # The flag above, held to a claim that can fail rather than to a comment.
        # `audible` asks the engine, not the stored preference — see
        # test_silence.py, where the same check guards every other run.
        reported = next(
            (json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")),
            None,
        )
        assert reported is not None, f"no --report line from the game:\n{result.stdout}"
        assert reported["audible"] is False, (
            "the game played sound on the developer's own desktop — --silent went "
            "missing from this run's command line again"
        )
