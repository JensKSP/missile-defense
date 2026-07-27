# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The xcb fallback in `app/main.cpp`, held to a claim that can be wrong.

The game asks for the xcb platform when it finds itself on a Wayland session.
That is a workaround for a defect in Qt, and the honest form of a workaround is
one that fails the day its cause is gone rather than quietly outliving it.

The claim being tested is narrow and does not mention this project: *a bare
`QVulkanWindow` cannot survive Qt's own window teardown under Wayland.* If that
becomes false, `app/main.cpp` should stop steering Wayland users to XWayland —
which costs them tearing, NVIDIA implementing no implicit sync — and this file
is where they find out.

Upstream: QTBUG-123214, reported 2024-03-12 against Qt 6.6.2, still untriaged
with no fix version, and `QWindowPrivate::destroy()` is unchanged through the
dev branch. So the expected outcome today is a crash.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from .harness import PROJECT_ROOT, app_environ, needs_app

pytestmark = [pytest.mark.e2e, needs_app]


def _witness() -> Path | None:
    # Release first, unlike the rest of this suite: the debug build carries
    # AddressSanitizer, and a sanitised witness can only ever skip.
    for build in ("build/release/app", "build/debug/app"):
        candidate = PROJECT_ROOT / build / "md_wayland_teardown"
        if candidate.exists():
            return candidate
    return None


def test_the_xcb_fallback_is_still_necessary(tmp_path: Path) -> None:
    witness = _witness()
    if witness is None:
        pytest.skip("md_wayland_teardown not built")
    if not os.environ.get("WAYLAND_DISPLAY"):
        # Not a failure and not something a virtual X server can stand in for:
        # the defect is in the Wayland platform plugin, so without a compositor
        # there is nothing to observe. CI is in this state; Jens's desktop is not.
        pytest.skip("no Wayland session — this can only be observed under a compositor")

    env = app_environ(tmp_path)
    env["QT_QPA_PLATFORM"] = "wayland"
    # LeakSanitizer would turn every run into exit 1 over libdbus allocations and
    # drown out the only number this test reads.
    env["ASAN_OPTIONS"] = f"{env.get('ASAN_OPTIONS', '')}:detect_leaks=0".lstrip(":")

    result = subprocess.run(
        [str(witness)], capture_output=True, text=True, timeout=120.0, env=env, check=False
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
        # ASan keeps freed blocks mapped in quarantine, so the stale read that
        # ends this program on a plain build succeeds under instrumentation. A
        # sanitised run surviving is not evidence of anything.
        pytest.skip(
            "built with AddressSanitizer, whose quarantine masks the use-after-free; "
            "configure a build without it to observe this"
        )

    # Negative on POSIX means killed by a signal; SIGSEGV is 11.
    assert result.returncode < 0, (
        f"a bare QVulkanWindow survived Qt's teardown under Wayland "
        f"(exit {result.returncode}) — QTBUG-123214 appears fixed in this Qt. "
        f"Drop the xcb fallback in app/main.cpp and delete this test with it.\n"
        f"{result.stdout}\n{result.stderr}"
    )
