# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Starting the game from the console. No Qt in here, so pytest can drive it.

Phase 1 of the console is read-only, and this is the one thing it *does*: hand a
recording to ``md_app --replay`` so a double-click in the episode list opens the
episode. Spawning the game is also the shape the later phases need — training
runs as a subprocess for the same reason (docs/ROADMAP.md, M8), and that keeps a
UI crash from taking the run with it.

Finding the binary is the fiddly part, so it is deliberate rather than a fixed
path: an explicit ``MD_APP``, then this checkout's build directories, then
``PATH`` for a system install from the ``.deb``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

#: <root>/python/md/ui/runner.py — the checkout, when the console runs from one.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Where a local build puts the game, best build first.
BUILD_PATHS = ("build/release/app", "build/debug/app")

#: An MSYS2 build links against Qt in the CLANG64 prefix and finds it on PATH,
#: which is there in the CLANG64 shell and absent everywhere else — including the
#: native interpreter the console is likely started from. Adding it back turns a
#: silent "nothing happened" into a window (docs/WINDOWS.md).
MSYS2_BIN = "clang64/bin"


class AppNotFound(RuntimeError):
    """The game binary could not be located — the message says how to fix it."""


class Process(Protocol):
    """Just enough of ``subprocess.Popen`` to know whether it is still up."""

    def poll(self) -> int | None: ...


#: Injected so tests can watch what would be launched without launching it.
Spawn = Callable[[list[str], Path, Mapping[str, str]], Process]


def _spawn(command: list[str], cwd: Path, env: Mapping[str, str]) -> Process:
    return subprocess.Popen(command, cwd=str(cwd), env=dict(env))


def app_binary(
    environ: Mapping[str, str] | None = None, *, root: Path | None = None
) -> Path | None:
    """Locate ``md_app``, or return ``None`` if this machine has no build of it."""
    env = os.environ if environ is None else environ
    root = PROJECT_ROOT if root is None else root
    exe = ".exe" if os.name == "nt" else ""

    override = env.get("MD_APP")
    if override:
        candidate = Path(override)
        return candidate if candidate.exists() else None
    for build in BUILD_PATHS:
        candidate = root / build / f"md_app{exe}"
        if candidate.exists():
            return candidate
    found = shutil.which("md_app", path=env.get("PATH"))
    return Path(found) if found else None


def launch_environ(
    environ: Mapping[str, str] | None = None, *, platform: str = sys.platform
) -> dict[str, str]:
    """The environment the game is started in — the two platform quirks, once.

    ``platform`` is a parameter so both branches are testable from either OS;
    a quirk that only one machine can check is a quirk that rots.
    """
    env = dict(os.environ if environ is None else environ)
    if platform == "linux":
        # An X11 window, so the result is screenshot-able — as tools/run_app does.
        env.setdefault("QT_QPA_PLATFORM", "xcb")
    elif platform == "win32":
        msys = Path(env.get("MSYS2_ROOT", "C:/msys64")) / MSYS2_BIN
        if msys.is_dir():
            env["PATH"] = f"{msys}{os.pathsep}{env.get('PATH', '')}"
    return env


class ReplayLauncher:
    """Opens recordings in the game, and keeps track of the windows it opened.

    The handles are kept only so finished children are reaped; the console never
    waits on the game, and closing the console leaves it running.
    """

    def __init__(
        self,
        *,
        root: Path | None = None,
        environ: Mapping[str, str] | None = None,
        spawn: Spawn = _spawn,
    ) -> None:
        self._root = PROJECT_ROOT if root is None else root
        self._environ = environ
        self._spawn = spawn
        self._children: list[Process] = []

    def launch(self, recording: Path) -> None:
        """Play ``recording`` in the game. Raises :class:`AppNotFound` if unbuilt."""
        binary = app_binary(self._environ, root=self._root)
        if binary is None:
            raise AppNotFound(
                "the game is not built here — run `cmake --build --preset release`, "
                "or point MD_APP at an md_app binary"
            )
        self._children = [child for child in self._children if child.poll() is None]
        self._children.append(
            self._spawn(
                [str(binary), "--replay", str(recording)],
                self._root,
                launch_environ(self._environ),
            )
        )

    @property
    def running(self) -> int:
        """How many game windows this console has open."""
        self._children = [child for child in self._children if child.poll() is None]
        return len(self._children)
