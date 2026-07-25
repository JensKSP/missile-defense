# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Starting things from the console: the game, and a training run. No Qt in here.

Both are subprocesses, and that is the architecture rather than an implementation
detail (docs/ROADMAP.md, M8). Training saturates the CPU for hours; inside a Qt
event loop that is a frozen window, and a UI crash would take the run with it.
Out of process, the console can be closed, can crash, or can be opened on a
directory synced from another machine, and the run does not notice.

Nothing here kills a run, either. Stopping is a *request* — see :mod:`md.control`
— so the loop finishes its update and writes a final checkpoint. The console has
no way to take that away.

Finding the game binary is the fiddly part, so it is deliberate rather than a
fixed path: an explicit ``MD_APP``, then this checkout's build directories, then
``PATH`` for a system install from the ``.deb``.
"""

from __future__ import annotations

import collections
import importlib.util
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import IO, Protocol

#: <root>/python/md/ui/runner.py — the checkout, when the console runs from one.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: The directory ``md`` itself sits in, so a spawned run can import it without
#: the console having been installed.
PACKAGE_PATH = Path(__file__).resolve().parents[2]

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


# ---- starting a training run -------------------------------------------------


class PipedProcess(Protocol):
    """A child whose output the console reads."""

    stdout: IO[str] | None

    def poll(self) -> int | None: ...


#: Injected so tests can drive the plumbing without starting a trainer.
SpawnPiped = Callable[[list[str], Path, Mapping[str, str]], PipedProcess]


def _spawn_piped(command: list[str], cwd: Path, env: Mapping[str, str]) -> PipedProcess:
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # one stream: a traceback belongs in the log too
        text=True,
        encoding="utf-8",  # agreed with PYTHONIOENCODING below, not guessed
        errors="replace",
        bufsize=1,
    )


def training_python(environ: Mapping[str, str] | None = None) -> str:
    """Which interpreter a run is started with.

    The console's own by default — on a training machine that is the one with
    torch, since it is also the one that could import PySide6. ``MD_PYTHON``
    overrides it for the split-interpreter case.
    """
    env = os.environ if environ is None else environ
    return env.get("MD_PYTHON") or sys.executable


def can_train() -> bool:
    """Whether *this* interpreter could run a trainer at all.

    ``find_spec`` locates torch without importing it, which matters twice: the
    console must never pull torch in (a test asserts it), and a console watching
    a remote run from a laptop with no torch should say Start is unavailable
    rather than spawning something that dies with an ImportError.
    """
    return importlib.util.find_spec("torch") is not None


def training_environ(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The run's environment: this checkout's ``python/`` on the import path.

    Its output encoding is pinned too. A pipe on Windows otherwise gets the
    legacy code page, and the trainer's progress lines are full of em dashes and
    arrows — read back as UTF-8 those come out as mojibake in the log pane.
    """
    env = dict(os.environ if environ is None else environ)
    env["PYTHONIOENCODING"] = "utf-8"
    existing = env.get("PYTHONPATH", "")
    if str(PACKAGE_PATH) not in existing.split(os.pathsep):
        joined = f"{PACKAGE_PATH}{os.pathsep}{existing}" if existing else str(PACKAGE_PATH)
        env["PYTHONPATH"] = joined
    return env


class TrainingRun:
    """A training run this console started, and the lines it has printed.

    The output is drained by a reader thread into a bounded buffer, and the UI
    picks it up on its ordinary timer tick — so a chatty run cannot block the
    event loop, and a long one cannot grow the window's memory without bound.
    """

    def __init__(
        self,
        command: list[str],
        *,
        cwd: Path,
        environ: Mapping[str, str] | None = None,
        spawn: SpawnPiped = _spawn_piped,
        keep: int = 2000,
    ) -> None:
        self.command = command
        self._process = spawn(command, cwd, training_environ(environ))
        self._lines: collections.deque[str] = collections.deque(maxlen=keep)
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        stream = self._process.stdout
        if stream is None:
            return
        for line in stream:
            with self._lock:
                self._lines.append(line.rstrip("\n"))

    def drain(self) -> list[str]:
        """Every line printed since the last call."""
        with self._lock:
            lines = list(self._lines)
            self._lines.clear()
        return lines

    def exit_code(self) -> int | None:
        """``None`` while it is still going."""
        return self._process.poll()

    @property
    def finished(self) -> bool:
        return self.exit_code() is not None

    def wait_for_output(self, *, timeout: float = 5.0) -> Iterator[str]:
        """Drain until the process ends — for tests, never for the event loop.

        The output stream closing and the process being reaped are two events,
        and in that order: without the second wait, the last line can arrive
        while the exit code is still ``None``.
        """
        self._reader.join(timeout)
        deadline = time.monotonic() + timeout
        while self._process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        yield from self.drain()
