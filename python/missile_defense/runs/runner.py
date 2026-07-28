# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Starting things from the trainer: the game, and a training run. No Qt in here.

Both are subprocesses, and that is the architecture rather than an implementation
detail (docs/ROADMAP.md, M8). Training saturates the CPU for hours; inside a Qt
event loop that is a frozen window, and a UI crash would take the run with it.
Out of process, the trainer can be closed, can crash, or can be opened on a
directory synced from another machine, and the run does not notice.

Nothing here kills a run, either. Stopping is a *request* — see :mod:`missile_defense.runs.control`
— so the loop finishes its update and writes a final checkpoint. The trainer has
no way to take that away.

Finding the game binary is the fiddly part, so it is deliberate rather than a
fixed path: an explicit ``MD_APP``, then this checkout's build directories, then
``PATH`` for a system install from the ``.deb``.

Finding the *interpreter* is the same kind of search and now has the same shape:
``MD_PYTHON``, then the runtime the trainer installed itself (:mod:`missile_defense.runs.runtime`),
then this interpreter if torch happens to be importable from it. Only starting a
run depends on the answer — attaching, browsing and replay never do.
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
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol

from . import paths, runtime

#: <root>/python/missile_defense/runs/runner.py — the checkout, when the trainer runs from one.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: The directory ``md`` itself sits in, so a spawned run can import it without
#: the trainer having been installed.
PACKAGE_PATH = Path(__file__).resolve().parents[2]

#: Where a local build puts the game, best build first.
BUILD_PATHS = ("build/release/app", "build/debug/app")

#: What the game is called. A build calls it `md_app`; every installer names it
#: `missile-defense`, which on Debian is on the default PATH for a login shell
#: but not necessarily for a desktop session — hence the explicit directory as
#: well as the PATH search.
BINARY_NAMES = ("md_app", "missile-defense")
SYSTEM_PATHS = ("/usr/games", "/usr/local/games", "/usr/bin")

#: Where the macOS disk image's drag-to-install leaves the game. Nothing inside a
#: bundle is ever on PATH, so an installed copy is only findable by looking here.
MACOS_APP_PATHS = ("/Applications", "~/Applications")

#: The bundle a build produces, and the one an installer leaves. They differ on
#: purpose (app/CMakeLists.txt): a checkout keeps the target's own name so the
#: layout matches every other platform, while the installed copy carries the name
#: Finder shows — which is the bundle's filename, not CFBundleDisplayName.
BUILD_BUNDLE = "md_app.app"
INSTALLED_BUNDLE = "Missile Defense.app"

#: An MSYS2 build links against Qt in the CLANG64 prefix and finds it on PATH,
#: which is there in the CLANG64 shell and absent everywhere else — including the
#: native interpreter the trainer is likely started from. Adding it back turns a
#: silent "nothing happened" into a window (docs/WINDOWS.md).
MSYS2_BIN = "clang64/bin"


#: What the trainer is called once installed. The Debian package and the
#: pyproject entry point agree on the name, so one search finds either.
TRAINER_NAMES = ("missile-defense-trainer",)

#: Where an installer leaves it. `/usr/games` is not searched: the trainer is
#: not a game and its Debian package puts it in `/usr/bin`.
TRAINER_SYSTEM_PATHS = ("/usr/bin", "/usr/local/bin")


#: Where the game records the interpreter it installed the trainer into, inside
#: the data directory. The C++ side writes and reads the same file — see
#: `record_file` in app/trainer.hpp — so the format is fixed by both: `key=value`
#: lines, unknown keys ignored so a newer writer cannot break an older reader.
RECORD_FILE = "trainer.conf"
RECORD_INTERPRETER_KEY = "interpreter"


def recorded_interpreter(data_dir: Path | None = None) -> Path | None:
    """The interpreter recorded in ``data_dir``, or ``None`` if there is no record.

    ``data_dir`` defaults to the same directory Qt's ``AppLocalDataLocation``
    resolves to, which is what the game passes its own lookup — the two must
    agree or the game and the trainer disagree about whether a trainer exists.
    """
    directory = paths.data_home() if data_dir is None else data_dir
    try:
        text = (directory / RECORD_FILE).read_text(encoding="utf-8")
    except OSError:
        return None  # never installed from the game, which is the ordinary case
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == RECORD_INTERPRETER_KEY:
            return Path(value.strip())
    return None


def record_interpreter(interpreter: Path, version: str, data_dir: Path | None = None) -> Path:
    """Write the record the lookup reads, and return where it was written.

    Called after a successful install. The version goes in beside the
    interpreter so the game can tell a trainer that is merely *older* than the
    wheel beside it from one that is missing, and offer the right thing.
    """
    directory = paths.data_home() if data_dir is None else data_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RECORD_FILE
    path.write_text(
        f"{RECORD_INTERPRETER_KEY}={interpreter}\nversion={version}\n", encoding="utf-8"
    )
    return path


class AppNotFound(RuntimeError):
    """The game binary could not be located — the message says how to fix it."""


class Process(Protocol):
    """Just enough of ``subprocess.Popen`` to know whether it is still up."""

    def poll(self) -> int | None: ...


#: Injected so tests can watch what would be launched without launching it.
Spawn = Callable[[list[str], Path, Mapping[str, str]], Process]


def _spawn(command: list[str], cwd: Path, env: Mapping[str, str]) -> Process:
    return subprocess.Popen(command, cwd=str(cwd), env=dict(env))


def _bundle_executable(containing: Path, bundle: str) -> Path:
    """The executable buried inside the macOS .app bundle in ``containing``."""
    return containing / bundle / "Contents" / "MacOS" / "md_app"


def app_binary(
    environ: Mapping[str, str] | None = None,
    *,
    root: Path | None = None,
    platform: str = sys.platform,
) -> Path | None:
    """Locate ``md_app``, or return ``None`` if this machine has no build of it.

    ``platform`` is a parameter for the same reason :func:`launch_environ` takes
    one: the macOS bundle layout stays testable from any OS.
    """
    env = os.environ if environ is None else environ
    root = PROJECT_ROOT if root is None else root
    exe = ".exe" if platform == "win32" else ""

    override = env.get("MD_APP")
    if override:
        candidate = Path(override)
        return candidate if candidate.exists() else None
    for build in BUILD_PATHS:
        # macOS builds a bundle, so the executable is nested (app/CMakeLists.txt).
        candidate = (
            _bundle_executable(root / build, BUILD_BUNDLE)
            if platform == "darwin"
            else root / build / f"md_app{exe}"
        )
        if candidate.exists():
            return candidate
    for name in BINARY_NAMES:
        found = shutil.which(f"{name}{exe}", path=env.get("PATH"))
        if found:
            return Path(found)
    if platform == "darwin":
        for directory in MACOS_APP_PATHS:
            candidate = _bundle_executable(Path(directory).expanduser(), INSTALLED_BUNDLE)
            if candidate.exists():
                return candidate
    for directory in SYSTEM_PATHS:
        candidate = Path(directory) / "missile-defense"
        if candidate.exists():
            return candidate
    return None


def trainer_executable(
    environ: Mapping[str, str] | None = None,
    *,
    root: Path | None = None,
    platform: str = sys.platform,
    data_dir: Path | None = None,
) -> Path | None:
    """Locate the trainer, or ``None`` if this install does not have one.

    **This lookup is the boundary between the two products.** The game adds its
    TRAIN AI entry only when this resolves, so on a game-only install — where
    there is no Python, no ``md`` package and no ``missile-defense-trainer`` — it must return
    ``None``, and the menu simply does not offer training. The C++ side searches
    the same places in the same order for exactly that reason: a disagreement
    between them is a menu entry that launches nothing, or a trainer that is
    installed and unreachable.

    Four places, most explicit first, mirroring :func:`app_binary`:

    1. ``MD_TRAINER`` — someone said which one. A path that does not exist is
       ``None`` rather than a fallback, because falling back would start a
       *different* trainer than the one that was named.
    2. the interpreter the game recorded when it installed the trainer itself,
       from ``trainer.conf`` in the data directory, run as
       ``-m missile_defense.ui``.
    3. ``missile-defense-trainer`` on ``PATH``, then the directories a
       distribution uses — the Debian answer, where apt owns both halves.
    4. this checkout's own ``python/missile_defense/ui``, run as
       ``-m missile_defense.ui`` — a developer has no installed launcher but does
       have the trainer, and the game should still offer it there.

    Step 2 replaced a search for a payload directory beside the game. Guessing
    where pip put things cannot be made to work — its scripts land in
    ``~/Library/Python/3.x/bin`` or ``%APPDATA%\\Python\\...\\Scripts``, neither is
    on ``PATH``, and a macOS app launched from the Finder inherits almost none of
    ``PATH`` anyway. So the game writes down the *interpreter* it installed into,
    which needs no scripts directory and no ``PATH`` at all.

    **Nothing in this package calls this.** It exists so the contract has two
    independent statements of itself, and `test_ui_runner.py` holds it to
    `app/trainer.cpp`. That is the point: one implementation cannot disagree with
    itself, and the failure this guards against — a menu entry that launches
    nothing — is invisible from either side alone.
    """
    env = os.environ if environ is None else environ
    root = PROJECT_ROOT if root is None else root
    exe = ".exe" if platform == "win32" else ""

    override = env.get("MD_TRAINER")
    if override:
        candidate = Path(override)
        return candidate if candidate.exists() else None
    recorded = recorded_interpreter(data_dir)
    # Skipped rather than fatal when it no longer exists: unlike MD_TRAINER
    # nobody asked for this one by name, so an interpreter that has since been
    # uninstalled must not hide an apt-installed trainer on PATH.
    if recorded is not None and recorded.exists():
        return recorded
    for name in TRAINER_NAMES:
        found = shutil.which(f"{name}{exe}", path=env.get("PATH"))
        if found:
            return Path(found)
    for directory in TRAINER_SYSTEM_PATHS:
        for name in TRAINER_NAMES:
            candidate = Path(directory) / f"{name}{exe}"
            if candidate.exists():
                return candidate
    if (root / "python" / "missile_defense" / "ui" / "__main__.py").exists():
        return Path(sys.executable)
    return None


def trainer_command(
    environ: Mapping[str, str] | None = None,
    *,
    root: Path | None = None,
    platform: str = sys.platform,
    data_dir: Path | None = None,
) -> list[str] | None:
    """The trainer as an argv, or ``None``.

    Adds ``-m missile_defense.ui`` where an interpreter runs it.

    Split from :func:`trainer_executable` because the game only needs to know
    *whether* there is one to decide its menu, while starting it needs the whole
    command — and two of the four answers above are not self-contained.
    """
    found = trainer_executable(environ, root=root, platform=platform, data_dir=data_dir)
    if found is None:
        return None
    # An interpreter is handed the module; a launcher is run as it is. The
    # recorded answer is an interpreter that is *not* this process's, so the test
    # cannot be `== sys.executable` any more.
    if found.name.startswith("missile-defense-"):
        return [str(found)]
    return [str(found), "-m", "missile_defense.ui"]


def launch_environ(
    environ: Mapping[str, str] | None = None, *, platform: str = sys.platform
) -> dict[str, str]:
    """The environment the game is started in — the platform quirks, once.

    ``platform`` is a parameter so both branches are testable from either OS;
    a quirk that only one machine can check is a quirk that rots.

    Linux imposes nothing. A game launched from the trainer should look exactly
    like the same game launched from the desktop, and on a Wayland session that
    means a Wayland window. This used to force xcb, from the days when
    ``QVulkanWindow`` could not survive Qt's teardown there; ``GameWindow::event``
    handles that now, and ``test_wayland_teardown.py`` says so out loud.
    """
    env = dict(os.environ if environ is None else environ)
    if platform == "win32":
        msys = Path(env.get("MSYS2_ROOT", "C:/msys64")) / MSYS2_BIN
        if msys.is_dir():
            env["PATH"] = f"{msys}{os.pathsep}{env.get('PATH', '')}"
    return env


class ReplayLauncher:
    """Opens recordings in the game, and keeps track of the windows it opened.

    The handles are kept only so finished children are reaped; the trainer never
    waits on the game, and closing the trainer leaves it running.
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
        self._launch("--replay", recording)

    def launch_match(self, manifest: Path) -> None:
        """Open a recorded match split-screen. Raises if the game is absent.

        Two agents, one seed, one clock — see `docs/API.md` §8. The manifest
        carries the names and the scores, so the screen can state what it is
        showing rather than leaving a viewer to assume.
        """
        self._launch("--match", manifest)

    def launch_pair(self, left: Path, right: Path) -> None:
        """Open two recordings side by side, with no manifest. Raises if unbuilt.

        The ad-hoc half of :meth:`launch_match`: two episodes exist and nothing
        wrote a tournament record for them, which is exactly the case while the
        tournament is *still running*. The names come from the recordings' own
        labels, and the screen claims no scores because there are none yet.
        """
        self._launch("--match-left", left, "--match-right", str(right))

    def launch_model(self, policy: Path, *, seed: int | None = None) -> None:
        """Open the game watching a promoted model. Raises if the game is absent.

        `--watch-model` and not a recording: the league's question is "how does
        this one play?", and a stored episode answers "how did it play once".

        ``seed`` pins the episode, which is what makes it possible to *watch a
        contest that is still running*: the evaluator and the game are both
        deterministic, so the same policy on the same seed is the same episode
        tick for tick. The window is a spectator and nothing more — it computes
        its own copy, the contest never waits for it, and closing it changes
        nothing.
        """
        extra = () if seed is None else ("--seed", str(seed))
        self._launch("--watch-model", policy, *extra)

    def _launch(self, flag: str, target: Path, *extra: str) -> None:
        """One spawn, however the game is being opened.

        Three copies of this drifted apart once already — the flag is the only
        thing that differs, and the "is it built?" message is the part a person
        actually reads.
        """
        binary = app_binary(self._environ, root=self._root)
        if binary is None:
            raise AppNotFound(
                "the game is not built here — run `cmake --build --preset release`, "
                "or point MD_APP at an md_app binary"
            )
        self._children = [child for child in self._children if child.poll() is None]
        self._children.append(
            self._spawn(
                [str(binary), flag, str(target), *extra],
                self._root,
                launch_environ(self._environ),
            )
        )

    @property
    def running(self) -> int:
        """How many game windows this trainer has open."""
        self._children = [child for child in self._children if child.poll() is None]
        return len(self._children)


# ---- starting a training run -------------------------------------------------


class PipedProcess(Protocol):
    """A child whose output the trainer reads."""

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


@dataclass(frozen=True)
class Interpreter:
    """The Python a run would be started with, and where it was found."""

    path: str
    #: For the tooltip, so "why is Start off?" and "which torch is that?" are
    #: answerable without reading this file.
    source: str


def find_interpreter(
    environ: Mapping[str, str] | None = None,
    *,
    store: runtime.Runtime | None = None,
) -> Interpreter | None:
    """Something that could run a trainer, or ``None`` if this machine has none.

    Three places, in order of how explicit each is:

    1. ``MD_PYTHON`` — someone said which one, so it is not second-guessed. It is
       the split-interpreter case on Windows (docs/WINDOWS.md), where the trainer
       and the trainer are deliberately different builds.
    2. the runtime the trainer installed and health-checked itself
       (:mod:`missile_defense.runs.runtime`) — the answer for anyone who installed a package.
    3. this interpreter, if torch happens to be importable from it — the developer
       case, and what this function used to be in its entirety.

    ``find_spec`` locates torch without importing it, which still matters: the
    trainer must never pull torch in, and a test asserts it.
    """
    env = os.environ if environ is None else environ
    explicit = env.get("MD_PYTHON")
    if explicit:
        return Interpreter(explicit, "MD_PYTHON")
    managed = (runtime.Runtime() if store is None else store).python()
    if managed is not None:
        return Interpreter(str(managed), "the runtime this trainer installed")
    if importlib.util.find_spec("torch") is not None:
        return Interpreter(sys.executable, "this interpreter")
    return None


def training_python(
    environ: Mapping[str, str] | None = None,
    *,
    store: runtime.Runtime | None = None,
) -> str:
    """Which interpreter a run is started with.

    Falls back to the trainer's own even when nothing can train, because the
    caller that builds a command line should not have to handle ``None`` for a
    case the UI has already disabled.
    """
    found = find_interpreter(environ, store=store)
    return found.path if found is not None else sys.executable


def can_train(
    environ: Mapping[str, str] | None = None,
    *,
    store: runtime.Runtime | None = None,
) -> bool:
    """Whether a run could be started at all.

    Only Start depends on this. Attaching to a run, browsing recordings and
    replaying them stay available with no runtime installed and no torch
    anywhere — watching a run synced from another machine is a supported way to
    use the trainer, and always was.
    """
    return find_interpreter(environ, store=store) is not None


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
    """A training run this trainer started, and the lines it has printed.

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
