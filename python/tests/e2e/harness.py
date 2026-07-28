# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Drive the shipped surfaces the way a user does. No tests in here.

Everything below starts a **real process** — the game binary, the trainer, the
trainer — against **real files** in a temporary directory, and reads back what a
person could observe. That process boundary is the whole definition of an e2e
test here: if a claim can be checked by calling a function, it belongs beside the
code as a unit test (docs/TESTING.md).

Three things are deliberate, and each is here because the obvious version of
these tests is flaky or lies:

* **No sleeps.** :func:`wait_until` polls with a deadline. A sleep long enough
  for a loaded CI runner wastes minutes on every developer's machine, and one
  tuned for a developer's machine is next month's CI flake.
* **Skip, never fail, on a missing optional half.** torch and PySide6 are
  optional by design — the game must never depend on either — so a machine
  without them reports *skipped* with the package named, not red.
* **Every run is bounded.** The game is given a frame budget, every subprocess a
  timeout. A hang has to surface as a failed test, not as a job that runs until
  the runner gives up.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

#: <root>/python/tests/e2e/harness.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: `missile_defense` itself, for the PYTHONPATH a spawned trainer needs.
PACKAGE_PATH = PROJECT_ROOT / "python"

#: Best build first, mirroring `missile_defense.runs.runner`. The debug build is preferred for
#: e2e when it exists: it is the one with `VK_LAYER_KHRONOS_validation` enabled,
#: which is what turns "did it render correctly?" into a checkable question
#: without capturing a single pixel.
BUILD_PATHS = ("build/debug/app", "build/release/app")

#: Generous, because a debug/ASan build on a loaded runner is slow — but finite,
#: because the failure this guards against is a window that never closes.
APP_TIMEOUT_S = 180.0

#: A training e2e trains for seconds. It is not checking that PPO learns
#: (test_ppo.py does that); it is checking that a run writes what it owes.
TRAIN_TIMEOUT_S = 600.0


# ---- skip guards -------------------------------------------------------------


def _have(module: str) -> bool:
    """Whether ``module`` could be imported, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # a namespace package shadowing the name
        return False


def app_binary() -> Path | None:
    """The game, from this checkout's builds. ``MD_APP`` overrides."""
    override = os.environ.get("MD_APP")
    if override:
        candidate = Path(override)
        return candidate if candidate.exists() else None
    suffix = ".exe" if sys.platform == "win32" else ""
    for build in BUILD_PATHS:
        candidate = PROJECT_ROOT / build / f"md_app{suffix}"
        if candidate.exists():
            return candidate
    return None


#: Set this to run the game tests on the real display, where you can watch them.
#: Off by default, and that is the point — see :func:`_display_wrapper`.
VISIBLE_ENV = "MD_E2E_VISIBLE"

#: A virtual X server: a real X11 display with no screen attached to it, which is
#: what lets a Vulkan swapchain exist without a window appearing on anyone's
#: desktop. `-a` picks a free display number, so parallel runs do not collide.
XVFB = ("xvfb-run", "-a", "--server-args=-screen 0 1280x720x24")


def wants_visible() -> bool:
    return os.environ.get(VISIBLE_ENV, "") not in ("", "0")


def _display_wrapper() -> list[str] | None:
    """How to give the game a display, or ``None`` if this machine cannot.

    **Invisible is the default, deliberately.** A test suite that throws game
    windows onto the screen of whoever is at the keyboard is a suite people stop
    running, and one that steals focus mid-run can lose them what they were
    typing. So the game is rendered into a virtual X server nobody is looking at,
    and the only way to see it is to ask, with ``MD_E2E_VISIBLE=1``.

    Qt's ``offscreen`` platform is not an option here even though it sounds like
    exactly this: it has no Vulkan support at all (*"This plugin does not support
    createPlatformVulkanInstance"*), so the game cannot start under it. Xvfb is a
    real X server and the swapchain is real; there is simply no monitor.
    """
    if sys.platform != "linux":
        return []  # Windows and macOS render to an ordinary hidden window
    if wants_visible():
        return [] if os.environ.get("DISPLAY") else None
    # Resolved to an absolute path, not left as a bare name. The packaging tests
    # run the staged game on a deliberately minimal `PATH` — proving the install
    # needs no interpreter — and `subprocess` resolves argv[0] against *that*
    # PATH, so a bare `xvfb-run` vanished with everything else in /usr/bin.
    # The wrapper is this suite's own scaffolding, not part of what is under
    # test, so it should not be subject to the environment being tested.
    found = shutil.which("xvfb-run")
    return [found, *XVFB[1:]] if found else None


#: Applied to whole modules or single tests. Written as constants rather than
#: fixtures so the reason is visible in the test file that skips.
needs_app = pytest.mark.skipif(
    app_binary() is None,
    reason="the game is not built here — cmake --build --preset release",
)
needs_display = pytest.mark.skipif(
    _display_wrapper() is None,
    reason=(
        "no way to render invisibly — install xvfb (`sudo apt install xvfb`), "
        f"or set {VISIBLE_ENV}=1 to run them on your own screen"
    ),
)
needs_torch = pytest.mark.skipif(not _have("torch"), reason="torch is not installed")
needs_native = pytest.mark.skipif(
    not _have("missile_defense._md_native"), reason="the native binding is not built"
)
needs_qt = pytest.mark.skipif(not _have("PySide6"), reason="PySide6 is not installed")

#: Building a wheel needs `build`. It is in `tools/bootstrap.py`'s DEV_TOOLS, so
#: a bootstrapped venv has it — this guard is for a bare interpreter, and it
#: names the fix rather than failing on an import nobody asked for.
#: The wheel-install suite builds a wheel, which compiles the extension — minutes
#: of work that duplicates what the `wheel` CI job already does. Opt-in so it
#: runs there and in a developer's own run, and never inside `app-e2e`, where it
#: would be a second copy of the same proof paid for twice.
needs_wheel_e2e = pytest.mark.skipif(
    os.environ.get("MD_WHEEL_E2E") != "1",
    reason="set MD_WHEEL_E2E=1 to build and install the wheel (slow; the CI `wheel` job does)",
)

needs_build = pytest.mark.skipif(
    not _have("build"),
    reason="the `build` package is not installed — re-run `python3 -m tools.bootstrap`",
)


# ---- waiting -----------------------------------------------------------------


def wait_until(
    condition: Callable[[], bool], *, timeout: float = 30.0, what: str = "condition"
) -> None:
    """Poll ``condition`` until it holds, or fail saying what never happened.

    The interval starts short and grows, so a condition that is already true
    costs a millisecond and one that takes twenty seconds does not spin a core
    for twenty seconds.
    """
    deadline = time.monotonic() + timeout
    interval = 0.005
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(interval)
        interval = min(interval * 1.5, 0.25)
    raise AssertionError(f"{what} did not happen within {timeout:.0f}s")


# ---- the game ----------------------------------------------------------------


@dataclass(frozen=True)
class AppRun:
    """What one run of the game did, as its `--report` describes it."""

    exit_code: int
    report: dict[str, object]
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        """Everything the run printed, both streams.

        **`xvfb-run` runs its command as `"$@" 2>&1`.** The two streams are
        merged into stdout before this process ever sees them, so `stderr` is
        empty for every run that goes through the wrapper — which is every run
        in CI. Assertions about what the game *said* have to read this.

        That is not only a test-ergonomics point: `validation_errors` greps for
        VUID messages, and reading `stderr` meant it found none and reported
        every run clean. The renderer check had been inert for as long as the
        suite has run under Xvfb.
        """
        return self.stdout + "\n" + self.stderr

    @property
    def models(self) -> int:
        """How many installed models the game says it can actually run."""
        return int(self.report.get("models", 0))

    @property
    def mode(self) -> str:
        return str(self.report.get("mode", ""))

    @property
    def state(self) -> str:
        return str(self.report.get("state", ""))

    @property
    def ticks(self) -> int:
        return int(self.report.get("ticks", 0))

    @property
    def score(self) -> int:
        return int(self.report.get("score", 0))

    @property
    def frames(self) -> int:
        return int(self.report.get("frames", 0))

    @property
    def cities_left(self) -> int:
        return int(self.report.get("cities_left", 0))

    @property
    def menu(self) -> list[str]:
        """The main menu's labels.

        The one part of the game whose *contents* depend on what else is
        installed beside it — TRAIN AI is there only where a training trainer
        was found — so it is how a packaging test tells the game-only product
        from the full one without a screenshot and a pair of eyes.
        """
        return [str(label) for label in self.report.get("menu", [])]


def app_environ(sandbox: Path) -> dict[str, str]:
    """The environment the game is started in for a test.

    ``sandbox`` is a temporary directory that stands in for the user's whole
    home. Everything the game persists is redirected into it — recordings, high
    scores, and the audio/fullscreen preferences QSettings writes — so a test can
    neither read the developer's own files nor leave anything of its own in them.
    Losing a high-score table to a test run would be a poor trade for a green tick.

    ``xcb`` for the same reason `tools/run_app.py` forces it: an X11 window is
    what Xvfb provides, and the Wayland plugin would have nothing to talk to.
    """
    env = dict(os.environ)
    if sys.platform == "linux":
        env["QT_QPA_PLATFORM"] = "xcb"
    for name, sub in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data")):
        directory = sandbox / sub
        directory.mkdir(parents=True, exist_ok=True)
        env[name] = str(directory)
    env["MD_RUNS_DIR"] = str(sandbox / "runs")
    # And the league beside it, so a test that promotes a model can see the game
    # pick it up — and so a test that promotes nothing does not find the
    # developer's own models in the browser.
    env["MD_MODELS_DIR"] = str(sandbox / "models")
    # The debug build is preferred for its Vulkan validation layer, and it also
    # carries LeakSanitizer — which at exit reports the graphics driver's and
    # dbus's allocations, none of which this project owns or can free. Keep every
    # other ASan check (use-after-free is the one worth having here); drop the
    # one whose findings are all in somebody else's library.
    env["ASAN_OPTIONS"] = f"{env.get('ASAN_OPTIONS', '')}:detect_leaks=0".lstrip(":")
    return env


def run_app(
    *args: str,
    sandbox: Path,
    frames: int = 240,
    until_done: bool = False,
    timeout: float = APP_TIMEOUT_S,
    expect_report: bool = True,
    binary: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppRun:
    """Run the game to a bounded end and return what it reported.

    Invisible and silent by default: rendered into a virtual X server (see
    :func:`_display_wrapper`) and started with ``--silent``, so a full run of this
    suite neither appears on the screen nor comes out of the speakers of whoever
    is at the machine.

    ``frames`` is always passed, including alongside ``until_done``: the frame
    budget is the backstop that turns a recording which never ends — or a window
    that never gets there — into a failed assertion rather than a hung job.

    ``binary`` and ``environ`` are for the packaging tests, which run the game
    out of a *staged install tree* rather than the build tree, on a `PATH` with
    no interpreter on it. Everything else uses the defaults, and should: those
    two arguments are the difference between "the game works" and "this
    particular install of the game works".
    """
    chosen = app_binary() if binary is None else binary
    assert chosen is not None, "the game is not built"
    wrapper = _display_wrapper()
    assert wrapper is not None, "no way to render this invisibly"

    command = [*wrapper, str(chosen), *args, "--frames", str(frames), "--silent"]
    if until_done:
        command.append("--until-done")
    if expect_report:
        command.append("--report")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=app_environ(sandbox) if environ is None else dict(environ),
        check=False,
    )
    report: dict[str, object] = {}
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("{"):
            report = json.loads(line)
            break
    run = AppRun(result.returncode, report, result.stdout, result.stderr)
    if expect_report:
        assert report, f"no --report line on stdout:\n{result.stdout}\n{result.stderr}"
    return run


def validation_errors(run: AppRun) -> list[str]:
    """Vulkan validation messages from this run, newest last.

    **There is no allow-list.** There used to be one, holding
    `VUID-vkAcquireNextImageKHR-semaphore-01779`, on the reasoning that the
    swapchain belongs to `QVulkanWindow` and so the semaphore was not ours to
    fix. The ownership half of that was right and the conclusion was wrong: an
    application cannot reorder Qt's acquire, but it *can* refuse to let Qt reuse
    a semaphore that is still busy, which is what `Renderer::submit` now does.
    See the comment there, and `test_vulkan_validation.py` for the evidence that
    the defect is Qt's.

    A baseline of "errors we have decided to live with" stops being a to-do and
    becomes background noise, and every renderer change after it is reviewed
    against a dirty baseline. If a VUID appears here, it is a bug.
    """
    # `output`, not `stderr`: under `xvfb-run` the layer's messages arrive on
    # stdout. See :attr:`AppRun.output` — this grepping the wrong stream is why
    # no validation error was ever reported by this suite.
    return [line for line in run.output.splitlines() if "VUID" in line]


def assert_clean(run: AppRun) -> None:
    """The two things that must hold of every run of the game.

    A Vulkan validation error is a failure and not a warning: it means the
    renderer did something the specification does not define, which is a bug that
    happens to have worked on this driver. Only the debug build enables the
    layer, so this bites exactly where it can.
    """
    assert run.exit_code == 0, f"exited {run.exit_code}\n{run.stderr}"
    offending = validation_errors(run)
    assert not offending, "Vulkan validation errors:\n" + "\n".join(offending)


# ---- a training run ----------------------------------------------------------

#: A run small enough to be a test and complete enough to produce every artifact.
#:
#: `max_ticks` is the load-bearing one. A recording is only written when an
#: episode *finishes*, and an uncapped episode is thousands of ticks — so without
#: this the run produces curves and checkpoints but no `.mdr`, and the replay
#: half of the suite would have nothing to play.
TINY_RUN: Mapping[str, str] = {
    "--envs": "4",
    "--steps": "48",
    "--updates": "8",
    "--max-ticks": "400",
    "--eval-every": "8",
    "--record-every": "2",
    "--checkpoint-every": "4",
    "--device": "cpu",
}


def train_environ() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    existing = env.get("PYTHONPATH", "")
    if str(PACKAGE_PATH) not in existing.split(os.pathsep):
        env["PYTHONPATH"] = (
            f"{PACKAGE_PATH}{os.pathsep}{existing}" if existing else str(PACKAGE_PATH)
        )
    return env


def train(
    out_dir: Path,
    *,
    overrides: Mapping[str, str] | None = None,
    python: str | None = None,
    timeout: float = TRAIN_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    """Run a real short training run into ``out_dir``.

    Real, not mocked: the point is that the artifacts a trainer and a game later
    read were written by the actual trainer, through the actual bindings, in a
    separate process — which is where the integration failures live.
    """
    return subprocess.run(
        train_command(out_dir, overrides=overrides, python=python),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=train_environ(),
        cwd=str(PROJECT_ROOT),
        check=False,
    )


def train_command(
    out_dir: Path,
    *,
    overrides: Mapping[str, str] | None = None,
    python: str | None = None,
) -> list[str]:
    """The command line a tiny run is started with."""
    settings = dict(TINY_RUN)
    settings.update(overrides or {})
    command: list[str] = [
        python or sys.executable,
        "-m",
        "missile_defense.training",
        "--out-dir",
        str(out_dir),
    ]
    for flag, value in settings.items():
        command += [flag, value]
    return command


def start_training(
    out_dir: Path,
    *,
    overrides: Mapping[str, str] | None = None,
    python: str | None = None,
) -> subprocess.Popen[str]:
    """The same run, handed back while it is still going.

    For the tests that have something to say to a *live* trainer — the control
    files (:mod:`missile_defense.runs.control`) are answers to questions asked mid-run, and a
    finished run cannot be asked.
    """
    return subprocess.Popen(  # noqa: S603 — our own command line, built above
        train_command(out_dir, overrides=overrides, python=python),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=train_environ(),
        cwd=str(PROJECT_ROOT),
    )


def recordings(run_dir: Path) -> list[Path]:
    """The watchable episodes a run dropped, oldest first."""
    return sorted(run_dir.glob("update-*.mdr"))


# ---- the scripted baseline ---------------------------------------------------


def agent_eval_binary() -> Path | None:
    """The scripted-baseline evaluator, from this checkout's builds.

    The other half of Task 11's claim. A run's statistics are written to
    `evals.csv` by the trainer and *printed* by this binary, from the same C++
    `Summary` — so "the numbers reach the files people read" is only proven by
    reading both, and this one has no Python in it at all.
    """
    suffix = ".exe" if sys.platform == "win32" else ""
    for build in ("build/debug/agent", "build/release/agent"):
        candidate = PROJECT_ROOT / build / f"md_agent_eval{suffix}"
        if candidate.exists():
            return candidate
    return None


needs_agent_eval = pytest.mark.skipif(
    agent_eval_binary() is None,
    reason="the evaluator is not built here — cmake --build --preset release",
)


def agent_eval(*args: str, timeout: float = 300.0) -> subprocess.CompletedProcess[str]:
    """Run the scripted-baseline evaluator the way ``poe eval`` does."""
    binary = agent_eval_binary()
    assert binary is not None, "the evaluator is not built"
    return subprocess.run(
        [str(binary), *args], capture_output=True, text=True, timeout=timeout, check=False
    )


# ---- a local package index ---------------------------------------------------


def build_wheel(destination: Path, *, timeout: float = 600.0) -> Path:
    """Build this project's wheel into ``destination`` and return it.

    What it is for: the runtime-setup e2e needs a package index to install
    *from*, and using the real one would make the test slow, non-hermetic, and a
    download of several gigabytes. A directory holding one wheel is a valid
    `--find-links` target, so the install path can be exercised end to end
    against something this repository produced a moment earlier.
    """
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(destination), str(PROJECT_ROOT)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    wheels = sorted(destination.glob("*.whl"))
    assert wheels, f"no wheel was built into {destination}"
    return wheels[-1]


def have_command(name: str) -> bool:
    return shutil.which(name) is not None


def as_lines(text: str) -> Sequence[str]:
    return [line for line in text.splitlines() if line.strip()]
