# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""A training runtime the trainer installs and owns. No Qt, no torch.

The trainer could always *watch* a run from anywhere, and could only *start* one
where torch already happened to be importable — so the answer to "I installed
this, how do I train?" was "open a terminal and pip install torch", which is the
one instruction an installed application should never have to give
(docs/ROADMAP.md, M8).

This module makes that a button. It sits beside :mod:`missile_defense.runs.control` and
:mod:`missile_defense.runs.paths` for the same reason they do: the trainer writes
it, the training loop
is started from it, and neither has to import the other to agree on where the
interpreter is.

**Planning is separate from doing.** :func:`recommend` is a pure function of the
machine and the vendor probes — no filesystem, no network — so "which build would
this box get?" is answerable in a test and printable in a dialog before anything
is downloaded. :class:`Runtime` does the effects, and takes its subprocess runner
as an argument.

**Installing is transactional.** Each install goes into its own new directory,
is health-checked by importing everything a run imports (:data:`IMPORT_ADVICE`),
and only then becomes current — by replacing a small marker file. Nothing mutates an
existing runtime, so a failed, cancelled or half-downloaded install cannot leave
the working one worse than it found it. Cancelling deletes only the directory it
was filling.

**A marker file, not a symlink.** ``current`` holds the name of the live
directory. A symlink would be the obvious Unix answer and needs a privilege on
Windows that a normal user does not have; ``os.replace`` over a text file is
atomic on both.

**The venv is built from the trainer's own interpreter**, which is what makes
``missile_defense._md_native`` importable inside it: the binding is a compiled extension
built for one CPython ABI, and a runtime on a different minor version would
install torch perfectly and then fail to import the simulation. The health check
is what catches that, and it is the reason the check imports the binding rather
than only torch.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlparse

from . import paths

#: Where ``md`` itself sits, so the health check can import it — and so a run
#: started from the managed interpreter can too, without ``md`` being installed
#: into it. Mirrors ``missile_defense.runs.runner.PACKAGE_PATH``.
PACKAGE_PATH = Path(__file__).resolve().parent.parent

#: The file naming the live runtime directory.
CURRENT_MARKER = "current"

#: What each runtime directory records about itself.
MANIFEST_NAME = "runtime.json"

#: Format of that manifest. Bumping it retires older runtimes rather than
#: guessing at their layout — reinstalling costs a download, not a run.
MANIFEST_SCHEMA = 1

#: Hosts this trainer will install from, and the whole of the trust decision.
#: Installing a package is running its code, so the index is not a preference to
#: be taken from an env var or a text field: it is either one of these two over
#: https, or there is no install plan. PyPI serves the CPU build; PyTorch's own
#: index serves the accelerator builds, which are not on PyPI at all.
ALLOWED_INDEX_HOSTS = ("pypi.org", "download.pytorch.org")


class UnsafeIndex(ValueError):
    """A plan named a package index this trainer will not install from."""


@dataclass(frozen=True)
class Backend:
    """One way of getting torch, and where it comes from."""

    name: str
    label: str
    #: What the dialog says about *why* you would pick this.
    detail: str
    index_url: str
    packages: tuple[str, ...]
    #: ``sys.platform`` values this build actually exists for.
    platforms: tuple[str, ...]
    #: Rough installed size, so the dialog can warn before a long download.
    gigabytes: float


#: What a training run imports before it does anything, in the order the health
#: check imports them, paired with what a person should do when that import is
#: the one that failed. Order matters twice: the binding is imported first
#: precisely so an ABI mismatch is not mistaken for a torch problem, and
#: :func:`_why_unhealthy` reads this top to bottom to name the culprit.
#:
#: **This list is the definition of a working runtime.** It drives the health
#: script below, and :data:`PACKAGES` exists to satisfy it — a module that a run
#: needs and this does not name is a runtime that installs, passes, and then
#: cannot train.
IMPORT_ADVICE = (
    (
        "missile_defense._md_native",
        "the simulation binding is missing or does not match this interpreter. "
        "pip cannot supply it — it is compiled from this checkout. Build it with "
        "`poe bindings`, then verify the runtime again.",
    ),
    (
        "torch",
        "torch installed but will not import. The usual cause is a wheel built "
        "for a different CUDA or Python version — see the log for the traceback.",
    ),
    (
        "numpy",
        "the runtime has torch but not numpy, which every run needs before it "
        "reaches the first update — repair it from this dialog, which reinstalls "
        "both.",
    ),
)

#: What pip is asked for. `missile_defense._md_native` is deliberately absent: it is compiled
#: from this checkout and no index has it.
#:
#: numpy is not incidental and not torch's problem. `missile_defense.sim.env` types its buffer
#: contract with `numpy.typing` and the trainer imports it on its first line, but
#: torch declares numpy *optional* — so `pip install torch` alone produced a
#: runtime that imported torch happily, passed a health check that asked no more
#: than that, and then died on `import numpy` in a subprocess whose output the
#: trainer had already moved on from. Every backend gets the same list, because
#: the requirement is the trainer's rather than the accelerator's.
PACKAGES = ("torch", "numpy")


#: cu130 because that is what this project's development box runs and documents
#: (docs/NVIDIA.md): the driver exposes a maximum CUDA version and the wheel must
#: not exceed it, and a Blackwell card needs cu128 or newer to have any kernel at
#: all. Newer wheels keep working on older drivers far less often than the
#: reverse, so this tracks the documented recipe rather than "latest".
CUDA = Backend(
    name="cuda",
    label="NVIDIA (CUDA)",
    detail="Uses the GPU. This project is optimizer-bound, so it is the whole win.",
    index_url="https://download.pytorch.org/whl/cu130",
    packages=PACKAGES,
    platforms=("linux", "win32"),
    gigabytes=4.5,
)

#: ROCm's Python story is Linux's. On Windows an AMD card has no supported path
#: — torch-directml pins an older torch — so that machine is offered the CPU
#: build honestly rather than an install that cannot work.
ROCM = Backend(
    name="rocm",
    label="AMD (ROCm)",
    detail="Uses the GPU on Linux. RDNA 2 may need HSA_OVERRIDE_GFX_VERSION.",
    index_url="https://download.pytorch.org/whl/rocm6.4",
    packages=PACKAGES,
    platforms=("linux",),
    gigabytes=5.0,
)

#: Always installable, never fast. Worth having as the floor: a run that trains
#: slowly still teaches you what the trainer does.
CPU = Backend(
    name="cpu",
    label="CPU only",
    detail="Works everywhere. Training is far slower — hours become days.",
    index_url="https://pypi.org/simple",
    packages=PACKAGES,
    platforms=("linux", "win32", "darwin"),
    gigabytes=1.0,
)

#: Tried in this order, so an accelerator wins over the fallback.
BACKENDS = (CUDA, ROCM, CPU)

#: Run inside the candidate runtime, and the only thing that decides whether an
#: install counts. Its imports are :data:`IMPORT_ADVICE`'s, generated rather than
#: restated: a list of what a run needs and a list of what is checked, kept side
#: by side and edited by hand, is two lists that will disagree — and the way they
#: disagree is a runtime that passes and cannot train.
HEALTH_SCRIPT = (
    "import json, sys\n"
    + "".join(f"import {module}\n" for module, _ in IMPORT_ADVICE)
    + "print(json.dumps({\n"
    "    'python': '.'.join(str(v) for v in sys.version_info[:3]),\n"
    "    'native': hasattr(missile_defense._md_native, 'VecEnv'),\n"
    "    'torch': torch.__version__,\n"
    "    'device': 'cuda' if torch.cuda.is_available() else 'cpu',\n"
    "}))\n"
)

#: The states the dialog renders. Anything not ``ready`` means Start stays off.
ABSENT = "absent"
READY = "ready"
BROKEN = "broken"
INCOMPLETE = "incomplete"


def venv_python(target: Path, *, platform: str = sys.platform) -> Path:
    """The interpreter inside a virtual environment at ``target``."""
    if platform == "win32":
        return target / "Scripts" / "python.exe"
    return target / "bin" / "python"


# ---- planning ----------------------------------------------------------------


@dataclass(frozen=True)
class SystemInfo:
    """The machine, as much of it as the choice depends on."""

    platform: str
    #: The interpreter a runtime is built from — the trainer's own, so the
    #: native binding's ABI matches.
    python: Path
    python_version: tuple[int, int]

    @classmethod
    def here(cls) -> SystemInfo:
        return cls(
            platform=sys.platform,
            python=Path(sys.executable),
            python_version=sys.version_info[:2],
        )


class BackendProbe(Protocol):
    """Does this machine have the hardware for one backend?

    Deliberately *not* :class:`missile_defense.ui.system.GpuProbe`: that one reads telemetry
    and needs a vendor Python package installed, which is precisely what a
    machine that has never trained does not have. What matters here is whether
    the *driver* is present, because the torch wheels bundle everything else
    (docs/NVIDIA.md).
    """

    # A read-only property rather than a plain attribute, so a frozen dataclass
    # satisfies it: a protocol declaring a mutable attribute is not implemented
    # by something immutable, and every probe here is immutable by design.
    @property
    def backend(self) -> str: ...

    def present(self) -> bool: ...


@dataclass(frozen=True)
class CommandProbe:
    """A driver detected by its command-line tool being on ``PATH``."""

    backend: str
    command: str

    def present(self) -> bool:
        return shutil.which(self.command) is not None


def default_probes() -> tuple[BackendProbe, ...]:
    """What ships: the two vendor tools that come with a working driver."""
    return (CommandProbe("cuda", "nvidia-smi"), CommandProbe("rocm", "rocm-smi"))


@dataclass(frozen=True)
class RuntimePlan:
    """What would be installed, and where. Nothing has happened yet."""

    backend: str
    python: Path
    target: Path
    packages: tuple[str, ...]
    index_url: str


def _next_target(root: Path, backend: str, version: tuple[int, int]) -> Path:
    """A directory that does not exist yet, named for what goes in it.

    Numbered rather than timestamped so a test can predict it, and so a directory
    listing sorts the way a human reads it. The name carries the Python version
    because that is the thing that silently invalidates a runtime: the binding is
    built for one ABI.
    """
    stem = f"{backend}-py{version[0]}.{version[1]}"
    index = 1
    while (root / f"{stem}-{index}").exists():
        index += 1
    return root / f"{stem}-{index}"


def backend_for(name: str) -> Backend:
    """The backend called ``name``, falling back to CPU rather than raising."""
    return next((b for b in BACKENDS if b.name == name), CPU)


def recommend(
    system: SystemInfo,
    probes: Sequence[BackendProbe],
    *,
    root: Path | None = None,
) -> RuntimePlan:
    """The install this machine should be offered.

    Pure: it reads the probes and the platform and computes a name. The only
    thing it touches on disk is which numbered directory is free, which is a
    question about the target rather than an effect on it.

    A probe answering yes is not enough — the backend has to exist for this
    platform too. An AMD card on Windows says yes to ROCm and gets the CPU build,
    because the alternative is an install that downloads for ten minutes and then
    cannot import.
    """
    root = paths.runtime_dir() if root is None else root
    detected = {probe.backend for probe in probes if probe.present()}
    chosen = next(
        (b for b in BACKENDS if b.name in detected and system.platform in b.platforms),
        CPU,
    )
    return RuntimePlan(
        backend=chosen.name,
        python=system.python,
        target=_next_target(root, chosen.name, system.python_version),
        packages=chosen.packages,
        index_url=chosen.index_url,
    )


def check_index(url: str) -> None:
    """Raise unless ``url`` is https on an allow-listed host."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_INDEX_HOSTS:
        raise UnsafeIndex(
            f"{url} is not an index this trainer installs from — "
            f"only https on {', '.join(ALLOWED_INDEX_HOSTS)}"
        )


# ---- status ------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeStatus:
    """What the store is right now, and what the dialog should offer."""

    state: str
    detail: str
    python: Path | None = None
    backend: str | None = None
    torch_version: str | None = None
    device: str | None = None

    @property
    def ready(self) -> bool:
        """Whether a run can be started. The only question the trainer asks."""
        return self.state == READY

    @property
    def repairable(self) -> bool:
        """Whether there is wreckage worth clearing before reinstalling."""
        return self.state in (BROKEN, INCOMPLETE)

    @property
    def removable(self) -> bool:
        return self.state != ABSENT


def _sign(payload: Mapping[str, object]) -> dict[str, object]:
    """The manifest plus a checksum over itself.

    Not a signature in the cryptographic sense — there is no key to sign with on
    a user's machine, and the threat here is not an attacker but a half-written
    file, a partial sync, or a directory edited by hand. A checksum over the
    canonical form catches all three, and catching them means a runtime that has
    been meddled with reads as broken rather than as trustworthy.
    """
    body = {key: value for key, value in payload.items() if key != "checksum"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return {**body, "checksum": hashlib.sha256(blob.encode("utf-8")).hexdigest()}


def _object(blob: str) -> dict[str, object] | None:
    """``blob`` parsed as a JSON object, or ``None`` if it is anything else."""
    try:
        parsed: object = json.loads(blob)
    except ValueError:
        return None
    return cast("dict[str, object]", parsed) if isinstance(parsed, dict) else None


def _verified(manifest: Path) -> dict[str, object] | None:
    """The manifest's contents, or ``None`` if it is missing, junk or edited."""
    try:
        payload = _object(manifest.read_text(encoding="utf-8"))
    except OSError:
        return None
    if payload is None or payload.get("schema") != MANIFEST_SCHEMA:
        return None
    expected = payload.get("checksum")
    return payload if isinstance(expected, str) and _sign(payload)["checksum"] == expected else None


@dataclass(frozen=True)
class Health:
    """What the candidate interpreter said when asked to import everything."""

    ok: bool
    detail: str
    torch_version: str | None = None
    device: str | None = None


#: Runs a command and streams its output a line at a time. Injected so every
#: state below is reachable in a test without a network or a real venv.
Runner = Callable[[list[str], Callable[[str], None]], int]


def _environ() -> dict[str, str]:
    """The environment a managed interpreter is invoked in.

    ``md`` is put on the path rather than installed into the runtime: the
    checkout (or the ``python3-md`` package) is the one copy, and a runtime that
    embedded its own would go stale the moment either changed.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    existing = env.get("PYTHONPATH", "")
    if str(PACKAGE_PATH) not in existing.split(os.pathsep):
        env["PYTHONPATH"] = (
            f"{PACKAGE_PATH}{os.pathsep}{existing}" if existing else str(PACKAGE_PATH)
        )
    return env


def _run(command: list[str], emit: Callable[[str], None]) -> int:
    """Run ``command``, feeding each line of its output to ``emit``."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # pip's failures belong in the same pane
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=_environ(),
    )
    assert process.stdout is not None
    for line in process.stdout:
        emit(line.rstrip("\n"))
    return process.wait()


def _last_json(lines: Iterable[str]) -> dict[str, object] | None:
    """The last line that is a JSON object — pip and CUDA both print chatter."""
    for line in reversed(list(lines)):
        payload = _object(line)
        if payload is not None:
            return payload
    return None


UNHEALTHY = "the installed runtime failed its check — see the log"

#: What :meth:`Runtime.install` says when it declines to start. Names the command
#: rather than the condition: "the binding is missing" is a diagnosis, and the
#: person reading it wants the cure.
NO_BINDING = (
    "the simulation binding is not built, so no runtime could pass its check — "
    "run `poe bindings` first, then install again"
)

#: What :meth:`Runtime.install` says when the binding exists but the interpreter
#: the runtime would be built from cannot load it. A different sentence from
#: :data:`NO_BINDING` because it is a different problem with a different cure:
#: `poe bindings` has already been run, just against the wrong interpreter.
WRONG_BINDING = (
    "the simulation binding exists but {python} cannot load it — it was built "
    "for a different interpreter or toolchain, and a runtime made from this one "
    "could not import it either. Rebuild it against this interpreter "
    "(`poe bindings -- win-native --python {python}` on Windows), then install again"
)

#: One import, in a fresh process, printing nothing. The question is whether the
#: extension *loads*, so nothing short of loading it will do.
BINDING_PROBE = "import missile_defense._md_native"

#: Whether a file for it is on the path at all — which is all `find_spec` can
#: say, and exactly the right tool for telling "never built" apart from "built
#: for something else" once :data:`BINDING_PROBE` has already said no.
BINDING_PRESENT_PROBE = (
    "import importlib.util as u, sys; "
    "sys.exit(0 if u.find_spec('missile_defense._md_native') else 1)"
)


def _missing_binding(python: Path, runner: Runner) -> str | None:
    """Why an install would be pointless, or ``None`` if it would not.

    **Asked of ``python``, not of this process, and by importing rather than by
    looking.** Both halves were wrong before and each cost the same thing.

    `importlib.find_spec` was the cheap way to ask, and it answers a different
    question: whether a file with the right suffix is on the path. A binding
    compiled by MSYS2 clang satisfies that and then fails to load in a
    python.org CPython, because it wants MinGW's runtime DLLs — so the guard
    passed, pip downloaded 1.9 GB of CUDA torch, and the health check at the end
    failed on the import the guard had declined to attempt. Measured here on
    2026-07-28: `find_spec` said `True`, `import` said `DLL load failed`.

    And the interpreter that has to load it is the one the runtime is built
    *from*, which is not necessarily the one the trainer runs in. Asking the
    trainer proves nothing about the venv it is about to create.

    Costs one interpreter start-up, against a download three orders of magnitude
    larger — and unlike an import in this process, it leaves nothing loaded.
    """
    if runner([str(python), "-c", BINDING_PROBE], _silent) == 0:
        return None
    # Present but unloadable is the more confusing of the two states, and worth
    # a second probe to name: `poe bindings` is the cure for one of them and has
    # already been done for the other. `find_spec` is the right tool *here* —
    # "is a file there" is exactly the question left, once the import has
    # already answered "does it load".
    present = runner([str(python), "-c", BINDING_PRESENT_PROBE], _silent) == 0
    return WRONG_BINDING.format(python=python) if present else NO_BINDING


def _why_unhealthy(lines: Iterable[str]) -> str:
    """Name the import that actually failed, rather than guessing torch.

    This used to report "could not import torch" whatever went wrong, which was
    wrong in the most common case and actively misleading in it: a source tree
    whose binding has not been built reports a perfect torch install as a torch
    failure, and the person goes looking in the wrong place. It cost a session.
    """
    text = "\n".join(lines)
    for module, advice in IMPORT_ADVICE:
        # Both spellings the interpreter uses: `No module named 'x'` for an
        # absent module, and a bare mention in an ImportError for one that is
        # present but unloadable — an ABI mismatch raises the latter.
        if f"No module named '{module}'" in text or ("ImportError" in text and module in text):
            return advice
    return UNHEALTHY


def _mtime(path: Path | None) -> float:
    """A file's modification time, or 0.0 when there is none.

    Stamps a verification against the manifest it was measured on, so
    installing, repairing or removing a runtime invalidates the result without
    anyone having to remember to say so.
    """
    try:
        return 0.0 if path is None else path.stat().st_mtime
    except OSError:
        return 0.0


def _silent(line: str) -> None:
    """Where an install's output goes when the caller did not ask for it."""


class Runtime:
    """The managed runtime store: what is installed, and how it got there.

    One directory holding numbered runtimes and a marker naming the live one.
    Every method is safe to call in any state — the trainer asks :meth:`status`
    on a timer and must never get an exception for a directory that a user has
    deleted underneath it.
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        runner: Runner = _run,
        platform: str = sys.platform,
    ) -> None:
        self._root = paths.runtime_dir() if root is None else root
        self._runner = runner
        self._platform = platform
        #: What :meth:`verify` last proved, as (interpreter, manifest mtime).
        #: Cleared rather than updated on failure, so a broken runtime is
        #: re-checked every time instead of being remembered as broken — the
        #: thing it was missing may since have been built.
        self._verified: tuple[str, float] | None = None

    @property
    def root(self) -> Path:
        return self._root

    # -- reading ---------------------------------------------------------------

    def _marker(self) -> Path:
        return self._root / CURRENT_MARKER

    def _installed(self) -> list[Path]:
        """Every directory that looks like a runtime, current or not."""
        if not self._root.is_dir():
            return []
        return sorted(p for p in self._root.iterdir() if p.is_dir())

    def _current(self) -> Path | None:
        try:
            name = self._marker().read_text(encoding="utf-8").strip()
        except OSError:
            return None
        candidate = self._root / name
        # The marker names a directory *inside* the store and nothing else: a
        # marker carrying "../.." must not turn remove() into a surprise.
        if not name or Path(name).name != name or not candidate.is_dir():
            return None
        return candidate

    def status(self) -> RuntimeStatus:
        """What the store is, without spawning anything.

        The health check ran at install time and its result is in the manifest;
        re-running it here would mean a subprocess every time the trainer ticks.
        What is checked instead is what can change while nobody is looking — the
        marker, the manifest's integrity, and whether the interpreter is still
        there.
        """
        current = self._current()
        if current is None:
            leftover = self._installed()
            if leftover:
                return RuntimeStatus(
                    INCOMPLETE,
                    f"an unfinished install is taking up space in {self._root}",
                )
            return RuntimeStatus(ABSENT, "no training runtime is installed yet")

        manifest = _verified(current / MANIFEST_NAME)
        if manifest is None:
            return RuntimeStatus(BROKEN, f"{current.name} has no usable manifest")

        interpreter = venv_python(current, platform=self._platform)
        backend = str(manifest.get("backend", ""))
        if not interpreter.exists():
            return RuntimeStatus(
                BROKEN, f"{current.name} has lost its interpreter", backend=backend
            )

        torch_version = manifest.get("torch")
        device = manifest.get("device")
        return RuntimeStatus(
            READY,
            f"{backend_for(backend).label} — torch {torch_version} on {device}",
            python=interpreter,
            backend=backend,
            torch_version=None if torch_version is None else str(torch_version),
            device=None if device is None else str(device),
        )

    def python(self) -> Path | None:
        """The interpreter to train with, or ``None`` if there is not one."""
        return self.status().python

    def verify(self, *, force: bool = False) -> RuntimeStatus:
        """:meth:`status`, but having actually asked the runtime to prove it.

        :meth:`status` believes the manifest and checks that the interpreter file
        exists. That is right for polling — the trainer asks it once a second —
        and wrong for deciding whether to offer Start, because everything it
        checks can be true of a runtime that no longer works: a torch deleted to
        reclaim disk, a CUDA driver downgraded under it, a binding rebuilt for a
        different Python. The trainer then shows Start, the button appears to do
        nothing, and the failure surfaces somewhere unrelated.

        So this runs the same health script the install had to pass. It costs a
        subprocess and an ``import torch``, which is why the result is cached
        against the manifest it was measured on: re-verifying is free until
        something actually installs, repairs or removes a runtime.
        """
        status = self.status()
        if not status.ready or status.python is None:
            return status
        stamp = self._stamp(status)
        if not force and stamp is not None and self._verified == stamp:
            return status
        health = self._health(status.python, _silent)
        if not health.ok:
            self._verified = None
            return replace(status, state=BROKEN, detail=health.detail, python=None)
        self._verified = stamp
        return status

    def _stamp(self, status: RuntimeStatus) -> tuple[str, float] | None:
        """What a verification is remembered against, or ``None`` if nothing is.

        The interpreter it proved and the modification time of the manifest it
        was measured on — so installing, repairing or removing a runtime
        invalidates the result without anyone having to remember to say so.
        """
        if status.python is None:
            return None
        current = self._current()
        return (
            str(status.python),
            _mtime(None if current is None else current / MANIFEST_NAME),
        )

    # -- writing ---------------------------------------------------------------

    def install(
        self,
        plan: RuntimePlan,
        *,
        on_output: Callable[[str], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> RuntimeStatus:
        """Carry out ``plan``, and return what the store is afterwards.

        Raises :class:`UnsafeIndex` before touching the disk — a bad index is a
        programming error rather than a state the user can be in, so it is the
        one failure here that is an exception instead of a status.

        Everything after that is transactional: the work happens in ``plan.target``
        and the marker is moved only once the interpreter has proved it can import
        both torch and the binding. Any failure, and any cancellation, deletes
        that directory and leaves whatever was current exactly as it was.
        """
        check_index(plan.index_url)
        emit = _silent if on_output is None else on_output
        # Asked before anything is downloaded, because the answer cannot change
        # by downloading. The binding is compiled from this checkout and is not
        # part of a runtime at all; pip has no way to supply it. Without it the
        # health check at the end of this method is certain to fail, and failing
        # deletes the directory — so leaving this until then spends five
        # gigabytes of CUDA torch to learn a fact that was knowable in a
        # millisecond, and then throws the torch away too. That is not
        # hypothetical: it happened to a checkout whose `.so` had been cleaned.
        if (missing := _missing_binding(plan.python, self._runner)) is not None:
            return replace(self.status(), detail=missing)
        # Annotated, or mypy reads the bare lambda as untyped and every call
        # through it becomes an untyped call in a typed context.
        stop: Callable[[], bool] = (lambda: False) if cancel is None else cancel
        interpreter = venv_python(plan.target, platform=self._platform)

        self._root.mkdir(parents=True, exist_ok=True)
        steps: tuple[tuple[str, list[str]], ...] = (
            ("create the environment", [str(plan.python), "-m", "venv", str(plan.target)]),
            (
                "install torch",
                [
                    str(interpreter),
                    "-m",
                    "pip",
                    "install",
                    "--index-url",
                    plan.index_url,
                    *plan.packages,
                ],
            ),
        )

        for what, command in steps:
            if stop():
                return self._abandon(plan.target, "install cancelled")
            if self._runner(command, emit) != 0:
                return self._abandon(plan.target, f"could not {what} — see the log")

        if stop():
            return self._abandon(plan.target, "install cancelled")
        health = self._health(interpreter, emit)
        if not health.ok:
            return self._abandon(plan.target, health.detail)

        self._write_manifest(plan, health)
        self._switch(plan.target)
        installed = self.status()
        # An install *is* a verification — a fresher one than :meth:`verify`
        # could produce, having just run the same script on the same
        # interpreter. Recorded so the dialog can ask what it now offers without
        # a second cold CUDA start, which after a five-minute download reads as
        # the window having hung at the finish line.
        self._verified = self._stamp(installed)
        return installed

    def repair(self) -> None:
        """Clear whatever is in the way, so an install can be tried again.

        It deletes rather than mends. A runtime is a pip install away from being
        recreated exactly, and the failure modes — a lost interpreter, a partial
        download, a manifest that no longer matches — are all cases where trusting
        a repair in place means trusting the thing that already broke once.
        """
        self.remove()

    def remove(self) -> None:
        """Delete every installed runtime and the marker. Never the root itself,
        which may be a directory the user chose and put other things in."""
        self._marker().unlink(missing_ok=True)
        for directory in self._installed():
            shutil.rmtree(directory, ignore_errors=True)
        self._verified = None  # there is nothing left that it was about

    # -- internals -------------------------------------------------------------

    def _abandon(self, target: Path, detail: str) -> RuntimeStatus:
        """Remove the directory this install was filling, and report why."""
        shutil.rmtree(target, ignore_errors=True)
        return replace(self.status(), detail=detail)

    def _health(self, interpreter: Path, emit: Callable[[str], None]) -> Health:
        lines: list[str] = []

        def record(line: str) -> None:
            lines.append(line)
            emit(line)

        code = self._runner([str(interpreter), "-c", HEALTH_SCRIPT], record)
        payload = _last_json(lines)
        if code != 0 or payload is None:
            return Health(False, _why_unhealthy(lines))
        if not payload.get("native"):
            return Health(False, "the installed runtime could not import the simulation binding")
        return Health(
            True,
            "ready",
            torch_version=str(payload.get("torch")),
            device=str(payload.get("device")),
        )

    def _write_manifest(self, plan: RuntimePlan, health: Health) -> None:
        manifest = _sign(
            {
                "schema": MANIFEST_SCHEMA,
                "backend": plan.backend,
                "index_url": plan.index_url,
                "packages": list(plan.packages),
                "torch": health.torch_version,
                "device": health.device,
            }
        )
        (plan.target / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _switch(self, target: Path) -> None:
        """Make ``target`` the current runtime, atomically."""
        temporary = self._root / f"{CURRENT_MARKER}.new"
        temporary.write_text(target.name, encoding="utf-8")
        os.replace(temporary, self._marker())
