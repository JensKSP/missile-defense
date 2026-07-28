# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Run one of the project's Python entry points in an interpreter that can.

This repo deliberately has more than one Python. The game is built with the
MSYS2 toolchain and the development venv follows it, but neither torch nor
PySide6 publishes a MinGW wheel — so the training loop and the trainer live in a
*native* interpreter beside it (docs/WINDOWS.md). `poe train` and `poe ui` used
whichever `python` came first on PATH, which on such a machine is the one that
can run neither, and the failure surfaced as an ImportError from inside a module.
That reads as a broken checkout rather than as the wrong interpreter, which is
the actual problem and a much easier one to fix.

So: find an interpreter that has what the module needs, and if there is none,
say which interpreters were looked at and what would fix each. Both halves
matter — the search is what makes it work on the machine that has the packages
somewhere, and the report is what makes it explicable on the machine that does
not have them anywhere.

Everywhere else this is a no-op: on a box with one Python and everything
installed, the first candidate is the running interpreter and it wins.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._util import PROJECT_ROOT

#: What each entry point needs before it can even start, and what installs it.
#: Import names, because that is what `find_spec` takes; the pip name is beside
#: it because they are not always the same word and the message quotes it.
#:
#: **numpy is here because it is a base dependency, not an extra**, and this
#: table is what decides whether an interpreter counts as able to run the entry
#: point at all. Listing only the headline package was enough for anyone who got
#: here through `pip install`, which brings numpy along — and wrong for everyone
#: who got the trainer from the Windows ZIP, where the payload is copied beside
#: the game and nothing resolves dependencies. On 2026-07-28 such an interpreter
#: was picked as usable, started the trainer, and died in `missile_defense.sim.policy_format` on
#: `import numpy`. An entry point's requirements are what it *imports*, not what
#: distinguishes it from the other entry point.
#:
#: `missile_defense._md_native` is deliberately absent. It is not a package any `pip install`
#: produces — `poe bindings` builds it — so naming it here would put a command
#: in the message that cannot work. The trainer, which truly cannot run without
#: it, gets it as a build dependency of the poe task instead.
REQUIREMENTS: dict[str, tuple[tuple[str, str], ...]] = {
    "missile_defense.training": (("torch", "torch"), ("numpy", "numpy")),
    "missile_defense.ui": (("PySide6", "PySide6"), ("numpy", "numpy")),
}

#: Asks an interpreter what it has, in one round trip: the version, then any of
#: the named modules it cannot find. `find_spec` locates a module without
#: importing it, which keeps the probe fast and side-effect free — importing
#: torch to ask whether torch is there costs seconds.
PROBE = """
import importlib.util as u, sys
missing = []
for name in sys.argv[1:]:
    try:
        found = u.find_spec(name) is not None
    except Exception:
        found = False   # a namespace package with a broken parent, and the like
    if not found:
        missing.append(name)
print(sys.version.split()[0], *missing)
sys.exit(1 if missing else 0)
"""

#: A probe is an interpreter start-up, so the search must stay short.
PROBE_TIMEOUT_S = 20.0

#: Long enough for the trainer to finish its update and write a final checkpoint
#: after a Ctrl-C, which is the whole reason this waits rather than exiting.
SHUTDOWN_GRACE_S = 30.0


@dataclass(frozen=True)
class Report:
    """What one candidate interpreter turned out to be."""

    python: str
    version: str
    missing: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return not self.missing


#: Injected so the search itself is testable without a machine that happens to
#: have the right interpreters lying around.
Probe = Callable[[str, Sequence[str]], Report | None]


def _probe(python: str, modules: Sequence[str]) -> Report | None:
    """Ask ``python`` about ``modules``, or ``None`` if it will not run at all."""
    try:
        finished = subprocess.run(
            [python, "-c", PROBE, *modules],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        # A stale `py -0p` entry, a dangling symlink, an interpreter that dies on
        # start-up: not a candidate, and not worth a word to the reader either.
        return None
    words = finished.stdout.split()
    if not words:
        return None
    return Report(python=python, version=words[0], missing=tuple(words[1:]))


def candidates(
    environ: Mapping[str, str] | None = None,
    *,
    platform: str = sys.platform,
    executable: str | None = None,
) -> list[str]:
    """Interpreters worth asking, best first, without duplicates.

    ``MD_PYTHON`` first because an explicit answer ends the question — it is the
    same variable the trainer already uses to pick the interpreter it starts a
    run with (``missile_defense.runs.runner.training_python``), so there is one thing to set
    rather than one per tool. Then the interpreter already running, so a machine
    with a single Python never pays for the search or gets surprised by it.
    Everything after that is discovery.
    """
    env = os.environ if environ is None else environ
    found: list[str] = []
    override = env.get("MD_PYTHON")
    if override:
        found.append(override)
    found.append(sys.executable if executable is None else executable)
    if platform == "win32":
        # The launcher knows about every interpreter installed the ordinary way,
        # including the ones deliberately kept off PATH.
        found.extend(_py_launcher_entries())
        # ... when it can be reached at all, which is not everywhere: `py` is not
        # on PATH in every shell, and where it is missing the launcher answers
        # nothing and an interpreter that can train goes unseen. The registry
        # holds the same list without the intermediary, and is where `py` itself
        # reads it from.
        #
        # Both are a fallback rather than the everyday path now: an installed
        # trainer has to find an interpreter it did not create, but a checkout
        # runs `poe` from its own venv, which is already the answer.
        found.extend(_registry_entries())
    for name in ("python3", "python"):
        on_path = shutil.which(name, path=env.get("PATH"))
        if on_path:
            found.append(on_path)
    return _unique(found)


def _py_launcher_entries() -> list[str]:
    """Paths from ``py -0p``, newest first — the Windows launcher's own list."""
    try:
        finished = subprocess.run(
            ["py", "-0p"], capture_output=True, text=True, timeout=PROBE_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [path for path in map(parse_launcher_line, finished.stdout.splitlines()) if path]


def _registry_entries() -> list[str]:
    """Every interpreter Windows has on record, per PEP 514.

    ``Software\\Python\\<Company>\\<Tag>`` under both hives, and both the
    ``ExecutablePath`` value and the install directory, because a distribution
    may set either. Anything unreadable is skipped rather than raised on: this is
    discovery, and one broken vendor key must not take the whole search with it.
    """
    try:
        import winreg  # noqa: PLC0415 — Windows only, and only on this path
    except ImportError:
        return []
    return _hive_entries(winreg)


def _hive_entries(winreg: Any) -> list[str]:
    """Walk both hives of ``Software\\Python``, as the module is handed in.

    ``winreg`` arrives as a parameter, and as ``Any``, because typeshed declares
    the module only for ``sys.platform == "win32"``: used directly, every name
    below is an "unknown attribute" on the Linux machine the gate runs on, which
    is where this went red. Hiding the block behind a ``sys.platform`` check
    would make it unreachable there and so type-checked on no machine CI has;
    this way only the registry API itself is taken on trust, and the walk around
    it — the loops, the string handling — is still checked everywhere.
    """

    def children(key: Any) -> list[str]:
        names: list[str] = []
        for index in range(1024):  # a bound, not an expectation
            try:
                names.append(str(winreg.EnumKey(key, index)))
            except OSError:
                break
        return names

    found: list[str] = []
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, r"Software\Python") as companies:
                for company in children(companies):
                    with winreg.OpenKey(companies, company) as tags:
                        for tag in children(tags):
                            found.extend(_registered_interpreter(winreg, tags, tag))
        except OSError:
            continue
    return found


def _registered_interpreter(winreg: Any, tags: Any, tag: str) -> list[str]:
    """The interpreter one PEP 514 tag points at, as zero, one or two paths."""
    try:
        with winreg.OpenKey(tags, rf"{tag}\InstallPath") as key:
            paths: list[str] = []
            try:
                executable, _ = winreg.QueryValueEx(key, "ExecutablePath")
                paths.append(str(executable))
            except OSError:
                pass
            try:
                directory, _ = winreg.QueryValueEx(key, "")
                paths.append(str(Path(str(directory)) / "python.exe"))
            except OSError:
                pass
            return paths
    except OSError:
        return []


def parse_launcher_line(line: str) -> str:
    """The interpreter path out of one ``py -0p`` row, or ``""`` for anything else.

    A row is ``-V:3.13 *   C:\\...\\python.exe``: a version tag, a ``*`` on the
    default, then the path. Split once and keep the whole remainder rather than
    taking the last word — ``C:\\Program Files\\Python313\\python.exe`` has a
    space in it, and a version-banner line has no tag at all.
    """
    parts = line.split(maxsplit=1)
    if len(parts) != 2 or not parts[0].startswith("-V:"):
        return ""
    rest = parts[1].strip()
    if rest.startswith("*"):
        rest = rest[1:].strip()
    return rest


def _unique(paths: Iterable[str]) -> list[str]:
    """Same file twice is one candidate — `python3` is usually a link."""
    seen: dict[str, str] = {}
    for path in paths:
        try:
            key = str(Path(path).resolve())
        except OSError:
            key = path
        seen.setdefault(key, path)
    return list(seen.values())


def survey(
    modules: Sequence[str], *, probe: Probe = _probe, pythons: Sequence[str] | None = None
) -> list[Report]:
    """Every candidate and what it has, stopping at the first that has everything.

    Stopping early is why the usual case costs nothing: on a machine where the
    running interpreter is the right one, exactly one probe happens. The rest of
    the list is only ever walked when something is genuinely missing, which is
    also when a full account of what was looked at earns its cost.
    """
    reports: list[Report] = []
    for python in candidates() if pythons is None else pythons:
        report = probe(python, modules)
        if report is None:
            continue
        reports.append(report)
        if report.usable:
            break
    return reports


def explain(module: str, requirements: Sequence[tuple[str, str]], reports: Sequence[Report]) -> str:
    """Why nothing here can run ``module``, and the command that changes that.

    Named packages and a real interpreter path to install them *into*, because
    the failure being described is precisely that this machine has several
    Pythons and the obvious one is the wrong one. "Install the dependencies"
    would be advice the reader has already tried.
    """
    packages = " ".join(package for _, package in requirements)
    task = "ui" if module == "missile_defense.ui" else "train"
    lines = [f"`{module}` needs {packages}, and no interpreter here has it."]
    if reports:
        width = max(len(report.python) for report in reports)
        lines.append("")
        lines.append("Looked at:")
        lines += [
            f"  {report.python:<{width}}  {report.version:<8}  no {', '.join(report.missing)}"
            for report in reports
        ]
    example = reports[0].python if reports else sys.executable
    lines += [
        "",
        "Install into whichever of those is the native build's interpreter:",
        f"    {example} -m pip install {packages}",
        "",
        "or point MD_PYTHON at one that already has them:",
        f"    MD_PYTHON=/path/to/python poe {task}",
    ]
    return "\n".join(lines)


def child_environ(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The child's environment: this checkout's ``python/`` on the import path.

    The chosen interpreter is a stranger to this source tree — it was picked for
    having torch or PySide6, not for being the venv — so nothing has put ``md``
    where it can find it.
    """
    env = dict(os.environ if environ is None else environ)
    package = str(PROJECT_ROOT / "python")
    existing = env.get("PYTHONPATH", "")
    if package not in existing.split(os.pathsep):
        env["PYTHONPATH"] = f"{package}{os.pathsep}{existing}" if existing else package
    return env


def forwarded(args: Sequence[str]) -> list[str]:
    """Drop poe's separator, which is not an argument to anything.

    `poe train -- --updates 20` is what the docs say and what the task's own
    help implies, and poe passes the `--` straight through. argparse then treats
    it as "everything after this is positional" and rejects `--updates` as an
    unrecognised argument — so the documented way to pass a flag failed, and
    failed inside the trainer where it read as the trainer's fault.
    """
    return list(args[1:]) if args and args[0] == "--" else list(args)


def run(python: str, module: str, args: Sequence[str]) -> int:
    """Run ``module`` and hand back its exit code — Ctrl-C included.

    A Ctrl-C in a terminal reaches the child as well as this process, and the
    trainer answers it by finishing its update. Exiting the moment the signal
    arrives would leave that half-done work orphaned and unreported, so this
    waits for the child rather than racing it.
    """
    command = [python, "-u", "-m", module, *args]
    process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=child_environ())
    try:
        return process.wait()
    except KeyboardInterrupt:
        try:
            return process.wait(timeout=SHUTDOWN_GRACE_S)
        except subprocess.TimeoutExpired:
            process.terminate()
            return 130


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        raise SystemExit(f"usage: python -m tools.launch <module> [args…]  ({_known()})")
    module, rest = args[0], forwarded(args[1:])
    requirements = REQUIREMENTS.get(module, ())

    reports = survey([name for name, _ in requirements])
    if not reports or not reports[-1].usable:
        print(explain(module, requirements, reports), file=sys.stderr)
        return 1

    chosen = reports[-1].python
    if len(reports) > 1:
        # Only when it is not the obvious one: a line of noise before every run
        # is how a useful message stops being read.
        print(f"+ {module} needs a different interpreter: {chosen}", file=sys.stderr)
    return run(chosen, module, rest)


def _known() -> str:
    return "known modules: " + ", ".join(sorted(REQUIREMENTS))


if __name__ == "__main__":
    raise SystemExit(main())
