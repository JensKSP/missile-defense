# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The quality gate, run concurrently, with one log per stage.

``poe check`` used to be a serial sequence of eleven tasks. On a 32-core machine
that spent most of its wall time on one core: the static checks are
single-threaded and add up to about thirteen seconds, and `clang-tidy` — which
has no parallelism of its own — took 141 of them by itself.

Two things this fixes, and the second is the reason it is a module rather than a
longer `sequence` in pyproject.toml.

**Concurrency.** Stages that need nothing from each other start together. The
ones that do share something declare it as a *resource* rather than as an edge in
a graph: `tidy` and `test` both drive `build/debug`, and two `ninja` runs in one
build directory corrupt it. Holding a lock per build tree is enough, and it lets
the stage list stay a flat table that anyone can add a line to.

**Output that survives being read by a machine.** Every stage writes its own
`build/gate/<stage>.log`, and the run writes `build/gate/summary.json`. There is
exactly one place that says whether the gate passed. That matters more than it
sounds: interleaved output from eleven concurrent stages is unreadable, and the
serial version was worse than unreadable — it ended with whatever the last stage
happened to print, so `poe check | tail` showed an inner stage's cheerful "All
checks passed!" for a run that had failed. A summary you cannot mistake for
progress chatter is the point.

Stages are named `poe` tasks and are run through it, so their definitions stay in
pyproject.toml and exist once. That costs an interpreter start per stage, which
is hidden by running them at the same time.

:data:`STAGES` and the `check-serial` task are two records of the same thing, and
`test_tools_gate.py` holds them to each other — a stage dropped from here would
otherwise make the gate faster *and* green while no longer checking anything.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import _util

#: Where the per-stage logs and the summary go. Under `build/` because it is
#: generated and already ignored; a directory of its own because "the log of the
#: last gate run" is a thing you want to be able to delete wholesale.
GATE_DIR = _util.PROJECT_ROOT / "build" / "gate"

#: How many lines of a failing stage's log are echoed at the end. Enough to carry
#: a compiler diagnostic or a pytest assertion, short enough that three failures
#: do not bury the summary that follows them.
TAIL_LINES = 25


@dataclass(frozen=True)
class Stage:
    """One `poe` task, and the build tree it needs to itself.

    ``resource`` is not a dependency: nothing here waits for another stage's
    *result*. It is exclusive access to a directory, because `cmake --build` in
    two processes over one build directory is a race with no diagnostic — it
    produces a corrupt tree and a confusing failure somewhere else.

    ``weight`` only orders the queue. Longest first, so the slowest stage starts
    at second zero instead of last; with `tidy` at eighteen seconds and
    everything else under eight, starting it late would add its whole duration to
    the run.
    """

    task: str
    resource: str | None = None
    weight: int = 1


#: The gate. Order here is documentation; `weight` decides what actually starts
#: first.
#:
#: `coverage` configures and builds a third tree of its own (`build/coverage`),
#: so it shares nothing with the other two and needs no lock beyond its own.
STAGES: tuple[Stage, ...] = (
    Stage("tidy", resource="debug", weight=100),
    Stage("test", resource="debug", weight=30),
    Stage("test-release", resource="release", weight=40),
    Stage("coverage", resource="coverage", weight=35),
    Stage("pytest", weight=20),
    Stage("typecheck", weight=10),
    Stage("format-check", weight=3),
    Stage("vulkan-shaders", weight=3),
    Stage("protocol-check", weight=2),
    Stage("fmt-py-chk", weight=1),
    Stage("lint", weight=1),
)


@dataclass
class Result:
    stage: str
    status: str
    exit_code: int
    duration_ms: int
    log: str
    #: The last lines of a failing stage's log. Always passed, never defaulted:
    #: an empty default would make "no tail captured" and "nothing to say"
    #: indistinguishable at the one place this is read.
    tail: list[str]


def _run(stage: Stage, locks: dict[str, threading.Lock]) -> Result:
    log_path = GATE_DIR / f"{stage.task}.log"
    lock = locks[stage.resource] if stage.resource else None
    if lock is not None:
        lock.acquire()
    start = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [sys.executable, "-m", "poethepoet", stage.task],
                cwd=str(_util.PROJECT_ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    finally:
        if lock is not None:
            lock.release()
    duration = int((time.monotonic() - start) * 1000)
    ok = completed.returncode == 0
    tail: list[str] = []
    if not ok:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = lines[-TAIL_LINES:]
    status = "pass" if ok else "fail"
    print(
        f"{'PASS' if ok else 'FAIL'}  {stage.task:<14} {duration / 1000:6.1f}s"
        + ("" if ok else f"  -> {log_path.relative_to(_util.PROJECT_ROOT)}"),
        file=sys.stderr,
        flush=True,
    )
    return Result(stage.task, status, completed.returncode, duration, str(log_path), tail)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    only = set(args) if args else None
    stages = [s for s in STAGES if only is None or s.task in only]
    if only is not None and not stages:
        print(f"no such stage: {', '.join(sorted(only))}", file=sys.stderr)
        return 2

    GATE_DIR.mkdir(parents=True, exist_ok=True)
    for stale in GATE_DIR.glob("*.log"):
        stale.unlink()

    locks = {name: threading.Lock() for name in {s.resource for s in stages if s.resource}}
    ordered = sorted(stages, key=lambda s: -s.weight)

    started = time.monotonic()
    # One worker per stage: they are mostly waiting on child processes that
    # parallelise internally, and the build-tree locks are what actually limit
    # how much compiling happens at once.
    with ThreadPoolExecutor(max_workers=len(ordered)) as pool:
        futures: list[Future[Result]] = [pool.submit(_run, s, locks) for s in ordered]
        results = [f.result() for f in futures]
    elapsed = int((time.monotonic() - started) * 1000)

    results.sort(key=lambda r: -r.duration_ms)
    failed = [r for r in results if r.status != "pass"]
    summary = {
        "passed": not failed,
        "elapsed_ms": elapsed,
        "stages": [
            {
                "stage": r.stage,
                "status": r.status,
                "exit_code": r.exit_code,
                "duration_ms": r.duration_ms,
                "log": str(Path(r.log).relative_to(_util.PROJECT_ROOT)),
            }
            for r in results
        ],
    }
    (GATE_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # The failures first and in full, because that is what the run is for. The
    # summary goes last so it is what a terminal is left showing.
    for result in failed:
        print(f"\n--- {result.stage} ---", file=sys.stderr)
        for line in result.tail:
            print(line, file=sys.stderr)

    print("\n" + "=" * 52, file=sys.stderr)
    for result in results:
        print(
            f"{result.status.upper():<5} {result.stage:<14} {result.duration_ms / 1000:6.1f}s",
            file=sys.stderr,
        )
    print("=" * 52, file=sys.stderr)
    verdict = "GATE PASSED" if not failed else f"GATE FAILED ({len(failed)} of {len(results)})"
    print(f"{verdict} in {elapsed / 1000:.1f}s", file=sys.stderr)
    print(f"logs: {GATE_DIR.relative_to(_util.PROJECT_ROOT)}/", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
