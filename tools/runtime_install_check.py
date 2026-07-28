# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Install a CPU training runtime the way the trainer does, and fail if it breaks.

Run with::

    python -m tools.runtime_install_check

The managed runtime is the one path nothing else in CI could see. The health
check that decides whether an install counts runs *inside* the interpreter pip
has just created, and that interpreter reaches ``missile_defense`` only through
``PYTHONPATH`` — a bare venv has none of this checkout's ``.pth`` files. So a
regression in that wiring is invisible to every other job, which imports the
package through the development venv and passes regardless, and it reaches a
person only at the end of a multi-gigabyte download.

CPU rather than an accelerator: it is the one backend that exists on all three
platforms (``runtime.CPU.platforms``), it needs no GPU on the runner, and what
is under test here is the mechanism rather than the speed.

The binding has to be built beside the package first (``poe bindings``), because
the runtime installer refuses to start without it — pip cannot supply a compiled
extension that is not on any index.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import missile_defense.runs.runtime as runtime


def check(root: Path, *, backend_name: str = "cpu") -> int:
    """Install a runtime into ``root`` and report whether it came up healthy."""
    backend = runtime.backend_for(backend_name)
    if sys.platform not in backend.platforms:
        print(f"{backend.label} does not exist for {sys.platform}; nothing to check")
        return 0

    plan = runtime.RuntimePlan(
        backend=backend.name,
        # This interpreter: the installer probes it for the binding before it
        # downloads anything, and creates the runtime venv from it.
        python=Path(sys.executable),
        target=root / f"{backend.name}-check",
        packages=backend.packages,
        index_url=backend.index_url,
    )
    print(f"installing {backend.label} from {plan.index_url} into {plan.target}")
    status = runtime.Runtime(root).install(plan, on_output=print)

    if status.state != runtime.READY:
        print(f"::error::the managed runtime did not come up: {status.detail}", file=sys.stderr)
        return 1
    print(f"runtime ready: {status.detail}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install a CPU training runtime the way the trainer does."
    )
    parser.add_argument(
        "--backend",
        default="cpu",
        help="which backend to install (default: cpu, the only one on all three platforms)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="where to put it (default: a temporary directory, removed afterwards)",
    )
    args = parser.parse_args(argv)

    if args.root is not None:
        args.root.mkdir(parents=True, exist_ok=True)
        return check(args.root, backend_name=args.backend)
    # A directory of its own, so a CI run never adopts or overwrites the
    # developer runtime that may already be on the machine.
    with tempfile.TemporaryDirectory(prefix="md-runtime-check-") as tmp:
        return check(Path(tmp), backend_name=args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
