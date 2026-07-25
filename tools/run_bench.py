# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Build (if needed) and run the throughput benchmark.

``poe bench`` measures the release build — the numbers that mean anything.
``poe profile`` builds the ``profile`` preset instead (same optimisation, plus
frame pointers so an external sampler can unwind stacks) and runs a long steady
workload to attach ``perf`` / WPA / VTune to. There is no in-code phase timing:
``Sim::step`` reads no clock, by design.

Extra arguments pass straight through, e.g. ``poe bench --threads 4 --csv``.
"""

from __future__ import annotations

import subprocess
import sys

from . import _util


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    profiling = "--profile" in args
    if profiling:
        # Selects the build, not a binary flag: --sample is what md_bench takes.
        args = [a for a in args if a != "--profile"]
        args.append("--sample")
    preset = "profile" if profiling else "release"
    binary = _util.PROJECT_ROOT / "build" / preset / "bench" / f"md_bench{_util.EXE}"
    if not binary.exists():
        _util.run(["cmake", "--preset", preset])
        _util.run(["cmake", "--build", "--preset", preset])
    return subprocess.call([str(binary), *args])


if __name__ == "__main__":
    raise SystemExit(main())
