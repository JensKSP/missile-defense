# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""LLVM source-based coverage for md::core, gated at a threshold percentage.

Builds the ``coverage`` CMake preset (core only, instrumented), runs the test
binaries, merges their profiles, reports line coverage of ``core/`` (production
sources only), and exits non-zero if it is below the threshold.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from typing import Any, cast

from . import _util

IGNORE: str = "-ignore-filename-regex=(tests/|_deps/|build/|catch2)"


def measure(threshold: float) -> int:
    profdata = _util.tool("llvm-profdata-21", "llvm-profdata")
    llvm_cov = _util.tool("llvm-cov-21", "llvm-cov")

    _util.run([sys.executable, "-m", "tools.cmake", "cmake", "--preset", "coverage"], capture=True)
    _util.run(
        [sys.executable, "-m", "tools.cmake", "cmake", "--build", "--preset", "coverage"],
        capture=True,
    )

    build = _util.PROJECT_ROOT / "build" / "coverage"
    unit = build / "core" / "tests" / f"md_core_unit_tests{_util.EXE}"
    e2e = build / "core" / "tests" / f"md_core_e2e_tests{_util.EXE}"

    prof = build / "prof"
    if prof.exists():
        shutil.rmtree(prof)
    prof.mkdir(parents=True)

    for label, binary in (("unit", unit), ("e2e", e2e)):
        env = dict(os.environ, LLVM_PROFILE_FILE=str(prof / f"{label}.profraw"))
        _util.run([str(binary)], env=env, capture=True)

    merged = prof / "merged.profdata"
    raws = sorted(str(p) for p in prof.glob("*.profraw"))
    _util.run([profdata, "merge", "-sparse", *raws, "-o", str(merged)])

    objects = ["-object", str(unit), "-object", str(e2e)]
    profile = f"-instr-profile={merged}"

    print()
    _util.run([llvm_cov, "report", *objects, profile, IGNORE])
    print()

    result = _util.run(
        [llvm_cov, "export", *objects, profile, IGNORE, "-summary-only"],
        capture=True,
        quiet=True,
    )
    summary = cast(dict[str, Any], json.loads(result.stdout))
    percent = float(summary["data"][0]["totals"]["lines"]["percent"])

    print(f"Total line coverage: {percent:.2f}% (threshold {threshold:g}%)")
    if percent + 1e-9 >= threshold:
        print("OK: coverage gate passed.")
        return 0
    print(
        f"FAIL: line coverage {percent:.2f}% is below the {threshold:g}% threshold.",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    threshold = float(args[0]) if args else 80.0
    return measure(threshold)


if __name__ == "__main__":
    raise SystemExit(main())
