# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""C++ quality tooling: clang-format, clang-tidy, and a build clean."""

from __future__ import annotations

import os
import shutil
import sys

from . import _util

CPP_DIRS: tuple[str, ...] = ("core", "app", "agent", "replay", "bindings")


def format_sources(*, fix: bool) -> None:
    files = _util.cpp_files(CPP_DIRS)
    if not files:
        return
    clang_format = _util.tool("clang-format-21", "clang-format")
    flags = ["-i"] if fix else ["--dry-run", "--Werror"]
    _util.run([clang_format, *flags, *files])


def tidy() -> None:
    # `module.cpp` is excluded: it is nanobind glue, and it only exists in the
    # compile database when the optional bindings are configured.
    clang_tidy = _util.tool("clang-tidy-21", "clang-tidy")
    files = _util.cpp_files(
        ("core", "app", "agent", "replay", "bindings"),
        exts=("cpp",),
        exclude=("/tests/", "miniaudio_impl", "module"),
    )

    # One clang-tidy process analysed all eighteen translation units in series —
    # 141 seconds on a 32-core machine, and the single largest thing in `poe
    # check` by a wide margin. clang-tidy has no parallelism of its own; the
    # answer is `run-clang-tidy`, which ships with LLVM and forks one process per
    # file over the same compile database. Measured here: 141s -> 18s, and the
    # remainder is one file (`app/game_window.cpp` alone takes 17.7s), so more
    # cores buy nothing beyond this.
    #
    # It is also the step that never gets cheaper on its own: nothing about it is
    # incremental, so every gate run re-analyses all eighteen files even when the
    # change was to a Python module.
    #
    # Optional rather than required, because it is a *speed* choice and the gate
    # must not stop working on a machine that ships clang-tidy without it. The
    # serial call below stays as the fallback, and the two are equivalent in what
    # they accept and reject — `test_quality.py` holds them to that.
    jobs = os.cpu_count() or 1
    parallel = _util.tool_optional("run-clang-tidy-21", "run-clang-tidy")
    if parallel is not None:
        # `-clang-tidy-binary` is spelled out rather than left to the default
        # search: `run-clang-tidy` looks for a bare `clang-tidy`, which on a
        # machine that only has the versioned name is a different tool or none.
        _util.run(
            [
                parallel,
                "-p",
                "build/debug",
                "-clang-tidy-binary",
                clang_tidy,
                "-quiet",
                "-warnings-as-errors=*",
                "-j",
                str(jobs),
                *files,
            ]
        )
        return
    _util.run([clang_tidy, "-p", "build/debug", "--warnings-as-errors=*", *files])


def clean() -> None:
    build = _util.PROJECT_ROOT / "build"
    if build.exists():
        shutil.rmtree(build)
    print(f"removed {build}")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    command = args[0] if args else ""
    if command == "format":
        format_sources(fix=True)
    elif command == "format-check":
        format_sources(fix=False)
    elif command == "tidy":
        tidy()
    elif command == "clean":
        clean()
    else:
        print("usage: python -m tools.quality {format|format-check|tidy|clean}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
