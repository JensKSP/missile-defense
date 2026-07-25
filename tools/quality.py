# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""C++ quality tooling: clang-format, clang-tidy, and a build clean."""

from __future__ import annotations

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
