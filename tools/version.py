# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Check that the version is spelled the same in every file that carries one.

Three files declare it, independently, and nothing makes them agree:

    CMakeLists.txt    project(... VERSION x.y.z)      the binary, and CPack's
                                                      .dmg / .exe / .zip names
    pyproject.toml    version = "x.y.z"               the Python package
    debian/changelog  missile-defense (x.y.z-r)       the .deb

A release publishes artifacts built from all three at once, so a disagreement
does not fail anything — it ships a disk image and a .deb that claim to be
different versions of the same program, which is discovered by a user. Run it
before tagging:

    poe version            # do the three agree?
    poe version v0.1.0     # and do they agree with the tag I am about to push?
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from . import _util

#: (file, regex). One capture group, holding the version. Anchored per file
#: rather than searched for loosely: `cmake_minimum_required(VERSION 3.25)` is a
#: version too, and matching it instead would be a silent wrong answer.
SOURCES: tuple[tuple[str, str], ...] = (
    ("CMakeLists.txt", r"^\s*VERSION\s+(\d+\.\d+\.\d+)\s*$"),
    ("pyproject.toml", r'^version\s*=\s*"(\d+\.\d+\.\d+)"'),
    ("debian/changelog", r"^missile-defense\s+\((\d+\.\d+\.\d+)-\d+\)"),
)


def read_versions(root: Path | None = None) -> dict[str, str]:
    """Map each declaring file to the version it declares."""
    base = _util.PROJECT_ROOT if root is None else root
    found: dict[str, str] = {}
    for relative, pattern in SOURCES:
        path = base / relative
        if not path.is_file():
            raise SystemExit(f"error: {relative} is missing")
        match = re.search(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
        if match is None:
            raise SystemExit(f"error: no version found in {relative}")
        found[relative] = match.group(1)
    return found


def check(tag: str | None = None, *, root: Path | None = None) -> int:
    """Print what each file says; return 0 if they agree (with ``tag`` if given)."""
    versions = read_versions(root)
    width = max(len(name) for name in versions)
    for name, version in versions.items():
        print(f"  {name:<{width}}  {version}")

    distinct = set(versions.values())
    if len(distinct) != 1:
        print(
            f"FAIL: {len(distinct)} different versions declared: {', '.join(sorted(distinct))}",
            file=sys.stderr,
        )
        return 1

    declared = distinct.pop()
    if tag is not None:
        # `v0.1.0` is the tag convention; the files carry the bare number.
        wanted = tag[1:] if tag.startswith("v") else tag
        print(f"  {'tag':<{width}}  {wanted}")
        if wanted != declared:
            print(
                f"FAIL: the tag says {wanted}, the tree says {declared}. "
                f"Tagging this would publish artifacts named {declared}.",
                file=sys.stderr,
            )
            return 1

    print(f"OK: {declared}, everywhere.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    return check(args[0] if args else None)


if __name__ == "__main__":
    raise SystemExit(main())
