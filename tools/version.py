# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Check that the version is spelled the same in every file that carries one.

Four files declare it, independently, and nothing makes them agree:

    CMakeLists.txt      project(... VERSION x.y.z)    the binary, and CPack's
                                                      .dmg / .exe / .zip names
    pyproject.toml      version = "x.y.z"             the Python package
    debian/changelog    missile-defense (x.y.z-r)     the .deb
    python/md/__init__  __version__ = "x.y.z"         what `md` reports at run
                                                      time, and what the training
                                                      trainer's About box shows

A release publishes artifacts built from all four at once, so a disagreement
does not fail anything — it ships a disk image and a .deb that claim to be
different versions of the same program, which is discovered by a user. The
fourth is the one most likely to drift, because nothing builds from it: it is
read at run time, so a stale value is not a broken build but a trainer that
tells its user the wrong version of the thing they are running. Run it
before tagging:

    poe version            # do the three agree?
    poe version v0.1.0     # and do they agree with the tag I am about to push?
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import _util

#: (file, regex). One capture group, holding the version. Anchored per file
#: rather than searched for loosely: `cmake_minimum_required(VERSION 3.25)` is a
#: version too, and matching it instead would be a silent wrong answer.
SOURCES: tuple[tuple[str, str], ...] = (
    ("CMakeLists.txt", r"^\s*VERSION\s+(\d+\.\d+\.\d+)\s*$"),
    ("pyproject.toml", r'^version\s*=\s*"(\d+\.\d+\.\d+)"'),
    ("debian/changelog", r"^missile-defense\s+\((\d+\.\d+\.\d+)-\d+\)"),
    ("python/md/__init__.py", r'^__version__\s*=\s*"(\d+\.\d+\.\d+)"'),
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


@dataclass(frozen=True)
class DevVersion:
    """One nightly version, spelled the way each ecosystem needs it.

    The tree always declares the version being worked *toward*, so a build off
    master is a pre-release of it and must sort *below* it. Each packaging system
    spells "below" differently, and getting it wrong is not cosmetic: a nightly
    that outranks the release it precedes is one `apt upgrade` from being pinned
    on a user's machine.
    """

    base: str  #: what the tree declares, e.g. "0.1.0"
    distance: int  #: commits since the last release tag
    commit: str  #: abbreviated hash, "g"-prefixed as git describe writes it

    @property
    def semver(self) -> str:
        """`0.1.0-dev.37+g1a2b3c4` — a hyphen makes it a pre-release (SemVer §9)."""
        return f"{self.base}-dev.{self.distance}+{self.commit}"

    @property
    def pep440(self) -> str:
        """`0.1.0.dev37+g1a2b3c4` — `.devN` sorts before the release (PEP 440)."""
        return f"{self.base}.dev{self.distance}+{self.commit}"

    @property
    def debian(self) -> str:
        """`0.1.0~dev37+g1a2b3c4` — `~` is the one character that sorts before nothing."""
        return f"{self.base}~dev{self.distance}+{self.commit}"

    @property
    def filename(self) -> str:
        """`0.1.0-dev37-g1a2b3c4` — no `+`, which URLs decode as a space."""
        return f"{self.base}-dev{self.distance}-{self.commit}"


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def dev_version(root: Path | None = None) -> DevVersion:
    """Derive the nightly version for the checkout at ``root``.

    Never hand-written: the distance and the hash come from git, so two nightlies
    of the same tree cannot disagree and one of a different tree cannot collide.
    """
    base = root if root is not None else _util.PROJECT_ROOT
    declared = set(read_versions(base).values())
    if len(declared) != 1:
        raise SystemExit("error: the tree disagrees with itself about its version")
    try:
        # --long so the format is stable even when HEAD *is* the tag.
        described = _git(base, "describe", "--tags", "--match", "v[0-9]*", "--long", "--abbrev=8")
        _, distance, commit = described.rsplit("-", 2)
    except subprocess.CalledProcessError:
        # No release tag yet — count from the root commit instead, which keeps
        # nightlies ordered from the very first one rather than starting at zero
        # once the first tag lands.
        distance = _git(base, "rev-list", "--count", "HEAD")
        commit = "g" + _git(base, "rev-parse", "--short=8", "HEAD")
    return DevVersion(declared.pop(), int(distance), commit)


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


#: `poe version --dev <flavour>` prints one rendering and nothing else, so CI can
#: capture it. Names match the attributes on DevVersion.
FLAVOURS = ("semver", "pep440", "debian", "filename")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--dev":
        flavour = args[1] if len(args) > 1 else "semver"
        if flavour not in FLAVOURS:
            raise SystemExit(f"error: unknown flavour {flavour!r}; pick one of {FLAVOURS}")
        print(getattr(dev_version(), flavour))
        return 0
    return check(args[0] if args else None)


if __name__ == "__main__":
    raise SystemExit(main())
