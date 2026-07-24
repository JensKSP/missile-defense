# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Shared helpers for the developer tooling: process running + file discovery."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
EXE: str = ".exe" if os.name == "nt" else ""


def tool(*names: str) -> str:
    """Return the first of ``names`` found on PATH, or exit with a clear message.

    Handles the Debian ``-21`` suffix vs. the plain name on other platforms.
    """
    found = tool_optional(*names)
    if found is None:
        raise SystemExit(f"error: none of these tools are on PATH: {', '.join(names)}")
    return found


def tool_optional(*names: str) -> str | None:
    """Return the first of ``names`` found on PATH, or ``None``."""
    for name in names:
        found = shutil.which(name)
        if found is not None:
            return found
    return None


def run(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    capture: bool = False,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command (no shell) from the project root by default."""
    if not quiet:
        print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.run(
        list(cmd),
        cwd=str(cwd if cwd is not None else PROJECT_ROOT),
        env=dict(env) if env is not None else None,
        check=check,
        text=True,
        capture_output=capture,
    )


def cpp_files(
    dirs: Sequence[str],
    *,
    exts: Sequence[str] = ("cpp", "hpp"),
    exclude: Sequence[str] = (),
) -> list[str]:
    """Recursively find C++ sources under ``dirs`` (paths relative to the root)."""
    files: list[str] = []
    for directory in dirs:
        base = PROJECT_ROOT / directory
        if not base.is_dir():
            continue
        for ext in exts:
            for path in base.rglob(f"*.{ext}"):
                rel = path.relative_to(PROJECT_ROOT).as_posix()
                if any(token in rel for token in exclude):
                    continue
                files.append(str(path))
    return sorted(files)


def app_binary() -> Path:
    """Path to the built game binary (release)."""
    return PROJECT_ROOT / "build" / "release" / "app" / f"md_app{EXE}"
