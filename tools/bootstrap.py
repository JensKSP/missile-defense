# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
"""Create the development venv with the console and its telemetry backends.

Run from a fresh checkout with::

    python3 -m tools.bootstrap

PyTorch remains separate because its correct wheel depends on the host GPU and
driver. The console is small enough to install consistently, including the
NVIDIA binding everywhere and AMD SMI on Linux.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ._util import PROJECT_ROOT

DEV_TOOLS = ("poethepoet", "ruff", "pytest", "mypy", "pyright")


def venv_python(venv: Path, *, platform: str = sys.platform) -> Path:
    """The interpreter a venv creates on this platform."""
    return venv / ("Scripts/python.exe" if platform == "win32" else "bin/python")


def install_command(venv: Path, *, root: Path = PROJECT_ROOT) -> list[str]:
    """The one pip invocation that defines a ready development console."""
    return [
        str(venv_python(venv)),
        "-m",
        "pip",
        "install",
        "--editable",
        f"{root}[console]",
        *DEV_TOOLS,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a development venv with tooling and console dependencies."
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=PROJECT_ROOT / ".venv",
        help="environment to create or update (default: .venv)",
    )
    args = parser.parse_args(argv)
    venv = args.venv.resolve()

    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    subprocess.run(install_command(venv), check=True)

    activate = venv / ("Scripts/activate" if sys.platform == "win32" else "bin/activate")
    print(f"ready: {venv}")
    print(f"activate it with: source {activate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
