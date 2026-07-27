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

#: `build` is here for one test: `tests/e2e/test_wheel_install.py` installs the
#: wheel this repository produces into a fresh interpreter, which is the only
#: check in the suite that runs the *shipped* package rather than the source
#: tree. Without it that whole file skips, and the packaging bugs it exists to
#: catch — a module left out, a binding not copied, a renamed entry point —
#: reach a release instead.
DEV_TOOLS = ("poethepoet", "ruff", "pytest", "mypy", "pyright", "build")

#: Constraints the *gate* needs, over and above what the package needs to run.
#:
#: `numpy<2.5` is a typing requirement rather than a runtime one. numpy 2.5
#: dropped Python 3.11 and writes PEP 695 `type` statements in its own stubs,
#: which mypy parses only when targeting 3.12+ — so a 2.5 numpy beside
#: `python_version = "3.11"` fails inside a stub file nothing in this repository
#: can edit. CI pins the same bound for the same reason; pinning it here means a
#: freshly bootstrapped venv passes `poe check` instead of failing on somebody
#: else's file.
#:
#: Pillow is what `tools/make_icon.py` needs, and installing it is the difference
#: between that script being type-checked and being waved through.
GATE_PINS = ("numpy<2.5", "pillow")


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
        *GATE_PINS,
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
