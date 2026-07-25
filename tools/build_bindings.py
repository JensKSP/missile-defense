# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Build the `_md_native` extension and put it beside the Python package.

CMake writes the module into the build tree, but `md.env` imports it as
``md._md_native`` — so it has to sit next to ``python/md/``. Copying it there (it
is gitignored) keeps ``pytest`` and an interactive session working straight from
a source checkout, with no install step and no PYTHONPATH juggling.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from . import _util


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    preset = args[0] if args else "release"

    # Point CMake at *this* interpreter: nanobind's headers come from the
    # environment the extension will be imported into, so a mismatch here shows up
    # much later as an import error.
    _util.run(
        [
            "cmake",
            "--preset",
            preset,
            "-DMD_BUILD_BINDINGS=ON",
            f"-DPython_EXECUTABLE={sys.executable}",
        ]
    )
    _util.run(["cmake", "--build", "--preset", preset, "--target", "_md_native"])

    built = sorted((_util.PROJECT_ROOT / "build" / preset / "python" / "md").glob("_md_native*"))
    modules = [p for p in built if p.suffix in {".so", ".pyd"}]
    if not modules:
        print(
            "error: the extension was not built — CMake skips it when Python or "
            "nanobind is missing (pip install nanobind).",
            file=sys.stderr,
        )
        return 1

    package: Path = _util.PROJECT_ROOT / "python" / "md"
    for stale in package.glob("_md_native*"):
        if stale.suffix in {".so", ".pyd"}:
            stale.unlink()
    for module in modules:
        shutil.copy2(module, package / module.name)
        print(f"copied {module.name} -> python/md/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
