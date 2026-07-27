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

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from . import _util


def _extension_suffix(interpreter: str) -> str:
    """What a *version-tagged* module for ``interpreter`` would be called.

    Asked of that interpreter rather than derived from the running one: the whole
    point of `--python` is that the two differ, and guessing here would delete a
    module belonging to a Python nobody built for.
    """
    probe = subprocess.run(
        [interpreter, "-c", "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.stdout.strip() if probe.returncode == 0 else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", nargs="?", default="release", help="CMake configure preset")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help=(
            "Interpreter to build against. Defaults to the running one. Point this "
            "at a native Windows CPython together with --preset win-native: torch "
            "ships no MinGW wheel, so training needs a module built for that ABI."
        ),
    )
    parsed = parser.parse_args(sys.argv[1:] if argv is None else argv)
    preset = parsed.preset

    # Point CMake at *this* interpreter: nanobind's headers come from the
    # environment the extension will be imported into, so a mismatch here shows up
    # much later as an import error.
    _util.run(
        [
            "cmake",
            "--preset",
            preset,
            "-DMD_BUILD_BINDINGS=ON",
            f"-DPython_EXECUTABLE={parsed.python}",
        ]
    )
    _util.run(["cmake", "--build", "--preset", preset, "--target", "_md_native"])

    output = _util.PROJECT_ROOT / "build" / preset / "python" / "md"
    built = sorted(output.glob("_md_native*"))
    modules = [p for p in built if p.suffix in {".so", ".pyd"}]
    if not modules:
        print(
            "error: the extension was not built — CMake skips it when Python or "
            "nanobind is missing (pip install nanobind).",
            file=sys.stderr,
        )
        return 1

    # The build tree keeps whatever an *earlier* configure produced: CMake links
    # to a new output name and never deletes the old one, so a checkout that
    # predates the stable ABI has both, and copying "everything built" would
    # carry the superseded module across as well. Drop it at the source.
    tagged = f"_md_native{_extension_suffix(parsed.python)}"
    if any(".abi3" in m.name for m in modules):
        for ghost in [m for m in modules if m.name == tagged]:
            ghost.unlink()
            modules.remove(ghost)

    package: Path = _util.PROJECT_ROOT / "python" / "md"
    # Clear only what this build supersedes: the exact names it is about to write,
    # plus the untagged form of each, which is a stale shadow of the same module.
    # A module carrying a *different* ABI tag is deliberately left alone. Windows
    # keeps an MSYS2/MinGW build and an MSVC one side by side (see
    # docs/TRAINING.md) because torch ships no MinGW wheel, and an interpreter
    # imports only the tag that is its own — so deleting the other one breaks the
    # other toolchain and buys nothing.
    superseded = {m.name for m in modules} | {f"_md_native{m.suffix}" for m in modules}
    # One exception, and it is the reverse of the rule above. An abi3 module and
    # a version-tagged one for the *same* interpreter are not two toolchains
    # coexisting, they are one module and its ghost — and the ghost wins, because
    # `importlib` tries the version-tagged suffix before `.abi3`. So a checkout
    # that had built the tagged module once would keep importing it for ever,
    # including the run that was meant to prove the stable ABI works.
    if any(".abi3" in m.name for m in modules):
        superseded |= {tagged}
    for stale in package.glob("_md_native*"):
        if stale.name in superseded:
            stale.unlink()
    for module in modules:
        shutil.copy2(module, package / module.name)
        print(f"copied {module.name} -> python/md/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
