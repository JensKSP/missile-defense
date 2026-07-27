# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The type-checker skip lists must describe reality, not somebody's memory.

PySide6 is optional — the console is LGPLv3 and never a dependency of the game —
so the widget modules are checked on a machine that has it, and skipped where it
is absent. Both checkers are told which those are by hand, in `pyproject.toml`.

**That enumeration drifted and cost days of red CI.** `analysis`, `library`,
`league` and `storage` were added without being listed, and the failure mode is
the worst kind: subclassing a Qt class resolves to `Any` only when PySide6 is
*missing*, so every developer machine stayed green while CI — the one
environment without it — failed on every push. Nobody looks at a list they are
not told about.

So the list is checked instead of trusted. A module that imports PySide6 must be
skipped; one that does not must be checked. Adding either kind now fails here,
with the line to add, rather than in CI three commits later.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI = PROJECT_ROOT / "python" / "md" / "ui"


def _config() -> dict[str, object]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _qt_modules() -> set[str]:
    """Every `md.ui` module whose source mentions PySide6.

    Textual on purpose: importing them to find out needs the very dependency
    whose absence this is about.
    """
    return {path.stem for path in sorted(UI.glob("*.py")) if "PySide6" in path.read_text("utf-8")}


def _pure_modules() -> set[str]:
    return {path.stem for path in sorted(UI.glob("*.py"))} - _qt_modules()


def _mypy_skipped() -> set[str]:
    config = _config()
    tool = config["tool"]
    assert isinstance(tool, dict)
    mypy = tool["mypy"]
    assert isinstance(mypy, dict)
    overrides = mypy["overrides"]
    assert isinstance(overrides, list)

    skipped: set[str] = set()
    for override in overrides:
        assert isinstance(override, dict)
        if not override.get("ignore_errors"):
            continue
        modules = override["module"]
        names = [modules] if isinstance(modules, str) else modules
        assert isinstance(names, list)
        for name in names:
            if isinstance(name, str) and name.startswith("md.ui."):
                skipped.add(name.removeprefix("md.ui."))
    return skipped


def _pyright_ignored() -> set[str]:
    config = _config()
    tool = config["tool"]
    assert isinstance(tool, dict)
    pyright = tool["pyright"]
    assert isinstance(pyright, dict)
    ignored = pyright["ignore"]
    assert isinstance(ignored, list)
    return {
        Path(str(entry)).stem
        for entry in ignored
        if str(entry).startswith("python/md/ui/") and str(entry).endswith(".py")
    }


def test_every_qt_module_is_skipped_by_mypy() -> None:
    missing = sorted(_qt_modules() - _mypy_skipped())
    assert not missing, (
        "these md.ui modules import PySide6 and are not in the mypy override, so "
        "the gate fails wherever PySide6 is absent (which is CI, and only CI):\n"
        + "\n".join(f'  "md.ui.{name}",' for name in missing)
    )


def test_every_qt_module_is_ignored_by_pyright() -> None:
    missing = sorted(_qt_modules() - _pyright_ignored())
    assert not missing, (
        "these md.ui modules import PySide6 and are not in pyright's ignore list:\n"
        + "\n".join(f'  "python/md/ui/{name}.py",' for name in missing)
    )


def test_the_qt_free_half_is_still_actually_checked() -> None:
    """The other direction, which matters just as much.

    "Skip the widgets" is a narrow exemption for a missing dependency. A
    Qt-free module quietly inheriting it would be unchecked code hiding behind
    an excuse that does not apply to it.
    """
    pure = _pure_modules()
    assert pure, "no Qt-free modules found — the detection is wrong, not the tree"
    for skipped, checker in ((_mypy_skipped(), "mypy"), (_pyright_ignored(), "pyright")):
        waved = sorted(pure & skipped)
        assert not waved, f"{checker} skips these, but they do not import PySide6: {waved}"


def test_the_two_checkers_skip_the_same_files() -> None:
    # Two lists, one rule. If they disagree, one of them is wrong and the next
    # person has no way to tell which.
    only_mypy = sorted(_mypy_skipped() - _pyright_ignored())
    only_pyright = sorted(_pyright_ignored() - _mypy_skipped())
    assert not only_mypy and not only_pyright, (
        f"mypy-only: {only_mypy}; pyright-only: {only_pyright}"
    )
