# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The trainer is not a trainer, and this is what keeps it that way.

Scope creep into ``md.ui`` would be the easy mistake — a "quick" model summary
that loads a checkpoint, and suddenly the window owns GPU memory and a crash
takes the run with it. So the rule is structural rather than remembered
(docs/ROADMAP.md, M8, risk 3): **``md.ui`` must never import torch.**

It is checked twice, because either check alone has a hole. The import check is
the real one but passes trivially in an interpreter that has no torch installed —
which is exactly the interpreter this project's quality gate runs in on Windows.
The source scan does not care whether torch exists.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import md
import md.ui

PACKAGE = Path(md.ui.__file__).parent
IMPORT_PATH = str(Path(md.__file__).parents[1])

#: Importing torch anywhere under md.ui would make the trainer a trainer;
#: md.train and md.ppo are just torch by another name.
FORBIDDEN = ("torch", "md.train", "md.ppo")

CHECK = """
import sys

import md.ui
import md.ui.runner
import md.ui.sources
import md.ui.theme

try:  # the widgets, where PySide6 is installed — the trainer is optional
    import md.ui.app
    import md.ui.charts
    import md.ui.runtime_dialog
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("PySide6"):
        raise

print("torch" in sys.modules)
"""


def test_importing_the_trainer_does_not_import_torch() -> None:
    # A fresh interpreter, so nothing another test imported can mask the answer.
    result = subprocess.run(
        [sys.executable, "-c", CHECK],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": IMPORT_PATH},
    )
    assert result.stdout.strip() == "False", result.stderr


def test_no_module_under_md_ui_imports_torch() -> None:
    for path in sorted(PACKAGE.glob("*.py")):
        for name in _imported_by(path):
            assert not name.startswith(FORBIDDEN), f"{path.name} imports {name}"


def _imported_by(path: Path) -> list[str]:
    """Every module name ``path`` imports, absolute imports only."""
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names
