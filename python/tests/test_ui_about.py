# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for what the trainer says about itself.

Two jobs, and only one of them is cosmetic. The version answers "which build am
I looking at?", which a bug report is useless without. The component list is the
LGPL notice: the trainer runs on PySide6 and Qt Charts, and a user is entitled to
be told so *by the program*, not only by a file in a repository they may never
have seen (THIRD_PARTY_LICENSES.md).

No Qt here. ``missile_defense.ui.about`` assembles text and nothing else, which is what lets
this run on a machine with no display and no PySide6 at all.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import missile_defense
from missile_defense.ui import about


def test_it_reports_the_packages_own_version() -> None:
    assert about.version() == missile_defense.__version__


def test_it_names_the_author_and_the_licence() -> None:
    text = about.summary()
    assert "Jens Köhler" in text
    assert "MIT" in text


def test_the_notice_names_pyside6_and_its_licence() -> None:
    # The obligation this satisfies: PySide6 and Qt Charts are LGPL-3.0 and the
    # trainer is the only place a user meets them.
    named = {component.name: component for component in about.components()}
    assert "PySide6" in named
    assert named["PySide6"].licence == "LGPL-3.0-only"
    assert "Qt Charts" in named["PySide6"].role


def test_a_component_that_is_not_installed_is_reported_rather_than_raised() -> None:
    # Every optional half is optional by design, so About must render on the
    # thinnest possible install — the one where torch, psutil and both GPU
    # probes are all absent — instead of being the one screen that crashes.
    missing = about.Component(
        name="nothing-at-all", version=None, licence="BSD-3-Clause", role="a package nobody has"
    )
    assert missing.installed is False
    assert "not installed" in about.render_component(missing)


def test_it_points_at_the_full_inventory() -> None:
    # About is a summary; the file is the authority, and the two must not drift
    # into being two different claims.
    assert "THIRD_PARTY_LICENSES.md" in about.summary()


def test_finding_a_version_does_not_import_the_package() -> None:
    # `missile_defense.ui` must never import torch (docs/ROADMAP.md M8, risk 3), and "report
    # which torch the managed runtime installed" is exactly the innocent-looking
    # feature that would break it. importlib.metadata reads the *metadata*, so it
    # answers without loading a multi-gigabyte extension module.
    check = (
        "import sys\n"
        "from missile_defense.ui import about\n"
        "about.components()\n"
        "about.summary()\n"
        "print('torch' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", check],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(missile_defense.__file__).parents[1])},
    )
    assert result.stdout.strip() == "False", result.stderr
