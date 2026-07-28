# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Fixtures shared by the e2e suite.

The expensive one is :func:`trained_run`, and it is session-scoped on purpose: a
real training run is the input to most of what follows — the trainer has to have
something to attach to, and the replay tests need a recording that an actual
trainer wrote — but running one per test would turn minutes into an hour.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from .harness import recordings, train


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory a real, short training run has finished writing into.

    Real, and *finished*: the run is waited on rather than sampled while live, so
    every artifact a completed run owes is present and no test races the trainer
    for a file it is halfway through writing.
    """
    pytest.importorskip("torch", reason="torch is not installed")
    out_dir = tmp_path_factory.mktemp("trained") / "run"
    result = train(out_dir)
    assert result.returncode == 0, f"the trainer failed:\n{result.stdout}\n{result.stderr}"
    assert recordings(out_dir), "the run wrote no recording — see TINY_RUN's max-ticks"
    return out_dir


@pytest.fixture(scope="session")
def qt_app() -> Iterator[object]:
    """One offscreen QApplication for the whole session.

    Offscreen rather than a virtual X server, because Qt Widgets needs no
    graphics device — only the game does, and only because it is Vulkan. So the
    trainer tests run invisibly anywhere, with nothing to install.
    """
    pytest.importorskip("PySide6", reason="PySide6 is not installed")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415 — optional dependency

    app = QApplication.instance() or QApplication([])
    yield app
    app.quit()  # type: ignore[union-attr]
