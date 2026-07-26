# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
"""The environment bootstrap installs the project, console, and dev tools."""

from __future__ import annotations

from pathlib import Path

from tools.bootstrap import DEV_TOOLS, install_command, venv_python


def test_the_console_extra_is_part_of_the_bootstrap(tmp_path: Path) -> None:
    command = install_command(tmp_path / ".venv", root=tmp_path / "project")

    assert "--editable" in command
    assert f"{tmp_path / 'project'}[console]" in command
    assert set(DEV_TOOLS).issubset(command)


def test_the_venv_interpreter_is_platform_native(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    assert venv_python(venv, platform="linux") == venv / "bin/python"
    assert venv_python(venv, platform="win32") == venv / "Scripts/python.exe"
