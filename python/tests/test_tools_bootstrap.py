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


def test_bootstrap_installs_what_the_binding_needs_to_build(tmp_path: Path) -> None:
    """The gap that made a bootstrapped console unable to train.

    nanobind is in `[build-system].requires`, which covers building the *wheel*
    — pip makes an isolated environment for that. It does not cover
    `cmake --build --target _md_native`, which `poe bindings` runs against this
    venv, and without it CMake never creates the target at all.

    The symptom was two steps away and named the wrong thing: the console's
    runtime installer downloaded five gigabytes of CUDA torch, health-checked it
    by importing the binding, found none, and reported the torch install as
    failed. `.github/workflows/ci.yml` runs the real thing; this fails in a
    second rather than in ten minutes.
    """
    command = install_command(tmp_path / ".venv")
    assert any(argument.startswith("nanobind") for argument in command), (
        f"bootstrap would produce a venv that cannot build _md_native: {command}"
    )
