# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
"""The environment bootstrap installs the project, trainer, and dev tools."""

from __future__ import annotations

from pathlib import Path

from tools.bootstrap import (
    DEV_TOOLS,
    TEST_INDEX,
    install_command,
    torch_install_command,
    venv_python,
)


def test_the_trainer_extra_is_part_of_the_bootstrap(tmp_path: Path) -> None:
    command = install_command(tmp_path / ".venv", root=tmp_path / "project")

    assert "--editable" in command
    assert f"{tmp_path / 'project'}[trainer]" in command
    assert set(DEV_TOOLS).issubset(command)


def test_the_venv_interpreter_is_platform_native(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    assert venv_python(venv, platform="linux") == venv / "bin/python"
    assert venv_python(venv, platform="win32") == venv / "Scripts/python.exe"


def test_bootstrap_installs_what_the_binding_needs_to_build(tmp_path: Path) -> None:
    """The gap that made a bootstrapped trainer unable to train.

    nanobind is in `[build-system].requires`, which covers building the *wheel*
    — pip makes an isolated environment for that. It does not cover
    `cmake --build --target _md_native`, which `poe bindings` runs against this
    venv, and without it CMake never creates the target at all.

    The symptom was two steps away and named the wrong thing: the trainer's
    runtime installer downloaded five gigabytes of CUDA torch, health-checked it
    by importing the binding, found none, and reported the torch install as
    failed. `.github/workflows/ci.yml` runs the real thing; this fails in a
    second rather than in ten minutes.
    """
    command = install_command(tmp_path / ".venv")
    assert any(argument.startswith("nanobind") for argument in command), (
        f"bootstrap would produce a venv that cannot build _md_native: {command}"
    )


def test_torch_is_installed_for_tests_and_from_its_own_index() -> None:
    """A second pip call, on purpose, and the CPU wheel on purpose too.

    `--index-url` applies to a whole command, so folding torch into the main
    install would resolve *every* package against the PyTorch index instead of
    PyPI. `--extra-index-url` would work and is the wrong tool: it makes both
    indexes eligible for every name, which is the dependency-confusion shape.

    The CPU wheel because the alternative is three gigabytes of CUDA that would
    be the second copy on a development machine — the trainer's managed runtime
    already holds the one real training uses, and `runner.find_interpreter`
    prefers it, so a venv torch is what tests import and never what a run gets.
    """
    command = torch_install_command(Path("/tmp/venv"))
    assert "--index-url" in command
    assert command[command.index("--index-url") + 1].endswith("/cpu")
    assert "torch" in command
    # And it is not in the invocation that installs everything else.
    main_install = install_command(Path("/tmp/venv"))
    assert "torch" not in main_install
    assert "--index-url" not in main_install


def test_the_index_torch_comes_from_is_one_the_project_already_trusts() -> None:
    """The same allow-list the managed runtime is held to.

    `md.runs.runtime` refuses to install from anywhere but these two hosts, and a
    development bootstrap reaching somewhere else would be the project trusting a
    host in one place that it rejects in another.
    """
    from urllib.parse import urlparse  # noqa: PLC0415 — one call, one test

    from missile_defense.runs.runtime import ALLOWED_INDEX_HOSTS  # noqa: PLC0415

    assert urlparse(TEST_INDEX).hostname in ALLOWED_INDEX_HOSTS
