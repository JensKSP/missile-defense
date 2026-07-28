# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for picking the interpreter that can run a module — without running one.

``tools.launch`` takes its probe as an argument for the same reason
``missile_defense.ui.runner`` takes its spawn function: what matters is the order it asks in
and what it says when the answer is no, and neither needs a second Python
installed to check. The one thing that *is* machine-shaped — the ``py -0p``
output on Windows — is parsed by a pure function, so it is checked here against
the lines a real launcher prints.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from tools import launch
from tools.launch import Report, candidates, child_environ, explain, parse_launcher_line, survey


class FakeProbe:
    """An interpreter census this machine does not have to actually contain."""

    def __init__(self, has: dict[str, Sequence[str]], *, dead: Sequence[str] = ()) -> None:
        self._has = has
        self._dead = set(dead)
        self.asked: list[str] = []

    def __call__(self, python: str, modules: Sequence[str]) -> Report | None:
        self.asked.append(python)
        if python in self._dead:
            return None
        installed = set(self._has.get(python, ()))
        return Report(
            python=python,
            version="3.12.1",
            missing=tuple(name for name in modules if name not in installed),
        )


# ---- the search order --------------------------------------------------------


def test_md_python_is_asked_before_anything_else() -> None:
    """An explicit answer ends the question — that is what the variable is for."""
    chosen = candidates({"MD_PYTHON": "/opt/native/python", "PATH": ""}, platform="linux")
    assert chosen[0] == "/opt/native/python"


def test_the_running_interpreter_comes_next() -> None:
    """A machine with one Python must never pay for the search."""
    chosen = candidates({"PATH": ""}, platform="linux", executable="/usr/bin/python3.11")
    assert chosen[0] == "/usr/bin/python3.11"


def test_the_same_interpreter_twice_is_one_candidate() -> None:
    chosen = candidates(
        {"MD_PYTHON": sys.executable, "PATH": ""}, platform="linux", executable=sys.executable
    )
    assert chosen == [sys.executable]


def test_the_survey_stops_at_the_first_usable_interpreter() -> None:
    probe = FakeProbe({"/b": ["torch"]})
    reports = survey(["torch"], probe=probe, pythons=["/a", "/b", "/c"])

    assert probe.asked == ["/a", "/b"]  # /c was never started
    assert reports[-1].usable and reports[-1].python == "/b"


def test_an_interpreter_that_will_not_run_is_not_reported() -> None:
    """A stale `py -0p` entry is not a finding, so it is not worth a line."""
    probe = FakeProbe({"/b": ["PySide6"]}, dead=["/a"])
    reports = survey(["PySide6"], probe=probe, pythons=["/a", "/b"])

    assert [report.python for report in reports] == ["/b"]


def test_nothing_usable_reports_every_interpreter_it_looked_at() -> None:
    probe = FakeProbe({})
    reports = survey(["torch"], probe=probe, pythons=["/a", "/b"])

    assert [report.python for report in reports] == ["/a", "/b"]
    assert all(report.missing == ("torch",) for report in reports)


# ---- what it says when there is no answer ------------------------------------


def test_the_failure_names_the_package_the_interpreters_and_the_fix() -> None:
    reports = [
        Report("/usr/bin/python3", "3.14.6", ("PySide6",)),
        Report("/opt/native/python", "3.12.1", ("PySide6",)),
    ]
    message = explain("missile_defense.ui", (("PySide6", "PySide6"),), reports)

    assert "PySide6" in message
    assert "/usr/bin/python3" in message and "/opt/native/python" in message
    assert "pip install PySide6" in message
    assert "MD_PYTHON" in message
    assert "poe ui" in message


def test_the_failure_still_helps_when_no_interpreter_answered() -> None:
    message = explain("missile_defense.train", (("torch", "torch"),), [])

    assert "pip install torch" in message
    assert "poe train" in message


# ---- the Windows launcher's list ---------------------------------------------


def test_a_py_launcher_row_yields_its_path() -> None:
    line = " -V:3.12          C:\\Users\\jens\\AppData\\Local\\Programs\\Python312\\python.exe"
    assert parse_launcher_line(line).endswith("Python312\\python.exe")


def test_the_star_on_the_default_is_not_part_of_the_path() -> None:
    line = " -V:3.13 *        C:\\Python313\\python.exe"
    assert parse_launcher_line(line) == "C:\\Python313\\python.exe"


def test_a_path_with_a_space_in_it_survives() -> None:
    """`C:\\Program Files\\…` is the reason this is not `line.split()[-1]`."""
    line = " -V:3.12 *        C:\\Program Files\\Python312\\python.exe"
    assert parse_launcher_line(line) == "C:\\Program Files\\Python312\\python.exe"


def test_a_line_that_is_not_an_entry_is_skipped() -> None:
    assert parse_launcher_line("Installed Pythons found by py Launcher") == ""
    assert parse_launcher_line("") == ""


# ---- the child's import path -------------------------------------------------


def test_the_checkout_is_put_on_the_import_path() -> None:
    """The chosen interpreter is a stranger to this source tree."""
    env = child_environ({})
    assert str(launch.PROJECT_ROOT / "python") in env["PYTHONPATH"].split(os.pathsep)


def test_an_existing_pythonpath_is_kept() -> None:
    env = child_environ({"PYTHONPATH": "/somewhere/else"})
    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries[0] == str(launch.PROJECT_ROOT / "python")
    assert "/somewhere/else" in entries


def test_the_checkout_is_not_added_twice() -> None:
    package = str(launch.PROJECT_ROOT / "python")
    env = child_environ({"PYTHONPATH": package})
    assert env["PYTHONPATH"] == package


# ---- what reaches the module -------------------------------------------------


def test_poes_separator_is_not_passed_on() -> None:
    """`poe train -- --updates 20` is the documented way to pass a flag."""
    assert launch.forwarded(["--", "--updates", "20"]) == ["--updates", "20"]


def test_only_the_leading_separator_is_dropped() -> None:
    # A later `--` may be an argument in its own right; that is the module's
    # business, not this launcher's.
    assert launch.forwarded(["--updates", "20", "--", "x"]) == ["--updates", "20", "--", "x"]


def test_no_arguments_stays_no_arguments() -> None:
    assert launch.forwarded([]) == []
