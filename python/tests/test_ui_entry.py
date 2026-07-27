# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for ``python -m md.ui``'s one job when it cannot start.

The entry point exists to turn a missing package into a sentence, and it had no
test at all until the day it shipped the opposite: everything except PySide6
went out as a stack trace, and the package that actually went missing was
``numpy`` — a *base* dependency, reached through ``md.league``. Started from the
game, which on Windows is a GUI-subsystem binary with no console attached, that
trace had nowhere to appear: the menu entry did nothing at all.

So what is asserted here is coverage of the message, not its wording: every
package it can be handed produces a named fix, on both kinds of interpreter.
"""

from __future__ import annotations

from md.ui.__main__ import explain, package_of


def test_a_missing_pyside6_names_pyside6_and_a_pip_command() -> None:
    message = explain("PySide6", managed=False, interpreter="/opt/py/python")
    assert "PySide6" in message
    assert "/opt/py/python -m pip install PySide6" in message


def test_a_missing_numpy_is_explained_rather_than_raised() -> None:
    # The regression. numpy is not an extra and not optional, so the message must
    # not describe it as one — a reader told "it is optional" about the package
    # their console just died on has been sent to look in the wrong place.
    message = explain("numpy", managed=False, interpreter="/opt/py/python")
    assert "numpy" in message
    assert "/opt/py/python -m pip install numpy" in message
    assert "optional" not in message


def test_an_externally_managed_interpreter_is_told_to_use_apt() -> None:
    # PEP 668: `pip install` is refused by design there, so naming it would send
    # the reader to a command their machine will decline.
    message = explain("PySide6", managed=True)
    assert "sudo apt install python3-pyside6.qtcharts" in message
    # The prose still says the words "pip install" — it is explaining what the
    # interpreter refuses. What must be absent is the runnable form of it.
    assert "-m pip install" not in message


def test_a_package_with_no_distribution_name_still_gets_an_apt_guess() -> None:
    # Better a conventional `python3-<name>` than silence: the table cannot list
    # every transitive import, and the shape of the answer is right even when the
    # exact package name is not.
    assert "sudo apt install python3-psutil" in explain("psutil", managed=True)


def test_a_submodule_import_names_the_package_that_installs_it() -> None:
    # `pip install PySide6.QtCharts` is a command that cannot succeed.
    assert package_of(ModuleNotFoundError(name="PySide6.QtCharts")) == "PySide6"
    assert package_of(ModuleNotFoundError(name="numpy")) == "numpy"


def test_an_error_naming_nothing_is_not_dressed_up_as_advice() -> None:
    # `main` re-raises this case rather than printing a sentence with a blank in
    # it: with no name there is nothing the message can say that the traceback
    # does not say better.
    assert package_of(ModuleNotFoundError("something went wrong")) == ""
